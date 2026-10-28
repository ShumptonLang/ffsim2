"""Assemble everything the simulator needs from a LeagueState.

The result is plain arrays indexed by (week, team, player) so the engine can
stay vectorized and decision modules can swap rosters cheaply.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from ffsim.data.projections import ProjectionSource
from ffsim.league import SLOT_ELIGIBILITY, LeagueState, PlayoffFormat
from ffsim.model.bayes import TalentModel, prior_rate
from ffsim.model.correlation import assign_roles
from ffsim.scoring import score
from ffsim.sim.lineups import best_lineup


@dataclass
class SimInputs:
    team_ids: list[int]  # roster_ids, defines team index order
    team_labels: list[str]
    rosters: dict[int, list[str]]  # roster_id -> lineup-eligible player ids
    slots: list[str]
    weeks: list[int]  # all simulated weeks (regular remainder + playoffs), ascending
    regular_weeks: list[int]  # simulated regular-season weeks
    playoffs: PlayoffFormat
    matchups: dict[int, list[tuple[int, int]]]  # week -> [(team_idx, team_idx)]
    base_half_wins: np.ndarray  # (T,) 2*wins + ties so far
    base_points_for: np.ndarray  # (T,)
    divisions: list[int | None]
    median_match: bool
    means: dict[int, dict[str, float]]  # week -> player_id -> projected points
    positions: dict[str, tuple[str, ...]]  # player_id -> fantasy positions
    forced_playoff_winners: dict[int, set[int]] = field(default_factory=dict)
    actual_points: dict[int, np.ndarray] = field(default_factory=dict)  # scored playoff weeks
    # Injury designation for the first simulated week (only when it is the current NFL week).
    statuses: dict[str, str] = field(default_factory=dict)
    # Injured player -> (week he could return, games missed): out until availability says
    returning: dict[str, tuple[int, int]] = field(default_factory=dict)
    sleeper_means: dict[int, dict[str, float]] = field(default_factory=dict)  # before blending
    rates: dict[str, float] = field(default_factory=dict)  # Bayesian per-game rate
    # week -> player_id -> (NFL team, opponent) / correlation role
    nfl_games: dict[int, dict[str, tuple[str, str]]] = field(default_factory=dict)
    roles: dict[int, dict[str, str]] = field(default_factory=dict)
    free_agents: list[str] = field(default_factory=list)  # candidate pickups, best first
    roster_capacity: int = 0  # lineup-eligible roster spots (excludes IR / taxi)

    @property
    def n_teams(self) -> int:
        return len(self.team_ids)

    def team_index(self, roster_id: int) -> int:
        return self.team_ids.index(roster_id)

    def with_rosters(self, rosters: Mapping[int, list[str]]) -> SimInputs:
        """Copy with some rosters replaced (for trade / waiver evaluation)."""
        return replace(self, rosters={**self.rosters, **rosters})

    @property
    def universe(self) -> list[str]:
        """Every player the simulation may need: rostered players plus free agents."""
        seen = {p for ps in self.rosters.values() for p in ps}
        return sorted(seen | set(self.free_agents))

    def ros_points(self, player_id: str, weeks: list[int] | None = None) -> float:
        """Projected points over the remaining simulated weeks (or the given weeks)."""
        return sum(self.means.get(w, {}).get(player_id, 0.0) for w in (weeks or self.weeks))

    def team_plan(
        self, week: int, players: list[str], starters: list[str | None] | None = None
    ) -> list[list[str]]:
        """Per starting slot, candidates in order of preference for one team-week.

        The first candidate is the starter: ``starters`` (one per slot) when given,
        else the projection-optimal lineup. The rest are eligible bench players by
        projection, used when the starter turns out to be unavailable.
        """
        m = self.means.get(week, {})
        lineup = (
            starters
            if starters is not None
            else best_lineup(self.slots, players, m, self.positions)
        )
        chosen = {p for p in lineup if p}
        bench = sorted(
            (p for p in players if p not in chosen and m.get(p, 0.0) > 0),
            key=lambda p: -m.get(p, 0.0),
        )
        plan = []
        for slot, starter in zip(self.slots, lineup, strict=True):
            ok = SLOT_ELIGIBILITY.get(slot, frozenset())
            backups = [p for p in bench if any(fp in ok for fp in self.positions.get(p, ()))]
            plan.append(([starter] if starter else []) + backups)
        return plan

    def lineup_plans(self) -> dict[int, dict[int, list[list[str]]]]:
        """week -> roster_id -> per starting slot, candidates in order of preference."""
        return {
            w: {rid: self.team_plan(w, players) for rid, players in self.rosters.items()}
            for w in self.weeks
        }

    def lineups(self) -> dict[int, dict[int, list[str]]]:
        """week -> roster_id -> starting player ids (projection-optimal)."""
        out: dict[int, dict[int, list[str]]] = {}
        for w in self.weeks:
            m = self.means.get(w, {})
            out[w] = {
                rid: [p for p in best_lineup(self.slots, players, m, self.positions) if p]
                for rid, players in self.rosters.items()
            }
        return out


def week_means(
    source: ProjectionSource,
    state: LeagueState,
    weeks: list[int],
) -> tuple[
    dict[int, dict[str, float]],
    dict[int, dict[str, tuple[str, str]]],
    dict[str, tuple[int, int]],
]:
    """Projected points per player for each week, scored with league settings.

    Also returns each projected player's (NFL team, opponent) per week. A
    player missing or projected 0 in a published week is on bye or out. A week
    with no projections at all (not published yet) has no game info and reuses
    each player's latest projection from a week his team played, so a bye in
    the last published week does not carry forward as a zero.

    An injured player (one with an injury designation) projected 0 in the last
    published week gets his latest healthy projection instead, and is returned
    in the third value as ``pid -> (first unpublished week, games missed)`` so
    the availability model can decide when he comes back.
    """
    settings = state.league.scoring_settings
    out: dict[int, dict[str, float]] = {}
    games: dict[int, dict[str, tuple[str, str]]] = {}
    returning: dict[str, tuple[int, int]] = {}
    last: dict[str, float] = {}
    healthy: dict[str, float] = {}  # latest projection > 0 in a week his team played
    missed: dict[str, int] = {}  # trailing published games projected 0 (byes skipped)
    for w in weeks:
        proj = source.weekly_projections(state.league.season, w)
        games[w] = {}
        if not proj:
            if not returning:
                for pid, k in missed.items():
                    if pid in healthy and state.player(pid).injury_status:
                        returning[pid] = (w, k)
            out[w] = last | {p: healthy[p] for p in returning}
            continue
        m = {}
        for pid, pw in proj.items():
            pos = state.player(pid).primary_position
            m[pid] = max(0.0, score(pw.stats, settings, pos)) if pw.stats.get("gp", 1) > 0 else 0.0
            if pw.team and pw.opponent:
                games[w][pid] = (pw.team, pw.opponent)
        # Sleeper lists bye players with their team but no game; keep their old value
        playing = {t for g in games[w].values() for t in g}
        bye = {p for p, pw in proj.items() if playing and pw.team and pw.team not in playing}
        last = {p: v for p, v in last.items() if p in bye} | {
            p: v for p, v in m.items() if p not in bye
        }
        for pid, v in m.items():
            if pid in bye:
                continue
            if v > 0:
                healthy[pid] = v
                missed.pop(pid, None)
            else:
                missed[pid] = missed.get(pid, 0) + 1
        out[w] = m
    return out, games, returning


def player_rates(
    state: LeagueState,
    source: ProjectionSource,
    talent: TalentModel,
    players: set[str],
) -> dict[str, float]:
    """Posterior per-game rate for each player: preseason prior + games played so far."""
    lg = state.league
    settings = lg.scoring_settings
    priors: dict[str, float] = {}
    for pid, pw in source.season_projections(lg.season).items():
        if pid in players:
            pos = state.player(pid).primary_position
            rate = prior_rate(pw.stats, settings, pos)
            if rate is not None:
                priors[pid] = rate
    games: dict[str, list[float]] = {p: [] for p in players}
    for w in range(lg.start_week, lg.last_scored_week + 1):
        for pid, pw in source.weekly_stats(lg.season, w, final=True).items():
            if pid in games and pw.stats.get("gp", 0) > 0:
                pos = state.player(pid).primary_position
                games[pid].append(score(pw.stats, settings, pos))
    out: dict[str, float] = {}
    for pid in players:
        g = games[pid]
        pos = state.player(pid).primary_position
        rate = talent.rate(pos, priors.get(pid), float(np.mean(g)) if g else 0.0, len(g))
        if rate is not None:
            out[pid] = rate
    return out


def build_inputs(
    state: LeagueState,
    source: ProjectionSource,
    winners_bracket: list[dict] | None = None,
    talent: TalentModel | None = None,
    n_free_agents: int = 0,
) -> SimInputs:
    """Assemble simulation inputs.

    ``n_free_agents`` > 0 also brings in that many of the best unrostered players
    per starting position (by projected points over the simulated weeks), so
    waiver pickups can be evaluated against the same random draws.
    """
    lg = state.league
    po = lg.playoffs
    first = lg.last_scored_week + 1
    last = min(po.last_week, 18)
    weeks = list(range(first, last + 1))
    regular = [w for w in weeks if w in lg.regular_season_weeks]

    team_ids = sorted(state.teams)
    idx = {rid: i for i, rid in enumerate(team_ids)}
    teams = [state.teams[r] for r in team_ids]
    matchups = {w: [(idx[m.home], idx[m.away]) for m in state.schedule.get(w, [])] for w in regular}

    forced: dict[int, set[int]] = {}
    for row in winners_bracket or []:
        r, w = row.get("r"), row.get("w")
        # Sleeper rounds are 1-based. Placement games (p = 3, 5, ...) are off the
        # title path; p = 1 is the championship itself.
        if r and w and int(w) in idx and row.get("p") in (None, 1):
            forced.setdefault(int(r) - 1, set()).add(idx[int(w)])

    rosters = {t.roster_id: list(t.active_players) for t in teams}
    rostered = {p for ps in rosters.values() for p in ps}
    sleeper, games, returning = week_means(source, state, weeks) if weeks else ({}, {}, {})
    roles = {
        w: assign_roles(
            (p, t[0], state.player(p).primary_position, sleeper[w].get(p, 0.0))
            for p, t in wg.items()
        )
        for w, wg in games.items()
    }
    free_agents: list[str] = []
    if n_free_agents and weeks:
        free_agents = _free_agents(state, sleeper, rostered, lg.starter_slots, n_free_agents)
    universe = rostered | set(free_agents)
    rates: dict[str, float] = {}
    means = sleeper
    if talent is not None and weeks:
        rates = player_rates(state, source, talent, universe)
        means = {
            w: {
                p: talent.centre(state.player(p).primary_position, m, rates.get(p))
                for p, m in wm.items()
            }
            for w, wm in sleeper.items()
        }
    statuses: dict[str, str] = {}
    if weeks and weeks[0] == state.nfl_week:
        for p in universe:
            st = state.player(p).injury_status
            if st:
                statuses[p] = st
    return SimInputs(
        team_ids=team_ids,
        team_labels=[t.label for t in teams],
        rosters=rosters,
        slots=list(lg.starter_slots),
        weeks=weeks,
        regular_weeks=regular,
        playoffs=po,
        matchups=matchups,
        base_half_wins=np.array([2 * t.record.wins + t.record.ties for t in teams], dtype=float),
        base_points_for=np.array([t.record.points_for for t in teams]),
        divisions=[t.division for t in teams] if lg.divisions else [],
        median_match=bool(lg.raw_settings.get("league_average_match")),
        means=means,
        positions={p: state.player(p).fantasy_positions for p in universe},
        forced_playoff_winners=forced,
        actual_points={
            w: np.array([pts.get(r, 0.0) for r in team_ids])
            for w, pts in state.playoff_points.items()
        },
        statuses=statuses,
        returning={p: r for p, r in returning.items() if p in universe},
        sleeper_means=sleeper,
        nfl_games={w: {p: g for p, g in wg.items() if p in universe} for w, wg in games.items()},
        roles={w: {p: r for p, r in wr.items() if p in universe} for w, wr in roles.items()},
        rates=rates,
        free_agents=free_agents,
        roster_capacity=sum(1 for s in lg.roster_positions if s not in ("IR", "TAXI")),
    )


def _free_agents(
    state: LeagueState,
    sleeper: Mapping[int, Mapping[str, float]],
    rostered: set[str],
    slots: Sequence[str],
    per_position: int,
) -> list[str]:
    """Best unrostered players at each position that can start in this league."""
    startable = {p for s in slots for p in SLOT_ELIGIBILITY.get(s, ())}
    totals: dict[str, float] = {}
    for wm in sleeper.values():
        for pid, m in wm.items():
            if pid not in rostered and m > 0:
                totals[pid] = totals.get(pid, 0.0) + m
    by_pos: dict[str, list[tuple[float, str]]] = {}
    for pid, tot in totals.items():
        pos = state.player(pid).primary_position
        if pos in startable:
            by_pos.setdefault(pos, []).append((tot, pid))
    out: list[tuple[float, str]] = []
    for members in by_pos.values():
        out.extend(sorted(members, reverse=True)[:per_position])
    return [pid for _, pid in sorted(out, reverse=True)]
