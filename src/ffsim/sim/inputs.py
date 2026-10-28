"""Assemble everything the simulator needs from a LeagueState.

The result is plain arrays indexed by (week, team, player) so the engine can
stay vectorized and decision modules can swap rosters cheaply.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from ffsim.data.projections import ProjectionSource
from ffsim.league import SLOT_ELIGIBILITY, LeagueState, PlayoffFormat
from ffsim.model.bayes import TalentModel, prior_rate
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
    sleeper_means: dict[int, dict[str, float]] = field(default_factory=dict)  # before blending
    rates: dict[str, float] = field(default_factory=dict)  # Bayesian per-game rate

    @property
    def n_teams(self) -> int:
        return len(self.team_ids)

    def team_index(self, roster_id: int) -> int:
        return self.team_ids.index(roster_id)

    def with_rosters(self, rosters: Mapping[int, list[str]]) -> SimInputs:
        """Copy with some rosters replaced (for trade / waiver evaluation)."""
        return replace(self, rosters={**self.rosters, **rosters})

    def lineup_plans(self) -> dict[int, dict[int, list[list[str]]]]:
        """week -> roster_id -> per starting slot, candidates in order of preference.

        The first candidate is the projection-optimal starter (if any); the rest
        are eligible bench players by projection, used when the starter is out.
        """
        out: dict[int, dict[int, list[list[str]]]] = {}
        for w in self.weeks:
            m = self.means.get(w, {})
            out[w] = {}
            for rid, players in self.rosters.items():
                lineup = best_lineup(self.slots, players, m, self.positions)
                starters = {p for p in lineup if p}
                bench = sorted(
                    (p for p in players if p not in starters and m.get(p, 0.0) > 0),
                    key=lambda p: -m.get(p, 0.0),
                )
                plan = []
                for slot, starter in zip(self.slots, lineup, strict=True):
                    ok = SLOT_ELIGIBILITY.get(slot, frozenset())
                    backups = [
                        p for p in bench if any(fp in ok for fp in self.positions.get(p, ()))
                    ]
                    plan.append(([starter] if starter else []) + backups)
                out[w][rid] = plan
        return out

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
) -> dict[int, dict[str, float]]:
    """Projected points per player for each week, scored with league settings.

    A week with no projections at all (not published yet) reuses the most recent
    week that has them; a player missing from a published week is on bye/out.
    """
    settings = state.league.scoring_settings
    out: dict[int, dict[str, float]] = {}
    last: dict[str, float] = {}
    for w in weeks:
        proj = source.weekly_projections(state.league.season, w)
        if not proj:
            out[w] = dict(last)
            continue
        m = {
            pid: max(0.0, score(pw.stats, settings, state.player(pid).primary_position))
            if pw.stats.get("gp", 1) > 0
            else 0.0
            for pid, pw in proj.items()
        }
        out[w] = last = m
    return out


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
) -> SimInputs:
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
    sleeper = week_means(source, state, weeks) if weeks else {}
    rates: dict[str, float] = {}
    means = sleeper
    if talent is not None and weeks:
        rates = player_rates(state, source, talent, rostered)
        means = {
            w: {
                p: talent.centre(state.player(p).primary_position, m, rates.get(p))
                for p, m in wm.items()
            }
            for w, wm in sleeper.items()
        }
    statuses: dict[str, str] = {}
    if weeks and weeks[0] == state.nfl_week:
        for p in rostered:
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
        positions={p: state.player(p).fantasy_positions for p in rostered},
        forced_playoff_winners=forced,
        actual_points={
            w: np.array([pts.get(r, 0.0) for r in team_ids])
            for w, pts in state.playoff_points.items()
        },
        statuses=statuses,
        sleeper_means=sleeper,
        rates=rates,
    )
