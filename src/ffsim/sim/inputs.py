"""Assemble everything the simulator needs from a LeagueState.

The result is plain arrays indexed by (week, team, player) so the engine can
stay vectorized and decision modules can swap rosters cheaply.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from ffsim.data.projections import ProjectionSource
from ffsim.league import LeagueState, PlayoffFormat
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

    @property
    def n_teams(self) -> int:
        return len(self.team_ids)

    def team_index(self, roster_id: int) -> int:
        return self.team_ids.index(roster_id)

    def with_rosters(self, rosters: Mapping[int, list[str]]) -> SimInputs:
        """Copy with some rosters replaced (for trade / waiver evaluation)."""
        return replace(self, rosters={**self.rosters, **rosters})

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


def build_inputs(
    state: LeagueState,
    source: ProjectionSource,
    winners_bracket: list[dict] | None = None,
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
        means=week_means(source, state, weeks) if weeks else {},
        positions={p: state.player(p).fantasy_positions for p in rostered},
        forced_playoff_winners=forced,
        actual_points={
            w: np.array([pts.get(r, 0.0) for r in team_ids])
            for w, pts in state.playoff_points.items()
        },
    )
