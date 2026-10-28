"""Shared helpers for decision modules."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from ffsim.league import SLOT_ELIGIBILITY, LeagueState, Team
from ffsim.sim.engine import Evaluator, Outcomes, paired_delta
from ffsim.sim.lineups import best_lineup

T = TypeVar("T")


@dataclass
class Context:
    state: LeagueState
    ev: Evaluator
    team: Team

    @property
    def rid(self) -> int:
        return self.team.roster_id

    @property
    def ti(self) -> int:
        return self.ev.team_idx[self.rid]


@dataclass
class Delta:
    """Change in a team's odds vs. the baseline (paired, same random draws)."""

    champion: float
    champion_se: float
    playoffs: float
    playoffs_se: float


def delta(ev: Evaluator, alt: Outcomes, team_index: int) -> Delta:
    n = alt.n
    base = ev.baseline(n)
    po = ev.inputs.playoffs.teams
    dc, sc = paired_delta(base.champion == team_index, alt.champion == team_index)
    dp, sp = paired_delta(
        base.made_playoffs(po)[:, team_index], alt.made_playoffs(po)[:, team_index]
    )
    return Delta(dc, sc, dp, sp)


def successive_halving(
    candidates: Sequence[T],
    score: Callable[[T, int], float],
    stages: Sequence[tuple[int, int]],
) -> list[tuple[T, float]]:
    """Screen many candidates cheaply, then re-score the survivors with more sims.

    ``stages`` is [(n_sims, keep), ...]; the last stage's ``keep`` is the number
    returned. Scores are higher-is-better; the final ranking uses the last stage.
    """
    pool = list(candidates)
    scored: list[tuple[T, float]] = []
    for n, keep in stages:
        scored = sorted(((c, score(c, n)) for c in pool), key=lambda cs: -cs[1])
        pool = [c for c, _ in scored[:keep]]
    return scored[: stages[-1][1]] if stages else []


def ros_lineup_points(ev: Evaluator, players: Sequence[str]) -> float:
    """Projected starting-lineup points summed over the simulated weeks."""
    inp = ev.inputs
    total = 0.0
    for w in inp.weeks:
        m = inp.means.get(w, {})
        lineup = best_lineup(inp.slots, list(players), m, inp.positions)
        total += sum(m.get(p, 0.0) for p in lineup if p)
    return total


def resolve_players(state: LeagueState, names: Iterable[str], pool: Iterable[str]) -> list[str]:
    """Match player names (case-insensitive, full or partial) within ``pool``."""
    pool = list(pool)
    out = []
    for raw in names:
        q = raw.strip().lower()
        if not q:
            continue
        exact = [p for p in pool if state.player(p).name.lower() == q or p == raw.strip()]
        partial = [p for p in pool if q in state.player(p).name.lower()]
        hits = exact or partial
        if len(hits) != 1:
            choices = ", ".join(sorted(state.player(p).name for p in hits)) or "none"
            raise LookupError(f"'{raw}' matches {len(hits)} players ({choices})")
        out.append(hits[0])
    return out


def drop_candidates(ev: Evaluator, roster: Sequence[str], keep: Iterable[str] = ()) -> list[str]:
    """Players ordered from least to most valuable (projected points left this season)."""
    protected = set(keep)
    return sorted((p for p in roster if p not in protected), key=lambda p: ev.inputs.ros_points(p))


def fit_roster(
    ev: Evaluator, roster: list[str], keep: Iterable[str] = ()
) -> tuple[list[str], list[str]]:
    """Trim a roster to capacity by dropping its least valuable players."""
    cap = ev.inputs.roster_capacity or len(roster)
    over = len(roster) - cap
    if over <= 0:
        return roster, []
    drops = drop_candidates(ev, roster, keep)[:over]
    return [p for p in roster if p not in drops], drops


def replacement_levels(ev: Evaluator) -> dict[str, float]:
    """Rest-of-season points of the best non-starter at each position, league-wide.

    A rough "perceived value" yardstick: managers judge trades by points above
    what they could find on a typical bench.
    """
    inp = ev.inputs
    starters_per_pos: dict[str, float] = {}
    for slot in inp.slots:
        elig = SLOT_ELIGIBILITY.get(slot, frozenset())
        for pos in elig:
            starters_per_pos[pos] = starters_per_pos.get(pos, 0.0) + 1.0 / len(elig)
    pts: dict[str, list[float]] = {}
    for roster in inp.rosters.values():
        for p in roster:
            pos = (inp.positions.get(p) or ("?",))[0]
            pts.setdefault(pos, []).append(inp.ros_points(p))
    out = {}
    for pos, vals in pts.items():
        k = int(round(starters_per_pos.get(pos, 0.0) * inp.n_teams))
        vals = sorted(vals, reverse=True)
        out[pos] = vals[k] if k < len(vals) else 0.0
    return out


def perceived_value(ev: Evaluator, players: Iterable[str], repl: dict[str, float]) -> float:
    inp = ev.inputs
    return float(
        sum(
            max(0.0, inp.ros_points(p) - repl.get((inp.positions.get(p) or ("?",))[0], 0.0))
            for p in players
        )
    )


def score_quantiles(points: np.ndarray, qs: Sequence[float] = (0.1, 0.9)) -> list[float]:
    return [float(v) for v in np.quantile(points, qs)]
