"""Vectorized lineup setting for many rosters at once.

The static engine solves one assignment per (week, team) with the Hungarian
algorithm. Here lineups are needed per path, per team, per week and for every
roster a manager considers, so slots are filled greedily instead: most
restrictive slot first, each taking the best eligible player left. That is
exact when the slots' eligibility sets are nested or disjoint (QB, RB, WR, TE,
FLEX, SUPER_FLEX, IDP groups, IDP_FLEX) and every player has one position; other
layouts use ``exact_lineups`` where exactness matters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ffsim.league import SLOT_ELIGIBILITY


@dataclass
class SlotSpec:
    slots: list[str]
    order: list[int]  # slot indices, most restrictive first
    elig: np.ndarray  # (P + 1, S) bool; the last row is the empty roster spot
    exact: bool  # greedy filling is optimal for this league and universe

    @classmethod
    def build(
        cls, slots: Sequence[str], players: Sequence[str], positions: Mapping[str, Sequence[str]]
    ) -> SlotSpec:
        sets = [SLOT_ELIGIBILITY.get(s, frozenset()) for s in slots]
        order = sorted(range(len(slots)), key=lambda i: (len(sets[i]), i))
        elig = np.zeros((len(players) + 1, len(slots)), dtype=bool)
        multi = False
        startable = frozenset().union(*sets) if sets else frozenset()
        for j, p in enumerate(players):
            fps = set(positions.get(p, ())) & startable
            multi |= len(fps) > 1
            for i, ok in enumerate(sets):
                elig[j, i] = bool(fps & ok)
        laminar = all(a <= b or b <= a or not (a & b) for a in sets for b in sets)
        return cls(list(slots), order, elig, laminar and not multi)


def greedy_lineups(
    values: np.ndarray, elig: np.ndarray, order: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Best lineups for many rosters at once.

    ``values`` (..., R): each roster member's value (-inf = can't play).
    ``elig`` (..., R, S): member may fill slot. Returns (total (...), starters
    (..., S) member index or -1 for an empty slot).
    """
    shape = values.shape[:-1]
    used = np.zeros(values.shape, dtype=bool)
    total = np.zeros(shape, dtype=np.float64)
    starters = np.full((*shape, elig.shape[-1]), -1, dtype=np.intp)
    members = np.arange(values.shape[-1])
    for s in order:
        v = np.where(elig[..., s] & ~used, values, -np.inf)
        j = np.argmax(v, axis=-1)
        best = np.take_along_axis(v, j[..., None], axis=-1)[..., 0]
        ok = best >= -1e-6  # like best_lineup, an empty slot beats a negative projection
        total += np.where(ok, best, 0.0)
        starters[..., s] = np.where(ok, j, -1)
        used |= (members == j[..., None]) & ok[..., None]
    return total, starters


def exact_lineups(values: np.ndarray, elig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same as ``greedy_lineups`` but solved exactly, one assignment per roster."""
    shape = values.shape[:-1]
    R, S = elig.shape[-2:]
    flat_v = values.reshape(-1, R)
    flat_e = elig.reshape(-1, R, S)
    total = np.zeros(len(flat_v))
    starters = np.full((len(flat_v), S), -1, dtype=np.intp)
    big = -1e9
    for n in range(len(flat_v)):
        v = flat_v[n]
        ok = flat_e[n] & np.isfinite(v)[:, None]
        if not ok.any():
            continue
        score = np.where(ok.T, np.nan_to_num(v, neginf=0.0)[None, :], big)  # (S, R)
        score = np.concatenate([score, np.full((S, S), -1e-6)], axis=1)  # empty-slot dummies
        rows, cols = linear_sum_assignment(score, maximize=True)
        for i, j in zip(rows, cols, strict=True):
            if j < R and score[i, j] > big and v[j] >= -1e-6:
                starters[n, i] = j
                total[n] += v[j]
    return total.reshape(shape), starters.reshape(*shape, S)


def lineups(
    values: np.ndarray, elig: np.ndarray, spec: SlotSpec, exact: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    if exact and not spec.exact:
        return exact_lineups(values, elig)
    return greedy_lineups(values, elig, spec.order)
