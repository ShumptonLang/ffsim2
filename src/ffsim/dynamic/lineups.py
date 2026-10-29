"""Vectorized lineup setting for many rosters at once.

The static engine solves one assignment per (week, team) with the Hungarian
algorithm. Here lineups are needed per path, per team, per week and for every
roster a manager considers, so slots are filled greedily instead: most
restrictive slot first, each taking the best eligible player left. That is
exact when the slots' eligibility sets are nested or disjoint (QB, RB, WR, TE,
FLEX, SUPER_FLEX, IDP groups, IDP_FLEX) and every player has one position; other
layouts use ``exact_lineups`` where exactness matters.

Roster valuation calls this millions of times; with numba installed
(``pip install ffsim[fast]``) a compiled kernel does it, otherwise numpy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ffsim.league import SLOT_ELIGIBILITY

try:  # optional accelerator
    import numba
except ImportError:  # pragma: no cover - exercised when numba is absent
    numba = None  # type: ignore[assignment]


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
    left = np.array(values, dtype=np.float32, order="C")  # used members become -inf
    by_slot = np.ascontiguousarray(np.moveaxis(elig, -1, 0))  # (S, ..., R)
    total = np.zeros(shape, dtype=np.float64)
    starters = np.full((*shape, elig.shape[-1]), -1, dtype=np.intp)
    for s in order:
        v = np.where(by_slot[s], left, -np.inf)
        j = np.argmax(v, axis=-1)[..., None]
        best = np.take_along_axis(v, j, axis=-1)[..., 0]
        ok = best >= -1e-6  # like best_lineup, an empty slot beats a negative projection
        total += np.where(ok, best, 0.0)
        starters[..., s] = np.where(ok, j[..., 0], -1)
        np.put_along_axis(
            left, j, np.where(ok[..., None], -np.inf, np.take_along_axis(left, j, -1)), -1
        )
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


def _greedy_totals_numpy(values: np.ndarray, elig: np.ndarray, order: Sequence[int]) -> np.ndarray:
    M, N, R = values.shape
    total, _ = greedy_lineups(values, elig[:, None], order)
    return total


if numba is not None:

    @numba.njit(cache=True, parallel=True)
    def _greedy_totals_numba(values, elig, order):  # type: ignore[no-untyped-def]
        M, N, R = values.shape
        out = np.zeros((M, N))
        for m in numba.prange(M):
            used = np.zeros(R, dtype=np.bool_)
            for n in range(N):
                used[:] = False
                tot = 0.0
                for s in order:
                    best = -np.inf
                    jb = -1
                    for r in range(R):
                        if elig[m, r, s] and not used[r]:
                            v = values[m, n, r]
                            if v > best:
                                best = v
                                jb = r
                    if jb >= 0 and best >= -1e-6:
                        tot += best
                        used[jb] = True
                out[m, n] = tot
        return out


def greedy_totals(
    values: np.ndarray, elig: np.ndarray, order: Sequence[int], accelerate: bool = True
) -> np.ndarray:
    """Lineup totals only, for ``values`` (M, N, R) sharing ``elig`` (M, R, S) across N."""
    if accelerate and numba is not None:
        return _greedy_totals_numba(
            np.ascontiguousarray(values, dtype=np.float32),
            np.ascontiguousarray(elig),
            np.asarray(order, dtype=np.int64),
        )
    return _greedy_totals_numpy(values, elig, order)
