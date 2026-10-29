"""What a manager expects a roster to be worth, given what he can observe.

A roster's value at decision step t is

    V = sum over weeks w >= t of  L[team, w] * E[lineup points in week w]

where L is the team's title-odds leverage per point in each week (estimated from
the simulation itself) and the expectation is over the manager's beliefs about
each player's scoring and over imagined injury scenarios drawn from the
availability model, conditioned on the injuries he knows about. Imagined
scenarios are what give bench depth, handcuffs and bye cover their value.
"""

from __future__ import annotations

import numpy as np

from ffsim.dynamic.beliefs import Beliefs
from ffsim.dynamic.lineups import SlotSpec, greedy_lineups, greedy_totals
from ffsim.dynamic.paths import PathDraws
from ffsim.model.availability import SEASON_ENDING, SEASON_OUT, AvailabilityModel
from ffsim.model.errors import ErrorModel
from ffsim.sim.engine import DrawSetup


def expected_points(
    draws: PathDraws, beliefs: Beliefs, setup: DrawSetup, errors: ErrorModel, t: int
) -> np.ndarray:
    """(B, W - t, P) expected points for weeks t.. as seen at decision step t."""
    known = draws.known_log(t)[:, t:]
    means = setup.means[None, t:] * np.exp(known)
    shift, scale = beliefs.posterior()
    return errors.expected_points(means, shift[:, None, :], scale[:, None, :], setup.positions)


def imagined_availability(
    model: AvailabilityModel,
    setup: DrawSetup,
    absence: np.ndarray | None,
    t: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(B, k, W - t, P) imagined availability for weeks t.., from what is known at t.

    ``absence`` (B, P) is the observable absence state after week t - 1 (None at
    the first step, when only injury designations are known). Absences of
    unknown length are drawn from the duration distribution conditioned on the
    games already missed.
    """
    W, P = setup.active.shape
    B = 1 if absence is None else absence.shape[0]
    pmf = np.asarray(model.durations, dtype=np.float64)
    pmf = pmf / pmf.sum()
    values = np.array(list(range(1, len(pmf))) + [SEASON_ENDING])
    remaining = np.zeros((B, k, P), dtype=np.int32)
    if absence is not None:
        st = np.broadcast_to(absence[:, None, :], (B, k, P))
        remaining[st == SEASON_OUT] = SEASON_ENDING
        hurt = (st >= 0) & (st != SEASON_OUT)
        if hurt.any():
            missed = st[hurt].astype(np.int64)
            finite = values[:-1]
            # P(d = v | d >= max(missed, 1), not season-ending), sampled by inverse CDF.
            lo = np.maximum(missed, 1)
            w = np.where(finite[None, :] >= lo[:, None], pmf[None, :-1], 0.0)
            tot = w.sum(1, keepdims=True)
            w = np.where(tot > 0, w / np.where(tot > 0, tot, 1), 0.0)
            u = rng.random(len(missed))
            idx = np.minimum((np.cumsum(w, 1) < u[:, None]).sum(1), len(finite) - 1)
            d = np.where(tot[:, 0] > 0, finite[idx], lo)
            remaining[hurt] = np.maximum(d - missed, 0)
    hazard = setup.hazard
    out = np.zeros((B, k, W - t, P), dtype=bool)
    for j, w_ in enumerate(range(t, W)):
        plays = setup.active[w_]
        if absence is None and w_ == 0:
            ok = rng.random((B, k, P)) < setup.first_play
        else:
            gone = (remaining > 0) & plays
            remaining -= gone
            ok = ~gone
        out[:, :, j] = ok & plays
        hit = out[:, :, j] & (remaining == 0) & (rng.random((B, k, P)) < hazard)
        if hit.any():
            remaining[hit] = values[rng.choice(len(values), size=int(hit.sum()), p=pmf)]
    return out


class RosterValuer:
    """Per-week expected lineup points for many candidate rosters at once."""

    def __init__(self, exp_pts: np.ndarray, avail: np.ndarray, spec: SlotSpec) -> None:
        B, Wr, P = exp_pts.shape
        self.B, self.Wr, self.P = B, Wr, P
        # Transposed with a sentinel row P for open roster spots.
        self.e = np.concatenate([exp_pts, np.zeros((B, Wr, 1), np.float32)], -1).transpose(0, 2, 1)
        a = np.concatenate([avail, np.zeros((*avail.shape[:3], 1), bool)], -1)
        self.a = a.transpose(0, 3, 1, 2)  # (B, P + 1, K, Wr)
        self.spec = spec
        self.rows = np.arange(B)
        self._ev: np.ndarray | None = None

    def week_values(
        self, rosters: np.ndarray, rows: np.ndarray | None = None, k: int | None = None
    ) -> np.ndarray:
        """(n, C, Wr) expected lineup points for candidate rosters (n, C, R).

        ``rows`` selects the paths (default all); ``k`` uses only the first k
        imagined scenarios (a cheaper, noisier screen).
        """
        idx = np.where(rosters >= 0, rosters, self.P)
        b = (self.rows if rows is None else rows)[:, None, None]
        e = self.e[b, idx]  # (n, C, R, Wr)
        a = self.a[b, idx, :k]  # (n, C, R, K, Wr)
        vals = np.where(a, e[:, :, :, None, :], -np.inf).transpose(0, 1, 3, 4, 2)
        n, C, K, Wr, R = vals.shape
        total = greedy_totals(
            vals.reshape(n * C, K * Wr, R), self.spec.elig[idx].reshape(n * C, R, -1),
            self.spec.order,
        )  # fmt: skip
        return total.reshape(n, C, K, Wr).mean(axis=2)

    def expected_lineup(self, rosters: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        """(n, T, Wr, S) starter values for rosters (n, T, R) on expected availability."""
        idx = np.where(rosters >= 0, rosters, self.P)
        b = (self.rows if rows is None else rows)[:, None, None]
        e = self.e[b, idx] * self.a[b, idx].mean(axis=3)  # (B, T, R, Wr)
        vals = e.transpose(0, 1, 3, 2)
        vals = np.where(idx[:, :, None, :] < self.P, vals, -np.inf)
        elig = self.spec.elig[idx][:, :, None]
        _, starters = greedy_lineups(vals, elig, self.spec.order)
        sv = np.take_along_axis(vals, np.maximum(starters, 0), axis=-1)
        return np.where(starters >= 0, sv, 0.0)

    def expected_values(self, rows: np.ndarray | None = None) -> np.ndarray:
        """(n, P, Wr) each player's expected points times availability."""
        if self._ev is None:
            self._ev = self.e[:, : self.P] * self.a[:, : self.P].mean(axis=2)
        return self._ev if rows is None else self._ev[rows]
