"""Projection-based lineup setting.

Managers set lineups from projections, not outcomes, so each team-week gets the
lineup that maximizes projected points subject to slot eligibility. That is a
bipartite assignment (slots x players), solved exactly with the Hungarian
algorithm; FLEX/SUPER_FLEX/IDP_FLEX need no special-casing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ffsim.league import SLOT_ELIGIBILITY

_INELIGIBLE = -1e9


def best_lineup(
    slots: Sequence[str],
    candidates: Sequence[str],
    means: Mapping[str, float],
    positions: Mapping[str, Sequence[str]],
) -> list[str | None]:
    """Player id for each starting slot (None if no eligible player is left).

    ``positions`` maps player_id -> fantasy positions. Players with no
    projection (bye, out) count as 0 but can still fill an otherwise empty slot,
    which matches how an empty slot scores.
    """
    if not slots:
        return []
    n_s, n_p = len(slots), len(candidates)
    value = np.full((n_s, n_p + n_s), _INELIGIBLE)
    value[:, n_p:] = -1e-6  # "empty" dummies: worse than any eligible real player
    for j, pid in enumerate(candidates):
        m = float(means.get(pid, 0.0))
        fps = positions.get(pid, ())
        for i, slot in enumerate(slots):
            if any(p in SLOT_ELIGIBILITY.get(slot, ()) for p in fps):
                value[i, j] = m
    rows, cols = linear_sum_assignment(value, maximize=True)
    out: list[str | None] = [None] * n_s
    for i, j in zip(rows, cols, strict=True):
        if j < n_p and value[i, j] > _INELIGIBLE:
            out[i] = candidates[j]
    return out
