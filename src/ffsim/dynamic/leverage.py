"""Title-odds leverage: how much one more point in week w is worth to each team.

Estimated from simulated seasons by finite differences with common random
numbers: add +/- ``delta`` points to one team-week in every path, replay
standings and playoffs, and difference the team's title odds. Noisy per team
for long shots, so each team's week profile is shrunk toward the league's.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ffsim.sim.engine import season_outcomes
from ffsim.sim.inputs import SimInputs

DELTA = 5.0
PROFILE_SHRINK = 0.5
FLOOR = 1e-6  # keeps eliminated teams' managers making sensible moves


def estimate_leverage(
    inputs: SimInputs,
    week_idx: Mapping[int, int],
    team_pts: np.ndarray,
    delta: float = DELTA,
    shrink: float = PROFILE_SHRINK,
) -> np.ndarray:
    """(T, W) d P(title) / d points, per team and simulated week."""
    S, W, T = team_pts.shape
    raw = np.zeros((T, W))
    for wi in range(W):
        for ti in range(T):
            tp = team_pts.copy()
            tp[:, wi, ti] += delta
            up = (season_outcomes(inputs, week_idx, tp).champion == ti).mean()
            tp[:, wi, ti] -= 2 * delta
            down = (season_outcomes(inputs, week_idx, tp).champion == ti).mean()
            raw[ti, wi] = (up - down) / (2 * delta)
    raw = np.maximum(raw, 0.0)
    totals = raw.sum(1, keepdims=True)
    profiles = np.where(totals > 0, raw / np.where(totals > 0, totals, 1.0), 0.0)
    league = profiles.mean(0, keepdims=True)
    if league.sum() == 0:
        league = np.full((1, W), 1.0 / W)
    mixed = (1 - shrink) * profiles + shrink * league
    return np.maximum(totals, FLOOR) * mixed
