"""What in-simulation managers believe about players, path by path.

The engine draws each player's latent value as z_w = sqrt(rho) * eta + sqrt(1 - rho) * eps_w
with one talent factor eta per season. A manager who has seen n games (as the
latent values he can infer from the scores) has the normal posterior

    E[z_future] = rho * n * zbar / (1 + (n - 1) * rho)
    Var[z_future] = 1 - rho + rho / (1 + n * rho / (1 - rho))

and expects ``ErrorModel.expected_points`` under it. Before any simulated game
this is exactly the projection the simulation started from (whose own Bayesian
blend already used the real games played so far), so beliefs only add what the
simulated season reveals. Everyone sees every score, so beliefs are shared by
all managers in a path.
"""

from __future__ import annotations

import numpy as np


class Beliefs:
    def __init__(self, n_paths: int, rho: np.ndarray) -> None:
        self.rho = np.clip(rho.astype(np.float64), 0.0, 0.95)
        P = len(rho)
        self.n = np.zeros((n_paths, P), dtype=np.float64)
        self.zsum = np.zeros((n_paths, P), dtype=np.float64)

    def observe(self, z_obs: np.ndarray, played: np.ndarray) -> None:
        """Add one week of observed latent values (n_paths, P) where ``played``."""
        self.n += played
        self.zsum += np.where(played, z_obs, 0.0)

    def posterior(self) -> tuple[np.ndarray, np.ndarray]:
        """(shift, scale) of each player's future latent value, (n_paths, P)."""
        rho, n = self.rho, self.n
        shift = rho * self.zsum / (1.0 + np.maximum(n - 1.0, 0.0) * rho)
        shift = np.where(n > 0, shift, 0.0)
        var = 1.0 - rho + rho / (1.0 + n * rho / np.maximum(1.0 - rho, 1e-9))
        return shift, np.sqrt(var)

    def take(self, rows: np.ndarray) -> Beliefs:
        out = Beliefs.__new__(Beliefs)
        out.rho, out.n, out.zsum = self.rho, self.n[rows].copy(), self.zsum[rows].copy()
        return out
