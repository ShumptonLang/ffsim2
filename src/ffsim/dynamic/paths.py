"""Outcome draws for closed-loop paths, with projection news revealed over time.

Points and availability come from the static engine's ``draw_batch``, so with
no projection news a path's outcomes are exactly the static engine's. Projection
news (``RevealModel``) is drawn first, from its own random stream: for week w
an increment with variance s2(k) - s2(k - 1) is revealed when the lead (weeks
before the game) drops below k, so s2(lead) is always what is still unknown. The week's true
projection, used to draw its points, includes all of them; a manager deciding
at lead k sees only those already revealed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ffsim.model.models import ModelSet
from ffsim.sim.engine import DrawSetup, draw_batch
from ffsim.sim.inputs import SimInputs


@dataclass
class PathDraws:
    pts: np.ndarray  # (B, W, P) fantasy points
    avail: np.ndarray  # (B, W, P) plays that week
    z_obs: np.ndarray  # (B, W, P) latent value a manager can infer from the score
    absence: np.ndarray  # (B, W, P) int8 observable absence state after each week
    news: np.ndarray | None  # (B, W, P, K + 1) [..., k] = news still unrevealed at lead k
    var_total: np.ndarray | None  # (W, P) news variance at the start
    var_lead: np.ndarray | None  # (W, W, P) [t, w] news variance still unknown at decision t

    @property
    def n(self) -> int:
        return self.pts.shape[0]

    def known_log(self, t: int) -> np.ndarray:
        """log(projection as seen at decision t / starting projection), (B, W, P).

        Mean-preserving: E[true projection | seen at t] = starting x exp(this).
        """
        B, W, P = self.pts.shape
        if self.news is None:
            return np.zeros((B, W, P), dtype=np.float32)
        assert self.var_total is not None and self.var_lead is not None
        K = self.news.shape[-1] - 1
        lead = np.clip(np.arange(W) - t, 0, K)  # weeks are consecutive simulation steps
        unseen = np.take_along_axis(
            self.news, np.broadcast_to(lead[None, :, None, None], (B, W, P, 1)), axis=-1
        )[..., 0]
        seen = self.news[..., K] - unseen
        return seen - self.var_total / 2 + self.var_lead[t] / 2


def news_variances(
    inputs: SimInputs, models: ModelSet, players: list[str]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per (lead level, week, player) increment variances, and the start variance."""
    rev = models.reveal
    K = rev.max_lead
    W, P = len(inputs.weeks), len(players)
    pos = [(inputs.positions.get(p) or (None,))[0] for p in players]
    published = [bool(inputs.nfl_games.get(w)) for w in inputs.weeks]
    # var_at[l, w, p]: variance still unknown at lead l (l = 0..K, capped).
    var_at = np.zeros((K + 1, W, P))
    for wi in range(W):
        for j, ps in enumerate(pos):
            for lead in range(1, K + 1):
                var_at[lead, wi, j] = rev.var(ps, lead, published[wi])
    return var_at, var_at[np.clip(np.arange(W), 0, K), np.arange(W)], K


def draw_paths(
    setup: DrawSetup,
    inputs: SimInputs,
    models: ModelSet,
    players: list[str],
    rng: np.random.Generator,
    rng_news: np.random.Generator,
    n: int,
    news: bool = True,
) -> PathDraws:
    var_at, var_total, K = news_variances(inputs, models, players) if news else (None, None, 0)
    W, P = setup.means.shape
    cum = None
    var_lead = None
    log_true = None
    if var_at is not None and K > 0 and var_total is not None and var_total.any():
        lead0 = np.clip(np.arange(W), 0, K)  # lead of week w at the first decision
        inc_var = np.zeros((K, W, P))
        for level in range(1, K + 1):
            inc_var[level - 1] = np.where(
                level <= lead0[:, None], var_at[level] - var_at[level - 1], 0.0
            )
        inc = rng_news.standard_normal((n, W, P, K)).astype(np.float32)
        inc *= np.sqrt(np.maximum(inc_var, 0.0)).transpose(1, 2, 0)[None].astype(np.float32)
        # cum[..., l] = sum of increments at levels <= l (still unknown at lead l).
        cum = np.zeros((n, W, P, K + 1), dtype=np.float32)
        cum[..., 1:] = np.cumsum(inc, axis=-1)
        var_lead = np.zeros((W, W, P))
        for t in range(W):
            lead = np.clip(np.arange(W) - t, 0, K)
            var_lead[t] = var_at[lead, np.arange(W)]
        log_true = cum[..., K] - var_total / 2
    batch = draw_batch(setup, models, rng, n, log_reveal=log_true, keep_state=True)
    assert batch.z is not None and batch.absence is not None
    true_means = setup.means[None] * (np.exp(log_true) if log_true is not None else 1.0)
    z_obs = models.errors.observable_latent(batch.z, true_means, setup.positions)
    return PathDraws(
        pts=batch.pts,
        avail=batch.avail,
        z_obs=z_obs,
        absence=batch.absence,
        news=cum,
        var_total=var_total if cum is not None else None,
        var_lead=var_lead,
    )
