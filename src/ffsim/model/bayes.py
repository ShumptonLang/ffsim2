"""Bayesian projection updating and rest-of-season talent persistence.

Centre. Each player's per-game scoring rate gets a normal-normal update: the
preseason projection is a prior worth ``k`` games, so

    rate = (k * prior + n * observed_mean) / (k + n)

and early-season results are shrunk instead of taken at face value. Sleeper's
weekly projections already react to news, so the week-w centre blends the two:

    centre_w = (1 - lam) * sleeper_w + lam * rate      (0 when sleeper_w is 0)

``lam`` and ``k`` are fitted per position on how well the blend predicts the
*next* week's actual score. In 2023-2025 data, Sleeper's offensive projections
leave little to add (lam ~ 0.1) while IDP projections improve a lot (lam ~ 0.5).

Persistence. A player's misses against projection are correlated across the
rest of the season (role changes, true talent the projection hasn't caught).
In each simulated season every player draws one latent talent factor shared by
all remaining weeks: z_w = sqrt(rho) * eta + sqrt(1 - rho) * eps_w. Marginal
weekly distributions are unchanged; only their co-movement is. ``rho`` is the
fitted correlation of a player's latent residuals across future weeks, measured
against a centre fixed at the cutoff week (as the simulation's centres are).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ffsim.data.history import PlayerSeason
from ffsim.model.errors import ErrorModel
from ffsim.scoring import score

GAMES_PER_SEASON = 17
_TIER_FLAG = re.compile(r"^(pts_allow|yds_allow)_\d")
# Players below this projection are rarely started; they're left out of fitting.
STARTABLE = {"DL": 3.0, "LB": 3.0, "DB": 3.0}
STARTABLE_DEFAULT = 5.0
LAMBDA_GRID = np.round(np.arange(0.0, 0.751, 0.05), 2)
K_GRID = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
MIN_FIT_ROWS = 300
PERSIST_HORIZON = 6


@dataclass
class TalentModel:
    blend: dict[str, tuple[float, float]] = field(default_factory=dict)  # pos -> (lam, k)
    rho: dict[str, float] = field(default_factory=dict)  # pos -> persistence
    default_blend: tuple[float, float] = (0.1, 4.0)
    default_rho: float = 0.15

    def lam_k(self, pos: str | None) -> tuple[float, float]:
        lam, k = self.blend.get(pos or "", self.default_blend)
        return float(lam), float(k)

    def rho_for(self, pos: str | None) -> float:
        return float(self.rho.get(pos or "", self.default_rho))

    def rate(self, pos: str | None, prior: float | None, obs_mean: float, n: int) -> float | None:
        """Posterior per-game rate; None when there's no prior and no games."""
        _, k = self.lam_k(pos)
        if prior is None:
            return obs_mean if n else None
        return (k * prior + n * obs_mean) / (k + n)

    def centre(self, pos: str | None, sleeper: float, rate: float | None) -> float:
        if sleeper <= 0 or rate is None:
            return max(sleeper, 0.0)
        lam, _ = self.lam_k(pos)
        return max((1 - lam) * sleeper + lam * rate, 0.0)

    def to_dict(self) -> dict:
        return {
            "blend": {p: list(v) for p, v in self.blend.items()},
            "rho": self.rho,
            "default_blend": list(self.default_blend),
            "default_rho": self.default_rho,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> TalentModel:
        return cls(
            blend={p: (float(v[0]), float(v[1])) for p, v in d.get("blend", {}).items()},
            rho={p: float(v) for p, v in d.get("rho", {}).items()},
            default_blend=tuple(d.get("default_blend", (0.1, 4.0))),  # type: ignore[arg-type]
            default_rho=float(d.get("default_rho", 0.15)),
        )


def prior_rate(
    season_stats: Mapping[str, float], settings: Mapping[str, float], position: str | None
) -> float | None:
    """Per-game prior from a full-season projection.

    The season line is scaled to an average game *before* scoring, so per-game
    thresholds (300-yard bonuses, points-allowed tiers) are judged per game.
    Sleeper's season lines are irregular: players list gp=18, and team defenses
    list gp=1 with stray tier flags (e.g. ``pts_allow_0``) instead of points
    allowed. Games outside 10-17 are treated as a full season, and season-level
    tier flags are dropped (tiers are re-derived from per-game points allowed).
    """
    gp = season_stats.get("gp") or GAMES_PER_SEASON
    g = gp if 10 <= gp <= GAMES_PER_SEASON else GAMES_PER_SEASON
    per_game = {k: v / g for k, v in season_stats.items() if k != "gp" and not _TIER_FLAG.match(k)}
    rate = score(per_game, settings, position)
    return rate if rate > 0 else None


def _startable(pos: str, proj: float) -> bool:
    return proj >= STARTABLE.get(pos, STARTABLE_DEFAULT)


@dataclass
class _Cutoff:
    pos: str
    prior: float
    n: int
    obs: float
    proj_next: float
    act_next: float
    later: list[tuple[float, float]]  # (sleeper proj, actual) for weeks c+1..c+H played


def cutoffs(
    seasons: Mapping[tuple[int, str], PlayerSeason], priors: Mapping[tuple[int, str], float]
) -> list[_Cutoff]:
    out = []
    for key, ps in seasons.items():
        prior = priors.get(key)
        if prior is None:
            continue
        wk = ps.weeks
        for c in range(1, 17):
            nxt = wk.get(c + 1)
            if not nxt or not nxt[2] or not _startable(ps.position, nxt[0]):
                continue
            obs = [a for w, (_, a, played) in wk.items() if w <= c and played]
            later = [
                (wk[w][0], wk[w][1])
                for w in range(c + 1, c + 1 + PERSIST_HORIZON)
                if w in wk and wk[w][2]
            ]
            out.append(
                _Cutoff(
                    ps.position,
                    prior,
                    len(obs),
                    float(np.mean(obs)) if obs else 0.0,
                    nxt[0],
                    nxt[1],
                    later,
                )  # fmt: skip
            )
    return out


def _fit_blend(cuts: list[_Cutoff]) -> tuple[float, float, float, float]:
    """Grid-search (lam, k) minimizing next-week squared error.

    Returns (lam, k, mse_sleeper_only, mse_best).
    """
    a = np.array([(c.prior, c.n, c.obs, c.proj_next, c.act_next) for c in cuts])
    prior, n, obs, proj, act = a.T
    base = float(np.mean((act - proj) ** 2))
    best = (0.0, K_GRID[0], base)
    for k in K_GRID:
        rate = (k * prior + n * obs) / (k + n)
        for lam in LAMBDA_GRID:
            mse = float(np.mean((act - ((1 - lam) * proj + lam * rate)) ** 2))
            if mse < best[2] - 1e-12:
                best = (float(lam), float(k), mse)
    return best[0], best[1], base, best[2]


def _fit_rho(cuts: list[_Cutoff], talent: TalentModel, errors: ErrorModel) -> float | None:
    """Pooled correlation of latent residuals across future weeks of one cutoff."""
    keep, centres = [], []
    for c in cuts:
        if len(c.later) < 3:
            continue
        centre = talent.centre(c.pos, c.proj_next, talent.rate(c.pos, c.prior, c.obs, c.n))
        if centre > 0:
            keep.append(c)
            centres.append(centre)
    acts = np.array([a for c in keep for _, a in c.later])
    means = np.array([m for c, m in zip(keep, centres, strict=True) for _ in c.later])
    pos = [c.pos for c in keep for _ in c.later]
    z = errors.to_latent_many(acts, means, pos) if len(acts) else np.array([])
    groups, start = [], 0
    for c in keep:
        groups.append(z[start : start + len(c.later)])
        start += len(c.later)
    if len(groups) < 50:
        return None
    allz = np.concatenate(groups)
    mu, var = allz.mean(), allz.var()
    num = den = 0.0
    for g in groups:
        d = g - mu
        num += d.sum() ** 2 - (d * d).sum()
        den += len(d) * (len(d) - 1)
    return float(np.clip(num / den / var, 0.0, 0.8))


def fit_talent(
    seasons: Mapping[tuple[int, str], PlayerSeason],
    priors: Mapping[tuple[int, str], float],
    errors: ErrorModel,
) -> tuple[TalentModel, dict[str, dict[str, float]]]:
    """Fit blend weights and persistence per position. Returns (model, diagnostics)."""
    cuts = cutoffs(seasons, priors)
    by_pos: dict[str, list[_Cutoff]] = defaultdict(list)
    for c in cuts:
        by_pos[c.pos].append(c)

    model = TalentModel()
    diags: dict[str, dict[str, float]] = {}
    if cuts:
        lam, k, _, _ = _fit_blend(cuts)
        model.default_blend = (lam, k)
    for pos, cs in sorted(by_pos.items()):
        if len(cs) < MIN_FIT_ROWS:
            continue
        lam, k, base, best = _fit_blend(cs)
        model.blend[pos] = (lam, k)
        diags[pos] = {
            "n": float(len(cs)),
            "lam": lam,
            "k": k,
            "mse_sleeper": base,
            "mse_blend": best,
        }

    rhos = []
    for pos, cs in sorted(by_pos.items()):
        rho = _fit_rho(cs, model, errors)
        if rho is not None:
            model.rho[pos] = round(rho, 4)
            rhos.append(rho)
            diags.setdefault(pos, {"n": float(len(cs))})["rho"] = rho
    if rhos:
        model.default_rho = round(float(np.median(rhos)), 4)
    return model, diags
