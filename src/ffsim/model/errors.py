"""Projection-error model: turns a projected mean into a full outcome distribution.

Fantasy scoring is right-skewed with a lump of near-zero games, so a normal curve
is a poor fit. For every position the model stores *quantile functions* of a
normalized outcome, tabulated on a grid of standard-normal latent values z:

* ratio positions (QB RB WR TE DL LB DB): outcome = projection x ratio(z)
* additive positions (K DEF, which can go negative): outcome = projection + resid(z)

Tables are fitted per projection bucket (low-projection players are much more
volatile, relative to their mean) and blended linearly between bucket centers,
so the shape changes smoothly with the projection. Every table is normalized so
the distribution mean equals the projection; shifting the centre is the job of
the Bayesian layer.

Two families build the tables:

* ``parametric``: dud mass + lognormal for ratio positions; shifted gamma for K/DEF
* ``bootstrap``: empirical quantiles of historical ratios/residuals

Sampling takes latent normals ``z`` (independent or correlated) and maps them
through the tables, which is what lets the copula layer reuse this model.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import special, stats

from ffsim.data.history import HistoryRow

RATIO_POSITIONS = ("QB", "RB", "WR", "TE", "DL", "LB", "DB")
ADDITIVE_POSITIONS = ("K", "DEF")
FAMILIES = ("parametric", "bootstrap")

Z_MIN, Z_MAX, Z_POINTS = -4.0, 4.0, 401
Z_GRID = np.linspace(Z_MIN, Z_MAX, Z_POINTS)
_DZ = (Z_MAX - Z_MIN) / (Z_POINTS - 1)
# Integration weights for E[f(Z)] on the grid (trapezoid on the normal density).
_W = stats.norm.pdf(Z_GRID) * _DZ
_W[[0, -1]] *= 0.5
_W /= _W.sum()
U_GRID = special.ndtr(Z_GRID)

DUD_RATIO = 0.15  # ratio below which a game counts as a "dud" (early exit, no role)
MIN_RATIO_PROJ = 1.0  # ratio positions: ignore history rows projected below this
N_BUCKETS = {"ratio": 4, "additive": 3}
MIN_BUCKET_ROWS = 60


@dataclass
class PositionModel:
    kind: str  # "ratio" | "additive"
    centers: list[float]  # median projection of each bucket (ascending)
    tables: list[list[float]]  # one quantile table per bucket, on Z_GRID
    diagnostics: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._centers = np.asarray(self.centers, dtype=np.float64)
        self._tables = np.asarray(self.tables, dtype=np.float64)

    def blend(self, means: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bucket indices (i0, i1) and weight w for each projected mean."""
        c = self._centers
        if len(c) == 1:
            z = np.zeros(means.shape, dtype=np.intp)
            return z, z, np.zeros(means.shape)
        i1 = np.clip(np.searchsorted(c, means), 1, len(c) - 1)
        i0 = i1 - 1
        gap = c[i1] - c[i0]
        w = np.clip((means - c[i0]) / np.where(gap > 0, gap, 1.0), 0.0, 1.0)
        return i0, i1, w


@dataclass
class ErrorModel:
    family: str
    positions: dict[str, PositionModel]
    meta: dict[str, object] = field(default_factory=dict)

    # -- persistence ---------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {
                "family": self.family,
                "meta": self.meta,
                "positions": {k: asdict(v) for k, v in self.positions.items()},
            }
        )

    @classmethod
    def from_json(cls, text: str) -> ErrorModel:
        d = json.loads(text)
        return cls(
            family=d["family"],
            meta=d.get("meta", {}),
            positions={k: PositionModel(**v) for k, v in d["positions"].items()},
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> ErrorModel:
        return cls.from_json(path.read_text())

    @classmethod
    def default(cls) -> ErrorModel:
        """Model fitted on 2023-2025 Sleeper projections vs. actuals (PPR scoring)."""
        from ffsim.model.models import ModelSet

        return ModelSet.default().errors

    # -- sampling ----------------------------------------------------------------

    def _position_model(self, pos: str | None) -> PositionModel:
        if pos in self.positions:
            return self.positions[pos]  # type: ignore[index]
        # Unknown/rare positions (e.g. FB) borrow the RB shape.
        return self.positions.get("RB") or next(iter(self.positions.values()))

    def sample(
        self, z: np.ndarray, means: np.ndarray, positions: Iterable[str | None]
    ) -> np.ndarray:
        """Map latent normals to fantasy points.

        ``z`` has shape (..., P) and ``means`` broadcasts against it; positions
        has length P. Players with a mean of 0 (bye, out) score exactly 0.
        """
        positions = list(positions)
        means = np.asarray(means, dtype=np.float64)
        out = np.zeros(np.broadcast_shapes(z.shape, means.shape), dtype=np.float32)
        zf = np.clip((z - Z_MIN) / _DZ, 0.0, Z_POINTS - 1)
        k = np.minimum(zf.astype(np.intp), Z_POINTS - 2)
        frac = zf - k

        by_pos: dict[str | None, list[int]] = defaultdict(list)
        for j, pos in enumerate(positions):
            by_pos[pos].append(j)
        for pos, cols in by_pos.items():
            pm = self._position_model(pos)
            m = np.broadcast_to(means, out.shape)[..., cols]
            i0, i1, w = pm.blend(m)
            kk, ff = k[..., cols], frac[..., cols]
            t = pm._tables
            v0 = t[i0, kk] * (1 - ff) + t[i0, kk + 1] * ff
            v1 = t[i1, kk] * (1 - ff) + t[i1, kk + 1] * ff
            v = v0 * (1 - w) + v1 * w
            pts = m * v if pm.kind == "ratio" else m + v
            out[..., cols] = np.where(m > 0, pts, 0.0)
        return out

    def quantile(self, u: np.ndarray | float, mean: float, position: str) -> np.ndarray:
        """Inverse CDF for a single player-week (for reports and tests)."""
        z = special.ndtri(np.clip(np.asarray(u, dtype=np.float64), 1e-12, 1 - 1e-12))
        return self.sample(np.atleast_1d(z)[:, None], np.array([mean]), [position])[:, 0]

    def to_latent(self, points: float, mean: float, position: str) -> float:
        """Latent z whose quantile equals ``points`` (inverse of ``sample``).

        Where the quantile function is flat (e.g. the dud mass), the midpoint of
        the flat stretch is returned.
        """
        q = self.sample(Z_GRID[:, None], np.array([mean]), [position])[:, 0].astype(np.float64)
        lo = int(np.searchsorted(q, points, side="left"))
        hi = int(np.searchsorted(q, points, side="right"))
        if hi > lo:  # exact hit on one or more grid values
            return float((Z_GRID[lo] + Z_GRID[hi - 1]) / 2)
        if lo == 0:
            return float(Z_GRID[0])
        if lo == Z_POINTS:
            return float(Z_GRID[-1])
        f = (points - q[lo - 1]) / (q[lo] - q[lo - 1])
        return float(Z_GRID[lo - 1] + f * (Z_GRID[lo] - Z_GRID[lo - 1]))

    def mean(self, mean: float, position: str) -> float:
        """Distribution mean evaluated on the grid (equals ``mean`` by construction)."""
        vals = self.sample(Z_GRID[:, None], np.array([mean]), [position])[:, 0]
        return float(np.dot(_W, vals))


# --- fitting ---------------------------------------------------------------------


def _normalize(table: np.ndarray, kind: str) -> np.ndarray:
    table = np.maximum.accumulate(table)  # enforce a monotone quantile function
    mu = float(np.dot(_W, table))
    return table / mu if kind == "ratio" else table - mu


def _ratio_parametric(r: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    dud = r < DUD_RATIO
    p = float(dud.mean())
    d = float(r[dud].mean()) if dud.any() else 0.0
    logs = np.log(r[~dud])
    sigma = float(logs.std())
    mu = np.log((1 - p * d) / (1 - p)) - sigma**2 / 2
    u_body = np.clip((U_GRID - p) / (1 - p), 1e-12, 1 - 1e-12)
    table = np.where(p > U_GRID, d, np.exp(mu + sigma * special.ndtri(u_body)))
    return table, {"p_dud": p, "dud_ratio": d, "sigma_log": sigma}


def _additive_parametric(
    actual: np.ndarray, resid: np.ndarray, center: float
) -> tuple[np.ndarray, dict[str, float]]:
    s = float(resid.std())
    floor = float(np.percentile(actual, 0.5)) - 0.5
    span = max(center - floor, 0.5)
    k = (span / s) ** 2
    theta = s**2 / span
    table = floor + special.gammaincinv(k, U_GRID) * theta - center
    return table, {"sd": s, "floor": floor, "gamma_shape": k}


def fit_position(proj: np.ndarray, actual: np.ndarray, kind: str, family: str) -> PositionModel:
    if kind == "ratio":
        keep = proj >= MIN_RATIO_PROJ
        proj, actual = proj[keep], actual[keep]
    n_b = max(1, min(N_BUCKETS[kind], len(proj) // MIN_BUCKET_ROWS))
    # Equal-count buckets by projection rank (robust to many identical projections).
    chunks = np.array_split(np.argsort(proj, kind="stable"), n_b)

    centers, tables, diags = [], [], []
    for idx in chunks:
        m, a = proj[idx], actual[idx]
        center = float(np.median(m))
        diag: dict[str, float] = {
            "n": float(len(m)),
            "lo": float(m.min()),
            "hi": float(m.max()),
            "center": center,
        }
        if kind == "ratio":
            r = a / m
            diag["mean_ratio"] = float(r.mean())
            if family == "parametric":
                table, extra = _ratio_parametric(r)
            else:
                table, extra = np.quantile(r, U_GRID), {}
        else:
            e = a - m
            diag["mean_resid"] = float(e.mean())
            if family == "parametric":
                table, extra = _additive_parametric(a, e - e.mean(), center)
            else:
                table, extra = np.quantile(e, U_GRID), {}
        diag.update(extra)
        centers.append(center)
        tables.append(_normalize(np.asarray(table, dtype=np.float64), kind).round(5).tolist())
        diags.append({k: round(v, 4) for k, v in diag.items()})
    return PositionModel(kind=kind, centers=centers, tables=tables, diagnostics=diags)


def fit_error_model(
    rows: Iterable[HistoryRow],
    settings: Mapping[str, float],
    family: str = "parametric",
    meta: Mapping[str, object] | None = None,
) -> ErrorModel:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if not row.played:
            continue
        m, a = row.scored(settings)
        if m > 0:
            pairs[row.position].append((m, a))

    positions: dict[str, PositionModel] = {}
    for pos in RATIO_POSITIONS + ADDITIVE_POSITIONS:
        data = np.asarray(pairs.get(pos, []), dtype=np.float64)
        if len(data) < MIN_BUCKET_ROWS:
            continue
        kind = "ratio" if pos in RATIO_POSITIONS else "additive"
        positions[pos] = fit_position(data[:, 0], data[:, 1], kind, family)
    return ErrorModel(family=family, positions=positions, meta=dict(meta or {}))


def coverage(
    model: ErrorModel, rows: Iterable[HistoryRow], settings: Mapping[str, float]
) -> dict[str, dict[str, float]]:
    """Share of actual outcomes below the model's 10th/50th/90th percentiles, by position.

    A calibrated model gives ~0.10 / 0.50 / 0.90.
    """
    acc: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row.played:
            m, a = row.scored(settings)
            if m > 0 and row.position in model.positions:
                acc[row.position].append((m, a))
    z = special.ndtri(np.array([0.1, 0.5, 0.9]))
    out: dict[str, dict[str, float]] = {}
    for pos, pairs in acc.items():
        arr = np.asarray(pairs)
        q = model.sample(np.broadcast_to(z, (len(arr), 3)).T, arr[:, 0][None, :], [pos] * len(arr))
        below = (arr[:, 1][None, :] < q).mean(axis=1)
        out[pos] = {"n": float(len(arr)), "q10": below[0], "q50": below[1], "q90": below[2]}
    return out
