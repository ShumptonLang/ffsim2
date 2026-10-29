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
from collections.abc import Iterable, Mapping, Sequence
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
# Grids for expected outcomes under a shifted/scaled latent normal (beliefs).
SHIFT_GRID = np.linspace(-3.0, 3.0, 121)
SCALE_GRID = np.linspace(0.3, 1.0, 15)
_DS = SHIFT_GRID[1] - SHIFT_GRID[0]
_DC = SCALE_GRID[1] - SCALE_GRID[0]

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
        self._shifted: np.ndarray | None = None
        self._runs: tuple[np.ndarray, np.ndarray] | None = None

    def shifted_means(self) -> np.ndarray:
        """E[table(shift + scale * Z)] per bucket on (SHIFT_GRID, SCALE_GRID).

        Used for beliefs: when a manager's posterior on a player's latent value
        is N(shift, scale^2), his expected outcome is projection x this (ratio)
        or projection + this (additive).
        """
        if self._shifted is None:
            # Same quadrature the tables were normalized with, so (0, 1) gives 1 / 0.
            nodes, weights = Z_GRID, _W
            z = SHIFT_GRID[:, None, None] + SCALE_GRID[None, :, None] * nodes[None, None, :]
            out = np.empty((len(self._tables), len(SHIFT_GRID), len(SCALE_GRID)))
            for b, t in enumerate(self._tables):
                out[b] = np.interp(z, Z_GRID, t) @ weights
            self._shifted = out
        return self._shifted

    def flat_runs(self) -> tuple[np.ndarray, np.ndarray]:
        """Per bucket and grid interval, the grid extent of the flat stretch it lies in.

        (-1, -1) where the quantile function is strictly increasing. A score in a
        flat stretch (e.g. the dud mass) only says the latent value is somewhere
        in that stretch.
        """
        if self._runs is None:
            n_b, n_z = self._tables.shape
            lo = np.full((n_b, n_z - 1), -1, dtype=np.intp)
            hi = np.full((n_b, n_z - 1), -1, dtype=np.intp)
            for b, t in enumerate(self._tables):
                flat = np.diff(t) == 0
                k = 0
                while k < n_z - 1:
                    if not flat[k]:
                        k += 1
                        continue
                    end = k
                    while end + 1 < n_z - 1 and flat[end + 1]:
                        end += 1
                    lo[b, k : end + 1] = k
                    hi[b, k : end + 1] = end + 1
                    k = end + 1
            self._runs = (lo, hi)
        return self._runs

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
        z = np.broadcast_to(z, out.shape)
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
        """Latent z whose quantile equals ``points`` (inverse of ``sample``)."""
        return float(self.to_latent_many(np.array([points]), np.array([mean]), [position])[0])

    def to_latent_many(
        self, points: np.ndarray, means: np.ndarray, positions: Sequence[str | None]
    ) -> np.ndarray:
        """Vectorized inverse of ``sample`` for many player-weeks.

        Where the quantile function is flat (e.g. the dud mass), the midpoint of
        the flat stretch is returned.
        """
        points = np.asarray(points, dtype=np.float64)
        out = np.empty(len(points))
        chunk = 4096
        for start in range(0, len(points), chunk):
            sl = slice(start, start + chunk)
            q = self.sample(Z_GRID[:, None], means[None, sl], positions[sl]).astype(np.float64)
            for k, (col, x) in enumerate(zip(q.T, points[sl], strict=True)):
                lo = int(np.searchsorted(col, x, side="left"))
                hi = int(np.searchsorted(col, x, side="right"))
                if hi > lo:  # exact hit on one or more grid values
                    z = (Z_GRID[lo] + Z_GRID[hi - 1]) / 2
                elif lo == 0:
                    z = Z_GRID[0]
                elif lo == Z_POINTS:
                    z = Z_GRID[-1]
                else:
                    f = (x - col[lo - 1]) / (col[lo] - col[lo - 1])
                    z = Z_GRID[lo - 1] + f * (Z_GRID[lo] - Z_GRID[lo - 1])
                out[start + k] = z
        return out

    def expected_points(
        self,
        means: np.ndarray,
        shift: np.ndarray,
        scale: np.ndarray,
        positions: Sequence[str | None],
    ) -> np.ndarray:
        """E[points] when the latent value is N(shift, scale^2) instead of N(0, 1).

        All arrays broadcast to (..., P); ``positions`` has length P. With
        shift 0 and scale 1 this is the projection itself.
        """
        means = np.asarray(means, dtype=np.float64)
        shape = np.broadcast_shapes(means.shape, np.shape(shift), np.shape(scale))
        out = np.zeros(shape, dtype=np.float32)
        means = np.broadcast_to(means, shape)
        sf = np.clip((np.broadcast_to(shift, shape) - SHIFT_GRID[0]) / _DS, 0, len(SHIFT_GRID) - 1)
        cf = np.clip((np.broadcast_to(scale, shape) - SCALE_GRID[0]) / _DC, 0, len(SCALE_GRID) - 1)
        si = np.minimum(sf.astype(np.intp), len(SHIFT_GRID) - 2)
        ci = np.minimum(cf.astype(np.intp), len(SCALE_GRID) - 2)
        sw, cw = sf - si, cf - ci
        by_pos: dict[str | None, list[int]] = defaultdict(list)
        for j, pos in enumerate(positions):
            by_pos[pos].append(j)
        for pos, cols in by_pos.items():
            pm = self._position_model(pos)
            g = pm.shifted_means()
            m = means[..., cols]
            i0, i1, w = pm.blend(m)
            s0, s1, c0, c1 = si[..., cols], si[..., cols] + 1, ci[..., cols], ci[..., cols] + 1
            a, c = sw[..., cols], cw[..., cols]

            def at(b: np.ndarray, g=g, s0=s0, s1=s1, c0=c0, c1=c1, a=a, c=c) -> np.ndarray:
                return (
                    g[b, s0, c0] * (1 - a) * (1 - c)
                    + g[b, s1, c0] * a * (1 - c)
                    + g[b, s0, c1] * (1 - a) * c
                    + g[b, s1, c1] * a * c
                )

            v = at(i0) * (1 - w) + at(i1) * w
            pts = m * v if pm.kind == "ratio" else m + v
            out[..., cols] = np.where(m > 0, pts, 0.0)
        return out

    def observable_latent(
        self, z: np.ndarray, means: np.ndarray, positions: Sequence[str | None]
    ) -> np.ndarray:
        """The latent value a manager can infer from the score ``sample(z, means)``.

        Equal to ``z`` (clipped to the grid) where the quantile function is
        strictly increasing; in a flat stretch (e.g. the dud mass) only the
        stretch is known, and its midpoint is returned, matching ``to_latent_many``.
        """
        zc = np.clip(z, Z_MIN, Z_MAX)
        means = np.broadcast_to(np.asarray(means, dtype=np.float64), zc.shape)
        k = np.minimum(((zc - Z_MIN) / _DZ).astype(np.intp), Z_POINTS - 2)
        out = zc.astype(np.float32, copy=True)
        by_pos: dict[str | None, list[int]] = defaultdict(list)
        for j, pos in enumerate(positions):
            by_pos[pos].append(j)
        for pos, cols in by_pos.items():
            pm = self._position_model(pos)
            run_lo, run_hi = pm.flat_runs()
            i0, i1, w = pm.blend(means[..., cols])
            kk = k[..., cols]
            lo0, hi0, lo1, hi1 = run_lo[i0, kk], run_hi[i0, kk], run_lo[i1, kk], run_hi[i1, kk]
            use0, use1 = w < 1, w > 0
            flat = ((lo0 >= 0) | ~use0) & ((lo1 >= 0) | ~use1)
            lo = np.maximum(np.where(use0, lo0, 0), np.where(use1, lo1, 0))
            hi = np.minimum(np.where(use0, hi0, Z_POINTS - 1), np.where(use1, hi1, Z_POINTS - 1))
            mid = (Z_GRID[np.clip(lo, 0, Z_POINTS - 1)] + Z_GRID[np.clip(hi, 0, Z_POINTS - 1)]) / 2
            out[..., cols] = np.where(flat, mid, out[..., cols])
        return out

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
