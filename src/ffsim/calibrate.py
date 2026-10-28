"""Out-of-sample calibration checks (leave one season out).

For each held-out season the models are fitted on the other seasons and scored
on the held-out one:

* outcome distributions, per position: share of actual scores
  below the 10th/50th/90th percentiles (should be ~10/50/90%) and CRPS (lower
  is better; it rewards both calibration and sharpness);
* Bayesian blend: next-week squared error of Sleeper's projection alone vs. the
  blended centre;
* correlation: observed role-pair correlations in the held-out season vs. the
  model's.

``calibrated_models`` refits on all the checked seasons.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import special

from ffsim.data.history import HistoryRow, HistoryStore, player_seasons
from ffsim.model.bayes import (
    STARTABLE,
    STARTABLE_DEFAULT,
    TalentModel,
    cutoffs,
    fit_talent,
    prior_rate,
)
from ffsim.model.correlation import ROLE_SLOTS, fit_loadings, pair_correlations
from ffsim.model.errors import ErrorModel, fit_error_model
from ffsim.model.models import ModelSet, fit_models

U_LEVELS = np.linspace(0.01, 0.99, 99)
KEY_PAIRS = [
    ("QB", "WR1", True), ("QB", "TE1", True), ("QB", "RB1", True), ("RB1", "RB2", True),
    ("QB", "QB", False), ("DEF", "QB", False), ("DEF", "RB1", False),
]  # fmt: skip


def _group(role: str) -> str:
    return next(p for p, rs in ROLE_SLOTS.items() if role in rs)


def _startable(pos: str, proj: float) -> bool:
    return proj >= STARTABLE.get(pos, STARTABLE_DEFAULT)


@dataclass
class DistScore:
    n: int = 0
    crps: float = 0.0
    below: np.ndarray = field(default_factory=lambda: np.zeros(3))  # share below q10/50/90

    def merge(self, other: DistScore) -> DistScore:
        n = self.n + other.n
        if n == 0:
            return DistScore()
        w1, w2 = self.n / n, other.n / n
        return DistScore(n, w1 * self.crps + w2 * other.crps, w1 * self.below + w2 * other.below)


def score_distributions(
    model: ErrorModel, rows: Iterable[HistoryRow], settings: Mapping[str, float]
) -> dict[str, DistScore]:
    """Held-out CRPS and coverage per position, on startable player-weeks."""
    by_pos: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if not r.played or r.position not in model.positions:
            continue
        proj, act = r.scored(settings)
        if proj > 0 and _startable(r.position, proj):
            by_pos[r.position].append((proj, act))
    z = special.ndtri(U_LEVELS)
    out = {}
    for pos, pairs in by_pos.items():
        arr = np.asarray(pairs)
        m, a = arr[:, 0], arr[:, 1]
        q = model.sample(np.broadcast_to(z[:, None], (len(z), len(m))), m[None, :], [pos] * len(m))
        diff = a[None, :] - q
        pinball = np.maximum(U_LEVELS[:, None] * diff, (U_LEVELS[:, None] - 1) * diff)
        crps = float(2 * pinball.mean())
        idx = [9, 49, 89]  # u = 0.10, 0.50, 0.90
        below = (a[None, :] < q[idx]).mean(axis=1)
        out[pos] = DistScore(len(m), crps, below)
    return out


def _priors(store: HistoryStore, seasons: Iterable[int], settings: Mapping[str, float]) -> dict:
    priors = {}
    for season in seasons:
        for pid, (pos, stats) in store.preseason(season).items():
            rate = prior_rate(stats, settings, pos)
            if rate is not None:
                priors[(season, pid)] = rate
    return priors


def blend_mse(talent: TalentModel, cuts: list) -> dict[str, tuple[int, float, float]]:
    """pos -> (n, MSE of Sleeper's next-week projection, MSE of the blended centre)."""
    acc: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for c in cuts:
        rate = talent.rate(c.pos, c.prior, c.obs, c.n)
        centre = talent.centre(c.pos, c.proj_next, rate)
        acc[c.pos].append(((c.act_next - c.proj_next) ** 2, (c.act_next - centre) ** 2))
    return {p: (len(v), float(np.mean([x for x, _ in v])), float(np.mean([y for _, y in v])))
            for p, v in acc.items()}  # fmt: skip


@dataclass
class CalibrationReport:
    seasons: list[int]
    dist: dict[str, DistScore]  # pos -> score
    blend: dict[str, tuple[int, float, float]]
    corr: list[tuple[str, str, bool, float, float, int]]  # a, b, same, observed, model, n


def leave_one_season_out(
    store: HistoryStore,
    seasons: Sequence[int],
    settings: Mapping[str, float],
) -> CalibrationReport:
    rows_by = {s: list(store.rows([s])) for s in seasons}
    priors = _priors(store, seasons, settings)
    dist: dict[str, DistScore] = {}
    blend: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    corr_acc: dict[tuple[str, str, bool], list[tuple[float, float, int]]] = defaultdict(list)

    for held in seasons:
        train = [r for s in seasons if s != held for r in rows_by[s]]
        test = rows_by[held]
        errors = fit_error_model(train, settings)
        for pos, sc in score_distributions(errors, test, settings).items():
            dist[pos] = dist.get(pos, DistScore()).merge(sc)

        train_seasons, _ = player_seasons(train, settings)
        test_seasons, _ = player_seasons(test, settings)
        talent, _ = fit_talent(train_seasons, priors, errors)
        for pos, v in blend_mse(talent, cutoffs(test_seasons, priors)).items():
            blend[pos].append(v)

        model, _ = fit_loadings(pair_correlations(train, settings, errors), talent.rho)
        observed = pair_correlations(test, settings, errors)
        for a, b, same in KEY_PAIRS:
            key = tuple(sorted((a, b)))
            table = observed.same if same else observed.opp
            if key in table:
                c, n = table[key]  # type: ignore[index]
                # what the simulator produces once talent persistence is layered on
                att = np.sqrt((1 - talent.rho_for(_group(a))) * (1 - talent.rho_for(_group(b))))
                corr_acc[(key[0], key[1], same)].append((c, att * model.implied(a, b, same), n))

    blend_out = {}
    for pos, vals in blend.items():
        n = sum(v[0] for v in vals)
        blend_out[pos] = (
            n,
            sum(v[0] * v[1] for v in vals) / n,
            sum(v[0] * v[2] for v in vals) / n,
        )
    corr_out = []
    for (a, b, same), cvals in corr_acc.items():
        n = sum(v[2] for v in cvals)
        obs = sum(v[0] * v[2] for v in cvals) / n
        mod = sum(v[1] * v[2] for v in cvals) / n
        corr_out.append((a, b, same, obs, mod, n))
    return CalibrationReport(list(seasons), dist, blend_out, corr_out)


def calibrated_models(
    store: HistoryStore,
    seasons: Sequence[int],
    settings: Mapping[str, float],
) -> ModelSet:
    """Fit on all seasons."""
    models, _ = fit_models(store, seasons, settings)
    models.meta["calibrated"] = True
    return models
