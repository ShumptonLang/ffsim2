"""Projection reveal: how much of a future week's projection is still unknown.

A manager deciding in week t sees week w's projection only as it stands today.
By game week it will have moved: matchup news, role changes, injuries around the
player. Some of that movement is learning about the player's talent from games
played in between, which the in-simulation beliefs already model; the rest is
news that arrives on its own. This model holds the variance of that remaining
news on the log scale, per position and lead time k = w - t:

    log proj_w(at game week) = log proj_w(as seen at t) + news,  news ~ N(-s2/2, s2)

so the expected projection is unchanged (the static engine is the s2 = 0 case).
Two curves are kept: ``sigma2`` for weeks Sleeper has already projected (the
matchup is priced in) and ``sigma2_unpub`` for weeks not yet projected, where
the simulator carries the latest projection forward and the matchup is news too.

History only stores the projection made for each game week, so news is
measured from the drift between game-week projections k weeks apart, with the
part explained by performance in between removed (V(k), the variogram):

* unpublished weeks: the simulator carries week t's projection forward, so the
  news still to come is exactly that drift: s2_unpub(k) = V(k);
* published weeks: drift is first adjusted for the change of opponent. Week t's
  own one-off component (half of V extrapolated to k = 0) is not news about
  week w, so s2(k) = V(k) - V(0) / 2.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from ffsim.data.history import HistoryRow

MAX_LEAD = 6
MIN_ROWS = 200
OPP_SHRINK = 10.0  # pseudo-games shrinking opponent effects toward zero
# Projections below these are rarely started; they're left out of fitting.
STARTABLE = {"DL": 3.0, "LB": 3.0, "DB": 3.0, "K": 4.0, "DEF": 3.0}
STARTABLE_DEFAULT = 5.0


@dataclass
class RevealModel:
    sigma2: dict[str, list[float]] = field(default_factory=dict)  # pos -> var for lead 1..
    sigma2_unpub: dict[str, list[float]] = field(default_factory=dict)

    def var(self, pos: str | None, lead: int, published: bool = True) -> float:
        """Variance of projection news still to come ``lead`` weeks before the game."""
        if lead <= 0:
            return 0.0
        curves = self.sigma2 if published else self.sigma2_unpub
        curve = curves.get(pos or "") or curves.get("*") or []
        if not curve:
            return 0.0
        return float(curve[min(lead, len(curve)) - 1])

    @property
    def max_lead(self) -> int:
        lens = [len(v) for d in (self.sigma2, self.sigma2_unpub) for v in d.values()]
        return max(lens, default=0)

    def to_dict(self) -> dict:
        return {"sigma2": self.sigma2, "sigma2_unpub": self.sigma2_unpub}

    @classmethod
    def from_dict(cls, d: Mapping) -> RevealModel:
        return cls(
            sigma2={p: [float(x) for x in v] for p, v in d.get("sigma2", {}).items()},
            sigma2_unpub={p: [float(x) for x in v] for p, v in d.get("sigma2_unpub", {}).items()},
        )


def _startable(pos: str, proj: float) -> bool:
    return proj >= STARTABLE.get(pos, STARTABLE_DEFAULT)


def _residual_var(e: np.ndarray, perf: np.ndarray) -> float:
    """Variance of ``e`` left after regressing out performance (OLS with intercept)."""
    x = np.column_stack([np.ones_like(perf), perf])
    beta, *_ = np.linalg.lstsq(x, e, rcond=None)
    return float(np.var(e - x @ beta))


def fit_reveal(
    rows: Iterable[HistoryRow], settings: Mapping[str, float]
) -> tuple[RevealModel, dict[str, dict[str, float]]]:
    """Fit news variance by position and lead from game-week projection history."""
    series: dict[tuple[int, str], dict[int, tuple[float, float, bool, str | None]]] = defaultdict(
        dict
    )
    pos_of: dict[tuple[int, str], str] = {}
    for r in rows:
        proj, act = r.scored(settings)
        series[(r.season, r.player_id)][r.week] = (proj, act, r.played, r.opponent)
        pos_of[(r.season, r.player_id)] = r.position

    # Opponent effects: mean log projection relative to the player's season mean.
    opp_sum: dict[tuple[int, str, str], float] = defaultdict(float)
    opp_n: dict[tuple[int, str, str], int] = defaultdict(int)
    for key, wk in series.items():
        pos = pos_of[key]
        good = [(p, o) for p, _, _, o in wk.values() if o and _startable(pos, p)]
        if len(good) < 4:
            continue
        mean_log = float(np.mean([np.log(p) for p, _ in good]))
        for p, o in good:
            ok = (key[0], pos, o)
            opp_sum[ok] += np.log(p) - mean_log
            opp_n[ok] += 1
    opp = {k: opp_sum[k] / (opp_n[k] + OPP_SHRINK) for k in opp_sum}

    pub: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    unpub: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for key, wk in series.items():
        pos = pos_of[key]
        for w, (pw, _, played_w, ow) in wk.items():
            if not played_w or not _startable(pos, pw):
                continue
            for k in range(1, MAX_LEAD + 1):
                t = w - k
                if t not in wk:
                    continue
                pt, _, _, ot = wk[t]
                if not _startable(pos, pt):
                    continue
                between = [
                    np.clip((wk[u][1] - wk[u][0]) / wk[u][0], -1.0, 3.0)
                    for u in range(t, w)
                    if u in wk and wk[u][2] and wk[u][0] > 0
                ]
                perf = float(np.mean(between)) if between else 0.0
                drift = float(np.log(pw) - np.log(pt))
                shift = opp.get((key[0], pos, ow or ""), 0.0) - opp.get(
                    (key[0], pos, ot or ""), 0.0
                )
                pub[(pos, k)].append((drift - shift, perf))
                unpub[(pos, k)].append((drift, perf))

    model = RevealModel()
    diags: dict[str, dict[str, float]] = {}
    for pos in sorted({p for p, _ in pub}):
        curves = []
        for table, own in ((pub, True), (unpub, False)):
            vals = []
            for k in range(1, MAX_LEAD + 1):
                obs = table.get((pos, k), [])
                if len(obs) < MIN_ROWS:
                    break
                a = np.array(obs)
                vals.append(_residual_var(a[:, 0], a[:, 1]))
            if own and vals:
                v0 = max(2 * vals[0] - vals[1], 0.0) if len(vals) > 1 else 0.0
                vals = [max(v - v0 / 2, 0.0) for v in vals]
            curves.append([round(float(v), 5) for v in np.maximum.accumulate(vals)] if vals else [])
        if curves[0]:
            model.sigma2[pos] = curves[0]
            diags[pos] = {"n": float(len(pub[(pos, 1)])), "s2_lead1": curves[0][0]}
        if curves[1]:
            model.sigma2_unpub[pos] = curves[1]
            diags.setdefault(pos, {})["s2_unpub_lead1"] = curves[1][0]
    return model, diags
