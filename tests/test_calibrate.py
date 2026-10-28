import numpy as np
import pytest

from ffsim.calibrate import CalibrationReport, DistScore, score_distributions
from ffsim.data.history import HistoryRow
from ffsim.model.models import ModelSet


def _rows_from_model(model, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        m = float(rng.uniform(6, 20))
        z = rng.standard_normal((1, 1))
        a = float(model.sample(z, np.array([m]), ["WR"])[0, 0])
        rows.append(HistoryRow(2030, 1, f"p{i}", "WR", None, None, {"x": m}, {"x": a}, True))
    return rows


def test_model_scored_on_its_own_draws_is_calibrated():
    model = ModelSet.default().errors
    sc = score_distributions(model, _rows_from_model(model), {"x": 1.0})["WR"]
    assert sc.n == 6000
    assert sc.below == pytest.approx([0.1, 0.5, 0.9], abs=0.025)
    # a model that is too narrow scores worse on CRPS than the true one
    narrow = ModelSet.default().errors
    pm = narrow.positions["WR"]
    pm._tables = 1 + (pm._tables - 1) * 0.5
    worse = score_distributions(narrow, _rows_from_model(model), {"x": 1.0})["WR"]
    assert worse.crps > sc.crps
    # too narrow: more outcomes fall below its 10th and above its 90th percentile
    assert worse.below[0] > 0.13 and worse.below[2] < 0.87


def test_unplayed_and_unstartable_rows_skipped():
    model = ModelSet.default().errors
    rows = [
        HistoryRow(2030, 1, "a", "WR", None, None, {"x": 2.0}, {"x": 5.0}, True),  # not startable
        HistoryRow(2030, 1, "b", "WR", None, None, {"x": 9.0}, {"x": 0.0}, False),  # inactive
    ]
    assert score_distributions(model, rows, {"x": 1.0}) == {}


def test_merge_and_choose_families():
    a = DistScore(100, 4.0, np.array([0.1, 0.5, 0.9]))
    b = DistScore(300, 2.0, np.array([0.2, 0.6, 1.0]))
    m = a.merge(b)
    assert m.n == 400 and m.crps == pytest.approx(2.5)
    assert m.below == pytest.approx([0.175, 0.575, 0.975])
    rep = CalibrationReport(
        [2030],
        {
            "parametric": {"WR": DistScore(1, 3.0), "QB": DistScore(1, 4.0)},
            "bootstrap": {"WR": DistScore(1, 3.1), "QB": DistScore(1, 3.9)},
        },  # fmt: skip
        {},
        [],
    )
    assert rep.choose_families() == {"QB": "bootstrap", "WR": "parametric"}
