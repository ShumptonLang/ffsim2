import numpy as np
import pytest

from ffsim.data.history import HistoryRow
from ffsim.model.errors import Z_GRID, ErrorModel, fit_error_model

SETTINGS = {"rec": 1.0, "fgm": 3.0}


@pytest.fixture(scope="module")
def model():
    return ErrorModel.default()


def test_default_covers_all_positions(model):
    assert set(model.positions) == {"QB", "RB", "WR", "TE", "K", "DEF", "DL", "LB", "DB"}
    assert model.positions["DEF"].kind == "additive"


@pytest.mark.parametrize(
    ("pos", "mean"),
    [("QB", 18), ("RB", 2), ("RB", 15), ("WR", 30), ("TE", 7), ("K", 8), ("DEF", 6), ("LB", 7)],
)
def test_mean_matches_projection(model, pos, mean):
    assert model.mean(mean, pos) == pytest.approx(mean, rel=0.01)


@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DEF", "DL", "LB", "DB"])
def test_quantile_monotone_and_skewed(model, pos):
    q = model.quantile(np.linspace(0.001, 0.999, 500), 10.0, pos)
    assert np.all(np.diff(q) >= -1e-6)
    median = model.quantile(0.5, 10.0, pos)[0]
    assert median < 10.0  # right skew: median below mean


def test_def_can_score_negative_and_zero_mean_scores_zero(model):
    assert model.quantile(0.001, 4.0, "DEF")[0] < 0
    z = np.random.default_rng(0).standard_normal((100, 2))
    pts = model.sample(z, np.array([0.0, 10.0]), ["WR", "DEF"])
    assert np.all(pts[:, 0] == 0) and np.any(pts[:, 1] != 0)


def test_monte_carlo_mean_and_shape(model):
    z = np.random.default_rng(1).standard_normal((50_000, 3, 2))
    means = np.array([[12.0, 5.0], [0.0, 7.0], [20.0, 9.0]])
    pts = model.sample(z, means[None], ["RB", "K"])
    assert pts.shape == (50_000, 3, 2)
    assert pts.mean(axis=0) == pytest.approx(means, abs=0.15)


def test_unknown_position_borrows_rb_shape(model):
    assert model.mean(6.0, "FB") == pytest.approx(6.0, rel=0.01)


def _synthetic_rows(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        m = float(rng.uniform(4, 20))
        ratio = float(np.exp(rng.normal(-0.18, 0.6)))
        rows.append(
            HistoryRow(2025, 1, f"p{i}", "WR", None, None, {"rec": m}, {"rec": m * ratio}, True)
        )
        kick = float(rng.gamma(4.0, 2.0))
        rows.append(
            HistoryRow(2025, 1, f"k{i}", "K", None, None, {"fgm": 8 / 3}, {"fgm": kick / 3}, True)
        )
    rows.append(HistoryRow(2025, 1, "x", "WR", None, None, {"rec": 9.0}, {"rec": 0.0}, False))
    return rows


@pytest.mark.parametrize("family", ["parametric", "bootstrap"])
def test_fit_recovers_synthetic_spread(family):
    m = fit_error_model(_synthetic_rows(), SETTINGS, family)
    assert set(m.positions) == {"WR", "K"}
    wr = m.positions["WR"]
    assert len(wr.centers) == 4 and wr.centers == sorted(wr.centers)
    # true ratio is lognormal(sigma=0.6): check the model's 10/90 quantiles
    lo, hi = m.quantile([0.1, 0.9], 10.0, "WR")
    true = 10 * np.exp(np.array([-1.2816, 1.2816]) * 0.6) / np.exp(0.6**2 / 2) * np.exp(0)
    assert lo == pytest.approx(true[0], rel=0.12)
    assert hi == pytest.approx(true[1], rel=0.12)
    assert m.mean(10.0, "WR") == pytest.approx(10.0, rel=0.01)
    # K residual sd ~ gamma sd = 2*sqrt(4) = 4
    k = m.quantile(np.linspace(0.0005, 0.9995, 2000), 8.0, "K")
    assert k.std() == pytest.approx(4.0, rel=0.15)


def test_json_roundtrip():
    m = fit_error_model(_synthetic_rows(500), SETTINGS)
    m2 = ErrorModel.from_json(m.to_json())
    z = Z_GRID[:, None]
    assert np.allclose(m.sample(z, np.array([9.0]), ["WR"]), m2.sample(z, np.array([9.0]), ["WR"]))


@pytest.mark.parametrize(("pos", "mean"), [("WR", 12.0), ("DEF", 7.0), ("QB", 20.0)])
def test_to_latent_inverts_sample(model, pos, mean):
    for z in (-1.5, -0.3, 0.0, 0.8, 2.0):
        pts = float(model.sample(np.array([[z]]), np.array([mean]), [pos])[0, 0])
        assert model.to_latent(pts, mean, pos) == pytest.approx(z, abs=0.03)


def test_to_latent_flat_region_midpoint(model):
    # a dud score sits in the flat dud mass; its latent value is well below 0
    assert model.to_latent(0.0, 12.0, "RB") < -1.0


def test_bad_family():
    with pytest.raises(ValueError):
        fit_error_model([], SETTINGS, "normal")
