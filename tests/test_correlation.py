import numpy as np
import pytest

from ffsim.data.history import HistoryRow
from ffsim.model.correlation import (
    N_FACTORS,
    ROLES,
    CorrelationModel,
    PairStats,
    assign_roles,
    fit_loadings,
    pair_correlations,
    sample_weekly,
    week_layout,
)
from ffsim.model.models import ModelSet


def test_assign_roles():
    roles = assign_roles(
        [
            ("a", "BUF", "WR", 12.0),
            ("b", "BUF", "WR", 15.0),
            ("c", "BUF", "WR", 9.0),
            ("d", "BUF", "WR", 3.0),
            ("e", "KC", "WR", 4.0),
            ("f", "BUF", "QB", 20.0),
            ("g", "BUF", "QB", 0.0),  # no projection: no role
            ("h", None, "RB", 9.0),  # no team: no role
            ("i", "BUF", "FB", 2.0),  # unknown group
        ]
    )
    assert roles == {"b": "WR1", "a": "WR2", "c": "WR3", "d": "WR3", "e": "WR1", "f": "QB"}


TOY = CorrelationModel(
    {
        "QB": (0.3, 0.6, 0.3, 0.0, 0.0, 0.0),
        "WR1": (0.2, 0.4, 0.3, 0.1, 0.3, 0.0),
        "WR2": (0.2, 0.3, 0.3, 0.1, -0.3, 0.0),
        "DEF": (-0.1, 0.1, 0.0, -0.6, 0.0, 0.0),
    }
)


def test_implied_and_legacy_padding():
    assert TOY.implied("QB", "WR1", True) == pytest.approx(0.06 + 0.24 + 0.09)
    assert TOY.implied("QB", "DEF", False) == pytest.approx(-0.03 + 0.6 * -0.6 + 0.0)
    assert TOY.implied("WR1", "WR2", True) == pytest.approx(TOY.implied("WR2", "WR1", True))
    assert TOY.loading("XX") == (0.0,) * N_FACTORS
    legacy = CorrelationModel.from_dict({"loadings": {"QB": [0.1, 0.2, 0.3, 0.4]}})
    assert legacy.loading("QB") == (0.1, 0.2, 0.3, 0.4, 0.0, 0.0)
    assert CorrelationModel.from_dict(TOY.to_dict()).loadings == TOY.loadings


def test_sampled_correlations_match_model():
    players = ["qb", "wr1", "wr2", "def_opp", "qb_opp", "loner"]
    teams = {
        "qb": ("BUF", "KC"),
        "wr1": ("BUF", "KC"),
        "wr2": ("BUF", "KC"),
        "def_opp": ("KC", "BUF"),
        "qb_opp": ("KC", "BUF"),
    }
    roles = {"qb": "QB", "wr1": "WR1", "wr2": "WR2", "def_opp": "DEF", "qb_opp": "QB"}
    lay = week_layout(TOY, players, teams, roles)
    z = sample_weekly(np.random.default_rng(0), 200_000, lay)
    c = np.corrcoef(z.T)
    assert np.allclose(z.std(0), 1.0, atol=0.01)
    assert c[0, 1] == pytest.approx(TOY.implied("QB", "WR1", True), abs=0.01)
    assert c[1, 2] == pytest.approx(TOY.implied("WR1", "WR2", True), abs=0.01)
    assert c[0, 3] == pytest.approx(TOY.implied("QB", "DEF", False), abs=0.01)
    assert c[0, 4] == pytest.approx(TOY.implied("QB", "QB", False), abs=0.01)
    assert abs(c[5, :5]).max() < 0.01  # no game info -> independent


def test_fit_loadings_reproduces_targets():
    roles = ["QB", "WR1", "WR2", "DEF"]
    same, opp = {}, {}
    for i, a in enumerate(roles):
        for b in roles[i:]:
            key = tuple(sorted((a, b)))
            if a != b or a in ("WR1",):
                same[key] = (TOY.implied(a, b, True), 5000)
            opp[key] = (TOY.implied(a, b, False), 5000)
    model, diags = fit_loadings(PairStats(same, opp))
    err = [d["implied"] - d["target"] for d in diags]
    assert np.sqrt(np.mean(np.square(err))) < 0.02
    assert model.implied("QB", "WR1", True) == pytest.approx(
        TOY.implied("QB", "WR1", True), abs=0.03
    )
    assert model.implied("QB", "DEF", False) == pytest.approx(
        TOY.implied("QB", "DEF", False), abs=0.03
    )


def test_fit_divides_out_talent_persistence():
    pairs = PairStats({("QB", "WR1"): (0.2, 5000)}, {})
    _, diags = fit_loadings(pairs, rho={"QB": 0.1, "WR": 0.2})
    assert diags[0]["target"] == pytest.approx(0.2 / np.sqrt(0.9 * 0.8))
    assert diags[0]["observed"] == pytest.approx(0.2)


def test_pair_correlations_recover_truth_from_history():
    """Generate correlated weekly outcomes, then estimate role correlations back."""
    errors = ModelSet.default().errors
    rng = np.random.default_rng(1)
    players = ["qb_a", "wr_a", "def_b", "qb_b", "wr_b", "def_a"]
    pos = {"qb": "QB", "wr": "WR", "def": "DEF"}
    rows = []
    for g in range(3000):
        t = {"a": f"T{2 * g}", "b": f"T{2 * g + 1}"}
        teams = {p: (t[p[-1]], t["b" if p[-1] == "a" else "a"]) for p in players}
        roles = {p: {"qb": "QB", "wr": "WR1", "def": "DEF"}[p.split("_")[0]] for p in players}
        z = sample_weekly(rng, 1, week_layout(TOY, players, teams, roles))[0]
        means = np.array([18.0, 14.0, 7.0, 18.0, 14.0, 7.0])
        posl = [pos[p.split("_")[0]] for p in players]
        pts = errors.sample(z[None, :], means[None, :], posl)[0]
        for p, m, a, ps in zip(players, means, pts, posl, strict=True):
            rows.append(
                HistoryRow(
                    2025, g, p, ps, teams[p][0], teams[p][1], {"x": m}, {"x": float(a)}, True
                )
            )
    stats = pair_correlations(rows, {"x": 1.0}, errors)
    assert stats.same[("QB", "WR1")][0] == pytest.approx(TOY.implied("QB", "WR1", True), abs=0.06)
    assert stats.opp[("DEF", "QB")][0] == pytest.approx(TOY.implied("DEF", "QB", False), abs=0.06)
    assert stats.opp[("QB", "QB")][0] == pytest.approx(TOY.implied("QB", "QB", False), abs=0.06)


def test_default_models_include_fitted_correlation():
    corr = ModelSet.default().correlation
    assert set(ROLES) <= set(corr.loadings)
    assert corr.implied("QB", "WR1", True) > 0.15
    assert corr.implied("DEF", "QB", False) < -0.25
    assert corr.implied("RB1", "RB2", True) < 0
