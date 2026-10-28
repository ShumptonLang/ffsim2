import numpy as np
import pytest

from ffsim.data.history import PlayerSeason
from ffsim.model.bayes import TalentModel, fit_talent, prior_rate
from ffsim.model.errors import ErrorModel


def test_rate_shrinks_toward_prior():
    m = TalentModel(blend={"WR": (0.5, 4.0)})
    assert m.rate("WR", 15.0, 25.0, 0) == pytest.approx(15.0)
    assert m.rate("WR", 15.0, 25.0, 4) == pytest.approx(20.0)
    assert m.rate("WR", 15.0, 25.0, 400) == pytest.approx(25.0, abs=0.3)
    assert m.rate("WR", None, 12.0, 3) == 12.0
    assert m.rate("WR", None, 0.0, 0) is None


def test_centre_blends_and_respects_zero_weeks():
    m = TalentModel(blend={"LB": (0.6, 2.0)})
    assert m.centre("LB", 5.0, 10.0) == pytest.approx(8.0)
    assert m.centre("LB", 0.0, 10.0) == 0.0  # bye / ruled out stays out
    assert m.centre("LB", 5.0, None) == 5.0
    assert m.centre("QB", 20.0, 10.0) == pytest.approx(19.0)  # default blend (0.1, 4)


def test_prior_rate_scores_an_average_game():
    s = {"pass_yd": 0.04, "bonus_pass_yd_300": 3}
    assert prior_rate({"pass_yd": 4250, "gp": 17}, s, "QB") == pytest.approx(10.0)  # no bonus
    assert prior_rate({"pass_yd": 5440, "gp": 17}, s, "QB") == pytest.approx(15.8)  # 320/gm
    assert prior_rate({"pass_yd": 4250, "gp": 18}, s, "QB") == pytest.approx(10.0)  # capped
    assert prior_rate({"pass_yd": 0, "gp": 17}, s, "QB") is None


def test_roundtrip():
    m = TalentModel(blend={"WR": (0.3, 12.0)}, rho={"WR": 0.15}, default_rho=0.1)
    m2 = TalentModel.from_dict(m.to_dict())
    assert m2.lam_k("WR") == (0.3, 12.0) and m2.rho_for("WR") == 0.15
    assert m2.rho_for("XX") == 0.1


def _league(sleeper_knows: bool, n_players=400, seed=0):
    """Synthetic seasons. Each player has a persistent true rate that differs from the prior.

    If ``sleeper_knows``, weekly projections equal the true rate; otherwise they
    are the stale prior, so observed results carry information.
    """
    rng = np.random.default_rng(seed)
    seasons, priors = {}, {}
    for i in range(n_players):
        prior = float(rng.uniform(8, 18))
        true = prior * float(np.exp(rng.normal(0, 0.3)))
        weeks = {}
        for w in range(1, 15):
            proj = true if sleeper_knows else prior
            act = true * float(np.exp(rng.normal(-0.08, 0.4)))
            weeks[w] = (proj, act, True)
        key = (2025, f"p{i}")
        seasons[key] = PlayerSeason(2025, f"p{i}", "WR", "XX", weeks)
        priors[key] = prior
    return seasons, priors


def test_fit_learns_to_trust_results_when_projections_are_stale():
    errors = ErrorModel.default()
    stale, priors = _league(sleeper_knows=False)
    m, diag = fit_talent(stale, priors, errors)
    lam, k = m.lam_k("WR")
    assert lam >= 0.4 and diag["WR"]["mse_blend"] < diag["WR"]["mse_sleeper"]
    assert m.rho_for("WR") > 0.1  # misses persist when the centre is stale

    fresh, priors = _league(sleeper_knows=True)
    m2, _ = fit_talent(fresh, priors, errors)
    assert m2.lam_k("WR")[0] <= 0.1
    assert m2.rho_for("WR") < 0.08
