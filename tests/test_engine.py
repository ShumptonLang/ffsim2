import numpy as np
import pytest

from ffsim.data.projections import SleeperProjectionSource
from ffsim.league import (
    PlayoffFormat,
    PlayoffRoundType,
    PlayoffSeedType,
    load_league_state,
)
from ffsim.model.availability import AvailabilityModel
from ffsim.model.bayes import TalentModel
from ffsim.model.models import ModelSet
from ffsim.sim.engine import Evaluator, _team_scores, simulate
from ffsim.sim.inputs import build_inputs
from tests.conftest import LEAGUE_ID
from tests.synth import independent, synthetic_inputs


@pytest.fixture(scope="module")
def model():
    return ModelSet.default()


def test_identical_teams_have_equal_title_odds(model):
    res = simulate(synthetic_inputs(), model, n_sims=20_000, seed=1)
    assert res.champion.sum() == pytest.approx(1.0)
    assert res.playoffs.sum() == pytest.approx(4.0)
    assert res.final.sum() == pytest.approx(2.0)
    assert np.allclose(res.champion, 1 / 6, atol=4 * res.se(res.champion).max())


def test_stronger_team_favored_and_clinched_team_in(model):
    inp = synthetic_inputs(strengths=[20, 10, 10, 10, 10, 10], base=[0, 0, 0, 0, 0, 40])
    res = simulate(inp, model, n_sims=5_000, seed=2)
    assert res.playoffs[5] == 1.0
    assert res.bye.sum() == 0  # 4-team bracket has no byes
    assert res.champion[0] == res.champion.max()
    assert res.week_win[10][0] > 0.75


def test_seed_reproducible(model):
    inp = synthetic_inputs()
    a = simulate(inp, model, n_sims=2_000, seed=7)
    b = simulate(inp, model, n_sims=2_000, seed=7, batch_size=500)
    c = simulate(inp, model, n_sims=2_000, seed=8)
    assert np.array_equal(a.champion, simulate(inp, model, n_sims=2_000, seed=7).champion)
    assert a.champion.sum() == pytest.approx(b.champion.sum())
    assert not np.array_equal(a.champion, c.champion)


def test_median_match_adds_wins(model):
    inp = synthetic_inputs()
    base = simulate(inp, model, n_sims=2_000, seed=3)
    inp.median_match = True
    med = simulate(inp, model, n_sims=2_000, seed=3)
    assert med.mean_wins.sum() == pytest.approx(2 * base.mean_wins.sum(), rel=0.01)


def test_two_week_rounds(model):
    inp = synthetic_inputs(playoff_teams=2)
    inp.playoffs = PlayoffFormat(2, 13, PlayoffRoundType.TWO_WEEKS_PER_ROUND, PlayoffSeedType.FIXED)
    inp.weeks = [10, 11, 12, 13, 14]
    for w in (13, 14):
        inp.means[w] = inp.means[10]
    res = simulate(inp, model, n_sims=2_000, seed=4)
    assert res.champion.sum() == pytest.approx(1.0)


def test_fixture_league_end_to_end(client, model):
    state = load_league_state(client, LEAGUE_ID)
    inp = build_inputs(state, SleeperProjectionSource(client), client.winners_bracket(LEAGUE_ID))
    assert inp.weeks == [3, 4, 5] and inp.regular_weeks == [3]
    assert inp.means[4] == inp.means[3]  # unpublished weeks reuse the latest projections
    assert inp.forced_playoff_winners == {}  # bracket not decided yet
    # IR player excluded from lineups
    ir = state.teams[2].reserve[0]
    assert all(ir not in team for wk in inp.lineups().values() for team in wk.values())
    res = simulate(inp, model, n_sims=3_000, seed=0)
    assert res.playoffs.sum() == pytest.approx(2.0)
    assert res.champion.sum() == pytest.approx(1.0)
    # team 1 is 2-0; one game left; top-2 make it -> team 1 always in
    assert res.playoffs[inp.team_index(1)] == 1.0


def test_persistence_makes_favorites_less_certain(model):
    inp = synthetic_inputs(strengths=[13, 10, 10, 10, 10, 10])
    flat = simulate(inp, independent(model), n_sims=8_000, seed=5)
    sticky = ModelSet(
        model.errors, TalentModel(default_rho=0.6), AvailabilityModel(default_hazard=0)
    )
    persistent = simulate(inp, sticky, n_sims=8_000, seed=5)
    assert persistent.champion[0] < flat.champion[0] - 0.03


def test_injuries_lower_scores(model):
    inp = synthetic_inputs()
    healthy = simulate(inp, independent(model), n_sims=3_000, seed=6)
    hurt = ModelSet(
        model.errors, TalentModel(default_rho=0.0), AvailabilityModel(default_hazard=0.3)
    )
    injured = simulate(inp, hurt, n_sims=3_000, seed=6)
    last = inp.weeks[-1]
    assert injured.week_points[last].mean() < 0.9 * healthy.week_points[last].mean()
    assert injured.week_points[inp.weeks[0]].mean() == pytest.approx(
        healthy.week_points[inp.weeks[0]].mean(), rel=0.02
    )


def test_bench_substitution_and_single_use():
    pts = np.array([[10.0, 7.0, 5.0], [10.0, 7.0, 5.0]], dtype=np.float32)
    avail = np.array([[True, True, True], [False, True, True]])
    # two slots; slot 0 starter p0 (backups p1, p2); slot 1 starter p1 (backup p2)
    plans = [[[0, 1, 2], [1, 2]]]
    out = _team_scores(pts, avail, plans)
    # sim 2: p0 out -> p1 moves into slot 0, so slot 1 falls back to p2
    assert out[:, 0].tolist() == [17.0, 12.0]


def test_questionable_starter_sometimes_sits(client, model):
    state = load_league_state(client, LEAGUE_ID)
    inp = build_inputs(state, SleeperProjectionSource(client), [])
    assert inp.statuses == {"1005": "Questionable"}


def test_bayesian_rates_and_centres(client, model):
    state = load_league_state(client, LEAGUE_ID)
    inp = build_inputs(state, SleeperProjectionSource(client), [], model.talent)
    lam, k = model.talent.lam_k("QB")
    # QB 1001: preseason 340 pts / 17 = 20 per game; weeks 1-2 scored 10 and 12
    assert inp.rates["1001"] == pytest.approx((k * 20 + 2 * 11) / (k + 2))
    sleeper = inp.sleeper_means[3]["1001"]
    assert inp.means[3]["1001"] == pytest.approx((1 - lam) * sleeper + lam * inp.rates["1001"])
    # LB 1011: prior 6/game, scored 12 twice; no week-3 projection -> stays 0
    lam_lb, k_lb = model.talent.lam_k("LB")
    assert inp.rates["1011"] == pytest.approx((k_lb * 6 + 2 * 12) / (k_lb + 2))
    assert inp.means[3].get("1011", 0.0) == 0.0
    # players with no prior and no games get no rate
    assert "1002" not in inp.rates


def _stack_inputs():
    """Team 1 stacks QB+WR1+WR2 from BUF; team 2 has the same roles spread over 3 NFL teams."""
    inp = synthetic_inputs(n_teams=2, playoff_teams=2)
    inp.slots = ["QB", "WR", "WR"]
    inp.rosters = {1: ["q1", "w1", "w2"], 2: ["q2", "w3", "w4"]}
    inp.positions = {"q1": ("QB",), "q2": ("QB",), "w1": ("WR",), "w2": ("WR",),
                     "w3": ("WR",), "w4": ("WR",)}  # fmt: skip
    for w in inp.weeks:
        inp.means[w] = {"q1": 20, "q2": 20, "w1": 14, "w2": 12, "w3": 14, "w4": 12}
        inp.nfl_games[w] = {
            "q1": ("BUF", "MIA"), "w1": ("BUF", "MIA"), "w2": ("BUF", "MIA"),
            "q2": ("KC", "LV"), "w3": ("DET", "GB"), "w4": ("SF", "LA"),
        }  # fmt: skip
        inp.roles[w] = {"q1": "QB", "w1": "WR1", "w2": "WR2", "q2": "QB", "w3": "WR1", "w4": "WR2"}
    return inp


def test_stacked_team_has_wider_score_distribution(model):
    inp = _stack_inputs()
    scores = Evaluator(inp, model, n_sims=40_000, seed=0).base_team_pts[:, 0, :]
    assert scores[:, 0].mean() == pytest.approx(scores[:, 1].mean(), rel=0.02)
    assert scores[:, 0].std() > 1.08 * scores[:, 1].std()
    # without correlation the two teams are statistically identical
    s2 = Evaluator(inp, independent(model), n_sims=40_000, seed=0).base_team_pts[:, 0, :]
    assert s2[:, 0].std() == pytest.approx(s2[:, 1].std(), rel=0.03)


def test_fixture_roles_and_games(client, model):
    state = load_league_state(client, LEAGUE_ID)
    inp = build_inputs(state, SleeperProjectionSource(client), [], model.talent)
    assert inp.nfl_games[3]["1001"] == ("BUF", "KC")
    assert inp.roles[3]["1001"] == "QB"
    assert inp.roles[3]["BUF"] == "DEF"
    assert inp.nfl_games[4] == {}  # unpublished week: no game info -> independent
    res = simulate(inp, model, n_sims=1_000, seed=0)
    assert res.champion.sum() == pytest.approx(1.0)
