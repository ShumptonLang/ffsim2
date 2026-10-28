import numpy as np
import pytest

from ffsim.data.projections import SleeperProjectionSource
from ffsim.league import (
    PlayoffFormat,
    PlayoffRoundType,
    PlayoffSeedType,
    load_league_state,
)
from ffsim.model.errors import ErrorModel
from ffsim.sim.engine import simulate
from ffsim.sim.inputs import SimInputs, build_inputs
from tests.conftest import LEAGUE_ID


@pytest.fixture(scope="module")
def model():
    return ErrorModel.default()


def synthetic_inputs(n_teams=6, weeks=(10, 11, 12), playoff_teams=4, strengths=None, base=None):
    strengths = strengths or [10.0] * n_teams
    po = PlayoffFormat(playoff_teams, 13, PlayoffRoundType.ONE_WEEK, PlayoffSeedType.FIXED)
    all_weeks = list(weeks) + [w for r in po.round_weeks() for w in r]
    rosters, means, positions = {}, {}, {}
    for t in range(n_teams):
        ids = [f"t{t}_{k}" for k in range(3)]
        rosters[t + 1] = ids
        for p in ids:
            positions[p] = ("WR",)
    for w in all_weeks:
        means[w] = {p: strengths[int(p[1:].split("_")[0])] for p in positions}
    pairs = [(i, n_teams - 1 - i) for i in range(n_teams // 2)]
    return SimInputs(
        team_ids=list(range(1, n_teams + 1)),
        team_labels=[f"T{i}" for i in range(n_teams)],
        rosters=rosters,
        slots=["WR", "WR", "FLEX"],
        weeks=all_weeks,
        regular_weeks=list(weeks),
        playoffs=po,
        matchups={w: pairs for w in weeks},
        base_half_wins=np.array(base if base is not None else [0.0] * n_teams),
        base_points_for=np.zeros(n_teams),
        divisions=[],
        median_match=False,
        means=means,
        positions=positions,
    )


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
