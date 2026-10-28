import numpy as np
import pytest

from ffsim.decisions.common import (
    Context,
    delta,
    resolve_players,
    successive_halving,
)
from ffsim.decisions.lineup import lineup_options
from ffsim.decisions.trades import apply_trade, evaluate_trade, find_trades
from ffsim.decisions.waivers import faab_bid, waiver_options
from ffsim.league import (
    League,
    LeagueState,
    Player,
    PlayoffFormat,
    PlayoffRoundType,
    PlayoffSeedType,
    Record,
    Team,
)
from ffsim.model.models import ModelSet
from ffsim.sim.engine import Evaluator, paired_delta
from ffsim.sim.inputs import SimInputs

POS = {"q": "QB", "w": "WR", "r": "RB"}
# roster_id -> {player_id: projection}. Team 1 is weak at WR2 and has a useless RB.
TEAMS = {
    1: {"q1": 16.0, "w1a": 13.0, "w1b": 6.0, "r1": 3.0},
    2: {"q2": 17.0, "w2a": 13.0, "w2b": 11.0, "r2": 8.0},
    3: {"q3": 15.0, "w3a": 12.0, "w3b": 10.0, "r3": 7.0},
    4: {"q4": 16.0, "w4a": 12.0, "w4b": 10.0, "r4": 6.0},
}
FREE = {"fa_w": 11.0, "fa_bad": 1.0}


def _pos(pid: str) -> str:
    return POS[pid[0]] if not pid.startswith("fa") else "WR"


def make_inputs() -> SimInputs:
    po = PlayoffFormat(2, 13, PlayoffRoundType.ONE_WEEK, PlayoffSeedType.FIXED)
    weeks = [10, 11, 12, 13]
    proj = {p: m for t in TEAMS.values() for p, m in t.items()} | FREE
    return SimInputs(
        team_ids=[1, 2, 3, 4],
        team_labels=["Me", "Two", "Three", "Four"],
        rosters={rid: list(ps) for rid, ps in TEAMS.items()},
        slots=["QB", "WR", "FLEX"],
        weeks=weeks,
        regular_weeks=[10, 11, 12],
        playoffs=po,
        matchups={10: [(0, 1), (2, 3)], 11: [(0, 2), (1, 3)], 12: [(0, 3), (1, 2)]},
        base_half_wins=np.zeros(4),
        base_points_for=np.zeros(4),
        divisions=[],
        median_match=False,
        means={w: dict(proj) for w in weeks},
        positions={p: (_pos(p),) for p in proj},
        free_agents=list(FREE),
        roster_capacity=4,
    )


def make_state() -> LeagueState:
    lg = League.from_sleeper(
        {"league_id": "1", "roster_positions": ["QB", "WR", "FLEX", "BN"], "settings": {}}
    )
    teams = {
        rid: Team(rid, f"u{rid}", f"user{rid}", name, tuple(ps), (), (), (), Record())
        for (rid, ps), name in zip(TEAMS.items(), ["Me", "Two", "Three", "Four"], strict=True)
    }
    players = {
        p: Player(p, f"Player {p}", _pos(p), (_pos(p),), "XX")
        for p in [*FREE, *[p for t in TEAMS.values() for p in t]]
    }
    return LeagueState(lg, teams, {}, players, nfl_week=10)


@pytest.fixture(scope="module")
def ctx():
    inp = make_inputs()
    ev = Evaluator(inp, ModelSet.default(), n_sims=6_000, seed=3)
    state = make_state()
    return Context(state, ev, state.teams[1])


def test_common_random_numbers_make_no_op_exact(ctx):
    ev = ctx.ev
    same = ev.run(rosters={1: list(TEAMS[1])})
    d = delta(ev, same, 0)
    assert d.champion == 0 and d.champion_se == 0


def test_only_changed_team_is_rescored(ctx):
    ev = ctx.ev
    tp = ev.team_points(rosters={1: ["q1", "w1a", "fa_w", "r1"]})
    assert np.array_equal(tp[:, :, 1:], ev.base_team_pts[:, :, 1:])
    assert tp[:, :, 0].mean() > ev.base_team_pts[:, :, 0].mean()


def test_paired_se_much_smaller_than_independent(ctx):
    ev = ctx.ev
    p = (ev.baseline().champion == 0).mean()
    unpaired = np.sqrt(2 * p * (1 - p) / ev.n_sims)
    # small edge (a slightly better bench body): pairing makes it measurable
    small = delta(ev, ev.run(rosters={1: ["q1", "w1a", "w1b", "fa_bad"]}), 0)
    assert small.champion_se < 0.3 * unpaired
    big = delta(ev, ev.run(rosters={1: ["q1", "w1a", "fa_w", "r1"]}), 0)
    assert big.champion > 5 * big.champion_se


def test_paired_delta():
    assert paired_delta(np.array([1, 0, 1, 0]), np.array([1, 1, 1, 0])) == pytest.approx(
        (0.25, 0.25)
    )


def test_lineup_options(ctx):
    opts = lineup_options(ctx, week=10)
    labels = [o.label for o in opts]
    assert "Projection-optimal lineup" in labels
    base = next(o for o in opts if o.is_baseline)
    assert base.delta.champion == 0
    worse = next(o for o in opts if o.label == "Start Player r1 over Player w1b")
    assert worse.mean < base.mean
    assert worse.p_win <= base.p_win
    assert [o.delta.champion for o in opts] == sorted(
        (o.delta.champion for o in opts), reverse=True
    )


def test_waivers_pick_the_upgrade_and_drop_the_dead_spot(ctx):
    opts = waiver_options(ctx, limit=2, screen_sims=2_000)
    best = opts[0]
    assert (best.add, best.drop) == ("fa_w", "r1")
    assert best.delta.champion > 2 * best.delta.champion_se
    assert opts[-1].delta.champion < best.delta.champion


def test_faab_bid():
    assert faab_bid(0.0, 3, 100) == 0
    assert faab_bid(0.01, 0, 100) == 10
    assert faab_bid(0.01, 1, 100) == 15
    assert faab_bid(0.20, 5, 100) == 100
    assert faab_bid(0.0001, 0, 100) == 1
    assert faab_bid(0.05, 2, 0) == 0


def test_apply_trade_auto_drops(ctx):
    rosters, my_drops, their_drops = apply_trade(ctx, 2, ["w1b"], ["w2a", "r2"])
    assert len(rosters[1]) == 4 and my_drops == ["r1"]
    assert "w2a" in rosters[1] and "r2" in rosters[1]
    assert rosters[2] == ["q2", "w2b", "w1b"] and their_drops == []


def test_trade_moves_odds_between_the_two_teams(ctx):
    r = evaluate_trade(ctx, 2, ["w1b"], ["w2a"])
    assert r.me.champion > 0 > r.partner_delta.champion
    assert r.my_lineup_delta == pytest.approx(4 * 7.0)
    # partner loses w2a (13); his RB (8) moves into FLEX rather than w1b (6)
    assert r.partner_lineup_delta == pytest.approx(-4 * 5.0)


def test_find_trades_only_acceptable_and_helpful(ctx):
    results = find_trades(ctx, max_size=2, limit=5, stages=[(1_000, 20), (6_000, 5)])
    for r in results:
        assert r.me.champion > 0
        assert r.partner_lineup_delta >= -1e-6
        assert r.partner_value_delta >= 0


def test_resolve_players():
    state = make_state()
    pool = ["w1a", "w1b", "q1"]
    assert resolve_players(state, ["player q1", " Player w1A"], pool) == ["q1", "w1a"]
    with pytest.raises(LookupError, match="matches 2"):
        resolve_players(state, ["player w1"], pool)
    with pytest.raises(LookupError, match="matches 0"):
        resolve_players(state, ["nobody"], pool)


def test_successive_halving_keeps_the_best():
    calls = []

    def score(c, n):
        calls.append((c, n))
        return -abs(c - 7) + (0.0 if n > 10 else np.sin(c))  # noisy at low n

    out = successive_halving(range(20), score, [(10, 6), (100, 2)])
    assert out[0][0] == 7
    assert sum(1 for _, n in calls if n == 100) == 6
