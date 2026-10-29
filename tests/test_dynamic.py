"""Closed-loop (dynamic) season simulation."""

import numpy as np
import pytest

from ffsim.data.history import HistoryRow
from ffsim.decisions.common import paired_delta
from ffsim.dynamic.beliefs import Beliefs
from ffsim.dynamic.claims import Claim, resolve
from ffsim.dynamic.decide import DynamicOptions, make_dynamic_plan
from ffsim.dynamic.leverage import estimate_leverage
from ffsim.dynamic.lineups import SlotSpec, exact_lineups, greedy_lineups, greedy_totals
from ffsim.dynamic.paths import draw_paths
from ffsim.dynamic.policy import GreedyPolicy, PolicyOptions
from ffsim.dynamic.rules import WaiverRules
from ffsim.dynamic.season import DynamicSeason, Move
from ffsim.dynamic.state import FREE, PathLeague
from ffsim.league import PlayoffFormat, PlayoffRoundType, PlayoffSeedType
from ffsim.model.availability import SEASON_OUT, AvailabilityModel
from ffsim.model.models import ModelSet
from ffsim.model.reveal import RevealModel, fit_reveal
from ffsim.sim.engine import Evaluator, season_outcomes
from ffsim.sim.inputs import SimInputs
from ffsim.sim.lineups import best_lineup
from tests.synth import independent, synthetic_inputs
from tests.test_decisions import make_inputs


@pytest.fixture(scope="module")
def models():
    return ModelSet.default()


def league(rosters, means, positions, slots, free=(), capacity=0, weeks=(10, 11, 12)):
    """Small league: 4 teams, round robin, 2-team one-week playoff after ``weeks``."""
    po = PlayoffFormat(2, weeks[-1] + 1, PlayoffRoundType.ONE_WEEK, PlayoffSeedType.FIXED)
    all_weeks = list(weeks) + [w for r in po.round_weeks() for w in r]
    pairs = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]
    return SimInputs(
        team_ids=[1, 2, 3, 4],
        team_labels=["Me", "Two", "Three", "Four"],
        rosters={rid: list(ps) for rid, ps in rosters.items()},
        slots=list(slots),
        weeks=all_weeks,
        regular_weeks=list(weeks),
        playoffs=po,
        matchups={w: pairs[i % 3] for i, w in enumerate(weeks)},
        base_half_wins=np.zeros(4),
        base_points_for=np.zeros(4),
        divisions=[],
        median_match=False,
        means={w: {p: means[p](w) for p in positions} for w in all_weeks},
        positions={p: (pos,) for p, pos in positions.items()},
        free_agents=list(free),
        roster_capacity=capacity,
    )


# -- engine equivalence -----------------------------------------------------------------


def test_never_move_reproduces_static_engine(models):
    inp = synthetic_inputs()
    ev = Evaluator(inp, models, n_sims=400, seed=4, batch_size=400)
    season = DynamicSeason(inp, models, WaiverRules())
    draws = draw_paths(
        season.setup, inp, models, season.players,
        np.random.default_rng(4), np.random.default_rng(1), 400, news=False,
    )  # fmt: skip
    run = season.run(draws, None, learn=False)
    assert np.array_equal(run.team_pts, ev.base_team_pts)


def test_availability_state_does_not_change_draws():
    m = AvailabilityModel()
    active = np.ones((6, 30), dtype=bool)
    hz, first = np.full(30, 0.2), np.full(30, 0.9)
    plain = m.sample(np.random.default_rng(3), 500, active, hz, first)
    avail, state = m.sample_with_state(np.random.default_rng(3), 500, active, hz, first)
    assert np.array_equal(plain, avail)
    # Absent after week w and not back in week w + 1 is consistent with the state.
    assert not (avail[:, 1:] & (state[:, :-1] == SEASON_OUT)).any()
    newly = (state[:, :-1] == 0) & avail[:, :-1]
    assert newly.any() and not (newly & avail[:, 1:]).any()


# -- lineups, beliefs ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "slots",
    [
        ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "K", "DEF", "DL", "LB", "DB",
         "IDP_FLEX"],
        ["QB", "RB", "WR", "WR", "FLEX", "FLEX"],
    ],
)  # fmt: skip
def test_greedy_lineups_match_hungarian(slots):
    rng = np.random.default_rng(0)
    pool = ["QB", "RB", "WR", "TE", "K", "DEF", "DL", "LB", "DB"]
    players = [f"p{i}" for i in range(40)]
    positions = {p: (pool[rng.integers(len(pool))],) for p in players}
    spec = SlotSpec.build(slots, players, positions)
    assert spec.exact
    for _ in range(200):
        roster = list(rng.choice(40, size=16, replace=False))
        vals = rng.gamma(2, 5, size=16)
        vals[rng.random(16) < 0.1] = -np.inf
        total, _ = greedy_lineups(vals, spec.elig[roster], spec.order)
        exact, _ = exact_lineups(vals, spec.elig[roster])
        means = {players[r]: v for r, v in zip(roster, vals, strict=True) if np.isfinite(v)}
        ref = sum(means[p] for p in best_lineup(slots, list(means), means, positions) if p)
        assert total == pytest.approx(ref) and exact == pytest.approx(ref)


def test_compiled_and_numpy_lineup_totals_agree():
    rng = np.random.default_rng(2)
    slots = ["QB", "RB", "WR", "FLEX", "SUPER_FLEX", "K"]
    players = [f"p{i}" for i in range(30)]
    pool = ["QB", "RB", "WR", "TE", "K"]
    spec = SlotSpec.build(slots, players, {p: (pool[i % 5],) for i, p in enumerate(players)})
    idx = rng.integers(0, 31, size=(20, 12))  # includes the empty-spot row
    vals = rng.gamma(2, 5, size=(20, 7, 12)).astype(np.float32)
    vals[rng.random(vals.shape) < 0.1] = -np.inf
    fast = greedy_totals(vals, spec.elig[idx], spec.order)
    slow = greedy_totals(vals, spec.elig[idx], spec.order, accelerate=False)
    assert np.allclose(fast, slow)


def test_overlapping_flex_layout_is_flagged_inexact():
    spec = SlotSpec.build(["REC_FLEX", "WRRB_FLEX"], ["a"], {"a": ("WR",)})
    assert not spec.exact


def test_expected_points_and_observable_latent(models):
    e = models.errors
    pos = ["QB", "RB", "WR", "TE", "K", "DEF", "LB"]
    means = np.array([18.0, 9.0, 4.0, 7.0, 8.0, 7.0, 6.0])
    assert e.expected_points(means, np.zeros(7), np.ones(7), pos) == pytest.approx(means, rel=1e-5)
    z = np.random.default_rng(0).standard_normal((300_000, 1)).astype(np.float32)
    mc = e.sample(0.3 + 0.9 * z, means, pos).mean(0)
    ep = e.expected_points(means, np.full(7, 0.3), np.full(7, 0.9), pos)
    assert ep == pytest.approx(mc, rel=0.02)
    zz = np.random.default_rng(1).standard_normal((500, 7)).astype(np.float32)
    pts = e.sample(zz, means, pos)
    obs = e.observable_latent(zz, means, pos)
    for j in range(7):
        ref = e.to_latent_many(pts[:, j].astype(float), np.full(500, means[j]), [pos[j]] * 500)
        assert np.abs(obs[:, j] - ref).max() < 0.02


def test_beliefs_match_normal_update():
    rho = np.array([0.0, 0.2, 0.5])
    b = Beliefs(1, rho)
    zs = [np.array([[1.0, 1.0, 1.0]]), np.array([[0.4, -0.2, 2.0]])]
    for z in zs:
        b.observe(z, np.ones((1, 3), dtype=bool))
    shift, scale = b.posterior()
    for j, r in enumerate(rho):
        # Brute force: z_i = sqrt(r) eta + sqrt(1 - r) e_i; predictive for z_new.
        n, tot = 2, sum(float(z[0, j]) for z in zs)
        prec = 1 + n * r / (1 - r)
        e_eta = np.sqrt(r) / (1 - r) * tot / prec
        assert shift[0, j] == pytest.approx(np.sqrt(r) * e_eta)
        assert scale[0, j] ** 2 == pytest.approx(r / prec + 1 - r)


# -- claims -------------------------------------------------------------------------------


def _claims_state(budgets=(100, 100, 100, 100)):
    inp = league(
        {1: ["a"], 2: ["b"], 3: ["c"], 4: ["d"]},
        {p: (lambda w: 5.0) for p in "abcdxy"},
        {p: "WR" for p in "abcdxy"},
        ["WR"],
        free=["x", "y"],
        capacity=1,
    )
    pidx = {p: i for i, p in enumerate(inp.universe)}
    rules = WaiverRules(
        "faab", dict(zip([1, 2, 3, 4], budgets, strict=True)), 0, {1: 4, 2: 3, 3: 2, 4: 1}
    )
    return PathLeague.initial(inp, pidx, rules, 1), pidx


def test_faab_highest_bid_then_priority():
    st, ix = _claims_state()
    claims = [
        Claim(0, ix["x"], ix["a"], bid=10),
        Claim(1, ix["x"], ix["b"], bid=12),
        Claim(2, ix["x"], ix["c"], bid=12),  # ties team 1, has better priority
        Claim(3, ix["y"], ix["d"], bid=150),  # over budget
    ]
    won = resolve(st, 0, claims, "faab", step=0)
    assert [(c.team, c.add) for c in won] == [(2, ix["x"])]
    assert st.budget[0].tolist() == [100, 100, 88, 100]
    assert st.owner[0, ix["c"]] == FREE and st.claimable_from[0, ix["c"]] == 1
    # The dropped player is on waivers this step and claimable at the next.
    assert not resolve(st, 0, [Claim(0, ix["c"], ix["a"])], "faab", step=0)
    assert resolve(st, 0, [Claim(0, ix["c"], ix["a"])], "faab", step=1)


def test_rolling_priority_moves_claimant_to_back():
    st, ix = _claims_state()
    claims = [Claim(3, ix["x"], ix["d"]), Claim(3, ix["y"], ix["d"], rank=1),
              Claim(2, ix["y"], ix["c"])]  # fmt: skip
    won = resolve(st, 0, claims, "rolling", step=0)
    assert [(c.team, c.add) for c in won] == [(3, ix["x"]), (2, ix["y"])]
    # Start [4, 3, 2, 1]; team 3 claims and goes last, then team 2 does.
    assert st.priority[0].tolist() == [2, 1, 4, 3]


# -- information discipline ------------------------------------------------------------


def test_news_is_mean_preserving_and_known_at_game_week(models):
    inp = make_inputs()
    m = ModelSet(models.errors, reveal=RevealModel({"WR": [0.1, 0.2]}, {"WR": [0.1, 0.2]}))
    season = DynamicSeason(inp, m, WaiverRules())
    draws = season.draw(0, 4000)
    assert draws.news is not None
    W = season.n_weeks
    true = draws.known_log(W - 1)  # every week's news revealed by the last step
    wr = [season.pidx[p] for p in ("w1a", "fa_w")]
    assert np.exp(true[:, 2, wr]).mean() == pytest.approx(1.0, abs=0.02)
    assert np.allclose(draws.known_log(0)[:, 0], 0.0)  # this week's news is in hand
    assert draws.known_log(0)[:, 2, wr].std() == pytest.approx(0.0, abs=1e-6)  # two weeks out
    assert draws.known_log(1)[:, 2, wr].std() > 0.2
    assert np.allclose(draws.known_log(2)[:, 2], true[:, 2], atol=1e-5)


def test_decisions_ignore_future_outcomes(models):
    inp = make_inputs()
    season = DynamicSeason(inp, models, WaiverRules(), PolicyOptions(scenarios=4))
    draws = season.draw(2, 200)
    base = season.run(draws, None)
    pol = GreedyPolicy(estimate_leverage(inp, season.week_idx, base.team_pts), None, 0, False,
                       season.opts)  # fmt: skip
    first = season.first_decisions(pol)
    t = 1
    a = season.run(draws, pol, seed=5, first=first)
    rng = np.random.default_rng(9)
    for name in ("pts", "z_obs", "avail", "absence"):
        arr = getattr(draws, name)
        arr[:, t + 1 :] = arr[rng.permutation(draws.n), t + 1 :]
    b = season.run(draws, pol, seed=5, first=first)
    early = [(m.path, m.step, m.team, m.add, m.drop) for m in a.moves if m.step <= t + 1]
    assert early == [(m.path, m.step, m.team, m.add, m.drop) for m in b.moves if m.step <= t + 1]


# -- the motivating cases ---------------------------------------------------------------


def kicker_bye_inputs():
    """My RB1 is on bye in week 11 (RB2 fills in), my K in week 12.

    A worse free-agent kicker is only useful as bye cover. Adding him now means
    dropping RB2 before his week-11 start; waiting gets both.
    """
    flat = lambda v: lambda w: v  # noqa: E731
    means = {
        "rb1": lambda w: 0.0 if w == 11 else 12.0,
        "rb2": flat(4.0),
        "k1": lambda w: 0.0 if w == 12 else 8.0,
        "k2": flat(6.0),
    }
    positions = {"rb1": "RB", "rb2": "RB", "k1": "K", "k2": "K"}
    rosters = {1: ["rb1", "rb2", "k1"]}
    for t in (2, 3, 4):
        rosters[t] = [f"r{t}a", f"r{t}b", f"k{t}x"]
        means |= {f"r{t}a": flat(12.0), f"r{t}b": flat(6.0), f"k{t}x": flat(8.5)}
        positions |= {f"r{t}a": "RB", f"r{t}b": "RB", f"k{t}x": "K"}
    return league(rosters, means, positions, ["RB", "K"], free=["k2"], capacity=3)


def _first_move_delta(season, policy, first, draws, move):
    base = season.run(draws, policy, focal=0, first=first)
    alt = season.run(draws, policy, focal=0, forced=move, first=first)
    out = [season_outcomes(season.inputs, season.week_idx, r.team_pts).champion == 0
           for r in (base, alt)]  # fmt: skip
    return paired_delta(*out), base


def test_full_roster_drops_someone_even_if_a_rival_is_over_the_limit(models):
    """Rosters are padded to the largest; the padding isn't an open spot."""
    inp = make_inputs()
    inp.rosters[2] = [*inp.rosters[2], "fa_bad"]  # team 2 is one over the limit
    inp.free_agents = ["fa_w"]
    season = DynamicSeason(inp, models, WaiverRules(), PolicyOptions(scenarios=4))
    draws = season.draw(1, 200)
    never = season.run(draws, None)
    lev = estimate_leverage(inp, season.week_idx, never.team_pts)
    first = season.first_decisions(GreedyPolicy(lev, None, 0, False, season.opts))
    mine = first[0].claims[0]
    assert mine and all(c.drop >= 0 for c in mine)
    run = season.run(draws, GreedyPolicy(lev, None, 0, False, season.opts), first=first)
    assert any(m.team == 0 and m.step == 0 for m in run.moves)  # the claim went through


def test_kicker_for_bye_waits_until_the_bye_week(models):
    inp = kicker_bye_inputs()
    m = independent(models)
    season = DynamicSeason(inp, m, WaiverRules(), PolicyOptions(scenarios=2))
    draws = season.draw(3, 3000)
    never = season.run(draws, None)
    pol = GreedyPolicy(estimate_leverage(inp, season.week_idx, never.team_pts), None, 0, False,
                       season.opts)  # fmt: skip
    first = season.first_decisions(pol)
    assert first[0].claims[0] == [] and first[0].stash[0] is not None  # deferred, not claimed
    (d, se), base = _first_move_delta(season, pol, first, draws, Move("claim", "k2", "rb2"))
    my_adds = [(mv.step, season.players[mv.add]) for mv in base.moves if mv.team == 0]
    assert my_adds and all(a == (2, "k2") for a in my_adds)  # week 12, the bye week
    assert len(my_adds) == draws.n
    assert d < -2 * se  # adding the kicker now (dropping RB2 before his start) is worse
    ev = Evaluator(inp, m, n_sims=3000, seed=3)
    static = ev.run(rosters={1: ["rb1", "k1", "k2"]})
    s_d, s_se = paired_delta(ev.baseline().champion == 0, static.champion == 0)
    assert s_d > 2 * s_se  # ... although with fixed rosters it looks like a gain


def streaming_inputs():
    flat = lambda v: lambda w: v  # noqa: E731
    rosters = {t: [f"d{t}a", f"d{t}b"] for t in (1, 2, 3, 4)}
    free = [f"fa{i}" for i in range(6)]
    names = [p for ps in rosters.values() for p in ps] + free
    return league(rosters, {p: flat(7.0) for p in names}, {p: "DEF" for p in names}, ["DEF"],
                  free=free, capacity=2)  # fmt: skip


@pytest.mark.parametrize("s2, streams", [(0.4, True), (0.0, False)])
def test_streaming_emerges_only_when_matchups_vary(models, s2, streams):
    inp = streaming_inputs()
    base = independent(models)
    m = ModelSet(base.errors, base.talent, base.availability, base.correlation,
                 reveal=RevealModel({"DEF": [s2] * 4}, {"DEF": [s2] * 4}))  # fmt: skip
    season = DynamicSeason(inp, m, WaiverRules(), PolicyOptions(scenarios=2, max_moves=1))
    draws = season.draw(1, 300)
    never = season.run(draws, None)
    pol = GreedyPolicy(estimate_leverage(inp, season.week_idx, never.team_pts), None, 0, False,
                       season.opts)  # fmt: skip
    run = season.run(draws, pol)
    adds = sum(1 for mv in run.moves if mv.team == 0) / draws.n
    assert (adds > 1.0) if streams else (adds == 0)


def test_my_policy_beats_standing_still(models):
    inp = make_inputs()
    season = DynamicSeason(inp, models, WaiverRules(), PolicyOptions(scenarios=4))
    draws = season.draw(4, 1500)
    never = season.run(draws, None)
    lev = estimate_leverage(inp, season.week_idx, never.team_pts)
    lev[1:] = 0.0  # rivals never move
    run = season.run(draws, GreedyPolicy(lev, None, 0, False, season.opts))
    out = [season_outcomes(inp, season.week_idx, r.team_pts).champion == 0 for r in (never, run)]
    d, se = paired_delta(*out)
    assert d > 2 * se


def test_make_dynamic_plan_end_to_end(models):
    inp = kicker_bye_inputs()
    opts = DynamicOptions.fast(fit_paths=300, screen_paths=300, eval_paths=600)
    plan = make_dynamic_plan(inp, independent(models), WaiverRules(), 1, opts,
                             extra=[Move("claim", "k2", "rb2")])  # fmt: skip
    policy = [o for o in plan.options if o.is_policy]
    assert len(policy) == 1 and policy[0].delta.champion == 0
    now = next(o for o in plan.options if o.move.add == "k2")
    assert now.delta.champion < 0
    assert plan.planned and plan.planned[0].player == "k2" and plan.planned[0].median_week == 12
    assert plan.adds_per_position == {"K": pytest.approx(1.0)}


# -- reveal fit and persistence --------------------------------------------------------


def test_fit_reveal_recovers_planted_news():
    rng = np.random.default_rng(0)
    q, se2 = 0.01, 0.04  # level random-walk step, one-off weekly component
    rows = []
    for k in range(400):
        level = np.cumsum(rng.normal(0, np.sqrt(q), 17))
        proj = 15.0 * np.exp(level + rng.normal(0, np.sqrt(se2), 17))
        for w in range(17):
            rows.append(
                HistoryRow(2030, w + 1, f"p{k}", "WR", "AAA", None, {"rec": float(proj[w])},
                           {"rec": float(proj[w])}, True)
            )  # fmt: skip
    model, _ = fit_reveal(rows, {"rec": 1.0})
    pub, unpub = model.sigma2["WR"], model.sigma2_unpub["WR"]
    assert pub[0] == pytest.approx(q + se2, rel=0.25)
    assert pub[2] == pytest.approx(3 * q + se2, rel=0.25)
    assert unpub[0] == pytest.approx(q + 2 * se2, rel=0.25)


def test_models_round_trip_with_reveal(models):
    m = ModelSet(models.errors, reveal=RevealModel({"WR": [0.1, 0.2]}, {}))
    back = ModelSet.from_json(m.to_json())
    assert back.reveal.var("WR", 2) == 0.2 and back.reveal.var("WR", 9) == 0.2
    assert back.reveal.var("QB", 1) == 0.0
    assert ModelSet.default().reveal is not None


# -- FAAB --------------------------------------------------------------------------------


def test_faab_prices_spend_each_budget():
    from ffsim.dynamic.decide import faab_prices

    budgets = np.array([100.0, 50.0, 100.0])
    first = faab_prices(None, budgets, np.array([0.5, 0.5, 0.0]))
    assert first.tolist() == [200.0, 100.0, 200.0]  # a team that bought nothing bids hardest
    damped = faab_prices(first, budgets, np.array([0.25, 0.5, 1.0]))
    assert damped == pytest.approx([np.sqrt(200 * 400), 100.0, np.sqrt(200 * 100)])


def test_winning_bid_beats_the_best_rival(models):
    from ffsim.dynamic.decide import _with_winning_bid
    from ffsim.dynamic.policy import TeamDecision

    inp = make_inputs()
    rules = WaiverRules("faab", {1: 60, 2: 60, 3: 60, 4: 60}, 1, {1: 3, 2: 1, 3: 2, 4: 4})
    season = DynamicSeason(inp, models, rules)
    ix = season.pidx
    none = TeamDecision([[]], [None])
    first = [
        none,
        TeamDecision([[Claim(1, ix["fa_w"], ix["r2"], bid=17)]], [None]),
        TeamDecision([[]], [Claim(2, ix["fa_w"], ix["r3"], bid=20)]),  # contested stash
        none,
    ]
    out = _with_winning_bid(season, first, 0, Move("claim", "fa_w", "r1", 5))
    assert [m.bid for m in out] == [5, 21]
    # Holding better priority than the top bidder, matching the bid is enough.
    rules2 = WaiverRules("faab", rules.budgets, 1, {1: 1, 2: 3, 3: 2, 4: 4})
    season2 = DynamicSeason(inp, models, rules2)
    assert [
        m.bid for m in _with_winning_bid(season2, first, 0, Move("claim", "fa_w", "r1", 5))
    ] == [5, 20]
