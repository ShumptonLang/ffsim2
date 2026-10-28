import numpy as np
import pytest

from ffsim import backtest as B
from ffsim.league import League, PlayoffFormat, PlayoffRoundType, PlayoffSeedType
from ffsim.model.models import ModelSet
from ffsim.sim.lineups import best_lineup

NFL = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
GROUPS = [("QB", 1, 17.0), ("RB", 2, 10.0), ("WR", 3, 11.0), ("TE", 1, 7.0), ("K", 1, 8.0)]


def synthetic_season(weeks=10, seed=0) -> B.SeasonData:
    """Six NFL teams; each team's players have fixed skill; real 'actuals' drawn per week."""
    rng = np.random.default_rng(seed)
    proj, actual, team = {}, {}, {}
    positions, prior = {}, {}
    skill = {}
    for t in NFL:
        positions[t] = "DEF"
        prior[t] = 7.0
        skill[t] = 7.0
        for pos, n, base in GROUPS:
            for k in range(n):
                pid = f"{t}_{pos}{k}"
                positions[pid] = pos
                skill[pid] = base * (1 - 0.15 * k) * float(np.exp(rng.normal(0, 0.25)))
                prior[pid] = skill[pid] * float(np.exp(rng.normal(0, 0.2)))
    opponents = {t: {} for t in NFL}
    for w in range(1, weeks + 1):
        order = NFL[w % 6 :] + NFL[: w % 6]
        pairs = [(order[0], order[1]), (order[2], order[3])] + (
            [] if w == 5 else [(order[4], order[5])]  # week 5: two teams on bye
        )
        proj[w], actual[w], team[w] = {}, {}, {}
        for a, b in pairs:
            opponents[a][w], opponents[b][w] = b, a
            for t in (a, b):
                for pid in [p for p in positions if p == t or p.startswith(t + "_")]:
                    proj[w][pid] = skill[pid]
                    actual[w][pid] = max(0.0, skill[pid] * float(np.exp(rng.normal(-0.1, 0.45))))
                    team[w][pid] = t
    return B.SeasonData(2030, proj, actual, team, opponents, positions, prior)


def small_league(teams=4, playoff_teams=2) -> League:
    return League.from_sleeper(
        {
            "league_id": "x",
            "total_rosters": teams,
            "roster_positions": ["QB", "RB", "WR", "WR", "FLEX", "K", "DEF", "BN", "BN"],
            "scoring_settings": {"rec": 1},
            "settings": {"playoff_week_start": 9, "playoff_teams": playoff_teams},
        }
    )


def test_round_robin():
    sched = B.round_robin([1, 2, 3, 4], range(1, 7))
    for pairs in sched.values():
        seen = [t for p in pairs for t in p]
        assert sorted(seen) == [1, 2, 3, 4]
    meetings = {frozenset(p) for w in (1, 2, 3) for p in sched[w]}
    assert len(meetings) == 6  # everyone meets everyone in n-1 weeks
    odd = B.round_robin([1, 2, 3], range(1, 4))
    assert all(len(p) == 1 for p in odd.values())


def test_mock_draft_builds_legal_full_rosters():
    data = synthetic_season()
    lg = small_league()
    slots = list(lg.starter_slots)
    rosters = B.mock_draft(data, 4, slots, 9, np.random.default_rng(0))
    taken = [p for ps in rosters.values() for p in ps]
    assert len(taken) == len(set(taken)) == 36
    for ps in rosters.values():
        lineup = best_lineup(slots, ps, {p: 1.0 for p in ps}, {p: (data.positions[p],) for p in ps})
        assert all(lineup), "every starting slot can be filled"
        assert sum(data.positions[p] == "K" for p in ps) == 1
        assert sum(data.positions[p] == "QB" for p in ps) <= 2


def test_realized_outcome_from_points():
    rp = B.Replay(
        label="t",
        season=2030,
        slots=["QB"],
        playoffs=PlayoffFormat(2, 3, PlayoffRoundType.ONE_WEEK, PlayoffSeedType.FIXED),
        capacity=1,
        team_ids=[1, 2, 3, 4],
        rosters={1: {1: [], 2: [], 3: [], 4: []}},
        schedule={1: [(1, 2), (3, 4)], 2: [(1, 3), (2, 4)]},
        points={
            1: {1: 100, 2: 90, 3: 80, 4: 70},
            2: {1: 100, 2: 95, 3: 90, 4: 60},
            3: {1: 50, 2: 120, 3: 0, 4: 0},
        },  # fmt: skip
    )
    made, champ = rp.realized()
    assert made == {1, 2} and champ == 2
    # no games left -> records-only baseline is certain about the playoffs
    po, ch = B.baseline_probs(rp, 2, 2_000, np.random.default_rng(0))
    assert po.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert ch[0] == pytest.approx(0.5, abs=0.05)


def test_inputs_at_uses_only_past_information():
    data = synthetic_season()
    lg = small_league()
    rp = B.mock_replay(data, lg, np.random.default_rng(1), "m")
    models = ModelSet.default()
    inp = B.inputs_at(rp, data, 4, models)
    assert inp.weeks == [5, 6, 7, 8, 9] and inp.regular_weeks == [5, 6, 7, 8]
    assert inp.base_half_wins.sum() == 2 * 4 * 2  # 4 weeks x 2 games x 2 half-wins
    # week 5 byes: players on bye teams get no projection or game
    bye_teams = {t for t in NFL if 5 not in data.opponents[t]}
    for p in inp.universe:
        team = data.last_team(p, 5)
        if team in bye_teams:
            assert p not in inp.means[5] and p not in inp.nfl_games[5]
    qb = next(p for p in inp.universe if data.positions[p] == "QB")
    assert inp.roles[5].get(qb) == "QB"


def test_predict_end_to_end():
    data = synthetic_season()
    rp = B.mock_replay(data, small_league(), np.random.default_rng(2), "m")
    pred = B.predict(rp, data, 4, ModelSet.default(), n_sims=800)
    assert pred.p_playoffs.sum() == pytest.approx(2.0)
    assert pred.p_champion.sum() == pytest.approx(1.0)
    assert pred.made.sum() == 2 and pred.won.sum() == 1


def test_scores_and_interval():
    p = np.array([0.9, 0.1])
    y = np.array([1.0, 0.0])
    assert B.brier(p, y) == pytest.approx(0.01)
    assert B.log_loss(np.array([1.0]), np.array([0.0])) == pytest.approx(-np.log(1e-3))
    preds = [
        B.Prediction(
            f"r{i}",
            2030,
            3,
            [1, 2],
            np.array([0.8, 0.2]),
            np.array([0.6, 0.4]),
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0]),
        )  # fmt: skip
        for i in range(5)
    ]
    sc = B.score_predictions(preds, "playoffs")
    assert sc.brier == pytest.approx(0.04) and sc.brier_base == pytest.approx(0.25)
    assert sc.skill == pytest.approx(1 - 0.04 / 0.25)
    lo, hi = B.skill_interval(preds, "playoffs")
    assert lo == pytest.approx(sc.skill) and hi == pytest.approx(sc.skill)
    bins = B.calibration_bins(preds)
    assert [(round(m, 2), o, n) for _, m, o, n in bins] == [(0.2, 0.0, 5), (0.8, 1.0, 5)]


def test_sleeper_replay(tmp_path):
    from ffsim.data.sleeper import SleeperClient
    from tests.conftest import FixtureTransport

    lid = "700"
    routes = {
        f"v1/league/{lid}": {
            "league_id": lid, "name": "Old", "season": "2025", "status": "complete",
            "total_rosters": 4, "roster_positions": ["QB", "WR", "BN"],
            "settings": {"playoff_week_start": 3, "playoff_teams": 2, "start_week": 1},
        },
        f"v1/league/{lid}/winners_bracket": [
            {"r": 1, "m": 1, "t1": 2, "t2": 3, "w": 3, "l": 2, "p": 1},
        ],
    }  # fmt: skip
    for w in (1, 2, 3):
        routes[f"v1/league/{lid}/matchups/{w}"] = [
            {"roster_id": r, "matchup_id": 1 if r < 3 else 2, "points": 100 + r * w,
             "players": [f"p{r}", f"x{w}"]}
            for r in (1, 2, 3, 4)
        ]  # fmt: skip
    with SleeperClient(tmp_path, transport=FixtureTransport(routes)) as c:
        rp, lg = B.sleeper_replay(c, lid)
    assert rp.team_ids == [1, 2, 3, 4]
    assert rp.regular_weeks == [1, 2]
    assert rp.schedule[1] == [(1, 2), (3, 4)]
    assert rp.roster_at(2)[1] == ["p1", "x2"]
    assert rp.realized() == ({2, 3}, 3)
