import numpy as np
import pytest

from ffsim.data.history import PlayerSeason
from ffsim.model.availability import SEASON_ENDING, AvailabilityModel, fit_availability


def test_status_probabilities():
    m = AvailabilityModel()
    assert m.play_prob(None) == 1.0
    assert m.play_prob("Questionable") == 0.75
    assert m.play_prob("Out") == 0.0
    assert m.play_prob("Weird") == 1.0
    assert m.hazard_for("DEF") == 0.0


def test_no_hazard_means_available_whenever_active():
    m = AvailabilityModel()
    active = np.array([[True, False], [True, True], [False, True]])
    av = m.sample(np.random.default_rng(0), 50, active, np.zeros(2), np.ones(2))
    assert av.shape == (50, 3, 2)
    assert np.array_equal(av, np.broadcast_to(active, av.shape))


def test_first_week_designation():
    m = AvailabilityModel()
    active = np.ones((2, 2), dtype=bool)
    av = m.sample(np.random.default_rng(1), 20_000, active, np.zeros(2), np.array([0.75, 0.0]))
    assert av[:, 0, 0].mean() == pytest.approx(0.75, abs=0.02)
    assert not av[:, 0, 1].any()
    assert av[:, 1].all()  # designation affects only the first week


def test_injury_durations():
    one_week = AvailabilityModel(durations=[1.0] + [0.0] * 10)
    active = np.ones((6, 1), dtype=bool)
    av = one_week.sample(np.random.default_rng(2), 10, active, np.ones(1), np.ones(1))
    # hurt every game he plays -> misses exactly the next one
    assert av[:, :, 0].tolist() == [[True, False, True, False, True, False]] * 10

    season = AvailabilityModel(durations=[0.0] * 10 + [1.0])
    av = season.sample(np.random.default_rng(3), 10, active, np.ones(1), np.ones(1))
    assert av[:, :, 0].tolist() == [[True] + [False] * 5] * 10


def test_injured_player_return_given_games_missed():
    m = AvailabilityModel()  # durations: 1 game 0.43, 2 games 0.24, 3 games 0.11, ...
    # out in the published week 0, projected again from week 1, one game missed
    active = np.array([[False, True], [True, True], [True, True], [True, True]])
    av = m.sample(
        np.random.default_rng(5), 20_000, active, np.zeros(2), np.ones(2),
        np.array([1, -1]), np.array([1, 0]),
    )  # fmt: skip
    pmf = np.array(m.durations) / sum(m.durations)
    # absence of exactly d games means back in week d
    assert av[:, 1, 0].mean() == pytest.approx(pmf[0], abs=0.015)
    assert av[:, 3, 0].mean() == pytest.approx(pmf[:3].sum(), abs=0.015)
    assert not (av[:, 2, 0] & ~av[:, 3, 0]).any()  # once back, stays back with no hazard
    assert av[:, :, 1].all()
    # three games missed already: only absences of 3+ remain, so returns are slower
    av3 = m.sample(
        np.random.default_rng(6), 20_000, active, np.zeros(2), np.ones(2),
        np.array([1, -1]), np.array([3, 0]),
    )  # fmt: skip
    assert av3[:, 1, 0].mean() == pytest.approx(pmf[2] / pmf[2:].sum(), abs=0.015)


def test_bye_weeks_do_not_count_toward_absence():
    m = AvailabilityModel(durations=[1.0] + [0.0] * 10)
    active = np.array([[True], [False], [True], [True]])
    av = m.sample(np.random.default_rng(4), 5, active, np.array([1.0]), np.ones(1))
    # hurt week 0, bye week 1, misses week 2 (next team game), back week 3
    assert av[:, :, 0].tolist() == [[True, False, False, True]] * 5


def test_fit_availability():
    seasons, team_weeks = {}, {}
    sched = list(range(1, 18))
    for i in range(600):
        weeks = {w: (10.0, 10.0, True) for w in sched}
        if i % 10 == 0:  # misses weeks 5-6
            weeks[5] = weeks[6] = (10.0, 0.0, False)
        if i % 30 == 1:  # gone from week 3 on
            for w in range(3, 18):
                weeks.pop(w)
        seasons[(2025, f"p{i}")] = PlayerSeason(2025, f"p{i}", "WR", "XX", weeks)
    team_weeks[(2025, "XX")] = sched
    m, diag = fit_availability(seasons, team_weeks)
    assert diag["WR"]["games"] > 3000
    assert 0.0 < m.hazard_for("WR") < 0.02
    d = dict(zip(list(range(1, 11)) + [SEASON_ENDING], m.durations, strict=True))
    assert d[2] == pytest.approx(60 / 80) and d[SEASON_ENDING] == pytest.approx(20 / 80)
