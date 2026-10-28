import numpy as np

from ffsim.league import PlayoffFormat, PlayoffRoundType, PlayoffSeedType
from ffsim.sim.playoffs import bracket_order, play_bracket, seed_teams


def fmt(teams, seed_type=PlayoffSeedType.FIXED):
    return PlayoffFormat(teams, 15, PlayoffRoundType.ONE_WEEK, seed_type)


def test_bracket_order():
    assert bracket_order(4) == [1, 4, 2, 3]
    assert bracket_order(8) == [1, 8, 4, 5, 2, 7, 3, 6]


def test_seeding_wins_then_points_for():
    half_wins = np.array([[10, 12, 10, 8]])
    pf = np.array([[900.0, 800.0, 950.0, 1000.0]])
    assert seed_teams(half_wins, pf).tolist() == [[1, 2, 0, 3]]


def test_division_winners_take_top_seeds():
    half_wins = np.array([[12, 10, 8, 4]])
    pf = np.zeros((1, 4))
    # team 3 wins division 2 despite the worst record
    assert seed_teams(half_wins, pf, [1, 1, 1, 2]).tolist() == [[0, 3, 1, 2]]


def scores_by_seed(order_strength, n_rounds):
    """Team i's score is its strength in every round."""
    s = np.array([order_strength], dtype=float)
    return [s.copy() for _ in range(n_rounds)]


def test_favorites_win_out():
    seeds = np.array([[0, 1, 2, 3, 4, 5]])
    strength = [60, 50, 40, 30, 20, 10]
    finalists, champ = play_bracket(seeds, scores_by_seed(strength, 3), fmt(6))
    assert sorted(finalists[0].tolist()) == [0, 1] and champ[0] == 0


def test_fixed_vs_reseed_after_upset():
    # seed 6 (team 5) upsets seed 3; seed 6 is strong enough to beat seed 2 but not seed 1.
    seeds = np.array([[0, 1, 2, 3, 4, 5]])
    strength = [100, 50, 10, 30, 20, 60]
    f_fixed, c_fixed = play_bracket(seeds, scores_by_seed(strength, 3), fmt(6))
    # fixed: 2 plays 6 (6 wins), 1 plays W(4,5) -> final 1 v 6
    assert sorted(f_fixed[0].tolist()) == [0, 5] and c_fixed[0] == 0
    f_re, c_re = play_bracket(seeds, scores_by_seed(strength, 3), fmt(6, PlayoffSeedType.RESEED))
    # reseed: 1 plays 6 (1 wins), 2 plays 4 (2 wins) -> final 1 v 2
    assert sorted(f_re[0].tolist()) == [0, 1] and c_re[0] == 0


def test_forced_winners_override_scores():
    seeds = np.array([[0, 1, 2, 3]])
    strength = [40, 30, 20, 10]
    finalists, champ = play_bracket(
        seeds, scores_by_seed(strength, 2), fmt(4), forced_winners={0: {3, 1}}
    )
    assert sorted(finalists[0].tolist()) == [1, 3] and champ[0] == 1


def test_vectorized_over_sims():
    seeds = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    s = np.array([[40, 30, 20, 10], [40, 30, 20, 10]], dtype=float)
    _, champ = play_bracket(seeds, [s, s], fmt(4))
    assert champ.tolist() == [0, 0]
