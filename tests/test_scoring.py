import pytest

from ffsim.scoring import profile, score, score_rounded

PPR = {
    "pass_yd": 0.04, "pass_td": 4, "pass_int": -2, "rush_yd": 0.1, "rush_td": 6,
    "rec": 1, "rec_yd": 0.1, "rec_td": 6, "fum_lost": -2,
}  # fmt: skip


def test_base_rule_ppr_qb_line():
    stats = {"pass_yd": 300, "pass_td": 2, "pass_int": 1, "rush_yd": 20, "pts_ppr": 99}
    assert score(stats, PPR) == pytest.approx(12 + 8 - 2 + 2)


def test_zero_weight_and_unknown_stats_ignored():
    assert score({"rec": 5, "rec_air_yd": 80}, {"rec": 0, "rec_yd": 0.1}) == 0


def test_te_premium_applies_only_to_tes():
    s = {**PPR, "bonus_rec_te": 0.5}
    stats = {"rec": 6, "rec_yd": 60}
    assert score(stats, s, position="TE") == pytest.approx(6 + 6 + 3)
    assert score(stats, s, position="WR") == pytest.approx(12)


def test_te_premium_not_double_counted_when_stat_line_has_it():
    s = {"rec": 1, "bonus_rec_te": 0.5}
    assert score({"rec": 4, "bonus_rec_te": 4}, s, position="TE") == pytest.approx(6)


def test_threshold_bonus_derived_from_yards():
    s = {"rush_yd": 0.1, "bonus_rush_yd_100": 3, "bonus_rush_rec_yd_150": 2}
    assert score({"rush_yd": 99}, s) == pytest.approx(9.9)
    assert score({"rush_yd": 100}, s) == pytest.approx(13)
    assert score({"rush_yd": 100, "rec_yd": 50}, s) == pytest.approx(15)


def test_threshold_bonus_not_double_counted():
    s = {"rush_yd": 0.1, "bonus_rush_yd_100": 3}
    assert score({"rush_yd": 120, "bonus_rush_yd_100": 1}, s) == pytest.approx(15)


DST = {
    "sack": 1, "int": 2, "def_td": 6, "pts_allow_0": 10, "pts_allow_1_6": 7,
    "pts_allow_7_13": 4, "pts_allow_14_20": 1, "pts_allow_21_27": 0,
    "pts_allow_28_34": -1, "pts_allow_35p": -4, "yds_allow_0_100": 5, "yds_allow_550p": -7,
}  # fmt: skip


@pytest.mark.parametrize(
    ("pts", "expected"),
    [(0, 10), (6, 7), (7, 4), (13.9, 4), (20, 1), (27, 0), (34, -1), (35, -4), (52, -4)],
)
def test_points_allowed_tier_derived(pts, expected):
    assert score({"pts_allow": pts}, DST) == pytest.approx(expected)


def test_yards_allowed_tiers():
    assert score({"yds_allow": 80}, DST) == pytest.approx(5)
    assert score({"yds_allow": 600}, DST) == pytest.approx(-7)
    assert score({"yds_allow": 300}, DST) == 0


def test_tier_flags_in_stat_line_take_precedence():
    # Actual stat lines may carry the tier flag; it must not be added twice.
    assert score({"pts_allow": 10, "pts_allow_7_13": 1, "sack": 3}, DST) == pytest.approx(7)


def test_idp_scoring():
    s = {"idp_tkl_solo": 1, "idp_tkl_ast": 0.5, "idp_sack": 4, "idp_int": 3, "idp_pass_def": 1.5}
    stats = {"idp_tkl_solo": 6, "idp_tkl_ast": 3, "idp_sack": 1, "idp_pass_def": 2}
    assert score(stats, s, position="LB") == pytest.approx(6 + 1.5 + 4 + 3)


def test_negative_kicker_and_rounding():
    s = {"fgm_40_49": 4, "xpm": 1, "fgmiss": -1}
    assert score_rounded({"fgm_40_49": 1, "xpm": 2, "fgmiss": 2}, s) == 4.0
    assert score_rounded({"rec_yd": 13.333}, {"rec_yd": 0.1}) == 1.33


def test_profile_label():
    p = profile({**PPR, "bonus_rec_te": 0.5, "idp_sack": 4})
    assert p.label == "PPR, TE+0.5, IDP"
    assert profile({"rec": 0.5, "pass_td": 6}).label == "Half-PPR, 6pt pass TD"


KICK = {"fgm_40_49": 4, "fgm_50_59": 5, "fgm_60p": 6, "fgmiss": -1, "xpm": 1}


def test_kicker_projection_buckets_mapped_to_league_keys():
    proj = {"fgm_40_49": 0.5, "fgm_50p": 0.3, "fgmiss_40_49": 0.1, "fgmiss_50p": 0.2, "xpm": 2}
    assert score(proj, KICK) == pytest.approx(2 + 1.5 - 0.3 + 2)


def test_kicker_actual_line_not_double_counted():
    act = {"fgm_50_59": 1, "fgm_50p": 1, "fgmiss": 1, "fgmiss_50_59": 1, "fgmiss_50p": 1}
    assert score(act, KICK) == pytest.approx(5 - 1)
    assert score({"fgm_50_59": 1, "fgm_60p": 1}, {"fgm_50p": 5}) == pytest.approx(10)
