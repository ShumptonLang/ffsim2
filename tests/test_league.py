import pytest

from ffsim.league import (
    League,
    Player,
    PlayoffFormat,
    PlayoffRoundType,
    PlayoffSeedType,
    TeamNotFound,
    load_league_state,
    parse_matchups,
    slot_accepts,
)
from tests.conftest import LEAGUE_ID


@pytest.fixture
def state(client):
    return load_league_state(client, LEAGUE_ID)


def test_league_settings(state):
    lg = state.league
    assert lg.season == 2026 and lg.total_rosters == 4
    assert lg.is_superflex and lg.has_idp
    assert lg.bench_slots == 3
    assert "BN" not in lg.starter_slots and "IR" not in lg.starter_slots
    assert len(lg.starter_slots) == 14
    assert lg.unsupported_slots == ()
    assert lg.regular_season_weeks == range(1, 4)
    assert lg.last_scored_week == 2
    assert lg.previous_league_id == "800000000000000001"
    assert lg.playoffs.round_weeks() == [(4, 5)]  # two-week championship


@pytest.mark.parametrize(
    ("teams", "round_type", "byes", "weeks"),
    [
        (6, PlayoffRoundType.ONE_WEEK, 2, [(15,), (16,), (17,)]),
        (4, PlayoffRoundType.ONE_WEEK, 0, [(15,), (16,)]),
        (8, PlayoffRoundType.TWO_WEEKS_PER_ROUND, 0, [(15, 16), (17, 18), (19, 20)]),
        (6, PlayoffRoundType.TWO_WEEK_CHAMPIONSHIP, 2, [(15,), (16,), (17, 18)]),
    ],
)
def test_playoff_format(teams, round_type, byes, weeks):
    po = PlayoffFormat(teams, 15, round_type, PlayoffSeedType.FIXED)
    assert po.byes == byes
    assert po.round_weeks() == weeks
    assert po.last_week == weeks[-1][-1]


def test_superflex_detected_from_two_qb_slots():
    lg = League.from_sleeper({"league_id": "1", "roster_positions": ["QB", "QB", "RB", "BN"]})
    assert lg.is_superflex and not lg.has_idp


def test_unsupported_slot_reported():
    lg = League.from_sleeper({"league_id": "1", "roster_positions": ["QB", "HOLDER"]})
    assert lg.unsupported_slots == ("HOLDER",)


def test_teams_and_records(state):
    t1 = state.teams[1]
    assert t1.label == "Team Number 1" and t1.display_name == "manager1"
    assert t1.record.points_for == pytest.approx(236.40)
    assert t1.record.points_against == pytest.approx(199.90)
    assert state.teams[4].label == "manager4"  # no team_name -> display name
    t2 = state.teams[2]
    assert len(t2.reserve) == 1 and t2.reserve[0] not in t2.active_players


def test_schedule(state):
    assert set(state.schedule) == {1, 2, 3}
    m1 = state.matchup_for(1, 1)
    assert (m1.home, m1.away) == (1, 2) and m1.played
    assert m1.home_points == pytest.approx(121.4)
    m3 = state.matchup_for(1, 3)
    assert m3.opponent(1) == 4 and not m3.played
    assert state.remaining_regular_weeks == [3]


def test_parse_matchups_skips_unpaired_and_null_ids():
    rows = [
        {"roster_id": 1, "matchup_id": 1, "points": 90},
        {"roster_id": 2, "matchup_id": 1, "points": 80},
        {"roster_id": 3, "matchup_id": None, "points": 0},
        {"roster_id": 4, "matchup_id": 2, "points": 0},
    ]
    ms = parse_matchups(5, rows, scored=True)
    assert len(ms) == 1 and ms[0].home_points == 90


def test_standings_order(state):
    assert [t.roster_id for t in state.standings()] == [1, 3, 2, 4]


@pytest.mark.parametrize("query", ["1", "manager1", "MANAGER1", "Team Number 1", "u1", "number 1"])
def test_find_team(state, query):
    assert state.find_team(query).roster_id == 1


def test_find_team_errors(state):
    with pytest.raises(TeamNotFound, match="ambiguous"):
        state.find_team("Team Number")
    with pytest.raises(TeamNotFound, match="no team"):
        state.find_team("nobody")


def test_players(state):
    lb = state.player("1011")
    assert lb.fantasy_positions == ("LB",)
    assert state.player("BUF").name == "BUF City Defense"
    assert state.player("missing").name.startswith("Unknown")
    raw_cb = Player.from_sleeper("x", {"position": "CB", "first_name": "A", "last_name": "B"})
    assert raw_cb.fantasy_positions == ("DB",) and raw_cb.name == "A B"


def test_slot_eligibility():
    assert slot_accepts("SUPER_FLEX", ["QB"])
    assert not slot_accepts("FLEX", ["QB"])
    assert slot_accepts("IDP_FLEX", ["DB"])
    assert not slot_accepts("IDP_FLEX", ["K"])
    assert not slot_accepts("UNKNOWN", ["QB"])
