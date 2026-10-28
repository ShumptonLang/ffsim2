"""Checks against the real Sleeper API. Opt in with FFSIM_LIVE_LEAGUE=<league_id>."""

import os

import pytest

from ffsim.data.projections import SleeperProjectionSource
from ffsim.data.sleeper import SleeperClient
from ffsim.league import load_league_state
from ffsim.scoring import score

LIVE_LEAGUE = os.environ.get("FFSIM_LIVE_LEAGUE")
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not LIVE_LEAGUE, reason="set FFSIM_LIVE_LEAGUE to run"),
]


def test_scoring_matches_sleeper_players_points(tmp_path):
    """Our scorer must reproduce Sleeper's own per-player points for a scored week."""
    with SleeperClient(tmp_path) as client:
        state = load_league_state(client, LIVE_LEAGUE)
        week = state.league.last_scored_week
        if week < 1:
            pytest.skip("no scored weeks yet")
        stats = SleeperProjectionSource(client).weekly_stats(state.league.season, week, final=True)
        rows = client.matchups(LIVE_LEAGUE, week, final=True)
        mismatches = []
        checked = 0
        for row in rows:
            for pid, official in (row.get("players_points") or {}).items():
                if pid not in stats:
                    continue
                ours = score(
                    stats[pid].stats, state.league.scoring_settings,
                    state.player(pid).primary_position,
                )  # fmt: skip
                checked += 1
                if abs(ours - official) > 0.011:
                    mismatches.append((pid, state.player(pid).name, ours, official))
        assert checked > 0
        assert not mismatches, mismatches[:10]
