import os
import time

import pytest

from ffsim.data.projections import SleeperProjectionSource, parse_rows
from ffsim.data.sleeper import SleeperClient, SleeperError
from tests.conftest import LEAGUE_ID


def test_responses_are_cached(client, transport):
    client.league(LEAGUE_ID)
    client.league(LEAGUE_ID)
    assert len(transport.requests) == 1


def test_refresh_bypasses_cache(tmp_path, transport):
    with SleeperClient(tmp_path, transport=transport) as c:
        c.league(LEAGUE_ID)
    with SleeperClient(tmp_path, transport=transport, refresh=True) as c:
        c.league(LEAGUE_ID)
    assert len(transport.requests) == 2


def test_ttl_expiry(client, transport):
    client.rosters(LEAGUE_ID)
    for f in (client.cache_dir / "http").glob("*.json"):
        old = time.time() - 3600
        os.utime(f, (old, old))
    client.rosters(LEAGUE_ID)
    assert len(transport.requests) == 2


def test_final_weeks_never_expire(client, transport):
    client.matchups(LEAGUE_ID, 1, final=True)
    for f in (client.cache_dir / "http").glob("*.json"):
        os.utime(f, (0, 0))
    client.matchups(LEAGUE_ID, 1, final=True)
    assert len(transport.requests) == 1


def test_http_error_raises(client):
    with pytest.raises(SleeperError, match="404"):
        client.league("does-not-exist")


def test_projections_request_and_parse(client, transport):
    proj = SleeperProjectionSource(client).weekly_projections(2026, 3, positions=["QB", "DL"])
    req = transport.requests[-1]
    assert req.url.host == "api.sleeper.com"
    assert req.url.params.get_list("position[]") == ["QB", "DL"]
    assert req.url.params["season_type"] == "regular"
    qb = proj["1001"]
    assert qb.stats["pass_td"] == pytest.approx(1.9) and qb.opponent == "KC"


def test_parse_rows_dict_form_and_non_numeric():
    rows = {"42": {"rec": 3, "flag": True, "note": "x"}}
    assert parse_rows(rows, 2026, 1)["42"].stats == {"rec": 3.0}
