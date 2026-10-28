import pytest
from typer.testing import CliRunner

from ffsim import cli
from ffsim.data.sleeper import SleeperClient
from tests.conftest import LEAGUE_ID

runner = CliRunner()


@pytest.fixture(autouse=True)
def fixture_client(monkeypatch, tmp_path, transport):
    def factory(cache_dir=None, refresh=False):
        return SleeperClient(tmp_path / "cache", refresh=refresh, transport=transport)

    monkeypatch.setattr(cli, "SleeperClient", factory)
    monkeypatch.setenv("COLUMNS", "200")


def test_league_command():
    res = runner.invoke(cli.app, ["league", "-l", LEAGUE_ID, "-m", "manager1"])
    assert res.exit_code == 0, res.output
    assert "Fixture Superflex IDP League" in res.output
    assert "PPR, TE+0.5, IDP" in res.output and "Superflex" in res.output
    assert "weeks 4/5" in res.output
    assert "W 121.40-98.60" in res.output  # week 1 vs roster 2
    assert "W 115.00-101.30" in res.output  # week 2 vs roster 3


def test_roster_command_scores_projections():
    res = runner.invoke(cli.app, ["roster", "-l", LEAGUE_ID, "-m", "1", "-w", "3"])
    assert res.exit_code == 0, res.output
    # QB 1001: 265*.04 + 1.9*4 - 0.7*2 + 2.8 + 0.3*6 = 21.4
    assert "21.40" in res.output
    # DEF BUF: 2.6 + 1.6 + 0 (22 allowed) = 4.2
    assert "4.20" in res.output
    assert "Free Agent" not in res.output


def test_unknown_manager_exits_nonzero():
    res = runner.invoke(cli.app, ["league", "-l", LEAGUE_ID, "-m", "zzz"])
    assert res.exit_code == 1 and "no team matches" in res.output
