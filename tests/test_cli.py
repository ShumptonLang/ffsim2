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


def test_odds_command():
    res = runner.invoke(cli.app, ["odds", "-l", LEAGUE_ID, "-m", "manager1", "-n", "2000"])
    assert res.exit_code == 0, res.output
    assert "Season outlook (2,000 simulations)" in res.output
    assert "#1" in res.output and "Out" in res.output  # seed distribution table
    assert "default (run `ffsim fit`)" in res.output


@pytest.mark.parametrize(
    "args",
    [
        ["lineup", "-l", LEAGUE_ID, "-m", "manager1", "-n", "300"],
        ["waivers", "-l", LEAGUE_ID, "-m", "manager1", "-n", "300", "--top", "3"],
        ["trade", "-l", LEAGUE_ID, "-m", "1", "-n", "300", "--give", "T1 WR4", "--get", "T2 WR4"],
        ["find-trades", "-l", LEAGUE_ID, "-m", "1", "-n", "300", "--max-size", "1"],
        ["plan", "-l", LEAGUE_ID, "-m", "1", "-n", "300"],
    ],
)
def test_decision_commands_run(args):
    res = runner.invoke(cli.app, args)
    assert res.exit_code == 0, res.output
    assert "Δ" in res.output or "No acceptable trade" in res.output


def test_plan_dynamic_shows_both_engines():
    res = runner.invoke(
        cli.app,
        ["plan", "-l", LEAGUE_ID, "-m", "1", "-n", "300", "--dynamic", "--fast", "--paths", "200"],
    )
    assert res.exit_code == 0, res.output
    assert "Waiver pickups" in res.output  # the static plan is unchanged and still shown
    assert "Dynamic (managers claim and drop all season)" in res.output
    assert "Δ Title (dynamic)" in res.output and "Δ Title (static)" in res.output
    assert "Policy:" in res.output


def test_waivers_lists_free_agent():
    res = runner.invoke(cli.app, ["waivers", "-l", LEAGUE_ID, "-m", "1", "-n", "300"])
    assert "Free Agent" in res.output and "(open spot)" in res.output


def test_trade_reports_both_sides():
    res = runner.invoke(
        cli.app,
        ["trade", "-l", LEAGUE_ID, "-m", "1", "-n", "300", "--give", "T1 WR4", "--get", "T2 WR4"],
    )
    assert "Trade with Team Number 2" in res.output
    bad = runner.invoke(
        cli.app, ["trade", "-l", LEAGUE_ID, "-m", "1", "--give", "nobody", "--get", "T2 WR4"]
    )
    assert bad.exit_code == 1 and "matches 0" in bad.output


def test_backtest_command(monkeypatch):
    from ffsim import backtest as B
    from ffsim.model.models import ModelSet
    from tests.test_backtest import small_league, synthetic_season

    monkeypatch.setattr(cli.SeasonData, "load", classmethod(lambda cls, *a: synthetic_season()))
    monkeypatch.setattr(cli, "fit_models", lambda *a, **k: (ModelSet.default(), {}))
    monkeypatch.setattr(
        cli,
        "mock_replay",
        lambda data, lg, rng, label: B.mock_replay(data, small_league(), rng, label),
    )
    res = runner.invoke(
        cli.app,
        [
            "backtest",
            "-l",
            LEAGUE_ID,
            "--seasons",
            "2030",
            "--leagues",
            "2",
            "--cutoffs",
            "3,5",
            "-n",
            "300",
        ],  # fmt: skip
    )
    assert res.exit_code == 0, res.output
    assert "Backtest: P(playoffs)" in res.output and "Calibration of P(playoffs)" in res.output
