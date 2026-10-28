"""``ffsim`` command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ffsim import report
from ffsim.data.projections import SleeperProjectionSource
from ffsim.data.sleeper import SleeperClient, SleeperError
from ffsim.league import LeagueState, Team, TeamNotFound, load_league_state
from ffsim.scoring import score

app = typer.Typer(no_args_is_help=True, help="Monte Carlo championship odds for Sleeper leagues.")
console = Console()

LeagueOpt = Annotated[str, typer.Option("--league", "-l", help="Sleeper league ID.")]
ManagerOpt = Annotated[
    str | None,
    typer.Option("--manager", "-m", help="Display name, team name, user ID or roster ID."),
]
CacheOpt = Annotated[Path | None, typer.Option("--cache-dir", help="Cache directory.")]
RefreshOpt = Annotated[bool, typer.Option("--refresh", help="Ignore cached API responses.")]


def _load(
    league_id: str, cache_dir: Path | None, refresh: bool
) -> tuple[SleeperClient, LeagueState]:
    client = SleeperClient(cache_dir, refresh=refresh)
    try:
        return client, load_league_state(client, league_id)
    except SleeperError as exc:
        client.close()
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def _team(state: LeagueState, manager: str | None) -> Team | None:
    if manager is None:
        return None
    try:
        return state.find_team(manager)
    except TeamNotFound as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@app.command()
def league(
    league_id: LeagueOpt,
    manager: ManagerOpt = None,
    cache_dir: CacheOpt = None,
    refresh: RefreshOpt = False,
) -> None:
    """League format, playoff structure, standings and (with --manager) schedule."""
    client, state = _load(league_id, cache_dir, refresh)
    with client:
        team = _team(state, manager)
        report.league_summary(state, console)
        console.print(report.standings_table(state, team))
        if team:
            console.print(report.schedule_table(state, team))


@app.command()
def roster(
    league_id: LeagueOpt,
    manager: Annotated[str, typer.Option("--manager", "-m", help="Team to show.")],
    week: Annotated[
        int | None, typer.Option("--week", "-w", help="NFL week (default: current).")
    ] = None,
    cache_dir: CacheOpt = None,
    refresh: RefreshOpt = False,
) -> None:
    """A team's roster with Sleeper projections scored under this league's settings."""
    client, state = _load(league_id, cache_dir, refresh)
    with client:
        team = _team(state, manager)
        assert team is not None
        wk = week or max(state.nfl_week, state.league.last_scored_week + 1)
        source = SleeperProjectionSource(client)
        try:
            proj = source.weekly_projections(state.league.season, wk)
        except SleeperError as exc:
            console.print(f"[yellow]Projections unavailable: {exc}[/yellow]")
            proj = {}
        points = {
            pid: score(pw.stats, state.league.scoring_settings, state.player(pid).primary_position)
            for pid, pw in proj.items()
            if pid in team.players
        }
        console.print(report.roster_table(state, team, wk, points))


if __name__ == "__main__":
    app()
