"""``ffsim`` command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ffsim import report
from ffsim.data.history import HistoryStore
from ffsim.data.projections import SleeperProjectionSource
from ffsim.data.sleeper import SleeperClient, SleeperError
from ffsim.league import LeagueState, Team, TeamNotFound, load_league_state
from ffsim.model.errors import coverage, fit_error_model, load_error_model, model_path
from ffsim.scoring import score
from ffsim.sim.engine import simulate
from ffsim.sim.inputs import build_inputs

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


@app.command()
def odds(
    league_id: LeagueOpt,
    manager: ManagerOpt = None,
    sims: Annotated[int, typer.Option("--sims", "-n", help="Number of simulations.")] = 10_000,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 0,
    family: Annotated[str, typer.Option("--family", help="Error model family.")] = "parametric",
    cache_dir: CacheOpt = None,
    refresh: RefreshOpt = False,
) -> None:
    """Playoff, bye and championship odds for every team."""
    client, state = _load(league_id, cache_dir, refresh)
    with client:
        team = _team(state, manager)
        try:
            bracket = client.winners_bracket(league_id)
        except SleeperError:
            bracket = []
        inputs = build_inputs(state, SleeperProjectionSource(client), bracket)
        model, fitted = load_error_model(client.cache_dir, state.league.scoring_settings, family)
    if not inputs.weeks:
        console.print("[yellow]Season is complete; nothing left to simulate.[/yellow]")
        raise typer.Exit(0)
    result = simulate(inputs, model, n_sims=sims, seed=seed)
    report.league_summary(state, console)
    console.print(
        f"Error model: {model.family}, "
        + ("fitted to this league's scoring" if fitted else "default (run `ffsim fit-errors`)")
    )
    console.print(report.odds_table(result, state, team))
    if team:
        console.print(report.seed_table(result, state, team))
        if inputs.regular_weeks:
            console.print(report.focal_weeks_table(result, inputs, team))
    console.print(
        "[dim]Milestone 2 baseline: players are simulated independently with no "
        "rest-of-season talent uncertainty, so odds are more concentrated than reality.[/dim]"
    )


@app.command("fit-errors")
def fit_errors(
    league_id: LeagueOpt,
    seasons: Annotated[
        str, typer.Option("--seasons", help="Comma-separated past seasons.")
    ] = "2023,2024,2025",
    family: Annotated[str, typer.Option("--family", help="parametric | bootstrap")] = "parametric",
    cache_dir: CacheOpt = None,
    refresh: RefreshOpt = False,
) -> None:
    """Fit the projection-error model to this league's scoring from past seasons."""
    client, state = _load(league_id, cache_dir, refresh)
    with client:
        years = [int(s) for s in seasons.split(",") if s.strip()]
        console.print(f"Loading {years} projections and stats (first run downloads ~1 min)…")
        rows = list(HistoryStore(client).rows(years))
        settings = state.league.scoring_settings
        model = fit_error_model(rows, settings, family, meta={"seasons": years})
        path = model_path(client.cache_dir, settings, family)
        model.save(path)
    console.print(report.error_model_table(model, coverage(model, rows, settings)))
    console.print(f"Saved to {path}")


if __name__ == "__main__":
    app()
