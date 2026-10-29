"""``ffsim`` command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console

from ffsim import report
from ffsim.backtest import Replay, SeasonData, mock_replay, predict, sleeper_replay
from ffsim.calibrate import calibrated_models, leave_one_season_out
from ffsim.data.history import HistoryStore
from ffsim.data.projections import SleeperProjectionSource
from ffsim.data.sleeper import SleeperClient, SleeperError
from ffsim.decisions.common import Context, resolve_players
from ffsim.decisions.lineup import lineup_options
from ffsim.decisions.plan import Plan, make_plan
from ffsim.decisions.trades import apply_trade, evaluate_trade, find_trades
from ffsim.decisions.waivers import waiver_options
from ffsim.dynamic.decide import DynamicOptions, make_dynamic_plan, static_comparison
from ffsim.dynamic.rules import WaiverRules
from ffsim.dynamic.season import Move
from ffsim.league import League, LeagueState, Team, TeamNotFound, load_league_state
from ffsim.model.errors import coverage
from ffsim.model.models import ModelSet, fit_models, load_models, models_path
from ffsim.scoring import score
from ffsim.sim.engine import Evaluator, simulate
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
        try:
            models, fitted = load_models(client.cache_dir, state.league.scoring_settings, family)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        inputs = build_inputs(state, SleeperProjectionSource(client), bracket, models.talent)
    if not inputs.weeks:
        console.print("[yellow]Season is complete; nothing left to simulate.[/yellow]")
        raise typer.Exit(0)
    result = simulate(inputs, models, n_sims=sims, seed=seed)
    report.league_summary(state, console)
    console.print(
        f"Models: {models.errors.family}, "
        + ("fitted to this league's scoring" if fitted else "default (run `ffsim fit`)")
    )
    console.print(report.odds_table(result, state, team))
    if team:
        console.print(report.seed_table(result, state, team))
        if inputs.regular_weeks:
            console.print(report.focal_weeks_table(result, inputs, team))


@app.command()
def fit(
    league_id: LeagueOpt,
    seasons: Annotated[
        str, typer.Option("--seasons", help="Comma-separated past seasons.")
    ] = "2023,2024,2025",
    family: Annotated[str, typer.Option("--family", help="parametric | bootstrap")] = "parametric",
    cache_dir: CacheOpt = None,
    refresh: RefreshOpt = False,
) -> None:
    """Fit outcome, talent and availability models to this league's scoring."""
    client, state = _load(league_id, cache_dir, refresh)
    with client:
        years = [int(s) for s in seasons.split(",") if s.strip()]
        console.print(f"Loading {years} projections and stats (first run downloads ~1 min)…")
        store = HistoryStore(client)
        settings = state.league.scoring_settings
        models, diags = fit_models(store, years, settings, family)
        models.meta["league_id"] = league_id
        path = models_path(client.cache_dir, settings, family)
        models.save(path)
        cov = coverage(models.errors, store.rows(years), settings)
    console.print(report.error_model_table(models.errors, cov))
    console.print(report.talent_table(models, diags))
    console.print(report.correlation_table(diags["correlation"]))  # type: ignore[arg-type]
    console.print(report.reveal_table(models))
    console.print(f"Saved to {path}")


SimsOpt = Annotated[int, typer.Option("--sims", "-n", help="Number of simulations.")]
SeedOpt = Annotated[int, typer.Option("--seed", help="Random seed.")]
MeOpt = Annotated[str, typer.Option("--manager", "-m", help="Your team (name or roster ID).")]


def _context(
    league_id: str,
    manager: str,
    sims: int,
    seed: int,
    cache_dir: Path | None,
    refresh: bool,
    free_agents: int = 0,
) -> Context:
    """Load the league and draw shared simulations for decision commands."""
    client, state = _load(league_id, cache_dir, refresh)
    with client:
        team = _team(state, manager)
        assert team is not None
        try:
            bracket = client.winners_bracket(league_id)
        except SleeperError:
            bracket = []
        models, fitted = load_models(client.cache_dir, state.league.scoring_settings)
        inputs = build_inputs(
            state, SleeperProjectionSource(client), bracket, models.talent, free_agents
        )
    if not inputs.weeks:
        console.print("[yellow]Season is complete; nothing left to decide.[/yellow]")
        raise typer.Exit(0)
    if not fitted:
        console.print("[dim]Using default models; run `ffsim fit` for your league's scoring.[/dim]")
    with console.status(f"Simulating {sims:,} seasons…"):
        ev = Evaluator(inputs, models, sims, seed)
    return Context(state, ev, team)


def _faab(ctx: Context) -> tuple[bool, int]:
    lg = ctx.state.league
    is_faab = lg.waiver_type == 2
    return is_faab, max(0, lg.waiver_budget - ctx.team.waiver_budget_used)


@app.command()
def lineup(
    league_id: LeagueOpt,
    manager: MeOpt,
    week: Annotated[int | None, typer.Option("--week", "-w", help="Week to set.")] = None,
    sims: SimsOpt = 10_000,
    seed: SeedOpt = 0,
    cache_dir: CacheOpt = None,
    refresh: RefreshOpt = False,
) -> None:
    """Start/sit options ranked by change in championship probability."""
    ctx = _context(league_id, manager, sims, seed, cache_dir, refresh)
    wk = week or ctx.ev.inputs.weeks[0]
    console.print(report.lineup_table(lineup_options(ctx, wk), ctx.state, wk))
    console.print(_NOISE_NOTE)


@app.command()
def waivers(
    league_id: LeagueOpt,
    manager: MeOpt,
    top: Annotated[int, typer.Option("--top", help="Pickups to show.")] = 10,
    per_position: Annotated[
        int, typer.Option("--per-position", help="Free agents considered per position.")
    ] = 8,
    sims: SimsOpt = 10_000,
    seed: SeedOpt = 0,
    cache_dir: CacheOpt = None,
    refresh: RefreshOpt = False,
) -> None:
    """Free-agent pickups ranked by change in championship probability."""
    ctx = _context(league_id, manager, sims, seed, cache_dir, refresh, free_agents=per_position)
    faab, budget = _faab(ctx)
    with console.status("Evaluating pickups…"):
        opts = waiver_options(ctx, limit=top, faab=faab, budget_left=budget)
    console.print(report.waiver_table(opts, ctx.state, faab))
    if faab:
        console.print(
            f"[dim]FAAB left: ${budget}. Bids are a heuristic: they scale the remaining "
            "budget by the title-odds gain and by how many rivals would also gain.[/dim]"
        )
    console.print(_NOISE_NOTE)


@app.command()
def trade(
    league_id: LeagueOpt,
    manager: MeOpt,
    give: Annotated[str, typer.Option("--give", help="Comma-separated players you send.")],
    get: Annotated[str, typer.Option("--get", help="Comma-separated players you receive.")],
    partner: Annotated[
        str | None, typer.Option("--with", help="Trade partner (default: owner of --get).")
    ] = None,
    sims: SimsOpt = 10_000,
    seed: SeedOpt = 0,
    cache_dir: CacheOpt = None,
    refresh: RefreshOpt = False,
) -> None:
    """Evaluate a specific trade for both sides."""
    ctx = _context(league_id, manager, sims, seed, cache_dir, refresh)
    rosters = ctx.ev.inputs.rosters
    try:
        give_ids = resolve_players(ctx.state, give.split(","), rosters[ctx.rid])
        if partner:
            other = ctx.state.find_team(partner).roster_id
        else:
            everyone = [p for r, ps in rosters.items() if r != ctx.rid for p in ps]
            first = resolve_players(ctx.state, get.split(",")[:1], everyone)[0]
            other = next(r for r, ps in rosters.items() if first in ps)
        get_ids = resolve_players(ctx.state, get.split(","), rosters[other])
    except (LookupError, TeamNotFound) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    result = evaluate_trade(ctx, other, give_ids, get_ids)
    console.print(report.trade_detail(result, ctx.state, ctx.team))
    console.print(_NOISE_NOTE)


@app.command("find-trades")
def find_trades_cmd(
    league_id: LeagueOpt,
    manager: MeOpt,
    max_size: Annotated[int, typer.Option("--max-size", help="Max players per side.")] = 2,
    partner: Annotated[str | None, typer.Option("--with", help="Only this partner.")] = None,
    top: Annotated[int, typer.Option("--top", help="Trades to show.")] = 10,
    sims: SimsOpt = 10_000,
    seed: SeedOpt = 0,
    cache_dir: CacheOpt = None,
    refresh: RefreshOpt = False,
) -> None:
    """Search for trades that raise your title odds and the partner would accept."""
    ctx = _context(league_id, manager, sims, seed, cache_dir, refresh)
    partners = [ctx.state.find_team(partner).roster_id] if partner else None
    with console.status("Searching trades…"):
        results = find_trades(ctx, max_size=max_size, limit=top, partners=partners)
    if not results:
        console.print("No acceptable trade found that raises your title odds.")
        return
    console.print(report.trade_table(results, ctx.state))
    console.print(
        "[dim]Only offers that don't cost the partner perceived value or projected "
        "lineup points are considered.[/dim]"
    )
    console.print(_NOISE_NOTE)


@app.command()
def plan(
    league_id: LeagueOpt,
    manager: MeOpt,
    sims: SimsOpt = 10_000,
    seed: SeedOpt = 0,
    cache_dir: CacheOpt = None,
    refresh: RefreshOpt = False,
    dynamic: Annotated[
        bool,
        typer.Option(
            "--dynamic",
            help="Also rank moves with every manager claiming and dropping all season "
            "(slow: minutes to hours).",
        ),
    ] = False,
    fast: Annotated[
        bool, typer.Option("--fast", help="Dynamic: small, quick settings for a first look.")
    ] = False,
    paths: Annotated[
        int | None, typer.Option("--paths", help="Dynamic: held-out seasons for final scoring.")
    ] = None,
    iterations: Annotated[
        int | None, typer.Option("--iterations", help="Dynamic: policy-iteration rounds.")
    ] = None,
    workers: Annotated[int, typer.Option("--workers", help="Dynamic: worker processes.")] = 1,
    pool: Annotated[
        int, typer.Option("--free-agents", help="Dynamic: free agents per position simulated.")
    ] = 25,
) -> None:
    """This week's path to the title: odds, lineup, best pickups and trade ideas."""
    ctx = _context(league_id, manager, sims, seed, cache_dir, refresh, free_agents=8)
    faab, budget = _faab(ctx)
    with console.status("Building plan…"):
        p = make_plan(ctx, faab, budget)
    i = p.odds.index(ctx.rid)
    console.print(
        f"[bold]{ctx.team.label}[/bold]: title {100 * p.odds.champion[i]:.1f}% · "
        f"playoffs {100 * p.odds.playoffs[i]:.1f}% · bye {100 * p.odds.bye[i]:.1f}%"
    )
    wk = ctx.ev.inputs.weeks[0]
    console.print(report.lineup_table(p.lineup, ctx.state, wk))
    console.print(report.waiver_table(p.waivers, ctx.state, faab))
    if p.trades:
        console.print(report.trade_table(p.trades, ctx.state))
    console.print(_NOISE_NOTE)
    if not dynamic:
        return
    opts = DynamicOptions.fast() if fast else DynamicOptions()
    opts.seed, opts.workers = seed, workers
    if paths:
        opts.eval_paths = paths
    if iterations is not None:
        opts.iterations = iterations
    _dynamic_plan(ctx, p, league_id, cache_dir, refresh, opts, pool, faab)


def _dynamic_plan(
    ctx: Context,
    static: Plan,
    league_id: str,
    cache_dir: Path | None,
    refresh: bool,
    opts: DynamicOptions,
    pool: int,
    faab: bool,
) -> None:
    client, state = _load(league_id, cache_dir, refresh)
    with client:
        try:
            bracket = client.winners_bracket(league_id)
        except SleeperError:
            bracket = []
        models, _ = load_models(client.cache_dir, state.league.scoring_settings)
        inputs = build_inputs(state, SleeperProjectionSource(client), bracket, models.talent, pool)
    extra = [Move("claim", w.add, w.drop, w.bid or 0) for w in static.waivers]
    for tr in static.trades:
        new, _, _ = apply_trade(ctx, tr.partner, tr.give, tr.get)
        extra.append(
            Move(
                "trade",
                partner=tr.partner,
                give=tuple(tr.give),
                get=tuple(tr.get),
                rosters={r: tuple(ps) for r, ps in new.items()},
            )
        )
    rules = WaiverRules.from_league(state)
    with console.status("Dynamic plan…") as status:
        dp = make_dynamic_plan(
            inputs,
            models,
            rules,
            ctx.rid,
            opts,
            extra,
            progress=lambda msg: status.update(f"Dynamic plan: {msg}…"),
        )
        static_comparison(ctx, dp.options)
    i = dp.odds.index(ctx.rid)
    console.print(
        f"[bold]Dynamic[/bold] (managers claim and drop all season): title "
        f"{100 * dp.odds.champion[i]:.1f}% · playoffs {100 * dp.odds.playoffs[i]:.1f}% "
        f"· bye {100 * dp.odds.bye[i]:.1f}%"
    )
    console.print(report.dynamic_table(dp, ctx.state, faab))
    if dp.planned:
        console.print(report.planned_moves_table(dp, ctx.state))
    if dp.adds_per_position:
        per = ", ".join(f"{pos} {n:.1f}" for pos, n in dp.adds_per_position.items())
        console.print(f"[dim]Your adds per season under the policy: {per}.[/dim]")
    console.print(
        "[dim]Dynamic Δ: your first move this week, then the same policy every manager "
        "follows, vs. that policy from now; scored on seasons not used to pick the moves. "
        "Static Δ: the same move with rosters fixed for the rest of the season.[/dim]"
    )


@app.command()
def calibrate(
    league_id: LeagueOpt,
    seasons: Annotated[
        str, typer.Option("--seasons", help="Comma-separated past seasons.")
    ] = "2023,2024,2025",
    apply: Annotated[
        bool, typer.Option("--apply", help="Save models using the best family per position.")
    ] = False,
    cache_dir: CacheOpt = None,
    refresh: RefreshOpt = False,
) -> None:
    """Leave-one-season-out checks of distributions, blending and correlation."""
    client, state = _load(league_id, cache_dir, refresh)
    with client:
        years = [int(s) for s in seasons.split(",") if s.strip()]
        store = HistoryStore(client)
        settings = state.league.scoring_settings
        with console.status("Fitting and scoring each held-out season…"):
            rep = leave_one_season_out(store, years, settings)
        for table in report.calibration_tables(rep):
            console.print(table)
        choice = rep.choose_families()
        if apply:
            with console.status("Refitting on all seasons…"):
                models = calibrated_models(store, years, settings, choice)
            models.meta["league_id"] = league_id
            path = models_path(client.cache_dir, settings, "parametric")
            models.save(path)
            console.print(f"Saved models ({choice}) to {path}")
        else:
            console.print("[dim]Run with --apply to save models using the ✓ families.[/dim]")


@app.command()
def backtest(
    league_id: LeagueOpt,
    seasons: Annotated[
        str, typer.Option("--seasons", help="Comma-separated past seasons.")
    ] = "2023,2024,2025",
    leagues: Annotated[
        int, typer.Option("--leagues", help="Mock leagues per season (0 to skip).")
    ] = 10,
    cutoffs: Annotated[
        str, typer.Option("--cutoffs", help="Weeks after which to predict.")
    ] = "3,6,9,12",
    replay: Annotated[
        list[str] | None, typer.Option("--replay", help="Completed Sleeper league ID to replay.")
    ] = None,
    sims: SimsOpt = 2_000,
    refit: Annotated[
        bool, typer.Option("--refit/--no-refit", help="Fit models without the tested season.")
    ] = True,
    seed: SeedOpt = 0,
    cache_dir: CacheOpt = None,
    refresh: RefreshOpt = False,
) -> None:
    """Replay past seasons and score predicted playoff and title odds."""
    client, state = _load(league_id, cache_dir, refresh)
    with client:
        store = HistoryStore(client)
        years = [int(s) for s in seasons.split(",") if s.strip()]
        cuts = [int(c) for c in cutoffs.split(",") if c.strip()]
        rng = np.random.default_rng(seed)

        jobs: list[tuple[Replay, League]] = []
        for season in years:
            for k in range(leagues):
                data = SeasonData.load(store, season, state.league.scoring_settings)
                jobs.append(
                    (mock_replay(data, state.league, rng, f"mock {season} #{k + 1}"), state.league)
                )
        for rid in replay or []:
            try:
                jobs.append(sleeper_replay(client, rid))
            except SleeperError as exc:
                console.print(f"[red]Skipping {rid}: {exc}[/red]")

        model_cache: dict[tuple[int, tuple], ModelSet] = {}
        data_cache: dict[tuple[int, tuple], SeasonData] = {}
        preds = []
        with console.status("Replaying seasons…") as status:
            for rep, lg in jobs:
                settings = lg.scoring_settings
                key = (rep.season, tuple(sorted(settings.items())))
                if key not in data_cache:
                    data_cache[key] = SeasonData.load(store, rep.season, settings)
                if key not in model_cache:
                    train = [y for y in years if y != rep.season] if refit else []
                    if train:
                        status.update(f"Fitting models without {rep.season}…")
                        model_cache[key], _ = fit_models(store, train, settings)
                    else:
                        model_cache[key], _ = load_models(client.cache_dir, settings)
                for cut in cuts:
                    if cut + 1 >= rep.playoffs.start_week:
                        continue
                    status.update(f"{rep.label}: predicting after week {cut}…")
                    preds.append(predict(rep, data_cache[key], cut, model_cache[key], sims, seed))
    if not preds:
        console.print("Nothing to score.")
        raise typer.Exit(1)
    console.print(
        f"{len(jobs)} season replays in this league's format "
        f"({state.league.total_rosters} teams, {state.league.playoffs.teams}-team playoffs)."
    )
    for table in report.backtest_tables(preds):
        console.print(table)
    console.print(
        "[dim]Brier: mean squared error of the probabilities (lower is better). Skill: "
        "improvement over a records-only baseline that treats every remaining game as a "
        "coin flip. Mock leagues are drafted from preseason projections and played out with "
        "real NFL scores; rosters stay fixed all season. Intervals resample whole "
        "leagues.[/dim]"
    )


_NOISE_NOTE = (
    "[dim]Δ values are percentage points of probability from paired simulations "
    "(same random draws). Dimmed values are within two standard errors of zero.[/dim]"
)


if __name__ == "__main__":
    app()
