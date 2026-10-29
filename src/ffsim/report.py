"""Rich-table rendering for CLI output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from ffsim.backtest import Prediction, calibration_bins, score_predictions, skill_interval
from ffsim.calibrate import CalibrationReport
from ffsim.decisions.lineup import LineupOption
from ffsim.decisions.trades import TradeResult
from ffsim.decisions.waivers import WaiverOption
from ffsim.dynamic.decide import DynamicPlan
from ffsim.dynamic.season import Move
from ffsim.league import LeagueState, Team
from ffsim.model.errors import ErrorModel
from ffsim.model.models import ModelSet
from ffsim.scoring import profile
from ffsim.sim.engine import SimResult
from ffsim.sim.inputs import SimInputs


def league_summary(state: LeagueState, console: Console) -> None:
    lg = state.league
    po = lg.playoffs
    console.print(f"[bold]{lg.name}[/bold]  ({lg.league_id}, {lg.season}, status: {lg.status})")
    console.print(
        f"{lg.total_rosters} teams · {profile(lg.scoring_settings).label}"
        + (" · Superflex" if lg.is_superflex else "")
    )
    console.print("Starters: " + " ".join(lg.starter_slots) + f"  · Bench {lg.bench_slots}")
    if lg.unsupported_slots:
        console.print(f"[yellow]Unsupported slots: {', '.join(lg.unsupported_slots)}[/yellow]")
    rounds = " → ".join("/".join(str(w) for w in r) for r in po.round_weeks())
    console.print(
        f"Playoffs: {po.teams} teams ({po.byes} byes), {po.seed_type.name.lower()} seeding, "
        f"weeks {rounds}"
    )
    console.print(
        f"Last scored week: {lg.last_scored_week} · NFL week: {state.nfl_week} · "
        f"remaining regular-season weeks: {state.remaining_regular_weeks or 'none'}"
    )


def standings_table(state: LeagueState, focal: Team | None = None) -> Table:
    t = Table(title="Standings")
    for col in ("#", "Team", "Manager", "W-L-T", "PF", "PA"):
        t.add_column(col, justify="right" if col in {"#", "PF", "PA"} else "left")
    for i, team in enumerate(state.standings(), 1):
        r = team.record
        style = "bold cyan" if focal and team.roster_id == focal.roster_id else None
        t.add_row(
            str(i),
            team.label,
            team.display_name,
            f"{r.wins}-{r.losses}-{r.ties}",
            f"{r.points_for:.2f}",
            f"{r.points_against:.2f}",
            style=style,
        )
    return t


def schedule_table(state: LeagueState, team: Team) -> Table:
    t = Table(title=f"{team.label}: regular-season schedule")
    for col in ("Week", "Opponent", "Result"):
        t.add_column(col)
    for week in state.league.regular_season_weeks:
        m = state.matchup_for(team.roster_id, week)
        if m is None:
            t.add_row(str(week), "—", "")
            continue
        opp = state.teams[m.opponent(team.roster_id)]  # type: ignore[index]
        result = ""
        if m.played:
            mine, theirs = (
                (m.home_points, m.away_points)
                if m.home == team.roster_id
                else (m.away_points, m.home_points)
            )
            wl = "W" if mine > theirs else "L" if mine < theirs else "T"  # type: ignore[operator]
            result = f"{wl} {mine:.2f}-{theirs:.2f}"
        t.add_row(str(week), opp.label, result)
    return t


def roster_table(
    state: LeagueState,
    team: Team,
    week: int,
    projected: Mapping[str, float] | None = None,
) -> Table:
    title = f"{team.label}: roster" + (f" (week {week} projections)" if projected else "")
    t = Table(title=title)
    for col in ("Player", "Pos", "NFL", "Status", "Slot", "Proj"):
        t.add_column(col, justify="right" if col == "Proj" else "left")
    starters = set(team.starters)

    def sort_key(pid: str) -> tuple[int, float]:
        return (0 if pid in starters else 1, -(projected or {}).get(pid, 0.0))

    for pid in sorted(team.players, key=sort_key):
        p = state.player(pid)
        if pid in team.reserve:
            slot = "IR"
        elif pid in team.taxi:
            slot = "TAXI"
        else:
            slot = "START" if pid in starters else "BN"
        proj = f"{projected[pid]:.2f}" if projected and pid in projected else ""
        t.add_row(
            p.name,
            "/".join(p.fantasy_positions) or (p.position or "?"),
            p.team or "FA",
            p.injury_status or "",
            slot,
            proj,
        )
    return t


def odds_table(result: SimResult, state: LeagueState, focal: Team | None = None) -> Table:
    t = Table(title=f"Season outlook ({result.n_sims:,} simulations)")
    for col in ("Team", "Record", "Proj W", "Playoffs", "Bye", "Final", "Champion", "±"):
        t.add_column(col, justify="left" if col in {"Team", "Record"} else "right")
    for i in np.argsort(-result.champion, kind="stable"):
        team = state.teams[result.team_ids[i]]
        r = team.record
        style = "bold cyan" if focal and team.roster_id == focal.roster_id else None
        t.add_row(
            team.label,
            f"{r.wins}-{r.losses}-{r.ties}",
            f"{result.mean_wins[i]:.1f}",
            _pct(result.playoffs[i]),
            _pct(result.bye[i]),
            _pct(result.final[i]),
            _pct(result.champion[i]),
            f"{100 * result.se(result.champion)[i]:.1f}",
            style=style,
        )
    return t


def focal_weeks_table(result: SimResult, inputs: SimInputs, focal: Team) -> Table:
    ti = inputs.team_index(focal.roster_id)
    t = Table(title=f"{focal.label}: remaining regular season")
    for col in ("Week", "Opponent", "Proj pts", "Opp proj", "P(win)"):
        t.add_column(col, justify="left" if col in {"Week", "Opponent"} else "right")
    for w in inputs.regular_weeks:
        opp = next(
            (b if a == ti else a for a, b in inputs.matchups.get(w, []) if ti in (a, b)), None
        )
        t.add_row(
            str(w),
            inputs.team_labels[opp] if opp is not None else "—",
            f"{result.week_points[w][ti]:.1f}",
            f"{result.week_points[w][opp]:.1f}" if opp is not None else "",
            _pct(result.week_win[w][ti]),
        )
    return t


def seed_table(result: SimResult, state: LeagueState, focal: Team) -> Table:
    i = result.index(focal.roster_id)
    probs = result.seed_probs[i]
    t = Table(title=f"{focal.label}: playoff seed distribution")
    for k in range(len(probs)):
        t.add_column(f"#{k + 1}", justify="right")
    t.add_column("Out", justify="right")
    t.add_row(*[_pct(p) for p in probs], _pct(1 - probs.sum()))
    return t


def _pct(p: float) -> str:
    if p <= 0:
        return "0%"
    if p < 0.001:
        return "<0.1%"
    return f"{100 * p:.1f}%"


def error_model_table(model: ErrorModel, cov: dict[str, dict[str, float]]) -> Table:
    t = Table(title=f"Projection-error model ({model.family})")
    for col in ("Pos", "Bucket", "n", "Center", "Bias", "Below p10", "p50", "p90"):
        t.add_column(col, justify="left" if col in {"Pos", "Bucket"} else "right")
    for pos, pm in model.positions.items():
        c = cov.get(pos, {})
        for b, d in enumerate(pm.diagnostics):
            bias = d.get("mean_ratio", d.get("mean_resid", 0.0))
            first = b == 0
            t.add_row(
                pos if first else "",
                f"{d['lo']:.1f}–{d['hi']:.1f}",
                f"{int(d['n'])}",
                f"{d['center']:.1f}",
                f"{bias:.2f}" + ("x" if pm.kind == "ratio" else ""),
                _pct(c["q10"]) if first and c else "",
                _pct(c["q50"]) if first and c else "",
                _pct(c["q90"]) if first and c else "",
            )
    return t


def talent_table(models: ModelSet, diags: Mapping[str, Any]) -> Table:
    t = Table(title="Talent updating, persistence and availability")
    for col in ("Pos", "Blend λ", "Prior k", "Next-wk MSE", "Persistence ρ", "Miss next game"):
        t.add_column(col, justify="left" if col == "Pos" else "right")
    talent, avail = diags.get("talent", {}), diags.get("availability", {})
    for pos in sorted(set(talent) | set(avail)):
        lam, k = models.talent.lam_k(pos)
        d = talent.get(pos, {})
        mse = f"{d['mse_sleeper']:.1f} → {d['mse_blend']:.1f}" if "mse_sleeper" in d else ""
        t.add_row(
            pos,
            f"{lam:.2f}",
            f"{k:g}",
            mse,
            f"{models.talent.rho_for(pos):.2f}",
            _pct(models.availability.hazard_for(pos)),
        )
    return t


KEY_PAIRS = [
    ("QB", "WR1", True), ("QB", "WR2", True), ("QB", "TE1", True), ("QB", "RB1", True),
    ("RB1", "RB2", True), ("WR1", "WR2", True), ("DEF", "RB1", True), ("K", "QB", True),
    ("QB", "QB", False), ("WR1", "WR1", False), ("QB", "WR1", False), ("DEF", "QB", False),
    ("DEF", "RB1", False), ("DEF", "DEF", False), ("LB1", "RB1", False), ("DB1", "WR1", False),
]  # fmt: skip


def correlation_table(diags: list[dict[str, Any]]) -> Table:
    t = Table(title="Within-game correlation (latent residuals)")
    for col in ("Pair", "Side", "Observed", "Model", "n"):
        t.add_column(col, justify="left" if col in {"Pair", "Side"} else "right")
    by_key = {(d["a"], d["b"], d["same_team"]): d for d in diags}
    for a, b, same in KEY_PAIRS:
        lo, hi = sorted((a, b))
        d = by_key.get((lo, hi, same))
        if d is None:
            continue
        # "Model" shows the weekly correlation the sim produces including talent attenuation.
        attenuated = d["implied"] * d["observed"] / d["target"] if d["target"] else d["implied"]
        t.add_row(
            f"{a}–{b}",
            "same team" if same else "opponents",
            f"{d['observed']:+.2f}",
            f"{attenuated:+.2f}",
            f"{d['n']:,}",
        )
    return t


def fmt_delta(d: float, se: float) -> str:
    """Percentage-point change with its standard error; dimmed if within noise."""
    text = f"{100 * d:+.2f} ± {100 * se:.2f}"
    return f"[dim]{text}[/dim]" if abs(d) < 2 * se or (d == 0 and se == 0) else text


def lineup_table(options: list[LineupOption], state: LeagueState, week: int) -> Table:
    t = Table(title=f"Week {week} lineup options (Δ vs. projection-optimal lineup)")
    for col in ("Option", "Δ Title odds (pts)", "Δ Playoffs", "P(win wk)", "Avg", "10th–90th"):
        t.add_column(col, justify="left" if col == "Option" else "right")
    for o in options:
        t.add_row(
            o.label,
            fmt_delta(o.delta.champion, o.delta.champion_se),
            fmt_delta(o.delta.playoffs, o.delta.playoffs_se),
            _pct(o.p_win),
            f"{o.mean:.1f}",
            f"{o.p10:.0f}–{o.p90:.0f}",
            style="bold" if o.is_baseline else None,
        )
    return t


def waiver_table(options: list[WaiverOption], state: LeagueState, faab: bool) -> Table:
    t = Table(title="Waiver pickups (Δ title odds vs. no move)")
    cols = ["Add", "Drop", "Δ Title odds (pts)", "Δ Playoffs", "ROS pts +/−"]
    if faab:
        cols += ["Rivals", "Bid"]
    for col in cols:
        t.add_column(col, justify="left" if col in {"Add", "Drop"} else "right")
    for o in options:
        row = [
            _player_label(state, o.add),
            _player_label(state, o.drop) if o.drop else "(open spot)",
            fmt_delta(o.delta.champion, o.delta.champion_se),
            fmt_delta(o.delta.playoffs, o.delta.playoffs_se),
            f"{o.ros_add:.0f} / {o.ros_drop:.0f}",
        ]
        if faab:
            row += [str(len(o.rivals)) if o.bid is not None else "", f"${o.bid}" if o.bid else ""]
        t.add_row(*row)
    return t


def trade_table(results: list[TradeResult], state: LeagueState) -> Table:
    t = Table(title="Trade ideas (Δ title odds; both sides)")
    for col in (
        "Partner",
        "You give",
        "You get",
        "Δ You",
        "Δ Them",
        "Δ Your lineup",
        "Δ Their lineup",
    ):
        t.add_column(col, justify="left" if col in {"Partner", "You give", "You get"} else "right")
    for r in results:
        t.add_row(
            state.teams[r.partner].label,
            ", ".join(state.player(p).name for p in r.give),
            ", ".join(state.player(p).name for p in r.get),
            fmt_delta(r.me.champion, r.me.champion_se),
            fmt_delta(r.partner_delta.champion, r.partner_delta.champion_se),
            f"{r.my_lineup_delta:+.0f}",
            f"{r.partner_lineup_delta:+.0f}",
        )
    return t


def trade_detail(r: TradeResult, state: LeagueState, me: Team) -> Table:
    partner = state.teams[r.partner]
    t = Table(title=f"Trade with {partner.label}")
    t.add_column("")
    t.add_column(me.label, justify="right")
    t.add_column(partner.label, justify="right")
    names = lambda ps: ", ".join(state.player(p).name for p in ps) or "—"  # noqa: E731
    t.add_row("Receives", names(r.get), names(r.give))
    t.add_row("Auto-drops (roster limit)", names(r.my_drops), names(r.partner_drops))
    t.add_row(
        "Δ Title odds (pts)",
        fmt_delta(r.me.champion, r.me.champion_se),
        fmt_delta(r.partner_delta.champion, r.partner_delta.champion_se),
    )
    t.add_row(
        "Δ Playoff odds (pts)",
        fmt_delta(r.me.playoffs, r.me.playoffs_se),
        fmt_delta(r.partner_delta.playoffs, r.partner_delta.playoffs_se),
    )
    t.add_row(
        "Δ Projected lineup pts (rest of season)",
        f"{r.my_lineup_delta:+.1f}",
        f"{r.partner_lineup_delta:+.1f}",
    )
    return t


def reveal_table(models: ModelSet) -> Table:
    rev = models.reveal
    t = Table(title="Projection news still to come (sd of log projection; used by --dynamic)")
    for col in ("Pos", "1 wk out", "3 wks out", "Not yet projected, 1 wk", "3 wks"):
        t.add_column(col, justify="left" if col == "Pos" else "right")
    for pos in sorted(set(rev.sigma2) | set(rev.sigma2_unpub)):
        t.add_row(
            pos,
            *(f"{100 * rev.var(pos, k, pub) ** 0.5:.0f}%" for pub in (True, False) for k in (1, 3)),
        )
    return t


def _move_label(move: Move, state: LeagueState) -> str:
    if move.kind == "none":
        return "No move this week"
    if move.kind == "trade":
        give = ", ".join(state.player(p).name for p in move.give)
        get = ", ".join(state.player(p).name for p in move.get)
        partner = state.teams[move.partner].label if move.partner in state.teams else "?"
        return f"Trade {give} to {partner} for {get}"
    text = f"Add {_player_label(state, move.add or '')}"
    return text + (f", drop {state.player(move.drop).name}" if move.drop else " (open spot)")


def dynamic_table(plan: DynamicPlan, state: LeagueState, faab: bool) -> Table:
    t = Table(
        title=f"This week's moves with managers acting all season "
        f"(Δ title odds vs. the policy, {plan.n_paths:,} held-out seasons)"
    )
    cols = ["Move", "Δ Title (dynamic)", "Δ Title (static)", "Δ Playoffs", "Claim won"]
    if faab:
        cols.append("Bid")
    for col in cols:
        t.add_column(col, justify="left" if col == "Move" else "right")
    for o in plan.options:
        if o.policy_moves is not None:
            what = "; ".join(_move_label(m, state) for m in o.policy_moves) or "no move"
            label = f"Policy: {what}"
            bid = ", ".join(f"${m.bid}" for m in o.policy_moves) if faab else ""
        else:
            label = _move_label(o.move, state)
            bid = f"${o.move.bid}" if o.move.kind == "claim" else ""
        static = (
            fmt_delta(o.static_champion, o.static_se or 0.0)
            if o.static_champion is not None
            else "—"
        )
        row = [
            label,
            fmt_delta(o.delta.champion, o.delta.champion_se),
            static,
            fmt_delta(o.delta.playoffs, o.delta.playoffs_se),
            _pct(o.p_won) if o.p_won is not None else "",
        ]
        if faab:
            row.append(bid)
        t.add_row(*row, style="bold" if o.is_policy else None)
    return t


def planned_moves_table(plan: DynamicPlan, state: LeagueState) -> Table:
    t = Table(title="What the policy goes on to do for you (baseline seasons)")
    for col in ("Adds", "Share of seasons", "Typical week"):
        t.add_column(col, justify="left" if col == "Adds" else "right")
    for a in plan.planned:
        t.add_row(_player_label(state, a.player), _pct(a.share), str(a.median_week))
    return t


def _player_label(state: LeagueState, pid: str) -> str:
    p = state.player(pid)
    tag = "/".join(p.fantasy_positions) or (p.position or "?")
    return f"{p.name} ({tag}, {p.team or 'FA'})"


def calibration_tables(rep: CalibrationReport) -> list[Table]:
    seasons = ", ".join(str(x) for x in rep.seasons)
    dist = Table(title=f"Outcome distributions, held out one season at a time ({seasons})")
    for col in ("Pos", "n", "Family", "CRPS", "Below p10", "p50", "p90", "Best"):
        dist.add_column(col, justify="left" if col in {"Pos", "Family", "Best"} else "right")
    choice = rep.choose_families()
    for pos in sorted(choice):
        for k, (fam, d) in enumerate(sorted(rep.dist.items())):
            if pos not in d:
                continue
            sc = d[pos]
            dist.add_row(
                pos if k == 0 else "",
                f"{sc.n:,}" if k == 0 else "",
                fam,
                f"{sc.crps:.3f}",
                _pct(sc.below[0]),
                _pct(sc.below[1]),
                _pct(sc.below[2]),
                "✓" if choice[pos] == fam else "",
            )
    blend = Table(title="Bayesian blend: next-week squared error, held out")
    for col in ("Pos", "n", "Sleeper only", "Blended", "Change"):
        blend.add_column(col, justify="left" if col == "Pos" else "right")
    for pos, (n, mse_s, mse_b) in sorted(rep.blend.items()):
        if n < 50:
            continue
        change = f"{100 * (mse_b - mse_s) / mse_s:+.1f}%"
        blend.add_row(pos, f"{n:,}", f"{mse_s:.2f}", f"{mse_b:.2f}", change)
    corr = Table(title="Within-game correlation, held out")
    for col in ("Pair", "Side", "Observed", "Model", "n"):
        corr.add_column(col, justify="left" if col in {"Pair", "Side"} else "right")
    for a, b, same, obs, mod, n in rep.corr:
        corr.add_row(
            f"{a}–{b}", "same team" if same else "opponents", f"{obs:+.2f}", f"{mod:+.2f}", f"{n:,}"
        )
    return [dist, blend, corr]


def backtest_tables(preds: list[Prediction]) -> list[Table]:
    out = []
    for target in ("playoffs", "champion"):
        t = Table(title=f"Backtest: P({target}) vs. what happened")
        for col in (
            "Cutoff week",
            "Teams",
            "Brier",
            "Records-only",
            "Skill",
            "90% interval",
            "Log loss",
            "Records-only ",
        ):
            t.add_column(col, justify="right")
        cuts = sorted({p.cutoff for p in preds})
        for cut in [*cuts, None]:
            sel = [p for p in preds if cut is None or p.cutoff == cut]
            sc = score_predictions(sel, target)
            lo, hi = skill_interval(sel, target)
            t.add_row(
                "all" if cut is None else str(cut),
                f"{sc.n_teams:,}",
                f"{sc.brier:.4f}",
                f"{sc.brier_base:.4f}",
                f"{100 * sc.skill:+.1f}%",
                f"{100 * lo:+.0f}% to {100 * hi:+.0f}%",
                f"{sc.log_loss:.3f}",
                f"{sc.log_loss_base:.3f}",
                style="bold" if cut is None else None,
            )
        out.append(t)
    cal = Table(title="Calibration of P(playoffs)")
    for col in ("Predicted bin", "Mean predicted", "Observed", "Teams"):
        cal.add_column(col, justify="right")
    for lo, mean_p, obs, n in calibration_bins(preds):
        cal.add_row(f"≥ {100 * lo:.0f}%", _pct(mean_p), _pct(obs), f"{n:,}")
    out.append(cal)
    return out
