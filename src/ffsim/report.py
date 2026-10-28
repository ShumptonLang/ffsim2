"""Rich-table rendering for CLI output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

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
