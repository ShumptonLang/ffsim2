"""Rich-table rendering for CLI output."""

from __future__ import annotations

from collections.abc import Mapping

from rich.console import Console
from rich.table import Table

from ffsim.league import LeagueState, Team
from ffsim.scoring import profile


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
