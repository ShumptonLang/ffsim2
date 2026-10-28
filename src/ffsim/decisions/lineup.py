"""Start/sit: candidate lineups for one week ranked by change in title odds.

The baseline is the projection-optimal lineup. Alternatives swap one starter for
one bench player (re-solving slot assignments so flex moves are allowed), plus
the lineup currently set in Sleeper. Each is scored against the same draws, so a
lineup with a lower projection can still rank first when its extra variance or
stacking is worth more to this team's title chances.
"""

from __future__ import annotations

from dataclasses import dataclass

from ffsim.decisions.common import Context, Delta, delta, score_quantiles
from ffsim.sim.lineups import best_lineup


@dataclass
class LineupOption:
    label: str
    starters: list[str | None]
    delta: Delta
    p_win: float
    mean: float
    p10: float
    p90: float
    projected: float  # sum of starters' projections

    @property
    def is_baseline(self) -> bool:
        return self.label == "Projection-optimal lineup"


def lineup_options(ctx: Context, week: int | None = None, limit: int = 12) -> list[LineupOption]:
    ev, rid, ti = ctx.ev, ctx.rid, ctx.ti
    inp = ev.inputs
    w = week or inp.weeks[0]
    m = inp.means.get(w, {})
    roster = inp.rosters[rid]
    base = best_lineup(inp.slots, roster, m, inp.positions)
    filled = sum(1 for p in base if p)
    starters = {p for p in base if p}
    bench = [p for p in roster if p not in starters and m.get(p, 0.0) > 0]

    candidates: list[tuple[str, list[str | None]]] = [("Projection-optimal lineup", base)]
    seen = {frozenset(starters)}
    current = _current_sleeper_lineup(ctx, w)
    if current is not None and frozenset(p for p in current if p) not in seen:
        candidates.append(("Your current Sleeper lineup", current))
        seen.add(frozenset(p for p in current if p))
    for out_p in sorted(starters, key=lambda p: m.get(p, 0.0)):
        for in_p in sorted(bench, key=lambda p: -m.get(p, 0.0)):
            pool = (starters - {out_p}) | {in_p}
            if frozenset(pool) in seen:
                continue
            alt = best_lineup(inp.slots, sorted(pool), m, inp.positions)
            if sum(1 for p in alt if p) < filled:
                continue  # in_p can't legally replace out_p
            seen.add(frozenset(pool))
            label = f"Start {ctx.state.player(in_p).name} over {ctx.state.player(out_p).name}"
            candidates.append((label, alt))

    options = []
    for label, lineup in candidates:
        starters_map = {} if label == "Projection-optimal lineup" else {(w, rid): lineup}
        out = ev.run(starters=starters_map)
        pts = out.team_pts[:, ev.week_idx[w], ti]
        p10, p90 = score_quantiles(pts)
        won = out.week_won.get(w)
        options.append(
            LineupOption(
                label=label,
                starters=lineup,
                delta=delta(ev, out, ti),
                p_win=float(won[:, ti].mean()) if won is not None else float("nan"),
                mean=float(pts.mean()),
                p10=p10,
                p90=p90,
                projected=sum(m.get(p, 0.0) for p in lineup if p),
            )
        )
    options.sort(key=lambda o: (-o.delta.champion, -o.p_win))
    return options[:limit]


def _current_sleeper_lineup(ctx: Context, week: int) -> list[str | None] | None:
    """The lineup set in Sleeper, if it is for the week being decided."""
    inp = ctx.ev.inputs
    if week != ctx.state.nfl_week or len(ctx.team.starters) != len(inp.slots):
        return None
    roster = set(inp.rosters[ctx.rid])
    lineup: list[str | None] = [p if p in roster else None for p in ctx.team.starters]
    return lineup if any(lineup) else None
