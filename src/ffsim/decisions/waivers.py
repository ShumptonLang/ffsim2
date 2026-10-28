"""Waiver pickups ranked by change in title odds, with a FAAB bid heuristic."""

from __future__ import annotations

from dataclasses import dataclass, field

from ffsim.decisions.common import Context, Delta, delta, drop_candidates, successive_halving

MAX_DROPS = 4
RIVAL_INTEREST = 0.005  # a rival "wants" a player if he adds >= 0.5 pts of title odds
FULL_BUDGET_DELTA = 0.05  # a pickup worth +5 pts of title odds justifies the whole budget


@dataclass
class WaiverOption:
    add: str
    drop: str | None
    delta: Delta
    ros_add: float
    ros_drop: float
    rivals: list[int] = field(default_factory=list)  # roster_ids that would also gain
    bid: int | None = None


def _with_add(ctx: Context, roster: list[str], add: str, drop: str | None) -> list[str]:
    return [p for p in roster if p != drop] + [add]


def waiver_options(
    ctx: Context,
    limit: int = 10,
    screen_sims: int = 2_000,
    faab: bool = False,
    budget_left: int = 0,
) -> list[WaiverOption]:
    ev, rid, ti = ctx.ev, ctx.rid, ctx.ti
    inp = ev.inputs
    roster = inp.rosters[rid]
    cap = inp.roster_capacity or len(roster) + 1
    drops: list[str | None] = (
        [None] if len(roster) < cap else list(drop_candidates(ev, roster)[:MAX_DROPS])
    )
    moves = [(fa, d) for fa in inp.free_agents for d in drops]

    def score(move: tuple[str, str | None], n: int) -> float:
        new = _with_add(ctx, roster, *move)
        return delta(ev, ev.run(rosters={rid: new}, n=n), ti).champion

    n_full = ev.n_sims
    ranked = successive_halving(
        moves,
        score,
        [(min(screen_sims, n_full), max(limit * 3, 20)), (n_full, limit * 2)],
    )
    best_per_add: dict[str, tuple[str | None, float]] = {}
    for (fa, d), s in ranked:
        if fa not in best_per_add or s > best_per_add[fa][1]:
            best_per_add[fa] = (d, s)

    options = []
    for fa, (d, _) in sorted(best_per_add.items(), key=lambda kv: -kv[1][1])[:limit]:
        out = ev.run(rosters={rid: _with_add(ctx, roster, fa, d)})
        options.append(
            WaiverOption(
                add=fa,
                drop=d,
                delta=delta(ev, out, ti),
                ros_add=inp.ros_points(fa),
                ros_drop=inp.ros_points(d) if d else 0.0,
            )
        )
    if faab:
        for opt in options[:5]:
            if opt.delta.champion > 0:
                opt.rivals = rival_interest(ctx, opt.add, screen_sims)
                opt.bid = faab_bid(opt.delta.champion, len(opt.rivals), budget_left)
    return options


def rival_interest(ctx: Context, add: str, n: int) -> list[int]:
    """Other teams whose own title odds rise meaningfully by adding this player."""
    ev = ctx.ev
    out = []
    for rid, roster in ev.inputs.rosters.items():
        if rid == ctx.rid:
            continue
        cap = ev.inputs.roster_capacity or len(roster) + 1
        drop = None if len(roster) < cap else drop_candidates(ev, roster)[0]
        new = [p for p in roster if p != drop] + [add]
        d = delta(ev, ev.run(rosters={rid: new}, n=min(n, ev.n_sims)), ev.team_idx[rid])
        if d.champion >= RIVAL_INTEREST:
            out.append(rid)
    return out


def faab_bid(delta_champion: float, rivals: int, budget_left: int) -> int:
    """Heuristic bid: scale budget by the title-odds gain and by competition.

    A gain of FULL_BUDGET_DELTA (5 points of title odds) is worth the whole
    remaining budget; with no interested rivals, half of that is enough to win
    most claims; with one rival, three quarters.
    """
    if budget_left <= 0 or delta_champion <= 0:
        return 0
    competition = {0: 0.5, 1: 0.75}.get(rivals, 1.0)
    share = min(1.0, delta_champion / FULL_BUDGET_DELTA) * competition
    return max(1, min(budget_left, round(budget_left * share)))
