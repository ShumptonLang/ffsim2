"""Trade evaluation and search, scored by change in title odds.

A trade changes *both* rosters, so it can lift a rival or a likely playoff
opponent; both sides' odds are reported. ``find_trades`` searches 1-for-1,
2-for-1, 1-for-2 and 2-for-2 deals and keeps only those the other manager would
plausibly accept, judged the way most managers judge offers:

* he doesn't give up perceived value (rest-of-season points above replacement),
  and doesn't gain so much that the offer looks lopsided;
* his projected starting lineup doesn't get worse.

Among those, deals are ranked by the change in *your* title odds.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from ffsim.decisions.common import (
    Context,
    Delta,
    delta,
    fit_roster,
    perceived_value,
    replacement_levels,
    ros_lineup_points,
    successive_halving,
)

TRADE_POOL = 8  # most valuable players per roster considered for trades
MAX_SCREENED = 2_000  # value-filtered offers checked for the partner's lineup
MAX_CANDIDATES = 400  # acceptable offers simulated
FAIRNESS = 0.35  # partner gains at most this share of the value he gives up (keeps offers sane)


@dataclass
class TradeResult:
    partner: int
    give: list[str]
    get: list[str]
    my_drops: list[str]
    partner_drops: list[str]
    me: Delta
    partner_delta: Delta
    my_lineup_delta: float  # projected starting-lineup points over the simulated weeks
    partner_lineup_delta: float
    partner_value_delta: float  # perceived value (points above replacement)


def apply_trade(
    ctx: Context, partner: int, give: list[str], get: list[str]
) -> tuple[dict[int, list[str]], list[str], list[str]]:
    rosters = ctx.ev.inputs.rosters
    mine = [p for p in rosters[ctx.rid] if p not in give] + list(get)
    theirs = [p for p in rosters[partner] if p not in get] + list(give)
    mine, my_drops = fit_roster(ctx.ev, mine, keep=get)
    theirs, their_drops = fit_roster(ctx.ev, theirs, keep=give)
    return {ctx.rid: mine, partner: theirs}, my_drops, their_drops


def evaluate_trade(
    ctx: Context, partner: int, give: list[str], get: list[str], n: int | None = None
) -> TradeResult:
    ev = ctx.ev
    new, my_drops, their_drops = apply_trade(ctx, partner, give, get)
    out = ev.run(rosters=new, n=n)
    repl = replacement_levels(ev)
    old = ev.inputs.rosters
    return TradeResult(
        partner=partner,
        give=list(give),
        get=list(get),
        my_drops=my_drops,
        partner_drops=their_drops,
        me=delta(ev, out, ctx.ti),
        partner_delta=delta(ev, out, ev.team_idx[partner]),
        my_lineup_delta=ros_lineup_points(ev, new[ctx.rid]) - ros_lineup_points(ev, old[ctx.rid]),
        partner_lineup_delta=ros_lineup_points(ev, new[partner])
        - ros_lineup_points(ev, old[partner]),
        partner_value_delta=perceived_value(ev, give, repl) - perceived_value(ev, get, repl),
    )


def find_trades(
    ctx: Context,
    max_size: int = 2,
    limit: int = 10,
    partners: list[int] | None = None,
    stages: list[tuple[int, int]] | None = None,
) -> list[TradeResult]:
    ev = ctx.ev
    inp = ev.inputs
    repl = replacement_levels(ev)

    def pool(rid: int) -> list[str]:
        return sorted(inp.rosters[rid], key=lambda p: -inp.ros_points(p))[:TRADE_POOL]

    def groups(players: list[str]) -> list[tuple[str, ...]]:
        return [g for k in range(1, max_size + 1) for g in itertools.combinations(players, k)]

    mine = groups(pool(ctx.rid))
    candidates: list[tuple[float, int, tuple[str, ...], tuple[str, ...]]] = []
    for partner in partners or [r for r in inp.team_ids if r != ctx.rid]:
        for give in mine:
            v_give = perceived_value(ev, give, repl)
            for get in groups(pool(partner)):
                v_get = perceived_value(ev, get, repl)
                gain = v_give - v_get  # partner's perceived gain
                if gain < 0 or gain > FAIRNESS * max(v_get, 1.0):
                    continue
                candidates.append((gain, partner, give, get))
    # Closest-to-even offers first (most likely accepted, least value given away).
    candidates.sort(key=lambda c: c[0])
    lineup_cache: dict[frozenset[str], float] = {}

    def lineup_pts(players: list[str]) -> float:
        key = frozenset(players)
        if key not in lineup_cache:
            lineup_cache[key] = ros_lineup_points(ev, players)
        return lineup_cache[key]

    acceptable = []
    for c in candidates[:MAX_SCREENED]:
        _, partner, give, get = c
        new, _, _ = apply_trade(ctx, partner, list(give), list(get))
        if lineup_pts(new[partner]) >= lineup_pts(inp.rosters[partner]) - 1e-6:
            acceptable.append(c)
        if len(acceptable) >= MAX_CANDIDATES:
            break
    candidates = acceptable

    def score(c: tuple[float, int, tuple[str, ...], tuple[str, ...]], n: int) -> float:
        _, partner, give, get = c
        new, _, _ = apply_trade(ctx, partner, list(give), list(get))
        return delta(ev, ev.run(rosters=new, n=n), ctx.ti).champion

    full = ev.n_sims
    stages = stages or [(min(1_000, full), 40), (min(4_000, full), 15), (full, limit)]
    ranked = successive_halving(candidates, score, stages)
    results = [
        evaluate_trade(ctx, partner, list(give), list(get))
        for (_, partner, give, get), s in ranked
        if s > 0
    ]
    results.sort(key=lambda r: -r.me.champion)
    return results[:limit]
