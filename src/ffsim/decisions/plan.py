"""``ffsim plan``: the week's path to the title in one report."""

from __future__ import annotations

from dataclasses import dataclass

from ffsim.decisions.common import Context
from ffsim.decisions.lineup import LineupOption, lineup_options
from ffsim.decisions.trades import TradeResult, find_trades
from ffsim.decisions.waivers import WaiverOption, waiver_options
from ffsim.sim.engine import SimResult


@dataclass
class Plan:
    odds: SimResult
    lineup: list[LineupOption]
    waivers: list[WaiverOption]
    trades: list[TradeResult]


def make_plan(
    ctx: Context, faab: bool, budget_left: int, max_trade_size: int = 2, top: int = 3
) -> Plan:
    ev = ctx.ev
    return Plan(
        odds=ev.summary(ev.baseline()),
        lineup=lineup_options(ctx, limit=top + 1),
        waivers=waiver_options(ctx, limit=top, faab=faab, budget_left=budget_left),
        trades=find_trades(ctx, max_size=max_trade_size, limit=top),
    )
