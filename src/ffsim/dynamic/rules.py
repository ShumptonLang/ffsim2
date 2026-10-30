"""League waiver rules as the dynamic simulation applies them."""

from __future__ import annotations

from dataclasses import dataclass, field

from ffsim.league import LeagueState

# Sleeper ``settings.waiver_type``.
WAIVER_KINDS = {0: "rolling", 1: "reverse", 2: "faab"}


@dataclass(frozen=True)
class WaiverRules:
    kind: str = "rolling"  # "faab" | "rolling" (claimant goes to the back) | "reverse" (standings)
    budgets: dict[int, int] = field(default_factory=dict)  # roster_id -> FAAB left
    bid_min: int = 0
    priority: dict[int, int] = field(default_factory=dict)  # roster_id -> position (1 = first)

    @property
    def faab(self) -> bool:
        return self.kind == "faab"

    @classmethod
    def from_league(cls, state: LeagueState) -> WaiverRules:
        lg = state.league
        kind = WAIVER_KINDS.get(lg.waiver_type, "rolling")
        teams = sorted(state.teams.values(), key=lambda t: t.roster_id)
        prio = {t.roster_id: t.waiver_position for t in teams}
        if sorted(prio.values()) != list(range(1, len(teams) + 1)):
            # Missing or inconsistent positions: fall back to worst record first.
            order = sorted(teams, key=lambda t: (t.record.win_pct, t.record.points_for))
            prio = {t.roster_id: i + 1 for i, t in enumerate(order)}
        return cls(
            kind=kind,
            budgets={t.roster_id: max(0, lg.waiver_budget - t.waiver_budget_used) for t in teams},
            bid_min=lg.waiver_bid_min,
            priority=prio,
        )
