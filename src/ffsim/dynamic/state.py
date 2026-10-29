"""Per-path league state: rosters, free agents, FAAB and waiver order."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ffsim.dynamic.rules import WaiverRules
from ffsim.sim.inputs import SimInputs

FREE = -1  # owner code: unrostered


@dataclass
class PathLeague:
    rosters: np.ndarray  # (B, T, R) universe index, -1 = open spot
    owner: np.ndarray  # (B, P) team index or FREE
    claimable_from: np.ndarray  # (B, P) first decision step a free player can be claimed
    budget: np.ndarray  # (B, T) FAAB left
    priority: np.ndarray  # (B, T) waiver position, 1 = first
    half_wins: np.ndarray  # (B, T) record so far, for reverse-standings waivers
    points_for: np.ndarray  # (B, T)
    capacity: int  # lineup-eligible roster spots

    @property
    def n(self) -> int:
        return self.rosters.shape[0]

    @classmethod
    def initial(
        cls, inputs: SimInputs, pidx: dict[str, int], rules: WaiverRules, n: int
    ) -> PathLeague:
        T, P = inputs.n_teams, len(pidx)
        cap = inputs.roster_capacity or max(len(r) for r in inputs.rosters.values())
        R = max(cap, max(len(r) for r in inputs.rosters.values()))
        rosters = np.full((T, R), -1, dtype=np.int32)
        owner = np.full(P, FREE, dtype=np.int16)
        for ti, rid in enumerate(inputs.team_ids):
            for k, p in enumerate(inputs.rosters[rid]):
                rosters[ti, k] = pidx[p]
                owner[pidx[p]] = ti
        budget = np.array([rules.budgets.get(rid, 0) for rid in inputs.team_ids])
        prio = np.array([rules.priority.get(rid, i + 1) for i, rid in enumerate(inputs.team_ids)])
        return cls(
            rosters=np.broadcast_to(rosters, (n, T, R)).copy(),
            owner=np.broadcast_to(owner, (n, P)).copy(),
            claimable_from=np.zeros((n, P), dtype=np.int16),
            budget=np.broadcast_to(budget, (n, T)).astype(np.int32),
            priority=np.broadcast_to(prio, (n, T)).astype(np.int32),
            half_wins=np.broadcast_to(inputs.base_half_wins, (n, T)).astype(np.float64),
            points_for=np.broadcast_to(inputs.base_points_for, (n, T)).astype(np.float64),
            capacity=cap,
        )

    def copy(self) -> PathLeague:
        return PathLeague(
            self.rosters.copy(),
            self.owner.copy(),
            self.claimable_from.copy(),
            self.budget.copy(),
            self.priority.copy(),
            self.half_wins.copy(),
            self.points_for.copy(),
            self.capacity,
        )

    def take(self, rows: np.ndarray | list[int]) -> PathLeague:
        return PathLeague(
            self.rosters[rows],
            self.owner[rows],
            self.claimable_from[rows],
            self.budget[rows],
            self.priority[rows],
            self.half_wins[rows],
            self.points_for[rows],
            self.capacity,
        )

    def roster_size(self) -> np.ndarray:
        return (self.rosters >= 0).sum(-1)

    def available(self, step: int) -> np.ndarray:
        """(B, P) players any team may claim at this decision step."""
        return (self.owner == FREE) & (self.claimable_from <= step)
