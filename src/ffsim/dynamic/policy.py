"""The roster policy every simulated manager follows.

Each decision step a manager looks for the add/drop that most raises his roster
value (``value.RosterValuer``), repeating up to ``max_moves`` times. There are
no position-specific rules; two generic ones keep the greedy search from being
myopic about timing:

* Defer a move that doesn't help this week. Making it next week keeps the
  current player for one more week and sees one more week of news; the next
  decision re-checks it. A move that only pays off in a later week (a kicker
  for a bye three weeks out) is therefore made the week it starts paying.
* Unless a rival is about to take the player: a deferred move whose player is
  in another manager's claims this step is claimed now.

FAAB bids are ``price`` dollars per unit of value gained, with ``price`` fitted
so that budgets are spent over the season (money is worthless afterwards).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ffsim.dynamic.claims import Claim
from ffsim.dynamic.lineups import SlotSpec
from ffsim.dynamic.state import PathLeague
from ffsim.dynamic.value import RosterValuer

EPS = 1e-9


@dataclass
class PolicyOptions:
    top_adds: int = 8  # free agents fully evaluated per manager and step
    top_drops: int = 4  # roster spots considered for dropping
    max_moves: int = 2  # adds per manager per step
    scenarios: int = 8  # imagined injury scenarios behind each roster value


@dataclass
class TeamDecision:
    claims: list[list[Claim]]  # per path, in preference order
    stash: list[Claim | None]  # per path, the best deferred move (claimed only if contested)


class GreedyPolicy:
    def __init__(
        self,
        leverage: np.ndarray,
        price: float | None,
        bid_min: int,
        faab: bool,
        opts: PolicyOptions,
    ) -> None:
        self.leverage = leverage  # (T, W)
        self.price = price
        self.bid_min = bid_min
        self.faab = faab
        self.opts = opts

    def bid(self, gain: float, budget: int) -> int:
        if not self.faab:
            return 0
        raw = 0 if self.price is None else int(round(self.price * gain))
        return int(min(max(raw, self.bid_min), budget))

    def decide(
        self,
        state: PathLeague,
        valuer: RosterValuer,
        spec: SlotSpec,
        team: int,
        step: int,
        exclude: np.ndarray | None = None,
    ) -> TeamDecision:
        """Claims (and a stash candidate) for one team in every path."""
        o = self.opts
        B = state.n
        L = self.leverage[team, step:].astype(np.float64)
        claims: list[list[Claim]] = [[] for _ in range(B)]
        stash: list[Claim | None] = [None] * B
        if L.sum() <= 0:
            return TeamDecision(claims, stash)
        roster = state.rosters[:, team].copy()  # (B, R)
        R = roster.shape[1]
        avail = state.available(step).copy()
        if exclude is not None:
            avail &= ~exclude
        active = np.ones(B, dtype=bool)
        groups, group_of = np.unique(spec.elig[:-1], axis=0, return_inverse=True)
        group_of = group_of.reshape(-1)
        for move in range(o.max_moves):
            # Leave-one-out losses pick the drop candidates (open spots lose nothing).
            loo = np.broadcast_to(roster[:, None, :], (B, R + 1, R)).copy()
            loo[:, np.arange(R), np.arange(R)] = -1
            vals = valuer.week_values(loo)  # (B, R + 1, Wr); last row = current roster
            base = vals[:, R]
            loss = ((base[:, None, :] - vals[:, :R]) * L).sum(-1)
            loss += np.where(roster >= 0, 1e-12, 0.0)  # on ties, fill an open spot first
            n_drop = min(o.top_drops, R)
            drops = np.argsort(loss, axis=1, kind="stable")[:, :n_drop]  # (B, D) slots
            # Cheap screen for adds: points over the weakest eligible starter.
            start = valuer.expected_lineup(roster[:, None, :])[:, 0]  # (B, Wr, S)
            thr = np.stack(
                [np.where(g, start, np.inf).min(-1) for g in groups], axis=-1
            )  # (B, Wr, G)
            thr = np.where(np.isfinite(thr), thr, 0.0)
            ev = valuer.expected_values()  # (B, P, Wr)
            gain = np.maximum(ev - thr.transpose(0, 2, 1)[:, group_of], 0.0) @ L  # (B, P)
            gain = np.where(avail & spec.elig[:-1].any(1)[None], gain, -1.0)
            n_add = min(o.top_adds, gain.shape[1])
            adds = np.argpartition(-gain, n_add - 1, axis=1)[:, :n_add]  # (B, A)
            add_ok = np.take_along_axis(gain, adds, 1) > 0
            if not add_ok.any():
                break
            # Full values for every (add, drop) pair.
            A, D = adds.shape[1], drops.shape[1]
            cand = np.broadcast_to(roster[:, None, None, :], (B, A, D, R)).copy()
            ai, di = np.meshgrid(np.arange(A), np.arange(D), indexing="ij")
            cand[np.arange(B)[:, None, None], ai[None], di[None], drops[:, None, :]] = adds[
                :, :, None
            ]
            cvals = valuer.week_values(cand.reshape(B, A * D, R))  # (B, A*D, Wr)
            diff = cvals - base[:, None, :]
            dv = (diff * L).sum(-1)
            now = diff[:, :, 0] * L[0]
            dv = np.where(np.repeat(add_ok, D, axis=1), dv, -np.inf)
            immediate = (dv > EPS) & (now > EPS)
            best_any = np.argmax(dv, axis=1)
            best_now = np.argmax(np.where(immediate, dv, -np.inf), axis=1)
            has_now = immediate.any(1) & active
            for b in np.flatnonzero(active):
                pick = best_now[b] if has_now[b] else best_any[b]
                g = float(dv[b, pick])
                if g <= EPS:
                    continue
                a_i, d_i = divmod(int(pick), D)
                slot = int(drops[b, d_i])
                c = Claim(
                    team=team,
                    add=int(adds[b, a_i]),
                    drop=int(roster[b, slot]),
                    bid=self.bid(g, int(state.budget[b, team])),
                    rank=move,
                    gain=g,
                )
                if has_now[b]:
                    claims[b].append(c)
                    roster[b, slot] = c.add
                    avail[b, c.add] = False
                elif move == 0:
                    stash[b] = c
            active &= has_now
            if not active.any():
                break
        return TeamDecision(claims, stash)


def contested(decisions: list[TeamDecision], b: int) -> list[Claim]:
    """Each team's claims in path ``b``, with stashes added when a rival wants the player."""
    wanted: dict[int, set[int]] = {}
    for d in decisions:
        for c in d.claims[b]:
            wanted.setdefault(c.add, set()).add(c.team)
    out: list[Claim] = []
    for d in decisions:
        out.extend(d.claims[b])
        s = d.stash[b]
        if s is not None and wanted.get(s.add, set()) - {s.team}:
            out.append(Claim(s.team, s.add, s.drop, s.bid, len(d.claims[b]), s.gain))
    return out
