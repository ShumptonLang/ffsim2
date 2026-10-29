"""Waiver claim resolution under FAAB, rolling or reverse-standings priority."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ffsim.dynamic.state import FREE, PathLeague


@dataclass(frozen=True)
class Claim:
    team: int  # team index
    add: int  # universe index
    drop: int  # universe index, or -1 to fill an open spot
    bid: int = 0
    rank: int = 0  # the team's own preference order (0 = first)
    gain: float = 0.0  # the claimant's value gain (title-odds units)


def _valid(state: PathLeague, b: int, c: Claim, step: int) -> bool:
    if state.owner[b, c.add] != FREE or state.claimable_from[b, c.add] > step:
        return False
    if c.drop >= 0:
        if state.owner[b, c.drop] != c.team:
            return False
    elif (state.rosters[b, c.team] >= 0).sum() >= state.capacity:
        return False
    return c.bid <= state.budget[b, c.team]


def apply_claim(state: PathLeague, b: int, c: Claim, step: int) -> None:
    roster = state.rosters[b, c.team]
    if c.drop >= 0:
        k = int(np.flatnonzero(roster == c.drop)[0])
        state.owner[b, c.drop] = FREE
        state.claimable_from[b, c.drop] = step + 1  # on waivers until the next run
    else:
        k = int(np.flatnonzero(roster < 0)[0])
    roster[k] = c.add
    state.owner[b, c.add] = c.team
    state.budget[b, c.team] -= c.bid


def resolve(state: PathLeague, b: int, claims: list[Claim], kind: str, step: int) -> list[Claim]:
    """Award claims in path ``b`` and update the state. Returns the claims won."""
    won: list[Claim] = []
    if not claims:
        return won
    if kind == "faab":
        prio = state.priority[b]
        for c in sorted(claims, key=lambda c: (-c.bid, prio[c.team], c.team, c.rank)):
            if _valid(state, b, c, step):
                apply_claim(state, b, c, step)
                won.append(c)
        return won
    pending = {t: sorted((c for c in claims if c.team == t), key=lambda c: c.rank)
               for t in {c.team for c in claims}}  # fmt: skip
    if kind == "reverse":
        record = state.half_wins[b] * 1e6 + state.points_for[b]
        order = sorted(pending, key=lambda t: (record[t], t))
    else:
        order = sorted(pending, key=lambda t: (state.priority[b, t], t))
    progress = True
    while progress:
        progress = False
        for t in list(order):
            queue = pending[t]
            while queue and not _valid(state, b, queue[0], step):
                queue.pop(0)
            if not queue:
                continue
            c = queue.pop(0)
            apply_claim(state, b, c, step)
            won.append(c)
            progress = True
            if kind == "rolling":
                # Successful claimant goes to the back of the line.
                p = state.priority[b]
                p[p > p[t]] -= 1
                p[t] = len(p)
                order = sorted(pending, key=lambda t: (state.priority[b, t], t))
                break
    return won
