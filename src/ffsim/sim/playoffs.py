"""Seeding and playoff brackets, vectorized over simulations.

Seeding follows Sleeper: wins (ties count half), then points for. With
divisions, division winners take the top seeds (ordered by record), then the
remaining teams by record.

Brackets are played in seed space. Round 1 uses the standard bracket order
(1v8, 4v5, 2v7, 3v6 for 8 slots; seeds beyond ``teams`` are byes). With fixed
seeding the bracket tree is kept; with reseeding, survivors are re-paired
best-vs-worst each later round. A round spanning two weeks sums both weeks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ffsim.league import PlayoffFormat, PlayoffSeedType


def bracket_order(size: int) -> list[int]:
    order = [1]
    while len(order) < size:
        m = len(order) * 2
        order = [x for s in order for x in (s, m + 1 - s)]
    return order


def seed_teams(
    half_wins: np.ndarray,
    points_for: np.ndarray,
    divisions: Sequence[int | None] | None = None,
) -> np.ndarray:
    """Team indices ordered by seed, shape (S, T)."""
    key = half_wins.astype(np.float64) * 1e5 + points_for
    if divisions and any(d is not None for d in divisions):
        div = np.asarray([-1 if d is None else int(d) for d in divisions])
        bonus = np.zeros_like(key)
        for d in np.unique(div[div >= 0]):
            members = np.flatnonzero(div == d)
            winner = members[np.argmax(key[:, members], axis=1)]
            bonus[np.arange(len(key)), winner] = 1e12
        key = key + bonus
    return np.argsort(-key, axis=1, kind="stable")


def play_bracket(
    seeds: np.ndarray,
    round_scores: Sequence[np.ndarray],
    fmt: PlayoffFormat,
    forced_winners: Mapping[int, set[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Play the bracket.

    ``seeds``: (S, T) team indices by seed. ``round_scores[r]``: (S, T) team
    scores for round r (already summed over the round's weeks).
    ``forced_winners[r]``: team indices known to have won round r (completed rounds).
    Returns (finalists (S, 2) team indices, champion (S,) team index).
    """
    n_sims = seeds.shape[0]
    size = 2**fmt.rounds
    rows = np.arange(n_sims)[:, None]
    # slot -> seed number (1-based); 0 marks a bye.
    order = np.array([s if s <= fmt.teams else 0 for s in bracket_order(size)])
    slot_seed = np.broadcast_to(order, (n_sims, size)).copy()
    finalists = np.zeros((n_sims, 2), dtype=np.intp)

    for r in range(fmt.rounds):
        if r > 0 and fmt.seed_type == PlayoffSeedType.RESEED:
            ss = np.sort(np.where(slot_seed == 0, 10**6, slot_seed), axis=1)
            k = ss.shape[1]
            paired = np.empty_like(ss)
            paired[:, 0::2] = ss[:, : k // 2]
            paired[:, 1::2] = ss[:, ::-1][:, : k // 2]
            slot_seed = np.where(paired == 10**6, 0, paired)

        a, b = slot_seed[:, 0::2], slot_seed[:, 1::2]
        team_a = seeds[rows, np.maximum(a, 1) - 1]
        team_b = seeds[rows, np.maximum(b, 1) - 1]
        if r == fmt.rounds - 1:
            finalists = np.stack([team_a[:, 0], team_b[:, 0]], axis=1)
        sc = round_scores[r]
        score_a = np.where(a > 0, sc[rows, team_a], -np.inf)
        score_b = np.where(b > 0, sc[rows, team_b], -np.inf)
        a_wins = score_a >= score_b
        forced = (forced_winners or {}).get(r)
        if forced:
            fa = np.isin(team_a, list(forced)) & (a > 0)
            fb = np.isin(team_b, list(forced)) & (b > 0)
            a_wins = np.where(fa, True, np.where(fb, False, a_wins))
        slot_seed = np.where(a_wins, a, b)

    champion = seeds[np.arange(n_sims), slot_seed[:, 0] - 1]
    return finalists, champion
