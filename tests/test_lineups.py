import itertools

import numpy as np
import pytest

from ffsim.league import SLOT_ELIGIBILITY
from ffsim.sim.lineups import best_lineup


def brute_force(slots, players, means, positions):
    best = -1.0
    n = len(slots)
    pool = list(players) + [None] * n
    for perm in itertools.permutations(range(len(pool)), n):
        chosen = [pool[i] for i in perm]
        ok = all(
            p is None or any(fp in SLOT_ELIGIBILITY[s] for fp in positions[p])
            for s, p in zip(slots, chosen, strict=True)
        )
        if ok:
            best = max(best, sum(means.get(p, 0.0) for p in chosen if p))
    return best


@pytest.mark.parametrize("seed", range(15))
def test_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    slots = ["QB", "WR", "FLEX", "SUPER_FLEX", "IDP_FLEX"]
    groups = ["QB", "RB", "WR", "TE", "LB", "DB", "DL"]
    players = [f"p{i}" for i in range(7)]
    positions = {p: (str(rng.choice(groups)),) for p in players}
    means = {p: float(rng.uniform(0, 25)) for p in players if rng.random() > 0.15}
    lineup = best_lineup(slots, players, means, positions)
    total = sum(means.get(p, 0.0) for p in lineup if p)
    assert total == pytest.approx(brute_force(slots, players, means, positions))
    assert len({p for p in lineup if p}) == len([p for p in lineup if p])
    for s, p in zip(slots, lineup, strict=True):
        if p:
            assert positions[p][0] in SLOT_ELIGIBILITY[s]


def test_superflex_prefers_second_qb():
    slots = ["QB", "WR", "SUPER_FLEX"]
    positions = {"qb1": ("QB",), "qb2": ("QB",), "wr1": ("WR",), "wr2": ("WR",)}
    means = {"qb1": 20, "qb2": 17, "wr1": 15, "wr2": 12}
    assert best_lineup(slots, list(positions), means, positions) == ["qb1", "wr1", "qb2"]


def test_empty_slot_when_no_eligible_player():
    out = best_lineup(["QB", "K"], ["a"], {"a": 10}, {"a": ("QB",)})
    assert out == ["a", None]


def test_bye_player_fills_only_if_needed():
    positions = {"a": ("RB",), "b": ("RB",)}
    assert best_lineup(["RB"], ["a", "b"], {"b": 3.0}, positions) == ["b"]
    assert best_lineup(["RB", "RB"], ["a", "b"], {"b": 3.0}, positions) in (["a", "b"], ["b", "a"])
