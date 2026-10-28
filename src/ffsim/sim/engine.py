"""Monte Carlo league-season engine.

For each batch of simulations: draw player points for every simulated week,
score projection-set lineups, play the remaining regular season, seed the
playoffs, play the bracket, and accumulate per-team outcome counts.

Milestone 2 draws players independently from the projection-error model. The
latent-normal interface (``z``) is where the correlation copula (milestone 4)
and the talent/availability layers (milestone 3) plug in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ffsim.model.errors import ErrorModel
from ffsim.sim.inputs import SimInputs
from ffsim.sim.playoffs import play_bracket, seed_teams


@dataclass
class SimResult:
    team_ids: list[int]
    team_labels: list[str]
    n_sims: int
    playoffs: np.ndarray  # (T,) P(make playoffs)
    bye: np.ndarray  # (T,) P(first-round bye)
    final: np.ndarray  # (T,) P(reach championship)
    champion: np.ndarray  # (T,) P(win title)
    seed_probs: np.ndarray  # (T, n_playoff_teams) P(seed k+1)
    mean_wins: np.ndarray  # (T,) expected final regular-season wins (ties = 1/2)
    week_win: dict[int, np.ndarray] = field(default_factory=dict)  # week -> (T,) P(win)
    week_points: dict[int, np.ndarray] = field(default_factory=dict)  # week -> (T,) E[points]

    def se(self, p: np.ndarray) -> np.ndarray:
        return np.sqrt(p * (1 - p) / max(self.n_sims, 1))

    def index(self, roster_id: int) -> int:
        return self.team_ids.index(roster_id)


def _lineup_matrices(inputs: SimInputs, players: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Starter masks (W, T, P) and projected means (W, P) over ``players``."""
    pidx = {p: i for i, p in enumerate(players)}
    lineups = inputs.lineups()
    W, T, P = len(inputs.weeks), inputs.n_teams, len(players)
    mask = np.zeros((W, T, P), dtype=np.float32)
    means = np.zeros((W, P), dtype=np.float64)
    for wi, w in enumerate(inputs.weeks):
        m = inputs.means.get(w, {})
        for p, i in pidx.items():
            means[wi, i] = m.get(p, 0.0)
        for ti, rid in enumerate(inputs.team_ids):
            for p in lineups[w][rid]:
                mask[wi, ti, pidx[p]] = 1.0
    return mask, means


def simulate(
    inputs: SimInputs,
    model: ErrorModel,
    n_sims: int = 10_000,
    seed: int | None = 0,
    batch_size: int = 2_000,
) -> SimResult:
    T = inputs.n_teams
    po = inputs.playoffs
    weeks = inputs.weeks
    lineups_players = sorted(
        {p for ps in inputs.lineups().values() for team in ps.values() for p in team}
    )
    mask, means = _lineup_matrices(inputs, lineups_players)
    positions = [(inputs.positions.get(p) or (None,))[0] for p in lineups_players]
    wpos = {w: i for i, w in enumerate(weeks)}
    round_weeks = po.round_weeks()

    n_seeds = max(po.teams, 1)
    acc = {
        "playoffs": np.zeros(T),
        "bye": np.zeros(T),
        "final": np.zeros(T),
        "champion": np.zeros(T),
        "wins": np.zeros(T),
        "seed": np.zeros((T, n_seeds)),
    }
    week_win = {w: np.zeros(T) for w in inputs.regular_weeks}
    week_pts = {w: np.zeros(T) for w in weeks}

    rng = np.random.default_rng(seed)
    done = 0
    while done < n_sims:
        B = min(batch_size, n_sims - done)
        z = rng.standard_normal((B, len(weeks), len(lineups_players)), dtype=np.float32)
        pts = model.sample(z, means[None, :, :], positions)  # (B, W, P)
        team_pts = np.einsum("bwp,wtp->bwt", pts, mask, optimize=True)  # (B, W, T)
        for w, wi in wpos.items():
            week_pts[w] += team_pts[:, wi, :].sum(0)

        half_wins = np.broadcast_to(inputs.base_half_wins.astype(float), (B, T)).copy()
        pf = np.broadcast_to(inputs.base_points_for.astype(float), (B, T)).copy()
        for w in inputs.regular_weeks:
            s = team_pts[:, wpos[w], :]
            pf += s
            won = np.zeros((B, T))
            for a, b in inputs.matchups.get(w, []):
                won[:, a] = 2 * (s[:, a] > s[:, b]) + (s[:, a] == s[:, b])
                won[:, b] = 2 * (s[:, b] > s[:, a]) + (s[:, a] == s[:, b])
            if inputs.median_match:
                med = np.median(s, axis=1, keepdims=True)
                won += 2 * (s > med) + (s == med)
            half_wins += won
            week_win[w] += (won == 2).sum(0)
        acc["wins"] += (half_wins / 2).sum(0)

        seeds = seed_teams(half_wins, pf, inputs.divisions)  # (B, T)
        playoff_teams = seeds[:, : po.teams]
        np.add.at(acc["playoffs"], playoff_teams.ravel(), 1)
        np.add.at(acc["bye"], seeds[:, : po.byes].ravel(), 1)
        for k in range(min(po.teams, T)):
            np.add.at(acc["seed"][:, k], seeds[:, k], 1)

        if po.teams > 1:
            round_scores = []
            for rw in round_weeks:
                sc = np.zeros((B, T))
                for w in rw:
                    if w in wpos:
                        sc += team_pts[:, wpos[w], :]
                    elif w in inputs.actual_points:
                        sc += inputs.actual_points[w]
                round_scores.append(sc)
            finalists, champ = play_bracket(seeds, round_scores, po, inputs.forced_playoff_winners)
            np.add.at(acc["final"], finalists.ravel(), 1)
            np.add.at(acc["champion"], champ, 1)
        else:
            np.add.at(acc["champion"], seeds[:, 0], 1)
        done += B

    n = float(n_sims)
    return SimResult(
        team_ids=list(inputs.team_ids),
        team_labels=list(inputs.team_labels),
        n_sims=n_sims,
        playoffs=acc["playoffs"] / n,
        bye=acc["bye"] / n,
        final=acc["final"] / n,
        champion=acc["champion"] / n,
        seed_probs=acc["seed"] / n,
        mean_wins=acc["wins"] / n,
        week_win={w: v / n for w, v in week_win.items()},
        week_points={w: v / n for w, v in week_pts.items()},
    )
