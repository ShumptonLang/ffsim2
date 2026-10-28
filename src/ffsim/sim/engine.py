"""Monte Carlo league-season engine.

For each batch of simulated seasons:

1. latent normals per player-week: a per-player talent factor shared by all
   remaining weeks (``rho`` from the talent model) plus a weekly part that is
   correlated within NFL games through the factor copula;
2. fantasy points via the projection-error model's quantile functions;
3. availability (this week's injury designations, new injuries later);
4. team scores from projection-set lineups, with bench substitutions for
   starters who are unavailable;
5. remaining regular season, seeding, and the playoff bracket.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ffsim.model.correlation import WeekLayout, sample_weekly, week_layout
from ffsim.model.models import ModelSet
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


@dataclass
class _Prepared:
    players: list[str]
    positions: list[str | None]
    means: np.ndarray  # (W, P)
    rho: np.ndarray  # (P,)
    hazard: np.ndarray  # (P,)
    first_play: np.ndarray  # (P,)
    plans: list[list[list[list[int]]]]  # [week][team][slot] -> candidate player indices
    layouts: list[WeekLayout]  # correlation factor layout per week


def prepare(inputs: SimInputs, models: ModelSet) -> _Prepared:
    plans_by_id = inputs.lineup_plans()
    players = sorted(
        {p for wk in plans_by_id.values() for plan in wk.values() for cands in plan for p in cands}
    )
    pidx = {p: i for i, p in enumerate(players)}
    positions = [(inputs.positions.get(p) or (None,))[0] for p in players]
    means = np.array(
        [[inputs.means.get(w, {}).get(p, 0.0) for p in players] for w in inputs.weeks],
        dtype=np.float64,
    ).reshape(len(inputs.weeks), len(players))
    plans = [
        [[[pidx[p] for p in cands] for cands in plans_by_id[w][rid]] for rid in inputs.team_ids]
        for w in inputs.weeks
    ]
    return _Prepared(
        players=players,
        positions=positions,
        means=means,
        rho=np.array([models.talent.rho_for(pos) for pos in positions]),
        hazard=np.array([models.availability.hazard_for(pos) for pos in positions]),
        first_play=np.array(
            [models.availability.play_prob(inputs.statuses.get(p)) for p in players]
        ),
        plans=plans,
        layouts=[
            week_layout(
                models.correlation, players, inputs.nfl_games.get(w, {}), inputs.roles.get(w, {})
            )
            for w in inputs.weeks
        ],
    )


def _team_scores(pts: np.ndarray, avail: np.ndarray, plans: list[list[list[int]]]) -> np.ndarray:
    """Scores (B, T) for one week: each slot takes its first available candidate."""
    B = pts.shape[0]
    out = np.zeros((B, len(plans)), dtype=np.float32)
    for t, plan in enumerate(plans):
        used: dict[int, np.ndarray] = {}
        total = np.zeros(B, dtype=np.float32)
        for cands in plan:
            filled = np.zeros(B, dtype=bool)
            for c in cands:
                ok = avail[:, c] & ~filled
                if c in used:
                    ok &= ~used[c]
                    used[c] |= ok
                else:
                    used[c] = ok
                total += np.where(ok, pts[:, c], 0.0)
                filled |= ok
        out[:, t] = total
    return out


def simulate(
    inputs: SimInputs,
    models: ModelSet,
    n_sims: int = 10_000,
    seed: int | None = 0,
    batch_size: int = 2_000,
) -> SimResult:
    T = inputs.n_teams
    po = inputs.playoffs
    weeks = inputs.weeks
    W = len(weeks)
    prep = prepare(inputs, models)
    P = len(prep.players)
    active = prep.means > 0
    wpos = {w: i for i, w in enumerate(weeks)}
    round_weeks = po.round_weeks()
    sqrt_rho = np.sqrt(prep.rho).astype(np.float32)
    sqrt_1m = np.sqrt(1 - prep.rho).astype(np.float32)

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
        talent = rng.standard_normal((B, 1, P), dtype=np.float32)
        weekly = np.stack([sample_weekly(rng, B, lay) for lay in prep.layouts], axis=1)
        z = sqrt_rho * talent + sqrt_1m * weekly.reshape(B, W, P)
        pts = models.errors.sample(z, prep.means[None, :, :], prep.positions)  # (B, W, P)
        avail = models.availability.sample(rng, B, active, prep.hazard, prep.first_play)
        team_pts = np.zeros((B, W, T), dtype=np.float32)
        for wi in range(W):
            team_pts[:, wi] = _team_scores(pts[:, wi], avail[:, wi], prep.plans[wi])
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
        np.add.at(acc["playoffs"], seeds[:, : po.teams].ravel(), 1)
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
