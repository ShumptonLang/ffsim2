"""Monte Carlo league-season engine with common random numbers.

``Evaluator`` draws every player's outcomes once for a fixed universe of players
(all rostered players, plus free agents when waivers are being evaluated):

1. latent normals per player-week: a per-player talent factor shared by all
   remaining weeks (``rho`` from the talent model) plus a weekly part that is
   correlated within NFL games through the factor copula;
2. fantasy points via the projection-error model's quantile functions;
3. availability (this week's injury designations, new injuries later).

Any roster or lineup scenario is then scored against those *same* draws: team
scores from lineups (with bench substitutions for unavailable starters), the
remaining regular season, seeding, and the playoff bracket. Because baseline
and alternative share every random number, the difference in a team's title
odds is a paired estimate with a far smaller standard error than two
independent runs, which is what makes small edges between decisions visible.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ffsim.model.correlation import sample_weekly, week_layout
from ffsim.model.models import ModelSet
from ffsim.sim.inputs import SimInputs
from ffsim.sim.playoffs import play_bracket, seed_teams

Starters = Mapping[tuple[int, int], list[str | None]]  # (week, roster_id) -> player per slot


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
class Outcomes:
    """Per-simulation results for one scenario."""

    team_pts: np.ndarray  # (S, W, T)
    half_wins: np.ndarray  # (S, T) final regular-season wins x2
    seeds: np.ndarray  # (S, T) team indices ordered by seed
    finalists: np.ndarray  # (S, 2)
    champion: np.ndarray  # (S,)
    week_won: dict[int, np.ndarray]  # regular week -> (S, T) bool

    @property
    def n(self) -> int:
        return len(self.champion)

    def made_playoffs(self, n_playoff: int) -> np.ndarray:
        T = self.seeds.shape[1]
        out = np.zeros((self.n, T), dtype=bool)
        np.put_along_axis(out, self.seeds[:, :n_playoff], True, axis=1)
        return out


def paired_delta(base: np.ndarray, alt: np.ndarray) -> tuple[float, float]:
    """Mean difference of per-simulation indicators and its standard error."""
    d = alt.astype(np.float64) - base.astype(np.float64)
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return float(d.mean()), se


def _team_scores(pts: np.ndarray, avail: np.ndarray, plans: list[list[list[int]]]) -> np.ndarray:
    """Scores (S, T) for one week: each slot takes its first available candidate."""
    S = pts.shape[0]
    out = np.zeros((S, len(plans)), dtype=np.float32)
    for t, plan in enumerate(plans):
        used: dict[int, np.ndarray] = {}
        total = np.zeros(S, dtype=np.float32)
        for cands in plan:
            filled = np.zeros(S, dtype=bool)
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


class Evaluator:
    def __init__(
        self,
        inputs: SimInputs,
        models: ModelSet,
        n_sims: int = 10_000,
        seed: int | None = 0,
        batch_size: int = 2_000,
    ) -> None:
        self.inputs = inputs
        self.models = models
        self.n_sims = n_sims
        self.players = inputs.universe
        self.pidx = {p: i for i, p in enumerate(self.players)}
        self.team_idx = {rid: i for i, rid in enumerate(inputs.team_ids)}
        self.week_idx = {w: i for i, w in enumerate(inputs.weeks)}
        self.pts, self.avail = self._draw(n_sims, seed, batch_size)
        self.base_team_pts = self._score_rosters(inputs.rosters, {})
        self._baseline: dict[int, Outcomes] = {}

    # -- draws ---------------------------------------------------------------------

    def _draw(
        self, n_sims: int, seed: int | None, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        inp, models = self.inputs, self.models
        W, P = len(inp.weeks), len(self.players)
        positions = [(inp.positions.get(p) or (None,))[0] for p in self.players]
        means = np.array(
            [[inp.means.get(w, {}).get(p, 0.0) for p in self.players] for w in inp.weeks],
            dtype=np.float64,
        ).reshape(W, P)
        rho = np.array([models.talent.rho_for(pos) for pos in positions], dtype=np.float32)
        hazard = np.array([models.availability.hazard_for(pos) for pos in positions])
        first_play = np.array(
            [models.availability.play_prob(inp.statuses.get(p)) for p in self.players]
        )
        layouts = [
            week_layout(models.correlation, self.players, inp.nfl_games.get(w, {}),
                        inp.roles.get(w, {}))
            for w in inp.weeks
        ]  # fmt: skip
        active = means > 0
        pts = np.zeros((n_sims, W, P), dtype=np.float32)
        avail = np.zeros((n_sims, W, P), dtype=bool)
        rng = np.random.default_rng(seed)
        for start in range(0, n_sims, batch_size):
            B = min(batch_size, n_sims - start)
            talent = rng.standard_normal((B, 1, P), dtype=np.float32)
            weekly = np.zeros((B, W, P), dtype=np.float32)
            for wi, lay in enumerate(layouts):
                weekly[:, wi] = sample_weekly(rng, B, lay)
            z = np.sqrt(rho) * talent + np.sqrt(1 - rho) * weekly
            sl = slice(start, start + B)
            pts[sl] = models.errors.sample(z, means[None, :, :], positions)
            avail[sl] = models.availability.sample(rng, B, active, hazard, first_play)
        return pts, avail

    # -- scoring -------------------------------------------------------------------

    def _plan_indices(self, plan: list[list[str]]) -> list[list[int]]:
        try:
            return [[self.pidx[p] for p in cands] for cands in plan]
        except KeyError as exc:
            raise KeyError(f"player {exc} is not in the simulated universe") from None

    def _score_rosters(
        self,
        rosters: Mapping[int, list[str]],
        starters: Starters,
        n: int | None = None,
        weeks: list[int] | None = None,
    ) -> np.ndarray:
        """Team points (n, W, T) for the given teams; other columns are zero."""
        n = n or self.n_sims
        W, T = len(self.inputs.weeks), self.inputs.n_teams
        out = np.zeros((n, W, T), dtype=np.float32)
        rids = list(rosters)
        if not rids:
            return out
        for w in weeks or self.inputs.weeks:
            wi = self.week_idx[w]
            plans = [
                self._plan_indices(self.inputs.team_plan(w, rosters[rid], starters.get((w, rid))))
                for rid in rids
            ]
            scores = _team_scores(self.pts[:n, wi], self.avail[:n, wi], plans)
            for k, rid in enumerate(rids):
                out[:, wi, self.team_idx[rid]] = scores[:, k]
        return out

    def team_points(
        self,
        rosters: Mapping[int, list[str]] | None = None,
        starters: Starters | None = None,
        n: int | None = None,
    ) -> np.ndarray:
        """Team points (n, W, T) with some rosters and/or lineups changed.

        Only the changed team-weeks are re-scored; everything else is the baseline.
        """
        n = n or self.n_sims
        rosters = dict(rosters or {})
        starters = dict(starters or {})
        tp = self.base_team_pts[:n].copy()
        if rosters:
            new = self._score_rosters(rosters, starters, n)
            for rid in rosters:
                ti = self.team_idx[rid]
                tp[:, :, ti] = new[:, :, ti]
        for w, rid in starters:
            if rid in rosters:
                continue
            wi, ti = self.week_idx[w], self.team_idx[rid]
            new = self._score_rosters({rid: self.inputs.rosters[rid]}, starters, n, weeks=[w])
            tp[:, wi, ti] = new[:, wi, ti]
        return tp

    # -- season & playoffs -----------------------------------------------------------

    def outcomes(self, team_pts: np.ndarray) -> Outcomes:
        inp = self.inputs
        po = inp.playoffs
        S, _, T = team_pts.shape
        half_wins = np.broadcast_to(inp.base_half_wins.astype(float), (S, T)).copy()
        pf = np.broadcast_to(inp.base_points_for.astype(float), (S, T)).copy()
        week_won: dict[int, np.ndarray] = {}
        for w in inp.regular_weeks:
            s = team_pts[:, self.week_idx[w], :]
            pf += s
            won = np.zeros((S, T))
            for a, b in inp.matchups.get(w, []):
                won[:, a] = 2 * (s[:, a] > s[:, b]) + (s[:, a] == s[:, b])
                won[:, b] = 2 * (s[:, b] > s[:, a]) + (s[:, a] == s[:, b])
            head_to_head = won.copy()
            if inp.median_match:
                med = np.median(s, axis=1, keepdims=True)
                won += 2 * (s > med) + (s == med)
            half_wins += won
            week_won[w] = head_to_head == 2
        seeds = seed_teams(half_wins, pf, inp.divisions)
        if po.teams > 1:
            round_scores = []
            for rw in po.round_weeks():
                sc = np.zeros((S, T))
                for w in rw:
                    if w in self.week_idx:
                        sc += team_pts[:, self.week_idx[w], :]
                    elif w in inp.actual_points:
                        sc += inp.actual_points[w]
                round_scores.append(sc)
            finalists, champion = play_bracket(seeds, round_scores, po, inp.forced_playoff_winners)
        else:
            finalists = np.stack([seeds[:, 0], seeds[:, 0]], axis=1)
            champion = seeds[:, 0]
        return Outcomes(team_pts, half_wins, seeds, finalists, champion, week_won)

    def run(
        self,
        rosters: Mapping[int, list[str]] | None = None,
        starters: Starters | None = None,
        n: int | None = None,
    ) -> Outcomes:
        if not rosters and not starters:
            return self.baseline(n)
        return self.outcomes(self.team_points(rosters, starters, n))

    def baseline(self, n: int | None = None) -> Outcomes:
        n = n or self.n_sims
        if n not in self._baseline:
            self._baseline[n] = self.outcomes(self.base_team_pts[:n])
        return self._baseline[n]

    # -- summaries -----------------------------------------------------------------

    def summary(self, out: Outcomes) -> SimResult:
        inp = self.inputs
        po = inp.playoffs
        S, T = out.n, inp.n_teams
        made = out.made_playoffs(po.teams)
        bye = out.made_playoffs(po.byes) if po.byes else np.zeros((S, T), dtype=bool)
        seed_probs = np.zeros((T, max(po.teams, 1)))
        for k in range(min(po.teams, T)):
            seed_probs[:, k] = np.bincount(out.seeds[:, k], minlength=T) / S
        final = np.zeros(T)
        if po.teams > 1:
            final = np.bincount(out.finalists.ravel(), minlength=T) / S
        return SimResult(
            team_ids=list(inp.team_ids),
            team_labels=list(inp.team_labels),
            n_sims=S,
            playoffs=made.mean(0),
            bye=bye.mean(0),
            final=final,
            champion=np.bincount(out.champion, minlength=T) / S,
            seed_probs=seed_probs,
            mean_wins=(out.half_wins / 2).mean(0),
            week_win={w: v.mean(0) for w, v in out.week_won.items()},
            week_points={w: out.team_pts[:, wi].mean(0) for w, wi in self.week_idx.items()},
        )


def simulate(
    inputs: SimInputs,
    models: ModelSet,
    n_sims: int = 10_000,
    seed: int | None = 0,
    batch_size: int = 2_000,
) -> SimResult:
    ev = Evaluator(inputs, models, n_sims, seed, batch_size)
    return ev.summary(ev.baseline())
