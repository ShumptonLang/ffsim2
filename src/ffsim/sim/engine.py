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


@dataclass
class DrawSetup:
    """Per-player inputs to the outcome draws, computed once per universe."""

    means: np.ndarray  # (W, P) projected points
    positions: list[str | None]
    rho: np.ndarray  # (P,) talent persistence
    hazard: np.ndarray  # (P,) weekly injury hazard
    first_play: np.ndarray  # (P,) chance of playing the first simulated week
    layouts: list  # per week, factor-copula layout
    active: np.ndarray  # (W, P) team plays and projection > 0

    @classmethod
    def build(cls, inputs: SimInputs, models: ModelSet, players: list[str]) -> DrawSetup:
        W, P = len(inputs.weeks), len(players)
        positions = [(inputs.positions.get(p) or (None,))[0] for p in players]
        means = np.array(
            [[inputs.means.get(w, {}).get(p, 0.0) for p in players] for w in inputs.weeks],
            dtype=np.float64,
        ).reshape(W, P)
        layouts = [
            week_layout(models.correlation, players, inputs.nfl_games.get(w, {}),
                        inputs.roles.get(w, {}))
            for w in inputs.weeks
        ]  # fmt: skip
        return cls(
            means=means,
            positions=positions,
            rho=np.array([models.talent.rho_for(pos) for pos in positions], dtype=np.float32),
            hazard=np.array([models.availability.hazard_for(pos) for pos in positions]),
            first_play=np.array(
                [models.availability.play_prob(inputs.statuses.get(p)) for p in players]
            ),
            layouts=layouts,
            active=means > 0,
        )


@dataclass
class DrawBatch:
    pts: np.ndarray  # (B, W, P) fantasy points
    avail: np.ndarray  # (B, W, P) plays that week
    z: np.ndarray | None = None  # (B, W, P) latent normals, if kept
    absence: np.ndarray | None = None  # (B, W, P) games missed in current absence, if kept


def draw_batch(
    setup: DrawSetup,
    models: ModelSet,
    rng: np.random.Generator,
    n: int,
    log_reveal: np.ndarray | None = None,
    keep_state: bool = False,
) -> DrawBatch:
    """Draw ``n`` seasons of outcomes for every player in the universe.

    ``log_reveal`` (n, W, P), if given, scales each player-week's projection by
    ``exp(log_reveal)`` before drawing points (projection news not yet known
    when the simulation starts). ``keep_state`` also returns the latent normals
    and the injury-absence state that in-simulation managers can observe.
    """
    W, P = setup.means.shape
    talent = rng.standard_normal((n, 1, P), dtype=np.float32)
    weekly = np.zeros((n, W, P), dtype=np.float32)
    for wi, lay in enumerate(setup.layouts):
        weekly[:, wi] = sample_weekly(rng, n, lay)
    z = np.sqrt(setup.rho) * talent + np.sqrt(1 - setup.rho) * weekly
    means = setup.means[None, :, :]
    if log_reveal is not None:
        means = means * np.exp(log_reveal)
    pts = models.errors.sample(z, means, setup.positions)
    if keep_state:
        avail, absence = models.availability.sample_with_state(
            rng, n, setup.active, setup.hazard, setup.first_play
        )
        return DrawBatch(pts, avail, z, absence)
    avail = models.availability.sample(rng, n, setup.active, setup.hazard, setup.first_play)
    return DrawBatch(pts, avail)


def season_outcomes(inp: SimInputs, week_idx: Mapping[int, int], team_pts: np.ndarray) -> Outcomes:
    """Standings, seeds and playoff results from team points (S, W, T)."""
    po = inp.playoffs
    S, _, T = team_pts.shape
    half_wins = np.broadcast_to(inp.base_half_wins.astype(float), (S, T)).copy()
    pf = np.broadcast_to(inp.base_points_for.astype(float), (S, T)).copy()
    week_won: dict[int, np.ndarray] = {}
    for w in inp.regular_weeks:
        s = team_pts[:, week_idx[w], :]
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
                if w in week_idx:
                    sc += team_pts[:, week_idx[w], :]
                elif w in inp.actual_points:
                    sc += inp.actual_points[w]
            round_scores.append(sc)
        finalists, champion = play_bracket(seeds, round_scores, po, inp.forced_playoff_winners)
    else:
        finalists = np.stack([seeds[:, 0], seeds[:, 0]], axis=1)
        champion = seeds[:, 0]
    return Outcomes(team_pts, half_wins, seeds, finalists, champion, week_won)


def summarize(inp: SimInputs, week_idx: Mapping[int, int], out: Outcomes) -> SimResult:
    """Probabilities and averages across simulations for one scenario."""
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
        week_points={w: out.team_pts[:, wi].mean(0) for w, wi in week_idx.items()},
    )


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
        setup = DrawSetup.build(self.inputs, self.models, self.players)
        W, P = setup.means.shape
        pts = np.zeros((n_sims, W, P), dtype=np.float32)
        avail = np.zeros((n_sims, W, P), dtype=bool)
        rng = np.random.default_rng(seed)
        for start in range(0, n_sims, batch_size):
            B = min(batch_size, n_sims - start)
            batch = draw_batch(setup, self.models, rng, B)
            sl = slice(start, start + B)
            pts[sl] = batch.pts
            avail[sl] = batch.avail
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
        return season_outcomes(self.inputs, self.week_idx, team_pts)

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
        return summarize(self.inputs, self.week_idx, out)


def simulate(
    inputs: SimInputs,
    models: ModelSet,
    n_sims: int = 10_000,
    seed: int | None = 0,
    batch_size: int = 2_000,
) -> SimResult:
    ev = Evaluator(inputs, models, n_sims, seed, batch_size)
    return ev.summary(ev.baseline())
