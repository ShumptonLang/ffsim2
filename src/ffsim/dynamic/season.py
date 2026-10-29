"""Play closed-loop seasons: decide, claim, set lineups, score, learn, repeat.

Each decision step comes before that week's games: every manager proposes
claims from what he can observe, claims are resolved under the league's rules
(players dropped go on waivers until the next step), lineups are set from
beliefs among the players available that week (so bench players cover for the
inactive, as in the static engine), the week is scored from the drawn points,
and everyone updates beliefs from the scores.

At the first step every path has the same information, so the first decisions
are made once (with more imagined scenarios) and shared, which also lets a
forced first move be compared against the policy's with common random numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ffsim.dynamic.beliefs import Beliefs
from ffsim.dynamic.claims import Claim, apply_claim, resolve
from ffsim.dynamic.lineups import SlotSpec, lineups
from ffsim.dynamic.paths import PathDraws, draw_paths, news_variances, unknown_news_var
from ffsim.dynamic.policy import GreedyPolicy, PolicyOptions, TeamDecision, contested
from ffsim.dynamic.rules import WaiverRules
from ffsim.dynamic.state import FREE, PathLeague
from ffsim.dynamic.value import (
    RosterValuer,
    expected_points,
    imagined_availability,
    imagined_points,
    known_means,
)
from ffsim.model.models import ModelSet
from ffsim.sim.engine import DrawSetup
from ffsim.sim.inputs import SimInputs

FIRST_STEP_SCENARIO_FACTOR = 8


@dataclass(frozen=True)
class Move:
    """A first-week action for the focal team."""

    kind: str  # "none" | "claim" | "trade"
    add: str | None = None
    drop: str | None = None
    bid: int = 0
    partner: int | None = None  # roster_id
    give: tuple[str, ...] = ()
    get: tuple[str, ...] = ()
    rosters: dict[int, tuple[str, ...]] = field(default_factory=dict)  # trade: new rosters


@dataclass
class MoveRecord:
    path: int
    step: int
    team: int
    add: int
    drop: int
    bid: int
    gain: float


@dataclass
class SeasonRun:
    team_pts: np.ndarray  # (B, W, T)
    moves: list[MoveRecord]
    first_claim_won: np.ndarray | None = None  # (B,) the forced claim succeeded


class DynamicSeason:
    """Everything fixed across paths: universe, draw setup, slot layout, rules."""

    def __init__(
        self,
        inputs: SimInputs,
        models: ModelSet,
        rules: WaiverRules,
        opts: PolicyOptions | None = None,
        news: bool = True,
    ) -> None:
        self.inputs = inputs
        self.models = models
        self.rules = rules
        self.opts = opts or PolicyOptions()
        self.news = news
        self.players = inputs.universe
        self.pidx = {p: i for i, p in enumerate(self.players)}
        self.team_idx = {rid: i for i, rid in enumerate(inputs.team_ids)}
        self.week_idx = {w: i for i, w in enumerate(inputs.weeks)}
        self.setup = DrawSetup.build(inputs, models, self.players)
        self.spec = SlotSpec.build(inputs.slots, self.players, inputs.positions)
        W, P = self.setup.means.shape
        self.news_var = np.zeros((W, W, P))  # [t, w] news still unknown at step t
        if news:
            var_at, _, K = news_variances(inputs, models, self.players)
            if K > 0:
                self.news_var = unknown_news_var(var_at, K)

    @property
    def n_weeks(self) -> int:
        return len(self.inputs.weeks)

    def draw(self, seed: int | np.random.SeedSequence, n: int) -> PathDraws:
        ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        outcomes, news = ss.spawn(2)
        return draw_paths(
            self.setup,
            self.inputs,
            self.models,
            self.players,
            np.random.default_rng(outcomes),
            np.random.default_rng(news),
            n,
            news=self.news,
        )

    # -- one season ------------------------------------------------------------------

    def run(
        self,
        draws: PathDraws,
        policy: GreedyPolicy | None,
        seed: int = 0,
        focal: int | None = None,
        forced: Move | None = None,
        first: list[TeamDecision] | None = None,
        learn: bool = True,
        kind: str | None = None,
    ) -> SeasonRun:
        """Play every path. ``policy`` None means nobody ever moves.

        ``forced`` replaces the focal team's first-step decision. ``first`` reuses
        precomputed first-step decisions (see ``first_decisions``). ``kind``
        overrides the league's waiver processing.
        """
        kind = kind or self.rules.kind
        inp = self.inputs
        B, W, T = draws.n, self.n_weeks, inp.n_teams
        state = PathLeague.initial(inp, self.pidx, self.rules, B)
        beliefs = Beliefs(B, self.setup.rho)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
        team_pts = np.zeros((B, W, T), dtype=np.float32)
        moves: list[MoveRecord] = []
        claim_won = None
        if forced is not None and forced.kind == "trade":
            self._apply_trade(state, forced)
        for t in range(W):
            means = known_means(draws, self.setup, t)
            exp_pts = expected_points(means, beliefs, self.setup, self.models.errors)
            if policy is not None:
                if t == 0:
                    decisions = first if first is not None else self.first_decisions(policy, seed)
                    won = self._first_step(state, decisions, focal, forced, moves, kind)
                    claim_won = won
                else:
                    absence = draws.absence[:, t - 1]
                    avail = imagined_availability(
                        self.models.availability, self.setup, absence, t, self.opts.scenarios, rng
                    )
                    valuer = RosterValuer(
                        self._imagined(means, exp_pts, beliefs, t, self.opts.scenarios, rng),
                        avail,
                        self.spec,
                    )
                    decisions = [policy.decide(state, valuer, self.spec, ti, t) for ti in range(T)]
                    for b in range(B):
                        for c in resolve(state, b, contested(decisions, b), kind, t):
                            moves.append(MoveRecord(b, t, c.team, c.add, c.drop, c.bid, c.gain))
            team_pts[:, t] = self._score_week(state, exp_pts[:, 0], draws, t)
            self._update_standings(state, team_pts[:, t], t)
            if learn:
                played = draws.avail[:, t] & self.setup.active[t]
                beliefs.observe(draws.z_obs[:, t], played)
        return SeasonRun(team_pts, moves, claim_won)

    def first_decisions(
        self, policy: GreedyPolicy, seed: int = 0, focal: int | None = None, n_options: int = 0
    ) -> list[TeamDecision]:
        """Every team's first-step decision, made once from the shared information.

        The ``focal`` team also gets its ``n_options`` best first moves.
        """
        state = PathLeague.initial(self.inputs, self.pidx, self.rules, 1)
        beliefs = Beliefs(1, self.setup.rho)
        means = self.setup.means[None]
        exp_pts = expected_points(means, beliefs, self.setup, self.models.errors)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 11]))
        k = self.opts.scenarios * FIRST_STEP_SCENARIO_FACTOR
        avail = imagined_availability(self.models.availability, self.setup, None, 0, k, rng)
        valuer = RosterValuer(self._imagined(means, exp_pts, beliefs, 0, k, rng), avail, self.spec)
        return [
            policy.decide(state, valuer, self.spec, ti, 0, n_options=n_options * (ti == focal))
            for ti in range(self.inputs.n_teams)
        ]

    def _imagined(
        self,
        means: np.ndarray,
        exp_pts: np.ndarray,
        beliefs: Beliefs,
        t: int,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Per-scenario expected points for the valuer (or today's, if switched off)."""
        if not self.opts.imagine_projections:
            return exp_pts
        return imagined_points(
            means, self.news_var[t, t:], beliefs, self.setup.active[t:],
            self.models.errors, self.setup.positions, k, rng,
        )  # fmt: skip

    def _first_step(
        self,
        state: PathLeague,
        decisions: list[TeamDecision],
        focal: int | None,
        forced: Move | None,
        moves: list[MoveRecord],
        kind: str,
    ) -> np.ndarray | None:
        """Resolve the shared first-step claims once and copy the result to every path."""
        one = state.take([0])
        claims = contested(decisions, 0)
        forced_claim = None
        if forced is not None and focal is not None:
            claims = [c for c in claims if c.team != focal]
            if forced.kind == "claim" and forced.add is not None:
                forced_claim = Claim(
                    team=focal,
                    add=self.pidx[forced.add],
                    drop=self.pidx[forced.drop] if forced.drop else -1,
                    bid=forced.bid,
                )
                claims.append(forced_claim)
        won = resolve(one, 0, claims, kind, 0)
        for c in won:
            for b in range(state.n):
                moves.append(MoveRecord(b, 0, c.team, c.add, c.drop, c.bid, c.gain))
        for name in ("rosters", "owner", "claimable_from", "budget", "priority"):
            getattr(state, name)[:] = getattr(one, name)[0]
        if forced_claim is None:
            return None
        return np.full(state.n, forced_claim in won)

    def _apply_trade(self, state: PathLeague, move: Move) -> None:
        for rid, players in move.rosters.items():
            ti = self.team_idx[rid]
            old = state.rosters[0, ti]
            for p in old[old >= 0]:
                if state.owner[0, p] == ti:
                    state.owner[:, p] = FREE
                    state.claimable_from[:, p] = 1
            new = np.full(old.shape, -1, dtype=np.int32)
            idx = [self.pidx[p] for p in players][: len(new)]
            new[: len(idx)] = idx
            state.rosters[:, ti] = new
        for rid, players in move.rosters.items():
            for p in players:
                state.owner[:, self.pidx[p]] = self.team_idx[rid]
                state.claimable_from[:, self.pidx[p]] = 0

    def _score_week(
        self, state: PathLeague, exp_now: np.ndarray, draws: PathDraws, t: int
    ) -> np.ndarray:
        """(B, T) points: lineups set from beliefs among players available this week."""
        B, T, R = state.rosters.shape
        P = len(self.players)
        idx = np.where(state.rosters >= 0, state.rosters, P)
        b = np.arange(B)[:, None, None]
        exp_ext = np.concatenate([exp_now, np.zeros((B, 1), np.float32)], 1)
        av_ext = np.concatenate([draws.avail[:, t], np.zeros((B, 1), bool)], 1)
        pts_ext = np.concatenate([draws.pts[:, t], np.zeros((B, 1), np.float32)], 1)
        vals = np.where(av_ext[b, idx], exp_ext[b, idx], -np.inf)  # (B, T, R)
        elig = self.spec.elig[idx]
        _, starters = lineups(vals, elig, self.spec, exact=True)
        got = np.take_along_axis(pts_ext[b, idx], np.maximum(starters, 0), axis=-1)
        return np.where(starters >= 0, got, 0.0).sum(-1)

    def _update_standings(self, state: PathLeague, pts: np.ndarray, t: int) -> None:
        w = self.inputs.weeks[t]
        if w not in self.inputs.regular_weeks:
            return
        state.points_for += pts
        for a, b in self.inputs.matchups.get(w, []):
            state.half_wins[:, a] += 2 * (pts[:, a] > pts[:, b]) + (pts[:, a] == pts[:, b])
            state.half_wins[:, b] += 2 * (pts[:, b] > pts[:, a]) + (pts[:, a] == pts[:, b])


def won_gains(run: SeasonRun, n_teams: int, n_paths: int) -> np.ndarray:
    """(T,) mean value gained per path through claims won, per team."""
    out = np.zeros(n_teams)
    for m in run.moves:
        out[m.team] += m.gain
    return out / max(n_paths, 1)


__all__ = ["DynamicSeason", "Move", "SeasonRun", "apply_claim", "won_gains"]
