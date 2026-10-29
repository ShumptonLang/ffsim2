"""``ffsim plan --dynamic``: rank this week's moves with managers acting all season.

1. Fit: play seasons where nobody moves (the static engine's world) to estimate
   each team's leverage, then alternate policy and refit (policy iteration);
   with FAAB, the bid price is refit from the value claims bought.
2. Rank: on fresh paths, compare "my first move is X, then the policy" against
   "the policy from now", with common random numbers. Candidates are screened
   on one set of paths and the finalists re-scored on another, so the reported
   gains aren't the lucky draws that picked them.
3. Explain: what the policy goes on to do for my team in the baseline paths
   (e.g. "adds the kicker in week 12 in 85% of seasons").
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from ffsim.decisions.common import Context, Delta, delta
from ffsim.dynamic.claims import Claim
from ffsim.dynamic.leverage import estimate_leverage
from ffsim.dynamic.policy import GreedyPolicy, PolicyOptions, TeamDecision
from ffsim.dynamic.rules import WaiverRules
from ffsim.dynamic.season import DynamicSeason, Move, SeasonRun, won_gains
from ffsim.model.models import ModelSet
from ffsim.sim.engine import Outcomes, SimResult, paired_delta, season_outcomes, summarize
from ffsim.sim.inputs import SimInputs


@dataclass
class DynamicOptions:
    fit_paths: int = 1_000
    screen_paths: int = 600
    eval_paths: int = 3_000
    batch: int = 250
    iterations: int = 2
    candidates: int = 10
    finalists: int = 5
    workers: int = 1
    seed: int = 0
    policy: PolicyOptions = field(default_factory=PolicyOptions)

    @classmethod
    def fast(cls, **kw: object) -> DynamicOptions:
        base: dict[str, object] = dict(
            fit_paths=200,
            screen_paths=200,
            eval_paths=600,
            batch=200,
            iterations=1,
            candidates=8,
            finalists=4,
            policy=PolicyOptions(top_adds=5, top_drops=3, max_moves=1, scenarios=4),
        )
        base.update(kw)
        return cls(**base)  # type: ignore[arg-type]


@dataclass
class DynamicOption:
    label: str
    move: Move
    delta: Delta
    static_champion: float | None = None  # static engine's Δ title odds, when comparable
    static_se: float | None = None
    p_won: float | None = None  # share of paths where the claim went through
    policy_moves: list[Move] | None = None  # set on the baseline row: the policy's own claims

    @property
    def is_policy(self) -> bool:
        return self.policy_moves is not None


@dataclass
class PlannedAdd:
    player: str
    share: float  # share of baseline paths in which my team adds him
    median_week: int


@dataclass
class DynamicPlan:
    odds: SimResult  # baseline: the policy from now, held-out paths
    iterations: list[float]  # my title odds per policy iteration (fit paths)
    options: list[DynamicOption]
    planned: list[PlannedAdd]
    adds_per_position: dict[str, float]  # my mean adds per season, by position
    n_paths: int


# -- batch execution -------------------------------------------------------------------

_SEASON: DynamicSeason | None = None


def _init_worker(season: DynamicSeason) -> None:
    global _SEASON
    _SEASON = season


BatchJob = tuple[
    int, int, int, int, GreedyPolicy | None, int | None, Move | None, list[TeamDecision] | None
]


def _run_batch(args: BatchJob) -> SeasonRun:
    seed, set_id, index, n, policy, focal, forced, first = args
    assert _SEASON is not None
    ss = np.random.SeedSequence([seed, set_id, index])
    draws = _SEASON.draw(ss, n)
    return _SEASON.run(
        draws, policy, seed=seed * 1000 + index, focal=focal, forced=forced, first=first
    )


class Runner:
    """Runs a set of paths in batches, optionally across processes."""

    def __init__(self, season: DynamicSeason, opts: DynamicOptions) -> None:
        self.season = season
        self.opts = opts
        self.pool: ProcessPoolExecutor | None = None
        if opts.workers > 1:
            self.pool = ProcessPoolExecutor(
                opts.workers, initializer=_init_worker, initargs=(season,)
            )
        else:
            _init_worker(season)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()

    def run(
        self,
        set_id: int,
        n: int,
        policy: GreedyPolicy | None,
        focal: int | None = None,
        forced: Move | None = None,
        first: list[TeamDecision] | None = None,
    ) -> SeasonRun:
        sizes = [min(self.opts.batch, n - s) for s in range(0, n, self.opts.batch)]
        jobs = [
            (self.opts.seed, set_id, i, b, policy, focal, forced, first)
            for i, b in enumerate(sizes)
        ]
        if self.pool is not None:
            parts = list(self.pool.map(_run_batch, jobs))
        else:
            _init_worker(self.season)
            parts = [_run_batch(j) for j in jobs]
        moves = []
        offset = 0
        for part, b in zip(parts, sizes, strict=True):
            for m in part.moves:
                m.path += offset
                moves.append(m)
            offset += b
        won = [p.first_claim_won for p in parts]
        return SeasonRun(
            np.concatenate([p.team_pts for p in parts]),
            moves,
            np.concatenate(won) if all(w is not None for w in won) else None,  # type: ignore[arg-type]
        )


# -- the plan --------------------------------------------------------------------------

FIT_SET, SCREEN_SET, EVAL_SET = 1, 2, 3


def fit_policy(
    runner: Runner, season: DynamicSeason, focal: int, opts: DynamicOptions,
    progress: Callable[[str], None] = lambda s: None,
) -> tuple[GreedyPolicy, list[float]]:  # fmt: skip
    """Policy iteration: leverage from never-move seasons, then refit under the policy."""
    inp, rules = season.inputs, season.rules
    progress("fitting leverage (no moves)")
    run = runner.run(FIT_SET, opts.fit_paths, None)
    odds = [float((outcomes(season, run).champion == focal).mean())]
    leverage = estimate_leverage(inp, season.week_idx, run.team_pts)
    price: float | None = None
    budget = float(sum(rules.budgets.values()))
    policy = GreedyPolicy(leverage, price, rules.bid_min, rules.faab, opts.policy)
    for it in range(opts.iterations):
        progress(f"policy iteration {it + 1}/{opts.iterations}")
        run = runner.run(FIT_SET, opts.fit_paths, policy)
        odds.append(float((outcomes(season, run).champion == focal).mean()))
        new = estimate_leverage(inp, season.week_idx, run.team_pts)
        leverage = 0.5 * leverage + 0.5 * new
        if rules.faab:
            spent = float(won_gains(run, inp.n_teams, opts.fit_paths).sum())
            price = budget / spent if spent > 0 else None
        policy = GreedyPolicy(leverage, price, rules.bid_min, rules.faab, opts.policy)
    return policy, odds


def outcomes(season: DynamicSeason, run: SeasonRun) -> Outcomes:
    return season_outcomes(season.inputs, season.week_idx, run.team_pts)


def _claim_move(season: DynamicSeason, c: Claim) -> Move:
    p = season.players
    return Move("claim", p[c.add], p[c.drop] if c.drop >= 0 else None, c.bid)


def candidate_moves(
    season: DynamicSeason, first: list[TeamDecision], focal: int, extra: list[Move], limit: int
) -> tuple[list[Move], list[Move]]:
    """First moves worth comparing against the policy, and the policy's own claims.

    Candidates: doing nothing, the policy's deferred "stash" move, each of its
    claims on its own, and ``extra`` (e.g. the static plan's pickups and trades).
    """
    mine = first[focal]
    policy_moves = [_claim_move(season, c) for c in mine.claims[0]]
    moves: list[Move] = [Move("none")]
    if mine.stash[0] is not None:
        moves.append(_claim_move(season, mine.stash[0]))
    if len(policy_moves) > 1:
        moves.extend(policy_moves)
    moves.extend(extra)
    seen: set[tuple[str | None, ...]] = set()
    out = []
    for m in moves:
        key = (m.kind, m.add, m.drop, *m.give, *m.get)
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out[:limit], policy_moves


def make_dynamic_plan(
    inputs: SimInputs,
    models: ModelSet,
    rules: WaiverRules,
    focal_rid: int,
    opts: DynamicOptions,
    extra: list[Move] | None = None,
    progress: Callable[[str], None] = lambda s: None,
) -> DynamicPlan:
    season = DynamicSeason(inputs, models, rules, opts.policy)
    focal = season.team_idx[focal_rid]
    runner = Runner(season, opts)
    try:
        policy, odds = fit_policy(runner, season, focal, opts, progress)
        first = season.first_decisions(policy, opts.seed)
        extra_ok = [
            m for m in (extra or [])
            if all(p in season.pidx for p in (m.add, m.drop, *m.give, *m.get) if p)
        ]  # fmt: skip
        moves, policy_moves = candidate_moves(season, first, focal, extra_ok, opts.candidates)
        progress(f"screening {len(moves)} first moves")
        base = runner.run(SCREEN_SET, opts.screen_paths, policy, focal, None, first)
        base_champ = outcomes(season, base).champion == focal
        screened = []
        for m in moves:
            alt = runner.run(SCREEN_SET, opts.screen_paths, policy, focal, m, first)
            d, _ = paired_delta(base_champ, outcomes(season, alt).champion == focal)
            screened.append((d, m))
        screened.sort(key=lambda dm: -dm[0])
        finalists = [m for _, m in screened[: opts.finalists]]
        progress(f"scoring {len(finalists)} finalists on fresh paths")
        base = runner.run(EVAL_SET, opts.eval_paths, policy, focal, None, first)
        base_out = outcomes(season, base)
        options = [_option(season, base_out, focal, m, runner, opts, policy, first)
                   for m in finalists]  # fmt: skip
        options.append(
            DynamicOption(
                policy_label(policy_moves),
                Move("policy"),
                Delta(0.0, 0.0, 0.0, 0.0),
                policy_moves=policy_moves,
            )
        )
        options.sort(key=lambda o: (-o.delta.champion, not o.is_policy))
        planned, per_pos = _planned(season, base, focal, opts.eval_paths)
        return DynamicPlan(
            odds=summarize(season.inputs, season.week_idx, base_out),
            iterations=odds,
            options=options,
            planned=planned,
            adds_per_position=per_pos,
            n_paths=opts.eval_paths,
        )
    finally:
        runner.close()


def _option(
    season: DynamicSeason, base: Outcomes, focal: int, move: Move, runner: Runner,
    opts: DynamicOptions, policy: GreedyPolicy, first: list[TeamDecision],
) -> DynamicOption:  # fmt: skip
    alt_run = runner.run(EVAL_SET, opts.eval_paths, policy, focal, move, first)
    alt = outcomes(season, alt_run)
    po = season.inputs.playoffs.teams
    dc, sc = paired_delta(base.champion == focal, alt.champion == focal)
    dp, sp = paired_delta(base.made_playoffs(po)[:, focal], alt.made_playoffs(po)[:, focal])
    p_won = None
    if alt_run.first_claim_won is not None:
        p_won = float(alt_run.first_claim_won.mean())
    return DynamicOption(label(move), move, Delta(dc, sc, dp, sp), p_won=p_won)


def policy_label(moves: list[Move]) -> str:
    if not moves:
        return "Policy: no move this week"
    return "Policy: " + "; ".join(label(m) for m in moves)


def label(move: Move) -> str:
    if move.kind == "none":
        return "No move this week"
    if move.kind == "trade":
        return f"Trade {', '.join(move.give)} for {', '.join(move.get)}"
    return f"Add {move.add}" + (f", drop {move.drop}" if move.drop else "")


def _planned(
    season: DynamicSeason, run: SeasonRun, focal: int, n_paths: int, top: int = 8
) -> tuple[list[PlannedAdd], dict[str, float]]:
    weeks: dict[int, list[int]] = defaultdict(list)
    per_pos: Counter[str] = Counter()
    for m in run.moves:
        if m.team != focal:
            continue
        weeks[m.add].append(season.inputs.weeks[m.step])
        pos = (season.inputs.positions.get(season.players[m.add]) or ("?",))[0]
        per_pos[pos] += 1
    ranked = sorted(weeks.items(), key=lambda kv: -len(kv[1]))[:top]
    planned = [
        PlannedAdd(season.players[p], len(set_paths(run, focal, p)) / n_paths,
                   int(np.median(w)))
        for p, w in ranked
    ]  # fmt: skip
    return planned, {k: v / n_paths for k, v in per_pos.most_common()}


def set_paths(run: SeasonRun, focal: int, player: int) -> set[int]:
    return {m.path for m in run.moves if m.team == focal and m.add == player}


def static_comparison(ctx: Context, options: list[DynamicOption]) -> None:
    """Fill each option's Δ title odds under the static engine (fixed rosters), if possible.

    The policy row's static counterpart is its claims applied all at once.
    """
    ev, rid = ctx.ev, ctx.rid
    universe = set(ev.players)
    for o in options:
        moves = o.policy_moves if o.policy_moves is not None else [o.move]
        rosters: dict[int, list[str]] = {rid: list(ev.inputs.rosters[rid])}
        ok = True
        for m in moves:
            if m.kind == "claim" and m.add in universe:
                mine = [p for p in rosters[rid] if p != m.drop] + [m.add]
                rosters[rid] = mine
            elif m.kind == "trade" and all(p in universe for p in m.give + m.get):
                rosters.update({r: list(ps) for r, ps in m.rosters.items()})
            elif m.kind not in ("none", "policy"):
                ok = False
        if not ok:
            continue
        d = delta(ev, ev.run(rosters=rosters), ctx.ti)
        o.static_champion, o.static_se = d.champion, d.champion_se
