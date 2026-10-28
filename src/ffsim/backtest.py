"""Backtesting: replay past NFL seasons and score the model's predictions.

Two kinds of replay share one prediction path:

* **Mock leagues** (``mock_replays``): leagues in a given format (teams, slots,
  scoring, playoffs) are drafted from preseason projections with some noise,
  then the season plays out with real historical scores, each team starting
  its projection-optimal lineup. This needs no historical Sleeper league, so it
  works for any league format and any season with history data.
* **Real leagues** (``sleeper_replay``): a completed Sleeper league, with rosters
  taken from each week's matchups and the real playoff results.

At each cutoff week the model predicts every team's playoff and title odds
using only what was known then: next week's projections, results through the
cutoff, preseason priors, and the NFL schedule. A records-only baseline (every
remaining game a coin flip) is scored the same way. Brier score and log loss
measure accuracy; skill = 1 - Brier(model) / Brier(baseline).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ffsim.data.history import HistoryStore
from ffsim.data.sleeper import SleeperClient
from ffsim.league import (
    SLOT_ELIGIBILITY,
    League,
    PlayoffFormat,
    parse_matchups,
)
from ffsim.model.bayes import prior_rate
from ffsim.model.correlation import assign_roles
from ffsim.model.models import ModelSet
from ffsim.sim.engine import Evaluator
from ffsim.sim.inputs import SimInputs
from ffsim.sim.lineups import best_lineup
from ffsim.sim.playoffs import play_bracket, seed_teams

LAST_WEEK = 17
POSITION_CAPS = {"QB": 2, "TE": 2, "K": 1, "DEF": 1}


# --- historical season data ------------------------------------------------------------


@dataclass
class SeasonData:
    """One NFL season of projections and results, scored for one league's settings."""

    season: int
    proj: dict[int, dict[str, float]]  # week -> player -> projected points
    actual: dict[int, dict[str, float]]  # week -> player -> points (played only)
    team: dict[int, dict[str, str]]  # week -> player -> NFL team
    opponents: dict[str, dict[int, str]]  # NFL team -> week -> opponent
    positions: dict[str, str]
    prior: dict[str, float]  # preseason per-game rate

    @classmethod
    def load(cls, store: HistoryStore, season: int, settings: Mapping[str, float]) -> SeasonData:
        proj: dict[int, dict[str, float]] = defaultdict(dict)
        actual: dict[int, dict[str, float]] = defaultdict(dict)
        team: dict[int, dict[str, str]] = defaultdict(dict)
        opponents: dict[str, dict[int, str]] = defaultdict(dict)
        positions: dict[str, str] = {}
        for r in store.rows([season], range(1, LAST_WEEK + 1)):
            m, a = r.scored(settings)
            proj[r.week][r.player_id] = max(m, 0.0)
            if r.played:
                actual[r.week][r.player_id] = a
            if r.team:
                team[r.week][r.player_id] = r.team
                if r.opponent:
                    opponents[r.team][r.week] = r.opponent
            positions[r.player_id] = r.position
        prior = {}
        for pid, (pos, stats) in store.preseason(season).items():
            rate = prior_rate(stats, settings, pos)
            if rate is not None:
                prior[pid] = rate
                positions.setdefault(pid, pos)
        return cls(season, dict(proj), dict(actual), dict(team), dict(opponents), positions, prior)

    def last_team(self, pid: str, week: int) -> str | None:
        for w in range(week, 0, -1):
            t = self.team.get(w, {}).get(pid)
            if t:
                return t
        return None

    def last_proj(self, pid: str, week: int) -> float:
        for w in range(week, 0, -1):
            m = self.proj.get(w, {}).get(pid)
            if m:
                return m
        return 0.0


# --- a season to replay ---------------------------------------------------------------


@dataclass
class Replay:
    label: str
    season: int
    slots: list[str]
    playoffs: PlayoffFormat
    capacity: int
    team_ids: list[int]
    rosters: dict[int, dict[int, list[str]]]  # week -> roster_id -> players (lineup-eligible)
    schedule: dict[int, list[tuple[int, int]]]  # regular week -> [(roster_id, roster_id)]
    points: dict[int, dict[int, float]]  # week -> roster_id -> actual points
    median_match: bool = False
    divisions: list[int | None] = field(default_factory=list)
    # Realized outcome; computed from points when not given (mock leagues).
    playoff_teams: set[int] | None = None
    champion: int | None = None

    @property
    def regular_weeks(self) -> list[int]:
        return sorted(self.schedule)

    def roster_at(self, week: int) -> dict[int, list[str]]:
        known = [w for w in self.rosters if w <= week]
        return self.rosters[max(known)] if known else self.rosters[min(self.rosters)]

    def realized(self) -> tuple[set[int], int]:
        if self.playoff_teams is not None and self.champion is not None:
            return self.playoff_teams, self.champion
        idx = {rid: i for i, rid in enumerate(self.team_ids)}
        T = len(self.team_ids)
        hw, pf = np.zeros((1, T)), np.zeros((1, T))
        for w in self.regular_weeks:
            for a, b in self.schedule[w]:
                pa, pb = self.points[w].get(a, 0.0), self.points[w].get(b, 0.0)
                hw[0, idx[a]] += 2 * (pa > pb) + (pa == pb)
                hw[0, idx[b]] += 2 * (pb > pa) + (pa == pb)
            for rid in self.team_ids:
                pf[0, idx[rid]] += self.points[w].get(rid, 0.0)
        seeds = seed_teams(hw, pf, self.divisions)
        made = {self.team_ids[i] for i in seeds[0, : self.playoffs.teams]}
        rounds = [
            np.array([[sum(self.points.get(w, {}).get(r, 0.0) for w in rw) for r in self.team_ids]])
            for rw in self.playoffs.round_weeks()
        ]
        _, champ = play_bracket(seeds, rounds, self.playoffs)
        return made, self.team_ids[int(champ[0])]


def round_robin(team_ids: Sequence[int], weeks: Sequence[int]) -> dict[int, list[tuple[int, int]]]:
    """Circle-method schedule, repeated as needed (odd counts get a bye team)."""
    ids = list(team_ids) + ([-1] if len(team_ids) % 2 else [])
    n = len(ids)
    rounds = []
    arr = ids[:]
    for _ in range(n - 1):
        rounds.append([(arr[i], arr[n - 1 - i]) for i in range(n // 2)])
        arr = [arr[0], arr[-1], *arr[1:-1]]
    return {
        w: [(a, b) for a, b in rounds[k % len(rounds)] if a != -1 and b != -1]
        for k, w in enumerate(weeks)
    }


def mock_draft(
    data: SeasonData,
    n_teams: int,
    slots: Sequence[str],
    capacity: int,
    rng: np.random.Generator,
    noise: float = 0.15,
) -> dict[int, list[str]]:
    """Snake draft by noisy preseason value, filling starting slots before depth."""
    startable = {p for s in slots for p in SLOT_ELIGIBILITY.get(s, ())}
    caps = dict(POSITION_CAPS)
    if "SUPER_FLEX" in slots:
        caps["QB"] = 3
    pool = [
        (rate * np.exp(rng.normal(0, noise)), pid)
        for pid, rate in data.prior.items()
        if data.positions.get(pid) in startable
    ]
    pool.sort(reverse=True)
    rosters: dict[int, list[str]] = {t: [] for t in range(1, n_teams + 1)}
    taken: set[str] = set()

    def open_positions(roster: list[str]) -> tuple[set[str], int]:
        """Positions that could fill a still-empty starting slot, and how many are empty."""
        filled = best_lineup(
            list(slots), roster, {p: 1.0 for p in roster}, {p: (data.positions[p],) for p in roster}
        )
        empty = [s for s, p in zip(slots, filled, strict=True) if p is None]
        return {pos for s in empty for pos in SLOT_ELIGIBILITY.get(s, ())}, len(empty)

    order = list(range(1, n_teams + 1))
    for rnd in range(capacity):
        for t in order if rnd % 2 == 0 else order[::-1]:
            roster = rosters[t]
            open_pos, n_empty = open_positions(roster)
            must_fill = capacity - len(roster) <= n_empty
            for _, pid in pool:
                if pid in taken:
                    continue
                pos = data.positions[pid]
                if sum(1 for p in roster if data.positions[p] == pos) >= caps.get(pos, 99):
                    continue
                if must_fill and pos not in open_pos:
                    continue
                roster.append(pid)
                taken.add(pid)
                break
    return rosters


def mock_replay(
    data: SeasonData,
    league: League,
    rng: np.random.Generator,
    label: str,
) -> Replay:
    slots = list(league.starter_slots)
    capacity = sum(1 for s in league.roster_positions if s not in ("IR", "TAXI"))
    n = league.total_rosters
    rosters = mock_draft(data, n, slots, capacity, rng)
    team_ids = sorted(rosters)
    regular = list(league.regular_season_weeks)
    last = min(league.playoffs.last_week, LAST_WEEK)
    points: dict[int, dict[int, float]] = {}
    for w in range(1, last + 1):
        proj = data.proj.get(w, {})
        positions = {p: (data.positions[p],) for ps in rosters.values() for p in ps}
        points[w] = {}
        for rid, ps in rosters.items():
            lineup = best_lineup(slots, ps, proj, positions)
            points[w][rid] = sum(data.actual.get(w, {}).get(p, 0.0) for p in lineup if p)
    return Replay(
        label=label,
        season=data.season,
        slots=slots,
        playoffs=league.playoffs,
        capacity=capacity,
        team_ids=team_ids,
        rosters={1: rosters},
        schedule=round_robin(team_ids, regular),
        points=points,
        median_match=bool(league.raw_settings.get("league_average_match")),
    )


def sleeper_replay(client: SleeperClient, league_id: str) -> tuple[Replay, League]:
    """A completed Sleeper league: weekly rosters, scores and the real bracket."""
    league = League.from_sleeper(client.league(league_id))
    last = min(league.playoffs.last_week, LAST_WEEK)
    rosters: dict[int, dict[int, list[str]]] = {}
    points: dict[int, dict[int, float]] = {}
    schedule: dict[int, list[tuple[int, int]]] = {}
    team_ids: set[int] = set()
    for w in range(league.start_week, last + 1):
        rows = client.matchups(league_id, w, final=True)
        rosters[w] = {int(r["roster_id"]): [str(p) for p in r.get("players") or []] for r in rows}
        points[w] = {int(r["roster_id"]): float(r.get("points") or 0.0) for r in rows}
        team_ids |= set(points[w])
        if w in league.regular_season_weeks:
            schedule[w] = [(m.home, m.away) for m in parse_matchups(w, rows, scored=True)]
    bracket = client.winners_bracket(league_id)
    made = {int(r[k]) for r in bracket for k in ("t1", "t2") if isinstance(r.get(k), int)}
    champ = next((int(r["w"]) for r in bracket if r.get("p") == 1 and r.get("w")), None)
    replay = Replay(
        label=f"{league.name} ({league.season})",
        season=league.season,
        slots=list(league.starter_slots),
        playoffs=league.playoffs,
        capacity=sum(1 for s in league.roster_positions if s not in ("IR", "TAXI")),
        team_ids=sorted(team_ids),
        rosters=rosters,
        schedule=schedule,
        points=points,
        median_match=bool(league.raw_settings.get("league_average_match")),
        playoff_teams=made or None,
        champion=champ,
    )
    return replay, league


# --- predictions at a cutoff ----------------------------------------------------------


def inputs_at(replay: Replay, data: SeasonData, cutoff: int, models: ModelSet) -> SimInputs:
    """Simulation inputs using only information available after ``cutoff`` week."""
    po = replay.playoffs
    nxt = cutoff + 1
    weeks = list(range(nxt, min(po.last_week, LAST_WEEK) + 1))
    regular = [w for w in replay.regular_weeks if w > cutoff]
    rosters = replay.roster_at(nxt)
    idx = {rid: i for i, rid in enumerate(replay.team_ids)}
    players = sorted({p for ps in rosters.values() for p in ps if p in data.positions})
    rosters = {rid: [p for p in ps if p in data.positions] for rid, ps in rosters.items()}

    hw = np.zeros(len(replay.team_ids))
    pf = np.zeros(len(replay.team_ids))
    for w in replay.regular_weeks:
        if w > cutoff:
            continue
        pts = replay.points.get(w, {})
        for a, b in replay.schedule[w]:
            pa, pb = pts.get(a, 0.0), pts.get(b, 0.0)
            hw[idx[a]] += 2 * (pa > pb) + (pa == pb)
            hw[idx[b]] += 2 * (pb > pa) + (pa == pb)
        for rid in replay.team_ids:
            pf[idx[rid]] += pts.get(rid, 0.0)

    # Projections as of the cutoff: next week's, carried forward to later weeks.
    talent = models.talent
    means: dict[int, dict[str, float]] = {w: {} for w in weeks}
    games: dict[int, dict[str, tuple[str, str]]] = {w: {} for w in weeks}
    for p in players:
        pos = data.positions[p]
        team = data.last_team(p, nxt)
        ref = data.last_proj(p, nxt)
        obs = [data.actual[w][p] for w in range(1, nxt) if p in data.actual.get(w, {})]
        rate = talent.rate(pos, data.prior.get(p), float(np.mean(obs)) if obs else 0.0, len(obs))
        for w in weeks:
            opp = data.opponents.get(team or "", {}).get(w)
            if opp is None:
                continue  # bye (or no team)
            base = data.proj.get(nxt, {}).get(p, 0.0) if w == nxt else ref
            means[w][p] = talent.centre(pos, base, rate)
            games[w][p] = (team, opp)  # type: ignore[assignment]
    role_rows = [
        (p, data.team.get(nxt, {}).get(p) or data.last_team(p, nxt), data.positions.get(p), m)
        for p, m in data.proj.get(nxt, {}).items()
    ]
    roles = assign_roles(role_rows)
    return SimInputs(
        team_ids=list(replay.team_ids),
        team_labels=[f"Team {rid}" for rid in replay.team_ids],
        rosters=rosters,
        slots=list(replay.slots),
        weeks=weeks,
        regular_weeks=regular,
        playoffs=po,
        matchups={w: [(idx[a], idx[b]) for a, b in replay.schedule[w]] for w in regular},
        base_half_wins=hw,
        base_points_for=pf,
        divisions=replay.divisions,
        median_match=replay.median_match,
        means=means,
        positions={p: (data.positions[p],) for p in players},
        nfl_games=games,
        roles={w: {p: roles[p] for p in players if p in roles} for w in weeks},
        roster_capacity=replay.capacity,
    )


def baseline_probs(
    replay: Replay, cutoff: int, n_sims: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Records-only odds: every remaining game (and playoff game) is a coin flip."""
    T = len(replay.team_ids)
    idx = {rid: i for i, rid in enumerate(replay.team_ids)}
    hw = np.zeros((n_sims, T))
    pf = np.zeros((n_sims, T))
    for w in replay.regular_weeks:
        pts = replay.points.get(w, {})
        if w <= cutoff:
            for a, b in replay.schedule[w]:
                pa, pb = pts.get(a, 0.0), pts.get(b, 0.0)
                hw[:, idx[a]] += 2 * (pa > pb) + (pa == pb)
                hw[:, idx[b]] += 2 * (pb > pa) + (pa == pb)
            pf += np.array([pts.get(r, 0.0) for r in replay.team_ids])
        else:
            for a, b in replay.schedule[w]:
                win = rng.random(n_sims) < 0.5
                hw[:, idx[a]] += 2 * win
                hw[:, idx[b]] += 2 * ~win
    seeds = seed_teams(hw, pf + rng.random((n_sims, T)) * 1e-3, replay.divisions)
    made = np.zeros((n_sims, T), dtype=bool)
    np.put_along_axis(made, seeds[:, : replay.playoffs.teams], True, axis=1)
    rounds = [rng.random((n_sims, T)) for _ in replay.playoffs.round_weeks()]
    _, champ = play_bracket(seeds, rounds, replay.playoffs)
    return made.mean(0), np.bincount(champ, minlength=T) / n_sims


@dataclass
class Prediction:
    replay: str
    season: int
    cutoff: int
    team_ids: list[int]
    p_playoffs: np.ndarray
    p_champion: np.ndarray
    base_playoffs: np.ndarray
    base_champion: np.ndarray
    made: np.ndarray  # 0/1 per team
    won: np.ndarray  # 0/1 per team


def predict(
    replay: Replay,
    data: SeasonData,
    cutoff: int,
    models: ModelSet,
    n_sims: int,
    seed: int = 0,
) -> Prediction:
    inp = inputs_at(replay, data, cutoff, models)
    ev = Evaluator(inp, models, n_sims=n_sims, seed=seed)
    res = ev.summary(ev.baseline())
    base_po, base_ch = baseline_probs(replay, cutoff, n_sims, np.random.default_rng(seed))
    made, champ = replay.realized()
    return Prediction(
        replay=replay.label,
        season=replay.season,
        cutoff=cutoff,
        team_ids=list(replay.team_ids),
        p_playoffs=res.playoffs,
        p_champion=res.champion,
        base_playoffs=base_po,
        base_champion=base_ch,
        made=np.array([rid in made for rid in replay.team_ids], dtype=float),
        won=np.array([rid == champ for rid in replay.team_ids], dtype=float),
    )


# --- scoring --------------------------------------------------------------------------


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-3) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass
class Score:
    n_teams: int
    brier: float
    brier_base: float
    log_loss: float
    log_loss_base: float

    @property
    def skill(self) -> float:
        return 1.0 - self.brier / self.brier_base if self.brier_base else 0.0


def score_predictions(preds: Iterable[Prediction], target: str) -> Score:
    preds = list(preds)
    if target == "playoffs":
        p = np.concatenate([x.p_playoffs for x in preds])
        b = np.concatenate([x.base_playoffs for x in preds])
        y = np.concatenate([x.made for x in preds])
    else:
        p = np.concatenate([x.p_champion for x in preds])
        b = np.concatenate([x.base_champion for x in preds])
        y = np.concatenate([x.won for x in preds])
    return Score(len(y), brier(p, y), brier(b, y), log_loss(p, y), log_loss(b, y))


def skill_interval(
    preds: Sequence[Prediction],
    target: str,
    n_boot: int = 1_000,
    level: float = 0.9,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap interval for skill, resampling whole replays.

    Teams in one league share a single outcome (exactly k make the playoffs),
    so replays, not teams, are the independent units.
    """
    by_replay: dict[str, list[Prediction]] = defaultdict(list)
    for p in preds:
        by_replay[p.replay].append(p)
    groups = list(by_replay.values())
    if len(groups) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    skills = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        sample = [p for i in pick for p in groups[i]]
        skills.append(score_predictions(sample, target).skill)
    lo, hi = np.quantile(skills, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def calibration_bins(
    preds: Iterable[Prediction], bins: Sequence[float] = (0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)
) -> list[tuple[float, float, float, int]]:
    """(bin low, mean predicted, observed rate, n) for playoff predictions."""
    preds = list(preds)
    p = np.concatenate([x.p_playoffs for x in preds])
    y = np.concatenate([x.made for x in preds])
    out = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        sel = (p >= lo) & (p < hi)
        if sel.any():
            out.append((lo, float(p[sel].mean()), float(y[sel].mean()), int(sel.sum())))
    return out
