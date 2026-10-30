"""Player availability: this week's injury designations and future injuries.

Sleeper's weekly projections already encode *known* absences (byes, players
ruled out, IR return timelines) as zero projections. This model adds what they
don't:

* this week's designation risk: a Questionable player plays with probability
  ~0.75 (configurable; Sleeper projects them at full value);
* new injuries in future weeks: each week a player who plays may get hurt and
  miss the next ``d`` team games. The weekly hazard per position and the absence
  length distribution are fitted from 2023-2025 data (a regular starter who
  played week w but was inactive for his team's next game).

In-game injuries in the week they happen are already part of the outcome
distribution's "dud" mass, so they are not modelled again here. Unavailable
starters are replaced from the bench by the lineup logic in the engine.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ffsim.data.history import PlayerSeason

SEASON_ENDING = 99  # absence lasting the rest of the season
MAX_DURATION = 10  # longer absences are pooled into SEASON_ENDING
SEASON_OUT = 127  # absence state code for a known season-ending injury

DEFAULT_STATUS_PLAY: dict[str, float] = {
    "Questionable": 0.75,
    "Doubtful": 0.10,
    "Out": 0.0,
    "IR": 0.0,
    "PUP": 0.0,
    "Sus": 0.0,
    "NA": 0.0,
    "DNR": 0.0,
    "COV": 0.0,
}
# Projection a player needs to count as a regular when fitting the hazard; this
# keeps backups who lose snaps to a returning starter from looking "injured".
REGULAR = {"QB": 12.0, "RB": 6.0, "WR": 6.0, "TE": 5.0, "K": 5.0, "DL": 3.0, "LB": 3.0, "DB": 3.0}


@dataclass
class AvailabilityModel:
    hazard: dict[str, float] = field(default_factory=dict)  # pos -> P(miss next game | played)
    # P(absence lasts d games) for d = 1..MAX_DURATION, then P(rest of season).
    durations: list[float] = field(
        default_factory=lambda: [0.43, 0.24, 0.11, 0.07, 0.04, 0.02, 0.01, 0.01, 0.01, 0.0, 0.06]
    )
    status_play: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_STATUS_PLAY))
    default_hazard: float = 0.06

    def hazard_for(self, pos: str | None) -> float:
        if pos == "DEF":
            return 0.0
        return float(self.hazard.get(pos or "", self.default_hazard))

    def play_prob(self, status: str | None) -> float:
        if not status:
            return 1.0
        return float(self.status_play.get(status, 1.0))

    def _duration_values(self) -> np.ndarray:
        return np.array(list(range(1, MAX_DURATION + 1)) + [SEASON_ENDING])

    def sample(
        self,
        rng: np.random.Generator,
        n_sims: int,
        active: np.ndarray,
        hazards: np.ndarray,
        first_week_play: np.ndarray,
    ) -> np.ndarray:
        """Availability (n_sims, W, P) as booleans (see ``sample_with_state``)."""
        return self._sample(rng, n_sims, active, hazards, first_week_play, False)[0]

    def sample_with_state(
        self,
        rng: np.random.Generator,
        n_sims: int,
        active: np.ndarray,
        hazards: np.ndarray,
        first_week_play: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Availability plus the absence state managers can observe."""
        avail, state = self._sample(rng, n_sims, active, hazards, first_week_play, True)
        assert state is not None
        return avail, state

    def _sample(
        self,
        rng: np.random.Generator,
        n_sims: int,
        active: np.ndarray,
        hazards: np.ndarray,
        first_week_play: np.ndarray,
        return_state: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Availability (n_sims, W, P) as booleans.

        ``active`` (W, P): the player's team plays and his projection is > 0.
        ``hazards`` (P,): weekly injury hazard. ``first_week_play`` (P,): chance
        he plays the first simulated week given his designation.

        With ``return_state``, also returns what a manager knows about absences after
        each week's games, as int8 (n_sims, W, P): -1 healthy, ``SEASON_OUT`` for
        a season-ending injury (announced), otherwise the number of team games
        missed so far in the current absence (0 right after the injury). The
        absence's drawn length stays hidden. The random draws are identical
        either way.
        """
        W, P = active.shape
        avail = np.zeros((n_sims, W, P), dtype=bool)
        state = np.full((n_sims, W, P), -1, dtype=np.int8) if return_state else None
        if W == 0:
            return avail, state
        pmf = np.asarray(self.durations, dtype=np.float64)
        pmf = pmf / pmf.sum()
        values = self._duration_values()
        remaining = np.zeros((n_sims, P), dtype=np.int32)
        missed = np.zeros((n_sims, P), dtype=np.int32)
        absent = np.zeros((n_sims, P), dtype=bool)
        season_out = np.zeros((n_sims, P), dtype=bool)
        for w in range(W):
            plays_team = np.broadcast_to(active[w], (n_sims, P))
            if w == 0:
                ok = rng.random((n_sims, P)) < first_week_play
            else:
                out = (remaining > 0) & plays_team
                remaining = remaining - out
                missed += out
                ok = ~out
            avail[:, w, :] = ok & plays_team
            absent &= ~avail[:, w, :]
            if w < W - 1:
                hurt = avail[:, w, :] & (remaining == 0) & (rng.random((n_sims, P)) < hazards)
                if hurt.any():
                    d = values[rng.choice(len(values), size=int(hurt.sum()), p=pmf)]
                    remaining[hurt] = d
                    if state is not None:
                        season_out[hurt] = d == SEASON_ENDING
                        missed[hurt] = 0
                        absent |= hurt
            if state is not None:
                known = np.where(season_out, SEASON_OUT, np.minimum(missed, 120))
                state[:, w, :] = np.where(absent, known, -1)
        return avail, state

    def to_dict(self) -> dict:
        return {
            "hazard": self.hazard,
            "durations": self.durations,
            "status_play": self.status_play,
            "default_hazard": self.default_hazard,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> AvailabilityModel:
        m = cls()
        m.hazard = {p: float(v) for p, v in d.get("hazard", {}).items()}
        if d.get("durations"):
            m.durations = [float(v) for v in d["durations"]]
        m.status_play = {**DEFAULT_STATUS_PLAY, **d.get("status_play", {})}
        m.default_hazard = float(d.get("default_hazard", m.default_hazard))
        return m


def fit_availability(
    seasons: Mapping[tuple[int, str], PlayerSeason],
    team_weeks: Mapping[tuple[int, str], list[int]],
) -> tuple[AvailabilityModel, dict[str, dict[str, float]]]:
    games: Counter[str] = Counter()
    misses: Counter[str] = Counter()
    lengths: Counter[int] = Counter()
    for ps in seasons.values():
        if ps.position == "DEF" or not ps.team:
            continue
        # Week 18 is excluded: playoff-bound teams rest starters, and fantasy seasons
        # end by week 17.
        sched = [w for w in team_weeks.get((ps.season, ps.team), []) if w < 18]
        wk = ps.weeks
        for i, w in enumerate(sched[:-1]):
            row = wk.get(w)
            if not row or not row[2] or row[0] < REGULAR.get(ps.position, 5.0):
                continue
            games[ps.position] += 1
            nxt = wk.get(sched[i + 1])
            if nxt and nxt[2]:
                continue
            misses[ps.position] += 1
            rest = sched[i + 1 :]
            d = next((j for j, g in enumerate(rest) if g in wk and wk[g][2]), None)
            if d is None:
                # Never returned. Only call it season-ending if enough games were left.
                if len(rest) > MAX_DURATION:
                    lengths[SEASON_ENDING] += 1
            elif d > MAX_DURATION:
                lengths[SEASON_ENDING] += 1
            else:
                lengths[d] += 1

    model = AvailabilityModel()
    diags: dict[str, dict[str, float]] = {}
    for pos, n in games.items():
        if n >= 200:
            model.hazard[pos] = round(misses[pos] / n, 4)
            diags[pos] = {"games": float(n), "hazard": model.hazard[pos]}
    total_games = sum(games.values())
    if total_games:
        model.default_hazard = round(sum(misses.values()) / total_games, 4)
    total = sum(lengths.values())
    if total >= 50:
        counts = [lengths[d] for d in range(1, MAX_DURATION + 1)] + [lengths[SEASON_ENDING]]
        model.durations = [round(c / total, 4) for c in counts]
    return model, diags
