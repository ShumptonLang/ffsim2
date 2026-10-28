"""Historical (projection, actual) pairs for fitting the projection-error model.

Each season-week is fetched once from the Sleeper app endpoints and stored in a
compact JSON file (only stats that can carry fantasy points, plus position), so
the multi-megabyte raw payloads don't sit in the HTTP cache.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ffsim.data.projections import DEFAULT_POSITIONS, StatLine, _numeric_stats
from ffsim.data.sleeper import APP_BASE, SleeperClient
from ffsim.league import Player
from ffsim.scoring import score

# Stat keys that never score in any league; dropped to keep the history small.
_JUNK_PREFIXES = ("adp_", "pos_adp_", "pos_rank_", "rank_", "fan_pts_allow")
_JUNK_KEYS = frozenset({"pts_ppr", "pts_half_ppr", "pts_std"})
# Zero is meaningful for these (a shutout is pts_allow == 0).
_KEEP_ZERO = frozenset({"pts_allow", "yds_allow"})


@dataclass(frozen=True)
class HistoryRow:
    season: int
    week: int
    player_id: str
    position: str  # fantasy group: QB RB WR TE K DEF DL LB DB
    team: str | None
    opponent: str | None
    proj: StatLine
    actual: StatLine
    played: bool

    def scored(self, settings: Mapping[str, float]) -> tuple[float, float]:
        return (
            score(self.proj, settings, self.position),
            score(self.actual, settings, self.position),
        )


def _compact(stats: Mapping[str, float]) -> StatLine:
    return {
        k: v
        for k, v in stats.items()
        if (v or k in _KEEP_ZERO) and k not in _JUNK_KEYS and not k.startswith(_JUNK_PREFIXES)
    }


def _position(row: Mapping[str, Any]) -> str | None:
    raw = row.get("player") or {}
    p = Player.from_sleeper(str(row.get("player_id")), raw)
    return p.primary_position


class HistoryStore:
    def __init__(self, client: SleeperClient, root: Path | None = None) -> None:
        self.client = client
        self.root = root or client.cache_dir / "history"

    def _path(self, season: int, week: int) -> Path:
        return self.root / f"{season}_{week:02d}.json"

    def _fetch(self, kind: str, season: int, week: int) -> list[dict[str, Any]]:
        params = {"season_type": "regular", "position[]": list(DEFAULT_POSITIONS)}
        url = f"{APP_BASE}/{kind}/nfl/{season}/{week}"
        # Raw payloads are large; bypass the HTTP cache and keep only the compact form.
        prev = self.client.use_cache
        self.client.use_cache = False
        try:
            return self.client.get(url, params=params) or []
        finally:
            self.client.use_cache = prev

    def week(self, season: int, week: int) -> list[HistoryRow]:
        path = self._path(season, week)
        if not path.exists():
            proj = {str(r["player_id"]): r for r in self._fetch("projections", season, week)}
            act = {str(r["player_id"]): r for r in self._fetch("stats", season, week)}
            out: dict[str, Any] = {}
            for pid, prow in proj.items():
                pos = _position(prow)
                pstats = _compact(_numeric_stats(prow.get("stats")))
                if not pos or not pstats:
                    continue
                astats = _numeric_stats((act.get(pid) or {}).get("stats"))
                out[pid] = {
                    "pos": pos,
                    "team": prow.get("team"),
                    "opp": prow.get("opponent"),
                    "proj": pstats,
                    "act": _compact(astats),
                    "played": astats.get("gp", 0) > 0,
                }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out))
        data = json.loads(path.read_text())
        return [
            HistoryRow(
                season=season,
                week=week,
                player_id=pid,
                position=r["pos"],
                team=r.get("team"),
                opponent=r.get("opp"),
                proj=r["proj"],
                actual=r["act"],
                played=r["played"],
            )
            for pid, r in data.items()
        ]

    def rows(
        self, seasons: Iterable[int], weeks: Iterable[int] = range(1, 19)
    ) -> Iterator[HistoryRow]:
        weeks = list(weeks)
        for season in seasons:
            for week in weeks:
                yield from self.week(season, week)

    def preseason(self, season: int) -> dict[str, tuple[str, StatLine]]:
        """Full-season preseason projections: player_id -> (position, stats)."""
        path = self.root / f"{season}_preseason.json"
        if not path.exists():
            params = {"season_type": "regular", "position[]": list(DEFAULT_POSITIONS)}
            prev = self.client.use_cache
            self.client.use_cache = False
            try:
                raw = self.client.get(f"{APP_BASE}/projections/nfl/{season}", params=params) or []
            finally:
                self.client.use_cache = prev
            out = {}
            for row in raw:
                pos = _position(row)
                stats = _compact(_numeric_stats(row.get("stats")))
                if pos and stats:
                    out[str(row["player_id"])] = [pos, stats]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out))
        return {pid: (v[0], v[1]) for pid, v in json.loads(path.read_text()).items()}


@dataclass
class PlayerSeason:
    season: int
    player_id: str
    position: str
    team: str | None
    # week -> (projected points, actual points, played)
    weeks: dict[int, tuple[float, float, bool]]


def player_seasons(
    rows: Iterable[HistoryRow], settings: Mapping[str, float]
) -> tuple[dict[tuple[int, str], PlayerSeason], dict[tuple[int, str], list[int]]]:
    """Group history into player-seasons (scored with ``settings``).

    Also returns each (season, NFL team)'s game weeks, inferred from which weeks
    the team had any projected players (bye weeks are absent).
    """
    seasons: dict[tuple[int, str], PlayerSeason] = {}
    team_weeks: dict[tuple[int, str], set[int]] = {}
    for r in rows:
        key = (r.season, r.player_id)
        ps = seasons.get(key)
        if ps is None:
            ps = seasons[key] = PlayerSeason(r.season, r.player_id, r.position, r.team, {})
        m, a = r.scored(settings)
        ps.weeks[r.week] = (m, a, r.played)
        if r.team:
            team_weeks.setdefault((r.season, r.team), set()).add(r.week)
    return seasons, {k: sorted(v) for k, v in team_weeks.items()}
