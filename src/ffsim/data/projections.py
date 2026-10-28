"""Projection and stat-line sources.

The engine consumes raw *stat lines* (``{"pass_yd": 251.3, "pass_td": 1.7, ...}``)
and scores them with the league's own settings, so any source only needs to
return stats keyed by Sleeper player_id.

``SleeperProjectionSource`` uses the undocumented ``api.sleeper.com`` endpoints
that back the Sleeper app. They are unversioned and may change without notice.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from ffsim.data.sleeper import APP_BASE, DAY, HOUR, SleeperClient

StatLine = dict[str, float]

DEFAULT_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DEF", "DL", "LB", "DB")


@dataclass(frozen=True)
class PlayerWeek:
    player_id: str
    season: int
    week: int
    stats: StatLine
    team: str | None = None
    opponent: str | None = None


class ProjectionSource(Protocol):
    def weekly_projections(
        self, season: int, week: int, positions: Iterable[str] = DEFAULT_POSITIONS
    ) -> dict[str, PlayerWeek]: ...

    def weekly_stats(
        self, season: int, week: int, positions: Iterable[str] = DEFAULT_POSITIONS
    ) -> dict[str, PlayerWeek]: ...


def _numeric_stats(raw: dict[str, Any] | None) -> StatLine:
    out: StatLine = {}
    for k, v in (raw or {}).items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int | float):
            out[k] = float(v)
    return out


def parse_rows(rows: Any, season: int, week: int) -> dict[str, PlayerWeek]:
    """Parse a projections/stats payload.

    The app endpoint returns a list of ``{"player_id", "stats", "team", "opponent", ...}``;
    some older variants return ``{player_id: stats}``. Both are accepted.
    """
    out: dict[str, PlayerWeek] = {}
    if isinstance(rows, dict):
        for pid, stats in rows.items():
            out[str(pid)] = PlayerWeek(str(pid), season, week, _numeric_stats(stats))
        return out
    for row in rows or []:
        pid = row.get("player_id")
        if pid is None:
            continue
        out[str(pid)] = PlayerWeek(
            player_id=str(pid),
            season=season,
            week=week,
            stats=_numeric_stats(row.get("stats")),
            team=row.get("team"),
            opponent=row.get("opponent"),
        )
    return out


class SleeperProjectionSource:
    def __init__(self, client: SleeperClient, season_type: str = "regular") -> None:
        self.client = client
        self.season_type = season_type

    def _fetch(
        self, kind: str, season: int, week: int, positions: Iterable[str], ttl: float | None
    ) -> dict[str, PlayerWeek]:
        params = {"season_type": self.season_type, "position[]": list(positions)}
        rows = self.client.get(f"{APP_BASE}/{kind}/nfl/{season}/{week}", params=params, ttl=ttl)
        return parse_rows(rows, season, week)

    def weekly_projections(
        self, season: int, week: int, positions: Iterable[str] = DEFAULT_POSITIONS
    ) -> dict[str, PlayerWeek]:
        return self._fetch("projections", season, week, positions, ttl=3 * HOUR)

    def weekly_stats(
        self,
        season: int,
        week: int,
        positions: Iterable[str] = DEFAULT_POSITIONS,
        *,
        final: bool = False,
    ) -> dict[str, PlayerWeek]:
        return self._fetch("stats", season, week, positions, ttl=None if final else HOUR)

    def season_projections(
        self, season: int, positions: Iterable[str] = DEFAULT_POSITIONS
    ) -> dict[str, PlayerWeek]:
        """Full-season (preseason prior) projections; week is recorded as 0."""
        params = {"season_type": self.season_type, "position[]": list(positions)}
        rows = self.client.get(f"{APP_BASE}/projections/nfl/{season}", params=params, ttl=DAY)
        return parse_rows(rows, season, 0)
