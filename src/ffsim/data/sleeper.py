"""Sleeper API client with an on-disk JSON cache.

Documented read-only endpoints live under ``https://api.sleeper.app/v1``. The
projection/stat endpoints used by the Sleeper app live under
``https://api.sleeper.com`` and are undocumented; see ``projections.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

V1_BASE = "https://api.sleeper.app/v1"
APP_BASE = "https://api.sleeper.com"

MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR


def default_cache_dir() -> Path:
    root = os.environ.get("FFSIM_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME")
    return Path(root) / "ffsim" if root else Path.home() / ".cache" / "ffsim"


class SleeperError(RuntimeError):
    pass


class SleeperClient:
    """Thin GET wrapper around the Sleeper API.

    Responses are cached as JSON files keyed by URL + params. Each call passes a
    TTL; ``refresh=True`` bypasses reads (but still writes) the cache, and
    ``ttl=None`` means the response never expires (used for completed weeks).
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        *,
        refresh: bool = False,
        use_cache: bool = True,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.cache_dir = cache_dir or default_cache_dir()
        self.refresh = refresh
        self.use_cache = use_cache
        self._http = httpx.Client(
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": "ffsim/0.1"},
            follow_redirects=True,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> SleeperClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- core -----------------------------------------------------------------

    def _cache_path(self, url: str, params: Mapping[str, Any] | None) -> Path:
        key = url + "?" + json.dumps(params or {}, sort_keys=True, default=str)
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        return self.cache_dir / "http" / f"{digest}.json"

    def get(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        ttl: float | None = HOUR,
    ) -> Any:
        path = self._cache_path(url, params)
        if self.use_cache and not self.refresh and path.exists():
            fresh = ttl is None or (time.time() - path.stat().st_mtime) < ttl
            if fresh:
                return json.loads(path.read_text())

        try:
            resp = self._http.get(url, params=params)
        except httpx.HTTPError as exc:
            raise SleeperError(f"GET {url} failed: {exc}") from exc
        if resp.status_code != 200:
            raise SleeperError(f"GET {url} returned HTTP {resp.status_code}")
        data = resp.json()

        if self.use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(path)
        return data

    def _v1(self, path: str, *, ttl: float | None = HOUR) -> Any:
        return self.get(f"{V1_BASE}/{path.lstrip('/')}", ttl=ttl)

    # -- documented v1 endpoints ------------------------------------------------

    def nfl_state(self) -> dict[str, Any]:
        return self._v1("state/nfl", ttl=HOUR)

    def league(self, league_id: str) -> dict[str, Any]:
        data = self._v1(f"league/{league_id}", ttl=HOUR)
        if not data:
            raise SleeperError(f"league {league_id} not found")
        return data

    def rosters(self, league_id: str) -> list[dict[str, Any]]:
        return self._v1(f"league/{league_id}/rosters", ttl=15 * MINUTE)

    def users(self, league_id: str) -> list[dict[str, Any]]:
        return self._v1(f"league/{league_id}/users", ttl=DAY)

    def matchups(self, league_id: str, week: int, *, final: bool = False) -> list[dict[str, Any]]:
        """Matchups for a week. ``final=True`` caches forever (week fully scored)."""
        return self._v1(f"league/{league_id}/matchups/{week}", ttl=None if final else 15 * MINUTE)

    def transactions(self, league_id: str, week: int) -> list[dict[str, Any]]:
        return self._v1(f"league/{league_id}/transactions/{week}", ttl=15 * MINUTE)

    def winners_bracket(self, league_id: str) -> list[dict[str, Any]]:
        return self._v1(f"league/{league_id}/winners_bracket", ttl=15 * MINUTE)

    def losers_bracket(self, league_id: str) -> list[dict[str, Any]]:
        return self._v1(f"league/{league_id}/losers_bracket", ttl=15 * MINUTE)

    def user(self, username_or_id: str) -> dict[str, Any]:
        return self._v1(f"user/{username_or_id}", ttl=DAY)

    def user_leagues(self, user_id: str, season: int | str) -> list[dict[str, Any]]:
        return self._v1(f"user/{user_id}/leagues/nfl/{season}", ttl=HOUR)

    def players(self) -> dict[str, dict[str, Any]]:
        """All NFL players (~5MB). Sleeper asks callers to fetch this at most daily."""
        return self._v1("players/nfl", ttl=DAY)
