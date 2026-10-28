from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ffsim.data.sleeper import SleeperClient

FIXTURES = Path(__file__).parent / "fixtures"
LEAGUE_ID = "900000000000000001"


class FixtureTransport(httpx.MockTransport):
    """Serves a fixture file keyed by URL path (host and leading slash stripped)."""

    def __init__(self, routes: dict[str, object]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []
        super().__init__(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = request.url.path.lstrip("/")
        if key not in self.routes:
            return httpx.Response(404, json=None)
        return httpx.Response(200, json=self.routes[key])


@pytest.fixture
def routes() -> dict[str, object]:
    return json.loads((FIXTURES / "sleeper_superflex_idp.json").read_text())


@pytest.fixture
def transport(routes) -> FixtureTransport:
    return FixtureTransport(routes)


@pytest.fixture
def client(tmp_path, transport) -> SleeperClient:
    with SleeperClient(tmp_path / "cache", transport=transport) as c:
        yield c
