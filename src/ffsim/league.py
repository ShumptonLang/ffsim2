"""League domain model built from Sleeper API payloads.

Covers roster slots (incl. SUPER_FLEX and IDP slots), playoff structure,
teams/records, and the weekly H2H schedule. Everything here is parsed from raw
JSON so it can be built from live API calls or recorded fixtures alike.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from ffsim.data.sleeper import SleeperClient

# --- roster slots ----------------------------------------------------------------

SLOT_ELIGIBILITY: dict[str, frozenset[str]] = {
    "QB": frozenset({"QB"}),
    "RB": frozenset({"RB"}),
    "WR": frozenset({"WR"}),
    "TE": frozenset({"TE"}),
    "K": frozenset({"K"}),
    "DEF": frozenset({"DEF"}),
    "FLEX": frozenset({"RB", "WR", "TE"}),
    "WRRB_FLEX": frozenset({"WR", "RB"}),
    "REC_FLEX": frozenset({"WR", "TE"}),
    "SUPER_FLEX": frozenset({"QB", "RB", "WR", "TE"}),
    "DL": frozenset({"DL"}),
    "LB": frozenset({"LB"}),
    "DB": frozenset({"DB"}),
    "IDP_FLEX": frozenset({"DL", "LB", "DB"}),
}
NON_STARTING_SLOTS = frozenset({"BN", "IR", "TAXI"})
IDP_GROUPS = frozenset({"DL", "LB", "DB"})

# Raw NFL positions -> Sleeper fantasy position groups (fallback when a player
# record has no ``fantasy_positions``).
_POSITION_GROUP: dict[str, str] = {
    "DE": "DL", "DT": "DL", "NT": "DL", "EDGE": "DL",
    "ILB": "LB", "OLB": "LB", "MLB": "LB",
    "CB": "DB", "S": "DB", "SS": "DB", "FS": "DB",
}  # fmt: skip


def slot_accepts(slot: str, fantasy_positions: Iterable[str]) -> bool:
    eligible = SLOT_ELIGIBILITY.get(slot, frozenset())
    return any(p in eligible for p in fantasy_positions)


# --- players ---------------------------------------------------------------------


@dataclass(frozen=True)
class Player:
    player_id: str
    name: str
    position: str | None
    fantasy_positions: tuple[str, ...]
    team: str | None
    status: str | None = None
    injury_status: str | None = None
    depth_chart_position: str | None = None
    depth_chart_order: int | None = None

    @property
    def primary_position(self) -> str | None:
        return self.fantasy_positions[0] if self.fantasy_positions else self.position

    @classmethod
    def from_sleeper(cls, player_id: str, raw: Mapping[str, Any]) -> Player:
        pos = raw.get("position")
        fps = raw.get("fantasy_positions") or ([pos] if pos else [])
        groups = tuple(dict.fromkeys(_POSITION_GROUP.get(p, p) for p in fps))
        # Team defenses (player_id "KC") have no full_name: "Kansas City" + "Chiefs".
        name = (
            raw.get("full_name")
            or f"{raw.get('first_name') or ''} {raw.get('last_name') or ''}".strip()
            or str(player_id)
        )
        return cls(
            player_id=str(player_id),
            name=name,
            position=pos,
            fantasy_positions=groups,
            team=raw.get("team"),
            status=raw.get("status"),
            injury_status=raw.get("injury_status"),
            depth_chart_position=raw.get("depth_chart_position"),
            depth_chart_order=raw.get("depth_chart_order"),
        )

    @classmethod
    def unknown(cls, player_id: str) -> Player:
        return cls(player_id=player_id, name=f"Unknown ({player_id})", position=None,
                   fantasy_positions=(), team=None)  # fmt: skip


# --- league settings ---------------------------------------------------------------


class PlayoffRoundType(IntEnum):
    ONE_WEEK = 0  # every round is one week
    TWO_WEEK_CHAMPIONSHIP = 1  # one-week rounds, two-week final
    TWO_WEEKS_PER_ROUND = 2


class PlayoffSeedType(IntEnum):
    FIXED = 0  # bracket fixed by seed
    RESEED = 1  # top remaining seed plays lowest remaining seed each round


@dataclass(frozen=True)
class PlayoffFormat:
    teams: int
    start_week: int
    round_type: PlayoffRoundType
    seed_type: PlayoffSeedType

    @property
    def rounds(self) -> int:
        return max(1, math.ceil(math.log2(self.teams))) if self.teams > 1 else 0

    @property
    def byes(self) -> int:
        """Top seeds that skip round 1 (e.g. 6-team bracket -> 2 byes)."""
        return 2**self.rounds - self.teams if self.teams > 1 else 0

    def round_weeks(self) -> list[tuple[int, ...]]:
        """NFL weeks covered by each playoff round, in order."""
        out: list[tuple[int, ...]] = []
        week = self.start_week
        for r in range(self.rounds):
            final = r == self.rounds - 1
            two = self.round_type == PlayoffRoundType.TWO_WEEKS_PER_ROUND or (
                final and self.round_type == PlayoffRoundType.TWO_WEEK_CHAMPIONSHIP
            )
            span = 2 if two else 1
            out.append(tuple(range(week, week + span)))
            week += span
        return out

    @property
    def last_week(self) -> int:
        rw = self.round_weeks()
        return rw[-1][-1] if rw else self.start_week - 1


@dataclass(frozen=True)
class League:
    league_id: str
    name: str
    season: int
    status: str
    total_rosters: int
    roster_positions: tuple[str, ...]
    scoring_settings: Mapping[str, float]
    playoffs: PlayoffFormat
    start_week: int = 1
    last_scored_week: int = 0
    waiver_budget: int = 0
    waiver_type: int = 0  # 0 rolling, 1 reverse standings, 2 FAAB
    waiver_bid_min: int = 0
    divisions: int = 0
    previous_league_id: str | None = None
    raw_settings: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def starter_slots(self) -> tuple[str, ...]:
        return tuple(s for s in self.roster_positions if s not in NON_STARTING_SLOTS)

    @property
    def bench_slots(self) -> int:
        return sum(1 for s in self.roster_positions if s == "BN")

    @property
    def unsupported_slots(self) -> tuple[str, ...]:
        return tuple(sorted({s for s in self.starter_slots if s not in SLOT_ELIGIBILITY}))

    @property
    def is_superflex(self) -> bool:
        return "SUPER_FLEX" in self.roster_positions or self.roster_positions.count("QB") > 1

    @property
    def has_idp(self) -> bool:
        return any(s in IDP_GROUPS or s == "IDP_FLEX" for s in self.roster_positions)

    @property
    def regular_season_weeks(self) -> range:
        return range(self.start_week, self.playoffs.start_week)

    @classmethod
    def from_sleeper(cls, raw: Mapping[str, Any]) -> League:
        s = raw.get("settings") or {}
        playoff_start = int(s.get("playoff_week_start") or 15)
        return cls(
            league_id=str(raw["league_id"]),
            name=raw.get("name") or str(raw["league_id"]),
            season=int(raw.get("season") or 0),
            status=raw.get("status") or "unknown",
            total_rosters=int(raw.get("total_rosters") or s.get("num_teams") or 0),
            roster_positions=tuple(raw.get("roster_positions") or ()),
            scoring_settings={k: float(v) for k, v in (raw.get("scoring_settings") or {}).items()},
            playoffs=PlayoffFormat(
                teams=int(s.get("playoff_teams") or 0),
                start_week=playoff_start,
                round_type=PlayoffRoundType(int(s.get("playoff_round_type") or 0)),
                seed_type=PlayoffSeedType(int(s.get("playoff_seed_type") or 0)),
            ),
            start_week=int(s.get("start_week") or 1),
            last_scored_week=int(s.get("last_scored_leg") or 0),
            waiver_budget=int(s.get("waiver_budget") or 0),
            waiver_type=int(s.get("waiver_type") or 0),
            waiver_bid_min=int(s.get("waiver_bid_min") or 0),
            divisions=int(s.get("divisions") or 0),
            previous_league_id=raw.get("previous_league_id") or None,
            raw_settings=s,
        )


# --- teams & schedule ---------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: float = 0.0
    points_against: float = 0.0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_pct(self) -> float:
        return (self.wins + 0.5 * self.ties) / self.games if self.games else 0.0


@dataclass(frozen=True)
class Team:
    roster_id: int
    owner_id: str | None
    display_name: str
    team_name: str | None
    players: tuple[str, ...]
    starters: tuple[str, ...]
    reserve: tuple[str, ...]
    taxi: tuple[str, ...]
    record: Record
    division: int | None = None
    waiver_budget_used: int = 0
    co_owner_ids: tuple[str, ...] = ()
    waiver_position: int = 0  # 1 = first in line; 0 if unknown

    @property
    def label(self) -> str:
        return self.team_name or self.display_name

    @property
    def active_players(self) -> tuple[str, ...]:
        """Players eligible for lineups (excludes IR and taxi)."""
        out = set(self.reserve) | set(self.taxi)
        return tuple(p for p in self.players if p not in out)


def _points(settings: Mapping[str, Any], key: str) -> float:
    return float(settings.get(key) or 0) + float(settings.get(f"{key}_decimal") or 0) / 100.0


def build_teams(
    rosters: Iterable[Mapping[str, Any]], users: Iterable[Mapping[str, Any]]
) -> dict[int, Team]:
    users_by_id = {u["user_id"]: u for u in users}
    teams: dict[int, Team] = {}
    for r in rosters:
        s = r.get("settings") or {}
        owner = r.get("owner_id")
        user = users_by_id.get(owner, {}) if owner else {}
        meta = user.get("metadata") or {}
        rid = int(r["roster_id"])
        teams[rid] = Team(
            roster_id=rid,
            owner_id=owner,
            display_name=user.get("display_name") or f"Team {rid}",
            team_name=meta.get("team_name") or None,
            players=tuple(str(p) for p in r.get("players") or ()),
            starters=tuple(str(p) for p in r.get("starters") or ()),
            reserve=tuple(str(p) for p in r.get("reserve") or ()),
            taxi=tuple(str(p) for p in r.get("taxi") or ()),
            record=Record(
                wins=int(s.get("wins") or 0),
                losses=int(s.get("losses") or 0),
                ties=int(s.get("ties") or 0),
                points_for=_points(s, "fpts"),
                points_against=_points(s, "fpts_against"),
            ),
            division=s.get("division"),
            waiver_budget_used=int(s.get("waiver_budget_used") or 0),
            co_owner_ids=tuple(r.get("co_owners") or ()),
            waiver_position=int(s.get("waiver_position") or 0),
        )
    return teams


@dataclass(frozen=True)
class Matchup:
    week: int
    home: int  # roster_id (Sleeper has no home/away; this is just the lower roster_id)
    away: int
    home_points: float | None = None
    away_points: float | None = None

    @property
    def played(self) -> bool:
        return self.home_points is not None and self.away_points is not None

    def opponent(self, roster_id: int) -> int | None:
        if roster_id == self.home:
            return self.away
        if roster_id == self.away:
            return self.home
        return None


def parse_matchups(week: int, rows: Iterable[Mapping[str, Any]], scored: bool) -> list[Matchup]:
    """Pair roster entries by ``matchup_id``. Points are kept only for scored weeks."""
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        mid = row.get("matchup_id")
        if mid is not None:
            groups[int(mid)].append(row)
    out: list[Matchup] = []
    for mid in sorted(groups):
        pair = sorted(groups[mid], key=lambda r: int(r["roster_id"]))
        if len(pair) != 2:
            continue
        a, b = pair
        out.append(
            Matchup(
                week=week,
                home=int(a["roster_id"]),
                away=int(b["roster_id"]),
                home_points=float(a.get("points") or 0.0) if scored else None,
                away_points=float(b.get("points") or 0.0) if scored else None,
            )
        )
    return out


# --- league state ----------------------------------------------------------------------


class TeamNotFound(LookupError):
    pass


@dataclass
class LeagueState:
    league: League
    teams: dict[int, Team]
    schedule: dict[int, list[Matchup]]
    players: dict[str, Player]
    nfl_week: int
    nfl_season_type: str = "regular"
    # week -> roster_id -> points, for scored playoff weeks (regular weeks are in schedule)
    playoff_points: dict[int, dict[int, float]] = field(default_factory=dict)

    def player(self, player_id: str) -> Player:
        return self.players.get(player_id) or Player.unknown(player_id)

    @property
    def remaining_regular_weeks(self) -> list[int]:
        return [w for w in self.league.regular_season_weeks if w > self.league.last_scored_week]

    def standings(self) -> list[Team]:
        """Sleeper-style order: wins (ties count half), then points for."""
        return sorted(
            self.teams.values(),
            key=lambda t: (t.record.wins + 0.5 * t.record.ties, t.record.points_for),
            reverse=True,
        )

    def matchup_for(self, roster_id: int, week: int) -> Matchup | None:
        return next((m for m in self.schedule.get(week, []) if m.opponent(roster_id)), None)

    def find_team(self, query: str | int) -> Team:
        """Resolve a manager by roster_id, user_id, display name or team name."""
        q = str(query).strip()
        if q.isdigit() and int(q) in self.teams:
            return self.teams[int(q)]
        ql = q.lower()
        for t in self.teams.values():
            names = {t.owner_id or "", t.display_name, t.team_name or ""}
            if ql in {n.lower() for n in names if n}:
                return t
        matches = [
            t
            for t in self.teams.values()
            if ql in t.display_name.lower() or (t.team_name and ql in t.team_name.lower())
        ]
        if len(matches) == 1:
            return matches[0]
        hint = ", ".join(sorted(t.display_name for t in self.teams.values()))
        if matches:
            raise TeamNotFound(f"'{query}' is ambiguous; matches several teams. Teams: {hint}")
        raise TeamNotFound(f"no team matches '{query}'. Teams: {hint}")

    def rostered_ids(self) -> set[str]:
        return {p for t in self.teams.values() for p in t.players}


def load_league_state(client: SleeperClient, league_id: str) -> LeagueState:
    league = League.from_sleeper(client.league(league_id))
    teams = build_teams(client.rosters(league_id), client.users(league_id))
    state = client.nfl_state()
    raw_players = client.players()

    schedule: dict[int, list[Matchup]] = {}
    for week in league.regular_season_weeks:
        scored = week <= league.last_scored_week
        schedule[week] = parse_matchups(
            week, client.matchups(league_id, week, final=scored), scored
        )

    playoff_points: dict[int, dict[int, float]] = {}
    for week in range(league.playoffs.start_week, league.last_scored_week + 1):
        rows = client.matchups(league_id, week, final=True)
        playoff_points[week] = {int(r["roster_id"]): float(r.get("points") or 0.0) for r in rows}

    wanted = {p for t in teams.values() for p in t.players}
    players = {pid: Player.from_sleeper(pid, raw) for pid, raw in raw_players.items()}
    for pid in wanted - players.keys():
        players[pid] = Player.unknown(pid)

    return LeagueState(
        league=league,
        teams=teams,
        schedule=schedule,
        players=players,
        nfl_week=int(state.get("week") or state.get("display_week") or 0),
        nfl_season_type=state.get("season_type") or "regular",
        playoff_points=playoff_points,
    )
