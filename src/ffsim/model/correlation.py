"""Within-week correlation: a factor-model Gaussian copula.

Every NFL game and team gets latent factors each simulated week:

* ``G`` (per game): game environment / shootout, shared by both teams;
* ``O`` (per team): offensive success;
* ``P`` (per team): pass-heavy vs run-heavy split;
* ``S1``, ``S2`` (per team): usage-share contrasts among teammates (targets
  going to WR1 rather than WR2, carries to RB1 rather than RB2).

A player's latent weekly residual is

    z = g*G[game] + o*O[own] + p*P[own] + x*O[opponent] + s1*S1[own] + s2*S2[own]
        + sqrt(1 - sum of squared loadings)*e

with loadings set by his *role* (QB, RB1, WR2, DEF, LB1, ...; roles come from
projection rank on his team that week). That yields, e.g., QB-WR1 stacking,
shootout correlation across the two sides, team defenses moving against the
opposing offense (negative x), and teammates competing for touches (share
factors, which can make RB1-RB2 negative while QB-WR stays positive).

Because z feeds the projection-error model's quantile functions, marginal
weekly distributions are unchanged; only co-movement changes. A factor model is
O(players) to sample and always positive-definite. Loadings are fitted by least
squares to role-pair correlations of latent residuals in 2023-2025 data, with
extra weight on pairs among the roles that drive lineup decisions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from ffsim.data.history import HistoryRow
from ffsim.model.errors import ErrorModel

# Roles per position group; extra players take the last role (WR4 -> WR3).
ROLE_SLOTS: dict[str, tuple[str, ...]] = {
    "QB": ("QB",),
    "RB": ("RB1", "RB2"),
    "WR": ("WR1", "WR2", "WR3"),
    "TE": ("TE1", "TE2"),
    "K": ("K",),
    "DEF": ("DEF",),
    "DL": ("DL1", "DL2"),
    "LB": ("LB1", "LB2"),
    "DB": ("DB1", "DB2"),
}
ROLES: tuple[str, ...] = tuple(r for rs in ROLE_SLOTS.values() for r in rs)
FACTORS = ("game", "own_offense", "own_pass", "opp_offense", "share_1", "share_2")
N_FACTORS = len(FACTORS)
MIN_PAIRS = 150
MAX_LOADING_SQ = 0.9
# Loadings are only identified up to rotations that leave implied correlations
# unchanged; a light ridge picks a small such set.
RIDGE = 2.0
# Pairs among these roles decide stacking and start/sit, so they are fitted 3x harder.
KEY_ROLES = frozenset({"QB", "RB1", "WR1", "WR2", "TE1", "DEF"})
KEY_WEIGHT = 3.0
_ZERO = (0.0,) * N_FACTORS


def assign_roles(players: Iterable[tuple[str, str | None, str | None, float]]) -> dict[str, str]:
    """Roles from (player_id, team, position group, projection) for one week."""
    groups: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
    for pid, team, pos, proj in players:
        if team and pos in ROLE_SLOTS and proj > 0:
            groups[(team, pos)].append((proj, pid))
    roles: dict[str, str] = {}
    for (_, pos), members in groups.items():
        slots = ROLE_SLOTS[pos]
        for rank, (_, pid) in enumerate(sorted(members, key=lambda m: (-m[0], m[1]))):
            roles[pid] = slots[min(rank, len(slots) - 1)]
    return roles


@dataclass
class CorrelationModel:
    loadings: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def loading(self, role: str | None) -> tuple[float, ...]:
        if role is None:
            return _ZERO
        return self.loadings.get(role, _ZERO)

    def implied(self, a: str, b: str, same_team: bool) -> float:
        la, lb = self.loading(a), self.loading(b)
        if same_team:
            return float(np.dot(la, lb))
        return la[0] * lb[0] + la[1] * lb[3] + la[3] * lb[1]

    def to_dict(self) -> dict:
        return {"loadings": {r: list(v) for r, v in self.loadings.items()}}

    @classmethod
    def from_dict(cls, d: Mapping) -> CorrelationModel:
        loads = {}
        for r, v in d.get("loadings", {}).items():
            vals = [float(x) for x in v]
            loads[r] = tuple(vals + [0.0] * (N_FACTORS - len(vals)))
        return cls(loads)


# --- fitting -------------------------------------------------------------------------


@dataclass
class PairStats:
    """Role-pair correlation estimates from history."""

    same: dict[tuple[str, str], tuple[float, int]]  # sorted role pair -> (corr, n)
    opp: dict[tuple[str, str], tuple[float, int]]


def pair_correlations(
    rows: Iterable[HistoryRow],
    settings: Mapping[str, float],
    errors: ErrorModel,
) -> PairStats:
    rows = [r for r in rows if r.played and r.team and r.opponent]
    proj_act = [r.scored(settings) for r in rows]
    keep = [i for i, (m, _) in enumerate(proj_act) if m > 0]
    rows = [rows[i] for i in keep]
    m = np.array([proj_act[i][0] for i in keep])
    a = np.array([proj_act[i][1] for i in keep])
    z = errors.to_latent_many(a, m, [r.position for r in rows])

    by_week: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_week[(r.season, r.week)].append(i)
    role = np.empty(len(rows), dtype=object)
    for idx in by_week.values():
        roles = assign_roles((str(i), rows[i].team, rows[i].position, m[i]) for i in idx)
        for i in idx:
            role[i] = roles.get(str(i))

    # Standardize z per role so products estimate correlations.
    stats: dict[str, tuple[float, float]] = {}
    for rname in ROLES:
        sel = z[role == rname]
        if len(sel) > 30:
            stats[rname] = (float(sel.mean()), float(sel.std()) or 1.0)
    zs = np.array(
        [
            (z[i] - stats[role[i]][0]) / stats[role[i]][1] if role[i] in stats else np.nan
            for i in range(len(rows))
        ]
    )

    sums: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    games: dict[tuple[int, int, frozenset[str]], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if not np.isnan(zs[i]):
            games[(r.season, r.week, frozenset((r.team or "", r.opponent or "")))].append(i)
    for idx in games.values():
        for u in range(len(idx)):
            i = idx[u]
            for j in idx[u + 1 :]:
                kind = "same" if rows[i].team == rows[j].team else "opp"
                ri, rj = sorted((role[i], role[j]))
                acc = sums[(kind, ri, rj)]
                acc[0] += zs[i] * zs[j]
                acc[1] += 1
    same: dict[tuple[str, str], tuple[float, int]] = {}
    opp: dict[tuple[str, str], tuple[float, int]] = {}
    for (kind, ri, rj), (total, n) in sums.items():
        target = same if kind == "same" else opp
        target[(ri, rj)] = (total / n, int(n))
    return PairStats(same, opp)


def fit_loadings(
    pairs: PairStats,
    rho: Mapping[str, float] | None = None,
    min_pairs: int = MIN_PAIRS,
    seed: int = 0,
) -> tuple[CorrelationModel, list[dict[str, object]]]:
    """Least-squares loadings matching observed role-pair correlations.

    Observed correlations include the persistent talent component, which the
    simulator layers on top (z = sqrt(rho)*eta + sqrt(1-rho)*weekly). Targets are
    divided by sqrt((1-rho_i)(1-rho_j)) so the combination reproduces them.
    """

    def pos_of(role: str) -> str:
        return next(p for p, rs in ROLE_SLOTS.items() if role in rs)

    def scale(a: str, b: str) -> float:
        if not rho:
            return 1.0
        ra, rb = rho.get(pos_of(a), 0.0), rho.get(pos_of(b), 0.0)
        return float(1.0 / np.sqrt(max((1 - ra) * (1 - rb), 1e-6)))

    obs = [
        (a, b, True, c * scale(a, b), n) for (a, b), (c, n) in pairs.same.items() if n >= min_pairs
    ] + [
        (a, b, False, c * scale(a, b), n) for (a, b), (c, n) in pairs.opp.items() if n >= min_pairs
    ]
    if not obs:
        return CorrelationModel(), []
    roles = sorted({r for a, b, *_ in obs for r in (a, b)})
    ridx = {r: i for i, r in enumerate(roles)}
    ia = np.array([ridx[o[0]] for o in obs])
    ib = np.array([ridx[o[1]] for o in obs])
    same = np.array([o[2] for o in obs])
    target = np.array([o[3] for o in obs])
    weight = np.sqrt(np.minimum(np.array([o[4] for o in obs], dtype=float), 20_000.0))
    weight *= np.array(
        [KEY_WEIGHT if o[0] in KEY_ROLES and o[1] in KEY_ROLES else 1.0 for o in obs]
    )

    def implied(L: np.ndarray) -> np.ndarray:
        A, B = L[ia], L[ib]
        same_c = (A * B).sum(1)
        opp_c = A[:, 0] * B[:, 0] + A[:, 1] * B[:, 3] + A[:, 3] * B[:, 1]
        return np.where(same, same_c, opp_c)

    def resid(flat: np.ndarray) -> np.ndarray:
        L = flat.reshape(len(roles), N_FACTORS)
        over = np.maximum((L**2).sum(1) - MAX_LOADING_SQ, 0.0)
        return np.concatenate([(implied(L) - target) * weight, over * 1e3, RIDGE * L.ravel()])

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(4):  # multi-start: the problem has many local optima
        sol = least_squares(resid, rng.normal(0.0, 0.2, len(roles) * N_FACTORS), method="trf")
        if best is None or sol.cost < best.cost:
            best = sol
    assert best is not None
    L = best.x.reshape(len(roles), N_FACTORS)
    model = CorrelationModel({r: tuple(round(float(v), 4) for v in L[i]) for r, i in ridx.items()})
    fitted = implied(L)
    diags: list[dict[str, object]] = [
        {"a": o[0], "b": o[1], "same_team": o[2], "observed": o[3] / scale(o[0], o[1]),
         "target": o[3], "implied": float(fitted[k]), "n": o[4]}
        for k, o in enumerate(obs)
    ]  # fmt: skip
    return model, diags


def fit_correlation(
    rows: Iterable[HistoryRow],
    settings: Mapping[str, float],
    errors: ErrorModel,
    rho: Mapping[str, float] | None = None,
) -> tuple[CorrelationModel, list[dict[str, object]]]:
    return fit_loadings(pair_correlations(rows, settings, errors), rho)


# --- sampling ---------------------------------------------------------------------


@dataclass
class WeekLayout:
    """Factor indices and loadings for P players in one simulated week."""

    game: np.ndarray  # (P,) game index, -1 if none
    team: np.ndarray  # (P,) team index, -1 if none
    opp: np.ndarray  # (P,) opponent team index, -1 if none
    load: np.ndarray  # (P, N_FACTORS)
    n_games: int
    n_teams: int


def week_layout(
    model: CorrelationModel,
    players: Sequence[str],
    teams: Mapping[str, tuple[str, str]],  # player -> (team, opponent) for this week
    roles: Mapping[str, str],
) -> WeekLayout:
    team_ids: dict[str, int] = {}
    game_ids: dict[frozenset[str], int] = {}
    P = len(players)
    game = np.full(P, -1)
    team = np.full(P, -1)
    opp = np.full(P, -1)
    load = np.zeros((P, N_FACTORS))
    for i, pid in enumerate(players):
        to = teams.get(pid)
        if not to or not to[0] or not to[1]:
            continue
        t, o = to
        team[i] = team_ids.setdefault(t, len(team_ids))
        opp[i] = team_ids.setdefault(o, len(team_ids))
        game[i] = game_ids.setdefault(frozenset(to), len(game_ids))
        load[i] = model.loading(roles.get(pid))
    return WeekLayout(game, team, opp, load, len(game_ids), len(team_ids))


def sample_weekly(rng: np.random.Generator, n_sims: int, layout: WeekLayout) -> np.ndarray:
    """Correlated standard-normal weekly latents, shape (n_sims, P)."""
    P = len(layout.game)
    game = rng.standard_normal((n_sims, layout.n_games + 1), dtype=np.float32)
    off = rng.standard_normal((n_sims, layout.n_teams + 1), dtype=np.float32)
    pas = rng.standard_normal((n_sims, layout.n_teams + 1), dtype=np.float32)
    sh1 = rng.standard_normal((n_sims, layout.n_teams + 1), dtype=np.float32)
    sh2 = rng.standard_normal((n_sims, layout.n_teams + 1), dtype=np.float32)
    # index -1 hits the extra column, which is zeroed (players with no game)
    for f in (game, off, pas, sh1, sh2):
        f[:, -1] = 0.0
    g, o, p, x, s1, s2 = (layout.load[:, k].astype(np.float32) for k in range(N_FACTORS))
    idio = np.sqrt(np.clip(1.0 - (layout.load**2).sum(1), 0.0, 1.0)).astype(np.float32)
    eps = rng.standard_normal((n_sims, P), dtype=np.float32)
    return (
        g * game[:, layout.game]
        + o * off[:, layout.team]
        + p * pas[:, layout.team]
        + x * off[:, layout.opp]
        + s1 * sh1[:, layout.team]
        + s2 * sh2[:, layout.team]
        + idio * eps
    )
