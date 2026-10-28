"""Score a stat line with a Sleeper league's ``scoring_settings``.

Most Sleeper scoring keys share their name with a stat key (``pass_yd``, ``rec``,
``idp_sack`` ...), so the base rule is ``sum(stats[k] * settings[k])``. That
covers offense, kickers, team defense and IDP alike. A few kinds of setting are
not raw stats and are derived here when the stat line doesn't carry them:

* position-specific reception bonuses (``bonus_rec_te`` = TE premium, ...)
* threshold bonuses (``bonus_rush_yd_100``, ``bonus_pass_yd_300`` ...)
* team-defense points/yards-allowed tiers (``pts_allow_7_13``, ``yds_allow_550p`` ...)

Derived terms are evaluated on whatever values the stat line holds. For a
*projection* those are means, so thresholds and tiers are applied to the mean
(e.g. a 92-yard projection earns no 100-yard bonus). The simulator handles the
real nonlinearity through outcome distributions; this function is exact for
actual stat lines.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

POSITION_REC_BONUS: dict[str, str] = {
    "bonus_rec_te": "TE",
    "bonus_rec_rb": "RB",
    "bonus_rec_wr": "WR",
}

_THRESHOLD_RE = re.compile(
    r"^bonus_(pass_yd|rush_yd|rec_yd|rush_rec_yd|pass_cmp|rush_att|rec)_(\d+)$"
)
_TIER_RANGE_RE = re.compile(r"^(pts_allow|yds_allow)_(\d+)_(\d+)$")
_TIER_PLUS_RE = re.compile(r"^(pts_allow|yds_allow)_(\d+)p$")
_TIER_ZERO_RE = re.compile(r"^(pts_allow|yds_allow)_0$")

_KICK_BUCKETS = ("0_19", "20_29", "30_39", "40_49")


def _kicking_aggregates(stats: Mapping[str, float]) -> dict[str, float]:
    """Fill kicking keys that sources split differently.

    Sleeper projections use ``fgm_50p`` and per-distance ``fgmiss_*`` while many
    leagues score ``fgm_50_59``/``fgm_60p`` and a flat ``fgmiss``. Only missing
    keys are filled, so actual stat lines (which carry both forms) are unchanged.
    60+ yard makes are rare enough that a lone ``fgm_50p`` is credited to 50-59.
    """
    out: dict[str, float] = {}
    for kind in ("fgm", "fgmiss"):
        b50, b5059, b60 = f"{kind}_50p", f"{kind}_50_59", f"{kind}_60p"
        if b50 in stats and b5059 not in stats and b60 not in stats:
            out[b5059] = stats[b50]
        elif b50 not in stats and (b5059 in stats or b60 in stats):
            out[b50] = stats.get(b5059, 0.0) + stats.get(b60, 0.0)
        if kind not in stats:
            long = stats.get(b50, out.get(b50, 0.0))
            parts = [stats.get(f"{kind}_{b}", 0.0) for b in _KICK_BUCKETS]
            if any(f"{kind}_{b}" in stats for b in _KICK_BUCKETS) or long:
                out[kind] = sum(parts) + long
    return out


def _threshold_base(stat: str, stats: Mapping[str, float]) -> float | None:
    if stat == "rush_rec_yd":
        if "rush_yd" not in stats and "rec_yd" not in stats:
            return None
        return stats.get("rush_yd", 0.0) + stats.get("rec_yd", 0.0)
    return stats.get(stat)


def _tier_bounds(key: str) -> tuple[str, float, float] | None:
    """Return (family, lo, hi) inclusive bounds for a tier key, else None."""
    if m := _TIER_RANGE_RE.match(key):
        return m.group(1), float(m.group(2)), float(m.group(3))
    if m := _TIER_PLUS_RE.match(key):
        return m.group(1), float(m.group(2)), float("inf")
    if m := _TIER_ZERO_RE.match(key):
        return m.group(1), 0.0, 0.0
    return None


def _in_tier(value: float, lo: float, hi: float) -> bool:
    # Points allowed are integers in practice; yards tiers in Sleeper are written
    # "100_199" so both bounds are inclusive. A projected (fractional) value is
    # floored so it always lands in exactly one tier.
    v = float(int(value)) if value >= 0 else 0.0
    return lo <= v <= hi


def score(
    stats: Mapping[str, float],
    settings: Mapping[str, float],
    position: str | None = None,
) -> float:
    """Fantasy points for one stat line under ``settings``."""
    total = 0.0
    if any(k.startswith("fg") for k in stats):
        stats = {**stats, **_kicking_aggregates(stats)}
    # If the stat line already carries tier flags, the base rule scores them.
    tier_families_present = {b[0] for k in stats if (b := _tier_bounds(k))}

    for key, weight in settings.items():
        if not weight:
            continue
        if key in stats:
            total += stats[key] * weight
            continue

        if key in POSITION_REC_BONUS:
            if position == POSITION_REC_BONUS[key]:
                total += stats.get("rec", 0.0) * weight
            continue

        if m := _THRESHOLD_RE.match(key):
            base = _threshold_base(m.group(1), stats)
            if base is not None and base >= float(m.group(2)):
                total += weight
            continue

        if (bounds := _tier_bounds(key)) is not None:
            family, lo, hi = bounds
            if family in tier_families_present or family not in stats:
                continue
            if _in_tier(stats[family], lo, hi):
                total += weight
            continue

    return total


def score_rounded(
    stats: Mapping[str, float], settings: Mapping[str, float], position: str | None = None
) -> float:
    """Score rounded to 2 decimals, matching how Sleeper displays points."""
    return round(score(stats, settings, position), 2)


@dataclass(frozen=True)
class ScoringProfile:
    """Coarse description of a scoring system, for reports."""

    ppr: float
    te_premium: float
    pass_td: float
    has_idp: bool

    @property
    def label(self) -> str:
        base = {1.0: "PPR", 0.5: "Half-PPR", 0.0: "Standard"}.get(self.ppr, f"{self.ppr:g}-PPR")
        parts = [base]
        if self.te_premium:
            parts.append(f"TE+{self.te_premium:g}")
        if self.pass_td and self.pass_td != 4:
            parts.append(f"{self.pass_td:g}pt pass TD")
        if self.has_idp:
            parts.append("IDP")
        return ", ".join(parts)


def profile(settings: Mapping[str, float]) -> ScoringProfile:
    return ScoringProfile(
        ppr=float(settings.get("rec", 0.0)),
        te_premium=float(settings.get("bonus_rec_te", 0.0)),
        pass_td=float(settings.get("pass_td", 0.0)),
        has_idp=any(k.startswith("idp_") and v for k, v in settings.items()),
    )


# Sleeper's default PPR scoring plus common IDP settings. Used to fit the shipped
# default error model; leagues fit their own with ``ffsim fit-errors``.
SLEEPER_PPR_DEFAULTS: dict[str, float] = {
    "pass_yd": 0.04, "pass_td": 4, "pass_int": -1, "pass_2pt": 2,
    "rush_yd": 0.1, "rush_td": 6, "rush_2pt": 2,
    "rec": 1, "rec_yd": 0.1, "rec_td": 6, "rec_2pt": 2,
    "fum_lost": -2, "fum_rec_td": 6,
    "fgm_0_19": 3, "fgm_20_29": 3, "fgm_30_39": 3, "fgm_40_49": 4, "fgm_50_59": 5, "fgm_60p": 6,
    "fgmiss": -1, "xpm": 1, "xpmiss": -1,
    "def_td": 6, "def_st_td": 6, "st_td": 6, "sack": 1, "int": 2, "fum_rec": 2, "ff": 1,
    "safe": 2, "blk_kick": 2, "def_st_ff": 1, "def_st_fum_rec": 1, "st_ff": 1, "st_fum_rec": 1,
    "pts_allow_0": 10, "pts_allow_1_6": 7, "pts_allow_7_13": 4, "pts_allow_14_20": 1,
    "pts_allow_21_27": 0, "pts_allow_28_34": -1, "pts_allow_35p": -4,
    "idp_tkl_solo": 1, "idp_tkl_ast": 0.5, "idp_tkl_loss": 1, "idp_sack": 4, "idp_qb_hit": 0.5,
    "idp_int": 3, "idp_ff": 3, "idp_fum_rec": 2, "idp_pass_def": 1.5, "idp_def_td": 6,
    "idp_safe": 2, "idp_blk_kick": 2,
}  # fmt: skip
