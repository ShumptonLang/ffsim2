"""Hand-built SimInputs for engine and decision tests."""

import numpy as np

from ffsim.league import PlayoffFormat, PlayoffRoundType, PlayoffSeedType
from ffsim.model.availability import AvailabilityModel
from ffsim.model.bayes import TalentModel
from ffsim.model.correlation import CorrelationModel
from ffsim.model.models import ModelSet
from ffsim.sim.inputs import SimInputs


def independent(models: ModelSet) -> ModelSet:
    """No talent persistence or injuries (milestone 2 behaviour)."""
    return ModelSet(
        models.errors,
        TalentModel(default_rho=0.0),
        AvailabilityModel(default_hazard=0.0),
        CorrelationModel(),
    )


def synthetic_inputs(n_teams=6, weeks=(10, 11, 12), playoff_teams=4, strengths=None, base=None):
    strengths = strengths or [10.0] * n_teams
    po = PlayoffFormat(playoff_teams, 13, PlayoffRoundType.ONE_WEEK, PlayoffSeedType.FIXED)
    all_weeks = list(weeks) + [w for r in po.round_weeks() for w in r]
    rosters, means, positions = {}, {}, {}
    for t in range(n_teams):
        ids = [f"t{t}_{k}" for k in range(3)]
        rosters[t + 1] = ids
        for p in ids:
            positions[p] = ("WR",)
    for w in all_weeks:
        means[w] = {p: strengths[int(p[1:].split("_")[0])] for p in positions}
    pairs = [(i, n_teams - 1 - i) for i in range(n_teams // 2)]
    return SimInputs(
        team_ids=list(range(1, n_teams + 1)),
        team_labels=[f"T{i}" for i in range(n_teams)],
        rosters=rosters,
        slots=["WR", "WR", "FLEX"],
        weeks=all_weeks,
        regular_weeks=list(weeks),
        playoffs=po,
        matchups={w: pairs for w in weeks},
        base_half_wins=np.array(base if base is not None else [0.0] * n_teams),
        base_points_for=np.zeros(n_teams),
        divisions=[],
        median_match=False,
        means=means,
        positions=positions,
    )
