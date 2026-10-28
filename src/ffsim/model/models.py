"""The fitted model set (outcome errors, talent, availability) and its persistence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from ffsim.data.history import HistoryStore, player_seasons
from ffsim.model.availability import AvailabilityModel, fit_availability
from ffsim.model.bayes import TalentModel, fit_talent, prior_rate
from ffsim.model.correlation import CorrelationModel, fit_correlation
from ffsim.model.errors import ErrorModel, fit_error_model

DEFAULT_FILE = "default_models.json"


@dataclass
class ModelSet:
    errors: ErrorModel
    talent: TalentModel = field(default_factory=TalentModel)
    availability: AvailabilityModel = field(default_factory=AvailabilityModel)
    correlation: CorrelationModel = field(default_factory=CorrelationModel)
    meta: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "meta": self.meta,
                "errors": json.loads(self.errors.to_json()),
                "talent": self.talent.to_dict(),
                "availability": self.availability.to_dict(),
                "correlation": self.correlation.to_dict(),
            }
        )

    @classmethod
    def from_json(cls, text: str) -> ModelSet:
        d = json.loads(text)
        return cls(
            errors=ErrorModel.from_json(json.dumps(d["errors"])),
            talent=TalentModel.from_dict(d.get("talent", {})),
            availability=AvailabilityModel.from_dict(d.get("availability", {})),
            correlation=CorrelationModel.from_dict(d.get("correlation", {})),
            meta=d.get("meta", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> ModelSet:
        return cls.from_json(path.read_text())

    @classmethod
    def default(cls) -> ModelSet:
        """Models fitted on 2023-2025 Sleeper data with default PPR (+IDP) scoring."""
        return cls.from_json(resources.files("ffsim.model").joinpath(DEFAULT_FILE).read_text())


def models_path(cache_dir: Path, settings: Mapping[str, float]) -> Path:
    key = json.dumps(sorted((k, float(v)) for k, v in settings.items() if v))
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return cache_dir / "models" / f"{digest}.json"


def load_models(cache_dir: Path, settings: Mapping[str, float]) -> tuple[ModelSet, bool]:
    """League-fitted models if ``ffsim fit`` has been run, else the defaults."""
    path = models_path(cache_dir, settings)
    if path.exists():
        return ModelSet.load(path), True
    return ModelSet.default(), False


def fit_models(
    store: HistoryStore,
    seasons: Iterable[int],
    settings: Mapping[str, float],
) -> tuple[ModelSet, dict[str, object]]:
    seasons = list(seasons)
    rows = list(store.rows(seasons))
    errors = fit_error_model(rows, settings)
    by_player, team_weeks = player_seasons(rows, settings)
    priors: dict[tuple[int, str], float] = {}
    for season in seasons:
        for pid, (pos, stats) in store.preseason(season).items():
            rate = prior_rate(stats, settings, pos)
            if rate is not None:
                priors[(season, pid)] = rate
    talent, talent_diag = fit_talent(by_player, priors, errors)
    availability, avail_diag = fit_availability(by_player, team_weeks)
    correlation, corr_diag = fit_correlation(rows, settings, errors, talent.rho)
    meta: dict[str, object] = {"seasons": seasons}
    return ModelSet(errors, talent, availability, correlation, meta), {
        "talent": talent_diag,
        "availability": avail_diag,
        "correlation": corr_diag,
    }
