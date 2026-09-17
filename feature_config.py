"""Define validated feature, reliability, and review settings and verify extraction identity."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path


def _unit(name: str, value: float, lower_open: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if not (0 < value <= 1 if lower_open else 0 <= value <= 1):
        raise ValueError(f"{name} must lie in {'(0, 1]' if lower_open else '[0, 1]'}.")


def _positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")


def _integer(name: str, value: int, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}.")


@dataclass(frozen=True)
class FeatureConfig:
    scar_threshold: float = 0.5
    baseline_weight: float = 0.5
    slice_extent_threshold: float = 0.05
    near_threshold_margin: float = 0.05
    upper_tail_quantile: float = 0.9
    top_slice_count: int = 3

    def __post_init__(self):
        for name in ("scar_threshold", "slice_extent_threshold"):
            _unit(name, getattr(self, name), lower_open=True)
        for name in ("baseline_weight", "near_threshold_margin", "upper_tail_quantile"):
            _unit(name, getattr(self, name))
        _integer("top_slice_count", self.top_slice_count)


@dataclass(frozen=True)
class ReliabilityConfig:
    minimum_icc: float
    minimum_spearman: float
    minimum_fold_spearman: float
    absolute_mae_floor: float
    median_mae_multiplier: float

    def __post_init__(self):
        for name in ("minimum_icc", "minimum_spearman", "minimum_fold_spearman", "absolute_mae_floor"):
            _unit(name, getattr(self, name))
        _positive("median_mae_multiplier", self.median_mae_multiplier)


@dataclass(frozen=True)
class QCConfig:
    upper_quantile: float
    lower_quantile: float

    def __post_init__(self):
        _unit("upper_quantile", self.upper_quantile)
        _unit("lower_quantile", self.lower_quantile)
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError("lower_quantile must be smaller than upper_quantile.")


@dataclass(frozen=True)
class ReviewConfig:
    measurement_threshold: float = 0.5
    consensus_weights: tuple[float, ...] = (1 / 6,) * 6
    baseline_weight: float = 0.5
    near_threshold_margin: float = 0.05
    upper_tail_quantile: float = 0.9
    priority_quantile: float | None = None
    grid_size: int = 4
    n_estimators: int = 100
    min_samples_leaf: int = 1
    max_features: float = 1.0
    empty_correct_weight: float = 1.0
    empty_false_positive_weight: float = 1.0
    positive_weight_maximum: float = 0.0
    positive_area_scale: float = 1.0
    dice_smoothing: float = 1.0

    def __post_init__(self):
        _unit("measurement_threshold", self.measurement_threshold, lower_open=True)
        for name in ("baseline_weight", "near_threshold_margin", "upper_tail_quantile"):
            _unit(name, getattr(self, name))
        if self.priority_quantile is not None:
            _unit("priority_quantile", self.priority_quantile)
        _unit("max_features", self.max_features, lower_open=True)
        _integer("grid_size", self.grid_size, minimum=2)
        _integer("n_estimators", self.n_estimators)
        _integer("min_samples_leaf", self.min_samples_leaf)
        for name in ("empty_correct_weight", "empty_false_positive_weight", "positive_area_scale", "dice_smoothing"):
            _positive(name, getattr(self, name))
        if (isinstance(self.positive_weight_maximum, bool)
                or not isinstance(self.positive_weight_maximum, (int, float))
                or not math.isfinite(self.positive_weight_maximum)
                or self.positive_weight_maximum < 0):
            raise ValueError("positive_weight_maximum must be non-negative and finite.")
        if len(self.consensus_weights) != 6:
            raise ValueError("consensus_weights must contain six expert weights.")
        for value in self.consensus_weights:
            _unit("consensus_weights", value)
        if not math.isclose(sum(self.consensus_weights), 1.0, abs_tol=1e-12):
            raise ValueError("consensus_weights must sum to one.")
        object.__setattr__(self, "consensus_weights", tuple(self.consensus_weights))


def config_from_dict(kind: type, values: dict):
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object.")
    unknown = set(values) - {field.name for field in fields(kind)}
    if unknown:
        raise ValueError(f"Unknown {kind.__name__} settings: {sorted(unknown)}.")
    missing = {field.name for field in fields(kind)
               if field.default is MISSING and field.default_factory is MISSING} - values.keys()
    if missing:
        raise ValueError(f"{kind.__name__} requires explicit settings: {sorted(missing)}.")
    return kind(**values)


def load_config(kind: type, path: Path | None):
    values = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
    return config_from_dict(kind, values)


def feature_config_metadata(config: FeatureConfig) -> dict[str, str]:
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"feature_config_json": encoded, "feature_config_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()}


def validate_feature_config(frame, expected: FeatureConfig | dict | None = None) -> FeatureConfig:
    names = ("feature_config_json", "feature_config_sha256")
    if frame.empty or not set(names).issubset(frame.columns):
        raise ValueError("Feature rows must contain extraction configuration JSON and SHA256 identity.")
    if any(frame[name].isna().any() or frame[name].nunique() != 1 for name in names):
        raise ValueError("All feature rows must share one extraction configuration and identity.")
    raw = str(frame[names[0]].iloc[0])
    config = config_from_dict(FeatureConfig, json.loads(raw))
    metadata = feature_config_metadata(config)
    if raw != metadata[names[0]] or str(frame[names[1]].iloc[0]) != metadata[names[1]]:
        raise ValueError("Feature configuration identity does not match its canonical values.")
    if expected is not None:
        if isinstance(expected, dict) and "feature_config_json" in expected:
            expected = config_from_dict(FeatureConfig, json.loads(expected["feature_config_json"]))
        elif isinstance(expected, dict):
            expected = config_from_dict(FeatureConfig, expected)
        if config != expected:
            raise ValueError("Feature extraction configuration differs from the frozen model.")
    return config


def feature_table_fingerprint(frame, feature_columns) -> str:
    import pandas as pd

    validate_feature_config(frame)
    names = tuple(feature_columns)
    if not names or len(names) != len(set(names)):
        raise ValueError("Feature fingerprint requires a non-empty unique feature list.")
    required = ("patient_id", "imaging_fold", *names, "baseline_model_sha256",
                "curriculum_model_sha256", "feature_config_sha256")
    if not set(required).issubset(frame.columns):
        raise ValueError("Feature fingerprint inputs lack patient, fold, feature, or model identity columns.")
    if frame.patient_id.isna().any() or frame.patient_id.map(lambda value: not str(value).strip()).any():
        raise ValueError("Feature fingerprint requires non-empty patient IDs.")
    folds = pd.to_numeric(frame.imaging_fold, errors="coerce")
    if not folds.isin(range(5)).all():
        raise ValueError("Feature fingerprint requires integer imaging folds from 0 through 4.")
    normalized = frame.loc[:, list(required)].copy()
    normalized["patient_id"] = normalized.patient_id.astype(str)
    normalized["imaging_fold"] = folds.astype(int)
    if normalized.duplicated(["patient_id", "imaging_fold"]).any():
        raise ValueError("Feature fingerprint requires unique patient and fold pairs.")
    for name in names:
        normalized[name] = pd.to_numeric(normalized[name], errors="raise").astype(float)
        if not normalized[name].map(math.isfinite).all():
            raise ValueError("Feature fingerprint requires finite feature values.")
    for name in ("baseline_model_sha256", "curriculum_model_sha256", "feature_config_sha256"):
        if not normalized[name].map(lambda value: isinstance(value, str) and len(value) == 64
                                     and all(character in "0123456789abcdef" for character in value)).all():
            raise ValueError("Feature fingerprint requires lowercase SHA256 model and configuration identities.")
    normalized = normalized.reset_index(drop=True).sort_values(["patient_id", "imaging_fold"], kind="stable")
    rows = normalized.to_dict("records")
    payload = {"schema_version": 1, "feature_columns": list(names), "rows": rows}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
