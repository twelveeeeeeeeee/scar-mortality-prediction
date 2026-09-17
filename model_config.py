"""Validate reusable Cox settings and explicitly supplied model-selection policies."""

from dataclasses import dataclass
import math
from numbers import Integral, Real


def _finite_number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class CoxConfig:
    penalties: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.1, 1.0)
    l1_ratios: tuple[float, ...] = (0.1, 0.5, 0.9)
    categorical_min_frequency: int | float | None = None
    fusion_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    def __post_init__(self):
        for name in ("penalties", "l1_ratios", "fusion_weights"):
            values = getattr(self, name)
            if (not isinstance(values, (tuple, list)) or not values
                    or not all(_finite_number(value) for value in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"{name} must contain unique finite numeric values")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        if min(self.penalties) <= 0:
            raise ValueError("Cox penalties must be positive")
        if min(self.l1_ratios) <= 0 or max(self.l1_ratios) > 1:
            raise ValueError("Cox l1 ratios must be in (0, 1]")
        if min(self.fusion_weights) < 0 or 0.0 not in self.fusion_weights:
            raise ValueError("Fusion weights must be nonnegative and include the clinical-only zero fallback")
        frequency = self.categorical_min_frequency
        if frequency is not None and not (
                isinstance(frequency, Integral) and not isinstance(frequency, bool) and frequency >= 1
                or _finite_number(frequency) and not isinstance(frequency, Integral) and 0 < frequency < 1):
            raise ValueError("Categorical minimum frequency must be None, a positive integer, or a fraction in (0, 1)")


@dataclass(frozen=True)
class SelectionConfig:
    lambda_c_index_margin: float
    lambda_brier_margin: float
    fold_c_index_margin: float
    fold_auc_margin: float
    fold_brier_margin: float
    familywise_c_index_margin: float
    practical_auc_gain: float
    practical_brier_gain: float
    benefit_probability_minimum: float
    ranking_brier_weight: float
    ranking_c_index_weight: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not _finite_number(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.benefit_probability_minimum > 1:
            raise ValueError("Minimum benefit probability must be in [0, 1]")
        if any(getattr(self, name) > 1 for name in (
                "lambda_c_index_margin", "lambda_brier_margin", "fold_c_index_margin",
                "fold_auc_margin", "fold_brier_margin", "familywise_c_index_margin",
                "practical_auc_gain", "practical_brier_gain")):
            raise ValueError("Metric margins and required gains must be in [0, 1]")
