"""Extract probability-based scar phenotypes, outcome-free reliability, and review flags."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from feature_config import FeatureConfig, ReliabilityConfig, QCConfig, feature_config_metadata, validate_feature_config

CANDIDATE_FEATURES = (
    "scar_burden_hard", "scar_burden_soft", "scar_positive_slice_fraction_all",
    "scar_positive_slice_fraction_active", "scar_extent_fraction_active",
    "max_slice_burden", "upper_quantile_active_slice_burden", "top_active_slice_burden_mean",
    "longest_positive_run_fraction_active_span", "normalized_scar_span_active",
    "scar_center_distance_from_active_mid", "first_last_third_scar_fraction_abs",
    "middle_third_scar_fraction", "max_edge_third_scar_fraction",
    "abs_slice_burden_slope_active",
)
ADMITTED_FEATURES = (
    "scar_burden_hard", "scar_burden_soft", "scar_extent_fraction_active",
)
QC_UPPER_FEATURES = (
    "baseline_curriculum_burden_abs_delta",
    "baseline_curriculum_mean_abs_probability_delta_in_container",
    "baseline_curriculum_upper_quantile_abs_probability_delta_in_container",
    "baseline_curriculum_hard_disagreement_fraction_in_container",
    "mean_blend_multiclass_entropy_in_container",
    "near_threshold_fraction_in_container",
)
QC_LOWER_FEATURES = ("mean_scar_threshold_margin_in_container",)
QC_FEATURES = QC_UPPER_FEATURES + QC_LOWER_FEATURES


class AbstentionError(ValueError):
    pass


def validate_probabilities(array: np.ndarray) -> np.ndarray:
    probability = np.asarray(array, dtype=np.float64)
    if probability.ndim != 4 or probability.shape[1] != 3:
        raise ValueError("Probability arrays must have shape (ordered_slice, 3, y, x).")
    if any(size == 0 for size in probability.shape):
        raise ValueError("Probability arrays cannot have an empty dimension.")
    if not np.isfinite(probability).all():
        raise ValueError("Probability arrays must contain finite values.")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("Probabilities must lie in [0, 1].")
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=0, atol=5e-3):
        raise ValueError("Class probabilities must sum to one within tolerance 0.005.")
    return probability


def _longest_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _summarize_masks(scar: np.ndarray, container: np.ndarray,
                     soft_scar: float, soft_container: float,
                     config: FeatureConfig = FeatureConfig()) -> dict[str, float]:
    scar_counts = scar.sum(axis=(1, 2)).astype(float)
    container_counts = container.sum(axis=(1, 2)).astype(float)
    if container_counts.sum() <= 0 or soft_container <= 0:
        raise AbstentionError("empty_anatomical_container")
    burden = np.divide(scar_counts, container_counts, out=np.zeros_like(scar_counts),
                       where=container_counts > 0)
    active_indices = np.flatnonzero(container_counts > 0)
    start, stop = int(active_indices[0]), int(active_indices[-1]) + 1
    active_scar, active_burden = scar_counts[start:stop], burden[start:stop]
    positive = active_scar > 0
    length = len(active_scar)
    coordinate = np.linspace(0, 1, length) if length > 1 else np.array([0.5])
    positive_indices = np.flatnonzero(positive)
    center = float(np.average(coordinate, weights=active_scar)) if positive.any() else 0.5
    span = ((positive_indices[-1] - positive_indices[0]) / (length - 1)
            if positive.any() and length > 1 else 0.0)
    thirds = np.minimum((((np.arange(length) + 0.5) / length) * 3).astype(int), 2)
    thirds_scar = np.array([active_scar[thirds == index].sum() for index in range(3)])
    thirds_fraction = thirds_scar / max(float(active_scar.sum()), 1.0)
    slope = abs(float(np.polyfit(coordinate, active_burden, 1)[0])) if length > 1 else 0.0
    values = (
        scar_counts.sum() / container_counts.sum(), soft_scar / soft_container,
        (scar_counts > 0).mean(), positive.mean(), (active_burden >= config.slice_extent_threshold).mean(),
        active_burden.max(), np.quantile(active_burden, config.upper_tail_quantile),
        np.sort(active_burden)[-min(config.top_slice_count, length):].mean(),
        _longest_run(positive) / length, span, abs(center - 0.5),
        abs(thirds_fraction[0] - thirds_fraction[2]), thirds_fraction[1],
        max(thirds_fraction[0], thirds_fraction[2]), slope,
    )
    return dict(zip(CANDIDATE_FEATURES, map(float, values)))


def extract_patient_features(baseline: np.ndarray, curriculum: np.ndarray,
                             anatomy_valid: bool = True,
                             config: FeatureConfig = FeatureConfig()) -> dict[str, float]:
    if not isinstance(anatomy_valid, (bool, np.bool_)):
        raise ValueError("anatomy_valid must be an explicit boolean.")
    if not anatomy_valid:
        raise AbstentionError("invalid_anatomical_container")
    baseline = validate_probabilities(baseline)
    curriculum = validate_probabilities(curriculum)
    if baseline.shape != curriculum.shape:
        raise ValueError("Baseline and curriculum arrays must have identical shapes and ordering.")
    blend = config.baseline_weight * baseline + (1 - config.baseline_weight) * curriculum
    scar = blend[:, 2] >= config.scar_threshold
    container = (np.argmax(blend, axis=1) == 1) | scar
    result = _summarize_masks(scar, container, float(blend[:, 2].sum()),
                              float((blend[:, 1] + blend[:, 2]).sum()), config)
    baseline_hard = baseline[:, 2][container] >= config.scar_threshold
    curriculum_hard = curriculum[:, 2][container] >= config.scar_threshold
    delta = np.abs(baseline[:, 2][container] - curriculum[:, 2][container])
    probabilities = np.moveaxis(blend, 1, -1)[container]
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-8, 1)), axis=1)
    margin = np.abs(blend[:, 2][container] - config.scar_threshold)
    result.update({
        "baseline_curriculum_burden_abs_delta": float(abs(baseline_hard.mean() - curriculum_hard.mean())),
        "baseline_curriculum_mean_abs_probability_delta_in_container": float(delta.mean()),
        "baseline_curriculum_upper_quantile_abs_probability_delta_in_container": float(np.quantile(delta, config.upper_tail_quantile)),
        "baseline_curriculum_hard_disagreement_fraction_in_container": float(np.logical_xor(baseline_hard, curriculum_hard).mean()),
        "mean_blend_multiclass_entropy_in_container": float(entropy.mean()),
        "mean_scar_threshold_margin_in_container": float(margin.mean()),
        "near_threshold_fraction_in_container": float((margin <= config.near_threshold_margin).mean()),
    })
    return result


def extract_reference_features(labels: np.ndarray, anatomy_valid: bool = True,
                               config: FeatureConfig = FeatureConfig()) -> dict[str, float]:
    labels = np.asarray(labels)
    if labels.ndim != 3 or any(size == 0 for size in labels.shape):
        raise ValueError("Reference labels must have shape (ordered_slice, y, x).")
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Reference labels must encode background=0, myocardium=1, scar=2.")
    if not anatomy_valid:
        raise AbstentionError("invalid_reference_container")
    scar, container = labels == 2, labels > 0
    return _summarize_masks(scar, container, float(scar.sum()), float(container.sum()), config)


def icc_absolute_agreement(automatic: np.ndarray, reference: np.ndarray) -> float:
    values = np.column_stack([automatic, reference]).astype(float)
    n, k = values.shape
    if n < 3 or not np.isfinite(values).all():
        return float("nan")
    grand, row_mean, column_mean = values.mean(), values.mean(axis=1), values.mean(axis=0)
    ms_rows = k * np.sum((row_mean - grand) ** 2) / (n - 1)
    ms_columns = n * np.sum((column_mean - grand) ** 2) / (k - 1)
    residual = values - row_mean[:, None] - column_mean[None, :] + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if abs(denominator) > 1e-15 else float("nan")


def _spearman(automatic: np.ndarray, reference: np.ndarray) -> float:
    if len(automatic) < 3 or np.ptp(automatic) == 0 or np.ptp(reference) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(automatic, reference).statistic)


def reliability_screen(automatic: pd.DataFrame, reference: pd.DataFrame,
                       bootstraps: int = 2000, seed: int = 42,
                       config: ReliabilityConfig | None = None) -> pd.DataFrame:
    if not isinstance(config, ReliabilityConfig):
        raise ValueError("An explicit ReliabilityConfig policy is required.")
    required = {"patient_id", "imaging_fold", *CANDIDATE_FEATURES}
    if not required.issubset(automatic.columns):
        raise ValueError("Automatic table lacks patient_id, imaging_fold, or candidate features.")
    if not {"patient_id", *CANDIDATE_FEATURES}.issubset(reference.columns):
        raise ValueError("Reference table lacks patient_id or candidate features.")
    if automatic.patient_id.duplicated().any() or reference.patient_id.duplicated().any():
        raise ValueError("Reliability requires one development OOF row per patient.")
    if set(automatic.patient_id) != set(reference.patient_id):
        raise ValueError("Automatic and reference patient sets must match exactly.")
    if "split" in automatic and set(automatic.split) != {"development_oof"}:
        raise ValueError("Reliability admission accepts development OOF patients only.")
    reference = reference.set_index("patient_id").loc[automatic.patient_id]
    folds = np.asarray(automatic.imaging_fold)
    unique_folds = np.unique(folds)
    if len(unique_folds) != 5 or pd.isna(folds).any():
        raise ValueError("Reliability admission requires exactly five imaging folds.")
    if bootstraps < 0:
        raise ValueError("Bootstrap count must be non-negative.")
    rng = np.random.default_rng(seed)
    results = []
    for feature in CANDIDATE_FEATURES:
        x, y = automatic[feature].to_numpy(float), reference[feature].to_numpy(float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite reliability inputs for {feature}.")
        icc, rho = icc_absolute_agreement(x, y), _spearman(x, y)
        fold_rho = [_spearman(x[folds == fold], y[folds == fold]) for fold in unique_folds]
        fold_mae = [float(np.mean(np.abs(x[folds == fold] - y[folds == fold]))) for fold in unique_folds]
        error_limit = max(config.absolute_mae_floor, config.median_mae_multiplier * float(np.median(fold_mae)))
        reasons = []
        if not np.isfinite(icc) or icc < config.minimum_icc:
            reasons.append("icc_below_minimum")
        if not np.isfinite(rho) or rho < config.minimum_spearman:
            reasons.append("spearman_below_minimum")
        if not np.isfinite(fold_rho).all() or min(fold_rho) < config.minimum_fold_spearman:
            reasons.append("undefined_or_negative_fold_correlation")
        if max(fold_mae) > error_limit:
            reasons.append("fold_mae_exceeds_guard")
        samples = []
        for _ in range(bootstraps):
            indices = rng.integers(len(x), size=len(x))
            value = icc_absolute_agreement(x[indices], y[indices])
            if np.isfinite(value):
                samples.append(value)
        lower, upper = np.quantile(samples, [0.025, 0.975]) if samples else (np.nan, np.nan)
        results.append({
            "feature": feature, "icc_a1": icc, "icc_lower_95": lower,
            "icc_upper_95": upper, "valid_bootstraps": len(samples), "spearman": rho,
            "minimum_fold_spearman": min(fold_rho) if np.isfinite(fold_rho).all() else np.nan,
            "maximum_fold_mae": max(fold_mae), "fold_mae_limit": error_limit,
            "mae": float(np.mean(np.abs(x - y))), "admitted": not reasons,
            "exclusion_reasons": ";".join(reasons),
        })
    return pd.DataFrame(results)


def fit_qc_thresholds(development_oof: pd.DataFrame,
                      config: QCConfig | None = None) -> dict[str, Any]:
    if not isinstance(config, QCConfig):
        raise ValueError("An explicit QCConfig quantile policy is required.")
    if "split" not in development_oof or set(development_oof.split) != {"development_oof"}:
        raise ValueError("QC thresholds must use development OOF rows only.")
    if development_oof.patient_id.duplicated().any():
        raise ValueError("QC fitting requires exactly one OOF row per patient.")
    if len(development_oof) < 2:
        raise ValueError("QC fitting requires at least two development patients.")
    if "prediction_status" in development_oof and not (development_oof.prediction_status == "ok").all():
        raise ValueError("Invalid containers must be resolved before freezing development QC.")
    if not np.isfinite(development_oof[list(QC_FEATURES)].to_numpy(float)).all():
        raise ValueError("QC fitting requires finite values.")
    thresholds = {
        "upper": {name: float(development_oof[name].quantile(config.upper_quantile)) for name in QC_UPPER_FEATURES},
        "lower": {name: float(development_oof[name].quantile(config.lower_quantile)) for name in QC_LOWER_FEATURES},
        "qc_config": {"upper_quantile": config.upper_quantile, "lower_quantile": config.lower_quantile},
    }
    if "feature_config_json" in development_oof or "feature_config_sha256" in development_oof:
        thresholds.update(feature_config_metadata(validate_feature_config(development_oof)))
    return thresholds


def apply_qc(features: dict[str, Any], thresholds: dict[str, dict[str, float]],
             prediction_status: str = "ok") -> dict[str, Any]:
    if prediction_status != "ok":
        return {"review_flag": True, "review_reasons": "abstention_invalid_container"}
    reasons = []
    for tail, names in (("upper", QC_UPPER_FEATURES), ("lower", QC_LOWER_FEATURES)):
        for name in names:
            value, threshold = float(features[name]), float(thresholds[tail][name])
            if not np.isfinite(value) or not np.isfinite(threshold):
                raise ValueError("QC inputs and thresholds must be finite.")
            if (tail == "upper" and value > threshold) or (tail == "lower" and value < threshold):
                reasons.append(f"{tail}:{name}")
    return {"review_flag": bool(reasons), "review_reasons": ";".join(reasons)}
