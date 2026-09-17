"""Select mortality fusion candidates using development-only error and stability gates."""

from __future__ import annotations

from dataclasses import asdict
import numpy as np

from model_config import CoxConfig, SelectionConfig
from survival_metrics import HORIZONS, as_survival_matrix, evaluate_metrics, km_curve, validate_outcomes


LAMBDA_GRID = CoxConfig().fusion_weights
COMPARISON_METRICS = ("harrell_c_index", "3_year_auc", "3_year_brier")


def select_lambda(candidate_metrics, *, config: SelectionConfig):
    if not isinstance(config, SelectionConfig):
        raise TypeError("Lambda selection requires an explicit SelectionConfig")
    if 0.0 not in candidate_metrics:
        raise ValueError("Lambda selection requires the clinical-only lambda=0 fallback")
    if any(not np.isfinite(weight) or weight < 0 for weight in candidate_metrics):
        raise ValueError("Fusion weights must be finite and nonnegative")
    baseline = candidate_metrics[0.0]
    required = ("harrell_c_index", "1_year_brier", "3_year_brier", "5_year_brier")
    def finite_metrics(metrics):
        return all(metrics.get(key) is not None and np.isfinite(metrics[key]) for key in required)

    baseline_available = finite_metrics(baseline)
    rows = []
    for weight, metrics in sorted(candidate_metrics.items()):
        finite = finite_metrics(metrics)
        passed = bool(baseline_available and finite
                      and metrics["harrell_c_index"] >= baseline["harrell_c_index"] - config.lambda_c_index_margin
                      and metrics["1_year_brier"] <= baseline["1_year_brier"] + config.lambda_brier_margin
                      and metrics["5_year_brier"] <= baseline["5_year_brier"] + config.lambda_brier_margin)
        rows.append({"lambda": float(weight), "guard_pass": passed, **metrics})
    policy = {**asdict(config),
              "objective": "minimum_3_year_ipcw_brier", "tie_break": "smaller_lambda"}
    if not baseline_available:
        return {"selected_lambda": 0.0, "selection_available": False,
                "reason": "unavailable_selection_metrics", "candidates": rows, "policy": policy}
    eligible = [row for row in rows if row["guard_pass"]]
    selected = min(eligible, key=lambda row: (row["3_year_brier"], row["lambda"]))
    return {"selected_lambda": selected["lambda"], "selection_available": True,
            "reason": "minimum_eligible_brier" if selected["lambda"] else "clinical_fallback_best",
            "candidates": rows, "policy": policy}


def compare_candidates(time, event, clinical_risk, clinical_survival, candidates,
                       imaging_folds, reference_time, reference_event,
                       horizons=HORIZONS, samples=2000, seed=42,
                       familywise_alpha=0.05, *, config: SelectionConfig):
    if not isinstance(config, SelectionConfig):
        raise TypeError("Candidate comparison requires an explicit SelectionConfig")
    time, event = validate_outcomes(time, event)
    reference_time, reference_event = validate_outcomes(reference_time, reference_event)
    clinical_risk = np.asarray(clinical_risk, dtype=float)
    clinical_survival = as_survival_matrix(clinical_survival, horizons)
    imaging_folds = np.asarray(imaging_folds)
    if imaging_folds.shape != time.shape or len(np.unique(imaging_folds)) != 5:
        raise ValueError("Imaging allocation must contain five folds and one value per patient")
    if not candidates or samples < 1 or not 0 < familywise_alpha < 1:
        raise ValueError("Candidates, bootstrap count and familywise alpha must be valid")
    censoring = km_curve(reference_time, ~reference_event)
    reference = evaluate_metrics(time, event, clinical_risk, clinical_survival,
                                 reference_time, reference_event, horizons, censoring)
    prepared = {}
    for name, candidate in candidates.items():
        risk = np.asarray(candidate["risk"], dtype=float)
        survival = as_survival_matrix(candidate["survival"], horizons)
        point = evaluate_metrics(time, event, risk, survival, reference_time,
                                 reference_event, horizons, censoring)
        prepared[name] = {"risk": risk, "survival": survival,
                          "delta": {key: point[key] - reference[key] for key in COMPARISON_METRICS},
                          "draws": {key: [] for key in COMPARISON_METRICS}}
    generator = np.random.default_rng(seed)
    for _ in range(samples):
        indices = generator.integers(0, len(time), size=len(time))
        ref = evaluate_metrics(time[indices], event[indices], clinical_risk[indices],
                               clinical_survival[indices], reference_time, reference_event,
                               horizons, censoring)
        for candidate in prepared.values():
            metrics = evaluate_metrics(time[indices], event[indices], candidate["risk"][indices],
                                       candidate["survival"][indices], reference_time,
                                       reference_event, horizons, censoring)
            for key in COMPARISON_METRICS:
                delta = metrics[key] - ref[key]
                if np.isfinite(delta):
                    candidate["draws"][key].append(float(delta))
    comparisons_count = len(candidates) * len(COMPARISON_METRICS)
    tail = familywise_alpha / (2 * comparisons_count)
    output = {}
    for name, candidate in prepared.items():
        metrics = {}
        for key, values in candidate["draws"].items():
            array = np.asarray(values, dtype=float)
            metrics[key] = {
                "point": float(candidate["delta"][key]) if np.isfinite(candidate["delta"][key]) else None,
                "lower_familywise": float(np.quantile(array, tail)) if len(array) else None,
                "upper_familywise": float(np.quantile(array, 1 - tail)) if len(array) else None,
                "probability_of_benefit": float(np.mean(array < 0 if "brier" in key else array > 0)) if len(array) else None,
                "valid_resamples": len(array),
                "invalid_resamples": samples - len(array),
            }
        folds = []
        for fold in np.unique(imaging_folds):
            mask = imaging_folds == fold
            ref = evaluate_metrics(time[mask], event[mask], clinical_risk[mask],
                                   clinical_survival[mask], reference_time, reference_event,
                                   horizons, censoring)
            val = evaluate_metrics(time[mask], event[mask], candidate["risk"][mask],
                                   candidate["survival"][mask], reference_time, reference_event,
                                   horizons, censoring)
            delta = {key: val[key] - ref[key] for key in COMPARISON_METRICS}
            passed = bool(all(np.isfinite(value) for value in delta.values())
                          and delta["harrell_c_index"] >= -config.fold_c_index_margin
                          and delta["3_year_auc"] >= -config.fold_auc_margin
                          and delta["3_year_brier"] <= config.fold_brier_margin)
            folds.append({"fold": str(fold), "guard_pass": passed,
                          "delta": {key: float(value) if np.isfinite(value) else None for key, value in delta.items()}})
        finite_bounds = all(metrics[key]["lower_familywise"] is not None for key in COMPARISON_METRICS)
        interval_gate = bool(finite_bounds
                             and metrics["3_year_auc"]["lower_familywise"] > 0
                             and metrics["3_year_brier"]["upper_familywise"] < 0
                             and metrics["harrell_c_index"]["lower_familywise"] > -config.familywise_c_index_margin)
        delta = candidate["delta"]
        practical_gate = bool(all(np.isfinite(value) for value in delta.values())
                              and finite_bounds
                              and delta["3_year_auc"] >= config.practical_auc_gain
                              and delta["3_year_brier"] <= -config.practical_brier_gain
                              and delta["harrell_c_index"] >= 0
                              and metrics["3_year_auc"]["probability_of_benefit"] >= config.benefit_probability_minimum
                              and metrics["3_year_brier"]["probability_of_benefit"] >= config.benefit_probability_minimum)
        score = (delta["3_year_auc"] - config.ranking_brier_weight * delta["3_year_brier"]
                 + config.ranking_c_index_weight * delta["harrell_c_index"])
        output[name] = {"metrics": metrics, "imaging_fold_guards": folds,
                        "familywise_gate": interval_gate, "practical_gate": practical_gate,
                        "advance_to_holdout": bool(all(row["guard_pass"] for row in folds)
                                                   and (interval_gate or practical_gate)),
                        "ranking_score": float(score) if np.isfinite(score) else None}
    eligible = [name for name, row in output.items() if row["advance_to_holdout"]]
    ranked = sorted(eligible, key=lambda name: (-output[name]["ranking_score"], name))
    return {"selected_candidate": ranked[0] if ranked else "clinical_only",
            "ranked_passing_candidates": ranked, "comparisons": output,
            "shared_bootstrap_samples": samples, "bootstrap_seed": seed,
            "policy": asdict(config),
            "familywise_alpha": familywise_alpha,
            "familywise_comparisons": comparisons_count,
            "familywise_interval_level": 1 - familywise_alpha / comparisons_count}
