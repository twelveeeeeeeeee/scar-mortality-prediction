"""Evaluate censored mortality predictions with fixed development-reference weighting."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


HORIZONS = (1.0, 3.0, 5.0)


def validate_outcomes(time, event):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    if time.ndim != 1 or event.shape != time.shape or not len(time):
        raise ValueError("Outcome arrays must have the same nonempty one-dimensional shape")
    if not np.isfinite(time).all() or np.any(time <= 0):
        raise ValueError("Follow-up times must be finite and positive")
    if not np.isin(event, (0, 1, False, True)).all():
        raise ValueError("Event indicators must be binary")
    return time, event.astype(bool)


def km_curve(time, event):
    time, event = validate_outcomes(time, event)
    unique, inverse, counts = np.unique(time, return_inverse=True, return_counts=True)
    failures = np.bincount(inverse, weights=event.astype(float), minlength=len(unique))
    at_risk = len(time) - np.r_[0, np.cumsum(counts[:-1])]
    return unique, np.cumprod(1.0 - failures / at_risk)


def km_predict(curve, query, left_limit=False):
    times, values = curve
    query = np.asarray(query, dtype=float)
    indices = np.searchsorted(times, query, side="left" if left_limit else "right") - 1
    return np.where(indices >= 0, values[np.maximum(indices, 0)], 1.0)


def harrell_concordance(time, event, risk):
    time, event = validate_outcomes(time, event)
    risk = np.asarray(risk, dtype=float)
    if risk.shape != time.shape or not np.isfinite(risk).all():
        raise ValueError("Risk scores must be finite and match outcomes")
    comparable = event[:, None] & (time[None, :] > time[:, None])
    count = int(comparable.sum())
    if not count:
        return float("nan"), 0
    differences = risk[:, None] - risk[None, :]
    numerator = np.count_nonzero(comparable & (differences > 0))
    numerator += 0.5 * np.count_nonzero(comparable & (differences == 0))
    return float(numerator / count), count


def as_survival_matrix(survival, horizons=HORIZONS):
    horizons = np.asarray(horizons, dtype=float)
    if horizons.ndim != 1 or not len(horizons) or not np.isfinite(horizons).all() or np.any(horizons <= 0):
        raise ValueError("Evaluation horizons must be a nonempty vector of finite positive times")
    if isinstance(survival, Mapping):
        columns = []
        for horizon in horizons:
            possible = (horizon, str(horizon), f"{horizon:g}_year")
            key = next((key for key in possible if key in survival), None)
            if key is None:
                raise ValueError(f"Missing survival predictions for horizon {horizon}")
            columns.append(np.asarray(survival[key], dtype=float))
        matrix = np.column_stack(columns)
    else:
        matrix = np.asarray(survival, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(horizons):
        raise ValueError("Survival predictions require one column per horizon")
    if not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("Survival probabilities must be finite and within [0, 1]")
    if any(b <= a for a, b in zip(horizons, horizons[1:])):
        raise ValueError("Horizons must be strictly increasing")
    if np.any(np.diff(matrix, axis=1) > 1e-10):
        raise ValueError("Survival probability must not increase over time")
    return matrix


def weighted_auc(case_scores, case_weights, control_scores, control_weights):
    if not len(case_scores) or not len(control_scores):
        return float("nan")
    total = float(np.sum(case_weights) * np.sum(control_weights))
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    differences = np.asarray(case_scores)[:, None] - np.asarray(control_scores)[None, :]
    weights = np.asarray(case_weights)[:, None] * np.asarray(control_weights)[None, :]
    return float(np.sum(weights * ((differences > 0) + 0.5 * (differences == 0))) / total)


def evaluate_metrics(time, event, risk, survival, reference_time, reference_event,
                     horizons=HORIZONS, censoring_curve=None):
    time, event = validate_outcomes(time, event)
    reference_time, reference_event = validate_outcomes(reference_time, reference_event)
    matrix = as_survival_matrix(survival, horizons)
    if matrix.shape[0] != len(time):
        raise ValueError("Prediction rows must match outcome rows")
    censoring = censoring_curve if censoring_curve is not None else km_curve(
        reference_time, ~reference_event
    )
    observed_curve = km_curve(time, event)
    c_index, _ = harrell_concordance(time, event, risk)
    result = {"harrell_c_index": c_index}
    for index, horizon in enumerate(horizons):
        case = event & (time <= horizon)
        control = time > horizon
        g_case = km_predict(censoring, time[case], left_limit=True)
        g_horizon = float(km_predict(censoring, horizon))
        valid_weights = bool(horizon <= np.max(reference_time)
                             and (case.any() or control.any())
                             and np.all(g_case > 0) and g_horizon > 0)
        prediction = matrix[:, index]
        if valid_weights:
            case_weights = 1.0 / g_case
            control_weights = np.full(int(control.sum()), 1.0 / g_horizon) if control.any() else np.empty(0)
            scores = np.asarray(risk, dtype=float)
            auc = weighted_auc(scores[case], case_weights, scores[control], control_weights)
            brier = float((np.sum(case_weights * prediction[case] ** 2)
                           + np.sum(control_weights * (1.0 - prediction[control]) ** 2)) / len(time))
        else:
            auc = brier = float("nan")
        prefix = f"{horizon:g}_year"
        result[f"{prefix}_auc"] = auc
        result[f"{prefix}_brier"] = brier
        observed_supported = horizon <= np.max(time) or observed_curve[1][-1] == 0
        result[f"{prefix}_calibration_error"] = float(
            np.mean(1.0 - prediction) - (1.0 - km_predict(observed_curve, horizon))
        ) if observed_supported else float("nan")
    return result


def fixed_prediction_bootstrap(time, event, risk, survival, reference_time,
                               reference_event, horizons=HORIZONS, samples=2000,
                               seed=42):
    time, event = validate_outcomes(time, event)
    reference_time, reference_event = validate_outcomes(reference_time, reference_event)
    risk = np.asarray(risk, dtype=float)
    matrix = as_survival_matrix(survival, horizons)
    if samples < 1:
        raise ValueError("Bootstrap sample count must be positive")
    censoring = km_curve(reference_time, ~reference_event)
    point = evaluate_metrics(time, event, risk, matrix, reference_time, reference_event,
                             horizons, censoring)
    draws = {key: [] for key in point}
    generator = np.random.default_rng(seed)
    for _ in range(samples):
        indices = generator.integers(0, len(time), size=len(time))
        metrics = evaluate_metrics(time[indices], event[indices], risk[indices], matrix[indices],
                                   reference_time, reference_event, horizons, censoring)
        for key, value in metrics.items():
            if np.isfinite(value):
                draws[key].append(float(value))
    output = {}
    for key, values in draws.items():
        output[key] = {
            "point": float(point[key]) if np.isfinite(point[key]) else None,
            "lower_95": float(np.quantile(values, 0.025)) if values else None,
            "upper_95": float(np.quantile(values, 0.975)) if values else None,
            "valid_resamples": len(values),
            "invalid_resamples": samples - len(values),
        }
    counts = {f"{horizon:g}_year": {"deaths": int(np.sum(event & (time <= horizon))),
                                    "controls": int(np.sum(time > horizon)),
                                    "early_censored": int(np.sum((~event) & (time <= horizon))),
                                    "reference_support": bool(horizon <= np.max(reference_time)),
                                    "reference_censoring_survival": float(km_predict(censoring, horizon)),
                                    "observed_followup_support": bool(horizon <= np.max(time) or km_curve(time, event)[1][-1] == 0),
                                    "auc_available": bool(np.isfinite(point[f"{horizon:g}_year_auc"])),
                                    "brier_available": bool(np.isfinite(point[f"{horizon:g}_year_brier"])),
                                    "calibration_available": bool(np.isfinite(point[f"{horizon:g}_year_calibration_error"]))}
              for horizon in horizons}
    return {"subjects": len(time), "events": int(event.sum()), "horizon_counts": counts, "bootstrap_samples": samples,
            "bootstrap_seed": seed, "metrics": output}
