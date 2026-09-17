"""Evaluate fusion weights with all component fitting and calibration confined to inner training folds."""

import numpy as np

from model_config import CoxConfig
from survival import (HORIZONS, CalibratedRisk,
                      stratified_splits, tune_component, validate_outcomes)
from survival_metrics import evaluate_metrics, harrell_concordance


def inner_fusion_metrics(clinical, scar_frame, time, event, splits, seed,
                         tuning_folds=4, fusion_weights=None, cox_config=CoxConfig()):
    if not isinstance(cox_config, CoxConfig):
        raise TypeError("Nested fusion requires a CoxConfig")
    fusion_weights = cox_config.fusion_weights if fusion_weights is None else fusion_weights
    time, event = validate_outcomes(time, event)
    count = len(time)
    if len(clinical) != count or len(scar_frame) != count:
        raise ValueError("Clinical, scar and outcome rows must agree")
    fusion_weights = tuple(float(weight) for weight in fusion_weights)
    if (not fusion_weights or len(set(fusion_weights)) != len(fusion_weights)
            or not np.isfinite(fusion_weights).all() or min(fusion_weights) < 0 or 0.0 not in fusion_weights):
        raise ValueError("Fusion pool must contain unique finite nonnegative weights including zero")
    coverage = np.zeros(count, dtype=int)
    normalized_splits = []
    for train, valid in splits:
        train = np.asarray(train, dtype=int)
        valid = np.asarray(valid, dtype=int)
        if (train.ndim != 1 or valid.ndim != 1 or not len(train) or not len(valid)
                or np.any(train < 0) or np.any(train >= count)
                or np.any(valid < 0) or np.any(valid >= count)
                or len(np.unique(train)) != len(train)
                or len(np.unique(valid)) != len(valid)
                or np.intersect1d(train, valid).size
                or len(train) + len(valid) != count):
            raise ValueError("Each inner split must be a disjoint complete partition")
        coverage[valid] += 1
        normalized_splits.append((train, valid))
    if not np.all(coverage == 1):
        raise ValueError("Each patient must receive exactly one inner held-out prediction")
    risks = {weight: np.full(count, np.nan) for weight in fusion_weights}
    briers = {weight: {f"{horizon:g}_year_brier": 0.0 for horizon in HORIZONS}
              for weight in fusion_weights}
    for fold, (train, valid) in enumerate(normalized_splits):
        subfolds = stratified_splits(event[train], tuning_folds, seed + 10000 + fold)
        clinical_train = clinical.iloc[train]
        scar_train = scar_frame.iloc[train]
        c_model, c_oof, _ = tune_component(clinical_train, time[train], event[train],
                                           "clinical", subfolds, config=cox_config)
        s_model, s_oof, _ = tune_component(scar_train, time[train], event[train],
                                           "scar", subfolds, config=cox_config)
        zc_train = c_model.standardized(clinical_train)
        zs_train = s_model.standardized(scar_train)
        zc_valid = c_model.standardized(clinical.iloc[valid])
        zs_valid = s_model.standardized(scar_frame.iloc[valid])
        for weight in fusion_weights:
            calibration = CalibratedRisk.fit(c_oof + weight * s_oof,
                                              zc_train + weight * zs_train,
                                              time[train], event[train])
            q_valid = zc_valid + weight * zs_valid
            eta_valid = calibration.gamma * q_valid
            survival = calibration.survival(q_valid)
            metrics = evaluate_metrics(time[valid], event[valid], eta_valid, survival,
                                        time[train], event[train], HORIZONS)
            risks[weight][valid] = eta_valid
            for key in briers[weight]:
                briers[weight][key] += len(valid) * metrics[key] / count
    return {weight: {"harrell_c_index": harrell_concordance(time, event, risks[weight])[0],
                     **briers[weight]}
            for weight in fusion_weights}
