"""Select, calibrate, and freeze mortality models using development-only nested validation."""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from survival import (HORIZONS, SCAR, CalibratedRisk,
                      clinical_frame, stratified_splits,
                      tune_component, validate_outcomes)
from nested_fusion import inner_fusion_metrics
from selection import compare_candidates, select_lambda
from feature_config import QCConfig, feature_table_fingerprint, load_config, validate_feature_config
from model_config import CoxConfig, SelectionConfig
from dataclasses import asdict


def read_csv(path):
    return pd.read_csv(path, dtype={"patient_id": str}, keep_default_na=False, na_values=[""])


def unique_index(frame, label):
    if "patient_id" not in frame or frame.patient_id.isna().any() or frame.patient_id.duplicated().any():
        raise ValueError(f"{label} requires unique nonmissing patient_id")
    return frame.set_index("patient_id", drop=False)


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), indent=2, allow_nan=False), encoding="utf-8")


def load_development(clinical_path, outcome_path, partition_path, feature_paths, reliability_paths=None):
    partition = unique_index(read_csv(partition_path), "Partitions")
    if not set(partition.role).issubset({"development", "holdout"}):
        raise ValueError("Partition role must be development or holdout")
    selected = partition[partition.role == "development"]
    assigned_folds = pd.to_numeric(selected.imaging_fold, errors="raise")
    if selected.empty or assigned_folds.isna().any() or not np.equal(assigned_folds, assigned_folds.astype(int)).all() or set(assigned_folds.astype(int)) != set(range(5)):
        raise ValueError("Development needs all five fixed imaging folds, numbered 0 through 4")
    clinical = unique_index(read_csv(clinical_path), "Clinical table")
    outcomes = unique_index(read_csv(outcome_path), "Outcomes")
    ids = selected.index
    missing = set(ids) - set(clinical.index.intersection(outcomes.index))
    if missing:
        raise ValueError("Development clinical data or outcomes are incomplete")
    c = clinical_frame(clinical.loc[ids]).reset_index(drop=True)
    time, event = validate_outcomes(outcomes.loc[ids, "time"], outcomes.loc[ids, "event"])
    conditions, qc_frames, path_contracts = {}, {}, {}
    for name, path in feature_paths.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Condition names may contain only letters, digits, underscores, and hyphens")
        features = unique_index(read_csv(path), name)
        if set(features.index) != set(ids):
            raise ValueError(f"{name}: supply exactly one OOF row per development patient")
        features = features.loc[ids]
        if "split" not in features or not features["split"].eq("development_oof").all():
            raise ValueError(f"{name}: nondevelopment features are forbidden during fitting")
        if not np.array_equal(pd.to_numeric(features.imaging_fold, errors="raise"), assigned_folds):
            raise ValueError(f"{name}: feature rows do not match fixed imaging-fold allocation")
        if not features.prediction_status.eq("ok").all():
            raise ValueError(f"{name}: invalid anatomical containers must be resolved before fitting")
        if not np.isfinite(features.loc[:, SCAR].to_numpy(dtype=float)).all():
            raise ValueError(f"{name}: admitted scar inputs are incomplete")
        if ((features.loc[:, SCAR] < 0) | (features.loc[:, SCAR] > 1)).any().any():
            raise ValueError(f"{name}: scar features must be fractions")
        config = validate_feature_config(features)
        if reliability_paths is not None:
            if name not in reliability_paths:
                raise ValueError(f"{name}: reliability report is required")
            report = read_csv(reliability_paths[name])
            validate_feature_config(report, expected=config)
            expected_features = feature_table_fingerprint(features, SCAR)
            if "development_features_sha256" not in report or not report.development_features_sha256.eq(expected_features).all():
                raise ValueError(f"{name}: reliability report belongs to different development feature rows")
            if report.feature.duplicated().any():
                raise ValueError(f"{name}: duplicate reliability feature rows")
            report = report.set_index("feature")
            for feature in SCAR:
                if feature not in report.index or str(report.loc[feature, "admitted"]).strip().lower() not in {"true", "1"}:
                    raise ValueError(f"{name}: required scar feature has not passed reliability admission: {feature}")
        contract = {}
        for fold, rows in features.groupby("imaging_fold"):
            contract[int(fold)] = {}
            for column in ("baseline_model_sha256", "curriculum_model_sha256"):
                values = rows[column].dropna().unique()
                if rows[column].isna().any() or len(values) != 1 or len(str(values[0])) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in str(values[0])):
                    raise ValueError(f"{name}: one valid {column} is required per imaging path")
                contract[int(fold)][column] = str(values[0]).lower()
        frame = c.copy()
        for feature in SCAR:
            frame[feature] = features[feature].to_numpy(dtype=float)
        conditions[name] = frame
        qc_frames[name] = features.reset_index(drop=True)
        path_contracts[name] = contract
    return ids.to_numpy(), c, time, event, selected.imaging_fold.astype(int).to_numpy(), conditions, qc_frames, path_contracts


def run_development(clinical, conditions, time, event, seed=42, repeats=5, outer_folds=5, inner_folds=4,
                    cox_config=CoxConfig(), selection_config=None):
    if not isinstance(selection_config, SelectionConfig):
        raise ValueError("An explicit SelectionConfig is required before development fitting.")
    if not isinstance(cox_config, CoxConfig):
        raise ValueError("Cox settings must be a validated CoxConfig.")
    if any(type(value) is not int or value < minimum for value, minimum in
           ((repeats, 1), (outer_folds, 2), (inner_folds, 2))):
        raise ValueError("Repeats must be positive and cross-validation fold counts must be at least two.")
    n = len(time)
    predictions = {"clinical": {"risk": np.zeros(n), "survival": np.zeros((n, 3))}}
    components = {"clinical": np.zeros(n)}
    for name in conditions:
        components[name] = np.zeros(n)
        for method in ("early", "late"):
            predictions[f"{name}:{method}"] = {"risk": np.zeros(n), "survival": np.zeros((n, 3))}
    selections = []
    for repeat in range(repeats):
        outer = stratified_splits(event, outer_folds, seed + repeat)
        for fold, (train, valid) in enumerate(outer):
            inner = stratified_splits(event[train], inner_folds, seed + 1000 * repeat + fold + 100)
            c_model, c_oof, c_history = tune_component(clinical.iloc[train], time[train], event[train], "clinical", inner, config=cox_config)
            zc_train = c_model.standardized(clinical.iloc[train])
            zc_valid = c_model.standardized(clinical.iloc[valid])
            c_cal = CalibratedRisk.fit(c_oof, zc_train, time[train], event[train])
            components["clinical"][valid] += zc_valid
            predictions["clinical"]["risk"][valid] += c_cal.gamma * zc_valid
            predictions["clinical"]["survival"][valid] += c_cal.survival(zc_valid)
            for name, frame in conditions.items():
                s_model, s_oof, s_history = tune_component(frame.iloc[train], time[train], event[train], "scar", inner, config=cox_config)
                zs_train, zs_valid = s_model.standardized(frame.iloc[train]), s_model.standardized(frame.iloc[valid])
                components[name][valid] += zs_valid
                lambda_metrics = inner_fusion_metrics(clinical.iloc[train], frame.iloc[train],
                                                     time[train], event[train], inner,
                                                     seed + repeat * 1000 + fold,
                                                     tuning_folds=inner_folds, cox_config=cox_config)
                lambda_selection = select_lambda(lambda_metrics, config=selection_config)
                selected_weight = lambda_selection["selected_lambda"]
                weight = float(selected_weight)
                q_train, q_valid = zc_train + weight * zs_train, zc_valid + weight * zs_valid
                cal = CalibratedRisk.fit(c_oof + weight * s_oof, q_train, time[train], event[train])
                target = predictions[f"{name}:late"]
                target["risk"][valid] += cal.gamma * q_valid
                target["survival"][valid] += cal.survival(q_valid)
                early, early_oof, e_history = tune_component(frame.iloc[train], time[train], event[train], "early", inner, config=cox_config)
                e_train, e_valid = early.standardized(frame.iloc[train]), early.standardized(frame.iloc[valid])
                e_cal = CalibratedRisk.fit(early_oof, e_train, time[train], event[train])
                predictions[f"{name}:early"]["risk"][valid] += e_cal.gamma * e_valid
                predictions[f"{name}:early"]["survival"][valid] += e_cal.survival(e_valid)
                selections.append({"repeat": repeat, "outer_fold": fold, "condition": name,
                                   "selected_lambda": selected_weight, "lambda_selection": lambda_selection, "inner_lambda_metrics": lambda_metrics,
                                   "clinical_tuning": c_history, "scar_tuning": s_history, "early_tuning": e_history})
            print(f"Completed repeat {repeat + 1}/{repeats}, outer fold {fold + 1}/{outer_folds}", flush=True)
    for values in predictions.values():
        values["risk"] /= repeats
        values["survival"] /= repeats
    for key in components:
        components[key] /= repeats
    return predictions, components, selections


def freeze_model(clinical, frame, time, event, c_oof, s_oof, qc_thresholds, path_contract, patient_ids, seed,
                 development_gate_passed=True, cox_config=CoxConfig(), selection_config=None, inner_folds=4):
    if not isinstance(selection_config, SelectionConfig):
        raise ValueError("An explicit SelectionConfig is required before freezing a model.")
    if not isinstance(cox_config, CoxConfig):
        raise ValueError("Cox settings must be a validated CoxConfig.")
    splits = stratified_splits(event, inner_folds, seed)
    selection_metrics = inner_fusion_metrics(clinical, frame, time, event, splits, seed,
                                            tuning_folds=inner_folds, cox_config=cox_config)
    selection = select_lambda(selection_metrics, config=selection_config)
    selected_weight = float(selection["selected_lambda"])
    weight = selected_weight if development_gate_passed else 0.0
    c_model, _, _ = tune_component(clinical, time, event, "clinical", splits, config=cox_config)
    zc = c_model.standardized(clinical)
    clinical_cal = CalibratedRisk.fit(c_oof, zc, time, event)
    s_model, fused = None, clinical_cal
    if weight > 0:
        s_model, _, _ = tune_component(frame, time, event, "scar", splits, config=cox_config)
        zs = s_model.standardized(frame)
        fused = CalibratedRisk.fit(c_oof + weight * s_oof, zc + weight * zs, time, event)
    metadata = {key: qc_thresholds[key] for key in ("feature_config_json", "feature_config_sha256")}
    if not development_gate_passed:
        reason = "development_gate_fallback"
    elif not selection.get("selection_available", True):
        reason = "unavailable_selection_metrics"
    else:
        reason = "cross_validated_selection"
    selection.update(deployed_lambda=weight, development_gate_passed=bool(development_gate_passed), deployment_reason=reason)
    return {"schema_version": 3, "recipe": "late_fusion" if weight > 0 else "clinical_only", "clinical": c_model, "scar": s_model,
            "calibration": fused, "clinical_calibration": clinical_cal, "fusion_weight": weight,
            "fusion_pool": cox_config.fusion_weights, "fusion_selection": selection,
            "cox_config": asdict(cox_config), "selection_config": asdict(selection_config), "inner_folds": inner_folds,
            "horizons": HORIZONS, "qc_thresholds": qc_thresholds if weight > 0 else None,
            "imaging_paths": path_contract if weight > 0 else {}, **metadata,
            "development_patient_ids": list(patient_ids), "reference_time": time, "reference_event": event,
            "numeric_zero_policy": "retain_source_values", "seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", required=True)
    parser.add_argument("--outcomes", required=True, help="patient_id,time (years),event")
    parser.add_argument("--partitions", required=True, help="patient_id,role,imaging_fold")
    parser.add_argument("--features", action="append", required=True, metavar="CONDITION=CSV",
                        help="Repeat for each externally supplied development OOF imaging condition")
    parser.add_argument("--reliability", action="append", required=True, metavar="CONDITION=CSV",
                        help="Matching development reliability report with all three required scar features admitted")
    parser.add_argument("--freeze-condition", help="Optional preselected imaging condition; otherwise choose the highest-ranked passing late-fusion condition")
    parser.add_argument("--qc-config", type=Path, required=True, help="Required JSON quantile policy for estimating development review thresholds")
    parser.add_argument("--selection-config", type=Path, required=True, help="Required JSON with prespecified development gate and ranking policies")
    parser.add_argument("--cox-config", type=Path, help="Optional JSON overrides for generic Cox grids, category grouping and fusion candidates")
    parser.add_argument("--output", required=True, help="New runtime output directory; must not exist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()
    cox_config = load_config(CoxConfig, args.cox_config)
    selection_config = load_config(SelectionConfig, args.selection_config)
    qc_config = load_config(QCConfig, args.qc_config)
    if args.repeats < 1 or args.outer_folds < 2 or args.inner_folds < 2 or args.bootstraps < 1:
        parser.error("Repeats/bootstrap counts must be positive and fold counts must be at least two.")
    feature_paths = dict(item.split("=", 1) for item in args.features)
    reliability_paths = dict(item.split("=", 1) for item in args.reliability)
    if len(feature_paths) != len(args.features) or len(reliability_paths) != len(args.reliability):
        parser.error("Duplicate condition names are not allowed")
    if set(feature_paths) != set(reliability_paths):
        parser.error("Feature and reliability conditions must match exactly")
    if args.freeze_condition is not None and args.freeze_condition not in feature_paths:
        parser.error("Freeze condition must have a development feature table")
    output = Path(args.output)
    ids, clinical, time, event, folds, conditions, qc, paths = load_development(
        args.clinical, args.outcomes, args.partitions, feature_paths, reliability_paths)
    output.mkdir(parents=True, exist_ok=False)
    predictions, components, selections = run_development(
        clinical, conditions, time, event, args.seed, args.repeats, args.outer_folds, args.inner_folds,
        cox_config=cox_config, selection_config=selection_config)
    baseline = predictions["clinical"]
    comparisons = compare_candidates(time, event, baseline["risk"], baseline["survival"],
                                     {key: value for key, value in predictions.items() if key != "clinical"},
                                     folds, time, event, samples=args.bootstraps, seed=args.seed, config=selection_config)
    write_json(output / "development_comparisons.json", comparisons)
    write_json(output / "nested_selections.json", selections)
    passing_late = [name.split(":")[0] for name in comparisons["ranked_passing_candidates"] if name.endswith(":late")]
    condition = args.freeze_condition or (passing_late[0] if passing_late else sorted(conditions)[0])
    gate = comparisons["comparisons"][condition + ":late"]
    from scar_features import fit_qc_thresholds
    frozen = freeze_model(clinical, conditions[condition], time, event, components["clinical"],
                          components[condition], fit_qc_thresholds(qc[condition], qc_config),
                          paths[condition], ids, args.seed, development_gate_passed=gate["advance_to_holdout"],
                          cox_config=cox_config, selection_config=selection_config, inner_folds=args.inner_folds)
    frozen["validation_config"] = {"repeats": args.repeats, "outer_folds": args.outer_folds,
                                   "inner_folds": args.inner_folds, "bootstraps": args.bootstraps, "seed": args.seed}
    frozen["condition"] = condition if frozen["fusion_weight"] > 0 else None
    frozen["development_source_condition"] = condition
    frozen["development_gate_passed"] = gate["advance_to_holdout"]
    joblib.dump(frozen, output / "mortality_model.joblib")
    write_json(output / "fusion_selection.json", frozen["fusion_selection"])
    for key, values in predictions.items():
        table = pd.DataFrame({"patient_id": ids, "risk": values["risk"]})
        for index, horizon in enumerate(HORIZONS):
            table[f"survival_{int(horizon)}y"] = values["survival"][:, index]
        table.to_csv(output / (key.replace(":", "_") + "_oof.csv"), index=False)
    frozen["clinical"].coefficients().to_csv(output / "clinical_coefficients.csv", index=False)
    if frozen["scar"] is not None:
        frozen["scar"].coefficients().to_csv(output / "scar_coefficients.csv", index=False)
    print(json.dumps({"recipe": frozen["recipe"], "fusion_weight": frozen["fusion_weight"],
                      "condition": frozen["condition"], "development_gate_passed": frozen["development_gate_passed"]}))


if __name__ == "__main__":
    main()
