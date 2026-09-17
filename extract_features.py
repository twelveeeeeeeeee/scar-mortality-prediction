"""Provide command-line probability phenotyping, development reliability, and frozen review QC."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from feature_config import (
    FeatureConfig, ReliabilityConfig, QCConfig, load_config,
    feature_config_metadata, validate_feature_config, feature_table_fingerprint,
)

from scar_features import (
    ADMITTED_FEATURES, CANDIDATE_FEATURES, QC_FEATURES, AbstentionError,
    apply_qc, extract_patient_features, extract_reference_features,
    fit_qc_thresholds, reliability_screen, validate_probabilities,
)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"patient_id": str}, keep_default_na=False)


def _new_file(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)


def _resolve(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _boolean(value: object) -> bool:
    text = str(value).lower().strip()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError("anatomy_valid must be true/false or 1/0.")


def _load_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise ValueError(f"Archive lacks required key {key}: {path}.")
        return np.asarray(archive[key])


def _check_manifest(frame: pd.DataFrame, contract: dict) -> None:
    required = {"patient_id", "imaging_fold", "split", "anatomy_valid", "baseline_path", "curriculum_path",
                "baseline_model_sha256", "curriculum_model_sha256"}
    if not required.issubset(frame):
        raise ValueError(f"Manifest requires columns: {sorted(required)}.")
    if frame.empty or (frame.patient_id.str.strip() == "").any():
        raise ValueError("Manifest cannot contain empty patient IDs or zero rows.")
    fold_values = pd.to_numeric(frame.imaging_fold, errors="coerce")
    if not fold_values.isin(range(5)).all():
        raise ValueError("imaging_fold must contain integer identifiers from 0 through 4.")
    frame = frame.copy()
    frame["imaging_fold"] = fold_values.astype(int)
    if frame.duplicated(["patient_id", "imaging_fold"]).any():
        raise ValueError("Duplicate patient/fold probability stacks.")
    for name in ("baseline_model_sha256", "curriculum_model_sha256"):
        if not frame[name].map(lambda value: bool(re.fullmatch(r"[a-f0-9]{64}", str(value)))).all():
            raise ValueError(f"{name} must contain lowercase SHA256 model identities.")
        if (frame.groupby("imaging_fold")[name].nunique() != 1).any():
            raise ValueError("Each imaging fold must use one frozen model per branch.")
    if any(type(item["fold"]) is not int or item["fold"] not in range(5) for item in contract["folds"]):
        raise ValueError("Contract fold identifiers must be integers from 0 through 4.")
    fold_map = {item["fold"]: item for item in contract["folds"]}
    if len(contract["folds"]) != 5 or set(fold_map) != set(range(5)):
        raise ValueError("The imaging contract must describe each of five folds exactly once.")
    training_sets = {fold: set(map(str, item["training_patient_ids"])) for fold, item in fold_map.items()}
    validation_sets = {fold: set(map(str, item["validation_patient_ids"])) for fold, item in fold_map.items()}
    holdout = set(map(str, contract.get("holdout_patient_ids", [])))
    all_development = set().union(*validation_sets.values())
    if sum(map(len, validation_sets.values())) != len(all_development):
        raise ValueError("Imaging validation patients must belong to exactly one fold.")
    if holdout & all_development:
        raise ValueError("Holdout and development patient sets overlap.")
    for fold in fold_map:
        if not validation_sets[fold] or len(validation_sets[fold]) != len(fold_map[fold]["validation_patient_ids"]):
            raise ValueError("Each imaging validation fold must be non-empty and contain unique patients.")
        if len(training_sets[fold]) != len(fold_map[fold]["training_patient_ids"]):
            raise ValueError("Imaging training lists must contain unique patients.")
        if training_sets[fold] & (validation_sets[fold] | holdout):
            raise ValueError("Imaging training includes an unseen or holdout patient.")
        if training_sets[fold] != all_development - validation_sets[fold]:
            raise ValueError("Each imaging training fold must equal the other four development folds.")
    for row in frame.itertuples():
        fold = int(row.imaging_fold)
        if fold not in fold_map:
            raise ValueError("Unknown imaging fold.")
        if row.split == "development_oof":
            if row.patient_id not in validation_sets[fold]:
                raise ValueError("Development probabilities are not assigned to the patient's unseen fold.")
        elif row.split in {"holdout", "new_patient"}:
            if row.patient_id in all_development:
                raise ValueError("Inference patient overlaps imaging development.")
            if row.split == "holdout" and row.patient_id not in holdout:
                raise ValueError("Holdout patient is absent from the frozen contract.")
        else:
            raise ValueError("split must be development_oof, holdout, or new_patient.")
    development = frame.loc[frame.split == "development_oof"]
    if development.patient_id.duplicated().any():
        raise ValueError("Development patients need exactly one OOF probability stack.")
    for _, group in frame.loc[frame.split != "development_oof"].groupby("patient_id"):
        if group.split.nunique() != 1 or group.anatomy_valid.map(_boolean).nunique() != 1:
            raise ValueError("Each inference patient must have consistent split and anatomy validity across folds.")
        if set(group.imaging_fold.astype(int)) != set(fold_map):
            raise ValueError("Inference patients require probability stacks from all five imaging folds.")


def extract(args: argparse.Namespace) -> None:
    config = load_config(FeatureConfig, getattr(args, "feature_config", None))
    metadata = feature_config_metadata(config)
    frame = _read_csv(args.manifest)
    contract = json.loads(args.imaging_contract.read_text(encoding="utf-8"))
    _check_manifest(frame, contract)
    _new_file(args.output)
    rows = []
    for row in frame.itertuples():
        output = {"patient_id": row.patient_id, "imaging_fold": int(row.imaging_fold), "split": row.split,
                  "prediction_status": "ok", "abstention_reason": "",
                  "baseline_model_sha256": row.baseline_model_sha256,
                  "curriculum_model_sha256": row.curriculum_model_sha256, **metadata}
        try:
            baseline = _load_array(_resolve(row.baseline_path, args.manifest.parent), "probabilities")
            curriculum = _load_array(_resolve(row.curriculum_path, args.manifest.parent), "probabilities")
            output.update(extract_patient_features(baseline, curriculum, _boolean(row.anatomy_valid), config))
        except AbstentionError as error:
            output.update(dict.fromkeys((*CANDIDATE_FEATURES, *QC_FEATURES), np.nan))
            output.update(prediction_status="abstained", abstention_reason=str(error))
        rows.append(output)
    pd.DataFrame(rows).to_csv(args.output, index=False)


def reliability(args: argparse.Namespace) -> None:
    automatic, reference_manifest = _read_csv(args.automatic), _read_csv(args.reference_manifest)
    feature_config = validate_feature_config(automatic)
    config = load_config(ReliabilityConfig, getattr(args, "reliability_config", None))
    if "split" not in automatic or set(automatic.split) != {"development_oof"}:
        raise ValueError("Reliability accepts development OOF rows only.")
    if set(reference_manifest.patient_id) != set(automatic.patient_id):
        raise ValueError("Reference patients must match the development OOF table exactly.")
    if reference_manifest.patient_id.duplicated().any():
        raise ValueError("Reference manifest must contain one row per patient.")
    reference_rows = []
    for row in reference_manifest.itertuples():
        labels = _load_array(_resolve(row.labels_path, args.reference_manifest.parent), "labels")
        reference_rows.append({"patient_id": row.patient_id, **extract_reference_features(labels, _boolean(row.anatomy_valid), feature_config)})
    results = reliability_screen(automatic, pd.DataFrame(reference_rows), args.bootstraps, args.seed, config)
    for name, value in feature_config_metadata(feature_config).items():
        results[name] = value
    results["development_features_sha256"] = feature_table_fingerprint(automatic, ADMITTED_FEATURES)
    results["reliability_config_json"] = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    _new_file(args.output)
    results.to_csv(args.output, index=False)
    admitted = results.loc[results.admitted, "feature"].tolist()
    print(json.dumps({"admitted_features": admitted, "core_features": list(ADMITTED_FEATURES),
                      "matches_core_feature_set": set(admitted) == set(ADMITTED_FEATURES)}))


def qc_fit(args: argparse.Namespace) -> None:
    development = _read_csv(args.development)
    validate_feature_config(development)
    thresholds = fit_qc_thresholds(development, load_config(QCConfig, getattr(args, "qc_config", None)))
    _new_file(args.output)
    args.output.write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")


def qc_apply(args: argparse.Namespace) -> None:
    frame = _read_csv(args.features)
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    if not {"feature_config_json", "feature_config_sha256"}.issubset(thresholds):
        raise ValueError("QC thresholds must include extraction configuration identity.")
    validate_feature_config(frame, thresholds)
    reviews = [apply_qc(row, thresholds, row["prediction_status"]) for row in frame.to_dict("records")]
    _new_file(args.output)
    pd.concat([frame, pd.DataFrame(reviews)], axis=1).to_csv(args.output, index=False)


def calibrate_threshold(args: argparse.Namespace) -> None:
    config = load_config(FeatureConfig, getattr(args, "feature_config", None))
    candidates = sorted(set(float(value) for value in args.candidates.split(",")))
    if not candidates:
        raise ValueError("At least one candidate threshold is required.")
    for value in candidates:
        replace(config, scar_threshold=value)
    frame, references = _read_csv(args.manifest), _read_csv(args.reference_manifest)
    contract = json.loads(args.imaging_contract.read_text(encoding="utf-8"))
    _check_manifest(frame, contract)
    if set(frame.split) != {"development_oof"}:
        raise ValueError("Threshold calibration accepts development OOF probability stacks only.")
    if not {"patient_id", "labels_path", "anatomy_valid"}.issubset(references):
        raise ValueError("Reference manifest requires patient_id, labels_path, and anatomy_valid.")
    if references.patient_id.duplicated().any() or set(references.patient_id) != set(frame.patient_id):
        raise ValueError("Reference patients must match calibration development patients exactly.")
    references = references.set_index("patient_id")
    scores = {value: [] for value in candidates}
    for row in frame.itertuples():
        reference = references.loc[row.patient_id]
        if not _boolean(row.anatomy_valid) or not _boolean(reference.anatomy_valid):
            raise ValueError("Resolve invalid anatomy before calibrating a mask threshold.")
        baseline = validate_probabilities(_load_array(_resolve(row.baseline_path, args.manifest.parent), "probabilities"))
        curriculum = validate_probabilities(_load_array(_resolve(row.curriculum_path, args.manifest.parent), "probabilities"))
        labels = _load_array(_resolve(reference.labels_path, args.reference_manifest.parent), "labels")
        extract_reference_features(labels, config=config)
        if baseline.shape != curriculum.shape or labels.shape != (baseline.shape[0], *baseline.shape[2:]):
            raise ValueError("Calibration probabilities and manual labels must have identical ordered image geometry.")
        probability = config.baseline_weight * baseline[:, 2] + (1 - config.baseline_weight) * curriculum[:, 2]
        target = labels == 2
        for value in candidates:
            mask = probability >= value
            denominator = int(mask.sum() + target.sum())
            scores[value].append(float(2 * np.logical_and(mask, target).sum() / denominator) if denominator else 1.0)
    means = {value: float(np.mean(score)) for value, score in scores.items()}
    selected = min(candidates, key=lambda value: (-means[value], abs(value - config.scar_threshold), value))
    calibrated = replace(config, scar_threshold=selected)
    _new_file(args.output)
    args.output.write_text(json.dumps(asdict(calibrated), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_scar_threshold": selected, "mean_patient_dice": means,
                      "patients": len(frame), "selection_source": "development_oof_manual_labels",
                      **feature_config_metadata(calibrated)}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scar phenotyping and outcome-free measurement QC.")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("extract", help="Extract all 15 candidates from paired probability stacks.")
    command.add_argument("--manifest", type=Path, required=True,
                         help="CSV: patient_id, imaging_fold, split, anatomy_valid, baseline_path, curriculum_path, baseline_model_sha256, curriculum_model_sha256. Paths point to NPZ archives with probabilities shaped (slice,3,y,x), channels background/myocardium/scar, in identical ordered image space. Model hashes identify the upstream fitted models used for these probabilities.")
    command.add_argument("--imaging-contract", type=Path, required=True,
                         help="JSON: folds list containing fold, training_patient_ids, validation_patient_ids; holdout_patient_ids list. Exactly five disjoint folds are required.")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--feature-config", type=Path,
                         help="JSON overrides for FeatureConfig; actual threshold, extent, quantile, and top-slice definitions and their identity are recorded in every output row.")
    command.set_defaults(function=extract)
    command = commands.add_parser("reliability", help="Apply ICC, rank, and fold gates using development references only.")
    command.add_argument("--automatic", type=Path, required=True)
    command.add_argument("--reference-manifest", type=Path, required=True,
                         help="CSV: patient_id, labels_path, anatomy_valid; NPZ labels have shape (slice,y,x), values 0/1/2 for background/myocardium/scar.")
    command.add_argument("--bootstraps", type=int, default=2000)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--reliability-config", type=Path, required=True,
                         help="JSON with explicit minimum_icc, minimum_spearman, minimum_fold_spearman, absolute_mae_floor, and median_mae_multiplier.")
    command.set_defaults(function=reliability)
    command = commands.add_parser("qc-fit", help="Estimate and freeze configured quantile thresholds from development OOF features.")
    command.add_argument("--development", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--qc-config", type=Path, required=True,
                         help="JSON with explicit upper_quantile and lower_quantile review policies.")
    command.set_defaults(function=qc_fit)
    command = commands.add_parser("qc-apply", help="Add review flags without changing phenotype values or mortality scores.")
    command.add_argument("--features", type=Path, required=True)
    command.add_argument("--thresholds", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=qc_apply)
    command = commands.add_parser("calibrate-threshold", help="Select a scar mask threshold using development OOF manual segmentation labels only.")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--imaging-contract", type=Path, required=True)
    command.add_argument("--reference-manifest", type=Path, required=True)
    command.add_argument("--candidates", required=True, help="Comma-separated candidate probability cutoffs in (0,1]. Equal mean patient Dice prefers the current cutoff, then the lower cutoff.")
    command.add_argument("--feature-config", type=Path)
    command.add_argument("--output", type=Path, required=True, help="Selected FeatureConfig JSON for subsequent extraction, reliability, and training.")
    command.set_defaults(function=calibrate_threshold)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
