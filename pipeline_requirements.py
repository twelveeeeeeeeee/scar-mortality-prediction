"""Describe mortality workflow prerequisites and validate available inputs before execution."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path


DEPENDENCIES = ("numpy", "pandas", "scipy", "scikit-learn", "scikit-survival", "joblib")


def describe():
    return {
        "runtime": {
            "python": ">=3.12",
            "installation": "python install_dependencies.py",
            "mortality_dependencies": list(DEPENDENCIES),
            "segmentation_dependencies": "Provided by the external adapter; no segmentation framework is bundled.",
            "execution": "Use the same dependency versions for model fitting and model loading.",
        },
        "segmentation_boundary": {
            "architecture_included": False,
            "checkpoints_included": False,
            "branch_roles": ["baseline", "curriculum"],
            "source_identity": "A branch role does not identify an architecture or source implementation.",
            "existing_probabilities_route": "Paired precomputed NPZ stacks can be used directly without segmentation checkpoints or adapters.",
            "raw_images_route": "Supply ten real model records, frozen checkpoints, preprocessing, an importable adapter, and image geometry metadata.",
            "registry": {
                "schema_version": 1,
                "models": "Exactly one record for each fold 0-4 and each baseline/curriculum branch.",
                "record_fields": ["fold", "branch", "architecture", "source_identifier", "source_revision",
                                  "checkpoint_path", "checkpoint_sha256", "adapter", "adapter_options"],
                "adapter": "module:function, called with keyword arguments image_path, checkpoint_path, model, metadata",
                "adapter_return": "dict: probabilities=(slice,3,y,x), geometry_id matching input, class_order=[background,myocardium,scar]",
            },
        },
        "input_schemas": {
            "clinical": "One row per patient: patient_id, age, lvef, egfr, sex, cardiac_surgery, cied, va, hx_va, pci, respiratory_history, diabetes.",
            "outcomes": "One row per patient: patient_id, time in positive years, event=0/1 for all-cause death. Outcomes are not used for segmentation, threshold calibration, or prediction.",
            "clinical_preparation": "Alternatively supply index_mri_date, death_date and the clinical fields to prepare_cohort.py. --administrative-cutoff is required; follow-up ascertainment must support the supplied date. Optional expected cohort/fold counts are explicit inputs.",
            "missing_numeric_values": "Blank numeric fields are imputed inside training folds. Source-coded numeric zeros are currently preserved; confirmed missing LVEF/eGFR zeros must be converted to blank in a separately versioned external input before preparation.",
            "partitions": "patient_id, role=development|holdout, imaging_fold=0-4 for development and empty for holdout. Patient allocation is externally supplied.",
            "imaging_contract": "JSON folds: [{fold, training_patient_ids, validation_patient_ids}], holdout_patient_ids. Validation sets are disjoint; each training set is exactly the other four development folds and excludes holdout.",
            "probability_manifest": "patient_id, imaging_fold, split=development_oof|holdout|new_patient, anatomy_valid, baseline_path, curriculum_path, baseline_model_sha256, curriculum_model_sha256.",
            "probability_npz": "Key probabilities: finite array (slice,3,y,x), channels background/myocardium/scar, class probabilities sum to one, both branches use identical ordered image coordinates.",
            "image_manifest": "patient_id, imaging_fold, split, anatomy_valid, image_path, geometry_id, slices, height, width. Used only by segmentation_interface.py.",
            "reference_manifest": "patient_id, labels_path, anatomy_valid; each NPZ has labels=(slice,y,x), integers 0/1/2. Development references only for threshold calibration and reliability.",
            "scar_features": "Development: one OOF row per patient from the fold that excluded that patient. Scar-enabled inference: five rows per patient from the same frozen imaging paths. Configuration and checkpoint identities must match model training.",
            "reliability_results": "One CSV per imaging condition from extract_features.py reliability, with feature/configuration identities and all three required scar features admitted. Training requires --reliability CONDITION=CSV for each --features CONDITION=CSV.",
            "feature_configuration": "Optional FeatureConfig JSON sets scar_threshold, baseline_weight, slice_extent_threshold, near_threshold_margin, upper_tail_quantile, and top_slice_count. Defaults use equal two-expert weights and a 0.5 mask cutoff. Feature tables carry canonical configuration JSON and SHA256.",
            "study_policies": "ReliabilityConfig, QCConfig and SelectionConfig have required fields with no study-derived defaults. Supply complete JSON policies before running reliability, QC or training. Config-template prints null for fields that must be filled; null is not a usable policy value.",
            "review_configuration": "ReviewConfig uses six equal weights, a 0.5 mask cutoff and unit sample weights. The review-priority quantile has no default and must be explicitly supplied through --review-config. Local quality fitting uses a generic 100-tree ExtraTrees model with minimum leaf size 1 and max_features 1.0.",
            "cox_configuration": "Optional CoxConfig JSON controls logarithmically spaced penalty candidates, l1 ratios, categorical grouping and a uniform fusion candidate grid. Category-frequency grouping is disabled by default. The selected settings and actual policies are stored in the frozen model.",
            "retained_definitions": "The 5% slice-burden extent, upper-tail feature quantile, top-three-slice summary and spatial grid remain configurable phenotype definitions. Five imaging paths, three class channels, 1/3/5-year horizons, time conversion and numerical tolerances retain their mathematical or protocol roles.",
        },
        "workflow": [
            "Install dependencies in a dedicated environment and externally assemble patient-level clinical/outcome tables plus fixed partitions.",
            "Provide imaging training membership and derive the imaging contract, or supply an independently verified contract.",
            "Use paired precomputed probabilities, or validate the external segmentation registry and export probabilities with segmentation_interface.py.",
            "Optionally calibrate the scar probability threshold using development OOF probabilities and development reference masks; freeze the resulting feature configuration.",
            "Extract development OOF scar features using that configuration. Use development reference masks to assess feature reliability; freeze development QC thresholds.",
            "Run train_mortality.py with development features, matching reliability tables, and explicit selection/QC policies. Cox hyperparameters and the fusion weight are selected using development resampling only. The optional freeze-condition restricts candidate choice; without it the best eligible imaging condition is selected.",
            "Extract five-path holdout or new-patient features with the same feature configuration and frozen segmentation models.",
            "Run predict_mortality.py using the frozen mortality model and baseline clinical inputs. A positive frozen fusion weight also requires five-path scar features; a zero fusion weight uses clinical inputs alone. Save the generated prediction receipt before outcome linkage.",
            "For evaluation, run evaluate_mortality.py with the frozen predictions, receipt, matching held-out outcomes, and development censoring reference in the model artifact.",
        ],
        "execution_conditions": [
            "Every image, mask, table, trained model, and checkpoint is an external runtime input; this source repository contains no patient data or trained models.",
            "Scar-enabled prediction requires valid anatomical containers; invalid cases abstain. Clinical-only prediction does not require imaging. Training currently requires all development cases to have valid required scar features.",
            "All-cause deaths and censored subjects must both occur in sufficient numbers in every nested training/validation stratum. Full resampling also needs comparable event pairs and convergent Cox fits; constant-score calibration is supported.",
            "Censoring-aware evaluation at 1, 3, and 5 years requires adequate follow-up and censoring-distribution support; unsupported estimates must be treated as unavailable.",
            "Reliability estimation and threshold calibration require manual development masks. Prediction from an already frozen model does not require manual masks or outcome fields.",
            "Threshold-selection Dice and subsequent reliability measured on the same development references are selection diagnostics, not independent performance estimates. Freeze the threshold before evaluating untouched subjects.",
            "The generic fusion candidate pool is uniformly spaced from zero to one and can be replaced through CoxConfig. A clinical-only model is frozen if no scar candidate passes the explicitly supplied development gate or the selected weight is zero. Meaningful full-pool selection requires supported Brier estimates through five years; unavailable gate metrics trigger the explicit clinical-only fallback.",
            "Generic defaults are starting settings, not validated operating points. Changing configuration requires new feature extraction, reliability assessment and model fitting; use only development data for tuning and freeze choices before holdout evaluation.",
            "A completed preflight checks the declared inputs only; it does not prove checkpoint provenance, segmentation accuracy, Cox convergence, or clinical performance.",
            "All output paths must be new, with enough disk space for probability archives and runtime models. Keep patient-level runtime artifacts outside the source repository.",
        ],
        "commands": {
            "describe": "python pipeline_requirements.py describe",
            "environment": "python pipeline_requirements.py preflight --stage environment",
            "configuration_template": "python pipeline_requirements.py config-template --kind selection",
            "preparation": "python prepare_cohort.py --input cohort.csv --manifest fixed_partitions.csv --imaging-training-membership training_membership.csv --administrative-cutoff YYYY-MM-DD --output-dir prepared",
            "registry": "python segmentation_interface.py validate --registry registry.json",
            "segmentation": "python segmentation_interface.py export --registry registry.json --image-manifest images.csv --imaging-contract imaging_contract.json --output-dir probability_export",
            "probability_preflight": "python pipeline_requirements.py preflight --stage probabilities --probability-manifest probability_manifest.csv --imaging-contract imaging_contract.json",
            "threshold_calibration": "python extract_features.py calibrate-threshold --manifest development_probability_manifest.csv --imaging-contract imaging_contract.json --reference-manifest reference_manifest.csv --candidates 0.1,0.3,0.5,0.7,0.9 --output feature_config.json",
            "extract": "python extract_features.py extract --manifest probability_manifest.csv --imaging-contract imaging_contract.json --feature-config feature_config.json --output scar_features.csv",
            "reliability": "python extract_features.py reliability --automatic development_features.csv --reference-manifest reference_manifest.csv --reliability-config reliability_policy.json --output reliability.csv",
            "training": "python train_mortality.py --clinical clinical.csv --outcomes outcomes.csv --partitions partitions.csv --features CONDITION=development_features.csv --reliability CONDITION=reliability.csv --selection-config selection_policy.json --qc-config qc_policy.json --output trained_model",
            "prediction": "python predict_mortality.py --model trained_model/mortality_model.joblib --clinical unseen_clinical.csv --features unseen_features.csv --output predictions.csv",
            "clinical_only_prediction": "python predict_mortality.py --model trained_model/mortality_model.joblib --clinical unseen_clinical.csv --output predictions.csv",
            "evaluation": "python evaluate_mortality.py --model trained_model/mortality_model.joblib --predictions predictions.csv --receipt predictions.receipt.json --outcomes unseen_outcomes.csv --output evaluation.json",
        },
    }


def configuration_template(kind):
    from dataclasses import MISSING, fields
    from feature_config import FeatureConfig, ReliabilityConfig, QCConfig, ReviewConfig
    from model_config import CoxConfig, SelectionConfig
    schemas = {"feature": FeatureConfig, "reliability": ReliabilityConfig, "qc": QCConfig,
               "review": ReviewConfig, "cox": CoxConfig, "selection": SelectionConfig}
    if kind not in schemas:
        raise ValueError("Unknown configuration kind.")
    output = {}
    for field in fields(schemas[kind]):
        if field.default is not MISSING:
            output[field.name] = field.default
        elif field.default_factory is not MISSING:
            output[field.name] = field.default_factory()
        else:
            output[field.name] = None
    return output


def environment_status():
    packages, errors = {}, []
    if sys.version_info < (3, 12):
        errors.append("Python 3.12 or newer is required.")
    for package in DEPENDENCIES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
            errors.append(f"Missing package: {package}.")
    return {"python": sys.version.split()[0], "packages": packages}, errors


def required(args, *names):
    for name in names:
        if not getattr(args, name, None):
            raise ValueError(f"--{name.replace('_', '-')} is required for stage {args.stage}.")


def check_probabilities(manifest_path, contract_path):
    import numpy as np
    import pandas as pd
    from extract_features import _check_manifest
    from segmentation_interface import CLASS_ORDER, resolve_path, validate_adapter_output

    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path, dtype={"patient_id": str}, keep_default_na=False)
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    _check_manifest(frame, contract)
    for row in frame.to_dict("records"):
        shape = None
        for branch in ("baseline", "curriculum"):
            path = resolve_path(row[f"{branch}_path"], manifest_path.parent)
            with np.load(path, allow_pickle=False) as archive:
                if "probabilities" not in archive:
                    raise ValueError(f"Probability key absent in {path}.")
                values = np.asarray(archive["probabilities"])
            if values.ndim != 4 or values.shape[1] != 3 or any(size == 0 for size in values.shape):
                raise ValueError("Probability arrays require nonempty (slice,3,y,x) dimensions.")
            if shape is None:
                shape = values.shape
            if values.shape != shape:
                raise ValueError("Paired branch probability shapes do not match.")
            validate_adapter_output({"probabilities": values, "geometry_id": "precomputed", "class_order": CLASS_ORDER},
                                    {"expected_shape": shape, "geometry_id": "precomputed"})
    return {"validated_probability_rows": len(frame), "patients": frame.patient_id.nunique(),
            "geometry_note": "Matching shapes were checked; common ordered image coordinates must be established upstream."}


def preflight(args):
    environment, errors = environment_status()
    result = {"stage": args.stage, "environment": environment, "checks": {}, "errors": errors}
    if not errors:
        try:
            if args.stage == "segmentation":
                required(args, "registry", "image_manifest", "imaging_contract")
                from segmentation_interface import build_export_plan, validate_registry
                models = validate_registry(args.registry)
                plans, _ = build_export_plan(args.image_manifest, args.imaging_contract, models)
                result["checks"] = {"validated_models": len(models), "image_rows": len(plans)}
            elif args.stage == "probabilities":
                required(args, "probability_manifest", "imaging_contract")
                result["checks"] = check_probabilities(args.probability_manifest, args.imaging_contract)
            elif args.stage == "training":
                required(args, "clinical", "outcomes", "partitions", "features", "reliability", "selection_config", "qc_config")
                from train_mortality import load_development
                from survival import stratified_splits
                from feature_config import load_config, QCConfig
                from model_config import CoxConfig, SelectionConfig
                load_config(SelectionConfig, args.selection_config)
                load_config(QCConfig, args.qc_config)
                cox_config = load_config(CoxConfig, args.cox_config)
                if args.repeats < 1 or args.outer_folds < 2 or args.inner_folds < 2:
                    raise ValueError("Repeats must be positive and cross-validation counts must be at least two.")
                features = {}
                for entry in args.features:
                    if "=" not in entry:
                        raise ValueError("--features requires CONDITION=CSV.")
                    name, path = entry.split("=", 1)
                    if not name or not path or name in features:
                        raise ValueError("Feature conditions must be nonempty and unique.")
                    features[name] = path
                reliability = {}
                for entry in args.reliability:
                    if "=" not in entry:
                        raise ValueError("--reliability requires CONDITION=CSV.")
                    name, path = entry.split("=", 1)
                    if not name or not path or name in reliability:
                        raise ValueError("Reliability conditions must be nonempty and unique.")
                    reliability[name] = path
                    if not Path(path).is_file():
                        raise FileNotFoundError(f"Reliability table is unavailable: {path}.")
                if set(reliability) != set(features):
                    raise ValueError("Reliability and feature conditions must match exactly.")
                loaded = load_development(args.clinical, args.outcomes, args.partitions, features, reliability)
                ids, _, _, event = loaded[:4]
                for repeat in range(args.repeats):
                    for outer_fold, (outer_train, _) in enumerate(stratified_splits(event, args.outer_folds, args.seed + repeat)):
                        inner = stratified_splits(event[outer_train], args.inner_folds, args.seed + 1000 * repeat + outer_fold + 100)
                        for inner_fold, (inner_train, _) in enumerate(inner):
                            stratified_splits(event[outer_train][inner_train], args.inner_folds,
                                              args.seed + repeat * 1000 + outer_fold + 10000 + inner_fold)
                for fold, (train, _) in enumerate(stratified_splits(event, args.inner_folds, args.seed)):
                    stratified_splits(event[train], args.inner_folds, args.seed + 10000 + fold)
                result["checks"] = {"development_patients": len(ids), "deaths": int(event.sum()),
                                    "conditions": sorted(features), "nested_event_strata_checked": True,
                                    "explicit_policies_validated": True, "fusion_candidates": cox_config.fusion_weights}
        except (ValueError, OSError, KeyError, ImportError, AttributeError) as error:
            result["errors"].append(str(error))
    result["preflight_passed"] = not result["errors"]
    result["scope"] = "Available inputs and event strata only; model training and performance are not validated by preflight."
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("describe", help="Print runtime inputs, schemas, model boundaries, and workflow commands.")
    template = commands.add_parser("config-template", help="Print a JSON template; fill required null policy fields before use.")
    template.add_argument("--kind", required=True, choices=("feature", "reliability", "qc", "review", "cox", "selection"))
    command = commands.add_parser("preflight", help="Check installed dependencies and inputs without fitting a model.")
    command.add_argument("--stage", choices=("environment", "segmentation", "probabilities", "training"), default="environment")
    for name in ("registry", "image-manifest", "imaging-contract", "probability-manifest", "clinical", "outcomes", "partitions", "selection-config", "qc-config", "cox-config"):
        command.add_argument(f"--{name}", type=Path)
    command.add_argument("--features", action="append", metavar="CONDITION=CSV")
    command.add_argument("--reliability", action="append", metavar="CONDITION=CSV")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--repeats", type=int, default=5)
    command.add_argument("--outer-folds", type=int, default=5)
    command.add_argument("--inner-folds", type=int, default=4)
    args = parser.parse_args(argv)
    result = (describe() if args.command == "describe" else configuration_template(args.kind)
              if args.command == "config-template" else preflight(args))
    print(json.dumps(result, indent=2))
    return 0 if args.command != "preflight" or result["preflight_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
