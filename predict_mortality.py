"""Apply five frozen imaging paths and one calibrated fusion transform to unseen patients."""

import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scar_features import apply_qc
from feature_config import validate_feature_config
from survival import SCAR, clinical_frame
from train_mortality import read_csv, unique_index, write_json


def predict(model, clinical, features=None):
    if model.get("schema_version") not in (2, 3):
        raise ValueError("Unsupported model schema; refit using the current training scripts")
    weight = float(model["fusion_weight"])
    if not np.isfinite(weight) or weight not in model["fusion_pool"]:
        raise ValueError("Frozen fusion weight is outside its selection pool")
    if (weight == 0 and model["recipe"] != "clinical_only") or (weight > 0 and model["recipe"] != "late_fusion"):
        raise ValueError("Frozen prediction recipe is inconsistent with its fusion weight")
    clinical = unique_index(clinical, "Inference clinical table")
    if set(clinical.index) & set(model["development_patient_ids"]):
        raise ValueError("Inference patients overlap mortality development")
    for column in ("event", "time", "death_date", "index_mri_date", "labels_path"):
        if (features is not None and column in features) or column in clinical:
            raise ValueError(f"Outcome and manual-reference fields are forbidden at inference: {column}")
    if weight > 0:
        if features is None:
            raise ValueError("This model requires five frozen imaging-path feature rows per patient")
        if set(features.patient_id) != set(clinical.index):
            raise ValueError("Clinical and imaging patient sets must match exactly")
        if features.duplicated(["patient_id", "imaging_fold"]).any():
            raise ValueError("Duplicate patient/imaging-path rows")
        if not set(features.split).issubset({"holdout", "new_patient"}):
            raise ValueError("Inference features must be holdout or new_patient rows")
        validate_feature_config(features, expected=model)
    clinical_x = clinical_frame(clinical)
    results = []
    for patient_id in clinical.index:
        output = {"patient_id": patient_id, "prediction_status": "ok", "abstention_reason": "",
                  "review_flag": False, "review_reasons": "", "fusion_weight": weight,
                  "prediction_mode": model["recipe"]}
        zc = float(model["clinical"].standardized(clinical_x.loc[[patient_id]])[0])
        zs, reasons, q = np.nan, [], zc
        if weight > 0:
            rows = features.loc[features.patient_id == patient_id].sort_values("imaging_fold")
            fold_values = pd.to_numeric(rows.imaging_fold, errors="raise")
            if len(rows) != 5 or set(fold_values) != set(model["imaging_paths"]):
                raise ValueError("Each patient requires the same five frozen imaging paths")
            for row in rows.to_dict("records"):
                expected = model["imaging_paths"][int(row["imaging_fold"])]
                for column, value in expected.items():
                    if str(row[column]).lower() != value:
                        raise ValueError(f"Imaging model identity changed for {column}")
            invalid = rows.prediction_status.ne("ok")
            if invalid.any():
                output.update(prediction_status="abstained", abstention_reason="required_imaging_path_invalid",
                              review_flag=True, review_reasons="abstention_invalid_container")
                results.append(output)
                continue
            values = rows.loc[:, SCAR].to_numpy(dtype=float)
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                output.update(prediction_status="abstained", abstention_reason="invalid_scar_phenotypes",
                              review_flag=True, review_reasons="abstention_invalid_container")
                results.append(output)
                continue
            reviews = [apply_qc(row, model["qc_thresholds"]) for row in rows.to_dict("records")]
            reasons = sorted({reason for review in reviews for reason in review["review_reasons"].split(";") if reason})
            zs = float(model["scar"].standardized(rows).mean())
            q = zc + weight * zs
        survival = model["calibration"].survival(np.array([q]), model["horizons"])[0]
        baseline = model["clinical_calibration"].survival(np.array([zc]), model["horizons"])[0]
        output.update(clinical_standardized_score=zc, scar_standardized_score=zs, fused_score=q,
                      risk=model["calibration"].gamma * q,
                      clinical_risk=model["clinical_calibration"].gamma * zc,
                      review_flag=bool(reasons), review_reasons=";".join(reasons))
        for horizon, value, control in zip(model["horizons"], survival, baseline):
            output[f"survival_{int(horizon)}y"] = float(value)
            output[f"death_probability_{int(horizon)}y"] = float(1 - value)
            output[f"clinical_survival_{int(horizon)}y"] = float(control)
        results.append(output)
    columns = ["patient_id", "prediction_status", "abstention_reason", "review_flag", "review_reasons", "fusion_weight", "prediction_mode",
               "clinical_standardized_score", "scar_standardized_score", "fused_score", "risk", "clinical_risk"]
    for horizon in model["horizons"]:
        columns.extend([f"survival_{int(horizon)}y", f"death_probability_{int(horizon)}y", f"clinical_survival_{int(horizon)}y"])
    return pd.DataFrame(results).reindex(columns=columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trusted model artifact produced by train_mortality.py")
    parser.add_argument("--clinical", required=True, help="Patient IDs and 11 baseline covariates; no outcomes")
    parser.add_argument("--features", help="Required for a nonzero fusion weight: exactly five frozen-path rows per unseen patient")
    parser.add_argument("--output", required=True, help="New prediction CSV; no outcomes are accepted")
    args = parser.parse_args()
    output = Path(args.output)
    receipt = output.with_suffix(".receipt.json")
    if output.exists() or receipt.exists():
        raise FileExistsError("Prediction output and receipt must be new files")
    model = joblib.load(args.model)
    result = predict(model, read_csv(args.clinical), read_csv(args.features) if args.features else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    write_json(receipt, {"model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                         "predictions_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                         "patients": len(result), "predicted": int(result.prediction_status.eq("ok").sum()),
                         "horizons_years": model["horizons"], "fusion_weight": model["fusion_weight"],
                         "recipe": model["recipe"]})


if __name__ == "__main__":
    main()
