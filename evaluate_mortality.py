"""Join frozen predictions to unseen outcomes and estimate censoring-aware holdout performance."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from survival_metrics import fixed_prediction_bootstrap, validate_outcomes
from train_mortality import read_csv, unique_index, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trusted frozen training artifact supplies development censoring data")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--receipt", required=True, help="SHA256 receipt created before outcome linkage")
    parser.add_argument("--outcomes", required=True, help="patient_id,time (years),event")
    parser.add_argument("--output", required=True, help="New evaluation JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstraps < 1:
        parser.error("--bootstraps must be positive.")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Evaluation output must be new")
    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    for path, key in ((args.model, "model_sha256"), (args.predictions, "predictions_sha256")):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != receipt[key]:
            raise ValueError("Frozen prediction receipt does not match the supplied artifacts")
    model = joblib.load(args.model)
    predictions = unique_index(read_csv(args.predictions), "Predictions")
    outcomes = unique_index(read_csv(args.outcomes), "Evaluation outcomes")
    if set(predictions.index) != set(outcomes.index):
        raise ValueError("Outcome and prediction patient sets must match exactly")
    if set(predictions.index) & set(model["development_patient_ids"]):
        raise ValueError("Evaluation patients overlap model development")
    valid = predictions.prediction_status.eq("ok")
    selected = predictions.loc[valid]
    if selected.empty:
        raise ValueError("No evaluable mortality predictions; all patients abstained")
    target = outcomes.loc[selected.index]
    time, event = validate_outcomes(target.time, target.event)
    results = {"patients": len(predictions), "predicted": len(selected), "abstained": int((~valid).sum()),
               "coverage": float(valid.mean()), "endpoint": "all_cause_mortality", "time_unit": "years",
               "evaluation_scope": "all_patients" if valid.all() else "predicted_subset_with_abstentions_reported",
               "model_sha256": receipt["model_sha256"], "recipe": model["recipe"],
               "fusion_weight": model["fusion_weight"]}
    for name, risk_column, prefix in (("selected_model", "risk", ""), ("clinical_only", "clinical_risk", "clinical_")):
        survival = np.column_stack([selected[f"{prefix}survival_{int(horizon)}y"] for horizon in model["horizons"]])
        results[name] = fixed_prediction_bootstrap(time, event, selected[risk_column].to_numpy(), survival,
                                                   model["reference_time"], model["reference_event"],
                                                   horizons=model["horizons"], samples=args.bootstraps, seed=args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, results)


if __name__ == "__main__":
    main()
