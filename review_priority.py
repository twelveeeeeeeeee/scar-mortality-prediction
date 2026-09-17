"""Cross-fit a labelled-set local Dice regressor and derive patient review priority."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

from scar_features import validate_probabilities
from feature_config import ReviewConfig, load_config


EXPERTS = ("baseline", "curriculum", "residual_encoder", "two_point_five_d", "scar_focused", "long_schedule")


def _dice(first: np.ndarray, second: np.ndarray, smoothing: float = 1.0) -> float:
    intersection = np.logical_and(first, second).sum()
    denominator = first.sum() + second.sum()
    return float((2 * intersection + smoothing) / (denominator + smoothing)) if denominator + smoothing > 0 else 1.0


def build_quality_rows(expert_probability: np.ndarray, baseline: np.ndarray,
                       curriculum: np.ndarray, labels: np.ndarray,
                       config: ReviewConfig = ReviewConfig()) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    baseline, curriculum = validate_probabilities(baseline), validate_probabilities(curriculum)
    probabilities = np.asarray(expert_probability, dtype=float)
    labels = np.asarray(labels)
    if baseline.shape != curriculum.shape or probabilities.shape != (baseline.shape[0], 6, *baseline.shape[2:]):
        raise ValueError("Expert scar probabilities must have shape (slice,6,y,x) aligned with full baseline/curriculum arrays.")
    if not np.isfinite(probabilities).all() or probabilities.min() < 0 or probabilities.max() > 1:
        raise ValueError("Expert scar probabilities must be finite and in [0,1].")
    if not np.allclose(probabilities[:, 0], baseline[:, 2], atol=1e-7, rtol=1e-6) or not np.allclose(probabilities[:, 1], curriculum[:, 2], atol=1e-7, rtol=1e-6):
        raise ValueError("The first two expert channels must match baseline and curriculum scar probabilities.")
    if labels.shape != (baseline.shape[0], *baseline.shape[2:]) or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Reference labels must be aligned integer background/myocardium/scar labels.")
    if min(labels.shape[1:]) < config.grid_size:
        raise ValueError("Each image dimension must contain at least grid_size pixels.")
    myocardium = np.argmax(config.baseline_weight * baseline + (1 - config.baseline_weight) * curriculum, axis=1) == 1
    consensus = np.average(probabilities, axis=1, weights=np.asarray(config.consensus_weights))
    consensus_scar = consensus >= config.measurement_threshold
    rows, targets, sample_weights = [], [], []
    row_edges = np.linspace(0, labels.shape[1], config.grid_size + 1, dtype=int)
    col_edges = np.linspace(0, labels.shape[2], config.grid_size + 1, dtype=int)
    for z in range(len(labels)):
        scar_probability = probabilities[z]
        masks = scar_probability >= config.measurement_threshold
        container = myocardium[z] | consensus_scar[z]
        count = max(1, int(container.sum()))
        mean_probability, std_probability = scar_probability.mean(axis=0), scar_probability.std(axis=0)
        vote = masks.mean(axis=0)
        clipped = np.clip(vote, 1e-7, 1 - 1e-7)
        entropy = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped)) / np.log(2)
        slice_burden = float(consensus_scar[z].sum() / count)
        slice_disagreement = float(((vote > 0) & (vote < 1) & container).sum() / count)
        for grid_row in range(config.grid_size):
            for grid_column in range(config.grid_size):
                selection = np.s_[row_edges[grid_row]:row_edges[grid_row + 1], col_edges[grid_column]:col_edges[grid_column + 1]]
                reference = labels[z][selection] == 2
                cell_mean = mean_probability[selection]
                for expert in range(6):
                    probability, predicted = scar_probability[expert][selection], masks[expert][selection]
                    identity = np.eye(6)[expert].tolist()
                    rows.append([
                        *identity, grid_row / (config.grid_size - 1), grid_column / (config.grid_size - 1),
                        z / (len(labels) - 1) if len(labels) > 1 else 0.5,
                        myocardium[z][selection].mean(), container[selection].mean(),
                        probability.mean(), probability.std(), np.quantile(probability, config.upper_tail_quantile), probability.max(),
                        predicted.mean(), (np.abs(probability - config.measurement_threshold) <= config.near_threshold_margin).mean(),
                        cell_mean.mean(), std_probability[selection].mean(), np.quantile(cell_mean, config.upper_tail_quantile),
                        vote[selection].mean(), entropy[selection].mean(), np.abs(probability - cell_mean).mean(),
                        (predicted == (cell_mean >= config.measurement_threshold)).mean(),
                        (masks[expert] & container).sum() / count, slice_burden, slice_disagreement,
                    ])
                    targets.append(_dice(predicted, reference, config.dice_smoothing))
                    if not reference.any():
                        weight = config.empty_correct_weight if not predicted.any() else config.empty_false_positive_weight
                    else:
                        weight = 1.0 + min(reference.sum() / max(1.0, reference.size / config.positive_area_scale), config.positive_weight_maximum)
                    sample_weights.append(weight)
    return (np.asarray(rows, dtype=float), np.asarray(targets), np.asarray(sample_weights),
            1.0 - _dice(consensus_scar, labels == 2, smoothing=0.0))


def crossfit_review(patients: list[dict], seed: int = 42, n_jobs: int = 1,
                    config: ReviewConfig = ReviewConfig()) -> pd.DataFrame:
    if config.priority_quantile is None:
        raise ValueError("Set priority_quantile explicitly before fitting review priority.")
    if len({patient["patient_id"] for patient in patients}) != len(patients):
        raise ValueError("Each labelled patient must occur exactly once.")
    folds = sorted({patient["quality_fold"] for patient in patients})
    if folds != list(range(5)):
        raise ValueError("The local quality experiment requires five patient-level folds.")
    outputs = []
    for fold in folds:
        train = [patient for patient in patients if patient["quality_fold"] != fold]
        test = [patient for patient in patients if patient["quality_fold"] == fold]
        model = ExtraTreesRegressor(n_estimators=config.n_estimators, min_samples_leaf=config.min_samples_leaf, max_features=config.max_features,
                                    random_state=seed, n_jobs=n_jobs)
        model.fit(np.concatenate([patient["x"] for patient in train]),
                  np.concatenate([patient["target"] for patient in train]),
                  sample_weight=np.concatenate([patient["weight"] for patient in train]))
        for patient in test:
            quality = model.predict(patient["x"])
            outputs.append({"patient_id": patient["patient_id"], "quality_fold": fold,
                            "review_priority": float(np.quantile(1.0 - quality, config.priority_quantile)),
                            "dice_error": patient["dice_error"]})
    return pd.DataFrame(outputs)


def _load(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        return archive[key]


def main() -> None:
    parser = argparse.ArgumentParser(description="Labelled-set review priority; independent of mortality prediction and holdout QC.")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="CSV: patient_id, quality_fold, expert_path, baseline_path, curriculum_path, labels_path. Expert NPZ scar_probabilities=(slice,6,y,x), ordered baseline/curriculum/residual_encoder/2.5D/scar_focused/long_schedule; full NPZ probabilities=(slice,3,y,x), labels NPZ labels=(slice,y,x).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--review-config", type=Path, required=True,
                        help="JSON with explicit priority_quantile; optional overrides for measurement threshold, expert weights, local grid, and regressor.")
    args = parser.parse_args()
    config = load_config(ReviewConfig, args.review_config)
    if config.priority_quantile is None:
        raise ValueError("The review configuration must define priority_quantile explicitly.")
    if args.bootstraps < 0:
        raise ValueError("Bootstrap count must be non-negative.")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}.")
    frame = pd.read_csv(args.manifest, dtype={"patient_id": str})
    required = {"patient_id", "quality_fold", "expert_path", "baseline_path", "curriculum_path", "labels_path"}
    if not required.issubset(frame) or frame.empty or frame.patient_id.isna().any():
        raise ValueError(f"Manifest requires non-empty columns {sorted(required)}.")
    if not pd.to_numeric(frame.quality_fold, errors="coerce").isin(range(5)).all():
        raise ValueError("quality_fold must contain integer identifiers from 0 through 4.")
    if "split" in frame and not frame.split.eq("development_oof").all():
        raise ValueError("Review-quality fitting accepts development OOF labels only.")
    patients = []
    for row in frame.itertuples():
        paths = {}
        for name in ("expert_path", "baseline_path", "curriculum_path", "labels_path"):
            value = Path(getattr(row, name))
            paths[name] = value if value.is_absolute() else args.manifest.parent / value
        x, target, weight, error = build_quality_rows(
            _load(paths["expert_path"], "scar_probabilities"), _load(paths["baseline_path"], "probabilities"),
            _load(paths["curriculum_path"], "probabilities"), _load(paths["labels_path"], "labels"), config)
        patients.append({"patient_id": row.patient_id, "quality_fold": int(row.quality_fold),
                         "x": x, "target": target, "weight": weight, "dice_error": error})
    results = crossfit_review(patients, args.seed, args.n_jobs, config)
    results["review_config_json"] = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    x, y = results.review_priority.to_numpy(), results.dice_error.to_numpy()
    rho = float(spearmanr(x, y).statistic) if np.ptp(x) and np.ptp(y) else None
    rng, samples = np.random.default_rng(args.seed), []
    for _ in range(args.bootstraps):
        indices = rng.integers(len(x), size=len(x))
        if np.ptp(x[indices]) and np.ptp(y[indices]):
            samples.append(float(spearmanr(x[indices], y[indices]).statistic))
    interval = np.quantile(samples, [0.025, 0.975]).tolist() if samples else [None, None]
    print(json.dumps({"spearman": rho, "bootstrap_95_interval": interval, "valid_bootstraps": len(samples)}))


if __name__ == "__main__":
    main()
