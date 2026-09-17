"""Connect externally supplied segmentation models to validated, reproducible probability-stack exports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


BRANCHES = ("baseline", "curriculum")
CLASS_ORDER = ("background", "myocardium", "scar")
REGISTRY_FIELDS = {
    "fold", "branch", "architecture", "source_identifier", "source_revision",
    "checkpoint_path", "checkpoint_sha256", "adapter", "adapter_options",
}
IMAGE_FIELDS = {
    "patient_id", "imaging_fold", "split", "anatomy_valid", "image_path",
    "geometry_id", "slices", "height", "width",
}
MANIFEST_FIELDS = (
    "patient_id", "imaging_fold", "split", "anatomy_valid", "baseline_path",
    "curriculum_path", "baseline_model_sha256", "curriculum_model_sha256",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value, base):
    value = Path(value)
    return value if value.is_absolute() else Path(base) / value


def concrete_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must identify a concrete external model or implementation.")
    text = value.strip()
    if text.lower() in {"todo", "tbd", "none", "unknown", "placeholder", "your_model", "your_checkpoint"} or "<" in text or ">" in text:
        raise ValueError(f"{name} contains a placeholder; supply the real external value.")
    return text


def load_adapter(specification):
    if not isinstance(specification, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", specification):
        raise ValueError("adapter must be an importable module:function identifier.")
    module, name = specification.split(":")
    adapter = getattr(importlib.import_module(module), name)
    if not callable(adapter):
        raise ValueError(f"Adapter is not callable: {specification}.")
    return adapter


def validate_registry(path, check_adapters=True):
    path = Path(path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1 or not isinstance(registry.get("models"), list):
        raise ValueError("Registry requires schema_version=1 and a models list.")
    records = {}
    for model in registry["models"]:
        if not isinstance(model, dict) or not REGISTRY_FIELDS.issubset(model):
            raise ValueError(f"Each model requires fields: {sorted(REGISTRY_FIELDS)}.")
        fold = model["fold"]
        if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5) or model["branch"] not in BRANCHES:
            raise ValueError("Model fold must be an integer from 0 to 4 and branch must be baseline or curriculum.")
        identity = (fold, model["branch"])
        if identity in records:
            raise ValueError("Duplicate fold/branch registry record.")
        record = dict(model)
        for name in ("architecture", "source_identifier", "source_revision", "checkpoint_path", "adapter"):
            record[name] = concrete_text(record[name], name)
        if not isinstance(record["adapter_options"], dict):
            raise ValueError("adapter_options must be a JSON object.")
        expected = str(record["checkpoint_sha256"])
        if not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ValueError("checkpoint_sha256 must contain 64 lowercase hexadecimal characters.")
        checkpoint = resolve_path(record["checkpoint_path"], path.parent).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"External segmentation checkpoint is unavailable: {checkpoint}.")
        if sha256_file(checkpoint) != expected:
            raise ValueError(f"Checkpoint SHA256 mismatch for fold {fold}, branch {model['branch']}.")
        record["checkpoint_path"] = str(checkpoint)
        if check_adapters:
            load_adapter(record["adapter"])
        records[identity] = record
    if set(records) != {(fold, branch) for fold in range(5) for branch in BRANCHES}:
        raise ValueError("Registry must contain exactly five folds times two segmentation branches.")
    return records


def positive_integer(value, field):
    text = str(value)
    if not re.fullmatch(r"[1-9][0-9]*", text):
        raise ValueError(f"{field} must be a positive integer.")
    return int(text)


def build_export_plan(image_manifest, imaging_contract, records):
    from extract_features import _check_manifest

    image_manifest = Path(image_manifest)
    frame = pd.read_csv(image_manifest, dtype=str, keep_default_na=False)
    if not IMAGE_FIELDS.issubset(frame):
        raise ValueError(f"Image manifest requires fields: {sorted(IMAGE_FIELDS)}.")
    plans, manifest = [], []
    for index, row in enumerate(frame.to_dict("records")):
        if not re.fullmatch(r"[0-4]", row["imaging_fold"]):
            raise ValueError("imaging_fold must be an integer from 0 to 4.")
        fold = int(row["imaging_fold"])
        if row["anatomy_valid"].lower() not in {"true", "false", "1", "0"}:
            raise ValueError("anatomy_valid must be true/false or 1/0.")
        image = resolve_path(row["image_path"], image_manifest.parent).resolve()
        if not image.exists():
            raise FileNotFoundError(f"External image input is unavailable: {image}.")
        geometry_id = concrete_text(row["geometry_id"], "geometry_id")
        shape = (positive_integer(row["slices"], "slices"), 3,
                 positive_integer(row["height"], "height"), positive_integer(row["width"], "width"))
        metadata = {key: row[key] for key in ("patient_id", "split", "anatomy_valid")}
        metadata.update(imaging_fold=fold, geometry_id=geometry_id, expected_shape=shape, class_order=CLASS_ORDER)
        output = {key: row[key] for key in ("patient_id", "split", "anatomy_valid")}
        output["imaging_fold"] = fold
        for branch in BRANCHES:
            output[f"{branch}_path"] = f"probabilities/{index:06d}_{branch}.npz"
            output[f"{branch}_model_sha256"] = records[(fold, branch)]["checkpoint_sha256"]
        manifest.append(output)
        plans.append({"image_path": image, "metadata": metadata, "manifest_row": output})
    contract = json.loads(Path(imaging_contract).read_text(encoding="utf-8"))
    _check_manifest(pd.DataFrame(manifest, columns=MANIFEST_FIELDS), contract)
    return plans, manifest


def validate_adapter_output(result, metadata):
    if not isinstance(result, dict) or not {"probabilities", "geometry_id", "class_order"}.issubset(result):
        raise ValueError("Adapter must return probabilities, geometry_id, and class_order.")
    if result["geometry_id"] != metadata["geometry_id"]:
        raise ValueError("Adapter output geometry differs from the declared ordered input image space.")
    if tuple(result["class_order"]) != CLASS_ORDER:
        raise ValueError("Adapter channels must be ordered background, myocardium, scar.")
    probabilities = np.asarray(result["probabilities"], dtype=np.float32)
    if probabilities.shape != tuple(metadata["expected_shape"]):
        raise ValueError("Adapter output shape must match (slice, 3, y, x) in the declared image space.")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Adapter probabilities must be finite values from 0 to 1.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-5):
        raise ValueError("Adapter class probabilities must sum to one at every voxel.")
    return probabilities


def export_probabilities(registry_path, image_manifest, imaging_contract, output_dir):
    records = validate_registry(registry_path)
    plans, manifest = build_export_plan(image_manifest, imaging_contract, records)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "probabilities").mkdir()
    adapters = {spec: load_adapter(spec) for spec in {record["adapter"] for record in records.values()}}
    for plan in plans:
        metadata = plan["metadata"]
        for branch in BRANCHES:
            record = records[(metadata["imaging_fold"], branch)]
            result = adapters[record["adapter"]](
                image_path=plan["image_path"], checkpoint_path=Path(record["checkpoint_path"]),
                model=dict(record), metadata=dict(metadata),
            )
            probabilities = validate_adapter_output(result, metadata)
            target = output / plan["manifest_row"][f"{branch}_path"]
            np.savez_compressed(target, probabilities=probabilities)
    with (output / "probability_manifest.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    receipt = {
        "schema_version": 1,
        "registry_sha256": sha256_file(registry_path),
        "image_manifest_sha256": sha256_file(image_manifest),
        "imaging_contract_sha256": sha256_file(imaging_contract),
        "probability_manifest_sha256": sha256_file(output / "probability_manifest.csv"),
        "model_sources": [record for _, record in sorted(records.items())],
        "probability_files": {row[f"{branch}_path"]: sha256_file(output / row[f"{branch}_path"])
                              for row in manifest for branch in BRANCHES},
    }
    with (output / "segmentation_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return output / "probability_manifest.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "No architecture, weights, or preprocessing implementation is bundled. Provide an importable adapter with "
        "signature adapter(*, image_path, checkpoint_path, model, metadata). It must load the declared checkpoint, "
        "apply its own frozen preprocessing, and return a dict containing probabilities shaped (slice,3,y,x), "
        "class_order=['background','myocardium','scar'], and the unchanged geometry_id. The output must return "
        "to the common ordered image space declared in the image manifest. The registry records the actual "
        "architecture, source_identifier, source_revision, and checkpoint hash; baseline/curriculum are branch roles."
    ))
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("validate", help="Verify all ten external checkpoint identities and adapters.")
    command.add_argument("--registry", type=Path, required=True)
    command = commands.add_parser("export", help="Produce probability archives and an extraction-compatible manifest.")
    command.add_argument("--registry", type=Path, required=True)
    command.add_argument("--image-manifest", type=Path, required=True)
    command.add_argument("--imaging-contract", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        records = validate_registry(args.registry)
        print(json.dumps({"validated_models": len(records), "architecture_included": False}))
    else:
        print(export_probabilities(args.registry, args.image_manifest, args.imaging_contract, args.output_dir))


if __name__ == "__main__":
    main()
