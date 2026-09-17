"""Prepare baseline clinical covariates, all-cause survival outcomes, and externally assigned patient partitions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path


NUMERIC_FIELDS = ("age", "lvef", "egfr")
CATEGORICAL_FIELDS = (
    "sex", "cardiac_surgery", "cied", "va", "hx_va", "pci",
    "respiratory_history", "diabetes",
)
CLINICAL_FIELDS = NUMERIC_FIELDS + CATEGORICAL_FIELDS
ALIASES = {
    "patient_id": ("patient_id", "study_subject_id", "StudySubjectID", "subject_id"),
    "age": ("age", "age_years"),
    "lvef": ("lvef", "ef", "ejection_fraction", "LVEF (%)"),
    "egfr": ("egfr", "estimated_glomerular_filtration_rate"),
    "sex": ("sex", "gender"),
    "cardiac_surgery": ("cardiac_surgery", "previous_cardiac_surgery", "PreviousCardiacSurgery"),
    "cied": ("cied", "cardiac_implantable_electronic_device"),
    "va": ("va", "ventricular_arrhythmia", "ventricular_arrhythmia_status"),
    "hx_va": ("hx_va", "history_of_va", "history_of_ventricular_arrhythmia"),
    "pci": ("pci", "hx_pci", "previous_pci", "previous_percutaneous_coronary_intervention"),
    "respiratory_history": ("respiratory_history", "hx_resp", "respiratory_disease"),
    "diabetes": ("diabetes", "hx_dm", "diabetes_history"),
    "index_mri_date": ("index_mri_date", "mri_date", "index_date", "date_of_mri"),
    "death_date": ("death_date", "date_of_death", "date of death", "deathdate"),
    "event": ("event", "dead_dec25", "dead Dec 25", "death_status"),
    "role": ("role", "split", "stage1_split", "partition"),
    "imaging_fold": ("imaging_fold", "image_fold", "held_out_imaging_fold"),
}
ROLE_ALIASES = {
    "development": "development", "dev": "development", "train": "development",
    "training": "development", "val": "development", "validation": "development",
    "imaging_training": "development", "labelled_validation": "development",
    "holdout": "holdout", "sealed_holdout": "holdout", "test": "holdout",
    "sealed_internal_holdout": "holdout",
}


def key(value):
    return re.sub(r"[^a-z0-9]", "", value.strip().lower())


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or any(not field.strip() for field in reader.fieldnames):
            raise ValueError("Every CSV column must have a nonempty name.")
        normalized = [key(field) for field in reader.fieldnames]
        if len(set(normalized)) != len(normalized):
            raise ValueError("CSV column names must be unique after normalization.")
        rows = list(reader)
        if not rows:
            raise ValueError("Input CSV has no patient rows.")
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError("Input contains rows with an inconsistent number of columns.")
        return reader.fieldnames, rows


def resolve_columns(fieldnames, required, optional=()):
    available = {key(field): field for field in fieldnames}
    result = {}
    for field in tuple(required) + tuple(optional):
        matches = sorted({available[key(alias)] for alias in ALIASES[field] if key(alias) in available})
        if not matches and field in required:
            raise ValueError(f"Required input column is absent: {field}")
        if len(matches) > 1:
            if field == "role" and set(matches) == {"role", "split"}:
                result[field] = "role"
                continue
            raise ValueError(f"Multiple columns map to {field}: {matches}")
        if matches:
            result[field] = matches[0]
    return result


def canonical_rows(rows, columns):
    result = []
    seen = set()
    for number, row in enumerate(rows, start=2):
        item = {field: row[column].strip() for field, column in columns.items()}
        patient = item["patient_id"]
        if not patient:
            raise ValueError(f"Empty patient identifier at CSV line {number}.")
        if patient in seen:
            raise ValueError(f"Duplicate patient identifier at CSV line {number}.")
        seen.add(patient)
        result.append(item)
    return result


def parse_date(value, date_format, field, row_number, required):
    if not value:
        if required:
            raise ValueError(f"Missing {field} at CSV line {row_number}.")
        return None
    try:
        return datetime.strptime(value, date_format).date()
    except ValueError as error:
        raise ValueError(f"Invalid {field} at CSV line {row_number}; expected {date_format}.") from error


def parse_status(value, row_number):
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "1.0", "true", "yes", "dead", "deceased"}:
        return True
    if normalized in {"0", "0.0", "false", "no", "alive", "living"}:
        return False
    raise ValueError(f"Invalid death status at CSV line {row_number}.")


def prepare_clinical_outcomes(rows, date_format, numeric_missing, administrative_cutoff):
    cutoff = datetime.strptime(administrative_cutoff, "%Y-%m-%d").date()
    clinical = []
    outcomes = []
    for number, row in enumerate(rows, start=2):
        covariates = {"patient_id": row["patient_id"]}
        for field in NUMERIC_FIELDS:
            raw = row[field]
            if raw.lower() in numeric_missing:
                covariates[field] = ""
                continue
            try:
                value = float(raw)
            except ValueError as error:
                raise ValueError(f"Invalid numeric value in {field} at CSV line {number}.") from error
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite numeric value in {field} at CSV line {number}.")
            covariates[field] = value
        for field in CATEGORICAL_FIELDS:
            covariates[field] = row[field]
        start = parse_date(row["index_mri_date"], date_format, "index_mri_date", number, True)
        death = parse_date(row["death_date"], date_format, "death_date", number, False)
        status = parse_status(row.get("event", ""), number)
        if status is True and death is None:
            raise ValueError(f"Death status lacks a death date at CSV line {number}.")
        if death is not None and death <= start:
            raise ValueError(f"Death must occur after the index MRI at CSV line {number}.")
        event = death is not None and death <= cutoff
        if status is False and event:
            raise ValueError(f"Alive status conflicts with a death by cutoff at CSV line {number}.")
        end = death if event else cutoff
        days = (end - start).days
        if days <= 0:
            raise ValueError(f"Follow-up must be positive at CSV line {number}.")
        clinical.append(covariates)
        outcomes.append({"patient_id": row["patient_id"], "time": days / 365.25, "event": int(event)})
    return clinical, outcomes


def prepare_partitions(path, patient_ids, expected_counts=None, expected_fold_size=None):
    if expected_counts is not None:
        if (not isinstance(expected_counts, dict) or set(expected_counts) != {"development", "holdout"}
                or any(type(value) is not int or value < 1 for value in expected_counts.values())):
            raise ValueError("Expected counts require positive integer development and holdout sizes.")
    if expected_fold_size is not None and (type(expected_fold_size) is not int or expected_fold_size < 1):
        raise ValueError("Expected fold size must be a positive integer.")
    fieldnames, raw = read_csv(path)
    columns = resolve_columns(fieldnames, ("patient_id", "role", "imaging_fold"))
    if "role" in fieldnames and "split" in fieldnames:
        for number, row in enumerate(raw, start=2):
            if row["role"].strip().lower() != row["split"].strip().lower():
                raise ValueError(f"Conflicting role and split at manifest line {number}.")
    rows = canonical_rows(raw, columns)
    if {row["patient_id"] for row in rows} != set(patient_ids):
        raise ValueError("Partition manifest and cohort must contain exactly the same patient identifiers.")
    result = []
    counts = {"development": 0, "holdout": 0}
    folds = set()
    for number, row in enumerate(rows, start=2):
        raw_role = row["role"].lower().replace("-", "_").replace(" ", "_")
        role = ROLE_ALIASES.get(raw_role)
        if role is None:
            raise ValueError(f"Unknown role at manifest line {number}.")
        counts[role] += 1
        raw_fold = row["imaging_fold"]
        if role == "development":
            if raw_fold not in {"0", "1", "2", "3", "4"}:
                raise ValueError(f"Development imaging_fold must be an integer from 0 to 4 at manifest line {number}.")
            fold = int(raw_fold)
            folds.add(fold)
        else:
            if raw_fold:
                raise ValueError(f"Holdout imaging_fold must be empty at manifest line {number}.")
            fold = ""
        result.append({"patient_id": row["patient_id"], "role": role, "split": role, "imaging_fold": fold})
    if folds != set(range(5)):
        raise ValueError("External development allocation must contain all five imaging folds (0 to 4).")
    if not counts["holdout"]:
        raise ValueError("The supplied study partition requires a nonempty holdout role.")
    if expected_counts is not None and counts != expected_counts:
        raise ValueError("Patient counts differ from the explicitly expected development/holdout sizes.")
    if expected_fold_size is not None:
        if any(sum(row["imaging_fold"] == fold for row in result) != expected_fold_size for fold in range(5)):
            raise ValueError("Imaging allocation differs from the explicitly expected unseen fold size.")
    return result


def validate_training_membership(path, partitions):
    fields, rows = read_csv(path)
    columns = resolve_columns(fields, ("patient_id", "imaging_fold"))
    observed = {fold: set() for fold in range(5)}
    for number, row in enumerate(rows, start=2):
        patient = row[columns["patient_id"]].strip()
        raw_fold = row[columns["imaging_fold"]].strip()
        if not patient or raw_fold not in {"0", "1", "2", "3", "4"}:
            raise ValueError(f"Invalid training membership at CSV line {number}.")
        fold = int(raw_fold)
        if patient in observed[fold]:
            raise ValueError(f"Duplicate patient-fold training membership at CSV line {number}.")
        observed[fold].add(patient)
    for fold in range(5):
        expected = {row["patient_id"] for row in partitions if row["role"] == "development" and row["imaging_fold"] != fold}
        if observed[fold] != expected:
            raise ValueError(f"Imaging path {fold} training membership must equal the other four development folds and exclude all holdout patients.")


def write_csv(path, rows, fieldnames):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Required cohort fields: patient_id, age, lvef, egfr, sex, cardiac_surgery,\n"
            "cied, va, hx_va, pci, respiratory_history, diabetes, index_mri_date, death_date.\n"
            "An optional event/death-status field is checked against dates. Blank death_date\n"
            "means no recorded death unless death status is positive. Supply an administrative\n"
            "cutoff supported by follow-up ascertainment. Time is expressed in years (365.25 days).\n"
            "Common legacy headers (StudySubjectID, EF, Gender, PreviousCardiacSurgery,\n"
            "hx_PCI, hx_resp, hx_DM) are accepted. Unknown categorical levels and numeric zeros\n"
            "are preserved. Blank numeric values are left missing for fold-local imputation.\n"
            "The external partition manifest needs patient_id, role (or split), imaging_fold.\n"
            "Development imaging folds are 0-4; holdout fold cells are empty. Training/validation\n"
            "roles map to development; test maps to holdout. No original allocation is recreated.\n"
            "Optional training membership lists each real training patient and imaging_fold;\n"
            "a patient appears in four paths, and holdout patients appear in none. Synthetic\n"
            "training instances do not belong in the mortality cohort or membership table.\n"
            "Runtime outputs are clinical.csv, outcomes.csv, partitions.csv. Supplying validated\n"
            "training membership additionally writes imaging_contract.json. Existing files\n"
            "are never overwritten. Keep these private runtime data outside the source repository."
        ),
    )
    parser.add_argument("--input", type=Path, required=True, help="External patient-level clinical and outcome CSV.")
    parser.add_argument("--manifest", type=Path, required=True, help="External fixed patient partition manifest.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New runtime directory for prepared CSV files.")
    parser.add_argument("--administrative-cutoff", required=True, help="Explicit administrative censoring date in ISO YYYY-MM-DD format.")
    parser.add_argument("--date-format", default="%Y-%m-%d", help="Exact datetime.strptime input format; default ISO YYYY-MM-DD dates.")
    parser.add_argument("--numeric-missing-token", action="append", default=[], help="Additional case-insensitive numeric missing token; repeat as needed. Zero cannot be a missing token.")
    parser.add_argument("--imaging-training-membership", type=Path, help="Optional actual real-patient training membership for each frozen imaging path.")
    parser.add_argument("--expected-development", type=int, help="Optional expected development count; requires --expected-holdout.")
    parser.add_argument("--expected-holdout", type=int, help="Optional expected holdout count; requires --expected-development.")
    parser.add_argument("--expected-fold-size", type=int, help="Optional expected unseen development count in every imaging fold.")
    args = parser.parse_args(argv)
    if (args.expected_development is None) != (args.expected_holdout is None):
        parser.error("Supply both --expected-development and --expected-holdout, or neither.")
    expected_counts = None if args.expected_development is None else {
        "development": args.expected_development, "holdout": args.expected_holdout}
    missing = {""} | {token.strip().lower() for token in args.numeric_missing_token}
    for token in missing - {""}:
        try:
            number = float(token)
        except ValueError:
            continue
        if math.isfinite(number):
            parser.error("Numeric missing tokens must not be finite numeric values; source-coded zeros must be preserved.")
    try:
        fields, raw = read_csv(args.input)
        columns = resolve_columns(fields, ("patient_id",) + CLINICAL_FIELDS + ("index_mri_date", "death_date"), ("event",))
        rows = canonical_rows(raw, columns)
        clinical, outcomes = prepare_clinical_outcomes(rows, args.date_format, missing, args.administrative_cutoff)
        partitions = prepare_partitions(args.manifest, [row["patient_id"] for row in rows], expected_counts, args.expected_fold_size)
        if args.imaging_training_membership:
            validate_training_membership(args.imaging_training_membership, partitions)
        destinations = [args.output_dir / name for name in ("clinical.csv", "outcomes.csv", "partitions.csv")]
        if args.imaging_training_membership:
            destinations.append(args.output_dir / "imaging_contract.json")
        if any(path.exists() for path in destinations):
            raise ValueError("Output files already exist; choose a new runtime output directory.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(destinations[0], clinical, ("patient_id",) + CLINICAL_FIELDS)
        write_csv(destinations[1], outcomes, ("patient_id", "time", "event"))
        write_csv(destinations[2], partitions, ("patient_id", "role", "split", "imaging_fold"))
        if args.imaging_training_membership:
            contract = {
                "folds": [
                    {
                        "fold": fold,
                        "training_patient_ids": sorted(row["patient_id"] for row in partitions if row["role"] == "development" and row["imaging_fold"] != fold),
                        "validation_patient_ids": sorted(row["patient_id"] for row in partitions if row["imaging_fold"] == fold),
                    } for fold in range(5)
                ],
                "holdout_patient_ids": sorted(row["patient_id"] for row in partitions if row["role"] == "holdout"),
            }
            with destinations[3].open("x", encoding="utf-8") as stream:
                json.dump(contract, stream, indent=2)
                stream.write("\n")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Prepared {len(clinical)} patients; cutoff {args.administrative_cutoff}; source-coded zeros preserved.")
    print("Imaging training membership validated." if args.imaging_training_membership else "Patient partitions validated; actual imaging training membership was not supplied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

