# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Assign the frozen subject split to inspected CSV metadata; never open media."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from .inspect_caucafall import ACTIVITY_LABELS, DEFAULT_ROOT, subject_key, validate_output
except ImportError:
    from inspect_caucafall import ACTIVITY_LABELS, DEFAULT_ROOT, subject_key, validate_output

DEFAULT_CONFIG = Path("configs/dataset_split.json")
DEFAULT_INVENTORY = Path("artifacts/dataset_inspection/inventory.csv")
DEFAULT_MANIFEST = Path("artifacts/dataset_inspection/split_inventory.csv")
SPLIT_SIZES = {"train": 6, "validation": 2, "test": 2}


def subject_assignments(config: dict[str, Any]) -> dict[str, str]:
    """Reject overlaps, omissions, duplicate IDs; explicit membership is authoritative."""
    if (config.get("dataset_name"), config.get("dataset_version"),
        config.get("split_strategy")) != (
            "CAUCAFall", "V5", "fixed_subject_independent"):
        raise ValueError("Expected CAUCAFall V5 fixed subject-independent split")
    assignments = {}
    for split, count in SPLIT_SIZES.items():
        subjects = config.get(split)
        if not isinstance(subjects, list) or len(subjects) != count:
            raise ValueError(f"Expected {count} subject IDs in {split}")
        for subject in subjects:
            if type(subject) is not int or not 1 <= subject <= 10:
                raise ValueError(f"Invalid subject ID: {subject}")
            name = f"Subject.{subject}"
            if name in assignments:
                raise ValueError(f"Duplicate/overlapping subject: {name}")
            assignments[name] = split
    if set(assignments) != {f"Subject.{i}" for i in range(1, 11)}:
        raise ValueError("Split must cover all ten subjects")
    return assignments


def assign_rows(rows: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, str]]:
    """Preserve original metadata, validate 100 subject/activity pairs, add split."""
    assignments = subject_assignments(config)
    seen_pairs = set()
    seen_paths = set()
    result = []
    required = {"subject", "activity", "label", "relative_video_path", "fps",
                "width", "height", "frame_count", "duration_seconds", "readable"}
    for row in rows:
        if not required.issubset(row) or None in row or any(v is None for v in row.values()):
            raise ValueError("Malformed or incomplete inventory row")
        subject, activity = row["subject"], row["activity"]
        if subject not in assignments or activity not in ACTIVITY_LABELS:
            raise ValueError(f"Unknown subject/activity: {subject}/{activity}")
        if row["label"] != ACTIVITY_LABELS[activity]:
            raise ValueError(f"Binary label mismatch: {subject}/{activity}")
        split = assignments[subject]
        if "split" in row and row["split"] != split:
            raise ValueError(f"Conflicting existing split: {subject}")
        path = PurePosixPath(row["relative_video_path"])
        if (len(path.parts) != 3 or path.parts[:2] != (subject, activity)
                or path.suffix.lower() != ".avi" or "\\" in str(path)
                or path.as_posix() != row["relative_video_path"]):
            raise ValueError(f"Path does not match subject/activity: {path}")
        pair = (subject, activity)
        if pair in seen_pairs or path in seen_paths:
            raise ValueError(f"Duplicate subject/activity or video: {path}")
        seen_pairs.add(pair)
        seen_paths.add(path)
        for key in ("fps", "width", "height", "frame_count", "duration_seconds"):
            value = float(row[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {key}: {path}")
        if row["readable"] != "True" or row.get("error", ""):
            raise ValueError(f"Video did not pass inspection: {path}")
        result.append({**row, "split": split})
    expected_pairs = {(subject, activity) for subject in assignments for activity in ACTIVITY_LABELS}
    if seen_pairs != expected_pairs:
        raise ValueError("Inventory must contain exactly one AVI for each of 100 subject/activity pairs")
    counts = Counter((row["split"], row["label"]) for row in result)
    for split, subjects in SPLIT_SIZES.items():
        if any(counts[split, label] != subjects * 5 for label in ("fall", "non_fall")):
            raise ValueError(f"Unexpected class counts in {split}")
    return sorted(result, key=lambda row: (subject_key(row["subject"]), row["activity"],
                                           row["relative_video_path"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        validate_output(DEFAULT_ROOT, args.output.parent)
        if args.output.is_symlink() or args.output.resolve() in (
                args.inventory.resolve(), args.config.resolve()):
            raise ValueError("Output must not overwrite an input or follow a symlink")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        with args.inventory.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            if not fields or len(fields) != len(set(fields)):
                raise ValueError("Missing or duplicate CSV header")
            rows = assign_rows(list(reader), config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[f for f in fields if f != "split"] + ["split"],
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for split in SPLIT_SIZES:
        selected = [row for row in rows if row["split"] == split]
        print(f"{split}: {len(selected)} videos; {dict(Counter(r['label'] for r in selected))}")
    print(f"No subject overlap; all {len(rows)} videos assigned once. Manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
