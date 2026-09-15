# /// script
# requires-python = ">=3.12"
# dependencies = ["opencv-python-headless==4.12.0.88", "numpy==2.2.6"]
# ///
"""Read-only CAUCAFall V5 structure, annotation and sequential media inspection."""

import argparse
import csv
import json
import math
import platform
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("data/raw/caucafall_v5/CAUCAFall")
DEFAULT_OUTPUT = Path("artifacts/dataset_inspection")
ACTIVITY_LABELS = {
    "Fall backwards": "fall", "Fall forward": "fall", "Fall left": "fall",
    "Fall right": "fall", "Fall sitting": "fall", "Hop": "non_fall",
    "Kneel": "non_fall", "Pick up object": "non_fall", "Sit down": "non_fall",
    "Walk": "non_fall",
}
FIELDS = ["subject", "activity", "label", "relative_video_path", "fps", "width",
          "height", "frame_count", "duration_seconds", "opened", "decoded_first_frame",
          "readable", "decoded_frame_count", "reached_end_normally",
          "decode_failure_before_expected_end", "frame_count_matches", "error"]


def subject_key(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"Subject\.(\d+)", name)
    return (int(match[1]) if match else 10**9, name)


def activity_label(activity: str) -> str:
    """Reject unknown source activities rather than silently assigning a label."""
    return ACTIVITY_LABELS[activity]


def validate_structure(root: Path, videos: list[Path]) -> list[str]:
    issues = []
    expected = {f"Subject.{i}" for i in range(1, 11)}
    actual = {p.name for p in root.iterdir() if p.is_dir()}
    for name in sorted(expected - actual, key=subject_key):
        issues.append(f"Missing subject: {name}")
    for name in sorted(actual - expected):
        issues.append(f"Unexpected subject directory: {name}")
    for subject in sorted(expected & actual, key=subject_key):
        activities = {p.name for p in (root / subject).iterdir() if p.is_dir()}
        for activity in sorted(set(ACTIVITY_LABELS) - activities):
            issues.append(f"Missing activity: {subject}/{activity}")
        for activity in sorted(activities - set(ACTIVITY_LABELS)):
            issues.append(f"Unexpected activity: {subject}/{activity}")
        for activity in sorted(set(ACTIVITY_LABELS) & activities):
            count = sum(p.parent == root / subject / activity for p in videos)
            if count != 1:
                issues.append(f"Expected 1 AVI, found {count}: {subject}/{activity}")
    for video in videos:
        parts = video.relative_to(root).parts
        if len(parts) != 3 or parts[0] not in expected or parts[1] not in ACTIVITY_LABELS:
            issues.append(f"Unexpected AVI location: {video.relative_to(root).as_posix()}")
    return issues


def classify_txt(path: Path, content: str) -> str:
    """Recognize observed class-id + four normalized box values; not pose data.

    Empty or malformed TXT is 'other_txt', even if a PNG shares its basename.
    Coordinate interpretation is YOLO-like, inferred from the observed format.
    """
    if path.name.lower() == "classes.txt":
        return "classes_metadata"
    lines = [line.split() for line in content.splitlines() if line.strip()]
    if not lines:
        return "other_txt"
    for fields in lines:
        if len(fields) != 5 or not fields[0].isdigit():
            return "other_txt"
        try:
            values = [float(value) for value in fields[1:]]
        except ValueError:
            return "other_txt"
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            return "other_txt"
    return "frame_annotation"


def inspect_annotations(root: Path, files: list[Path]) -> dict[str, Any]:
    pngs = {p.with_suffix(""): p for p in files if p.suffix.lower() == ".png"}
    texts = {p.with_suffix(""): p for p in files if p.suffix.lower() == ".txt"}
    kinds: dict[str, list[str]] = {k: [] for k in
                                 ("classes_metadata", "frame_annotation", "other_txt")}
    contents = {}
    errors = []
    samples = {}
    class_contents: Counter[str] = Counter()
    class_ids: Counter[str] = Counter()
    invalid_ids = []
    for stem, path in texts.items():
        relative = path.relative_to(root).as_posix()
        try:
            content = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            errors.append(f"{relative}: {exc}")
            kinds["other_txt"].append(relative)
            continue
        contents[stem] = content
        kind = classify_txt(path, content)
        kinds[kind].append(relative)
        samples.setdefault(kind, {"path": relative, "content": content})
        if kind == "classes_metadata":
            class_contents[content] += 1
        if kind == "frame_annotation":
            ids = [line.split()[0] for line in content.splitlines() if line.strip()]
            class_ids.update(ids)
            metadata = path.parent / "classes.txt"
            try:
                names = metadata.read_text(encoding="utf-8-sig").splitlines()
                if any(int(i) >= len(names) for i in ids):
                    invalid_ids.append(relative)
            except (OSError, UnicodeError):
                invalid_ids.append(relative)
    rel = lambda paths: sorted(p.relative_to(root).as_posix() for p in paths)
    orphan_png = rel(p for stem, p in pngs.items() if stem not in texts)
    orphan_txt = rel(p for stem, p in texts.items()
                     if stem not in pngs and p.name.lower() != "classes.txt")
    annotation_set = set(kinds["frame_annotation"])
    missing_metadata = [f"Subject.{i}/{activity}/classes.txt"
                        for i in range(1, 11) for activity in ACTIVITY_LABELS
                        if not (root / f"Subject.{i}" / activity / "classes.txt").is_file()]
    return {
        "counts": {kind: len(paths) for kind, paths in kinds.items()},
        "matched_frame_annotations": sum(p.relative_to(root).as_posix() in annotation_set
                                         for stem, p in texts.items() if stem in pngs),
        "png_without_txt": orphan_png,
        "non_classes_txt_without_png": orphan_txt,
        "orphan_frame_annotations": [p for p in orphan_txt if p in annotation_set],
        "png_with_other_txt": rel(p for stem, p in pngs.items() if stem in texts
                                   and texts[stem].relative_to(root).as_posix() not in annotation_set),
        "other_txt": kinds["other_txt"], "read_errors": errors,
        "classes_content_distribution": dict(class_contents),
        "frame_class_id_distribution": dict(class_ids),
        "invalid_or_missing_class_mapping": invalid_ids,
        "missing_classes_metadata": missing_metadata, "representative_samples": samples,
        "orphan_txt_contents": {p: contents.get((root / p).with_suffix("")) for p in orphan_txt},
    }


def inspect_video(root: Path, path: Path, cv2: Any) -> dict[str, Any]:
    parts = path.relative_to(root).parts
    subject, activity = (parts[0], parts[1]) if len(parts) >= 3 else ("", "")
    row: dict[str, Any] = dict.fromkeys(FIELDS)
    row.update(subject=subject, activity=activity, label=ACTIVITY_LABELS.get(activity, "unknown"),
               relative_video_path=path.relative_to(root).as_posix(), opened=False,
               decoded_first_frame=False, readable=False, decoded_frame_count=0,
               reached_end_normally=False, decode_failure_before_expected_end=None,
               frame_count_matches=None, error="")
    capture = cv2.VideoCapture()
    try:
        row["opened"] = bool(capture.open(str(path)))
        if not row["opened"]:
            row["error"] = "VideoCapture could not open video"
            return row
        for name, prop in [("fps", cv2.CAP_PROP_FPS), ("width", cv2.CAP_PROP_FRAME_WIDTH),
                           ("height", cv2.CAP_PROP_FRAME_HEIGHT),
                           ("frame_count", cv2.CAP_PROP_FRAME_COUNT)]:
            value = capture.get(prop)
            row[name] = value if math.isfinite(value) and value > 0 else None
        if row["fps"] and row["frame_count"]:
            row["duration_seconds"] = row["frame_count"] / row["fps"]
        while True:
            success, frame = capture.read()
            if not success or frame is None or not frame.size:
                break
            row["decoded_frame_count"] += 1
        row["decoded_first_frame"] = row["decoded_frame_count"] > 0
        expected = row["frame_count"]
        if expected is not None:
            row["frame_count_matches"] = row["decoded_frame_count"] == expected
            row["decode_failure_before_expected_end"] = row["decoded_frame_count"] < expected
            row["reached_end_normally"] = row["frame_count_matches"]
        row["readable"] = row["decoded_first_frame"] and row["reached_end_normally"]
        if not row["decoded_first_frame"]:
            row["error"] = "First frame could not be decoded"
        elif any(row[key] is None for key in ("fps", "width", "height", "frame_count")):
            row["error"] = "Missing or invalid media metadata"
        elif not row["frame_count_matches"]:
            row["error"] = "Decoded frame count differs from metadata"
    except cv2.error as exc:
        row["error"] = str(exc)
        row["decoded_first_frame"] = row["decoded_frame_count"] > 0
        if row["frame_count"] is not None:
            row["decode_failure_before_expected_end"] = row["decoded_frame_count"] < row["frame_count"]
    finally:
        capture.release()
    return row


def media_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def bounds(key: str) -> list[float] | None:
        values = [row[key] for row in rows if row[key] is not None]
        return [min(values), max(values)] if values else None
    return {
        "fps_distribution": dict(sorted(Counter(str(row["fps"]) for row in rows).items())),
        "resolution_distribution": dict(Counter(f'{row["width"]}x{row["height"]}' for row in rows)),
        "frame_count_range": bounds("frame_count"),
        "duration_seconds_range": bounds("duration_seconds"),
        "opened": sum(row["opened"] for row in rows),
        "decoded_first_frame": sum(row["decoded_first_frame"] for row in rows),
        "decoded_frame_count": sum(row["decoded_frame_count"] for row in rows),
        "reached_end_normally": sum(row["reached_end_normally"] for row in rows),
        "issues": [{"path": row["relative_video_path"], "error": row["error"]}
                   for row in rows if row["error"]],
    }


def validate_output(root: Path, output: Path) -> None:
    """Refuse reports inside the input tree or the repository's immutable raw tree."""
    raw = Path(__file__).resolve().parents[2] / "data/raw"
    for protected in (root.resolve(), raw.resolve()):
        if output.resolve().is_relative_to(protected):
            raise ValueError("Report output must be outside the dataset and data/raw")
    for name in ("inventory.csv", "summary.json", "summary.md"):
        if (output / name).is_symlink():
            raise ValueError(f"Refusing symlink report destination: {output / name}")


def write_reports(output: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    annotations = summary["annotations"]
    lines = ["# CAUCAFall V5 inspection", "", f'Validation: **{summary["validation"]}**.', "",
             "Sequential decode to read() failure; normal end means decoded count equals metadata. OpenCV does not distinguish EOF from a decoder failure; count agreement is not proof of visual integrity.",
             "FPS and frame counts come from OpenCV metadata; duration = frame_count / FPS (seconds).",
             "Paths are relative to the supplied CAUCAFall root. No raw files are changed.", "",
             "## Counts and media", "", "```json",
             json.dumps({key: summary[key] for key in ("counts", "labels", "media")}, indent=2),
             "```", "", "## Annotation findings", "", "```json",
             json.dumps(annotations, indent=2), "```", "", "## Structure issues", ""]
    lines.extend(f"- {issue}" for issue in summary["structure_issues"])
    if not summary["structure_issues"]:
        lines.append("None.")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"Dataset root does not exist: {args.root}")
    try:
        validate_output(args.root, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        import cv2
    except ImportError:
        parser.error("OpenCV missing. Run: uv run ml/datasets/inspect_caucafall.py")
    files = sorted((p for p in args.root.rglob("*") if p.is_file()),
                   key=lambda p: (subject_key(p.relative_to(args.root).parts[0]), p.as_posix()))
    videos = [p for p in files if p.suffix.lower() == ".avi"]
    structure = validate_structure(args.root, videos)
    annotations = inspect_annotations(args.root, files)
    rows = [inspect_video(args.root, path, cv2) for path in videos]
    media = media_summary(rows)
    warnings = any(annotations[key] for key in
                   ("png_without_txt", "non_classes_txt_without_png", "other_txt",
                    "read_errors", "png_with_other_txt", "missing_classes_metadata",
                    "invalid_or_missing_class_mapping"))
    summary = {
        "dataset": "CAUCAFall V5 (locally supplied download; version not independently authenticated)",
        "runtime": {"python": platform.python_version(), "opencv": cv2.__version__},
        "counts": {"subjects": len([p for p in args.root.iterdir() if p.is_dir()]),
                   "activity_directories": sum(p.is_dir() for s in args.root.iterdir()
                                               if s.is_dir() for p in s.iterdir()),
                   **{suffix[1:]: sum(p.suffix.lower() == suffix for p in files)
                      for suffix in (".avi", ".png", ".txt")}},
        "labels": dict(Counter(row["label"] for row in rows)),
        "structure_issues": structure, "annotations": annotations, "media": media,
        "validation": "FAIL" if structure or media["issues"] else
                      "PASS_WITH_ANNOTATION_WARNINGS" if warnings else "PASS",
    }
    write_reports(args.output, rows, summary)
    print(f'{summary["validation"]}: {len(videos)} AVI; '
          f'{media["decoded_first_frame"]} decoded first frames; {media["decoded_frame_count"]} total frames. Reports: {args.output}')
    return 1 if summary["validation"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
