"""Stage 3.1a: Train-only missingness topology and sensitivity from Stage 3.1 tables."""

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.datasets.pose_raw import timestamp_ms
from ml.preprocessing import contract
from ml.preprocessing.characterize_pose import BASELINE, OUTPUT, REPO, distribution

VERSION = "stage31a_missingness_addendum_v1"
TOPOLOGIES = ("internal", "leading", "trailing", "whole_video")
METRICS = ("mean_video_missing_fraction", "median_video_missing_fraction",
           "mean_longest_missing_support_ms", "median_longest_missing_support_ms",
           "mean_missing_run_count")
NAMES = ("missing_runs_topology.csv", "missing_topology_summary.csv",
         "subject_class_missingness.csv", "subject_class_differences.csv",
         "missingness_sensitivity.csv", "stage31a_summary.json", "stage31a_REPORT.md")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Invalid CSV header: {path}")
        return list(reader)


def _csv_bytes(rows: list[dict[str, Any]], columns: list[str]) -> bytes:
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(row[key], separators=(",", ":"), allow_nan=False)
                         if isinstance(row[key], (list, dict)) else row[key] for key in columns})
    return out.getvalue().encode("utf-8")


def _json_bytes(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def classify_run(run: dict[str, Any], video: dict[str, Any], frame_ms: int) -> dict[str, Any]:
    """Classify an existing run from source indices; add support and anchor timing."""
    total = int(video["frames"])
    start, end = int(run["start_frame"]), int(run["end_frame"])
    count = int(run["frames"])
    if not 0 <= start <= end < total or count != end - start + 1:
        raise ValueError("Invalid missing-run indices/count")
    first = int(run["start_ms"])
    end_exclusive = int(run["end_exclusive_ms"])
    last = end_exclusive - frame_ms
    support = end_exclusive - first
    if (frame_ms <= 0 or first != timestamp_ms(start, 1000 / frame_ms)
            or last != timestamp_ms(end, 1000 / frame_ms)
            or support != count * frame_ms or int(run["duration_ms"]) != support):
        raise ValueError("Missing-run timing disagrees with frozen source rate")
    if start == 0 and end == total - 1:
        topology = "whole_video"
    elif start == 0:
        topology = "leading"
    elif end == total - 1:
        topology = "trailing"
    else:
        topology = "internal"
    left = first - frame_ms if topology == "internal" else None
    right = last + frame_ms if topology == "internal" else None
    return dict(**run, topology=topology, missing_frame_count=count,
                first_missing_timestamp_ms=first, last_missing_timestamp_ms=last,
                missing_support_ms=support, left_observed_timestamp_ms=left,
                right_observed_timestamp_ms=right,
                anchor_gap_ms=right - left if topology == "internal" else None)


def topology_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_frames = sum(int(r["missing_frame_count"]) for r in runs)
    result = []
    for topology in TOPOLOGIES:
        selected = [r for r in runs if r["topology"] == topology]
        frames = sum(int(r["missing_frame_count"]) for r in selected)
        result.append(dict(topology=topology, runs=len(selected), missing_frames=frames,
                           fraction_of_all_missing_frames=frames / total_frames if total_frames else None,
                           run_length_frames=distribution([int(r["missing_frame_count"]) for r in selected]),
                           missing_support_ms=distribution([int(r["missing_support_ms"]) for r in selected]),
                           anchor_gap_ms=distribution([int(r["anchor_gap_ms"]) for r in selected])
                           if topology == "internal" else None))
    return result


def class_summary(videos: list[dict[str, Any]]) -> dict[str, Any]:
    """Equal-video descriptors; longest gap is zero when a video has no missing run."""
    if not videos:
        raise ValueError("Empty class group")
    missing = np.array([float(v["missing_fraction"]) for v in videos])
    longest = np.array([int(v["longest_missing_run_ms"] or 0) for v in videos])
    run_counts = np.array([int(v["missing_run_count"]) for v in videos])
    frames = sum(int(v["frames"]) for v in videos)
    detected = sum(int(v["detected_frames"]) for v in videos)
    return dict(videos=len(videos), frames=frames, detected_frames=detected,
                missing_frames=frames - detected, pooled_pose_availability=detected / frames,
                mean_video_missing_fraction=float(missing.mean()),
                median_video_missing_fraction=float(np.median(missing)),
                mean_longest_missing_support_ms=float(longest.mean()),
                median_longest_missing_support_ms=float(np.median(longest)),
                mean_missing_run_count=float(run_counts.mean()))


def subject_class_tables(videos: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_subject: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in videos:
        if row["split"] != "train":
            raise contract.ContractError("Stage 3.1a subject analysis accepts Train only")
        by_subject[int(row["subject_id"])][row["video_label"]].append(row)
    summaries, differences = [], []
    for subject in sorted(by_subject):
        groups = by_subject[subject]
        if set(groups) != {"fall", "non_fall"} or any(len(groups[label]) != 5 for label in groups):
            raise ValueError("Expected five fall and five non-fall Train videos per subject")
        fall = class_summary(groups["fall"])
        non_fall = class_summary(groups["non_fall"])
        summaries.extend((dict(subject_id=subject, video_label=label, **stats)
                          for label, stats in (("fall", fall), ("non_fall", non_fall))))
        differences.append(dict(subject_id=subject, **{
            f"fall_minus_non_fall_{metric}": fall[metric] - non_fall[metric] for metric in METRICS}))
    return summaries, differences


def sensitivity(videos: list[dict[str, Any]], whole_video_sources: set[str]
                ) -> tuple[list[dict[str, Any]], list[str]]:
    """Predeclared one-step sensitivity: omit only complete no-pose videos."""
    if any(v["split"] != "train" for v in videos):
        raise contract.ContractError("Stage 3.1a sensitivity accepts Train only")
    all_paths = {v["source_video"] for v in videos}
    actual_whole = {v["source_video"] for v in videos if int(v["detected_frames"]) == 0}
    if whole_video_sources != actual_whole or not whole_video_sources <= all_paths:
        raise ValueError("Whole-video topology and zero-detected video identities disagree")
    omitted = sorted(whole_video_sources)
    result = []
    for view, selected in (("all_train", videos),
                           ("excluding_whole_video_missing", [v for v in videos
                                                               if v["source_video"] not in whole_video_sources])):
        for label in ("fall", "non_fall"):
            group = [v for v in selected if v["video_label"] == label]
            result.append(dict(view=view, video_label=label,
                               omitted_source_videos=omitted if view != "all_train" else [],
                               **class_summary(group)))
    return result, omitted


def _validate_original(sources: tuple[contract.PoseSource, ...],
                       summary: dict[str, Any], videos: list[dict[str, str]],
                       runs: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (len(sources) != 60 or summary.get("analysis_version") != "stage31_train_pose_characterization_v1"
            or summary.get("provenance", {}).get("stage2_run_id") != contract.RUN_ID
            or summary["provenance"].get("stage30_baseline_commit") != BASELINE
            or summary["provenance"].get("split") != "train"
            or summary["provenance"].get("analysis_code_sha256") !=
            _hash(REPO / "ml/preprocessing/characterize_pose.py")):
        raise ValueError("Unexpected Stage 3.1 source identity")
    identities = {s.source_video: s for s in sources}
    if len(identities) != len(videos) or {v["source_video"] for v in videos} != set(identities):
        raise ValueError("Stage 3.1 video table does not match guarded Train sources")
    if sum(int(v["frames"]) for v in videos) != summary["inventory"]["frames"]:
        raise ValueError("Stage 3.1 frame totals disagree")
    prepared_videos = {}
    for row in videos:
        source = identities[row["source_video"]]
        if (row["split"] != "train" or int(row["subject_id"]) != source.subject_id
                or row["video_label"] != source.video_label or row["activity"] != source.activity):
            raise ValueError("Non-Train or mismatched Stage 3.1 video row")
        prepared_videos[source.source_video] = row
    selected_runs = defaultdict(list)
    enriched = []
    for run in runs:
        name = run["source_video"]
        if name not in prepared_videos or run["split"] != "train":
            raise ValueError("Non-Train or unknown Stage 3.1 missing run")
        video = prepared_videos[name]
        if run["video_label"] != video["video_label"] or int(run["subject_id"]) != int(video["subject_id"]):
            raise ValueError("Missing run video identity mismatch")
        frames = int(video["frames"])
        duration = int(video["duration_ms"])
        frame_ms = timestamp_ms(1, 20.0) - timestamp_ms(0, 20.0)
        if frames <= 0 or duration != frames * frame_ms:
            raise ValueError("Video duration disagrees with frozen 20 FPS timestamps")
        item = classify_run(run, video, frame_ms)
        selected_runs[name].append(item)
        enriched.append(item)
    if len(enriched) != summary["missing_runs"]["count"]:
        raise ValueError("Stage 3.1 missing-run count disagrees")
    for name, video in prepared_videos.items():
        ordered = sorted(selected_runs[name], key=lambda r: int(r["start_frame"]))
        if (len(ordered) != int(video["missing_run_count"])
                or sum(r["missing_frame_count"] for r in ordered) != int(video["missing_frames"])):
            raise ValueError("Missing-run/video count disagreement")
        for left, right in zip(ordered, ordered[1:]):
            if int(right["start_frame"]) <= int(left["end_frame"]) + 1:
                raise ValueError("Overlapping or unsegmented missing runs")
    return list(prepared_videos.values()), enriched


def _report(topology: list[dict[str, Any]], differences: list[dict[str, Any]],
            sensitivity_rows: list[dict[str, Any]], omitted: list[str]) -> str:
    lines = ["# Stage 3.1a — Train-only missingness addendum", "",
             "Status: HOLD addendum for external Gate Review. This extends the existing "
             "Stage 3.1 analysis without changing its original files or the dataset.", "",
             "## Missing-run topology", "",
             "| Type | Runs | Missing frames | Share of missing frames | Median support (ms) | Maximum support (ms) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in topology:
        d = row["missing_support_ms"]
        lines.append(f"| {row['topology']} | {row['runs']} | {row['missing_frames']} | "
                     f"{row['fraction_of_all_missing_frames']:.1%} | {d['p50'] if d['count'] else 'NA'} | "
                     f"{d['max'] if d['count'] else 'NA'} |")
    lines += ["", "`missing_support_ms` is the timestamp support of missing samples: "
              "last missing timestamp minus first missing timestamp plus one 20 FPS frame period. "
              "For N missing frames it is N × 50 ms. For internal gaps, `anchor_gap_ms` "
              "is the right observed timestamp minus the left observed timestamp; "
              "it is a different quantity. Non-internal anchors are null.", "",
              "## Subject-stratified fall minus non-fall differences", "",
              "Each subject contributes five fall and five non-fall videos. Values below are descriptive "
              "differences of equal-video summaries; no significance test was run.", "",
              "| Subject | Mean missing fraction | Median missing fraction | Mean longest support (ms) | "
              "Median longest support (ms) | Mean run count |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in differences:
        values = [row[f"fall_minus_non_fall_{metric}"] for metric in METRICS]
        lines.append(f"| {row['subject_id']} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    rows = {(r["view"], r["video_label"]): r for r in sensitivity_rows}
    lines += ["", "## Whole-video-missing sensitivity", "",
              f"Omitted **only in the sensitivity view**: {', '.join(omitted) if omitted else 'none'}. "
              "The original 60-video dataset and Stage 3.1 tables remain unchanged.", "",
              "| View | Class | Videos | Mean video missing fraction | Mean longest support (ms) | Mean run count |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for view in ("all_train", "excluding_whole_video_missing"):
        for label in ("fall", "non_fall"):
            row = rows[view, label]
            lines.append(f"| {view} | {label} | {row['videos']} | "
                         f"{row['mean_video_missing_fraction']:.4f} | "
                         f"{row['mean_longest_missing_support_ms']:.1f} | "
                         f"{row['mean_missing_run_count']:.2f} |")
    lines += ["", "Frames are nested within videos, and videos within subjects. "
              "Class differences can reflect subject, activity or tracking characteristics. "
              "This diagnostic establishes neither causality nor predictive usefulness. "
              "No preprocessing threshold or exclusion policy is selected. "
              "Validation/Test arrays were not loaded.", ""]
    return "\n".join(lines)


def build(split: str = "train", base: Path = OUTPUT) -> dict[str, bytes]:
    """Build an addendum from already reviewed Train tables; never open pose arrays."""
    if split != "train":
        raise contract.ContractError("Stage 3.1a accepts Train only")
    sources = contract.select_sources(("train",), purpose="exploration")
    paths = {name: base / name for name in ("summary.json", "videos.csv", "missing_runs.csv")}
    summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
    videos, runs = _validate_original(sources, summary, _read_csv(paths["videos.csv"]),
                                      _read_csv(paths["missing_runs.csv"]))
    topology = topology_summary(runs)
    if sum(r["missing_frames"] for r in topology) != summary["inventory"]["missing_frames"]:
        raise ValueError("Topology missing-frame total disagrees")
    subject_rows, differences = subject_class_tables(videos)
    if {r["subject_id"] for r in differences} != set(contract.FROZEN_SUBJECTS["train"]):
        raise ValueError("Unexpected subject scope")
    whole = {r["source_video"] for r in runs if r["topology"] == "whole_video"}
    sensitivity_rows, omitted = sensitivity(videos, whole)
    addendum = {
        "analysis_version": VERSION, "status": "hold_addendum_pending_external_gate_review",
        "provenance": {
            "stage30_baseline_commit": BASELINE, "stage2_run_id": contract.RUN_ID,
            "source_stage31_version": summary["analysis_version"], "split": "train",
            "subject_ids": list(contract.FROZEN_SUBJECTS["train"]),
            "code_sha256": _hash(Path(__file__)),
            "stage30_contract_code_sha256": _hash(REPO / "ml/preprocessing/contract.py"),
            "stage30_config_sha256": _hash(contract.CONFIG),
            "source_stage31_file_sha256": {name: _hash(path) for name, path in paths.items()},
            "stage2_manifest_sha256": summary["provenance"]["stage2_manifest_sha256"],
            "pose_arrays_loaded": False,
        },
        "definitions": {
            "topology": {
                "internal": "observed pose before and after the run",
                "leading": "run begins at frame zero and has an observed pose after",
                "trailing": "run ends at final frame and has an observed pose before",
                "whole_video": "every frame has pose_detected=false",
            },
            "missing_support_ms": "last_missing_timestamp_ms - first_missing_timestamp_ms + source frame period; N*50 ms at frozen 20 FPS",
            "anchor_gap_ms": "internal only: right_observed_timestamp_ms - left_observed_timestamp_ms; null otherwise",
            "class_difference": "fall minus non_fall within each subject; equal-video means/medians, not frame-independent tests",
            "sensitivity": "single predeclared exclusion of whole-video-missing videos, only in diagnostic view",
            "quantiles": "NumPy linear percentile, null for empty; sample count accompanies every distribution",
        },
        "topology": topology, "subject_class_differences": differences,
        "sensitivity": {"omitted_source_videos": omitted, "rows": sensitivity_rows},
        "interpretation": "Descriptive Train-only diagnostic; nested frames/videos, possible subject/activity/tracking confounding; no causal or predictive claim, no policy selection.",
    }
    return {
        "missing_runs_topology.csv": _csv_bytes(runs, list(runs[0])) if runs else b"",
        "missing_topology_summary.csv": _csv_bytes(topology, list(topology[0])),
        "subject_class_missingness.csv": _csv_bytes(subject_rows, list(subject_rows[0])),
        "subject_class_differences.csv": _csv_bytes(differences, list(differences[0])),
        "missingness_sensitivity.csv": _csv_bytes(sensitivity_rows, list(sensitivity_rows[0])),
        "stage31a_summary.json": _json_bytes(addendum),
        "stage31a_REPORT.md": _report(topology, differences, sensitivity_rows, omitted).encode("utf-8"),
    }


def publish(files: dict[str, bytes], base: Path = OUTPUT) -> None:
    """Add only new files in the existing analysis namespace; preserve originals."""
    if set(files) != set(NAMES) or not base.is_dir() or base.resolve() != base.absolute():
        raise ValueError("Unexpected addendum output directory or file set")
    for name in files:
        contract.validate_output_path(base / name)
    created = []
    with tempfile.TemporaryDirectory(prefix=".stage31a-", dir=base) as tmp:
        staged = Path(tmp)
        try:
            for name, content in files.items():
                (staged / name).write_bytes(content)
            for name in files:
                dest = base / name
                os.link(staged / name, dest)  # Atomic no-overwrite publication per file.
                created.append(dest)
        except BaseException:
            for dest in created:
                dest.unlink()
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", help="Stage 3.1a accepts train only")
    args = parser.parse_args()
    files = build(args.split)
    publish(files)
    print(json.dumps({"output": str(OUTPUT), "files": sorted(files), "status": "addendum_only"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
