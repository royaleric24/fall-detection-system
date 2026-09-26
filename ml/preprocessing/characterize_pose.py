"""Stage 3.1: descriptive Train-only raw pose analysis, without transformations."""

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.datasets.pose_raw import load_validated
from ml.preprocessing import contract

REPO = Path(__file__).resolve().parents[2]
VERSION = "stage31_train_pose_characterization_v1"
BASELINE = "9d1912cd4c1566438a60496d5d055a43189d3a66"
OUTPUT = REPO / "artifacts/preprocessing" / VERSION
SOURCE_FPS = 20.0  # Frozen CAUCAFall source timing; verified per manifest row.
FRAME_MS = 50  # 1000 / 20; descriptive duration of one source frame.
PERCENTILES = (5, 10, 25, 50, 75, 90, 95, 99)
RUN_PERCENTILES = (50, 75, 90, 95, 99)
LANDMARKS = {"left_shoulder": 11, "right_shoulder": 12, "left_hip": 23, "right_hip": 24}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: Any, percentiles: tuple[int, ...] = PERCENTILES) -> dict[str, Any]:
    """Finite descriptive distribution; population SD and NumPy linear quantiles."""
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(data).all():
        raise ValueError("Distribution requires finite observed values")
    result: dict[str, Any] = {"count": int(data.size), "mean": None, "std_population": None,
                               "min": None, "max": None}
    result.update({f"p{p}": None for p in percentiles})
    if data.size:
        result.update(mean=float(data.mean()), std_population=float(data.std(ddof=0)),
                      min=float(data.min()), max=float(data.max()))
        result.update({f"p{p}": float(np.percentile(data, p, method="linear"))
                       for p in percentiles})
    return result


def missing_runs(mask: np.ndarray, timestamps: np.ndarray,
                 frame_ms: int = FRAME_MS) -> list[dict[str, Any]]:
    """Inclusive runs with end-exclusive elapsed duration from source timestamps."""
    if mask.ndim != 1 or timestamps.shape != mask.shape or mask.dtype != np.dtype("bool"):
        raise ValueError("Expected same-length 1-D boolean mask and timestamps")
    if frame_ms <= 0 or (timestamps.size and (timestamps[0] != 0 or np.any(np.diff(timestamps) <= 0))):
        raise ValueError("Invalid source timing")
    total_ms = int(timestamps[-1]) + frame_ms if timestamps.size else 0
    result = []
    start = None
    for index in range(mask.size + 1):
        is_missing = index < mask.size and not bool(mask[index])
        if is_missing and start is None:
            start = index
        elif not is_missing and start is not None:
            end = index - 1
            duration = int(timestamps[end] - timestamps[start]) + frame_ms
            result.append(dict(start_frame=start, end_frame=end, frames=index - start,
                               duration_ms=duration, start_ms=int(timestamps[start]),
                               end_exclusive_ms=int(timestamps[end]) + frame_ms,
                               start_progress=float(timestamps[start] / total_ms),
                               end_progress=float((timestamps[end] + frame_ms) / total_ms)))
            start = None
    return result


def video_statistics(source: contract.PoseSource, arrays: dict[str, np.ndarray]
                     ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Describe one validated raw video; never alter or persist its arrays."""
    detected = arrays["pose_detected"]
    timestamps = arrays["timestamp_ms"]
    landmarks = arrays["landmarks"][detected]
    frame_count = int(detected.size)
    detected_count = int(detected.sum())
    runs = missing_runs(detected, timestamps)
    durations = [r["duration_ms"] for r in runs]
    lengths = [r["frames"] for r in runs]
    progress_bins = [0] * 5
    duration_ms = int(timestamps[-1]) + FRAME_MS
    for index in np.flatnonzero(~detected):
        progress = (int(timestamps[index]) + FRAME_MS / 2) / duration_ms
        progress_bins[min(int(progress * 5), 4)] += 1
    row = dict(subject_id=source.subject_id, source_video=source.source_video,
               source_npz=source.source_npz.relative_to(REPO).as_posix(),
               activity=source.activity, video_label=source.video_label, split=source.split,
               frames=frame_count, duration_ms=duration_ms, detected_frames=detected_count,
               missing_frames=frame_count - detected_count,
               pose_availability=detected_count / frame_count,
               missing_fraction=(frame_count - detected_count) / frame_count,
               missing_run_count=len(runs), shortest_missing_run_frames=min(lengths, default=None),
               shortest_missing_run_ms=min(durations, default=None),
               median_missing_run_ms=float(np.median(durations)) if durations else None,
               longest_missing_run_frames=max(lengths, default=None),
               longest_missing_run_ms=max(durations, default=None),
               missing_run_durations_frames=lengths, missing_run_durations_ms=durations,
               missing_progress_quintile_frames=progress_bins,
               zero_detected_poses=detected_count == 0)
    run_rows = [dict(subject_id=source.subject_id, source_video=source.source_video,
                     video_label=source.video_label, split=source.split, **run) for run in runs]
    # Raw image-relative x/y distances only. No 3-D metric interpretation or normalization.
    xy = landmarks[:, :, :2].astype(np.float64, copy=False)
    left_shoulder, right_shoulder = xy[:, 11], xy[:, 12]
    left_hip, right_hip = xy[:, 23], xy[:, 24]
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    hip_midpoint = (left_hip + right_hip) / 2
    geometry = {
        "shoulder_width_xy": np.linalg.norm(left_shoulder - right_shoulder, axis=1),
        "hip_width_xy": np.linalg.norm(left_hip - right_hip, axis=1),
        "shoulder_to_hip_midpoint_torso_xy": np.linalg.norm(shoulder_midpoint - hip_midpoint, axis=1),
    }
    values = dict(raw_channels=landmarks, visibility=landmarks[:, :, 3], geometry=geometry)
    return row, run_rows, values


def grouped_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Frame totals plus equal-video descriptive means; no frame independence claim."""
    frames = sum(r["frames"] for r in rows)
    detected = sum(r["detected_frames"] for r in rows)
    return dict(videos=len(rows), frames=frames, detected_frames=detected,
                missing_frames=frames - detected, pooled_pose_availability=detected / frames if frames else None,
                mean_video_pose_availability=float(np.mean([r["pose_availability"] for r in rows])) if rows else None,
                mean_video_missing_fraction=float(np.mean([r["missing_fraction"] for r in rows])) if rows else None,
                mean_video_longest_missing_run_ms=float(np.mean([r["longest_missing_run_ms"] or 0 for r in rows])) if rows else None,
                mean_video_missing_run_count=float(np.mean([r["missing_run_count"] for r in rows])) if rows else None,
                zero_detected_pose_videos=sum(r["zero_detected_poses"] for r in rows))


def _group_table(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return [dict(**{key: name}, **grouped_summary(group)) for name, group in sorted(groups.items())]


def _csv_bytes(rows: list[dict[str, Any]], columns: list[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(row[key], separators=(",", ":"), allow_nan=False)
                         if isinstance(row[key], (list, dict)) else row[key] for key in columns})
    return output.getvalue().encode("utf-8")


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _train_manifest_metadata(sources: tuple[contract.PoseSource, ...],
                             config: dict[str, Any]) -> dict[str, tuple[float, int]]:
    """Read only selected Train metadata after the Stage 3.0 selector passed."""
    manifest = REPO / config["source"]["evidence_root"] / "manifest.csv"
    if sha256(manifest) != config["source"]["manifest_sha256"]:
        raise ValueError("Official manifest changed after Train selection")
    selected = {s.source_video for s in sources}
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["source_relative_video_path"] in selected]
    if len(rows) != len(sources) or any(row["split"] != "train" for row in rows):
        raise ValueError("Train manifest scope mismatch")
    result = {}
    for row in rows:
        fps = float(row["source_fps"])
        count = int(row["expected_frame_count"])
        if fps != SOURCE_FPS or count <= 0 or row["status"] != "complete":
            raise ValueError("Unexpected Train timing/count/status")
        result[row["source_relative_video_path"]] = (fps, count)
    if len(result) != len(sources):
        raise ValueError("Duplicate Train manifest source")
    return result


def analyze(split: str = "train") -> dict[str, bytes]:
    """Build files in memory; refuse Validation/Test before any pose or metadata I/O."""
    if split != "train":
        raise contract.ContractError("Stage 3.1 accepts Train only")
    sources = contract.select_sources(("train",), purpose="exploration")
    if len(sources) != 60 or {s.subject_id for s in sources} != set(contract.FROZEN_SUBJECTS["train"]):
        raise ValueError("Unexpected Train scope")
    config = contract.read_contract()
    if any(s.split != "train" or s.stage2_run_id != contract.RUN_ID for s in sources):
        raise ValueError("Non-Train or wrong-run source")
    metadata = _train_manifest_metadata(sources, config)
    video_rows = []
    run_rows = []
    visibility_parts = []
    frame_mean_parts = []
    frame_min_parts = []
    channel_parts: dict[str, list[np.ndarray]] = {name: [] for name in ("x", "y", "z", "visibility")}
    geometry_parts: dict[str, list[np.ndarray]] = {name: [] for name in
                                                        ("shoulder_width_xy", "hip_width_xy",
                                                         "shoulder_to_hip_midpoint_torso_xy")}
    npz_hashes = {}
    for source in sources:
        fps, count = metadata[source.source_video]
        before_hash = sha256(source.source_npz)
        # Existing Stage 2 validator enforces all four keys, shapes, dtypes, timing,
        # finite detected rows and all-NaN missing rows before any statistics.
        arrays = load_validated(source.source_npz, fps, count)
        row, runs, values = video_statistics(source, arrays)
        video_rows.append(row)
        run_rows.extend(runs)
        visibility = values["visibility"]
        visibility_parts.append(visibility)
        if visibility.size:
            frame_mean_parts.append(visibility.mean(axis=1))
            frame_min_parts.append(visibility.min(axis=1))
        for index, name in enumerate(channel_parts):
            channel_parts[name].append(values["raw_channels"][:, :, index].reshape(-1))
        for name, samples in values["geometry"].items():
            geometry_parts[name].append(samples)
        after_hash = sha256(source.source_npz)
        if before_hash != after_hash:
            raise ValueError(f"Source NPZ changed during analysis: {source.source_video}")
        npz_hashes[source.source_video] = after_hash
    if len(video_rows) != 60 or sum(r["frames"] for r in video_rows) <= 0:
        raise ValueError("Incomplete Train analysis")
    videos = sorted(video_rows, key=lambda r: (r["subject_id"], r["activity"], r["source_video"]))
    runs = sorted(run_rows, key=lambda r: (r["subject_id"], r["source_video"], r["start_frame"]))
    observed_visibility = np.concatenate(visibility_parts, axis=0)
    landmark_rows = []
    for joint in range(33):
        landmark_rows.append(dict(landmark_index=joint,
                                  **distribution(observed_visibility[:, joint])))
    channels = {}
    for name, parts in channel_parts.items():
        samples = np.concatenate(parts)
        channels[name] = dict(**distribution(samples),
                              outside_unit_interval_count=int(((samples < 0) | (samples > 1)).sum())
                              if name in ("x", "y", "visibility") else None)
    geometry = {}
    for name, parts in geometry_parts.items():
        samples = np.concatenate(parts)
        geometry[name] = dict(**distribution(samples), exact_zero_count=int((samples == 0).sum()),
                              nonpositive_count=int((samples <= 0).sum()))
    subject_rows = _group_table(videos, "subject_id")
    activity_rows = _group_table(videos, "activity")
    class_rows = _group_table(videos, "video_label")
    longest = [r["longest_missing_run_ms"] or 0 for r in videos]
    summary = {
        "analysis_version": VERSION,
        "provenance": {
            "stage30_baseline_commit": BASELINE,
            "analysis_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "analysis_code_sha256": sha256(Path(__file__)),
            "stage30_contract_code_sha256": sha256(REPO / "ml/preprocessing/contract.py"),
            "stage30_config_sha256": sha256(contract.CONFIG),
            "stage2_run_id": contract.RUN_ID,
            "stage2_manifest_sha256": config["source"]["manifest_sha256"],
            "stage2_extraction_run_sha256": config["source"]["run_sha256"],
            "source_npz_sha256_by_video": dict(sorted(npz_hashes.items())),
            "dataset": "CAUCAFall V5; locally supplied, origin not independently authenticated",
            "split": "train", "subject_ids": list(contract.FROZEN_SUBJECTS["train"]),
            "python": platform.python_version(), "numpy": np.__version__,
        },
        "definitions": {
            "pose_availability": "detected frames / all source frames; not fall-classifier accuracy",
            "missing_run": "maximal consecutive pose_detected=false frames; inclusive frame indices",
            "missing_run_duration_ms": "timestamp_ms[end] - timestamp_ms[start] + 50 ms source frame period",
            "video_duration_ms": "last timestamp_ms + 50 ms source frame period",
            "progress_bins": "missing frame midpoint (timestamp_ms+25 ms)/video_duration_ms in five fixed descriptive intervals [0,.2),...,[.8,1]",
            "run_progress": "run start timestamp/video duration and end-exclusive timestamp/video duration",
            "quantiles": "NumPy percentile(method=linear) on observed finite values; null for empty samples; p99 exploratory when n is small",
            "std": "population standard deviation (ddof=0); frames within a video are correlated",
            "geometry": "2-D Euclidean distances using raw image-relative x/y, indices 11/12 shoulders and 23/24 hips; no z meters",
            "class_comparison": "descriptive video-level equal-weight means; pooled availability separately; no causality or predictive-performance claim",
            "empty_runs": "shortest/median/longest are null for a no-missing video; class mean longest treats such videos as zero",
            "no_policy_thresholds": True,
        },
        "inventory": dict(subjects=len({r["subject_id"] for r in videos}),
                          **grouped_summary(videos)),
        "subjects": subject_rows, "activities": activity_rows,
        "missingness_shortcut_risk_diagnostic": {
            "classes": class_rows,
            "video_level_missing_fraction": {name: distribution([r["missing_fraction"] for r in videos if r["video_label"] == name])
                                             for name in ("fall", "non_fall")},
            "video_level_longest_run_ms_including_zero": {name: distribution([r["longest_missing_run_ms"] or 0
                                                                              for r in videos if r["video_label"] == name])
                                                        for name in ("fall", "non_fall")},
            "video_level_run_count": {name: distribution([r["missing_run_count"] for r in videos if r["video_label"] == name])
                                      for name in ("fall", "non_fall")},
            "interpretation": "Descriptive Train-only shortcut-risk diagnostic; not causal evidence, model accuracy, or a predictive-performance experiment.",
        },
        "missing_runs": {
            "count": len(runs),
            "duration_frames": distribution([r["frames"] for r in runs], RUN_PERCENTILES),
            "duration_ms": distribution([r["duration_ms"] for r in runs], RUN_PERCENTILES),
            "duration_quantile_sample_warning": len(runs) < 20,
            "missing_frame_progress_quintile_counts": [sum(r["missing_progress_quintile_frames"][i] for r in videos)
                                                       for i in range(5)],
            "videos_with_no_missing_runs": sum(r["missing_run_count"] == 0 for r in videos),
            "videos_with_zero_detected_poses": sum(r["zero_detected_poses"] for r in videos),
            "video_level_longest_duration_ms_including_zero": distribution(longest, RUN_PERCENTILES),
        },
        "visibility": {
            "detected_frames": int(observed_visibility.shape[0]),
            "per_frame_mean_across_33": distribution(np.concatenate(frame_mean_parts)),
            "per_frame_min_across_33": distribution(np.concatenate(frame_min_parts)),
            "per_landmark_csv": "landmark_visibility.csv",
        },
        "raw_channels": channels,
        "body_geometry": {
            "left_right_hip_midpoint_finite_frames": int(observed_visibility.shape[0]),
            "left_right_shoulder_midpoint_finite_frames": int(observed_visibility.shape[0]),
            "quantities": geometry,
            "near_zero_policy_tolerance": None,
        },
        "integrity": {
            "validated_npzs": len(videos), "schema_validation": "pose_raw.load_validated, allow_pickle=False",
            "detected_nan_count": 0, "detected_inf_count": 0,
            "finite_values_on_missing_frames": 0, "shape_dtype_timestamp_violations": 0,
            "note": "Any violation stops analysis before publication; zeros here follow completed strict validation.",
        },
        "status": "analysis_only_pending_external_gate_review",
    }
    files = {"summary.json": _json_bytes(summary),
             "videos.csv": _csv_bytes(videos, list(videos[0])),
             "missing_runs.csv": _csv_bytes(runs, list(runs[0]) if runs else
                                             ["subject_id", "source_video", "video_label", "split", "start_frame", "end_frame", "frames", "duration_ms", "start_ms", "end_exclusive_ms", "start_progress", "end_progress"]),
             "landmark_visibility.csv": _csv_bytes(landmark_rows, list(landmark_rows[0])),
             "subjects.csv": _csv_bytes(subject_rows, list(subject_rows[0])),
             "activities.csv": _csv_bytes(activity_rows, list(activity_rows[0])),
             "classes.csv": _csv_bytes(class_rows, list(class_rows[0]))}
    files["REPORT.md"] = _report(summary, videos).encode("utf-8")
    return files


def _report(summary: dict[str, Any], videos: list[dict[str, Any]]) -> str:
    inv = summary["inventory"]
    missing = summary["missing_runs"]
    by_class = {row["video_label"]: row for row in summary["missingness_shortcut_risk_diagnostic"]["classes"]}
    zero = [r["source_video"] for r in videos if r["zero_detected_poses"]]
    geometry = summary["body_geometry"]["quantities"]
    return (f"# Stage 3.1 Train-only pose characterization\n\n"
            f"Analysis version: `{VERSION}`. Source: official Stage 2 run `{contract.RUN_ID}`. "
            f"Stage 3.0 baseline: `{BASELINE}`. Status: pending external Gate Review.\n\n"
            "All results are descriptive and Train-only. Pose availability is MediaPipe pose return rate, "
            "not fall-detection accuracy or robustness. Frame/joint samples within videos are correlated.\n\n"
            f"## Inventory and missingness\n\n"
            f"{inv['subjects']} subjects, {inv['videos']} videos, {inv['frames']} frames; "
            f"{inv['detected_frames']} detected and {inv['missing_frames']} missing. "
            f"Pooled pose availability: {inv['pooled_pose_availability']:.4%}.\n\n"
            f"{missing['count']} missing runs; median {missing['duration_ms']['p50']} ms, "
            f"p95 {missing['duration_ms']['p95']} ms, maximum {missing['duration_ms']['max']} ms. "
            f"Zero-detected-pose Train videos: {len(zero)} ({', '.join(zero) if zero else 'none'}). "
            f"Missing-frame counts by progress quintile: {missing['missing_frame_progress_quintile_counts']}.\n\n"
            "## Missingness shortcut-risk diagnostic\n\n"
            f"Fall videos: {by_class['fall']['videos']}, pooled availability "
            f"{by_class['fall']['pooled_pose_availability']:.4%}, mean per-video missing fraction "
            f"{by_class['fall']['mean_video_missing_fraction']:.4f}, mean longest gap "
            f"{by_class['fall']['mean_video_longest_missing_run_ms']:.1f} ms, mean run count "
            f"{by_class['fall']['mean_video_missing_run_count']:.2f}.\n\n"
            f"Non-fall videos: {by_class['non_fall']['videos']}, pooled availability "
            f"{by_class['non_fall']['pooled_pose_availability']:.4%}, mean per-video missing fraction "
            f"{by_class['non_fall']['mean_video_missing_fraction']:.4f}, mean longest gap "
            f"{by_class['non_fall']['mean_video_longest_missing_run_ms']:.1f} ms, mean run count "
            f"{by_class['non_fall']['mean_video_missing_run_count']:.2f}.\n\n"
            "These differences cannot establish causality or a predictive shortcut. "
            "A later mask-only ablation would need a separately approved evaluation.\n\n"
            "## Visibility and geometry\n\n"
            f"Visibility was summarized for {summary['visibility']['detected_frames']} detected frames; "
            "see `landmark_visibility.csv` for all 33 joints and `summary.json` for per-frame summaries. "
            "No visibility threshold was selected.\n\n"
            f"Raw x/y shoulder width median: {geometry['shoulder_width_xy']['p50']:.6g}; "
            f"hip width median: {geometry['hip_width_xy']['p50']:.6g}; "
            f"shoulder-to-hip midpoint torso median: {geometry['shoulder_to_hip_midpoint_torso_xy']['p50']:.6g}. "
            "These are diagnostic image-relative 2-D distances, not chosen normalization scales. "
            "Exact-zero counts and full distributions are in `summary.json`.\n\n"
            "## Integrity and limits\n\n"
            "All selected NPZs passed the frozen raw schema validation. Reported zero violation counts "
            "follow from that validation; no raw values were repaired, clipped or rejected by range. "
            "The source download's V5 identity was locally supplied and not independently authenticated. "
            "All Stage 3.0 preprocessing decisions remain `UNDECIDED`. No Validation/Test arrays were loaded.\n")


def publish(files: dict[str, bytes], output: Path = OUTPUT) -> None:
    """Publish one complete new analysis directory; never overwrite a prior run."""
    contract.validate_output_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".stage31-", dir=output.parent))
    try:
        for name, content in files.items():
            if name != Path(name).name:
                raise ValueError("Invalid artifact filename")
            (staging / name).write_bytes(content)
        if output.exists():
            raise FileExistsError(f"Refusing existing output: {output}")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", help="Stage 3.1 accepts train only")
    args = parser.parse_args()
    files = analyze(args.split)
    publish(files)
    print(json.dumps({"output": str(OUTPUT), "files": sorted(files), "status": "analysis_only"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
