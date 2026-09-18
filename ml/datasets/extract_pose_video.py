# /// script
# requires-python = "==3.13.15"
# dependencies = ["mediapipe==0.10.35", "opencv-contrib-python==4.12.0.88", "numpy==2.2.6"]
# ///
"""Stage 2.2: extract exactly one permitted CAUCAFall AVI, never held-out 6/7."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import tempfile
import platform
import subprocess
import uuid
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .assign_dataset_split import assign_rows
    from .pose_raw import (ExtractionError, SCHEMA_VERSION, TIMESTAMP_METHOD,
                           encode_pose, publish_npz, timestamp_ms)
    from .pose_runtime import MODEL_URL, MODEL_SHA256, missing_runs, pose_options
except ImportError:
    from assign_dataset_split import assign_rows
    from pose_raw import (ExtractionError, SCHEMA_VERSION, TIMESTAMP_METHOD,
                          encode_pose, publish_npz, timestamp_ms)
    from pose_runtime import MODEL_URL, MODEL_SHA256, missing_runs, pose_options

REPO = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO / "data/raw/caucafall_v5/CAUCAFall"
OUTPUT_ROOT = REPO / "data/interim/caucafall_v5/pose_raw_v1"
RESULT_ROOT = REPO / "artifacts/pose_extraction/single_video"
MODEL = REPO / "ml/checkpoints/pose_landmarker_full.task"
INVENTORY = REPO / "artifacts/dataset_inspection/inventory.csv"
SPLIT_CONFIG = REPO / "configs/dataset_split.json"
LOGGER = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_metadata(source: Path) -> dict[str, Any]:
    """Check identity before reading source bytes; frozen 6/7 are always refused."""
    source = source.absolute()
    try:
        relative = source.relative_to(SOURCE_ROOT)
        if len(relative.parts) != 3 or relative.suffix.lower() != ".avi":
            raise ValueError("Expected Subject.N/activity/video.avi inside CAUCAFall root")
        if relative.parts[0] in ("Subject.6", "Subject.7"):
            raise ValueError("Held-out Subjects 6/7 are forbidden in Stage 2.2")
        if source.resolve() != source or not source.is_file():
            raise ValueError("Source must be an existing canonical path without symlink redirection")
        config = json.loads(SPLIT_CONFIG.read_text())
        with INVENTORY.open(newline="") as handle:
            rows = assign_rows(list(csv.DictReader(handle)), config)
        matches = [row for row in rows if row["relative_video_path"] == relative.as_posix()]
        if len(matches) != 1 or matches[0]["split"] == "test":
            raise ValueError("Source must match one inspected train/validation video")
        row = matches[0]
        dimensions = [float(row[key]) for key in ("width", "height", "frame_count")]
        if any(not math.isfinite(v) or v <= 0 or not v.is_integer() for v in dimensions):
            raise ValueError("Expected positive integral dimensions and frame count")
        return dict(subject_id=int(row["subject"].split(".")[1]),
                    original_activity=row["activity"], binary_label=row["label"], split=row["split"],
                    source_relative_video_path=relative.as_posix(), source_fps=float(row["fps"]),
                    width=int(dimensions[0]), height=int(dimensions[1]),
                    expected_frame_count=int(dimensions[2]))
    except (OSError, ValueError, KeyError) as exc:
        raise ExtractionError("invalid_source_metadata", str(exc)) from exc


def verify_capture(capture: Any, metadata: dict[str, Any], cv2: Any) -> None:
    """Compare current OpenCV metadata with the inspected source, without defaults."""
    if not capture.isOpened():
        raise ExtractionError("video_decode_failure", "Cannot open source video")
    for key, prop in (("source_fps", cv2.CAP_PROP_FPS), ("width", cv2.CAP_PROP_FRAME_WIDTH),
                      ("height", cv2.CAP_PROP_FRAME_HEIGHT),
                      ("expected_frame_count", cv2.CAP_PROP_FRAME_COUNT)):
        value = capture.get(prop)
        if not math.isfinite(value) or value <= 0 or value != metadata[key]:
            raise ExtractionError("invalid_source_metadata", f"{key}: {value} != {metadata[key]}")


def decode_records(capture: Any, landmarker: Any, metadata: dict[str, Any],
                   progress: dict[str, Any], cv2: Any, mp: Any) -> dict[str, np.ndarray]:
    """Process sequential source frames; failures never become no-pose frames."""
    detected, landmarks = [], []
    expected, fps = metadata["expected_frame_count"], metadata["source_fps"]
    while True:
        index = len(detected)
        try:
            ok, frame = capture.read()
        except Exception as exc:
            raise ExtractionError("video_decode_failure", str(exc), index) from exc
        if not ok:
            if index != expected:
                raise ExtractionError("video_decode_failure",
                                      f"Decoded {index}; expected {expected}", index)
            break
        progress["decoded_frame_count"] += 1
        if (frame is None or getattr(frame, "shape", None) != (metadata["height"], metadata["width"], 3)
                or index >= expected):
            raise ExtractionError("video_decode_failure", "Invalid frame shape or extra frame", index)
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            raise ExtractionError("video_decode_failure", str(exc), index) from exc
        try:
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms(index, fps))
        except Exception as exc:
            raise ExtractionError("inference_runtime_failure", str(exc), index) from exc
        try:
            present, values = encode_pose(result)
        except ExtractionError as exc:
            exc.frame_index = index
            raise
        detected.append(present)
        landmarks.append(values)
        progress["extracted_frame_count"] += 1
        progress["pose_detected_count" if present else "pose_missing_count"] += 1
    return dict(frame_index=np.arange(expected, dtype=np.int32),
                timestamp_ms=np.array([timestamp_ms(i, fps) for i in range(expected)], dtype=np.int64),
                pose_detected=np.array(detected, dtype=np.bool_),
                landmarks=np.array(landmarks, dtype=np.float32))


def extract_arrays(source: Path, model: Path, metadata: dict[str, Any],
                   progress: dict[str, Any], cv2: Any, mp: Any) -> dict[str, np.ndarray]:
    """Own and release one capture and one fresh tracker."""
    capture = cv2.VideoCapture()
    try:
        try:
            capture.open(str(source))
            verify_capture(capture, metadata, cv2)
        except cv2.error as exc:
            raise ExtractionError("video_decode_failure", str(exc)) from exc
        try:
            with mp.tasks.vision.PoseLandmarker.create_from_options(pose_options(model, mp)) as tracker:
                return decode_records(capture, tracker, metadata, progress, cv2, mp)
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError("inference_runtime_failure", str(exc)) from exc
    finally:
        capture.release()


def output_path(root: Path, relative: Path) -> Path:
    """Reject raw-tree output, symlink redirection, and existing destinations."""
    path = root.absolute() / relative
    if (path.resolve() != path or path.is_relative_to(REPO / "data/raw")
            or path.exists() or path.is_symlink()):
        raise ExtractionError("output_persistence_failure", f"Unsafe or existing output: {path}")
    return path


def runtime_provenance(model: Path) -> dict[str, Any]:
    """Small per-video evidence, not the full dataset manifest lifecycle."""
    git = lambda *args: subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()
    sources = [INVENTORY, SPLIT_CONFIG, REPO / "docs/dataset_protocol.md",
               REPO / "docs/pose_extraction_contract.md", Path(__file__),
               REPO / "ml/datasets/pose_raw.py", REPO / "ml/datasets/pose_runtime.py",
               REPO / "ml/datasets/assign_dataset_split.py", REPO / "ml/datasets/inspect_caucafall.py"]
    return dict(contract_version=1, dataset_protocol_version=2,
                git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain")),
                file_sha256={p.relative_to(REPO).as_posix(): sha256(p) for p in sources},
                python=platform.python_version(), platform=platform.platform(),
                packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                model_identity="PoseLandmarker Full float16 v1", model_url=MODEL_URL,
                model_sha256=sha256(model), running_mode="VIDEO", delegate="CPU", num_poses=1,
                detection_confidence=.5, presence_confidence=.5, tracking_confidence=.5,
                segmentation=False, timestamp_method=TIMESTAMP_METHOD,
                source_root=str(SOURCE_ROOT), source_download_provenance="Locally supplied; unverified",
                input_policy="Original resolution; BGR to RGB; no resampling; fresh tracker per video")


def publish_video_pair(output: Path, report: Path, arrays: dict[str, np.ndarray],
                       metadata: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Stage both files; publish validated NPZ, then result marker; roll back errors.

    Two paths cannot be renamed atomically together. An uncatchable process kill
    between publication steps can leave an orphan NPZ without a completion marker.
    Consumers require the NPZ and its matching complete result; neither alone
    establishes completion. Crash recovery/resume remains deferred.
    """
    try:
        for path in (output, report):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Refusing existing output: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".pose-pair-") as npz_dir, \
                tempfile.TemporaryDirectory(dir=report.parent, prefix=".pose-pair-") as json_dir:
            staged_npz, staged_report = Path(npz_dir) / "pose.npz", Path(json_dir) / "result.json"
            committed = False
            try:
                reloaded = publish_npz(staged_npz, arrays, metadata["source_fps"],
                                       metadata["expected_frame_count"])
                complete = dict(result, status="complete", error_type=None, error_message=None,
                                error_frame_index=None,
                                arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype))
                                        for k, v in reloaded.items()},
                                missing_frame_intervals=missing_runs(
                                    np.flatnonzero(~reloaded["pose_detected"]).tolist()))
                staged_report.write_text(json.dumps(complete, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
                if json.loads(staged_report.read_text(encoding="utf-8")) != complete:
                    raise ValueError("Result record failed reload validation")
                # Persist both staged files before linking either final destination.
                for staged in (staged_npz, staged_report):
                    with staged.open("rb") as handle:
                        os.fsync(handle.fileno())
                os.link(staged_npz, output)
                os.link(staged_report, report)  # Completion marker is published LAST.
                committed = True
                return complete
            finally:
                if not committed:
                    # samefile also covers an interrupt immediately after link(),
                    # without ever deleting another writer's existing destination.
                    # Remove our completion marker before removing our NPZ.
                    for staged, final in ((staged_report, report), (staged_npz, output)):
                        if staged.exists() and final.exists() and not final.is_symlink() \
                                and staged.samefile(final):
                            final.unlink()
    except (OSError, ValueError) as exc:
        raise ExtractionError("output_persistence_failure", str(exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Exactly one AVI path, relative to cwd or absolute")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result: dict[str, Any] = dict(schema_version=SCHEMA_VERSION, run_id=str(uuid.uuid4()),
                                dataset_name="CAUCAFall", dataset_version="V5", status="running",
                                requested_source=str(args.source), decoded_frame_count=0,
                                extracted_frame_count=0, pose_detected_count=0, pose_missing_count=0,
                                timestamp_method=TIMESTAMP_METHOD)
    report = None
    phase = "invalid_source_metadata"
    try:
        metadata = source_metadata(args.source)
        result.update(metadata)
        relative = Path(metadata["source_relative_video_path"])
        output = output_path(args.output_root, relative.with_suffix(".npz"))
        report = output_path(RESULT_ROOT, relative.with_suffix(".json"))
        result.update(output_root=str(args.output_root.absolute()),
                      output_relative_path=relative.with_suffix(".npz").as_posix(),
                      source_sha256=sha256(args.source))
        phase = "provenance_mismatch"
        if sha256(args.model) != MODEL_SHA256:
            raise ExtractionError(phase, "Model SHA-256 differs from approved Full float16 v1")
        result["provenance"] = runtime_provenance(args.model)
        phase = "inference_runtime_failure"
        import cv2
        import mediapipe as mp
        actual = (platform.python_version(), mp.__version__,
                  importlib.metadata.version("opencv-contrib-python"), np.__version__)
        if actual != ("3.13.15", "0.10.35", "4.12.0.88", "2.2.6"):
            raise ExtractionError("provenance_mismatch", f"Unexpected runtime: {actual}")
        result["provenance"]["opencv_runtime"] = cv2.__version__
        arrays = extract_arrays(args.source, args.model, metadata, result, cv2, mp)
        phase = "provenance_mismatch"
        if sha256(args.model) != MODEL_SHA256:
            raise ExtractionError(phase, "Model changed during extraction")
        if sha256(args.source) != result["source_sha256"]:
            raise ExtractionError("provenance_mismatch", "Source changed during extraction")
        phase = "output_persistence_failure"
        result = publish_video_pair(output, report, arrays, metadata, result)
    except KeyboardInterrupt:
        result.update(status="incomplete", error_type="interrupted_run", error_message="Interrupted",
                      error_frame_index=None)
    except (ExtractionError, OSError, ValueError, ImportError, subprocess.SubprocessError) as exc:
        result.update(status="failed", error_type=getattr(exc, "error_type", phase),
                      error_message=str(exc), error_frame_index=getattr(exc, "frame_index", None))
    LOGGER.info(json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
