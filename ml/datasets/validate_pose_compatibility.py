# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["mediapipe==0.10.35", "opencv-contrib-python==4.12.0.88", "numpy==2.2.6"]
# ///
"""Stage 1.3: fixed ten-clip pose smoke test, retaining summaries only."""

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

try:
    from .inspect_caucafall import DEFAULT_ROOT, activity_label, validate_output
    from .pose_runtime import MODEL_URL, missing_runs, pose_options
except ImportError:
    from inspect_caucafall import DEFAULT_ROOT, activity_label, validate_output
    from pose_runtime import MODEL_URL, missing_runs, pose_options

# Fixed before running: all ten activities, five subjects; not a statistical sample.
SAMPLES = [(1, "Fall forward"), (2, "Fall backwards"), (3, "Fall left"),
           (4, "Fall right"), (5, "Fall sitting"), (1, "Sit down"),
           (2, "Kneel"), (3, "Pick up object"), (4, "Walk"), (5, "Hop")]
# Standard body connections, excluding face details to keep overlays legible.
CONNECTIONS = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
               (23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),
               (28,30),(30,32),(15,17),(15,19),(16,18),(16,20)]


def compatibility_decision(rows: list[dict[str, Any]]) -> str:
    # Mechanical gate only; the report must also include visual review.
    if not rows or any(r["errors"] or not r["valid_pose_frames"] or
                       r["invalid_landmark_frames"] or
                       r["decoded_frames"] != r["metadata_frames"] for r in rows):
        return "FAIL"
    if any(r["frames_without_pose"] for r in rows):
        return "PASS WITH WARNINGS"
    return "PASS"


def save_overlay(path: Path, frame: Any, landmarks: Any, caption: str, cv2: Any) -> None:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    if landmarks:
        points = [(round(lm.x * w), round(lm.y * h)) for lm in landmarks]
        for a, b in CONNECTIONS:
            if landmarks[a].visibility >= .5 and landmarks[b].visibility >= .5:
                cv2.line(canvas, points[a], points[b], (0, 220, 0), 2)
        for lm, point in zip(landmarks, points):
            if lm.visibility >= .5:
                cv2.circle(canvas, point, 3, (0, 0, 255), -1)
    cv2.rectangle(canvas, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
    if path.is_symlink() or not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 80]):
        raise OSError(f"Could not safely write diagnostic: {path}")


def inspect_pose(root: Path, path: Path, model: Path, output: Path,
                 confidence: float, mp: Any, cv2: Any) -> dict[str, Any]:
    subject, activity, _ = path.relative_to(root).parts
    row = dict(subject=subject, activity=activity, label=activity_label(activity),
               relative_video_path=path.relative_to(root).as_posix(), decoded_frames=0,
               valid_pose_frames=0, frames_without_pose=0, invalid_landmark_frames=0,
               metadata_frames=None, fps=None, errors=[])
    missing = []
    visibility = []
    diagnostic_names = []
    saved_missing = False
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        expected = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(expected) or expected <= 0:
            raise ValueError("Invalid FPS/frame count")
        row.update(fps=fps, metadata_frames=expected)
        options = pose_options(model, mp, confidence)
        # New tracker per clip prevents carrying subject state between unrelated videos.
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None or not frame.size:
                    break
                index = row["decoded_frames"]
                row["decoded_frames"] += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(index * 1000 / fps))
                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                valid = landmarks is not None and len(landmarks) == 33 and all(
                    math.isfinite(v) for lm in landmarks for v in (lm.x, lm.y, lm.z, lm.visibility))
                if valid:
                    row["valid_pose_frames"] += 1
                    visibility.extend(lm.visibility for lm in landmarks)
                else:
                    row["frames_without_pose"] += 1
                    missing.append(index)
                    if landmarks is not None:
                        row["invalid_landmark_frames"] += 1
                # Three clips, two fixed positions each, plus first missing frame per clip.
                representative = activity in ("Fall forward", "Kneel", "Walk") and index in (
                    int(expected * .25), int(expected * .75))
                first_missing = not valid and not saved_missing
                if representative or first_missing:
                    name = f'{subject}_{activity.replace(" ", "_")}_{index:04d}.jpg'
                    save_overlay(output / name, frame, landmarks if valid else None,
                                 f'{subject} {activity} frame {index} | pose={valid}', cv2)
                    diagnostic_names.append(name)
                    if first_missing:
                        saved_missing = True
    except (ValueError, RuntimeError, cv2.error) as exc:
        row["errors"].append(str(exc))
    finally:
        capture.release()
    runs = missing_runs(missing)
    row.update(pose_detection_rate=row["valid_pose_frames"] / row["decoded_frames"]
               if row["decoded_frames"] else 0,
               visibility_min=min(visibility) if visibility else None,
               visibility_mean=sum(visibility) / len(visibility) if visibility else None,
               visibility_max=max(visibility) if visibility else None,
               missing_frame_intervals=runs,
               longest_missing_run=max((b-a+1 for a,b in runs), default=0),
               diagnostics=diagnostic_names)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model", type=Path, default=Path("ml/checkpoints/pose_landmarker_full.task"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/pose_compatibility"))
    parser.add_argument("--confidence", type=float, default=.5)
    args = parser.parse_args()
    if not args.root.is_dir() or not args.model.is_file():
        parser.error("Dataset root and downloaded official model must exist")
    if not 0 <= args.confidence <= 1:
        parser.error("Confidence must be in [0, 1]")
    validate_output(args.root, args.output)
    for name in ("pose_summary.json", "pose_inventory.csv"):
        if (args.output / name).is_symlink():
            parser.error("Refusing symlink report destination")
    videos = []
    for subject, activity in SAMPLES:
        matches = sorted((args.root / f"Subject.{subject}" / activity).glob("*.avi"))
        if len(matches) != 1:
            parser.error(f"Expected one AVI: Subject.{subject}/{activity}")
        videos.append(matches[0])
    import cv2
    import mediapipe as mp
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in videos:
        row = inspect_pose(args.root, path, args.model, args.output, args.confidence, mp, cv2)
        rows.append(row)
        print(f'{row["subject"]} {row["activity"]}: {row["valid_pose_frames"]}/{row["decoded_frames"]}', flush=True)
    total = sum(r["decoded_frames"] for r in rows)
    valid = sum(r["valid_pose_frames"] for r in rows)
    summary = dict(runtime=dict(python=platform.python_version(), mediapipe=mp.__version__,
                               opencv=cv2.__version__, platform=platform.platform()),
                   model_url=MODEL_URL, model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                   config=dict(mode="VIDEO", delegate="CPU", num_poses=1, confidence=args.confidence,
                               timestamps="round(zero_based_frame_index * 1000 / measured_fps)",
                               input="BGR to RGB; original resolution and FPS; no resampling"),
                   criterion="FAIL for decode/inference/landmark errors or any clip with zero valid poses; WARN for any missing poses; otherwise PASS. Visual review required separately.",
                   decision=compatibility_decision(rows), decoded_frames=total, valid_pose_frames=valid,
                   pose_detection_rate=valid/total if total else 0, videos=rows)
    (args.output / "pose_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    fields = [key for key in rows[0] if key not in ("missing_frame_intervals", "diagnostics", "errors")]
    with (args.output / "pose_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f'{summary["decision"]}: {valid}/{total} frames; visual review still required.')
    return 1 if summary["decision"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
