# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.2.6", "mediapipe==0.10.35", "opencv-contrib-python==4.12.0.88", "paho-mqtt==2.1.0"]
# ///
"""OpenCV/MediaPipe edge publisher with optional cloud prediction preview."""

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.mqtt import MQTTConnection, identifier, validate_pose
from ml.datasets.extract_pose_video import source_metadata, verify_capture
from ml.datasets.pose_raw import encode_pose, timestamp_ms
from ml.datasets.pose_runtime import MODEL_SHA256, pose_options
from ml.preprocessing.pose_sequence import PreprocessingConfig, StreamingPreprocessor

REPO = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


class LatestPrediction:
    """MQTT callback stores an immutable snapshot; OpenCV stays on the main thread."""

    def __init__(self, source_id: str, sequence_id: str) -> None:
        self.source_id, self.sequence_id = source_id, sequence_id
        self._lock = Lock()
        self._value: tuple[float, str, int] | None = None

    def on_message(self, client: Any, userdata: Any, message: Any) -> None:
        try:
            prediction = json.loads(message.payload)
            if not isinstance(prediction, dict):
                raise ValueError("Expected prediction object")
            if (prediction.get("source_id") != self.source_id
                    or prediction.get("sequence_id") != self.sequence_id):
                return
            probability = prediction.get("fall_probability")
            label, length = prediction.get("predicted_label"), prediction.get("buffer_length")
            if (type(prediction.get("schema_version")) is not int or prediction["schema_version"] != 1
                    or type(probability) not in (int, float) or not 0 <= probability <= 1
                    or label not in ("fall", "non_fall") or type(length) is not int or length <= 0):
                raise ValueError("Invalid prediction display fields")
            with self._lock:
                self._value = (float(probability), label, length)
        except (ValueError, TypeError, UnicodeError) as exc:
            LOGGER.warning("Ignored malformed prediction: %s", exc)

    def snapshot(self) -> tuple[float, str, int] | None:
        with self._lock:
            return self._value


def preview_frame(frame: Any, prediction: tuple[float, str, int] | None, cv2: Any,
                  *, pose_detected: bool) -> bool:
    """Draw only after pose publication; return True when the user presses q."""
    height, width = frame.shape[:2]
    scale = min(width / 640, height / 480)
    bottom = height / scale
    panel_top = bottom - 262
    white, muted = (242, 244, 246), (173, 184, 193)
    green, red, amber = (105, 218, 105), (82, 82, 245), (90, 202, 245)
    connected = prediction is not None  # Received a prediction this sequence; no heartbeat semantics.
    status_color = amber if not connected else (green if prediction[1] == "non_fall" else red)

    def point(x: float, y: float) -> tuple[int, int]:
        return round(x * scale), round(y * scale)

    def text(value: str, x: float, y: float, size: float = .5,
             color: tuple[int, int, int] = white, thickness: int = 1) -> None:
        cv2.putText(frame, value, point(x, y), cv2.FONT_HERSHEY_SIMPLEX, size * scale,
                    color, max(1, round(thickness * scale)), cv2.LINE_AA)

    overlay = frame.copy()
    background = (22, 28, 34)
    cv2.rectangle(overlay, (0, 0), (width - 1, round(48 * scale)), background, -1)
    cv2.rectangle(overlay, point(16, panel_top), point(382, panel_top + 224), background, -1)
    cv2.rectangle(overlay, (0, round((bottom - 34) * scale)), (width - 1, height - 1), background, -1)
    cv2.addWeighted(overlay, .72, frame, .28, 0, dst=frame)
    text("FALL DETECTION SYSTEM", 16, 31, .65, thickness=2)
    text("CLOUD", width / scale - 94, 31, .5)
    cv2.circle(frame, point(width / scale - 112, 26), max(1, round(5 * scale)), green if connected else amber, -1)
    cv2.rectangle(frame, point(16, panel_top), point(20, panel_top + 224), status_color, -1)
    if connected:
        text("STATUS", 32, panel_top + 31, .42, muted)
        text("NORMAL" if prediction[1] == "non_fall" else "FALL", 169, panel_top + 35, .85, status_color, 2)
    else:
        text("INITIALIZING CLOUD MODEL...", 32, panel_top + 32, .5, amber, 2)
    text("Fall Probability:", 32, panel_top + 73, color=muted)
    text(f"{prediction[0]:.1%}" if connected else "--", 265, panel_top + 73, .6, thickness=2)
    cv2.rectangle(frame, point(32, panel_top + 89), point(366, panel_top + 101), (52, 63, 74), -1)
    fill_width = round(334 * prediction[0]) if connected else 0
    if fill_width > 0:
        cv2.rectangle(frame, point(32, panel_top + 89), point(32 + fill_width, panel_top + 101), status_color, -1)
    text("Buffer:", 32, panel_top + 133, color=muted)
    text(f"{prediction[2]} frames" if connected else "--", 170, panel_top + 133)
    text("Pose:", 32, panel_top + 164, color=muted)
    text("DETECTED" if pose_detected else "MISSING", 170, panel_top + 164, color=green if pose_detected else amber)
    text("Cloud:", 32, panel_top + 195, color=muted)
    text("CONNECTED" if connected else "WAITING", 170, panel_top + 195, color=green if connected else amber)
    text("Press Q to exit", 16, bottom - 12)
    cv2.imshow("Cloud Fall Detection", frame)
    return cv2.waitKey(1) & 0xFF == ord("q")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Nonnegative camera index or canonical non-test AVI path")
    parser.add_argument("--source-id", default=os.getenv("SOURCE_ID", "edge-01"))
    parser.add_argument("--sequence-id", default=uuid.uuid4().hex)
    parser.add_argument("--pose-model", type=Path,
                        default=Path(os.getenv("POSE_MODEL_PATH", str(REPO / "ml/checkpoints/pose_landmarker_full.task"))))
    parser.add_argument("--camera-fps", type=float, default=float(os.getenv("CAMERA_FPS", "20")))
    parser.add_argument("--unpaced", action="store_true", help="Decode AVI as fast as possible for smoke tests")
    parser.add_argument("--preview", action="store_true", help="Show video with latest cloud prediction; q to quit")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    source_id, sequence_id = identifier(args.source_id), identifier(args.sequence_id)
    camera = args.source.isdecimal()
    # This identity guard runs before importing/OpenCV-opening any dataset media.
    metadata = None if camera else source_metadata(Path(args.source).absolute())
    source = int(args.source) if camera else str(Path(args.source).absolute())
    selected = json.loads((REPO / "artifacts/evaluation/stage5/final_ml_config.json").read_text())
    config = PreprocessingConfig(**selected["training_config"]["preprocessing"])
    if config.normalize_pose or config.max_forward_fill_frames != 5:
        raise ValueError("Stage 6 requires frozen no-normalization/five-frame-fill configuration")
    with args.pose_model.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != MODEL_SHA256:
            raise ValueError("PoseLandmarker model identity mismatch")
    import cv2
    import mediapipe as mp
    fps = metadata["source_fps"] if metadata else args.camera_fps
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid source/camera FPS")
    latest = LatestPrediction(source_id, sequence_id)
    connection = None
    sequence_started = completed = False
    frames = detected_frames = 0
    preprocessor = StreamingPreprocessor(config)
    common = dict(schema_version=1, source_id=source_id, sequence_id=sequence_id)
    last_mp_time = last_epoch_time = -1
    capture = cv2.VideoCapture(source)
    try:
        if not camera:
            verify_capture(capture, metadata, cv2)
        elif not capture.isOpened():
            raise ValueError("Camera unavailable")
        connection = MQTTConnection(on_message=latest.on_message if args.preview else None)
        if args.preview:
            connection.subscription = f"{connection.prefix}/prediction/{source_id}"
        started = time.monotonic()
        connection.start()
        topic = f"{connection.prefix}/pose/{source_id}"
        connection.publish(topic, dict(common, message_type="sequence_start"))
        sequence_started = True
        with mp.tasks.vision.PoseLandmarker.create_from_options(pose_options(args.pose_model, mp)) as tracker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    if camera or frames != metadata["expected_frame_count"]:
                        raise ValueError("Camera offline or AVI decoded-frame count mismatch")
                    break
                if not camera and frames >= metadata["expected_frame_count"]:
                    raise ValueError("Unexpected extra AVI frame")
                media_time = timestamp_ms(frames, fps) if not camera else max(last_mp_time+1, int((time.monotonic()-started)*1000))
                last_mp_time = media_time
                transmitted_time = media_time if not camera else max(last_epoch_time+1, time.time_ns()//1_000_000)
                last_epoch_time = transmitted_time
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detected, landmarks = encode_pose(tracker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), media_time))
                features, _ = preprocessor.step(landmarks, detected)
                message = dict(common, message_type="frame", frame_index=frames,
                               timestamp_ms=transmitted_time, timestamp_kind="unix_epoch" if camera else "clip_relative",
                               fps=fps, pose_detected=detected, features=features.tolist())
                connection.publish(topic, validate_pose(message))
                frames += 1
                detected_frames += int(detected)
                if args.preview and preview_frame(frame, latest.snapshot(), cv2, pose_detected=detected):
                    break
                if not camera and not args.unpaced:
                    time.sleep(max(0, started + frames/fps - time.monotonic()))
        completed = True
    except KeyboardInterrupt:
        completed = True
        LOGGER.info("Edge interrupted; stopping")
    finally:
        try:
            if completed and sequence_started:
                connection.publish(topic, dict(common, message_type="sequence_end"))
        finally:
            try:
                capture.release()
            finally:
                try:
                    if connection is not None:
                        connection.close()
                finally:
                    if args.preview:
                        cv2.destroyAllWindows()
    summary = dict(source_id=source_id, sequence_id=sequence_id, frames_published=frames,
                   pose_detected_frames=detected_frames, missing_pose_frames=frames-detected_frames,
                   source_fps=fps, timestamp_kind="unix_epoch" if camera else "clip_relative",
                   authorized_source=metadata["source_relative_video_path"] if metadata else "camera",
                   errors=0)
    LOGGER.info("edge_summary %s", json.dumps(summary))
    if args.summary:
        with args.summary.open("x") as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)
            handle.write("\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info("Edge interrupted; stopped")
