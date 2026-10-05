# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.2.6", "mediapipe==0.10.35", "opencv-contrib-python==4.12.0.88", "paho-mqtt==2.1.0"]
# ///
"""Stage 6 OpenCV/MediaPipe edge publisher, camera index or authorized CAUCAFall AVI."""

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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.mqtt import MQTTConnection, identifier, validate_pose
from ml.datasets.extract_pose_video import source_metadata, verify_capture
from ml.datasets.pose_raw import encode_pose, timestamp_ms
from ml.datasets.pose_runtime import MODEL_SHA256, pose_options
from ml.preprocessing.pose_sequence import PreprocessingConfig, StreamingPreprocessor

REPO = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Nonnegative camera index or canonical non-test AVI path")
    parser.add_argument("--source-id", default=os.getenv("SOURCE_ID", "edge-01"))
    parser.add_argument("--sequence-id", default=uuid.uuid4().hex)
    parser.add_argument("--pose-model", type=Path,
                        default=Path(os.getenv("POSE_MODEL_PATH", str(REPO / "ml/checkpoints/pose_landmarker_full.task"))))
    parser.add_argument("--camera-fps", type=float, default=float(os.getenv("CAMERA_FPS", "20")))
    parser.add_argument("--unpaced", action="store_true", help="Decode AVI as fast as possible for smoke tests")
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
    capture = cv2.VideoCapture(source)
    if not camera:
        verify_capture(capture, metadata, cv2)
    elif not capture.isOpened():
        raise ValueError("Camera unavailable")
    fps = metadata["source_fps"] if metadata else args.camera_fps
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid source/camera FPS")
    connection = MQTTConnection()
    frames = detected_frames = 0
    preprocessor = StreamingPreprocessor(config)
    common = dict(schema_version=1, source_id=source_id, sequence_id=sequence_id)
    started = time.monotonic()
    last_mp_time = last_epoch_time = -1
    connection.start()
    topic = f"{connection.prefix}/pose/{source_id}"
    try:
        connection.publish(topic, dict(common, message_type="sequence_start"))
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
                if not camera and not args.unpaced:
                    time.sleep(max(0, started + frames/fps - time.monotonic()))
        connection.publish(topic, dict(common, message_type="sequence_end"))
    finally:
        capture.release()
        connection.close()
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
    main()
