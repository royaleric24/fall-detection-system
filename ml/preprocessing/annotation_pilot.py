"""Stage 3.2f Train-only pilot freeze and sequential AVI integrity checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


REPO = Path(__file__).resolve().parents[2]
PARENT = "59052bab724fa501bc2938c822388356399046ea"
RUN_ID = "stage32f_train_pilot_v1"
ARTIFACT_ROOT = REPO / "artifacts/temporal_annotation" / RUN_ID
SOURCE_ROOT = REPO / "data/raw/caucafall_v5/CAUCAFall"
STAGE3 = REPO / "configs/stage3_contract.json"
MANIFEST = REPO / "artifacts/pose_extraction/runs/4df7dd5f-fb38-4908-8233-1a81deb8dc05/manifest.csv"
TRAIN_SUBJECTS = (1, 2, 3, 4, 8, 9)
FALL = ("Fall backwards", "Fall forward", "Fall left", "Fall right", "Fall sitting")
NON_FALL = ("Hop", "Kneel", "Pick up object", "Sit down", "Walk")
FROZEN_FILES = ("pilot_private_manifest.json", "pilot_annotator_manifest.json",
                "pilot_selection_provenance.json", "presentation_order.json")


class PilotError(ValueError):
    """A pilot constraint or immutable source check failed."""


@dataclass(frozen=True)
class TrainVideo:
    subject_id: int
    activity: str
    label: str
    source_video: str
    source_sha256: str
    expected_frame_count: int
    source_fps: int
    width: int
    height: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_train_metadata(manifest: Path = MANIFEST, contract: Path = STAGE3) -> list[TrainVideo]:
    """Use Stage 2 identity metadata only; never read a pose array or AVI here."""
    source = json.loads(contract.read_text(encoding="utf-8"))["source"]
    if sha256_file(manifest) != source["manifest_sha256"]:
        raise PilotError("Official Stage 2 manifest digest mismatch")
    videos: list[TrainVideo] = []
    with manifest.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "train":
                continue
            subject = int(row["subject_id"])
            activity = row["original_activity"]
            label = row["binary_label"]
            path = Path(row["source_relative_video_path"])
            if (subject not in TRAIN_SUBJECTS or path.is_absolute() or ".." in path.parts
                    or path.parts[:2] != (f"Subject.{subject}", activity)
                    or activity not in (FALL if label == "fall" else NON_FALL)):
                raise PilotError("Invalid Train identity in frozen manifest")
            fps = float(row["source_fps"])
            if fps != 20 or row["status"] != "complete":
                raise PilotError("Train timeline/status differs from frozen Stage 2")
            videos.append(TrainVideo(subject, activity, label, path.as_posix(),
                                     row["source_sha256"], int(row["expected_frame_count"]),
                                     20, int(row["width"]), int(row["height"])))
    if len(videos) != 60 or len({v.source_video for v in videos}) != 60:
        raise PilotError("Expected 60 unique Train metadata rows")
    return videos


def select_pilot(videos: list[TrainVideo]) -> list[TrainVideo]:
    """One per subject/class; first five activities once, sixth repeats first."""
    by_key = {(v.subject_id, v.activity): v for v in videos}
    if len(by_key) != 60 or {v.subject_id for v in videos} != set(TRAIN_SUBJECTS):
        raise PilotError("Incomplete or duplicate Train subject/activity grid")
    chosen: list[TrainVideo] = []
    for label, activities in (("fall", FALL), ("non_fall", NON_FALL)):
        for index, subject in enumerate(TRAIN_SUBJECTS):
            activity = activities[index % len(activities)]
            video = by_key.get((subject, activity))
            if video is None or video.label != label:
                raise PilotError("Pilot activity constraint cannot be satisfied")
            chosen.append(video)
    return chosen


def _digest_id(domain: str, video: TrainVideo) -> str:
    identity = f"{RUN_ID}|{PARENT}|{video.subject_id}|{video.activity}|{video.source_video}"
    return hashlib.sha256(f"{domain}|{identity}".encode("utf-8")).hexdigest()


def build_manifests(chosen: list[TrainVideo]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if len(chosen) != 12 or len({v.source_video for v in chosen}) != 12:
        raise PilotError("Pilot requires exactly 12 unique full Train videos")
    private = []
    for video in chosen:
        clip_id = "clip-" + _digest_id("neutral-id", video)[:16]
        private.append(dict(neutral_clip_id=clip_id, source_video=video.source_video,
                            source_avi_sha256=video.source_sha256, subject_id=video.subject_id,
                            split="train", activity=video.activity, dataset_label=video.label,
                            expected_frame_count=video.expected_frame_count, source_fps=video.source_fps,
                            source_fps_provenance="frozen_Stage_2_AVI_timeline",
                            width=video.width, height=video.height,
                            selection_rule="canonical_subject_activity_grid_v1",
                            parent_commit=PARENT, protocol_version="stage32e_v1"))
    if len({row["neutral_clip_id"] for row in private}) != 12:
        raise PilotError("Neutral clip ID collision")
    keyed = sorted(zip(chosen, private), key=lambda item: (_digest_id("presentation", item[0]), item[1]["neutral_clip_id"]))
    ordered = [row for _, row in keyed]
    labels = [row["dataset_label"] for row in ordered]
    if all(labels[i] != labels[i + 1] for i in range(11)):
        ordered[1], ordered[2] = ordered[2], ordered[1]
    public = [dict(neutral_clip_id=row["neutral_clip_id"], frame_count=row["expected_frame_count"],
                   duration_ms=row["expected_frame_count"] * 50) for row in ordered]
    return private, public, [row["neutral_clip_id"] for row in ordered]


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def freeze_pilot(output: Path = ARTIFACT_ROOT) -> Path:
    """Freeze metadata and order before any selected AVI is opened."""
    videos = load_train_metadata()
    selected = select_pilot(videos)
    private, public, order = build_manifests(selected)
    output.mkdir(parents=True, exist_ok=False)
    _write_json_new(output / FROZEN_FILES[0], private)
    _write_json_new(output / FROZEN_FILES[1], public)
    _write_json_new(output / FROZEN_FILES[2], {
        "run_id": RUN_ID, "parent_commit": PARENT, "protocol_version": "stage32e_v1",
        "metadata_source": str(MANIFEST.relative_to(REPO)),
        "metadata_sha256": sha256_file(MANIFEST),
        "eligible_scope": "frozen_Train_only_60_videos",
        "selection_rule": "sort subjects numerically (1,2,3,4,8,9); for each class use canonical alphabetic activity order; assign first five subjects distinct activities and sixth the first activity; one video per subject/activity; no visual, pose, missingness, model or annotation inputs",
        "presentation_rule": "sort by SHA-256 of domain presentation, run ID, parent and canonical source identity; if perfectly alternating classes, swap positions 2 and 3; freeze order",
        "neutral_id_rule": "clip- plus first 16 SHA-256 hex characters of domain neutral-id, run ID, parent and canonical source identity",
        "purpose": "protocol_and_tool_validation_pilot_not_statistically_representative"
    })
    _write_json_new(output / FROZEN_FILES[3], order)
    _write_json_new(output / "freeze_record.json", {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_file_sha256": {name: sha256_file(output / name) for name in FROZEN_FILES},
        "decode_verification_at_freeze": "NOT_RUN"
    })
    return output


def load_frozen(output: Path = ARTIFACT_ROOT) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    freeze = json.loads((output / "freeze_record.json").read_text(encoding="utf-8"))
    for name, expected in freeze["frozen_file_sha256"].items():
        if name not in FROZEN_FILES or sha256_file(output / name) != expected:
            raise PilotError("Frozen pilot manifest changed")
    private = json.loads((output / FROZEN_FILES[0]).read_text(encoding="utf-8"))
    public = json.loads((output / FROZEN_FILES[1]).read_text(encoding="utf-8"))
    order = json.loads((output / FROZEN_FILES[3]).read_text(encoding="utf-8"))
    if len(private) != 12 or len(public) != 12 or len(order) != 12:
        raise PilotError("Frozen pilot size mismatch")
    if [row["neutral_clip_id"] for row in public] != order or {row["neutral_clip_id"] for row in private} != set(order):
        raise PilotError("Frozen private/public/order mismatch")
    return private, public, order


def iter_indexed_frames(path: Path, expected: int, fps: int, width: int, height: int) -> Iterator[tuple[int, Any]]:
    """Yield only sequentially decoded frames; never use CAP_PROP_POS_FRAMES."""
    import cv2

    capture = cv2.VideoCapture()
    index = 0
    try:
        if not capture.open(str(path)) or not capture.isOpened():
            raise PilotError("Selected AVI could not open")
        checks = ((cv2.CAP_PROP_FPS, fps), (cv2.CAP_PROP_FRAME_WIDTH, width),
                  (cv2.CAP_PROP_FRAME_HEIGHT, height), (cv2.CAP_PROP_FRAME_COUNT, expected))
        if any(capture.get(prop) != value for prop, value in checks):
            raise PilotError("Selected AVI metadata differs from frozen Stage 2")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.shape != (height, width, 3) or index >= expected:
                raise PilotError(f"Invalid or extra decoded frame at index {index}")
            yield index, frame
            index += 1
        if index != expected:
            raise PilotError(f"Decoded {index} frames; expected {expected}")
    finally:
        capture.release()


class IndexedFrameSource:
    """In-memory indexed frames obtained only by a complete sequential decode."""

    def __init__(self, frames: list[Any]):
        if not frames:
            raise PilotError("Indexed frame source cannot be empty")
        self._frames = frames
        self.index = 0

    @classmethod
    def open_avi(cls, path: Path, expected: int, fps: int, width: int, height: int) -> "IndexedFrameSource":
        pairs = list(iter_indexed_frames(path, expected, fps, width, height))
        if [index for index, _ in pairs] != list(range(expected)):
            raise PilotError("Sequential frame indices are not contiguous")
        return cls([frame for _, frame in pairs])

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def at(self, index: int) -> tuple[int, Any]:
        result = self.peek(index)
        self.index = index
        return result

    def peek(self, index: int) -> tuple[int, Any]:
        """Read an indexed cached frame without changing the navigation cursor."""
        if type(index) is not int or not 0 <= index < self.frame_count:
            raise PilotError("Frame index out of range")
        return index, self._frames[index]

    def next(self) -> tuple[int, Any]:
        return self.at(min(self.index + 1, self.frame_count - 1))

    def previous(self) -> tuple[int, Any]:
        return self.at(max(self.index - 1, 0))


def verify_pilot(output: Path = ARTIFACT_ROOT, source_root: Path = SOURCE_ROOT) -> list[dict[str, Any]]:
    """Only after freeze, mechanically decode the 12 selected Train AVIs."""
    private, _, order = load_frozen(output)
    by_id = {row["neutral_clip_id"]: row for row in private}
    results = []
    for clip_id in order:
        row = by_id[clip_id]
        if row["split"] != "train" or row["subject_id"] not in TRAIN_SUBJECTS:
            raise PilotError("Non-Train clip in frozen pilot")
        relative = Path(row["source_video"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != f"Subject.{row['subject_id']}":
            raise PilotError("Unsafe selected source identity")
        path = source_root / relative
        if sha256_file(path) != row["source_avi_sha256"]:
            raise PilotError("Selected AVI bytes differ from frozen Stage 2 hash")
        indices = [index for index, _ in iter_indexed_frames(path, row["expected_frame_count"],
                     row["source_fps"], row["width"], row["height"])]
        if indices != list(range(row["expected_frame_count"])):
            raise PilotError("Sequential decoder silently skipped a frame")
        results.append(dict(neutral_clip_id=clip_id, source_video=row["source_video"],
                            source_sha256=row["source_avi_sha256"], opened=True,
                            expected_frame_count=row["expected_frame_count"],
                            decoded_frame_count=len(indices), first_frame_index=indices[0],
                            last_frame_index=indices[-1], contiguous_frame_indices=True,
                            source_fps=row["source_fps"], result="MATCH"))
    import cv2
    _write_json_new(output / "decode_integrity.json", results)
    _write_json_new(output / "tool_runtime_provenance.json", {
        "python_version": sys.version, "opencv_version": cv2.__version__,
        "opencv_distribution_version": importlib.metadata.version("opencv-contrib-python"),
        "platform": platform.platform(), "decoder": "OpenCV_VideoCapture_sequential_read",
        "frame_index_rule": "zero_based_sequential_read_ordinal_no_random_seek",
        "protocol_version": "stage32e_v1", "parent_commit": PARENT,
        "pilot_manifest_version": RUN_ID, "mediapipe_required": False
    })
    _write_json_new(output / "validation_record.json", {
        "pilot_frozen_before_decode": True, "selected_count": 12,
        "all_selected_AVI_hashes_match": True, "all_sequential_counts_match": True,
        "all_indices_contiguous_from_zero": True, "frame_count_mismatches": 0,
        "limitation": "OpenCV read termination cannot by itself distinguish EOF from decoder failure when count matches"
    })
    return results


def validate_artifacts(output: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    """Check frozen pilot and decode artifacts without opening any AVI."""
    private, public, order = load_frozen(output)
    subjects = Counter((row["subject_id"], row["dataset_label"]) for row in private)
    if subjects != Counter({(subject, label): 1 for subject in TRAIN_SUBJECTS
                            for label in ("fall", "non_fall")}):
        raise PilotError("Frozen pilot subject/class balance mismatch")
    for label, expected in (("fall", set(FALL)), ("non_fall", set(NON_FALL))):
        if {row["activity"] for row in private if row["dataset_label"] == label} != expected:
            raise PilotError("Frozen pilot activity coverage mismatch")
    if (any(row["split"] != "train" or row["subject_id"] not in TRAIN_SUBJECTS
            or row["protocol_version"] != "stage32e_v1" or row["parent_commit"] != PARENT
            for row in private)
            or any(set(row) != {"neutral_clip_id", "frame_count", "duration_ms"}
                   or row["duration_ms"] != row["frame_count"] * 50 for row in public)):
        raise PilotError("Frozen private/public provenance mismatch")
    results = json.loads((output / "decode_integrity.json").read_text(encoding="utf-8"))
    if len(results) != 12 or [row["neutral_clip_id"] for row in results] != order:
        raise PilotError("Decode artifact order/count mismatch")
    by_id = {row["neutral_clip_id"]: row for row in private}
    for row in results:
        source = by_id[row["neutral_clip_id"]]
        expected = source["expected_frame_count"]
        if (row["source_video"] != source["source_video"]
                or row["source_sha256"] != source["source_avi_sha256"]
                or row["expected_frame_count"] != expected
                or row["decoded_frame_count"] != expected
                or row["first_frame_index"] != 0
                or row["last_frame_index"] != expected - 1
                or row["contiguous_frame_indices"] is not True
                or row["result"] != "MATCH"):
            raise PilotError("Decode artifact differs from frozen pilot")
    runtime = json.loads((output / "tool_runtime_provenance.json").read_text(encoding="utf-8"))
    validation = json.loads((output / "validation_record.json").read_text(encoding="utf-8"))
    if (runtime["parent_commit"] != PARENT or runtime["pilot_manifest_version"] != RUN_ID
            or runtime["protocol_version"] != "stage32e_v1"
            or validation["selected_count"] != 12 or validation["frame_count_mismatches"] != 0
            or validation["pilot_frozen_before_decode"] is not True):
        raise PilotError("Runtime/validation provenance mismatch")
    return {"selected_clips": 12, "fall_clips": 6, "non_fall_clips": 6,
            "decoded_frames": sum(row["decoded_frame_count"] for row in results),
            "frame_count_mismatches": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "validate"))
    args = parser.parse_args()
    if args.action == "freeze":
        print(freeze_pilot())
    elif args.action == "verify":
        print(f"verified {len(verify_pilot())} frozen Train clips")
    else:
        print(json.dumps(validate_artifacts(), sort_keys=True))


if __name__ == "__main__":
    main()
