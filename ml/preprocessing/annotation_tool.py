"""Local Stage 3.2f AVI annotation tool; run only in a later authorized pilot."""

from __future__ import annotations

import argparse
import copy
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.preprocessing.annotation_pilot import (
    ARTIFACT_ROOT, SOURCE_ROOT, IndexedFrameSource, PilotError, load_frozen, sha256_file,
)
from ml.preprocessing.manual_annotation_contract import AnnotationContractError, validate_record


TOOL_VERSION = "stage32f_local_tk_v1"
BOUNDARIES = ("fall_transition_start", "grounded_start", "recovery_start")
COORDINATES = ("earliest_plausible_frame", "preferred_frame", "latest_plausible_frame")
EVENTS = ("fall_observed", "no_fall_observed", "uncertain")
STATUSES = ("observed", "left_censored", "right_censored", "not_observed", "unjudgeable", "not_applicable")
REASONS = ("occlusion", "subject_out_of_frame", "motion_blur", "ambiguous_transition_start",
           "ambiguous_grounded_state", "ambiguous_recovery", "insufficient_clip_context", "other")
PUBLIC_KEYS = frozenset({"neutral_clip_id", "frame_count", "duration_ms"})


def annotator_payload(public_row: dict[str, Any]) -> dict[str, Any]:
    """Fail closed at the private-to-annotator data boundary."""
    if set(public_row) != PUBLIC_KEYS:
        raise PilotError("Annotator payload contains private or missing fields")
    if not isinstance(public_row["neutral_clip_id"], str) or not public_row["neutral_clip_id"].startswith("clip-"):
        raise PilotError("Invalid neutral clip ID")
    if type(public_row["frame_count"]) is not int or public_row["frame_count"] <= 0:
        raise PilotError("Invalid public frame count")
    if public_row["duration_ms"] != public_row["frame_count"] * 50:
        raise PilotError("Invalid derived duration")
    return dict(public_row)


class DraftStore:
    """Private draft storage; final records are exclusive-create and immutable by tool policy."""

    def __init__(self, root: Path):
        self.root = root

    def _path(self, record_id: str, state: str) -> Path:
        if not record_id or not all(c.isalnum() or c in "-_" for c in record_id):
            raise AnnotationContractError("Invalid annotation_record_id for storage")
        return self.root / f"{record_id}.{state.lower()}.json"

    def save_draft(self, record: dict[str, Any]) -> Path:
        record_id = record["annotation_record_id"]
        final = self._path(record_id, "FINALIZED")
        if final.exists():
            raise AnnotationContractError("Finalized original cannot be overwritten")
        self.root.mkdir(parents=True, exist_ok=True)
        draft = self._path(record_id, "DRAFT")
        temporary = self.root / f".{record_id}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump({"state": "DRAFT", "record": record}, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, draft)
        finally:
            temporary.unlink(missing_ok=True)
        return draft

    def load_draft(self, record_id: str) -> dict[str, Any]:
        if self._path(record_id, "FINALIZED").exists():
            raise AnnotationContractError("Finalized original cannot be resumed")
        wrapper = json.loads(self._path(record_id, "DRAFT").read_text(encoding="utf-8"))
        if wrapper.get("state") != "DRAFT" or wrapper.get("record", {}).get("annotation_record_id") != record_id:
            raise AnnotationContractError("Invalid draft identity/state")
        return wrapper["record"]

    def finalize(self, record: dict[str, Any]) -> Path:
        validate_record(record)
        record_id = record["annotation_record_id"]
        self.root.mkdir(parents=True, exist_ok=True)
        final = self._path(record_id, "FINALIZED")
        descriptor = os.open(final, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"state": "FINALIZED", "record": record}, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            final.unlink(missing_ok=True)
            raise
        self._path(record_id, "DRAFT").unlink(missing_ok=True)
        return final


class AnnotationSession:
    """UI-independent manual state; never derives a label from private provenance."""

    def __init__(self, private_row: dict[str, Any], public_row: dict[str, Any],
                 frames: IndexedFrameSource, store: DraftStore,
                 annotator_id: str, session_id: str, annotation_run_id: str,
                 draft_record: dict[str, Any] | None = None):
        self.public = annotator_payload(public_row)
        if (private_row["neutral_clip_id"] != self.public["neutral_clip_id"]
                or private_row["expected_frame_count"] != frames.frame_count
                or self.public["frame_count"] != frames.frame_count):
            raise PilotError("Private/public/frame-source identity mismatch")
        self.frames = frames
        self.store = store
        self.finalized = False
        if draft_record is None:
            self.record = {
                "annotation_record_id": "ann-" + uuid.uuid4().hex,
                "protocol_version": "stage32e_v1",
                "source": {
                    "source_video": private_row["source_video"],
                    "neutral_clip_id": self.public["neutral_clip_id"],
                    "subject_id": private_row["subject_id"],
                    "split": private_row["split"],
                    "source_avi_sha256": private_row["source_avi_sha256"],
                    "frame_count": frames.frame_count,
                    "source_fps": 20,
                    "source_fps_provenance": "frozen_Stage_2_AVI_timeline",
                },
                "annotator": {"pseudonymous_id": annotator_id, "session_id": session_id},
                "event_presence": None,
                "event_presence_reason_flags": [],
                "boundaries": {name: {"status": "not_applicable", "reason_flags": []} for name in BOUNDARIES},
                "creation": {
                    "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "annotation_run_id": annotation_run_id,
                    "tool_version": TOOL_VERSION,
                },
            }
        else:
            if (draft_record["source"]["neutral_clip_id"] != self.public["neutral_clip_id"]
                    or draft_record["source"]["source_video"] != private_row["source_video"]
                    or draft_record["annotator"]["pseudonymous_id"] != annotator_id):
                raise PilotError("Draft does not belong to this neutral clip/annotator")
            self.record = copy.deepcopy(draft_record)

    def _ensure_editable(self) -> None:
        if self.finalized:
            raise AnnotationContractError("Finalized original cannot be edited")

    def set_event_presence(self, value: str) -> None:
        self._ensure_editable()
        if value not in EVENTS:
            raise AnnotationContractError("Invalid event_presence")
        self.record["event_presence"] = value
        if value == "fall_observed":
            statuses = ("unjudgeable", "unjudgeable", "not_observed")
        elif value == "uncertain":
            statuses = ("unjudgeable", "unjudgeable", "not_applicable")
        else:
            statuses = ("not_applicable",) * 3
        self.record["boundaries"] = {name: {"status": status, "reason_flags": []}
                                     for name, status in zip(BOUNDARIES, statuses)}

    def set_status(self, boundary: str, status: str) -> None:
        self._ensure_editable()
        if boundary not in BOUNDARIES or status not in STATUSES:
            raise AnnotationContractError("Invalid boundary or status")
        event = self.record["event_presence"]
        if event == "no_fall_observed" and status != "not_applicable":
            raise AnnotationContractError("No-fall boundaries are not applicable")
        if event == "uncertain" and status != ("not_applicable" if boundary == "recovery_start" else "unjudgeable"):
            raise AnnotationContractError("Uncertain event boundary status is fixed by protocol")
        if event == "fall_observed" and (status == "not_applicable" or (status == "not_observed" and boundary != "recovery_start")):
            raise AnnotationContractError("Fall-observed boundary status is invalid")
        if event is None:
            raise AnnotationContractError("Select event_presence first")
        flags = self.record["boundaries"][boundary]["reason_flags"]
        self.record["boundaries"][boundary] = {"status": status, "reason_flags": flags}

    def mark(self, boundary: str, coordinate: str) -> None:
        self._ensure_editable()
        if boundary not in BOUNDARIES or coordinate not in COORDINATES:
            raise AnnotationContractError("Invalid boundary coordinate")
        target = self.record["boundaries"][boundary]
        if target["status"] != "observed":
            raise AnnotationContractError("Only observed boundaries accept frame coordinates")
        target[coordinate] = self.frames.index

    def set_reason_flags(self, scope: str, flags: list[str]) -> None:
        self._ensure_editable()
        if any(flag not in REASONS for flag in flags) or len(set(flags)) != len(flags):
            raise AnnotationContractError("Invalid reason flags")
        if scope == "event":
            self.record["event_presence_reason_flags"] = list(flags)
        elif scope in BOUNDARIES:
            self.record["boundaries"][scope]["reason_flags"] = list(flags)
        else:
            raise AnnotationContractError("Invalid reason scope")

    def set_note(self, note: str) -> None:
        self._ensure_editable()
        if not isinstance(note, str):
            raise AnnotationContractError("Note must be text")
        self.record["annotator_note"] = note

    def save_draft(self) -> Path:
        self._ensure_editable()
        return self.store.save_draft(self.record)

    def finalize(self) -> Path:
        self._ensure_editable()
        path = self.store.finalize(self.record)
        self.finalized = True
        return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neutral-clip-id", required=True)
    parser.add_argument("--annotator-id", required=True, help="Pseudonymous ID only")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--annotation-run-id", required=True)
    parser.add_argument("--draft-root", type=Path, required=True)
    parser.add_argument("--resume-record-id")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    private, public, _ = load_frozen(ARTIFACT_ROOT)
    by_id = {row["neutral_clip_id"]: row for row in private}
    visible = {row["neutral_clip_id"]: row for row in public}
    row = by_id[args.neutral_clip_id]
    if row["split"] != "train":
        raise PilotError("Only frozen Train pilot clips are available")
    path = SOURCE_ROOT / row["source_video"]
    if sha256_file(path) != row["source_avi_sha256"]:
        raise PilotError("Frozen selected AVI identity mismatch")
    frames = IndexedFrameSource.open_avi(path, row["expected_frame_count"], row["source_fps"], row["width"], row["height"])
    store = DraftStore(args.draft_root)
    draft = store.load_draft(args.resume_record_id) if args.resume_record_id else None
    session = AnnotationSession(row, visible[args.neutral_clip_id], frames, store,
                                args.annotator_id, args.session_id, args.annotation_run_id, draft)
    from ml.preprocessing.annotation_web import serve_web

    serve_web(session, args.port)


if __name__ == "__main__":
    main()
