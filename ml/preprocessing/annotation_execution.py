"""Stage 3.2g dual-annotator execution controls; no agreement calculations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ml.preprocessing.annotation_pilot import ARTIFACT_ROOT, SOURCE_ROOT, IndexedFrameSource, load_frozen, sha256_file
from ml.preprocessing.annotation_tool import AnnotationSession, DraftStore, TOOL_VERSION
from ml.preprocessing.manual_annotation_contract import AnnotationContractError, validate_record


REPO = Path(__file__).resolve().parents[2]
PILOT_VERSION = "stage32f_train_pilot_v1"
PROTOCOL_VERSION = "stage32e_v1"
ANNOTATORS = ("A01", "A02")
REAL_ROOT = REPO / "data/annotation_runs" / PILOT_VERSION
SYNTHETIC_DIRNAME = "synthetic_onboarding"
PINNED_SHA256 = {
    "pilot_private_manifest.json": "9df7208206036fc28f390c5ef3ccafcb9aaca967bd9439d207407ed8fccb61e8",
    "pilot_annotator_manifest.json": "508bbb7aaaf1884dadc6d84287a4dba6cf24df05d63cc41f0aa7caac608f8891",
    "presentation_order.json": "c10b2a5be2501db9a4e177ebfd514cf7e9935bdcc0c44539e89694b063172791",
}


class ExecutionError(ValueError):
    """A Stage 3.2g isolation or integrity requirement failed."""


def _utc(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _approved(annotator_id: str) -> None:
    if annotator_id not in ANNOTATORS:
        raise ExecutionError("Unapproved pseudonymous annotator ID")


@dataclass(frozen=True)
class PilotSpec:
    """Private frozen provenance for integrity checking, never exposed to UI."""

    private_by_id: Mapping[str, Mapping[str, Any]]
    presentation_order: tuple[str, ...]
    private_manifest_sha256: str
    presentation_order_sha256: str
    version: str = PILOT_VERSION
    protocol_version: str = PROTOCOL_VERSION
    tool_version: str = TOOL_VERSION

    @classmethod
    def from_frozen(cls, root: Path = ARTIFACT_ROOT) -> "PilotSpec":
        for name, expected in PINNED_SHA256.items():
            if sha256_file(root / name) != expected:
                raise ExecutionError("Stage 3.2f pilot identity changed")
        private, public, order = load_frozen(root)
        if len(order) != 12 or len(set(order)) != 12 or [row["neutral_clip_id"] for row in public] != order:
            raise ExecutionError("Expected exactly 12 frozen ordered pilot clips")
        return cls({row["neutral_clip_id"]: row for row in private}, tuple(order),
                   PINNED_SHA256["pilot_private_manifest.json"],
                   PINNED_SHA256["presentation_order.json"])

    def source_fields(self, clip_id: str) -> dict[str, Any]:
        try:
            row = self.private_by_id[clip_id]
        except KeyError as exc:
            raise ExecutionError("Clip is outside the frozen Train pilot") from exc
        return {
            "source_video": row["source_video"],
            "neutral_clip_id": row["neutral_clip_id"],
            "subject_id": row["subject_id"],
            "split": row["split"],
            "source_avi_sha256": row["source_avi_sha256"],
            "frame_count": row["expected_frame_count"],
            "source_fps": row["source_fps"],
            "source_fps_provenance": row["source_fps_provenance"],
        }


@dataclass(frozen=True)
class FinalizedBlob:
    """A file identity and immutable byte snapshot supplied to the pure checker."""

    relative_path: str
    content: bytes


@dataclass(frozen=True)
class CompletionResult:
    complete: bool
    counts: Mapping[str, int]
    file_sha256: Mapping[str, str]
    errors: tuple[str, ...]


def check_completion(spec: PilotSpec, files: Mapping[str, Sequence[FinalizedBlob]]) -> CompletionResult:
    """Validate structural completeness only; never compare annotation semantics."""
    errors: list[str] = []
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    record_ids: set[str] = set()
    if set(files) != set(ANNOTATORS):
        errors.append("annotator_set_mismatch")
    expected = set(spec.presentation_order)
    if len(expected) != 12 or len(spec.private_by_id) != 12 or expected != set(spec.private_by_id):
        errors.append("pilot_identity_mismatch")
    for annotator in ANNOTATORS:
        seen: set[str] = set()
        rows = files.get(annotator, ())
        counts[annotator] = 0
        for blob in rows:
            path = blob.relative_path
            if path in hashes or not path.startswith(f"{annotator}/finalized/"):
                errors.append("file_identity_mismatch")
                continue
            hashes[path] = hashlib.sha256(blob.content).hexdigest()
            try:
                wrapper = json.loads(blob.content)
                if set(wrapper) != {"state", "record"} or wrapper["state"] != "FINALIZED":
                    raise ExecutionError("finalized_state_invalid")
                record = wrapper["record"]
                if not isinstance(record, dict):
                    raise ExecutionError("record_invalid")
                validate_record(record)
                clip_id = record["source"]["neutral_clip_id"]
                record_id = record["annotation_record_id"]
                if path != f"{annotator}/finalized/{record_id}.finalized.json":
                    raise ExecutionError("file_identity_mismatch")
                if record_id in record_ids:
                    raise ExecutionError("duplicate_record_id")
                record_ids.add(record_id)
                if record["annotator"]["pseudonymous_id"] != annotator:
                    raise ExecutionError("annotator_ownership_mismatch")
                if record["protocol_version"] != spec.protocol_version:
                    raise ExecutionError("protocol_version_mismatch")
                if record["creation"]["annotation_run_id"] != spec.version:
                    raise ExecutionError("pilot_version_mismatch")
                if record["creation"]["tool_version"] != spec.tool_version:
                    raise ExecutionError("tool_version_mismatch")
                if clip_id not in expected:
                    raise ExecutionError("unexpected_clip_id")
                if record["source"] != spec.source_fields(clip_id):
                    raise ExecutionError("source_identity_mismatch")
                if clip_id in seen:
                    raise ExecutionError("duplicate_finalized_original")
                seen.add(clip_id)
                counts[annotator] += 1
            except (ValueError, TypeError, KeyError, AttributeError, AnnotationContractError) as exc:
                code = str(exc) if isinstance(exc, ExecutionError) else "invalid_finalized_record"
                errors.append(code)
        if seen != expected:
            errors.append(f"incomplete_{annotator}")
    return CompletionResult(not errors, counts, hashes, tuple(sorted(set(errors))))


def next_required_clip(spec: PilotSpec, annotator_id: str,
                       finalized: Sequence[FinalizedBlob]) -> str | None:
    """Require each annotator's finalized clips to form a prefix of one frozen order."""
    _approved(annotator_id)
    seen: set[str] = set()
    for blob in finalized:
        if not blob.relative_path.startswith(f"{annotator_id}/finalized/"):
            raise ExecutionError("Another annotator's final is unavailable")
        try:
            wrapper = json.loads(blob.content)
            if wrapper["state"] != "FINALIZED":
                raise ExecutionError("Non-final record in finalized storage")
            record = wrapper["record"]
            validate_record(record)
            clip_id = record["source"]["neutral_clip_id"]
            if (record["annotator"]["pseudonymous_id"] != annotator_id
                    or record["creation"]["annotation_run_id"] != spec.version
                    or record["creation"]["tool_version"] != spec.tool_version
                    or record["source"] != spec.source_fields(clip_id)
                    or blob.relative_path != f"{annotator_id}/finalized/{record['annotation_record_id']}.finalized.json"
                    or clip_id in seen):
                raise ExecutionError("Invalid or duplicate final in presentation sequence")
            seen.add(clip_id)
        except (ValueError, TypeError, KeyError, AttributeError, AnnotationContractError) as exc:
            raise ExecutionError("Invalid finalized presentation sequence") from exc
    prefix = set(spec.presentation_order[:len(seen)])
    if seen != prefix:
        raise ExecutionError("Finalized clips do not follow frozen presentation order")
    return spec.presentation_order[len(seen)] if len(seen) < len(spec.presentation_order) else None


def raw_freeze_record(spec: PilotSpec, result: CompletionResult, *, frozen_at_utc: str,
                      code_commit: str) -> dict[str, Any]:
    """Construct future raw-record freeze metadata; caller must exclusive-create it."""
    if not result.complete or result.counts != {"A01": 12, "A02": 12}:
        raise ExecutionError("Dual completion gate has not passed")
    if not _utc(frozen_at_utc) or re.fullmatch(r"[0-9a-f]{40}", code_commit) is None:
        raise ExecutionError("Invalid freeze timestamp or code commit")
    return {
        "pilot_version": spec.version,
        "protocol_version": spec.protocol_version,
        "tool_version": spec.tool_version,
        "annotator_ids": list(ANNOTATORS),
        "expected_clip_ids": list(spec.presentation_order),
        "presentation_order_sha256": spec.presentation_order_sha256,
        "pilot_private_manifest_sha256": spec.private_manifest_sha256,
        "finalized_file_sha256": dict(sorted(result.file_sha256.items())),
        "completed_at_utc": frozen_at_utc,
        "code_commit": code_commit,
    }


def require_agreement_ready(spec: PilotSpec, files: Mapping[str, Sequence[FinalizedBlob]],
                            freeze: Mapping[str, Any] | None, *, code_commit: str) -> None:
    """Future analysis entry points must call this before reading annotation values."""
    result = check_completion(spec, files)
    if not result.complete or freeze is None:
        raise ExecutionError("Agreement blocked until dual completion and raw freeze")
    if freeze.get("code_commit") != code_commit:
        raise ExecutionError("Raw annotation freeze code commit mismatch")
    expected = raw_freeze_record(spec, result, frozen_at_utc=freeze.get("completed_at_utc", ""),
                                 code_commit=freeze.get("code_commit", ""))
    if dict(freeze) != expected:
        raise ExecutionError("Raw annotation freeze does not match current files")


def validate_protocol_question(question: Mapping[str, Any]) -> None:
    required = {"question_id", "annotator_id", "neutral_clip_id", "protocol_section",
                "category", "description", "created_at_utc", "status"}
    if set(question) != required | ({"resolution"} if "resolution" in question else set()):
        raise ExecutionError("Invalid protocol-question fields")
    _approved(question["annotator_id"])
    if (question["category"] not in {"procedural_tool", "semantic_protocol"}
            or not _utc(question["created_at_utc"])
            or not all(isinstance(question[key], str) and question[key].strip()
                       for key in ("question_id", "protocol_section", "description"))
            or (question["neutral_clip_id"] is not None
                and not isinstance(question["neutral_clip_id"], str))):
        raise ExecutionError("Invalid protocol question")
    if question["category"] == "semantic_protocol":
        if question["status"] != "OPEN" or "resolution" in question:
            raise ExecutionError("Semantic question stays open during active pilot")
    elif question["status"] not in {"OPEN", "ANSWERED"} or (question["status"] == "ANSWERED") != ("resolution" in question):
        raise ExecutionError("Invalid procedural question status")


def validate_technical_incident(incident: Mapping[str, Any]) -> None:
    required = {"incident_id", "annotator_id", "neutral_clip_id", "category", "blocking",
                "description", "created_at_utc", "status"}
    optional = {"resolved_at_utc", "resolution_summary"}
    if not required <= set(incident) or set(incident) - required - optional:
        raise ExecutionError("Invalid technical-incident fields")
    _approved(incident["annotator_id"])
    if (not isinstance(incident["neutral_clip_id"], str)
            or not isinstance(incident["incident_id"], str)
            or not isinstance(incident["description"], str)
            or not incident["description"].strip()
            or incident["category"] not in {"decode_failure", "frame_display_mismatch", "tool_crash",
                                                 "corrupt_state", "source_hash_mismatch", "frame_count_mismatch",
                                                 "wrong_clip", "other_tool_failure"}
            or type(incident["blocking"]) is not bool or not _utc(incident["created_at_utc"])):
        raise ExecutionError("Invalid technical incident")
    correctness_categories = {"decode_failure", "frame_display_mismatch", "tool_crash",
                              "corrupt_state", "source_hash_mismatch", "frame_count_mismatch", "wrong_clip"}
    if incident["category"] in correctness_categories and not incident["blocking"]:
        raise ExecutionError("Correctness-relevant technical incident must block finalization")
    resolved = incident["status"] == "RESOLVED"
    if (incident["status"] not in {"OPEN", "RESOLVED"}
            or set(incident) & optional != (optional if resolved else set())):
        raise ExecutionError("Invalid technical incident resolution state")
    if resolved and (not _utc(incident["resolved_at_utc"]) or not incident["resolution_summary"].strip()):
        raise ExecutionError("Invalid technical incident resolution")


class ControlLogs:
    """Separate operator-controlled question and incident files; no annotation fields."""

    def __init__(self, root: Path):
        self.root = root / "_control"

    def _write_new(self, area: str, identity: str, record: Mapping[str, Any]) -> Path:
        if re.fullmatch(r"[A-Za-z0-9_-]+", identity) is None:
            raise ExecutionError("Invalid log identity")
        folder = self.root / area
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{identity}.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True, indent=2)
            stream.write("\n")
        return path

    def add_question(self, question: Mapping[str, Any]) -> Path:
        validate_protocol_question(question)
        return self._write_new("protocol_questions", question["question_id"], question)

    def add_incident(self, incident: Mapping[str, Any]) -> Path:
        validate_technical_incident(incident)
        return self._write_new("technical_incidents", incident["incident_id"], incident)

    def resolve_incident(self, incident_id: str, *, resolved_at_utc: str,
                         resolution_summary: str) -> Path:
        if re.fullmatch(r"[A-Za-z0-9_-]+", incident_id) is None:
            raise ExecutionError("Invalid incident identity")
        path = self.root / "technical_incidents" / f"{incident_id}.json"
        if path.is_symlink():
            raise ExecutionError("Symlinked incident log rejected")
        incident = json.loads(path.read_text(encoding="utf-8"))
        validate_technical_incident(incident)
        if incident["status"] != "OPEN":
            raise ExecutionError("Incident is already resolved")
        updated = dict(incident, status="RESOLVED", resolved_at_utc=resolved_at_utc,
                       resolution_summary=resolution_summary)
        validate_technical_incident(updated)
        temporary = path.with_name(f".{incident_id}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(updated, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def has_open_blocker(self, annotator_id: str, clip_id: str) -> bool:
        folder = self.root / "technical_incidents"
        for path in folder.glob("*.json"):
            if path.is_symlink():
                raise ExecutionError("Symlinked incident log rejected")
            incident = json.loads(path.read_text(encoding="utf-8"))
            validate_technical_incident(incident)
            if (incident["annotator_id"] == annotator_id and incident["neutral_clip_id"] == clip_id
                    and incident["blocking"] and incident["status"] == "OPEN"):
                return True
        return False


class IsolatedStore(DraftStore):
    """Per-annotator storage adapter for the frozen Stage 3.2f session."""

    def __init__(self, run_root: Path, annotator_id: str, clip_id: str, run_id: str,
                 control: ControlLogs, expected_source: Mapping[str, Any]):
        _approved(annotator_id)
        if run_root.is_symlink() or (run_root / annotator_id).is_symlink():
            raise ExecutionError("Symlinked annotator workspace rejected")
        super().__init__(run_root / annotator_id)
        self.annotator_id = annotator_id
        self.clip_id = clip_id
        self.run_id = run_id
        self.control = control
        self.expected_source = dict(expected_source)
        for name in ("drafts", "finalized", "logs"):
            if (self.root / name).is_symlink():
                raise ExecutionError("Symlinked annotator storage rejected")
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def _path(self, record_id: str, state: str) -> Path:
        super()._path(record_id, state)
        if state not in {"DRAFT", "FINALIZED"}:
            raise ExecutionError("Unknown annotation storage state")
        return self.root / ("drafts" if state == "DRAFT" else "finalized") / f"{record_id}.{state.lower()}.json"

    def _ownership(self, record: Mapping[str, Any]) -> None:
        if (record["annotator"]["pseudonymous_id"] != self.annotator_id
                or record["source"] != self.expected_source
                or record["protocol_version"] != PROTOCOL_VERSION
                or record["creation"]["annotation_run_id"] != self.run_id
                or record["creation"]["tool_version"] != TOOL_VERSION):
            raise ExecutionError("Record ownership or frozen version mismatch")

    def save_draft(self, record: dict[str, Any]) -> Path:
        self._ownership(record)
        self._reject_if_clip_finalized()
        return super().save_draft(record)

    def load_draft(self, record_id: str) -> dict[str, Any]:
        self._reject_if_clip_finalized()
        record = super().load_draft(record_id)
        self._ownership(record)
        return record

    def _reject_if_clip_finalized(self) -> None:
        for path in (self.root / "finalized").glob("*.finalized.json"):
            if path.is_symlink():
                raise ExecutionError("Symlinked finalized record rejected")
            wrapper = json.loads(path.read_text(encoding="utf-8"))
            if wrapper["record"]["source"]["neutral_clip_id"] == self.clip_id:
                raise ExecutionError("Clip already has a finalized original for this annotator")

    def finalize(self, record: dict[str, Any]) -> Path:
        self._ownership(record)
        if self.control.has_open_blocker(self.annotator_id, self.clip_id):
            raise ExecutionError("Blocking unresolved technical incident prevents finalization")
        self._reject_if_clip_finalized()
        return super().finalize(record)


class ExecutionWorkspace:
    """The only Stage 3.2g application entry for independent pilot sessions."""

    def __init__(self, root: Path, *, synthetic: bool):
        if root.is_symlink():
            raise ExecutionError("Symlinked execution workspace rejected")
        resolved = root.resolve()
        if synthetic:
            if resolved.name != SYNTHETIC_DIRNAME or resolved == REAL_ROOT.resolve():
                raise ExecutionError("Synthetic onboarding requires a separate synthetic_onboarding root")
        elif resolved != REAL_ROOT.resolve():
            raise ExecutionError("Real pilot workspace must use the fixed Git-ignored run root")
        self.root = resolved
        self.synthetic = synthetic
        self.control = ControlLogs(resolved)

    def session(self, spec: PilotSpec, annotator_id: str, clip_id: str, frames: IndexedFrameSource,
                public_row: dict[str, Any], *, session_id: str, resume_record_id: str | None = None,
                code_commit: str | None = None, decoder_version: str | None = None) -> AnnotationSession:
        _approved(annotator_id)
        if not isinstance(session_id, str) or not session_id.strip():
            raise ExecutionError("Explicit session ID required")
        if clip_id not in spec.private_by_id:
            raise ExecutionError("Clip outside configured workspace pilot")
        if self.synthetic and spec.version == PILOT_VERSION:
            raise ExecutionError("Synthetic onboarding cannot use the real pilot identity")
        if not self.synthetic and spec != PilotSpec.from_frozen():
            raise ExecutionError("Real workspace requires exact frozen Stage 3.2f pilot")
        if not self.synthetic and (code_commit is None or re.fullmatch(r"[0-9a-f]{40}", code_commit) is None
                                   or not decoder_version):
            raise ExecutionError("Real session requires code commit and decoder version")
        if not self.synthetic:
            pending = next_required_clip(spec, annotator_id, collect_finalized(self.root)[annotator_id])
            if clip_id != pending:
                raise ExecutionError("Clip is not next in frozen presentation order")
        private = spec.private_by_id[clip_id]
        store = IsolatedStore(self.root, annotator_id, clip_id, spec.version, self.control,
                              spec.source_fields(clip_id))
        draft = store.load_draft(resume_record_id) if resume_record_id else None
        session = AnnotationSession(private, public_row, frames, store, annotator_id,
                                    session_id, spec.version, draft)
        log = {
            "annotator_id": annotator_id,
            "neutral_clip_id": clip_id,
            "session_id": session_id,
            "pilot_version": spec.version,
            "protocol_version": spec.protocol_version,
            "tool_version": spec.tool_version,
            "code_commit": code_commit or "SYNTHETIC_UNCOMMITTED",
            "opencv_version": decoder_version,
            "started_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "synthetic_onboarding": self.synthetic,
        }
        path = store.root / "logs" / f"{uuid.uuid4().hex}.session.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(log, stream, sort_keys=True, indent=2)
            stream.write("\n")
        return session


def collect_finalized(run_root: Path) -> dict[str, list[FinalizedBlob]]:
    """Future I/O adapter; reject unexpected files and symlinks before pure checking."""
    result: dict[str, list[FinalizedBlob]] = {}
    if run_root.exists():
        if run_root.is_symlink() or any(path.name not in {*ANNOTATORS, "_control"}
                                       for path in run_root.iterdir()):
            raise ExecutionError("Unexpected annotator workspace entry")
    for annotator in ANNOTATORS:
        folder = run_root / annotator / "finalized"
        if (run_root / annotator).is_symlink() or folder.is_symlink():
            raise ExecutionError("Symlinked finalized directory rejected")
        entries: list[FinalizedBlob] = []
        if folder.exists():
            for path in sorted(folder.iterdir()):
                if path.is_symlink() or not path.is_file() or not path.name.endswith(".finalized.json"):
                    raise ExecutionError("Unexpected finalized storage entry")
                entries.append(FinalizedBlob(f"{annotator}/finalized/{path.name}", path.read_bytes()))
        result[annotator] = entries
    return result


def write_raw_freeze_new(run_root: Path, spec: PilotSpec,
                         files: Mapping[str, Sequence[FinalizedBlob]], *,
                         frozen_at_utc: str, code_commit: str) -> Path:
    """Future exclusive-create freeze write; never called on real data in Stage 3.2g."""
    if run_root.resolve() == REAL_ROOT.resolve():
        if spec != PilotSpec.from_frozen() or files != collect_finalized(run_root):
            raise ExecutionError("Real raw freeze requires exact frozen pilot and current finalized files")
    record = raw_freeze_record(spec, check_completion(spec, files), frozen_at_utc=frozen_at_utc,
                               code_commit=code_commit)
    folder = run_root / "_control" / "raw_freezes"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "raw_annotation_freeze.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def main() -> None:
    """Future controlled launcher; Stage 3.2g does not invoke it on real AVI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-id", required=True, choices=ANNOTATORS)
    parser.add_argument("--neutral-clip-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--resume-record-id")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    spec = PilotSpec.from_frozen()
    workspace = ExecutionWorkspace(REAL_ROOT, synthetic=False)
    if args.neutral_clip_id not in spec.private_by_id:
        raise ExecutionError("Clip is outside the frozen Train pilot")
    if args.neutral_clip_id != next_required_clip(
            spec, args.annotator_id, collect_finalized(REAL_ROOT)[args.annotator_id]):
        raise ExecutionError("Clip is not next in frozen presentation order")
    if not args.session_id.strip():
        raise ExecutionError("Explicit session ID required")
    _, public, _ = load_frozen()
    public_by_id = {row["neutral_clip_id"]: row for row in public}
    row = spec.private_by_id[args.neutral_clip_id]
    source = SOURCE_ROOT / row["source_video"]
    if sha256_file(source) != row["source_avi_sha256"]:
        raise ExecutionError("Frozen source AVI hash mismatch")
    frames = IndexedFrameSource.open_avi(source, row["expected_frame_count"], row["source_fps"],
                                         row["width"], row["height"])
    import cv2

    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    session = workspace.session(spec, args.annotator_id, args.neutral_clip_id, frames,
                                public_by_id[args.neutral_clip_id], session_id=args.session_id,
                                resume_record_id=args.resume_record_id,
                                code_commit=code_commit, decoder_version=cv2.__version__)
    from ml.preprocessing.annotation_web import serve_web

    serve_web(session, args.port)


if __name__ == "__main__":
    main()
