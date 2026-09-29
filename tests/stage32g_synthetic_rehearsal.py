"""TEST ONLY: persistent-path rehearsal inside a disposable synthetic namespace.

Reads frozen pilot metadata, never source pixels. No agreement statistics exist
here. Reuses the reviewed controller/store rather than the real AVI launcher.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from ml.preprocessing import annotation_pilot
from ml.preprocessing.annotation_execution import (
    ANNOTATORS, REPO, ControlLogs, ExecutionError, FinalizedBlob, IsolatedStore,
    PilotSpec, check_completion, collect_finalized, next_required_clip,
    raw_freeze_record, require_agreement_ready, write_raw_freeze_new,
)
from ml.preprocessing.annotation_pilot import IndexedFrameSource
from ml.preprocessing.annotation_tool import AnnotationSession
from ml.preprocessing.manual_annotation_contract import AnnotationContractError, validate_record


FIXTURE_UTC = "2026-09-29T00:00:00Z"
TEST_MARKER = "SYNTHETIC REHEARSAL TEST ONLY - NOT HUMAN ANNOTATION"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rejected(action: Callable[[], Any], exceptions: tuple[type[Exception], ...] = (ExecutionError,)) -> bool:
    try:
        action()
    except exceptions:
        return True
    raise AssertionError("Expected fail-closed rejection")


def _fixture_session(root: Path, spec: PilotSpec, owner: str, clip: str, number: int,
                     control: ControlLogs) -> AnnotationSession:
    row = spec.private_by_id[clip]
    store = IsolatedStore(root, owner, clip, spec.version, control, spec.source_fields(clip))
    public = {"neutral_clip_id": clip, "frame_count": row["expected_frame_count"],
              "duration_ms": row["expected_frame_count"] * 50}
    # These placeholders are never rendered, encoded or decoded.
    frames = IndexedFrameSource([TEST_MARKER] * row["expected_frame_count"])
    session = AnnotationSession(row, public, frames, store, owner,
                                f"synthetic-rehearsal-session-{owner}", spec.version)
    session.record["annotation_record_id"] = f"synthetic-rehearsal-{owner}-{number:02d}"
    session.record["creation"]["created_at_utc"] = FIXTURE_UTC
    session.set_note(TEST_MARKER)
    if owner == "A01":
        session.set_event_presence("no_fall_observed")
    else:
        # Intentionally distinct, schema-valid fixture decisions. No comparison
        # or disagreement magnitude is calculated from the two streams.
        session.set_event_presence("uncertain")
        session.set_reason_flags("event", ["other"])
        session.set_reason_flags("fall_transition_start", ["other"])
        session.set_reason_flags("grounded_start", ["other"])
    validate_record(session.record)
    return session


def _mutated_blob(original: FinalizedBlob, change: Callable[[dict[str, Any]], None]) -> FinalizedBlob:
    wrapper = json.loads(original.content)
    change(wrapper["record"])
    record_id = wrapper["record"]["annotation_record_id"]
    owner = original.relative_path.split("/")[0]
    return FinalizedBlob(f"{owner}/finalized/{record_id}.finalized.json",
                         json.dumps(wrapper, indent=2).encode() + b"\n")


def run_rehearsal(root: Path, *, code_commit: str) -> dict[str, Any]:
    """Exercise reviewed storage -> collector -> completion -> freeze -> guard."""
    _require(root.name == "synthetic_rehearsal", "Explicit synthetic rehearsal namespace required")
    _require(not root.exists(), "Rehearsal must start in a fresh disposable directory")
    protocol = REPO / "configs/stage32e_manual_annotation_protocol.json"
    protocol_before = hashlib.sha256(protocol.read_bytes()).hexdigest()
    original_open = Path.open
    def metadata_or_fixture_only(path: Path, *args: Any, **kwargs: Any) -> Any:
        if (path.resolve().is_relative_to(REPO / "data")
                or path.suffix.lower() in {".avi", ".png", ".txt", ".npz", ".npy", ".tflite", ".pt", ".pth", ".onnx"}):
            raise AssertionError("Rehearsal attempted forbidden data-content access")
        return original_open(path, *args, **kwargs)
    with patch.object(IndexedFrameSource, "open_avi", side_effect=AssertionError("No AVI opening")), \
            patch.object(annotation_pilot, "iter_indexed_frames", side_effect=AssertionError("No AVI decoding")), \
            patch.object(Path, "open", metadata_or_fixture_only):
        spec = PilotSpec.from_frozen()
        control = ControlLogs(root)
        sessions: dict[str, list[AnnotationSession]] = {}
        for owner in ANNOTATORS:
            sessions[owner] = []
            for number, clip in enumerate(spec.presentation_order, 1):
                session = _fixture_session(root, spec, owner, clip, number, control)
                session.save_draft()
                sessions[owner].append(session)

        ownership = {}
        for owner, other in (("A01", "A02"), ("A02", "A01")):
            record_id = sessions[other][0].record["annotation_record_id"]
            ownership[f"{owner}_cannot_load_{other}"] = _rejected(
                lambda owner=owner, record_id=record_id: sessions[owner][0].store.load_draft(record_id),
                (FileNotFoundError, ExecutionError, AnnotationContractError))
        ownership["third_annotator_rejected"] = _rejected(
            lambda: IsolatedStore(root, "A03", spec.presentation_order[0], spec.version,
                                  control, spec.source_fields(spec.presentation_order[0])))

        # Logs are separate files. An open incident must block finalization while
        # leaving the deterministic annotation decisions byte-for-byte unchanged.
        first = sessions["A01"][0]
        semantic_before = copy.deepcopy(first.record)
        question_path = control.add_question({
            "question_id": "synthetic-rehearsal-question", "annotator_id": "A01",
            "neutral_clip_id": spec.presentation_order[0], "protocol_section": "synthetic example",
            "category": "semantic_protocol", "description": TEST_MARKER,
            "created_at_utc": FIXTURE_UTC, "status": "OPEN",
        })
        incident_path = control.add_incident({
            "incident_id": "synthetic-rehearsal-incident", "annotator_id": "A01",
            "neutral_clip_id": spec.presentation_order[0], "category": "frame_display_mismatch",
            "blocking": True, "description": TEST_MARKER, "created_at_utc": FIXTURE_UTC, "status": "OPEN",
        })
        blocker_rejected = _rejected(first.finalize)
        _require(first.record == semantic_before, "Technical incident changed annotation semantics")
        _require("_control" in question_path.parts and "_control" in incident_path.parts,
                 "Control logs must be outside annotation records")
        control.resolve_incident("synthetic-rehearsal-incident", resolved_at_utc=FIXTURE_UTC,
                                 resolution_summary=TEST_MARKER)

        before_completion = False
        for owner in ANNOTATORS:
            for session in sessions[owner]:
                current = collect_finalized(root)[owner]
                _require(next_required_clip(spec, owner, current) == session.public["neutral_clip_id"],
                         "Rehearsal violated common frozen order")
                finalized_path = session.finalize()
                wrapper = json.loads(finalized_path.read_bytes())
                validate_record(wrapper["record"])
                _require(wrapper["state"] == "FINALIZED" and
                         wrapper["record"]["annotator_note"] == TEST_MARKER,
                         "Final record lost synthetic identity")
            if owner == "A01":
                actual_incomplete = collect_finalized(root)
                _require(check_completion(spec, actual_incomplete).counts == {"A01": 12, "A02": 0},
                         "Expected chronological pre-completion disk state")
                before_completion = _rejected(lambda: require_agreement_ready(
                    spec, actual_incomplete, None, code_commit=code_commit))

        files = collect_finalized(root)
        results = {}
        for name, n1, n2 in (("A", 11, 12), ("B", 12, 0), ("C", 12, 11), ("D", 12, 12)):
            subset = {"A01": files["A01"][:n1], "A02": files["A02"][:n2]}
            result = check_completion(spec, subset)
            _require(result.complete == (name == "D"), "Unexpected structural completion result")
            results[name] = {"counts": dict(result.counts), "complete": result.complete,
                             "integrity_errors": list(result.errors)}

        incomplete = {"A01": files["A01"][:11], "A02": files["A02"]}
        _rejected(lambda: require_agreement_ready(spec, incomplete, None, code_commit=code_commit))
        before_freeze = _rejected(lambda: require_agreement_ready(spec, files, None, code_commit=code_commit))
        freeze_path = write_raw_freeze_new(root, spec, files, frozen_at_utc=FIXTURE_UTC, code_commit=code_commit)
        freeze_bytes = freeze_path.read_bytes()
        freeze = json.loads(freeze_bytes)
        _require(freeze["protocol_version"] == spec.protocol_version and freeze["pilot_version"] == spec.version
                 and freeze["annotator_ids"] == list(ANNOTATORS)
                 and freeze["expected_clip_ids"] == list(spec.presentation_order)
                 and freeze["presentation_order_sha256"] == spec.presentation_order_sha256
                 and len(freeze["finalized_file_sha256"]) == 24
                 and freeze["finalized_file_sha256"] == dict(check_completion(spec, files).file_sha256),
                 "Raw freeze identity/hash capture is incomplete")
        second_freeze = _rejected(lambda: write_raw_freeze_new(
            root, spec, files, frozen_at_utc=FIXTURE_UTC, code_commit=code_commit), (FileExistsError,))
        _require(freeze_path.read_bytes() == freeze_bytes, "Second freeze changed original bytes")
        require_agreement_ready(spec, files, freeze, code_commit=code_commit)

        # The supported store rejects owner changes and another original; no
        # in-place file-edit API is used or claimed to be OS-level protection.
        finalized_before = {blob.relative_path: blob.content for group in files.values() for blob in group}
        changed_owner = copy.deepcopy(first.record)
        changed_owner["annotator"]["pseudonymous_id"] = "A02"
        ownership["owner_change_after_finalization_rejected"] = _rejected(
            lambda: first.store.finalize(changed_owner))
        duplicate = copy.deepcopy(first.record)
        duplicate["annotation_record_id"] += "-duplicate-test-only"
        ownership["duplicate_original_rejected"] = _rejected(lambda: first.store.finalize(duplicate))
        for owner, other in (("A01", "A02"), ("A02", "A01")):
            ownership[f"{owner}_cannot_resume_{other}_final"] = _rejected(
                lambda owner=owner, other=other: sessions[owner][0].store.load_draft(
                    sessions[other][0].record["annotation_record_id"]),
                (FileNotFoundError, ExecutionError, AnnotationContractError))
        _require(finalized_before == {blob.relative_path: blob.content
                                     for group in collect_finalized(root).values() for blob in group},
                 "Ownership checks changed final bytes")

        integrity = {}
        mutations = {
            "wrong_protocol_version": lambda r: r.update(protocol_version="synthetic-wrong-protocol"),
            "wrong_pilot_version": lambda r: r["creation"].update(annotation_run_id="synthetic-wrong-pilot"),
            "wrong_neutral_clip_id": lambda r: r["source"].update(neutral_clip_id=spec.presentation_order[1]),
        }
        for name, change in mutations.items():
            changed = {owner: list(group) for owner, group in files.items()}
            changed["A01"][0] = _mutated_blob(changed["A01"][0], change)
            integrity[name] = not check_completion(spec, changed).complete
        missing = {"A01": files["A01"][:-1], "A02": files["A02"]}
        integrity["missing_expected_clip"] = not check_completion(spec, missing).complete
        extra = {owner: list(group) for owner, group in files.items()}
        def extra_clip(record: dict[str, Any]) -> None:
            record["annotation_record_id"] += "-extra-test-only"
            record["source"]["neutral_clip_id"] = "clip-synthetic-unexpected"
        extra["A01"].append(_mutated_blob(files["A01"][0], extra_clip))
        integrity["unexpected_extra_clip"] = not check_completion(spec, extra).complete
        wrong_order = dict(freeze, expected_clip_ids=list(reversed(freeze["expected_clip_ids"])))
        integrity["presentation_order_mismatch"] = _rejected(lambda: require_agreement_ready(
            spec, files, wrong_order, code_commit=code_commit))
        wrong_order_hash = dict(freeze, presentation_order_sha256="0" * 64)
        integrity["presentation_order_hash_mismatch"] = _rejected(lambda: require_agreement_ready(
            spec, files, wrong_order_hash, code_commit=code_commit))
        _require(all(integrity.values()), "An integrity violation was accepted")

        with tempfile.TemporaryDirectory(prefix="stage32g-synthetic-tamper-") as tmp:
            tamper_root = Path(tmp) / "synthetic_rehearsal"
            shutil.copytree(root, tamper_root)
            selected = files["A01"][0].relative_path
            target = tamper_root / selected
            target.chmod(0o644)
            target.write_bytes(target.read_bytes() + b"\n")
            tampered = collect_finalized(tamper_root)
            tampered_result = check_completion(spec, tampered)
            _require(tampered_result.complete, "Whitespace tamper should remain schema-valid")
            sha_mismatch = hashlib.sha256(target.read_bytes()).hexdigest() != freeze["finalized_file_sha256"][selected]
            freeze_validation_failed = raw_freeze_record(
                spec, tampered_result, frozen_at_utc=FIXTURE_UTC, code_commit=code_commit) != freeze
            guard_denied = _rejected(lambda: require_agreement_ready(
                spec, tampered, freeze, code_commit=code_commit))
            _require(sha_mismatch and freeze_validation_failed and guard_denied, "Tamper was not detected")

        _require(hashlib.sha256(protocol.read_bytes()).hexdigest() == protocol_before,
                 "Semantic question mutated frozen protocol")
        _require(all(not list((root / owner / "drafts").glob("*.json")) for owner in ANNOTATORS),
                 "Finalized drafts were not removed")
        return {
            "synthetic_test_only": True,
            "namespace": "<temporary_directory>/synthetic_rehearsal",
            "fixtures_retained": False,
            "frozen_neutral_clip_ids": list(spec.presentation_order),
            "pilot_version": spec.version,
            "protocol_version": spec.protocol_version,
            "presentation_order_sha256": spec.presentation_order_sha256,
            "finalized_counts": dict(check_completion(spec, files).counts),
            "deterministic_schema_valid_distinct_decisions": True,
            "completion_cases": results,
            "raw_freeze_test_only": freeze,
            "raw_freeze_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
            "second_freeze_rejected_unchanged": second_freeze,
            "agreement_access_guard": {"before_dual_completion": "DENY" if before_completion else "ERROR",
                                       "after_completion_before_freeze": "DENY" if before_freeze else "ERROR",
                                       "after_matching_raw_freeze": "ALLOW_PREREQUISITES_ONLY"},
            "tamper_detection": {"sha256_mismatch": sha_mismatch,
                                 "freeze_validation_failed": freeze_validation_failed,
                                 "agreement_access_denied": guard_denied, "isolated_copy_discarded": True},
            "ownership_rejections": ownership,
            "integrity_rejections": integrity,
            "independent_control_logs": {"blocking_incident_rejected_finalization": blocker_rejected,
                                         "annotation_semantics_unchanged": True,
                                         "frozen_protocol_unchanged": True},
            "no_agreement_metric_computed": True,
            "no_real_data_content_accessed": True,
            "dataset_content_access_guard_active": True,
            "no_genuine_annotation_created": True,
        }
