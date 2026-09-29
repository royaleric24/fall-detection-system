"""Stage 3.2g synthetic-only execution and completion tests."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.preprocessing.annotation_pilot import IndexedFrameSource
from ml.preprocessing.annotation_execution import (
    ANNOTATORS, PILOT_VERSION, PROTOCOL_VERSION, ControlLogs, ExecutionError,
    ExecutionWorkspace, FinalizedBlob, PilotSpec, check_completion,
    collect_finalized, next_required_clip, raw_freeze_record, require_agreement_ready,
    validate_protocol_question, validate_technical_incident, write_raw_freeze_new,
)
from ml.preprocessing.annotation_tool import TOOL_VERSION
from ml.preprocessing.annotation_web import annotator_state
from ml.preprocessing.manual_annotation_contract import AnnotationContractError


NOW = "2026-09-27T00:00:00Z"
CODE_COMMIT = "a" * 40


def mock_spec():
    rows = {}
    order = []
    for number in range(1, 13):
        clip = f"clip-synthetic-{number:02d}"
        order.append(clip)
        rows[clip] = {
            "neutral_clip_id": clip, "source_video": f"Subject.1/Mock/synthetic-{number:02d}.avi",
            "source_avi_sha256": "0" * 64, "subject_id": 1, "split": "train",
            "expected_frame_count": 9, "source_fps": 20,
            "source_fps_provenance": "frozen_Stage_2_AVI_timeline",
        }
    return PilotSpec(rows, tuple(order), "1" * 64, "2" * 64,
                     version="stage32g_synthetic_onboarding_v1")


def mock_record(spec, annotator, clip, number):
    return {
        "annotation_record_id": f"ann-{annotator}-{number:02d}",
        "protocol_version": PROTOCOL_VERSION,
        "source": spec.source_fields(clip),
        "annotator": {"pseudonymous_id": annotator, "session_id": f"synthetic-{annotator}"},
        "event_presence": "no_fall_observed",
        "event_presence_reason_flags": [],
        "boundaries": {name: {"status": "not_applicable", "reason_flags": []}
                       for name in ("fall_transition_start", "grounded_start", "recovery_start")},
        "creation": {"created_at_utc": NOW, "annotation_run_id": spec.version,
                     "tool_version": TOOL_VERSION},
    }


def blob(annotator, record):
    path = f"{annotator}/finalized/{record['annotation_record_id']}.finalized.json"
    content = json.dumps({"state": "FINALIZED", "record": record}, sort_keys=True).encode()
    return FinalizedBlob(path, content)


def full_files(spec):
    return {annotator: [blob(annotator, mock_record(spec, annotator, clip, index))
                        for index, clip in enumerate(spec.presentation_order, 1)]
            for annotator in ANNOTATORS}


def mock_session(workspace, spec, annotator="A01", clip=None, resume=None):
    clip = clip or spec.presentation_order[0]
    public = {"neutral_clip_id": clip, "frame_count": 9, "duration_ms": 450}
    return workspace.session(spec, annotator, clip, IndexedFrameSource([f"mock-{i}" for i in range(9)]),
                             public, session_id=f"synthetic-session-{annotator}", resume_record_id=resume)


class Stage32gExecutionTests(unittest.TestCase):
    def setUp(self):
        self.spec = mock_spec()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "synthetic_onboarding"
        self.workspace = ExecutionWorkspace(self.root, synthetic=True)

    def test_end_to_end_frozen_metadata_synthetic_rehearsal(self):
        from tests.stage32g_synthetic_rehearsal import run_rehearsal

        baseline = "6f0af44710121afd43a8f9e077a5bdd09bd8dec5"
        result = run_rehearsal(self.root.parent / "synthetic_rehearsal", code_commit=baseline)
        self.assertEqual(result["finalized_counts"], {"A01": 12, "A02": 12})
        self.assertEqual([result["completion_cases"][case]["complete"] for case in "ABCD"],
                         [False, False, False, True])
        self.assertEqual(result["agreement_access_guard"], {
            "before_dual_completion": "DENY", "after_completion_before_freeze": "DENY",
            "after_matching_raw_freeze": "ALLOW_PREREQUISITES_ONLY"})
        evidence_path = Path("artifacts/temporal_annotation/stage32g_execution_v1/synthetic_rehearsal_validation.json")
        self.assertEqual(json.loads(evidence_path.read_text())["rehearsal"], result)

    def test_frozen_pilot_is_exactly_twelve_train_clips_in_pinned_order(self):
        frozen = PilotSpec.from_frozen()
        self.assertEqual(len(frozen.presentation_order), 12)
        self.assertEqual(set(frozen.presentation_order), set(frozen.private_by_id))
        self.assertEqual(frozen.version, PILOT_VERSION)
        self.assertEqual(frozen.protocol_version, PROTOCOL_VERSION)
        self.assertEqual(frozen.tool_version, TOOL_VERSION)
        self.assertEqual(frozen.presentation_order_sha256,
                         hashlib.sha256((Path("artifacts/temporal_annotation/stage32f_train_pilot_v1") /
                                         "presentation_order.json").read_bytes()).hexdigest())
        self.assertTrue(all(row["split"] == "train" and row["subject_id"] in {1, 2, 3, 4, 8, 9}
                            for row in frozen.private_by_id.values()))
        self.assertFalse(any(row["subject_id"] in {5, 6, 7, 10} for row in frozen.private_by_id.values()))

    def test_next_clip_enforces_one_presentation_order_for_each_annotator(self):
        self.assertEqual(next_required_clip(self.spec, "A01", []), self.spec.presentation_order[0])
        first = blob("A01", mock_record(self.spec, "A01", self.spec.presentation_order[0], 1))
        self.assertEqual(next_required_clip(self.spec, "A01", [first]), self.spec.presentation_order[1])
        second = blob("A01", mock_record(self.spec, "A01", self.spec.presentation_order[1], 2))
        with self.assertRaises(ExecutionError):
            next_required_clip(self.spec, "A01", [second])
        with self.assertRaises(ExecutionError):
            next_required_clip(self.spec, "A02", [first])
        self.assertIsNone(next_required_clip(self.spec, "A01", full_files(self.spec)["A01"]))

    def test_unknown_pseudonym_and_unknown_clip_fail_closed(self):
        for identity in ("A03", "eric", "root", "", "A01/../A02"):
            with self.subTest(identity=identity), self.assertRaises(ExecutionError):
                mock_session(self.workspace, self.spec, identity)
        with self.assertRaises(ExecutionError):
            mock_session(self.workspace, self.spec, clip="clip-validation")

    def test_workspaces_and_public_state_are_separated_and_blinded(self):
        first = mock_session(self.workspace, self.spec, "A01")
        second = mock_session(self.workspace, self.spec, "A02")
        self.assertNotEqual(first.store.root, second.store.root)
        for owner in ANNOTATORS:
            for directory in ("drafts", "finalized", "logs"):
                self.assertTrue((self.root / owner / directory).is_dir())
        public = annotator_state(first)
        self.assertNotIn("source_video", public)
        self.assertNotIn("subject_id", public)
        self.assertNotIn("dataset_label", public)
        self.assertNotIn("A02", json.dumps(public))
        logs = list((self.root / "A01" / "logs").glob("*.json"))
        self.assertEqual(len(logs), 1)
        session_log = json.loads(logs[0].read_text())
        self.assertEqual(session_log["annotator_id"], "A01")
        self.assertEqual(session_log["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(session_log["pilot_version"], self.spec.version)
        self.assertEqual(session_log["tool_version"], TOOL_VERSION)
        self.assertEqual(session_log["code_commit"], "SYNTHETIC_UNCOMMITTED")
        self.assertTrue(session_log["synthetic_onboarding"])

    def test_owner_can_revise_draft_other_owner_cannot_load_it(self):
        first = mock_session(self.workspace, self.spec, "A01")
        first.set_event_presence("no_fall_observed")
        first.set_note("first synthetic draft")
        path = first.save_draft()
        self.assertIn("/A01/drafts/", str(path))
        record_id = first.record["annotation_record_id"]
        first.set_note("revised synthetic draft")
        first.save_draft()
        resumed = mock_session(self.workspace, self.spec, "A01", resume=record_id)
        self.assertEqual(resumed.record["annotator_note"], "revised synthetic draft")
        with self.assertRaises(FileNotFoundError):
            mock_session(self.workspace, self.spec, "A02", resume=record_id)
        with self.assertRaises((ExecutionError, AnnotationContractError, FileNotFoundError)):
            mock_session(self.workspace, self.spec, "A02", resume="../A01/drafts/" + record_id)

    def test_finalized_original_is_exclusive_and_cannot_be_reopened(self):
        session = mock_session(self.workspace, self.spec)
        session.set_event_presence("no_fall_observed")
        path = session.finalize()
        before = path.read_bytes()
        self.assertIn("/A01/finalized/", str(path))
        self.assertEqual(json.loads(before)["state"], "FINALIZED")
        with self.assertRaises(Exception):
            session.finalize()
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(Exception):
            mock_session(self.workspace, self.spec, "A01", resume=session.record["annotation_record_id"])
        another = mock_session(self.workspace, self.spec, "A01")
        another.set_event_presence("no_fall_observed")
        with self.assertRaisesRegex(ExecutionError, "already has a finalized"):
            another.finalize()

    def test_completion_rejects_partial_and_single_annotator(self):
        files = full_files(self.spec)
        files["A01"].pop()
        partial = check_completion(self.spec, files)
        self.assertFalse(partial.complete)
        self.assertEqual(partial.counts, {"A01": 11, "A02": 12})
        with self.assertRaises(ExecutionError):
            require_agreement_ready(self.spec, files, None, code_commit=CODE_COMMIT)
        files = full_files(self.spec)
        files["A02"] = []
        self.assertFalse(check_completion(self.spec, files).complete)
        self.assertFalse(check_completion(self.spec, {"A01": files["A01"]}).complete)

    def test_dual_12_of_12_passes_structural_completion_without_agreement(self):
        files = full_files(self.spec)
        with patch("ml.preprocessing.annotation_execution.validate_record", wraps=__import__(
                "ml.preprocessing.annotation_execution", fromlist=["validate_record"]).validate_record) as validator:
            result = check_completion(self.spec, files)
        self.assertEqual(validator.call_count, 24)
        self.assertTrue(result.complete)
        self.assertEqual(result.counts, {"A01": 12, "A02": 12})
        self.assertEqual(len(result.file_sha256), 24)
        self.assertEqual(result.errors, ())
        self.assertNotIn("agreement", repr(result).lower())
        self.assertNotIn("event_presence", repr(result))

    def test_unexpected_duplicate_protocol_and_invalid_record_fail(self):
        mutations = []
        extra = full_files(self.spec)
        bad = mock_record(self.spec, "A01", self.spec.presentation_order[0], 99)
        bad["source"]["neutral_clip_id"] = "clip-validation-test"
        extra["A01"].append(blob("A01", bad))
        mutations.append(extra)
        duplicate = full_files(self.spec)
        duplicate["A01"].append(blob("A01", mock_record(self.spec, "A01", self.spec.presentation_order[0], 99)))
        mutations.append(duplicate)
        protocol = full_files(self.spec)
        record = json.loads(protocol["A01"][0].content)["record"]
        record["protocol_version"] = "wrong"
        protocol["A01"][0] = blob("A01", record)
        mutations.append(protocol)
        invalid = full_files(self.spec)
        record = json.loads(invalid["A02"][0].content)["record"]
        record["event_presence"] = "invalid"
        invalid["A02"][0] = blob("A02", record)
        mutations.append(invalid)
        for files in mutations:
            with self.subTest(case=len(files["A01"]) + len(files["A02"])):
                self.assertFalse(check_completion(self.spec, files).complete)

    def test_pilot_tool_source_owner_and_state_mismatch_fail(self):
        cases = []
        for field, value in (("annotation_run_id", "wrong-pilot"), ("tool_version", "wrong-tool")):
            files = full_files(self.spec)
            record = json.loads(files["A01"][0].content)["record"]
            record["creation"][field] = value
            files["A01"][0] = blob("A01", record)
            cases.append(files)
        files = full_files(self.spec)
        record = json.loads(files["A01"][0].content)["record"]
        record["annotator"]["pseudonymous_id"] = "A02"
        files["A01"][0] = blob("A01", record)
        cases.append(files)
        files = full_files(self.spec)
        record = json.loads(files["A01"][0].content)["record"]
        record["source"]["source_avi_sha256"] = "f" * 64
        files["A01"][0] = blob("A01", record)
        cases.append(files)
        files = full_files(self.spec)
        wrapper = json.loads(files["A01"][0].content)
        wrapper["state"] = "DRAFT"
        files["A01"][0] = FinalizedBlob(files["A01"][0].relative_path, json.dumps(wrapper).encode())
        cases.append(files)
        for case in cases:
            self.assertFalse(check_completion(self.spec, case).complete)

    def test_raw_freeze_and_agreement_guard_require_exact_current_hashes(self):
        files = full_files(self.spec)
        with self.assertRaises(ExecutionError):
            require_agreement_ready(self.spec, files, None, code_commit=CODE_COMMIT)
        result = check_completion(self.spec, files)
        freeze = raw_freeze_record(self.spec, result, frozen_at_utc=NOW, code_commit=CODE_COMMIT)
        require_agreement_ready(self.spec, files, freeze, code_commit=CODE_COMMIT)
        changed = copy.deepcopy(files)
        changed["A01"][0] = FinalizedBlob(changed["A01"][0].relative_path,
                                         changed["A01"][0].content + b" ")
        with self.assertRaises(ExecutionError):
            require_agreement_ready(self.spec, changed, freeze, code_commit=CODE_COMMIT)
        with self.assertRaises(ExecutionError):
            require_agreement_ready(self.spec, files, dict(freeze, code_commit="b" * 40),
                                    code_commit=CODE_COMMIT)
        with self.assertRaises(ExecutionError):
            raw_freeze_record(self.spec, check_completion(self.spec, {"A01": files["A01"]}),
                              frozen_at_utc=NOW, code_commit=CODE_COMMIT)

    def test_synthetic_freeze_write_is_exclusive_and_collector_rejects_extra(self):
        files = full_files(self.spec)
        path = write_raw_freeze_new(self.root, self.spec, files, frozen_at_utc=NOW, code_commit=CODE_COMMIT)
        self.assertTrue(path.exists())
        with self.assertRaises(FileExistsError):
            write_raw_freeze_new(self.root, self.spec, files, frozen_at_utc=NOW, code_commit=CODE_COMMIT)
        self.assertEqual(collect_finalized(self.root), {"A01": [], "A02": []})
        (self.root / "A03").mkdir()
        with self.assertRaises(ExecutionError):
            collect_finalized(self.root)

    def test_synthetic_onboarding_cannot_use_real_pilot_namespace(self):
        with self.assertRaises(ExecutionError):
            ExecutionWorkspace(self.root.parent / "stage32f_train_pilot_v1", synthetic=True)
        with self.assertRaises(ExecutionError):
            ExecutionWorkspace(self.root, synthetic=False)
        with self.assertRaises(ExecutionError):
            mock_session(self.workspace, PilotSpec.from_frozen())
        self.assertFalse((self.root.parent / "stage32f_train_pilot_v1").exists())

    def test_protocol_questions_are_separate_and_semantic_remains_open(self):
        path = Path("configs/stage32e_manual_annotation_protocol.json")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        question = {"question_id": "synthetic-q1", "annotator_id": "A01",
                    "neutral_clip_id": self.spec.presentation_order[0],
                    "protocol_section": "boundary definition", "category": "semantic_protocol",
                    "description": "synthetic example", "created_at_utc": NOW, "status": "OPEN"}
        validate_protocol_question(question)
        stored = self.workspace.control.add_question(question)
        self.assertIn("_control/protocol_questions", str(stored))
        with self.assertRaises(ExecutionError):
            validate_protocol_question(dict(question, status="ANSWERED", resolution="change meaning"))
        procedural = dict(question, question_id="synthetic-q2", category="procedural_tool",
                          status="ANSWERED", resolution="Use the previous-frame button")
        validate_protocol_question(procedural)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertNotIn("event_presence", stored.read_text())

    def test_technical_incident_is_separate_and_blocking(self):
        session = mock_session(self.workspace, self.spec)
        session.set_event_presence("no_fall_observed")
        incident = {"incident_id": "synthetic-i1", "annotator_id": "A01",
                    "neutral_clip_id": self.spec.presentation_order[0],
                    "category": "frame_display_mismatch", "blocking": True,
                    "description": "synthetic display mismatch", "created_at_utc": NOW, "status": "OPEN"}
        validate_technical_incident(incident)
        path = self.workspace.control.add_incident(incident)
        self.assertIn("_control/technical_incidents", str(path))
        self.assertNotIn("event_presence", path.read_text())
        self.assertNotIn("unjudgeable", path.read_text())
        with self.assertRaisesRegex(ExecutionError, "Blocking unresolved"):
            session.finalize()
        self.assertFalse(any((self.root / "A01" / "finalized").iterdir()))
        resolved = dict(incident, status="RESOLVED", resolved_at_utc=NOW,
                        resolution_summary="synthetic issue resolved")
        validate_technical_incident(resolved)
        self.workspace.control.resolve_incident("synthetic-i1", resolved_at_utc=NOW,
                                                resolution_summary="synthetic issue resolved")
        self.assertEqual(json.loads(path.read_text()), resolved)
        self.assertTrue(session.finalize().exists())


if __name__ == "__main__":
    unittest.main()
