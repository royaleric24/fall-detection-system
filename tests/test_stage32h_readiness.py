"""Readiness tests: temporary namespaces and generated frames, never real AVI."""

from __future__ import annotations

import copy
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ml.preprocessing import annotation_readiness as readiness
from ml.preprocessing.annotation_execution import ExecutionError, PilotSpec, require_agreement_ready
from ml.preprocessing.annotation_pilot import IndexedFrameSource, PilotError, load_frozen
from ml.preprocessing.annotation_tool import BOUNDARIES, COORDINATES, annotator_payload
from ml.preprocessing.annotation_web import annotator_state, frame_png_bytes, perform_action


class Stage32hReadinessTests(unittest.TestCase):
    RUNTIME_COMMIT = "a" * 40  # Mock commit identity used only in temporary tests.

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="stage32h-unit-test-")
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name).resolve()
        self.root = base / "genuine_test_namespace"
        self.synthetic = base / "synthetic_onboarding"
        for name, value in (("REAL_ROOT", self.root), ("ONBOARDING_ROOT", self.synthetic)):
            patcher = patch.object(readiness, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(readiness, "_head", return_value=readiness.PARENT)
        patcher.start()
        self.addCleanup(patcher.stop)

    def bind_fixture(self):
        # Commit-tree inspection is exercised separately; no test binds the real run.
        with patch.object(readiness, "_verify_readiness_commit"):
            return readiness.bind_runtime(self.RUNTIME_COMMIT)

    def bound_launch(self, identity, owner="A01", **changes):
        values = dict(run_id=identity["run_id"], annotator_id=owner,
                      workspace=self.root / owner, pilot_version=readiness.PILOT_VERSION,
                      protocol_version=readiness.PROTOCOL_VERSION, tool_version=readiness.TOOL_VERSION,
                      execution_version=readiness.EXECUTION_VERSION)
        values.update(changes)
        with patch.object(readiness, "_verify_readiness_commit"):
            return readiness.prepare_launch(**values)

    def test_initialization_creates_one_identity_empty_workspaces_and_zero_records(self):
        result = readiness.initialize_run()
        self.assertEqual(result["finalized_counts"], {"A01": 0, "A02": 0})
        self.assertEqual(result["expected_per_annotator"], 12)
        self.assertEqual(result["annotation_record_count"], 0)
        self.assertEqual(result["genuine_session_count"], 0)
        self.assertEqual(result["agreement_state"], "LOCKED")
        self.assertNotIn("event_presence", result)
        self.assertNotIn("boundaries", result)
        for root in (self.root, self.synthetic):
            for owner in ("A01", "A02"):
                for area in ("drafts", "finalized", "logs"):
                    self.assertEqual(list((root / owner / area).iterdir()), [])
        self.assertEqual(list(self.root.rglob("*.json")), [readiness._active_path()])
        before = readiness._active_path().read_bytes()
        self.assertEqual(readiness.initialize_run(), result)
        self.assertEqual(readiness._active_path().read_bytes(), before)
        self.assertEqual(len(list((self.root / "_control/run_initialization").iterdir())), 1)

    def test_existing_unbound_workspace_is_not_adopted(self):
        self.root.mkdir()
        with self.assertRaisesRegex(ExecutionError, "Unbound"):
            readiness.initialize_run()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_namespace_alias_and_symlink_fail_before_initialization(self):
        with patch.object(readiness, "ONBOARDING_ROOT", self.root / "synthetic_onboarding"):
            with self.assertRaisesRegex(ExecutionError, "separate"):
                readiness.initialize_run()
        self.synthetic.mkdir()
        self.root.symlink_to(self.synthetic, target_is_directory=True)
        with self.assertRaisesRegex(ExecutionError, "Symlink"):
            readiness.initialize_run()
        self.assertEqual(list(self.synthetic.iterdir()), [])

    def test_frozen_versions_order_and_code_match_parent(self):
        result = readiness.initialize_run()
        frozen = result["frozen"]
        self.assertEqual(result["parent_commit"], readiness.PARENT)
        self.assertEqual(frozen["tool_code_commit"], readiness.PARENT)
        spec = PilotSpec.from_frozen()
        self.assertEqual(frozen["expected_neutral_clip_ids_in_presentation_order"],
                         list(spec.presentation_order))
        self.assertEqual(len(spec.presentation_order), 12)
        self.assertEqual(frozen["presentation_order_sha256"], spec.presentation_order_sha256)
        self.assertEqual(frozen["protocol_version"], "stage32e_v1")
        self.assertEqual(frozen["pilot_version"], "stage32f_train_pilot_v1")
        self.assertEqual(frozen["tool_version"], "stage32f_local_tk_v1")
        self.assertEqual(frozen["execution_protocol_version"], "stage32g_execution_v1")
        self.assertEqual(set(frozen["file_sha256"]), set(readiness.FROZEN_FILES))
        self.assertEqual(result["record_annotation_run_id"], spec.version)

    def test_unknown_annotator_rejected_before_launch_or_onboarding(self):
        result = readiness.initialize_run()
        for owner in ("A03", "eric", "", "A01/../A02"):
            with self.subTest(owner=owner), self.assertRaises(ExecutionError):
                self.bound_launch(result, owner)
            with self.subTest(owner=owner), self.assertRaises(ExecutionError):
                readiness.synthetic_session(owner)
        self.assertEqual(readiness.empty_readiness()["annotation_record_count"], 0)

    def test_launch_is_run_owner_workspace_and_frozen_version_bound_without_execution(self):
        result = readiness.initialize_run()
        self.bind_fixture()
        original_run = subprocess.run

        def metadata_only(command, **kwargs):
            if command[0] != "git":
                raise AssertionError("Launch forbidden")
            return original_run(command, **kwargs)

        with patch("subprocess.run", side_effect=metadata_only), \
                patch.object(IndexedFrameSource, "open_avi", side_effect=AssertionError("AVI forbidden")):
            for owner in ("A01", "A02"):
                command = self.bound_launch(result, owner)
                self.assertEqual(command[command.index("--annotator-id") + 1], owner)
                self.assertEqual(command[command.index("--neutral-clip-id") + 1],
                                 result["frozen"]["expected_neutral_clip_ids_in_presentation_order"][0])
                self.assertTrue(command[command.index("--session-id") + 1].startswith(
                    result["run_id"] + "-" + owner + "-"))
        for change in ({"run_id": "wrong"}, {"workspace": self.root / "A02"},
                       {"workspace": self.synthetic / "A01"}, {"pilot_version": "changed"},
                       {"protocol_version": "changed"}, {"tool_version": "changed"},
                       {"execution_version": "changed"}):
            with self.subTest(change=change), self.assertRaises(ExecutionError):
                self.bound_launch(result, **change)
        self.assertEqual(readiness.empty_readiness(), result)

    def test_tampered_active_run_or_additional_identity_fails_closed(self):
        result = readiness.initialize_run()
        path = readiness._active_path()
        original = path.read_bytes()
        path.chmod(0o644)
        for key, value in (("agreement_state", "UNLOCKED"), ("parent_commit", "a" * 40),
                           ("record_annotation_run_id", result["run_id"]),
                           ("session_id_rule", "unbound")):
            changed = copy.deepcopy(result)
            changed[key] = value
            path.write_text(json.dumps(changed))
            with self.subTest(key=key), self.assertRaises(ExecutionError):
                self.bound_launch(result)
        changed = copy.deepcopy(result)
        changed["frozen"]["expected_neutral_clip_ids_in_presentation_order"].reverse()
        path.write_text(json.dumps(changed))
        with self.assertRaises(ExecutionError):
            self.bound_launch(result)
        path.write_bytes(original)
        (path.parent / "second_run.json").write_text("{}")
        with self.assertRaisesRegex(ExecutionError, "additional"):
            self.bound_launch(result)

    def test_initialization_and_prepared_launch_need_no_dataset_or_avi_access(self):
        original_open = io.open

        def guarded_open(file, *args, **kwargs):
            if isinstance(file, (str, Path)):
                path = Path(file).resolve()
                if path == readiness.REPO / "data" or readiness.REPO / "data" in path.parents:
                    raise AssertionError("Dataset contents forbidden")
                if path.suffix.lower() in {".avi", ".npy", ".npz", ".png", ".txt"}:
                    raise AssertionError("Media and pose contents forbidden")
            return original_open(file, *args, **kwargs)

        with patch("io.open", side_effect=guarded_open), \
                patch.object(IndexedFrameSource, "open_avi", side_effect=AssertionError("AVI forbidden")), \
                patch("ml.preprocessing.annotation_pilot.iter_indexed_frames",
                      side_effect=AssertionError("Decoder forbidden")):
            result = readiness.initialize_run()
            self.bind_fixture()
            self.bound_launch(result)
        self.assertFalse(result["readiness_checks"]["dataset_open_required"])

    def test_blinded_public_projection_and_no_heldout_clip_selection(self):
        result = readiness.initialize_run()
        spec = PilotSpec.from_frozen()
        self.assertTrue(all(row["split"] == "train" and row["subject_id"] in {1, 2, 3, 4, 8, 9}
                            for row in spec.private_by_id.values()))
        for clip in ("clip-validation", "clip-test", "Subject.6", "Subject.7", "Subject.5", "Subject.10"):
            with self.subTest(clip=clip), self.assertRaises(ExecutionError):
                spec.source_fields(clip)
        _, public, _ = load_frozen()
        for row in public:
            self.assertEqual(set(annotator_payload(row)), {"neutral_clip_id", "frame_count", "duration_ms"})
            for key in ("subject_id", "activity", "dataset_label", "source_video", "source_path",
                        "PNG_TXT", "pose", "MediaPipe", "model", "A02_records"):
                with self.subTest(key=key), self.assertRaises(PilotError):
                    annotator_payload(dict(row, **{key: "private"}))
        session = readiness.synthetic_session("A01")
        state = annotator_state(session)
        self.assertEqual(set(state), {"neutral_clip_id", "frame_count", "duration_ms", "frame_index",
                                     "time_ms", "event_presence", "event_presence_reason_flags",
                                     "boundaries", "annotator_note", "finalized"})
        self.assertNotIn("A02", json.dumps(state))
        self.assertEqual(readiness.empty_readiness(), result)

    def test_agreement_remains_locked_and_frozen_guard_denies_empty_streams(self):
        result = readiness.initialize_run()
        with self.assertRaisesRegex(ExecutionError, "blocked"):
            require_agreement_ready(PilotSpec.from_frozen(), {"A01": [], "A02": []}, None,
                                    code_commit=readiness.PARENT)
        self.assertEqual(readiness.empty_readiness()["agreement_state"], "LOCKED")
        self.assertEqual(result["agreement_state"], "LOCKED")

    def test_readiness_rejects_any_genuine_record_session_or_raw_freeze(self):
        readiness.initialize_run()
        for area in ("A01/drafts", "A01/finalized", "A02/logs", "_control/raw_freezes"):
            path = self.root / area / "unexpected.json"
            path.write_text("{}")
            with self.subTest(area=area), self.assertRaises(ExecutionError):
                readiness.empty_readiness()
            path.unlink()
        readiness.empty_readiness()

    def test_synthetic_onboarding_navigation_resume_boundaries_and_finalization_are_isolated(self):
        result = readiness.initialize_run()
        with patch.object(IndexedFrameSource, "open_avi", side_effect=AssertionError("AVI forbidden")):
            session = readiness.synthetic_session("A01")
            self.assertEqual(session.record["creation"]["annotation_run_id"], readiness.ONBOARDING_VERSION)
            self.assertNotIn(session.public["neutral_clip_id"], PilotSpec.from_frozen().presentation_order)
            self.assertTrue(frame_png_bytes(session.frames.peek(0)[1]).startswith(b"\x89PNG"))
            perform_action(session, {"verb": "step", "amount": 1})
            self.assertEqual(annotator_state(session)["frame_index"], 1)
            perform_action(session, {"verb": "jump", "index": 7})
            self.assertEqual(annotator_state(session)["time_ms"], 350)
            session.set_event_presence("fall_observed")
            for boundary, index in zip(BOUNDARIES, (2, 5, 9)):
                session.set_status(boundary, "observed")
                session.frames.at(index)
                for coordinate in COORDINATES:
                    session.mark(boundary, coordinate)
            session.set_note("SYNTHETIC UNIT TEST ONLY: no human or real clip")
            session.save_draft()
            record_id = session.record["annotation_record_id"]
            resumed = readiness.synthetic_session("A01", resume_record_id=record_id)
            self.assertEqual(resumed.record, session.record)
            with self.assertRaises(FileNotFoundError):
                readiness.synthetic_session("A02", resume_record_id=record_id)
            final = resumed.finalize()
            before = final.read_bytes()
            with self.assertRaises(Exception):
                resumed.finalize()
            self.assertEqual(final.read_bytes(), before)
            self.assertTrue(final.is_relative_to(self.synthetic / "A01/finalized"))
            self.assertEqual(readiness.empty_readiness(), result)

    def test_genuine_launcher_requires_post_freeze_binding_before_any_video(self):
        result = readiness.initialize_run()
        with patch.object(IndexedFrameSource, "open_avi", side_effect=AssertionError("AVI forbidden")):
            for owner in ("A01", "A02"):
                with self.subTest(owner=owner), self.assertRaisesRegex(ExecutionError, "must be bound"):
                    self.bound_launch(result, owner)
        self.assertEqual(readiness.empty_readiness(), result)

    def test_runtime_binding_preserves_run_and_marker_and_is_not_replaceable(self):
        result = readiness.initialize_run()
        original = readiness._active_path().read_bytes()
        binding = self.bind_fixture()
        self.assertEqual(binding["run_id"], result["run_id"])
        self.assertEqual(binding["runtime_commit"], self.RUNTIME_COMMIT)
        self.assertEqual(binding["parent_commit"], readiness.PARENT)
        self.assertEqual(binding["runtime_requirements"], readiness.RUNTIME_REQUIREMENTS)
        self.assertEqual(readiness._active_path().read_bytes(), original)
        before = readiness._binding_path().read_bytes()
        self.assertEqual(self.bind_fixture(), binding)
        self.assertEqual(readiness._binding_path().read_bytes(), before)
        with patch.object(readiness, "_verify_readiness_commit"), self.assertRaises(ExecutionError):
            readiness.bind_runtime("b" * 40)
        self.assertEqual(readiness.empty_readiness(), result)

    def test_wrong_head_rejected_by_genuine_launcher_before_video_or_child_process(self):
        result = readiness.initialize_run()
        self.bind_fixture()
        arguments = dict(run_id=result["run_id"], annotator_id="A01", workspace=self.root / "A01",
                         pilot_version=readiness.PILOT_VERSION, protocol_version=readiness.PROTOCOL_VERSION,
                         tool_version=readiness.TOOL_VERSION, execution_version=readiness.EXECUTION_VERSION)
        original_run = subprocess.run

        def metadata_only(command, **kwargs):
            if command[0] != "git":
                raise AssertionError("Child launch forbidden")
            return original_run(command, **kwargs)

        with patch.object(IndexedFrameSource, "open_avi", side_effect=AssertionError("AVI forbidden")), \
                patch("subprocess.run", side_effect=metadata_only):
            for wrong in (readiness.PARENT, "b" * 40):
                with patch.object(readiness, "_head", return_value=wrong), \
                        self.assertRaisesRegex(ExecutionError, "HEAD does not match"):
                    readiness.prepare_launch(**arguments)
        self.assertEqual(readiness.empty_readiness(), result)

    def test_runtime_requirements_fail_closed_independent_of_cache_path(self):
        result = readiness.initialize_run()
        self.bind_fixture()
        with patch.object(IndexedFrameSource, "open_avi", side_effect=AssertionError("AVI forbidden")):
            with patch("platform.python_version", return_value="3.13.14"), \
                    self.assertRaisesRegex(ExecutionError, "Python runtime"):
                self.bound_launch(result)
            with patch("importlib.metadata.version", return_value="4.12.0.87"), \
                    self.assertRaisesRegex(ExecutionError, "OpenCV runtime"):
                self.bound_launch(result)
            with patch("cv2.__version__", "wrong"), self.assertRaisesRegex(ExecutionError, "OpenCV runtime"):
                self.bound_launch(result)
        self.assertEqual(readiness.verify_runtime(), readiness.RUNTIME_REQUIREMENTS)

    def test_commit_guard_requires_committed_readiness_bytes_and_clean_index(self):
        def git_metadata(command, **kwargs):
            self.assertEqual(command[0], "git")
            if command[1] == "rev-parse":
                return readiness.PARENT + "\n"
            relative = command[2].split(":", 1)[1]
            return (readiness.REPO / relative).read_bytes()

        with patch.object(readiness, "_head", return_value=self.RUNTIME_COMMIT), \
                patch("subprocess.check_output", side_effect=git_metadata), \
                patch("subprocess.run", return_value=SimpleNamespace(returncode=0)):
            readiness._verify_readiness_commit(self.RUNTIME_COMMIT)
            with patch("subprocess.run", return_value=SimpleNamespace(returncode=1)), \
                    self.assertRaisesRegex(ExecutionError, "working tree/index"):
                readiness._verify_readiness_commit(self.RUNTIME_COMMIT)
            with patch.object(readiness, "sha256_file", return_value="0" * 64), \
                    self.assertRaisesRegex(ExecutionError, "differs from bound"):
                readiness._verify_readiness_commit(self.RUNTIME_COMMIT)

    def test_runtime_binding_rejects_run_or_marker_provenance_tampering(self):
        result = readiness.initialize_run()
        binding = self.bind_fixture()
        path = readiness._binding_path()
        path.chmod(0o644)
        for key, value in (("run_id", "wrong"), ("active_run_sha256", "0" * 64),
                           ("runtime_requirements", {}), ("parent_commit", "b" * 40)):
            changed = dict(binding, **{key: value})
            path.write_text(json.dumps(changed))
            with self.subTest(key=key), self.assertRaisesRegex(ExecutionError, "binding identity"):
                self.bound_launch(result)
        self.assertEqual(readiness.empty_readiness(), result)


if __name__ == "__main__":
    unittest.main()
