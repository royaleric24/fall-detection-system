"""Stage 2.6 orchestration tests: synthetic files, no inference or raw-data reads."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.datasets import preflight_pose_run as batch


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root / "sources"
        self.source.mkdir()
        self.metadata = [dict(subject_id=6, original_activity="Walk", binary_label="non_fall",
                              split="test", source_relative_video_path="Subject.6/Walk/WalkS6.avi",
                              source_fps=20.0, width=720, height=480, expected_frame_count=2)]
        self.video = self.source / self.metadata[0]["source_relative_video_path"]
        self.video.parent.mkdir(parents=True)
        self.video.write_bytes(b"synthetic source bytes, not media")

    def test_checksums_use_bytes_and_detect_changes(self):
        before = batch.source_checksums(self.metadata, self.source)
        self.assertEqual(before[0]["sha256"], batch.core.sha256(self.video))
        self.video.write_bytes(b"changed")
        self.assertNotEqual(before, batch.source_checksums(self.metadata, self.source))

    def test_extra_missing_duplicate_and_redirected_sources_rejected(self):
        extra = self.source / "extra.AVI"
        extra.write_bytes(b"extra")
        with self.assertRaises(ValueError):
            batch.source_checksums(self.metadata, self.source)
        extra.unlink()
        with self.assertRaises(ValueError):
            batch.source_checksums(self.metadata * 2, self.source)
        self.video.unlink()
        with self.assertRaises(ValueError):
            batch.source_checksums(self.metadata, self.source)
        target = self.root / "outside.avi"
        target.write_bytes(b"target")
        self.video.symlink_to(target)
        with self.assertRaises(ValueError):
            batch.source_checksums(self.metadata, self.source)

    def test_pending_manifest_has_unknown_counters_and_run_identity(self):
        checksums = batch.source_checksums(self.metadata, self.source)
        row, = batch.manifest_rows(self.metadata, checksums, "new-run")
        self.assertEqual(set(row), set(batch.FIELDS))
        self.assertEqual(row["run_id"], "new-run")
        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["split"], "test")
        self.assertEqual(row["output_relative_path"], "Subject.6/Walk/WalkS6.npz")
        for key in ("extracted_frame_count", "pose_detected_count", "pose_missing_count"):
            self.assertEqual(row[key], "")
        with self.assertRaises(ValueError):
            batch.manifest_rows(self.metadata, [], "new-run")

    def test_real_inventory_scope_without_reading_raw_sources(self):
        # Metadata only; intercept development helper's source-path existence check.
        with patch.object(batch.core, "source_metadata") as helper:
            def metadata_for(source):
                path = source.relative_to(batch.core.SOURCE_ROOT).as_posix()
                row = next(r for r in batch.read_csv(batch.core.INVENTORY)
                           if r["relative_video_path"] == path)
                config = json.loads(batch.core.SPLIT_CONFIG.read_text())
                subject = int(row["subject"].split(".")[1])
                return dict(subject_id=subject, original_activity=row["activity"],
                            binary_label=row["label"], split=next(k for k in ("train", "validation") if subject in config[k]),
                            source_relative_video_path=path, source_fps=float(row["fps"]),
                            width=int(float(row["width"])), height=int(float(row["height"])),
                            expected_frame_count=int(float(row["frame_count"])))
            helper.side_effect = metadata_for
            rows = batch.inventory_metadata()
        self.assertEqual(len(rows), 100)
        self.assertEqual(sum(r["expected_frame_count"] for r in rows), 19877)
        self.assertEqual(helper.call_count, 80)
        self.assertEqual({r["subject_id"] for r in rows if r["split"] == "test"}, {6, 7})
        self.assertEqual([r["subject_id"] for r in rows], sorted(r["subject_id"] for r in rows))

    def test_dirty_state_refused_before_sources_or_runtime(self):
        with patch.object(batch, "git", return_value="?? new.py"), \
                patch.object(batch, "inventory_metadata") as inv, \
                patch.object(batch, "runtime") as runtime:
            with self.assertRaisesRegex(ValueError, "clean worktree"):
                batch.prepare()
            inv.assert_not_called()
            runtime.assert_not_called()

    def test_changed_checkpoint_file_refused(self):
        with patch.object(batch.core, "REPO", self.root), \
                patch.object(batch, "FROZEN_FILES", ("core.py",)), \
                patch.object(batch, "git"), \
                patch.object(batch.subprocess, "check_output", return_value=b"frozen"):
            (self.root / "core.py").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
                batch.verify_checkpoint()

    def test_runtime_or_model_mismatch_refused(self):
        p = dict(python="3.13.15", packages={"mediapipe": "0.10.35",
                 "opencv-contrib-python": "4.12.0.88", "numpy": "2.2.6"}, model_sha256="wrong")
        with patch.object(batch.core, "runtime_provenance", return_value=p):
            with self.assertRaisesRegex(ValueError, "Model checksum"):
                batch.runtime()
            p["packages"]["mediapipe"] = "wrong"
            with self.assertRaisesRegex(ValueError, "Runtime differs"):
                batch.runtime()

    def prepare_fixture(self, dirty=False):
        from contextlib import ExitStack
        stack = ExitStack()
        self.addCleanup(stack.close)
        script = self.root / "ml/datasets/preflight_pose_run.py"
        script.parent.mkdir(parents=True)
        script.write_bytes(b"test orchestration")
        model = self.root / "model.task"
        model.write_bytes(b"test model")
        def git(*args):
            return ("?? new.py" if dirty else "") if args[0] == "status" else "committed-head"
        for obj, key, value in ((batch.core, "REPO", self.root), (batch.core, "SOURCE_ROOT", self.source),
                                (batch.core, "MODEL", model), (batch, "__file__", str(script))):
            stack.enter_context(patch.object(obj, key, value))
        stack.enter_context(patch.object(batch, "git", side_effect=git))
        stack.enter_context(patch.object(batch, "verify_checkpoint", return_value={}))
        stack.enter_context(patch.object(batch, "inventory_metadata", return_value=self.metadata))
        stack.enter_context(patch.object(batch, "runtime", return_value={"model_sha256": batch.core.sha256(model)}))
        tests = stack.enter_context(patch.object(batch, "run_tests", return_value={"command": ["test"], "exit_code": 0, "output": "OK"}))
        return tests

    def test_check_only_writes_nothing_even_when_dirty(self):
        self.prepare_fixture(dirty=True)
        result = batch.prepare(check_only=True)
        self.assertFalse(result["artifacts_created"])
        self.assertIsNone(result["run_id"])
        self.assertFalse((self.root / "artifacts").exists())

    def test_preflight_publishes_only_pending_new_namespace(self):
        self.prepare_fixture()
        result = batch.prepare()
        run_dir = Path(result["run_directory"])
        self.assertEqual(run_dir.parent, self.root / "artifacts/pose_extraction/runs")
        provenance = json.loads((run_dir / "extraction_run.json").read_text())
        self.assertFalse(provenance["git_dirty"])
        self.assertEqual(provenance["git_commit"], "committed-head")
        self.assertEqual(provenance["status"], "pending")
        self.assertEqual(provenance["run_id"], result["run_id"])
        self.assertEqual(batch.read_csv(run_dir / "manifest.csv")[0]["status"], "pending")
        self.assertEqual(list((run_dir / "results").iterdir()), [])
        self.assertFalse(Path(result["output_root"]).exists())
        self.assertFalse(list(run_dir.parent.glob(".preflight-*")))

    def test_failed_tests_publish_no_run(self):
        tests = self.prepare_fixture()
        tests.side_effect = ValueError("tests failed")
        with self.assertRaisesRegex(ValueError, "tests failed"):
            batch.prepare()
        self.assertFalse((self.root / "artifacts").exists())

    def test_changed_source_during_tests_publish_no_run(self):
        tests = self.prepare_fixture()
        def change():
            self.video.write_bytes(b"changed during tests")
            return {"output": "OK"}
        tests.side_effect = change
        with self.assertRaisesRegex(ValueError, "Source/model changed"):
            batch.prepare()
        self.assertFalse((self.root / "artifacts").exists())

    def test_failed_publication_cleans_staging(self):
        self.prepare_fixture()
        with patch.object(batch.os, "rename", side_effect=OSError("injected")):
            with self.assertRaisesRegex(OSError, "injected"):
                batch.prepare()
        self.assertEqual(list((self.root / "artifacts/pose_extraction/runs").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
