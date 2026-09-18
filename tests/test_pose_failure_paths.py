"""Hermetic CLI and publication failure injection; no native inference or raw data."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from ml.datasets import extract_pose_video as extractor
from ml.datasets import pose_raw


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.output, self.report = self.root / "outputs/a.npz", self.root / "results/a.json"
        self.arrays = dict(frame_index=np.arange(2, dtype=np.int32),
                           timestamp_ms=np.array([0, 50], dtype=np.int64),
                           pose_detected=np.array([True, False], dtype=bool),
                           landmarks=np.full((2, 33, 4), np.nan, dtype=np.float32))
        self.arrays["landmarks"][0] = .1
        self.metadata = dict(source_fps=20., expected_frame_count=2)
        self.result = dict(schema_version="pose_raw_v1", status="running", provenance={"test": True})

    def publish(self):
        return extractor.publish_video_pair(self.output, self.report, self.arrays,
                                            self.metadata, self.result)

    def assert_clean(self):
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())
        self.assertEqual([p for p in self.root.rglob("*") if p.is_file()], [])
        self.assertEqual(list(self.root.rglob(".pose*")), [])

    def test_success_publishes_validated_npz_before_result_marker(self):
        original = os.link
        publications = []
        def link(source, target):
            if target == self.output:
                self.assertFalse(self.report.exists())
                publications.append("npz")
            elif target == self.report:
                pose_raw.load_validated(self.output, 20, 2)
                publications.append("result")
            return original(source, target)
        with patch.object(extractor.os, "link", side_effect=link):
            result = self.publish()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(publications, ["npz", "result"])
        pose_raw.load_validated(self.output, 20, 2)
        self.assertEqual(json.loads(self.report.read_text()), result)
        self.assertEqual(list(self.root.rglob(".pose*")), [])

    def test_partial_result_write_failure_leaves_no_final_files(self):
        original = Path.write_text
        def write(path, content, **kwargs):
            original(path, content[:10], **kwargs)
            raise OSError("injected partial result write")
        with patch.object(Path, "write_text", write), self.assertRaises(pose_raw.ExtractionError) as caught:
            self.publish()
        self.assertEqual(caught.exception.error_type, "output_persistence_failure")
        self.assert_clean()

    def test_corrupt_result_reload_leaves_no_final_files(self):
        with patch.object(Path, "read_text", return_value="{}"), self.assertRaises(pose_raw.ExtractionError):
            self.publish()
        self.assert_clean()

    def test_result_serialization_failure_leaves_no_final_files(self):
        self.result["invalid"] = float("nan")
        with self.assertRaises(pose_raw.ExtractionError):
            self.publish()
        self.assert_clean()

    def test_result_and_npz_publication_failures_roll_back(self):
        original = os.link
        for destination in (self.report, self.output):
            def link(source, target):
                if target == destination:
                    self.assertFalse(self.report.exists())
                    self.assertEqual(self.output.exists(), destination == self.report)
                    raise OSError("injected publication failure")
                return original(source, target)
            with self.subTest(destination=destination), patch.object(extractor.os, "link", side_effect=link), \
                    self.assertRaises(pose_raw.ExtractionError):
                self.publish()
            self.assert_clean()

    def test_uncatchable_exit_before_result_publication_leaves_only_orphan_npz(self):
        # A child process exits without Python cleanup, modelling the exact crash
        # boundary. The outer fixture removes its leftover staging directories.
        script = '''
import os, sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ml.datasets.extract_pose_video import publish_video_pair
root = Path(sys.argv[1])
output, report = root / "outputs/a.npz", root / "results/a.json"
arrays = dict(frame_index=np.arange(2, dtype=np.int32),
              timestamp_ms=np.array([0, 50], dtype=np.int64),
              pose_detected=np.zeros(2, dtype=bool),
              landmarks=np.full((2, 33, 4), np.nan, dtype=np.float32))
original = os.link
def link(source, target):
    if target == report:
        os._exit(23)
    return original(source, target)
with patch("os.link", side_effect=link):
    publish_video_pair(output, report, arrays,
                       dict(source_fps=20., expected_frame_count=2),
                       dict(schema_version="pose_raw_v1", status="running"))
'''
        process = subprocess.run([sys.executable, "-B", "-c", script, str(self.root)],
                                 cwd=extractor.REPO, capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 23, process.stderr)
        pose_raw.load_validated(self.output, 20, 2)
        self.assertFalse(self.report.exists(), "Orphan NPZ must have no complete result marker")

    def test_interrupt_after_either_final_link_rolls_back(self):
        original = os.link
        for destination in (self.report, self.output):
            def link(source, target):
                original(source, target)
                if target == destination:
                    raise KeyboardInterrupt()
            with self.subTest(destination=destination), patch.object(extractor.os, "link", side_effect=link), \
                    self.assertRaises(KeyboardInterrupt):
                self.publish()
            self.assert_clean()

    def test_fsync_failure_cleans_staged_files(self):
        with patch.object(extractor.os, "fsync", side_effect=OSError("sync failure")), \
                self.assertRaises(pose_raw.ExtractionError):
            self.publish()
        self.assert_clean()

    def test_existing_file_or_dangling_symlink_is_preserved(self):
        for destination in (self.report, self.output):
            destination.parent.mkdir(parents=True, exist_ok=True)
            for symlink in (False, True):
                if symlink:
                    destination.symlink_to(self.root / "absent")
                else:
                    destination.write_bytes(b"existing evidence")
                with self.assertRaises(pose_raw.ExtractionError):
                    self.publish()
                if symlink:
                    self.assertTrue(destination.is_symlink())
                else:
                    self.assertEqual(destination.read_bytes(), b"existing evidence")
                destination.unlink()
                self.assert_clean()

    def test_concurrent_destination_is_not_removed_by_rollback(self):
        original = os.link
        for destination in (self.report, self.output):
            def link(source, target):
                if target == destination:
                    target.write_bytes(b"other writer")
                return original(source, target)
            with self.subTest(destination=destination), patch.object(extractor.os, "link", side_effect=link), \
                    self.assertRaises(pose_raw.ExtractionError):
                self.publish()
            self.assertEqual(destination.read_bytes(), b"other writer")
            destination.unlink()
            self.assert_clean()


class CLIFailureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.source = self.root / "source.avi"; self.source.write_bytes(b"synthetic source")
        self.model = self.root / "model.task"; self.model.write_bytes(b"synthetic model")
        self.output = self.root / "outputs/Subject.2/Walk/video.npz"
        self.report = self.root / "results/Subject.2/Walk/video.json"
        self.metadata = dict(source_fps=20., expected_frame_count=2, width=720, height=480,
                             subject_id=2, original_activity="Walk", binary_label="non_fall",
                             split="train", source_relative_video_path="Subject.2/Walk/video.avi")
        self.arrays = dict(frame_index=np.arange(2, dtype=np.int32),
                           timestamp_ms=np.array([0, 50], dtype=np.int64),
                           pose_detected=np.zeros(2, dtype=bool),
                           landmarks=np.full((2, 33, 4), np.nan, dtype=np.float32))

    def invoke(self, effect=None):
        def extract(*args):
            if effect:
                effect()
            args[3].update(decoded_frame_count=2, extracted_frame_count=2, pose_missing_count=2)
            return self.arrays
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["extract", str(self.source), "--model",
                                str(self.model), "--output-root", str(self.root / "outputs")]))
            stack.enter_context(patch.object(extractor, "source_metadata", return_value=self.metadata))
            stack.enter_context(patch.object(extractor, "RESULT_ROOT", self.root / "results"))
            stack.enter_context(patch.object(extractor, "MODEL_SHA256", hashlib.sha256(b"synthetic model").hexdigest()))
            stack.enter_context(patch.object(extractor, "runtime_provenance", return_value={}))
            stack.enter_context(patch.object(extractor.platform, "python_version", return_value="3.13.15"))
            stack.enter_context(patch.object(extractor.importlib.metadata, "version", return_value="4.12.0.88"))
            stack.enter_context(patch.dict(sys.modules, {"cv2": NS(__version__="4.12.0"),
                                                        "mediapipe": NS(__version__="0.10.35")}))
            inference = stack.enter_context(patch.object(extractor, "extract_arrays", side_effect=extract))
            logger = stack.enter_context(patch.object(extractor.LOGGER, "info"))
            code = extractor.main()
            result = json.loads(logger.call_args.args[0])
            return code, result, inference.call_count

    def assert_failure(self, code, result, error_type, status="failed"):
        self.assertNotEqual(code, 0)
        self.assertEqual(result["status"], status)
        self.assertEqual(result["error_type"], error_type)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())
        self.assertEqual(list(self.root.rglob(".pose*")), [])

    def test_cli_success(self):
        code, result, calls = self.invoke()
        self.assertEqual((code, result["status"], calls), (0, "complete", 1))
        pose_raw.load_validated(self.output, 20, 2)
        self.assertEqual(json.loads(self.report.read_text()), result)

    def test_cli_model_hash_mismatch_before_inference(self):
        self.model.write_bytes(b"wrong model")
        code, result, calls = self.invoke()
        self.assert_failure(code, result, "provenance_mismatch")
        self.assertEqual(calls, 0)

    def test_cli_source_and_model_changed_during_extraction(self):
        for target in (self.source, self.model):
            original = target.read_bytes()
            code, result, _ = self.invoke(lambda: target.write_bytes(b"changed"))
            self.assert_failure(code, result, "provenance_mismatch")
            target.write_bytes(original)

    def test_cli_interruption(self):
        def interrupt():
            raise KeyboardInterrupt()
        code, result, _ = self.invoke(interrupt)
        self.assert_failure(code, result, "interrupted_run", status="incomplete")

    def test_cli_known_failure_preserves_category_and_frame(self):
        def fail():
            raise pose_raw.ExtractionError("inference_runtime_failure", "test failure", 1)
        code, result, _ = self.invoke(fail)
        self.assert_failure(code, result, "inference_runtime_failure")
        self.assertEqual(result["error_frame_index"], 1)

    def test_cli_result_write_failure_is_nonzero(self):
        with patch.object(Path, "write_text", side_effect=OSError("result write failure")):
            code, result, _ = self.invoke()
        self.assert_failure(code, result, "output_persistence_failure")

    def test_cli_result_publication_failure_is_nonzero(self):
        original = os.link
        def link(source, target):
            if target == self.report:
                raise OSError("result publication failure")
            return original(source, target)
        with patch.object(extractor.os, "link", side_effect=link):
            code, result, _ = self.invoke()
        self.assert_failure(code, result, "output_persistence_failure")

    def test_cli_existing_result_prevents_inference_and_is_preserved(self):
        self.report.parent.mkdir(parents=True); self.report.write_bytes(b"prior result")
        code, result, calls = self.invoke()
        self.assertEqual((code, result["status"], calls), (1, "failed", 0))
        self.assertEqual(self.report.read_bytes(), b"prior result")
        self.assertFalse(self.output.exists())

    def test_cli_orphan_npz_is_not_accepted_as_complete(self):
        pose_raw.publish_npz(self.output, self.arrays, 20, 2)
        original = self.output.read_bytes()
        code, result, calls = self.invoke()
        self.assertEqual((code, result["status"], calls), (1, "failed", 0))
        self.assertEqual(result["error_type"], "output_persistence_failure")
        self.assertFalse(self.report.exists())
        self.assertEqual(self.output.read_bytes(), original)
        self.assertEqual(list(self.root.rglob(".pose*")), [])


class ExistingEvidenceTests(unittest.TestCase):
    def test_stage22_reports_and_available_npzs(self):
        repo = Path(__file__).resolve().parents[1]
        for stem, total, valid, missing in (
                ("Subject.2/Fall backwards/FallBackwardsS2", 119, 119, []),
                ("Subject.1/Fall forward/FallForwardS1", 190, 94, [[0, 91], [133, 133], [136, 138]])):
            report = json.loads((repo / "artifacts/pose_extraction/single_video" / (stem + ".json")).read_text())
            self.assertEqual((report["decoded_frame_count"], report["pose_detected_count"]), (total, valid))
            self.assertEqual(report["missing_frame_intervals"], missing)
            output = repo / "data/interim/caucafall_v5/pose_raw_v1" / (stem + ".npz")
            if output.exists():  # Ignored local evidence is optional on a clean checkout.
                arrays = pose_raw.load_validated(output, 20, total)
                self.assertEqual(int(arrays["pose_detected"].sum()), valid)
                indices = [i for start, end in missing for i in range(start, end + 1)]
                self.assertEqual(np.flatnonzero(~arrays["pose_detected"]).tolist(), indices)
                self.assertTrue(np.isnan(arrays["landmarks"][indices]).all())


if __name__ == "__main__":
    unittest.main()
