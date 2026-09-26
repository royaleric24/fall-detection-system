"""Synthetic Stage 3.1 descriptive-statistics and Train-only access tests."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ml.datasets import pose_raw
from ml.preprocessing import characterize_pose as a
from ml.preprocessing import contract


def arrays(mask: list[bool]) -> dict[str, np.ndarray]:
    count = len(mask)
    landmarks = np.full((count, 33, 4), np.nan, dtype=np.float32)
    for index, detected in enumerate(mask):
        if detected:
            landmarks[index] = np.array([0.5, 0.5, -0.1, 0.8], dtype=np.float32)
            landmarks[index, 11, :2] = [0.3, 0.2]
            landmarks[index, 12, :2] = [0.7, 0.2]
            landmarks[index, 23, :2] = [0.4, 0.7]
            landmarks[index, 24, :2] = [0.6, 0.7]
    return dict(frame_index=np.arange(count, dtype=np.int32),
                timestamp_ms=np.arange(count, dtype=np.int64) * 50,
                pose_detected=np.array(mask, dtype=np.bool_), landmarks=landmarks)


def source(number: int = 8) -> contract.PoseSource:
    return contract.PoseSource("CAUCAFall", "V5", contract.RUN_ID,
                               f"Subject.{number}/Walk/WalkS{number}.avi",
                               a.REPO / f"data/interim/test/Subject.{number}/Walk/WalkS{number}.npz",
                               number, "Walk", "train", "non_fall")


class Stage31Tests(unittest.TestCase):
    def test_multiple_runs_duration_and_progress(self):
        data = arrays([False, False, True, False, True, False, False])
        runs = a.missing_runs(data["pose_detected"], data["timestamp_ms"])
        self.assertEqual([(r["start_frame"], r["end_frame"], r["frames"], r["duration_ms"])
                          for r in runs], [(0, 1, 2, 100), (3, 3, 1, 50), (5, 6, 2, 100)])
        self.assertEqual(runs[0]["start_progress"], 0)
        self.assertEqual(runs[-1]["end_progress"], 1)
        self.assertEqual(runs[1]["start_ms"], 150)
        self.assertEqual(runs[1]["end_exclusive_ms"], 200)

    def test_all_detected(self):
        data = arrays([True] * 4)
        self.assertEqual(a.missing_runs(data["pose_detected"], data["timestamp_ms"]), [])
        row, runs, values = a.video_statistics(source(), data)
        self.assertEqual(runs, [])
        self.assertEqual((row["detected_frames"], row["missing_run_count"], row["duration_ms"]), (4, 0, 200))
        self.assertIsNone(row["longest_missing_run_ms"])
        self.assertEqual(values["visibility"].shape, (4, 33))

    def test_all_missing(self):
        data = arrays([False] * 4)
        row, runs, values = a.video_statistics(source(), data)
        self.assertEqual((row["detected_frames"], row["missing_frames"], row["missing_run_count"]), (0, 4, 1))
        self.assertEqual((runs[0]["start_frame"], runs[0]["end_frame"], runs[0]["duration_ms"]), (0, 3, 200))
        self.assertTrue(row["zero_detected_poses"])
        self.assertEqual(row["missing_progress_quintile_frames"], [1, 1, 0, 1, 1])
        self.assertEqual(values["visibility"].shape, (0, 33))

    def test_beginning_end_and_intermittent_runs(self):
        for mask, expected in [([False, True, True], [(0, 0)]),
                               ([True, True, False], [(2, 2)]),
                               ([True, False, True, False, True], [(1, 1), (3, 3)])]:
            data = arrays(mask)
            runs = a.missing_runs(data["pose_detected"], data["timestamp_ms"])
            with self.subTest(mask=mask):
                self.assertEqual([(r["start_frame"], r["end_frame"]) for r in runs], expected)
                self.assertEqual(sum(r["duration_ms"] for r in runs), 50 * sum(not x for x in mask))

    def test_distribution_empty_and_linear_quantiles(self):
        self.assertIsNone(a.distribution([])["p50"])
        stats = a.distribution([0, 10, 20, 30])
        self.assertEqual(stats["count"], 4)
        for key, expected in (("p25", 7.5), ("p50", 15), ("p95", 28.5)):
            self.assertAlmostEqual(stats[key], expected)
        with self.assertRaisesRegex(ValueError, "finite"):
            a.distribution([np.nan])

    def test_raw_schema_and_missing_pose_invariants(self):
        data = arrays([True, False])
        pose_raw.validate_arrays(data, 20, 2)
        data["landmarks"][1, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "Missing poses"):
            pose_raw.validate_arrays(data, 20, 2)
        data = arrays([True, False]); data["landmarks"][0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "Detected poses"):
            pose_raw.validate_arrays(data, 20, 2)
        data = arrays([True, False]); data["timestamp_ms"] = data["timestamp_ms"].astype(np.int32)
        with self.assertRaisesRegex(ValueError, "timestamp_ms"):
            pose_raw.validate_arrays(data, 20, 2)

    def test_raw_geometry_is_diagnostic_only(self):
        data = arrays([True, False])
        before = data["landmarks"].copy()
        row, _, values = a.video_statistics(source(), data)
        self.assertEqual(row["detected_frames"], 1)
        self.assertAlmostEqual(values["geometry"]["shoulder_width_xy"][0], 0.4)
        self.assertAlmostEqual(values["geometry"]["hip_width_xy"][0], 0.2)
        self.assertAlmostEqual(values["geometry"]["shoulder_to_hip_midpoint_torso_xy"][0], 0.5)
        np.testing.assert_array_equal(data["landmarks"], before)

    def test_validation_and_test_rejected_before_selector_or_loader(self):
        with patch.object(a.contract, "select_sources") as selector, patch.object(a, "load_validated") as loader:
            for split in ("validation", "test", "all", "Train"):
                with self.subTest(split=split), self.assertRaisesRegex(contract.ContractError, "Train only"):
                    a.analyze(split)
            selector.assert_not_called()
            loader.assert_not_called()

    def test_train_selection_precedes_pose_loading(self):
        with patch.object(a.contract, "select_sources", return_value=()) as selector, \
                patch.object(a, "load_validated") as loader:
            with self.assertRaisesRegex(ValueError, "Train scope"):
                a.analyze()
            selector.assert_called_once_with(("train",), purpose="exploration")
            loader.assert_not_called()

    def test_deterministic_summary_and_official_provenance(self):
        subjects = contract.FROZEN_SUBJECTS["train"]
        selected = tuple(contract.PoseSource("CAUCAFall", "V5", contract.RUN_ID,
                           f"Subject.{subject_id}/activity/sample{slot}.avi",
                           a.REPO / f"data/interim/test/Subject.{subject_id}/activity/sample{slot}.npz",
                           subject_id, "activity", "train", "fall" if slot < 5 else "non_fall")
                         for subject_id in subjects for slot in range(10))
        meta = {s.source_video: (20.0, 2) for s in selected}
        data = arrays([True, False])
        with patch.object(a.contract, "select_sources", return_value=selected), \
                patch.object(a, "_train_manifest_metadata", return_value=meta), \
                patch.object(a, "load_validated", return_value=data), \
                patch.object(a, "sha256", return_value="synthetic-digest"):
            first = a.analyze()
            second = a.analyze()
        self.assertEqual(first, second)
        summary = json.loads(first["summary.json"])
        self.assertEqual(summary["provenance"]["stage2_run_id"], contract.RUN_ID)
        self.assertEqual(summary["provenance"]["stage30_baseline_commit"], a.BASELINE)
        self.assertEqual(summary["provenance"]["split"], "train")
        self.assertEqual(summary["inventory"]["videos"], 60)
        self.assertEqual(summary["inventory"]["frames"], 120)
        self.assertEqual(summary["missing_runs"]["count"], 60)
        self.assertTrue(all("Subject.10" not in content.decode() and "Subject.6" not in content.decode()
                            for content in first.values()))


if __name__ == "__main__":
    unittest.main()
