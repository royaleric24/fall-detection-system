"""Synthetic topology, duration, subject and sensitivity tests for Stage 3.1a."""

import unittest
from unittest.mock import patch

import numpy as np

from ml.preprocessing import contract
from ml.preprocessing import characterize_pose as original
from ml.preprocessing import missingness_addendum as a


def run_and_video(mask: list[bool]):
    observed = np.asarray(mask, dtype=np.bool_)
    times = np.arange(len(mask), dtype=np.int64) * 50
    runs = original.missing_runs(observed, times)
    video = dict(frames=len(mask), duration_ms=len(mask) * 50,
                 detected_frames=int(observed.sum()), missing_frames=int((~observed).sum()),
                 missing_fraction=float((~observed).mean()), missing_run_count=len(runs),
                 longest_missing_run_ms=max((r["duration_ms"] for r in runs), default=""),
                 split="train", video_label="fall", subject_id=8, source_video="Subject.8/Fall forward/a.avi")
    return runs, video


def class_video(subject: int, label: str, slot: int, missing_fraction: float,
                *, whole: bool = False) -> dict:
    return dict(subject_id=subject, source_video=f"Subject.{subject}/{label}/sample{slot}.avi",
                split="train", video_label=label, frames=100,
                detected_frames=0 if whole else round(100 * (1 - missing_fraction)),
                missing_fraction=missing_fraction, missing_run_count=1 if missing_fraction else 0,
                longest_missing_run_ms=5000 if whole else (500 if missing_fraction else ""))


class Stage31aTests(unittest.TestCase):
    def test_all_topologies_are_exclusive_and_exhaustive(self):
        cases = [([True, False, True], "internal"),
                 ([False, True, True], "leading"),
                 ([True, True, False], "trailing"),
                 ([False, False, False], "whole_video")]
        for mask, expected in cases:
            runs, video = run_and_video(mask)
            self.assertEqual(len(runs), 1)
            item = a.classify_run(runs[0], video, 50)
            with self.subTest(mask=mask):
                self.assertEqual(item["topology"], expected)
                self.assertIn(item["topology"], a.TOPOLOGIES)
        self.assertEqual(len(a.TOPOLOGIES), len(set(a.TOPOLOGIES)))

    def test_single_and_multi_frame_missing_support(self):
        for mask, count, support in [([True, False, True], 1, 50),
                                     ([True, False, False, False, True], 3, 150)]:
            runs, video = run_and_video(mask)
            item = a.classify_run(runs[0], video, 50)
            with self.subTest(mask=mask):
                self.assertEqual(item["missing_frame_count"], count)
                self.assertEqual(item["missing_support_ms"], support)
                self.assertEqual(item["first_missing_timestamp_ms"], 50)
                self.assertEqual(item["last_missing_timestamp_ms"], 50 * count)

    def test_internal_anchor_gap_distinct_from_missing_support(self):
        runs, video = run_and_video([True, False, False, True])
        item = a.classify_run(runs[0], video, 50)
        self.assertEqual((item["left_observed_timestamp_ms"], item["right_observed_timestamp_ms"]), (0, 150))
        self.assertEqual(item["anchor_gap_ms"], 150)
        self.assertEqual(item["missing_support_ms"], 100)

    def test_noninternal_anchor_fields_are_null(self):
        for mask in ([False, True], [True, False], [False, False]):
            runs, video = run_and_video(mask)
            item = a.classify_run(runs[0], video, 50)
            with self.subTest(mask=mask):
                self.assertIsNone(item["left_observed_timestamp_ms"])
                self.assertIsNone(item["right_observed_timestamp_ms"])
                self.assertIsNone(item["anchor_gap_ms"])

    def test_invalid_run_indices_or_timing_rejected(self):
        runs, video = run_and_video([True, False, True])
        bad = dict(runs[0], frames=2)
        with self.assertRaisesRegex(ValueError, "indices"):
            a.classify_run(bad, video, 50)
        bad = dict(runs[0], duration_ms=100)
        with self.assertRaisesRegex(ValueError, "timing"):
            a.classify_run(bad, video, 50)

    def test_topology_summary_counts_and_distributions(self):
        masks = ([True, False, True], [False, True], [True, False, False], [False, False, False])
        items = [a.classify_run(run, video, 50) for mask in masks
                 for runs, video in [run_and_video(mask)] for run in runs]
        rows = a.topology_summary(items)
        self.assertEqual([r["topology"] for r in rows], list(a.TOPOLOGIES))
        self.assertEqual([r["runs"] for r in rows], [1, 1, 1, 1])
        self.assertEqual([r["missing_frames"] for r in rows], [1, 1, 2, 3])
        self.assertAlmostEqual(sum(r["fraction_of_all_missing_frames"] for r in rows), 1)
        self.assertEqual(rows[0]["anchor_gap_ms"]["p50"], 100)
        self.assertTrue(all(r["anchor_gap_ms"] is None for r in rows[1:]))

    def test_subject_class_summaries_and_differences_are_train_only(self):
        videos = [class_video(subject, label, slot, 0.2 if label == "fall" else 0.1)
                  for subject in (8, 4) for label in ("fall", "non_fall") for slot in range(5)]
        rows, diffs = a.subject_class_tables(videos)
        self.assertEqual((len(rows), len(diffs)), (4, 2))
        self.assertEqual({r["videos"] for r in rows}, {5})
        self.assertTrue(all(abs(r["fall_minus_non_fall_mean_video_missing_fraction"] - 0.1) < 1e-12
                            for r in diffs))
        for split in ("validation", "test"):
            bad = [dict(videos[0], split=split)] + videos[1:]
            with self.subTest(split=split), self.assertRaisesRegex(contract.ContractError, "Train only"):
                a.subject_class_tables(bad)

    def test_sensitivity_omits_only_whole_video_cases(self):
        all_video = class_video(8, "fall", 0, 1.0, whole=True)
        other_fall = class_video(8, "fall", 1, 0.2)
        non_fall = class_video(8, "non_fall", 0, 0.1)
        videos = [all_video, other_fall, non_fall]
        rows, omitted = a.sensitivity(videos, {all_video["source_video"]})
        self.assertEqual(omitted, [all_video["source_video"]])
        by_key = {(r["view"], r["video_label"]): r for r in rows}
        self.assertEqual(by_key["all_train", "fall"]["videos"], 2)
        self.assertEqual(by_key["excluding_whole_video_missing", "fall"]["videos"], 1)
        self.assertEqual(by_key["excluding_whole_video_missing", "non_fall"]["videos"], 1)
        self.assertAlmostEqual(by_key["excluding_whole_video_missing", "fall"]["mean_video_missing_fraction"], 0.2)
        self.assertEqual(len(videos), 3)  # Diagnostic view leaves the source list intact.
        with self.assertRaisesRegex(ValueError, "disagree"):
            a.sensitivity(videos, set())

    def test_nontrain_rejected_before_source_or_artifact_read(self):
        with patch.object(a.contract, "select_sources") as selector, \
                patch.object(a, "_read_csv") as reader:
            for split in ("validation", "test"):
                with self.subTest(split=split), self.assertRaisesRegex(contract.ContractError, "Train only"):
                    a.build(split)
            selector.assert_not_called()
            reader.assert_not_called()

    def test_train_selection_precedes_artifact_read(self):
        with patch.object(a.contract, "select_sources", side_effect=ValueError("guard called")) as selector, \
                patch.object(a, "_read_csv") as reader:
            with self.assertRaisesRegex(ValueError, "guard called"):
                a.build()
            selector.assert_called_once_with(("train",), purpose="exploration")
            reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
