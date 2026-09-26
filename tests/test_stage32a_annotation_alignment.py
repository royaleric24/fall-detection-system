"""Synthetic images/annotations only; never open actual source media in unit tests."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ml.preprocessing import audit_annotation_alignment as a
from ml.preprocessing import contract
from ml.preprocessing import review_alignment_evidence as review


def image(value):
    return np.full((4, 5, 3), value, dtype=np.uint8)


def annotations(labels):
    return [dict(**a.filename_identity(f"f{i:04}.txt"),
                 **a.parse_annotation(f"{label} 0.5 0.5 0.4 0.6"))
            for i, label in enumerate(labels)]


class Stage32aTests(unittest.TestCase):
    def test_zero_based_numbering(self):
        r = a.numbering(["f0002.png", "f0000.png", "f0001.png"])
        self.assertEqual((r["first_index"], r["last_index"], r["index_minus_ordinal"]), (0, 2, [0]))
        self.assertTrue(r["contiguous"])

    def test_one_based_numbering(self):
        r = a.numbering(["0001.png", "0002.png"])
        self.assertEqual(r["index_minus_ordinal"], [1])

    def test_subject_digits_not_silently_removed(self):
        r = a.numbering(["cfs800001.png", "cfs800002.png"])
        self.assertEqual(r["index_minus_ordinal"], [800001])
        self.assertEqual(a.filename_identity("cfs800002a.txt")["suffix"], "a")

    def test_ordinal_is_independent_of_filename_index(self):
        hashes = [a.pixel_digest(image(i)) for i in range(3)]
        for names in (["0.png", "1.png", "2.png"], ["s800001.png", "s800002.png", "s800003.png"]):
            self.assertEqual(len(a.ordered_names(names)), 3)
            self.assertEqual({i: j for i, (j, _) in a.exact_correspondence(hashes, hashes).items()}, {0: 0, 1: 1, 2: 2})

    def test_equal_sequences(self):
        result = self.audit([10, 20, 30], [10, 20, 30])
        self.assertEqual(result["inventory"]["exact_annotation_mapped_frames"], 3)
        self.assertEqual([r["stage2_timestamp_ms"] for r in result["correspondence"]], [0, 50, 100])
        self.assertTrue(all(r["correspondence_status"] == "exact" for r in result["correspondence"]))

    def test_png_without_txt(self):
        result = self.audit([10, 20], [10, 20], missing_txt={1})
        self.assertEqual(result["inventory"]["png_without_txt_count"], 1)
        row = result["correspondence"][1]
        self.assertEqual(row["correspondence_status"], "missing_annotation")
        self.assertIsNone(row["annotation_class_id"])

    def test_txt_without_png_and_suffix_not_auto_repaired(self):
        result = self.audit([10, 20], [10, 20], missing_txt={1}, extra_txt=["f0001a.txt"])
        self.assertEqual(result["inventory"]["txt_without_png_count"], 1)
        self.assertEqual(result["correspondence"][-1]["correspondence_status"], "missing_png")
        self.assertIsNone(result["correspondence"][1]["annotation_class_id"])
        self.assertEqual(result["issues"][0]["candidate_same_numeric_index"], ["f0001a.txt"])

    def test_leading_extra_png(self):
        mapping = a.exact_correspondence(["a", "b", "c"], ["x", "a", "b", "c"])
        self.assertEqual([mapping[i][0] for i in range(3)], [1, 2, 3])

    def test_trailing_extra_png_is_retained_and_compared(self):
        result = self.audit([10, 20], [10, 20, 30])
        self.assertEqual(len(result["correspondence"]), 3)
        self.assertEqual(result["correspondence"][-1]["row_kind"], "unmapped_png")
        self.assertTrue(any(r["png_ordinal_index"] == 2 for r in result["local"]))

    def test_internal_missing_png(self):
        mapping = a.exact_correspondence(["a", "b", "c", "d"], ["a", "c", "d"])
        self.assertNotIn(1, mapping)
        self.assertEqual(mapping[2][0], 1)
        self.assertEqual(mapping[3][0], 2)

    def test_duplicate_image_and_index_detection(self):
        self.assertEqual(a.duplicate_groups(["a", "b", "a"]), [[0, 2]])
        self.assertEqual(a.numbering(["f001.png", "f001 - copia.png"])["duplicate_indices"], [1])
        self.assertEqual(a.numbering(["f001.png", "f003.png"])["gaps"], [[2, 2]])
        mapping = a.exact_correspondence(["a", "b"], ["a", "a", "b"])
        self.assertNotIn(0, mapping)
        self.assertIn(1, mapping)

    def test_unequal_counts_never_truncated(self):
        result = self.audit([10, 20, 30], [10, 20])
        self.assertEqual(len(result["direct"]), 3)
        avi = [r for r in result["correspondence"] if r["row_kind"] == "avi_frame"]
        self.assertEqual([r["stage2_frame_index"] for r in avi], [0, 1, 2])
        self.assertEqual(avi[-1]["correspondence_status"], "unresolved")

    def test_invalid_class_id(self):
        for value in ("2", "-1", "0.0", "nan"):
            self.assertFalse(a.parse_annotation(f"{value} 0.5 0.5 0.2 0.2")["valid"])

    def test_malformed_annotation_rows(self):
        for text in ("", "0 1 2", "0 0.5 0.5 0.3 inf", "0 0.5 0.5 0 0.1", "0 a b c d", "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2"):
            self.assertFalse(a.parse_annotation(text)["valid"], text)

    def test_class_transitions_and_semantics(self):
        r = a.sequence_summary(annotations([0, 0, 1, 1, 0, 1]), "fall")
        self.assertEqual((r["transitions_0_to_1"], r["transitions_1_to_0"]), (2, 1))
        self.assertFalse(r["class1_one_contiguous_interval"])
        self.assertEqual((r["first_class1_annotation_index"], r["last_class1_annotation_index"]), (2, 5))
        self.assertEqual(a.SEMANTICS[1], "grounded_fall_state")
        self.assertTrue(a.sequence_summary(annotations([0, 1]), "non_fall")["adl_contains_class1"])

    def test_invalid_or_ambiguous_sequence_suppresses_transitions(self):
        rows = annotations([0, 1]); rows[1]["valid"] = False
        self.assertIsNone(a.sequence_summary(rows, "fall")["transitions_0_to_1"])

    def test_validation_rejected_before_source_io(self):
        self.assert_forbidden("validation")

    def test_test_rejected_before_source_io(self):
        self.assert_forbidden("test")

    def test_train_selection_precedes_media_access(self):
        with patch.object(a.contract, "select_sources", return_value=()) as selector, patch.object(a, "audit_video") as audit:
            with self.assertRaisesRegex(ValueError, "Train scope"):
                a.analyze()
            selector.assert_called_once_with(("train",), purpose="exploration")
            audit.assert_not_called()

    def test_unresolved_nonexact_never_fabricates_a_mapping(self):
        result = self.audit([10, 20], [11, 21])
        self.assertTrue(result["local"])
        self.assertEqual(result["inventory"]["exact_annotation_mapped_frames"], 0)
        for row in result["correspondence"]:
            if row["row_kind"] == "avi_frame":
                self.assertEqual(row["correspondence_status"], "unresolved")
                self.assertIsNone(row["png_filename"])
                self.assertIsNone(row["annotation_class_id"])

    def test_crossed_exact_matches_remain_unresolved(self):
        self.assertEqual(a.exact_correspondence(["a", "b", "c"], ["a", "c", "b"]), {})

    def test_metrics_exact_nonexact_and_shape(self):
        exact = a.image_metrics(image(0), image(0))
        self.assertTrue(exact["exact"]); self.assertIsNone(exact["psnr_db"])
        d = a.image_metrics(image(10), image(12))
        self.assertEqual((d["mae"], d["max_abs_error"], d["mse"]), (2, 2, 4))
        self.assertAlmostEqual(d["psnr_db"], 10*np.log10(255**2/4))
        self.assertFalse(a.image_metrics(image(1), np.zeros((3, 3, 3), dtype=np.uint8))["same_dimensions"])

    def test_changed_decode_count_disables_stage2_mapping(self):
        result = self.audit([10, 20], [10, 20], expected_count=3)
        self.assertEqual(result["inventory"]["exact_image_mapped_frames"], 0)

    def test_candidate_ranking_preserves_ties_without_acceptance(self):
        rows = [dict(source_video="synthetic", stage2_frame_index="0", png_ordinal_index=str(j),
                     mae=str(value), mse=str(value)) for j, value in enumerate([1, 1, 2])]
        result = review.candidate_rankings(rows)[0]
        self.assertEqual(result["best_mae_png_ordinals"], [0, 1])
        self.assertEqual(result["candidate_png_ordinals"], [0, 1, 2])
        self.assertEqual(result["correspondence_status"], "unresolved")

    def test_offset_summary_reports_uncompared_elements(self):
        rows = [dict(source_video="synthetic", offset="0", mae="2", mse="4")]
        inventory = [dict(source_video="synthetic", decoded_avi_frame_count="1", png_count="2")]
        result = next(r for r in review.offset_summaries(rows, inventory) if r["png_ordinal_minus_avi_index"] == 0)
        self.assertEqual((result["comparable_pairs"], result["avi_frames_not_compared"], result["png_frames_not_compared"]), (1, 0, 1))

    def assert_forbidden(self, split):
        with patch.object(a.contract, "select_sources") as selector, patch.object(a, "audit_video") as audit:
            with self.assertRaisesRegex(contract.ContractError, "Train only"):
                a.analyze(split)
            selector.assert_not_called(); audit.assert_not_called()

    def audit(self, avi_values, png_values, missing_txt=None, extra_txt=None, expected_count=None):
        missing_txt = missing_txt or set()
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp).resolve(); directory = raw / "Subject.8/Walk"
            directory.mkdir(parents=True)
            video = directory / "WalkS8.avi"; video.write_bytes(b"synthetic-video-placeholder")
            (directory / "classes.txt").write_text("nofall\nfall\n")
            for i, value in enumerate(png_values):
                a.cv2.imwrite(str(directory / f"f{i:04}.png"), image(value))
                if i not in missing_txt:
                    (directory / f"f{i:04}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            for name in extra_txt or []:
                (directory / name).write_text("0 0.5 0.5 0.2 0.2\n")
            source = contract.PoseSource("CAUCAFall", "V5", contract.RUN_ID, "Subject.8/Walk/WalkS8.avi",
                                         raw / "unused.npz", 8, "Walk", "train", "non_fall")
            count = expected_count if expected_count is not None else len(avi_values)
            meta = dict(frame_count=str(count))
            decoded = dict(fps=20, width=5, height=4, frame_count=count, backend="synthetic",
                           matches_stage2_count=count == len(avi_values))
            with patch.object(a, "RAW", raw), patch.object(a, "REPO", raw), \
                    patch.object(a, "decode_avi", return_value=([image(v) for v in avi_values], decoded)):
                return a.audit_video(source, meta, a.file_digest(video))


if __name__ == "__main__":
    unittest.main()
