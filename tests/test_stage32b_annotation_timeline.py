"""Synthetic-only tests: no real source image, annotation, video or pose access."""

import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from ml.preprocessing import audit_annotation_timeline as a


def pair(names, annotations=None, aliases=None):
    annotations = annotations if annotations is not None else [Path(n).stem + ".txt" for n in names]
    return a.pair_sequence(names, annotations, aliases or {})


class Stage32bTests(unittest.TestCase):
    def test_numeric_order_independent_of_input_order(self):
        rows, status = pair(["f10.png", "f2.png", "f1.png"])
        self.assertEqual([r["png_filename"] for r in rows], ["f1.png", "f2.png", "f10.png"])
        self.assertEqual([r["png_ordinal_index"] for r in rows], [0, 1, 2])
        self.assertEqual(status, "verified")

    def test_filename_index_is_not_ordinal_or_avi_index(self):
        rows, _ = pair(["cfs800001.png", "cfs800003.png"])
        self.assertEqual([r["png_filename_index"] for r in rows], [800001, 800003])
        self.assertEqual([r["png_ordinal_index"] for r in rows], [0, 1])
        self.assertTrue(all(r["avi_frame_index"] is None for r in rows))

    def test_gap_is_reported_without_inventing_a_sample(self):
        names = ["f001.png", "f003.png"]
        self.assertEqual(a.prior.numbering(names)["gaps"], [[2, 2]])
        rows, status = pair(names)
        self.assertEqual((len(rows), status), (2, "verified"))

    def test_duplicate_numeric_index_blocks_verified_order(self):
        names = ["f001.png", "f001 - copia.png"]
        self.assertEqual(a.prior.numbering(names)["duplicate_indices"], [1])
        rows, status = pair(names)
        self.assertEqual(status, "unresolved")
        self.assertTrue(all(r["annotation_filename"] is None for r in rows))

    def test_all_four_explicit_suffix_aliases(self):
        for aliases in a.ALIASES.values():
            png, txt = next(iter(aliases.items()))
            rows, status = pair([png], [txt], aliases)
            self.assertEqual(status, "verified")
            self.assertEqual(rows[0]["png_filename"], png)
            self.assertEqual(rows[0]["annotation_filename"], txt)
            self.assertEqual(rows[0]["annotation_pairing_status"], "verified_explicit_sequence_alias")

    def test_unlisted_suffix_is_not_normalized(self):
        rows, status = pair(["f001.png"], ["f001a.txt"])
        self.assertEqual(status, "unresolved")
        self.assertIsNone(rows[0]["annotation_filename"])

    def test_alias_rejects_competing_identity(self):
        rows, status = pair(["f001.png"], ["f001a.txt", "f001b.txt"], {"f001.png": "f001a.txt"})
        self.assertEqual(status, "unresolved")
        self.assertIsNone(rows[0]["annotation_filename"])

    def test_alias_preserves_prefix_and_digit_spelling(self):
        for txt in ("f01a.txt", "g001a.txt"):
            rows, status = pair(["f001.png"], [txt], {"f001.png": txt})
            self.assertEqual(status, "unresolved")
            self.assertIsNone(rows[0]["annotation_filename"])

    def test_unparsed_and_mixed_prefixes_are_unresolved(self):
        for names in (["cover.png"], ["a001.png", "b002.png"]):
            self.assertEqual(pair(names)[1], "unresolved")

    def test_exact_pixel_duplicates(self):
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        hashes = [a.prior.pixel_digest(x) for x in (image, image.copy())]
        rows = a.duplicate_audit(hashes, ["f1.png", "f2.png"], "decoded_pixel_sha256")
        self.assertEqual(rows[0]["ordinals"], [0, 1])
        self.assertEqual(rows[0]["excess_identical_copies"], 1)

    def test_boundary_duplicate_pairs(self):
        rows = a.duplicate_audit(["a", "a", "b", "c", "c"], list("12345"), "file_sha256")
        self.assertTrue(rows[0]["leading_boundary_duplicate"])
        self.assertFalse(rows[0]["trailing_boundary_duplicate"])
        self.assertTrue(rows[1]["trailing_boundary_duplicate"])
        self.assertEqual(rows[1]["internal_adjacent_pairs"], [])

    def test_internal_and_nonadjacent_duplicates(self):
        rows = a.duplicate_audit(["a", "b", "b", "c", "b", "d"], list("123456"), "decoded_pixel_sha256")
        self.assertEqual(rows[0]["ordinals"], [1, 2, 4])
        self.assertEqual(rows[0]["internal_adjacent_pairs"], [[1, 2]])
        self.assertFalse(rows[0]["leading_boundary_duplicate"])
        self.assertFalse(rows[0]["trailing_boundary_duplicate"])

    def test_one_pixel_difference_is_not_a_duplicate(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        near = image.copy(); near[0, 0, 0] = 1
        hashes = [a.prior.pixel_digest(x) for x in (image, near)]
        self.assertEqual(a.duplicate_audit(hashes, ["f1.png", "f2.png"], "decoded_pixel_sha256"), [])

    def test_one_and_zero_based_names_keep_zero_based_ordinals(self):
        for names in (["f0.png", "f1.png"], ["f1.png", "f2.png"]):
            rows, status = pair(names)
            self.assertEqual(status, "verified")
            self.assertEqual([r["png_ordinal_index"] for r in rows], [0, 1])

    def test_verified_order_does_not_verify_timebase(self):
        rows, status = pair(["f1.png", "f2.png"])
        self.assertEqual(status, "verified")
        self.assertTrue(all(r["annotation_timebase_status"] == "unresolved" for r in rows))
        self.assertIsNone(a.timing_diagnostics(101, 100, 20)["verified_png_fps"])

    def test_no_fabricated_timestamp(self):
        rows, _ = pair(["f1.png", "f2.png", "f3.png"])
        self.assertEqual([r["png_timestamp_ms"] for r in rows], [None, None, None])

    def test_duration_and_sample_interval_conventions(self):
        r = a.timing_diagnostics(101, 100, 20)
        self.assertEqual(r["avi_count_duration_seconds"], 5)
        self.assertEqual(r["avi_first_last_sample_span_seconds"], 4.95)
        self.assertEqual(r["png_count_over_avi_count_duration"], 20.2)
        self.assertEqual(r["png_intervals_over_avi_count_duration"], 20)
        self.assertAlmostEqual(r["png_count_over_avi_first_last_span"], 101/4.95)
        self.assertAlmostEqual(r["png_intervals_over_avi_first_last_span"], 100/4.95)

    def test_grounded_fall_transitions_are_ordinal_only(self):
        r = a.ordinal_labels([{"annotation_class_id": v} for v in [0, 0, 1, 1, 0]])
        self.assertEqual((r["first_grounded_fall_ordinal"], r["last_grounded_fall_ordinal"]), (2, 3))
        self.assertEqual(r["transitions_0_to_1"], [2])
        self.assertEqual(r["transitions_1_to_0"], [4])
        self.assertTrue(r["class1_one_interval"])
        self.assertFalse(any("ms" in key or "onset" in key for key in r))
        self.assertEqual(a.prior.SEMANTICS[1], "grounded_fall_state")

    def assert_subject_rejected(self, subject, split):
        with patch.object(a.contract, "select_sources") as selector:
            with self.assertRaises(a.contract.ContractError):
                a.analyze(split)
            selector.assert_not_called()
        with patch.object(zipfile, "ZipFile") as opener:
            with self.assertRaises(a.contract.ContractError):
                a.workbook_train_counts(Path("absent.xlsx"), (subject,))
            opener.assert_not_called()
        with self.assertRaises(a.contract.ContractError):
            a.require_train_subject(subject, "train")

    def test_validation_rejected_before_io(self):
        for subject in (10, 5):
            self.assert_subject_rejected(subject, "validation")

    def test_test_rejected_before_io(self):
        for subject in (6, 7):
            self.assert_subject_rejected(subject, "test")

    def test_scope_selection_precedes_image_io(self):
        with patch.object(a.contract, "select_sources", return_value=()) as selector, patch.object(a, "audit_sequence") as audit:
            with self.assertRaisesRegex(ValueError, "Train scope"):
                a.analyze()
            selector.assert_called_once_with(("train",), purpose="exploration")
            audit.assert_not_called()

    def test_png_metadata_is_not_a_capture_clock(self):
        _, encoded = cv2.imencode(".png", np.zeros((2, 2, 3), dtype=np.uint8))
        r = a.png_metadata(encoded.tobytes())
        self.assertEqual(r["ancillary_records"], [])
        self.assertIn("IHDR", r["chunk_types"])
        data = bytes([7, 234, 1, 1, 0, 0, 0])
        chunk = struct.pack(">I", len(data)) + b"tIME" + data + b"\0"*4
        content = encoded.tobytes()
        r = a.png_metadata(content[:-12] + chunk + content[-12:])
        self.assertIn("not a verified frame capture timestamp", r["ancillary_records"][0]["interpretation"])

    def test_file_and_pixel_identity_are_distinct(self):
        image = np.zeros((20, 20, 3), np.uint8)
        encoded = [cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, n])[1].tobytes() for n in (0, 9)]
        self.assertNotEqual(a.prior.digest(encoded[0]), a.prior.digest(encoded[1]))
        decoded = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED) for b in encoded]
        self.assertEqual(a.prior.pixel_digest(decoded[0]), a.prior.pixel_digest(decoded[1]))

    def test_workbook_resolves_only_selected_train_cells(self):
        ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        strings = ["Subject 1:", "Activity", "Frames", "Hop"]
        cells = ['<c r="C8" t="s"><v>0</v></c>', '<c r="C10" t="s"><v>1</v></c>',
                 '<c r="D10" t="s"><v>2</v></c>']
        for row in range(11, 21):
            cells.extend([f'<c r="C{row}" t="s"><v>3</v></c>', f'<c r="D{row}"><v>12</v></c>'])
        # Subject 5 value would fail if dereferenced; it must remain unread.
        cells.append('<c r="D41" t="s"><v>99999</v></c>')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"synthetic.xlsx"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("xl/sharedStrings.xml", f'<sst xmlns="{ns}">' + ''.join(f'<si><t>{s}</t></si>' for s in strings) + '</sst>')
                z.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{ns}"><sheetData><row>' + ''.join(cells) + '</row></sheetData></worksheet>')
            rows = a.workbook_train_counts(path, (1,))
        self.assertEqual(len(rows), 10)
        self.assertEqual({r["subject_id"] for r in rows}, {1})
        self.assertEqual(sum(r["documented_frames"] for r in rows), 120)
        self.assertEqual(rows[0]["frames_cell"], "D11")

    def test_sequence_integration_retains_duplicate_and_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            rel = Path("Subject.2/Fall backwards/FallBackwardsS2.avi")
            directory = root/rel.parent; directory.mkdir(parents=True)
            aliases = {"cas200091 - copia.png": "cas200091.txt"}
            names = ["cas200090.png", "cas200091 - copia.png", "cas200092.png"]
            for name, value in zip(names, [10, 20, 20]):
                cv2.imwrite(str(directory/name), np.full((3, 3, 4), value, np.uint8))
                target = aliases.get(name, Path(name).stem + ".txt")
                (directory/target).write_text("1 0.5 0.5 0.2 0.2")
            (directory/"classes.txt").write_text("nofall\nfall\n")
            ledger = {p.relative_to(root).as_posix(): a.prior.file_digest(p) for p in directory.iterdir()}
            source = a.contract.PoseSource("CAUCAFall", "V5", a.contract.RUN_ID, rel.as_posix(),
                root/"never_load.npz", 2, "Fall backwards", "train", "fall")
            with patch.object(a, "RAW", root), patch.object(cv2, "VideoCapture", side_effect=AssertionError("AVI forbidden")), patch.object(np, "load", side_effect=AssertionError("pose forbidden")):
                r = a.audit_sequence(source, ledger, {"decoded_avi_frame_count": "2", "fps": "20"},
                    {"subject_id": 2, "activity": "Fall backwards", "documented_frames": 3, "frames_cell": "J16"})
            self.assertEqual(r["inventory"]["annotation_order_status"], "verified")
            self.assertEqual(r["inventory"]["explicit_sequence_alias_pairs"], 1)
            self.assertEqual(r["inventory"]["decoded_pixel_duplicate_excess"], 1)
            self.assertEqual(len(r["records"]), 3)
            self.assertTrue(r["inventory"]["trailing_boundary_duplicate"])
            self.assertTrue(all(x["png_timestamp_ms"] is None for x in r["records"]))


if __name__ == "__main__":
    unittest.main()
