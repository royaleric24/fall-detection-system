"""Small synthetic fixtures test logic, never supply reported dataset metrics."""

import tempfile
import unittest
from pathlib import Path

from ml.datasets.inspect_caucafall import (
    ACTIVITY_LABELS, activity_label, classify_txt, inspect_annotations,
    inspect_video, subject_key, validate_output, validate_structure,
)


class InspectionTests(unittest.TestCase):
    def test_binary_labels_preserve_source_names(self):
        self.assertEqual(len(ACTIVITY_LABELS), 10)
        for name, label in ACTIVITY_LABELS.items():
            self.assertEqual(activity_label(name), "fall" if name.startswith("Fall ") else "non_fall")
        with self.assertRaises(KeyError):
            activity_label("Unknown")

    def test_numeric_subject_order(self):
        self.assertEqual(sorted(["Subject.10", "Subject.2", "Subject.1"], key=subject_key),
                         ["Subject.1", "Subject.2", "Subject.10"])

    def test_annotation_classification(self):
        self.assertEqual(classify_txt(Path("classes.txt"), "nofall\nfall\n"), "classes_metadata")
        self.assertEqual(classify_txt(Path("frame.txt"), "1 0.5 0.5 0.2 0.3\n"), "frame_annotation")
        for content in ("", "notes", "1 nan 0.5 0.2 0.3", "1 2 0.5 0.2 0.3"):
            self.assertEqual(classify_txt(Path("frame.txt"), content), "other_txt")

    def test_metadata_and_orphans_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in {"classes.txt": "nofall\nfall\n", "paired.png": "",
                                  "paired.txt": "0 .5 .5 .2 .3", "orphan.png": "",
                                  "orphan_annotation.txt": "1 .5 .5 .2 .3",
                                  "notes.txt": "notes", "bad.png": "", "bad.txt": "notes"}.items():
                (root / name).write_text(content)
            result = inspect_annotations(root, sorted(root.iterdir()))
            self.assertEqual(result["png_without_txt"], ["orphan.png"])
            self.assertEqual(result["orphan_frame_annotations"], ["orphan_annotation.txt"])
            self.assertEqual(result["non_classes_txt_without_png"], ["notes.txt", "orphan_annotation.txt"])
            self.assertEqual(result["png_with_other_txt"], ["bad.png"])
            self.assertEqual(result["matched_frame_annotations"], 1)

    def test_structure_missing_duplicate_and_unexpected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = []
            for i in range(1, 11):
                for activity in ACTIVITY_LABELS:
                    path = root / f"Subject.{i}" / activity / "video.avi"
                    path.parent.mkdir(parents=True)
                    path.touch()
                    videos.append(path)
            self.assertEqual(validate_structure(root, videos), [])
            missing = videos.pop()
            missing.unlink()
            duplicate = videos[0].with_name("duplicate.avi")
            duplicate.touch()
            videos.append(duplicate)
            unexpected = root / "stray.avi"
            unexpected.touch()
            videos.append(unexpected)
            issues = validate_structure(root, videos)
            self.assertTrue(any("found 0" in s for s in issues))
            self.assertTrue(any("found 2" in s for s in issues))
            self.assertTrue(any("Unexpected AVI" in s for s in issues))
            missing.parent.rmdir()
            self.assertTrue(any("Missing activity" in s for s in validate_structure(root, videos)))

    def test_output_cannot_write_raw_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                validate_output(root, root / "report")
            with self.assertRaises(ValueError):
                validate_output(root, Path("data/raw/reports"))

    def test_open_metadata_without_decode_is_not_readable(self):
        class Capture:
            released = False
            def open(self, path): return True
            def get(self, prop): return 30.0
            def read(self): return False, None
            def release(self): self.released = True
        capture = Capture()
        class FakeCV:
            CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_COUNT = range(4)
            error = RuntimeError
            def VideoCapture(self): return capture
        result = inspect_video(Path("dataset"), Path("dataset/Subject.1/Hop/video.avi"), FakeCV())
        self.assertTrue(result["opened"])
        self.assertFalse(result["readable"])
        self.assertFalse(result["decoded_first_frame"])
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
