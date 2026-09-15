"""Decision/accounting tests only; the actual experiment uses real MediaPipe."""
import unittest
from ml.datasets.validate_pose_compatibility import SAMPLES, compatibility_decision, missing_runs
from ml.datasets.inspect_caucafall import inspect_video


class CompatibilityTests(unittest.TestCase):
    def test_missing_intervals(self):
        self.assertEqual(missing_runs([0, 1, 4, 8, 9]), [[0, 1], [4, 4], [8, 9]])
        self.assertEqual(missing_runs([]), [])

    def test_fixed_subset_scope(self):
        self.assertEqual(len(SAMPLES), 10)
        self.assertEqual(len({a for _, a in SAMPLES}), 10)
        self.assertEqual(len({s for s, _ in SAMPLES}), 5)

    def test_decision_gate(self):
        row = dict(errors=[], valid_pose_frames=10, invalid_landmark_frames=0,
                   decoded_frames=10, metadata_frames=10, frames_without_pose=0)
        self.assertEqual(compatibility_decision([row]), "PASS")
        self.assertEqual(compatibility_decision([{**row, "frames_without_pose": 2,
                                                 "valid_pose_frames": 8}]), "PASS WITH WARNINGS")
        for changes in ({"valid_pose_frames": 0}, {"invalid_landmark_frames": 1},
                        {"errors": ["failure"]}, {"decoded_frames": 9}):
            self.assertEqual(compatibility_decision([{**row, **changes}]), "FAIL")

    def test_sequential_decode_counts(self):
        from pathlib import Path
        class Frame:
            size = 10
        class Capture:
            def __init__(self, count): self.remaining = count; self.released = False
            def open(self, path): return True
            def get(self, prop): return 3.0
            def read(self):
                if self.remaining:
                    self.remaining -= 1
                    return True, Frame()
                return False, None
            def release(self): self.released = True
        class CV:
            CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_COUNT = range(4)
            error = RuntimeError
            def VideoCapture(self): return capture
        for count in (2, 3, 4):
            capture = Capture(count)
            row = inspect_video(Path('data'), Path('data/Subject.1/Hop/video.avi'), CV())
            self.assertEqual(row['decoded_frame_count'], count)
            self.assertEqual(row['reached_end_normally'], count == 3)
            self.assertEqual(row['decode_failure_before_expected_end'], count < 3)
            self.assertEqual(row['frame_count_matches'], count == 3)
            self.assertEqual(row['readable'], count == 3)
            self.assertTrue(capture.released)


if __name__ == '__main__':
    unittest.main()
