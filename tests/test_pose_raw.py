"""Stage 2.2 synthetic schema/I/O/failure tests; no real MediaPipe or raw video reads."""

import tempfile
import unittest
import io
import zipfile
import warnings
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from ml.datasets import extract_pose_video as extractor
from ml.datasets import pose_raw
from ml.datasets.pose_runtime import pose_options


def pose(count=33):
    return NS(pose_landmarks=[[NS(x=1.2, y=-.1, z=-.3, visibility=.01)
                               for _ in range(count)]])


def example_arrays():
    valid, values = pose_raw.encode_pose(pose())
    return dict(frame_index=np.arange(2, dtype=np.int32),
                timestamp_ms=np.array([0, 50], dtype=np.int64),
                pose_detected=np.array([valid, False], dtype=np.bool_),
                landmarks=np.stack([values, np.full((33, 4), np.nan, dtype=np.float32)]))


class SchemaTests(unittest.TestCase):
    def test_timestamps(self):
        self.assertEqual([pose_raw.timestamp_ms(i, 20) for i in range(5)], [0, 50, 100, 150, 200])
        for fps in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                pose_raw.timestamp_ms(0, fps)

    def test_timestamp_rounding_and_long_sequence(self):
        self.assertEqual([pose_raw.timestamp_ms(i, 400) for i in range(5)], [0, 2, 5, 8, 10])
        times = np.array([pose_raw.timestamp_ms(i, 20) for i in range(1000)])
        np.testing.assert_array_equal(np.diff(times), np.full(999, 50))
        self.assertEqual(times[-1], 49950)

    def test_reject_invalid_frame_index_and_count(self):
        for index in (-1, .5, True, float("nan")):
            with self.subTest(index=index), self.assertRaises(ValueError):
                pose_raw.timestamp_ms(index, 20)
        for count in (0, -1, 2.0, True, float("nan")):
            with self.subTest(count=count), self.assertRaises(ValueError):
                pose_raw.validate_arrays(example_arrays(), 20, count)

    def test_all_missing_all_detected_and_infinities(self):
        for present in (False, True):
            arrays = example_arrays()
            arrays["pose_detected"][:] = present
            arrays["landmarks"][:] = .01 if present else np.nan
            pose_raw.validate_arrays(arrays, 20, 2)
            arrays["landmarks"][0, 0, 0] = np.inf
            with self.assertRaises(ValueError):
                pose_raw.validate_arrays(arrays, 20, 2)

    def test_malformed_containers_and_nonscalar_values(self):
        cases = [NS(pose_landmarks=[None]), NS(pose_landmarks=[[]]),
                 NS(pose_landmarks=""), NS(pose_landmarks={})]
        result = pose()
        for landmark in result.pose_landmarks[0]:
            for field in ("x", "y", "z", "visibility"):
                setattr(landmark, field, [0.1])
        cases.append(result)
        for result in cases:
            with self.subTest(result=result), self.assertRaises(pose_raw.ExtractionError):
                pose_raw.encode_pose(result)

    def test_corrupt_truncated_and_non_npz_inputs(self):
        stream = io.BytesIO(); np.savez(stream, **example_arrays())
        npy = io.BytesIO(); np.save(npy, np.arange(3))
        for payload in (b"", b"not an archive", stream.getvalue()[:100], npy.getvalue()):
            with self.subTest(length=len(payload)), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.npz"; path.write_bytes(payload)
                with self.assertRaises(ValueError):
                    pose_raw.load_validated(path, 20, 2)

    def test_duplicate_archive_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(path, **example_arrays())
            with zipfile.ZipFile(path, "a") as archive, warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr("frame_index.npy", b"duplicate")
            with self.assertRaises(ValueError):
                pose_raw.load_validated(path, 20, 2)

    def test_valid_pose_preserves_order_coordinates_and_low_visibility(self):
        result = pose()
        result.pose_landmarks[0][7].x = .77
        detected, values = pose_raw.encode_pose(result)
        self.assertTrue(detected)
        self.assertEqual(values.shape, (33, 4))
        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(values[0], np.array([1.2, -.1, -.3, .01], dtype=np.float32))
        self.assertEqual(values[7, 0], np.float32(.77))

    def test_missing_is_only_empty_pose_list(self):
        detected, values = pose_raw.encode_pose(NS(pose_landmarks=[]))
        self.assertFalse(detected)
        self.assertEqual(values.shape, (33, 4))
        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(np.isnan(values).all())

    def test_reject_malformed_not_missing(self):
        bad = [pose(32), pose(34), NS(), NS(pose_landmarks=None),
               NS(pose_landmarks=[pose().pose_landmarks[0]] * 2)]
        for field, value in (("x", np.nan), ("y", np.inf), ("visibility", np.nan), ("z", 1e100)):
            result = pose()
            setattr(result.pose_landmarks[0][0], field, value)
            bad.append(result)
        result = pose()
        del result.pose_landmarks[0][0].visibility
        bad.append(result)
        for result in bad:
            with self.subTest(result=result), self.assertRaises(pose_raw.ExtractionError) as caught:
                pose_raw.encode_pose(result)
            self.assertEqual(caught.exception.error_type, "malformed_pose_result")

    def test_schema_rejects_names_dtypes_shapes_indices_times_and_nan_mismatch(self):
        pose_raw.validate_arrays(example_arrays(), 20, 2)
        cases = []
        a = example_arrays(); a["frame_label"] = np.zeros(2); cases.append(a)
        a = example_arrays(); del a["timestamp_ms"]; cases.append(a)
        for key in pose_raw.ARRAY_DTYPES:
            a = example_arrays(); a[key] = a[key].astype(object); cases.append(a)
            a = example_arrays(); a[key] = a[key][:-1]; cases.append(a)
        for key, values in (("frame_index", [1, 2]), ("frame_index", [0, 2]),
                            ("timestamp_ms", [0, 100]), ("timestamp_ms", [0, 0])):
            a = example_arrays(); a[key][:] = values; cases.append(a)
        a = example_arrays(); a["landmarks"][0, 0, 0] = np.nan; cases.append(a)
        a = example_arrays(); a["landmarks"][1] = 0; cases.append(a)
        a = example_arrays(); a["landmarks"][1, 0, 0] = 1; cases.append(a)
        a = example_arrays(); a["landmarks"] = a["landmarks"][:, :32]; cases.append(a)
        for i, arrays in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                pose_raw.validate_arrays(arrays, 20, 2)

    def test_publish_reload_without_pickle_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.npz"
            arrays = pose_raw.publish_npz(path, example_arrays(), 20, 2)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), set(pose_raw.ARRAY_DTYPES))
                for key in archive.files:
                    np.testing.assert_array_equal(arrays[key], archive[key])
            original = path.read_bytes()
            with self.assertRaises(pose_raw.ExtractionError):
                pose_raw.publish_npz(path, example_arrays(), 20, 2)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_reload_or_publication_does_not_publish_or_leave_temporary(self):
        for target, error in (("load_validated", ValueError("bad archive")),
                              ("os.link", OSError("disk error"))):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "pose.npz"
                with patch(f"ml.datasets.pose_raw.{target}", side_effect=error):
                    with self.assertRaises(pose_raw.ExtractionError) as caught:
                        pose_raw.publish_npz(path, example_arrays(), 20, 2)
                self.assertEqual(caught.exception.error_type, "output_persistence_failure")
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_object_archive_is_rejected_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            arrays = example_arrays(); arrays["landmarks"] = arrays["landmarks"].astype(object)
            np.savez(path, **arrays)
            with self.assertRaises(ValueError):
                pose_raw.load_validated(path, 20, 2)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.metadata = dict(source_fps=20., width=3, height=2, expected_frame_count=3)
        self.progress = dict(decoded_frame_count=0, extracted_frame_count=0,
                             pose_detected_count=0, pose_missing_count=0)
        self.frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(3)]
        self.timestamps = []
        self.results = [NS(pose_landmarks=[]), NS(pose_landmarks=[]), pose()]
        self.mp = NS(Image=lambda **kwargs: kwargs["data"], ImageFormat=NS(SRGB=1))
        self.cv = NS(COLOR_BGR2RGB=1, cvtColor=lambda frame, code: frame[..., ::-1])

    def run_decode(self, inference_error=None, decode_error=False):
        def read():
            if decode_error:
                raise RuntimeError("decoder failed")
            return (True, self.frames.pop(0)) if self.frames else (False, None)
        def detect(image, timestamp):
            self.timestamps.append(timestamp)
            if inference_error:
                raise inference_error
            return self.results.pop(0)
        return extractor.decode_records(NS(read=read), NS(detect_for_video=detect),
                                        self.metadata, self.progress, self.cv, self.mp)

    def test_missing_run_retains_every_frame_and_timestamp(self):
        arrays = self.run_decode()
        pose_raw.validate_arrays(arrays, 20, 3)
        self.assertEqual(self.timestamps, [0, 50, 100])
        self.assertEqual(arrays["pose_detected"].tolist(), [False, False, True])
        self.assertTrue(np.isnan(arrays["landmarks"][:2]).all())
        self.assertEqual(self.progress, dict(decoded_frame_count=3, extracted_frame_count=3,
                                            pose_detected_count=1, pose_missing_count=2))

    def test_wrong_decode_count_and_exception_are_failures(self):
        for count in (2, 4):
            self.setUp()
            self.frames = [self.frames[0]] * count
            with self.assertRaises(pose_raw.ExtractionError) as caught:
                self.run_decode()
            self.assertEqual(caught.exception.error_type, "video_decode_failure")
        self.setUp()
        with self.assertRaises(pose_raw.ExtractionError) as caught:
            self.run_decode(decode_error=True)
        self.assertEqual(caught.exception.error_type, "video_decode_failure")

    def test_inference_and_malformed_failures_are_not_counted_as_missing(self):
        for error_type in ("inference_runtime_failure", "malformed_pose_result"):
            self.setUp()
            self.results[0] = pose(32)
            with self.assertRaises(pose_raw.ExtractionError) as caught:
                self.run_decode(RuntimeError("inference failed") if error_type.startswith("inference") else None)
            self.assertEqual(caught.exception.error_type, error_type)
            self.assertEqual(caught.exception.frame_index, 0)
            self.assertEqual(self.progress["pose_missing_count"], 0)
            self.assertEqual(self.progress["extracted_frame_count"], 0)

    def test_capture_metadata_mismatch(self):
        cv = NS(CAP_PROP_FPS=0, CAP_PROP_FRAME_WIDTH=1, CAP_PROP_FRAME_HEIGHT=2, CAP_PROP_FRAME_COUNT=3)
        values = [20., 3., 2., 3.]
        capture = NS(isOpened=lambda: True, get=lambda prop: values[prop])
        extractor.verify_capture(capture, self.metadata, cv)
        for i in range(4):
            old = values[i]; values[i] = old + 1
            with self.assertRaises(pose_raw.ExtractionError):
                extractor.verify_capture(capture, self.metadata, cv)
            values[i] = old

    def test_invalid_decoded_frame_shape(self):
        for frame in (None, object(), np.zeros((2, 3)), np.zeros((2, 3, 4)), np.zeros((0, 3, 3))):
            self.setUp(); self.frames[0] = frame
            with self.subTest(frame=frame), self.assertRaises(pose_raw.ExtractionError) as caught:
                self.run_decode()
            self.assertEqual(caught.exception.error_type, "video_decode_failure")
            self.assertEqual(caught.exception.frame_index, 0)
            self.assertEqual(self.timestamps, [])
            self.assertEqual(self.progress["pose_missing_count"], 0)

    def test_early_and_extra_frame_error_indices_and_counts(self):
        for count, expected_index, extracted in ((0, 0, 0), (2, 2, 2), (4, 3, 3)):
            self.setUp(); self.frames = [self.frames[0]] * count
            with self.subTest(count=count), self.assertRaises(pose_raw.ExtractionError) as caught:
                self.run_decode()
            self.assertEqual(caught.exception.error_type, "video_decode_failure")
            self.assertEqual(caught.exception.frame_index, expected_index)
            self.assertEqual(self.progress["extracted_frame_count"], extracted)
            self.assertEqual(len(self.timestamps), extracted)

    def test_invalid_capture_metadata_and_unopened_source(self):
        cv = NS(CAP_PROP_FPS=0, CAP_PROP_FRAME_WIDTH=1, CAP_PROP_FRAME_HEIGHT=2, CAP_PROP_FRAME_COUNT=3)
        for index in range(4):
            for bad in (0, -1, np.nan, np.inf):
                values = [20., 3., 2., 3.]; values[index] = bad
                with self.subTest(index=index, bad=bad), self.assertRaises(pose_raw.ExtractionError) as caught:
                    extractor.verify_capture(NS(isOpened=lambda: True, get=lambda prop: values[prop]),
                                             self.metadata, cv)
                self.assertEqual(caught.exception.error_type, "invalid_source_metadata")
        with self.assertRaises(pose_raw.ExtractionError) as caught:
            extractor.verify_capture(NS(isOpened=lambda: False), self.metadata, cv)
        self.assertEqual(caught.exception.error_type, "video_decode_failure")

    def test_capture_released_on_tracker_failure_and_interruption(self):
        from unittest.mock import Mock
        for failure in (RuntimeError("tracker failed"), KeyboardInterrupt()):
            capture = Mock()
            cv = NS(VideoCapture=lambda: capture, error=RuntimeError)
            mp = NS(tasks=NS(vision=NS(PoseLandmarker=NS(create_from_options=Mock(side_effect=failure)))))
            with patch.object(extractor, "verify_capture"), patch.object(extractor, "pose_options"), \
                    self.assertRaises(KeyboardInterrupt if isinstance(failure, KeyboardInterrupt)
                                      else pose_raw.ExtractionError):
                extractor.extract_arrays(Path("fake.avi"), Path("fake.task"), self.metadata,
                                         self.progress, cv, mp)
            capture.release.assert_called_once()

    def test_source_identity_and_holdout_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(extractor, "SOURCE_ROOT", root):
                source = root / "Subject.2/Fall backwards/FallBackwardsS2.avi"
                source.parent.mkdir(parents=True); source.touch()
                metadata = extractor.source_metadata(source)
                self.assertEqual((metadata["subject_id"], metadata["split"]), (2, "train"))
                for path in (root / "Subject.6/Walk/no.avi", root / "Subject.7/Walk/no.avi",
                             root.parent / "outside.avi", source.with_name("unknown.avi")):
                    with self.assertRaises(pose_raw.ExtractionError):
                        extractor.source_metadata(path)
                link = source.with_name("link.avi"); link.symlink_to(source)
                with self.assertRaises(pose_raw.ExtractionError):
                    extractor.source_metadata(link)

    def test_output_refuses_raw_and_symlink(self):
        with self.assertRaises(pose_raw.ExtractionError):
            extractor.output_path(extractor.REPO / "data/raw", Path("pose.npz"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "alias").symlink_to(root, target_is_directory=True)
            with self.assertRaises(pose_raw.ExtractionError):
                extractor.output_path(root / "alias", Path("pose.npz"))

    def test_shared_mediapipe_configuration(self):
        base = lambda **kwargs: kwargs
        base.Delegate = NS(CPU="cpu")
        mp = NS(tasks=NS(BaseOptions=base, vision=NS(
            PoseLandmarkerOptions=lambda **kwargs: kwargs, RunningMode=NS(VIDEO="video"))))
        options = pose_options(Path("model.task"), mp)
        self.assertEqual(options, dict(base_options=dict(model_asset_path="model.task", delegate="cpu"),
                                      running_mode="video", num_poses=1, min_pose_detection_confidence=.5,
                                      min_pose_presence_confidence=.5, min_tracking_confidence=.5,
                                      output_segmentation_masks=False))


if __name__ == "__main__":
    unittest.main()
