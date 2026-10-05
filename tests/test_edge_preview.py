"""Local preview checks using synthetic frames and mocked camera/MQTT/GUI only."""

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from edge import main as edge


def prediction_message(**changes):
    prediction = dict(schema_version=1, source_id="webcam-demo", sequence_id="demo-sequence",
                      fall_probability=.014, predicted_label="non_fall", buffer_length=200)
    prediction.update(changes)
    return SimpleNamespace(payload=json.dumps(prediction).encode())


class PredictionTests(unittest.TestCase):
    def test_latest_callback_snapshot_and_invalid_messages(self):
        latest = edge.LatestPrediction("webcam-demo", "demo-sequence")
        self.assertIsNone(latest.snapshot())
        thread = Thread(target=latest.on_message, args=(None, None, prediction_message()))
        thread.start()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(latest.snapshot(), (.014, "non_fall", 200))
        for change in (dict(source_id="other"), dict(sequence_id="old-sequence"),
                       dict(schema_version=True), dict(fall_probability=True),
                       dict(fall_probability=float("nan")), dict(fall_probability=float("inf")),
                       dict(fall_probability=1.1), dict(predicted_label="invalid"),
                       dict(buffer_length=True), dict(buffer_length=0), dict(buffer_length=None)):
            with self.subTest(change=change):
                latest.on_message(None, None, prediction_message(**change))
                self.assertEqual(latest.snapshot(), (.014, "non_fall", 200))
        for payload in (b'{"broken":', b'[]', b'\xff'):
            latest.on_message(None, None, SimpleNamespace(payload=payload))
            self.assertEqual(latest.snapshot(), (.014, "non_fall", 200))
        latest.on_message(None, None, prediction_message(fall_probability=.9, predicted_label="fall", buffer_length=86))
        self.assertEqual(latest.snapshot(), (.9, "fall", 86))

    def test_waiting_overlay_prediction_overlay_and_q(self):
        cv2, frame = MagicMock(), np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.waitKey.return_value = -1
        self.assertFalse(edge.preview_frame(frame, None, cv2, pose_detected=False))
        lines = [call.args[1] for call in cv2.putText.call_args_list]
        for expected in ("INITIALIZING CLOUD MODEL...", "--", "MISSING", "WAITING", "Press Q to exit"):
            self.assertIn(expected, lines)
        self.assertFalse(any("%" in line for line in lines))
        cv2.putText.reset_mock()
        cv2.waitKey.return_value = ord("q")
        self.assertTrue(edge.preview_frame(frame, (.014, "non_fall", 200), cv2, pose_detected=True))
        lines = [call.args[1] for call in cv2.putText.call_args_list]
        for expected in ("FALL DETECTION SYSTEM", "NORMAL", "1.4%", "200 frames", "DETECTED", "CONNECTED"):
            self.assertIn(expected, lines)
        cv2.imshow.assert_called_with("Cloud Fall Detection", frame)
        cv2.waitKey.assert_called_with(1)

    def test_status_uses_cloud_label_and_bar_spans_zero_to_one(self):
        # Deliberately differing probabilities/labels ensure the UI does not reclassify.
        for probability, label, status, color in ((0., "fall", "FALL", (82, 82, 245)),
                                                  (1., "non_fall", "NORMAL", (105, 218, 105)),
                                                  (.014, "non_fall", "NORMAL", (105, 218, 105))):
            with self.subTest(probability=probability, label=label):
                cv2 = MagicMock()
                cv2.waitKey.return_value = -1
                edge.preview_frame(np.zeros((480, 640, 3), dtype=np.uint8),
                                   (probability, label, 200), cv2, pose_detected=True)
                status_call = next(call for call in cv2.putText.call_args_list if call.args[1] == status)
                self.assertEqual(status_call.args[5], color)
                bars = [call for call in cv2.rectangle.call_args_list if call.args[1] == (32, 307)]
                self.assertEqual(len(bars), 2 if probability > 0 else 1)
                if probability > 0:
                    self.assertEqual(bars[-1].args[2:4], ((32 + round(334 * probability), 319), color))

    def test_preview_cli_help(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["edge.main", "--help"]), redirect_stdout(output):
            with self.assertRaises(SystemExit) as result:
                edge.main()
        self.assertEqual(result.exception.code, 0)
        self.assertIn("--preview", output.getvalue())


class EdgeExitTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.model = Path(temporary) / "synthetic.task"
        self.model.write_bytes(b"synthetic pose model for mocked tracker")
        self.enterContext(patch.object(edge, "MODEL_SHA256", hashlib.sha256(self.model.read_bytes()).hexdigest()))
        self.argv = ["edge.main", "--source", "0", "--source-id", "webcam-demo",
                     "--sequence-id", "demo-sequence", "--camera-fps", "20",
                     "--pose-model", str(self.model), "--preview"]
        self.enterContext(patch.object(sys, "argv", self.argv))
        self.cv2, self.mp, self.capture = MagicMock(), MagicMock(), MagicMock()
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.capture.isOpened.return_value = True
        self.capture.read.return_value = (True, self.frame)
        self.cv2.VideoCapture.return_value = self.capture
        self.cv2.waitKey.return_value = ord("q")
        self.cv2.cvtColor.return_value = self.frame.copy()
        self.enterContext(patch.dict(sys.modules, cv2=self.cv2, mediapipe=self.mp))
        self.enterContext(patch.object(edge, "pose_options", return_value=object()))
        self.landmarks = np.full((33, 4), .25, dtype=np.float32)
        self.enterContext(patch.object(edge, "encode_pose", return_value=(True, self.landmarks)))
        self.metadata = self.enterContext(patch.object(edge, "source_metadata", side_effect=AssertionError("Unexpected dataset access")))
        self.connection = MagicMock(prefix="fall", subscription=None)
        self.constructor = self.enterContext(patch.object(edge, "MQTTConnection", return_value=self.connection))
        self.published = []
        self.connection.publish.side_effect = lambda topic, message: self.published.append((topic, message))

    def assert_released(self, preview=True):
        self.capture.release.assert_called_once_with()
        self.connection.close.assert_called_once_with()
        if preview:
            self.cv2.destroyAllWindows.assert_called_once_with()
        else:
            self.cv2.destroyAllWindows.assert_not_called()

    def test_preview_q_subscribes_and_preserves_frame_payload(self):
        def publish(topic, message):
            self.published.append((topic, message))
            if message["message_type"] == "frame":
                callback = self.constructor.call_args.kwargs["on_message"]
                callback(None, None, prediction_message())
        self.connection.publish.side_effect = publish
        self.connection.start.side_effect = lambda: self.assertEqual(self.connection.subscription, "fall/prediction/webcam-demo")
        edge.main()
        self.cv2.VideoCapture.assert_called_once_with(0)
        self.metadata.assert_not_called()
        self.assertEqual([m["message_type"] for _, m in self.published], ["sequence_start", "frame", "sequence_end"])
        self.assertTrue(all(topic == "fall/pose/webcam-demo" for topic, _ in self.published))
        message = self.published[1][1]
        self.assertEqual(message["features"], [.25] * 132)
        self.assertEqual((message["frame_index"], message["fps"], message["timestamp_kind"]), (0, 20., "unix_epoch"))
        self.assertTrue(message["pose_detected"])
        lines = [call.args[1] for call in self.cv2.putText.call_args_list]
        for expected in ("1.4%", "NORMAL", "200 frames", "DETECTED", "CONNECTED"):
            self.assertIn(expected, lines)
        tracker = self.mp.tasks.vision.PoseLandmarker.create_from_options.return_value.__enter__.return_value
        tracker.detect_for_video.assert_called_once()
        self.assert_released()

    def test_missing_pose_is_displayed_without_changing_payload(self):
        with patch.object(edge, "encode_pose", return_value=(False, np.full((33, 4), np.nan, dtype=np.float32))):
            edge.main()
        message = self.published[1][1]
        self.assertFalse(message["pose_detected"])
        self.assertEqual(message["features"], [0.] * 132)
        lines = [call.args[1] for call in self.cv2.putText.call_args_list]
        self.assertIn("MISSING", lines)
        self.assertIn("INITIALIZING CLOUD MODEL...", lines)
        self.assertIn("WAITING", lines)
        self.assert_released()

    def test_ctrl_c_sends_sequence_end_and_returns_without_exception(self):
        self.capture.read.side_effect = KeyboardInterrupt
        edge.main()
        self.assertEqual([m["message_type"] for _, m in self.published], ["sequence_start", "sequence_end"])
        self.assert_released()

    def test_mqtt_start_failure_still_releases_resources_without_end(self):
        self.connection.start.side_effect = ConnectionError("Synthetic startup failure")
        with self.assertRaises(ConnectionError):
            edge.main()
        self.assertEqual(self.published, [])
        self.assert_released()

    def test_sequence_end_failure_still_releases_resources(self):
        def publish(topic, message):
            if message["message_type"] == "sequence_end":
                raise ConnectionError("Synthetic end publication failure")
            self.published.append((topic, message))
        self.connection.publish.side_effect = publish
        with self.assertRaises(ConnectionError):
            edge.main()
        self.assert_released()

    def test_existing_headless_avi_eof_path(self):
        self.argv.remove("--preview")
        self.argv[self.argv.index("--source") + 1] = "/private/tmp/synthetic.avi"
        self.argv.append("--unpaced")
        self.metadata.side_effect = None
        self.metadata.return_value = dict(source_fps=20., expected_frame_count=1,
                                          source_relative_video_path="synthetic.avi")
        self.capture.read.side_effect = [(True, self.frame), (False, None)]
        with patch.object(edge, "verify_capture"):
            edge.main()
        self.constructor.assert_called_once_with(on_message=None)
        self.assertIsNone(self.connection.subscription)
        self.assertEqual([m["message_type"] for _, m in self.published], ["sequence_start", "frame", "sequence_end"])
        self.cv2.imshow.assert_not_called()
        self.cv2.waitKey.assert_not_called()
        self.assert_released(preview=False)


if __name__ == "__main__":
    unittest.main()
