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
                      fall_probability=.014, predicted_label="non_fall", buffer_length=200, frame_index=85)
    prediction.update(changes)
    return SimpleNamespace(payload=json.dumps(prediction).encode())


class PredictionTests(unittest.TestCase):
    def test_new_prediction_arrival_time_ignores_duplicates_and_late_messages(self):
        latest = edge.LatestPrediction("webcam-demo", "demo-sequence")
        self.assertEqual(latest.snapshot_with_received_at(), (None, None))
        with patch.object(edge.time, "monotonic", return_value=10.) as clock:
            latest.on_message(None, None, prediction_message(frame_index=85))
            clock.return_value = 20.
            latest.on_message(None, None, prediction_message(frame_index=85))
            latest.on_message(None, None, prediction_message(frame_index=75))
            self.assertEqual(latest.snapshot_with_received_at()[1], 10.)
            latest.on_message(None, None, prediction_message(frame_index=95))
            raw, received_at = latest.snapshot_with_received_at()
            self.assertEqual(raw.frame_index, 95)
            self.assertEqual(received_at, 20.)

    def test_latest_callback_snapshot_and_invalid_messages(self):
        latest = edge.LatestPrediction("webcam-demo", "demo-sequence")
        self.assertIsNone(latest.snapshot())
        thread = Thread(target=latest.on_message, args=(None, None, prediction_message()))
        thread.start()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(latest.snapshot(), (.014, "non_fall", 200, 85))
        for change in (dict(source_id="other"), dict(sequence_id="old-sequence"),
                       dict(schema_version=True), dict(fall_probability=True),
                       dict(fall_probability=float("nan")), dict(fall_probability=float("inf")),
                       dict(fall_probability=1.1), dict(predicted_label="invalid"),
                       dict(buffer_length=True), dict(buffer_length=0), dict(buffer_length=None),
                       dict(frame_index=True), dict(frame_index=-1), dict(frame_index=None), dict(frame_index=1.5)):
            with self.subTest(change=change):
                latest.on_message(None, None, prediction_message(**change))
                self.assertEqual(latest.snapshot(), (.014, "non_fall", 200, 85))
        for payload in (b'{"broken":', b'[]', b'\xff'):
            latest.on_message(None, None, SimpleNamespace(payload=payload))
            self.assertEqual(latest.snapshot(), (.014, "non_fall", 200, 85))
        latest.on_message(None, None, prediction_message(fall_probability=.9, predicted_label="fall", buffer_length=86, frame_index=95))
        self.assertEqual(latest.snapshot(), (.9, "fall", 86, 95))
        latest.on_message(None, None, prediction_message(frame_index=85))
        self.assertEqual(latest.snapshot(), (.9, "fall", 86, 95))

    def test_waiting_overlay_prediction_overlay_and_q(self):
        cv2, frame = MagicMock(), np.zeros((480, 640, 3), dtype=np.uint8)
        application = edge.PoseValidityGate(5)
        application.step(False, 0, None)
        cv2.waitKey.return_value = -1
        self.assertFalse(edge.preview_frame(frame, None, cv2, pose_detected=False, application=application))
        lines = [call.args[1] for call in cv2.putText.call_args_list]
        for expected in ("INITIALIZING CLOUD MODEL...", "--", "MISSING", "WAITING", "Press Q to exit"):
            self.assertIn(expected, lines)
        self.assertFalse(any("%" in line for line in lines))
        cv2.putText.reset_mock()
        cv2.waitKey.return_value = ord("q")
        prediction = edge.Prediction(.014, "non_fall", 200, 85)
        application.step(True, 85, prediction)
        self.assertTrue(edge.preview_frame(frame, prediction, cv2, pose_detected=True, application=application))
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
                prediction = edge.Prediction(probability, label, 200, 85)
                application = edge.PoseValidityGate(5)
                application.step(True, 85, prediction)
                edge.preview_frame(np.zeros((480, 640, 3), dtype=np.uint8),
                                   prediction, cv2, pose_detected=True, application=application)
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


class PoseValidityTests(unittest.TestCase):
    def test_missing_boundary_preserves_raw_fall(self):
        latest = edge.LatestPrediction("webcam-demo", "demo-sequence")
        latest.on_message(None, None, prediction_message(fall_probability=.72, predicted_label="fall", frame_index=100))
        raw = latest.snapshot()
        application = edge.PoseValidityGate(edge.PreprocessingConfig().max_forward_fill_frames)
        application.step(True, 100, raw)
        for missing_count in range(1, 6):
            application.step(False, 100 + missing_count, raw)
            self.assertEqual(application.consecutive_missing_frames, missing_count)
            self.assertEqual(application.application_status, "FALL")
            self.assertTrue(application.prediction_actionable)
        application.step(False, 106, raw)
        self.assertEqual(application.application_status, "POSE_LOST")
        self.assertFalse(application.prediction_actionable)
        self.assertIs(latest.snapshot(), raw)
        self.assertEqual(raw, (.72, "fall", 200, 100))

    def test_initializing_and_leading_missing(self):
        application = edge.PoseValidityGate(5)
        application.step(True, 0, None)
        self.assertEqual(application.application_status, "INITIALIZING")
        self.assertFalse(application.prediction_actionable)
        for index in range(1, 7):
            application.step(False, index, None)
        self.assertEqual(application.application_status, "POSE_LOST")
        application.step(True, 7, None)
        self.assertEqual(application.application_status, "RECOVERING")
        self.assertFalse(application.prediction_actionable)

    def test_recovery_waits_for_return_frame_and_resumes_raw_normal_or_fall(self):
        for label, probability, fresh_index, expected in (("non_fall", .014, 6, "NORMAL"),
                                                        ("fall", .975236, 15, "FALL")):
            with self.subTest(label=label):
                application = edge.PoseValidityGate(5)
                stale = edge.Prediction(.72, "fall", 200, 5)
                for index in range(6):
                    application.step(False, index, stale)
                application.step(True, 6, stale)
                self.assertEqual(application.recovery_frame_index, 6)
                self.assertEqual(application.consecutive_missing_frames, 0)
                for prediction in (None, stale):
                    application.step(True, 7, prediction)
                    self.assertEqual(application.application_status, "RECOVERING")
                    self.assertFalse(application.prediction_actionable)
                    self.assertEqual(application.recovery_frame_index, 6)
                fresh = edge.Prediction(probability, label, 200, fresh_index)
                application.step(True, max(8, fresh_index), fresh)
                self.assertEqual(application.application_status, expected)
                self.assertTrue(application.prediction_actionable)
                self.assertIsNone(application.recovery_frame_index)
                self.assertEqual(fresh.predicted_label, label)
                self.assertEqual(fresh.fall_probability, probability)

    def test_repeated_loss_records_a_new_recovery_frame(self):
        application = edge.PoseValidityGate(5)
        raw = edge.Prediction(.72, "fall", 200, 5)
        for index in range(6):
            application.step(False, index, raw)
        application.step(True, 6, raw)
        for index in range(7, 12):
            application.step(False, index, raw)
            self.assertEqual(application.application_status, "RECOVERING")
            self.assertFalse(application.prediction_actionable)
        application.step(False, 12, raw)
        self.assertEqual(application.application_status, "POSE_LOST")
        application.step(True, 13, edge.Prediction(.014, "non_fall", 200, 6))
        self.assertEqual(application.recovery_frame_index, 13)
        self.assertEqual(application.application_status, "RECOVERING")
        self.assertFalse(application.prediction_actionable)

    def test_suppressed_overlay_keeps_raw_fall_visible(self):
        application = edge.PoseValidityGate(5)
        raw = edge.Prediction(.72, "fall", 200, 5)
        for index in range(6):
            application.step(False, index, raw)
        for detected, expected in ((False, "POSE LOST"), (True, "RECOVERING")):
            with self.subTest(status=expected):
                application.step(detected, 6, raw)
                cv2 = MagicMock()
                cv2.waitKey.return_value = -1
                edge.preview_frame(np.zeros((480, 640, 3), dtype=np.uint8), raw, cv2,
                                   pose_detected=detected, application=application)
                lines = [call.args[1] for call in cv2.putText.call_args_list]
                for text in (expected, "FALL", "72.0%", "CONNECTED"):
                    self.assertIn(text, lines)
                self.assertNotIn("NORMAL", lines)
                self.assertTrue(any("SUPPRESSED" in line for line in lines))
                raw_label = next(call for call in cv2.putText.call_args_list if call.args[1] == "FALL")
                self.assertEqual(raw_label.args[5], (173, 184, 193))
                self.assertFalse(application.prediction_actionable)
                if detected:
                    self.assertIn("Raw Cloud (stale):", lines)


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

    def test_pose_loss_and_recovery_do_not_reset_sequence_or_change_features(self):
        self.cv2.waitKey.side_effect = [-1] * 8 + [ord("q")]
        missing = np.full((33, 4), np.nan, dtype=np.float32)
        observations = [(True, self.landmarks)] + [(False, missing)] * 6 + [(True, self.landmarks)] * 2
        states = []
        original_preview = edge.preview_frame

        def publish(topic, message):
            self.published.append((topic, message))
            if message["message_type"] == "frame":
                index = message["frame_index"]
                if index in (0, 7, 8):
                    callback = self.constructor.call_args.kwargs["on_message"]
                    callback(None, None, prediction_message(frame_index={0: 0, 7: 6, 8: 7}[index],
                        fall_probability=.014 if index == 8 else .72,
                        predicted_label="non_fall" if index == 8 else "fall"))

        def preview(*args, **kwargs):
            application = kwargs["application"]
            states.append((application.application_status, application.prediction_actionable))
            return original_preview(*args, **kwargs)

        self.connection.publish.side_effect = publish
        with patch.object(edge, "encode_pose", side_effect=observations), patch.object(edge, "preview_frame", side_effect=preview):
            edge.main()
        self.assertEqual(states, [("FALL", True)] * 6 + [("POSE_LOST", False), ("RECOVERING", False), ("NORMAL", True)])
        messages = [message for _, message in self.published]
        self.assertEqual([message["message_type"] for message in messages], ["sequence_start"] + ["frame"] * 9 + ["sequence_end"])
        self.assertEqual({message["sequence_id"] for message in messages}, {"demo-sequence"})
        for index, message in enumerate(messages[1:-1]):
            self.assertEqual(message["frame_index"], index)
            self.assertEqual(message["pose_detected"], index not in range(1, 7))
            self.assertEqual(message["features"], [0.] * 132 if index == 6 else [.25] * 132)
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

    def test_dashboard_without_preview_subscribes_and_copies_authoritative_gate(self):
        from frontend.dashboard import DashboardState
        self.argv.remove("--preview")
        self.argv.append("--dashboard")
        self.capture.read.side_effect = [(True, self.frame), KeyboardInterrupt]
        self.connection.connected.is_set.return_value = True
        server = MagicMock(port=8767)
        created_states = []

        def state(*args):
            created_states.append(DashboardState(*args))
            return created_states[-1]

        def publish(topic, message):
            self.published.append((topic, message))
            if message["message_type"] == "frame":
                callback = self.constructor.call_args.kwargs["on_message"]
                callback(None, None, prediction_message())

        self.connection.publish.side_effect = publish
        self.connection.start.side_effect = lambda: self.assertEqual(self.connection.subscription, "fall/prediction/webcam-demo")
        with patch("frontend.dashboard.DashboardState", side_effect=state), \
             patch("frontend.dashboard.DashboardServer", return_value=server), \
             patch.object(edge, "PoseValidityGate", wraps=edge.PoseValidityGate) as gate:
            edge.main()
        gate.assert_called_once_with(5)
        value = created_states[0].snapshot()
        self.assertEqual(value["application_status"], "NORMAL")
        self.assertTrue(value["prediction_actionable"])
        self.assertEqual(value["raw_prediction"]["fall_probability"], .014)
        self.assertEqual(value["cloud_prediction_status"], "ACTIVE")
        self.assertEqual(set(self.published[1][1]), {
            "schema_version", "source_id", "sequence_id", "message_type", "frame_index",
            "timestamp_ms", "timestamp_kind", "fps", "pose_detected", "features",
        })
        self.assertEqual(self.published[1][1]["features"], [.25] * 132)
        server.start.assert_called_once()
        server.close.assert_called_once()
        self.cv2.imshow.assert_not_called()
        self.assert_released(preview=False)

    def test_dashboard_cleanup_on_q_ctrl_c_avi_eof_and_startup_failure(self):
        scenarios = ("q", "ctrl_c", "avi_eof", "mqtt_start", "dashboard_start", "sequence_end")
        for scenario in scenarios:
            with self.subTest(exit=scenario):
                self.capture.reset_mock()
                self.connection.reset_mock()
                self.cv2.reset_mock()
                self.capture.read.side_effect = None
                self.capture.read.return_value = (True, self.frame)
                self.connection.start.side_effect = None
                self.connection.publish.side_effect = lambda *args: None
                self.argv[:] = ["edge.main", "--source", "0", "--pose-model", str(self.model), "--preview", "--dashboard"]
                server = MagicMock(port=8767)
                if scenario == "ctrl_c":
                    self.capture.read.side_effect = KeyboardInterrupt
                elif scenario == "avi_eof":
                    self.argv[self.argv.index("--source") + 1] = "/private/tmp/synthetic.avi"
                    self.argv.append("--unpaced")
                    self.metadata.side_effect = None
                    self.metadata.return_value = dict(source_fps=20., expected_frame_count=0,
                                                      source_relative_video_path="synthetic.avi")
                    self.capture.read.return_value = (False, None)
                elif scenario == "mqtt_start":
                    self.connection.start.side_effect = ConnectionError("Synthetic MQTT failure")
                elif scenario == "dashboard_start":
                    server.start.side_effect = RuntimeError("Synthetic Dashboard failure")
                elif scenario == "sequence_end":
                    def publish(topic, message):
                        if message["message_type"] == "sequence_end":
                            raise ConnectionError("Synthetic end failure")
                    self.connection.publish.side_effect = publish
                with patch("frontend.dashboard.DashboardServer", return_value=server), \
                     patch.object(edge, "verify_capture"):
                    if scenario in ("mqtt_start", "dashboard_start", "sequence_end"):
                        with self.assertRaises((ConnectionError, RuntimeError)):
                            edge.main()
                    else:
                        edge.main()
                server.close.assert_called_once()
                self.assert_released()


if __name__ == "__main__":
    unittest.main()
