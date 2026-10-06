"""Synthetic authority, freshness, HTTP and concurrency checks; no external services."""

import unittest
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from frontend.dashboard import DashboardServer, DashboardState, create_app


def application(status="INITIALIZING", actionable=False, missing=0, recovery=None):
    return SimpleNamespace(application_status=status, prediction_actionable=actionable,
                           consecutive_missing_frames=missing, recovery_frame_index=recovery)


def prediction(index=85):
    return SimpleNamespace(fall_probability=.72, predicted_label="fall",
                           buffer_length=86, frame_index=index)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.connected = Event()
        self.state = DashboardState("synthetic-edge", "synthetic-sequence", self.connected,
                                    clock=lambda: self.now)
        self.client = create_app(self.state).test_client()

    def update(self, gate, raw=None, received_at=None, index=90):
        self.state.update(frame_index=index, pose_detected=True, application=gate,
                          prediction=raw, prediction_received_at=received_at)

    def test_initial_snapshot_has_no_invented_prediction(self):
        value = self.state.snapshot()
        self.assertEqual(value["application_status"], "INITIALIZING")
        self.assertFalse(value["prediction_actionable"])
        self.assertIsNone(value["frame_index"])
        self.assertIsNone(value["pose_detected"])
        self.assertTrue(all(v is None for v in value["raw_prediction"].values()))
        self.assertEqual(value["cloud_prediction_status"], "WARMING_UP")
        self.assertIsNone(value["prediction_age_seconds"])

    def test_copies_all_application_states_without_recomputing(self):
        for status in ("INITIALIZING", "NORMAL", "FALL", "POSE_LOST", "RECOVERING"):
            with self.subTest(status=status):
                # Intentionally incompatible inputs prove the bridge does not decide status.
                gate = application(status, actionable=True, missing=99, recovery=800)
                self.update(gate)
                value = self.state.snapshot()
                self.assertEqual(value["application_status"], status)
                self.assertTrue(value["prediction_actionable"])
                self.assertEqual(value["consecutive_missing_frames"], 99)
                self.assertEqual(value["recovery_frame_index"], 800)

    def test_suppressed_raw_fall_preserved_in_json(self):
        gate = application("POSE_LOST", missing=6)
        self.state.update(frame_index=91, pose_detected=False, application=gate,
                          prediction=prediction(), prediction_received_at=100.)
        value = self.client.get("/api/state").get_json()
        self.assertEqual(value["application_status"], "POSE_LOST")
        self.assertFalse(value["prediction_actionable"])
        self.assertFalse(value["pose_detected"])
        self.assertEqual(value["raw_prediction"], dict(fall_probability=.72,
                         predicted_label="fall", buffer_length=86, frame_index=85))

    def test_http_page_api_static_and_no_post(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"APPLICATION STATUS", page.data)
        self.assertIn(b"RAW CLOUD MODEL", page.data)
        self.assertIn(b"Latest Inference Context", page.data)
        response = self.client.get("/api/state")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(self.client.post("/api/state").status_code, 405)
        for filename in ("dashboard.js", "dashboard.css"):
            with self.client.get(f"/static/{filename}") as response:
                self.assertEqual(response.status_code, 200)

    def test_only_allowlisted_scalars_are_exposed(self):
        gate = application("NORMAL", True)
        gate.features = [123.] * 132
        gate.MQTT_PASSWORD = "unused-synthetic-field"
        self.update(gate, prediction(), 100.)
        value = self.client.get("/api/state").get_json()
        self.assertEqual(set(value), {
            "source_id", "sequence_id", "frame_index", "pose_detected",
            "application_status", "prediction_actionable", "consecutive_missing_frames",
            "recovery_frame_index", "raw_prediction", "mqtt_connected",
            "cloud_prediction_status", "prediction_age_seconds", "prediction_stale_after_seconds",
        })
        self.assertEqual(set(value["raw_prediction"]), {
            "fall_probability", "predicted_label", "buffer_length", "frame_index",
        })

    def test_freshness_ages_without_polling_or_gate_updates(self):
        gate = application("FALL", True)
        self.update(gate, prediction(), 100.)
        self.assertEqual(self.state.snapshot()["cloud_prediction_status"], "ACTIVE")
        self.now = 104.999
        self.assertEqual(self.state.snapshot()["cloud_prediction_status"], "ACTIVE")
        self.now = 105.
        for _ in range(3):
            value = self.client.get("/api/state").get_json()
            self.assertEqual(value["cloud_prediction_status"], "STALE")
            self.assertEqual(value["prediction_age_seconds"], 5.)
            self.assertEqual(value["application_status"], "FALL")
            self.assertTrue(value["prediction_actionable"])
        self.update(gate, prediction(95), 105.)
        self.assertEqual(self.state.snapshot()["cloud_prediction_status"], "ACTIVE")

    def test_transport_is_live_even_without_edge_updates(self):
        self.assertFalse(self.state.snapshot()["mqtt_connected"])
        self.connected.set()
        self.assertTrue(self.state.snapshot()["mqtt_connected"])
        self.connected.clear()
        self.assertFalse(self.state.snapshot()["mqtt_connected"])
        self.assertEqual(self.state.snapshot()["cloud_prediction_status"], "WARMING_UP")

    def test_snapshot_does_not_keep_gate_or_mutable_raw_reference(self):
        gate = application("FALL", True)
        self.update(gate, prediction(), 100.)
        gate.application_status = "POSE_LOST"
        value = self.state.snapshot()
        value["raw_prediction"]["predicted_label"] = "non_fall"
        value["application_status"] = "NORMAL"
        next_value = self.state.snapshot()
        self.assertEqual(next_value["application_status"], "FALL")
        self.assertEqual(next_value["raw_prediction"]["predicted_label"], "fall")

    def test_concurrent_snapshots_are_coherent(self):
        errors = []

        def write():
            try:
                for index in range(2000):
                    self.update(application("FALL", True, missing=index, recovery=index),
                                prediction(index), 100., index=index)
            except BaseException as exc:
                errors.append(exc)

        thread = Thread(target=write)
        thread.start()
        for _ in range(2000):
            value = self.state.snapshot()
            if value["frame_index"] is not None:
                self.assertEqual(value["frame_index"], value["raw_prediction"]["frame_index"])
                self.assertEqual(value["frame_index"], value["consecutive_missing_frames"])
                self.assertEqual(value["frame_index"], value["recovery_frame_index"])
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_embedded_server_binds_loopback_and_shuts_down(self):
        stopped = Event()
        mock_server = MagicMock(server_port=8767)
        mock_server.serve_forever.side_effect = lambda **kwargs: stopped.wait(2)
        mock_server.shutdown.side_effect = stopped.set
        with patch("frontend.dashboard.make_server", return_value=mock_server) as make:
            server = DashboardServer(self.state)
            server.start()
            try:
                self.assertEqual(make.call_args.args[0], "127.0.0.1")
                self.assertTrue(server._thread.is_alive())
                self.assertFalse(server.app.debug)
            finally:
                server.close()
            self.assertFalse(server._thread.is_alive())
            server.close()
        mock_server.server_close.assert_called_once()

    def test_thread_start_failure_closes_bound_socket(self):
        mock_server = MagicMock(server_port=8767)
        with patch("frontend.dashboard.make_server", return_value=mock_server), \
             patch("frontend.dashboard.Thread") as thread:
            thread.return_value.start.side_effect = RuntimeError("Synthetic thread startup failure")
            server = DashboardServer(self.state)
            with self.assertRaises(RuntimeError):
                server.start()
            server.close()
        mock_server.server_close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
