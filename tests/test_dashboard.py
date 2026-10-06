"""Synthetic authority, freshness, HTTP and concurrency checks; no external services."""

import unittest
import json
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from frontend.dashboard import PROBABILITY_HISTORY_LIMIT, RECENT_EVENT_LIMIT, DashboardServer, DashboardState, create_app


def application(status="INITIALIZING", actionable=False, missing=0, recovery=None):
    return SimpleNamespace(application_status=status, prediction_actionable=actionable,
                           consecutive_missing_frames=missing, recovery_frame_index=recovery)


def prediction(index=85):
    return SimpleNamespace(fall_probability=.72, predicted_label="fall",
                           buffer_length=86, frame_index=index)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.wall_now = 1791291200.0
        self.connected = Event()
        self.state = DashboardState("synthetic-edge", "synthetic-sequence", self.connected,
                                    clock=lambda: self.now, wall_clock=lambda: self.wall_now)
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
        self.assertEqual(value["probability_history"], [])
        self.assertEqual(value["recent_events"], [])

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
            "probability_history",
            "recent_events",
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

    def test_history_deduplicates_new_indices_and_rejects_old_indices(self):
        gate = application("FALL", True)
        for _ in range(3):
            self.update(gate, prediction(85), 100.)
        self.update(gate, prediction(75), 100.)
        new = prediction(95)
        new.fall_probability = .18
        self.update(gate, new, 100.)
        self.assertEqual(self.state.snapshot()["probability_history"], [
            dict(frame_index=85, fall_probability=.72),
            dict(frame_index=95, fall_probability=.18),
        ])

    def test_history_is_bounded_and_api_remains_small(self):
        gate = application("NORMAL", True)
        for index in range(PROBABILITY_HISTORY_LIMIT + 5):
            self.update(gate, prediction(85 + index * 10), 100.)
        value = self.client.get("/api/state").get_json()
        history = value["probability_history"]
        self.assertEqual(len(history), PROBABILITY_HISTORY_LIMIT)
        self.assertEqual(history[0]["frame_index"], 135)
        self.assertEqual(history[-1]["frame_index"], 85 + (PROBABILITY_HISTORY_LIMIT + 4) * 10)
        self.assertLess(len(json.dumps(value).encode()), 8192)

    def test_polling_or_same_index_updates_cannot_add_or_replace_points(self):
        gate = application("NORMAL", True)
        raw = prediction()
        self.update(gate, raw, 100.)
        expected = [dict(frame_index=85, fall_probability=.72)]
        for _ in range(5):
            value = self.client.get("/api/state").get_json()
            self.assertEqual(value["probability_history"], expected)
        raw.fall_probability = .81
        self.update(gate, raw, 100.)
        self.assertEqual(self.state.snapshot()["probability_history"], expected)

    def test_history_survives_browser_reload_and_returns_independent_copies(self):
        self.update(application("FALL", True), prediction(), 100.)
        history = self.state.snapshot()["probability_history"]
        history[0]["fall_probability"] = 0.
        history.append(dict(frame_index=95, fall_probability=1.))
        refreshed_client = create_app(self.state).test_client()
        refreshed_client.get("/")
        self.assertEqual(refreshed_client.get("/api/state").get_json()["probability_history"], [
            dict(frame_index=85, fall_probability=.72),
        ])

    def test_history_retains_suppressed_fall(self):
        self.update(application("POSE_LOST", missing=6), prediction(), 100.)
        value = self.client.get("/api/state").get_json()
        self.assertEqual(value["application_status"], "POSE_LOST")
        self.assertFalse(value["prediction_actionable"])
        self.assertEqual(value["raw_prediction"]["predicted_label"], "fall")
        self.assertEqual(value["probability_history"], [dict(frame_index=85, fall_probability=.72)])

    def test_probability_is_not_reclassified_and_threshold_is_only_a_chart_reference(self):
        raw = prediction()
        raw.fall_probability = 0.
        self.update(application("NORMAL", False), raw, 100.)
        value = self.client.get("/api/state").get_json()
        self.assertEqual(value["application_status"], "NORMAL")
        self.assertEqual(value["raw_prediction"]["predicted_label"], "fall")
        self.assertEqual(value["probability_history"][0]["fall_probability"], 0.)
        self.assertIn(b"50% reference", self.client.get("/").data)
        with self.client.get("/static/dashboard.js") as response:
            js = response.data.decode()
        self.assertIn("const THRESHOLD_REFERENCE = 0.5", js)
        self.assertIn("statusNode.dataset.status = state.application_status", js)
        self.assertIn("raw.predicted_label.replace", js)
        self.assertIn("probabilityHistory = state.probability_history", js)
        self.assertNotRegex(js, r"fall_probability\s*(?:>=|>)")
        self.assertNotRegex(js, r"probabilityHistory\.(?:push|unshift)")

    def test_application_transition_events_are_deduplicated(self):
        for status in ("NORMAL", "NORMAL", "FALL", "FALL", "FALL"):
            self.update(application(status), prediction(), 100.)
        events = self.state.snapshot()["recent_events"]
        self.assertEqual(events, [dict(type="fall", label="FALL", message="Application entered FALL",
                                      observed_at_ms=1791291200000)])

    def test_pose_lost_event_does_not_change_raw_fall(self):
        self.update(application("FALL", True), prediction(), 100.)
        for _ in range(100):
            self.update(application("POSE_LOST", missing=6), prediction(), 100.)
        value = self.client.get("/api/state").get_json()
        self.assertEqual(len(value["recent_events"]), 1)
        self.assertEqual(value["recent_events"][0]["type"], "pose_lost")
        self.assertEqual(value["recent_events"][0]["label"], "POSE LOST")
        self.assertEqual(value["raw_prediction"]["predicted_label"], "fall")
        self.assertEqual(value["raw_prediction"]["fall_probability"], .72)
        self.assertFalse(value["prediction_actionable"])

    def test_recovery_events_distinguish_normal_from_fall(self):
        for final in ("NORMAL", "FALL"):
            with self.subTest(final=final):
                state = DashboardState("synthetic", "sequence", Event(), wall_clock=lambda: self.wall_now)
                for status in ("POSE_LOST", "RECOVERING", final):
                    state.update(frame_index=90, pose_detected=True, prediction=prediction(),
                                 application=application(status), prediction_received_at=100.)
                events = state.snapshot()["recent_events"]
                self.assertEqual(events[1]["type"], "recovering")
                self.assertEqual(events[0]["type"], "recovered" if final == "NORMAL" else "fall")
                self.assertEqual(events[0]["message"], "Application returned to NORMAL" if final == "NORMAL"
                                 else "Fresh cloud prediction restored FALL state")
                self.assertEqual(len(events), 2)

    def test_first_observed_state_is_only_a_baseline(self):
        self.client.get("/api/state")
        self.update(application("NORMAL", True), prediction(), 100.)
        self.assertEqual(self.state.snapshot()["recent_events"], [])

    def test_recent_events_are_bounded_and_newest_first(self):
        self.update(application("NORMAL"))
        for index in range(RECENT_EVENT_LIMIT + 5):
            self.wall_now += 1.
            self.update(application("FALL" if index % 2 == 0 else "NORMAL"))
        events = self.state.snapshot()["recent_events"]
        self.assertEqual(len(events), RECENT_EVENT_LIMIT)
        self.assertEqual(events[0]["observed_at_ms"], 1791291225000)
        self.assertEqual(events[-1]["observed_at_ms"], 1791291206000)
        self.assertTrue(all(a["observed_at_ms"] > b["observed_at_ms"] for a, b in zip(events, events[1:])))

    def test_api_reads_and_browser_refresh_do_not_create_events(self):
        self.update(application("NORMAL"))
        self.update(application("FALL"))
        expected = self.state.snapshot()["recent_events"]
        refreshed = create_app(self.state).test_client()
        for _ in range(10):
            refreshed.get("/")
            self.assertEqual(refreshed.get("/api/state").get_json()["recent_events"], expected)

    def test_event_snapshots_are_independent_and_use_wall_clock(self):
        self.update(application("NORMAL"))
        self.now += 500.
        self.wall_now += 2.
        self.update(application("FALL"))
        value = self.state.snapshot()
        self.assertEqual(value["recent_events"][0]["observed_at_ms"], 1791291202000)
        value["recent_events"][0]["type"] = "normal"
        value["recent_events"].clear()
        self.assertEqual(self.state.snapshot()["recent_events"][0]["type"], "fall")

    def test_initializing_transport_and_freshness_do_not_create_events(self):
        self.update(application("NORMAL"), prediction(), 100.)
        self.update(application("INITIALIZING"))
        self.connected.set()
        self.now = 110.
        self.state.snapshot()
        self.connected.clear()
        self.state.snapshot()
        self.assertEqual(self.state.snapshot()["recent_events"], [])

    def test_alert_strip_is_hidden_by_default_and_uses_only_application_state(self):
        page = self.client.get("/").data.decode()
        self.assertIn('<div id="fall-alert" role="alert" hidden>', page)
        self.assertIn('<strong>FALL DETECTED</strong>', page)
        with self.client.get("/static/dashboard.js") as response:
            js = response.data.decode()
        self.assertEqual(js.count("document.getElementById('fall-alert')"), 1)
        self.assertIn("document.getElementById('fall-alert').hidden = state.application_status !== 'FALL';", js)
        self.assertIn("renderRecentEvents(state.recent_events)", js)
        self.update(application("POSE_LOST"), prediction(), 100.)
        value = self.client.get("/api/state").get_json()
        self.assertEqual(value["application_status"], "POSE_LOST")
        self.assertEqual(value["raw_prediction"]["predicted_label"], "fall")
        self.update(application("FALL", True), prediction(), 100.)
        self.assertEqual(self.client.get("/api/state").get_json()["application_status"], "FALL")


if __name__ == "__main__":
    unittest.main()
