"""Loopback Dashboard reads copies of authoritative Edge state, never decides falls."""

import time
from collections import deque
from threading import Event, Lock, Thread
from typing import Any, Callable, Protocol

from flask import Flask, jsonify, render_template
from werkzeug.serving import BaseWSGIServer, make_server

DEFAULT_PORT = 8767
PREDICTION_STALE_SECONDS = 5.0  # UI health only; never changes the application gate.
PROBABILITY_HISTORY_LIMIT = 60
RECENT_EVENT_LIMIT = 20


class PredictionView(Protocol):
    fall_probability: float
    predicted_label: str
    buffer_length: int
    frame_index: int


class ApplicationView(Protocol):
    application_status: str
    prediction_actionable: bool
    consecutive_missing_frames: int
    recovery_frame_index: int | None


class DashboardState:
    """One Edge writer copies scalar fields; HTTP readers get independent snapshots."""

    def __init__(self, source_id: str, sequence_id: str, connected: Event,
                 *, clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time) -> None:
        self._lock = Lock()
        self._connected = connected
        self._clock = clock
        self._wall_clock = wall_clock
        self._recent_events: deque[dict[str, str | int]] = deque(maxlen=RECENT_EVENT_LIMIT)
        self._last_event_status: str | None = None
        self._prediction_received_at: float | None = None
        self._probability_history: deque[dict[str, int | float]] = deque(maxlen=PROBABILITY_HISTORY_LIMIT)
        self._last_history_frame: int | None = None
        self._value: dict[str, Any] = dict(
            source_id=source_id, sequence_id=sequence_id, frame_index=None,
            pose_detected=None, application_status="INITIALIZING",
            prediction_actionable=False, consecutive_missing_frames=0,
            recovery_frame_index=None,
            raw_prediction=dict(fall_probability=None, predicted_label=None,
                                buffer_length=None, frame_index=None),
        )

    def update(self, *, frame_index: int, pose_detected: bool,
               prediction: PredictionView | None, application: ApplicationView,
               prediction_received_at: float | None) -> None:
        # Only the Edge main thread calls this, immediately after application.step().
        raw = dict(fall_probability=None, predicted_label=None,
                   buffer_length=None, frame_index=None)
        if prediction is not None:
            raw = dict(fall_probability=prediction.fall_probability,
                       predicted_label=prediction.predicted_label,
                       buffer_length=prediction.buffer_length,
                       frame_index=prediction.frame_index)
        with self._lock:
            status = application.application_status
            if self._last_event_status is not None and status != self._last_event_status:
                self._record_application_transition(self._last_event_status, status)
            self._last_event_status = status
            if (prediction is not None
                    and (self._last_history_frame is None or prediction.frame_index > self._last_history_frame)):
                self._probability_history.append(dict(frame_index=prediction.frame_index,
                                                     fall_probability=prediction.fall_probability))
                self._last_history_frame = prediction.frame_index
            self._value.update(
                frame_index=frame_index, pose_detected=pose_detected,
                application_status=application.application_status,
                prediction_actionable=application.prediction_actionable,
                consecutive_missing_frames=application.consecutive_missing_frames,
                recovery_frame_index=application.recovery_frame_index,
                raw_prediction=raw,
            )
            self._prediction_received_at = prediction_received_at

    def _record_application_transition(self, previous: str, current: str) -> None:
        """Called under the state lock; describes copied states, never decides them."""
        descriptions = {
            "FALL": ("fall", "FALL", "Fresh cloud prediction restored FALL state" if previous == "RECOVERING"
                     else "Application entered FALL"),
            "POSE_LOST": ("pose_lost", "POSE LOST", "Pose unavailable; prediction suppressed"),
            "RECOVERING": ("recovering", "RECOVERING", "Pose restored; waiting for fresh cloud prediction"),
            "NORMAL": ("recovered" if previous in ("POSE_LOST", "RECOVERING") else "normal",
                       "RECOVERED" if previous in ("POSE_LOST", "RECOVERING") else "NORMAL",
                       "Application returned to NORMAL"),
        }
        if current in descriptions:
            kind, label, message = descriptions[current]
            self._recent_events.append(dict(type=kind, label=label, message=message,
                                           observed_at_ms=int(self._wall_clock() * 1000)))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            value = dict(self._value, raw_prediction=dict(self._value["raw_prediction"]))
            value["probability_history"] = [dict(point) for point in self._probability_history]
            value["recent_events"] = [dict(event) for event in reversed(self._recent_events)]
            received_at = self._prediction_received_at
        age = None if received_at is None else max(0.0, self._clock() - received_at)
        value.update(
            mqtt_connected=self._connected.is_set(),
            cloud_prediction_status=("WARMING_UP" if age is None else
                                     "STALE" if age >= PREDICTION_STALE_SECONDS else "ACTIVE"),
            prediction_age_seconds=age,
            prediction_stale_after_seconds=PREDICTION_STALE_SECONDS,
        )
        return value


def create_app(state: DashboardState) -> Flask:
    app = Flask(__name__)
    app.config["DEBUG"] = False

    @app.get("/")
    def index():
        return render_template("dashboard.html")

    @app.get("/api/state")
    def get_state():
        response = jsonify(state.snapshot())
        response.headers["Cache-Control"] = "no-store"
        return response

    return app


class DashboardServer:
    """Embedded WSGI server: no Flask CLI, reloader, extra process or camera."""

    def __init__(self, state: DashboardState, port: int = DEFAULT_PORT) -> None:
        self.app = create_app(state)
        self.port = port
        self._server: BaseWSGIServer | None = None
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("Dashboard already started")
        self._server = make_server("127.0.0.1", self.port, self.app, threaded=True)
        self.port = self._server.server_port
        self._thread = Thread(target=self._server.serve_forever,
                              kwargs={"poll_interval": .1}, name="dashboard-http", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self._server.server_close()
            self._server = None
            raise

    def close(self) -> None:
        if self._server is not None:
            try:
                if self._thread is not None and self._thread.is_alive():
                    self._server.shutdown()
                    self._thread.join(timeout=2)
            finally:
                self._server.server_close()
                self._server = None
