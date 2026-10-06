# Stage 8.1 local Dashboard

The existing Edge main thread owns capture, pose extraction, frozen preprocessing,
MQTT publication and the single authoritative `PoseValidityGate`. Immediately after
that gate steps, Edge copies scalar fields into a locked `DashboardState`. An embedded
Flask WSGI server runs in a background thread and serves read-only snapshots. It never
calls the gate or makes fall decisions. GRU inference continues on the existing VPS.

## Launch

From the repository root, with the existing secure MQTT environment already set:

```bash
uv run --python 3.12 --script edge/main.py --source 0 --source-id webcam-demo --preview --dashboard
uv run --python 3.12 --script edge/main.py --source 0 --source-id webcam-demo --dashboard
```

Open `http://127.0.0.1:8767/`. Use `--dashboard-port 8768` if needed. Binding is always
loopback, with debug/reloader disabled. Both modes use the existing prediction topic;
there is no extra MQTT client or backend-ready observer. Q exits with OpenCV preview;
Ctrl+C exits either mode. AVI completion and errors also close the Dashboard server.
Python >=3.12 and pinned `Flask==3.1.2` are declared with the existing PEP 723 dependencies.

## Fields

`GET /` serves local HTML/CSS/vanilla JS; `GET /api/state` returns JSON with `no-store`.
The browser polls 350 ms after each completed request, without overlapping requests.
No credentials or pose feature arrays enter the HTTP snapshot.

- `source_id`, `sequence_id`, `frame_index`: active Edge identity and current published
  frame, zero-based. Frame/pose fields are null before the first processed frame.
- `pose_detected`: current raw pose observation, not whether forward-filled features exist.
- `application_status`, `prediction_actionable`, `consecutive_missing_frames`,
  `recovery_frame_index`: copies from the authoritative gate. States are INITIALIZING,
  NORMAL, FALL, POSE_LOST and RECOVERING. No Dashboard safety logic recomputes them.
- `raw_prediction`: `fall_probability`, `predicted_label`, `buffer_length`, `frame_index`.
  Values are null before the first cloud prediction. Raw fall/non_fall output remains
  visible even when the gate suppresses it. Buffer means latest inference context,
  not a live query of the current server buffer.
- `mqtt_connected`: reads the existing connection Event at snapshot time. It describes
  MQTT transport only, not Backend health.
- `cloud_prediction_status`: WARMING_UP before a prediction; ACTIVE while the last new
  prediction is younger than 5 seconds; STALE at age >=5 seconds.
- `prediction_age_seconds`: monotonic elapsed time since the MQTT callback accepted
  a strictly newer prediction frame index. Equal-index duplicates and late messages
  do not refresh the clock; browser polling and Edge copies do not refresh it.
- `prediction_stale_after_seconds`: the UI-only constant 5.0. This is deliberately
  conservative relative to the nominal inference cadence. It changes no gate, raw
  prediction, threshold, sequence, telemetry or MQTT behaviour.

## Limits

Snapshots of application/raw state update after each Edge gate step. HTTP cannot
make a blocked camera or publish loop progress. Transport and prediction age remain
separate from the copied application values; STALE does not change actionable.
If HTTP becomes unavailable, the page dims and labels its values as the last snapshot.
The existing MQTT reconnect/sequence restart policy is unchanged. Backend readiness
is not a heartbeat, and this Dashboard makes no claim about systemd liveness.

The server is for a single local course demo, not public hosting. No charts, event
history, database, browser MQTT or CDN are included. Application FALL is the existing
gate output; this change adds no consecutive-prediction confirmation logic.
