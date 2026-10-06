# Stage 8.1 / 8.2 / 8.3 local Dashboard

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
- `probability_history`: at most 60 pairs of `frame_index` / `fall_probability`,
  stored in a locked in-memory deque in `DashboardState`. Only a strictly newer
  prediction frame index observed by the Edge state bridge appends a point. Equal
  indices, old indices, repeated Edge copies and HTTP polling never append points.
  Browser refresh retains history; a new Edge runtime creates an empty history.

## Layout and trend

The dark desktop page separates the authoritative application decision from the
raw cloud model output. NORMAL is green, FALL has a strong red panel and visible
FALL DETECTED text, POSE_LOST/RECOVERING are amber, and INITIALIZING is neutral.
Actionable is YES/NO. When raw output exists but actionable is false, a suppression
note explains the pose-validity gate while retaining the raw label and probability.

The Canvas chart uses only server-owned history, with frame index on X and 0–100%
probability on Y. The dashed 50% reference line is display-only; neither the chart
nor the page derives labels or application decisions from this line. Classification
still uses the existing Backend label. These are cloud inference samples, not
camera frames or samples generated at browser polling frequency. There is no fake
0% point before the first prediction, and no interpolation that adds stored points.
The history follows the existing latest-prediction bridge: it does not recover
predictions overwritten between Edge updates, or fill in missing samples.

## Recent Events and FALL banner

`recent_events` in `/api/state` is a newest-first copy of a locked in-memory deque
of at most 20 events; the page renders the latest 8. Only changed application states
observed during the Edge's existing `DashboardState.update()` create events. The
first update establishes the baseline without an event; repeated states, browser
refreshes, API reads and freshness changes do not create events. Initializing
transitions are not recorded. NORMAL does not clear earlier FALL events.

Events describe entry into FALL, POSE_LOST, RECOVERING and NORMAL. Returning to
NORMAL from POSE_LOST/RECOVERING is labelled RECOVERED. RECOVERING -> FALL is a FALL
event with "Fresh cloud prediction restored FALL state", never RECOVERED.
`observed_at_ms` is Dashboard wall-clock observation time in Unix milliseconds,
not camera or cloud inference time; the browser formats it as local HH:MM:SS.
Events clear on Edge restart and are never written to files or browser storage.

The red FALL DETECTED strip appears only when the copied `application_status` is
FALL. Raw FALL or probability >=0.5 alone cannot trigger it. POSE_LOST/RECOVERING
retain amber state cards, suppression text and visible raw output. No audio,
notifications, modal, acknowledgement workflow or extra fall decision is added.
MQTT status is read live at snapshot time, not copied by each Edge update; this
Gate adds no MQTT events or observer, and no ACTIVE/STALE events.

## Limits

Snapshots of application/raw state update after each Edge gate step. HTTP cannot
make a blocked camera or publish loop progress. Transport and prediction age remain
separate from the copied application values; STALE does not change actionable.
If HTTP becomes unavailable, the page dims and labels its values as the last snapshot.
The existing MQTT reconnect/sequence restart policy is unchanged. Backend readiness
is not a heartbeat, and this Dashboard makes no claim about systemd liveness.

The server is for a single local course demo, not public hosting. No persistent
event history, database, browser MQTT or CDN are included. Application FALL is the existing
gate output; this change adds no consecutive-prediction confirmation logic.
