# Stage 6 — edge pose to MQTT rolling GRU inference

Local engineering gate: **PASS**. Starts from Stage 5
`e7715cfb86cede7abf02217b319f379a5bc121fe`. The selected Stage 5 checkpoint is
`ml/checkpoints/stage5_no_normalization_best.pt`; the backend checks its exact
SHA256 against the committed final ML config and fails if missing or substituted.
No model retraining, threshold tuning, dashboard or alert logic is implemented.

## Commands and configuration

Run from repository root. Export the values in `.env.example` into each process;
Python does not automatically read `.env`. Use MQTT_HOST/PORT/USERNAME/PASSWORD,
MQTT_TOPIC_PREFIX, SOURCE_ID, CHECKPOINT_PATH and POSE_MODEL_PATH. Credentials are
optional for a loopback smoke broker and required by the supplied VPS broker config.
The frozen preprocessing/model/threshold come from Stage 5 final_ml_config.json.

```bash
# Separate backend process (dependencies pinned in the PEP 723 script).
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache uv run backend/app/main.py
# Separate edge process; AVI is paced at source FPS unless --unpaced is given.
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache uv run edge/main.py --source 'data/raw/caucafall_v5/CAUCAFall/Subject.1/Fall backwards/FallBackwardsS1.avi'
# Live camera is also supported; this hardware path has not been tested here.
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache uv run edge/main.py --source 0
```

AVI access reuses the existing inventory-based source guard before OpenCV opens
media. Currently file input is restricted to canonical train/validation CAUCAFall
AVIs; Subject.6/7 and redirected/symlink paths are refused. Frame indices retain
source decode order. All frames are processed without resampling. A count mismatch
or pose extraction/runtime error fails clearly instead of becoming missing pose.
OpenCV read termination alone cannot distinguish EOF from decoder failure; AVI
count verification catches early/extra termination. Missing MediaPipe pose is
encoded using existing pose_raw_v1 semantics.

Streaming and offline preprocessing now share `CausalPoseFill`. Spatial feature
conversion calls the existing transform. The selected configuration remains
normalize_pose=False, max_forward_fill_frames=5, x/y/z/visibility, 132 float32
features. Leading missing frames and gaps beyond five frames are zero. No future
frame is used. A new sequence creates fresh tracking and fill state.

## MQTT protocol

Paho 2.1.0 uses VERSION2 callbacks and MQTT 3.1.1. All publications use QoS1,
retain=False. Broker transport is TCP; no image/video bytes are sent.

- `fall/pose/<source_id>`: control and feature messages.
- `fall/prediction/<source_id>`: inference results.
- `fall/status/backend`: readiness message, after subscription acknowledgment.

The prefix is configurable. Source/sequence IDs allow letters, digits, `_`, `-`,
`.` (1–80 characters). A producer must send `sequence_start` before frames and
`sequence_end` after an AVI. Controls contain schema_version=1, message_type,
source_id, sequence_id. Each restart uses a new sequence_id (UUID by default).

```json
{"schema_version":1,"message_type":"frame","source_id":"edge-01","sequence_id":"demo-001","frame_index":123,"timestamp_ms":6150,"timestamp_kind":"clip_relative","fps":20,"pose_detected":true,"features":["132 finite numeric values"]}
```

`features` above is schematic; the wire message contains exactly 132 numeric
values in MediaPipe order, flattened x/y/z/visibility. pose_detected refers to
the current raw observation; a false value can coexist with a causally filled
feature vector. Timestamp units are milliseconds. AVI uses the existing
`round(frame_index * 1000 / source_fps)` convention and `clip_relative`; camera
uses Unix epoch milliseconds and `unix_epoch`, clamped to strictly increase.
MediaPipe camera tracking uses a separate monotonic relative timestamp. Camera
FPS is declared by CAMERA_FPS; achieved throughput may differ.

```json
{"schema_version":1,"source_id":"edge-01","sequence_id":"demo-001","frame_index":85,"timestamp_ms":4250,"timestamp_kind":"clip_relative","buffer_length":86,"fall_probability":0.975236,"threshold":0.5,"predicted_label":"fall"}
```

The backend rejects invalid schemas/types, topic/source mismatches, nonfinite or
wrong-length features and oversized payloads without terminating its loop.
Network callbacks enqueue messages; the main thread performs CPU inference and
publishes results, avoiding a publish-wait inside the network callback.

## Buffer policy and failures

Each active source has independent sequence state, capped at 64 active sources.
Before 86 consecutive frames, the backend only collects. It infers at frame 86
(zero-based index 85) and every 10 subsequent new frames, keeping at most the latest
200. Inference receives the true current length without synthetic padding.
At 20 FPS this is 4.3 s minimum, 10 s maximum, 0.5 s update interval.

Same-sequence start redelivery is idempotent; a new sequence replaces old state.
Sequence end releases state. Duplicate/late frames and wrong sequence IDs are
ignored. A frame-index gap resets rolling context and requires 86 new consecutive
frames. FPS/timestamp convention cannot change mid-sequence. Backend reconnect
clears context; restart edge with a new sequence after a connection failure.
The edge fails if publication cannot be acknowledged, rather than reporting a
successful stream. Queue overflow is counted and logged; resulting frame gaps
reset history. Camera-offline/pose-runtime failure stops the edge with an error.

**Known mismatch:** training uses full labeled clips; runtime uses rolling context.
Stage 6 introduces no new labels or training to solve that mismatch. These are
classification probabilities, not confirmed fall alerts. Camera operation, long
network outages and deployed cloud behavior remain unverified locally.

## Executed smoke evidence

`artifacts/integration/stage6/smoke.json`: real Mosquitto 2.0.22 broker, separate
OpenCV/MediaPipe edge and PyTorch backend processes, loopback TCP. Backend uses
Python 3.12.14 / NumPy 2.3.5 / torch 2.8.0 / Paho 2.1.0; edge uses Python 3.12.14 /
NumPy 2.2.6 / MediaPipe 0.10.35 / OpenCV-contrib 4.12.0.88 / Paho 2.1.0. Existing
Full float16 PoseLandmarker model and its frozen hash are reused.

Authorized Subject.1/Fall backwards AVI: 125 decoded/published/received frames,
125 pose detections; predictions at indices 85/95/105/115, first probability
0.9752359390, true context 86. Four prediction messages were observed through
MQTT. No unexpected edge/backend errors, buffer gaps or queue overflow occurred.
One intentionally malformed payload was rejected, followed by successful inference.
AVI and selected checkpoint hashes match before/after. Subjects 6/7 were untouched.
No inference accuracy/tuning decision was made from this smoke clip.

The first attempt failed before AVI access because the harness resolved a venv
interpreter symlink to the base Python. The harness now preserves the venv path;
the corrected run passed. Failed-attempt logs remain local under
`/private/tmp/fall-stage6-failed-environment/`; ordinary logs are Git-ignored.

84 focused/related tests passed, including streaming/offline equality on synthetic
gaps and one authorized raw NPZ, schema round-trip/rejection, source isolation,
minimum/stride/maximum context, source guards, checkpoint identity and actual
finite GRU inference. No camera or remote VPS test is claimed.

```bash
python -m unittest tests.test_stage6_runtime tests.test_stage5_evaluation tests.test_stage4_gru tests.test_stage33_pose_sequence tests.test_stage3_contract tests.test_dataset_split tests.test_pose_raw -q
# Run from a backend venv with torch/NumPy/Paho; edge-python must keep its venv path.
python scripts/stage6_smoke.py --broker /path/to/mosquitto --edge-python /path/to/edge-venv/bin/python --output artifacts/integration/stage6_recheck
```

The harness binds an ephemeral loopback port, runs/cleans up three OS processes,
observes actual MQTT predictions and saves a compact summary. The supplied broker
path must be Mosquitto 2.0.22 for the recorded version label; local broker was built
in /private/tmp with no TLS to avoid modifying system services.

## Ubuntu VPS deployment — pending infrastructure access

The selected deployment approach is **Mosquitto package + Python venv/systemd**.
No remote credentials/server were available, so the following configuration exists
but remote deployment has not been executed or verified. Use Ubuntu 24.04 LTS;
record the chosen provider, VPS CPU architecture/address and actual package versions
when deployment occurs. A CPU PyTorch wheel is sufficient (x86_64 or supported ARM64).

On the VPS, place the repository at `/opt/fall-detection` and separately copy the
Git-ignored selected checkpoint into `ml/checkpoints/`. Verify SHA256 is
`3c1fbf4be59f705f003c06f0e1b5698dec5c3d81e19c3345ab35231a237132f6`.
The cloud does not need AVI/raw pose data or the MediaPipe model.

```bash
sudo apt-get update
sudo apt-get install -y mosquitto mosquitto-clients python3-venv
sudo useradd --system --home-dir /opt/fall-detection --shell /usr/sbin/nologin fall
sudo chown -R fall:fall /opt/fall-detection
sudo -u fall python3 -m venv /opt/fall-detection/.venv
sudo -u fall /opt/fall-detection/.venv/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
sudo -u fall /opt/fall-detection/.venv/bin/pip install numpy==2.3.5 paho-mqtt==2.1.0
sudo mosquitto_passwd -c /etc/mosquitto/fall.passwd fall-demo
sudo chown root:mosquitto /etc/mosquitto/fall.passwd
sudo chmod 640 /etc/mosquitto/fall.passwd
sudo install -m 644 /opt/fall-detection/infrastructure/mosquitto/fall.conf /etc/mosquitto/conf.d/fall.conf
sudo install -m 644 /opt/fall-detection/infrastructure/systemd/fall-backend.service /etc/systemd/system/fall-backend.service
```

Create `/etc/fall-detection.env` as root, mode 600, using the .env.example fields.
Backend MQTT_HOST=127.0.0.1, port 1883, username fall-demo, and the password entered
above. CHECKPOINT_PATH is `/opt/fall-detection/ml/checkpoints/stage5_no_normalization_best.pt`.
Keep credentials out of Git and command-line arguments. The service runs as `fall`.

```bash
sudo systemctl daemon-reload
sudo systemctl restart mosquitto
sudo systemctl enable --now fall-backend
sudo journalctl -u fall-backend -n 30 --no-pager
```

Allow TCP 1883 in the VPS firewall/security group **from the edge IP only**; preserve
SSH access. Edge MQTT_HOST becomes the real VPS address and uses the same minimal
credentials. Start the edge and check backend journal plus prediction-topic receipt.
Only the Mosquitto port is exposed; there is no REST/WebSocket service in this stage.
Transport is plaintext for the course MVP; TLS/production authentication are deferred.
Cloud completion requires a real remote run and cannot be inferred from local PASS.

Protocol references: [Paho client API](https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html),
[Mosquitto password authentication](https://www.mosquitto.org/documentation/authentication-methods/).
