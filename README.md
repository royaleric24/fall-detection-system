# Fall Detection System

A real-time edge-cloud fall detection system for **COMP6131 – Internet of Things Essentials**.

The current course V1 uses a camera-only pipeline: RGB frames are processed locally with OpenCV and MediaPipe, only pose features are transmitted over MQTT, a real Ubuntu VPS performs rolling GRU inference, and the Mac edge application applies a pose-validity safety gate before presenting the result in a local web dashboard.

## 1. Current V1 Status

The implemented end-to-end path is:

```text
Mac camera / authorized AVI
        ↓
OpenCV
        ↓
MediaPipe Pose (33 landmarks)
        ↓
Causal pose preprocessing
        ↓
132-D pose feature vector
        ↓
MQTT over the Internet
        ↓
Ubuntu VPS
  Mosquitto broker
        ↓
Rolling GRU backend
        ↓
MQTT prediction
        ↓
Mac Edge application
        ↓
PoseValidityGate
        ↓
DashboardState
        ↓
Local Flask dashboard
        ↓
Browser
```

Current V1 achievements:

- [x] Real webcam input on macOS
- [x] MediaPipe Pose extraction
- [x] Causal missing-pose preprocessing
- [x] Subject-independent CAUCAFall train/validation split
- [x] GRU temporal classifier
- [x] Controlled normalization ablation
- [x] MQTT edge-cloud transport
- [x] Real Ubuntu VPS deployment
- [x] Mosquitto authentication
- [x] systemd-managed cloud inference backend
- [x] Real webcam → Internet MQTT → VPS GRU → MQTT prediction
- [x] Application-level pose-validity false-alarm suppression
- [x] Local real-time Flask dashboard
- [x] Probability trend, health status, recent events and FALL alert UX
- [x] Real prerecorded FALL positive-control E2E validation

The following originally planned components are **not part of the current V1 implementation**:

- FastAPI REST backend
- WebSocket transport
- PostgreSQL
- React / Node frontend
- persistent event storage
- IMU / vision-IMU fusion
- public web hosting

They were intentionally deferred because they are not required for the working course-demo path.

## 2. Implemented Architecture

### Edge: Mac

`edge/main.py` owns the real-time edge loop:

1. Capture webcam frames or an authorized CAUCAFall AVI.
2. Run MediaPipe Pose.
3. Convert each frame into 33 landmarks × `x/y/z/visibility` = **132 features**.
4. Apply the frozen causal missing-pose preprocessing.
5. Publish the feature frame through MQTT.
6. Receive cloud predictions.
7. Apply the authoritative `PoseValidityGate`.
8. Optionally render the OpenCV preview and local dashboard.

Raw video is **not uploaded to the VPS**.

### Cloud: Ubuntu VPS

The real VPS runs:

```text
Mosquitto
+
backend/app/main.py
+
backend/app/inference.py
```

The backend:

- subscribes to pose telemetry;
- maintains independent per-source rolling buffers;
- runs the frozen GRU on CPU;
- publishes fall probability and raw label through MQTT.

The current backend is a long-running MQTT inference process, **not** a FastAPI/Uvicorn service.

### Dashboard: Mac localhost

The dashboard runs in the same Edge process as a background Flask service:

```text
http://127.0.0.1:8767/
```

It reads a locked snapshot of the already-computed application state. It does **not** create a second fall-decision state machine.

The browser uses HTML/CSS/vanilla JavaScript and polls `GET /api/state`. No browser MQTT credentials, CDN, WebSocket, database or external frontend framework are required.

See [Dashboard runtime and semantics](docs/stage8_dashboard.md).

## 3. Frozen ML Configuration

The selected model is:

```text
Model:                  GRU
Input size:             132
Hidden size:            64
Layers:                 1
Bidirectional:          false
Classification threshold: 0.5
```

Frozen preprocessing:

```text
normalize_pose:              false
max_forward_fill_frames:     5
causal_missing_pose_handling: true
feature_order:               x, y, z, visibility
landmarks:                   33
```

Selected checkpoint:

```text
ml/checkpoints/stage5_no_normalization_best.pt
SHA-256:
3c1fbf4be59f705f003c06f0e1b5698dec5c3d81e19c3345ab35231a237132f6
```

The backend verifies the checkpoint and frozen preprocessing/model metadata at startup.

### Validation-only model selection result

On the fixed 20-clip Validation split (Subjects 5 and 10):

| Variant | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| No normalization — selected | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Normalized Stage 4 baseline | 0.9000 | 1.0000 | 0.8000 | 0.8889 |

This is a small Validation-set course-project result, **not** a statistical generalization claim. Subjects 6 and 7 remain the sealed held-out Test split in the frozen configuration.

See [Stage 5 report](artifacts/evaluation/stage5/REPORT.md) and [final ML configuration](artifacts/evaluation/stage5/final_ml_config.json).

## 4. Online Inference Policy

The cloud backend uses a rolling sequence buffer:

```text
Minimum context: 86 consecutive frames
Inference stride: 10 new frames
Maximum context: 200 frames
```

At the nominal camera declaration of 20 FPS:

- first prediction is available after 86 frames (about 4.3 s);
- new predictions are produced every 10 new frames (about 0.5 s);
- the latest 200 feature frames are retained.

A frame-index gap resets the rolling context. Duplicate or late messages do not enter the inference history.

Training uses full labeled clips while runtime uses rolling context; this remains a known system limitation.

## 5. MQTT Protocol

Default topic prefix:

```text
fall
```

Topics:

```text
fall/pose/<source_id>
fall/prediction/<source_id>
fall/status/backend
```

All current publications use QoS 1 and `retain=False`.

### Pose frame

```json
{
  "schema_version": 1,
  "message_type": "frame",
  "source_id": "edge-01",
  "sequence_id": "demo-001",
  "frame_index": 123,
  "timestamp_ms": 6150,
  "timestamp_kind": "clip_relative",
  "fps": 20.0,
  "pose_detected": true,
  "features": ["132 finite numeric values"]
}
```

For webcam input, `timestamp_kind` is `unix_epoch`. `pose_detected` refers to the raw observation; short missing runs can still have causally forward-filled features.

### Cloud prediction

```json
{
  "schema_version": 1,
  "source_id": "edge-01",
  "sequence_id": "demo-001",
  "frame_index": 85,
  "timestamp_ms": 4250,
  "timestamp_kind": "clip_relative",
  "buffer_length": 86,
  "fall_probability": 0.975236,
  "threshold": 0.5,
  "predicted_label": "fall"
}
```

`buffer_length` is the context length used for that prediction, not a live query of the current server buffer.

## 6. Application-Level Pose Validity Gate

The cloud model produces a **raw** fall probability and raw label. The Mac edge application then applies a pose-validity gate without modifying the model output.

Application states:

```text
INITIALIZING
NORMAL
FALL
POSE_LOST
RECOVERING
```

Rules:

- Missing pose for frames 1–5 uses the existing causal preprocessing policy.
- When consecutive missing pose exceeds 5 frames:
  - application state becomes `POSE_LOST`;
  - `prediction_actionable = false`.
- When pose returns after prolonged loss:
  - state becomes `RECOVERING`;
  - the application waits for a fresh cloud prediction whose frame index covers the recovery frame.
- After that fresh prediction:
  - the application returns to `NORMAL` or `FALL` according to the raw cloud label;
  - `prediction_actionable = true`.

A raw cloud `FALL` remains visible during `POSE_LOST` or `RECOVERING`, but it does not trigger the application FALL alert while the prediction is suppressed.

This behavior is designed to prevent the known prolonged-pose-loss false alarm from being presented as an actionable fall.

## 7. Dashboard

The local dashboard shows:

- authoritative application status;
- raw cloud fall probability;
- raw cloud label;
- actionable YES/NO;
- current pose status;
- MQTT transport status;
- cloud prediction freshness (`WARMING_UP`, `ACTIVE`, `STALE`);
- latest inference context and prediction frame;
- up to 60 real cloud probability samples;
- up to 20 in-memory application transition events;
- a prominent FALL banner only when `application_status == "FALL"`.

Probability history is de-duplicated by prediction `frame_index`; browser polling does not create fake model samples.

Recent Events are generated from real application-state transitions, not from HTTP polling. They are in-memory only and clear when the Edge process restarts.

## 8. Quick Start

### Requirements

- Python 3.12
- `uv`
- MediaPipe PoseLandmarker model at the configured `POSE_MODEL_PATH`
- selected GRU checkpoint at the configured `CHECKPOINT_PATH`
- access to the deployed MQTT broker/backend

The Edge and Backend scripts use PEP 723 dependency declarations. Python does not automatically load `.env`.

### Configure the current Terminal

From the repository root:

```bash
export MQTT_HOST=<vps-host>
export MQTT_PORT=1883
export MQTT_USERNAME=<mqtt-username>
export MQTT_TOPIC_PREFIX=fall
export POSE_MODEL_PATH="$PWD/ml/checkpoints/pose_landmarker_full.task"

read -rs 'MQTT_PASSWORD?MQTT password: '
echo
export MQTT_PASSWORD
```

Do not commit credentials. The password exists only in the current shell environment and child processes unless you deliberately persist it elsewhere.

### Check the VPS

Using your configured SSH target:

```bash
ssh <vps-alias> 'printf "Mosquitto: "; systemctl is-active mosquitto; printf "Backend: "; systemctl is-active fall-backend'
```

Expected:

```text
Mosquitto: active
Backend: active
```

To watch real backend inference:

```bash
ssh <vps-alias>
sudo journalctl -u fall-backend -f
```

Pressing `Ctrl+C` stops log following; it does not stop the backend service.

### Run the real webcam demo

```bash
uv run --python 3.12 --script edge/main.py \
  --source 0 \
  --source-id webcam-demo \
  --preview \
  --dashboard
```

Open:

```text
http://127.0.0.1:8767/
```

The initial state may be `INITIALIZING / WARMING_UP` until the rolling buffer reaches 86 consecutive frames.

### Run the prerecorded FALL positive control

If the local CAUCAFall V5 data is available:

```bash
uv run --python 3.12 --script edge/main.py \
  --source "data/raw/caucafall_v5/CAUCAFall/Subject.1/Fall backwards/FallBackwardsS1.avi" \
  --source-id fall-control \
  --preview \
  --dashboard
```

This uses a Train subject, not the sealed held-out Test subjects.

Do not ask a person to physically fall for the demo.

## 9. Real E2E Validation

The final demo path was validated with a real Mac webcam, public MQTT transport, the deployed VPS GRU backend and a real browser dashboard.

### Webcam run

Observed in the real browser:

- MQTT transport connected;
- cloud prediction active;
- real probability-history samples;
- `POSE_LOST → RECOVERING → RECOVERED`;
- normal operation after recovery.

The VPS journal independently confirmed real predictions for the webcam source.

### Prerecorded FALL positive control

For:

```text
Subject.1/Fall backwards/FallBackwardsS1.avi
```

the real VPS produced:

| Frame | Context | P(fall) |
| ---: | ---: | ---: |
| 85 | 86 | 0.975236 |
| 95 | 96 | 0.979456 |
| 105 | 106 | 0.981400 |
| 115 | 116 | 0.982003 |

The real browser showed:

```text
Raw label: FALL
Raw probability: 98.2%
Application status: FALL
Prediction actionable: YES
Pose: DETECTED
FALL DETECTED banner: visible
```

The focused Stage 8 dashboard/edge/runtime regression suite passed:

```text
54 passed, 0 failures, 0 errors
```

## 10. Dataset and Split

The project uses CAUCAFall V5.

The fixed subject-independent split is:

```text
Train:      1, 2, 3, 4, 8, 9
Validation: 5, 10
Test:       6, 7
```

The held-out Test subjects are protected from tuning by the dataset protocol.

See:

- [Dataset protocol](docs/dataset_protocol.md)
- [Split configuration](configs/dataset_split.json)
- [Dataset inspection](artifacts/dataset_inspection/summary.md)

## 11. Repository Structure

```text
fall-detection-system/
├── edge/
│   └── main.py                  # camera/AVI, MediaPipe, MQTT, gate, dashboard bridge
├── common/
│   └── mqtt.py                  # shared MQTT protocol and connection wrapper
├── backend/
│   └── app/
│       ├── main.py              # long-running MQTT backend
│       └── inference.py         # rolling GRU inference
├── frontend/
│   ├── dashboard.py             # DashboardState + local Flask server
│   ├── templates/
│   │   └── dashboard.html
│   └── static/
│       ├── dashboard.css
│       └── dashboard.js
├── ml/
│   ├── datasets/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── checkpoints/
├── infrastructure/
│   ├── mosquitto/
│   └── systemd/
├── artifacts/
├── configs/
├── docs/
├── tests/
├── scripts/
├── AGENTS.md
├── README.md
└── .env.example
```

## 12. Current Deployment

The current course-demo deployment uses:

```text
Mac
  Edge + MediaPipe + application gate + local Dashboard

Internet MQTT

Ubuntu VPS
  Mosquitto
  Python virtual environment
  systemd fall-backend service
  CPU GRU inference
```

The VPS does not host the Dashboard. The browser connects only to the Mac loopback Flask service.

MQTT credentials are required by the deployed broker. Credentials must remain outside Git.

The course MVP currently uses TCP MQTT without TLS. Production-grade TLS, public web hosting, stronger authentication and hardened network policy are outside the V1 scope.

## 13. Known Limitations

- One camera and one monitored person are the target scenario.
- Multi-person tracking is not implemented.
- Raw cloud predictions can still be wrong; the pose-validity gate only addresses the demonstrated prolonged-pose-loss failure mode.
- Training uses full clips while deployed inference uses rolling context.
- Dashboard events are in-memory and are not persisted.
- The Dashboard is localhost-only and has no production authentication.
- The deployed MQTT transport is not a production security design.
- No FastAPI, WebSocket, PostgreSQL or React layer is used in the current V1.
- No IMU or multimodal fusion is implemented.
- Subjects 6 and 7 remain reserved for final held-out evaluation in the frozen ML configuration.
- The system is a course prototype, not a medical device and not intended for medical diagnosis.

## 14. Deferred Work

Potential extensions include:

- held-out final Test evaluation when the evaluation plan authorizes it;
- broader real-camera robustness testing;
- latency/FPS measurement;
- TLS-secured MQTT;
- persistent event storage;
- public authenticated dashboard/API;
- multi-person handling;
- IMU integration and vision-IMU fusion.

For the current course V1, priority is a complete, reproducible and demonstrable edge-cloud system rather than additional infrastructure.
