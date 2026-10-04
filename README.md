# Fall Detection System

A real-time edge-cloud fall detection system based on temporal human-pose modeling, robust multi-stage alert handling, and optional vision-IMU fusion.

## 1. Project Goal

The project aims to build a complete real-time fall detection pipeline rather than an isolated offline classifier.

The core system must support:

- real-time camera input;
- edge-side human-pose extraction and preprocessing;
- skeleton time-series transmission through MQTT;
- cloud-side temporal inference;
- multi-stage false-alarm suppression;
- persistent event storage;
- REST API and WebSocket services;
- a real-time web dashboard;
- latency, ablation, and robustness evaluation.

The first production-capable version must work with **camera-only input**. IMU sensing is an optional enhancement and must not be a hard dependency of the core system.

## 2. Frozen V1 Architecture

```text
Camera
  ↓
OpenCV
  ↓
MediaPipe Pose
  ↓
Skeleton Normalization
  ↓
MQTT
  ↓
Cloud Sequence Buffer
  ↓
Temporal Model (GRU initial baseline; TCN primary candidate)
  ↓
Fall Probability
  ↓
Robust Alert State Machine
  ↓
PostgreSQL
  ↓
FastAPI REST + WebSocket
  ↓
React Dashboard
```

### Edge responsibilities

- capture camera frames;
- run pose estimation;
- normalize skeleton coordinates;
- attach timestamps and device metadata;
- publish telemetry through MQTT;
- maintain heartbeat/device status;
- later: provide a lightweight local fallback rule.

### Cloud responsibilities

- receive MQTT telemetry;
- maintain per-device temporal buffers;
- run temporal fall inference;
- execute the event/alert state machine;
- store devices, telemetry summaries, and fall events;
- expose REST and WebSocket interfaces;
- provide the backend used by the dashboard.

### Web responsibilities

- device/system status;
- current fall probability;
- current event state;
- live pose/camera preview where enabled;
- latency/FPS metrics;
- event history and acknowledgement.

## 3. Scope

### Must-have

- [ ] Real-time camera input
- [ ] MediaPipe-based pose extraction
- [ ] Skeleton normalization
- [ ] Skeleton temporal modeling
- [ ] MQTT edge-cloud transport
- [ ] Cloud inference
- [ ] FastAPI backend
- [ ] PostgreSQL database
- [ ] WebSocket real-time updates
- [ ] React dashboard
- [ ] Multi-stage alert logic
- [ ] Precision / Recall / F1 evaluation
- [ ] Detection-latency evaluation
- [ ] False-alarm evaluation
- [ ] Ablation study
- [ ] Robustness testing

### Enhancement

- [ ] IMU input: accelerometer + gyroscope
- [ ] 1D CNN IMU encoder
- [ ] Vision-IMU feature fusion
- [ ] Modality dropout
- [ ] Sensor-missing fallback
- [ ] WebRTC full live-video streaming

### Explicitly out of scope for V1

- multiple-camera tracking;
- multi-person fall reasoning;
- face recognition;
- mobile application;
- Kubernetes / Kafka;
- GPU cloud requirement;
- continuous raw 1080p video upload;
- medical diagnosis or medical-grade claims.

The V1 target scenario is **one camera, one monitored person, indoor environment**.

## 4. Main Data Flow

At approximately 15 FPS, the edge device converts each RGB frame into 33 human pose landmarks. Coordinates are normalized to reduce dependence on image resolution, subject position, and subject-camera distance.

Example MQTT topic:

```text
fall/edge-001/pose
```

Example payload shape:

```json
{
  "device_id": "edge-001",
  "timestamp": 0,
  "frame_id": 0,
  "pose_detected": true,
  "pose_confidence": 0.0,
  "landmarks": []
}
```

The cloud buffers consecutive poses into a temporal window. The initial target is:

```text
15 FPS × 2 s = 30 frames
```

A temporal model receives approximately:

```text
[time, joints, features]
= [30, 33, F]
```

and outputs a fall probability.

## 5. Alert State Machine

The classifier must not trigger an alarm directly from a single threshold crossing.

```text
NORMAL
  ↓
SUSPECTED
  ↓
CONFIRMED
  ↓
ALERT
  ↓
ACKNOWLEDGED / RECOVERED
```

The final decision may combine:

- model probability;
- temporal persistence;
- body vertical velocity;
- body orientation;
- pose confidence;
- post-fall inactivity.

Thresholds and timing parameters must be configurable and evaluated rather than hard-coded without evidence.

## 6. Evaluation

### Classification

- Precision
- Recall
- F1 score
- Confusion matrix

### System performance

- edge preprocessing latency;
- network latency;
- cloud inference latency;
- end-to-end pipeline latency;
- effective FPS;
- alert-confirmation delay reported separately.

### Robustness

The V1 evaluation should include controlled degradation of:

- lighting;
- occlusion;
- camera FPS;
- network delay;
- packet loss;
- missing/unreliable pose frames.

### Ablation

Minimum planned comparison:

1. temporal model only;
2. + temporal smoothing;
3. + physical/motion features;
4. + complete alert state machine.

If IMU is implemented, add:

- vision only;
- IMU only;
- vision + IMU;
- missing-modality tests.

## 7. Repository Structure

```text
fall-detection-system/
├── edge/
│   ├── camera/
│   ├── pose/
│   ├── preprocessing/
│   ├── mqtt/
│   └── main.py                 # created when Stage 2/3 starts
├── ml/
│   ├── datasets/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── checkpoints/
├── backend/
│   └── app/
│       ├── api/
│       ├── mqtt/
│       ├── inference/
│       ├── events/
│       ├── database/
│       └── main.py             # created during backend stage
├── frontend/
├── infrastructure/
│   ├── compose.yaml            # created during cloud stage
│   ├── mosquitto/
│   └── nginx/
├── tests/
├── scripts/
├── configs/
├── docs/
│   └── architecture.md
├── README.md
├── AGENTS.md
├── .gitignore
└── .env.example
```

## 8. Development Stages

```text
Stage 0  System definition and repository
Stage 1  Dataset exploration
Stage 2  Dataset → pose extraction
Stage 3  Pose preprocessing / visualization
Stage 4  GRU / TCN offline baseline
Stage 5  Webcam → real-time local inference
Stage 6  Edge → MQTT → Cloud
Stage 7  FastAPI + PostgreSQL
Stage 8  Web dashboard
Stage 9  Robust alert handling
Stage 10 Robustness + ablation experiments
Stage 11 Optional IMU integration
Stage 12 Optional multimodal fusion
```

Do not begin cloud deployment before the local camera → pose → temporal model loop has been validated.

## 9. Planned Deployment

Initial deployment target:

```text
Alibaba Cloud Simple Application Server
Hong Kong region
Ubuntu 24.04 LTS
Docker + Docker Compose
```

Expected containers later:

- Mosquitto
- FastAPI backend + inference
- PostgreSQL
- React/Nginx

## 10. Research Questions

- **RQ1:** Can skeleton-based temporal modeling provide reliable real-time fall detection?
- **RQ2:** Can an edge-cloud architecture reduce transferred data while preserving real-time performance?
- **RQ3:** Can multi-stage event handling reduce false alarms compared with direct model-threshold alarms?
- **RQ4:** How do lighting, occlusion, frame-rate degradation, and network impairment affect F1 and latency?
- **RQ5 (optional):** Does vision-IMU fusion improve performance and robustness under missing or degraded modalities?

## 11. Current Status

**Stage 3.3 is complete: minimal clip-level binary supervision and deterministic
pose preprocessing are ready for the Stage 4 GRU baseline.** The verified Stage 2
`pose_raw_v1` run remains immutable. The dataset retains the frozen subject split,
excludes Test 6/7, uses causal forward-fill (up to 5 frames), hip-centered torso-scaled XYZ
and visibility (132 features), and provides a PyTorch Dataset/DataLoader with
full clips and padding/lengths. The normalization switch supports the later ablation. There
are 59 usable Train clips and 20 Validation clips; one all-missing Train clip is
explicitly excluded. No model training or held-out evaluation has started.

The course-project sprint closes further Stage 3.2 temporal-annotation research;
Stage 3.2j is retained at `63793aa50f91d1df3429ac081ae23b7dc6dd9d6c` without
genuine review. Historical Stage 3.0 build guards remain unchanged. See the
[Stage 3.3 usage, feature contract and validation](docs/stage33_minimal_preprocessing.md)
and [executed readiness audit](artifacts/preprocessing/stage33_causal/audit.json).

The local CAUCAFall V5 download passes structural and full sequential media validation:
10 subjects × 10 activities = 100 AVI videos, all 19,877 frames decoded with metadata-count agreement at reported 20 FPS and 720 × 480 resolution. Six PNGs and four frame TXT
annotations have unmatched basenames; all 100 `classes.txt` files are separate
metadata. Raw data is unchanged.

See [inspection usage and annotation exceptions](ml/datasets/README.md),
[generated summary](artifacts/dataset_inspection/summary.md), and
[video inventory](artifacts/dataset_inspection/inventory.csv).

Stage 1.3 pose compatibility: **PASS WITH WARNINGS**, using the official MediaPipe
Tasks API on ten clips. Valid 33-landmark poses were returned for 1,504/1,724
frames (87.24% pose availability, not accuracy or robustness evidence).
Subject.1 / Fall forward remains a key warning: 94/190 valid poses (49.47%),
with a longest missing run of 92 frames (4.60 seconds). Stage 1.4 defines
missing-pose and low-visibility policy in the dataset protocol. See the [compatibility report](artifacts/pose_compatibility/REPORT.md)
for missing-pose intervals, runtime workaround, and required follow-up protocol.

Stage 1.4 — dataset protocol and subject-independent split — is complete.
**Stage 1 has passed Gate Review.** The [dataset protocol](docs/dataset_protocol.md) freezes labels, subject
membership and raw pose/missingness rules; the [split config](configs/dataset_split.json)
and [100-video split manifest](artifacts/dataset_inspection/split_inventory.csv)
record train/validation/test assignments (60/20/20 videos). All five architecture
Stage 1 questions have documented answers, with compatibility warnings retained.

Gate-review correction: the final fixed subject-independent split is train
8/4/3/9/1/2, validation 10/5, test 6/7. Subjects 1–5 were explored in Stage 1.3;
Subjects 6/7 are reserved for final evaluation under the protocol
[held-out test policy](docs/dataset_protocol.md#held-out-test-policy). Seed 42
describes only the original candidate, not the final adjusted assignment.
