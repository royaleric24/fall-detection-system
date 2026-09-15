# System Architecture

## 1. Purpose

This document defines the Stage 0 architecture for the Fall Detection System. It is the architectural source of truth until the team explicitly approves a revision.

## 2. Architectural Goals

The system should demonstrate all of the following in one integrated prototype:

1. real-time sensing;
2. temporal modeling;
3. edge-cloud partitioning;
4. real cloud backend functionality;
5. robust false-alarm handling;
6. measurable end-to-end latency;
7. robustness and ablation evaluation;
8. optional multimodal expansion.

## 3. Context Diagram

```text
                          ┌─────────────────────┐
                          │    Web Dashboard    │
                          │                     │
                          │ Device / Event /    │
                          │ Metrics Monitoring  │
                          └──────────▲──────────┘
                                     │
                              HTTPS / WebSocket
                                     │
┌────────────────────────────────────┴─────────────────────────────────┐
│                              CLOUD                                  │
│                                                                     │
│  ┌─────────────┐      ┌──────────────────────────────────────────┐   │
│  │ Mosquitto   │ ───▶ │ FastAPI Backend                          │   │
│  │ MQTT Broker │      │                                          │   │
│  └─────────────┘      │ Device Buffers → TCN/GRU → Event Engine │   │
│                       │ REST API / WebSocket / Device Management │   │
│                       └──────────────────┬───────────────────────┘   │
│                                          │                           │
│                                  ┌───────▼────────┐                  │
│                                  │  PostgreSQL    │                  │
│                                  │ devices/events │                  │
│                                  └────────────────┘                  │
└────────────────────────────────────▲─────────────────────────────────┘
                                     │
                                     │ MQTT
                                     │
┌────────────────────────────────────┴─────────────────────────────────┐
│                               EDGE                                  │
│                                                                     │
│ Camera → OpenCV → MediaPipe Pose → Normalize → Telemetry Publisher │
│                                                                     │
│ Optional later: IMU → filter/window/features → MQTT                 │
└─────────────────────────────────────────────────────────────────────┘
```

## 4. Why Skeleton-Based Temporal Modeling

The primary V1 representation is a time sequence of human-pose landmarks, not raw RGB classification.

This choice is intended to:

- reduce sensitivity to background appearance;
- reduce bandwidth between edge and cloud;
- provide a compact time-series representation;
- improve privacy compared with continuous raw-video upload;
- align offline dataset preprocessing with real-time camera preprocessing.

It does **not** eliminate domain shift. Pose estimation can still degrade under low light, unusual camera angles, occlusion, motion blur, and partial-body visibility. These failures are therefore explicit robustness targets rather than assumed solved problems.

## 5. Edge Layer

### 5.1 Responsibilities

The edge process owns:

- camera acquisition;
- frame timestamps;
- pose estimation;
- landmark confidence extraction;
- skeleton normalization;
- telemetry serialization;
- MQTT publishing;
- heartbeat/status reporting.

### 5.2 Initial operating target

Initial target parameters:

```text
Camera FPS:       15 FPS
Temporal window:  2 seconds
Window length:    30 frames
Person count:     1
Environment:      indoor
```

These are initial engineering defaults, not final experimentally optimized values.

### 5.3 Pose representation

Logical raw landmark representation per frame:

```text
[J, 4]
```

where:

- `J = 33` landmarks;
- features initially contain `x, y, z, visibility`.

The exact model feature tensor may later include normalized coordinates, confidence, velocity, orientation, or selected derived features.

### 5.4 Normalization

Normalization should make the representation less dependent on camera resolution and absolute screen location.

The initial design is:

1. compute hip/pelvis center;
2. translate coordinates relative to this center;
3. calculate a body scale such as torso length or shoulder width;
4. divide translated coordinates by the scale;
5. preserve confidence/visibility separately;
6. define behavior when the scale is invalid or critical landmarks are missing.

The exact formula must be unit-tested before dataset-wide preprocessing.

## 6. MQTT Transport

### 6.1 Planned topics

```text
fall/{device_id}/pose
fall/{device_id}/status
fall/{device_id}/heartbeat
fall/{device_id}/imu       # optional later
fall/{device_id}/command   # optional later
```

### 6.2 Pose message

Provisional schema:

```json
{
  "schema_version": 1,
  "device_id": "edge-001",
  "timestamp_ms": 0,
  "frame_id": 0,
  "pose_detected": true,
  "pose_confidence": 0.0,
  "landmarks": []
}
```

The schema will be formalized before Stage 6.

### 6.3 Transport failure cases

The system must later define behavior for:

- duplicate messages;
- out-of-order messages;
- temporary disconnects;
- missing frames;
- stale telemetry;
- backend restart.

## 7. Cloud Backend

### 7.1 Runtime components

Planned single-server deployment:

```text
Ubuntu 24.04 LTS
Docker Compose
├── Mosquitto
├── FastAPI backend + inference
├── PostgreSQL
└── React/Nginx
```

### 7.2 Sequence buffering

The backend maintains a separate temporal buffer for each device.

Initial logical model input:

```text
X ∈ R^[T × J × F]
```

with initial `T = 30`, `J = 33`.

Sliding-window stride must be configurable. It must not be chosen only for model accuracy because stride also changes inference frequency and latency.

### 7.3 Inference

Initial model sequence:

1. establish GRU baseline;
2. establish TCN model;
3. compare F1, latency, complexity, and real-time behavior;
4. select the deployment model from measured results.

TCN is the current primary deployment candidate, not a guaranteed winner.

### 7.4 Event Engine

Model inference outputs a probability or score. It does not directly create an alert.

Provisional state machine:

```text
NORMAL
  │
  │ abnormal evidence
  ▼
SUSPECTED
  │
  │ persistent / corroborated evidence
  ▼
CONFIRMED
  │
  │ post-fall verification
  ▼
ALERT
  │
  ├──▶ ACKNOWLEDGED
  └──▶ RECOVERED
```

Potential evidence includes:

- fall-model probability;
- repeated positive windows;
- rapid downward hip motion;
- large body-orientation change;
- low post-event motion;
- pose-estimation confidence.

Thresholds and dwell times must be configuration values and later evaluated through ablation.

## 8. Database

Initial conceptual entities:

### Device

```text
id
name
status
last_seen
camera_status
imu_status
```

### FallEvent

```text
id
device_id
started_at
confirmed_at
alerted_at
confidence
status
inference_mode
detection_latency_ms
```

### TelemetrySummary

```text
device_id
timestamp
fall_probability
pose_confidence
processing_latency_ms
network_latency_ms
```

Raw high-rate telemetry does not need to be stored indefinitely in PostgreSQL. Storage policy will be defined when the backend is implemented.

## 9. Backend Interfaces

Initial planned REST endpoints:

```text
GET  /api/devices
GET  /api/devices/{id}
GET  /api/events
GET  /api/events/{id}
POST /api/events/{id}/ack
GET  /api/system/status
```

Planned real-time interface:

```text
WS /ws/live
```

The WebSocket should transmit real system state such as:

- device connectivity;
- fall probability;
- event state;
- latency;
- pose availability;
- inference mode.

## 10. Frontend

Minimum V1 dashboard:

```text
┌────────────────────────────────────────────────────────┐
│ Fall Detection Monitor                                 │
├───────────────────────────┬────────────────────────────┤
│ Camera / Pose Preview     │ Current State              │
│                           │ NORMAL / SUSPECTED / ...   │
│                           │ Fall probability           │
│                           │ Camera / IMU status        │
├───────────────────────────┼────────────────────────────┤
│ Probability Timeline      │ Runtime Metrics            │
│                           │ FPS / latency / packet     │
├───────────────────────────┴────────────────────────────┤
│ Event History                                          │
└────────────────────────────────────────────────────────┘
```

The dashboard must consume actual backend data.

## 11. Optional IMU Extension

The optional multimodal path is:

```text
Accelerometer + Gyroscope
          ↓
       Windowing
          ↓
       1D CNN
          ↓
    IMU embedding
          \ 
           \ 
Vision TCN → Fusion MLP → Fall probability
```

The initial preferred fusion strategy is feature-level/intermediate or late fusion rather than raw heterogeneous-signal concatenation.

Missing-modality support may later use explicit modality masks and modality dropout during training.

## 12. Evaluation Architecture

### 12.1 Classification metrics

- Precision
- Recall
- F1
- Confusion matrix

### 12.2 System metrics

Measure separately:

```text
T_edge
T_network
T_cloud_inference
T_event_processing
T_pipeline
T_alert_confirmation
```

Do not conflate computational pipeline latency with intentional alert confirmation time.

### 12.3 Robustness dimensions

Planned controlled tests:

- normal vs low lighting;
- increasing partial occlusion;
- FPS degradation;
- injected network delay;
- packet loss;
- intermittent missing pose frames;
- optional IMU missing/degraded.

### 12.4 Ablation

Minimum:

```text
A: temporal classifier only
B: A + temporal smoothing
C: B + physical/motion evidence
D: C + complete state machine/post-fall verification
```

Optional multimodal:

```text
Vision only
IMU only
Vision + IMU
Vision missing
IMU missing
```

## 13. Non-Functional Requirements

### Reliability

A missing pose frame, temporary MQTT disconnect, or unavailable optional IMU must not crash the entire process.

### Security

- no credentials committed to Git;
- production MQTT/HTTP should later use authentication and TLS;
- `.env` is local/deployment-only;
- avoid unnecessary continuous raw-video storage.

### Observability

Important state transitions and failures must be logged with timestamps and device identifiers.

### Configurability

At minimum the following should not be buried as magic numbers in code:

- FPS;
- sequence length;
- sliding-window stride;
- fall threshold;
- smoothing parameters;
- state-machine timing thresholds;
- broker address;
- model path.

## 14. Stage Boundaries

### Stage 0 — complete

Architecture and repository freeze.

### Stage 1 — current

Dataset exploration and validation. Stage 1.1–1.4 are complete; Stage 1 is
**READY TO PASS GATE REVIEW**, not automatically approved. Stage 2 has not started.
[Dataset protocol](dataset_protocol.md) records the answers and limitations for
each question below, the frozen subject split, and raw missing-pose policy.

Questions to answer before training:

- Which dataset modalities are actually required?
- How are subjects/actions/trials organized?
- How will subject-independent splitting be implemented?
- Can the dataset video be reliably converted using the same pose pipeline as the real camera?
- What class labels should V1 use?

### Stage 2+

Implementation proceeds only after Stage 1 establishes a reproducible data protocol.

## 15. Architecture Change Policy

An architectural decision should change only when at least one of the following applies:

- the current design cannot satisfy a project requirement;
- an implementation dependency makes it impractical;
- measured results show a clear limitation;
- the team deliberately changes project scope.

Any substantial change should update this document and `README.md` in the same change set.
