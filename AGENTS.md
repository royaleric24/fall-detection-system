# AGENTS.md

This file defines collaboration rules for AI coding agents and human contributors working on the Fall Detection System repository.

## 1. Project Objective

Build a complete real-time edge-cloud fall detection system with:

- camera-based pose input;
- temporal skeleton modeling;
- real cloud backend functionality;
- robust false-alarm handling;
- real-time dashboard;
- reproducible evaluation;
- optional IMU multimodal enhancement.

The project is not merely an offline machine-learning benchmark.

## 2. Frozen Design Decisions

Unless explicitly changed by the team, preserve these decisions:

1. V1 must work with camera-only input.
2. RGB frames are converted to human-pose landmarks before temporal classification.
3. Edge performs camera capture, pose extraction, normalization, and telemetry publishing.
4. Cloud performs sequence buffering, model inference, event-state logic, persistence, API, and WebSocket service.
5. MQTT is the primary edge-cloud telemetry protocol.
6. FastAPI is the backend framework.
7. PostgreSQL is the main persistent database.
8. React is the dashboard frontend.
9. TCN is the primary temporal-model candidate; GRU is the initial baseline/comparison model.
10. IMU integration is an enhancement, not a blocking V1 dependency.
11. Do not introduce Kubernetes, Kafka, microservice decomposition, or other infrastructure without a demonstrated project need.

## 3. Engineering Principles

### 3.1 Build the smallest complete vertical slice first

Prefer a functioning path:

```text
camera → pose → temporal inference → event → backend → dashboard
```

over multiple disconnected partially implemented modules.

### 3.2 Avoid premature complexity

Do not introduce a new framework, service, dependency, queue, database, or model architecture unless it solves a concrete requirement.

### 3.3 Keep interfaces explicit

Every cross-module interface must document:

- input schema;
- output schema;
- units;
- coordinate convention;
- timestamp convention;
- missing-data behavior;
- failure behavior.

### 3.4 Reproducibility

Training/evaluation code must record:

- dataset version/source;
- split strategy;
- random seed;
- preprocessing configuration;
- model configuration;
- checkpoint;
- metrics.

Avoid subject leakage between training and test sets.

### 3.5 Measure before optimizing

Do not claim a component improves robustness, latency, or F1 without an experiment.

## 4. Coding Rules

### Python

- Prefer Python 3.12+ unless a dependency requires otherwise.
- Use type hints for public functions and important data structures.
- Keep functions small and single-purpose.
- Avoid hidden global state.
- Use `pathlib` for filesystem paths.
- Use structured logging instead of scattered `print()` statements in production modules.
- Add tests for normalization, buffering, event-state logic, and message schemas.

### Configuration

Do not hard-code deployment-specific values such as:

- broker host;
- database password;
- device ID;
- camera index;
- thresholds;
- model paths.

Use configuration files and/or environment variables.

Never commit secrets or real credentials.

### Data and model artifacts

Do not commit:

- raw datasets;
- large processed datasets;
- `.pt`, `.pth`, `.onnx`, or large checkpoints;
- generated video files;
- private test recordings.

Keep metadata, scripts, manifests, and reproducible download/preprocessing instructions instead.

## 5. ML Rules

### 5.1 Representation

The primary V1 model input is a normalized skeleton sequence rather than raw RGB.

Expected logical shape:

```text
[T, J, F]
```

where initially:

- `T ≈ 30` frames;
- `J = 33` landmarks;
- `F` contains normalized coordinates and selected confidence/motion features.

The exact feature definition must be documented before training.

### 5.2 Dataset splitting

Use subject-independent splitting whenever subject identity is available.

Never randomly split frames from the same recorded sequence across train/test.

### 5.3 Baselines first

Before adding complex architectures, establish reproducible baselines:

1. simple heuristic baseline where useful;
2. GRU baseline;
3. TCN model.

A more complex model is justified only after baseline limitations are measured.

### 5.4 Metrics

At minimum report:

- Precision;
- Recall;
- F1;
- confusion matrix.

Fall detection must not rely on accuracy alone because class imbalance can make accuracy misleading.

## 6. Real-Time System Rules

### Time

All telemetry must include timestamps.

Prefer Unix epoch milliseconds for transmitted event timestamps unless the team explicitly standardizes another representation.

### Buffering

Cloud sequence buffering must be device-specific.

Out-of-order, late, duplicated, and missing messages must have defined behavior.

### Failure handling

A temporary missing pose must not crash the pipeline.

The system should distinguish at least:

- person not detected;
- low pose confidence;
- camera offline;
- MQTT disconnected;
- backend unavailable.

## 7. Alert Logic Rules

Never trigger a production alert directly from one model output.

The system must use an event-state machine such as:

```text
NORMAL → SUSPECTED → CONFIRMED → ALERT → ACKNOWLEDGED/RECOVERED
```

The event engine may combine:

- model probability;
- persistence across windows;
- vertical motion;
- body orientation;
- post-fall inactivity;
- pose confidence.

All thresholds must be configurable and later evaluated through ablation/testing.

## 8. Backend Rules

The cloud deployment must run real backend functionality.

At minimum the backend is expected to provide:

- MQTT telemetry consumption;
- temporal sequence buffering;
- inference execution;
- event-state evaluation;
- database persistence;
- REST endpoints;
- WebSocket updates.

Do not create an "inference microservice" solely for architectural appearance. Keep inference inside the backend until scaling or isolation creates a real need.

## 9. Frontend Rules

The dashboard must display real backend state, not hard-coded demo values.

Minimum planned widgets/views:

- device connectivity;
- camera/pose status;
- current inference mode;
- current fall probability;
- current alert state;
- recent event history;
- latency/FPS metrics.

## 10. Testing Priorities

Prioritize automated tests for components whose failure can silently invalidate experiments:

1. skeleton normalization;
2. temporal-window construction;
3. dataset split integrity;
4. MQTT payload validation;
5. event-state transitions;
6. latency timestamp calculations.

## 11. Experiment Discipline

Robustness tests must alter one factor at a time where possible.

Planned factors include:

- lighting;
- occlusion;
- frame rate;
- network delay;
- packet loss;
- missing pose data;
- optional missing IMU modality.

Ablation results should compare the same dataset split and evaluation protocol.

## 12. AI Agent Workflow

When an AI agent receives a task:

1. Read `README.md` and `docs/architecture.md` before making architectural changes.
2. Identify the current project stage.
3. Modify only the smallest relevant module.
4. Do not silently change frozen architecture decisions.
5. State assumptions if requirements are ambiguous.
6. Add or update tests where appropriate.
7. Update documentation when schemas or architecture change.
8. Do not generate fake metrics, benchmark values, experiment results, device output, or citations.
9. Do not claim code was tested unless it was actually executed.
10. Prefer simple, inspectable implementations suitable for graduate-level learning and demonstration.

## 13. Definition of Done for V1

A V1 vertical slice is complete only when a real camera can drive the following path end to end:

```text
real camera
→ edge pose extraction
→ MQTT
→ real cloud backend
→ temporal inference
→ event-state machine
→ database
→ WebSocket/API
→ dashboard
```

The core V1 must remain functional even if the optional IMU subsystem is unavailable.
