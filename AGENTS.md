# AGENTS.md

This file defines collaboration rules for AI coding agents and human contributors working on the **Real-Time Edge–Cloud Vision-Based Fall Detection System** repository.

The project is a graduate-level course project for:

**COMP6131 — Internet of Things Essentials**

Current final-delivery deadline:

```text
2026-10-07
```

---

# 1. Project Objective

Build a complete and demonstrable real-time **edge–cloud vision-based fall detection system**:

```text
Camera / AVI
→ edge pose extraction
→ pose preprocessing
→ MQTT
→ real cloud server
→ temporal inference
→ fall probability
→ alert logic
→ dashboard
```

The final system must demonstrate:

- camera-based human-pose input;
- temporal skeleton modeling;
- real edge–cloud communication;
- real cloud-server backend functionality;
- binary fall detection;
- basic false-alarm suppression;
- real-time or near-real-time inference;
- dashboard visualization;
- reproducible model evaluation;
- at least one meaningful ablation or robustness experiment.

The project is **not merely an offline machine-learning benchmark**.

A real cloud-server component is a **mandatory course requirement**.

A purely local or LAN-only backend may be used during development, but it does **not** satisfy the final V1 cloud requirement.

The objective is a complete, credible, working course-project system, not a publication-grade research platform or production-scale cloud architecture.

---

# 2. Current Delivery Mode

The project is now in:

# Final Delivery Sprint

The priority order is:

```text
1. Complete working end-to-end system
2. Prevent subject leakage and Test contamination
3. Produce trustworthy core evaluation results
4. Deploy required real cloud-server functionality
5. Keep implementation reproducible
6. Add extra methodological sophistication only when it materially improves correctness
```

Do not spend substantial project time on research infrastructure that does not materially improve the final deliverable.

## 2.1 Strict Gate rule

Use strict methodology / provenance gates only when a task involves:

- subject or data leakage risk;
- Validation/Test contamination;
- final metric correctness;
- irreversible source-data mutation;
- frozen experimental provenance.

For normal engineering work use:

```text
design
→ implement
→ test
→ continue
```

Do not create publication-grade protocols for ordinary engineering tasks.

---

# 3. Current Project Stage

The Stage 3.2 temporal-annotation investigation is considered **sufficient for the course project**.

Completed:

```text
Stage 3.2h
Independent Dual Annotation
COMPLETE

Stage 3.2i
Inter-Annotator Agreement Analysis
COMPLETE / PASS

Stage 3.2j
Discrepancy Review Tool Implementation
COMPLETE / PASS
```

Deferred because of scope and deadline:

```text
genuine Stage 3.2j discrepancy review
Stage 3.2k formal adjudication
precise temporal-boundary supervision
```

Do not resume these deferred tasks unless explicitly requested.

The active project path is:

```text
Stage 3.3
Minimal Supervision + Preprocessing
↓
Stage 4
GRU Baseline
↓
Stage 5
Evaluation + Ablation + Robustness
↓
Stage 6
Real Cloud Deployment + Edge–Cloud MQTT Integration
↓
Stage 7
Dashboard + Alert Demo
↓
Stage 8
Final Evaluation + Report + Presentation
```

---

# 4. Scope Discipline

Before adding any new:

- model;
- framework;
- experiment;
- service;
- database feature;
- frontend framework;
- annotation process;
- infrastructure component;

ask:

```text
Does this materially improve:

- end-to-end completion?
- course requirements?
- evaluation correctness?
- cloud-system completeness?
- reproducibility?
```

If the answer is no, defer it.

## 4.1 Explicitly deferred / low-priority scope

Unless all P0 work is complete, do not spend time on:

- genuine Stage 3.2j review;
- Stage 3.2k adjudication;
- Transformer models;
- ST-GCN;
- multiple temporal model comparisons;
- large hyperparameter sweeps;
- multimodal IMU fusion;
- domain adaptation;
- advanced uncertainty modeling;
- Kubernetes;
- Kafka;
- distributed microservices;
- autoscaling;
- service mesh;
- production authentication;
- production observability stacks;
- publication-grade statistical analysis.

These may be documented as future work.

---

# 5. Frozen Core Design Decisions

Unless explicitly changed by the team, preserve the following decisions.

1. V1 must work with camera-only input.
2. RGB frames are converted to human-pose landmarks before temporal classification.
3. Edge performs:
   - camera/video capture;
   - pose extraction;
   - pose preprocessing;
   - MQTT telemetry publishing.
4. MQTT is the primary edge–cloud telemetry protocol.
5. A **real cloud server / VPS is mandatory for final V1**.
6. The cloud server performs at minimum:
   - MQTT telemetry reception or broker connectivity;
   - temporal sequence buffering;
   - GRU inference;
   - alert logic;
   - API and/or WebSocket service for the dashboard.
7. FastAPI remains the preferred backend framework if compatible with the existing codebase.
8. The primary V1 temporal model is a **small GRU-based classifier**.
9. Primary ML supervision is **clip-level binary fall / non-fall classification**.
10. Exact manually annotated temporal boundaries are **not primary training targets** for the final course-project baseline.
11. IMU integration is optional future work and must not block V1.
12. Do not introduce infrastructure that does not solve a demonstrated course requirement.

Existing:

```text
FastAPI
Mosquitto
React
PostgreSQL
```

components may be reused where already functional.

However:

```text
PostgreSQL
React
container orchestration
multiple microservices
production authentication
```

are not mandatory completion conditions unless the existing implementation depends on them.

The mandatory requirement is **real cloud-server functionality**, not production-scale cloud architecture.

---

# 6. Engineering Principles

## 6.1 Complete the smallest vertical slice first

Prefer:

```text
Camera / AVI
→ pose
→ preprocessing
→ MQTT
→ cloud GRU inference
→ alert
→ dashboard
```

over several disconnected partially completed components.

## 6.2 Avoid premature complexity

Do not introduce a new:

- framework;
- service;
- dependency;
- queue;
- database;
- model architecture;
- annotation workflow;

unless it solves a concrete project requirement.

## 6.3 Keep interfaces explicit

Important cross-module interfaces should document:

- input schema;
- output schema;
- units;
- coordinate convention;
- timestamp convention;
- missing-data behavior;
- failure behavior.

Do not block progress on excessive documentation when an interface is already simple, tested, and unambiguous.

## 6.4 Preserve working functionality

During the final sprint:

- avoid broad refactors;
- avoid package reorganizations;
- avoid renaming stable interfaces;
- avoid rewriting functional code for aesthetic reasons.

Prefer localized changes.

---

# 7. Reproducibility

Training and evaluation code must record at minimum:

- dataset/source identity;
- subject split;
- random seed;
- preprocessing configuration;
- model configuration;
- checkpoint identity/path;
- evaluation metrics.

Do not claim an experiment is reproducible unless the relevant configuration can actually be recovered.

Do not silently change experiment configuration between runs.

---

# 8. Data Integrity and Held-Out Protection

Subject-independent splitting is mandatory whenever subject identity is available.

Never randomly split frames from the same recorded sequence across train, Validation, or Test.

Verify:

```text
train_subjects ∩ validation_subjects = ∅
```

Test Subjects:

```text
6
7
```

remain sealed during:

- preprocessing design;
- model development;
- model selection;
- hyperparameter adjustment;
- threshold selection;
- ablation selection;
- robustness design.

Do not inspect or tune against Test Subjects 6 or 7 before the final model/configuration is frozen.

No AI coding agent may silently relax this rule.

---

# 9. Stage 3.2 Research Artifacts

Existing Stage 3.2 research artifacts must be preserved.

Do not delete, rewrite, or retroactively reinterpret them.

Current status:

```text
Stage 3.2h:
Independent dual annotation complete

Stage 3.2i:
Inter-annotator agreement analysis complete

Stage 3.2j:
Blind discrepancy-review infrastructure implemented
Genuine review deferred
```

The key course-project interpretation is:

> Clip-level fall presence was considerably more reproducible in the pilot than exact temporal boundary localization.

Therefore the final course-project baseline uses:

```text
clip-level binary fall / non-fall supervision
```

rather than precise `grounded_start`, `fall_transition_start`, or `recovery_start` supervision.

Do not restart temporal-boundary adjudication during the final sprint without explicit instruction.

---

# 10. Primary ML Task

The final primary ML task is:

```text
pose sequence
→ binary classification
→ fall / non-fall
```

Canonical label mapping:

```text
fall     → 1
non-fall → 0
```

Do not infer the primary label from precise temporal boundaries if a canonical clip-level label already exists.

---

# 11. Pose Representation

The primary V1 model input is a normalized skeleton sequence derived from MediaPipe pose landmarks rather than raw RGB.

Logical representation may be:

```text
[T, F]
```

or an equivalent structured tensor derived from:

```text
[T, J, F]
```

depending on the implementation.

The exact representation must be documented in the preprocessing code.

Do not assume the previous tentative:

```text
T ≈ 30
```

is mandatory.

Variable-length sequences with:

- padding;
- sequence lengths;
- masks;

are acceptable.

---

# 12. Minimal Preprocessing Strategy

Prefer the simplest deterministic pipeline compatible with existing `pose_raw_v1` data:

```text
pose_raw_v1
↓
coordinate / feature selection
↓
hip-centered translation normalization
↓
body-scale normalization
↓
simple missing-pose handling
↓
sequence padding / batching
↓
GRU-ready tensor
```

Avoid large feature-engineering studies.

---

# 13. Translation Normalization

Prefer pelvis / hip centering using existing MediaPipe hip landmarks.

Conceptually:

```text
hip_center =
(left_hip + right_hip) / 2
```

Then translate spatial coordinates relative to the hip center.

Use the actual landmark indices and feature schema already defined in the repository.

Do not guess indices.

Missing-hip fallback behavior must be simple, deterministic, and documented.

---

# 14. Scale Normalization

Use one stable body-scale reference compatible with the current landmark representation.

A torso-related scale is preferred if reliably available.

Requirements:

- deterministic;
- resistant to divide-by-zero;
- minimum valid scale defined;
- fallback behavior documented.

Do not choose the normalization strategy using Test performance.

---

# 15. Missing Pose Handling

A temporary missing pose must not crash the pipeline.

Prefer simple deterministic handling such as:

- short-gap linear interpolation;
- zero-filled normalized coordinates;
- validity mask where already supported.

Do not introduce during the final sprint:

- learned imputation;
- Kalman filtering;
- pose reconstruction networks;
- complex smoothing pipelines;

unless a concrete blocking issue requires them.

---

# 16. Primary Temporal Model

The primary temporal model for final delivery is:

# GRU

Preferred conceptual architecture:

```text
pose sequence
→ small GRU
→ final / pooled hidden representation
→ Linear
→ binary logit
→ P(fall)
```

The purpose is a robust, reproducible course-project baseline.

GRU is no longer merely a comparison model.

Do not implement:

```text
TCN
LSTM
Transformer
ST-GCN
```

unless:

```text
GRU training
+
evaluation
+
cloud integration
+
dashboard/demo
+
final deliverables
```

are already complete and stable.

---

# 17. Training Strategy

Keep training intentionally simple.

Only tune a small number of high-impact parameters, for example:

- GRU hidden size;
- learning rate;
- sequence handling;
- training epochs / early stopping;
- decision threshold where justified.

Do not perform large grid searches.

Validation data may be used for model/configuration selection.

Test Subjects 6/7 must not be used to select:

- preprocessing;
- architecture;
- hidden size;
- learning rate;
- threshold;
- sequence length;
- robustness settings.

---

# 18. Evaluation Metrics

At minimum report:

- Accuracy;
- Precision;
- Recall;
- F1;
- confusion matrix.

Primary attention should be given to:

```text
Fall-class Recall
Fall-class F1
```

Accuracy alone is insufficient.

Use the same split and evaluation protocol when comparing experiment variants.

---

# 19. Ablation Scope

Only one primary ablation is required unless time remains.

Preferred ablation:

```text
without normalization
vs
hip-centered + body-scale normalized pose
```

Keep:

- dataset split;
- model family;
- evaluation procedure;

the same across the comparison.

Do not build a large ablation matrix.

---

# 20. Robustness Scope

Keep robustness experiments small and interpretable.

Preferred robustness experiment:

## Missing pose / dropout

Simulate limited degradation such as:

```text
10%
20%
30%
```

frame or landmark dropout where straightforward.

Measure changes in:

```text
Recall
F1
```

Optional second experiment:

## Coordinate noise

Add small synthetic perturbation to normalized pose coordinates and measure performance degradation.

Do not attempt all of the following unless core delivery is already complete:

- lighting sweeps;
- large occlusion studies;
- frame-rate sweeps;
- extensive network-delay experiments;
- large packet-loss studies;
- IMU missing-modality experiments;
- domain adaptation.

One completed, interpretable robustness experiment is preferable to several unfinished experiments.

---

# 21. Required Real Cloud Architecture

A **real remotely deployed cloud server / VPS is a P0 requirement**.

The final runtime should follow:

```text
Camera / AVI
↓
Edge device / local machine
↓
OpenCV + MediaPipe Pose
↓
pose preprocessing
↓
MQTT
↓
REAL CLOUD SERVER / VPS
↓
sequence buffer
↓
GRU inference
↓
fall probability
↓
alert logic
↓
API / WebSocket
↓
dashboard
```

A local backend may be used during development and debugging.

A local-only backend does not satisfy final V1.

---

# 22. Minimal Cloud Server Scope

Use the simplest architecture satisfying the course requirement.

Preferred minimal deployment:

```text
Single Linux VPS
├── Mosquitto MQTT broker
├── FastAPI backend
├── GRU checkpoint
├── sequence buffer
├── inference logic
├── alert logic
└── API / WebSocket
```

Optional components on the same VPS:

```text
PostgreSQL
frontend static hosting
reverse proxy
```

only if already useful or easy to deploy.

Do not introduce:

- Kubernetes;
- Kafka;
- multiple backend services;
- distributed inference;
- load balancing;
- autoscaling;
- service mesh;
- production observability infrastructure.

A single real VPS is sufficient for the course project if it performs actual backend functionality.

---

# 23. Cloud Deployment Reproducibility

Record at minimum:

- cloud provider;
- operating system;
- server architecture where relevant;
- required ports;
- MQTT configuration;
- backend start command;
- Python/environment setup;
- model checkpoint used;
- required environment variables;
- firewall/security-group requirements.

Never commit:

- passwords;
- API keys;
- SSH private keys;
- broker credentials;
- database secrets.

---

# 24. MQTT and Telemetry

MQTT remains the primary edge–cloud telemetry protocol.

Telemetry should contain sufficient information to identify:

- source/device;
- timestamp;
- pose or processed features;
- validity / pose status where relevant.

Prefer Unix epoch milliseconds for transmitted timestamps unless the existing project schema already defines another standard.

Avoid redesigning a working MQTT schema during the final sprint without a concrete reason.

---

# 25. Sequence Buffering

Cloud-side sequence buffers must remain device-specific.

The minimum runtime must handle safely:

- temporary missing pose;
- incomplete sequence buffers;
- temporary MQTT interruption.

Advanced handling of:

- reordering;
- duplicates;
- sophisticated late-message policy;

is secondary unless already implemented.

Do not let advanced transport edge cases block the final demonstration.

---

# 26. Alert Logic MVP

Do not require a publication-grade or production-grade alert state machine.

The minimum acceptable mechanism is:

```text
if P(fall) >= threshold
for N consecutive predictions:
    trigger alert
```

Both:

```text
threshold
N
```

must be configurable.

This provides simple temporal confirmation against isolated probability spikes.

If a more complex existing state machine is already functional, it may be retained.

For example:

```text
NORMAL
→ SUSPECTED
→ CONFIRMED
→ ALERT
→ RECOVERED
```

But a complex state machine must not block delivery.

Do not require additional heuristics such as:

- vertical velocity;
- body angle;
- inactivity duration;
- recovery logic;

unless already implemented and useful.

---

# 27. Backend Scope

The real cloud backend must provide enough functionality to support the course demo.

Minimum required:

- MQTT telemetry consumption;
- temporal sequence buffering;
- GRU inference;
- fall probability production;
- alert evaluation;
- API and/or WebSocket output used by the dashboard.

These functions must run on the real cloud server in the final demonstration.

Persistence is optional unless required by the course rubric or existing implementation.

If PostgreSQL is already stable and useful, preserve it.

Do not begin a major database project during the final sprint.

Inference should remain inside the primary backend unless separation is genuinely required.

---

# 28. Frontend Scope

The dashboard must display **real backend/cloud state**, not hard-coded demo values.

Preferred information:

- device connectivity;
- camera / pose status;
- current fall probability;
- current fall / non-fall state;
- alert state;
- recent inference/event information;
- basic latency/FPS if already available.

Do not prioritize UI polish over:

- model correctness;
- MQTT communication;
- cloud inference;
- real dashboard data flow.

React may be retained if already functional.

Do not rewrite the frontend framework solely for the deadline.

---

# 29. Testing Priorities

Prioritize tests for failures that could invalidate model results or break the final demo.

## P0

1. dataset split integrity;
2. Test Subject 6/7 exclusion;
3. pose normalization;
4. no unexpected NaN / Inf;
5. sequence padding/batching;
6. model input/output compatibility;
7. MQTT payload compatibility;
8. edge → cloud communication;
9. cloud inference smoke test;
10. end-to-end inference path.

## P1

11. alert confirmation logic;
12. dashboard/backend communication;
13. latency timestamp calculations;
14. cloud restart/recovery smoke test if time permits.

Do not spend substantial final-sprint time increasing test coverage for already completed low-risk research tooling.

---

# 30. Experiment Discipline

Robustness and ablation experiments should alter one primary factor at a time where practical.

When comparing variants, preserve:

- dataset split;
- evaluation protocol;
- model family where relevant.

Do not fabricate:

- metrics;
- latency;
- benchmark results;
- screenshots;
- cloud output;
- test output;
- citations.

Do not claim a test was executed unless it was actually executed.

If an experiment is incomplete, state that it is incomplete.

---

# 31. Coding Rules

## Python

- Use the repository's currently supported Python version.
- Use type hints for public functions and important structures.
- Keep functions focused and inspectable.
- Avoid unnecessary hidden global state.
- Prefer `pathlib`.
- Use structured logging in runtime modules where practical.
- Do not perform broad refactors during the final sprint.

## Configuration

Do not hard-code deployment-specific values such as:

- broker host;
- broker port;
- database password;
- cloud server address;
- device ID;
- camera index;
- alert threshold;
- confirmation count;
- model/checkpoint path.

Use the repository's existing configuration mechanism and/or environment variables.

Never commit secrets.

---

# 32. Artifact Rules

Do not commit unnecessary:

- raw datasets;
- large processed datasets;
- private recordings;
- large generated videos;
- secrets.

Follow the repository's established conventions for:

- model checkpoints;
- evaluation artifacts;
- manifests;
- generated reports.

Keep reproducible scripts and metadata.

Do not rewrite the artifact policy during the final sprint unless needed.

---

# 33. AI Agent Workflow

When an AI coding agent receives a task:

1. Read this `AGENTS.md`.
2. Identify the current project stage.
3. Determine whether the task affects:
   - leakage;
   - Test data;
   - final metrics;
   - immutable source data.
4. If yes, apply strict validation.
5. Otherwise choose the simplest reliable implementation.
6. Modify only the smallest relevant modules.
7. Add focused tests where failure would materially affect correctness.
8. Avoid unrelated refactoring.
9. Preserve existing working functionality.
10. Do not silently change frozen design decisions.
11. Do not restart deferred Stage 3.2 research.
12. Do not introduce new model architectures unless explicitly requested.
13. Do not generate fake experiment or deployment results.
14. Do not claim code was tested unless it was actually executed.
15. Prefer completion over architectural elegance during the final sprint.
16. Keep the real cloud-server requirement in scope at all times.

If two technically valid approaches exist, prefer the one that is:

```text
simpler
more reliable
faster to verify
easier to demonstrate
```

unless the alternative materially improves correctness.

---

# 34. Current Priority Order

Use this priority order:

```text
P0
Stage 3.3 preprocessing is GRU-ready

P0
GRU trains and produces valid predictions

P0
No subject leakage / Test contamination

P0
Validation metrics are trustworthy

P0
Real cloud server is deployed and reachable

P0
Edge → MQTT → cloud communication works

P0
Cloud performs actual GRU inference

P0
End-to-end system works

P0
Final evaluation metrics are correct

P1
Dashboard + alert demonstration

P1
One meaningful ablation

P1
One meaningful robustness experiment

P1
Deployment / demo documentation

P2
Database/event-history persistence

P2
UI polish

P2
Additional robustness experiments

P3
Additional models

P3
Additional annotation research
```

Do not perform P2 or P3 work while critical P0 work remains incomplete.

---

# 35. Definition of Done for Final Course V1

The final course-project V1 is complete when a real or prerecorded camera/video source can drive:

```text
Camera / AVI
→ edge pose extraction
→ preprocessing
→ MQTT
→ real cloud server
→ sequence buffering
→ GRU inference
→ fall probability
→ alert confirmation
→ API / WebSocket
→ dashboard
```

The cloud-server step is mandatory.

A local-only backend is acceptable for development, but not for final completion.

In addition:

```text
[ ] preprocessing pipeline is reproducible
[ ] GRU training is reproducible

[ ] train/validation subject separation is verified
[ ] Test Subjects 6 and 7 were not used for tuning

[ ] core evaluation metrics are reported

[ ] real cloud server deployment works
[ ] edge successfully communicates with cloud through MQTT
[ ] cloud performs actual model inference
[ ] dashboard receives actual backend/cloud state

[ ] alert confirmation logic works

[ ] at least one ablation or robustness experiment is completed

[ ] end-to-end demo works

[ ] README/report explains:
    - system architecture
    - ML pipeline
    - edge-cloud design
    - cloud deployment
    - evaluation
    - limitations
```

The following are **not mandatory completion conditions** unless separately required by the course rubric:

```text
PostgreSQL
React
IMU integration
multiple ML models
production authentication
production scaling
```

**Real cloud-server functionality is a mandatory completion condition.**

---

# 36. Final Delivery Rule

When there is a conflict between:

```text
extra sophistication
```

and:

```text
a complete, tested, demonstrable system
```

choose the complete system.

Do not sacrifice:

- source-data integrity;
- subject-split correctness;
- held-out Test integrity;
- final metric correctness;
- reproducibility;
- mandatory cloud-server functionality;

for speed.

Everything else may be simplified when necessary to meet the course deadline.