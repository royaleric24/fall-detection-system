# Stage 6 remote deployment evidence — PASS

Frozen on 2026-10-05 (Asia/Shanghai). The actual remote smoke test passed
14/14 checks. This is deployment/transport/inference evidence, not an accuracy
evaluation or webcam test.

Mac AVI → OpenCV → MediaPipe → current causal preprocessing → Internet MQTT
→ Alibaba Cloud Ubuntu VPS → Mosquitto → systemd Python backend → rolling GRU
→ MQTT prediction → Mac subscriber.

## Recorded identity and runtime

| Item | Evidence |
| --- | --- |
| Provider | Alibaba Cloud (team-provided deployment identity) |
| OS / architecture | Ubuntu 22.04.5 LTS / x86_64 |
| Deployment Git commit | `6272c58b09f77f834c3fc1e755ad0c90b4955032` |
| Checkpoint | `ml/checkpoints/stage5_no_normalization_best.pt` |
| Checkpoint SHA-256 | `3c1fbf4be59f705f003c06f0e1b5698dec5c3d81e19c3345ab35231a237132f6` |
| Broker | Mosquitto active; installed package `2.0.11-1ubuntu1.2` |
| Backend | `fall-backend`, enabled, active/running |
| Backend PID / NRestarts | 6274 / 0, unchanged before/after smoke and at read-only freeze check |
| Backend invocation | `3f8b2bff05bb4d8087372f51eb87fc90` |
| Python | 3.12.14 on Mac edge and VPS backend |
| Source AVI | `Subject.1/Fall backwards/FallBackwardsS1.avi` (train) |
| source_id | `remote-smoke-065f410b15bd` |
| sequence_id | `3d13427b25d548ae89fab8cc040dd89e` |
| Edge published / pose detected / missing | 125 / 125 / 0 |
| VPS broker observed frame messages / unique indices | 125 / 125 (indices 0–124) |
| Predictions generated / received on Mac | 4 / 4 |
| Prediction frame indices (zero-based) | 85 / 95 / 105 / 115 |
| Buffer lengths | 86 / 96 / 106 / 116 |
| First probability / threshold / label | 0.9752359390258789 / 0.5 / fall |
| Backend/observer unexpected warnings or errors | none recorded during smoke |
| Edge unexpected errors / exit code | none / 0 |

**125 frames = VPS broker observed count.** The backend does not expose its
cumulative frames_received counter online. Its exact live count is unknown
(`null` in the original VPS result). Backend journal and prediction messages
prove it processed at least through frame 115 / buffer length 116; they do not
prove backend online frames_received=125. The service was kept running.

The AVI hash was unchanged before/after the smoke. No held-out Subject 6/7
data was accessed. The selected representation remains 132-D MediaPipe
x/y/z/visibility, normalize_pose=false, causal fill up to five missing frames,
minimum context 86, maximum context 200, stride 10, threshold 0.5.

## Nonfatal MediaPipe warnings

All three original warning lines are preserved in `mediapipe-warnings.txt` and
`result.json`: two feedback-manager warnings disabling feedback tensor support,
and one NORM_RECT/IMAGE_DIMENSIONS projection warning. These were nonfatal in
this run: edge exit code 0, all 125 poses published, and four remote predictions
returned. They are not recorded as errors.

## Saved files and provenance

- `result.json`: original Mac result, including checklist, prediction receipt,
  edge warnings, source hashes and embedded VPS result; preserved byte-for-byte.
- `vps-result.json`: original remote result, read with sudo from the existing
  root-owned temporary directory; preserved byte-for-byte and verified equal
  to the Mac result's embedded VPS object.
- `edge-summary.json`: original Mac summary, preserved byte-for-byte.
- `predictions.json`: four complete MQTT prediction examples extracted from
  the original Mac result and matched to the VPS observer.
- `backend-journal.txt`: smoke journal lines extracted from the VPS result.
- `vps-readonly-verification.txt`: separate read-only service/OS/package check
  at 2026-10-05T13:52:15Z (21:52:15 Asia/Shanghai). No inference was rerun.
- `mediapipe-warnings.txt`: the three original warning lines.
- `source-evidence-sha256.json`: hashes of the three byte-preserved originals.

Original locations existed and were read at freeze time:
Mac `/private/tmp/fall-phase9-5j0x21fp/evidence-2auww9kh/`,
VPS `/tmp/fall-phase9-evidence-cc4suikn/`. Temporary runtime scripts, raw media,
environment files, passwords and private keys are not included.

## Deployment operation references

Backend start command: `/opt/fall-detection/.venv/bin/python -m backend.app.main`,
working directory `/opt/fall-detection`, systemd user `fall`.
Python venv dependencies: torch 2.8.0 CPU, NumPy 2.3.5, Paho MQTT 2.1.0
(deployment setup is documented in `docs/stage6_mqtt_runtime.md`).
The actual deployed OS above supersedes that document's proposed Ubuntu 24.04
and its earlier pending-deployment status.

Mosquitto listens on TCP 1883 with authentication, anonymous access disabled;
edge uses the public VPS address recorded in the original result, backend uses
loopback. Keep TCP 1883 permitted from the demonstration edge IP in cloud
security groups/firewall and preserve administrative SSH access. This freeze
does not independently audit firewall rules. Credentials remain in the private
VPS `/etc/fall-detection.env` and the local process environment, outside Git.
Relevant variables are MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
MQTT_TOPIC_PREFIX and CHECKPOINT_PATH; no credential values are stored here.

This evidence commit changes documentation only. It does not change the
deployed code commit, backend, VPS, checkpoint, model or preprocessing. Dashboard
and alert confirmation are outside this Stage 6 smoke acceptance.
