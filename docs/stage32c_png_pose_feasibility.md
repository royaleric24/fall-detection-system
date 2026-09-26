# Stage 3.2c — frozen small-sample PNG pose feasibility

This is a bounded Train-only pilot pending external Gate Review. The frozen
parent is `8a43017170b1a9dff66476f83f95878a6c0945f8`. No Stage 2 or earlier
Stage 3 file is changed. The experiment asks whether official PNGs yield raw
MediaPipe pose observations while preserving exact annotation-native identity.
It does not establish a PNG physical timebase or a final pose runtime.

## Reproduction order

Run from the repository root. The output directory must not already exist.
Freezing must happen before any MediaPipe inference. The two commands are
separate phases so the pre-inference manifest is inspectable and hash-pinned.

```sh
MPLCONFIGDIR=/private/tmp/caucafall-mplconfig UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project --with mediapipe==0.10.35 --with opencv-contrib-python==4.12.0.88 --with numpy==2.2.6 python -B -m unittest tests.test_stage32c_png_pose tests.test_stage3_contract -v
MPLCONFIGDIR=/private/tmp/caucafall-mplconfig UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project --with mediapipe==0.10.35 --with opencv-contrib-python==4.12.0.88 --with numpy==2.2.6 python -B -m ml.preprocessing.pilot_png_pose freeze
MPLCONFIGDIR=/private/tmp/caucafall-mplconfig UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project --with mediapipe==0.10.35 --with opencv-contrib-python==4.12.0.88 --with numpy==2.2.6 python -B -m ml.preprocessing.pilot_png_pose run
git diff --check
```

The freeze phase reads only frozen Train metadata/artifacts, checks all Stage
3.2b artifact hashes, and uses the Stage 3 source selector to enforce Train
membership. The required eight complete sequences appear in task order:
Subject.2 Fall backwards; Subject.8 Hop; Subject.8 Pick up object;
Subject.9 Walk; Subject.4 Fall forward; Subject.4 Pick up object;
Subject.9 Fall sitting; Subject.8 Fall forward. It then selects the
lexicographically first eligible `fall` sequence and the lexicographically
first eligible `non_fall` sequence from the frozen Train inventory. Eligibility
requires verified order, zero explicit alias pairs, zero exact duplicate
groups, and absence from the eight required sequences. No pixel or pose
inspection enters sample selection. The repeatability subset uses the first
pilot `nofall` frame, first `grounded_fall_state` frame, and first exact
duplicate image pair in source-name/ordinal order; overlaps are kept once.
Its identities are also frozen before inference.

The freeze writes `pilot_sample_manifest.csv`,
`annotation_identity_manifest.csv`, `repeatability_subset.csv`, and
`sample_freeze.json` under
`artifacts/preprocessing/stage32c_png_pose_feasibility_v1/`. The JSON pins
their SHA-256 digests, Stage 3.2b source hashes, model hash, and extractor code
hash. The run phase refuses a changed freeze or source record. It verifies
every selected PNG and TXT byte hash and original annotation class against
the reviewed metadata before importing MediaPipe. It rejects any alias outside
the four explicit Stage 3.2b identities; no fuzzy matching is available.
Validation and Test subject content is never opened.

## Runtime and raw observation schema

Stage 2 uses PoseLandmarker `VIDEO` mode with clip-relative AVI timestamps.
The PNG physical timebase is unresolved, so this pilot uses the task's
stateless `IMAGE` mode and calls `detect(image)` without a timestamp. The
Stage 2 Full float16 v1 model asset and SHA-256, CPU delegate, one-pose limit,
0.5 detection/presence settings, and disabled segmentation are retained.
Tracking confidence is configured at 0.5 for parity but does not provide
temporal tracking in IMAGE mode. One runtime instance processes the full pilot;
two fresh instances process the frozen repeatability subset. This mode is
feasibility evidence, not approval for production real-time inference.

PNG bytes are decoded at original resolution using OpenCV
`imdecode(IMREAD_COLOR)` into 8-bit BGR, then converted via
`cv2.COLOR_BGR2RGB` for an SRGB `mp.Image`. No resizing or skeleton
transformation is performed. MediaPipe's original 33-joint ordering is kept.
The four channels are raw x, y, z and visibility; x/y are image-relative,
z is API-relative rather than calibrated meters. The Stage 2 `encode_pose`
helper preserves low visibility and encodes an empty pose as all NaN.
MediaPipe or pose-encoding exceptions are logged per ordinal and represented
as missing poses. A PNG decode/source-integrity failure stops the run.

Each selected sequence has one versioned pilot NPZ beneath `pose_npz/` with
exactly:

| Field | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `annotation_ordinal` | `[T]` | int32 | `0..T-1` PNG-native position |
| `pose_detected` | `[T]` | bool | One observed pose or no pose |
| `landmarks` | `[T,33,4]` | float32 | x, y, z, visibility |

Detected rows must be finite; missing rows must be entirely NaN. There is no
timestamp, inferred FPS, class ID or label in an NPZ. Labels live only in the
identity manifest. Every PNG and TXT retains subject/activity/sequence,
original name, byte hash and ordinal. The extractor receives image bytes only;
it does not receive a class ID. All exact duplicate PNGs retain their own
ordinal and output row. Reload validation checks the schema and completeness.

## Diagnostics and limits

The pilot reports per-sequence and overall observed pose availability, and
availability split by original annotation class: `nofall` and
`grounded_fall_state`. These are missingness diagnostics for a deliberately
selected stress pilot, not classifier metrics or a full-Train estimate.
Subject.8 Fall forward is compared only at the source level with Stage 2's
frozen 0/122 AVI pose result; no frame mapping or cause is inferred.

For selected frozen exact pixel-duplicate groups, pose detection, missing/
finite structure and float32 landmarks are compared exactly. Non-equal
detected pairs report maximum absolute difference without an acceptance
tolerance. The preselected repeatability subset is run twice more under the
same IMAGE configuration, each time with a fresh landmarker. Comparisons are
against the first full-pilot observation. All outcomes are retained; no
repeated full pilot or parameter tuning occurs.

Outputs include the pose NPZs, per-sequence availability, class-state
availability, duplicate and repeatability comparisons, runtime failures,
runtime provenance, summary and report. The pilot never constructs windows,
normalizes/interpolates/resamples poses, trains a model or decides downstream
preprocessing policy. All configuration decisions remain `UNDECIDED`. The
pilot is not committed and stops for external Gate Review.

On this host, the sandboxed IMAGE graph failed before inference because a
macOS graphics service was unavailable. The identical graph initialized
outside the sandbox; the frozen pilot completed there without changing its
sample, model or inference settings. `runtime_execution_notes.json` preserves
this execution anomaly. Another host must verify its own IMAGE runtime before
relying on this pilot's repeatability result.
