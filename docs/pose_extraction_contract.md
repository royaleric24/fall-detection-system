# Stage 2 raw pose extraction contract

Schema version: `pose_raw_v1`. Contract version: `1`.

Stage 2.1 defines this contract only. Stage 0 is complete and Stage 1 has passed
Gate Review. No Stage 2 extractor, manifest generator or bulk pose dataset is
implemented by this document. Stage 2.2 has not started.

[Dataset protocol version 2](dataset_protocol.md) remains authoritative for
CAUCAFall V5 identity, labels, split membership, held-out test restrictions,
missingness, low visibility and Stage 1 runtime evidence. This contract selects
the dense storage encoding and timestamp method within that protocol; it does
not revise frozen Stage 1 decisions. MUST and MUST NOT are requirements for the
future implementation and its tests. Paths below are repository-relative unless
another base is explicitly stated. No example path represents a generated file.

## 1. Stage boundary

```text
CAUCAFall AVI
→ OpenCV sequential decode
→ BGR to RGB
→ MediaPipe PoseLandmarker
→ raw pose record
```

Stage 2 raw extraction MAY perform only sequential video decoding, the color
conversion required by MediaPipe, pose inference, raw landmark persistence,
frame/timestamp persistence, missing-pose persistence, and provenance and
extraction-status recording. Each video starts a new PoseLandmarker tracker;
tracking state MUST NOT carry between source videos. Every successfully decoded
source frame is processed exactly once, in source order, preserving the original
spatial resolution without temporal resampling. VIDEO-mode timestamps represent
the source temporal rate using the frozen formula in section 5. Offline extraction
may execute faster or slower than real time; no `sleep()` or 20-FPS wall-clock
throttling is required.

Stage 2 MUST NOT perform any of the following; they belong to Stage 3 or later:

- resampling or skeleton normalization;
- interpolation, forward fill or backward fill;
- low-visibility joint rejection, visibility threshold tuning or joint masking;
- temporal smoothing, derived motion features, velocity calculation or
  body-orientation features;
- temporal-window construction or frame-level fall labeling;
- model training.

## 2. Source video rule

The source root is `data/raw/caucafall_v5/CAUCAFall`. Current measured properties
are **20 FPS** and **720 × 480 pixels** for all 100 AVIs. Preserve the measured
source FPS and resolution, and validate them against the inspected inventory;
do not silently substitute defaults when metadata is invalid or inconsistent.
Raw files MUST NOT be renamed, repaired, deleted or overwritten.

The real-time engineering target of **15 FPS / 30 frames / 2 seconds** is not a
CAUCAFall extraction parameter. Stage 2 MUST NOT resample to 15 FPS. Thirty
source frames at 20 FPS represent 1.5 seconds, not two seconds.

## 3. Storage unit and metadata association

Freeze the physical mapping as **one source AVI → one completed raw pose NPZ**.
The intended ignored output root is `data/interim/caucafall_v5/pose_raw_v1/`.
Mirror the source-relative subject/activity hierarchy and filename stem,
replacing only the `.avi` suffix with `.npz`. For example:

```text
data/raw/caucafall_v5/CAUCAFall/Subject.1/Fall forward/example.avi
→ data/interim/caucafall_v5/pose_raw_v1/Subject.1/Fall forward/example.npz
```

Do not create physical train/validation/test directory trees. Split membership
is explicit video metadata inherited from the subject. Reject output collisions
or paths escaping the output root; outputs MUST NOT resolve into `data/raw`.
Bulk NPZ files remain Git-ignored.

The four arrays below are stored in the NPZ. Video metadata lives in the manifest
row, joined through `output_relative_path`; `run_id` joins that row to run
provenance. An NPZ without its matching manifest/provenance is not a complete
deliverable. No pickled object array is required: readers must be able to load
the core arrays with `allow_pickle=False`. Compression is a storage choice and
does not alter the logical schema.

## 4. Core NPZ arrays

`T` is the number of successfully decoded source frames in the completed video,
including every ordinary no-pose frame. It is not a temporal-window length.

| Array key | Exact shape | NumPy dtype | Meaning |
| --- | --- | --- | --- |
| `frame_index` | `[T]` | `int32` | Zero-based source frame index |
| `timestamp_ms` | `[T]` | `int64` | Clip-relative milliseconds |
| `pose_detected` | `[T]` | `bool` | Authoritative valid-pose indicator |
| `landmarks` | `[T, 33, 4]` | `float32` | Raw x, y, z, visibility in original joint order |

All arrays MUST agree on `T`. `frame_index` MUST equal `0, 1, ..., T-1`:
it starts at zero and increases by exactly one for every successfully decoded
source frame, including no-pose frames. Never renumber to conceal missing poses.
The feature axis is frozen as:

```text
landmarks[..., 0] = x
landmarks[..., 1] = y
landmarks[..., 2] = z
landmarks[..., 3] = visibility
```

Preserve MediaPipe's original 33-landmark index order. Do not select, reorder,
add or mask joints. The required float32 storage conversion is the only numeric
representation conversion; it does not authorize geometric preprocessing.

## 5. Timestamp semantics

For CAUCAFall V5 Stage 2, freeze:

```python
timestamp_ms = round(frame_index * 1000 / source_fps)
```

Apply the formula per scalar frame index using Python's nearest-integer `round`
semantics (ties to even), then store as int64. At 20 FPS, timestamps are exactly
`0, 50, 100, 150, ...`, strictly increasing from clip-local zero. Use the same
timestamps for VIDEO-mode inference and persistence. Do not switch to decoder
presentation timestamps within this schema.

Record `timestamp_method = "derived_from_frame_index_and_source_fps"` in the
manifest and run provenance. These are **clip-relative timestamps**, not Unix
epoch timestamps, inference wall-clock times or fall-onset annotations. The
last timestamp is `(T-1) * 50` ms at 20 FPS; `T / source_fps` is clip duration.

## 6. Valid pose and coordinate semantics

For `pose_detected[t] == true`, `landmarks[t]` MUST have shape `(33, 4)` and all
132 values MUST be finite, including after float32 conversion. A valid result
contains exactly one pose with 33 landmarks under `num_poses = 1`.

Preserve raw MediaPipe API values: x/y are image-relative coordinates using
image width/height scales, with x increasing rightward and y downward. They are
not pixel coordinates and MUST NOT be clamped to the image bounds. z is
MediaPipe relative depth, not calibrated metric depth; preserve its API sign
and scale without conversion. Visibility is the returned per-joint value.

Do not normalize around the pelvis, scale by shoulders or torso, convert z to
meters, fabricate coordinates, or reject a joint for low visibility. No project
skeleton normalization is implied by the API's image-relative coordinates.

## 7. Missing-pose semantics

Only a successful inference call that returns no pose is an ordinary missing
pose. Retain that decoded frame in all four arrays and encode:

```text
pose_detected[t] = false
landmarks[t, :, :] = NaN
```

`pose_detected` is the authoritative missing-pose indicator. NaN is only the
dense-array encoding of absent measurements; it is neither interpolation nor
a fabricated skeleton. Required invariants are:

```text
pose_detected[t] == true  → all 132 landmark values are finite
pose_detected[t] == false → all 132 landmark values are NaN
```

Mixed finite/NaN rows violate the schema. Never use an all-zero skeleton for
missingness, remove the frame/video, interpolate, forward fill or backward fill.
Long missing runs remain visible, including the recorded 92-frame startup gap
in Subject.1 / Fall forward. No coverage cutoff authorizes dropping a video.

## 8. Low visibility

A valid returned 33-landmark pose remains `pose_detected = true` even when some
joints have low visibility. Preserve every coordinate and original visibility
value. Low visibility and no pose are distinct states. Stage 2 introduces no
joint visibility threshold, masking or rejection. The reference PoseLandmarker
detection/presence/tracking settings are task settings, not per-joint filters.

## 9. Extraction failures and completeness

Ordinary no-pose frames are valid records. The following are extraction failures
and MUST NOT be converted into `pose_detected = false`:

| `error_type` | Condition |
| --- | --- |
| `invalid_source_metadata` | Invalid/nonfinite/nonpositive FPS or dimensions, invalid frame count, or disagreement with inspected source metadata |
| `malformed_pose_result` | Wrong pose/joint count, missing required values, or nonfinite returned/stored values |
| `video_decode_failure` | Cannot open/decode video, decoder exception, or unexpected decoded count/termination |
| `inference_runtime_failure` | PoseLandmarker initialization or inference exception |
| `output_persistence_failure` | NPZ, manifest or provenance cannot be written or validated |
| `interrupted_run` | Extraction stops before completion without a successfully finalized video |
| `provenance_mismatch` | Source/model/config identity disagrees with the frozen run provenance |

Record source identity, failure category/message and the zero-based frame index
when known. For an inference failure, that index identifies the decoded frame
whose result could not be recorded; for early decode termination, it identifies
the next expected frame. Leave it absent for failures without a frame identity.

Stop the affected video on failure. A valid prefix may be retained only as
explicitly incomplete temporary output, never as the completed NPZ. Other
videos may proceed, but a run with any unfinished/failed video is incomplete.
If persistent error reporting itself fails, exit nonzero and report the failure
through the process log/stderr; never announce success.

OpenCV read termination alone cannot distinguish EOF from corruption. A video
can be complete only if sequential decoding ends with the positive integral
expected frame count from the inventory, current metadata agrees, every decoded
frame has a valid or ordinary missing record, all schema invariants pass, and
the output is successfully persisted and reloaded/validated. Count agreement
does not certify visual integrity of every frame.

Write and validate a temporary NPZ before publishing the final file and marking
its row complete. A filename's existence alone is not evidence of completion.
A future resume operation MUST validate schema, arrays/counts, source checksum
and run configuration before reusing a completed output. Restart an incomplete
video from frame zero with a fresh tracker; do not append after a lost tracker
state. Detailed implementation and interruption tests belong to Stage 2.2+.

A fully recorded video with no detected poses is structurally complete, with
its missing count exposed; this is not a claim of useful pose coverage or a
passing compatibility gate. Do not silently discard it or tune on test results.

## 10. Labels and split inheritance

Store only video-level `subject_id`, `original_activity`, `binary_label` and
`split`. `subject_id` is an integer 1–10 corresponding to `Subject.N`.
`original_activity` retains the exact source directory spelling. The five
activities Fall backwards, Fall forward, Fall left, Fall right and Fall sitting
map to `fall`; Hop, Kneel, Pick up object, Sit down and Walk map to `non_fall`.
Validate against the frozen dataset protocol and split config.

Do not create `frame_label` or assign active-fall labels to every frame in a fall
video. Fall onset and window supervision are not defined yet and require a
later policy before training.

The final explicit membership in [dataset_split.json](../configs/dataset_split.json)
is authoritative:

| Split | Subject IDs in frozen order | Videos |
| --- | --- | --- |
| `train` | 8, 4, 3, 9, 1, 2 | 60 |
| `validation` | 10, 5 | 20 |
| `test` | 6, 7 | 20 |

Every video, frame, skeleton, future processed sample and temporal window
inherits its source subject's split. Never randomly resplit frames/skeletons,
cross video/split boundaries, or regenerate final membership solely from seed
42; that seed records only the historical candidate.

## 11. Held-out test restrictions

Subject.6 and Subject.7 are held-out final test subjects. During Stage 2.2–2.5
development/debugging, do not inspect their outputs to modify extraction,
preprocessing, visibility handling, missing-data handling, feature engineering,
normalization or temporal-window rules. Use permitted train/validation sources
for development; learned preprocessing statistics remain training-only.

Once the pipeline is fully frozen, Stage 2.6 may mechanically apply the same
deterministic extractor to Subjects 6/7. Their statistics, visual outputs or
qualitative error analysis MUST NOT adapt the pipeline before final evaluation.
Mechanical checks of format, identity and completeness are permitted; failures
must remain explicit and cannot justify test-driven policy tuning. This is not
a claim of byte-identical inference across different platforms/runtimes.

## 12. Planned extraction manifest

Planned path: `artifacts/pose_extraction/manifest.csv`. UTF-8 CSV with a header
and LF line endings; one row per source AVI in the run's declared scope, including
failed, incomplete and pending videos. A full Stage 2.6 run requires all 100
unique source AVIs. A smaller development run must declare its scope and must
not be presented as the full dataset. Preserve numeric subject ordering then
activity/path ordering. No generator or sample run values are created here.

| Field | Type and meaning |
| --- | --- |
| `schema_version` | String, exactly `pose_raw_v1` |
| `run_id` | Nonempty unique run identifier matching extraction_run.json |
| `dataset_name` | String, `CAUCAFall` |
| `dataset_version` | String, `V5` (locally supplied identity) |
| `subject_id` | Integer 1–10 |
| `original_activity` | Exact frozen source activity name |
| `binary_label` | `fall` or `non_fall`, video-level only |
| `split` | `train`, `validation` or `test`, from subject membership |
| `source_relative_video_path` | POSIX path relative to `data/raw/caucafall_v5/CAUCAFall` |
| `source_sha256` | Lowercase 64-hex SHA-256 of actual source AVI bytes |
| `output_relative_path` | Intended final NPZ POSIX path relative to the declared output root |
| `source_fps` | Positive finite source frames/second; currently 20 |
| `width` | Positive integer source width in pixels; currently 720 |
| `height` | Positive integer source height in pixels; currently 480 |
| `expected_frame_count` | Positive integer from inspected inventory, checked against current metadata |
| `extracted_frame_count` | Nonnegative number of valid-or-missing frame records constructed |
| `pose_detected_count` | Nonnegative number of records with a valid returned pose |
| `pose_missing_count` | Nonnegative number of ordinary no-pose records |
| `timestamp_method` | Exactly `derived_from_frame_index_and_source_fps` |
| `status` | `pending`, `running`, `complete`, `failed` or `incomplete` |
| `error_type` | Failure category from section 9, otherwise empty |
| `error_message` | Actual diagnostic text, otherwise empty; no invented results |
| `error_frame_index` | Additional field: zero-based failure frame index when known, otherwise empty |

`pending` means not attempted; `running` means started but not finalized. Freeze
the terminal status/error mapping as follows:

- `complete`: successfully finalized and validated video satisfying section 9.
- `failed`: known extraction failure, such as invalid source metadata, malformed
  pose result, decode failure, inference failure, provenance mismatch or
  persistence failure.
- `incomplete`: interrupted or unfinished extraction that was not finalized.
  In particular, `error_type = interrupted_run` MUST map to `status = incomplete`.

A stale `running` record discovered after a terminated process MUST be treated
as unfinished/incomplete, never complete. Unavailable measurements or
hashes are empty CSV cells, not fabricated zeros; complete rows require all
non-error fields. Counters begin at zero once extraction starts and describe
constructed records, not a claim of successful persistence in failed runs.

For every complete row:

```text
T = expected_frame_count = extracted_frame_count
pose_detected_count + pose_missing_count = T
pose_detected_count = count(pose_detected == true)
pose_missing_count = count(pose_detected == false)
error_type = error_message = error_frame_index = empty
```

A malformed/inference-failure frame is excluded from both pose counters and
reported as an error, so it cannot inflate ordinary missingness. The full dataset
is complete only when all 100 unique rows and associated outputs are complete.

## 13. Planned run provenance

Planned path: `artifacts/pose_extraction/extraction_run.json`. Record actual run
values, not copied reference measurements. It MUST contain at least:

- `schema_version`, `contract_version`, `dataset_protocol_version`, `run_id`,
  declared source/output roots, video scope and overall completion status;
- split-config path and SHA-256; input inventory and split-inventory paths and
  SHA-256 values; contract/protocol paths and hashes identifying the exact text;
- `git_commit` and `git_dirty`, identifying the Git commit and dirty-worktree status;
- actual Python, MediaPipe, OpenCV and NumPy versions, including the OpenCV
  distribution/package version and the complete resolved dependency versions;
- PoseLandmarker model identity/version/source URL and actual model SHA-256;
- running mode, delegate, `num_poses`, pose detection confidence, pose presence
  confidence, tracking confidence and segmentation setting;
- timestamp method, source FPS policy, original-resolution BGR→RGB input policy
  and per-video tracker reset policy;
- runtime/platform information, including OS and machine architecture;
- source-download provenance and source-checksum-manifest path/hash when used.

A dirty worktree must remain traceable so the commit alone does not conceal
relevant uncommitted changes. Do not embed arbitrary large patches directly in
`extraction_run.json`. If a dirty diff is archived, prefer a separate artifact
referenced by `git_diff_artifact_path` and `git_diff_sha256` metadata. Stage 2.1
does not implement a Git archival system.

The single planned manifest/provenance pair describes one run. Preserve earlier
run metadata before replacing it; do not mix rows from incompatible runs. Reuse
on resume must remain traceable to the matching source/model/configuration.

The successful Stage 1.3 reference configuration is:

| Setting | Recorded reference |
| --- | --- |
| Python | 3.13.15 |
| MediaPipe | 0.10.35 |
| OpenCV distribution | opencv-contrib-python 4.12.0.88 |
| NumPy | 2.2.6 |
| Model | PoseLandmarker Full float16 v1 |
| Running mode / delegate | VIDEO / CPU |
| `num_poses` | 1 |
| Detection / presence / tracking confidence | 0.5 / 0.5 / 0.5 |
| Segmentation | Disabled |

Versioned model source:
`https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`

Recorded model SHA-256:

```text
5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1
```

Verify the actual model digest before extraction. The Stage 1.3 direct dependency
pins are not a complete environment lock. Its MediaPipe 1.0.1 initialization
failure and 0.10.35 workaround remain documented in the
[environment notes](../artifacts/pose_compatibility/environment_notes.md).
This evidence does not establish camera hardware compatibility, headless
portability, real-time throughput or LIVE_STREAM performance.

## 14. Source checksum preflight

Before Stage 2.6 full extraction, create a metadata-only source-video checksum
manifest covering exactly the 100 inventoried AVIs, with fields
`relative_video_path` and `sha256`. Paths are relative to the CAUCAFall source
root; hashes are SHA-256 over actual file bytes. Store the manifest outside raw
data (suggested: `artifacts/pose_extraction/source_checksums.csv`), record its
hash in run provenance, and verify extraction sources against it. Hashing test
AVIs for identity is a mechanical integrity check, not exploratory pose analysis.

Record the original download source; do not infer an unverified source URL or
claim authenticated V5 origin. A new checksum manifest establishes a current
baseline, not historical immutability before it existed. No source file may be
modified. Stage 2.1 creates neither this checksum manifest nor pose outputs.

## 15. Stage 2.1 acceptance checklist

Stage 2.1 is complete when all documentation checks below hold. These checkmarks
describe this specification, not executed extractor validation:

- [x] Raw pose schema `pose_raw_v1` and one-AVI/one-NPZ mapping documented.
- [x] Exact array dtypes and shapes documented.
- [x] Coordinate semantics and joint/feature order documented.
- [x] Timestamp units, origin, formula and metadata method documented.
- [x] Missing-pose indicator, NaN encoding and invariants documented.
- [x] Low-visibility preservation and distinction from missingness documented.
- [x] Extraction failures, statuses and completion conditions documented.
- [x] Video labels and frozen subject split inheritance documented.
- [x] Held-out test restrictions through Stage 2.6 documented.
- [x] Manifest, run provenance and source-checksum requirements documented.
- [x] Stage 2 and Stage 3+ responsibilities explicitly separated.
- [x] No Stage 2 extractor or bulk pose dataset generated by Stage 2.1.

Stage 2.2 requires a separate implementation task. Future tests must exercise
schema invariants, ordinary missing poses versus malformed/error outputs,
low-visibility preservation, timestamps, split/provenance validation, persistence,
interruption/resume and completeness without using held-out test outputs for
policy development. No model training has occurred; GRU remains the initial
baseline and TCN the primary candidate, with neither superiority nor performance
inferred from this contract.
