# CAUCAFall V5 dataset protocol — V1

Protocol version: 2 (Stage 1 gate-review split correction). Stage 1.4 defines dataset use; the architecture remains
[architecture.md](architecture.md). Stage 0 and Stage 1.1–1.3 are complete.
Stage 1.3 is **PASS WITH WARNINGS**. Stage 2 has not started. This document is
the Stage 1 source of truth for labels, split membership, and raw pose policy.
A change to frozen membership or labels requires an explicit protocol revision,
updated config/manifests/tests, and disclosure in subsequent experiments.

## Dataset identity, layout and immutable input

The primary dataset is **CAUCAFall V5**, as identified in the user's downloaded
copy. Actual video root: `data/raw/caucafall_v5/CAUCAFall`, relative to the
repository root. Version/source authenticity has not been independently verified;
record the original download source and a source checksum manifest before bulk
extraction. No pre-experiment checksum manifest exists, so historical byte-for-byte
immutability cannot be independently certified.

The outer download directory holds documentation such as `Dataset details.xlsx`
and JPEG figures. They are not videos or additional samples. The video tree is
`Subject.N/<original activity>/<video>.avi`, with Subject.1 through Subject.10,
ten activities per subject and one AVI per subject/activity pair. No further
trial subdivision was observed. Existing PNG/TXT files are auxiliary annotations.

Treat `data/raw` as read-only: no rename, repair, deletion, or overwrite. `data/`
remains Git-ignored; raw videos/images/annotations, model binaries, and future
bulk pose outputs must not be committed. Small metadata and diagnostics belong
under `artifacts/`, outside raw data. An ignored Finder `.DS_Store` is not an
annotation. Current tools leave it untouched.

## Measured source properties

Use the [inspection report](../artifacts/dataset_inspection/summary.md), rather
than assumptions from older publication descriptions:

| Property | Local V5 result |
| --- | --- |
| Subjects / activities per subject | 10 / 10 |
| AVI videos | 100 |
| Binary video labels | 50 fall / 50 non_fall |
| Source FPS | 20 for all videos |
| Resolution | 720 × 480 pixels for all videos |
| Frames per video | 86–292 |
| Calculated duration | 4.3–14.6 seconds |
| Sequentially decoded frames | 19,877 |
| Decode/count check | All 100 match OpenCV metadata counts |

OpenCV does not directly distinguish EOF from decoder failure. Sequential read
termination at the expected count establishes count agreement, not visual
integrity of every frame. Duration is frame_count / FPS, not an event timestamp.

The separate real-time engineering target remains **15 FPS, 2 seconds,
approximately 30 frames**. Source videos stay at **20 FPS** during raw extraction.
No resampling is prescribed here. Temporal-rate alignment is later preprocessing
work; never interpret 30 source frames at 20 FPS as two seconds.

## Frozen original-activity → binary V1 mapping

| Original activity | Binary video label |
| --- | --- |
| Fall backwards | fall |
| Fall forward | fall |
| Fall left | fall |
| Fall right | fall |
| Fall sitting | fall |
| Hop | non_fall |
| Kneel | non_fall |
| Pick up object | non_fall |
| Sit down | non_fall |
| Walk | non_fall |

Always preserve the original activity alongside the binary label. These are
**video-level activity labels**: a fall clip can include standing, movement and
post-fall frames. Do not automatically label every frame or future temporal
window in a fall clip as an active fall. Window/event supervision and temporal
onset conventions remain later design work, to be defined before training.

## Frozen subject-independent split

[configs/dataset_split.json](../configs/dataset_split.json) is the machine-readable
membership source for the final **fixed subject-independent split**. Explicit
subject lists are authoritative. The final assignment cannot be regenerated
solely from seed 42; no shuffle is performed by the assignment tool.

| Split | Subject IDs (in frozen order) | Subjects | Videos | fall | non_fall |
| --- | --- | --- | --- | --- | --- |
| train | 8, 4, 3, 9, 1, 2 | 6 | 60 | 30 | 30 |
| validation | 10, 5 | 2 | 20 | 10 | 10 |
| test | 6, 7 | 2 | 20 | 10 | 10 |

The split unit is the **subject**. Every video, frame, skeleton, augmented
sample and future temporal window inherits the originating subject's split.
Never split frames/windows randomly after preprocessing. Retain subject and
source-video provenance in derived records and validate against this mapping.
Do not concatenate across source-video or split boundaries. Fit learned
preprocessing statistics using training data only; choose model/hyperparameters
with validation, and reserve test for final evaluation. No cross-validation is
implemented. Subject-level cross-validation may later supplement reliability
assessment if time permits, with a separately documented evaluation protocol.

### Split history and methodological correction

1. A seed-42 candidate shuffle produced 8, 4, 3, 9, 6, 7, 10, 5, 1, 2:
   candidate train 8/4/3/9/6/7, validation 10/5, test 1/2.
2. Stage 1.3 exploratory pose compatibility validation had already processed and
   inspected Subjects 1–5. In particular, Subject.1 / Fall forward's 92-frame /
   4.60-second missing interval informed the missing-pose policy.
3. No classifier was trained, so this is not training leakage. However, the
   candidate test subjects were involved in policy development. Before any
   Stage 2 preprocessing or model training, the gate review deliberately moved
   Subjects 1 and 2 to train and Subjects 6 and 7 to test. Validation is unchanged.
4. Subjects 6 and 7 were not in the Stage 1.3 exploratory pose study and are now
   reserved for final held-out evaluation. They did undergo the all-subject
   structural/media checks; this is not a claim that their files were never read.

The config's `candidate_generation` object records the original seed and order
as history only. It does not determine the final explicit subject assignments.

### Held-out test policy

After this Stage 1 gate review, Subject.6 and Subject.7 are the frozen held-out
final test subjects. Before final evaluation, they must not be used for:

- preprocessing-policy tuning;
- feature engineering decisions;
- visibility-threshold selection;
- interpolation-rule selection;
- window-coverage threshold tuning;
- model selection;
- hyperparameter tuning;
- alert-threshold tuning;
- qualitative error analysis used to modify the system.

Stage 2 may mechanically preprocess them using the already-frozen pipeline,
because the same deterministic preprocessing must eventually be applied to all
splits. Their resulting statistics or visual outputs must not be used to adapt
the pipeline before final evaluation. Validation subjects may be used for
model/preprocessing selection where appropriate. Learned preprocessing statistics
must be fitted using training data only. Every derived sample retains its
originating subject's split.

## Manifest construction and integrity

Run from the repository root:

```sh
uv run --python 3.13.15 ml/datasets/assign_dataset_split.py
uv run --python 3.13.15 --no-project python -m unittest discover -s tests -v
```

The assignment script uses only the Python standard library and inline uv
metadata; no project environment or new dependency is needed. JSON is used for
the config so the standard library can parse it without adding a YAML package.

Input: `artifacts/dataset_inspection/inventory.csv` (the inspected inventory).
Output: `artifacts/dataset_inspection/split_inventory.csv`, retaining every input
column and adding `split` = `train`, `validation`, or `test`. This is a
deterministic extension of existing metadata, not a second media inspection.
`--config`, `--inventory`, and `--output` can override paths. Neither raw videos
nor MediaPipe are opened. The input inventory is preserved.

Fields include subject, original activity, binary `label`, `relative_video_path`
(relative to the CAUCAFall root), `fps` (source frames/second), `width`/`height`
(pixels), `frame_count`, `duration_seconds`, prior decode checks, and `split`.
Rows use numeric subject ordering then activity/path; membership is independent
of row order. CSV uses UTF-8 with LF line endings. There is exactly one row per AVI.

Validation rejects overlapping/duplicate/unknown subjects, incomplete coverage,
wrong split sizes, missing/duplicate video rows,
unknown activities, mismatched source paths/labels, invalid media metadata,
failed readability, and conflicting prior split assignments. It requires every
subject/activity pair and expected balanced counts. Failure exits nonzero before
writing the manifest. No record is silently dropped to force expected counts.
The tests freeze the exact requested membership in addition to checking leakage.

## Annotation policy

The main V1 data path is **AVI → MediaPipe Pose → skeleton time series**.
PNG/TXT YOLO-style boxes are not required supervision for this path.
The known warnings remain: six PNGs without exact-basename TXT, four non-classes
TXT without exact-basename PNG, and 100 `classes.txt` directory metadata files.
Exact lists are retained in the inspection summary. Do not repair or rename
these files or treat `classes.txt` as an orphan frame annotation. Any future
annotation-based experiment requires its own explicit exception policy.

## MediaPipe compatibility evidence and runtime

[Stage 1.3 report](../artifacts/pose_compatibility/REPORT.md): ten clips across five
subjects, 1,504 / 1,724 frames with valid poses (**87.24% availability**), all
successful detections with exactly 33 finite landmarks. This is not model
accuracy, fall-classifier performance, or proof of robustness.

**Subject.1 / Fall forward: 94 / 190 (49.47%), longest missing run 92 frames /
4.60 seconds (indices 0–91).** Long startup gaps and shorter floor-level/deeply
bent-pose gaps must remain visible. Low availability is not grounds to drop a
subject or change its split.

The successful run used Python **3.13.15**, MediaPipe **0.10.35** Tasks
PoseLandmarker, OpenCV-contrib **4.12.0.88**, NumPy **2.2.6**, Full float16 v1
model, VIDEO mode, one pose, CPU delegate, all three task confidence parameters
0.5 and segmentation disabled. These task parameters are not a per-joint
visibility rejection policy. The real-camera path is planned to share the
MediaPipe model/interface; camera hardware, LIVE_STREAM behavior, throughput,
and headless/platform portability have not been validated by this experiment.

MediaPipe 1.0.1 aborted during macOS arm64 CPU graph initialization. Its signature
is consistent with issue #6356, but that report used Python 3.14.5; identical
root cause is not established. The successful fallback retained Python 3.13.15,
the Tasks API and the same model family. See
[environment notes](../artifacts/pose_compatibility/environment_notes.md).

Inline script metadata pins direct dependencies. Transitive dependencies are
not locked, and default Python constraints are not patch pins. Select 3.13.15
explicitly and capture the complete resolved environment for future bulk runs.
Verify the versioned model against the recorded SHA-256:
`5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`.
Source FPS, task settings, model hash, package versions, protocol/config revision
and input-inventory provenance must accompany later extraction runs.

## Frozen Stage 2 raw pose policy (specification only)

For each successfully decoded source frame, preserve zero-based frame index and
clip-relative time. Preserve source presentation timestamps when reliable, or
record source FPS and `timestamp_ms = frame_index * 1000 / fps` with the method
explicitly identified. The measured constant 20 FPS yields 50-ms spacing from
zero. These are not Unix event timestamps. Never renumber frames to hide gaps.

For a valid returned pose, record `pose_detected = true` and preserve all 33
landmarks in MediaPipe's original index order, including x, y, z and visibility.
Keep API coordinates unmodified: image-relative x/y use width/height scales;
z is MediaPipe's relative depth, not calibrated metric depth. No project
normalization, clipping or coordinate fabrication is allowed in raw extraction.
The exact storage encoding will be implemented/tested in Stage 2.

For no returned pose, record `pose_detected = false`, retain the frame/time and
represent landmarks as explicitly absent (for example null, never zero-filled
coordinates). Do not interpolate, forward-fill, backward-fill, fabricate,
silently remove the frame, or remove the video. Long missing runs remain explicit.

A returned 33-landmark pose with low visibility is **not a missing pose**.
Preserve every joint and its visibility. Do not mask/reject low-visibility joints
or choose a new joint threshold in Stage 1.4. Visualization thresholds in old
overlays affect display only. Later filtering/weighting or possible short-gap
imputation requires a justified, tested preprocessing policy before training.

Malformed output (wrong count/nonfinite coordinates) or an inference/decode error
must be separately reported as an extraction failure, not mislabeled as ordinary
no-pose or silently skipped. Preserve available frame/source identity in error
records. An incomplete run must not be presented as a complete skeleton dataset.

## Stage gate and remaining implementation

The five architecture Stage 1 questions now have explicit answers:

| Architecture question | Answer / evidence |
| --- | --- |
| Required modalities? | RGB AVI only; IMU optional, boxes not needed for primary path |
| Subjects/actions/trials? | 10 subjects × 10 activities × 1 AVI; measured inventory |
| Subject-independent split? | Final fixed subject-independent membership; automated integrity checks |
| Compatible with planned pose pipeline? | PASS WITH WARNINGS; 33-landmark Tasks interface demonstrated on ten clips; real-camera equivalence remains an implementation validation |
| V1 labels? | Frozen ten-activity → fall/non_fall video mapping, originals preserved |

Once manifest/integrity tests pass, Stage 1 is **READY TO PASS GATE REVIEW**, not
automatically approved or marked complete. Compatibility warnings are accepted
as explicit protocol constraints, not resolved robustness claims.

Stage 2 still needs an authorized extraction implementation with the above raw
record/error semantics, complete environment/source provenance, persistence and
resume/completeness tests, then bulk extraction after gate approval and an explicit Stage 2 task. Stage 3 needs
normalization, feature definitions, visibility/missing-data treatment, temporal
rate/window/label design and tests before training. Neither stage is implemented
here. GRU remains the initial baseline and TCN the primary candidate.
