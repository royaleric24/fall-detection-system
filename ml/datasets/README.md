# CAUCAFall V5 — Stage 1 inspection

Current handoff: **Stage 2 complete/frozen; Stage 3.0 pending external Gate Review**.
Earlier sections below retain their historical stage scope. The official raw-pose
run is `4df7dd5f-fb38-4908-8233-1a81deb8dc05`; development NPZs are not substitutes.
See the [Stage 3 contract](../../docs/stage3_preprocessing_contract.md) and
[metadata/access guards](../preprocessing/contract.py). Future analysis must use
`select_sources()` (Train by default), explicitly request Validation when permitted,
and never bypass the Test guard. No preprocessing or training is authorized by
this handoff. Stage 2 scripts and hashed protocol/configuration files are unchanged.

Run from the repository root using uv and Python 3.12 or newer:

```sh
uv run ml/datasets/inspect_caucafall.py
uv run ml/datasets/inspect_caucafall.py --root /path/to/CAUCAFall --output artifacts/dataset_inspection
uv run --python 3.13 --no-project python -m unittest discover -s tests -v
```

There is no project-wide Python environment yet. The inspector declares pinned
OpenCV headless and NumPy dependencies in its inline script metadata; uv isolates
these dependencies. OpenCV is needed to inspect AVI metadata and decode real
frames. No GUI, pandas, pose extraction, training, resampling, or raw-data writes
are involved. On restricted hosts, `UV_CACHE_DIR=/private/tmp/caucafall-uv-cache`
can select a writable uv cache.

The default input is `data/raw/caucafall_v5/CAUCAFall`. The containing directory's
spreadsheet and JPEG documentation are outside the video tree and file counts.
The supplied download is identified as V5 by the user; the tool does not independently
authenticate dataset provenance. Keep the download source/version with future
experiment manifests.

Reports are Git-trackable metadata under `artifacts/dataset_inspection/`:

- `inventory.csv`: one row per discovered AVI, ordered numerically by subject;
  original activity names and frozen V1 `fall` / `non_fall` labels are retained.
- `summary.json`: counts, annotation samples, complete orphan lists, structural
  issues, runtime versions, media distributions, and validation status.
- `summary.md`: human-readable rendering of the same results.

Inventory paths are relative to the input root. FPS is frames/second; width and
height are pixels; frame count is reported by the container/OpenCV. Duration is
frame count divided by FPS, in seconds. Invalid/nonpositive/nonfinite metadata
becomes an empty CSV cell (JSON null). `opened` records VideoCapture opening;
`decoded_first_frame` records a nonempty real decoded frame. The inspector now
sequentially reads until OpenCV returns no frame. `decoded_frame_count` is the
actual count; `frame_count_matches` compares it to metadata.
`reached_end_normally` means count equality at termination;
`decode_failure_before_expected_end` flags early termination. OpenCV cannot
distinguish EOF from decoder failure directly, so these are count-based checks,
not proof of visual integrity. Missing metadata leaves comparison fields null.
`readable` requires a decoded frame and count agreement. `error` records
opening/decoding failures, invalid metadata, or count mismatches.
No event timestamps or skeleton coordinates are produced.

Expected structure is Subject.1 through Subject.10, each with the ten activities
listed in `ACTIVITY_LABELS` and exactly one direct AVI per activity. Unexpected
AVI locations are failures and remain visible in the inventory. Unknown activities
receive `unknown` in the inventory and trigger structural failure, rather than
being silently assigned a binary label.

TXT classification is content-based: `classes.txt` is metadata; nonempty lines
of an integer class ID plus four finite values in [0, 1] are frame annotations;
all other or unreadable TXT files are reported separately. The observed format
is consistent with YOLO-style `class_id center_x center_y width height` normalized
boxes, not skeletons; that coordinate interpretation is an inference. Class IDs
are checked against directory-local `classes.txt`. Exact relative basenames are
used for pairing: the inspector never renames or repairs files. A PNG whose TXT
is malformed is reported separately from a PNG with no TXT at all.

Exit codes: 0 for structural/media success (including annotation warnings),
1 for structural/media failure, 2 for invalid arguments or missing dependencies.
Always review annotation warnings before any work that consumes annotations.
Outputs inside the input dataset or repository `data/raw` are refused.

The 15 FPS / 30-frame / 2-second values in the architecture remain engineering
targets, not source-video assumptions. This inspection does not establish pose
extraction quality, a data split, or a training protocol. Stage 2 is not started.

## Observed annotation exceptions in the local download

There are 100 `classes.txt` files: 50 contain `nofall\nfall\n`, 48 contain
`nofall\n`, and two contain `nofall` without a final newline. All 19,999 remaining
TXT files contain one five-field annotation line. Class-ID counts are 13,610
for `0` and 6,389 for `1`; these are annotation counts, not video labels or model
metrics. Local class IDs must not be confused with the provisional activity-level
binary video labels: fall videos also contain non-fall frames.

The following are naming-artifact hypotheses, not verified repairs:

| PNG basename | Unmatched TXT basename in the same directory | Interpretation |
| --- | --- | --- |
| `Subject.2/Fall backwards/cas200091 - copia` | `cas200091` | Copy suffix suggests a naming artifact |
| `Subject.8/Hop/sals800096` | `sals800096a` | Extra `a` suffix suggests a naming artifact |
| `Subject.8/Pick up object/res800090` | `res800090a` | Extra `a` suffix suggests a naming artifact |
| `Subject.9/Walk/cams900140` | `cams900140w` | Extra `w` suffix suggests a naming artifact |
| `Subject.5/Kneel/ars500236` | None | Missing annotation; cause undetermined |
| `Subject.6/Walk/cams600260` | None | Missing annotation; cause undetermined |

The complete paths, including file extensions, are in the generated summary.
No files were changed. An explicit exception policy is needed before using these
PNG/TXT pairs for supervised work. These annotation mismatches do not by themselves
block a later AVI-based pose exploration step, provided video validation succeeds.

## Stage 1.3 — pose compatibility smoke validation

This is still Stage 1. The fixed subset is one clip per activity across subjects
1–5; see `SAMPLES` in `validate_pose_compatibility.py`. It was selected before
inference to cover all source activities, not to estimate population accuracy.
No split, normalization, resampling, skeleton export, or model training is done.

```sh
curl -L --fail --output ml/checkpoints/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
uv run ml/datasets/inspect_caucafall.py
uv run ml/datasets/validate_pose_compatibility.py
uv run --python 3.13 --no-project python -m unittest discover -s tests -v
```

The pose script uses a uv-isolated Python 3.13 environment with MediaPipe 0.10.35,
OpenCV-contrib 4.12.0.88 (required by MediaPipe), and NumPy 2.2.6. It does not
install both headless and contrib OpenCV into that environment. The standalone
media inspector retains its original isolated headless environment. There is
still no project-wide `pyproject.toml`. The model stays in the existing ignored
checkpoint directory; its official versioned URL and SHA-256 are recorded.

The official Tasks `PoseLandmarker` API in VIDEO mode uses the Full float16 v1
model on CPU, one person, all three confidence thresholds 0.5, and no segmentation.
`--confidence`, `--root`, `--model`, and `--output` are configurable. Every decoded
frame is submitted synchronously, converted BGR→RGB at original 720×480 resolution.
Timestamps are clip-relative milliseconds: `round(frame_index * 1000 / fps)`,
starting at zero. At measured 20 FPS they increase by 50 ms. A new tracker is
created for each video. A future camera path can use the same Tasks model and
RGB/33-landmark interface; this experiment does not validate camera hardware,
LIVE_STREAM frame dropping, or real-time throughput.

Valid detections have exactly 33 finite x/y/z/visibility values. API-normalized
image coordinates are retained transiently only; no project skeleton normalization
is applied. Missing poses count as missing and do not stop the clip. Visibility
min/mean/max aggregate all 33 landmarks of all valid frames; they measure model
visibility estimates, not landmark localization accuracy. Missing intervals are
inclusive zero-based frame indices. Diagnostics comprise two fixed positions
(25% and 75%) in a forward fall, kneel, and walk, plus at most the first missing
frame per clip. Overlays display landmarks/connections with visibility ≥0.5.

The mechanical gate fails for decode/inference errors, malformed landmarks, or
any clip with zero valid poses; warns for any missing pose; otherwise passes.
This avoids an invented percentage cutoff. A human visual review and inspection
of missing-run lengths must accompany the gate: near-total loss or loss of a
critical fall phase can still block pose-based downstream work.

See [experiment report](../../artifacts/pose_compatibility/REPORT.md) for measured
results and the final reviewed compatibility decision. Stage 1 remains current.
Stage 1.4 now freezes split membership and raw extraction policy in the
[dataset protocol](../../docs/dataset_protocol.md); later preprocessing/feature
design remains necessary before training.

Official references checked for this experiment:

- [Python setup](https://developers.google.com/edge/mediapipe/solutions/setup_python)
- [Pose Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [MediaPipe release/wheels](https://pypi.org/project/mediapipe/1.0.1/)

### Dependency reproducibility limits

Each script carries PEP 723 inline metadata (`# /// script`): `uv run <script>`
reads `requires-python` and exact direct dependency pins and resolves an isolated
script environment. No `pyproject.toml` is needed. The media script pins
opencv-python-headless 4.12.0.88 and numpy 2.2.6; the pose script pins mediapipe
0.10.35, opencv-contrib-python 4.12.0.88 and numpy 2.2.6.

There are currently no script lockfiles. Transitive dependencies may therefore
resolve to newer versions on a clean installation. The Python constraints also
do not pin a patch release. To select the measured interpreter explicitly, use
`uv run --python 3.13.15 <script>`. Thus the direct dependencies and model digest
are reproducible, but a byte-identical complete environment is not guaranteed.
The model URL is versioned and its SHA-256 is in the experiment summary; verify
that digest when reproducing the experiment. The four-frame context sheet was
created by the documented one-off OpenCV snippet, not by the pose validator.

Stage 1.4 is responsible for the missing-pose/low-visibility dataset policy.
Stage 1.3 counts missing frames explicitly and does not fill them, interpolate
them, or remove them from the pose-availability denominator.

## Stage 1.4 — frozen split and raw extraction protocol

[Dataset protocol](../../docs/dataset_protocol.md) is the Stage 1 source of truth.
[Split config](../../configs/dataset_split.json) fixes final subject membership:
train 8/4/3/9/1/2, validation 10/5, test 6/7. The seed-42 candidate is recorded
as history only; the gate-review adjustment protects Subjects 6/7 from exploratory
pose-policy development. The final split is not generated solely by seed 42. JSON avoids
adding a YAML dependency. Never reassign frames/windows independently of subjects.

```sh
uv run --python 3.13.15 ml/datasets/assign_dataset_split.py
```

This standard-library-only tool reads the existing inspection inventory and
writes `artifacts/dataset_inspection/split_inventory.csv` with all original
columns plus `split`. It does not read videos or run pose extraction. The
original inventory stays intact. Each of 100 subject/activity pairs must appear
exactly once, and split counts must be 60/20/20 with 30/10/10 examples of each
binary label. Invalid/overlapping assignments fail rather than dropping records.

Raw Stage 2 policy is now specified: keep all 33 returned landmarks, including
low visibility, and keep every missing frame/time explicitly without filling
or interpolation. No joint visibility rejection threshold is introduced.
Stage 1 is READY TO PASS GATE REVIEW; Stage 2 has not started.

## Stage 2.2 — single-video raw pose extraction

The [approved Stage 2 contract](../../docs/pose_extraction_contract.md) governs
the new single-video tool. The Stage 1 status statements above describe those
earlier tasks. Stage 2.1 has passed review; this command implements only the
Stage 2.2 single-video path, not full-dataset extraction or preprocessing.

Run from the repository root, first on the Stage 1.3 high-availability train clip,
then on the known missing-pose train clip:

```sh
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache MPLCONFIGDIR=/private/tmp/caucafall-matplotlib uv run --offline --python 3.13.15 ml/datasets/extract_pose_video.py 'data/raw/caucafall_v5/CAUCAFall/Subject.2/Fall backwards/FallBackwardsS2.avi'
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache MPLCONFIGDIR=/private/tmp/caucafall-matplotlib uv run --offline --python 3.13.15 ml/datasets/extract_pose_video.py 'data/raw/caucafall_v5/CAUCAFall/Subject.1/Fall forward/FallForwardS1.avi'
```

`--offline` uses an already populated uv cache; omit it when dependencies need
installation. Inline metadata pins Python 3.13.15, MediaPipe 0.10.35,
opencv-contrib-python 4.12.0.88 and NumPy 2.2.6. The official Full float16 v1 model
must already exist at `ml/checkpoints/pose_landmarker_full.task`; `--model` may
select another local copy with the same approved SHA-256. The shared
`pose_runtime.py` module retains Stage 1.3 VIDEO/CPU/one-pose settings with all three
confidence settings at 0.5 and segmentation disabled. No threshold tuning occurs.

The CLI accepts exactly one canonical AVI path under the expected source root.
It derives identity, labels, split and expected media properties from the inspected
inventory and frozen split config. It refuses Subjects 6/7 before opening their
videos. All source frames retain their original resolution and temporal indices;
VIDEO timestamps use the frozen formula, with no resampling or wall-clock throttle.

The ignored output root defaults to `data/interim/caucafall_v5/pose_raw_v1/`;
`--output-root` can change it. Subject/activity/stem hierarchy is mirrored. The
NPZ contains exactly the four contract arrays, without labels or Python objects.
`pose_raw.py` validates the in-memory arrays, writes a temporary NPZ and reloads
with `allow_pickle=False` to validate again. The CLI stages the required JSON
result too, verifies its contents and syncs both staged files. It publishes the
validated NPZ first and the result JSON last using filesystem links that refuse
overwrites. The final `status = complete` result JSON is the completion marker.
Caught failures or interruptions roll back this attempt's published files and
clean temporary artifacts; existing files belonging to another run are preserved.
Missing poses retain all-NaN landmarks; low-visibility detections remain
finite and present. Malformed output and decode/inference errors stop extraction
with a separate error category and frame index when known.

A small per-video JSON result is written under
`artifacts/pose_extraction/single_video/<subject>/<activity>/<stem>.json` and logged
as JSON. It records identity/split, counts, missing intervals, output location,
source/model hashes, runtime/package versions, Git commit/dirty status and hashes
of relevant implementation/configuration files. Source hashes are checked before
and after the selected video only; this is not the 100-video checksum preflight.
Both existing NPZ and existing result destinations are refused, so preserve or
move prior artifacts deliberately before repeating a run. No resume is implemented.

This per-video result is development evidence, not the planned full manifest and
`extraction_run.json` lifecycle; those remain deferred. A valid development NPZ
is not a complete bulk dataset deliverable. Caught interruptions report
`incomplete`/`interrupted_run`; known errors report `failed`. Native process
aborts cannot be caught by Python and may leave no result file; absence of a
successful result never establishes completion. A result-write failure exits
nonzero and leaves no final NPZ from that attempt. Failed/interrupted attempts
are reported through structured stderr logs, without a final result JSON.

Publication across two paths is not a single filesystem transaction. An
uncatchable termination between publication steps can leave an NPZ without a
result JSON. Such an orphan NPZ is incomplete; consumers must require both the
NPZ and its matching complete result JSON. The completion marker is never
deliberately published before its validated final NPZ exists. No crash
recovery/resume or full Stage 2.6 manifest lifecycle is implemented here.

On this macOS host, the first restricted-sandbox attempt aborted during native
MediaPipe graph initialization with graphics-context/service errors (exit 134).
The same pinned CPU setup succeeded outside the sandbox. This does not establish
headless portability or real-time LIVE_STREAM performance.

Run all lightweight tests with NumPy only; no MediaPipe inference or raw-video
reads are required:

```sh
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache uv run --offline --python 3.13.15 --no-project --with numpy==2.2.6 python -B -m unittest discover -s tests -v
```

Stage 2.3 extends the regression suite with timestamp/type checks, malformed and
missing-pose cases, corrupt archives, decode/metadata/inference failures,
source/model provenance changes, interruption and result-publication failure
injection. The two Stage 2.2 result JSONs and local ignored NPZs are preserved as
historical regression evidence. Tests reload the NPZs when present; they do not
rerun inference, read held-out videos or tune pose settings. Their recorded code
hashes identify the historical Stage 2.2 implementation, not subsequent fixes.
