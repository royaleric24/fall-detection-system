# Stage 3.1 — Train-only raw pose characterization

Status: analysis implementation for external Gate Review. No preprocessing
policy, derived dataset or classifier is produced. The frozen Stage 3.0 baseline
is `9d1912cd4c1566438a60496d5d055a43189d3a66`; Stage 2 source is only run
`4df7dd5f-fb38-4908-8233-1a81deb8dc05`.

Run from the repository root after the focused tests pass:

```sh
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project --with numpy==2.2.6 python -B -m unittest tests.test_stage31_characterization tests.test_stage3_contract -v
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project --with numpy==2.2.6 python -B -m ml.preprocessing.characterize_pose
```

The script calls `contract.select_sources(("train",), purpose="exploration")`
before any pose load and rejects every other `--split` before calling the selector.
It takes expected frame counts and source FPS only from selected Train manifest
rows, then uses `pose_raw.load_validated` with `allow_pickle=False`. No glob across
NPZ roots, no Validation/Test pose load, and no Stage 2 writes occur.

Output is a fresh, versioned directory:
`artifacts/preprocessing/stage31_train_pose_characterization_v1/`.
The script refuses an existing destination. Its files are:

| File | Content |
| --- | --- |
| `summary.json` | Definitions, provenance, totals, grouped summaries, quantiles, raw-channel/geometry diagnostics and integrity result |
| `videos.csv` | Every Train video; video label, counts, missing runs, durations, progress bins and source identity |
| `missing_runs.csv` | Every Train missing run with inclusive frames, end-exclusive duration and normalized progress |
| `subjects.csv`, `activities.csv`, `classes.csv` | Grouped frame totals and equal-video descriptive means |
| `landmark_visibility.csv` | Visibility distribution for each of 33 joints |
| `REPORT.md` | Concise interpretation of the machine-readable results |

The JSON records dataset/run identity, split, subject IDs, baseline and current
Git commits, code/config/manifest digests, each analyzed NPZ digest, Python/NumPy
versions and definitions. The original dataset's V5 identity remains locally
supplied rather than independently authenticated. CSVs contain only Train rows.
Raw masks remain in the immutable NPZs; source identity and per-video/run tables
support a later independently authorized mask-only ablation without duplicating
or altering a skeleton sequence.

## Statistic definitions

- A missing run is maximal consecutive `pose_detected=false`; frame endpoints
  are inclusive. Its elapsed duration is `timestamp_ms[end] -
  timestamp_ms[start] + 50 ms`. The additional 50 ms represents the last source
  frame's interval at the frozen 20 FPS. A video's duration is last timestamp
  plus 50 ms. This is source video time, not wall-clock or event timing.
- A frame's descriptive normalized progress is its midpoint time divided by
  video duration: `(timestamp_ms + 25 ms) / video_duration_ms`. Missing frames
  are counted in fixed quintile bins `[0,.2), [.2,.4), [.4,.6), [.6,.8),
  [.8,1]`. The bins describe location; they are not coverage or interpolation
  thresholds. Run boundaries use start time and end-exclusive time divided by
  video duration.
- Pose availability is detected/all source frames. A missing fraction is
  missing/all frames. For videos with no missing run, shortest/median/longest
  fields are null. Equal-video class means of longest duration include zero
  for such videos. Missing-run durations are stored in both frames and ms.
- Quantiles use NumPy's linear percentile method on observed finite values.
  Standard deviation is population `ddof=0`. Empty distributions have count
  zero and null descriptive values. A high quantile with few runs is unstable;
  the aggregate summary flags fewer than 20 runs, and sample counts accompany
  every distribution. No significance test is run. Frames and joints within a
  video are correlated, so video/subject summaries guide class interpretation.
- Visibility is analyzed only on detected frames. Each joint has count, mean,
  population SD, min/max and p5/p10/p25/p50/p75/p90/p95/p99. Per-frame mean
  and minimum across 33 joints are also summarized. Values are never masked.
- Candidate widths and torso length are 2-D Euclidean distances in the raw
  image-relative x/y channels. Indices 11/12 are shoulders; 23/24 are hips per
  [MediaPipe's PoseLandmark API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmark).
  Shoulder/hip midpoints are finite whenever their detected-frame joints are
  finite; availability and exact-zero/nonpositive scale counts are reported.
  No numerical near-zero tolerance is introduced, and z is never treated as
  metric depth. These quantities are diagnostics, not normalization choices.
- Raw x/y/z/visibility channel distributions and x/y/visibility counts outside
  `[0,1]` are descriptive. Out-of-range values are not rejected. The frozen
  schema validator fails on wrong keys, shapes, dtypes, timestamps, nonfinite
  detected values or finite missing values before publishing artifacts.

The class comparison is named **missingness shortcut-risk diagnostic**. It
reports pooled pose availability alongside equal-video missing fractions,
longest-run durations and run counts for Train fall and non-fall videos.
Differences alone do not prove causality or predictive shortcut behavior.
Pose availability is MediaPipe return rate, never fall-detection accuracy,
robustness or classifier performance.

All Stage 3.0 decisions remain `UNDECIDED`. No interpolation gap, visibility
threshold, coverage threshold, center/scale, target FPS, resampling method,
window or label rule, or optional motion feature is selected here.
