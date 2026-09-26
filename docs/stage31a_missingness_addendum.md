# Stage 3.1a — Train-only missingness addendum

This addendum addresses the external **HOLD** on the existing Stage 3.1
characterization. It adds descriptive topology, subject-stratified class
comparisons and one predeclared whole-video-missing sensitivity view. It does
not declare Stage 3.1 PASS, select a preprocessing policy or change the dataset.

The source remains official Stage 2 run
`4df7dd5f-fb38-4908-8233-1a81deb8dc05` and Stage 3.0 baseline
`9d1912cd4c1566438a60496d5d055a43189d3a66`. The script first invokes
`contract.select_sources(("train",), purpose="exploration")`, verifies the
existing Stage 3.1 Train-only tables against the selected source identities and
their original code hash, then derives the addendum **without reopening any pose
NPZ**. Validation and Test requests fail before source or artifact reads.

Run from the repository root:

```sh
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project --with numpy==2.2.6 python -B -m unittest tests.test_stage31a_addendum tests.test_stage31_characterization tests.test_stage3_contract -v
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project --with numpy==2.2.6 python -B -m ml.preprocessing.missingness_addendum
```

The script adds seven files under the existing
`artifacts/preprocessing/stage31_train_pose_characterization_v1/` directory:
`missing_runs_topology.csv`, `missing_topology_summary.csv`,
`subject_class_missingness.csv`, `subject_class_differences.csv`,
`missingness_sensitivity.csv`, `stage31a_summary.json` and
`stage31a_REPORT.md`. Existing Stage 3.1 files remain byte-identical. The
addendum JSON records the input table digests, addendum code digest, frozen
contract/config digests, run identity and Train scope. Output filenames refuse
overwrite, so a second invocation does not silently replace the review record.

## Topology and timing definitions

A missing run is one maximal consecutive `pose_detected=false` sequence, with
inclusive source-frame indices. Each run belongs to exactly one category:

- `whole_video`: every source frame is missing;
- `leading`: begins at frame 0 and an observed pose follows;
- `trailing`: ends at the final frame and an observed pose precedes it;
- `internal`: observed poses exist on both sides.

The classification checks whole-video first, then leading/trailing, then
internal. Per-topology tables give run count, missing-frame count and share of
all Train missing frames, plus length and time distributions. Empty groups have
zero count and null quantiles.

The frozen timestamp rule is `round(frame_index * 1000 / source_fps)` at
20 FPS, yielding a 50 ms source-frame period. For a run of N missing frames:

```text
missing_support_ms = last_missing_timestamp_ms
                   - first_missing_timestamp_ms
                   + source_frame_period_ms
                   = N × 50 ms
```

This is the temporal support of the missing samples. It is explicitly distinct
from the **internal-only** distance between observed anchors:

```text
anchor_gap_ms = right_observed_timestamp_ms
              - left_observed_timestamp_ms
```

For an internal N-frame run in this contiguous 20 FPS source,
`anchor_gap_ms = (N+1) × 50 ms`. Leading, trailing and whole-video runs have
null left/right anchor timestamps and null anchor gap. The addendum preserves
the original Stage 3.1 `duration_ms` field without changing its values and
adds explicit `missing_support_ms` and anchor columns in a separate run table.
Neither time quantity is selected as a future interpolation threshold.

## Class comparison and sensitivity

The subject-class table has one row per class for each Train subject (five
videos per class). It gives equal-video mean/median missing fraction, mean/median
longest missing support and mean missing-run count. A separate table reports
within-subject `fall - non_fall` differences for these five quantities.
Videos with no missing run contribute zero to longest-duration class summaries.
No significance test is performed.

The sensitivity table has two predefined views: all 60 Train videos, and all
Train videos except **only** those that are wholly missing. Both fall and
non-fall rows report video counts, pooled pose availability and equal-video
class summaries. The exact omitted path list accompanies the second view.
Omission is limited to this diagnostic calculation; the original video and
its Stage 3.1 row remain in the dataset and original artifacts.

Frames are nested within videos, and videos within subjects. Descriptive class
differences may reflect subject, activity or tracking characteristics. This
addendum establishes neither causality nor predictive usefulness. All Stage 3.0
preprocessing choices remain `UNDECIDED`; no model, thresholds, transformed
poses, temporal windows or frame/window labels are produced.
