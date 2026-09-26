# Stage 3.0 — handoff, preprocessing contract and leakage firewall

Contract version: **1**. Review status: **pending external Gate Review**.
This document records the requested Stage 3.0 constraints; it does not declare
Stage 3.0 PASS or authorize Stage 3.1, a derived-data build, or training.
The research/technical-review process owns unresolved methodology decisions.

Machine-readable source: [stage3_contract.json](../configs/stage3_contract.json).
Guard implementation: [contract.py](../ml/preprocessing/contract.py).
The architecture, camera-only V1, GRU baseline and TCN candidate are unchanged.

## 1. Immutable input and historical evidence

| Identity | Frozen value |
| --- | --- |
| Dataset | CAUCAFall V5, locally supplied; origin not independently authenticated |
| Stage 2 run | `4df7dd5f-fb38-4908-8233-1a81deb8dc05` |
| Execution baseline | `4b7557a5b3a8a4b40906b57a01a264bdeef9edb6` |
| Final evidence commit | `a28e0289c53463b3676b6c9275d4ee18a840fdae` |
| Input schema | `pose_raw_v1` |
| Pose root | `data/interim/caucafall_v5/pose_raw_v1_runs/4df7dd5f-fb38-4908-8233-1a81deb8dc05/` |
| Evidence root | `artifacts/pose_extraction/runs/4df7dd5f-fb38-4908-8233-1a81deb8dc05/` |

The official run declares `complete`. Never overwrite its NPZs, manifest,
provenance, result JSONs, checksums, reports or test logs. Development outputs
under older roots are not interchangeable with this official input.
The configuration pins SHA-256 digests of the official run JSON, manifest and
frozen split config. The selector also checks the inventory against the digest
in the official run. It validates complete source identity and manifest mapping;
it does not rerun extraction or claim a fresh array-integrity audit.

Earlier stage-status wording in the frozen dataset protocol/extraction contract
describes their historical creation stage. Those files remain byte-identical for
Stage 2 provenance. This document and the updated current-status entries describe
the Stage 3 handoff. Do not rewrite history to update status.

## 2. Raw schema, coordinates and missingness

One source video maps to one NPZ, containing exactly:

| Key | Shape | dtype |
| --- | --- | --- |
| `frame_index` | `[T]` | `int32` |
| `timestamp_ms` | `[T]` | `int64` |
| `pose_detected` | `[T]` | `bool` |
| `landmarks` | `[T,33,4]` | `float32` |

Channels are x, y, z, visibility in MediaPipe's original joint order. x/y are
image-relative width/height coordinates (right/down positive); z is API relative
depth, not calibrated meters. Retain out-of-bounds coordinates and low visibility.
`pose_detected=true` requires all 132 values finite; `false` requires all 132 NaN.
No-pose, low visibility and malformed/incomplete extraction are different states.
Missing pose must never silently become a zero skeleton or disappear from time.

All future representations must preserve original observation/missingness
provenance. If later approved transformations introduce imputed or invalidated
values, distinguish them explicitly from measured values in a separately reviewed
derived schema. No mask layout, fill strategy or rejection rule is selected here.
Use the existing `pose_raw.load_validated` after authorized source selection when
future work is permitted to load arrays. Stage 3.0 itself has no pose loader.

## 3. Frozen subject split and access policy

`configs/dataset_split.json` remains the authoritative membership source.
The code's exact-membership assertions detect drift; they do not create a new split.
Seed 42 describes a historical candidate, not the final assignment.

| Split | Subjects, in frozen order | Access |
| --- | --- | --- |
| Train | 8, 4, 3, 9, 1, 2 | Default future exploratory scope; sole source of fitted statistics |
| Validation | 10, 5 | Explicit selection for permitted development/selection; never fitting |
| Test | 6, 7 | Stage 3.0 identity/split/file-existence checks only; no pose loading or exploratory statistics |

Every frame, skeleton, feature, augmentation and window inherits its source
subject's split. Never randomly repartition derived samples. Never concatenate
videos or splits to construct windows. A future window has exactly one source
video, one subject and one split; retain its source time/frame mapping.

Test must not influence missingness exploration, interpolation/visibility
thresholds, normalization, resampling, window coverage or labels, features,
model/hyperparameter choice, alert thresholds, or qualitative error analysis used
to modify the system. Do not consume Test diagnostics from existing Stage 2
reports as a substitute for loading Test arrays. Their prior mechanical extraction
does not authorize policy analysis.

`select_sources()` defaults to Train and returns immutable `PoseSource` identity
records, including an explicitly named `video_label`, never a window label.
`select_sources(("validation",))` is an explicit Validation request.
`purpose="fit_statistics"` permits Train only; it selects sources, not a fitted
transform. `require_subject_access` checks real subject membership and rejects
mislabeling a Test subject as Train. Forbidden requests fail before filesystem I/O.
The selector reads manifest identity fields and returns no availability counters;
it never opens NPZs, videos, per-video results or aggregate reports.

No Test-loading override exists. A future final/frozen-application mode requires
a separately reviewed contract and deterministic pipeline, with unchanged
Train-fitted state and explicit authorization. Unknown modes currently fail.
No final evaluation workflow is implemented or implied by this reservation.

These are repository API safeguards, not an OS sandbox. Future analysis entry
points must call the guards before any pose load; direct `numpy.load` or an
external script can bypass an API guard and must not be used to bypass policy.
No analysis is authorized merely by having a selector available.

## 4. Processing dependencies and temporal rules

The required dependency order is:

1. Validate immutable source identity and split; enforce access purpose **before loading**.
2. In a separately authorized stage, characterize permitted Train inputs; any
   Validation involvement must be explicit and recorded.
3. Determine temporal supervision separately in Stage 3.2. Freeze all required
   preprocessing decisions, derived schema, and exact transform order through review.
4. If the approved pipeline learns statistics, fit them using Train only; record
   the fitting sources/configuration and freeze those values for Validation/Test.
5. Apply the approved timestamp-based transformations independently per video.
6. Construct and label within-video windows only under the approved supervision
   protocol; preserve observation masks, provenance and split inheritance.
7. Validate derived data and publish a new version/run with its own manifest.

Steps 2–7 are future dependencies, not implemented operations. In particular,
the relative algorithmic order of interpolation, normalization, resampling and
optional feature extraction remains `UNDECIDED`; choosing it is methodology.

Source timing remains 20 FPS with clip-relative milliseconds:
`timestamp_ms = round(frame_index * 1000 / source_fps)` using Python ties-to-even
rounding. These are not Unix event timestamps. Future gap durations, resampling,
window intervals and temporal label overlap must use actual timestamps and
document interval/end-point conventions, rather than frame counts alone.
Frame indices remain lineage information. The online target of 15 FPS / about
30 frames / two seconds does not freeze model FPS, duration or resampling.

## 5. Temporal supervision and unresolved decisions

The current `fall` / `non_fall` labels describe videos. A fall video may contain
pre-fall, transition and post-fall states. **A fall-video label does not make every
window an active-fall positive.** Stage 3.2 must define the supervision protocol,
its evidence/annotation sources, event timing and ambiguous cases before any
window labeling. No temporary broadcast-label shortcut is permitted.

All of the following are explicitly `UNDECIDED` in the configuration:

- missing-pose policy; visibility threshold;
- interpolation policy and maximum gap in milliseconds;
- normalization denominator and invalid-scale behavior;
- resampling method and target model FPS;
- window duration/stride in milliseconds and minimum pose coverage;
- temporal-supervision protocol, positive-window overlap threshold and ambiguous-window policy;
- optional motion features, exact transform order and derived schema version.

No value means "use a default". Disabling a future optional operation must itself
be an explicit reviewed decision. `read_contract` rejects missing decision keys;
`require_build_ready` reports unresolved decisions and always rejects construction
in Stage 3.0, even if a caller changes `build_enabled` or replaces every placeholder.
A later implementation must add decision-specific validation under review; arbitrary
non-placeholder values are not treated as approved parameters by this module.

## 6. Output versioning, paths and provenance

Contract version 1 is an infrastructure version, not a processed-data schema.
The derived schema version remains `UNDECIDED`. Future outputs use separate roots:

```text
data/processed/caucafall_v5/<derived_schema_version>/<new_run_id>/...
artifacts/preprocessing/<derived_schema_version>/<new_run_id>/...
```

No directories or artifacts are created by Stage 3.0. A policy/configuration or
code change requires a new derived run; incompatible schema/semantics require a
new schema version. Never mix runs or overwrite prior outputs. Store bulk data
under the already ignored `data/` tree; keep small reproducibility metadata in Git.

`validate_output_path` performs a read-only destination check. It rejects outputs
inside/above immutable raw, interim or extraction-evidence trees, rejects aliases,
symlinks and existing destinations, and requires a separate Stage 3 root. It does
not allocate versions/run IDs or implement publication. Future writers must repeat
checks at write time and publish without overwrite; the helper is not protection
against a concurrent filesystem change after validation.

Each derived sample/window must trace directly, or through immutable manifest joins, to:

- dataset name/version and source-download provenance limitation;
- Stage 2 run ID, manifest/run digests, execution/evidence commits;
- source NPZ path/digest and source-video path/checksum;
- subject ID, original activity, video label and authoritative split;
- source timestamps/frame mapping, and window bounds if applicable;
- derived schema, preprocessing contract/configuration version and configuration hash;
- actual preprocessing code commit when available, dirty status and relevant file
  hashes; if Git identity is unavailable, record that explicitly with code hashes;
- actual runtime/dependency versions, deterministic settings and any applicable seed;
- Train-only fitted-state identity/hash and fitting-source manifest, when applicable;
- supervision protocol/version and annotation lineage when labels are derived;
- run identity, validation/completion status and explicit failures/exclusions.

Do not claim a source NPZ digest was captured in Stage 2 if it was not. A later
authorized build records its input identity baseline without editing Stage 2.
Incomplete derived runs must not be reported as model-ready datasets. The derived
tensor layout and completion/publication implementation remain future work.

## 7. Stage 3 Gate structure

| Gate | Required review outcome before dependent work |
| --- | --- |
| Stage 3.0 | Review handoff identity, contract, firewall and regression evidence; external reviewer decides acceptance |
| Stage 3.1 | Separately authorize permitted-split characterization; record scope and evidence without Test exploration |
| Stage 3.2 | Separately determine/review temporal-supervision protocol; no broadcast video labels |
| Later preprocessing-policy gate | Review missingness, normalization, timing, features, transform order, schema and all unresolved decisions |
| Later implementation/build gate | Review deterministic code, Train-fitting controls, provenance, split/video isolation and tests before a dataset build |
| Stage 3 completion gate | Review derived-data integrity, reproducibility and limitations before any separately authorized training |

Only Stage 3.0 is implemented here. Later gate names state dependencies, not new
approved algorithms or detailed stage numbering. Test application remains blocked
until a future explicit frozen-application contract and authorization exist.

## 8. Focused validation

From the repository root, the new tests use only the standard library:

```sh
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project python -B -m unittest tests.test_stage3_contract -v
```

Tests cover exact/disjoint subject membership, default Train and explicit
Validation selection, Test rejection before I/O, Train-only fitting selection,
identity spoofing, source metadata mutation, unresolved/build locks, and output
path isolation including symlinks. Fixtures contain metadata and dummy path targets,
not simulated preprocessing. Test NPZ/video contents are never opened.

Relevant existing regressions (split/schema and Stage 2 run provenance/lifecycle):

```sh
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --offline --python 3.13.15 --no-project --with numpy==2.2.6 python -B -m unittest tests.test_dataset_split tests.test_pose_raw tests.test_pose_run_preflight tests.test_pose_run_executor -v
```

Stage 2's executor intentionally ties execution to its reviewed commit and full
implementation/test hash map. Adding Stage 3 tests changes that live map; do not
rerun the old executor on the official run to validate this handoff or rewrite its
hashes/logs. Historical run evidence stays associated with its historical code.
