# Stage 3.2i implementation and execution boundary

This implements the unchanged [analysis specification](stage32i_analysis_spec.md)
frozen at `013418a92494b8350cce8856e339c7ad3f44514a`. Implementation-freeze work
uses synthetic fixtures only. The first genuine agreement computation requires
separate authorization after the analysis-code commit exists.

## Interfaces

`ml/preprocessing/stage32i_agreement.py` separates five operations:

1. `load_genuine()` validates only the fixed Stage 3.2h source run, schema,
   24 FINALIZED originals, and raw-freeze identities. Its public validation
   summary contains counts and provenance only. It does not compute agreement.
2. `construct_pairs()` pairs `record.source.neutral_clip_id` after the existing
   completion checker succeeds. It never pairs by filenames, order or decisions.
3. `event_metrics()`, `status_metrics()`, `temporal_metrics()` and
   `boundary_summaries()` are deterministic calculations. These pure functions
   support synthetic sample sizes; the genuine path always requires 12 pairs.
4. `analyze_pairs()` creates the derived tables and audits their denominators.
5. `write_artifacts()` serializes the required 15 logical outputs.
   `run_genuine()` additionally manages code provenance and fresh post-analysis
   source verification. No source annotation is ever written.

Categories, matrix orientation and boundary order follow the specification.
Rows are A01 and columns A02. Table rows sort by canonical identity followed by
frozen boundary order. Within each clip the three boundaries remain repeated
measurements, never an enlarged independent sample.

Input boundaries use the existing Stage 3.2e schema: non-observed coordinates
must be **absent**, including `preferred_timestamp_ms`; explicit null is invalid
in the source. Observed coordinates are integer zero-based AVI indices with
`0 <= E <= P <= L < frame_count`. Milliseconds are derived using 20 FPS / 50 ms
per frame. The loader obtains counts/FPS from frozen pilot metadata and never
decodes video, loads pose arrays or visits held-out subjects.

Internally `None`, JSON `null` and CSV `NA` mean unavailable. Numeric zero remains
zero, and CSV booleans use `true`/`false`. Undefined coefficients carry an explicit
reason. For non-comparable rows, source statuses and any genuinely recorded
observed coordinates are retained for provenance, but **all derived temporal
metrics are NA**. Gap context is distinguished by `temporal_comparable` and
`interval_overlap_any`; adjacent disjoint intervals have a zero blank gap.

The Stage 3.2i `inter_interval_blank_gap_frames` never calls or changes the
historical Stage 3.2e endpoint-distance definition. Quartiles use an explicit
Hyndman-Fan Type 7 interpolation. Chance terms for kappa and AC1 are evaluated
with exact rational arithmetic before reporting floats; K is always 3.

Every summary has explicit denominators. The discrepancy inventory is one row
per flagged clip and boundary, with only the four objective flags in Section 22.
An event-presence flag may repeat across a clip's boundary rows; these rows are
not independent samples. No severity, adjudication or supervision fields exist.

## Provenance and failure behavior

The fixed source run is `stage32h-6747d0191488418d932955fda609ec0e`, under
`data/annotation_runs/stage32f_train_pilot_v1`. No alternative input root or
recursive dataset search is provided. Symlinked inputs and unexpected finalized
storage entries are rejected. The manifest digest is verified before loading
annotation records. The existing `require_agreement_ready()` performs the full
raw-freeze/schema check; it receives the **source runtime commit**
`725d6cd063c307fb50ddc2725ecfd1666c7f17b5`.

The genuine run is linked through the unchanged active marker, runtime binding
and each annotator's run-prefixed session ID. `creation.annotation_run_id` remains
the historical pilot version, not the genuine run ID. The historical launcher
requires HEAD to equal its runtime commit, so it is not used as an analysis
launcher. The analysis loader verifies the historical binding and frozen files
without rebinding or changing that launch rule.

`analysis_spec_version` is the full specification-freeze commit identity.
`analysis_code_commit` must be the current clean committed analysis checkout,
descended from the specification freeze and distinct from the source runtime.
The implementation and its tests must exist in that commit. Run ID and ISO 8601
UTC timestamp (`Z`) are explicit inputs, so no sampling or hidden random state
is involved. JSON uses sorted keys, indentation and no non-finite numbers.

Outputs are exclusively created under
`artifacts/temporal_annotation/stage32i_agreement/<analysis_run_id>/`.
An existing output directory is rejected. Until all output generation and fresh
source verification succeed, the manifest/report remain HOLD. Verification is
repeated after final certification metadata is written; failure changes the
completion claim to HOLD/FAIL and raises an error. It never repairs a source.
The process gate has no agreement-performance threshold.

## Commands

Use Python 3.13.15, consistent with the existing project runtime. The core
analyzer uses the Python standard library and existing repository modules.

Safe structural preflight (no agreement calculation):

```sh
python -B -m ml.preprocessing.stage32i_agreement preflight
```

Synthetic tests (the test harness blocks access to the repository `data/` tree):

```sh
python -B -m unittest tests.test_stage32i_agreement -v
```

Future genuine execution, **only after separate authorization**:

```sh
python -B -m ml.preprocessing.stage32i_agreement run \
  --analysis-code-commit "$ANALYSIS_CODE_FREEZE_SHA" \
  --analysis-run-id "$ANALYSIS_RUN_ID" \
  --analysis-timestamp "$ANALYSIS_UTC_TIMESTAMP" \
  --plots
```

`--plots` supports the Section 32 PNG diagnostics using optional Matplotlib.
Matplotlib 3.11.2 is the tested plotting version. To run the full synthetic
suite including PNG generation in an isolated environment:

```sh
uv run --python 3.13.15 --no-project --with matplotlib==3.11.2 \
  python -B -m unittest tests.test_stage32i_agreement -v
```

The flag fails before annotation loading if Matplotlib is unavailable. Without
it, the 15 required tables/JSON/report files are still supported and the manifest
records that plots were not requested. Plots contain individual observations,
fixed axes/direction and no quality bands. Matplotlib's version is recorded when
used. Plot tests use synthetic data and never create official result artifacts.

Do not execute `run` during implementation freeze, even as a smoke test.

## Implementation-freeze verification

Executed on Python 3.13.15: **51 synthetic tests passed**, including temporary
PNG generation with Matplotlib 3.11.2. The following existing regressions also
passed (**48 tests**), using the established OpenCV 4.12.0.88 runtime:

```sh
python -B -m unittest \
  tests.test_stage32e_manual_annotation_protocol \
  tests.test_stage32h_readiness \
  tests.test_stage32g_execution_protocol.Stage32gExecutionTests.test_frozen_pilot_is_exactly_twelve_train_clips_in_pinned_order \
  tests.test_stage32g_execution_protocol.Stage32gExecutionTests.test_completion_rejects_partial_and_single_annotator \
  tests.test_stage32g_execution_protocol.Stage32gExecutionTests.test_dual_12_of_12_passes_structural_completion_without_agreement \
  tests.test_stage32g_execution_protocol.Stage32gExecutionTests.test_unexpected_duplicate_protocol_and_invalid_record_fail \
  tests.test_stage32g_execution_protocol.Stage32gExecutionTests.test_pilot_tool_source_owner_and_state_mismatch_fail \
  tests.test_stage32g_execution_protocol.Stage32gExecutionTests.test_raw_freeze_and_agreement_guard_require_exact_current_hashes -v
```

Genuine preflight was separately exercised with computation functions replaced
by failing stubs and an input access allowlist: 24 original finalized files and
three run/freeze control files. It passed without calling any metric function.
No genuine agreement computation, official result artifact, new raw freeze,
adjudication or downstream supervision was produced during implementation freeze.
