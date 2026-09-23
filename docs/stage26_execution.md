# Stage 2.6 execution layer

This module consumes one existing official preflight run. It never assigns a run
ID or reuses development outputs. Source review and an execution commit are
required before creating the official run; neither real inference nor official
run creation is part of this implementation task.

## Future authorized invocation

After review and a separate explicitly authorized commit, create the run with
`preflight_pose_run.py` from that clean committed state. Then use its returned run
path and the full reviewed execution commit SHA:

```sh
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --python 3.13.15 ml/datasets/execute_pose_run.py artifacts/pose_extraction/runs/<run_id> --execution-commit <reviewed-full-commit-sha>
```

The commit argument is required; the executor does not infer approval from the
current HEAD. HEAD and preflight `git_commit` must equal that supplied SHA. The
frozen core/configuration files must still match checkpoint
`dd30d02100d3d77eb73ceb8b2e5ee8b90a6564e0` byte for byte. Preflight records the new
executor and all tests, so older preflight artifacts lacking that identity are
rejected rather than upgraded in place.

## Validation before inference

The executor validates a canonical UUID run directory under the repository's
`artifacts/pose_extraction/runs/`, the required inputs, and the matching ignored
`data/interim/caucafall_v5/pose_raw_v1_runs/<run_id>/` NPZ root. Symlink redirection,
namespace escape and unknown artifact files are rejected. It checks all 100
ordered manifest rows against the frozen metadata adapter, including labels,
splits, source properties, source SHA-256 and mirrored output paths.

The provenance identity checksum, complete expected implementation hash map,
model, runtime configuration and complete resolved package versions must match.
All 100 actual AVI byte checksums are checked against the run baseline before
inference and again in the final audit. The preflight test log must match its
recorded bytes and hash. No MediaPipe import happens until this gate passes.

Git status uses an exact file allowlist, not a blanket exemption for `artifacts/`:
only this run's declared input/output metadata, result paths, final reports and
execution sentinel may be dirty. Immutable preflight evidence within that list
still has its hashes/content checked. Unrelated tracked or untracked changes,
including code/config changes or another run's files, stop execution. Ignored
NPZs are checked directly for namespace, schema, identity association and ownership.

## Frozen production reuse

`preflight_pose_run.inventory_metadata`, `source_checksums`, `manifest_rows`,
`implementation_hashes`, `runtime` and `run_tests` supply shared orchestration
validation. The extractor's `sha256` and `output_path` guard identities and paths.
For each pending video the executor calls:

1. `extract_pose_video.extract_arrays`, which owns `verify_capture`, sequential
   decoding, BGR-to-RGB, tracker creation/reset, timestamps and pose encoding;
2. `extract_pose_video.publish_video_pair`, which validates/stages and publishes
   the NPZ before publishing its matching complete result JSON;
3. `pose_raw.load_validated`, which independently reloads the final archive with
   pickle disabled and validates all four arrays.

No decoding, landmark encoding, MediaPipe settings, timestamp generation or NPZ
publication implementation is duplicated. The executor independently counts the
NPZ mask and computes inclusive missing intervals and longest runs. It compares
these with the result and manifest, and checks run/source/output/provenance identity.
Subjects 6/7 use the same extraction primitives with validated full-run metadata;
the unchanged Stage 2.2 CLI guard is not used to authorize them.

## Ledger, failures and conservative resume

The manifest is the authoritative ordered ledger. Changes are written to a
same-directory temporary file, flushed/fsynced, then atomically replaced. A failed
write leaves the previous ledger intact; temporary files are removed on catchable
failures.

- `pending → running` is persisted before extraction starts.
- `running → complete` occurs only after both final artifacts independently pass.
- Known extraction errors become `failed` with exact category/message/frame index
  and available counters. Other pending videos may continue with unchanged code.
- A caught interruption becomes `incomplete` / `interrupted_run`; the run stops.
- Provenance changes, validation inconsistencies, unexpected defects or ledger
  persistence failures stop execution for review. No code is patched or resumed.

Completed rows are skipped only after full pair validation. Pending rows must
have neither final artifact. Failed rows are retained, not retried or replaced.
Stale `running` and `incomplete` rows are treated as unfinished and stop for
review. This implementation deliberately does not delete/restart their artifacts.
A pair published immediately before a crash is not inferred complete from a
stale ledger. No automatic promotion or historical-output reuse occurs.

An exclusive local `executor.lock` sentinel prevents concurrent writers. It is
removed after catchable exits. An uncatchable termination leaves it in place;
subsequent execution refuses it. Ownership/recovery needs separate review—there
is no automatic lock deletion or cross-file crash recovery. A crash can leave an
orphan NPZ or a valid pair beside a non-complete ledger; neither establishes full
run completion. Power-loss durability across several files is not claimed.

## Final audit and reports

After all pending rows are attempted, the lightweight suite runs again and its
output is stored in `tests_after.txt`. Code/config/source/model/runtime checks are
repeated; every complete pair is independently reloaded again. Only 100 validated
complete pairs plus successful final audits can make the run complete.

`summary.json` and `FULL_EXTRACTION_REPORT.md` include status counts, expected
frames, validated stored-frame counts, detected/missing totals, per-split/activity
summaries, per-video availability and longest missing runs, all missing intervals
(machine-readable), zero-pose videos, and failures. No long-run coverage cutoff
is introduced. In a non-complete run, overall availability is unavailable and
validated-pair totals are explicitly distinguished from failed-prefix counters.

Reports are written before the final `extraction_run.json` completion update.
Final test results, audit outcomes and Git status are recorded there. A completed
run reopened by the executor must still pass source/code/runtime/pair checks and
have matching final test evidence and reports.

Pose availability is diagnostic only. Subject.8 / Fall forward remains in the
scope even if all its poses are missing. Held-out statistics must not drive
thresholds, missingness policy, preprocessing, model or alert decisions. No
qualitative held-out analysis or Stage 3 work belongs to this executor.
