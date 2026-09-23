# Stage 2.6 preflight preparation

This layer prepares a fresh run namespace without running MediaPipe. It does not
implement a bulk execution/resume command, generate pose outputs, or claim full-run
completion. The approved pose-extraction contract remains unchanged.

## Review and commit boundary

Frozen core checkpoint: `dd30d02100d3d77eb73ceb8b2e5ee8b90a6564e0`.

First review the orchestration and tests and run the lightweight suite. Commit
these additions in a separate orchestration/preflight commit only after explicit
approval. Run-creating preflight refuses any dirty tracked or untracked files,
requires checkpoint ancestry, and compares frozen core/configuration bytes with
that checkpoint. Run provenance records the actual subsequent clean HEAD, as well
as the original core checkpoint; it never substitutes the older development run's
Git identity.

Read-only check (permitted before committing):

```sh
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --python 3.13.15 ml/datasets/preflight_pose_run.py --check
```

This hashes source AVI bytes, verifies exact inventory/split scope, inspects the
installed pinned runtime and model, and runs the lightweight tests. It creates no
run ID, run directory, persisted checksum manifest or NPZ. A successful check on a
dirty worktree is not a successful clean-run preflight.

After the reviewed orchestration commit, run-creating preflight is:

```sh
UV_CACHE_DIR=/private/tmp/caucafall-stage26-cache uv run --python 3.13.15 ml/datasets/preflight_pose_run.py
```

Neither command imports or initializes MediaPipe. OpenCV is imported only to
verify its runtime version; no video is decoded. Exact dependency versions and
the complete installed distribution list are recorded. The test suite uses the
same interpreter, with its exact command and output retained in run provenance
and `tests_before.txt`. Source/model/implementation bytes and Git state are
checked again after the tests, before publishing preflight artifacts.

## Run-scoped storage

A new UUID is generated only for successful run-creating preflight:

```text
artifacts/pose_extraction/runs/<run_id>/
  extraction_run.json
  manifest.csv
  source_checksums.csv
  tests_before.txt
  results/

data/interim/caucafall_v5/pose_raw_v1_runs/<run_id>/
  Subject.N/<activity>/<video>.npz
```

The NPZ root is declared but not created by preflight. Later per-video result JSONs
belong under `results/Subject.N/<activity>/<video>.json`. The future full extraction
report belongs at `<run directory>/FULL_EXTRACTION_REPORT.md`; no report claiming
extraction exists at preflight. All 100 manifest rows begin `pending`, with unknown
extraction counters left empty. There are 19,877 expected frames and 60/20/20
train/validation/test videos. All source paths and SHA-256 values are recorded in
numeric subject/activity/path order. This establishes a current byte baseline,
not authenticated download provenance or historical immutability.

Preflight artifacts are staged together and published into a fresh directory;
existing destinations are refused. Historical development records and NPZs are
never copied, relabeled, overwritten or treated as completed outputs of this run.

## Frozen function reuse and execution boundary

The adapter reuses `assign_rows` for all 100 inventory/split identities and
cross-checks the 80 train/validation metadata records against `source_metadata`.
The latter intentionally retains its Stage 2.2 test-subject rejection. For the
explicitly authorized full-run scope, the adapter maps the validated inventory's
metadata fields for Subjects 6/7 without changing that production function or
interpreting their pose data.

It reuses `sha256`, `output_path`, `runtime_provenance`, and frozen schema/model
constants. A later authorized execution layer must invoke the existing
`extract_arrays` (which owns `verify_capture`, sequential decoding and a fresh
tracker), `publish_video_pair` (validated NPZ first, complete result JSON last),
and `pose_raw.load_validated`. No extraction algorithm is duplicated here.

Before any future inference, revalidate this run's source/model/runtime/core and
orchestration identity, its pending manifest, clean starting commit evidence and
namespace ownership. Preflight-generated metadata itself makes the working tree
dirty; it must not conceal unrelated code changes. Source metadata is checked
against the actual capture by the frozen `extract_arrays` path before inference.
The full-run executor, manifest transitions, final independent validation and
aggregate report are not executed by this preflight-only entry point.

Subjects 6/7 are permitted for mechanical extraction/schema/completeness checks
only. Their outputs must not influence extraction, preprocessing, model or alert
policy. No Stage 3 work is authorized by this module.
