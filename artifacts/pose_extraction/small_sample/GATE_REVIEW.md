# Stage 2.5 — Small-Sample Validation & Pipeline Freeze Review

**Verdict: READY FOR STAGE 2.6**

Audit date: 2026-09-18. This verdict concerns engineering and data-contract readiness, not pose availability, classification performance, or portability. No evidence inconsistency or unresolved extraction-engineering blocker was found. Recommend authorizing Stage 2.6 through a separate, explicit task. This report does not start or execute Stage 2.6.

## Scope and method

This was a read-only audit of existing evidence, followed by creation of this report. No MediaPipe inference, raw-video reads, extraction, preprocessing, training, threshold changes, sample substitutions, or evidence regeneration occurred. Subjects 6/7 were not accessed; their membership was checked only in existing inventory/split metadata.

A temporary audit script outside the repository imported only the production NumPy schema module, `ml.datasets.pose_raw`, and reloaded all ten Stage 2.4 NPZs plus both Stage 2.2 NPZs with `load_validated`. Counts and inclusive missing intervals were independently recomputed from `pose_detected`, without calling the extractor's missing-run helper. SHA-256 digests were recomputed from the existing artifact and frozen-file bytes. No implementation or tests were changed.

## Sample integrity — PASS

[sample_definition.json](sample_definition.json) contains exactly ten unique videos. Every source-relative path exists exactly as selected in the inspected inventory; subject, original activity and binary label agree with that inventory. Subject membership agrees with `configs/dataset_split.json`.

- All ten original activities occur exactly once, with their original names.
- Five clips are `fall` and five are `non_fall`.
- Seven clips are train and three are validation.
- Selected subjects are 3, 4, 5, 8, 9 and 10; neither 6 nor 7 is selected.
- The ten entries in [execution.jsonl](execution.jsonl), [commands.txt](commands.txt), both summaries and the ten per-video logs match the predeclared set and order exactly.
- Predeclaration and the initial freeze precede every recorded execution. Every execution exited 0 and records `validation=PASS`. No failed video was replaced; no selected video was omitted or rerun in the recorded execution set.
- The two Stage 2.2 development videos are outside this sample and were not counted as new executions.

The sample definition was not edited; its current SHA-256 matches both freeze records and every per-video before/after hash map.

## Artifact-pair and schema integrity — PASS

All ten expected final NPZs exist with matching final result JSONs having `status=complete`. Source identity, split, dimensions, source FPS, output location, run ID, array metadata and recorded source checksum agree across the relevant inventory, result, execution and summary records. Every logged result JSON equals its corresponding final result JSON. Current NPZ and result-file hashes match the execution record and JSON summary.

Each NPZ passed the production `pose_raw_v1` validator with the inventory FPS and frame count. Every archive has exactly four unique arrays: `frame_index`, `timestamp_ms`, `pose_detected`, and `landmarks`. Dtypes and shapes match the frozen schema; no frame-level labels, object arrays or extra entries are present. Loading uses `allow_pickle=False`.

Frame indices are contiguous from zero. Timestamps obey the unchanged `round(frame_index * 1000 / source_fps)` formula and increase by 50 ms at the recorded 20 FPS. Detected landmarks are finite; every missing frame has all 132 landmark values set to NaN. For each video:

```text
T = expected_frame_count = decoded_frame_count = extracted_frame_count
pose_detected_count + pose_missing_count = T
```

Independent recomputation agrees with the per-video results and both Stage 2.4 summaries:

| Subject | Original activity | Split | T / decoded / extracted | Detected | Missing | Longest missing run (frames) |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | Fall forward | train | 122 / 122 / 122 | 0 | 122 | 122 |
| 9 | Fall backwards | train | 252 / 252 / 252 | 252 | 0 | 0 |
| 3 | Fall left | train | 171 / 171 / 171 | 147 | 24 | 12 |
| 4 | Fall right | train | 150 / 150 / 150 | 150 | 0 | 0 |
| 5 | Fall sitting | validation | 229 / 229 / 229 | 224 | 5 | 5 |
| 8 | Hop | train | 166 / 166 / 166 | 156 | 10 | 10 |
| 9 | Kneel | train | 216 / 216 / 216 | 203 | 13 | 7 |
| 3 | Pick up object | train | 186 / 186 / 186 | 132 | 54 | 18 |
| 10 | Sit down | validation | 234 / 234 / 234 | 220 | 14 | 4 |
| 10 | Walk | validation | 224 / 224 / 224 | 210 | 14 | 4 |

Totals: **1,950 frames; 1,694 detected; 256 missing; 10 complete, 0 failed, 0 incomplete, 0 unattempted.**

All independently reconstructed inclusive missing intervals equal the result records:

| Subject / activity | Missing intervals (inclusive frame indices) |
| --- | --- |
| 8 / Fall forward | [0, 121] |
| 9 / Fall backwards | none |
| 3 / Fall left | [0, 3], [5, 5], [7, 8], [78, 78], [83, 83], [101, 103], [105, 116] |
| 4 / Fall right | none |
| 5 / Fall sitting | [54, 58] |
| 8 / Hop | [0, 9] |
| 9 / Kneel | [0, 2], [101, 101], [106, 106], [109, 109], [112, 118] |
| 3 / Pick up object | [0, 4], [6, 6], [12, 16], [19, 19], [21, 21], [23, 23], [25, 25], [60, 60], [62, 74], [92, 93], [95, 112], [114, 118] |
| 10 / Sit down | [47, 49], [73, 76], [82, 83], [85, 85], [89, 91], [206, 206] |
| 10 / Walk | [0, 1], [3, 4], [7, 10], [12, 15], [81, 81], [107, 107] |

The result records store intervals, not a separate longest-run field. Longest runs derived from those intervals agree with the independently scanned NPZ masks and summary frame/second values. The summaries do not store interval lists; they were not treated as interval evidence.

## Provenance freeze — PASS

All 12 SHA-256 entries in [freeze_before.json](freeze_before.json) and [freeze_after.json](freeze_after.json) are identical. Each of the ten execution records repeats that same complete map before and after its run. Fresh hashes of the current files also match:

- `ml/datasets/extract_pose_video.py`, `pose_raw.py`, `pose_runtime.py`, `assign_dataset_split.py`, and `inspect_caucafall.py`;
- `docs/pose_extraction_contract.md` and `docs/dataset_protocol.md`;
- `configs/dataset_split.json`;
- `artifacts/dataset_inspection/inventory.csv` and `split_inventory.csv`;
- `ml/checkpoints/pose_landmarker_full.task`;
- the predeclared sample definition.

Per-video provenance agrees with the corresponding frozen code/configuration hashes and model digest. All ten records preserve Python 3.13.15, MediaPipe 0.10.35, OpenCV-contrib 4.12.0.88, NumPy 2.2.6, VIDEO, CPU, one pose, confidence 0.5/0.5/0.5, and segmentation disabled. Git provenance is commit `f7eecb0d0de246f932001daf910b4ebd9d0898a7`, dirty worktree `true`; the file hashes identify the uncommitted extraction implementation.

No provenance discrepancies were found or repaired. Source digests were cross-checked between existing records; raw videos were not rehashed in this audit. The original source-download provenance remains **locally supplied; unverified**. This audit does not authenticate its origin or establish historical raw-byte immutability before a checksum baseline existed.

## Stage 2.2 evidence preservation — PASS

All four existing files (two NPZs and two result JSONs) match their `prior_evidence_sha256` entries in both freeze records. Both NPZs also pass the current production schema validator, and their counts and missing intervals match their original results:

- Subject.1 / Fall forward: 190 frames, 94 detected, 96 missing; intervals [0, 91], [133, 133], [136, 138].
- Subject.2 / Fall backwards: 119 frames, 119 detected, 0 missing; no missing intervals.

Neither pair was modified, overwritten or rerun. Their historical extractor provenance is preserved, rather than rewritten to reference the later Stage 2.3 implementation.

## Test and finalization evidence — PASS with evidence limitation

[tests_before.txt](tests_before.txt) and [tests_after.txt](tests_after.txt) each contain the same 63 named tests, every one marked `ok`, followed by `Ran 63 tests` and `OK`. Both execution metadata files record exit 0 and the same command:

```sh
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache uv run --offline --python 3.13.15 --no-project --with numpy==2.2.6 python -B -m unittest discover -s tests -v
```

The before check finished at 2026-09-17 08:53:00.271932 UTC, before all ten runs. The after check began at 2026-09-17 15:53:04.476315 UTC, after all ten runs. The freeze snapshots bracket both checks; per-run hashes and current hashes confirm the same extractor/support code at the recorded checkpoints.

The Stage 2.4 report explicitly records no extractor/test changes. Both logs have identical test identities; the current five test modules contain those 63 test methods, and their filesystem modification times precede the before check (latest: 08:45:00.840242 UTC). This supports the recorded unchanged-test workflow. **Historical test-file hashes were not captured**, so byte-for-byte test-content identity throughout the interval cannot be independently proven from these artifacts. No contradictory evidence was found; no retrospective hashes were manufactured. No new tests were added or suite rerun during Stage 2.5.

The frozen publication code stages and validates both artifacts, publishes the final NPZ, and publishes the final complete result JSON last. Existing regression evidence covers publication order, failures, interruption, orphan rejection and output protection. Consumers must require both a validated NPZ and a matching complete result. An uncatchable termination can leave an orphan NPZ; it is incomplete. Full crash recovery/resume remains deferred.

## Data quality warning and stage boundaries

**DATA QUALITY WARNING — Subject.8 / Fall forward: 122 decoded, 0 pose detected, 122 missing, one 122-frame missing run [0, 121].**

This is a valid complete extraction with explicit missingness, not an extraction failure. No minimum coverage threshold is imposed. The video remains in the frozen sample and split; it was not dropped, replaced, interpolated or used to tune MediaPipe settings.

Stage 3 must design an explicit policy for long-run and all-window pose missingness. That policy is not designed here. Pose availability is diagnostic, not accuracy or readiness evidence. Existing native runtime warnings and the measured macOS execution-environment constraint remain documented in the Stage 2.4 report; this audit makes no new portability or performance claim.

## Authorization recommendation

Recommend **authorizing Stage 2.6** using the frozen extraction behavior and configuration. No unresolved engineering blocker was identified in the audited sample, schema, artifact pairs or provenance. The missing historical test hashes are an evidence limitation, not an observed modification or data-contract failure.

The separate Stage 2.6 task must still satisfy the frozen contract's planned full-run manifest/provenance requirements and source-checksum preflight covering exactly the 100 inventoried AVIs, while preserving existing evidence and completion-marker semantics. Those full-run facilities are not claimed to exist already. The current development CLI's held-out guard remains unchanged. Only a later explicit authorization may permit mechanical extraction of Subjects 6/7 under the frozen protocol; their results must not drive pipeline or policy tuning.

No Stage 2.6 work, full-dataset extraction or commit was performed by this review.

## Files inspected and changes made

Inspected the frozen contract/protocol and split, production schema/publication code, shared-runtime and supporting-code hashes, both inspected inventory hashes, model hash, all small-sample definition/freeze/execution/command/log/summary/report files, both before/after test logs and metadata, the five lightweight test modules, all ten Stage 2.4 NPZ/result pairs, and both Stage 2.2 NPZ/result pairs. Exact source and artifact paths are retained in the unchanged sample definition and JSON summary.

**Only this `GATE_REVIEW.md` was created in the repository for Stage 2.5.** Existing implementation, tests, documentation, configuration and evidence files were preserved. The worktree already contains uncommitted prior-stage changes; they are not Stage 2.5 changes.
