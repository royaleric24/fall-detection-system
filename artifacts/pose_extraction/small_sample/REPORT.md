# Stage 2.4 — Frozen Small-Sample Extraction

Execution complete; ready for Gate Review. Stage 2.1–2.3 were CLOSED / PASS on entry. Stage 2.5 was not started. No commit was made.

## Predeclared sample and selection

All ten entries in [sample_definition.json](sample_definition.json) were fixed before inference. The selection covers all ten original activities, five fall and five non_fall clips, six permitted subjects, seven train clips and three validation clips. Train subjects 8/9 extend beyond the original Stage 1.3 Subjects 1–5 subset. Selection used structural coverage, not newly observed pose availability. All destinations were absent before execution.

The Stage 2.2 Subject.1/Fall forward and Subject.2/Fall backwards pairs remain immutable prior evidence. Neither was rerun or counted among these ten new runs. Subjects 6/7 were not opened, hashed, decoded, inferred or visually inspected; their stored membership appears only in the existing split metadata/tests.

## Per-video results

Pose availability = detected / expected frames. It is diagnostic only, not accuracy or robustness evidence. No availability cutoff was introduced. Longest missing run uses inclusive source frame indices and is reported in frames / seconds.

| Subject | Original activity | Label | Split | Expected / decoded | Detected | Missing | Availability | Longest missing run | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | Fall forward | fall | train | 122 / 122 | 0 | 122 | 0.00% | 122 / 6.10 s | complete |
| 9 | Fall backwards | fall | train | 252 / 252 | 252 | 0 | 100.00% | 0 / 0.00 s | complete |
| 3 | Fall left | fall | train | 171 / 171 | 147 | 24 | 85.96% | 12 / 0.60 s | complete |
| 4 | Fall right | fall | train | 150 / 150 | 150 | 0 | 100.00% | 0 / 0.00 s | complete |
| 5 | Fall sitting | fall | validation | 229 / 229 | 224 | 5 | 97.82% | 5 / 0.25 s | complete |
| 8 | Hop | non_fall | train | 166 / 166 | 156 | 10 | 93.98% | 10 / 0.50 s | complete |
| 9 | Kneel | non_fall | train | 216 / 216 | 203 | 13 | 93.98% | 7 / 0.35 s | complete |
| 3 | Pick up object | non_fall | train | 186 / 186 | 132 | 54 | 70.97% | 18 / 0.90 s | complete |
| 10 | Sit down | non_fall | validation | 234 / 234 | 220 | 14 | 94.02% | 4 / 0.20 s | complete |
| 10 | Walk | non_fall | validation | 224 / 224 | 210 | 14 | 93.75% | 4 / 0.20 s | complete |

Totals: **10 selected, 10 complete, 0 failed, 0 incomplete, 0 unattempted; 1950 decoded frames, 1694 detected poses, 256 missing poses.**

Every row has extracted = decoded = expected frames. The [CSV](summary.csv) and [JSON](summary.json) include exact source-relative AVI paths, source SHA-256, output/result paths, run IDs and output hashes. NPZs are ignored under `data/interim/caucafall_v5/pose_raw_v1/`; result records remain under `artifacts/pose_extraction/single_video/` with mirrored subject/activity/stem hierarchy. No large datasets or model binaries were added to Git.

## Unexpected behavior and limitations

Subject.8 / Fall forward returned no poses: all 122 source frames persisted as explicit missing records. This is diagnostic availability, not an extraction failure. No threshold tuning, sample replacement, filtering or visual error analysis was performed.
The existing runtime emitted MediaPipe feedback-tensor and landmark-projection warnings in native logs. All ten processes exited 0; no decode, inference, malformed-output, publication or pair-validation failure occurred. CPU inference ran outside the restricted sandbox because earlier stages established a macOS graphics-context initialization requirement. No alternate runtime, model or threshold was tried. This does not establish headless portability, real-time throughput or LIVE_STREAM performance.

## Frozen implementation/configuration/model

Before-execution hashes were saved in [freeze_before.json](freeze_before.json). They were checked immediately before and after every selected video, recorded in [execution.jsonl](execution.jsonl), and matched again in [freeze_after.json](freeze_after.json). The extractor, supporting code, contract, protocol, split, inventories, model and sample definition were not edited during the run. The two prior evidence pairs also retain their original hashes.

| File | SHA-256 before extraction (unchanged after) |
| --- | --- |
| `ml/datasets/extract_pose_video.py` | `d8ba41d765e09355630914d74ccc6f32dcc041ab99e171af3efb7fcbdec0c44f` |
| `ml/datasets/pose_raw.py` | `f348b8bf26dac67160ac3ede67d3b49b4096a3607f2368615d21cfc951a57f67` |
| `ml/datasets/pose_runtime.py` | `7f00d55dfcbd3ed12442764c830f36ff6e7f3c6cac5691a98328ea8dd5cf37cf` |
| `ml/datasets/assign_dataset_split.py` | `b64c55bf29b0a567aeb2b7f365e92233d432614ffe957784557674786df5999d` |
| `ml/datasets/inspect_caucafall.py` | `aefb8c123d0a6457d5c37bdc202285d1c857d23941fe1e8bc1aac524d4d7b2fb` |
| `docs/pose_extraction_contract.md` | `4a4547bec9b40d5ba287aaa142b6ca613c0c3b5bdba57939060370a8b80bf783` |
| `docs/dataset_protocol.md` | `e8197e9fedc41185dac1bd4adae0f5552017fa9c4a36d3be53dc586c6d39a8bd` |
| `configs/dataset_split.json` | `c1de39a02dd1ca2fe6cb1bd06c594ed9149477b4bc65c8774de527bd53a140d3` |
| `artifacts/dataset_inspection/inventory.csv` | `855406c628aba7b0f6c53935393237ec72cb562b47381170ba0c72c921210bc8` |
| `artifacts/dataset_inspection/split_inventory.csv` | `323eb716b8714c6503ddba14f6a310214931872bb1c7ed4268f0ac7f9bbebb56` |
| `ml/checkpoints/pose_landmarker_full.task` | `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1` |
| `artifacts/pose_extraction/small_sample/sample_definition.json` | `38a4ee9e34a4e8e9fb0401e35bbc69c7012bc982bf984a32b0d6635076a01200` |

Git commit at freeze: `f7eecb0d0de246f932001daf910b4ebd9d0898a7`; worktree dirty: `True`. The worktree already contained uncommitted Stage 2.1–2.3 changes; hashes identify the executed code.

Runtime for all ten videos: Python 3.13.15, MediaPipe 0.10.35, opencv-contrib-python 4.12.0.88 (cv2 4.12.0), NumPy 2.2.6, Full float16 v1, VIDEO, CPU, one pose, confidence 0.5/0.5/0.5, segmentation disabled. A separate extractor process creates a fresh tracker for each source. Per-video results retain full resolved package versions and platform information.

## Commands actually executed

The following ten independent extractor commands ran sequentially from the repository root on the host, with native output captured in `logs/01.txt` through `logs/10.txt`. Exact commands and exit codes are also in [commands.txt](commands.txt) and [execution.jsonl](execution.jsonl).

```sh
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache MPLCONFIGDIR=/private/tmp/caucafall-matplotlib uv run --offline --python 3.13.15 ml/datasets/extract_pose_video.py 'data/raw/caucafall_v5/CAUCAFall/Subject.8/Fall forward/FallForwardS8.avi'
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache MPLCONFIGDIR=/private/tmp/caucafall-matplotlib uv run --offline --python 3.13.15 ml/datasets/extract_pose_video.py 'data/raw/caucafall_v5/CAUCAFall/Subject.9/Fall backwards/FallBackwardsS9.avi'
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache MPLCONFIGDIR=/private/tmp/caucafall-matplotlib uv run --offline --python 3.13.15 ml/datasets/extract_pose_video.py 'data/raw/caucafall_v5/CAUCAFall/Subject.3/Fall left/FallLeftS3.avi'
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache MPLCONFIGDIR=/private/tmp/caucafall-matplotlib uv run --offline --python 3.13.15 ml/datasets/extract_pose_video.py 'data/raw/caucafall_v5/CAUCAFall/Subject.4/Fall right/FallRightS4.avi'
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache MPLCONFIGDIR=/private/tmp/caucafall-matplotlib uv run --offline --python 3.13.15 ml/datasets/extract_pose_video.py 'data/raw/caucafall_v5/CAUCAFall/Subject.5/Fall sitting/FallSittingS5.avi'
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache MPLCONFIGDIR=/private/tmp/caucafall-matplotlib uv run --offline --python 3.13.15 ml/datasets/extract_pose_video.py data/raw/caucafall_v5/CAUCAFall/Subject.8/Hop/HopS8.avi
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache MPLCONFIGDIR=/private/tmp/caucafall-matplotlib uv run --offline --python 3.13.15 ml/datasets/extract_pose_video.py data/raw/caucafall_v5/CAUCAFall/Subject.9/Kneel/KneelS9.avi
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache MPLCONFIGDIR=/private/tmp/caucafall-matplotlib uv run --offline --python 3.13.15 ml/datasets/extract_pose_video.py 'data/raw/caucafall_v5/CAUCAFall/Subject.3/Pick up object/PickupobjectS3.avi'
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache MPLCONFIGDIR=/private/tmp/caucafall-matplotlib uv run --offline --python 3.13.15 ml/datasets/extract_pose_video.py 'data/raw/caucafall_v5/CAUCAFall/Subject.10/Sit down/SitDownS10.avi'
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache MPLCONFIGDIR=/private/tmp/caucafall-matplotlib uv run --offline --python 3.13.15 ml/datasets/extract_pose_video.py data/raw/caucafall_v5/CAUCAFall/Subject.10/Walk/WalkS10.avi
```

## Artifact and consistency validation

Every selected output had both a final NPZ and its matching final `status=complete` result JSON. Each NPZ was independently reloaded with `allow_pickle=False` through the frozen `load_validated` function. Checks covered exactly four arrays, exact dtypes/shapes, original frame indices, 50-ms source-rate timestamps, finite detected poses and all-NaN missing poses. Counts, labels/split, source checksum, output location, array metadata and missing intervals matched the final result. Source FPS/resolution matched the inventory (20 FPS, 720×480).

No resampling, normalization, interpolation/fill, visibility rejection, derived features, windows, frame labels or training ran. The frozen extractor implementation and passing frame/missingness invariants supply the evidence; no independent pose-quality assessment or tuning was performed. No output was repaired, replaced or rerun.

## Tests before and after

```sh
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache uv run --offline --python 3.13.15 --no-project --with numpy==2.2.6 python -B -m unittest discover -s tests -v
```

Before extraction: **63 tests passed**, exit 0 ([log](tests_before.txt), [execution metadata](tests_before.json)). After extraction: **63 tests passed**, exit 0 ([log](tests_after.txt), [execution metadata](tests_after.json)). No tests or extractor code were added/changed in Stage 2.4.

## Git state

`git diff --stat` below includes prior tracked Stage 2 changes and excludes untracked metadata/code. This Stage 2.4 task created only sample metadata/logs/summaries, ten per-video result JSONs and ten ignored NPZs.

```text
 README.md                                  | 15 +++--
 docs/architecture.md                       | 14 +++--
 ml/datasets/README.md                      | 88 ++++++++++++++++++++++++++++++
 ml/datasets/validate_pose_compatibility.py | 21 +------
 4 files changed, 109 insertions(+), 29 deletions(-)
```

`git status --short`:

```text
 M README.md
 M docs/architecture.md
 M ml/datasets/README.md
 M ml/datasets/validate_pose_compatibility.py
?? artifacts/pose_extraction/
?? docs/pose_extraction_contract.md
?? ml/datasets/extract_pose_video.py
?? ml/datasets/pose_raw.py
?? ml/datasets/pose_runtime.py
?? tests/test_pose_failure_paths.py
?? tests/test_pose_raw.py
```
