# Stage 5 — controlled normalization ablation and pose-frame dropout

Stage 5 starts from accepted Stage 4 commit
`a5fb6f23aa7965fa5b6519f0900642b4eedf40fb`. The normalized checkpoint was reused
without retraining or modification. Exactly one new model training run was made.
All metrics below are Validation results, with positive class **fall**, threshold
0.5, 20 clips (10 fall, 10 non_fall), subjects 5/10. Train remains 59 usable clips
(29/30), subjects 1/2/3/4/8/9; the same all-missing clip is excluded in both runs.
Test 6/7 remains sealed. No raw pose file was modified.

## Normalization ablation

The only configuration difference is `preprocessing.normalize_pose=False`.
Both runs use the same 132-input, 64-hidden, one-layer unidirectional GRU and linear
head; seed 42; Adam lr 0.001, weight decay 0.0001; batch 8; BCEWithLogitsLoss;
max 50 epochs; Validation-loss early stopping with patience 8 and strict improvement.
Causal forward-fill up to five frames, zero fill afterward, visibility handling,
full clip lengths, subject split, exclusions and threshold are unchanged. The
same training loop is reused; an explicit ablation flag permits the existing
Stage 3.3 normalization switch, preserving the Stage 4 default guard.

| Variant | Best epoch | Validation loss | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No normalization | 16 | 0.01966907 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Normalization (Stage 4) | 5 | 0.29209915 | 0.9000 | 1.0000 | 0.8000 | 0.8889 |

**Delta F1 = normalized - unnormalized = -0.1111** (−11.11 percentage points).
Confusion matrices, true rows/predicted columns, order [non_fall, fall]:
no normalization `[[10,0],[0,10]]`; normalization `[[10,0],[2,8]]`.
The new run stopped after 24 epochs and selected epoch 16 by loss, without separate
tuning. Checkpoint reloading reproduces its validation outputs exactly.

Normalization did not improve this controlled course-project comparison. The
unnormalized variant correctly classified two additional fall clips on this split.
This single-seed, 20-clip result is not a statistical significance or broad
camera/generalization claim; no additional tuning run was performed.

## Robustness of the normalized Stage 4 checkpoint

The fixed seed is 42. For each canonical clip ID, SHA256(seed:ID) initializes
NumPy PCG64 and a frame permutation. Select the first floor(rate × T) indices,
so 10% is a subset of 20%, independently of file iteration order. Selection is
across all source frames, including already-missing ones. Selected frames are
marked pose_detected=False and all-NaN **in memory before preprocessing**, then
passed through the unchanged causal fill/normalization. No labels, lengths or
clips are dropped. No future-frame interpolation or model retraining occurs.

| Nominal dropout | Accuracy | Precision | Recall | F1 | Delta F1 vs clean | Delta Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0.9000 | 1.0000 | 0.8000 | 0.8889 | 0.0000 | 0.0000 |
| 10% | 0.9000 | 1.0000 | 0.8000 | 0.8889 | 0.0000 | 0.0000 |
| 20% | 0.9000 | 1.0000 | 0.8000 | 0.8889 | 0.0000 | 0.0000 |

All three confusion matrices are `[[10,0],[2,8]]`. At 10%, 459/4,674 frames were
selected (9.8203%), introducing 398 additional missing frames. At 20%, 925 were
selected (19.7903%), introducing 799 additional missing frames. Fractions differ
slightly from nominal rates because counts are floored per clip. Some selected
frames were already missing. Probabilities/loss changed, while binary predictions
and classification metrics remained unchanged. Clean probabilities, predictions
and metrics exactly match the frozen Stage 4 CSV/summary.

The normalized model tolerated these particular synthetic frame-dropout masks
with causal fill on this Validation split. This does not establish robustness to
long outages, occlusion, other random seeds or different cameras.

## Final configuration decision

`final_ml_config.json` freezes **the unnormalized Stage 5 model**, based on its
higher clean Validation fall F1/Recall and lower loss under the controlled comparison.
This is the prompt's exception to the intended normalized default. It explicitly
references both the selected checkpoint and the unchanged canonical Stage 4
checkpoint; their preprocessing modes must not be mixed.

Selected: causal fill <=5 frames, normalize_pose=False, XYZ/visibility 132-D,
single-layer unidirectional GRU hidden 64, threshold 0.5. The selected checkpoint
is `ml/checkpoints/stage5_no_normalization_best.pt` (Git-ignored, not duplicated).
The configuration is fixed for later system integration and held-out evaluation.
Subjects 6/7 have not been evaluated. The requested robustness experiment belongs
to the normalized baseline only; **no dropout robustness result is claimed for
the selected unnormalized model**. No second robustness experiment is run.

## Verification and reproduction

75 tests passed: five new tests cover deterministic/nested dropout, input
immutability, causal fill, controlled ablation, pre-I/O subject guards and exact
clean-condition reproduction; related Stage 3.3/4 tests cover the unchanged model,
labels, lengths and metric formulae. The real experiment was run with a guard that
rejected Subject.6/7 and AVI paths and allowed NPZ opens only in read-only mode.
Only the same 80 authorized NPZ identities were accessed. Input fingerprints and
the normalized checkpoint SHA256 match before/after; all outputs are finite.

Code: `ml/evaluation/stage5.py`; config: `configs/stage5_no_normalization.json`.
Runtime is the same Python 3.12.14 / NumPy 2.3.5 / torch 2.8.0 CPU environment.

```bash
python -m unittest tests.test_stage5_evaluation tests.test_stage4_gru tests.test_stage33_pose_sequence tests.test_stage3_contract tests.test_dataset_split tests.test_pose_raw -q
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache uv run ml/evaluation/stage5.py
```

The runner refuses existing Stage 5 output/checkpoint paths; use a separate
checkout for a fresh reproduction, retaining the accepted Stage 4 checkpoint
and authorized raw inputs. It generates results and predictions; the final-config
record captures the subsequent evidence-based selection. The unnormalized
training summary/config and all 20 probabilities are under
`artifacts/training/stage5_no_normalization/`. Dropout probabilities are in
`dropout_predictions.csv`; full metrics, corruption counts and access log are in
`results.json`. No large tensors or checkpoint binaries are committed.

Stage 5 gate: **PASS**. Stage 6, MQTT, backend and cloud integration are not started.
