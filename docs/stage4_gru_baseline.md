# Stage 4 — small GRU baseline

Git preflight confirmed HEAD at canonical Stage 3.3 commit
`729b1d26c8935f8550e2bedef68e6e4ee034730d` and earlier Stage 3.3 commit
`2bc88a8cad4a3940df5c0e14944179a507446db4` as its ancestor. No history changes
were made. The user's uncommitted AGENTS.md is excluded from this stage commit.

## Model and training

`ml/models/gru.py`: CPU float32 features [B,T,132] plus CPU int64 lengths [B]
produce raw fall logits [B]. One unidirectional GRU with hidden size 64 and one
layer feeds `Linear(64,1)` from `hidden[-1]`. `pack_padded_sequence` excludes
trailing padding; no sigmoid inside the model. IDs, validity masks and boundary
annotations are not input features. Sigmoid is used only for probabilities.

The single fixed run uses `configs/stage4_gru.json`: Python/NumPy/torch seed 42,
batch size 8, Adam lr 0.001, weight decay 0.0001, BCEWithLogitsLoss, max 50 epochs,
patience 8, threshold 0.5. CPU, one torch thread, deterministic algorithms and a
seeded shuffle generator are recorded. Stage 3.3 preprocessing is unchanged:
XYZ/visibility, hip/torso normalization enabled, causal fill up to five frames,
then zero; full clip lengths retained.

The sole selection criterion is minimum Validation loss, with strict improvement
and ties retaining the earlier epoch. This gives a continuous signal on the small
20-clip Validation set. No F1 tie-break, threshold tuning, class balancing, model
search or ablation is performed. Early stopping counts eight epochs without loss
improvement. Loss is a sample mean; final Train/Validation metrics are calculated
in eval mode using the saved best epoch, not averaged training-batch predictions.

## Executed result

59 usable Train clips (29 fall/30 non_fall), subjects 1/2/3/4/8/9; 20 Validation
clips (10/10), subjects 5/10. The existing all-missing Train Subject.8 Fall forward
clip remains excluded. Exclusions and denominator counts are in summary.json.
Subjects 6/7 were never accessed; only 80 authorized NPZs were opened read-only.
Their aggregate fingerprint was identical before/after training. No subject
leakage, raw-data mutation, nonfinite loss or nonfinite gradient was observed.

One real training run completed 13 epochs; best epoch is 5 (Validation loss).

| Best checkpoint | Train | Validation |
| --- | ---: | ---: |
| Loss | 0.06896234 | 0.29209915 |
| Accuracy | 1.0000 | 0.9000 |
| Precision (fall) | 1.0000 | 1.0000 |
| Recall (fall) | 1.0000 | 0.8000 |
| F1 (fall) | 1.0000 | 0.8889 |

Confusion matrices use true rows and predicted columns, order [non_fall, fall]:
Train `[[30,0],[0,29]]`; Validation `[[10,0],[2,8]]`. Undefined precision/recall/F1
are explicitly zero. Probabilities >= 0.5 predict fall. Validation predicts both
classes, gets 18/20 correct and exceeds a constant-class classifier's 10/20 accuracy.
This indicates learning on this split; it is not a held-out Test or online result.
Validation loss rose after epoch 5, so the best checkpoint was retained rather
than the final epoch. No additional tuning run was started.

## Artifacts and reproduction

- `artifacts/training/stage4_gru_baseline/summary.json`: source identity, input
  fingerprint, split counts/exclusions, full config, runtime, seed, best epoch,
  history, best/last Validation metrics and checkpoint identity.
- `artifacts/training/stage4_gru_baseline/validation_predictions.csv`: 20 canonical
  clip IDs, subject IDs, labels, probabilities, fixed-threshold predictions, lengths.
- `ml/checkpoints/stage4_gru_best.pt`: the sole local canonical best checkpoint,
  156,309 bytes, SHA256
  `611693fd32d527fd9e84b293158cfa193fd24a9aa7593a83397c553f640ad9c0`.

The checkpoint is Git-ignored under the existing policy. It includes state_dict,
model config, preprocessing config, threshold, seed, best epoch, source fingerprint,
training-base commit and implementation SHA256s. The training-base commit is the
parent state present during the run; the implementation hashes identify the exact
Stage 4 source that is committed afterward. Runtime tested: Python 3.12.14,
NumPy 2.3.5, torch 2.8.0. The script pins dependencies using PEP 723.

From repository root, use new output and checkpoint paths (existing ones refused):

```bash
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache uv run --with numpy==2.3.5 --with torch==2.8.0 python -m unittest tests.test_stage4_gru tests.test_stage33_pose_sequence tests.test_stage3_contract tests.test_dataset_split tests.test_pose_raw -q
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache uv run ml/training/train_gru.py --output artifacts/training/stage4_recheck --checkpoint ml/checkpoints/stage4_recheck.pt
```

Inference loading:

```python
from pathlib import Path
import torch
from ml.models.gru import load_checkpoint

model, metadata = load_checkpoint(Path("ml/checkpoints/stage4_gru_best.pt"))
with torch.no_grad():
    probabilities = model(batch["features"], batch["lengths"]).sigmoid()
    predictions = probabilities >= metadata["classification_threshold"]
```

70 tests passed (7 new GRU tests plus 63 relevant regression tests), including
actual Stage 3.3 batches, unsorted lengths, padding-suffix invariance, individual
sequence equivalence, backward gradients, known metrics, threshold boundary,
checkpoint round-trip and training-path access guards. The real best checkpoint
was reloaded and reproduced all Validation metrics/probabilities/predictions exactly.

Stage 4 gate: PASS. Training uses full labeled clips; continuous online inference
will require a rolling sequence-buffer policy. That policy, normalization ablation,
robustness experiments, Test evaluation and cloud integration remain later work.
Stage 5 and Stage 6 were not started.
