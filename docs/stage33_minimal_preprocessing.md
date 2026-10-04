# Stage 3.3 — minimal clip supervision and preprocessing

The course-project sprint authorizes this builder directly. Stage 3.2j review
remains frozen at `63793aa50f91d1df3429ac081ae23b7dc6dd9d6c`; genuine review,
3.2k adjudication and temporal-boundary research are closed. Historical Stage 3.0
build-disabled files remain unchanged; their guards still protect old workflows.
This new dataset uses the existing verified source selector and frozen split.

## Input and output

Official immutable Stage 2 run: `4df7dd5f-fb38-4908-8233-1a81deb8dc05`,
`pose_raw_v1` NPZ with frame_index int32 [T], timestamp_ms int64 [T],
pose_detected bool [T], landmarks float32 [T,33,4] (x,y,z,visibility).
Detected frames must be finite; missing frames are all NaN. T varies; frames
retain original order and 20 FPS. Timestamps are clip-relative, derived from
frame index, not Unix event timestamps. No videos, PNGs, temporal annotations,
or held-out pose files are loaded.

The dataset uses canonical clip labels `fall=1`, `non_fall=0`, never boundary
annotations. Each sample contains float32 features [T,132], label, sequence_length,
bool joint mask [T,33], clip_id, subject_id, split and missing_pose_frames.
Identifiers are audit metadata only. Each joint contributes normalized x/y,
original visibility and validity (0/1). z is omitted because it is not calibrated
metric depth. This is the sole preprocessing variant.

## Deterministic transform

1. Multiply image-relative x by width/height; y stays in image-height units.
2. Hip center is midpoint of MediaPipe 23/24; shoulders are midpoint of 11/12,
   matching the repository's existing landmark mapping.
3. Subtract hip center and divide x/y by Euclidean shoulder-center to hip-center
   distance. No rotation, fitted statistics or optional motion features.
4. Scale below 0.001 image-height units is invalid. This fixed numerical floor
   was chosen before loading Train/Validation and was not tuned.
5. Missing joints use finite-coordinate/visibility availability, with no visibility
   threshold: low-confidence returned coordinates remain observed and visibility
   is exposed. A single available hip/shoulder substitutes for its pair midpoint;
   absent hip or shoulder pair makes the frame invalid. The frozen raw schema
   currently permits only whole-frame missingness; partial-joint fallback is
   tested for future callers, not fabricated into the raw data.
6. Invalid frames/joints have all-zero features and false validity. No interpolation,
   carry-forward, smoothing or reconstruction. Missing internal frames preserve time.
7. Exclude only clips with no valid normalized joint in any frame; record each
   exclusion. This rule applies identically to Train/Validation and changes no
   subject membership. Source inputs remain immutable.

Malformed archives, schema/identity mismatches, invalid labels and unauthorized
splits raise errors. Finite-output checks reject unexpected NaN/Inf. Both the
selector and loader reject Test before pose I/O. No statistics are fitted.

## Batching and next-stage use

The dataset eagerly caches the small authorized split and returns independent
arrays. `collate_sequences` right-pads full clips to batch maximum, returning
features [B,Tmax,132], int64 labels [B], int64 lengths [B], bool masks [B,Tmax,33]
and audit IDs. Lengths include internal missing frames and exclude batch padding.
Stage 4 must use packed sequences so padding cannot alter its final hidden state:

```python
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from ml.datasets.fall_sequence_dataset import FallSequenceDataset, collate_sequences

dataset = FallSequenceDataset("train")
batch = collate_sequences([dataset[0], dataset[1]])
x = torch.from_numpy(batch["features"])
y = torch.from_numpy(batch["labels"]).float()  # Binary-logit loss targets.
packed = pack_padded_sequence(x, torch.from_numpy(batch["lengths"]),
                             batch_first=True, enforce_sorted=False)
gru = torch.nn.GRU(input_size=132, hidden_size=32, batch_first=True)
_, hidden = gru(packed)
# Feed hidden[-1] into the binary classifier; keep IDs outside the model.
```

This torch example is a Stage 4 consumption recipe, not an executed GRU test.
Stage 3.3 requires only NumPy (tested with Python 3.12.14 / NumPy 2.3.5).
Reproduce with an environment containing NumPy 2.3.5:

```bash
python -m unittest tests.test_stage33_pose_sequence tests.test_stage3_contract tests.test_dataset_split tests.test_pose_raw -q
python -m ml.datasets.fall_sequence_dataset --output artifacts/preprocessing/stage33_recheck/audit.json
```

The audit refuses existing destinations. It stores source identities/checksums,
configuration, access log, counts and runtime NumPy version; no processed tensors
are committed. Dataset construction is the reproducible preprocessing path.

## Model-readiness result

See `artifacts/preprocessing/stage33_minimal/audit.json` for executed results.
Train subjects remain 8/4/3/9/1/2; Validation remains 10/5; Test 6/7 remains sealed.
60 input Train clips become 59 usable clips (29 fall, 30 non_fall); 20 Validation
clips remain usable (10/10). Train Subject.8/Fall forward/FallForwardS8.avi is
excluded because all 122 frames lack pose. No invalid samples remain.
Usable sequence min/median/max: Train 86/189/266, Validation 122/231.5/292.
Pre-filter missing-pose rates: Train 14.215%, Validation 14.934%.
All output values are finite. The guarded audit opened exactly 80 unique authorized
NPZs and no Subject.6/7 data or AVI. Tests include synthetic transforms, masks,
label mapping, split/access guards and actual authorized loader integration.

Limitations: clip presence is weak supervision and provides no event timing;
long pose gaps remain missing; image-plane normalization loses absolute hip
translation and does not solve camera domain shift. Full clips differ from future
30-frame online windows; Stage 4/5 must document the deployment aggregation choice
without treating every window in a fall clip as positive. No model training,
accuracy claim or test evaluation has been performed.
