# Stage 3.3 — causal XYZ preprocessing and PyTorch dataset

This revision implements the final sprint request: clip-level binary labels,
causal missing handling, original XYZ/visibility features, real PyTorch Dataset
and DataLoader, and one normalization switch. It supersedes the 2-D zero-fill
implementation in commit `2bc88a8cad4a3940df5c0e14944179a507446db4`.
The old `stage33_minimal/audit.json` remains historical evidence; the current
report and canonical sequence manifest are in `artifacts/preprocessing/stage33_causal/`.
Stage 3.2 research records and Stage 3.0 guards remain unchanged. No genuine
3.2j review, adjudication, model training or Stage 4 execution is performed.

## Representation and deterministic preprocessing

Immutable input is official Stage 2 run `4df7dd5f-fb38-4908-8233-1a81deb8dc05`:
`pose_raw_v1` NPZ contains frame_index int32 [T], timestamp_ms int64 [T],
pose_detected bool [T], landmarks float32 [T,33,4], ordered x/y/z/visibility.
Detected poses are finite; missing poses are all NaN. Times are clip-relative,
derived from frame index at source FPS (20), not Unix runtime event times.

`PreprocessingConfig(normalize_pose=True, max_forward_fill_frames=5, epsilon=1e-6)`:

1. Use the current detected pose. For a missing frame, copy the most recent
   observed pose for at most five consecutive frames. Filled poses do not reset
   the missing-run counter. Leading gaps and the sixth/subsequent missing frames
   are zero. No future-frame interpolation, time compression or cropping.
2. Use XYZ midpoint of hips 23/24 and shoulders 11/12, following the existing
   MediaPipe landmark mapping. Subtract hip XYZ center and divide XYZ by
   `max(norm(shoulder_center - hip_center), epsilon)`.
3. If a caller has one unavailable hip/shoulder, use its available partner; if
   neither hip or neither shoulder is available, mark the frame zero/invalid.
   The official raw loader still enforces its original whole-frame schema.
4. Keep visibility as a separate original feature, clipped to [0,1]. Low
   visibility is not treated as missing. Spatial XYZ alone is normalized.
5. For the later ablation, set `normalize_pose=False`: original XYZ coordinates
   with exactly the same causal fill and visibility handling. No ablation is run now.

Coordinate axes and relative units are MediaPipe's canonical image x/y and
relative z. The XYZ torso norm is a deterministic relative scale, not calibrated
physical distance. No aspect conversion, angle, velocity or added input feature.
Invalid joints become zeros with false masks; finite output is explicitly checked.
Malformed archives, labels, identity mismatches and unauthorized splits raise errors.

## Dataset, manifest and batching

`FallSequenceDataset` subclasses `torch.utils.data.Dataset`, eagerly caching the
small authorized split. Each sample returns CPU float32 features [T,132] (33 ×
x/y/z/visibility), integer label, sequence_length, bool mask [T,33], bool
observed_pose [T], clip_id, subject_id, split and missing/fill counts. mask includes
usable filled poses; observed_pose distinguishes actual observations. IDs and masks
are metadata and are not appended to model features. Returned tensors are cloned.

Canonical labels come from the verified Stage 2 manifest: `fall=1`, `non_fall=0`.
No temporal annotation is loaded. The minimal exported `sequences.csv` contains
sequence_id, subject_id, activity, label, pose_path (repository-relative), n_frames,
fps, split. It contains the 79 usable sequences. All-missing clips are excluded
using original detection alone, identically for both normalization settings;
exclusions stay in the audit. No subject is reassigned.

`collate_sequences` uses PyTorch `pad_sequence`, returning float32 features
[B,Tmax,132], int64 labels [B], int64 CPU lengths [B], bool masks, observed_pose,
and IDs. Original sequence lengths include internal missing frames; only batch
padding lies outside length. Stage 4 must pack before using a final hidden state:

```python
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from ml.datasets.fall_sequence_dataset import FallSequenceDataset, collate_sequences
from ml.preprocessing.pose_sequence import PreprocessingConfig

dataset = FallSequenceDataset("train", config=PreprocessingConfig(normalize_pose=True))
loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_sequences)
batch = next(iter(loader))
packed = pack_padded_sequence(batch["features"], batch["lengths"],
                             batch_first=True, enforce_sorted=False)
# Stage 4: GRU(input_size=132); binary-logit loss uses batch["labels"].float().
```

## Reproduction and executed checks

Runtime tested: Python 3.12.14, NumPy 2.3.5, torch 2.8.0 (CPU). PEP 723 dependencies
are pinned in the dataset script. From repository root:

```bash
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache uv run --with numpy==2.3.5 --with torch==2.8.0 python -m unittest tests.test_stage33_pose_sequence tests.test_stage3_contract tests.test_dataset_split tests.test_pose_raw -q
UV_CACHE_DIR=/private/tmp/caucafall-uv-cache uv run ml/datasets/fall_sequence_dataset.py --output artifacts/preprocessing/stage33_recheck
```

The output directory must be new. The audit verifies authorized NPZ fingerprints
before/after processing, records source identity, config, runtime and accessed IDs,
and exercises a real DataLoader plus `pack_padded_sequence` without a model.
The guarded smoke check allows only read-only NPZ access and refuses Subject.6/7
paths or AVI. All 80 authorized input NPZs were unchanged; 63 focused and related
regression tests passed (13 Stage 3.3 tests).

Train subjects: 8/4/3/9/1/2; Validation: 10/5; sealed Test: 6/7, never opened.
60 Train inputs become 59 usable clips (29 fall / 30 non_fall); Validation is 20
(10/10). All 122 frames of Subject.8/Fall forward/FallForwardS8.avi lack pose,
so that clip remains explicitly excluded. Usable length min/median/max is Train
86/189/266 and Validation 122/231.5/292. Input missing rates are 14.215%/14.934%.
Among retained clips, Train fills 315 missing frames and leaves 1,143 zero frames;
Validation fills 166 and leaves 532 zero frames. No NaN/Inf or unusable clips remain.
First DataLoader shapes: Train [4,190,132], Validation [4,247,132].

Limitations: causal fill holds stale poses for at most 250 ms at source 20 FPS;
long gaps remain zero. Clip labels give event presence, not timing. Relative XYZ
normalization removes absolute hip translation and does not resolve camera domain
shift. Online window aggregation remains a later-stage decision. The all-missing
exclusion must be reported in future evaluation denominators.
