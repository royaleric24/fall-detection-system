# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.3.5", "torch==2.8.0"]
# ///
"""Stage 3.3 PyTorch clip dataset, causal preprocessing and padded DataLoader.

Run: uv run ml/datasets/fall_sequence_dataset.py --output OUTPUT_DIRECTORY
"""

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.datasets.pose_raw import load_validated
from ml.preprocessing.contract import (REPO, PoseSource, read_contract, require_subject_access,
                                       select_sources, validate_output_path)
from ml.preprocessing.pose_sequence import (FEATURE_DIM, PreprocessingConfig,
                                            normalize_pose_sequence)

LABELS = {"non_fall": 0, "fall": 1}
MANIFEST_FIELDS = ("sequence_id", "subject_id", "activity", "label", "pose_path",
                   "n_frames", "fps", "split")


def binary_label(label: str) -> int:
    if label not in LABELS:
        raise ValueError(f"Unknown canonical clip label: {label}")
    return LABELS[label]


def validate_sources(sources: Sequence[PoseSource]) -> None:
    """Reject held-out, mismatched, overlapping or duplicate sources before pose I/O."""
    seen: set[str] = set()
    subjects: dict[str, set[int]] = {"train": set(), "validation": set()}
    for source in sources:
        require_subject_access(source.subject_id, source.split)
        if source.split not in subjects:
            raise ValueError("Only train/validation are authorized")
        binary_label(source.video_label)
        if source.source_video in seen:
            raise ValueError("Duplicate clip")
        seen.add(source.source_video)
        subjects[source.split].add(source.subject_id)
    if subjects["train"] & subjects["validation"]:
        raise ValueError("Train/validation subject leakage")


class FallSequenceDataset(Dataset):
    """Verified full clips, cached as independent float32 CPU PyTorch tensors.

    features [T,132] contain XYZ/visibility only. mask [T,33] marks usable joints
    (including causal fill); observed_pose [T] retains original detection. IDs
    never enter features. Entirely undetected clips are excluded consistently in
    both normalization variants; exclusions are recorded, no frames are cropped.
    """

    def __init__(self, split: str = "train", *,
                 config: PreprocessingConfig = PreprocessingConfig()) -> None:
        self.sources = select_sources((split,))  # Test rejected before any I/O.
        validate_sources(self.sources)
        self.config = config
        contract = read_contract()
        manifest = REPO / contract["source"]["evidence_root"] / "manifest.csv"
        content = manifest.read_bytes()
        if hashlib.sha256(content).hexdigest() != contract["source"]["manifest_sha256"]:
            raise ValueError("Source manifest changed")
        selected = {s.source_video for s in self.sources}
        self.metadata = {r["source_relative_video_path"]: r for r in csv.DictReader(
            content.decode().splitlines()) if r["source_relative_video_path"] in selected}
        if set(self.metadata) != selected:
            raise ValueError("Missing selected source metadata")
        self.input_sources = self.sources
        self.samples: list[dict[str, Any]] = []
        self.excluded: list[dict[str, Any]] = []
        self.manifest: list[dict[str, Any]] = []
        retained = []
        self.raw_frames = 0
        self.raw_missing_frames = 0
        for source in self.input_sources:
            sample = self._load_sample(source)
            self.raw_frames += sample["sequence_length"]
            self.raw_missing_frames += sample["missing_pose_frames"]
            # Based only on original detection, identical for normalization ablation.
            if not sample["observed_pose"].any():
                self.excluded.append(dict(sequence_id=source.source_video,
                                          subject_id=source.subject_id, split=source.split,
                                          label=sample["label"], reason="all_frames_pose_missing"))
                continue
            if not sample["mask"].any():
                raise ValueError(f"No usable processed pose: {source.source_video}")
            retained.append(source)
            self.samples.append(sample)
            row = self.metadata[source.source_video]
            self.manifest.append(dict(sequence_id=source.source_video,
                                      subject_id=source.subject_id, activity=source.activity,
                                      label=sample["label"],
                                      pose_path=source.source_npz.relative_to(REPO).as_posix(),
                                      n_frames=sample["sequence_length"],
                                      fps=float(row["source_fps"]), split=source.split))
        self.sources = tuple(retained)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in self.samples[index].items()}

    def _load_sample(self, source: PoseSource) -> dict[str, Any]:
        require_subject_access(source.subject_id, source.split)
        row = self.metadata[source.source_video]
        arrays = load_validated(source.source_npz, float(row["source_fps"]),
                                int(row["expected_frame_count"]))
        features, mask = normalize_pose_sequence(arrays["landmarks"], arrays["pose_detected"],
                                                 config=self.config)
        observed = arrays["pose_detected"].copy()
        return dict(features=torch.from_numpy(features), label=binary_label(source.video_label),
                    sequence_length=len(features), mask=torch.from_numpy(mask),
                    observed_pose=torch.from_numpy(observed), clip_id=source.source_video,
                    subject_id=source.subject_id, split=source.split,
                    missing_pose_frames=int((~observed).sum()),
                    forward_filled_frames=int((mask.any(axis=1) & ~observed).sum()))


def collate_sequences(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Right-pad full clips. Pack using CPU lengths before selecting GRU final state."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([s["sequence_length"] for s in samples], dtype=torch.int64)
    for sample, length in zip(samples, lengths.tolist()):
        if (length <= 0 or sample["features"].shape != (length, FEATURE_DIM)
                or sample["features"].dtype != torch.float32
                or sample["mask"].shape != (length, 33) or sample["mask"].dtype != torch.bool
                or sample["observed_pose"].shape != (length,)
                or sample["label"] not in (0, 1)
                or not torch.isfinite(sample["features"]).all()):
            raise ValueError("Invalid sample dimensions/values")
    return dict(features=pad_sequence([s["features"] for s in samples], batch_first=True),
                lengths=lengths,
                mask=pad_sequence([s["mask"] for s in samples], batch_first=True),
                observed_pose=pad_sequence([s["observed_pose"] for s in samples], batch_first=True),
                labels=torch.tensor([s["label"] for s in samples], dtype=torch.int64),
                clip_ids=[s["clip_id"] for s in samples],
                subject_ids=[s["subject_id"] for s in samples])


def source_fingerprint(sources: Sequence[PoseSource]) -> str:
    """Read only selected pose files; fingerprint identities and complete bytes."""
    validate_sources(sources)
    digest = hashlib.sha256()
    for source in sorted(sources, key=lambda s: s.source_video):
        digest.update(source.source_video.encode())
        with source.source_npz.open("rb") as handle:
            digest.update(hashlib.file_digest(handle, "sha256").digest())
    return digest.hexdigest()


def audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Authorized data smoke, real DataLoader/packing and source immutability check."""
    sources = select_sources(("train", "validation"))
    validate_sources(sources)
    before = source_fingerprint(sources)
    datasets = {split: FallSequenceDataset(split) for split in ("train", "validation")}
    result: dict[str, Any] = dict(
        stage="3.3", schema="fall_clip_xyz_causal_v2", feature_dimension=FEATURE_DIM,
        python_version=platform.python_version(), numpy_version=np.__version__,
        torch_version=torch.__version__, source_identity=read_contract()["source"],
        config=asdict(PreprocessingConfig()), test_subjects_accessed=[],
        access_log=[dict(sequence_id=s.source_video, subject_id=s.subject_id, split=s.split)
                    for s in sources], splits={})
    manifest = []
    for split, dataset in datasets.items():
        if not len(dataset):
            raise ValueError(f"No usable clips in {split}")
        samples = [dataset[i] for i in range(len(dataset))]
        lengths = [s["sequence_length"] for s in samples]
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                            collate_fn=collate_sequences)
        first = next(iter(loader))
        # Verify actual pack interface without starting model training or Stage 4.
        packed = pack_padded_sequence(first["features"], first["lengths"],
                                     batch_first=True, enforce_sorted=False)
        assert packed.data.shape == (int(first["lengths"].sum()), FEATURE_DIM)
        frames = sum(lengths)
        result["splits"][split] = dict(
            input_clips=len(dataset.input_sources), excluded=dataset.excluded, clips=len(samples),
            subjects=sorted({s["subject_id"] for s in samples}),
            class_counts=dict(Counter("fall" if s["label"] else "non_fall" for s in samples)),
            sequence_length=dict(min=min(lengths), median=float(np.median(lengths)), max=max(lengths)),
            input_frames=dataset.raw_frames,
            input_missing_pose_rate=dataset.raw_missing_frames / dataset.raw_frames,
            frames=frames, missing_pose_frames=sum(s["missing_pose_frames"] for s in samples),
            forward_filled_frames=sum(s["forward_filled_frames"] for s in samples),
            zero_filled_frames=sum(int((~s["mask"].any(dim=1)).sum()) for s in samples),
            invalid_samples=sum(not bool(s["mask"].any()) for s in samples),
            nonfinite_values=sum(int((~torch.isfinite(s["features"])).sum()) for s in samples),
            dataloader_batch_shape=list(first["features"].shape),
            dataloader_batch_lengths=first["lengths"].tolist(), packed_sequence_check="PASS")
        manifest.extend(dataset.manifest)
    after = source_fingerprint(sources)
    if before != after:
        raise ValueError("Raw pose files changed during smoke check")
    result["raw_pose_fingerprint_before"] = before
    result["raw_pose_fingerprint_after"] = after
    result["raw_pose_unchanged"] = True
    return result, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    args = parser.parse_args()
    output = validate_output_path(args.output)
    report, manifest = audit()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "audit.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "sequences.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    print(json.dumps(report["splits"], indent=2))
