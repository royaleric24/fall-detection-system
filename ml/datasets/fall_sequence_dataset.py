"""Stage 3.3 clip dataset and NumPy collator; directly convertible to GRU tensors.

Run audit: python -m ml.datasets.fall_sequence_dataset --output PATH
Dependencies: numpy. No torch dependency is required at this stage.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ml.datasets.pose_raw import load_validated
from ml.preprocessing.contract import (REPO, PoseSource, read_contract, require_subject_access,
                                       select_sources, validate_output_path)
from ml.preprocessing.pose_sequence import (FEATURE_DIM, PreprocessingConfig,
                                            normalize_pose_sequence)

LABELS = {"non_fall": 0, "fall": 1}


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


class FallSequenceDataset:
    """Map-style dataset: select verified sources first, load only chosen clips.

    Returns features [T,132], label integer, length, joint mask [T,33], clip_id,
    subject_id and split. IDs never enter features. Internal missing frames count
    toward length: lengths remove trailing batch padding only.
    """

    def __init__(self, split: str = "train", *,
                 config: PreprocessingConfig = PreprocessingConfig()) -> None:
        self.sources = select_sources((split,))  # Reject Test before any I/O.
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
        self.samples = []
        self.excluded = []
        retained = []
        self.raw_frames = 0
        self.raw_missing_frames = 0
        for source in self.input_sources:
            sample = self._load_sample(source)
            self.raw_frames += sample["sequence_length"]
            self.raw_missing_frames += sample["missing_pose_frames"]
            if not sample["mask"].any():
                self.excluded.append(dict(clip_id=source.source_video,
                                          subject_id=source.subject_id, split=source.split,
                                          label=sample["label"], reason="no_valid_normalized_pose"))
            else:
                retained.append(source)
                self.samples.append(sample)
        self.sources = tuple(retained)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Independent arrays protect cached deterministic preprocessing from callers.
        return {k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in self.samples[index].items()}

    def _load_sample(self, source: PoseSource) -> dict[str, Any]:
        require_subject_access(source.subject_id, source.split)
        row = self.metadata[source.source_video]
        arrays = load_validated(source.source_npz, float(row["source_fps"]),
                                int(row["expected_frame_count"]))
        features, mask = normalize_pose_sequence(
            arrays["landmarks"], arrays["pose_detected"],
            image_aspect=int(row["width"]) / int(row["height"]), config=self.config)
        return dict(features=features, label=binary_label(source.video_label),
                    sequence_length=len(features), mask=mask, clip_id=source.source_video,
                    subject_id=source.subject_id, split=source.split,
                    missing_pose_frames=int((~arrays["pose_detected"]).sum()))


def collate_sequences(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Right-pad full clips; GRU must pack with these lengths before final-state use."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    lengths = np.array([s["sequence_length"] for s in samples], dtype=np.int64)
    if np.any(lengths <= 0):
        raise ValueError("Empty sequences cannot be packed")
    features = np.zeros((len(samples), int(lengths.max()), FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((len(samples), int(lengths.max()), 33), dtype=bool)
    for i, (sample, length) in enumerate(zip(samples, lengths)):
        if (sample["features"].shape != (length, FEATURE_DIM)
                or sample["mask"].shape != (length, 33)
                or not np.isfinite(sample["features"]).all()):
            raise ValueError("Invalid sample dimensions/values")
        features[i, :length] = sample["features"]
        mask[i, :length] = sample["mask"]
    return dict(features=features, lengths=lengths, mask=mask,
                labels=np.array([s["label"] for s in samples], dtype=np.int64),
                clip_ids=[s["clip_id"] for s in samples],
                subject_ids=[s["subject_id"] for s in samples])


def audit() -> dict[str, Any]:
    """Minimal full model-readiness pass, exclusively over authorized 80 clips."""
    datasets = {split: FallSequenceDataset(split) for split in ("train", "validation")}
    validate_sources(tuple(s for ds in datasets.values() for s in ds.sources))
    result: dict[str, Any] = dict(stage="3.3", schema="fall_clip_v1", feature_dimension=FEATURE_DIM,
                                source_run_id=read_contract()["source"]["run_id"],
                                numpy_version=np.__version__,
                                source_identity=read_contract()["source"],
                                config=asdict(PreprocessingConfig()), test_subjects_accessed=[],
                                access_log=[], splits={})
    for split, dataset in datasets.items():
        samples = []
        result["access_log"].extend(dict(clip_id=s.source_video, subject_id=s.subject_id,
                                         split=s.split) for s in dataset.input_sources)
        for index in range(len(dataset)):
            sample = dataset[index]
            samples.append(sample)
        lengths = [s["sequence_length"] for s in samples]
        batch = collate_sequences(samples)
        frames = sum(lengths)
        result["splits"][split] = dict(
            input_clips=len(dataset.input_sources), excluded=dataset.excluded,
            clips=len(samples), subjects=sorted({s["subject_id"] for s in samples}),
            class_counts=dict(Counter("fall" if s["label"] else "non_fall" for s in samples)),
            sequence_length=dict(min=min(lengths), median=float(np.median(lengths)), max=max(lengths)),
            input_frames=dataset.raw_frames,
            input_missing_pose_rate=dataset.raw_missing_frames / dataset.raw_frames,
            frames=frames, missing_pose_frames=sum(s["missing_pose_frames"] for s in samples),
            missing_pose_rate=sum(s["missing_pose_frames"] for s in samples) / frames,
            invalid_normalized_frames=sum(int((~s["mask"].any(axis=1)).sum()) for s in samples),
            invalid_samples=sum(not s["mask"].any() for s in samples),
            nonfinite_values=int((~np.isfinite(batch["features"])).sum()),
            batch_shape=list(batch["features"].shape))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = validate_output_path(args.output)
    report = audit()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["splits"], indent=2))
