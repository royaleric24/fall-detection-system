# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.3.5", "torch==2.8.0"]
# ///
"""One normalization ablation and fixed-seed Validation pose-frame dropout."""

import copy
import csv
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.datasets.fall_sequence_dataset import FallSequenceDataset, collate_sequences, source_fingerprint
from ml.datasets.pose_raw import load_validated
from ml.models.gru import load_checkpoint
from ml.preprocessing.contract import REPO, require_subject_access, select_sources
from ml.preprocessing.pose_sequence import normalize_pose_sequence
from ml.training.train_gru import evaluate, set_seed, train_baseline

STAGE4_COMMIT = "a5fb6f23aa7965fa5b6519f0900642b4eedf40fb"
BASELINE_DIR = REPO / "artifacts/training/stage4_gru_baseline"
OUTPUT = REPO / "artifacts/evaluation/stage5"
ABLATION_OUTPUT = REPO / "artifacts/training/stage5_no_normalization"
ABLATION_CHECKPOINT = REPO / "ml/checkpoints/stage5_no_normalization_best.pt"
ABLATION_CONFIG = REPO / "configs/stage5_no_normalization.json"
DROPOUT_SEED = 42


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_ablation_config(baseline: dict[str, Any], ablation: dict[str, Any]) -> None:
    expected = copy.deepcopy(baseline)
    expected["preprocessing"]["normalize_pose"] = False
    if baseline["preprocessing"]["normalize_pose"] is not True or ablation != expected:
        raise ValueError("Ablation may change only preprocessing.normalize_pose")


def frame_dropout(arrays: dict[str, np.ndarray], sequence_id: str, rate: float, seed: int
                  ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Pure evaluation corruption: floor(rate*T) selected indices across all frames.

    A stable seed/clip hash initializes PCG64; permutation prefixes are nested
    across levels. Already-missing selected frames stay missing. Input is untouched.
    """
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Dropout rate must be between zero and one")
    key = hashlib.sha256(f"{seed}:{sequence_id}".encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(key[:8], "little")))
    frames = len(arrays["pose_detected"])
    selected = np.sort(rng.permutation(frames)[:int(rate * frames)])
    corrupted = {k: v.copy() for k, v in arrays.items()}
    corrupted["pose_detected"][selected] = False
    corrupted["landmarks"][selected] = np.nan
    return corrupted, selected


def dropout_samples(dataset: FallSequenceDataset, rate: float, seed: int
                    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep all clean Validation clips/labels/lengths, even if corruption loses pose."""
    # Validate every source before opening any NPZ, including all original subjects.
    for source in dataset.input_sources:
        require_subject_access(source.subject_id, source.split)
        if source.split != "validation":
            raise ValueError("Robustness evaluation permits Validation only")
    samples = []
    total_frames = selected_frames = newly_missing = 0
    selected_digest = hashlib.sha256()
    for source in dataset.sources:
        row = dataset.metadata[source.source_video]
        arrays = load_validated(source.source_npz, float(row["source_fps"]), int(row["expected_frame_count"]))
        corrupted, selected = frame_dropout(arrays, source.source_video, rate, seed)
        features, mask = normalize_pose_sequence(corrupted["landmarks"], corrupted["pose_detected"],
                                                 config=dataset.config)
        total_frames += len(features)
        selected_frames += len(selected)
        newly_missing += int(arrays["pose_detected"][selected].sum())
        selected_digest.update(json.dumps([source.source_video, selected.tolist()]).encode())
        samples.append(dict(features=torch.from_numpy(features), mask=torch.from_numpy(mask),
                            observed_pose=torch.from_numpy(corrupted["pose_detected"]),
                            sequence_length=len(features), label=int(source.video_label == "fall"),
                            clip_id=source.source_video, subject_id=source.subject_id))
    return samples, dict(seed=seed, nominal_rate=rate, frames=total_frames,
                         selected_frames=selected_frames, newly_missing_frames=newly_missing,
                         selected_fraction=selected_frames / total_frames,
                         selected_indices_sha256=selected_digest.hexdigest())


def check_clean_match(metrics: dict[str, Any], records: list[dict[str, Any]],
                      baseline: dict[str, Any], path: Path) -> None:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = [dict(sequence_id=r["sequence_id"], subject_id=int(r["subject_id"]),
                     label=int(r["label"]), probability=float(r["probability"]),
                     prediction=int(r["prediction"]), n_frames=int(r["n_frames"])) for r in rows]
    if metrics != baseline["best_checkpoint_metrics"]["validation"] or records != expected:
        raise ValueError("0% dropout must exactly reproduce frozen Stage 4 metrics/predictions")


def run_stage5() -> dict[str, Any]:
    subprocess.run(["git", "merge-base", "--is-ancestor", STAGE4_COMMIT, "HEAD"], cwd=REPO, check=True)
    if OUTPUT.exists() or ABLATION_OUTPUT.exists() or ABLATION_CHECKPOINT.exists():
        raise FileExistsError("Refusing to overwrite Stage 5 results/checkpoint")
    baseline = json.loads((BASELINE_DIR / "summary.json").read_text())
    ablation_config = json.loads(ABLATION_CONFIG.read_text())
    validate_ablation_config(baseline["config"], ablation_config)
    checkpoint_path = REPO / baseline["checkpoint"]["path"]
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != baseline["checkpoint"]["sha256"]:
        raise ValueError("Stage 4 checkpoint identity mismatch")
    # Model, Dataset, and preprocessing must still be the accepted Stage 4 versions.
    for name in ("ml/models/gru.py", "ml/datasets/fall_sequence_dataset.py",
                 "ml/preprocessing/pose_sequence.py"):
        if sha256(REPO / name) != baseline["implementation_sha256"][name]:
            raise ValueError(f"Frozen implementation changed: {name}")
    set_seed(DROPOUT_SEED)
    sources = select_sources(("train", "validation"))
    before = source_fingerprint(sources)
    if before != baseline["source_pose_fingerprint_before"]:
        raise ValueError("Stage 4 raw input identity changed")
    model, metadata = load_checkpoint(checkpoint_path)
    if metadata["training_config"] != baseline["config"]:
        raise ValueError("Checkpoint/config mismatch")
    dataset = FallSequenceDataset("validation")
    robustness = []
    all_predictions = []
    for rate in (0.0, 0.1, 0.2):
        samples, corruption = dropout_samples(dataset, rate, DROPOUT_SEED)
        loader = DataLoader(samples, batch_size=baseline["config"]["batch_size"], shuffle=False,
                            num_workers=0, collate_fn=collate_sequences)
        metrics, records = evaluate(model, loader, 0.5)
        if rate == 0:
            check_clean_match(metrics, records, baseline, BASELINE_DIR / "validation_predictions.csv")
        clean = baseline["best_checkpoint_metrics"]["validation"]
        robustness.append(dict(dropout=corruption, metrics=metrics,
                               delta_f1=metrics["f1"] - clean["f1"],
                               delta_recall=metrics["recall"] - clean["recall"]))
        all_predictions.extend(dict(dropout_rate=rate, **r) for r in records)
    # Exactly one new training run; the existing normalized checkpoint is reused.
    ablation = train_baseline(ABLATION_CONFIG, ABLATION_OUTPUT, ABLATION_CHECKPOINT,
                              normalization_ablation=True)
    if ablation["datasets"] != baseline["datasets"]:
        raise ValueError("Ablation changed dataset membership/counts/exclusions")
    after = source_fingerprint(sources)
    if before != after or sha256(checkpoint_path) != checkpoint_hash:
        raise ValueError("Raw inputs or normalized checkpoint changed")
    a = ablation["best_checkpoint_metrics"]["validation"]
    b = baseline["best_checkpoint_metrics"]["validation"]
    result = dict(stage=5, stage4_commit=STAGE4_COMMIT,
                  comparison_changed_field="preprocessing.normalize_pose",
                  normalized_checkpoint=baseline["checkpoint"],
                  ablation_summary=ABLATION_OUTPUT.relative_to(REPO).as_posix() + "/summary.json",
                  ablation=dict(without_normalization=dict(best_epoch=ablation["best_epoch"], metrics=a),
                                normalization=dict(best_epoch=baseline["best_epoch"], metrics=b),
                                delta_f1_normalized_minus_unnormalized=b["f1"] - a["f1"]),
                  robustness=robustness, clean_matches_stage4=True,
                  raw_pose_fingerprint_before=before, raw_pose_fingerprint_after=after,
                  raw_pose_unchanged=True, normalized_checkpoint_unchanged=True,
                  test_subjects_accessed=[],
                  evaluation_script_sha256=sha256(Path(__file__)),
                  access_log=[dict(sequence_id=s.source_video, subject_id=s.subject_id, split=s.split)
                              for s in sources])
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (OUTPUT / "dropout_predictions.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(all_predictions[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_predictions)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_stage5()
    print(json.dumps(dict(ablation=result["ablation"], robustness=result["robustness"]), indent=2))
