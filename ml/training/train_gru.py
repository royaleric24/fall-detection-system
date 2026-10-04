# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.3.5", "torch==2.8.0"]
# ///
"""One fixed Stage 4 GRU run; train/validation only, no ablation or threshold search."""

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.datasets.fall_sequence_dataset import (FallSequenceDataset, collate_sequences,
                                               source_fingerprint, validate_sources)
from ml.models.gru import FallGRU, load_checkpoint
from ml.preprocessing.contract import REPO, read_contract, select_sources
from ml.preprocessing.pose_sequence import PreprocessingConfig

STAGE33_COMMIT = "729b1d26c8935f8550e2bedef68e6e4ee034730d"
EARLIER_COMMIT = "2bc88a8cad4a3940df5c0e14944179a507446db4"
LOGGER = logging.getLogger(__name__)


def git_preflight() -> str:
    """Require the intended Stage 3.3 ancestor, never merge or rewrite history."""
    for earlier, later in ((EARLIER_COMMIT, STAGE33_COMMIT), (STAGE33_COMMIT, "HEAD")):
        result = subprocess.run(["git", "merge-base", "--is-ancestor", earlier, later],
                                cwd=REPO, capture_output=True)
        if result.returncode:
            raise ValueError("Repository is not based on canonical Stage 3.3")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    """Fall=1; confusion matrix rows=true, columns=predicted, order non_fall/fall.

    Undefined precision/recall/F1 are explicitly zero, never NaN.
    """
    if (not labels or len(labels) != len(predictions)
            or any(v not in (0, 1) for v in labels + predictions)):
        raise ValueError("Expected same-length nonempty binary labels/predictions")
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(positive_class="fall", accuracy=(tn + tp) / len(labels),
                precision=precision, recall=recall,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                confusion_matrix=[[tn, fp], [fn, tp]])


def load_datasets(config: dict[str, Any]) -> tuple[FallSequenceDataset, FallSequenceDataset]:
    """Hard-coded split scope prevents a caller from requesting Test training."""
    preprocessing = PreprocessingConfig(**config["preprocessing"])
    if not preprocessing.normalize_pose:
        raise ValueError("Stage 4 baseline requires normalize_pose=True")
    train = FallSequenceDataset("train", config=preprocessing)
    validation = FallSequenceDataset("validation", config=preprocessing)
    validate_sources(train.input_sources + validation.input_sources)
    if not len(train) or not len(validation):
        raise ValueError("Empty authorized split")
    return train, validation


def evaluate(model: FallGRU, loader: DataLoader, threshold: float
             ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    records = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["features"], batch["lengths"])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, batch["labels"].float(), reduction="sum")
            probabilities = logits.sigmoid()
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise ValueError("Nonfinite evaluation logits/loss")
            total_loss += float(loss)
            for i, probability in enumerate(probabilities.tolist()):
                records.append(dict(sequence_id=batch["clip_ids"][i],
                                    subject_id=batch["subject_ids"][i],
                                    label=int(batch["labels"][i]), probability=probability,
                                    prediction=int(probability >= threshold),
                                    n_frames=int(batch["lengths"][i])))
    metrics = binary_metrics([r["label"] for r in records], [r["prediction"] for r in records])
    metrics["loss"] = total_loss / len(records)
    return metrics, records


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _dataset_summary(dataset: FallSequenceDataset) -> dict[str, Any]:
    return dict(input_sequences=len(dataset.input_sources), sequences=len(dataset),
                subjects=sorted({s.subject_id for s in dataset.sources}),
                class_counts={"fall": sum(s["label"] == 1 for s in dataset.samples),
                              "non_fall": sum(s["label"] == 0 for s in dataset.samples)},
                excluded=dataset.excluded)


def train_baseline(config_path: Path, output: Path, checkpoint_path: Path) -> dict[str, Any]:
    """Write one canonical best checkpoint plus a small summary/history/predictions."""
    training_base_commit = git_preflight()
    config = json.loads(config_path.read_text())
    if (config["input_size"] != 132 or config["num_layers"] != 1
            or config["optimizer"] != "Adam" or config["loss"] != "BCEWithLogitsLoss"
            or config["selection_criterion"] != "validation_loss"
            or config["classification_threshold"] != 0.5):
        raise ValueError("Unsupported Stage 4 baseline configuration")
    # Limit output destinations; do not overwrite frozen data, evidence or checkpoints.
    for path, root in ((output, REPO / "artifacts/training"),
                       (checkpoint_path, REPO / "ml/checkpoints")):
        if (path.absolute() != path.resolve() or not path.resolve().is_relative_to(root)
                or path.exists()):
            raise ValueError(f"Invalid or existing output: {path}")
    set_seed(config["seed"])
    sources = select_sources(("train", "validation"))
    before = source_fingerprint(sources)
    train, validation = load_datasets(config)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True,
                              generator=generator, num_workers=0, collate_fn=collate_sequences)
    train_eval_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=False,
                                   num_workers=0, collate_fn=collate_sequences)
    validation_loader = DataLoader(validation, batch_size=config["batch_size"], shuffle=False,
                                   num_workers=0, collate_fn=collate_sequences)
    model = FallGRU(config["input_size"], config["hidden_size"], config["num_layers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["features"], batch["lengths"])
            loss = loss_fn(logits, batch["labels"].float())
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise ValueError("Nonfinite training loss/logits")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Nonfinite gradients")
            optimizer.step()
            total_loss += float(loss.detach()) * len(logits)
        val_metrics, _ = evaluate(model, validation_loader, config["classification_threshold"])
        history.append(dict(epoch=epoch, train_optimization_loss=total_loss / len(train),
                            validation=val_metrics))
        LOGGER.info("epoch=%d train_loss=%.6f val_loss=%.6f val_f1=%.4f", epoch,
                    total_loss / len(train), val_metrics["loss"], val_metrics["f1"])
        if val_metrics["loss"] < best_loss:  # Strict improvement; ties retain earlier epoch.
            best_loss, best_epoch = val_metrics["loss"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config["early_stopping_patience"]:
            break
    if best_state is None:
        raise ValueError("No valid best epoch")
    model.load_state_dict(best_state)
    train_metrics, _ = evaluate(model, train_eval_loader, config["classification_threshold"])
    validation_metrics, predictions = evaluate(model, validation_loader, config["classification_threshold"])
    after = source_fingerprint(sources)
    if before != after:
        raise ValueError("Raw source poses changed during training")
    implementation_files = ("ml/models/gru.py", "ml/training/train_gru.py",
                            "ml/datasets/fall_sequence_dataset.py", "ml/preprocessing/pose_sequence.py")
    implementation_sha256 = {name: _sha256(REPO / name) for name in implementation_files}
    checkpoint = dict(schema_version="fall_gru_v1", state_dict=best_state,
                      model_config=model.config, preprocessing=config["preprocessing"],
                      classification_threshold=config["classification_threshold"],
                      seed=config["seed"], best_epoch=best_epoch,
                      training_base_commit=training_base_commit, stage33_commit=STAGE33_COMMIT,
                      implementation_sha256=implementation_sha256, training_config=config,
                      source_pose_fingerprint=before)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("xb") as handle:
        torch.save(checkpoint, handle)
    reloaded, _ = load_checkpoint(checkpoint_path)
    reload_metrics, reload_predictions = evaluate(reloaded, validation_loader, config["classification_threshold"])
    if reload_metrics != validation_metrics or reload_predictions != predictions:
        raise ValueError("Checkpoint reload changed validation outputs")
    summary = dict(stage=4, config=config, stage33_commit=STAGE33_COMMIT,
                   training_base_commit=training_base_commit, implementation_sha256=implementation_sha256,
                   config_sha256=_sha256(config_path), source_identity=read_contract()["source"],
                   source_pose_fingerprint_before=before, source_pose_fingerprint_after=after,
                   raw_pose_unchanged=True, test_subjects_accessed=[],
                   access_log=[dict(sequence_id=s.source_video, subject_id=s.subject_id, split=s.split)
                               for s in sources],
                   runtime=dict(python=platform.python_version(), numpy=np.__version__,
                                torch=str(torch.__version__), device="cpu", threads=1,
                                deterministic_algorithms=True),
                   datasets=dict(train=_dataset_summary(train), validation=_dataset_summary(validation)),
                   epochs_completed=len(history), best_epoch=best_epoch,
                   stopping_reason="patience" if stale_epochs >= config["early_stopping_patience"] else "max_epochs",
                   best_checkpoint_metrics=dict(train=train_metrics, validation=validation_metrics),
                   last_epoch_validation_metrics=history[-1]["validation"],
                   checkpoint=dict(path=checkpoint_path.relative_to(REPO).as_posix(),
                                   sha256=_sha256(checkpoint_path), bytes=checkpoint_path.stat().st_size,
                                   reload_validation_match=True), history=history)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "validation_predictions.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(predictions[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(predictions)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/stage4_gru.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = train_baseline(args.config.resolve(), args.output.absolute(), args.checkpoint.absolute())
    print(json.dumps(dict(best_epoch=summary["best_epoch"], epochs=summary["epochs_completed"],
                         metrics=summary["best_checkpoint_metrics"], checkpoint=summary["checkpoint"]), indent=2))
