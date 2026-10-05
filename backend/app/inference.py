"""Per-source rolling inference for the frozen Stage 5 checkpoint, CPU only."""

import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ml.models.gru import load_checkpoint

REPO = Path(__file__).resolve().parents[2]


def load_selected_model():
    config = json.loads((REPO / "artifacts/evaluation/stage5/final_ml_config.json").read_text())
    path = Path(os.getenv("CHECKPOINT_PATH", str(REPO / config["checkpoint"]["path"])))
    if not path.is_file():
        raise FileNotFoundError(f"Selected Stage 5 checkpoint missing: {path}")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != config["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint SHA256 mismatch; normalized Stage 4 substitution is forbidden")
    torch.set_num_threads(1)
    model, metadata = load_checkpoint(path)
    if (metadata["preprocessing"] != config["training_config"]["preprocessing"]
            or metadata["model_config"] != {k: config["model"][k] for k in ("input_size", "hidden_size", "num_layers")}
            or metadata["classification_threshold"] != 0.5
            or metadata["preprocessing"]["normalize_pose"] is not False):
        raise ValueError("Frozen Stage 5 model/preprocessing mismatch")
    return model, config


@dataclass
class SourceBuffer:
    sequence_id: str
    features: deque = field(default_factory=lambda: deque(maxlen=200))
    last_index: int = -1
    last_timestamp: int = -1
    count: int = 0
    next_inference: int = 86
    timing: tuple | None = None


class InferenceEngine:
    """Infer at context 86 then every 10 consecutive new frames, retain latest 200."""

    def __init__(self, model, threshold: float = 0.5) -> None:
        if threshold != 0.5:
            raise ValueError("Threshold is frozen at 0.5")
        self.model = model
        self.threshold = threshold
        self.sources: dict[str, SourceBuffer] = {}
        self.frames_received = 0
        self.predictions = 0
        self.ignored = 0
        self.gap_resets = 0

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        source, sequence = message["source_id"], message["sequence_id"]
        kind = message["message_type"]
        if kind == "sequence_start":
            if source in self.sources and self.sources[source].sequence_id == sequence:
                return None  # QoS1 redelivery of start is idempotent.
            if len(self.sources) >= 64 and source not in self.sources:
                raise ValueError("Too many active sources")
            self.sources[source] = SourceBuffer(sequence)
            return None
        state = self.sources.get(source)
        if state is None or state.sequence_id != sequence:
            self.ignored += 1
            return None
        if kind == "sequence_end":
            del self.sources[source]
            return None
        index, timestamp = message["frame_index"], message["timestamp_ms"]
        timing = (message["timestamp_kind"], message["fps"])
        if index <= state.last_index or timestamp <= state.last_timestamp:
            self.ignored += 1  # QoS1 duplicates and late messages never enter history.
            return None
        if state.timing is not None and state.timing != timing:
            raise ValueError("FPS/timestamp convention changed inside sequence")
        if index != state.last_index + 1:
            state.features.clear()
            state.count, state.next_inference = 0, 86
            self.gap_resets += 1
        state.timing = timing
        state.last_index, state.last_timestamp = index, timestamp
        state.features.append(message["features"])
        state.count += 1
        self.frames_received += 1
        if state.count < state.next_inference:
            return None
        features = torch.tensor(list(state.features), dtype=torch.float32).unsqueeze(0)
        lengths = torch.tensor([features.shape[1]], dtype=torch.int64)
        with torch.no_grad():
            probability = float(self.model(features, lengths).sigmoid()[0])
        if not 0 <= probability <= 1:
            raise ValueError("Nonfinite/out-of-range inference probability")
        state.next_inference += 10
        self.predictions += 1
        return dict(schema_version=1, source_id=source, sequence_id=sequence,
                    frame_index=index, timestamp_ms=timestamp, timestamp_kind=message["timestamp_kind"],
                    buffer_length=len(state.features), fall_probability=probability,
                    threshold=self.threshold, predicted_label="fall" if probability >= self.threshold else "non_fall")
