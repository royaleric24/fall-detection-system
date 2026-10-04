"""Small unidirectional binary GRU; lengths exclude right-side batch padding."""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class FallGRU(nn.Module):
    """[B,T,input_size] + CPU int64 [B] lengths -> raw logits [B]."""

    def __init__(self, input_size: int = 132, hidden_size: int = 64,
                 num_layers: int = 1) -> None:
        super().__init__()
        self.config = dict(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          bidirectional=False)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if (features.ndim != 3 or features.shape[2] != self.config["input_size"]
                or lengths.shape != (features.shape[0],) or lengths.dtype != torch.int64
                or lengths.device.type != "cpu" or features.shape[0] == 0
                or torch.any(lengths <= 0) or torch.any(lengths > features.shape[1])):
            raise ValueError("Expected [B,T,F] and positive CPU int64 lengths <= T")
        packed = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.classifier(hidden[-1]).squeeze(-1)


def load_checkpoint(path: Path) -> tuple[FallGRU, dict[str, Any]]:
    """Load the simple Stage 4 checkpoint on CPU in eval mode; no sigmoid in model."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "fall_gru_v1":
        raise ValueError("Unsupported checkpoint schema")
    model = FallGRU(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint
