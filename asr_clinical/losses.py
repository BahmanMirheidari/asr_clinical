from __future__ import annotations

import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)
        pt = torch.exp(-ce_loss)
        return ((1.0 - pt) ** self.gamma * ce_loss).mean()


def make_loss(task: str, loss_name: str, class_weight_tensor=None, focal_gamma: float = 2.0):
    if task == "regression":
        return nn.MSELoss()
    if loss_name == "focal":
        return FocalLoss(weight=class_weight_tensor, gamma=focal_gamma)
    return nn.CrossEntropyLoss(weight=class_weight_tensor)


def build_y_grid(y_values, size=101, minimum=None, maximum=None):
    """Build a uniform RAFT grid on the scale used by the training labels."""
    values = torch.as_tensor(y_values, dtype=torch.float32).flatten()
    if values.numel() == 0:
        raise ValueError("Cannot build a regression target grid from no labels")
    lo = float(values.min()) if minimum is None else float(minimum)
    hi = float(values.max()) if maximum is None else float(maximum)
    if hi <= lo:
        hi = lo + 1.0
    if int(size) < 2:
        raise ValueError("y_grid_size must be at least 2")
    return torch.linspace(lo, hi, int(size))


def decode_expected_value(logits, y_grid, temperature=1.0):
    probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
    return (probs * y_grid.to(device=logits.device, dtype=logits.dtype)).sum(-1)


def raft_expected_value_loss(logits, targets, y_grid, temperature=1.0):
    prediction = decode_expected_value(logits, y_grid, temperature)
    return torch.nn.functional.mse_loss(prediction, targets.to(prediction.dtype))
