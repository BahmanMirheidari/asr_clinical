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
