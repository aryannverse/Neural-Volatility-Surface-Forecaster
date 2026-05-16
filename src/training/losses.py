from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SurfaceLossConfig:
    base_loss: str = "huber"  # mse | mae | huber
    smoothness_weight: float = 0.05


class SurfaceLoss(nn.Module):
    def __init__(self, cfg: SurfaceLossConfig | None = None):
        super().__init__()
        self.cfg = cfg or SurfaceLossConfig()

    def _base(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mode = self.cfg.base_loss.lower()
        if mode == "mse":
            return F.mse_loss(pred, target)
        if mode == "mae":
            return F.l1_loss(pred, target)
        return F.huber_loss(pred, target, delta=0.05)

    @staticmethod
    def _smoothness_penalty(pred: torch.Tensor) -> torch.Tensor:
        dx = pred[:, :, 1:] - pred[:, :, :-1]
        dy = pred[:, 1:, :] - pred[:, :-1, :]
        return (dx.pow(2).mean() + dy.pow(2).mean()) * 0.5

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self._base(pred, target)
        smooth = self._smoothness_penalty(pred)
        return base + self.cfg.smoothness_weight * smooth
