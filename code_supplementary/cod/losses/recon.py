from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class OccupancyReconstructionLoss(nn.Module):
    def __init__(self, vol_coeff: float = 1.0, near_coeff=0.1):
        super().__init__()

        self.vol_coeff = vol_coeff
        self.near_coeff = near_coeff

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                num_vol_points: int = -1, reduction: str = 'mean'):
        if num_vol_points < 0:
            num_vol_points = labels.size(1) // 2
        vol_loss = F.binary_cross_entropy_with_logits(logits[:, :num_vol_points], labels[:, :num_vol_points],
                                                      reduction=reduction)
        near_loss = F.binary_cross_entropy_with_logits(logits[:, num_vol_points:], labels[:, num_vol_points:],
                                                       reduction=reduction)
        recon_loss = self.vol_coeff * vol_loss + self.near_coeff * near_loss

        return recon_loss, vol_loss, near_loss


class UncertaintyLoss(nn.Module):

    def __init__(self, uncertainty_range: Tuple[float, float] = (0.01, 1)):
        super().__init__()
        self.start, self.end = uncertainty_range

    def forward(self, uncertainty: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):
        uncertainty = uncertainty.squeeze(-1)
        query_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        query_loss = (query_loss - self.start).clamp(0, self.end - self.start) / (self.end - self.start)
        return F.mse_loss(uncertainty, query_loss.detach())
