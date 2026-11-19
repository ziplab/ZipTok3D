"""Weighted prefix-budget sampling with an optional linear anneal between two weight vectors."""
from typing import Optional, Sequence
import torch


def normalize_weights(weights: Sequence[float], num_budgets: int) -> torch.Tensor:
    values = torch.as_tensor([float(w) for w in weights], dtype=torch.float64)
    if values.numel() != num_budgets:
        raise ValueError(f'expected {num_budgets} budget weights, got {values.numel()}')
    if not torch.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError('budget weights must be finite, nonneg and not all zero')
    return values / values.sum()


def scheduled_probs(epoch: float,
                    start: torch.Tensor,
                    final: Optional[torch.Tensor] = None,
                    anneal_epochs: Optional[Sequence[float]] = None) -> torch.Tensor:
    if final is None or anneal_epochs is None:
        return start
    e0, e1 = float(anneal_epochs[0]), float(anneal_epochs[1])
    if not 0 <= e0 < e1:
        raise ValueError('prefix_weight_anneal_epochs must satisfy 0 <= start < end')
    t = min(1.0, max(0.0, (float(epoch) - e0) / (e1 - e0)))
    probs = (1.0 - t) * start + t * final
    return probs / probs.sum()


def sample_lengths(budgets: torch.Tensor, probs: torch.Tensor, batch_size: int) -> torch.Tensor:
    index = torch.multinomial(probs.to(device=budgets.device, dtype=torch.float32),
                              batch_size, replacement=True)
    return budgets[index]
