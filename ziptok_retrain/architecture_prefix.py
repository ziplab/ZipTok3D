"""Sample exponential-grid prefix budgets and validate explicit token counts.

``exp`` chooses uniformly among 1, 2, 4, ... and the full token count. Each
sample retains a contiguous leading prefix; no latent values are zero-masked.
"""

from numbers import Integral
from typing import Optional, Union

import torch


def normalize_strategy(strategy: Optional[str]) -> Optional[str]:
    if strategy is None:
        return None
    value = str(strategy).lower()
    if value in {"none", "null", "off", "false"}:
        return None
    if value in {"exp", "exponential", "prefix_budget_uniform"}:
        return "exp"
    raise ValueError("nested_dropout_strategy must be 'exp' or null")


def prefix_budgets(num_latents: int) -> list:
    """Include the full budget even when it is not a power of two."""
    if isinstance(num_latents, bool) or not isinstance(num_latents, Integral) or num_latents < 1:
        raise ValueError("num_latents must be a positive integer")
    budgets = [1 << exponent for exponent in range(int(num_latents).bit_length())]
    if budgets[-1] != num_latents:
        budgets.append(int(num_latents))
    return budgets


def resolve_prefix_lengths(
    z: torch.Tensor,
    strategy: Optional[str],
    training: bool,
    active_num_latents: Optional[Union[int, torch.Tensor]] = None,
) -> torch.Tensor:
    """An explicit budget always wins; evaluation otherwise uses all tokens."""
    if z.ndim != 3 or z.size(0) < 1 or z.size(1) < 1:
        raise ValueError("latents must have nonempty [batch, tokens, channels] shape")
    batch_size, num_latents = z.shape[:2]
    if active_num_latents is not None:
        if isinstance(active_num_latents, bool):
            raise ValueError("active_num_latents must contain integer token counts")
        if isinstance(active_num_latents, Integral):
            lengths = torch.full((batch_size,), int(active_num_latents), dtype=torch.long, device=z.device)
        elif isinstance(active_num_latents, torch.Tensor):
            if active_num_latents.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
                raise ValueError("active_num_latents must contain integer token counts")
            if active_num_latents.ndim > 1 or active_num_latents.numel() not in {1, batch_size}:
                raise ValueError("active_num_latents needs one count or one count per sample")
            lengths = active_num_latents.to(device=z.device, dtype=torch.long).reshape(-1)
            if lengths.numel() == 1:
                lengths = lengths.expand(batch_size)
        else:
            raise ValueError("active_num_latents must be an integer or integer tensor")
        if torch.any((lengths < 1) | (lengths > num_latents)):
            raise ValueError(f"active_num_latents must be between 1 and {num_latents}")
        return lengths
    if training and normalize_strategy(strategy) == "exp":
        budgets = torch.tensor(prefix_budgets(num_latents), dtype=torch.long, device=z.device)
        return budgets[torch.randint(budgets.numel(), (batch_size,), device=z.device)]
    return torch.full((batch_size,), num_latents, dtype=torch.long, device=z.device)
