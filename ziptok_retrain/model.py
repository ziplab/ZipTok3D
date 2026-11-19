"""CompactLatentAutoencoder whose training-time prefix budgets follow a probability vector."""
from typing import Optional, Sequence
import torch
from .autoencoder import CompactLatentAutoencoder
from .prefix import normalize_weights, sample_lengths


class WeightedPrefixAutoencoder(CompactLatentAutoencoder):

    def __init__(self,
                 output_dim: int,
                 num_latents: int,
                 embed_dim: int,
                 query_dim: int,
                 encoder_params: dict,
                 decoder_params: dict,
                 use_learnable_pos: bool = False,
                 nested_dropout_strategy: str = "exp",
                 prefix_budget_weights: Optional[Sequence[float]] = None,
                 ):
        super().__init__(output_dim=output_dim, num_latents=num_latents, embed_dim=embed_dim,
                         query_dim=query_dim, encoder_params=encoder_params, decoder_params=decoder_params,
                         use_learnable_pos=use_learnable_pos, nested_dropout_strategy=nested_dropout_strategy)
        if self.nested_dropout_strategy != 'exp':
            raise ValueError('WeightedPrefixAutoencoder requires nested_dropout_strategy=exp')
        budgets = self.prefix_budgets
        weights = prefix_budget_weights if prefix_budget_weights is not None else [1.0] * len(budgets)
        self._budget_values = torch.tensor(budgets, dtype=torch.long)
        self._budget_probs = normalize_weights(weights, len(budgets))

    def set_budget_probs(self, probs: torch.Tensor) -> None:
        probs = torch.as_tensor(probs, dtype=torch.float64)
        self._budget_probs = normalize_weights(probs.tolist(), self._budget_values.numel())

    @property
    def budget_probs(self) -> torch.Tensor:
        return self._budget_probs

    def _prefix_lengths(self, z, active_num_latents=None):
        if active_num_latents is None and self.training:
            budgets = self._budget_values.to(z.device)
            return sample_lengths(budgets, self._budget_probs, z.size(0))
        return super()._prefix_lengths(z, active_num_latents)
