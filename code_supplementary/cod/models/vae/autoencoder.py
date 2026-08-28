import math
from typing import Optional, Union

import torch
from torch import nn

from pointops.functions import pointops
from .base import BaseAutoencoder
from .modules.pos import PointEmbed
from .modules.transformer import init_embedding
from .networks import CompactPointPatchEncoder, CompactTriplaneDecoder


class CompactLatentAutoencoder(BaseAutoencoder):
    """ZipTok3D stage-1 tokenizer built on the COD-VAE autoencoder."""

    def __init__(
        self,
        output_dim: int,
        num_latents: int,
        embed_dim: int,
        query_dim: int,
        encoder_params: dict,
        decoder_params: dict,
        use_learnable_pos: bool = False,
        nested_dropout_strategy: Optional[str] = "prefix_budget_uniform",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_latents = num_latents
        self.query_dim = query_dim
        self.nested_dropout_strategy = self._normalize_strategy(nested_dropout_strategy)
        self.use_learnable_pos = use_learnable_pos

        self.point_embed = PointEmbed(dim=embed_dim)
        self.norm_latent = nn.LayerNorm(embed_dim)
        self.encoder = CompactPointPatchEncoder(embed_dim=embed_dim, **encoder_params)
        self.decoder = CompactTriplaneDecoder(
            embed_dim=embed_dim, query_dim=query_dim, **decoder_params
        )
        self.head = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, output_dim),
        )
        self.latent_pos = (
            nn.Parameter(torch.empty(num_latents, embed_dim))
            if use_learnable_pos else None
        )
        self.reset_parameters()

    @staticmethod
    def _normalize_strategy(strategy):
        if strategy is None:
            return None
        strategy = str(strategy).lower()
        return None if strategy in {"none", "null", "false", "off"} else strategy

    @property
    def prefix_budgets(self):
        budgets = [2 ** i for i in range(int(math.log2(self.num_latents)) + 1)]
        if budgets[-1] != self.num_latents:
            budgets.append(self.num_latents)
        return budgets

    def reset_parameters(self):
        if self.latent_pos is not None:
            init_embedding(self.latent_pos, self.embed_dim ** -0.5)

    def make_prefix_mask(
        self,
        batch_size: int,
        num_latents: int,
        keep: Union[int, torch.Tensor],
        device,
    ) -> torch.Tensor:
        if isinstance(keep, int):
            keep = torch.full((batch_size,), keep, dtype=torch.long, device=device)
        else:
            keep = keep.to(device=device, dtype=torch.long).view(-1)
            if keep.numel() == 1:
                keep = keep.expand(batch_size)
        if keep.numel() != batch_size:
            raise ValueError("one prefix length is required per batch item")
        keep = keep.clamp_(1, num_latents)
        positions = torch.arange(num_latents, device=device).unsqueeze(0)
        return positions >= keep.unsqueeze(1)

    def generate_nested_dropout_mask(self, batch_size, num_latents, device=None):
        device = device or next(self.parameters()).device
        if self.nested_dropout_strategy is None:
            return torch.zeros(batch_size, num_latents, dtype=torch.bool, device=device)
        if self.nested_dropout_strategy == "uniform":
            keep = torch.randint(1, num_latents + 1, (batch_size,), device=device)
        elif self.nested_dropout_strategy in {"prefix_budget_uniform", "exponential"}:
            budgets = torch.tensor(
                [x for x in self.prefix_budgets if x <= num_latents],
                dtype=torch.long,
                device=device,
            )
            sampled = torch.randint(0, budgets.numel(), (batch_size,), device=device)
            keep = budgets[sampled]
        else:
            raise ValueError(f"unknown nested dropout strategy: {self.nested_dropout_strategy}")
        return self.make_prefix_mask(batch_size, num_latents, keep, device)

    def _resolve_mask(self, z, mask=None, active_num_latents=None):
        if mask is not None:
            mask = mask.to(device=z.device, dtype=torch.bool)
            if mask.shape != z.shape[:2]:
                raise ValueError(
                    f"prefix mask shape {tuple(mask.shape)} does not match "
                    f"latent shape {tuple(z.shape[:2])}"
                )
            return mask
        if active_num_latents is not None:
            return self.make_prefix_mask(
                z.size(0), z.size(1), int(active_num_latents), z.device
            )
        if self.training and self.nested_dropout_strategy is not None:
            return self.generate_nested_dropout_mask(z.size(0), z.size(1), z.device)
        return torch.zeros(z.shape[:2], dtype=torch.bool, device=z.device)

    def _prepare_decoder_inputs(
        self,
        z,
        mask=None,
        active_num_latents=None,
    ):
        if active_num_latents is not None:
            active = int(active_num_latents)
            if not 1 <= active <= z.size(1):
                raise ValueError(
                    f"active_num_latents must be in [1, {z.size(1)}], got {active}"
                )
            z = z[:, :active]
            if mask is not None:
                mask = mask[:, :active]
        return z, self._resolve_mask(z, mask=mask)

    def forward(
        self,
        pc,
        queries,
        active_num_latents: Optional[int] = None,
        num_decode_loops: Optional[int] = None,
        student_loop: Optional[int] = None,
    ):
        z, posterior = self.encode(pc)
        z, mask = self._prepare_decoder_inputs(
            z, active_num_latents=active_num_latents
        )
        planes, init_planes, uncertainty, student_planes = self.decode(
            z,
            mask=mask,
            num_loops=num_decode_loops,
            student_loop=student_loop,
        )
        outputs = {
            "logits": self.decode_queries(planes, queries),
            "planes": planes,
            "prefix_lengths": (~mask).sum(dim=1),
        }
        if init_planes is not None and self.training:
            outputs["init_out"] = self.decode_queries(init_planes, queries)
        if uncertainty is not None:
            outputs["uncertainty"] = self.decoder.decode_uncertainty(
                uncertainty, queries
            )
            outputs["uncertainty_planes"] = uncertainty
        if student_planes is not None:
            outputs["student_planes"] = student_planes
            outputs["student_logits"] = self.decode_queries(student_planes, queries)
        return outputs

    def encode_embed(self, pc):
        z = self.norm_latent(self._get_init_z(pc))
        return self.encoder(pc, z, self.point_embed)

    def encode_latents(self, z, **kwargs):
        return z, None

    def decode_latents(self, z, mask=None, active_num_latents=None, **kwargs):
        return self._prepare_decoder_inputs(
            z, mask=mask, active_num_latents=active_num_latents
        )

    def decode_embed(
        self,
        z,
        mask=None,
        active_num_latents=None,
        num_loops=None,
        num_decode_loops=None,
        student_loop=None,
        **kwargs,
    ):
        z, mask = self._prepare_decoder_inputs(
            z, mask=mask, active_num_latents=active_num_latents
        )
        loops = num_decode_loops if num_decode_loops is not None else num_loops
        return self._decode_physical_prefixes(
            z, mask, num_loops=loops, student_loop=student_loop
        )

    def _decode_physical_prefixes(self, z, mask, num_loops, student_loop):
        """Decode each budget group with a physical K-token prefix."""
        prefix_lengths = (~mask).sum(dim=1)
        if torch.any(prefix_lengths < 1):
            raise ValueError("every decoder input must retain at least one latent token")
        positions = torch.arange(z.size(1), device=z.device).unsqueeze(0)
        expected_mask = positions >= prefix_lengths.unsqueeze(1)
        if not torch.equal(mask, expected_mask):
            raise ValueError("decoder masks must represent contiguous leading prefixes")

        grouped = []
        grouped_indices = []
        for prefix_length_tensor in torch.unique(prefix_lengths, sorted=True):
            prefix_length = int(prefix_length_tensor.item())
            indices = torch.nonzero(
                prefix_lengths == prefix_length_tensor, as_tuple=False
            ).flatten()
            z_group = z.index_select(0, indices)[:, :prefix_length]
            grouped.append(self.decoder.decode(
                z_group,
                mask=None,
                num_loops=num_loops,
                student_loop=student_loop,
            ))
            grouped_indices.append(indices)

        concatenated_indices = torch.cat(grouped_indices, dim=0)
        restore_order = torch.argsort(concatenated_indices)
        restored = []
        for output_index in range(len(grouped[0])):
            parts = [outputs[output_index] for outputs in grouped]
            if parts[0] is None:
                if any(part is not None for part in parts):
                    raise RuntimeError("inconsistent optional decoder outputs across budgets")
                restored.append(None)
            else:
                restored.append(
                    torch.cat(parts, dim=0).index_select(0, restore_order)
                )
        return tuple(restored)

    def decode_queries(self, context, queries):
        features = self.decoder.decode_queries(context, queries)
        return self.head(features).squeeze(-1)

    def _get_init_z(self, pc):
        if self.latent_pos is None:
            return self.point_embed(pointops.fps(pc, self.num_latents))
        return self.latent_pos.unsqueeze(0).expand(pc.size(0), -1, -1)
