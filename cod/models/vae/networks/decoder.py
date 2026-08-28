"""COD-VAE triplane decoder with ZipTok3D's shared refinement loop.

The selection stage is executed once. At each pass, the shared Transformer
processes only the 192 selected triplane tokens and the physical K-token latent
prefix. The remaining 576 initialized triplane tokens bypass every recurrent
update and are restored at their original spatial locations.
"""

import functools
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from ..modules.blocks import (
    CrossTransformerBlock,
    GEGLU,
    StandardTransformerBlock,
    init_embedding,
)


class CompactTriplaneDecoder(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        query_dim: int,
        output_resolution: int,
        output_patch_size: int,
        num_layers: int,
        num_init_layers: int,
        num_loops: int = 1,
        keep_ratio: float = 0.5,
        use_conv_refine: bool = False,
        num_heads: int = 8,
        num_register: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        prune_dropout: float = 0.1,
    ):
        super().__init__()
        if num_loops < 1:
            raise ValueError("num_loops must be positive")

        self.embed_dim = embed_dim
        self.query_dim = query_dim
        self.output_resolution = output_resolution
        self.output_patch_size = output_patch_size
        self.plane_resolution = output_resolution // output_patch_size
        self.num_output_patches = 3 * self.plane_resolution ** 2
        self.keep_ratio = keep_ratio
        self.num_loops = num_loops

        self.init_transformer = CrossTransformerBlock(
            embed_dim=embed_dim,
            num_layers=num_init_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=prune_dropout,
            num_register=num_register,
        )
        # One physical block, recurrently reused.  This preserves the COD-VAE
        # parameter names while changing effective decoder depth from d to d*L.
        self.transformer = StandardTransformerBlock(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_register=num_register,
            use_ln=True,
        )

        patch_head_dim = query_dim * output_patch_size ** 2
        self.init_out = nn.Linear(embed_dim, patch_head_dim)
        self.decoder_out = nn.Linear(embed_dim, patch_head_dim)
        self.uncertainty_out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            GEGLU(),
            nn.Linear(embed_dim, 1),
        )

        self.conv_refine = None
        if use_conv_refine:
            activation = nn.LeakyReLU(0.2, inplace=True)
            conv = functools.partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)
            self.conv_refine = nn.Sequential(
                conv(query_dim, query_dim), activation,
                conv(query_dim, query_dim), activation,
                conv(query_dim, query_dim),
            )

        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))
        self.mask_pos = nn.Parameter(torch.empty(self.num_output_patches, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        scale = self.embed_dim ** -0.5
        init_embedding(self.mask_pos, scale)
        init_embedding(self.mask_token, scale)

    def decode(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_loops: Optional[int] = None,
        student_loop: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        loops = self.num_loops if num_loops is None else int(num_loops)
        if not 1 <= loops <= self.num_loops:
            raise ValueError(f"num_loops must be in [1, {self.num_loops}], got {loops}")
        if student_loop is not None and not 1 <= student_loop <= loops:
            raise ValueError(f"student_loop must be in [1, {loops}], got {student_loop}")

        batch_size = z.size(0)
        tokens = self.mask_pos.unsqueeze(0).expand(batch_size, -1, -1)
        init_tokens, final_tokens, uncertainty, sampled_tokens = self.decode_tokens(
            tokens=tokens,
            z=z,
            z_mask=mask,
            num_loops=loops,
            student_loop=student_loop,
        )

        init_patches = self.init_out(init_tokens)
        final_planes = self._tokens_to_planes(final_tokens, init_patches, uncertainty)
        init_planes = self.patches_to_planes(init_patches)
        student_planes = None
        if sampled_tokens is not None:
            student_planes = self._tokens_to_planes(sampled_tokens, init_patches, uncertainty)

        uncertainty_planes = uncertainty.view(
            uncertainty.size(0), 3, 1, self.plane_resolution, self.plane_resolution
        )
        if self.conv_refine is not None:
            final_planes = self._apply_conv_refine(final_planes)
            init_planes = self._apply_conv_refine(init_planes)
            if student_planes is not None:
                student_planes = self._apply_conv_refine(student_planes)

        return final_planes, init_planes, uncertainty_planes, student_planes

    def decode_tokens(
        self,
        tokens: torch.Tensor,
        z: torch.Tensor,
        z_mask: Optional[torch.Tensor],
        num_loops: int,
        student_loop: Optional[int],
    ):
        if z_mask is not None:
            z_mask = z_mask.to(device=z.device, dtype=torch.bool)
            if z_mask.shape != z.shape[:2]:
                raise ValueError(
                    f"latent mask shape {tuple(z_mask.shape)} does not match {tuple(z.shape[:2])}"
                )

        # COD-VAE selection S_omega.
        init_tokens = self.init_transformer(tokens, z, source_mask=z_mask)
        uncertainty = torch.sigmoid(self.uncertainty_out(init_tokens))
        selected, indices = self._select_by_uncertainty(
            init_tokens, uncertainty, self.keep_ratio
        )
        selected = selected + self.mask_token.unsqueeze(0)

        # ZipTok3D shared update g_theta. Z_:K is reinserted unchanged at each
        # pass, and only the 192-token selected state is carried forward.
        state = selected
        sampled_state = None
        for loop_idx in range(1, num_loops + 1):
            sequence = torch.cat([state, z], dim=1)

            key_padding_mask = None
            if z_mask is not None:
                fixed_mask = torch.zeros(
                    z.size(0), state.size(1), dtype=torch.bool, device=z.device
                )
                key_padding_mask = torch.cat([fixed_mask, z_mask], dim=1)

            updated = self.transformer(sequence, key_padding_mask=key_padding_mask)
            state = updated[:, :selected.size(1)]
            if student_loop == loop_idx:
                sampled_state = state

        final_tokens = self._restore_full_tokens(init_tokens, state, indices)
        sampled_tokens = None
        if sampled_state is not None:
            sampled_tokens = self._restore_full_tokens(init_tokens, sampled_state, indices)
        return init_tokens, final_tokens, uncertainty, sampled_tokens

    def _restore_full_tokens(self, init_tokens, selected, indices):
        full_tokens = init_tokens.clone()
        expanded = indices.expand(-1, -1, full_tokens.size(-1))
        return full_tokens.scatter(1, expanded, selected)

    def _tokens_to_planes(self, tokens, init_patches, uncertainty):
        residual = self.decoder_out(tokens)
        patches = init_patches + uncertainty * residual
        return self.patches_to_planes(patches)

    def _apply_conv_refine(self, planes):
        batch_size = planes.size(0)
        flat = planes.flatten(0, 1)
        flat = self.conv_refine(flat) + flat
        return flat.view(
            batch_size, 3, self.query_dim, self.output_resolution, self.output_resolution
        )

    def patches_to_planes(self, patches):
        patches = patches.view(
            patches.size(0), 3, self.plane_resolution, self.plane_resolution, -1
        )
        return rearrange(
            patches,
            "b i h w (p q d) -> b i d (h p) (w q)",
            p=self.output_patch_size,
            q=self.output_patch_size,
            d=self.query_dim,
        )

    def _select_by_uncertainty(self, tokens, uncertainty, ratio):
        indices = torch.argsort(uncertainty, dim=1, descending=True)
        num_remain = max(1, int(indices.size(1) * ratio))
        indices = indices[:, :num_remain]
        selected = torch.gather(tokens, 1, indices.expand(-1, -1, tokens.size(-1)))
        return selected, indices

    def decode_queries(self, planes: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        queries = queries.clamp(-1, 0.999)
        features = torch.zeros(
            queries.size(0), queries.size(1), self.query_dim,
            device=queries.device, dtype=planes.dtype,
        )
        return _decode_queries_with_plane(features, planes, queries, mode="sum")

    def decode_uncertainty(self, planes, queries):
        queries = queries.clamp(-1, 0.999)
        uncertainty = torch.ones(
            queries.size(0), queries.size(1), 1,
            device=queries.device, dtype=planes.dtype,
        )
        return _decode_queries_with_plane(uncertainty, planes, queries, mode="mult")


def _decode_queries_with_plane(query_features, planes, queries, mode="sum"):
    for axis in range(3):
        plane = planes[:, axis]
        plane_queries = torch.stack(
            [queries[..., j] for j in range(3) if j != axis], dim=-1
        ).to(dtype=plane.dtype)
        features = F.grid_sample(
            plane, plane_queries.unsqueeze(2), align_corners=False
        ).squeeze(-1).transpose(1, 2)
        if mode == "sum":
            query_features = query_features + features
        elif mode == "mult":
            query_features = query_features * features
        else:
            raise NotImplementedError(mode)
    return query_features
