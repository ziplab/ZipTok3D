"""COD-VAE decoder with optional recurrent refinement and one intermediate plane output.

num_loops=1 retains the original computation and state keys. Additional passes
reuse the same transformer, carrying selected/merged tokens while conditioning
on the original z each time. Initialization, pruning and merging happen once;
registers retain the transformer's original per-call creation/removal semantics.
"""

from typing import Tuple, List, Callable
from numbers import Integral
import functools

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from cod.models.vae.modules.blocks import StandardTransformerBlock, CrossTransformerBlock, init_embedding, GEGLU
from cod.models.vae.modules.transformer import CrossAttn


class CompactTriplaneDecoder(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 query_dim: int,
                 output_resolution: int,
                 output_patch_size: int,
                 num_layers: int,
                 num_init_layers: int,

                 keep_ratio: float = 0.5,
                 num_merged_tokens: int = -1,
                 use_conv_refine: bool = False,

                 ## transformer params
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 prune_dropout: float = 0.1,
                 num_loops: int = 1,

                 ):
        super().__init__()
        if isinstance(num_loops, bool) or not isinstance(num_loops, Integral) or num_loops < 1:
            raise ValueError("num_loops must be a positive integer")
        self.num_loops = int(num_loops)

        self.embed_dim = embed_dim
        self.query_dim = query_dim
        self.output_resolution = output_resolution
        self.output_patch_size = output_patch_size
        self.plane_resolution = output_resolution // output_patch_size
        self.num_output_patches = 3 * (self.plane_resolution ** 2)
        self.keep_ratio = keep_ratio

        self.init_transformer = CrossTransformerBlock(embed_dim=embed_dim, num_layers=num_init_layers,
                                                      num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=prune_dropout)
        self.transformer = StandardTransformerBlock(embed_dim=embed_dim, num_layers=num_layers,
                                                    num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                                                    use_ln=True,
                                                    )

        patch_head_dim = query_dim * (output_patch_size ** 2)
        self.init_out = nn.Linear(embed_dim, patch_head_dim)
        self.decoder_out = nn.Linear(embed_dim, patch_head_dim)
        self.uncertainty_out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            GEGLU(),
            nn.Linear(embed_dim, 1),
        )
        self.merging_module = None
        if num_merged_tokens > 0:
            self.merging_module = _MergingModule(embed_dim=embed_dim, num_heads=num_heads,
                                                 num_merged=num_merged_tokens)

        self.conv_refine = None
        if use_conv_refine:
            conv_activation = nn.LeakyReLU(0.2, inplace=True)
            conv_layer = functools.partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)
            self.conv_refine = nn.Sequential(
                conv_layer(query_dim, query_dim),
                conv_activation,
                conv_layer(query_dim, query_dim),
                conv_activation,
                conv_layer(query_dim, query_dim),
            )

        ## parameters
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))
        self.mask_pos = nn.Parameter(torch.empty(self.num_output_patches, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        scale = self.embed_dim ** -0.5
        init_embedding(self.mask_pos, scale)
        init_embedding(self.mask_token, scale)

    def decode(self, z: torch.Tensor, intermediate_step=None):
        """Return three outputs by default, or append planes from one earlier pass."""
        B = z.size(0)
        tokens = self.mask_pos.unsqueeze(0).expand(B, -1, -1)

        decoded = self.decode_tokens(tokens, z, intermediate_step=intermediate_step)
        init_tokens, tokens, uncertainty = decoded[:3]
        patches = self.decoder_out(tokens)
        init_patches = self.init_out(init_tokens)
        patches = init_patches + uncertainty * patches

        intermediate_planes = None
        if intermediate_step is not None:
            intermediate_patches = init_patches + uncertainty * self.decoder_out(decoded[3])
            intermediate_planes = self.patches_to_planes(intermediate_patches)
        planes = self.patches_to_planes(patches)
        uncertainty = uncertainty.view(uncertainty.size(0), 3, 1, self.plane_resolution, self.plane_resolution)
        init_planes = self.patches_to_planes(init_patches)

        if self.conv_refine is not None:
            planes = planes.flatten(0, 1)
            planes = self.conv_refine(planes) + planes
            planes = planes.view(B, 3, self.query_dim, self.output_resolution, self.output_resolution)

            init_planes = init_planes.flatten(0, 1)
            init_planes = self.conv_refine(init_planes) + init_planes
            init_planes = init_planes.view(B, 3, self.query_dim, self.output_resolution, self.output_resolution)

            if intermediate_planes is not None:
                intermediate_planes = intermediate_planes.flatten(0, 1)
                intermediate_planes = self.conv_refine(intermediate_planes) + intermediate_planes
                intermediate_planes = intermediate_planes.view(
                    B, 3, self.query_dim, self.output_resolution, self.output_resolution)

        if intermediate_step is not None:
            return planes, init_planes, uncertainty, intermediate_planes
        return planes, init_planes, uncertainty

    def patches_to_planes(self, patches):
        patches = patches.view(patches.size(0), 3, self.plane_resolution, self.plane_resolution, -1)
        planes = rearrange(patches, 'b i h w (p q d) -> b i d (h p) (w q)',
                           p=self.output_patch_size, q=self.output_patch_size, d=self.query_dim)

        return planes

    def _validate_intermediate_step(self, intermediate_step):
        if intermediate_step is None:
            return None
        if (isinstance(intermediate_step, bool) or not isinstance(intermediate_step, Integral)
                or not 1 <= intermediate_step < self.num_loops):
            raise ValueError("intermediate_step must be an integer from 1 to num_loops - 1")
        return int(intermediate_step)

    def decode_tokens(self, tokens, z, intermediate_step=None):
        intermediate_step = self._validate_intermediate_step(intermediate_step)
        ## prune decoder
        init_tokens = self.init_transformer(tokens, z)
        uncertainty = self.uncertainty_out(init_tokens)
        uncertainty = self.apply_uncertainty_activation(uncertainty)

        L_full = tokens.size(1)
        tokens, indices, prune_indices = self._select_by_uncertainty(init_tokens, uncertainty, self.keep_ratio)
        if self.mask_token is not None:
            tokens = tokens + self.mask_token.unsqueeze(0)

        L = tokens.size(1)
        if self.merging_module is not None:
            pruned = torch.gather(init_tokens, 1, prune_indices.expand(-1, -1, tokens.size(-1)))
            merged = self.merging_module(pruned)
            tokens = torch.cat([tokens, merged], dim=1)

        ## Refine selected and merged state; z remains the original conditioning.
        state_length = tokens.size(1)
        intermediate_tokens = None
        for step in range(1, self.num_loops + 1):
            x = torch.cat([tokens, z], dim=1)
            tokens = self.transformer(x)[:, :state_length]
            if step == intermediate_step:
                intermediate_tokens = self._restore_full_tokens(tokens[:, :L], init_tokens, indices, L_full)

        ## reconstructing full tokens
        full_tokens = self._restore_full_tokens(tokens[:, :L], init_tokens, indices, L_full)
        if intermediate_step is not None:
            return init_tokens, full_tokens, uncertainty, intermediate_tokens
        return init_tokens, full_tokens, uncertainty

    def _restore_full_tokens(self, tokens, init_tokens, indices, L_full):
        """Scatter selected state back, preserving the original unselected initialization."""
        full_tokens = torch.zeros(tokens.size(0), L_full, tokens.size(2),
                                  device=tokens.device, dtype=tokens.dtype)
        mask = torch.ones(tokens.size(0), L_full, device=tokens.device, dtype=torch.bool)
        mask.scatter_(1, indices.squeeze(-1), 0)

        full_tokens[mask] = init_tokens[mask]
        _indices = indices.expand(-1, -1, full_tokens.size(-1))
        full_tokens.scatter_(1, _indices, tokens)

        return full_tokens

    def _select_by_uncertainty(self, tokens, uncertainty, ratio):
        indices = torch.argsort(uncertainty, dim=1, descending=True)
        num_remain = int(indices.size(1) * ratio)
        prune_indices = indices[:, num_remain:]
        indices = indices[:, :num_remain]
        tokens = torch.gather(tokens, 1, indices.expand(-1, -1, tokens.size(-1)))

        return tokens, indices, prune_indices

    def decode_queries(self, planes: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        queries.clamp_(-1, 0.999)
        query_features = torch.zeros(queries.size(0), queries.size(1), self.query_dim,
                                     device=queries.device, dtype=queries.dtype)
        query_features = _decode_queries_with_plane(query_features, planes, queries, mode='sum')
        return query_features

    def decode_uncertainty(self, planes, queries):
        queries.clamp_(-1, 0.999)
        uncertainty = torch.ones(queries.size(0), queries.size(1), 1, device=queries.device, dtype=queries.dtype)
        uncertainty = _decode_queries_with_plane(uncertainty, planes, queries, mode='mult')
        return uncertainty

    def apply_uncertainty_activation(self, x: torch.Tensor):
        return F.sigmoid(x)


def _decode_queries_with_plane(query_features: torch.Tensor, planes: torch.Tensor,
                               queries: torch.Tensor, mode='sum') -> torch.Tensor:
    for i in range(3):
        plane = planes[:, i]
        _queries = torch.stack([queries[..., j] for j in range(3) if j != i], dim=-1)
        features = F.grid_sample(plane, _queries.unsqueeze(2)).squeeze(-1).transpose(1, 2)
        if mode == 'sum':
            query_features = query_features + features
        elif mode == 'mult':
            query_features = query_features * features
        else:
            raise NotImplementedError

    return query_features


class _MergingModule(nn.Module):

    def __init__(self, embed_dim: int, num_merged: int, num_heads: int = 8, ):
        super().__init__()

        self.embed_dim = embed_dim

        self.cross_attn = CrossAttn(d_model=embed_dim, n_head=num_heads, dropout=0)
        self.tokens = nn.Parameter(torch.empty(num_merged, embed_dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        scale = self.embed_dim ** -0.5
        init_embedding(self.tokens, scale)

    def forward(self, x):
        tokens = self.tokens.unsqueeze(0).expand(x.size(0), -1, -1)
        tokens = tokens + self.cross_attn(tokens, x)
        return tokens
