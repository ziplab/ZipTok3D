"""COD-VAE autoencoder with physical exp prefixes and optional recurrent intermediate logits.

Default single-pass decoding, learned/FPS initialization and parameter names
are preserved. Requested intermediate logits share the same encode, sampled
prefixes and recurrent trajectory as final logits; neither branch is detached.
"""

import functools
from typing import List, Tuple

import torch
from torch import nn

from pointops.functions import pointops
from cod.models.vae.base import BaseAutoencoder
from cod.models.vae.modules.pos import PointEmbed
from cod.models.vae.modules.transformer import init_embedding
from cod.models.vae.networks.encoder import CompactPointPatchEncoder
from .architecture_prefix import normalize_strategy, prefix_budgets, resolve_prefix_lengths
from .decoder import CompactTriplaneDecoder


class CompactLatentAutoencoder(BaseAutoencoder):

    def __init__(self,
                 output_dim: int,
                 num_latents: int,
                 embed_dim: int,
                 query_dim: int,
                 encoder_params: dict,
                 decoder_params: dict,
                 use_learnable_pos: bool = False,
                 nested_dropout_strategy: str = "exp",
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_latents = num_latents
        self.query_dim = query_dim

        self.use_learnable_pos = use_learnable_pos
        self.nested_dropout_strategy = normalize_strategy(nested_dropout_strategy)
        prefix_budgets(num_latents)  # Validate without adding checkpoint state.

        ## layers
        self.point_embed = PointEmbed(dim=embed_dim)
        self.norm_latent = nn.LayerNorm(embed_dim)
        encoder_params = dict(encoder_params)
        if encoder_params.pop("causal_self_attention", False):
            raise ValueError("the released training-compatible encoder is bidirectional")
        self.encoder = CompactPointPatchEncoder(embed_dim=embed_dim, **encoder_params)
        self.decoder = CompactTriplaneDecoder(embed_dim=embed_dim, query_dim=query_dim, **decoder_params)
        self.head = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, output_dim),
        )

        self.latent_pos = nn.Parameter(torch.empty(num_latents, embed_dim)) if use_learnable_pos else None
        self.reset_parameters()

    def reset_parameters(self):
        scale = self.embed_dim ** -0.5
        if self.latent_pos is not None:
            init_embedding(self.latent_pos, scale)

    def forward(self, pc, queries, active_num_latents=None, intermediate_step=None):
        self.decoder._validate_intermediate_step(intermediate_step)
        z, posterior = self.encode(pc)
        lengths = self._prefix_lengths(z, active_num_latents)
        decoded = self.decode(z, active_num_latents=lengths, intermediate_step=intermediate_step)
        patches, init_patches, uncertainty, z_recon = decoded[:4]
        out = self.decode_queries(patches, queries)

        outputs = {
            'logits': out,
            'prefix_lengths': lengths,
        }
        if intermediate_step is not None:
            outputs['intermediate_logits'] = self.decode_queries(decoded[4], queries)
        if (init_patches is not None) and self.training:
            init_out = self.decode_queries(init_patches, queries)
            outputs['init_out'] = init_out
        if uncertainty is not None:
            uncertainty_queries = self.decoder.decode_uncertainty(uncertainty, queries)
            outputs['uncertainty'] = uncertainty_queries
            outputs['uncertainty_planes'] = uncertainty

        return outputs

    def encode_embed(self, pc):
        z = self._get_init_z(pc)
        z = self.norm_latent(z)
        return self.encoder(pc, z, self.point_embed)

    def encode_latents(self, z):
        ### autoencoder model without channel compression
        return z, None

    def decode_latents(self, z):
        ### autoencoder model without channel compression
        return z

    def _prefix_lengths(self, z, active_num_latents=None):
        return resolve_prefix_lengths(
            z, self.nested_dropout_strategy, self.training, active_num_latents
        )

    @property
    def prefix_budgets(self):
        return prefix_budgets(self.num_latents)

    def decode(self, z, active_num_latents=None, intermediate_step=None):
        """Keep four original outputs; requesting a pass appends intermediate planes fifth."""
        return self.decode_embed(self.decode_latents(z), active_num_latents=active_num_latents,
                                 intermediate_step=intermediate_step)

    def decode_embed(self, z, active_num_latents=None, intermediate_step=None):
        self.decoder._validate_intermediate_step(intermediate_step)
        lengths = self._prefix_lengths(z, active_num_latents)
        budgets = torch.unique(lengths, sorted=True)
        decoder_options = {} if intermediate_step is None else {'intermediate_step': intermediate_step}
        if budgets.numel() == 1:
            # This path is exactly the COD-VAE decode call at the full budget.
            decoded = self.decoder.decode(z[:, :int(budgets.item())], **decoder_options)
        else:
            decoded_groups, batch_indices = [], []
            for budget in budgets.tolist():
                indices = torch.nonzero(lengths == budget, as_tuple=False).flatten()
                # Advanced indexing avoids a broken IndexSelectBackward kernel
                # in some CPU PyTorch builds; the selected values are identical.
                decoder_input = z[indices, :budget]
                decoded_groups.append(self.decoder.decode(decoder_input, **decoder_options))
                batch_indices.append(indices)
            restore_order = torch.argsort(torch.cat(batch_indices))
            decoded = tuple(
                torch.cat([group[index] for group in decoded_groups], dim=0)[restore_order]
                for index in range(3 if intermediate_step is None else 4)
            )
        context, init_context, uncertainty = decoded[:3]
        # Preserve the original uncompressed-latent passthrough in output four.
        if intermediate_step is not None:
            return context, init_context, uncertainty, z, decoded[3]
        return context, init_context, uncertainty, z

    def decode_queries(self, context, queries):
        query_features = self.decoder.decode_queries(context, queries)
        out = self.head(query_features).squeeze(-1)
        return out

    def _get_init_z(self, pc):
        if self.latent_pos is None:
            z = pointops.fps(pc, self.num_latents)
            z = self.point_embed(z)
        else:
            z = self.latent_pos.unsqueeze(0).expand(pc.size(0), -1, -1)
        return z
