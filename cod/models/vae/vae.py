from typing import Optional

import torch
from torch import nn

from .autoencoder import CompactLatentAutoencoder
from .base import BaseAutoencoder
from .modules.blocks import StandardTransformerBlock
from .modules.kl import DiagonalGaussianDistribution
from .modules.transformer import init_embedding


class CompactLatentVAE(BaseAutoencoder):
    """Stage-2 prefix VAE supporting prefixes up to 16 positions of width 32."""

    def __init__(
        self,
        num_latent_layers: int,
        latent_dim: int,
        mlp_ratio: float = 4.0,
        num_heads: int = 8,
        dropout: float = 0.1,
        active_num_latents: int = 16,
        latent_causal: bool = True,
        use_latent_pos_embedding: bool = True,
        embed_dim: int = -1,
        **autoencoder_kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.active_num_latents = active_num_latents
        self.latent_causal = latent_causal

        self.autoencoder = CompactLatentAutoencoder(
            embed_dim=embed_dim, **autoencoder_kwargs
        )
        self.latent_proj_in = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, latent_dim * 2),
        )
        self.latent_proj_out = nn.Sequential(
            nn.Linear(latent_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.stage2_latent_pos = (
            nn.Parameter(torch.empty(active_num_latents, embed_dim))
            if use_latent_pos_embedding else None
        )
        self.latent_decoder = StandardTransformerBlock(
            embed_dim=embed_dim,
            num_layers=num_latent_layers,
            out_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            # Zero register tokens matches the architecture reported in the
            # supplementary material.  This is behaviorally identical to the
            # earlier non-positive sentinel used here.
            num_register=0,
            causal=latent_causal,
        )
        if self.stage2_latent_pos is not None:
            init_embedding(self.stage2_latent_pos, embed_dim ** -0.5)

    def load_autoencoder_weights(self, state_dict):
        self.autoencoder.load_state_dict(state_dict, strict=True)
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()

    def train(self, mode=True):
        super().train(mode)
        # The tokenizer is a fixed teacher throughout stage 2.
        self.autoencoder.eval()
        return self

    def encode_embed(self, pc):
        with torch.no_grad():
            return self.autoencoder.encode_embed(pc)

    def encode_latents(
        self,
        z,
        active_num_latents: Optional[int] = None,
        sample_posterior: bool = True,
        **kwargs,
    ):
        active = self.active_num_latents if active_num_latents is None else int(active_num_latents)
        if not 1 <= active <= min(self.active_num_latents, z.size(1)):
            raise ValueError("invalid active_num_latents for stage-2 VAE")
        parameters = self.latent_proj_in(z[:, :active].float())
        posterior = DiagonalGaussianDistribution(parameters)
        latent = posterior.sample() if sample_posterior else posterior.mode()
        return latent, posterior

    def decode_latents(self, z, mask=None, **kwargs):
        features = self.latent_proj_out(z)
        if self.stage2_latent_pos is not None:
            features = features + self.stage2_latent_pos[:features.size(1)].unsqueeze(0)
        features = self.latent_decoder(features)
        prefix_mask = torch.zeros(features.shape[:2], dtype=torch.bool, device=features.device)
        return features, prefix_mask

    def decode_embed(self, z, mask=None, **kwargs):
        return self.autoencoder.decode_embed(z, mask=mask, **kwargs)

    def decode_queries(self, context, queries):
        return self.autoencoder.decode_queries(context, queries)

    def forward(
        self,
        pc,
        queries,
        active_num_latents: Optional[int] = None,
        num_decode_loops: Optional[int] = None,
        sample_posterior: bool = True,
    ):
        latent, posterior = self.encode(
            pc,
            active_num_latents=active_num_latents,
            sample_posterior=sample_posterior,
        )
        features, prefix_mask = self.decode_latents(latent)
        planes = self.decode_embed(
            features,
            mask=prefix_mask,
            num_decode_loops=num_decode_loops,
        )[0]
        return {
            "logits": self.decode_queries(planes, queries),
            "planes": planes,
            "latents": latent,
            "posterior": posterior,
        }
