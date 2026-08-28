from typing import Tuple, List, Callable

import torch
from torch import nn

from pointops.functions import pointops
from ..modules.blocks import ProgressiveEncoderBlock
from ..modules.transformer import ResidualCrossAttnBlock


class CompactPointPatchEncoder(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_patches: int,
                 num_blocks: int,
                 num_layers_per_block: int,
                 whether_casual: bool = False,

                 ## transformer params
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 ):
        super().__init__()

        # Upstream COD-VAE names this option ``whether_casual``.
        self.whether_casual = whether_casual

        self.num_patches = num_patches

        self.blocks = nn.ModuleList()
        self.norm_point = nn.LayerNorm(embed_dim)
        for _ in range(num_blocks):
            self.blocks.append(ProgressiveEncoderBlock(embed_dim=embed_dim, num_layers=num_layers_per_block,
                                                       num_heads=num_heads,
                                                       mlp_ratio=mlp_ratio, dropout=dropout))
        self.last_block = ResidualCrossAttnBlock(embed_dim, num_heads,
                                                 mlp_ratio=mlp_ratio,
                                                 dropout=dropout)

    def forward(self, pc: torch.Tensor, z: torch.Tensor, embed_fn: Callable[..., torch.Tensor]) -> torch.Tensor:
        point_features = embed_fn(pc)
        point_features = self.norm_point(point_features)
        patch_pos = pointops.fps(pc, self.num_patches)
        patches = embed_fn(patch_pos)
        patches = self.norm_point(patches)

        for i, encoder in enumerate(self.blocks):
            point_features, patches, z = encoder(point_features, patches, z)

        z = self.last_block(z, point_features)
        return z
