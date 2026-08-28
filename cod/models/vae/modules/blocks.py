import functools

import torch
from torch import nn

from .transformer import ResidualAttentionBlock, ResidualCrossAttnBlock, GEGLU, CrossAttn, FFN, init_embedding


class ProgressiveEncoderBlock(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_layers: int,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 rescale_cross_attn: bool = False,
                 attn_proj: bool = False,
                 update_patch: bool = True,
                 ):
        super().__init__()

        self.rescale_cross_attn = rescale_cross_attn
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        attn_out_dim = embed_dim if attn_proj else -1
        self.processing_layers = nn.ModuleList()
        self.points2patch = CrossAttn(embed_dim, num_heads, dropout=0, out_dim=attn_out_dim) if update_patch else None
        self.ln_ffn = nn.LayerNorm(embed_dim)
        self.patch_ffn = FFN(embed_dim, act_layer=GEGLU, mlp_ratio=mlp_ratio, dropout=dropout)

        for i in range(num_layers):
            self.processing_layers.append(ResidualAttentionBlock(embed_dim, num_heads,
                                                                 act_layer=GEGLU,
                                                                 mlp_ratio=mlp_ratio,
                                                                 dropout=dropout))
        self.patch2latents = ResidualCrossAttnBlock(embed_dim, num_heads,
                                                    act_layer=GEGLU,
                                                    mlp_ratio=mlp_ratio,
                                                    dropout=dropout)
        self.latents2points = CrossAttn(embed_dim, num_heads, dropout=0)

    def forward(self, point_features, patches, z):
        if self.points2patch is not None:
            patches = patches + self.points2patch(patches, point_features)
        patches = patches + self.patch_ffn(self.ln_ffn(patches))

        for layer in self.processing_layers:
            patches = layer(patches)

        z = self.patch2latents(z, patches)
        point_features = point_features + self.latents2points(point_features, z)

        return point_features, patches, z


class CrossTransformerBlock(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_layers: int,

                 use_ln: bool = True,
                 use_ffn: bool = True,

                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 num_register: int = 1,
                 ):
        super().__init__()
        mlp_ratio = mlp_ratio if use_ffn else 0

        self.embed_dim = embed_dim
        self.register = nn.Parameter(torch.empty(num_register, embed_dim)) if num_register > 0 else None

        ## layers
        self.ln_pre = nn.LayerNorm(embed_dim) if use_ln else None
        self.ln_source = nn.LayerNorm(embed_dim) if use_ln else None
        self.transformer = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer.append(ResidualCrossAttnBlock(embed_dim, num_heads,
                                                           act_layer=GEGLU,
                                                           mlp_ratio=mlp_ratio,
                                                           dropout=dropout))
        self.reset_parameters()

    def reset_parameters(self):
        scale = self.embed_dim ** -0.5
        if self.register is not None:
            init_embedding(self.register, scale)

    def forward(self, x, source, source_mask=None):
        L = x.size(1)
        if self.register is not None:
            register = self.register.unsqueeze(0).expand(x.size(0), -1, -1)
            x = torch.cat([x, register], dim=1)

        if self.ln_pre is not None:
            x = self.ln_pre(x)
        if self.ln_source is not None:
            source = self.ln_source(source)

        for i, layer in enumerate(self.transformer):
            x = layer(x, source, key_padding_mask=source_mask)

        x = x[:, :L]
        return x


class StandardTransformerBlock(nn.Module):

    def __init__(self, embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 input_dim: int = -1,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.,
                 num_register: int = 1,
                 use_ln: bool = False,
                 out_dim: int = -1,
                 causal: bool = False,
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.class_token = None
        self.class_pos = None
        self.num_register = num_register
        self.causal = causal
        self.ln_pre = nn.LayerNorm(embed_dim) if use_ln else None

        self.linear_out = None
        if out_dim > 0:
            self.linear_out = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, out_dim),
            )

        if num_register > 0:
            self.class_token = nn.Parameter(torch.empty(1, embed_dim))
            self.class_pos = nn.Parameter(torch.empty(num_register, embed_dim))

        ## layers
        self.input_proj = nn.Linear(input_dim, embed_dim) if input_dim > 0 else nn.Identity()
        self.transformer = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer.append(ResidualAttentionBlock(embed_dim, num_heads,
                                                           act_layer=GEGLU,
                                                           mlp_ratio=mlp_ratio,
                                                           dropout=dropout))

        self.reset_parameters()

    def reset_parameters(self):
        scale = self.embed_dim ** -0.5
        if self.class_token is not None:
            init_embedding(self.class_pos, scale)
            init_embedding(self.class_token, scale)

    def forward(self, patches, key_padding_mask=None):
        B, L, C = patches.size()
        x = self.input_proj(patches)
        if self.class_token is not None:
            cls_token = self.class_token + self.class_pos
            cls_token = cls_token.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([x, cls_token], dim=1)
            if key_padding_mask is not None:
                register_mask = torch.zeros(
                    B, self.num_register, dtype=torch.bool, device=x.device
                )
                key_padding_mask = torch.cat([key_padding_mask.bool(), register_mask], dim=1)
        if self.ln_pre is not None:
            x = self.ln_pre(x)

        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(
                torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device),
                diagonal=1,
            )

        for layer in self.transformer:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        if self.num_register > 0:
            x = x[:, :-self.num_register]

        if self.linear_out is not None:
            x = self.linear_out(x)
        return x
