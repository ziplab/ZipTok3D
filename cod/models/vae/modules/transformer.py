from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .timm_layers import DropPath


def init_embedding(embedding, scale):
    nn.init.normal_(embedding)
    embedding.data *= scale


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class CrossAttn(nn.Module):
    _debug = False

    @classmethod
    def enable_debug_mode(cls):
        cls._debug = True

    def __init__(
            self,
            d_model,
            n_head,
            norm_layer=nn.LayerNorm,
            dropout: float = 0,
            out_dim: int = -1,
    ):
        super().__init__()

        self.ln_cross = norm_layer(d_model)
        self.ln_source = norm_layer(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.to_out = nn.Linear(d_model, out_dim) if out_dim > 0 else nn.Identity()
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self._attn = None

    def forward(self, x: torch.Tensor, source: torch.Tensor,
                key_padding_mask: torch.Tensor = None):
        x = self.ln_cross(x)
        source = self.ln_source(source)
        outputs = self.cross_attn(
            x, source, source,
            key_padding_mask=key_padding_mask,
            need_weights=CrossAttn._debug,
            average_attn_weights=False,
        )
        attn_out = outputs[0]

        if CrossAttn._debug:
            attn_weights = outputs[1]
            self._attn = attn_weights
        return self.drop_path(self.to_out(attn_out))


class FFN(nn.Module):

    def __init__(self, d_model,
                 mlp_ratio=4.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 dropout: float = 0,
                 ):
        super().__init__()
        activation = act_layer()
        mlp_width = int(d_model * mlp_ratio)
        input_width = mlp_width
        if isinstance(activation, GEGLU):
            input_width *= 2

        self.ln = norm_layer(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, input_width),
            activation,
            nn.Linear(mlp_width, d_model),
        )
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.mlp(self.ln(x)))


class ResidualAttentionBlock(nn.Module):
    """
    Code borrowed from "https://github.com/bytedance/1d-tokenizer"
    """

    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio=4.0,
            act_layer=GEGLU,
            norm_layer=nn.LayerNorm,
            dropout: float = 0,
            use_gate: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.linear_gate = nn.Linear(d_model, d_model) if use_gate else None
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            activation = act_layer()
            mlp_width = int(d_model * mlp_ratio)
            input_width = mlp_width
            if isinstance(activation, GEGLU):
                input_width *= 2

            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, input_width)),
                ("gelu", activation),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor = None,
                  key_padding_mask: torch.Tensor = None):
        return self.attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None):
        attn_output = self.attention(
            x=self.ln_1(x),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        attn_output = self.drop_path(attn_output)
        if self.linear_gate is not None:
            gate = self.linear_gate(x)
            attn_output = torch.sigmoid(gate) * attn_output

        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class ResidualCrossAttnBlock(nn.Module):

    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio=4.0,
            act_layer=GEGLU,
            norm_layer=nn.LayerNorm,
            dropout: float = 0,
            use_self_attn: bool = True,
    ):
        super().__init__()

        self.ln_cross = norm_layer(d_model)
        self.ln_source = norm_layer(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

        if use_self_attn:
            self.ln_1 = norm_layer(d_model)
            self.self_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        else:
            self.ln_1 = None
            self.self_attn = None

        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            activation = act_layer()
            mlp_width = int(d_model * mlp_ratio)
            input_width = mlp_width
            if isinstance(activation, GEGLU):
                input_width *= 2

            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, input_width)),
                ("gelu", activation),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()

    def attention(self, x: torch.Tensor, attn_layer, source: torch.Tensor = None,
                  attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None):
        if source is None:
            source = x

        out = attn_layer(x, source, source,
                         attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                         need_weights=False, average_attn_weights=False)
        return out[0]

    def forward(self, x: torch.Tensor, source: torch.Tensor,
                attn_mask: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None,
                ):
        cross_attn_output = self.attention(self.ln_cross(x), self.cross_attn, self.ln_source(source),
                                           attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.drop_path(cross_attn_output)

        if self.self_attn is not None:
            attn_output = self.attention(self.ln_1(x), self.self_attn)
            x = x + self.drop_path(attn_output)

        if self.mlp_ratio > 0:
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x
