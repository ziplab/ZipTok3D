"""Class-conditional causal EDM over compact latent prefixes.

The architecture follows the latent-array Transformer used by 3DShape2VecSet:
each block combines token self-attention, category cross-attention, and
time-conditioned adaptive layer normalization. Token self-attention is causal,
so any physical prefix up to the maximum training length can be denoised
without suffix padding.
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _zero_module(module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


class PositionalEmbedding(nn.Module):
    def __init__(self, channels: int, max_positions: int = 10000):
        super().__init__()
        self.channels = channels
        self.max_positions = max_positions

    def forward(self, x):
        half = self.channels // 2
        frequencies = torch.arange(half, device=x.device, dtype=torch.float32)
        frequencies = frequencies / max(half, 1)
        frequencies = (1.0 / self.max_positions) ** frequencies
        angles = x.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
        if embedding.size(-1) < self.channels:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class Attention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        head_dim=64,
        dropout=0.0,
        causal=False,
    ):
        super().__init__()
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = heads * head_dim
        self.scale = head_dim ** -0.5
        self.heads = heads
        self.causal = bool(causal)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None):
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        batch_size = q.size(0)
        q, k, v = (
            tensor.view(batch_size, tensor.size(1), self.heads, -1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.heads, tensor.size(1), -1)
            for tensor in (q, k, v)
        )
        similarity = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
        if self.causal:
            if similarity.size(-2) != similarity.size(-1):
                raise ValueError("causal attention requires equal query and key lengths")
            causal_mask = torch.ones(
                similarity.shape[-2:], dtype=torch.bool, device=similarity.device
            ).triu(1)
            similarity = similarity.masked_fill(causal_mask, float("-inf"))
        weights = similarity.softmax(dim=-1)
        output = torch.einsum("b i j, b j d -> b i d", weights, v)
        output = output.view(batch_size, self.heads, output.size(1), -1)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, output.size(2), -1)
        return self.to_out(output)


class GEGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        values, gates = self.proj(x).chunk(2, dim=-1)
        return values * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * ratio)
        self.net = nn.Sequential(
            GEGLU(dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x, condition):
        scale, shift = self.proj(condition).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


class EDMTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        head_dim,
        mlp_ratio=4.0,
        dropout=0.0,
        causal=True,
    ):
        super().__init__()
        self.self_attn = Attention(
            dim,
            heads=heads,
            head_dim=head_dim,
            dropout=dropout,
            causal=causal,
        )
        self.cross_attn = Attention(
            dim, context_dim=dim, heads=heads, head_dim=head_dim, dropout=dropout
        )
        self.ff = FeedForward(dim, ratio=mlp_ratio, dropout=dropout)
        self.norm1 = AdaLayerNorm(dim)
        self.norm2 = AdaLayerNorm(dim)
        self.norm3 = AdaLayerNorm(dim)

    def forward(self, x, time_condition, class_condition):
        x = x + self.self_attn(self.norm1(x, time_condition))
        x = x + self.cross_attn(self.norm2(x, time_condition), class_condition)
        x = x + self.ff(self.norm3(x, time_condition))
        return x


class LatentArrayTransformer(nn.Module):
    def __init__(
        self,
        channels,
        num_latents,
        width,
        heads,
        head_dim,
        depth,
        time_channels,
        mlp_ratio,
        dropout,
        causal,
        use_token_pos_embedding,
        gradient_checkpointing,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.token_pos = (
            nn.Parameter(torch.empty(num_latents, width))
            if use_token_pos_embedding else None
        )
        self.proj_in = nn.Linear(channels, width, bias=False)
        self.blocks = nn.ModuleList([
            EDMTransformerBlock(
                width, heads, head_dim, mlp_ratio, dropout, causal=causal
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width)
        self.proj_out = _zero_module(nn.Linear(width, channels, bias=False))
        self.map_noise = PositionalEmbedding(time_channels)
        self.map_layer0 = nn.Linear(time_channels, width)
        self.map_layer1 = nn.Linear(width, width)
        if self.token_pos is not None:
            nn.init.normal_(self.token_pos, std=width ** -0.5)

    def forward(self, x, noise, condition):
        time = self.map_noise(noise).unsqueeze(1)
        time = F.silu(self.map_layer0(time))
        time = F.silu(self.map_layer1(time))
        x = self.proj_in(x)
        if self.token_pos is not None:
            x = x + self.token_pos[:x.size(1)].unsqueeze(0)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, time, condition, use_reentrant=False)
            else:
                x = block(x, time, condition)
        return self.proj_out(self.norm(x))


class SABERStage2EDM(nn.Module):
    """Class-conditional EDM with prefix-consistent causal self-attention."""

    def __init__(
        self,
        num_latents: int = 16,
        channels: int = 32,
        num_classes: int = 55,
        width: int = 384,
        num_heads: int = 8,
        head_dim: int = 48,
        depth: int = 16,
        time_channels: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        sigma_min: float = 0.0,
        sigma_max: float = float("inf"),
        sigma_data: float = 1.0,
        causal: bool = True,
        use_token_pos_embedding: bool = True,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        if num_heads * head_dim != width:
            raise ValueError("width must equal num_heads * head_dim")
        self.num_latents = num_latents
        self.channels = channels
        self.num_classes = num_classes
        self.causal = bool(causal)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.category_emb = nn.Embedding(num_classes, width)
        self.model = LatentArrayTransformer(
            channels=channels,
            num_latents=num_latents,
            width=width,
            heads=num_heads,
            head_dim=head_dim,
            depth=depth,
            time_channels=time_channels,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            causal=self.causal,
            use_token_pos_embedding=use_token_pos_embedding,
            gradient_checkpointing=gradient_checkpointing,
        )

    def _condition(self, labels, batch_size, device):
        if labels is None:
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        if labels.dtype.is_floating_point:
            return labels if labels.ndim == 3 else labels.unsqueeze(1)
        return self.category_emb(labels.long()).unsqueeze(1)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **kwargs):
        del force_fp32, kwargs
        if not 1 <= x.size(1) <= self.num_latents:
            raise ValueError(
                f"latent length must be in [1, {self.num_latents}], got {x.size(1)}"
            )
        x = x.float()
        sigma = sigma.float().reshape(-1, 1, 1)
        condition = self._condition(class_labels, x.size(0), x.device)
        c_skip = self.sigma_data ** 2 / (sigma.square() + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma.square() + self.sigma_data ** 2).sqrt()
        c_in = 1.0 / (self.sigma_data ** 2 + sigma.square()).sqrt()
        c_noise = sigma.log() / 4.0
        predicted = self.model(c_in * x, c_noise.flatten(), condition)
        return c_skip * x + c_out * predicted.float()

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    @torch.no_grad()
    def sample(
        self,
        class_labels,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        generator: Optional[torch.Generator] = None,
        num_latents: Optional[int] = None,
    ):
        active_num_latents = (
            self.num_latents if num_latents is None else int(num_latents)
        )
        if not 1 <= active_num_latents <= self.num_latents:
            raise ValueError(
                f"num_latents must be in [1, {self.num_latents}], "
                f"got {active_num_latents}"
            )
        noise = torch.randn(
            class_labels.size(0), active_num_latents, self.channels,
            device=class_labels.device, generator=generator,
        )
        return edm_sampler(
            self, noise, class_labels, num_steps, sigma_min, sigma_max, rho
        )


@torch.no_grad()
def edm_sampler(
    net,
    latents,
    class_labels=None,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
):
    sigma_min = max(float(sigma_min), float(net.sigma_min))
    sigma_max = min(float(sigma_max), float(net.sigma_max))
    indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    steps = (
        sigma_max ** (1 / rho)
        + indices / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    steps = torch.cat([net.round_sigma(steps), torch.zeros_like(steps[:1])])

    x_next = latents.to(torch.float64) * steps[0]
    for index, (current, following) in enumerate(zip(steps[:-1], steps[1:])):
        x_hat = x_next
        denoised = net(x_hat, current, class_labels).to(torch.float64)
        derivative = (x_hat - denoised) / current
        x_next = x_hat + (following - current) * derivative
        if index < num_steps - 1:
            corrected = net(x_next, following, class_labels).to(torch.float64)
            next_derivative = (x_next - corrected) / following
            x_next = x_hat + (following - current) * (
                0.5 * derivative + 0.5 * next_derivative
            )
    return x_next.float()
