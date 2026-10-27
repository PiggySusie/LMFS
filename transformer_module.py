from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(d_out, d_in, device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (d_in + d_out))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(vocab_size, d_model, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        rms = torch.sqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 / rms * self.weight).to(in_dtype)


def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        freqs = theta ** (
            -torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        )
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = einsum(positions, freqs, "seq, half -> seq half")
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq d_k"],
        token_positions: Int[Tensor, "... seq"],
    ) -> Float[Tensor, "... seq d_k"]:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        # Broadcast to match leading dims of x (e.g. heads).
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        out = torch.empty_like(x)
        out[..., 0::2] = rot_even
        out[..., 1::2] = rot_odd
        return out


def softmax(x: Float[Tensor, "..."], dim: int = -1) -> Float[Tensor, "..."]:
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = softmax(scores, dim=-1)
    return einsum(weights, V, "... q k, ... k d -> ... q d")


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope: RotaryPositionalEmbedding | None = None
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device
            )

    def forward(
        self,
        x: Float[Tensor, "... seq d_model"],
        token_positions: Int[Tensor, "... seq"] | None = None,
    ) -> Float[Tensor, "... seq d_model"]:
        *prefix, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        k = rearrange(k, "... seq (h d) -> ... h seq d", h=self.num_heads)
        v = rearrange(v, "... seq (h d) -> ... h seq d", h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
                for _ in prefix:
                    token_positions = token_positions.unsqueeze(0)
                token_positions = token_positions.expand(*prefix, seq_len)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        mask = ~causal
        attn = scaled_dot_product_attention(q, k, v, mask=mask)
        attn = rearrange(attn, "... h seq d -> ... seq (h d)")
        return self.output_proj(attn)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        token_positions: Int[Tensor, "batch seq"] | None = None,
    ) -> Float[Tensor, "batch seq d_model"]:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self, in_indices: Int[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch seq vocab"]:
        batch, seq_len = in_indices.shape
        x = self.token_embeddings(in_indices)
        positions = torch.arange(seq_len, device=in_indices.device)
        positions = positions.unsqueeze(0).expand(batch, seq_len)
        for layer in self.layers:
            x = layer(x, token_positions=positions)
        x = self.ln_final(x)
        return self.lm_head(x)
