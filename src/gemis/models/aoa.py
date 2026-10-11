"""Attention-over-attention (AoA) cross-attention, paper section 3.3.

For decoder step ``t`` in layer ``l`` (single head shown)::

    CAM = softmax(Q K_enc^T / sqrt(d_k))             cross-attention map, (T, N)
    SAM = softmax(Q K_dec^T / sqrt(d_k) + causal)    self-attention map,  (T, T)
    A   = softmax(Q K_enc^T / sqrt(d_k) + SAM @ CAM / sqrt(d_k))

``SAM @ CAM`` averages the cross-attention maps of the previous steps, weighted
by how much step ``t`` attends to each of them. The projection weights are the
pretrained BART cross-attention weights; AoA adds no parameters.

Difference from the paper's Eq. 10: the paper writes the first term without
the ``1/sqrt(d_k)`` scaling. It is kept here because the pretrained BART
weights assume scaled dot-product attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def additive_mask(mask: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    """Normalise an encoder padding mask to an additive (B, 1, 1|T, N) tensor.

    Accepts a (B, N) mask of 1/0 or booleans (1 = keep), or an already additive
    4-D float mask as built by ``BartDecoder``.
    """
    if mask is None:
        return None
    if mask.dim() == 2:
        mask = mask[:, None, None, :]
    if mask.dtype == torch.bool or not mask.is_floating_point():
        return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(
            ~mask.bool(), torch.finfo(dtype).min
        )
    return mask.to(dtype)


class AoACrossAttention(nn.Module):
    """Multi-head cross-attention with the AoA bias."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.dropout = dropout
        self.use_sam = True  # set to False to recover plain cross-attention

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    @classmethod
    def from_pretrained_attention(cls, attn: nn.Module, dropout: float) -> AoACrossAttention:
        """Build from a ``BartAttention`` module, reusing its projection layers."""
        aoa = cls(attn.embed_dim, attn.num_heads, dropout, bias=attn.q_proj.bias is not None)
        aoa.q_proj, aoa.k_proj, aoa.v_proj, aoa.out_proj = (
            attn.q_proj,
            attn.k_proj,
            attn.v_proj,
            attn.out_proj,
        )
        return aoa

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, T, D) decoder states
        encoder_hidden_states: torch.Tensor,  # (B, N, D)
        encoder_mask: torch.Tensor | None = None,  # additive, broadcastable to (B, H, T, N)
        self_attn_weights: torch.Tensor | None = None,  # (B, H, T, T), rows sum to 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = hidden_states.shape
        q = self._split(self.q_proj(hidden_states))
        k = self._split(self.k_proj(encoder_hidden_states))
        v = self._split(self.v_proj(encoder_hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale  # (B, H, T, N)
        if encoder_mask is not None:
            scores = scores + encoder_mask

        if self.use_sam and self_attn_weights is not None:
            cam = F.softmax(scores, dim=-1)  # padding already masked out
            scores = scores + torch.matmul(self_attn_weights, cam) / self.scale

        weights = F.softmax(scores, dim=-1)
        dropped = F.dropout(weights, p=self.dropout, training=self.training)
        out = torch.matmul(dropped, v).transpose(1, 2).reshape(b, t, self.embed_dim)
        return self.out_proj(out), weights
