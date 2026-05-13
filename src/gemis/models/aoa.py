"""Attention-over-Attention (AoA) cross-attention layer.

Replaces the standard cross-attention in each BART decoder layer.

Paper §3.3 formula:

    CAM_t^l = Softmax(Q_t^l · K_enc^T)
    SAM_t^l = (decoder self-attention weights at step t)     # passed in externally
    A_t^l   = Softmax(Q_t^l · K_enc^T  +  (SAM_t^l · CAM_t^l) / √d_k)
    V_t^l   = A_t^l · V_enc

SAM captures "which past decoder outputs matter for the current step"; dotting
SAM against CAM re-weights the cross-attention map toward encoder positions that
were important for the already-predicted intents.

The self-attention weights (SAM) are computed by BartAttention in the layer's
self-attention call and forwarded here by GEMISDecoderLayer — no extra projections
are needed and no weights are duplicated.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AoACrossAttention(nn.Module):
    """Multi-head AoA cross-attention.

    Args:
        embed_dim:  total embedding dimension (must equal BART hidden size).
        num_heads:  number of attention heads.
        dropout:    attention dropout probability.
        bias:       whether to include bias in Q/K/V/out projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.dropout = dropout

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(p=dropout)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, H, T, d_k)"""
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, d_k) → (B, T, D)"""
        B, H, T, dk = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * dk)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,          # (B, T_dec, D)  decoder states
        key_value_states: torch.Tensor,       # (B, T_enc, D)  encoder outputs
        attention_mask: Optional[torch.Tensor] = None,   # (B, 1, T_dec, T_enc) additive
        past_key_value: Optional[tuple] = None,          # cached (K_enc, V_enc) 2-tuple
        output_attentions: bool = False,
        layer_head_mask: Optional[torch.Tensor] = None,
        # AoA-specific: self-attention weights from the same decoder layer,
        # shape (B, H, T_dec, T_dec), already softmax'd.
        self_attn_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple]]:
        B, T_dec, D = hidden_states.shape

        # ── Q from decoder; K, V from encoder (or cache) ─────────────────────
        Q = self._split_heads(self.q_proj(hidden_states))     # (B, H, T_dec, dk)

        if past_key_value is not None:
            # reuse cached encoder projections (cross-attn K/V never change)
            K, V = past_key_value
        else:
            K = self._split_heads(self.k_proj(key_value_states))  # (B, H, T_enc, dk)
            V = self._split_heads(self.v_proj(key_value_states))  # (B, H, T_enc, dk)

        present_key_value: tuple = (K, V)  # always cache for subsequent steps

        # ── standard cross-attention scores (CAM) ────────────────────────────
        # cam_scores: (B, H, T_dec, T_enc)
        cam_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale

        # ── AoA bias: SAM · softmax(CAM) ────────────────────────────────────
        # self_attn_weights: (B, H, T_dec, T_dec)  — SAM from self-attention
        # cam_soft:          (B, H, T_dec, T_enc)  — historical cross-attention
        # product:           (B, H, T_dec, T_dec) x (B, H, T_dec, T_enc)
        #                  → (B, H, T_dec, T_enc)  — AoA bias
        if self_attn_weights is not None:
            cam_soft = F.softmax(cam_scores, dim=-1)
            aoa_bias = torch.matmul(self_attn_weights, cam_soft) / self.scale
            cam_scores = cam_scores + aoa_bias

        # ── encoder padding mask (additive) ──────────────────────────────────
        if attention_mask is not None:
            cam_scores = cam_scores + attention_mask

        # ── softmax + dropout ─────────────────────────────────────────────────
        attn_weights = F.softmax(cam_scores, dim=-1)   # (B, H, T_dec, T_enc)

        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask.view(1, -1, 1, 1)

        attn_weights = self.attn_dropout(attn_weights)

        # ── context vector ────────────────────────────────────────────────────
        attn_output = torch.matmul(attn_weights, V)    # (B, H, T_dec, dk)
        attn_output = self._merge_heads(attn_output)   # (B, T_dec, D)
        attn_output = self.out_proj(attn_output)

        attn_weights_out = attn_weights if output_attentions else None
        return attn_output, attn_weights_out, present_key_value
