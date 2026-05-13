"""GEMISDecoderLayer: BartDecoderLayer subclass that threads SAM into AoA.

Designed for transformers ≥ 4.40 where:
  - BartAttention.forward() returns (attn_output, attn_weights) — 2 values only.
  - Cache is a Cache object updated in-place; no kv-pair in the layer return.
  - Attention weights from eager_attention_forward are (B, H, T, T) — no reshape.
  - forward() accepts past_key_values (Cache) and cache_position.

Changes from the stock BartDecoderLayer:
  1. Self-attention is always called with output_attentions=True to capture SAM.
  2. SAM (B, H, T_dec, T_dec) is passed to AoACrossAttention as self_attn_weights.
  3. AoACrossAttention (encoder_attn) is called with the legacy interface and
     always recomputes K/V — it never updates the cache.  This is correct for
     training and for non-cached greedy generation (our generate() method passes
     the full decoded sequence at each step).

This layer is applied via __class__ reassignment in GEMISModel._inject_aoa()
so that no weights are re-initialized.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartDecoderLayer


class GEMISDecoderLayer(BartDecoderLayer):
    """BartDecoderLayer that passes decoder self-attention weights (SAM) to AoA."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_values=None,                              # Cache | None
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple:

        residual = hidden_states

        # ── Self-attention (transformers ≥4.40 returns 2 values) ─────────────
        # Always use output_attentions=True so we get the SAM for AoA.
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=True,          # must be True for AoA SAM
            cache_position=cache_position,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # SAM is already (B, H, T_dec, T_dec) in transformers ≥4.40
        # (eager_attention_forward returns heads-separated shape)
        self_attn_weights_4d: Optional[torch.Tensor] = self_attn_weights

        # ── AoA cross-attention ───────────────────────────────────────────────
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # Call AoACrossAttention with its own (legacy) interface.
            # We intentionally do NOT pass past_key_values / cache_position here —
            # AoACrossAttention always recomputes K, V from encoder hidden states.
            # This is correct for training and non-cached decoding.
            hidden_states, cross_attn_weights, _ = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=None,               # always recompute
                output_attentions=output_attentions,
                self_attn_weights=self_attn_weights_4d,   # ← AoA injection
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # ── Feed-forward ──────────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        # ── Return (no cache — Cache updated in-place by BartAttention) ───────
        outputs: tuple = (hidden_states,)
        if output_attentions:
            # Only expose self_attn_weights to caller when explicitly requested
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs
