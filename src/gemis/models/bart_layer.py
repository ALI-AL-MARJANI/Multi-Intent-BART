"""GEMISDecoderLayer: BartDecoderLayer subclass that threads SAM into AoA.

Compatible with transformers 4.40 – 4.57+.

Two API variants exist depending on the transformers version installed:
  - "new" (≥4.44): BartDecoder calls layer with past_key_values (plural, Cache),
                   BartAttention.forward() accepts past_key_values + cache_position.
  - "old" (<4.44): BartDecoder calls layer with past_key_value (singular, tuple),
                   BartAttention.forward() accepts past_key_value (singular).

GEMISDecoderLayer detects which API is active on the first forward call and
caches the result, so the check runs only once per process.
"""

from __future__ import annotations

import inspect
from typing import Optional

import torch
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartDecoderLayer


class GEMISDecoderLayer(BartDecoderLayer):
    """BartDecoderLayer that passes decoder self-attention weights (SAM) to AoA."""

    # Cached self-attn API: None = not yet detected, 'new' or 'old'
    _self_attn_api: Optional[str] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_values=None,    # new API: Cache | None
        past_key_value=None,     # old API: tuple | None
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,                # absorb any future unknown kwargs
    ) -> tuple:

        # Normalise to a single variable regardless of which name was passed
        pkv = past_key_values if past_key_values is not None else past_key_value

        residual = hidden_states

        # ── Detect self_attn API once ────────────────────────────────────────
        if GEMISDecoderLayer._self_attn_api is None:
            params = inspect.signature(self.self_attn.forward).parameters
            GEMISDecoderLayer._self_attn_api = (
                "new" if "past_key_values" in params else "old"
            )

        # ── Self-attention — always output_attentions=True for SAM ──────────
        if GEMISDecoderLayer._self_attn_api == "new":
            hidden_states, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                past_key_values=pkv,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=True,
                cache_position=cache_position,
            )
        else:
            sa_result = self.self_attn(
                hidden_states=hidden_states,
                past_key_value=pkv,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=True,
            )
            hidden_states = sa_result[0]
            # old API: (hidden, attn_weights, past_kv) — weights at index 1
            self_attn_weights = sa_result[1] if len(sa_result) > 1 else None

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        self_attn_weights_4d: Optional[torch.Tensor] = self_attn_weights

        # ── AoA cross-attention ──────────────────────────────────────────────
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, cross_attn_weights, _ = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=None,
                output_attentions=output_attentions,
                self_attn_weights=self_attn_weights_4d,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # ── Feed-forward ─────────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs: tuple = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs
