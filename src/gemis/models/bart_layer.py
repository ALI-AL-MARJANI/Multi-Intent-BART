"""GEMISDecoderLayer: a BART decoder layer whose cross-attention is AoA.

Self-attention is computed here from the layer's pretrained projections, so
the attention map (SAM) is always available and the layer does not depend on
the signature of ``BartAttention.forward``, which changes between transformers
releases. No key/value cache is used: ``GEMISModel`` re-runs the decoder on
the full prefix at every generation step (target sequences are short).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartDecoderLayer

from gemis.models.aoa import AoACrossAttention, additive_mask


class GEMISDecoderLayer(BartDecoderLayer):
    """Drop-in replacement obtained by re-classing a pretrained ``BartDecoderLayer``."""

    @classmethod
    def convert(cls, layer: BartDecoderLayer, attention_dropout: float) -> GEMISDecoderLayer:
        layer.encoder_attn = AoACrossAttention.from_pretrained_attention(layer.encoder_attn, attention_dropout)
        layer.__class__ = cls
        return layer  # type: ignore[return-value]

    def _self_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn = self.self_attn
        b, t, d = x.shape
        heads, head_dim = attn.num_heads, attn.embed_dim // attn.num_heads

        def split(y: torch.Tensor) -> torch.Tensor:
            return y.view(b, t, heads, head_dim).transpose(1, 2)

        q = split(attn.q_proj(x)) * head_dim**-0.5
        k = split(attn.k_proj(x))
        v = split(attn.v_proj(x))
        scores = torch.matmul(q, k.transpose(-1, -2))
        causal = torch.full((t, t), torch.finfo(scores.dtype).min, device=x.device).triu(1)
        weights = F.softmax(scores + causal, dim=-1)  # (B, H, T, T)
        dropped = F.dropout(weights, p=attn.dropout, training=self.training)
        out = torch.matmul(dropped, v).transpose(1, 2).reshape(b, t, d)
        return attn.out_proj(out), weights

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,  # ignored: the causal mask is built here
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = False,
        **_: object,  # cache / head-mask arguments of the various transformers versions
    ) -> tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states, sam = self._self_attention(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)

        cross_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states, cross_weights = self.encoder_attn(
                hidden_states,
                encoder_hidden_states,
                encoder_mask=additive_mask(encoder_attention_mask, hidden_states.dtype),
                self_attn_weights=sam,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = self.encoder_attn_layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.final_layer_norm(residual + hidden_states)

        if output_attentions:
            return hidden_states, sam, cross_weights
        return (hidden_states,)
