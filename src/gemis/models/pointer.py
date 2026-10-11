"""Pointer head, paper section 3.2 (Eq. 5-6).

    H~ = alpha * MLP(H_e) + (1 - alpha) * E_x
    logits_t = [ W h_t ; H~ h_t ]        ->  L label classes, then N pointers

``H_e`` are the encoder states, ``E_x`` the input token embeddings, ``h_t`` the
decoder state and ``W`` the embeddings of the label tokens. Padded encoder
positions are masked out.

Difference from the paper: ``alpha`` is a hyperparameter there (value not
reported); here it is ``sigmoid`` of a learned scalar initialised at 0.5.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointerHead(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(
        self,
        decoder_hidden: torch.Tensor,  # (B, T, D)
        label_embeddings: torch.Tensor,  # (L, D)
        encoder_hidden: torch.Tensor | None = None,  # (B, N, D)
        encoder_embeddings: torch.Tensor | None = None,  # (B, N, D)
        attention_mask: torch.Tensor | None = None,  # (B, N), 1 = real token
    ) -> torch.Tensor:
        """Logits over ``L`` labels, followed by ``N`` pointers if encoder states are given."""
        label_scores = decoder_hidden @ label_embeddings.T  # (B, T, L)
        if encoder_hidden is None:
            return label_scores

        alpha = self.alpha
        fused = alpha * self.mlp(encoder_hidden) + (1 - alpha) * encoder_embeddings
        pointer_scores = decoder_hidden @ fused.transpose(1, 2)  # (B, T, N)
        if attention_mask is not None:
            pointer_scores = pointer_scores.masked_fill(
                attention_mask[:, None, :] == 0, torch.finfo(pointer_scores.dtype).min
            )
        return torch.cat([label_scores, pointer_scores], dim=-1)
