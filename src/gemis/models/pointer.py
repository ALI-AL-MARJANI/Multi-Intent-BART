"""Pointer Network head for GEMIS.

At each decoder step t, the pointer network produces a distribution over
N + L tokens, where:
  - The first N positions correspond to encoder input tokens (pointer outputs).
  - The last L positions correspond to the extended vocabulary (intent/slot labels).

Architecture (paper §3.2):

    Ĥ_e = α · MLP(H_e) + (1 - α) · E_x
    logits = Softmax( [Ĥ_e ⊗ h_d_t ; W ⊗ h_d_t] )

where α is a learnable scalar, H_e are encoder hidden states, E_x are
input token embeddings, h_d_t is the decoder hidden state at step t, and
W is the label embedding matrix (rows = vocab embeddings for intent/slot labels).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointerNetwork(nn.Module):
    """Joint pointer + label classification head.

    Args:
        hidden_size:  dimensionality of BART hidden states / embeddings.
        vocab_size:   full vocabulary size (including added intent/slot tokens).
        mlp_hidden:   hidden size of the MLP projecting encoder states.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        mlp_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # α: learnable scalar in [0,1] controlling encoder-state vs. embedding fusion
        self.alpha_logit = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5

        # MLP to project encoder hidden states
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # Label scoring: projects decoder hidden state against vocab embeddings.
        # We do NOT create a separate weight matrix here — we reuse the model's
        # input embedding matrix (passed at forward time) for parameter sharing.
        # A small projection aligns decoder hidden dim with embedding dim.
        self.decoder_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        encoder_hidden: torch.Tensor,    # (B, N, D)  — encoder outputs
        encoder_embeddings: torch.Tensor, # (B, N, D)  — input token embeddings
        decoder_hidden: torch.Tensor,    # (B, T, D)  — decoder hidden states
        label_embeddings: torch.Tensor,  # (L, D)     — vocab embedding matrix
    ) -> torch.Tensor:
        """Return logits of shape (B, T, N+L) over pointer positions + label vocab."""
        alpha = torch.sigmoid(self.alpha_logit)  # scalar in (0,1)

        # Fused encoder representation: (B, N, D)
        h_enc_fused = alpha * self.mlp(encoder_hidden) + (1 - alpha) * encoder_embeddings

        # Project decoder hidden: (B, T, D)
        h_dec = self.decoder_proj(decoder_hidden)

        # --- pointer scores over N input positions ---
        # (B, T, D) × (B, N, D)^T → (B, T, N)
        pointer_scores = torch.bmm(h_dec, h_enc_fused.transpose(1, 2))

        # --- label scores over L label tokens ---
        # label_embeddings: (L, D) → (1, D, L) for broadcasting
        label_emb_t = label_embeddings.T.unsqueeze(0)  # (1, D, L)
        # (B, T, D) × (1, D, L) → (B, T, L)
        label_scores = torch.matmul(h_dec, label_emb_t)

        # concatenate: (B, T, N+L)
        logits = torch.cat([pointer_scores, label_scores], dim=-1)
        return logits

    # ── greedy decode helper ──────────────────────────────────────────────────

    @staticmethod
    def decode_step(
        logits: torch.Tensor,    # (B, N+L)
        n_input_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split logits into pointer and label parts; return argmax of each.

        Returns:
            predicted_ids: (B,) argmax over full N+L space.
            is_pointer:    (B,) boolean mask — True if predicted index < N.
        """
        predicted_ids = logits.argmax(dim=-1)
        is_pointer = predicted_ids < n_input_tokens
        return predicted_ids, is_pointer
