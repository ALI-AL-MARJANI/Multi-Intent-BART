"""GEMIS: main model combining BART, AoA decoder, and Pointer Network.

Architecture overview:
    1. A BART encoder-decoder backbone (facebook/bart-large or bart-base).
    2. Each BART decoder layer's cross-attention is replaced by AoACrossAttention.
    3. A PointerNetwork head sits on top of the decoder for final token prediction.
    4. The vocabulary is extended with intent/slot tokens whose embeddings are
       initialized as the mean of their subword embeddings.

Forward pass:
    - Input: tokenized utterance (B, N).
    - Decoder input: teacher-forced target sequence (B, T).
    - Output: logits (B, T, N+L) over pointer positions + label vocab.
    - Loss: standard seq2seq cross-entropy.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartAttention

from gemis.data.dataset import encoder_pos_to_word_idx
from gemis.models.aoa import AoACrossAttention
from gemis.models.bart_layer import GEMISDecoderLayer
from gemis.models.pointer import PointerNetwork
from gemis.models.vocab_utils import setup_vocab


class GEMISModel(nn.Module):
    """Full GEMIS model.

    Args:
        backbone_name:  HuggingFace model id (e.g. "facebook/bart-large").
        intent_labels:  list of intent class strings (e.g. ["PlayMusic", ...]).
        slot_labels:    list of slot type strings (e.g. ["track", ...]).
        tokenizer:      extended BART tokenizer (passed in to avoid circular deps).
        mlp_hidden:     hidden size in the pointer network MLP.
    """

    def __init__(
        self,
        backbone_name: str,
        intent_labels: list[str],
        slot_labels: list[str],
        tokenizer,
        mlp_hidden: int = 512,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.intent_labels = intent_labels
        self.slot_labels = slot_labels

        # ── load BART backbone ────────────────────────────────────────────────
        self.bart: BartForConditionalGeneration = (
            BartForConditionalGeneration.from_pretrained(backbone_name)
        )
        config: BartConfig = self.bart.config

        # ── extend vocabulary and init embeddings ─────────────────────────────
        setup_vocab(self.bart, tokenizer, intent_labels, slot_labels)
        # also resize the lm_head to match new vocab size
        self.bart.lm_head = nn.Linear(
            config.d_model, len(tokenizer), bias=False
        )

        # ── replace cross-attention layers in decoder with AoA ────────────────
        self._inject_aoa(config)

        # ── pointer network ───────────────────────────────────────────────────
        self.pointer = PointerNetwork(
            hidden_size=config.d_model,
            vocab_size=len(tokenizer),
            mlp_hidden=mlp_hidden,
        )

        self._vocab_size = len(tokenizer)

    # ── AoA injection ────────────────────────────────────────────────────────

    def _inject_aoa(self, config: BartConfig) -> None:
        """Upgrade every decoder layer to GEMISDecoderLayer + AoACrossAttention.

        Two operations per layer:
          1. Swap encoder_attn (BartAttention) → AoACrossAttention, copying
             pretrained weights so we start from the pretrained cross-attention.
          2. Reassign the layer's __class__ to GEMISDecoderLayer so that
             forward() threads the self-attention weights (SAM) into AoA.
             No weights are re-initialized — only the Python dispatch changes.
        """
        decoder = self.bart.model.decoder
        for layer in decoder.layers:
            old: BartAttention = layer.encoder_attn

            # 1. Build AoACrossAttention and warm-start from pretrained weights
            aoa = AoACrossAttention(
                embed_dim=config.d_model,
                num_heads=config.decoder_attention_heads,
                dropout=config.attention_dropout,
                bias=True,
            )
            aoa.q_proj.weight.data.copy_(old.q_proj.weight.data)
            aoa.k_proj.weight.data.copy_(old.k_proj.weight.data)
            aoa.v_proj.weight.data.copy_(old.v_proj.weight.data)
            aoa.out_proj.weight.data.copy_(old.out_proj.weight.data)
            if old.q_proj.bias is not None:
                aoa.q_proj.bias.data.copy_(old.q_proj.bias.data)
                aoa.k_proj.bias.data.copy_(old.k_proj.bias.data)
                aoa.v_proj.bias.data.copy_(old.v_proj.bias.data)
                aoa.out_proj.bias.data.copy_(old.out_proj.bias.data)
            layer.encoder_attn = aoa

            # 2. Upgrade layer class — zero weight impact, changes forward() only
            layer.__class__ = GEMISDecoderLayer

            # 3. Force eager attention on self_attn so SAM weights are always
            #    returned (SDPA / Flash silently returns None for output_attentions=True).
            #    Try the direct instance attribute first (transformers ≥4.40), then
            #    the config-based path as a fallback.
            if hasattr(layer.self_attn, "_attn_implementation"):
                layer.self_attn._attn_implementation = "eager"
            elif (
                hasattr(layer.self_attn, "config")
                and layer.self_attn.config is not None
                and hasattr(layer.self_attn.config, "_attn_implementation")
            ):
                layer.self_attn.config._attn_implementation = "eager"

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,           # (B, N)
        attention_mask: torch.Tensor,      # (B, N)
        decoder_input_ids: torch.Tensor,   # (B, T)
        labels: Optional[torch.Tensor] = None,        # (B, T)
        pointer_targets: Optional[torch.Tensor] = None, # (B, T) — encoder positions or -1
    ) -> dict[str, torch.Tensor]:
        """Run forward pass and optionally compute loss.

        Returns a dict with:
            "logits"  : (B, T, N+L) — raw scores for pointer + label positions
            "loss"    : scalar cross-entropy loss (only when labels provided)
        """
        B, N = input_ids.shape

        # ── encode ────────────────────────────────────────────────────────────
        encoder_outputs = self.bart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        encoder_hidden = encoder_outputs.last_hidden_state  # (B, N, D)

        # ── input token embeddings (for pointer fusion) ───────────────────────
        encoder_embeddings = self.bart.model.shared(input_ids)  # (B, N, D)

        # ── decode (BART decoder with AoA) ────────────────────────────────────
        # use_cache=False: we never use the KV-cache (full sequence passed each
        # step in generate(), teacher forcing in forward). Disabling it also
        # avoids an IndexError on transformers versions that expect the layer to
        # return a present_key_value at output[1].
        decoder_outputs = self.bart.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=attention_mask,
            use_cache=False,
        )
        decoder_hidden = decoder_outputs.last_hidden_state  # (B, T, D)

        # ── pointer network forward ───────────────────────────────────────────
        # label_embeddings: use shared embedding matrix rows (vocab_size, D)
        label_emb = self.bart.model.shared.weight  # (vocab_size, D)

        logits = self.pointer(
            encoder_hidden=encoder_hidden,
            encoder_embeddings=encoder_embeddings,
            decoder_hidden=decoder_hidden,
            label_embeddings=label_emb,
        )  # (B, T, N+vocab_size)

        output: dict[str, torch.Tensor] = {"logits": logits}

        # ── loss ──────────────────────────────────────────────────────────────
        if labels is not None:
            output["loss"] = self._compute_loss(logits, labels, pointer_targets, N)

        return output

    def _compute_loss(
        self,
        logits: torch.Tensor,              # (B, T, N+L)
        labels: torch.Tensor,              # (B, T)  vocab ids
        pointer_targets: Optional[torch.Tensor],  # (B, T)  encoder positions or -1
        n_input_tokens: int,
    ) -> torch.Tensor:
        """Seq2seq cross-entropy loss (equation 11 in paper).

        For tokens where pointer_targets[b,t] >= 0, the gold target is the
        pointer position (index into the first N logit slots).  Otherwise,
        the gold target is labels[b,t] mapped to the second N+vocab_size region.
        """
        B, T, NL = logits.shape

        # build unified target indices into the [0, N+L) space
        # default: use vocabulary index shifted into the label region
        unified_targets = labels.clone()  # (B, T)  — vocab ids

        # remap vocab targets to the [N, N+L) range
        mask_vocab = labels != -100
        unified_targets[mask_vocab] = (
            labels[mask_vocab] + n_input_tokens
        ).clamp(max=NL - 1)

        # remap pointer targets to [0, N)
        if pointer_targets is not None:
            mask_ptr = pointer_targets >= 0
            unified_targets[mask_ptr] = pointer_targets[mask_ptr]

        # ignore padding
        ignore_mask = labels == -100
        unified_targets[ignore_mask] = -100

        loss = nn.functional.cross_entropy(
            logits.view(B * T, NL),
            unified_targets.view(B * T),
            ignore_index=-100,
            reduction="mean",
        )
        return loss

    # ── greedy generation ─────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,       # (B, N)
        attention_mask: torch.Tensor,  # (B, N)
        max_new_tokens: int = 64,
        input_words_batch: Optional[list[list[str]]] = None,
        return_scores: bool = False,
    ) -> "list[list[int]] | dict":
        """Greedy autoregressive generation.

        When the pointer network selects encoder position p (absolute, BOS=0):
          - The OUTPUT token is the word-index integer k encoded as a string
            token (e.g. tokenizer.encode(" 2") → Ġ2).
          - The NEXT DECODER INPUT token is input_ids[:, p] (the actual wordpiece).

        Args:
            input_ids:          tokenised source, (B, N) including BOS.
            attention_mask:     (B, N).
            max_new_tokens:     decoding budget.
            input_words_batch:  optional list of word lists for each batch item,
                                used for accurate word-boundary detection with
                                multi-subword words.  If None, we approximate
                                word index as enc_pos - 1 (valid for single-
                                subword words, true for >95% of ATIS/SNIPS).

        Returns:
            If return_scores=False (default): List[List[int]] — generated output
            token ids (one list per item), where pointer steps hold the word-index
            integer token id, not the wordpiece.

            If return_scores=True: dict with keys:
                "generated_ids" : List[List[int]] — same as above.
                "intent_entropies": List[List[float]] — per-step softmax entropy
                    at each intent-position decode step, one list per batch item.
                    Empty list if no intent token was generated.
        """
        B, N = input_ids.shape
        device = input_ids.device
        tok = self.tokenizer

        encoder_outputs = self.bart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        encoder_hidden = encoder_outputs.last_hidden_state
        encoder_embeddings = self.bart.model.shared(input_ids)
        label_emb = self.bart.model.shared.weight

        # Two separate sequences:
        #   output_ids    — what we record (word-index integers for pointer steps)
        #   decoder_input — what the decoder reads (wordpieces for pointer steps)
        bos_col = torch.full((B, 1), tok.bos_token_id, dtype=torch.long, device=device)
        output_ids = bos_col.clone()
        decoder_input = bos_col.clone()

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        # intent token id range: any token whose text starts with "<intent:"
        intent_token_ids: set[int] = {
            tid for tok_str, tid in tok.get_vocab().items()
            if tok_str.startswith("<intent:")
        }

        # per-batch accumulator of entropy values at intent-position steps
        intent_entropies: list[list[float]] = [[] for _ in range(B)]

        for _ in range(max_new_tokens):
            dec_out = self.bart.model.decoder(
                input_ids=decoder_input,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=attention_mask,
                use_cache=False,
            )
            dec_hidden = dec_out.last_hidden_state  # (B, t, D)

            logits_step = self.pointer(
                encoder_hidden=encoder_hidden,
                encoder_embeddings=encoder_embeddings,
                decoder_hidden=dec_hidden[:, -1:, :],
                label_embeddings=label_emb,
            )  # (B, 1, N+L)

            logits_1d = logits_step.squeeze(1)   # (B, N+L)
            next_ids, is_ptr = PointerNetwork.decode_step(
                logits_1d, n_input_tokens=N
            )  # next_ids: (B,), is_ptr: (B,) bool

            # ── capture entropy at intent-position steps ──────────────────────
            if return_scores:
                probs = torch.softmax(logits_1d, dim=-1)  # (B, N+L)
                # entropy H = -sum(p * log(p+eps)), clamp for numerical safety
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (B,)
                for b in range(B):
                    if not finished[b] and next_ids[b].item() in intent_token_ids:
                        intent_entropies[b].append(entropy[b].item())

            # tokens for the OUTPUT sequence
            out_tokens = next_ids.clone()

            # tokens for the NEXT DECODER INPUT
            dec_tokens = next_ids.clone()

            if is_ptr.any():
                for b in range(B):
                    if not is_ptr[b]:
                        continue
                    enc_pos: int = next_ids[b].item()

                    # --- decode output: word-index integer ---
                    if input_words_batch is not None:
                        word_idx = encoder_pos_to_word_idx(
                            enc_pos, input_words_batch[b], tok
                        )
                    else:
                        word_idx = enc_pos - 1  # fast approx (BOS offset)

                    if word_idx >= 0:
                        # encode " 2" → single token Ġ2
                        pos_token_ids = tok.encode(f" {word_idx}", add_special_tokens=False)
                        out_tokens[b] = pos_token_ids[0] if pos_token_ids else next_ids[b]
                    else:
                        out_tokens[b] = next_ids[b]

                    # --- decoder input: actual wordpiece ---
                    if 0 < enc_pos < N:
                        dec_tokens[b] = input_ids[b, enc_pos]

            output_ids = torch.cat([output_ids, out_tokens.unsqueeze(1)], dim=1)
            decoder_input = torch.cat([decoder_input, dec_tokens.unsqueeze(1)], dim=1)

            eos = out_tokens == tok.eos_token_id
            finished |= eos
            if finished.all():
                break

        generated = [output_ids[b, 1:].tolist() for b in range(B)]

        if return_scores:
            return {
                "generated_ids": generated,
                "intent_entropies": intent_entropies,
            }
        return generated
