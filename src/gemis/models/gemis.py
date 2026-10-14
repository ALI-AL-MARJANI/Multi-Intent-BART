"""GEMIS: BART encoder-decoder + AoA cross-attention + pointer head.

Reimplementation of Li & Zhu (2026), "A Generative Model for Joint Multiple
Intent Detection and Slot Filling" (arXiv:2602.08322). No official code was
released; ``README.md`` lists the points where this implementation had to make
a choice the paper leaves open.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BartConfig, BartModel, PreTrainedTokenizerFast

from gemis.data.labels import LabelSpace
from gemis.data.tokenizer import add_label_tokens, init_label_embeddings
from gemis.models.bart_layer import GEMISDecoderLayer
from gemis.models.pointer import PointerHead


@dataclass
class GenerationOutput:
    """Greedy decoding result for a batch of ``B`` utterances.

    Attributes:
        sequences: class ids in ``[0, L + N)`` per utterance, ``</s>`` excluded.
        log_probs: log-probability of each emitted class, ``</s>`` included.
        entropies: entropy of the output distribution at each of those steps.
        first_step_label_logits: (B, L) label logits at the first step.
        first_step_max_prob: (B,) largest probability at the first step.
        encoder_mean: (B, D) mean of the encoder states over real tokens.
    """

    sequences: list[list[int]]
    log_probs: list[list[float]]
    entropies: list[list[float]]
    first_step_label_logits: torch.Tensor
    first_step_max_prob: torch.Tensor
    encoder_mean: torch.Tensor


class GEMISModel(nn.Module):
    """Joint multi-intent detection and slot filling as sequence generation.

    Args:
        tokenizer: BART tokenizer; label tokens are added to it in place.
        label_space: intents / slot types (and position classes if no pointer).
        backbone: Hugging Face id of the pretrained BART (e.g. ``facebook/bart-large``).
        bart_config: build a randomly initialised BART instead (tests, checkpoints).
        mlp_hidden: hidden size of the pointer MLP.
        use_aoa: replace decoder cross-attention with attention-over-attention.
        use_pointer: predict word positions by pointing at encoder states.
        label_init: ``subword_mean`` or ``random`` initialisation of label embeddings.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        label_space: LabelSpace,
        backbone: str | None = None,
        bart_config: BartConfig | None = None,
        mlp_hidden: int = 512,
        use_aoa: bool = True,
        use_pointer: bool = True,
        label_init: str = "subword_mean",
    ) -> None:
        super().__init__()
        if (backbone is None) == (bart_config is None) and bart_config is None:
            raise ValueError("give a backbone name or a bart_config")
        if not use_pointer and label_space.n_positions == 0:
            raise ValueError("use_pointer=False needs a LabelSpace with position classes")

        self.label_space = label_space
        self.backbone = backbone
        self.mlp_hidden = mlp_hidden
        self.use_aoa = use_aoa
        self.use_pointer = use_pointer
        self.label_init = label_init

        if bart_config is not None:
            self.bart = BartModel(bart_config)
        else:
            self.bart = BartModel.from_pretrained(backbone)
        config = self.bart.config

        label_token_ids = add_label_tokens(tokenizer, label_space)
        init_label_embeddings(self.bart, tokenizer, label_space, label_token_ids, label_init)
        self.register_buffer("label_token_ids", torch.tensor(label_token_ids), persistent=False)

        if use_aoa:
            for layer in self.bart.decoder.layers:
                GEMISDecoderLayer.convert(layer, config.attention_dropout)

        self.pointer = PointerHead(config.d_model, mlp_hidden, dropout=config.dropout)
        self.pad_token_id: int = tokenizer.pad_token_id
        self.decoder_start_token_id: int = (
            config.decoder_start_token_id if config.decoder_start_token_id is not None else tokenizer.bos_token_id
        )

    # ── checkpoints ──────────────────────────────────────────────────────────

    def export_config(self) -> dict[str, Any]:
        """Everything needed to rebuild the architecture (not the weights)."""
        bart_config = self.bart.config.to_dict()
        return {
            "backbone": self.backbone,
            "bart_config": bart_config,
            "mlp_hidden": self.mlp_hidden,
            "use_aoa": self.use_aoa,
            "use_pointer": self.use_pointer,
            "label_init": self.label_init,
            "intents": self.label_space.intents,
            "slots": self.label_space.slots,
            "n_positions": self.label_space.n_positions,
        }

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        tokenizer: PreTrainedTokenizerFast | None = None,
        map_location: str | torch.device = "cpu",
    ) -> tuple[GEMISModel, PreTrainedTokenizerFast, dict[str, Any]]:
        """Rebuild model and tokenizer from a file written by ``GEMISTrainer``."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        cfg = checkpoint["model_config"]
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
        bart_config = BartConfig.from_dict(cfg["bart_config"])
        # the saved config has the extended vocabulary; rebuild at the original size
        bart_config.vocab_size = len(tokenizer)
        model = cls(
            tokenizer=tokenizer,
            label_space=LabelSpace(cfg["intents"], cfg["slots"], cfg["n_positions"]),
            bart_config=bart_config,
            mlp_hidden=cfg["mlp_hidden"],
            use_aoa=cfg["use_aoa"],
            use_pointer=cfg["use_pointer"],
            label_init="random",
        )
        model.backbone, model.label_init = cfg["backbone"], cfg["label_init"]
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, tokenizer, checkpoint

    # ── building blocks ──────────────────────────────────────────────────────

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.bart.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def _decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.bart.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state

    def _logits(
        self,
        decoder_hidden: torch.Tensor,
        encoder_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embed = self.bart.get_input_embeddings()
        label_embeddings = embed(self.label_token_ids)
        if not self.use_pointer:
            return self.pointer(decoder_hidden, label_embeddings)
        return self.pointer(
            decoder_hidden,
            label_embeddings,
            encoder_hidden=encoder_hidden,
            encoder_embeddings=embed(input_ids),
            attention_mask=attention_mask,
        )

    # ── training ─────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, N)
        attention_mask: torch.Tensor,  # (B, N)
        decoder_input_ids: torch.Tensor,  # (B, T)
        labels: torch.Tensor | None = None,  # (B, T) class ids in [0, L + N), -100 = ignore
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced pass. Returns ``logits`` (B, T, L + N) and, with labels, ``loss``."""
        encoder_hidden = self.encode(input_ids, attention_mask)
        decoder_hidden = self._decode(decoder_input_ids, encoder_hidden, attention_mask)
        logits = self._logits(decoder_hidden, encoder_hidden, input_ids, attention_mask)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100)
        return output

    # ── inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 96,
    ) -> GenerationOutput:
        """Greedy decoding (the paper reports no gain from beam search).

        After a pointer step the decoder reads the pointed wordpiece; after a
        label step it reads the label token.
        """
        batch_size = input_ids.size(0)
        n_labels = len(self.label_space)
        device = input_ids.device

        encoder_hidden = self.encode(input_ids, attention_mask)
        mask = attention_mask.unsqueeze(-1).to(encoder_hidden.dtype)
        encoder_mean = (encoder_hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

        decoder_input = torch.full((batch_size, 1), self.decoder_start_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        chosen, chosen_logp, entropy, alive = [], [], [], []
        first_logits = first_max_prob = None

        for _ in range(max_new_tokens):
            decoder_hidden = self._decode(decoder_input, encoder_hidden, attention_mask)
            logits = self._logits(decoder_hidden[:, -1:], encoder_hidden, input_ids, attention_mask).squeeze(1).float()
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            next_class = log_probs.argmax(dim=-1)

            if first_logits is None:
                first_logits = logits[:, :n_labels]
                first_max_prob = probs.max(dim=-1).values

            alive.append(~finished)
            chosen.append(next_class)
            chosen_logp.append(log_probs.gather(1, next_class[:, None]).squeeze(1))
            entropy.append(-torch.where(probs > 0, probs * log_probs, 0.0).sum(dim=-1))

            is_pointer = next_class >= n_labels
            label_tokens = self.label_token_ids[next_class.clamp(max=n_labels - 1)]
            pointed = input_ids.gather(1, (next_class - n_labels).clamp(min=0)[:, None]).squeeze(1)
            next_token = torch.where(is_pointer, pointed, label_tokens)
            finished = finished | (next_class == self.label_space.eos_id)
            next_token = next_token.masked_fill(finished, self.pad_token_id)
            decoder_input = torch.cat([decoder_input, next_token[:, None]], dim=1)
            if finished.all():
                break

        alive_t = torch.stack(alive, 1).cpu()
        chosen_t = torch.stack(chosen, 1).cpu()
        logp_t = torch.stack(chosen_logp, 1).cpu()
        entropy_t = torch.stack(entropy, 1).cpu()

        sequences, log_probs_out, entropies_out = [], [], []
        for b in range(batch_size):
            keep = alive_t[b]
            classes = chosen_t[b][keep].tolist()
            log_probs_out.append(logp_t[b][keep].tolist())
            entropies_out.append(entropy_t[b][keep].tolist())
            if classes and classes[-1] == self.label_space.eos_id:
                classes = classes[:-1]
            sequences.append(classes)

        return GenerationOutput(
            sequences=sequences,
            log_probs=log_probs_out,
            entropies=entropies_out,
            first_step_label_logits=first_logits.cpu(),
            first_step_max_prob=first_max_prob.cpu(),
            encoder_mean=encoder_mean.float().cpu(),
        )
