"""Evaluation metrics for GEMIS.

Three metrics reported in the paper:
    1. Slot F1         — micro-averaged F1 over exact slot spans.
    2. Intent Accuracy — exact-set match of intent labels per utterance.
    3. Overall Accuracy — exact match of BOTH intents AND all slot spans.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerFast


@dataclass
class Frame:
    """Parsed semantic frame from a decoded target sequence."""
    intents: frozenset[str]
    # (start_word_idx, end_word_idx_exclusive, slot_type)
    slot_spans: list[tuple[int, int, str]]


def _clean(s: str) -> str:
    """Strip BPE Ġ/▁ prefixes and surrounding whitespace."""
    return s.lstrip("Ġ▁").strip()


def parse_target_sequence(
    token_ids: list[int],
    tokenizer: PreTrainedTokenizerFast,
) -> Frame:
    """Decode a generated token id sequence into a Frame.

    Target format: <intent:X> <intent:Y> ... start end <slot:Z> ...

    Position tokens decode to plain integers (after stripping Ġ prefix).
    Intent/slot tokens have the form <intent:X> or <slot:Z>.
    """
    intents: list[str] = []
    slot_spans: list[tuple[int, int, str]] = []

    # Decode each token to its text representation
    decoded: list[str] = [
        _clean(tokenizer.decode([tid], skip_special_tokens=False))
        for tid in token_ids
    ]

    i = 0
    # ── collect intents ───────────────────────────────────────────────────────
    while i < len(decoded):
        tok = decoded[i]
        if tok.startswith("<intent:") and tok.endswith(">"):
            intents.append(tok[len("<intent:"):-1])
            i += 1
        else:
            break

    # ── collect (start, end, slot_type) triples ───────────────────────────────
    while i + 2 < len(decoded):
        start_str = decoded[i]
        end_str = decoded[i + 1]
        slot_str = decoded[i + 2]

        try:
            start = int(start_str)
            end = int(end_str)
        except ValueError:
            i += 1
            continue

        if slot_str.startswith("<slot:") and slot_str.endswith(">"):
            slot_type = slot_str[len("<slot:"):-1]
            slot_spans.append((start, end, slot_type))
            i += 3
        else:
            i += 1

    return Frame(intents=frozenset(intents), slot_spans=slot_spans)


# ── metric functions ──────────────────────────────────────────────────────────

def intent_accuracy(golds: list[Frame], preds: list[Frame]) -> float:
    correct = sum(g.intents == p.intents for g, p in zip(golds, preds))
    return correct / len(golds) if golds else 0.0


def slot_f1(golds: list[Frame], preds: list[Frame]) -> float:
    total_tp = total_fp = total_fn = 0

    for gold, pred in zip(golds, preds):
        gold_set = set(gold.slot_spans)
        pred_set = set(pred.slot_spans)
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def overall_accuracy(golds: list[Frame], preds: list[Frame]) -> float:
    correct = sum(
        g.intents == p.intents and set(g.slot_spans) == set(p.slot_spans)
        for g, p in zip(golds, preds)
    )
    return correct / len(golds) if golds else 0.0


def compute_metrics(golds: list[Frame], preds: list[Frame]) -> dict[str, float]:
    return {
        "slot_f1":          slot_f1(golds, preds) * 100,
        "intent_accuracy":  intent_accuracy(golds, preds) * 100,
        "overall_accuracy": overall_accuracy(golds, preds) * 100,
    }
