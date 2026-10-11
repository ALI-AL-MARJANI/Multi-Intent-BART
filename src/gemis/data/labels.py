"""Label space: the ``L`` non-pointer classes the decoder can emit.

The output distribution at each step covers ``L + N`` classes (paper Eq. 6):
the first ``L`` are labels (end-of-sequence, intents, slot types), the last
``N`` are pointers to encoder positions. Labels come first so that a target
index does not depend on how much a batch is padded.

When the pointer network is disabled (ablation), word positions are emitted
as ``<pos:k>`` label classes instead.
"""

from __future__ import annotations

from dataclasses import dataclass

EOS = "</s>"


@dataclass(frozen=True)
class Item:
    """One decoded element of a target sequence."""

    kind: str  # "eos" | "intent" | "slot" | "pos"
    value: str | int | None = None


class LabelSpace:
    """Maps intents, slot types (and optional position classes) to class ids."""

    def __init__(self, intents: list[str], slots: list[str], n_positions: int = 0) -> None:
        self.intents = list(intents)
        self.slots = list(slots)
        self.n_positions = n_positions
        self._intent_offset = 1
        self._slot_offset = self._intent_offset + len(self.intents)
        self._pos_offset = self._slot_offset + len(self.slots)
        self._intent_ids = {name: self._intent_offset + i for i, name in enumerate(self.intents)}
        self._slot_ids = {name: self._slot_offset + i for i, name in enumerate(self.slots)}

    def __len__(self) -> int:
        return self._pos_offset + self.n_positions

    @property
    def eos_id(self) -> int:
        return 0

    @property
    def intent_ids(self) -> list[int]:
        return list(range(self._intent_offset, self._slot_offset))

    def intent_id(self, name: str) -> int:
        return self._intent_ids[name]

    def slot_id(self, name: str) -> int:
        return self._slot_ids[name]

    def pos_id(self, word_index: int) -> int:
        if not 0 <= word_index < self.n_positions:
            raise KeyError(f"word index {word_index} outside the {self.n_positions} position classes")
        return self._pos_offset + word_index

    def is_intent(self, class_id: int) -> bool:
        return self._intent_offset <= class_id < self._slot_offset

    def has_intent(self, name: str) -> bool:
        return name in self._intent_ids

    def has_slot(self, name: str) -> bool:
        return name in self._slot_ids

    def decode(self, class_id: int) -> Item:
        if class_id == 0:
            return Item("eos")
        if class_id < self._slot_offset:
            return Item("intent", self.intents[class_id - self._intent_offset])
        if class_id < self._pos_offset:
            return Item("slot", self.slots[class_id - self._slot_offset])
        if class_id < len(self):
            return Item("pos", class_id - self._pos_offset)
        raise IndexError(class_id)

    def token_strings(self) -> list[str]:
        """Tokenizer token for every class, in class-id order."""
        return (
            [EOS]
            + [f"<intent:{name}>" for name in self.intents]
            + [f"<slot:{name}>" for name in self.slots]
            + [f"<pos:{k}>" for k in range(self.n_positions)]
        )

    def init_texts(self) -> list[str | None]:
        """Text whose subword embeddings initialise each class token (None = keep)."""
        return [None] + list(self.intents) + list(self.slots) + [f" {k}" for k in range(self.n_positions)]
