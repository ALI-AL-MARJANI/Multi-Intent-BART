"""Turn a generated class-id sequence back into a semantic frame."""

from __future__ import annotations

from gemis.data.alignment import position_to_word
from gemis.data.dataset import Frame
from gemis.data.labels import Item, LabelSpace


def sequence_to_items(sequence: list[int], label_space: LabelSpace, word_starts: list[int]) -> list[Item]:
    """Class ids -> items; pointer classes become word indices."""
    n_labels = len(label_space)
    items: list[Item] = []
    for class_id in sequence:
        if class_id < n_labels:
            items.append(label_space.decode(class_id))
        else:
            items.append(Item("pos", position_to_word(class_id - n_labels, word_starts)))
    return items


def items_to_frame(items: list[Item]) -> Frame:
    """Parse ``intent* (pos pos slot)*``.

    Decoding is unconstrained, so the sequence may be malformed. Leading
    intents are collected; afterwards every well-formed ``pos pos slot``
    triple with ``0 <= start < end`` is kept and anything else is skipped.
    """
    intents: list[str] = []
    i = 0
    while i < len(items) and items[i].kind == "intent":
        intents.append(items[i].value)  # type: ignore[arg-type]
        i += 1

    slots: list[tuple[int, int, str]] = []
    while i + 2 < len(items):
        a, b, c = items[i : i + 3]
        if a.kind == "pos" and b.kind == "pos" and c.kind == "slot":
            if 0 <= a.value < b.value:  # type: ignore[operator]
                slots.append((a.value, b.value, c.value))  # type: ignore[arg-type]
            i += 3
        else:
            i += 1
    return Frame(frozenset(intents), frozenset(slots))


def decode_frame(sequence: list[int], label_space: LabelSpace, word_starts: list[int]) -> Frame:
    return items_to_frame(sequence_to_items(sequence, label_space, word_starts))
