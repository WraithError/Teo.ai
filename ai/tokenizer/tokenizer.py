"""
ai/tokenizer/tokenizer.py

Character-level tokenizer. No external dependencies, no subword merges —
every character in the training corpus gets one slot in the vocabulary.

Why character-level and not word/BPE (see docs/architecture.md Section 3):
  - Trivial to implement correctly from scratch.
  - Zero out-of-vocabulary problem — en/uz/mixed/code all just work,
    since they're built from the same Latin character set plus symbols.
  - Vocab stays tiny (under 100 tokens for Teo's current dataset),
    which matters a lot when the output layer is Dense(hidden, vocab_size) —
    smaller vocab = smaller, faster, easier-to-train output layer.
  - The cost: longer sequences than word-level for the same sentence,
    and the model has to learn spelling before it learns words. Acceptable
    tradeoff at this project's current scale. Revisit (BPE) once the
    dataset is large enough that sequence length becomes the bottleneck.

Special tokens occupy the first 5 vocabulary slots, fixed:
    0 = <PAD>   padding, ignored in loss
    1 = <UNK>   any character not seen during vocab-building
    2 = <START> beginning of a full exchange
    3 = <EOS>   end of a full exchange
    4 = <SEP>   boundary between the user's turn and Teo's turn
"""

from __future__ import annotations
import json
from pathlib import Path

PAD, UNK, START, EOS, SEP = 0, 1, 2, 3, 4
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<START>", "<EOS>", "<SEP>"]


class CharTokenizer:
    def __init__(self, char_to_id: dict[str, int], id_to_char: dict[int, str]) -> None:
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    @classmethod
    def build_from_texts(cls, texts: list[str]) -> "CharTokenizer":
        """Scan every string in `texts`, assign each unique character an id."""
        chars: set[str] = set()
        for t in texts:
            chars.update(t)

        char_to_id: dict[str, int] = {}
        for i, tok in enumerate(SPECIAL_TOKENS):
            char_to_id[tok] = i

        for i, ch in enumerate(sorted(chars), start=len(SPECIAL_TOKENS)):
            char_to_id[ch] = i

        id_to_char = {i: ch for ch, i in char_to_id.items()}
        return cls(char_to_id, id_to_char)

    def encode_chars(self, text: str) -> list[int]:
        """Plain character encoding — no special tokens added."""
        unk_id = self.char_to_id["<UNK>"]
        return [self.char_to_id.get(ch, unk_id) for ch in text]

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        out = []
        for i in ids:
            ch = self.id_to_char.get(int(i), "<UNK>")
            if strip_special and ch in SPECIAL_TOKENS:
                continue
            out.append(ch)
        return "".join(out)

    def encode_exchange(self, user_text: str, assistant_text: str) -> tuple[list[int], int]:
        """
        Build the full token sequence for one training example:
            <START> user_chars <SEP> assistant_chars <EOS>

        Returns (ids, sep_index) — sep_index is where <SEP> landed, so the
        training script knows where the assistant's turn begins (needed to
        mask the loss to only the assistant span — see train_language_model.py).
        """
        ids = [START] + self.encode_chars(user_text) + [SEP]
        sep_index = len(ids) - 1
        ids += self.encode_chars(assistant_text) + [EOS]
        return ids, sep_index

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char_to_id": self.char_to_id}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        char_to_id = data["char_to_id"]
        id_to_char = {int(i): ch for ch, i in char_to_id.items()}
        return cls(char_to_id, id_to_char)

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
