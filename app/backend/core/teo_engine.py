"""
teo_engine.py
-------------
The ONE file the app touches to talk to Teo's brain.
Everything upstream (routes/chat.py) only ever calls engine.get_response().
Everything downstream (tokenizer, LSTM, future Transformer) changes freely
without anything in app/ ever needing to change.

Why a class and not a function (Phase 1 was a bare function — flagged in
review: "teo_engine.py Is Stateless"):
A real model needs to load weights and a tokenizer ONCE, keep them in
memory, and optionally see conversation history. A pure function can't
hold that state between calls. A singleton instance can.

Current status: no trained model exists yet (training comes after the
language-model objective + real dataset volume are in place — see
docs/architecture.md Section 3). This class is fully wired for that
day; right now it runs in placeholder mode and says so honestly rather
than pretending to be smarter than it is.
"""

import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = ROOT / "ai" / "training" / "checkpoints"
TOKENIZER_PATH = ROOT / "datasets" / "tokenizer" / "vocab.json"
CARD_PATH = MODEL_DIR / "model_card.json"
WEIGHTS_PATH = MODEL_DIR / "best_weights.npz"
START, EOS, SEP = 2, 3, 4


class TeoEngine:
    """
    Singleton-style engine. Instantiated once at import time (see `engine`
    at the bottom of this file) so weights/tokenizer load only once per
    process, not once per request.
    """

    def __init__(self) -> None:
        self.tokenizer = None
        self.lstm = None
        self.dense = None
        self._load_attempted = False
        self._try_load()

    def _try_load(self) -> None:
        """Attempt to load a trained model + tokenizer. Safe no-op if absent."""
        self._load_attempted = True

        if TOKENIZER_PATH.exists():
            sys.path.insert(0, str(ROOT / "ai" / "tokenizer"))
            from tokenizer import CharTokenizer
            self.tokenizer = CharTokenizer.load(str(TOKENIZER_PATH))

        if CARD_PATH.exists() and WEIGHTS_PATH.exists():
            try:
                sys.path.insert(0, str(ROOT / "ai" / "nanonet"))
                from nanonet.layers.lstm import LSTM
                from nanonet.layers.dense import Dense
                from nanonet.utils.io import load_weights

                with open(CARD_PATH, encoding="utf-8") as f:
                    card = json.load(f)

                arch = card["architecture"]
                self.lstm = LSTM(
                    input_size=arch["vocab_size"],
                    hidden_size=arch["hidden_size"],
                    return_sequences=True,
                )
                self.dense = Dense(arch["hidden_size"], arch["vocab_size"])
                load_weights([self.lstm, self.dense], str(WEIGHTS_PATH))
                self.lstm.eval()
                self.dense.eval()
            except Exception:
                self.lstm = None
                self.dense = None

    def is_model_loaded(self) -> bool:
        return self.lstm is not None and self.dense is not None

    def _one_hot_sequence(self, ids: list[int], vocab_size: int):
        import numpy as np
        out = np.zeros((1, len(ids), vocab_size), dtype=np.float32)
        for t, idx in enumerate(ids):
            out[0, t, idx] = 1.0
        return out

    def _sample_token(self, logits, temperature: float = 0.8, greedy: bool = False) -> int:
        import numpy as np
        if greedy:
            return int(np.argmax(logits))
        scaled = logits / max(temperature, 1e-6)
        shifted = scaled - scaled.max()
        probs = np.exp(shifted)
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))

    def _generate(self, user_message: str, max_tokens: int = 300, temperature: float = 0.8) -> str:
        import numpy as np
        if self.tokenizer is None or self.lstm is None or self.dense is None:
            raise RuntimeError("Model is not usable")

        V = self.tokenizer.vocab_size
        prefix_ids = [START] + self.tokenizer.encode_chars(user_message) + [SEP]
        generated_ids: list[int] = []

        try:
            for _ in range(max_tokens):
                current_ids = prefix_ids + generated_ids
                x = self._one_hot_sequence(current_ids, V)
                lstm_out = self.lstm.forward(x)
                last_hidden = lstm_out[0, -1, :][np.newaxis, :]
                logits = self.dense.forward(last_hidden)[0]
                next_id = self._sample_token(logits, temperature=temperature)

                if next_id == EOS:
                    break
                if next_id == 0:  # PAD
                    continue
                generated_ids.append(next_id)

            return self.tokenizer.decode(generated_ids)
        except Exception as exc:
            raise RuntimeError("Generation failed") from exc

    def get_response(self, user_message: str, history: Optional[list] = None) -> str:
        """
        history: list of {"role": "user"|"teo", "content": str}, most recent last.
        Ignored until Phase 2 (Memory Layer) actually uses it — accepted now
        so the contract doesn't change later (see architecture.md Phase 2).
        """
        if self.is_model_loaded():
            try:
                return self._generate(user_message)
            except Exception:
                return (
                    "Accepted. Brain's not trained yet — you're talking to placeholder Teo. "
                    "Tokenizer + language-model training loop are in ai/training/. "
                    "Run that, drop a checkpoint in ai/training/checkpoints/, and I wake up."
                )

        if CARD_PATH.exists() and not WEIGHTS_PATH.exists():
            return (
                "Model card found but weights are missing. "
                "Run: python ai/training/train_language_model.py"
            )

        return (
            "Accepted. Brain's not trained yet — you're talking to placeholder Teo. "
            "Tokenizer + language-model training loop are in ai/training/. "
            "Run that, drop a checkpoint in ai/training/checkpoints/, and I wake up."
        )


# Single shared instance — import this, not the class, from routes/chat.py
engine = TeoEngine()


def get_response(user_message: str, history: Optional[list] = None) -> str:
    """Thin backward-compatible wrapper so existing imports don't break."""
    return engine.get_response(user_message, history)
