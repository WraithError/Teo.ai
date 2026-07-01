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

from pathlib import Path
from typing import Optional

MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "ai" / "training" / "checkpoints"
TOKENIZER_PATH = Path(__file__).resolve().parent.parent.parent.parent / "datasets" / "tokenizer" / "vocab.json"


class TeoEngine:
    """
    Singleton-style engine. Instantiated once at import time (see `engine`
    at the bottom of this file) so weights/tokenizer load only once per
    process, not once per request.
    """

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self._load_attempted = False
        self._try_load()

    def _try_load(self) -> None:
        """Attempt to load a trained model + tokenizer. Safe no-op if absent."""
        self._load_attempted = True

        if TOKENIZER_PATH.exists():
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "ai" / "tokenizer"))
            from tokenizer import CharTokenizer
            self.tokenizer = CharTokenizer.load(str(TOKENIZER_PATH))

        # Model checkpoint loading goes here once training produces one.
        # Left unimplemented on purpose — there is nothing to load yet,
        # and a fake load path would be worse than an honest placeholder.

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def get_response(self, user_message: str, history: Optional[list] = None) -> str:
        """
        history: list of {"role": "user"|"teo", "content": str}, most recent last.
        Ignored until Phase 2 (Memory Layer) actually uses it — accepted now
        so the contract doesn't change later (see architecture.md Phase 2).
        """
        if self.model is not None:
            # Real inference path — wired in once a checkpoint exists.
            raise NotImplementedError("Model loaded but inference path not yet implemented.")

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
