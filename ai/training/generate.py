"""
ai/training/generate.py

Load a trained Teo checkpoint and generate responses interactively.
Reads model_card.json to reconstruct the exact architecture, so you
don't have to manually match hyperparameters.

Usage:
    cd Teo.AI
    python ai/training/generate.py
    python ai/training/generate.py --temperature 0.8 --max-tokens 200

Flags:
    --temperature   Sampling temperature. Lower = more conservative.
                    0.5–0.9 works well. 1.0 = raw probabilities. Default 0.8.
    --max-tokens    Max characters Teo will generate per response. Default 300.
    --greedy        Use greedy decoding instead of sampling (always picks
                    the most likely character — more repetitive but stable).
"""

from __future__ import annotations
import argparse
import json
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "ai" / "nanonet"))
sys.path.insert(0, str(ROOT / "ai" / "tokenizer"))

from nanonet.layers.lstm  import LSTM
from nanonet.layers.dense import Dense
from nanonet.utils.io     import load_weights
from tokenizer            import CharTokenizer

CHECKPOINT_DIR = ROOT / "ai" / "training" / "checkpoints"
CARD_PATH      = CHECKPOINT_DIR / "model_card.json"
WEIGHTS_PATH   = CHECKPOINT_DIR / "best_weights.npz"
PAD, UNK, START, EOS, SEP = 0, 1, 2, 3, 4


def one_hot_sequence(ids: list[int], vocab_size: int) -> np.ndarray:
    """ids (list of int) → (1, len, vocab_size) float32"""
    T = len(ids)
    out = np.zeros((1, T, vocab_size), dtype=np.float32)
    for t, idx in enumerate(ids):
        out[0, t, idx] = 1.0
    return out


def sample_token(logits: np.ndarray, temperature: float, greedy: bool) -> int:
    """
    logits : (vocab_size,) raw scores from Dense
    Returns a single sampled token id.
    """
    if greedy:
        return int(np.argmax(logits))

    # Temperature scaling + softmax
    scaled = logits / max(temperature, 1e-6)
    shifted = scaled - scaled.max()
    probs = np.exp(shifted)
    probs /= probs.sum()

    return int(np.random.choice(len(probs), p=probs))


def generate(
    user_message: str,
    tokenizer: CharTokenizer,
    lstm: LSTM,
    dense: Dense,
    max_tokens: int,
    temperature: float,
    greedy: bool,
) -> str:
    """
    Generate Teo's response to user_message character by character.

    Strategy: feed the full prefix <START> user <SEP> to the LSTM, then
    sample from the output at the last position. Append the sampled token,
    re-feed the full prefix + generated chars, repeat until <EOS> or max_tokens.

    This is O(n²) in the generated length — acceptable since sequences are short.
    A stateful approach (caching h,c) is the right optimization for a larger model.
    """
    V = tokenizer.vocab_size

    # Build prefix: <START> user_chars <SEP>
    prefix_ids = [START] + tokenizer.encode_chars(user_message) + [SEP]
    generated_ids: list[int] = []

    for _ in range(max_tokens):
        current_ids = prefix_ids + generated_ids
        x = one_hot_sequence(current_ids, V)           # (1, T, V)

        # LSTM forward
        lstm_out = lstm.forward(x)                     # (1, T, hidden)

        # Dense on last position only
        last_hidden = lstm_out[0, -1, :]               # (hidden,)
        last_hidden_2d = last_hidden[np.newaxis, :]    # (1, hidden)
        logits = dense.forward(last_hidden_2d)[0]      # (vocab_size,)

        # Sample next token
        next_id = sample_token(logits, temperature, greedy)

        if next_id == EOS:
            break
        if next_id == PAD:
            continue

        generated_ids.append(next_id)

    return tokenizer.decode(generated_ids)


def load_model(
    card: dict,
) -> tuple[CharTokenizer, LSTM, Dense]:
    arch   = card["architecture"]
    V      = arch["vocab_size"]
    hidden = arch["hidden_size"]

    tokenizer_path = ROOT / "datasets" / "tokenizer" / "vocab.json"
    tokenizer = CharTokenizer.load(str(tokenizer_path))

    lstm  = LSTM(input_size=V, hidden_size=hidden, return_sequences=True)
    dense = Dense(hidden, V)

    load_weights([lstm, dense], str(WEIGHTS_PATH))

    # Switch to eval mode (disables Dropout if any)
    lstm.eval()
    dense.eval()

    return tokenizer, lstm, dense


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens",  type=int,   default=300, dest="max_tokens")
    p.add_argument("--greedy",      action="store_true")
    cfg = p.parse_args()

    if not CARD_PATH.exists() or not WEIGHTS_PATH.exists():
        print("No checkpoint found. Train first:")
        print("    python ai/training/train_language_model.py")
        sys.exit(1)

    with open(CARD_PATH, encoding="utf-8") as f:
        card = json.load(f)

    print("=" * 60)
    print("TEO — generation mode")
    print(f"  Checkpoint  : epoch {card['saved_epoch']}")
    print(f"  Val loss    : {card['val_loss']}  (ppl {card['perplexity']})")
    print(f"  Temperature : {cfg.temperature}")
    print(f"  Strategy    : {'greedy' if cfg.greedy else 'sampling'}")
    print("=" * 60)
    print("Type a message. Ctrl-C or 'exit' to quit.\n")

    tokenizer, lstm, dense = load_model(card)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        response = generate(
            user_input, tokenizer, lstm, dense,
            cfg.max_tokens, cfg.temperature, cfg.greedy
        )
        print(f"Teo: {response}\n")


if __name__ == "__main__":
    main()
