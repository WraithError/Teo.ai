"""
ai/training/generate.py

Load a trained Teo checkpoint and generate responses interactively.
Reads model_card.json to reconstruct the exact architecture.
Works with both old (CharLM_LSTM/one-hot) and new (Embedding) checkpoints.

Usage:
    cd Teo.AI
    python ai/training/generate.py
    python ai/training/generate.py --temperature 0.7 --max-tokens 200 --greedy
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

from nanonet.backend          import xp
from nanonet.layers.embedding import Embedding
from nanonet.layers.lstm      import LSTM
from nanonet.layers.dense     import Dense
from nanonet.utils.io         import load_weights
from tokenizer                import CharTokenizer

CHECKPOINT_DIR = ROOT / "ai" / "training" / "checkpoints"
CARD_PATH      = CHECKPOINT_DIR / "model_card.json"
WEIGHTS_PATH   = CHECKPOINT_DIR / "best_weights.npz"

PAD, UNK, START, EOS, SEP = 0, 1, 2, 3, 4


def sample_token(logits: np.ndarray, temperature: float, greedy: bool) -> int:
    if greedy:
        return int(np.argmax(logits))
    scaled  = logits / max(temperature, 1e-6)
    shifted = scaled - scaled.max()
    probs   = np.exp(shifted)
    probs  /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def generate(
    user_message: str, tokenizer: CharTokenizer,
    embed: Embedding, lstm: LSTM, dense: Dense,
    max_tokens: int, temperature: float, greedy: bool,
    arch_type: str,
) -> str:
    V = tokenizer.vocab_size
    prefix_ids  = [START] + tokenizer.encode_chars(user_message) + [SEP]
    generated   = []

    for _ in range(max_tokens):
        current = prefix_ids + generated
        ids_arr = np.array([current], dtype=np.int32)  # (1, T)

        if arch_type == "CharLM_LSTM":
            # Legacy one-hot path
            T = len(current)
            x = np.zeros((1, T, V), dtype=np.float32)
            for t, idx in enumerate(current):
                x[0, t, idx] = 1.0
            x_xp = xp.array(x)
        else:
            # New Embedding path
            x_xp = xp.array(ids_arr)            # (1, T)
            x_xp = embed.forward(x_xp)          # (1, T, embed_dim)

        lstm_out = lstm.forward(x_xp)            # (1, T, hidden)
        last_h   = lstm_out[0, -1:, :]           # (1, hidden)
        logits   = dense.forward(last_h)[0]      # (vocab_size,)

        if hasattr(logits, 'get'):               # CuPy → numpy
            logits = logits.get()

        next_id = sample_token(np.array(logits), temperature, greedy)

        if next_id == EOS:
            break
        if next_id == PAD:
            continue
        generated.append(next_id)

    return tokenizer.decode(generated)


def load_model(card: dict):
    arch      = card["architecture"]
    arch_type = arch.get("type", "CharLM_LSTM")
    V         = arch["vocab_size"]
    hidden    = arch["hidden_size"]
    embed_dim = arch.get("embed_dim", V)   # fallback for old checkpoints

    tokenizer = CharTokenizer.load(str(ROOT / "datasets" / "tokenizer" / "vocab.json"))

    if arch_type == "CharLM_LSTM":
        # Old one-hot architecture — embed is a dummy placeholder
        embed = Embedding(V, V)      # won't be used in generation
        lstm  = LSTM(input_size=V,  hidden_size=hidden, return_sequences=True)
    else:
        embed = Embedding(V, embed_dim)
        lstm  = LSTM(input_size=embed_dim, hidden_size=hidden, return_sequences=True)

    dense = Dense(hidden, V)
    load_weights([embed, lstm, dense], str(WEIGHTS_PATH))
    embed.eval()
    lstm.eval()
    dense.eval()
    return tokenizer, embed, lstm, dense, arch_type


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens",  type=int,   default=300, dest="max_tokens")
    p.add_argument("--greedy",      action="store_true")
    cfg = p.parse_args()

    if not CARD_PATH.exists() or not WEIGHTS_PATH.exists():
        print("No checkpoint. Train first:")
        print("  python ai/training/train_language_model.py")
        sys.exit(1)

    with open(CARD_PATH, encoding="utf-8") as f:
        card = json.load(f)

    arch_type = card["architecture"].get("type", "CharLM_LSTM")

    print("=" * 60)
    print("TEO — generation")
    print(f"  Epoch      : {card['saved_epoch']}")
    print(f"  Val loss   : {card['val_loss']}  (ppl {card['perplexity']})")
    print(f"  Arch       : {arch_type}")
    print(f"  Temperature: {cfg.temperature}")
    print("=" * 60)
    print("Type a message. 'exit' to quit.\n")

    tokenizer, embed, lstm, dense, arch_type = load_model(card)

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
            user_input, tokenizer, embed, lstm, dense,
            cfg.max_tokens, cfg.temperature, cfg.greedy, arch_type,
        )
        print(f"Teo: {response}\n")


if __name__ == "__main__":
    main()
