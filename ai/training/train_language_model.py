"""
ai/training/train_language_model.py

Character-level LSTM language model training for Teo.

What "language model" means here:
    Given a sequence of characters so far, predict the next character.
    This is called next-token prediction. It's the same objective that
    trained GPT-2, though the architecture and scale are very different.

Why "assistant-only loss":
    Each training example has the shape:
        <START> user chars <SEP> teo chars <EOS>
    We only compute loss on the Teo span (after <SEP>).
    We want Teo to learn to generate good responses, not to predict
    what users will type. This is the same principle as instruction tuning.

Architecture:
    one-hot(input_ids)               (batch, seq_len, vocab_size)
        → LSTM(vocab_size, hidden)   (batch, seq_len, hidden)
        → Dense(hidden, vocab_size)  (batch*seq_len, vocab_size) logits
        → CrossEntropyLoss           scalar
    Only the assistant-span logits contribute to the loss gradient.

Usage:
    cd Teo.AI
    python ai/training/train_language_model.py

    # or with options:
    python ai/training/train_language_model.py --epochs 300 --hidden-size 256

Files produced (in ai/training/checkpoints/):
    best_weights.npz    ← trained parameters (gitignored — can be large)
    model_card.json     ← architecture + training metadata (commit this)
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "ai" / "nanonet"))
sys.path.insert(0, str(ROOT / "ai" / "tokenizer"))

from nanonet.layers.lstm    import LSTM
from nanonet.layers.dense   import Dense
from nanonet.optimizers.adam import Adam
from nanonet.losses.cross_entropy import CrossEntropyLoss
from nanonet.utils.io import load_weights, save_weights
from tokenizer import CharTokenizer

VOCAB_PATH       = ROOT / "datasets" / "tokenizer" / "vocab.json"
DATASET_DIR      = ROOT / "datasets" / "train"
CHECKPOINT_DIR   = ROOT / "ai" / "training" / "checkpoints"
WEIGHTS_PATH     = CHECKPOINT_DIR / "best_weights.npz"


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

class Sample(NamedTuple):
    x:       np.ndarray   # int (seq_len,)  — input token ids
    y:       np.ndarray   # int (seq_len,)  — target token ids (x shifted by 1)
    mask:    np.ndarray   # bool (seq_len,) — True where loss should be computed


def load_dataset(tokenizer: CharTokenizer, max_seq_len: int) -> list[Sample]:
    samples: list[Sample] = []

    for fpath in sorted(DATASET_DIR.glob("*.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ids, sep_index = tokenizer.encode_exchange(
                    d["user"], d["assistant"]
                )

                # Truncate long sequences (rare with current dataset)
                ids = ids[:max_seq_len + 1]

                if len(ids) < 3:
                    continue   # pathologically short — skip

                x    = np.array(ids[:-1], dtype=np.int32)   # input
                y    = np.array(ids[1:],  dtype=np.int32)   # target = input shifted right

                # Loss mask: True at positions where y comes from the
                # assistant span (from sep_index onward in x)
                mask = np.zeros(len(x), dtype=bool)
                mask[sep_index:] = True

                samples.append(Sample(x=x, y=y, mask=mask))

    return samples


def pad_batch(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad a batch of variable-length samples to the same length."""
    max_len = max(s.x.shape[0] for s in samples)

    x_batch    = np.zeros((len(samples), max_len), dtype=np.int32)
    y_batch    = np.zeros((len(samples), max_len), dtype=np.int32)
    mask_batch = np.zeros((len(samples), max_len), dtype=bool)

    for i, s in enumerate(samples):
        L = s.x.shape[0]
        x_batch[i, :L]    = s.x
        y_batch[i, :L]    = s.y
        mask_batch[i, :L] = s.mask

    return x_batch, y_batch, mask_batch


def one_hot(ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """
    ids : (batch, seq_len) int
    out : (batch, seq_len, vocab_size) float32
    """
    out = np.zeros((*ids.shape, vocab_size), dtype=np.float32)
    b_idx = np.arange(ids.shape[0])[:, None]
    t_idx = np.arange(ids.shape[1])[None, :]
    out[b_idx, t_idx, ids] = 1.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def clip_gradients(layers: list, max_norm: float = 1.0) -> float:
    """
    Clip all gradients to max_norm (L2 global norm clipping).
    Standard practice for LSTM training — prevents exploding gradients.
    Returns the pre-clip norm so you can watch for gradient issues.
    """
    total_sq = sum(
        float(np.sum(g ** 2))
        for layer in layers
        for g in layer.grads.values()
    )
    total_norm = float(np.sqrt(total_sq))

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for layer in layers:
            for key in layer.grads:
                layer.grads[key] *= clip_coef

    return total_norm


def forward_and_loss(
    x_ids: np.ndarray,
    y_ids: np.ndarray,
    mask:  np.ndarray,
    lstm:  LSTM,
    dense: Dense,
    loss_fn: CrossEntropyLoss,
    vocab_size: int,
) -> tuple[float, int]:
    """
    Full forward pass + masked loss computation.
    Returns (loss_value, num_tokens_counted).
    """
    batch, seq_len = x_ids.shape

    # Encode inputs as one-hot vectors
    x_onehot = one_hot(x_ids, vocab_size)              # (batch, seq_len, vocab_size)

    # LSTM forward → all hidden states
    lstm_out = lstm.forward(x_onehot)                   # (batch, seq_len, hidden)

    # Dense forward on flattened sequence
    hidden_size = lstm_out.shape[2]
    flat = lstm_out.reshape(batch * seq_len, hidden_size)       # (B*T, hidden)
    logits_flat = dense.forward(flat)                   # (B*T, vocab_size)

    # Select only assistant-span positions for loss
    mask_flat    = mask.reshape(batch * seq_len)        # (B*T,) bool
    targets_flat = y_ids.reshape(batch * seq_len)       # (B*T,) int

    logits_masked  = logits_flat[mask_flat]             # (N_assistant, vocab_size)
    targets_masked = targets_flat[mask_flat]            # (N_assistant,)

    n_tokens = int(mask_flat.sum())
    if n_tokens == 0:
        return 0.0, 0

    loss_val = loss_fn.forward(logits_masked, targets_masked)
    return loss_val, n_tokens


def backward_pass(
    x_ids:     np.ndarray,
    mask:      np.ndarray,
    lstm:      LSTM,
    dense:     Dense,
    loss_fn:   CrossEntropyLoss,
    vocab_size: int,
) -> None:
    """
    Backprop through the masked loss → Dense → LSTM.
    Gradients for non-assistant positions are zeroed out.
    """
    batch, seq_len = x_ids.shape
    hidden_size = lstm.hidden_size

    # Gradient from loss (only for assistant-span positions)
    grad_masked = loss_fn.backward()                    # (N_assistant, vocab_size)
    mask_flat   = mask.reshape(batch * seq_len)         # (B*T,)

    # Scatter back to full flat gradient — zeros everywhere else
    grad_flat = np.zeros((batch * seq_len, vocab_size), dtype=np.float32)
    grad_flat[mask_flat] = grad_masked

    # Backprop through Dense
    grad_hidden_flat = dense.backward(grad_flat)        # (B*T, hidden)
    grad_hidden = grad_hidden_flat.reshape(batch, seq_len, hidden_size)

    # Backprop through LSTM (BPTT)
    lstm.backward(grad_hidden)                          # gradients stored in lstm.grads


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    lstm: LSTM, dense: Dense,
    epoch: int, train_loss: float, val_loss: float, perplexity: float,
    tokenizer: CharTokenizer, cfg: argparse.Namespace,
    path: Path,
) -> None:
    path.mkdir(parents=True, exist_ok=True)

    weights_path = path / "best_weights.npz"
    card_path    = path / "model_card.json"

    save_weights([lstm, dense], str(weights_path))

    card = {
        "version": "0.1",
        "saved_epoch": epoch,
        "train_loss": round(float(train_loss), 4),
        "val_loss": round(float(val_loss), 4),
        "perplexity": round(float(perplexity), 4),
        "architecture": {
            "type": "CharLM_LSTM",
            "vocab_size": tokenizer.vocab_size,
            "hidden_size": cfg.hidden_size,
        },
        "training": {
            "optimizer": "Adam",
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "max_seq_len": cfg.max_seq_len,
            "grad_clip": cfg.grad_clip,
        },
        "data": {
            "tokenizer": "datasets/tokenizer/vocab.json",
        },
    }
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: argparse.Namespace) -> None:
    print("=" * 60)
    print("TEO — LANGUAGE MODEL TRAINING")
    print("=" * 60)

    # ── Load tokenizer ──────────────────────────────────────────────────────
    if not VOCAB_PATH.exists():
        print(f"ERROR: {VOCAB_PATH} not found. Run build_vocab.py first.")
        sys.exit(1)

    tokenizer = CharTokenizer.load(str(VOCAB_PATH))
    V = tokenizer.vocab_size
    print(f"Vocab size  : {V}")

    # ── Load + split dataset ────────────────────────────────────────────────
    all_samples = load_dataset(tokenizer, cfg.max_seq_len)
    if not all_samples:
        print("ERROR: No samples loaded.")
        sys.exit(1)

    rng = np.random.default_rng(cfg.seed)
    indices = rng.permutation(len(all_samples))
    val_n   = max(1, int(len(all_samples) * cfg.val_split))
    train_i = indices[val_n:]
    val_i   = indices[:val_n]

    train_samples = [all_samples[i] for i in train_i]
    val_samples   = [all_samples[i] for i in val_i]

    print(f"Train / val : {len(train_samples)} / {len(val_samples)} samples")
    print(f"Hidden size : {cfg.hidden_size}")
    print(f"Epochs      : {cfg.epochs}  |  batch {cfg.batch_size}  |  lr {cfg.lr}")
    print()

    # ── Build model ──────────────────────────────────────────────────────────
    lstm  = LSTM(input_size=V, hidden_size=cfg.hidden_size, return_sequences=True)
    dense = Dense(cfg.hidden_size, V)            # raw logits, no activation
    if cfg.resume and WEIGHTS_PATH.exists():
        load_weights([lstm, dense], str(WEIGHTS_PATH))
        print(f"Resumed from {WEIGHTS_PATH}")
    opt   = Adam(lr=cfg.lr)
    loss_fn = CrossEntropyLoss(from_logits=True)
    layers  = [lstm, dense]

    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        rng.shuffle(train_samples)
        train_loss_sum = 0.0
        train_tokens   = 0

        for start in range(0, len(train_samples), cfg.batch_size):
            batch = train_samples[start : start + cfg.batch_size]
            x_ids, y_ids, mask = pad_batch(batch)

            # Forward
            loss_val, n_tok = forward_and_loss(
                x_ids, y_ids, mask, lstm, dense, loss_fn, V
            )
            if n_tok == 0:
                continue

            # Backward
            backward_pass(x_ids, mask, lstm, dense, loss_fn, V)

            # Gradient clip
            clip_gradients(layers, max_norm=cfg.grad_clip)

            # Update weights, zero grads
            opt.step(layers)
            for layer in layers:
                layer.zero_grad()

            train_loss_sum += loss_val * n_tok
            train_tokens   += n_tok

        train_loss = train_loss_sum / max(train_tokens, 1)

        # ── Validate ─────────────────────────────────────────────────────────
        val_loss_sum = 0.0
        val_tokens   = 0
        for start in range(0, len(val_samples), cfg.batch_size):
            batch = val_samples[start : start + cfg.batch_size]
            x_ids, y_ids, mask = pad_batch(batch)
            loss_val, n_tok = forward_and_loss(
                x_ids, y_ids, mask, lstm, dense, loss_fn, V
            )
            val_loss_sum += loss_val * n_tok
            val_tokens   += n_tok

        val_loss   = val_loss_sum / max(val_tokens, 1)
        perplexity = float(np.exp(min(val_loss, 30)))   # cap to avoid inf

        # ── Log ──────────────────────────────────────────────────────────────
        if epoch % cfg.log_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            marker  = " ← best" if val_loss < best_val_loss else ""
            print(
                f"  Epoch {epoch:4d}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"ppl={perplexity:.2f}  "
                f"[{elapsed:.0f}s]{marker}"
            )

        # ── Save best ────────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                lstm, dense, epoch, train_loss, val_loss, perplexity,
                tokenizer, cfg, CHECKPOINT_DIR
            )

        if cfg.dry_run and epoch >= 3:
            print("\n  [dry-run] Stopping after 3 epochs.")
            break

    print()
    print(f"Training complete. Best val_loss: {best_val_loss:.4f}  "
          f"ppl: {np.exp(min(best_val_loss,30)):.2f}")
    print(f"Checkpoint saved to: {CHECKPOINT_DIR}")
    print()
    print("Next: python ai/training/generate.py")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Teo's character-level language model.")
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch-size",  type=int,   default=16,   dest="batch_size")
    p.add_argument("--hidden-size", type=int,   default=128,  dest="hidden_size")
    p.add_argument("--lr",          type=float, default=3e-3)
    p.add_argument("--max-seq-len", type=int,   default=256,  dest="max_seq_len")
    p.add_argument("--val-split",   type=float, default=0.1,  dest="val_split")
    p.add_argument("--grad-clip",   type=float, default=1.0,  dest="grad_clip")
    p.add_argument("--log-every",   type=int,   default=10,   dest="log_every")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--resume",      action="store_true",
                   help="Resume from best_weights.npz if it exists.")
    p.add_argument("--dry-run",     action="store_true", dest="dry_run",
                   help="Run 3 epochs only — verify everything works before a long run.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
