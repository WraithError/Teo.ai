"""
ai/training/train_language_model.py

Character-level LSTM language model training for Teo.
Changes in this version vs previous:
  #1 — uses nanonet.backend (xp) instead of raw numpy import
  #3 — Embedding layer replaces one-hot encoding
  #5 — full checkpoint state: optimizer m/v/step, loss history, epoch
  #6 — per-epoch timing: forward / backward / optimizer / total

Architecture:
    token_ids  (batch, seq_len)
        → Embedding(vocab_size, embed_dim)   (batch, seq_len, embed_dim)
        → LSTM(embed_dim, hidden_size)       (batch, seq_len, hidden_size)
        → Dense(hidden_size, vocab_size)     (batch*seq_len, vocab_size)
        → CrossEntropyLoss [assistant span only]

Usage:
    cd Teo.AI
    python ai/training/train_language_model.py
    python ai/training/train_language_model.py --epochs 300 --embed-dim 64
    python ai/training/train_language_model.py --resume   # exact resume
    python ai/training/train_language_model.py --dry-run  # 3 epochs, verify
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np   # always numpy for data loading/IO ops
import numpy as _np  # alias so we can keep np for IO

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "ai" / "nanonet"))
sys.path.insert(0, str(ROOT / "ai" / "tokenizer"))

from nanonet.backend import xp, GPU, BACKEND
from nanonet.layers.embedding import Embedding
from nanonet.layers.lstm      import LSTM
from nanonet.layers.dense     import Dense
from nanonet.optimizers.adam  import Adam
from nanonet.losses.cross_entropy import CrossEntropyLoss
from nanonet.utils.io import save_weights, load_weights
from tokenizer import CharTokenizer

VOCAB_PATH     = ROOT / "datasets" / "tokenizer" / "vocab.json"
DATASET_DIR    = ROOT / "datasets" / "train"
CHECKPOINT_DIR = ROOT / "ai" / "training" / "checkpoints"
WEIGHTS_PATH   = CHECKPOINT_DIR / "best_weights.npz"
OPT_PATH       = CHECKPOINT_DIR / "optimizer_state.npz"   # #5: optimizer state
CARD_PATH      = CHECKPOINT_DIR / "model_card.json"


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

class Sample(NamedTuple):
    x:    np.ndarray   # (seq_len,) int
    y:    np.ndarray   # (seq_len,) int
    mask: np.ndarray   # (seq_len,) bool — True = compute loss here


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
                ids, sep_index = tokenizer.encode_exchange(d["user"], d["assistant"])
                ids = ids[:max_seq_len + 1]
                if len(ids) < 3:
                    continue
                x    = _np.array(ids[:-1], dtype=_np.int32)
                y    = _np.array(ids[1:],  dtype=_np.int32)
                mask = _np.zeros(len(x), dtype=bool)
                mask[sep_index:] = True
                samples.append(Sample(x=x, y=y, mask=mask))
    return samples


def pad_batch(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_len = max(s.x.shape[0] for s in samples)
    B = len(samples)
    x_batch    = _np.zeros((B, max_len), dtype=_np.int32)
    y_batch    = _np.zeros((B, max_len), dtype=_np.int32)
    mask_batch = _np.zeros((B, max_len), dtype=bool)
    for i, s in enumerate(samples):
        L = s.x.shape[0]
        x_batch[i, :L]    = s.x
        y_batch[i, :L]    = s.y
        mask_batch[i, :L] = s.mask
    return x_batch, y_batch, mask_batch


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def clip_gradients(layers: list, max_norm: float = 1.0) -> float:
    """L2 global norm clipping. Returns pre-clip norm."""
    total_sq = 0.0
    for layer in layers:
        for g in layer.grads.values():
            total_sq += float(xp.sum(g ** 2))
    total_norm = float(xp.sqrt(total_sq))
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for layer in layers:
            for key in layer.grads:
                layer.grads[key] *= clip_coef
    return total_norm


def forward_and_loss(
    x_ids: np.ndarray, y_ids: np.ndarray, mask: np.ndarray,
    embed: Embedding, lstm: LSTM, dense: Dense,
    loss_fn: CrossEntropyLoss, vocab_size: int,
) -> tuple[float, int]:
    """Full forward pass with Embedding + LSTM + Dense + masked loss."""
    batch, seq_len = x_ids.shape

    # Move to xp (no-op on CPU; copies to GPU on CuPy)
    x_xp    = xp.array(x_ids)
    y_xp    = xp.array(y_ids)
    mask_xp = xp.array(mask)

    x_embed  = embed.forward(x_xp)                          # (B, T, embed_dim)
    lstm_out = lstm.forward(x_embed)                         # (B, T, hidden)

    hidden_size = lstm_out.shape[2]
    flat        = lstm_out.reshape(batch * seq_len, hidden_size)
    logits_flat = dense.forward(flat)                        # (B*T, V)

    mask_flat    = mask_xp.reshape(batch * seq_len)
    targets_flat = y_xp.reshape(batch * seq_len)

    n_tokens = int(mask_flat.sum())
    if n_tokens == 0:
        return 0.0, 0

    loss_val = loss_fn.forward(
        logits_flat[mask_flat],
        targets_flat[mask_flat],
    )
    return float(loss_val), n_tokens


def backward_pass(
    x_ids: np.ndarray, mask: np.ndarray,
    embed: Embedding, lstm: LSTM, dense: Dense,
    loss_fn: CrossEntropyLoss, vocab_size: int,
) -> None:
    """Backward: loss → Dense → LSTM → Embedding."""
    batch, seq_len = x_ids.shape
    hidden_size = lstm.hidden_size

    mask_xp   = xp.array(mask)
    mask_flat = mask_xp.reshape(batch * seq_len)

    grad_masked = loss_fn.backward()                         # (N_asst, V)
    grad_flat   = xp.zeros((batch * seq_len, vocab_size), dtype=xp.float32)
    grad_flat[mask_flat] = grad_masked

    grad_hidden_flat = dense.backward(grad_flat)             # (B*T, hidden)
    grad_hidden      = grad_hidden_flat.reshape(batch, seq_len, hidden_size)

    dx_embed = lstm.backward(grad_hidden)                    # (B, T, embed_dim)
    embed.backward(dx_embed)                                 # updates embed.grads["W"]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O  (#5 — full state)
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    embed: Embedding, lstm: LSTM, dense: Dense, opt: Adam,
    epoch: int, train_loss: float, val_loss: float, perplexity: float,
    loss_history: list[dict], tokenizer: CharTokenizer, cfg: argparse.Namespace,
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Weights
    save_weights([embed, lstm, dense], str(WEIGHTS_PATH))

    # Optimizer state (#5)
    opt_data: dict = {"step_count": _np.array(opt._step_count)}
    for i, layer_m in opt._m.items():
        for key, arr in layer_m.items():
            opt_data[f"m_{i}_{key}"] = _np.array(arr)
    for i, layer_v in opt._v.items():
        for key, arr in layer_v.items():
            opt_data[f"v_{i}_{key}"] = _np.array(arr)
    _np.savez(str(OPT_PATH), **opt_data)

    # Model card
    card = {
        "version": "0.2",
        "saved_epoch": epoch,
        "train_loss": round(float(train_loss), 4),
        "val_loss": round(float(val_loss), 4),
        "perplexity": round(float(perplexity), 4),
        "architecture": {
            "type": "CharLM_Embedding_LSTM",
            "vocab_size": tokenizer.vocab_size,
            "embed_dim": cfg.embed_dim,
            "hidden_size": cfg.hidden_size,
        },
        "training": {
            "optimizer": "Adam",
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "max_seq_len": cfg.max_seq_len,
            "grad_clip": cfg.grad_clip,
            "backend": BACKEND,
        },
        "data": {"tokenizer": "datasets/tokenizer/vocab.json"},
        "loss_history": loss_history[-50:],  # keep last 50 entries
    }
    with open(CARD_PATH, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)


def load_checkpoint(embed: Embedding, lstm: LSTM, dense: Dense, opt: Adam) -> int:
    """
    Load weights + optimizer state. Returns the saved epoch number.
    Returns 0 if checkpoint architecture doesn't match (fresh start).
    """
    if not WEIGHTS_PATH.exists() or not CARD_PATH.exists():
        return 0

    with open(CARD_PATH, encoding="utf-8") as f:
        card = json.load(f)

    arch = card.get("architecture", {})

    # Architecture mismatch check (one-hot → embedding migration)
    if arch.get("type") == "CharLM_LSTM":
        print("  WARNING: checkpoint is from old one-hot architecture.")
        print("  Cannot resume — starting fresh with Embedding architecture.")
        return 0

    if (arch.get("embed_dim") != embed.embed_dim or
        arch.get("hidden_size") != lstm.hidden_size or
        arch.get("vocab_size") != embed.vocab_size):
        print("  WARNING: checkpoint architecture mismatch — starting fresh.")
        return 0

    load_weights([embed, lstm, dense], str(WEIGHTS_PATH))
    print(f"  Weights loaded from epoch {card['saved_epoch']}")

    # Restore optimizer state (#5)
    if OPT_PATH.exists():
        opt_data   = _np.load(str(OPT_PATH), allow_pickle=False)
        opt._step_count = int(opt_data["step_count"])
        import re
        for key in opt_data.files:
            m = re.match(r"^m_(\d+)_(.+)$", key)
            if m:
                i, k = int(m.group(1)), m.group(2)
                if i not in opt._m:
                    opt._m[i] = {}
                opt._m[i][k] = xp.array(opt_data[key])
            v = re.match(r"^v_(\d+)_(.+)$", key)
            if v:
                i, k = int(v.group(1)), v.group(2)
                if i not in opt._v:
                    opt._v[i] = {}
                opt._v[i][k] = xp.array(opt_data[key])
        print(f"  Optimizer state restored (step {opt._step_count})")

    return card.get("saved_epoch", 0)


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: argparse.Namespace) -> None:
    print("=" * 60)
    print("TEO — LANGUAGE MODEL TRAINING")
    print(f"  Backend : {BACKEND}  ({'GPU' if GPU else 'CPU'})")
    print("=" * 60)

    if not VOCAB_PATH.exists():
        print(f"ERROR: {VOCAB_PATH} not found. Run build_vocab.py first.")
        sys.exit(1)

    tokenizer = CharTokenizer.load(str(VOCAB_PATH))
    V = tokenizer.vocab_size
    print(f"Vocab size  : {V}")

    all_samples = load_dataset(tokenizer, cfg.max_seq_len)
    if not all_samples:
        print("ERROR: No samples loaded.")
        sys.exit(1)

    rng = _np.random.default_rng(cfg.seed)
    indices = rng.permutation(len(all_samples))
    val_n   = max(1, int(len(all_samples) * cfg.val_split))
    train_samples = [all_samples[i] for i in indices[val_n:]]
    val_samples   = [all_samples[i] for i in indices[:val_n]]

    print(f"Train / val : {len(train_samples)} / {len(val_samples)} samples")
    print(f"Architecture: Embedding({V}→{cfg.embed_dim}) → LSTM({cfg.embed_dim}→{cfg.hidden_size}) → Dense({cfg.hidden_size}→{V})")
    print(f"Epochs      : {cfg.epochs}  |  batch {cfg.batch_size}  |  lr {cfg.lr}")
    print()

    # Build model
    embed   = Embedding(V, cfg.embed_dim)
    lstm    = LSTM(input_size=cfg.embed_dim, hidden_size=cfg.hidden_size, return_sequences=True)
    dense   = Dense(cfg.hidden_size, V)
    opt     = Adam(lr=cfg.lr)
    loss_fn = CrossEntropyLoss(from_logits=True)
    layers  = [embed, lstm, dense]

    start_epoch = 0
    if cfg.resume:
        start_epoch = load_checkpoint(embed, lstm, dense, opt)

    best_val_loss  = float("inf")
    loss_history: list[dict] = []

    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        rng.shuffle(train_samples)
        train_loss_sum = 0.0
        train_tokens   = 0

        # ── #6: Timing accumulators ─────────────────────────────────────────
        t_fwd = t_bwd = t_opt = 0.0

        for start in range(0, len(train_samples), cfg.batch_size):
            batch = train_samples[start : start + cfg.batch_size]
            x_ids, y_ids, mask = pad_batch(batch)

            # Forward  (#6)
            t0 = time.perf_counter()
            loss_val, n_tok = forward_and_loss(
                x_ids, y_ids, mask, embed, lstm, dense, loss_fn, V
            )
            t_fwd += time.perf_counter() - t0
            if n_tok == 0:
                continue

            # Backward  (#6)
            t0 = time.perf_counter()
            backward_pass(x_ids, mask, embed, lstm, dense, loss_fn, V)
            t_bwd += time.perf_counter() - t0

            clip_gradients(layers, max_norm=cfg.grad_clip)

            # Optimizer  (#6)
            t0 = time.perf_counter()
            opt.step(layers)
            t_opt += time.perf_counter() - t0

            for layer in layers:
                layer.zero_grad()

            train_loss_sum += loss_val * n_tok
            train_tokens   += n_tok

        train_loss = train_loss_sum / max(train_tokens, 1)

        # Validation
        val_loss_sum = 0.0
        val_tokens   = 0
        for start in range(0, len(val_samples), cfg.batch_size):
            batch = val_samples[start : start + cfg.batch_size]
            x_ids, y_ids, mask = pad_batch(batch)
            loss_val, n_tok = forward_and_loss(
                x_ids, y_ids, mask, embed, lstm, dense, loss_fn, V
            )
            val_loss_sum += loss_val * n_tok
            val_tokens   += n_tok

        val_loss   = val_loss_sum / max(val_tokens, 1)
        perplexity = float(_np.exp(min(val_loss, 30)))

        # Loss history  (#5)
        loss_history.append({
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "val_loss": round(float(val_loss), 4),
            "ppl": round(perplexity, 4),
        })

        # Log  (#6: timing breakdown)
        if epoch % cfg.log_every == 0 or epoch == 1:
            t_total = t_fwd + t_bwd + t_opt
            marker  = " ← best" if val_loss < best_val_loss else ""
            print(
                f"  Epoch {epoch:4d}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  ppl={perplexity:.2f}"
                f"{marker}"
            )
            print(
                f"           fwd={t_fwd*1000:.0f}ms  "
                f"bwd={t_bwd*1000:.0f}ms  "
                f"opt={t_opt*1000:.0f}ms  "
                f"total={t_total*1000:.0f}ms"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                embed, lstm, dense, opt, epoch,
                train_loss, val_loss, perplexity,
                loss_history, tokenizer, cfg,
            )

        if cfg.dry_run and epoch >= start_epoch + 3:
            print("\n  [dry-run] Stopping after 3 epochs.")
            break

    print()
    print(f"Done. Best val_loss: {best_val_loss:.4f}  ppl: {_np.exp(min(best_val_loss,30)):.2f}")
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print("Next: python ai/training/generate.py")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Teo's character-level language model.")
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--embed-dim",   type=int,   default=64,   dest="embed_dim")
    p.add_argument("--hidden-size", type=int,   default=128,  dest="hidden_size")
    p.add_argument("--batch-size",  type=int,   default=16,   dest="batch_size")
    p.add_argument("--lr",          type=float, default=3e-3)
    p.add_argument("--max-seq-len", type=int,   default=256,  dest="max_seq_len")
    p.add_argument("--val-split",   type=float, default=0.1,  dest="val_split")
    p.add_argument("--grad-clip",   type=float, default=1.0,  dest="grad_clip")
    p.add_argument("--log-every",   type=int,   default=10,   dest="log_every")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--resume",      action="store_true",
                   help="Resume from best_weights.npz (architecture must match).")
    p.add_argument("--dry-run",     action="store_true", dest="dry_run",
                   help="Run 3 epochs only — verify pipeline before overnight run.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
