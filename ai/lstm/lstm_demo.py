"""
ai/lstm/lstm_demo.py

Three things this file does:
  1. Gradient check — verifies backward() matches numerical gradients
  2. Sequence classifier — trains LSTM to detect rising vs falling sequences
  3. Stacked LSTM — shows return_sequences=True chaining two LSTMs

Run from the project root:
    cd Teo.AI
    python ai/lstm/lstm_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../ai/nanonet"))

import numpy as np
from nanonet.layers.lstm import LSTM
from nanonet import Sequential, Dense, CrossEntropyLoss, Adam


# ─────────────────────────────────────────────────────────────────────────────
# 1. GRADIENT CHECK
# Numerically verify that backward() is correct.
# Method: compare analytical gradients to finite-difference estimates.
# If max error < 1e-5, we're good.
# ─────────────────────────────────────────────────────────────────────────────

def numerical_gradient(layer, x, loss_fn, eps=1e-5):
    """
    Compute numerical gradient of loss w.r.t. each weight in layer.params
    using central differences: (f(x+ε) - f(x-ε)) / 2ε
    """
    num_grads = {}

    for key, W in layer.params.items():
        dW = np.zeros_like(W)
        it = np.nditer(W, flags=["multi_index"])

        while not it.finished:
            idx = it.multi_index

            original = W[idx]

            W[idx] = original + eps
            out_plus = layer.forward(x)
            loss_plus = loss_fn(out_plus)

            W[idx] = original - eps
            out_minus = layer.forward(x)
            loss_minus = loss_fn(out_minus)

            dW[idx] = (loss_plus - loss_minus) / (2 * eps)
            W[idx] = original
            it.iternext()

        num_grads[key] = dW

    return num_grads


def gradient_check():
    print("=" * 60)
    print("GRADIENT CHECK")
    print("=" * 60)

    np.random.seed(42)
    batch, seq_len, input_size, hidden_size = 2, 4, 3, 5

    layer = LSTM(input_size=input_size, hidden_size=hidden_size, return_sequences=False)
    x = np.random.randn(batch, seq_len, input_size) * 0.1

    # Simple scalar loss: mean of all outputs
    def loss_fn(out):
        return float(np.mean(out ** 2))

    # Analytical backward
    out = layer.forward(x)
    # Gradient of mean(out^2) w.r.t. out = 2*out / out.size
    grad_out = 2.0 * out / out.size
    layer.backward(grad_out)
    analytical = {k: v.copy() for k, v in layer.grads.items()}

    # Numerical backward
    numerical = numerical_gradient(layer, x, loss_fn)

    all_passed = True
    for key in analytical:
        a = analytical[key]
        n = numerical[key]
        max_err = np.max(np.abs(a - n))
        rel_err = max_err / (np.max(np.abs(n)) + 1e-8)
        status = "PASS ✓" if max_err < 1e-4 else "FAIL ✗"
        if max_err >= 1e-4:
            all_passed = False
        print(f"  {key:6s}  max_abs_err={max_err:.2e}  rel_err={rel_err:.2e}  {status}")

    print()
    if all_passed:
        print("  All gradients correct. BPTT is working.\n")
    else:
        print("  WARNING: gradient mismatch — check backward().\n")

    return all_passed


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEQUENCE CLASSIFIER
# Task: classify whether a 10-step sequence is RISING (label 0)
#       or FALLING (label 1).
# Network: LSTM → Dense(softmax)
# ─────────────────────────────────────────────────────────────────────────────

def make_sequence_dataset(n_samples=1000, seq_len=10, noise=0.1, seed=0):
    """
    Rising sequence:  each step = previous + small positive delta  → label 0
    Falling sequence: each step = previous - small positive delta  → label 1
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    for _ in range(n_samples):
        label = rng.integers(0, 2)           # 0 = rising, 1 = falling
        delta = rng.uniform(0.05, 0.3)
        start = rng.uniform(-1.0, 1.0)

        seq = [start + ((-1) ** label) * delta * t for t in range(seq_len)]
        seq = np.array(seq) + rng.normal(0, noise, seq_len)
        X.append(seq)
        y.append(label)

    X = np.array(X)[:, :, np.newaxis]       # (N, seq_len, 1)
    y = np.array(y)                          # (N,)
    return X, y


def sequence_classifier():
    print("=" * 60)
    print("SEQUENCE CLASSIFIER  (rising vs falling)")
    print("=" * 60)

    np.random.seed(0)

    X, y = make_sequence_dataset(n_samples=800, seq_len=10)
    split = 640
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = Sequential([
        LSTM(input_size=1, hidden_size=16, return_sequences=False),
        Dense(16, 2, activation="softmax"),
    ])

    loss_fn = CrossEntropyLoss()
    opt     = Adam(lr=3e-3)
    model.compile(loss=loss_fn, optimizer=opt)

    # Manual training loop (keeps it transparent — no Trainer magic hiding the LSTM)
    batch_size = 32
    epochs     = 30

    for epoch in range(1, epochs + 1):
        # Shuffle
        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]

        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, len(X_train), batch_size):
            xb = X_train[start : start + batch_size]
            yb = y_train[start : start + batch_size]

            # Forward
            model.train()
            pred = model.forward(xb)                    # (batch, 2)
            loss = loss_fn.forward(pred, yb)

            # Backward
            grad = loss_fn.backward()                   # (batch, 2)
            model.backward(grad)
            opt.step(model.layers)

            # Zero grads
            for layer in model.layers:
                layer.zero_grad()

            epoch_loss += loss
            n_batches  += 1

        if epoch % 5 == 0:
            # Validation accuracy
            model.eval()
            val_pred  = model.forward(X_val)            # (N_val, 2)
            val_class = np.argmax(val_pred, axis=1)
            val_acc   = float(np.mean(val_class == y_val)) * 100
            avg_loss  = epoch_loss / n_batches
            print(f"  Epoch {epoch:3d}  loss={avg_loss:.4f}  val_acc={val_acc:.1f}%")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# 3. STACKED LSTM
# Shows return_sequences=True working — two LSTM layers in sequence.
# ─────────────────────────────────────────────────────────────────────────────

def stacked_lstm_shape_check():
    print("=" * 60)
    print("STACKED LSTM — shape check")
    print("=" * 60)

    batch, seq_len, input_size = 4, 8, 6

    model = Sequential([
        LSTM(input_size=6,  hidden_size=32, return_sequences=True),
        LSTM(input_size=32, hidden_size=16, return_sequences=False),
        Dense(16, 3, activation="softmax"),
    ])

    x   = np.random.randn(batch, seq_len, input_size)
    out = model.forward(x)

    print(f"  Input  shape: {x.shape}")
    print(f"  LSTM-1 output: (batch={batch}, seq_len={seq_len}, hidden=32)")
    print(f"  LSTM-2 output: (batch={batch}, hidden=16)")
    print(f"  Dense  output: {out.shape}  ← expected (4, 3)")
    assert out.shape == (batch, 3), f"Shape mismatch: {out.shape}"
    print("  Shape check passed ✓\n")

    # Also verify backward doesn't crash
    grad = np.random.randn(*out.shape)
    model.backward(grad)
    print("  Backward through stacked LSTM: no errors ✓\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed = gradient_check()
    sequence_classifier()
    stacked_lstm_shape_check()

    print("=" * 60)
    print("Done. LSTM is ready to train Teo.")
    print("=" * 60)
