"""
tools/test_teo.py

Minimal test suite. No pytest needed — just Python.
Covers the four critical contracts: nanonet, tokenizer, dataset, API route.

Usage:
    cd Teo.AI
    python tools/test_teo.py
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai" / "nanonet"))
sys.path.insert(0, str(ROOT / "ai" / "tokenizer"))
sys.path.insert(0, str(ROOT / "app" / "backend"))

PASS = "PASS ✓"
FAIL = "FAIL ✗"

results: list[tuple[str, bool, str]] = []


def test(name: str):
    """Decorator that runs a test function and records the result."""
    def decorator(fn):
        try:
            fn()
            results.append((name, True, ""))
        except Exception:
            results.append((name, False, traceback.format_exc()))
        return fn
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: LSTM forward shape + gradient check
# ─────────────────────────────────────────────────────────────────────────────

@test("LSTM: forward shape + gradient check")
def test_lstm():
    from nanonet.layers.lstm import LSTM

    np.random.seed(0)
    batch, seq_len, input_size, hidden_size = 2, 5, 10, 16

    layer = LSTM(input_size=input_size, hidden_size=hidden_size, return_sequences=True)
    x = np.random.randn(batch, seq_len, input_size) * 0.1

    # Shape check
    out = layer.forward(x)
    assert out.shape == (batch, seq_len, hidden_size), \
        f"Expected {(batch, seq_len, hidden_size)}, got {out.shape}"

    # Gradient check (simplified — just verify backward doesn't crash and
    # produces correct-shaped gradients)
    grad_out = np.ones_like(out) * 0.01
    dx = layer.backward(grad_out)
    assert dx.shape == x.shape, \
        f"dx shape {dx.shape} != x shape {x.shape}"
    assert "W_x" in layer.grads and layer.grads["W_x"].shape == layer.params["W_x"].shape
    assert "W_h" in layer.grads and layer.grads["W_h"].shape == layer.params["W_h"].shape


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Tokenizer roundtrip + special token positions
# ─────────────────────────────────────────────────────────────────────────────

@test("Tokenizer: build, encode, decode, exchange format")
def test_tokenizer():
    from tokenizer import CharTokenizer, PAD, UNK, START, EOS, SEP

    tok = CharTokenizer.build_from_texts([
        "Accepted! Let me check that. Hello world. Salom. code.py 123"
    ])

    # Special tokens at the expected positions
    assert tok.char_to_id["<PAD>"]   == PAD
    assert tok.char_to_id["<UNK>"]   == UNK
    assert tok.char_to_id["<START>"] == START
    assert tok.char_to_id["<EOS>"]   == EOS
    assert tok.char_to_id["<SEP>"]   == SEP

    # Roundtrip
    text = "Accepted! Let me check that."
    ids  = tok.encode_chars(text)
    out  = tok.decode(ids, strip_special=False)
    assert out == text, f"Roundtrip failed: {out!r} != {text!r}"

    # Exchange format
    ids_ex, sep_idx = tok.encode_exchange("help", "Sure.")
    assert ids_ex[0]       == START
    assert ids_ex[sep_idx] == SEP
    assert ids_ex[-1]      == EOS

    # UNK for unseen character
    unk_ids = tok.encode_chars("™")
    assert unk_ids[0] == UNK


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Dataset schema integrity
# ─────────────────────────────────────────────────────────────────────────────

@test("Dataset: schema valid, no duplicate IDs, all required fields")
def test_dataset():
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "validate_dataset.py")],
        capture_output=True, text=True
    )
    assert result.returncode == 0, \
        f"validate_dataset.py failed:\n{result.stdout}\n{result.stderr}"
    assert "All samples valid" in result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: teo_engine — get_response is callable and returns a non-empty string
# ─────────────────────────────────────────────────────────────────────────────

@test("API: teo_engine.get_response returns a string")
def test_engine():
    from core.teo_engine import get_response

    response = get_response("Hello Teo")
    assert isinstance(response, str), f"Expected str, got {type(response)}"
    assert len(response) > 0, "Response was empty"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: One-hot encoding dimensions
# ─────────────────────────────────────────────────────────────────────────────

@test("Training utils: one_hot produces correct shape and sum")
def test_one_hot():
    sys.path.insert(0, str(ROOT / "ai" / "training"))
    from train_language_model import one_hot

    ids = np.array([[1, 2, 3], [4, 0, 1]], dtype=np.int32)  # (2, 3)
    V   = 10
    out = one_hot(ids, V)

    assert out.shape == (2, 3, 10), f"Shape mismatch: {out.shape}"
    # Each row should be a valid one-hot vector
    assert np.allclose(out.sum(axis=-1), 1.0), "One-hot rows don't sum to 1"
    # Correct position should be 1
    assert out[0, 0, 1] == 1.0, "Wrong position set"
    assert out[1, 2, 1] == 1.0, "Wrong position set"


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("TEO TEST SUITE")
    print("=" * 60)

    for name, passed, tb in results:
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")
        if not passed and tb:
            # Print just the last line of the traceback for brevity
            lines = [l for l in tb.strip().splitlines() if l.strip()]
            print(f"         → {lines[-1]}")

    print()
    total  = len(results)
    passed = sum(1 for _, p, _ in results if p)
    print(f"  {passed}/{total} passed")

    failed = [(n, t) for n, p, t in results if not p]
    if failed:
        print()
        print("  Full tracebacks:")
        for name, tb in failed:
            print(f"\n  [{name}]")
            print(tb)
        return 1

    print()
    print("  All tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
