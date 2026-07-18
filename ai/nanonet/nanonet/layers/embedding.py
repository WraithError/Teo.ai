"""
nanonet/layers/embedding.py
----------------------------
Learned character embedding layer.

Replaces one-hot encoding. Instead of a sparse 98-dimensional vector
(98 zeros with a single 1), each token becomes a dense 64-dimensional
vector that the model learns during training.

Why this matters:
    one-hot: vocab_size=98 → 98 floats per token, mostly zeros, no
             learned similarity between characters. 'a' and 'e' look
             completely unrelated to the model.
    Embedding: embed_dim=64 → 64 floats per token, dense, and 'a'/'e'
             can end up close together in embedding space because they
             appear in similar contexts. Better representations, faster
             learning, less memory.

Input:  (batch, seq_len)         integer token ids
Output: (batch, seq_len, embed_dim)  float32 embeddings

Backward:
    Gradients accumulate into the embedding table using scatter-add
    (xp.add.at). Positions that weren't used in this batch get zero
    gradient — their embeddings don't change this step.
"""

from __future__ import annotations
import sys
import os

# Allow import from either inside the package or from training scripts
try:
    from nanonet.backend import xp
    from nanonet.core.base import Layer
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend import xp
    from core.base import Layer


class Embedding(Layer):
    """
    Lookup table that maps integer token ids to dense float vectors.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary (size of the lookup table).
    embed_dim  : int
        Dimensionality of each embedding vector. 64 is a good default
        for character-level models with vocab < 200.
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim

        # Small random init — large values slow convergence
        scale = 0.05
        self.params = {
            "W": (xp.random.randn(vocab_size, embed_dim) * scale).astype(xp.float32)
        }
        self.grads = {
            "W": xp.zeros((vocab_size, embed_dim), dtype=xp.float32)
        }

        self._cache_ids = None   # saved during forward, used in backward

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, ids: xp.ndarray) -> xp.ndarray:
        """
        ids : (batch, seq_len) int32/int64
        out : (batch, seq_len, embed_dim) float32
        """
        self._cache_ids = ids
        return self.params["W"][ids]

    # ── Backward ────────────────────────────────────────────────────────────

    def backward(self, dout: xp.ndarray) -> None:
        """
        dout : (batch, seq_len, embed_dim) float32
               upstream gradient from the LSTM layer

        Updates self.grads["W"] via scatter-add. Returns None because
        integer indices have no meaningful gradient.
        """
        self.grads["W"][:] = 0.0
        xp.add.at(self.grads["W"], self._cache_ids, dout)
        return None   # no dx — can't differentiate w.r.t. integer indices

    # ── Utilities ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"Embedding(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"
