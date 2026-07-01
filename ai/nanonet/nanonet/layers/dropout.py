"""
nanonet/layers/dropout.py
Dropout regularization layer.
"""

from __future__ import annotations
import numpy as np
from nanonet.core.base import Layer


class Dropout(Layer):
    """
    Randomly zeros activations during training with probability ``p``.
    Scales remaining activations by 1/(1-p) so the expected sum is unchanged
    (inverted dropout).  Has no effect during evaluation.

    Parameters
    ----------
    p : float
        Drop probability in [0, 1).

    Example
    -------
    >>> model.add(Dense(256, 128, activation="relu"))
    >>> model.add(Dropout(0.3))
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Drop probability must be in [0, 1), got {p}.")
        self.p = p
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            return x
        self._mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
        return x * self._mask / (1.0 - self.p)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0 or self._mask is None:
            return grad
        return grad * self._mask / (1.0 - self.p)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"
