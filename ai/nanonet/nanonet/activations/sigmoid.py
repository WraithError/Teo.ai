import numpy as np
from nanonet.core.base import Layer

class Sigmoid(Layer):
    """Sigmoid: f(x) = 1 / (1 + exp(-x)). Output in (0, 1)."""
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self._out
    def backward(self, grad: np.ndarray) -> np.ndarray:
        s = self._out
        return grad * s * (1.0 - s)
    def __repr__(self): return "Sigmoid()"
