import numpy as np
from nanonet.core.base import Layer

class Tanh(Layer):
    """Hyperbolic tangent: f(x) = tanh(x). Output in (-1, 1)."""
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = np.tanh(x)
        return self._out
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1.0 - self._out ** 2)
    def __repr__(self): return "Tanh()"
