import numpy as np
from nanonet.core.base import Layer

class ReLU(Layer):
    """Rectified Linear Unit: f(x) = max(0, x)."""
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._mask
    def __repr__(self): return "ReLU()"

class LeakyReLU(Layer):
    """Leaky ReLU: f(x) = x if x > 0 else alpha * x."""
    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return np.where(x > 0, x, self.alpha * x)
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * np.where(self._x > 0, 1.0, self.alpha)
    def __repr__(self): return f"LeakyReLU(alpha={self.alpha})"
