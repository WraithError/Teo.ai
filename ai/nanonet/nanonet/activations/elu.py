import numpy as np
from nanonet.core.base import Layer

class ELU(Layer):
    """Exponential Linear Unit: f(x) = x if x>0 else alpha*(exp(x)-1)."""
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return np.where(x > 0, x, self.alpha * (np.exp(np.clip(x, -500, 0)) - 1))
    def backward(self, grad: np.ndarray) -> np.ndarray:
        dact = np.where(self._x > 0, 1.0, self.alpha * np.exp(np.clip(self._x, -500, 0)))
        return grad * dact
    def __repr__(self): return f"ELU(alpha={self.alpha})"
