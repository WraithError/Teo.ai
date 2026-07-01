import numpy as np
from nanonet.core.base import Layer

class Softmax(Layer):
    """
    Softmax over the last axis.
    When used with CrossEntropyLoss, the combined backward is just (pred - target)/N.
    Pass ``combined=True`` (default) if paired with CrossEntropyLoss.
    """
    def __init__(self, combined_with_ce: bool = True) -> None:
        super().__init__()
        self.combined_with_ce = combined_with_ce
    def forward(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        self._out = e / e.sum(axis=-1, keepdims=True)
        return self._out
    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.combined_with_ce:
            return grad          # gradient already absorbed in CrossEntropyLoss
        # Full Jacobian-vector product
        s = self._out
        return s * (grad - (grad * s).sum(axis=-1, keepdims=True))
    def __repr__(self): return "Softmax()"
