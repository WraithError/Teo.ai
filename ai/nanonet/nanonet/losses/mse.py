import numpy as np
from nanonet.core.base import Loss

class MSELoss(Loss):
    """Mean Squared Error: mean((pred - target)^2). Best for regression."""
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._diff = pred - target
        return float(np.mean(self._diff ** 2))
    def backward(self) -> np.ndarray:
        return 2.0 * self._diff / self._diff.size
