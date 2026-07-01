import numpy as np
from nanonet.core.base import Loss

class HuberLoss(Loss):
    """Huber loss: MSE for small errors, MAE for large ones. Robust and smooth."""
    def __init__(self, delta: float = 1.0) -> None:
        self.delta = delta
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._diff = pred - target
        abs_diff = np.abs(self._diff)
        self._mask = abs_diff <= self.delta
        loss = np.where(self._mask, 0.5 * self._diff**2, self.delta * (abs_diff - 0.5 * self.delta))
        return float(np.mean(loss))
    def backward(self) -> np.ndarray:
        grad = np.where(self._mask, self._diff, self.delta * np.sign(self._diff))
        return grad / self._diff.size
