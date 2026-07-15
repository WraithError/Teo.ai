import numpy as np
from nanonet.core.base import Loss

class BCELoss(Loss):
    """Binary Cross-Entropy. Use with Sigmoid output for binary classification."""
    _EPS = 1e-9
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._pred, self._target = pred, target
        p = np.clip(pred, self._EPS, 1 - self._EPS)
        return float(-np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))
    def backward(self) -> np.ndarray:
        p = np.clip(self._pred, self._EPS, 1 - self._EPS)
        t = self._target
        return (-(t / p) + (1 - t) / (1 - p)) / p.shape[0]
