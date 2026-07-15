import numpy as np
from nanonet.core.base import Loss

class MAELoss(Loss):
    """Mean Absolute Error: mean(|pred - target|). Robust to outliers."""
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._diff = pred - target
        return float(np.mean(np.abs(self._diff)))
    def backward(self) -> np.ndarray:
        return np.sign(self._diff) / self._diff.size
