import numpy as np
from nanonet.core.base import Metric

class Accuracy(Metric):
    """
    Classification accuracy.

    - Binary  : pred shape (N,1) or (N,), threshold = 0.5
    - Multi   : pred shape (N,C), takes argmax
    """
    @property
    def name(self) -> str: return "accuracy"

    def compute(self, pred: np.ndarray, target: np.ndarray) -> float:
        if pred.ndim == 2 and pred.shape[1] > 1:
            p = pred.argmax(axis=1)
            t = target.argmax(axis=1) if target.ndim == 2 else target.astype(int)
        else:
            p = (pred.ravel() >= 0.5).astype(int)
            t = target.ravel().astype(int)
        return float(np.mean(p == t))
