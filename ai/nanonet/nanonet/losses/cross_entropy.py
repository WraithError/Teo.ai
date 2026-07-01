import numpy as np
from nanonet.core.base import Loss

class CrossEntropyLoss(Loss):
    """
    Categorical Cross-Entropy fused with Softmax backward.

    Accepts:
      - ``pred``   : raw logits (N, C)  –or–  softmax probabilities (N, C)
      - ``target`` : class indices (N,) –or–  one-hot vectors (N, C)

    Set ``from_logits=True`` (default) to apply softmax internally.
    """
    _EPS = 1e-9

    def __init__(self, from_logits: bool = True) -> None:
        self.from_logits = from_logits

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        if self.from_logits:
            e = np.exp(pred - pred.max(axis=-1, keepdims=True))
            self._prob = e / e.sum(axis=-1, keepdims=True)
        else:
            self._prob = pred

        if target.ndim == 1:               # class indices → one-hot
            oh = np.zeros_like(self._prob)
            oh[np.arange(len(target)), target.astype(int)] = 1
            self._target = oh
        else:
            self._target = target

        return float(-np.mean(np.sum(self._target * np.log(self._prob + self._EPS), axis=-1)))

    def backward(self) -> np.ndarray:
        # Fused softmax + CE gradient = (prob - one_hot) / N
        return (self._prob - self._target) / self._prob.shape[0]
