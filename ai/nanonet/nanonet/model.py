"""
nanonet/model.py
Sequential model — stack layers, compile, predict, save/load.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional
from nanonet.core.base import Layer, Loss, Optimizer, Metric
from nanonet.utils.io import save_weights, load_weights


class Sequential:
    """
    A sequential stack of layers.

    Build the model by passing a list of layers or using ``add()``,
    then call ``compile()`` to attach a loss, optimizer, and metrics.

    Example
    -------
    >>> model = Sequential([
    ...     Dense(784, 256, activation="relu"),
    ...     Dropout(0.3),
    ...     Dense(256, 10, activation="softmax"),
    ... ])
    >>> model.compile(loss=CrossEntropyLoss(), optimizer=Adam(lr=1e-3),
    ...               metrics=[Accuracy()])
    """

    def __init__(self, layers: Optional[List[Layer]] = None) -> None:
        self.layers: List[Layer] = list(layers) if layers else []
        self._loss:      Optional[Loss]      = None
        self._optimizer: Optional[Optimizer] = None
        self._metrics:   List[Metric]        = []

    # ── Building ─────────────────────────────────────────────────────────────

    def add(self, layer: Layer) -> "Sequential":
        """Append a layer and return self for chaining."""
        self.layers.append(layer)
        return self

    def compile(
        self,
        loss: Loss,
        optimizer: Optimizer,
        metrics: Optional[List[Metric]] = None,
    ) -> "Sequential":
        """Attach loss, optimizer and optional metrics. Returns self."""
        self._loss      = loss
        self._optimizer = optimizer
        self._metrics   = metrics or []
        return self

    # ── Forward / Backward ───────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    # ── Training mode ────────────────────────────────────────────────────────

    def train(self) -> "Sequential":
        for layer in self.layers:
            layer.train()
        return self

    def eval(self) -> "Sequential":
        for layer in self.layers:
            layer.eval()
        return self

    # ── Inference ────────────────────────────────────────────────────────────

    def _set_mode(self, training: bool) -> None:
        for layer in self.layers:
            layer.training = training

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Run inference in batches. Returns the full prediction array."""
        self._set_mode(False)
        preds = []
        for start in range(0, len(X), batch_size):
            preds.append(self.forward(X[start:start + batch_size]))
        self._set_mode(True)
        return np.concatenate(preds, axis=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Compute loss and all compiled metrics on (X, y). Returns a dict."""
        self._set_mode(False)
        pred  = self.forward(X)
        loss  = self._loss(pred, y) if self._loss else None
        self._set_mode(True)
        result = {}
        if loss is not None:
            result["loss"] = loss
        for m in self._metrics:
            result[m.name] = m.compute(pred, y)
        return result

    # ── Parameter access ─────────────────────────────────────────────────────

    def get_weights(self) -> dict:
        """Return a snapshot of all parameters (deep copy)."""
        return {
            f"layer{i}_{k}": v.copy()
            for i, layer in enumerate(self.layers)
            for k, v in layer.params.items()
        }

    def set_weights(self, weights: dict) -> None:
        """Restore parameters from a snapshot produced by ``get_weights()``."""
        for i, layer in enumerate(self.layers):
            for k in layer.params:
                key = f"layer{i}_{k}"
                if key in weights:
                    layer.params[k] = weights[key].copy()

    def num_params(self) -> int:
        return sum(layer.num_params() for layer in self.layers)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = "model.npz") -> None:
        save_weights(self.layers, path)

    def load(self, path: str = "model.npz") -> None:
        load_weights(self.layers, path)

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a table of layers and parameter counts."""
        line = "─" * 52
        print(f"\n{'Model Summary':^52}")
        print(line)
        print(f"  {'Layer':<30} {'Params':>10}")
        print(line)
        total = 0
        for layer in self.layers:
            n = layer.num_params()
            total += n
            print(f"  {repr(layer):<30} {n:>10,}")
        print(line)
        print(f"  {'Total parameters':<30} {total:>10,}")
        print(line + "\n")
