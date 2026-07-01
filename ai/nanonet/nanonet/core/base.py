"""
nanonet/core/base.py
Abstract base classes for all NanoNet components.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class Layer(ABC):
    """
    Base class for every layer (Dense, Dropout, BatchNorm, activations, …).

    Subclasses store learnable parameters in ``self.params`` and their
    corresponding gradients in ``self.grads`` — both plain dicts keyed by
    string names (e.g. "W", "b").  The training flag lets layers behave
    differently at inference time.
    """

    def __init__(self) -> None:
        self.params:   Dict[str, np.ndarray] = {}
        self.grads:    Dict[str, np.ndarray] = {}
        self.training: bool = True

    # ── Core interface ──────────────────────────────────────────────────────

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run the forward pass. Cache anything needed for backward."""

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backpropagate ``grad`` (dLoss/dOutput) through this layer.

        Must:
          - Populate ``self.grads`` for every key in ``self.params``.
          - Return dLoss/dInput (same shape as the forward input ``x``).
        """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    # ── Training / eval mode ────────────────────────────────────────────────

    def train(self) -> "Layer":
        """Switch to training mode. Returns self for chaining."""
        self.training = True
        return self

    def eval(self) -> "Layer":
        """Switch to inference mode. Returns self for chaining."""
        self.training = False
        return self

    # ── Parameter helpers ───────────────────────────────────────────────────

    def named_parameters(self) -> Dict[str, np.ndarray]:
        return self.params

    def named_gradients(self) -> Dict[str, np.ndarray]:
        return self.grads

    def zero_grad(self) -> None:
        """Zero all gradients in-place."""
        for key in self.grads:
            self.grads[key] = np.zeros_like(self.grads[key])

    def num_params(self) -> int:
        """Total number of scalar parameters in this layer."""
        return sum(p.size for p in self.params.values())

    # ── Optional regularization loss ────────────────────────────────────────

    def reg_loss(self) -> float:
        """Override to contribute a regularization term to the total loss."""
        return 0.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Loss(ABC):
    """Base class for all loss functions."""

    @abstractmethod
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute scalar loss; cache state needed for backward."""

    @abstractmethod
    def backward(self) -> np.ndarray:
        """Return dLoss/dPred (same shape as pred)."""

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        return self.forward(pred, target)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Optimizer(ABC):
    """
    Base class for all optimizers.

    ``step(layers)`` receives the full list of layers in the model.
    Implementations should skip layers with empty ``params``.
    """

    def __init__(self, lr: float) -> None:
        self.lr = lr
        self._step_count: int = 0

    @abstractmethod
    def step(self, layers: List[Layer]) -> None:
        """Update parameters using the gradients already stored in layers."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self.lr})"


class Initializer(ABC):
    """Base class for weight / bias initializers."""

    @abstractmethod
    def initialize(self, shape: tuple) -> np.ndarray:
        """Return a weight array of the given shape."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Metric(ABC):
    """Base class for evaluation metrics (accuracy, R², RMSE, …)."""

    @abstractmethod
    def compute(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute and return a scalar metric value."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short display name used in training logs (e.g. 'accuracy')."""

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        return self.compute(pred, target)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Callback(ABC):
    """
    Base class for training callbacks.

    Every hook receives the ``Trainer`` instance, giving full access to
    the model, optimizer, history, and current epoch / batch index.
    Override only the hooks you need.
    """

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_begin(self, epoch: int, trainer: Any) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, float], trainer: Any) -> None:
        pass

    def on_batch_begin(self, batch: int, trainer: Any) -> None:
        pass

    def on_batch_end(self, batch: int, logs: Dict[str, float], trainer: Any) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
