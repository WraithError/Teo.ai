"""
nanonet/layers/dense.py
Fully-connected (linear) layer with optional activation and regularization.
"""

from __future__ import annotations
import numpy as np
from nanonet.core.base import Layer
from nanonet.initializers import get_initializer


class Dense(Layer):
    """
    Fully-connected layer: output = activation(x @ W + b)

    Parameters
    ----------
    in_features  : int
        Number of input features.
    out_features : int
        Number of output neurons.
    activation   : str | Layer | None
        Activation to apply after the linear transform.
        Accepts a string key (``"relu"``, ``"sigmoid"``, …) or a Layer instance.
    initializer  : str | Initializer
        Weight initializer. Default ``"he"`` (good for ReLU).
    use_bias     : bool
        Whether to add a bias term.
    l1           : float
        L1 regularization coefficient on weights.
    l2           : float
        L2 regularization coefficient on weights.

    Examples
    --------
    >>> Dense(128, 64, activation="relu", l2=1e-4)
    >>> Dense(64, 10, activation=Softmax())
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation=None,
        initializer: str = "he",
        use_bias: bool = True,
        l1: float = 0.0,
        l2: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.use_bias     = use_bias
        self.l1           = l1
        self.l2           = l2

        # ── Weight initialization ───────────────────────────────────────────
        init = get_initializer(initializer)
        self.params["W"] = init.initialize((in_features, out_features))
        self.grads["W"]  = np.zeros_like(self.params["W"])

        if use_bias:
            self.params["b"] = np.zeros((1, out_features))
            self.grads["b"]  = np.zeros_like(self.params["b"])

        # ── Optional activation ─────────────────────────────────────────────
        if activation is not None:
            from nanonet.activations import get_activation
            self._activation = get_activation(activation)
        else:
            self._activation = None

        self._cache: dict = {}

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache["x"] = x
        z = x @ self.params["W"]
        if self.use_bias:
            z = z + self.params["b"]
        self._cache["z"] = z

        if self._activation is not None:
            return self._activation.forward(z)
        return z

    # ── Backward ────────────────────────────────────────────────────────────

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Backprop through activation
        if self._activation is not None:
            grad = self._activation.backward(grad)

        x = self._cache["x"]

        # Parameter gradients
        self.grads["W"] = x.T @ grad

        # Regularization gradients
        if self.l2 > 0.0:
            self.grads["W"] = self.grads["W"] + self.l2 * self.params["W"]
        if self.l1 > 0.0:
            self.grads["W"] = self.grads["W"] + self.l1 * np.sign(self.params["W"])

        if self.use_bias:
            self.grads["b"] = grad.sum(axis=0, keepdims=True)

        # Gradient w.r.t. input
        return grad @ self.params["W"].T

    # ── Regularization loss contribution ────────────────────────────────────

    def reg_loss(self) -> float:
        loss = 0.0
        if self.l2 > 0.0:
            loss += 0.5 * self.l2 * float(np.sum(self.params["W"] ** 2))
        if self.l1 > 0.0:
            loss += self.l1 * float(np.sum(np.abs(self.params["W"])))
        return loss

    def __repr__(self) -> str:
        act = self._activation.__class__.__name__ if self._activation else "None"
        return (f"Dense({self.in_features} → {self.out_features}, "
                f"activation={act}, l1={self.l1}, l2={self.l2})")
