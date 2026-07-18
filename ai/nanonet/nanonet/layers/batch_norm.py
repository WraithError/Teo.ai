"""
nanonet/layers/batch_norm.py
Batch Normalization layer (Ioffe & Szegedy, 2015).
"""

from __future__ import annotations
import numpy as np
from nanonet.core.base import Layer


class BatchNorm(Layer):
    """
    Normalizes each feature across the batch, then applies a learned
    scale (gamma) and shift (beta).

    During training  : normalizes over the current batch; updates running stats.
    During inference : uses stored running mean / variance.

    Parameters
    ----------
    num_features : int
        Number of input features (must match the last axis of the input).
    eps          : float
        Small constant for numerical stability.
    momentum     : float
        Momentum for running mean / variance update.

    Example
    -------
    >>> model.add(Dense(128, 64))
    >>> model.add(BatchNorm(64))
    >>> model.add(ReLU())
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.eps          = eps
        self.momentum     = momentum

        # Learnable parameters
        self.params["gamma"] = np.ones((1, num_features))
        self.params["beta"]  = np.zeros((1, num_features))
        self.grads["gamma"]  = np.zeros_like(self.params["gamma"])
        self.grads["beta"]   = np.zeros_like(self.params["beta"])

        # Running statistics (not trained by optimizer, updated in forward)
        self.running_mean = np.zeros((1, num_features))
        self.running_var  = np.ones((1,  num_features))

        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            mu  = x.mean(axis=0, keepdims=True)
            var = x.var(axis=0,  keepdims=True)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var

            x_hat = (x - mu) / np.sqrt(var + self.eps)
            self._cache = {"x": x, "x_hat": x_hat, "mu": mu, "var": var}
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.params["gamma"] * x_hat + self.params["beta"]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x     = self._cache["x"]
        x_hat = self._cache["x_hat"]
        mu    = self._cache["mu"]
        var   = self._cache["var"]
        N     = x.shape[0]

        self.grads["gamma"] = (grad * x_hat).sum(axis=0, keepdims=True)
        self.grads["beta"]  = grad.sum(axis=0, keepdims=True)

        dx_hat   = grad * self.params["gamma"]
        dvar     = (-0.5 * dx_hat * (x - mu) * (var + self.eps) ** -1.5).sum(axis=0, keepdims=True)
        dmu      = (-dx_hat / np.sqrt(var + self.eps)).sum(axis=0, keepdims=True) + dvar * (-2.0 * (x - mu)).mean(axis=0, keepdims=True)
        dx       = dx_hat / np.sqrt(var + self.eps) + dvar * 2.0 * (x - mu) / N + dmu / N

        return dx

    def __repr__(self) -> str:
        return f"BatchNorm({self.num_features}, eps={self.eps}, momentum={self.momentum})"
