import numpy as np
from nanonet.core.base import Optimizer, Layer

class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional Nesterov momentum.

    Parameters
    ----------
    lr       : float  Learning rate.
    momentum : float  Momentum factor (0 = plain SGD).
    nesterov : bool   Use Nesterov lookahead.
    """
    def __init__(self, lr: float = 0.01, momentum: float = 0.0, nesterov: bool = False):
        super().__init__(lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self._v: dict = {}

    def step(self, layers: list):
        self._step_count += 1
        for i, layer in enumerate(layers):
            if not layer.params:
                continue
            if i not in self._v:
                self._v[i] = {k: np.zeros_like(v) for k, v in layer.params.items()}
            for key in layer.params:
                g = layer.grads[key]
                v = self.momentum * self._v[i][key] - self.lr * g
                self._v[i][key] = v
                if self.nesterov:
                    layer.params[key] += self.momentum * v - self.lr * g
                else:
                    layer.params[key] += v
