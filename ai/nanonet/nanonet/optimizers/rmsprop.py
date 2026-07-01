import numpy as np
from nanonet.core.base import Optimizer

class RMSProp(Optimizer):
    """
    RMSProp optimizer — adaptive per-parameter learning rates.

    Parameters
    ----------
    lr    : float  Learning rate.
    rho   : float  Decay factor for the moving average of squared gradients.
    eps   : float  Numerical stability term.
    """
    def __init__(self, lr=0.001, rho=0.9, eps=1e-8):
        super().__init__(lr)
        self.rho = rho
        self.eps = eps
        self._eg2: dict = {}

    def step(self, layers: list):
        self._step_count += 1
        for i, layer in enumerate(layers):
            if not layer.params:
                continue
            if i not in self._eg2:
                self._eg2[i] = {k: np.zeros_like(v) for k, v in layer.params.items()}
            for key in layer.params:
                g = layer.grads[key]
                self._eg2[i][key] = self.rho * self._eg2[i][key] + (1 - self.rho) * g**2
                layer.params[key] -= self.lr * g / (np.sqrt(self._eg2[i][key]) + self.eps)
