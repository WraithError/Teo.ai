import numpy as np
from nanonet.core.base import Optimizer

class AdaGrad(Optimizer):
    """
    AdaGrad — adapts learning rate per parameter using accumulated squared gradients.
    Good for sparse data; can aggressively shrink lr over time.
    """
    def __init__(self, lr=0.01, eps=1e-8):
        super().__init__(lr)
        self.eps = eps
        self._g2: dict = {}

    def step(self, layers: list):
        self._step_count += 1
        for i, layer in enumerate(layers):
            if not layer.params:
                continue
            if i not in self._g2:
                self._g2[i] = {k: np.zeros_like(v) for k, v in layer.params.items()}
            for key in layer.params:
                g = layer.grads[key]
                self._g2[i][key] += g**2
                layer.params[key] -= self.lr * g / (np.sqrt(self._g2[i][key]) + self.eps)
