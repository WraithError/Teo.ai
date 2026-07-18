from nanonet.backend import xp as np  # GPU-ready: xp==numpy or xp==cupy
from nanonet.core.base import Optimizer

class Adam(Optimizer):
    """
    Adam optimizer (Kingma & Ba, 2015).

    Parameters
    ----------
    lr    : float  Learning rate (default 0.001).
    beta1 : float  1st-moment decay (default 0.9).
    beta2 : float  2nd-moment decay (default 0.999).
    eps   : float  Numerical stability term.
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self._m: dict = {}
        self._v: dict = {}

    def step(self, layers: list):
        self._step_count += 1
        t = self._step_count
        for i, layer in enumerate(layers):
            if not layer.params:
                continue
            if i not in self._m:
                self._m[i] = {k: np.zeros_like(v) for k, v in layer.params.items()}
                self._v[i] = {k: np.zeros_like(v) for k, v in layer.params.items()}
            for key in layer.params:
                g = layer.grads[key]
                self._m[i][key] = self.beta1 * self._m[i][key] + (1 - self.beta1) * g
                self._v[i][key] = self.beta2 * self._v[i][key] + (1 - self.beta2) * g**2
                m_hat = self._m[i][key] / (1 - self.beta1**t)
                v_hat = self._v[i][key] / (1 - self.beta2**t)
                layer.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
