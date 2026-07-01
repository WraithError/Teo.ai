import numpy as np
from nanonet.core.base import Initializer

class HeNormal(Initializer):
    """He (Kaiming) normal: std = sqrt(2 / fan_in). Best for ReLU."""
    def initialize(self, shape: tuple) -> np.ndarray:
        return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
