import numpy as np
from nanonet.core.base import Initializer

class XavierNormal(Initializer):
    """Glorot normal: std = sqrt(2 / (fan_in + fan_out)). Good for tanh/sigmoid."""
    def initialize(self, shape: tuple) -> np.ndarray:
        fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else 1
        return np.random.randn(*shape) * np.sqrt(2.0 / (fan_in + fan_out))

class XavierUniform(Initializer):
    """Glorot uniform variant."""
    def initialize(self, shape: tuple) -> np.ndarray:
        fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else 1
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)
