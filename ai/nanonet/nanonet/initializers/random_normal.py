import numpy as np
from nanonet.core.base import Initializer

class RandomNormal(Initializer):
    """Gaussian initializer with configurable mean and std."""
    def __init__(self, mean: float = 0.0, std: float = 0.01) -> None:
        self.mean = mean
        self.std = std
    def initialize(self, shape: tuple) -> np.ndarray:
        return np.random.randn(*shape) * self.std + self.mean

class Zeros(Initializer):
    def initialize(self, shape: tuple) -> np.ndarray:
        return np.zeros(shape)

class Ones(Initializer):
    def initialize(self, shape: tuple) -> np.ndarray:
        return np.ones(shape)
