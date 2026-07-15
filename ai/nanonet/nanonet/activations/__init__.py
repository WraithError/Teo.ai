from .relu import ReLU, LeakyReLU
from .elu import ELU
from .sigmoid import Sigmoid
from .tanh import Tanh
from .softmax import Softmax
from nanonet.core.base import Layer

_REGISTRY = {
    "relu": ReLU, "leaky_relu": LeakyReLU, "elu": ELU,
    "sigmoid": Sigmoid, "tanh": Tanh, "softmax": Softmax,
}

def get_activation(name_or_obj):
    if name_or_obj is None:
        return None
    if isinstance(name_or_obj, Layer):
        return name_or_obj
    key = name_or_obj.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown activation '{name_or_obj}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[key]()

__all__ = ["ReLU","LeakyReLU","ELU","Sigmoid","Tanh","Softmax","get_activation"]
