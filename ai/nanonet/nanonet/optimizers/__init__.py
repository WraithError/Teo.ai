from .sgd import SGD
from .adam import Adam
from .rmsprop import RMSProp
from .adagrad import AdaGrad
from nanonet.core.base import Optimizer

_REGISTRY = {
    "sgd": SGD, "adam": Adam, "rmsprop": RMSProp, "adagrad": AdaGrad,
}

def get_optimizer(name_or_obj, **kwargs) -> Optimizer:
    if isinstance(name_or_obj, Optimizer):
        return name_or_obj
    key = name_or_obj.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown optimizer '{name_or_obj}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)

__all__ = ["SGD", "Adam", "RMSProp", "AdaGrad", "get_optimizer"]
