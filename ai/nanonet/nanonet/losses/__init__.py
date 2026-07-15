from .mse import MSELoss
from .mae import MAELoss
from .huber import HuberLoss
from .bce import BCELoss
from .cross_entropy import CrossEntropyLoss
from nanonet.core.base import Loss

_REGISTRY = {
    "mse": MSELoss, "mae": MAELoss, "huber": HuberLoss,
    "bce": BCELoss, "binary_crossentropy": BCELoss,
    "crossentropy": CrossEntropyLoss, "cross_entropy": CrossEntropyLoss,
}

def get_loss(name_or_obj) -> Loss:
    if isinstance(name_or_obj, Loss):
        return name_or_obj
    key = name_or_obj.lower().replace("-", "_")
    if key not in _REGISTRY:
        raise ValueError(f"Unknown loss '{name_or_obj}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[key]()

__all__ = ["MSELoss","MAELoss","HuberLoss","BCELoss","CrossEntropyLoss","get_loss"]
