from .he import HeNormal
from .xavier import XavierNormal, XavierUniform
from .random_normal import RandomNormal, Zeros, Ones
from nanonet.core.base import Initializer

_REGISTRY = {
    "he": HeNormal, "he_normal": HeNormal,
    "xavier": XavierNormal, "xavier_normal": XavierNormal,
    "xavier_uniform": XavierUniform,
    "random_normal": RandomNormal,
    "zeros": Zeros, "ones": Ones,
}

def get_initializer(name_or_obj) -> Initializer:
    if isinstance(name_or_obj, Initializer):
        return name_or_obj
    key = name_or_obj.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown initializer '{name_or_obj}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[key]()

__all__ = ["HeNormal","XavierNormal","XavierUniform","RandomNormal","Zeros","Ones","get_initializer"]
