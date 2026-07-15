import numpy as np
from pathlib import Path

def save_weights(layers: list, path: str) -> None:
    """Serialize all layer parameters to a .npz file."""
    data = {}
    for i, layer in enumerate(layers):
        for key, val in layer.params.items():
            data[f"layer{i}_{key}"] = val
        # Also save running stats for BatchNorm
        if hasattr(layer, "running_mean"):
            data[f"layer{i}_running_mean"] = layer.running_mean
            data[f"layer{i}_running_var"]  = layer.running_var
    np.savez(path, **data)

def load_weights(layers: list, path: str) -> None:
    """Load parameters from a .npz file into layers (must match architecture)."""
    data = np.load(path)
    for i, layer in enumerate(layers):
        for key in layer.params:
            arr_key = f"layer{i}_{key}"
            if arr_key in data:
                layer.params[key] = data[arr_key]
        if hasattr(layer, "running_mean") and f"layer{i}_running_mean" in data:
            layer.running_mean = data[f"layer{i}_running_mean"]
            layer.running_var  = data[f"layer{i}_running_var"]
