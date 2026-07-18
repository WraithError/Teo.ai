"""
nanonet/backend.py
------------------
Hardware abstraction layer.

Import `xp` everywhere instead of `np` or `cp`. Same code runs on
CPU (NumPy) and GPU (CuPy) with zero changes to any other file.

Usage:
    from nanonet.backend import xp, GPU
    x = xp.zeros((3, 4))          # np.zeros on CPU, cp.zeros on GPU
    x = xp.random.randn(10, 64)   # works on both

When you get a real GPU later:
    pip install cupy-cuda12x        # match your CUDA version
    # That's it. Every layer, optimizer, and training script upgrades automatically.
"""

try:
    import cupy as xp
    GPU = True
    BACKEND = "cupy"
except ImportError:
    import numpy as xp
    GPU = False
    BACKEND = "numpy"

__all__ = ["xp", "GPU", "BACKEND"]
