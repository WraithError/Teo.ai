import time
from nanonet.core.base import Callback

class ProgressBar(Callback):
    """Prints a per-epoch progress line with loss and metric values."""
    def __init__(self, log_every: int = 1):
        self.log_every = log_every
        self._t0 = None

    def on_train_begin(self, trainer):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs, trainer):
        if epoch % self.log_every != 0:
            return
        elapsed = time.time() - self._t0
        parts = [f"Epoch {epoch:>4}"]
        for k, v in logs.items():
            parts.append(f"{k}: {v:.5f}")
        parts.append(f"({elapsed:.1f}s total)")
        print("  ".join(parts))
