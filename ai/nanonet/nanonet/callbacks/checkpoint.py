import numpy as np
from nanonet.core.base import Callback

class ModelCheckpoint(Callback):
    """
    Save the model whenever a monitored metric improves.

    Parameters
    ----------
    path    : str   File path (e.g. ``"best_model.npz"``).
    monitor : str   Metric key to watch.
    mode    : str   ``"min"`` or ``"max"``.
    verbose : bool  Print a message on each save.
    """
    def __init__(self, path="checkpoint.npz", monitor="val_loss",
                 mode="min", verbose=True):
        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self._best = None

    def on_train_begin(self, trainer):
        self._best = np.inf if self.mode == "min" else -np.inf

    def on_epoch_end(self, epoch, logs, trainer):
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = (current < self._best) if self.mode == "min" else (current > self._best)
        if improved:
            self._best = current
            trainer.model.save(self.path)
            if self.verbose:
                print(f"  [Checkpoint] Saved → {self.path}  ({self.monitor}: {current:.6f})")
