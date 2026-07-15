from nanonet.core.base import Callback
import numpy as np

class EarlyStopping(Callback):
    """
    Stop training when a monitored metric stops improving.

    Parameters
    ----------
    monitor   : str   Metric key to watch, e.g. ``"val_loss"`` or ``"val_accuracy"``.
    patience  : int   Epochs with no improvement before stopping.
    min_delta : float Minimum change to count as improvement.
    mode      : str   ``"min"`` (loss) or ``"max"`` (accuracy).
    restore_best_weights : bool  Restore the best weights when stopping.
    """
    def __init__(self, monitor="val_loss", patience=5, min_delta=1e-4,
                 mode="min", restore_best_weights=True):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self._best = None
        self._wait = 0
        self._best_weights = None

    def on_train_begin(self, trainer):
        self._best = np.inf if self.mode == "min" else -np.inf
        self._wait = 0

    def on_epoch_end(self, epoch, logs, trainer):
        current = logs.get(self.monitor)
        if current is None:
            return

        improved = (current < self._best - self.min_delta) if self.mode == "min" \
                   else (current > self._best + self.min_delta)

        if improved:
            self._best = current
            self._wait = 0
            if self.restore_best_weights:
                self._best_weights = trainer.model.get_weights()
        else:
            self._wait += 1
            if self._wait >= self.patience:
                trainer.stop_training = True
                print(f"\n[EarlyStopping] Stopped at epoch {epoch}. "
                      f"Best {self.monitor}: {self._best:.6f}")
                if self.restore_best_weights and self._best_weights:
                    trainer.model.set_weights(self._best_weights)
                    print("[EarlyStopping] Best weights restored.")
