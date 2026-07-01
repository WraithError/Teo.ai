from nanonet.core.base import Callback

class StepLR(Callback):
    """Multiply the learning rate by ``gamma`` every ``step_size`` epochs."""
    def __init__(self, step_size: int = 10, gamma: float = 0.5):
        self.step_size = step_size
        self.gamma = gamma

    def on_epoch_end(self, epoch, logs, trainer):
        if epoch % self.step_size == 0:
            trainer.optimizer.lr *= self.gamma
            print(f"  [StepLR] lr → {trainer.optimizer.lr:.6f}")


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving."""
    def __init__(self, monitor="val_loss", factor=0.5, patience=3,
                 min_lr=1e-6, mode="min"):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self._wait = 0
        self._best = None

    def on_train_begin(self, trainer):
        import numpy as np
        self._best = float("inf") if self.mode == "min" else float("-inf")
        self._wait = 0

    def on_epoch_end(self, epoch, logs, trainer):
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = (current < self._best) if self.mode == "min" else (current > self._best)
        if improved:
            self._best = current
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                new_lr = max(trainer.optimizer.lr * self.factor, self.min_lr)
                trainer.optimizer.lr = new_lr
                self._wait = 0
                print(f"  [ReduceLROnPlateau] lr → {new_lr:.6f}")
