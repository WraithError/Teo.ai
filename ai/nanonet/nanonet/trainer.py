"""
nanonet/trainer.py
Trainer — decouples the training loop from the model.
"""

from __future__ import annotations
import time
import numpy as np
from typing import List, Optional, Tuple

from nanonet.model import Sequential
from nanonet.core.base import Callback
from nanonet.data import Dataset, DataLoader


class History:
    """Stores per-epoch metrics for train and validation sets."""

    def __init__(self) -> None:
        self.history: dict = {}

    def update(self, logs: dict) -> None:
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def __getitem__(self, key: str):
        return self.history[key]

    def keys(self):
        return self.history.keys()

    def __repr__(self) -> str:
        keys = list(self.history.keys())
        epochs = len(next(iter(self.history.values()))) if self.history else 0
        return f"History(epochs={epochs}, metrics={keys})"


class Trainer:
    """
    Trains a Sequential model with full callback, validation, and history support.

    Parameters
    ----------
    model     : Sequential
    callbacks : list of Callback instances

    Example
    -------
    >>> trainer = Trainer(model, callbacks=[EarlyStopping(patience=5),
    ...                                     ModelCheckpoint("best.npz")])
    >>> history = trainer.fit(X_train, y_train, epochs=100,
    ...                       validation_data=(X_val, y_val))
    """

    def __init__(
        self,
        model: Sequential,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        self.model         = model
        self.callbacks     = callbacks or []
        self.stop_training = False
        self.history       = History()

        # Expose optimizer reference so callbacks can adjust lr
        self.optimizer = model._optimizer

    # ── Main fit loop ────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        shuffle: bool = True,
        verbose: bool = True,
        log_every: int = 1,
    ) -> History:
        """
        Train the model.

        Parameters
        ----------
        X               : Training inputs  (N, features)
        y               : Training targets (N, ...)
        epochs          : Number of full passes over the data.
        batch_size      : Mini-batch size.
        validation_data : Optional tuple (X_val, y_val) for validation metrics.
        shuffle         : Shuffle data before each epoch.
        verbose         : Print progress.
        log_every       : Print every N epochs.

        Returns
        -------
        History object with recorded metrics.
        """
        assert self.model._loss      is not None, "Call model.compile() first."
        assert self.model._optimizer is not None, "Call model.compile() first."

        loader = DataLoader(Dataset(X, y), batch_size=batch_size, shuffle=shuffle)
        self.stop_training = False
        self.optimizer     = self.model._optimizer

        self._fire("on_train_begin")

        for epoch in range(1, epochs + 1):
            self._fire("on_epoch_begin", epoch)
            self._set_mode(training=True)

            epoch_loss = 0.0
            batch_count = 0

            for batch_idx, (Xb, yb) in enumerate(loader):
                self._fire("on_batch_begin", batch_idx)

                # ── Forward ────────────────────────────────────────────────
                pred = self.model.forward(Xb)
                loss = self.model._loss.forward(pred, yb)

                # Add regularization loss from all layers
                reg  = sum(l.reg_loss() for l in self.model.layers)
                total_loss = loss + reg

                # ── Backward ───────────────────────────────────────────────
                grad = self.model._loss.backward()
                self.model.backward(grad)
                self.model._optimizer.step(self.model.layers)

                epoch_loss  += total_loss
                batch_count += 1

                self._fire("on_batch_end", batch_idx, {"loss": total_loss})

            avg_loss = epoch_loss / max(batch_count, 1)
            logs: dict = {"loss": avg_loss}

            # ── Training metrics ───────────────────────────────────────────
            if self.model._metrics:
                self._set_mode(training=False)
                train_pred = self.model.predict(X)
                self._set_mode(training=True)
                for m in self.model._metrics:
                    logs[m.name] = m.compute(train_pred, y)

            # ── Validation ────────────────────────────────────────────────
            if validation_data is not None:
                X_val, y_val = validation_data
                val_logs = self.model.evaluate(X_val, y_val)
                for k, v in val_logs.items():
                    logs[f"val_{k}"] = v

            self.history.update(logs)
            self._fire("on_epoch_end", epoch, logs)

            if verbose and (epoch % log_every == 0 or epoch == 1):
                self._print_epoch(epoch, epochs, logs)

            if self.stop_training:
                break

        self._fire("on_train_end")
        return self.history

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        return self.model.evaluate(X, y)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _fire(self, event: str, *args) -> None:
        for cb in self.callbacks:
            getattr(cb, event)(*args, self)

    @staticmethod
    def _print_epoch(epoch: int, total: int, logs: dict) -> None:
        parts = [f"Epoch {epoch:>{len(str(total))}}/{total}"]
        for k, v in logs.items():
            parts.append(f"{k}: {v:.5f}")
        print("  ".join(parts))

    def _set_mode(self, training: bool) -> None:
        """Directly toggle layer training flags without calling model.train()/eval()."""
        for layer in self.model.layers:
            layer.training = training
