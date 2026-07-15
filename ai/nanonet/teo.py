"""
teo.py
──────
Meet Teo — your personal neural network.

Teo is a thin, friendly wrapper around Sequential + Trainer that lets you
build, name, and train a model with minimal boilerplate.

  >>> teo = Teo("Teo", layers=[...])
  >>> teo.compile(loss=..., optimizer=..., metrics=[...])
  >>> history = teo.train(X, y, epochs=50)
  >>> teo.save()
  >>> print(teo)
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple

from nanonet.model import Sequential
from nanonet.trainer import Trainer, History
from nanonet.core.base import Layer, Loss, Optimizer, Metric, Callback


class Teo(Sequential):
    """
    A named neural network you can talk to.

    Parameters
    ----------
    name   : str             A name for your model (default "Teo").
    layers : list of Layer   The initial stack of layers (optional).

    Examples
    --------
    Build and train in a few lines:

    >>> from nanonet.layers import Dense, Dropout
    >>> from nanonet.losses import CrossEntropyLoss
    >>> from nanonet.optimizers import Adam
    >>> from nanonet.metrics import Accuracy
    >>> from nanonet.callbacks import EarlyStopping
    >>> from teo import Teo

    >>> teo = Teo("Teo", layers=[
    ...     Dense(784, 128, activation="relu"),
    ...     Dropout(0.2),
    ...     Dense(128, 10, activation="softmax"),
    ... ])

    >>> teo.compile(
    ...     loss=CrossEntropyLoss(),
    ...     optimizer=Adam(lr=1e-3),
    ...     metrics=[Accuracy()],
    ... )

    >>> history = teo.train(
    ...     X_train, y_train,
    ...     epochs=30,
    ...     validation_data=(X_val, y_val),
    ...     callbacks=[EarlyStopping(patience=5)],
    ... )

    >>> teo.save("teo.npz")
    """

    def __init__(
        self,
        name: str = "Teo",
        layers: Optional[List[Layer]] = None,
    ) -> None:
        super().__init__(layers=layers)
        self.name = name

    # ── Training shortcut ────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        callbacks: Optional[List[Callback]] = None,
        shuffle: bool = True,
        verbose: bool = True,
        log_every: int = 1,
    ) -> History:
        """
        Train Teo and return a History object.

        Parameters
        ----------
        X               : Training feature array  (N, features)
        y               : Training target array   (N, ...)
        epochs          : Full passes over the training data.
        batch_size      : Samples per mini-batch.
        validation_data : ``(X_val, y_val)`` evaluated after every epoch.
        callbacks       : List of Callback objects.
        shuffle         : Shuffle before each epoch.
        verbose         : Print epoch-level logs.
        log_every       : Only print every N epochs.

        Returns
        -------
        History  — dict-like object; access curves with ``history["loss"]``.
        """
        trainer = Trainer(self, callbacks=callbacks or [])
        return trainer.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            shuffle=shuffle,
            verbose=verbose,
            log_every=log_every,
        )

    # ── Convenience wrappers ─────────────────────────────────────────────────

    def think(self, X: np.ndarray) -> np.ndarray:
        """Alias for ``predict()``. Run inference on X."""
        return self.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Alias for ``evaluate()``. Return loss + metrics on (X, y)."""
        return self.evaluate(X, y)

    def save(self, path: Optional[str] = None) -> None:
        path = path or f"{self.name.lower()}.npz"
        super().save(path)
        print(f"[{self.name}] Weights saved → {path}")

    def load(self, path: Optional[str] = None) -> None:
        path = path or f"{self.name.lower()}.npz"
        super().load(path)
        print(f"[{self.name}] Weights loaded ← {path}")

    def __repr__(self) -> str:
        n = self.num_params()
        return (f"Teo(name='{self.name}', "
                f"layers={len(self.layers)}, "
                f"params={n:,})")
