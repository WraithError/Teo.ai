import numpy as np
from .dataset import Dataset

class DataLoader:
    """
    Iterates over a Dataset in mini-batches.

    Parameters
    ----------
    dataset    : Dataset
    batch_size : int
    shuffle    : bool  Shuffle before each epoch.
    drop_last  : bool  Drop the final incomplete batch.

    Usage
    -----
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    >>> for X_batch, y_batch in loader:
    ...     ...
    """
    def __init__(self, dataset: Dataset, batch_size: int = 32,
                 shuffle: bool = True, drop_last: bool = False) -> None:
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.drop_last  = drop_last

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        idx = np.random.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            yield self.dataset.X[batch_idx], self.dataset.y[batch_idx]

    def __repr__(self) -> str:
        return (f"DataLoader(dataset={self.dataset}, "
                f"batch_size={self.batch_size}, shuffle={self.shuffle})")
