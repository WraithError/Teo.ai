import numpy as np

class Dataset:
    """
    Wraps a pair of numpy arrays (X, y) into an iterable dataset.
    Subclass this to add transforms, augmentation, lazy loading, etc.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        assert len(X) == len(y), "X and y must have the same number of samples."
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __repr__(self) -> str:
        return f"Dataset(n={len(self)}, X={self.X.shape}, y={self.y.shape})"
