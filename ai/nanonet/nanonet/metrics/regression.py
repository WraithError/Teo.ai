import numpy as np
from nanonet.core.base import Metric

class RMSE(Metric):
    @property
    def name(self) -> str: return "rmse"
    def compute(self, pred, target) -> float:
        return float(np.sqrt(np.mean((pred - target)**2)))

class MAE(Metric):
    @property
    def name(self) -> str: return "mae"
    def compute(self, pred, target) -> float:
        return float(np.mean(np.abs(pred - target)))

class R2Score(Metric):
    """Coefficient of determination (1 = perfect, 0 = baseline mean, <0 = worse than mean)."""
    @property
    def name(self) -> str: return "r2"
    def compute(self, pred, target) -> float:
        ss_res = np.sum((target - pred)**2)
        ss_tot = np.sum((target - target.mean())**2)
        return float(1.0 - ss_res / (ss_tot + 1e-9))
