"""
NanoNet — a from-scratch neural network framework built on NumPy.
Use via the high-level Teo interface or build directly with the modules below.
"""

from nanonet.model   import Sequential
from nanonet.trainer import Trainer, History

from nanonet.layers      import Dense, Dropout, BatchNorm, LSTM
from nanonet.activations import ReLU, LeakyReLU, ELU, Sigmoid, Tanh, Softmax
from nanonet.losses      import MSELoss, MAELoss, HuberLoss, BCELoss, CrossEntropyLoss
from nanonet.optimizers  import SGD, Adam, RMSProp, AdaGrad
from nanonet.initializers import HeNormal, XavierNormal, XavierUniform, RandomNormal
from nanonet.metrics     import Accuracy, RMSE, MAE, R2Score
from nanonet.callbacks   import EarlyStopping, ModelCheckpoint, StepLR, ReduceLROnPlateau, ProgressBar
from nanonet.data        import Dataset, DataLoader

__all__ = [
    "Sequential", "Trainer", "History",
    "Dense", "Dropout", "BatchNorm", "LSTM",
    "ReLU", "LeakyReLU", "ELU", "Sigmoid", "Tanh", "Softmax",
    "MSELoss", "MAELoss", "HuberLoss", "BCELoss", "CrossEntropyLoss",
    "SGD", "Adam", "RMSProp", "AdaGrad",
    "HeNormal", "XavierNormal", "XavierUniform", "RandomNormal",
    "Accuracy", "RMSE", "MAE", "R2Score",
    "EarlyStopping", "ModelCheckpoint", "StepLR", "ReduceLROnPlateau", "ProgressBar",
    "Dataset", "DataLoader",
]
