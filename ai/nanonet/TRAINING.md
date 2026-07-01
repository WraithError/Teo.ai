# Teo — Training Guide
> A from-scratch neural network framework built on NumPy.
> No PyTorch. No TensorFlow. Just pure math you can read.

---

## Quick Start

```python
from teo import Teo
from nanonet import Dense, Dropout, CrossEntropyLoss, Adam, Accuracy
from nanonet.callbacks import EarlyStopping

teo = Teo("Teo", layers=[
    Dense(128, 64, activation="relu"),
    Dropout(0.3),
    Dense(64, 10, activation="softmax"),
])

teo.compile(
    loss=CrossEntropyLoss(),
    optimizer=Adam(lr=1e-3),
    metrics=[Accuracy()],
)

history = teo.train(X_train, y_train, epochs=50,
                    validation_data=(X_val, y_val))

teo.save()           # saves to teo.npz
teo.load()           # loads from teo.npz
preds = teo.think(X) # run inference
```

---

## Project Structure

```
nanonet/              ← core framework package
  core/base.py        ← abstract base classes (Layer, Loss, Optimizer…)
  layers/
    dense.py          ← fully-connected layer (with L1/L2 regularization)
    dropout.py        ← random zeroing during training
    batch_norm.py     ← batch normalization
  activations/
    relu.py           ← ReLU, LeakyReLU
    elu.py            ← ELU
    sigmoid.py        ← Sigmoid
    tanh.py           ← Tanh
    softmax.py        ← Softmax (fused CE backward)
  losses/
    mse.py            ← Mean Squared Error
    mae.py            ← Mean Absolute Error
    huber.py          ← Huber loss
    bce.py            ← Binary Cross-Entropy
    cross_entropy.py  ← Categorical Cross-Entropy
  optimizers/
    sgd.py            ← SGD + Nesterov momentum
    adam.py           ← Adam
    rmsprop.py        ← RMSProp
    adagrad.py        ← AdaGrad
  initializers/       ← He, Xavier, RandomNormal, Zeros, Ones
  metrics/            ← Accuracy, RMSE, MAE, R2Score
  callbacks/          ← EarlyStopping, ModelCheckpoint, StepLR, ReduceLROnPlateau
  data/               ← Dataset, DataLoader
  model.py            ← Sequential model
  trainer.py          ← Trainer (decoupled training loop)

teo.py                ← Teo — the friendly top-level interface
examples/
  xor_demo.py
  regression_demo.py
  multiclass_demo.py
TRAINING.md           ← this file
```

---

## Step-by-Step Training Guide

### 1. Prepare Your Data

Teo expects plain NumPy arrays:

```python
import numpy as np

X = np.array(...)   # shape: (N, features)    float
y = np.array(...)   # shape: (N,) or (N, C)   int or float

# Always normalize your inputs — it makes training much faster
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# Split train / validation
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
```

**Target format:**
| Task | y shape | Loss |
|---|---|---|
| Regression | `(N, 1)` | `MSELoss` / `MAELoss` |
| Binary classification | `(N, 1)` float 0/1 | `BCELoss` |
| Multi-class | `(N,)` int class indices | `CrossEntropyLoss` |

---

### 2. Choose Your Architecture

```python
from teo import Teo
from nanonet import Dense, Dropout, BatchNorm

teo = Teo("Teo", layers=[
    Dense(in_features, 128, activation="relu"),  # input layer
    BatchNorm(128),                              # normalize activations
    Dropout(0.3),                                # prevent overfitting
    Dense(128, 64, activation="relu"),           # hidden layer
    Dense(64, n_classes, activation="softmax"),  # output layer
])
```

**Activation guide:**
| Activation | Use when |
|---|---|
| `"relu"` | Default for hidden layers |
| `"leaky_relu"` | When neurons "die" (gradient = 0) |
| `"tanh"` | Hidden layers, regression tasks |
| `"sigmoid"` | Binary output (single neuron) |
| `"softmax"` | Multi-class output (last layer) |
| `None` | Regression output (linear) |

**Depth rules of thumb:**
- Simple / tabular data → 2–3 layers
- Complex patterns → 4–6 layers
- More than 6 → add BatchNorm between every two Dense layers

---

### 3. Compile

```python
from nanonet import Adam, CrossEntropyLoss, Accuracy

teo.compile(
    loss=CrossEntropyLoss(),
    optimizer=Adam(lr=1e-3),   # start here; tune if needed
    metrics=[Accuracy()],
)

teo.summary()  # prints layer table + param count
```

**Optimizer guide:**
| Optimizer | Best for |
|---|---|
| `Adam(lr=1e-3)` | Almost everything — start here |
| `SGD(lr=0.01, momentum=0.9)` | When you want more control |
| `RMSProp(lr=1e-3)` | Recurrent-style tasks |
| `AdaGrad(lr=0.01)` | Sparse / NLP data |

**Loss guide:**
| Loss | Task |
|---|---|
| `MSELoss()` | Regression |
| `MAELoss()` | Regression, robust to outliers |
| `HuberLoss(delta=1.0)` | Regression, mix of MSE+MAE |
| `BCELoss()` | Binary classification (Sigmoid output) |
| `CrossEntropyLoss()` | Multi-class classification |

---

### 4. Train

```python
from nanonet.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

history = teo.train(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=10),
        ModelCheckpoint("teo_best.npz", monitor="val_accuracy", mode="max"),
        ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5),
    ],
    log_every=10,
)
```

**Batch size guide:**
| Dataset size | Batch size |
|---|---|
| < 1 000 | 16–32 |
| 1 000 – 50 000 | 32–128 |
| > 50 000 | 128–512 |

**When loss is not going down:**
1. Lower the learning rate (try `lr=1e-4`)
2. Add more neurons or layers
3. Train for more epochs
4. Check your data normalization

**When validation loss goes up but train loss goes down (overfitting):**
1. Add `Dropout(0.2–0.5)` after hidden layers
2. Add `l2=1e-4` regularization to Dense layers
3. Reduce model size
4. Use `EarlyStopping`

---

### 5. Evaluate & Predict

```python
# Score on any dataset
scores = teo.score(X_val, y_val)
print(scores)  # {"loss": ..., "accuracy": ...}

# Run inference
predictions = teo.think(X_test)  # shape: (N, out_features)

# For classification — convert probabilities to class labels
classes = predictions.argmax(axis=1)
```

---

### 6. Save & Load

```python
teo.save("teo.npz")     # save weights
teo.load("teo.npz")     # restore weights (architecture must match!)
```

---

### 7. Plot Training Curves

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history["loss"],     label="train loss")
plt.plot(history["val_loss"], label="val loss")
plt.title("Loss"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["accuracy"],     label="train acc")
plt.plot(history["val_accuracy"], label="val acc")
plt.title("Accuracy"); plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()
```

---

## How to Add Your Own Components

Every component inherits from a base class in `nanonet/core/base.py`.

### Custom Layer

```python
from nanonet.core.base import Layer
import numpy as np

class MyLayer(Layer):
    def __init__(self, ...):
        super().__init__()
        # Add learnable params here:
        # self.params["W"] = ...
        # self.grads["W"]  = np.zeros_like(self.params["W"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x          # save for backward
        return ...               # return transformed x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Populate self.grads["W"] etc.
        return ...               # return dLoss/dInput
```

### Custom Activation

```python
from nanonet.core.base import Layer

class Swish(Layer):
    """Swish: x * sigmoid(x)"""
    def forward(self, x):
        self._s = 1 / (1 + np.exp(-x))
        self._x = x
        return x * self._s
    def backward(self, grad):
        return grad * (self._s + self._x * self._s * (1 - self._s))
```

### Custom Loss

```python
from nanonet.core.base import Loss

class LogCoshLoss(Loss):
    def forward(self, pred, target):
        self._diff = pred - target
        return float(np.mean(np.log(np.cosh(self._diff))))
    def backward(self):
        return np.tanh(self._diff) / self._diff.size
```

### Custom Optimizer

```python
from nanonet.core.base import Optimizer

class MyOptimizer(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def step(self, layers):
        self._step_count += 1
        for layer in layers:
            for key in layer.params:
                layer.params[key] -= self.lr * layer.grads[key]
```

### Custom Callback

```python
from nanonet.core.base import Callback

class LossLogger(Callback):
    def on_epoch_end(self, epoch, logs, trainer):
        with open("loss_log.csv", "a") as f:
            f.write(f"{epoch},{logs.get('loss', '')},{logs.get('val_loss', '')}\n")
```

---

## Dependency

```
numpy
```

Install it with:
```bash
pip install numpy
```

Run the examples:
```bash
python examples/xor_demo.py
python examples/regression_demo.py
python examples/multiclass_demo.py
```
