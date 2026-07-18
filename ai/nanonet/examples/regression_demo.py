"""
examples/regression_demo.py
───────────────────────────
Fit a sine wave with Teo using MSE loss and BatchNorm.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from teo import Teo
from nanonet import Dense, BatchNorm, MSELoss, Adam, RMSE, R2Score
from nanonet.callbacks import ReduceLROnPlateau, ProgressBar

np.random.seed(0)

# ── Dataset ───────────────────────────────────────────────────────────────────
N = 500
X = np.linspace(-np.pi, np.pi, N).reshape(-1, 1)
y = np.sin(X) + np.random.randn(N, 1) * 0.05   # noisy sine

split = int(0.8 * N)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ── Build Teo ─────────────────────────────────────────────────────────────────
teo = Teo("Teo", layers=[
    Dense(1,  64, activation="tanh"),
    BatchNorm(64),
    Dense(64, 64, activation="tanh"),
    Dense(64,  1),                      # linear output for regression
])

teo.compile(
    loss=MSELoss(),
    optimizer=Adam(lr=5e-3),
    metrics=[RMSE(), R2Score()],
)

teo.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training Teo on sine regression...\n")
history = teo.train(
    X_train, y_train,
    epochs=300,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[ReduceLROnPlateau(monitor="val_loss", patience=10, factor=0.5)],
    log_every=50,
)

# ── Final score ───────────────────────────────────────────────────────────────
scores = teo.score(X_val, y_val)
print("\nValidation results:")
for k, v in scores.items():
    print(f"  {k}: {v:.5f}")

teo.save()
