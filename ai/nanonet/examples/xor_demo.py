"""
examples/xor_demo.py
────────────────────
Train Teo to solve the XOR problem — a classic non-linearly separable task.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from teo import Teo
from nanonet import Dense, Dropout, BCELoss, Adam, Accuracy
from nanonet.callbacks import EarlyStopping, ProgressBar

np.random.seed(42)

# ── Dataset ──────────────────────────────────────────────────────────────────
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]],         dtype=float)

# ── Build Teo ─────────────────────────────────────────────────────────────────
teo = Teo("Teo", layers=[
    Dense(2, 16, activation="relu"),
    Dense(16, 8, activation="relu"),
    Dense(8,  1, activation="sigmoid"),
])

teo.compile(
    loss=BCELoss(),
    optimizer=Adam(lr=0.01),
    metrics=[Accuracy()],
)

teo.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training Teo on XOR...\n")
history = teo.train(
    X, y,
    epochs=500,
    batch_size=4,
    callbacks=[EarlyStopping(monitor="loss", patience=30, mode="min")],
    log_every=50,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\nResults:")
preds = teo.think(X)
for xi, yi, pi in zip(X, y, preds):
    label = "✓" if round(pi[0]) == yi[0] else "✗"
    print(f"  {label}  input={xi.astype(int)}  target={int(yi[0])}  pred={pi[0]:.4f}")

teo.save()
