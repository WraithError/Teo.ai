"""
examples/multiclass_demo.py
───────────────────────────
Classify the Iris dataset (3 classes) with Teo.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from teo import Teo
from nanonet import Dense, Dropout, CrossEntropyLoss, Adam, Accuracy
from nanonet.callbacks import EarlyStopping, ModelCheckpoint

np.random.seed(7)

# ── Toy multi-class dataset (mimics Iris structure) ───────────────────────────
def make_dataset(n_per_class=50, n_features=4, n_classes=3):
    X_list, y_list = [], []
    for c in range(n_classes):
        center = np.random.randn(n_features) * 3
        X_list.append(np.random.randn(n_per_class, n_features) + center)
        y_list.append(np.full(n_per_class, c))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx].astype(int)

X, y = make_dataset()

# Normalize features
X = (X - X.mean(0)) / (X.std(0) + 1e-8)

# Train / val split
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ── Build Teo ─────────────────────────────────────────────────────────────────
teo = Teo("Teo", layers=[
    Dense(4,  32, activation="relu"),
    Dropout(0.2),
    Dense(32, 16, activation="relu"),
    Dense(16,  3, activation="softmax"),
])

teo.compile(
    loss=CrossEntropyLoss(from_logits=False),   # Softmax already applied
    optimizer=Adam(lr=1e-3),
    metrics=[Accuracy()],
)

teo.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training Teo on multi-class classification...\n")
history = teo.train(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=15),
        ModelCheckpoint("teo_best.npz", monitor="val_accuracy", mode="max"),
    ],
    log_every=25,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
scores = teo.score(X_val, y_val)
print("\nFinal validation scores:")
for k, v in scores.items():
    print(f"  {k}: {v:.5f}")
