"""
Variational QNN classifier training: binary classification via
gradient descent on a square-loss objective using the product /
entangled ansätze defined in circuits.py. Uses PennyLane's Adam
optimizer with parameter-shift-free adjoint differentiation
(exact gradients, fast on state-vector simulators).
"""

import numpy as np
import pennylane as qml
import time
from sklearn.metrics import accuracy_score, roc_auc_score
from .circuits import make_qnn_circuit, init_ansatz_params


def train_qnn(X_train, y_train, X_test, y_test, n_qubits, n_layers,
              entangled, n_epochs=60, lr=0.1, batch_size=20,
              seed=0, verbose=True, track_loss_curve=True):
    """Train a variational QNN binary classifier via Adam on MSE loss
    between <Z_0> in [-1, 1] and the {-1, +1}-mapped label.

    Returns dict with trained params, test accuracy/AUC, loss curve,
    and timing -- everything needed for the results table and figures.
    """
    label = "entangled" if entangled else "product (non-entangled)"
    circuit = make_qnn_circuit(n_qubits, n_layers, entangled)
    y_train_pm = 2 * y_train - 1  # map {0,1} -> {-1,+1}
    y_test_pm = 2 * y_test - 1

    params = qml.numpy.array(
        init_ansatz_params(n_layers, n_qubits, seed=seed),
        requires_grad=True
    )

    def batch_loss(params, X_batch, y_batch):
        preds = qml.math.stack([circuit(x, params) for x in X_batch])
        return qml.math.mean((preds - y_batch) ** 2)

    opt = qml.AdamOptimizer(stepsize=lr)
    rng = np.random.default_rng(seed)

    loss_curve = []
    t0 = time.time()
    n_train = len(X_train)

    for epoch in range(n_epochs):
        idx = rng.permutation(n_train)
        epoch_losses = []
        for start in range(0, n_train, batch_size):
            batch_idx = idx[start:start + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train_pm[batch_idx]

            params, loss_val = opt.step_and_cost(
                lambda p: batch_loss(p, X_batch, y_batch), params
            )
            epoch_losses.append(loss_val)

        mean_loss = float(np.mean(epoch_losses))
        if track_loss_curve:
            loss_curve.append(mean_loss)
        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"    epoch {epoch:3d}  loss={mean_loss:.4f}")

    elapsed = time.time() - t0

    # evaluate
    def predict_scores(X):
        return np.array([circuit(x, params) for x in X])

    train_scores = predict_scores(X_train)
    test_scores = predict_scores(X_test)

    train_pred = (train_scores > 0).astype(int)
    test_pred = (test_scores > 0).astype(int)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    try:
        test_auc = roc_auc_score(y_test, test_scores)
    except ValueError:
        test_auc = float("nan")

    if verbose:
        print(f"  [{label:24s}] train_acc={train_acc:.4f}  "
              f"test_acc={test_acc:.4f}  test_auc={test_auc:.4f}  "
              f"time={elapsed:.1f}s")

    return {
        "params": params,
        "loss_curve": loss_curve,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_auc": test_auc,
        "time_sec": elapsed,
        "entangled": entangled,
        "n_epochs": n_epochs,
        "seed": seed,
    }


if __name__ == "__main__":
    from .data import load_dataset

    d = load_dataset("iris", n_qubits=4)
    print("QNN training, iris, 4 qubits, 2 layers")
    for ent in [False, True]:
        train_qnn(
            d["X_train"], d["y_train"], d["X_test"], d["y_test"],
            n_qubits=4, n_layers=2, entangled=ent,
            n_epochs=30, seed=0
        )
