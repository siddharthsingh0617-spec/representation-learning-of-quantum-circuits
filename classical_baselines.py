

import time
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def train_classical_baselines(X_train, y_train, X_test, y_test, seed=0):
    results = {}

    t0 = time.time()
    svm = SVC(kernel="rbf", probability=True, random_state=seed)
    svm.fit(X_train, y_train)
    pred = svm.predict(X_test)
    proba = svm.predict_proba(X_test)[:, 1]
    results["rbf_svm"] = {
        "accuracy": accuracy_score(y_test, pred),
        "auc": roc_auc_score(y_test, proba),
        "time_sec": time.time() - t0,
    }

    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=500,
                         random_state=seed)
    mlp.fit(X_train, y_train)
    pred = mlp.predict(X_test)
    proba = mlp.predict_proba(X_test)[:, 1]
    results["mlp"] = {
        "accuracy": accuracy_score(y_test, pred),
        "auc": roc_auc_score(y_test, proba),
        "time_sec": time.time() - t0,
    }

    return results


if __name__ == "__main__":
    from .data import load_dataset
    d = load_dataset("iris", n_qubits=4)
    res = train_classical_baselines(d["X_train"], d["y_train"],
                                     d["X_test"], d["y_test"])
    for k, v in res.items():
        print(f"{k}: acc={v['accuracy']:.4f}  auc={v['auc']:.4f}  "
              f"time={v['time_sec']:.3f}s")
