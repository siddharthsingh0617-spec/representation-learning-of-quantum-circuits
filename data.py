"""
Data loading and preprocessing utilities for the QML benchmark.

We use standard small-scale UCI datasets (Iris, Wine), reduced via PCA
to n_qubits dimensions and scaled to [0, pi] for angle encoding, which
is the standard preprocessing pipeline used in quantum kernel / QNN
literature (Havlicek et al. 2019; Schuld & Killoran 2019).
"""

import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_dataset(name: str, n_qubits: int, test_size: float = 0.3,
                  random_state: int = 42, binary: bool = True):
    """
    Load and preprocess a dataset for quantum ML experiments.

    Parameters
    ----------
    name : 'iris' or 'wine'
    n_qubits : number of qubits == number of PCA components to keep
    test_size : fraction held out for testing
    random_state : seed controlling PCA + split (NOT the model seed)
    binary : if True, collapse to a binary classification task
             (class 0 vs rest) for cleaner kernel/QNN benchmarking;
             if False, keep the full multiclass problem.

    Returns
    -------
    dict with X_train, X_test, y_train, y_test (angles in [0, pi]),
    plus metadata (feature names surrogate, class names).
    """
    if name == "iris":
        data = load_iris()
    elif name == "wine":
        data = load_wine()
    else:
        raise ValueError(f"Unknown dataset {name!r}")

    X, y = data.data, data.target

    if binary:
        # class 0 vs. rest -> balanced-ish binary problem, keeps the
        # kernel/QNN experiments cheap and the CKA analysis clean.
        y = (y == 0).astype(int)

    # Standardize then PCA-reduce to n_qubits dims (one feature per qubit,
    # single-angle encoding). This mirrors common QML preprocessing
    # (Havlicek et al. 2019, Suzuki et al. 2020).
    pca = PCA(n_components=n_qubits, random_state=random_state)
    X_reduced = pca.fit_transform(X)

    # Scale each PCA component independently to [0, pi] for angle encoding
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X_reduced)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state,
        stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "n_qubits": n_qubits,
        "pca_explained_var": pca.explained_variance_ratio_.sum(),
        "dataset_name": name,
        "n_classes": len(np.unique(y)),
        "class_balance_train": np.bincount(y_train) / len(y_train),
    }


if __name__ == "__main__":
    for ds in ["iris", "wine"]:
        d = load_dataset(ds, n_qubits=4)
        print(f"{ds}: train={d['X_train'].shape}, test={d['X_test'].shape}, "
              f"PCA var explained={d['pca_explained_var']:.3f}, "
              f"class balance={d['class_balance_train']}")
