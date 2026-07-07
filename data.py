

import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_dataset(name: str, n_qubits: int, test_size: float = 0.3,
                  random_state: int = 42, binary: bool = True):
   
    if name == "iris":
        data = load_iris()
    elif name == "wine":
        data = load_wine()
    else:
        raise ValueError(f"Unknown dataset {name!r}")

    X, y = data.data, data.target

    if binary:
 
        y = (y == 0).astype(int)

   
    pca = PCA(n_components=n_qubits, random_state=random_state)
    X_reduced = pca.fit_transform(X)

   
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
