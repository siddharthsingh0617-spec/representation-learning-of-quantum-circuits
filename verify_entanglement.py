

import numpy as np
import pennylane as qml
from .circuits import (feature_map_product, feature_map_entangled,
                        ansatz_product, ansatz_entangled,
                        init_ansatz_params)


def average_entanglement_entropy(circuit_fn, n_qubits, n_layers,
                                  n_samples=30, seed=0, is_ansatz=False):
    
    rng = np.random.default_rng(seed)
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    entropies = []
    for _ in range(n_samples):
        x = rng.uniform(0, np.pi, n_qubits)

        if is_ansatz:
            params = init_ansatz_params(n_layers, n_qubits,
                                         seed=rng.integers(1e6))

            @qml.qnode(dev)
            def circ(w=wires):
                qml.AngleEmbedding(x, wires=w, rotation="Y")
                circuit_fn(params, w)
                return [qml.vn_entropy(wires=[i]) for i in w]
        else:
            @qml.qnode(dev)
            def circ(w=wires):
                circuit_fn(x, w, n_layers=n_layers)
                return [qml.vn_entropy(wires=[i]) for i in w]

        entropies.append(np.mean(circ()))

    return float(np.mean(entropies)), float(np.std(entropies))


def run_manipulation_check(n_qubits=4, n_layers=2, n_samples=30, seed=0):

    results = {}

    for name, fn in [("feature_map_product", feature_map_product),
                      ("feature_map_entangled", feature_map_entangled)]:
        mean, std = average_entanglement_entropy(
            fn, n_qubits, n_layers, n_samples, seed, is_ansatz=False)
        results[name] = (mean, std)

    for name, fn in [("ansatz_product", ansatz_product),
                      ("ansatz_entangled", ansatz_entangled)]:
        mean, std = average_entanglement_entropy(
            fn, n_qubits, n_layers, n_samples, seed, is_ansatz=True)
        results[name] = (mean, std)

    return results


if __name__ == "__main__":
    res = run_manipulation_check()
    print("Manipulation check: mean single-qubit von Neumann entanglement "
          "entropy (nats)\n")
    print(f"{'Circuit':<25s} {'Mean':>10s} {'Std':>10s}")
    for k, (m, s) in res.items():
        print(f"{k:<25s} {m:>10.4f} {s:>10.4f}")
