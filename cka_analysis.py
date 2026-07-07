"""
Core analysis pipeline: extract layer-wise quantum representations
(Pauli expectation vectors) for a batch of inputs under trained
product/entangled QNN parameters, then compute CKA-style similarity
between them -- directly answering the project's central question:
"how does entanglement change what a quantum model represents?"

Three comparisons are performed, mirroring the structure of the
original CKA paper's experiments:

  1. SANITY CHECK (Sec 6.1 analogue): representations of the SAME
     condition (e.g. entangled) across different random-seed trained
     models should be highly similar layer-to-layer if the model is
     learning consistent structure.

  2. CROSS-CONDITION COMPARISON (this project's novel contribution):
     representations of entangled vs. non-entangled models, layer
     by layer -- do entangled circuits encode qualitatively different
     information than product circuits, or just a rotated/rescaled
     version of the same information?

  3. RANDOM VS TRAINED (Sec 6.3 analogue): representations of
     untrained (random params) vs trained models, to check that
     training changes the representation in a structured way.
"""

import numpy as np
import pennylane as qml
from .circuits import (default_device, ansatz_product, ansatz_entangled,
                        init_ansatz_params)
from .similarity import all_similarity_metrics


def extract_layerwise_representations(X, params, n_qubits, n_layers,
                                       entangled):
    """For a batch of inputs X (n_samples x n_qubits) and fixed
    ansatz parameters, return an array of shape
    (n_layers+1, n_samples, 3*n_qubits) giving the per-layer Pauli
    expectation-value representation of every sample.

    layer index 0 = representation right after data encoding (before
    any ansatz layer); layer index k = after k ansatz layers.
    """
    wires = list(range(n_qubits))
    dev = default_device(n_qubits)
    ansatz_layer_fn = ansatz_entangled if entangled else ansatz_product

    obs = []
    for w in wires:
        obs += [qml.PauliX(w), qml.PauliY(w), qml.PauliZ(w)]

    # Build one QNode per depth (0..n_layers) -- still cheap since
    # state-vector sim cost dominates, not QNode construction.
    qnodes = []
    for upto in range(n_layers + 1):
        @qml.qnode(dev)
        def circuit(x, params=params, upto=upto):
            qml.AngleEmbedding(x, wires=wires, rotation="Y")
            if upto > 0:
                ansatz_layer_fn(params[:upto], wires)
            return [qml.expval(o) for o in obs]
        qnodes.append(circuit)

    reps = np.zeros((n_layers + 1, len(X), 3 * n_qubits))
    for li, qnode in enumerate(qnodes):
        for si, x in enumerate(X):
            reps[li, si, :] = np.array(qnode(x))

    return reps


def layerwise_self_similarity(X, n_qubits, n_layers, entangled,
                               n_seeds=5, base_seed=100):
    """SANITY CHECK: train-free version -- compare representations of
    the SAME architecture (entangled or product) across different
    random parameter initializations, layer by layer. Quantifies how
    consistent the representational geometry is across random draws
    of the circuit, analogous to Section 6.1 of the CKA paper but
    applied to untrained (random-parameter) circuits, which is a
    well-defined and reproducible baseline for QML circuits."""
    all_reps = []
    for s in range(n_seeds):
        params = init_ansatz_params(n_layers, n_qubits,
                                     seed=base_seed + s)
        reps = extract_layerwise_representations(
            X, params, n_qubits, n_layers, entangled)
        all_reps.append(reps)
    all_reps = np.stack(all_reps)  # (n_seeds, n_layers+1, n_samples, dim)

    n_layers_total = all_reps.shape[1]
    cka_matrix = np.zeros((n_layers_total, n_seeds, n_seeds))
    for li in range(n_layers_total):
        for i in range(n_seeds):
            for j in range(n_seeds):
                if i == j:
                    cka_matrix[li, i, j] = 1.0
                elif j > i:
                    val = all_similarity_metrics(
                        all_reps[i, li], all_reps[j, li]
                    )["linear_cka"]
                    cka_matrix[li, i, j] = val
                    cka_matrix[li, j, i] = val

    # summarize: mean off-diagonal CKA per layer
    mean_self_cka = []
    for li in range(n_layers_total):
        mat = cka_matrix[li]
        off_diag = mat[~np.eye(n_seeds, dtype=bool)]
        mean_self_cka.append(off_diag.mean())

    return {
        "cka_matrix_per_layer": cka_matrix,
        "mean_self_cka_per_layer": np.array(mean_self_cka),
    }


def cross_condition_similarity(X, n_qubits, n_layers, n_seeds=5,
                                base_seed=100):
    """CORE COMPARISON: for each layer depth, compute CKA (and other
    indices) between product-circuit and entangled-circuit
    representations of the SAME inputs, averaged across n_seeds
    independent random parameter draws (paired by seed so both
    conditions see the 'same' random initialization scale)."""
    records = []
    for s in range(n_seeds):
        params = init_ansatz_params(n_layers, n_qubits,
                                     seed=base_seed + s)
        reps_product = extract_layerwise_representations(
            X, params, n_qubits, n_layers, entangled=False)
        reps_entangled = extract_layerwise_representations(
            X, params, n_qubits, n_layers, entangled=True)

        for li in range(n_layers + 1):
            metrics = all_similarity_metrics(
                reps_product[li], reps_entangled[li])
            metrics["layer"] = li
            metrics["seed"] = s
            records.append(metrics)

    return records


def trained_vs_random_similarity(X, trained_params, n_qubits, n_layers,
                                  entangled, n_random_seeds=5,
                                  base_seed=200):
    """Compare a TRAINED model's layerwise representations against
    untrained (random-parameter) models of the same architecture,
    quantifying how much training reorganizes the representation --
    analogous to Section 6.3 (trained vs. untrained networks) in the
    original CKA paper."""
    reps_trained = extract_layerwise_representations(
        X, trained_params, n_qubits, n_layers, entangled)

    cka_per_layer = []
    for li in range(n_layers + 1):
        vals = []
        for s in range(n_random_seeds):
            rand_params = init_ansatz_params(n_layers, n_qubits,
                                              seed=base_seed + s)
            reps_random = extract_layerwise_representations(
                X, rand_params, n_qubits, n_layers, entangled)
            val = all_similarity_metrics(
                reps_trained[li], reps_random[li])["linear_cka"]
            vals.append(val)
        cka_per_layer.append((np.mean(vals), np.std(vals)))

    return cka_per_layer


if __name__ == "__main__":
    from .data import load_dataset

    d = load_dataset("iris", n_qubits=4)
    X_sample = d["X_train"][:30]  # subsample for speed in this smoke test
    n_qubits, n_layers = 4, 2

    print("=== Self-similarity sanity check (product circuit) ===")
    res = layerwise_self_similarity(X_sample, n_qubits, n_layers,
                                     entangled=False, n_seeds=3)
    print("Mean self-CKA per layer:", res["mean_self_cka_per_layer"])

    print("\n=== Self-similarity sanity check (entangled circuit) ===")
    res = layerwise_self_similarity(X_sample, n_qubits, n_layers,
                                     entangled=True, n_seeds=3)
    print("Mean self-CKA per layer:", res["mean_self_cka_per_layer"])

    print("\n=== Cross-condition similarity (product vs entangled) ===")
    records = cross_condition_similarity(X_sample, n_qubits, n_layers,
                                          n_seeds=3)
    for r in records:
        if r["seed"] == 0:
            print(f"  layer={r['layer']}  linear_cka={r['linear_cka']:.4f}  "
                  f"rbf_cka={r['rbf_cka']:.4f}")
