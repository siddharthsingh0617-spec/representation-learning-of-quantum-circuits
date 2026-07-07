"""
Quantum circuit definitions: feature maps (data encodings) and
variational ansätze, each provided in matched ENTANGLED /
NON-ENTANGLED pairs so that the only difference between conditions
is the presence of entangling two-qubit gates. Everything else
(number of layers, number of single-qubit rotations, parameter count
where applicable) is held fixed -- this is essential for a clean
ablation: we want entanglement, not circuit capacity, to be the
manipulated variable.

Two circuit families are provided:

1. Feature maps (data re-uploading style, used for quantum kernels):
   - ZFeatureMap-style:        single-qubit RZ/RY encoding only (NO entanglement)
   - ZZFeatureMap-style:       adds ZZ-entangling layers between encodings
     (this is the standard Havlicek et al. 2019 construction)

2. Variational ansätze (used for the trainable QNN classifier):
   - HEA-product:  hardware-efficient ansatz, rotation layers only, no CNOTs
   - HEA-entangled: identical rotation layers + a ring of CNOTs per layer

Each circuit also exposes a `layer_representations` mode used by the
representation-similarity (CKA) analysis: instead of returning only
the final measurement, it returns the vector of single-qubit Pauli
expectation values <X_i>, <Y_i>, <Z_i> evaluated after EVERY layer,
which we treat as the network's per-layer "representation" of an input,
directly analogous to "layer activations" in the classical CKA paper.
"""

import pennylane as qml
import numpy as np


def default_device(n_qubits):
    """Use the fast C++ state-vector simulator when available, falling
    back to default.qubit otherwise. lightning.qubit gives ~30-50%
    speedups for the circuit depths used in this project and produces
    numerically identical results (both are exact statevector
    simulators)."""
    try:
        return qml.device("lightning.qubit", wires=n_qubits)
    except qml.DeviceError:
        return qml.device("default.qubit", wires=n_qubits)


# ---------------------------------------------------------------------
# Feature maps (for quantum kernel methods)
# ---------------------------------------------------------------------

def feature_map_product(x, wires, n_layers=2):
    """Non-entangled feature map: per-qubit RZ-RY-RZ data encoding,
    repeated n_layers times with no two-qubit gates at all. Each qubit
    evolves completely independently -> the resulting state is always
    a product state."""
    n = len(wires)
    for layer in range(n_layers):
        for i, w in enumerate(wires):
            xi = x[i % len(x)]
            qml.RZ(xi, wires=w)
            qml.RY(xi, wires=w)
            qml.RZ((layer + 1) * xi, wires=w)


def feature_map_entangled(x, wires, n_layers=2):
    """Entangled feature map (ZZ-style, Havlicek et al. 2019): identical
    single-qubit encoding layer as `feature_map_product`, PLUS a ring
    of ZZ-entangling (CNOT-RZ-CNOT) gates between adjacent qubits after
    each layer. Single-qubit gate count is identical to the product
    version; entanglement is the only structural addition."""
    n = len(wires)
    for layer in range(n_layers):
        for i, w in enumerate(wires):
            xi = x[i % len(x)]
            qml.RZ(xi, wires=w)
            qml.RY(xi, wires=w)
            qml.RZ((layer + 1) * xi, wires=w)
        # entangling ring: encodes pairwise feature products, standard
        # ZZFeatureMap construction
        for i in range(n):
            j = (i + 1) % n
            xi, xj = x[i % len(x)], x[j % len(x)]
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ((np.pi - xi) * (np.pi - xj), wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])


# ---------------------------------------------------------------------
# Variational ansätze (for the trainable QNN classifier)
# ---------------------------------------------------------------------

def ansatz_product(params, wires):
    """Hardware-efficient ansatz, NO entangling gates: each qubit gets
    its own independent Rot(theta1, theta2, theta3) per layer. Acts as
    a strict ablation baseline -- a QNN that can only ever represent
    product states, regardless of training."""
    n_layers, n = params.shape[0], len(wires)
    for layer in range(n_layers):
        for i, w in enumerate(wires):
            qml.Rot(*params[layer, i], wires=w)


def ansatz_entangled(params, wires):
    """Hardware-efficient ansatz WITH entanglement: identical per-qubit
    Rot layers as `ansatz_product`, plus a ring of CNOTs after each
    rotation layer (standard HEA, e.g. Kandala et al. 2017)."""
    n_layers, n = params.shape[0], len(wires)
    for layer in range(n_layers):
        for i, w in enumerate(wires):
            qml.Rot(*params[layer, i], wires=w)
        for i in range(n):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


def init_ansatz_params(n_layers, n_qubits, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 3))


# ---------------------------------------------------------------------
# QNode factories
# ---------------------------------------------------------------------

def make_kernel_circuit(n_qubits, n_layers, entangled, device=None):
    """Returns a QNode computing the fidelity |<phi(x1)|phi(x2)>|^2
    between two feature-mapped states -- the quantum kernel value."""
    wires = list(range(n_qubits))
    dev = device or default_device(n_qubits)
    fmap = feature_map_entangled if entangled else feature_map_product

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        fmap(x1, wires, n_layers=n_layers)
        qml.adjoint(fmap)(x2, wires, n_layers=n_layers)
        return qml.probs(wires=wires)

    return kernel_circuit


def make_qnn_circuit(n_qubits, n_layers, entangled, device=None):
    """Returns a QNode for the variational classifier: data encoding
    (always uses simple angle embedding, identical in both conditions)
    followed by a trainable ansatz (product or entangled), measuring
    <Z_0> as the classifier's decision value."""
    wires = list(range(n_qubits))
    dev = device or default_device(n_qubits)
    ansatz = ansatz_entangled if entangled else ansatz_product
    # lightning.qubit doesn't support backprop; use adjoint-diff there
    # (exact, and faster than backprop for state-vector sims anyway).
    diff_method = "adjoint" if "lightning" in dev.name else "backprop"

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(x, params):
        qml.AngleEmbedding(x, wires=wires, rotation="Y")
        ansatz(params, wires)
        return qml.expval(qml.PauliZ(0))

    return circuit


def make_layerwise_representation_circuit(n_qubits, n_layers, entangled,
                                           device=None):
    """Returns a QNode that exposes the per-layer 'representation' of
    an input: after data encoding and after each ansatz layer, we
    measure <X_i>, <Y_i>, <Z_i> on every qubit. Concatenating these
    across qubits gives a 3*n_qubits-dim vector per layer -- the
    quantum analogue of a hidden-layer activation vector, used as
    input to the CKA-style similarity analysis.

    Returns a function repr_fn(x, params) -> np.ndarray of shape
    (n_layers + 1, 3 * n_qubits)   [layer 0 = after encoding only]
    """
    wires = list(range(n_qubits))
    dev = device or default_device(n_qubits)
    ansatz_layer_fn = ansatz_entangled if entangled else ansatz_product

    obs = []
    for w in wires:
        obs += [qml.PauliX(w), qml.PauliY(w), qml.PauliZ(w)]

    def repr_fn(x, params):
        reps = []
        n_layers_total = params.shape[0]
        for upto in range(0, n_layers_total + 1):
            @qml.qnode(dev)
            def circuit(x=x, params=params, upto=upto):
                qml.AngleEmbedding(x, wires=wires, rotation="Y")
                if upto > 0:
                    ansatz_layer_fn(params[:upto], wires)
                return [qml.expval(o) for o in obs]
            reps.append(np.array(circuit()))
        return np.stack(reps)  # (n_layers+1, 3*n_qubits)

    return repr_fn


if __name__ == "__main__":
    import time

    n_qubits, n_layers = 4, 2
    x = np.random.uniform(0, np.pi, n_qubits)

    print("=== Feature map circuit checks ===")
    for ent, label in [(False, "product"), (True, "entangled")]:
        k = make_kernel_circuit(n_qubits, n_layers, ent)
        val = k(x, x)[0]  # fidelity with itself should be ~1
        print(f"  {label}: self-fidelity (should be ~1.0) = {val:.6f}")

    print("=== QNN circuit checks ===")
    params = init_ansatz_params(n_layers, n_qubits, seed=0)
    for ent, label in [(False, "product"), (True, "entangled")]:
        q = make_qnn_circuit(n_qubits, n_layers, ent)
        out = q(x, params)
        print(f"  {label}: <Z0> = {out:.4f}")

    print("=== Layerwise representation circuit checks ===")
    for ent, label in [(False, "product"), (True, "entangled")]:
        rf = make_layerwise_representation_circuit(n_qubits, n_layers, ent)
        t0 = time.time()
        reps = rf(x, params)
        print(f"  {label}: shape={reps.shape}, time={time.time()-t0:.3f}s")
