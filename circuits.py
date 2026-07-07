

import pennylane as qml
import numpy as np


def default_device(n_qubits):
    
    try:
        return qml.device("lightning.qubit", wires=n_qubits)
    except qml.DeviceError:
        return qml.device("default.qubit", wires=n_qubits)




def feature_map_product(x, wires, n_layers=2):

    n = len(wires)
    for layer in range(n_layers):
        for i, w in enumerate(wires):
            xi = x[i % len(x)]
            qml.RZ(xi, wires=w)
            qml.RY(xi, wires=w)
            qml.RZ((layer + 1) * xi, wires=w)


def feature_map_entangled(x, wires, n_layers=2):
  
    n = len(wires)
    for layer in range(n_layers):
        for i, w in enumerate(wires):
            xi = x[i % len(x)]
            qml.RZ(xi, wires=w)
            qml.RY(xi, wires=w)
            qml.RZ((layer + 1) * xi, wires=w)
    
        for i in range(n):
            j = (i + 1) % n
            xi, xj = x[i % len(x)], x[j % len(x)]
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ((np.pi - xi) * (np.pi - xj), wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])



def ansatz_product(params, wires):

    n_layers, n = params.shape[0], len(wires)
    for layer in range(n_layers):
        for i, w in enumerate(wires):
            qml.Rot(*params[layer, i], wires=w)


def ansatz_entangled(params, wires):
   
    n_layers, n = params.shape[0], len(wires)
    for layer in range(n_layers):
        for i, w in enumerate(wires):
            qml.Rot(*params[layer, i], wires=w)
        for i in range(n):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


def init_ansatz_params(n_layers, n_qubits, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 3))




def make_kernel_circuit(n_qubits, n_layers, entangled, device=None):
 
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
