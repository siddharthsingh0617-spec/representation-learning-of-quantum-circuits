

import argparse
import json
import os
import numpy as np

from .data import load_dataset
from .verify_entanglement import run_manipulation_check
from .quantum_kernel import train_quantum_kernel_svm
from .qnn_training import train_qnn
from .classical_baselines import train_classical_baselines
from .cka_analysis import (layerwise_self_similarity,
                            cross_condition_similarity,
                            trained_vs_random_similarity)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()
                if k not in ("model", "K_train", "K_test")}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run_dir(run_id, results_root="results"):
    d = os.path.join(results_root, run_id)
    os.makedirs(d, exist_ok=True)
    return d


def save_json(run_id, name, obj, results_root="results"):
    path = os.path.join(run_dir(run_id, results_root), f"{name}.json")
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)
    print(f"Saved {path}")


def load_json(run_id, name, results_root="results"):
    path = os.path.join(run_dir(run_id, results_root), f"{name}.json")
    with open(path) as f:
        return json.load(f)


def phase_manip(run_id, n_qubits, n_layers, **kw):
    res = run_manipulation_check(n_qubits=n_qubits, n_layers=n_layers,
                                  n_samples=20)
    save_json(run_id, "manip", res)


def phase_kernel_svm(run_id, dataset, n_qubits, n_layers, n_seeds,
                      kernel_svm_subsample, kernel_svm_test_subsample,
                      seed_start=0, seed_end=None, **kw):
    data = load_dataset(dataset, n_qubits=n_qubits)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    out_name = f"kernel_svm_seeds_{seed_start}_{seed_end}"
    results = {"product": [], "entangled": []}
    seed_end = seed_end if seed_end is not None else n_seeds
    for seed in range(seed_start, seed_end):
        rng = np.random.default_rng(seed)
        train_idx = rng.choice(len(X_train), size=min(
            kernel_svm_subsample, len(X_train)), replace=False)
        test_idx = rng.choice(len(X_test), size=min(
            kernel_svm_test_subsample, len(X_test)), replace=False)
        Xtr, ytr = X_train[train_idx], y_train[train_idx]
        Xte, yte = X_test[test_idx], y_test[test_idx]
        print(f"-- seed {seed} --")
        for entangled in [False, True]:
            res = train_quantum_kernel_svm(
                Xtr, ytr, Xte, yte, n_qubits, n_layers, entangled)
            res["seed"] = seed
            key = "entangled" if entangled else "product"
            results[key].append(to_jsonable(res))
    save_json(run_id, out_name, results)


def phase_qnn(run_id, dataset, n_qubits, n_layers, qnn_epochs,
              seed_start=0, seed_end=None, n_seeds=5, **kw):
    data = load_dataset(dataset, n_qubits=n_qubits)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    out_name = f"qnn_seeds_{seed_start}_{seed_end}"
    results = {"product": [], "entangled": []}
    params_out = {"product": None, "entangled": None}
    seed_end = seed_end if seed_end is not None else n_seeds
    for seed in range(seed_start, seed_end):
        print(f"-- seed {seed} --")
        for entangled in [False, True]:
            res = train_qnn(
                X_train, y_train, X_test, y_test, n_qubits, n_layers,
                entangled, n_epochs=qnn_epochs, seed=seed, verbose=False)
            key = "entangled" if entangled else "product"
            if seed == seed_start:
                params_out[key] = res["params"].tolist()
            print(f"  [{key:9s}] test_acc={res['test_accuracy']:.4f}  "
                  f"test_auc={res['test_auc']:.4f}  time={res['time_sec']:.1f}s")
            results[key].append(to_jsonable(res))
    save_json(run_id, out_name, results)
    save_json(run_id, f"qnn_params_seed{seed_start}", params_out)


def phase_classical(run_id, dataset, n_qubits, **kw):
    data = load_dataset(dataset, n_qubits=n_qubits)
    res = train_classical_baselines(data["X_train"], data["y_train"],
                                     data["X_test"], data["y_test"])
    save_json(run_id, "classical_baselines", res)


def phase_cka_self(run_id, dataset, n_qubits, n_layers, n_seeds, **kw):
    data = load_dataset(dataset, n_qubits=n_qubits)
    X_cka = data["X_test"]
    out = {}
    for entangled in [False, True]:
        key = "entangled" if entangled else "product"
        res = layerwise_self_similarity(X_cka, n_qubits, n_layers,
                                         entangled, n_seeds=n_seeds)
        out[key] = to_jsonable(res)
        print(f"{key}: mean self-CKA per layer = "
              f"{res['mean_self_cka_per_layer']}")
    save_json(run_id, "cka_self", out)


def phase_cka_cross(run_id, dataset, n_qubits, n_layers, n_seeds, **kw):
    data = load_dataset(dataset, n_qubits=n_qubits)
    X_cka = data["X_test"]
    records = cross_condition_similarity(X_cka, n_qubits, n_layers,
                                          n_seeds=n_seeds)
    save_json(run_id, "cka_cross", records)


def phase_cka_trained_random(run_id, dataset, n_qubits, n_layers,
                              n_seeds, **kw):
    data = load_dataset(dataset, n_qubits=n_qubits)
    X_cka = data["X_test"]
    params = load_json(run_id, "qnn_params_seed0")
    out = {}
    for entangled in [False, True]:
        key = "entangled" if entangled else "product"
        p = np.array(params[key])
        cka_per_layer = trained_vs_random_similarity(
            X_cka, p, n_qubits, n_layers, entangled,
            n_random_seeds=n_seeds)
        out[key] = cka_per_layer
        print(f"{key}: {cka_per_layer}")
    save_json(run_id, "cka_trained_random", out)


PHASES = {
    "manip": phase_manip,
    "kernel_svm": phase_kernel_svm,
    "qnn": phase_qnn,
    "classical": phase_classical,
    "cka_self": phase_cka_self,
    "cka_cross": phase_cka_cross,
    "cka_trained_random": phase_cka_trained_random,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("phase", choices=list(PHASES.keys()))
    parser.add_argument("--dataset", default="iris")
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--kernel_svm_subsample", type=int, default=60)
    parser.add_argument("--kernel_svm_test_subsample", type=int, default=30)
    parser.add_argument("--qnn_epochs", type=int, default=60)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=None)
    args = parser.parse_args()

    PHASES[args.phase](**vars(args))
