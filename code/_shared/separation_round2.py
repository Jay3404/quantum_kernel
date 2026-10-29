"""Training-cost and feature-map exploration with a separate validation gate."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import zz_feature_map
from qiskit_aer import AerSimulator
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from . import controlled as base
from .angle_encoding import AngleEncoding
from .full_data import dense_kernel, kernel_block, write_json
from .improved import local_lock, sha
from .separation import load_prepared

ROOT = base.ROOT


class RepeatedQuantumStates(base.QuantumStates):
    def __init__(self, alpha, reps):
        tick = time.perf_counter()
        self.backend = AerSimulator(method="statevector", precision="double", max_parallel_threads=1,
                                    max_parallel_experiments=1, runtime_parameter_bind_enable=True)
        circuit = zz_feature_map(7, reps=reps, entanglement="linear", alpha=alpha)
        circuit.save_statevector(label="state")
        self.circuit = transpile(circuit, self.backend, optimization_level=1, seed_transpiler=17)
        self.parameters = tuple(self.circuit.parameters)
        self.setup_seconds = time.perf_counter() - tick


def class_weights(y, policy):
    counts = np.bincount(y, minlength=2).astype(float)
    if not np.all(counts > 0):
        raise ValueError("Both training classes are required")
    if policy == "balanced":
        weight = len(y) / (2 * counts)
    elif policy in ("ratio_2", "ratio_5", "ratio_10"):
        weight = np.array([1., float(policy.split("_")[1])])
        weight /= np.dot(counts, weight) / len(y)
    else:
        raise ValueError("Unknown training cost policy")
    return {0: float(weight[0]), 1: float(weight[1])}


def midrank(reference, values):
    reference = np.sort(np.asarray(reference, dtype=float))
    if reference.ndim != 1 or not len(reference) or not np.isfinite(reference).all():
        raise ValueError("Finite training reference required")
    return (np.searchsorted(reference, values, "left") + np.searchsorted(reference, values, "right")) / (2 * len(reference))


def prepared_folder(output, dataset, seed):
    return Path(output) / "prepared" / dataset / f"seed_{seed:03d}"


def prepare(dataset, seed, protocol, output):
    folder = prepared_folder(output, dataset, seed)
    folder.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        arrays, context, audit = load_prepared(dataset, seed, protocol, "smoke")
        positions = np.arange(len(arrays["validation"]))
        tune, gate = train_test_split(positions, test_size=.5, stratify=arrays["y_validation"], random_state=seed + 1717)
        inputs = {k: arrays[k] for k in ("train", "risk_train", "y_train", "weights")}
        for part, index in (("tune", tune), ("gate", gate)):
            inputs[part] = arrays["validation"][index]
            inputs[f"risk_{part}"] = arrays["risk_validation"][index]
        inputs["y_tune"] = arrays["y_validation"][tune]
        source = ROOT / "code" / dataset / "results" / protocol["source_run"] / f"seed_{seed:03d}/prepared.npz"
        with np.load(source, allow_pickle=False) as original:
            reference = original["risk_train"]
        for part in ("train", "tune", "gate"):
            inputs[f"rank_{part}"] = midrank(reference, inputs[f"risk_{part}"])
        np.savez_compressed(folder / "candidate_inputs.npz", **inputs)
        np.savez_compressed(folder / "gate_labels.npz", y_gate=arrays["y_validation"][gate])
        record = {"dataset": dataset, "seed": seed, "source_audit": audit,
                  "tune_positions": tune.tolist(), "gate_positions": gate.tolist(),
                  "candidate_inputs_sha256": sha(folder / "candidate_inputs.npz"),
                  "gate_labels_sha256": sha(folder / "gate_labels.npz"),
                  "encoder": context["encoder"].parameters(), "direction": context["direction"].tolist(),
                  "risk_mean": context["risk_mean"], "gamma": context["gamma"],
                  "candidate_loaded_keys": list(inputs), "test_arrays_read": False}
        write_json(folder / "audit.json", record)
    return dataset, seed


def encoder_from(record):
    params = record["encoder"]
    return AngleEncoding(params["mode"], *(np.asarray(params[k]) for k in ("center", "scale", "minimum", "span")), params["train_rows"])


def encoded_features(arrays, record, method, strength, learner, engine):
    encoder = encoder_from(record)
    values = {}
    for part in ("train", "tune", "gate"):
        z, risk = arrays[part], arrays[f"risk_{part}"]
        if method == "feature_norm":
            z = z * (1 + strength * risk[:, None] * (7 * arrays["weights"]))
        value = encoder.transform(z)
        if method == "phase_shift":
            value += np.pi * strength * (risk - record["risk_mean"])[:, None] * np.asarray(record["direction"])
        if learner == "QSVC":
            value = engine(value)
            if method == "rank_axis":
                angle = np.pi * strength * arrays[f"rank_{part}"] / 2
                qubit = np.column_stack([np.cos(angle), np.sin(angle)])
                value = np.einsum("ni,nj->nij", value, qubit).reshape(len(value), -1)
        elif method == "rank_axis":
            value = np.column_stack([value, np.pi * strength * arrays[f"rank_{part}"]])
        if not np.isfinite(value).all():
            raise AssertionError("Nonfinite feature encoding")
        values[part] = value
    return values


def risk_kernel(left, right, learner, gamma):
    difference = np.pi * (np.asarray(left)[:, None] - np.asarray(right)[None, :])
    return np.cos(difference / 2) ** 2 if learner == "QSVC" else np.exp(-gamma * difference ** 2)


def job_folder(output, job):
    dataset, seed, learner, reps, alpha, factor = job
    name = f"reps_{reps}_alpha_{alpha:g}_gamma_{factor:g}"
    return Path(output) / "candidates" / dataset / f"seed_{seed:03d}" / learner / name


def rectangular(left, right, learner, gamma):
    result = np.empty((len(left), len(right)))
    for offset in range(0, len(left), 256):
        result[offset:offset+256] = kernel_block(left[offset:offset+256], right, learner, gamma)
    return result


def run_job(job, protocol, output, hashes):
    dataset, seed, learner, reps, alpha, factor = job
    folder = job_folder(output, job)
    folder.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1), local_lock(folder / "job.lock"):
        if (folder / "complete.json").exists():
            raise AssertionError("Do not overwrite or automatically retry a completed job")
        prepared = prepared_folder(output, dataset, seed)
        record = json.loads((prepared / "audit.json").read_text())
        if sha(prepared / "candidate_inputs.npz") != record["candidate_inputs_sha256"]:
            raise AssertionError("Candidate inputs changed")
        with np.load(prepared / "candidate_inputs.npz", allow_pickle=False) as saved:
            arrays = dict(saved)
        if any(key in arrays for key in ("y_gate", "y_test", "test", "risk_test")):
            raise AssertionError("Gate/test labels or test features reached a candidate worker")
        engine = RepeatedQuantumStates(alpha, reps) if learner == "QSVC" else None
        gamma = record["gamma"] * factor
        rows, margins = [], {"tune": [], "gate": []}
        total = time.perf_counter()
        conditions = [("baseline", 0.), *[(m, s) for m in protocol["methods"][1:] for s in protocol["lambdas"]]]
        baseline_kernels = None
        for method, strength in conditions:
            tick = time.perf_counter()
            if method == "rank_mixture":
                features_time = 0.
                kernels = {p: (1-strength)*baseline_kernels[p] + strength*risk_kernel(arrays[f"rank_{p}"], arrays["rank_train"], learner, gamma)
                           for p in ("train", "tune", "gate")}
            else:
                values = encoded_features(arrays, record, method, strength, learner, engine)
                features_time = time.perf_counter() - tick
                tick = time.perf_counter()
                kernels = {"train": dense_kernel(values["train"], learner, gamma, 1024, 10, 24)}
                for part in ("tune", "gate"):
                    kernels[part] = rectangular(values[part], values["train"], learner, gamma)
                del values
            kernel_time = time.perf_counter() - tick
            if method == "baseline":
                baseline_kernels = kernels
            for policy in protocol["weight_policies"]:
                weights = class_weights(arrays["y_train"], policy)
                for c in protocol["C_grid"]:
                    model = SVC(kernel="precomputed", C=c, class_weight=weights, cache_size=256)
                    tick = time.perf_counter()
                    model.fit(kernels["train"], arrays["y_train"])
                    fit_time = time.perf_counter() - tick
                    if model.fit_status_:
                        raise RuntimeError("SVC did not converge")
                    tick = time.perf_counter()
                    scores = {p: model.decision_function(kernels[p]) for p in margins}
                    inference_time = time.perf_counter() - tick
                    if not all(np.isfinite(v).all() for v in scores.values()):
                        raise AssertionError("Nonfinite margin")
                    threshold, _ = base.threshold_for_cost(arrays["y_tune"], scores["tune"], 0.)
                    metrics = base.metrics(arrays["y_tune"], scores["tune"], threshold)
                    rows.append({"dataset": dataset, "seed": seed, "learner": learner, "method": method, "lambda": strength,
                        "reps": reps, "alpha": alpha, "gamma_factor": factor, "C": c, "weight_policy": policy,
                        "weight_0": weights[0], "weight_1": weights[1], "threshold": threshold,
                        **{f"tune_{k}": v for k,v in metrics.items()}, "fit_rows": len(arrays["y_train"]),
                        "ensemble_members": 1, "model_fit_calls": 1, "n_support": int(model.n_support_.sum()),
                        "feature_seconds": features_time, "kernel_seconds": kernel_time, "fit_seconds": fit_time,
                        "inference_seconds": inference_time, "train_kernel_entries": len(arrays["y_train"])**2,
                        "tune_kernel_entries": len(arrays["y_tune"])*len(arrays["y_train"]),
                        "gate_kernel_entries": len(arrays["gate"])*len(arrays["y_train"]),
                        "quantum_qubits": (8 if method == "rank_axis" else 7) if engine else 0,
                        "circuit_setup_seconds": engine.setup_seconds if engine else 0.,
                        "risk_kernel_qubits": 1 if learner == "QSVC" and method == "rank_mixture" else 0,
                        "aer_simulated_qubits": 7 if engine else 0})
                    for part in margins:
                        margins[part].append(scores[part])
                    del model
            if method != "baseline":
                del kernels
            gc.collect()
        np.savez_compressed(folder / "margins.npz", **{k: np.asarray(v) for k,v in margins.items()})
        write_json(folder / "complete.json", {"job": list(job), "source_sha256": hashes, "protocol": protocol,
            "prepared_audit_sha256": sha(prepared / "audit.json"), "archive_sha256": sha(folder / "margins.npz"),
            "trials": rows, "elapsed_seconds": time.perf_counter()-total, "gate_labels_read": False, "test_arrays_read": False})
        print("completed", *job, flush=True)
    return job


def order(row):
    return (row["tune_cost"], row["lambda"], row["C"], row["reps"], row["alpha"], row["gamma_factor"], row["weight_policy"])


def family_pool(rows, family, learner):
    baseline = [r for r in rows if r["method"] == "baseline"]
    if family == "reference":
        return [r for r in baseline if r["weight_policy"] == "balanced" and (learner == "RBF-SVM" or (r["reps"] == 1 and r["alpha"] in (.5, 1.)))]
    if family == "cost_only":
        return [r for r in baseline if learner == "RBF-SVM" or (r["reps"] == 1 and r["alpha"] in (.5, 1.))]
    if family == "depth_only":
        return family_pool(rows, "reference", learner) if learner == "RBF-SVM" else [r for r in baseline if r["reps"] == 2 and r["weight_policy"] == "balanced"]
    if family == "tuned_backbone":
        return baseline
    if family in ("phase_shift", "feature_norm", "rank_axis", "rank_mixture"):
        return [r for r in rows if r["method"] == family]
    raise ValueError("Unknown candidate family")
