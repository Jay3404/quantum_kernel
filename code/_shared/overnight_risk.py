"""Nested risk/metric exploration with exact final SVMs and sealed test labels."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import gc
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.linalg import eigh
from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import LedoitWolf
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
from qiskit import transpile
from qiskit.circuit.library import zz_feature_map, z_feature_map
from qiskit_aer import AerSimulator

from . import controlled as base
from .angle_encoding import AngleEncoding
from .full_data import available_memory, write_json
from .improved import local_lock, sha
from .quantum_ten import RiskStates, fit_alignment
from .separation_round2 import midrank
from .single_811 import prepare_arrays

ROOT = base.ROOT
PARTS = ("train", "tune", "gate")
METRICS = {"lda_stretch", "within_whiten", "nca_metric", "pls_metric", "supervised_raw7", "lda_axis", "lmnn_metric"}
CIRCUITS = {"zz_phase", "reupload", "iqp_coupling", "aligned_risk"}
MIXTURES = {"rank_mixture", "logit_mixture", "uncertainty_mixture"}


class BackboneStates(base.QuantumStates):
    def __init__(self, alpha, reps, entangled=True):
        tick = time.perf_counter()
        self.backend = AerSimulator(method="statevector", precision="double", max_parallel_threads=1,
            max_parallel_experiments=1, runtime_parameter_bind_enable=True)
        circuit = zz_feature_map(7, reps=reps, entanglement="linear", alpha=alpha) if entangled else z_feature_map(7, reps=reps, alpha=alpha)
        circuit.save_statevector(label="state")
        self.circuit = transpile(circuit, self.backend, optimization_level=1, seed_transpiler=17)
        self.parameters = tuple(self.circuit.parameters)
        self.setup_seconds = time.perf_counter() - tick


def raw_teacher(X, y, split, seed, boosted=False):
    def estimator(offset, training_y):
        learner = XGBClassifier(n_estimators=160, max_depth=3, learning_rate=.05, n_jobs=1, tree_method="hist",
            eval_metric="logloss", random_state=offset, scale_pos_weight=float(np.sum(training_y == 0) / np.sum(training_y == 1))) if boosted else LogisticRegression(C=1., class_weight="balanced", max_iter=3000, random_state=offset)
        pipeline = Pipeline([("features", base.make_features(X)[:-1]), ("model", learner)])
        return pipeline if boosted else CalibratedClassifierCV(pipeline, method="sigmoid",
            cv=StratifiedKFold(3, shuffle=True, random_state=offset), ensemble=False, n_jobs=1)
    fit = np.asarray(split["train"])
    risk = np.zeros(len(fit))
    for index, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(fit, y[fit])):
        model = estimator(seed + index + 31, y[fit[a]])
        model.fit(X.iloc[fit[a]], y[fit[a]])
        risk[b] = model.predict_proba(X.iloc[fit[b]])[:, 1]
    model = estimator(seed + 37, y[fit])
    model.fit(X.iloc[fit], y[fit])
    return {"train": risk, "tune": model.predict_proba(X.iloc[split["validation"]])[:, 1],
        "gate": model.predict_proba(X.iloc[split["test"]])[:, 1]}


def prepared_folder(output, dataset, seed):
    return Path(output) / "prepared" / dataset / f"seed_{seed:04d}"


def job_folder(output, job):
    dataset, seed, learner, method, arm = job
    return Path(output) / "candidates" / dataset / f"seed_{seed:04d}" / learner / method / arm


def subset(indices, y, cap, seed):
    indices = np.asarray(indices)
    return indices if len(indices) <= cap else train_test_split(indices, train_size=cap, stratify=y[indices], random_state=seed)[0]


def fixed_test_split(y, seed, dataset):
    offsets = {"german_numeric": 0, "south_german": 1, "taiwan": 2, "hmeq": 3}
    development, test = train_test_split(np.arange(len(y)), test_size=.1, stratify=y,
        random_state=20261018 + offsets.get(dataset, 19))
    train, validation = train_test_split(development, test_size=len(test), stratify=y[development], random_state=seed + 103)
    return {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()}


def make_splits(y, seed, stage, cap, dataset="german_numeric"):
    outer = fixed_test_split(y, seed, dataset)
    if stage == "full":
        return outer, outer
    fit, hold = train_test_split(np.asarray(outer["train"]), test_size=.25, stratify=y[outer["train"]], random_state=seed + 211)
    tune, gate = train_test_split(hold, test_size=.5, stratify=y[hold], random_state=seed + 307)
    fit = subset(fit, y, cap, seed + 401)
    return outer, {"train": fit.tolist(), "validation": tune.tolist(), "test": gate.tolist()}


def prepare(dataset, seed, stage, protocol, output):
    folder = prepared_folder(output, dataset, seed)
    folder.mkdir(parents=True, exist_ok=True)
    with local_lock(folder / "prepare.lock"), threadpool_limits(limits=1):
        if (folder / "audit.json").exists():
            raise AssertionError("Existing preparation retained")
        X, y, source = base.load_data(dataset, str(ROOT / "data"))
        cap = protocol["discovery_train_cap"] if stage == "discovery" else protocol["confirmation_train_cap"]
        outer, split = make_splits(y, seed, stage, cap, dataset)
        old, times = prepare_arrays(X, y, split, seed)
        arrays = {"train": old["train"], "tune": old["validation"], "gate": old["test"],
            "y_train": old["y_train"], "y_tune": old["y_validation"], "weights": old["weights"]}
        features = base.make_features(X).fit(X.iloc[split["train"]])
        np.testing.assert_allclose(features.transform(X.iloc[split["train"]]), arrays["train"], atol=1e-10)
        for part, name in (("train", "train"), ("tune", "validation"), ("gate", "test")):
            arrays[f"risk_{part}"] = old[f"risk_{name}"]
            arrays[f"rank_{part}"] = midrank(old["risk_train"], arrays[f"risk_{part}"])
            arrays[f"raw_{part}"] = features[:-1].transform(X.iloc[split[name]])
        tick = time.perf_counter()
        for prefix, boosted in (("raw_lr", False), ("gbdt", True)):
            for part, risk in raw_teacher(X, y, split, seed, boosted).items():
                arrays[f"{prefix}_{part}"] = risk
            for part in PARTS:
                arrays[f"{prefix}_rank_{part}"] = midrank(arrays[f"{prefix}_train"], arrays[f"{prefix}_{part}"])
        times["raw_teachers_fit_and_prediction_seconds"] = time.perf_counter() - tick
        encoder = AngleEncoding.fit(arrays["train"], "linear_no_clip")
        angles = encoder.transform(arrays["train"])
        direction = (angles - angles.mean(0)).T @ (arrays["risk_train"] - arrays["risk_train"].mean())
        direction /= max(np.linalg.norm(direction), 1e-12)
        if not np.linalg.norm(direction):
            direction = np.ones(7) / np.sqrt(7)
        np.savez_compressed(folder / "inputs.npz", **arrays)
        write_json(folder / "split.json", {"outer": outer, "actual": split})
        if stage != "full":
            np.savez_compressed(folder / "confirmation_labels.npz", y_gate=y[split["test"]])
        record = {"dataset": dataset, "seed": seed, "stage": stage, "source": source,
            "input_sha256": sha(folder / "inputs.npz"), "split_sha256": sha(folder / "split.json"),
            "encoder": encoder.parameters(), "direction": direction.tolist(), "times": times,
            "train_rows": len(arrays["train"]), "tune_rows": len(arrays["tune"]), "gate_rows": len(arrays["gate"]),
            "training_row_ids": split["train"], "outer_test_labels_saved": False,
            "confirmation_labels_sha256": sha(folder / "confirmation_labels.npz") if stage != "full" else None}
        write_json(folder / "audit.json", record)
    return dataset, seed


def load_inputs(output, dataset, seed):
    folder = prepared_folder(output, dataset, seed)
    audit = json.loads((folder / "audit.json").read_text())
    assert sha(folder / "inputs.npz") == audit["input_sha256"]
    with np.load(folder / "inputs.npz", allow_pickle=False) as saved:
        arrays = dict(saved)
    assert not {"y_gate", "y_test", "test"} & set(arrays)
    return arrays, audit


def risk_values(arrays, arm, seed):
    probabilities, ranks = {}, {}
    for index, part in enumerate(PARTS):
        p, q = arrays[f"risk_{part}"].copy(), arrays[f"rank_{part}"].copy()
        if arm == "shuffled":
            positions = np.random.default_rng(seed + 7919 + index).permutation(len(p))
            p, q = p[positions], q[positions]
        elif arm == "constant":
            p[:], q[:] = arrays["risk_train"].mean(), arrays["rank_train"].mean()
        probabilities[part], ranks[part] = p, q
    return probabilities, ranks


def angle_encoder(record):
    p = record["encoder"]
    return AngleEncoding(p["mode"], *(np.asarray(p[k]) for k in ("center", "scale", "minimum", "span")), p["train_rows"])


def metric_embedding(arrays, method, q, arm, seed, protocol):
    scaler = StandardScaler().fit(arrays["train"])
    x = {p: scaler.transform(arrays[p]) for p in PARTS}
    y = arrays["y_train"].copy()
    if arm == "shuffled" and method in METRICS:
        y = np.random.default_rng(seed + 9173).permutation(y)
    record = {"method": method, "training_labels_shuffled": arm == "shuffled" and method in METRICS}
    if method in ("lda_stretch", "lda_axis"):
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(x["train"], y)
        direction = model.coef_[0] / max(np.linalg.norm(model.coef_[0]), 1e-12)
        values = {p: v + 2 * (v @ direction)[:, None] * direction for p, v in x.items()} if method == "lda_stretch" else {p: np.column_stack([v[:, :6], model.decision_function(v)]) for p, v in x.items()}
    elif method == "within_whiten":
        residual = x["train"] - np.vstack([x["train"][y == c].mean(0) for c in (0, 1)])[y]
        covariance = LedoitWolf().fit(residual).covariance_
        eig, vectors = eigh(covariance)
        transform = vectors @ np.diag(1 / np.sqrt(np.maximum(eig, 1e-6))) @ vectors.T
        values = {p: v @ transform for p, v in x.items()}
    elif method == "pls_metric":
        model = PLSRegression(n_components=7, scale=False).fit(arrays["raw_train"], y)
        values = {p: model.transform(arrays[f"raw_{p}"]) for p in PARTS}
    elif method == "supervised_raw7":
        scores = mutual_info_classif(arrays["raw_train"], y, random_state=seed)
        columns = np.argsort(-scores, kind="stable")[:7]
        values = {p: arrays[f"raw_{p}"][:, columns] for p in PARTS}
        record["columns"] = columns.tolist()
    else:
        if method == "nca_risk":
            x = {p: np.column_stack([v, np.pi * (q[p] - .5)]) for p, v in x.items()}
        positions = subset(np.arange(len(y)), y, protocol["metric_train_cap"], seed + 509)
        if method == "lmnn_metric":
            from metric_learn import LMNN
            model = LMNN(n_components=7, n_neighbors=3, min_iter=20, max_iter=80, random_state=seed)
        else:
            model = NeighborhoodComponentsAnalysis(n_components=7, init="identity", max_iter=60, tol=1e-5, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x["train"][positions], y[positions])
        record["fit_warnings"] = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
        values = {p: model.transform(v) for p, v in x.items()}
        record.update(fit_positions=positions.tolist(), iterations=int(getattr(model, "n_iter_", 0)))
    encoder = AngleEncoding.fit(values["train"], "linear_no_clip")
    record["encoder"] = encoder.parameters()
    return {p: encoder.transform(v) for p, v in values.items()}, record


def primitive_kernel(left, right, kind, gamma, gpu_device=None):
    use_gpu = gpu_device is not None and len(left) * len(right) >= 1_000_000
    if use_gpu:
        import cupy as cp
        lock = ROOT / ".overnight-locks" / f"gpu_{gpu_device}.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a") as handle, cp.cuda.Device(gpu_device):
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                right_gpu = cp.asarray(right)
                result = np.empty((len(left), len(right)), dtype=np.float64)
                for start in range(0, len(left), 1024):
                    a = cp.asarray(left[start:start + 1024])
                    if kind == "fidelity":
                        block = cp.clip(cp.abs(a @ right_gpu.conj().T) ** 2, 0, 1)
                    else:
                        distance = cp.maximum(cp.sum(a * a, 1)[:, None] + cp.sum(right_gpu * right_gpu, 1)[None, :] - 2 * a @ right_gpu.T, 0.)
                        block = cp.exp(-gamma * distance)
                    result[start:start + len(a)] = cp.asnumpy(block)
                    del a, block
                del right_gpu
                cp.get_default_memory_pool().free_all_blocks()
                return result
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
    result = np.empty((len(left), len(right)), dtype=np.float64)
    for start in range(0, len(left), 512):
        a = left[start:start + 512]
        if kind == "fidelity":
            value = base.fidelity(a, right)
        else:
            distance = np.maximum(np.sum(a * a, 1)[:, None] + np.sum(right * right, 1)[None, :] - 2 * a @ right.T, 0.)
            value = np.exp(-gamma * distance)
        result[start:start + len(a)] = value
    return result


def representations(arrays, record, method, learner, arm, seed, protocol):
    if method.startswith("raw_lr_") or method.startswith("gbdt_"):
        prefix = "raw_lr" if method.startswith("raw_lr_") else "gbdt"
        arrays = {**arrays, **{f"risk_{p}": arrays[f"{prefix}_{p}"] for p in PARTS},
            **{f"rank_{p}": arrays[f"{prefix}_rank_{p}"] for p in PARTS}}
    p, q = risk_values(arrays, arm, seed)
    angles = {part: angle_encoder(record).transform(arrays[part]) for part in PARTS}
    metric, metric_record = (metric_embedding(arrays, method, q, arm, seed, protocol)
        if method in METRICS or method == "nca_risk" else (None, None))
    aligned = None
    if method == "aligned_risk":
        altered = {**arrays, **{f"rank_{part}": q[part] for part in PARTS}}
        alpha, gain, history = fit_alignment(altered, {**record, "risk_mean": float(arrays["risk_train"].mean())}, q,
            1., seed, protocol, entangled=arm != "separable", classical=learner == "RBF-SVM")
        aligned = (alpha, gain, history)
    return {"p": p, "q": q, "angles": angles, "metric": metric, "metric_record": metric_record, "aligned": aligned}


def make_representation(arrays, record, context, method, learner, arm, strength, geometry):
    tick = time.perf_counter()
    p, q = context["p"], context["q"]
    method = method.removeprefix("raw_lr_").removeprefix("gbdt_")
    values = {part: value.copy() for part, value in context["angles"].items()}
    kind = "fidelity" if learner == "QSVC" else "rbf"
    kernel_mode = "base"
    risk_coordinate = q
    logit = {part: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6))) / 4 for part, v in p.items()}
    direction = np.asarray(record["direction"])
    for part in PARTS:
        z = arrays[part]
        if method in ("multiply", "feature_specific", "feature_normalized"):
            weights = np.ones(7) if method == "multiply" else arrays["weights"] * (7 if method == "feature_normalized" else 1)
            values[part] = angle_encoder(record).transform(z * (1 + strength * p[part][:, None] * weights))
        elif method in ("rank_shift", "logit_shift"):
            coordinate = q[part] - .5 if method == "rank_shift" else logit[part]
            values[part] += np.pi * strength * coordinate[:, None] * direction
        elif method == "residual_risk":
            values[part] = values[part] - strength * ((values[part] - context["angles"]["train"].mean(0)) @ direction)[:, None] * direction
        elif context["metric"] is not None:
            values[part] = context["metric"][part]
    circuit_details = []
    state_times = {part: 0. for part in PARTS}
    base_values = None
    if context["metric"] is not None or method == "residual_risk":
        kernel_mode = "metric_mix"
        base_values = context["angles"]
    if method in MIXTURES:
        kernel_mode = "risk_mix"
    if method in ("rank_product", "logit_product", "hellinger_product", "soft_gate"):
        kernel_mode = method
    if method.startswith("logit"):
        risk_coordinate = logit
    if method == "uncertainty_mixture":
        risk_coordinate = {part: np.column_stack([q[part], 4 * p[part] * (1 - p[part])]) for part in PARTS}
    alpha = geometry["alpha"]
    gain = 1.
    if context["aligned"] is not None:
        alpha, gain, _ = context["aligned"]
    if learner == "QSVC":
        # Each baseline matches the method's zero-strength circuit.
        engine_method = method if method in CIRCUITS - {"aligned_risk"} else "reupload" if method == "aligned_risk" else "standard"
        if engine_method == "standard":
            engine = BackboneStates(alpha, geometry["reps"], arm != "separable")
        else:
            engine = RiskStates(engine_method, arrays["weights"], direction, arm != "separable", alpha)
        if hasattr(engine, "qubits"):
            circuit_details.append({"qubits": engine.qubits, "depth": engine.depth, "gates": engine.gates})
        else:
            circuit_details.append({"qubits": 7, "depth": engine.circuit.depth(), "gates": dict(engine.circuit.count_ops())})
        for part in PARTS:
            state_tick = time.perf_counter()
            input_value = np.column_stack([values[part], np.pi * strength * gain * (q[part] - .5)]) if isinstance(engine, RiskStates) else values[part]
            values[part] = engine(input_value)
            state_times[part] += time.perf_counter() - state_tick
        if base_values is not None:
            for part, value in base_values.items():
                state_tick = time.perf_counter()
                base_values[part] = engine(np.column_stack([value, np.zeros(len(value))])) if isinstance(engine, RiskStates) else engine(value)
                state_times[part] += time.perf_counter() - state_tick
    elif method in CIRCUITS:
        # Declared classical analog: directed risk shift plus explicit risk axis.
        values = {part: np.column_stack([v + np.pi * strength * gain * (q[part] - .5)[:, None] * direction,
            np.pi * strength * gain * (q[part] - .5)]) for part, v in values.items()}
    if method in ("rank_append", "logit_append", "polynomial_risk", "residual_risk"):
        coordinate = logit if method == "logit_append" else {part: q[part] - .5 for part in PARTS}
        if learner == "QSVC":
            for part in PARTS:
                angle = np.pi * strength * coordinate[part]
                aux = np.column_stack([np.cos(angle / 2), np.sin(angle / 2)])
                values[part] = np.einsum("ni,nj->nij", values[part], aux).reshape(len(angle), -1)
            circuit_details[0]["qubits"] += 1
            if method == "polynomial_risk":
                kernel_mode = "rank_product"
        else:
            values = {part: np.column_stack([v, np.pi * strength * coordinate[part],
                strength * coordinate[part][:, None] * context["angles"][part]]) if method == "polynomial_risk" else np.column_stack([v, np.pi * strength * coordinate[part]]) for part, v in values.items()}
    gamma = geometry["gamma_factor"] / max(7 * context["angles"]["train"].var(), 1e-12) if learner == "RBF-SVM" else 1.
    if context["aligned"] is not None and learner == "RBF-SVM":
        gamma *= alpha
    return {"values": values, "base_values": base_values, "kind": kind, "gamma": float(gamma),
        "mode": kernel_mode, "strength": strength, "risk": risk_coordinate, "p": p, "q": q,
        "details": {"representation_seconds": time.perf_counter() - tick, "circuits": circuit_details,
            "state_seconds_by_part": state_times,
            "metric_fit": context["metric_record"], "alignment": context["aligned"][2] if context["aligned"] else None}}


def unpack_representation(value):
    return value[0] if isinstance(value, tuple) else value


def representation_kernel(rep, left, right, gpu_device=None):
    lp, li = left
    rp, ri = right
    a, b = rep["values"][lp][li], rep["values"][rp][ri]
    kernel = primitive_kernel(a, b, rep["kind"], rep["gamma"], gpu_device)
    mode, strength = rep["mode"], rep["strength"]
    if mode == "base":
        return kernel
    if mode == "metric_mix":
        base_kernel = primitive_kernel(rep["base_values"][lp][li], rep["base_values"][rp][ri], rep["kind"], rep["gamma"], gpu_device)
        return (1 - strength) * base_kernel + strength * kernel
    u, v = rep["risk"][lp][li], rep["risk"][rp][ri]
    difference = np.asarray(u)[:, None] - np.asarray(v)[None, :]
    squared = np.sum(difference ** 2, axis=2) if difference.ndim == 3 else difference ** 2
    risk_kernel = np.exp(-np.pi ** 2 * squared)
    if mode == "rank_product" or mode == "logit_product":
        return kernel * np.exp(-np.pi ** 2 * strength ** 2 * squared)
    if mode == "hellinger_product":
        p, q = rep["p"][lp][li], rep["p"][rp][ri]
        affinity = (np.sqrt(p[:, None] * q[None, :]) + np.sqrt((1 - p[:, None]) * (1 - q[None, :]))) ** 2
        return kernel * ((1 - strength) + strength * affinity)
    if mode == "soft_gate":
        p, q = rep["q"][lp][li], rep["q"][rp][ri]
        left_gate, right_gate = np.column_stack([1 - p, p]), np.column_stack([1 - q, q])
        left_gate /= np.linalg.norm(left_gate, axis=1, keepdims=True)
        right_gate /= np.linalg.norm(right_gate, axis=1, keepdims=True)
        return kernel * ((1 - strength) + strength * (left_gate @ right_gate.T))
    if mode == "risk_mix":
        return (1 - strength) * kernel + strength * risk_kernel
    raise ValueError(mode)


def nystrom_factor(kernel):
    eig, vectors = eigh((kernel + kernel.T) / 2)
    if eig.min() < -1e-7:
        raise AssertionError("Non-PSD landmark kernel")
    keep = eig > max(eig.max() * 1e-10, 1e-10)
    if not keep.any():
        raise AssertionError("Empty Nystrom feature space")
    return vectors[:, keep] / np.sqrt(eig[keep]), {"rank": int(keep.sum()), "min_eigenvalue": float(eig.min()), "max_eigenvalue": float(eig.max())}


def select_trial(rows, positive=None):
    pool = rows if positive is None else [row for row in rows if (row["lambda"] > 0) == positive]
    if not pool:
        raise AssertionError("Missing planned trial family")
    return min(pool, key=lambda r: (r["tune_cost"], r["lambda"], r["C"], r["geometry"]))


def geometry_diagnostic(rep, arrays, part="gate", cap=256):
    n = min(cap, len(arrays[part]))
    positions = np.linspace(0, len(arrays[part]) - 1, n, dtype=int)
    kernel = representation_kernel(rep, (part, positions), (part, positions))
    return {"positions": positions.tolist(), "kernel": kernel}


def run_job(job, stage, protocol, output, hashes):
    dataset, seed, learner, method, arm = job
    folder = job_folder(output, job)
    folder.mkdir(parents=True, exist_ok=True)
    with local_lock(folder / "job.lock"), threadpool_limits(limits=1):
        if stage == "full" and (Path(output) / "test_release.json").exists():
            raise AssertionError("No new fitting after full test release")
        if (folder / "complete.json").exists():
            raise AssertionError("Completed job retained; no silent retry")
        arrays, record = load_inputs(output, dataset, seed)
        if method == "reference":
            return run_reference(job, arrays, record, folder, hashes)
        started = time.perf_counter()
        context = representations(arrays, record, method, learner, arm, seed, protocol)
        strengths = [0.] if method == "baseline" else [0., *(protocol["lambdas"] if stage == "full" else protocol["smoke_lambdas"])]
        all_rows, scores, diagnostics = [], {"tune": [], "gate": []}, {}
        approximate = stage == "full" and len(arrays["train"]) > protocol["exact_search_max_rows"]
        landmarks = subset(np.arange(len(arrays["y_train"])), arrays["y_train"], protocol["nystrom_landmarks"], seed + 701)
        gpu = protocol["gpu_devices"][(seed // 10) % len(protocol["gpu_devices"])] if protocol["gpu_devices"] else None
        for geometry_id, geometry in enumerate(protocol["geometries"]):
            for strength in strengths:
                tick = time.perf_counter()
                rep = unpack_representation(make_representation(arrays, record, context, method, learner, arm, strength, geometry))
                kernel_times = {}
                kernels = {}
                if approximate:
                    gram = representation_kernel(rep, ("train", landmarks), ("train", landmarks), gpu)
                    factor, spectrum = nystrom_factor(gram)
                    for part in PARTS:
                        t = time.perf_counter()
                        kernels[part] = representation_kernel(rep, (part, slice(None)), ("train", landmarks), gpu) @ factor
                        kernel_times[part] = time.perf_counter() - t
                else:
                    spectrum = None
                    needed = sum(len(arrays[p]) for p in PARTS) * len(arrays["train"]) * 8
                    if available_memory() and available_memory() < needed * 2 + 24 * 2 ** 30:
                        raise MemoryError("Dense kernel memory reserve unavailable")
                    for part in PARTS:
                        t = time.perf_counter()
                        kernels[part] = representation_kernel(rep, (part, slice(None)), ("train", slice(None)), gpu)
                        kernel_times[part] = time.perf_counter() - t
                for c in protocol["C_grid"]:
                    model = LinearSVC(C=c, class_weight="balanced", dual=False, max_iter=5000, tol=1e-5, random_state=seed) if approximate else SVC(kernel="precomputed", C=c, class_weight="balanced", cache_size=256)
                    t = time.perf_counter()
                    model.fit(kernels["train"], arrays["y_train"])
                    fit_seconds = time.perf_counter() - t
                    if (approximate and model.n_iter_ >= 5000) or (not approximate and model.fit_status_):
                        raise RuntimeError("SVM search candidate did not converge")
                    prediction_times, margins = {}, {}
                    for part in ("tune", "gate"):
                        t = time.perf_counter()
                        margins[part] = model.decision_function(kernels[part])
                        prediction_times[part] = time.perf_counter() - t
                    threshold, _ = base.threshold_for_cost(arrays["y_tune"], margins["tune"], 0.)
                    metrics = base.metrics(arrays["y_tune"], margins["tune"], threshold)
                    index = len(all_rows)
                    all_rows.append({"dataset": dataset, "seed": seed, "learner": learner, "method": method, "arm": arm,
                        "lambda": strength, "C": c, "geometry": geometry_id, "threshold": threshold,
                        **{f"tune_{k}": v for k, v in metrics.items()}, "fit_rows": len(arrays["y_train"]),
                        "approximate_search": approximate, "fit_seconds": fit_seconds,
                        "kernel_seconds": kernel_times, "inference_seconds": prediction_times,
                        "training_seconds": fit_seconds + kernel_times["train"] + rep["details"]["state_seconds_by_part"]["train"],
                        "test_inference_seconds": kernel_times["gate"] + prediction_times["gate"] + rep["details"]["state_seconds_by_part"]["gate"],
                        "train_kernel_seconds": kernel_times["train"], "test_kernel_seconds": kernel_times["gate"],
                        "representation_details": rep["details"], "nystrom_spectrum": spectrum,
                        "score_index": index, "model_fit_calls": 1, "ensemble_members": 1})
                    for part in scores:
                        scores[part].append(margins[part])
                    del model
                checkpoint(folder, all_rows, scores)
                del kernels, rep
                gc.collect()
        choices = {"baseline": select_trial(all_rows, False)}
        if method != "baseline":
            choices["proposed"] = select_trial(all_rows, True)
        write_json(folder / "parameter_lock.json", {"choices": choices, "rule": "Tune labels only; exact refit parameters locked before exact model fit"})
        exact_scores = {}
        for role, choice in choices.items():
            if approximate:
                row, margins, diagnostic = exact_fit(arrays, record, context, job, choice, protocol, gpu)
                row["selection_proxy_trial"] = choice["score_index"]
                diagnostics[f"exact_{role}"] = diagnostic
            else:
                row = {**choice, "final_exact": True, "final_model_fit_calls": 1}
                margins = {p: scores[p][choice["score_index"]] for p in scores}
                selected_rep = make_representation(arrays, record, context, method, learner, arm, choice["lambda"], protocol["geometries"][choice["geometry"]])
                diagnostics[f"exact_{role}"] = geometry_diagnostic(selected_rep, arrays)
            exact_scores[f"{role}_tune"], exact_scores[f"{role}_gate"] = margins["tune"], margins["gate"]
            choices[role] = row
        np.savez_compressed(folder / "exact_margins.npz", **exact_scores)
        diag_arrays, diag_positions = {}, {}
        for key, diagnostic in diagnostics.items():
            diag_arrays[key] = diagnostic["kernel"]
            diag_positions[key] = diagnostic["positions"]
        np.savez_compressed(folder / "geometry.npz", **diag_arrays)
        write_json(folder / "complete.json", {"job": list(job), "stage": stage, "source_sha256": hashes,
            "input_sha256": record["input_sha256"], "trials": all_rows, "choices": choices,
            "margins_sha256": sha(folder / "margins.npz"), "exact_margins_sha256": sha(folder / "exact_margins.npz"),
            "parameter_lock_sha256": sha(folder / "parameter_lock.json"), "geometry_sha256": sha(folder / "geometry.npz"),
            "geometry_positions": diag_positions, "landmarks": landmarks.tolist() if approximate else None,
            "held_out_labels_read": False, "elapsed_seconds": time.perf_counter() - started})
        print("completed", stage, *job, flush=True)
    return job


def checkpoint(folder, rows, scores):
    temporary = folder / "margins.tmp.npz"
    np.savez_compressed(temporary, **{part: np.asarray(value) for part, value in scores.items()})
    os.replace(temporary, folder / "margins.npz")
    write_json(folder / "progress.json", {"candidate_count": len(rows),
        "trials": [{k: v for k, v in row.items() if k != "representation_details"} for row in rows],
        "margins_sha256": sha(folder / "margins.npz"), "complete": False, "held_out_labels_read": False})


@contextmanager
def exact_slot(protocol):
    root = ROOT / ".overnight-locks/exact_slots"
    root.mkdir(parents=True, exist_ok=True)
    handles = [None] * protocol["exact_workers"]
    acquired = None
    try:
        while acquired is None:
            for index in range(len(handles)):
                if handles[index] is None:
                    handles[index] = (root / f"{index}.lock").open("a")
                try:
                    fcntl.flock(handles[index], fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = index
                    break
                except BlockingIOError:
                    pass
            if acquired is None:
                time.sleep(2)
        yield
    finally:
        if acquired is not None:
            fcntl.flock(handles[acquired], fcntl.LOCK_UN)
        for handle in handles:
            if handle is not None:
                handle.close()


def exact_fit(arrays, record, context, job, choice, protocol, gpu):
    with exact_slot(protocol):
        return _exact_fit(arrays, record, context, job, choice, protocol, gpu)


def _exact_fit(arrays, record, context, job, choice, protocol, gpu):
    tick = time.perf_counter()
    rep = unpack_representation(make_representation(arrays, record, context, job[3], job[2], job[4], choice["lambda"], protocol["geometries"][choice["geometry"]]))
    n = len(arrays["train"])
    needed = (n + len(arrays["tune"]) + len(arrays["gate"])) * n * 8
    if available_memory() and available_memory() < 2 * needed + 24 * 2 ** 30:
        raise MemoryError("Exact final SVM memory reserve unavailable")
    kernels, kernel_times = {}, {}
    for part in PARTS:
        t = time.perf_counter()
        kernels[part] = representation_kernel(rep, (part, slice(None)), ("train", slice(None)), gpu)
        kernel_times[part] = time.perf_counter() - t
    kernel_seconds = time.perf_counter() - tick
    model = SVC(kernel="precomputed", C=choice["C"], class_weight="balanced", cache_size=256)
    tick = time.perf_counter()
    model.fit(kernels["train"], arrays["y_train"])
    fit_seconds = time.perf_counter() - tick
    if model.fit_status_:
        raise RuntimeError("Exact final SVC did not converge")
    tick = time.perf_counter()
    margins, decision_times = {}, {}
    for part in ("tune", "gate"):
        t = time.perf_counter()
        margins[part] = model.decision_function(kernels[part])
        decision_times[part] = time.perf_counter() - t
    inference_seconds = time.perf_counter() - tick
    threshold, _ = base.threshold_for_cost(arrays["y_tune"], margins["tune"], 0.)
    metrics = base.metrics(arrays["y_tune"], margins["tune"], threshold)
    row = {**choice, "threshold": threshold, **{f"tune_{k}": v for k, v in metrics.items()},
        "final_exact": True, "final_model_fit_calls": 1, "exact_kernel_seconds": kernel_seconds,
        "exact_fit_seconds": fit_seconds, "exact_inference_seconds": inference_seconds,
        "training_seconds": fit_seconds + kernel_times["train"] + rep["details"]["state_seconds_by_part"]["train"],
        "test_inference_seconds": kernel_times["gate"] + decision_times["gate"] + rep["details"]["state_seconds_by_part"]["gate"],
        "train_kernel_seconds": kernel_times["train"], "test_kernel_seconds": kernel_times["gate"],
        "representation_details": rep["details"]}
    diagnostic = geometry_diagnostic(rep, arrays)
    del kernels, model
    gc.collect()
    return row, margins, diagnostic


def run_reference(job, arrays, record, folder, hashes):
    started = time.perf_counter()
    rows, scores = [], {"tune": [], "gate": []}
    for raw in (False, True):
        x = {p: arrays[f"raw_{p}"] if raw else arrays[p] for p in PARTS}
        for learner, depth in (("RF", 6), ("RF", None), ("XGBoost", 3), ("XGBoost", 6)):
            model = RandomForestClassifier(n_estimators=200, max_depth=depth, class_weight="balanced", n_jobs=1, random_state=job[1]) if learner == "RF" else XGBClassifier(n_estimators=200, max_depth=depth, learning_rate=.05, tree_method="hist", n_jobs=1, random_state=job[1], eval_metric="logloss", scale_pos_weight=float(np.sum(arrays["y_train"] == 0) / np.sum(arrays["y_train"] == 1)))
            tick = time.perf_counter()
            model.fit(x["train"], arrays["y_train"])
            fit_seconds = time.perf_counter() - tick
            margins = {p: model.predict_proba(x[p])[:, 1] for p in scores}
            threshold, _ = base.threshold_for_cost(arrays["y_tune"], margins["tune"], 1 / 6)
            metrics = base.metrics(arrays["y_tune"], margins["tune"], threshold)
            rows.append({"dataset": job[0], "seed": job[1], "learner": learner, "method": "raw" if raw else "pca7",
                "arm": "normal", "lambda": 0., "C": 0., "geometry": 0, "depth": depth, "threshold": threshold,
                **{f"tune_{k}": v for k, v in metrics.items()}, "fit_rows": len(arrays["y_train"]), "fit_seconds": fit_seconds,
                "final_exact": True, "score_index": len(rows), "model_fit_calls": 1, "ensemble_members": 1})
            for p in scores:
                scores[p].append(margins[p])
    for raw in (False, True):
        margins = {p: arrays[f"raw_lr_{p}"] if raw else arrays[f"risk_{p}"] for p in scores}
        threshold, _ = base.threshold_for_cost(arrays["y_tune"], margins["tune"], 1 / 6)
        metrics = base.metrics(arrays["y_tune"], margins["tune"], threshold)
        rows.append({"dataset": job[0], "seed": job[1], "learner": "LR", "method": "raw" if raw else "pca7", "arm": "normal",
            "lambda": 0., "C": 0., "geometry": 0, "threshold": threshold, **{f"tune_{k}": v for k, v in metrics.items()},
            "fit_rows": len(arrays["y_train"]), "final_exact": True, "score_index": len(rows), "model_fit_calls": 0, "ensemble_members": 1})
        for p in scores:
            scores[p].append(margins[p])
    checkpoint(folder, rows, scores)
    choice = select_trial(rows)
    np.savez_compressed(folder / "exact_margins.npz", **{f"reference_{p}": scores[p][choice["score_index"]] for p in scores})
    write_json(folder / "parameter_lock.json", {"choices": {"reference": choice}, "rule": "Tune labels only"})
    write_json(folder / "complete.json", {"job": list(job), "source_sha256": hashes, "input_sha256": record["input_sha256"],
        "trials": rows, "choices": {"reference": choice}, "margins_sha256": sha(folder / "margins.npz"),
        "exact_margins_sha256": sha(folder / "exact_margins.npz"), "parameter_lock_sha256": sha(folder / "parameter_lock.json"),
        "held_out_labels_read": False, "elapsed_seconds": time.perf_counter() - started})
    print("completed reference", *job, flush=True)
    return job
