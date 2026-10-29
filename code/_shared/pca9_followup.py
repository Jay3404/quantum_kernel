"""Isolated dimension-consistent sensitivity campaign; upstream files are read-only."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import ctypes
import fcntl
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import zz_feature_map, z_feature_map
from qiskit_aer import AerSimulator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from . import controlled as base
from . import overnight_risk as old
from .angle_encoding import AngleEncoding
from .full_data import write_json
from .improved import sha, local_lock
from .separation_round2 import midrank

ROOT, PARTS = base.ROOT, ("train", "tune", "gate")
METHODS = ("gbdt_soft_gate", "gbdt_rank_append", "rank_mixture", "pls_metric")
REQUEST = ROOT / "docs/cmc_pca9_followup_request_20261019.json"
REQUEST_SHA = "c563a80547229a713e81e7d18fe75afea91d6f64ab1e2b2df51f94f1970d209d"
PROTOCOL = ROOT / "docs/cmc_pca9_protocol_20261020.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_array(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()


def release_heap():
    gc.collect()
    trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
    if trim:
        trim(0)


def peak_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if __import__("sys").platform == "darwin" else 2**20)


class CheckedPCA(PCA):
    def __init__(self, n_components=9, fit_log=None):
        super().__init__(n_components=n_components, svd_solver="full")
        self.fit_log = fit_log

    def _audit(self, values):
        tolerance = self.singular_values_[0] * max(values.shape) * np.finfo(float).eps
        assert len(self.singular_values_) == self.n_components
        assert self.singular_values_[-1] > tolerance, "Nested training rank does not permit the requested dimension"
        if self.fit_log:
            row = {"rows": len(values), "encoded_features": values.shape[1], "components": self.n_components,
                "ninth_singular_value": float(self.singular_values_[-1]), "rank_tolerance": float(tolerance),
                "training_input_sha256": digest_array(values)}
            with Path(self.fit_log).open("a") as handle:
                handle.write(json.dumps(row) + "\n")

    def fit(self, X, y=None):
        result = super().fit(X, y)
        self._audit(X)
        return result

    def fit_transform(self, X, y=None):
        result = super().fit_transform(X, y)
        self._audit(X)
        return result


def features(X, dimension, fit_log=None):
    raw = base.make_features(X)[:-1]
    return Pipeline([*raw.steps, ("pca", CheckedPCA(dimension, fit_log))])


def risk_estimator(X, seed, dimension, fit_log=None):
    model = Pipeline([("features", features(X, dimension, fit_log)),
        ("lr", LogisticRegression(C=1., class_weight="balanced", max_iter=3000, random_state=seed))])
    return CalibratedClassifierCV(model, method="sigmoid",
        cv=StratifiedKFold(3, shuffle=True, random_state=seed), ensemble=False, n_jobs=1)


def calibration_indices(ids, labels, seed):
    return [{"train_ids": ids[a].tolist(), "heldout_ids": ids[b].tolist()}
        for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(ids, labels)]


def assert_can_fit(output):
    output = Path(output)
    assert not (output / "selection_lock.json").exists(), "No training after global selection lock"
    assert not (output / "test_release.json").exists(), "No training after test release"


def prepare(dataset, seed, protocol, output):
    assert_can_fit(output)
    folder = old.prepared_folder(output, dataset, seed)
    folder.mkdir(parents=True, exist_ok=True)
    with local_lock(folder / "prepare.lock"), threadpool_limits(limits=1):
        assert not (folder / "audit.json").exists(), "No automatic preparation retry"
        parent = ROOT / "code/results" / protocol["upstream_run"] / "full"
        previous, previous_audit = old.load_inputs(parent, dataset, seed)
        parent_folder = old.prepared_folder(parent, dataset, seed)
        split_file = json.loads((parent_folder / "split.json").read_text())
        split = split_file["actual"]
        assert split == split_file["outer"]
        assert not set(split["train"]) & set(split["validation"] + split["test"])
        assert not set(split["validation"]) & set(split["test"])
        X, y, source = base.load_data(dataset, str(ROOT / "data"))
        assert source == previous_audit["source"]
        assert sum(map(len, split.values())) == len(X) and len(split["train"]) == round(.8 * len(X))
        dimension = protocol["pca_components"]
        fit_log = str(folder / "nested_pca_fits.jsonl")
        assert not Path(fit_log).exists(), "Preserve previous preparation attempts"
        fit = np.asarray(split["train"])
        tick = time.perf_counter()
        pipeline = features(X, dimension, fit_log)
        train = pipeline.fit_transform(X.iloc[fit])
        arrays = {"train": train, "y_train": y[fit], "y_tune": y[split["validation"]]}
        times = {"feature_fit_seconds": time.perf_counter() - tick}
        for part, name in (("tune", "validation"), ("gate", "test")):
            tick = time.perf_counter()
            arrays[part] = pipeline.transform(X.iloc[split[name]])
            times[part + "_feature_seconds"] = time.perf_counter() - tick
        tick = time.perf_counter()
        risk = np.zeros(len(fit))
        oof = []
        for index, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(fit, y[fit])):
            model = risk_estimator(X, seed + index + 1, dimension, fit_log)
            model.fit(X.iloc[fit[a]], y[fit[a]])
            risk[b] = model.predict_proba(X.iloc[fit[b]])[:, 1]
            oof.append({"train_ids": fit[a].tolist(), "heldout_ids": fit[b].tolist(),
                "calibration_seed": seed + index + 1,
                "calibration_folds": calibration_indices(fit[a], y[fit[a]], seed + index + 1)})
        model = risk_estimator(X, seed, dimension, fit_log)
        model.fit(X.iloc[fit], y[fit])
        estimator = model.calibrated_classifiers_[0].estimator
        np.testing.assert_allclose(estimator.named_steps["features"].transform(X.iloc[fit]), train, atol=1e-10)
        coefficients = np.abs(estimator.named_steps["lr"].coef_[0])
        arrays["weights"] = coefficients / coefficients.sum() if coefficients.sum() else np.ones(dimension) / dimension
        arrays["risk_train"] = risk
        times["risk_fit_seconds"] = time.perf_counter() - tick
        for part, name in (("tune", "validation"), ("gate", "test")):
            tick = time.perf_counter()
            arrays["risk_" + part] = model.predict_proba(X.iloc[split[name]])[:, 1]
            times[part + "_risk_seconds"] = time.perf_counter() - tick
        # Dimension-independent original-feature teachers are adopted only after input equality checks.
        teacher_provenance = {}
        for part, name in (("train", "train"), ("tune", "validation"), ("gate", "test")):
            arrays["rank_" + part] = midrank(risk, arrays["risk_" + part])
            raw = pipeline[:-1].transform(X.iloc[split[name]])
            np.testing.assert_array_equal(raw, previous["raw_" + part])
            arrays["raw_" + part] = raw
            for prefix in ("raw_lr", "gbdt"):
                for key in (prefix + "_" + part, prefix + "_rank_" + part):
                    arrays[key] = previous[key].copy()
                    teacher_provenance[key] = digest_array(arrays[key])
        for prefix in ("raw_lr", "gbdt"):
            for part in PARTS:
                np.testing.assert_array_equal(arrays[prefix + "_rank_" + part],
                    midrank(arrays[prefix + "_train"], arrays[prefix + "_" + part]))
        np.testing.assert_array_equal(arrays["y_train"], previous["y_train"])
        np.testing.assert_array_equal(arrays["y_tune"], previous["y_tune"])
        encoder = AngleEncoding.fit(train, "linear_no_clip")
        np.savez_compressed(folder / "inputs.npz", **arrays)
        write_json(folder / "split.json", split_file)
        assert sha(folder / "split.json") == previous_audit["split_sha256"]
        write_json(folder / "oof.json", {"folds": oof, "final_calibration_seed": seed,
            "final_calibration_folds": calibration_indices(fit, y[fit], seed),
            "outer_folds": 5, "inner_folds": 3, "sigmoid": True, "ensemble": False})
        logs = [json.loads(line) for line in Path(fit_log).read_text().splitlines()]
        assert len(logs) == 25 and all(row["components"] == dimension for row in logs)
        record = {"dataset": dataset, "seed": seed, "stage": "full", "source": source,
            "pca_components": dimension, "input_sha256": sha(folder / "inputs.npz"),
            "split_sha256": sha(folder / "split.json"), "encoder": encoder.parameters(), "times": times,
            "train_rows": len(fit), "tune_rows": len(arrays["tune"]), "gate_rows": len(arrays["gate"]),
            "training_row_ids": split["train"], "outer_test_labels_saved": False,
            "nested_pca_fits_sha256": sha(fit_log), "nested_pca_fit_count": len(logs),
            "oof_sha256": sha(folder / "oof.json"), "new_pca_lr_fit": True,
            "teacher_reuse": {"new_fit_calls": 0, "source_input_sha256": previous_audit["input_sha256"],
                "source_audit_path": str((parent_folder / "audit.json").relative_to(ROOT)),
                "source_audit_sha256": sha(parent_folder / "audit.json"), "arrays_sha256": teacher_provenance,
                "input_check": "Exact raw-feature matrices, original row order, labels, split, upstream frozen teacher code and hyperparameters",
                "timing": "No new teacher fit; original detailed teacher times are not reconstructed"}}
        write_json(folder / "audit.json", record)
        return {"dataset": dataset, "seed": seed, "nested_pca_fits": len(logs), "peak_gib": peak_gib()}


class DimensionStates(base.QuantumStates):
    def __init__(self, dimension, alpha, reps, entangled=True):
        self.dimension = dimension
        self.backend = AerSimulator(method="statevector", precision="double", max_parallel_threads=1,
            max_parallel_experiments=1, runtime_parameter_bind_enable=True)
        circuit = zz_feature_map(dimension, reps=reps, entanglement="linear", alpha=alpha) if entangled else z_feature_map(dimension, reps=reps, alpha=alpha)
        circuit.save_statevector(label="state")
        self.circuit = transpile(circuit, self.backend, optimization_level=1, seed_transpiler=17)
        self.parameters = tuple(self.circuit.parameters)
        assert len(self.parameters) == dimension

    def __call__(self, values):
        assert values.ndim == 2 and values.shape[1] == self.dimension and np.isfinite(values).all()
        return super().__call__(values)


def context(arrays, record, method, learner, arm, seed, protocol):
    assert method in (*METHODS, "baseline")
    source = arrays
    if method.startswith("gbdt_"):
        source = {**arrays, **{f"risk_{p}": arrays[f"gbdt_{p}"] for p in PARTS},
            **{f"rank_{p}": arrays[f"gbdt_rank_{p}"] for p in PARTS}}
    p, q = old.risk_values(source, arm, seed)
    angles = {part: old.angle_encoder(record).transform(arrays[part]) for part in PARTS}
    metric, metric_record = None, None
    if method == "pls_metric":
        labels = arrays["y_train"].copy()
        if arm == "shuffled":
            labels = np.random.default_rng(seed + 9173).permutation(labels)
        tick = time.perf_counter()
        model = PLSRegression(n_components=protocol["pca_components"], scale=False).fit(arrays["raw_train"], labels)
        values = {part: model.transform(arrays["raw_" + part]) for part in PARTS}
        encoder = AngleEncoding.fit(values["train"], "linear_no_clip")
        metric = {part: encoder.transform(values[part]) for part in PARTS}
        metric_record = {"components": protocol["pca_components"], "training_labels_shuffled": arm == "shuffled",
            "fit_seconds": time.perf_counter() - tick, "encoder": encoder.parameters(), "fit_rows": len(labels)}
    return {"p": p, "q": q, "angles": angles, "metric": metric, "metric_record": metric_record, "cache": {}}


def make_representation(arrays, record, ctx, method, learner, arm, strength, geometry):
    tick = time.perf_counter()
    dimension = arrays["train"].shape[1]
    assert dimension == record["pca_components"]
    values = ctx["metric"] if method == "pls_metric" else ctx["angles"]
    base_values = ctx["angles"] if method == "pls_metric" else None
    state_times = dict.fromkeys(PARTS, 0.)
    circuits, reused = [], False
    if learner == "QSVC":
        key = (geometry["alpha"], geometry["reps"])
        reused = key in ctx["cache"]
        if not reused:
            engine = DimensionStates(dimension, geometry["alpha"], geometry["reps"], arm != "separable")
            transformed = []
            for group in (values, base_values):
                if group is None:
                    transformed.append(None)
                    continue
                transformed_group = {}
                for part, data in group.items():
                    started = time.perf_counter()
                    transformed_group[part] = engine(data)
                    state_times[part] += time.perf_counter() - started
                transformed.append(transformed_group)
            details = {"qubits": dimension, "depth": engine.circuit.depth(), "gates": dict(engine.circuit.count_ops()),
                "backend": "CPU Aer double exact noiseless statevector", "entangled": arm != "separable",
                "measured_cold_state_seconds_by_part": dict(state_times)}
            ctx["cache"][key] = (*transformed, details)
        values, base_values, details = ctx["cache"][key]
        circuits = [dict(details)]
    values = {part: value.copy() for part, value in values.items()}
    if method == "gbdt_rank_append":
        for part in PARTS:
            angle = np.pi * strength * (ctx["q"][part] - .5)
            if learner == "QSVC":
                aux = np.column_stack([np.cos(angle / 2), np.sin(angle / 2)])
                values[part] = np.einsum("ni,nj->nij", values[part], aux).reshape(len(angle), -1)
            else:
                values[part] = np.column_stack([values[part], angle])
        if circuits:
            circuits[0].update(qubits=dimension + 1, auxiliary_independent_qubits=1)
    mode = {"gbdt_soft_gate": "soft_gate", "rank_mixture": "risk_mix", "pls_metric": "metric_mix"}.get(method, "base")
    gamma = geometry["gamma_factor"] / max(dimension * ctx["angles"]["train"].var(), 1e-12) if learner == "RBF-SVM" else 1.
    return {"values": values, "base_values": base_values, "kind": "fidelity" if learner == "QSVC" else "rbf",
        "gamma": float(gamma), "mode": mode, "strength": strength, "risk": ctx["q"], "q": ctx["q"], "p": ctx["p"],
        "details": {"representation_seconds": time.perf_counter() - tick, "circuits": circuits,
            "state_seconds_by_part": state_times, "state_cache_reused": reused,
            "measured_cold_state_seconds_by_part": details["measured_cold_state_seconds_by_part"] if learner == "QSVC" else dict(state_times),
            "metric_fit": ctx["metric_record"], "pca_components": dimension}}


def kernel(rep, left, right, gpu=None):
    part, selection = left
    positions = np.atleast_1d(np.arange(len(rep["values"][part]))[selection])
    rp, ri = right
    columns = len(np.atleast_1d(np.arange(len(rep["values"][rp]))[ri]))
    result = np.empty((len(positions), columns), dtype=np.float64)
    for start in range(0, len(positions), 512):
        chosen = positions[start:start + 512]
        result[start:start + len(chosen)] = old.representation_kernel(rep, (part, chosen), right, gpu)
    return result


def jobs(protocol):
    tasks = []
    for dataset, seed in itertools.product(protocol["datasets"], protocol["seeds"]):
        tasks.append((dataset, seed, "reference", "reference", "normal"))
        for learner in protocol["learners"]:
            tasks.append((dataset, seed, learner, "baseline", "normal"))
            for method in protocol["methods"]:
                arms = ["normal", "shuffled"] + ([] if method == "pls_metric" else ["constant"])
                if learner == "QSVC":
                    arms.append("separable")
                tasks.extend((dataset, seed, learner, method, arm) for arm in arms)
    return tasks


@contextmanager
def exact_slot(output, count):
    root = Path(output).parent / "exact_slots"
    root.mkdir(parents=True, exist_ok=True)
    handles = [(root / f"{index}.lock").open("a") for index in range(count)]
    selected = None
    try:
        while selected is None:
            for index, handle in enumerate(handles):
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    selected = index
                    break
                except BlockingIOError:
                    pass
            if selected is None:
                time.sleep(1)
        yield
    finally:
        for handle in handles:
            handle.close()


def matrices(rep, arrays, approximate, landmarks, gpu):
    tick = time.perf_counter()
    factor, spectrum = None, None
    if approximate:
        factor, spectrum = old.nystrom_factor(kernel(rep, ("train", landmarks), ("train", landmarks), gpu))
    setup = time.perf_counter() - tick
    kernels, timing = {}, {}
    for part in PARTS:
        tick = time.perf_counter()
        value = kernel(rep, (part, slice(None)), ("train", landmarks if approximate else slice(None)), gpu)
        kernels[part] = value @ factor if approximate else value
        timing[part] = time.perf_counter() - tick
    return kernels, timing, spectrum, setup


def fit_candidate(arrays, kernels, choice, approximate, seed):
    model = LinearSVC(C=choice["C"], class_weight="balanced", dual=False, max_iter=5000, tol=1e-5,
        random_state=seed) if approximate else SVC(kernel="precomputed", C=choice["C"], class_weight="balanced", cache_size=256)
    tick = time.perf_counter()
    model.fit(kernels["train"], arrays["y_train"])
    fitted = time.perf_counter() - tick
    assert not (model.n_iter_ >= 5000 if approximate else model.fit_status_), "SVM candidate did not converge"
    margins, timing = {}, {}
    for part in ("tune", "gate"):
        tick = time.perf_counter()
        margins[part] = model.decision_function(kernels[part])
        assert np.isfinite(margins[part]).all()
        timing[part] = time.perf_counter() - tick
    threshold, _ = base.threshold_for_cost(arrays["y_tune"], margins["tune"], 0.)
    metrics = base.metrics(arrays["y_tune"], margins["tune"], threshold)
    return {**choice, "threshold": threshold, **{"tune_" + k: v for k, v in metrics.items()},
        "fit_rows": len(arrays["train"]), "fit_seconds": fitted, "inference_seconds": timing,
        "approximate_search": approximate, "model_fit_calls": 1, "ensemble_members": 1}, margins


def exact_fit(arrays, record, ctx, job, choice, protocol, output, gpu):
    with exact_slot(output, protocol["exact_workers"]):
        rep = make_representation(arrays, record, ctx, job[3], job[2], job[4], choice["lambda"], protocol["geometries"][choice["geometry"]])
        kernels, timing, _, setup = matrices(rep, arrays, False, None, gpu)
        row, margins = fit_candidate(arrays, kernels, choice, False, job[1])
        row.update(final_exact=True, final_model_fit_calls=1, selection_was_approximate=True,
            exact_fit_seconds=row["fit_seconds"], exact_kernel_seconds=sum(timing.values()),
            kernel_seconds=timing, train_kernel_seconds=timing["train"], test_kernel_seconds=timing["gate"],
            training_seconds=row["fit_seconds"] + timing["train"] + rep["details"]["state_seconds_by_part"]["train"],
            test_inference_seconds=timing["gate"] + row["inference_seconds"]["gate"] + rep["details"]["state_seconds_by_part"]["gate"],
            representation_details=rep["details"], nystrom_setup_seconds=setup)
        del kernels
        release_heap()
        return row, margins, diagnostic(rep, arrays)


def diagnostic(rep, arrays):
    positions = np.linspace(0, len(arrays["gate"]) - 1, min(256, len(arrays["gate"])), dtype=int)
    return {"positions": positions.tolist(), "kernel": kernel(rep, ("gate", positions), ("gate", positions))}


def run_reference(job, arrays, record, folder, hashes):
    started = time.perf_counter()
    rows, scores = [], {"tune": [], "gate": []}
    for raw in (False, True):
        values = {p: arrays["raw_" + p] if raw else arrays[p] for p in PARTS}
        for learner, depth in (("RF", 6), ("RF", None), ("XGBoost", 3), ("XGBoost", 6)):
            model = RandomForestClassifier(n_estimators=200, max_depth=depth, class_weight="balanced", n_jobs=1,
                random_state=job[1]) if learner == "RF" else XGBClassifier(n_estimators=200, max_depth=depth,
                learning_rate=.05, tree_method="hist", n_jobs=1, random_state=job[1], eval_metric="logloss",
                scale_pos_weight=float(np.sum(arrays["y_train"] == 0) / np.sum(arrays["y_train"] == 1)))
            tick = time.perf_counter()
            model.fit(values["train"], arrays["y_train"])
            fit_seconds = time.perf_counter() - tick
            margins, timing = {}, {}
            for part in scores:
                tick = time.perf_counter()
                margins[part] = model.predict_proba(values[part])[:, 1]
                timing[part] = time.perf_counter() - tick
            row = reference_row(job, arrays, learner, raw, depth, margins, len(rows))
            row.update(fit_seconds=fit_seconds, training_seconds=fit_seconds,
                test_inference_seconds=timing["gate"], model_fit_calls=1)
            rows.append(row)
            for part in scores:
                scores[part].append(margins[part])
    for raw in (False, True):
        margins = {p: arrays[("raw_lr_" if raw else "risk_") + p] for p in scores}
        row = reference_row(job, arrays, "LR", raw, None, margins, len(rows))
        row.update(model_fit_calls=0, preparation_score_reuse=True)
        rows.append(row)
        for part in scores:
            scores[part].append(margins[part])
    old.checkpoint(folder, rows, scores)
    choice = old.select_trial(rows)
    np.savez_compressed(folder / "exact_margins.npz", **{f"reference_{p}": scores[p][choice["score_index"]] for p in scores})
    write_json(folder / "parameter_lock.json", {"choices": {"reference": choice}, "rule": "Validation only"})
    complete(folder, job, record, hashes, rows, {"reference": choice}, started)


def reference_row(job, arrays, learner, raw, depth, margins, index):
    threshold, _ = base.threshold_for_cost(arrays["y_tune"], margins["tune"], 1 / 6)
    return {"dataset": job[0], "seed": job[1], "learner": learner, "method": "raw" if raw else "pca9",
        "arm": "normal", "lambda": 0., "C": 0., "geometry": 0, "depth": depth, "threshold": threshold,
        **{"tune_" + k: v for k, v in base.metrics(arrays["y_tune"], margins["tune"], threshold).items()},
        "fit_rows": len(arrays["train"]), "final_exact": True, "score_index": index, "ensemble_members": 1}


def complete(folder, job, record, hashes, rows, choices, started, diagnostics=None, landmarks=None):
    files = {"margins_sha256": "margins.npz", "exact_margins_sha256": "exact_margins.npz",
        "parameter_lock_sha256": "parameter_lock.json"}
    if diagnostics:
        np.savez_compressed(folder / "geometry.npz", **{k: v["kernel"] for k, v in diagnostics.items()})
        files["geometry_sha256"] = "geometry.npz"
    write_json(folder / "complete.json", {"job": list(job), "stage": "full", "source_sha256": hashes,
        "input_sha256": record["input_sha256"], "trials": rows, "choices": choices,
        **{k: sha(folder / name) for k, name in files.items()},
        "geometry_positions": {k: v["positions"] for k, v in (diagnostics or {}).items()},
        "landmarks": landmarks.tolist() if landmarks is not None else None,
        "held_out_labels_read": False, "elapsed_seconds": time.perf_counter() - started,
        "worker_pid": os.getpid(), "peak_gib": peak_gib(), "execution_origin": "new_pca9_single_job_process"})


def run_job(job, protocol, output, hashes):
    assert_can_fit(output)
    folder = old.job_folder(output, job)
    folder.mkdir(parents=True, exist_ok=True)
    with local_lock(folder / "job.lock"), threadpool_limits(limits=1):
        assert not (folder / "complete.json").exists() and not (folder / "progress.json").exists(), "No automatic job retry"
        arrays, record = old.load_inputs(output, *job[:2])
        if job[3] == "reference":
            return run_reference(job, arrays, record, folder, hashes)
        started = time.perf_counter()
        ctx = context(arrays, record, job[3], job[2], job[4], job[1], protocol)
        strengths = [0.] if job[3] == "baseline" else [0., *protocol["lambdas"]]
        approximate = len(arrays["train"]) > protocol["exact_search_max_rows"]
        landmarks = old.subset(np.arange(len(arrays["y_train"])), arrays["y_train"], protocol["nystrom_landmarks"], job[1] + 701)
        gpu = protocol["gpu_devices"][(job[1] // 10) % len(protocol["gpu_devices"])] if protocol["gpu_devices"] else None
        rows, scores = [], {"tune": [], "gate": []}
        for gid, strength in itertools.product(range(len(protocol["geometries"])), strengths):
            rep = make_representation(arrays, record, ctx, job[3], job[2], job[4], strength, protocol["geometries"][gid])
            kernels, timing, spectrum, setup = matrices(rep, arrays, approximate, landmarks, gpu)
            for c in protocol["C_grid"]:
                choice = {"dataset": job[0], "seed": job[1], "learner": job[2], "method": job[3], "arm": job[4],
                    "lambda": strength, "C": c, "geometry": gid, "score_index": len(rows)}
                row, margins = fit_candidate(arrays, kernels, choice, approximate, job[1])
                row.update(kernel_seconds=timing, nystrom_spectrum=spectrum, nystrom_setup_seconds=setup,
                    train_kernel_seconds=timing["train"], test_kernel_seconds=timing["gate"], representation_details=rep["details"],
                    training_seconds=row["fit_seconds"] + timing["train"] + rep["details"]["state_seconds_by_part"]["train"],
                    test_inference_seconds=timing["gate"] + row["inference_seconds"]["gate"] + rep["details"]["state_seconds_by_part"]["gate"])
                rows.append(row)
                for part in scores:
                    scores[part].append(margins[part])
            old.checkpoint(folder, rows, scores)
            del kernels, rep
            release_heap()
        choices = {"baseline": old.select_trial(rows, False)}
        if job[3] != "baseline":
            choices["proposed"] = old.select_trial(rows, True)
        write_json(folder / "parameter_lock.json", {"choices": choices,
            "rule": "Validation-only parameters before exact fit; fresh exact validation threshold before global test release"})
        final, exact_scores, diagnostics = {}, {}, {}
        for role, choice in choices.items():
            if approximate:
                row, margins, diag = exact_fit(arrays, record, ctx, job, choice, protocol, output, gpu)
                row["selection_proxy_trial"] = choice["score_index"]
            else:
                row = {**choice, "final_exact": True, "final_model_fit_calls": 1, "selection_was_approximate": False}
                margins = {part: scores[part][choice["score_index"]] for part in scores}
                rep = make_representation(arrays, record, ctx, job[3], job[2], job[4], choice["lambda"], protocol["geometries"][choice["geometry"]])
                diag = diagnostic(rep, arrays)
            final[role] = row
            for part in scores:
                exact_scores[role + "_" + part] = margins[part]
            diagnostics["exact_" + role] = diag
        np.savez_compressed(folder / "exact_margins.npz", **exact_scores)
        complete(folder, job, record, hashes, rows, final, started, diagnostics, landmarks if approximate else None)
