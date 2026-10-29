"""Leakage-contained calibrated ensembles for the revised full-data campaign."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import gc
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from . import controlled as base
from .full_data import all_indices, dense_kernel, kernel_block, write_json

CORE = ("RBF-SVM", "QSVC", "PQK")
LEARNERS = (*CORE, "LR", "RF", "XGBoost", "risk_only")
POLICIES = ("cost_probability", "matched_backbone", "tuned_probability")
THRESHOLD = 1 / 6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


@contextmanager
def local_lock(path, blocking=False):
    # Keep advisory locks off the server's NFS workspace.
    directory = Path("/tmp/quantum-kernel-improved-locks")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (hashlib.sha256(str(Path(path).absolute()).encode()).hexdigest() + ".lock")
    with target.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError("Another process holds this improved job lock") from error
        yield


class TunedQuantumStates(base.QuantumStates):
    def __init__(self, alpha, entanglement):
        started = time.perf_counter()
        self.backend = AerSimulator(method="statevector", precision="double", max_parallel_threads=1,
                                    max_parallel_experiments=1, runtime_parameter_bind_enable=True)
        circuit = zz_feature_map(7, reps=1, entanglement=entanglement, alpha=alpha)
        circuit.save_statevector(label="state")
        self.circuit = transpile(circuit, self.backend, optimization_level=1, seed_transpiler=17)
        self.parameters = tuple(self.circuit.parameters)
        self.setup_seconds = time.perf_counter() - started


def local_observables(states):
    qubits = int(round(np.log2(states.shape[1])))
    result = np.empty((len(states), 3 * qubits))
    for q, axis in itertools.product(range(qubits), range(3)):
        label = "I" * (qubits - q - 1) + "XYZ"[axis] + "I" * q
        operator = SparsePauliOp.from_list([(label, 1)]).to_matrix(sparse=True)
        for start in range(0, len(states), 512):
            batch = states[start:start + 512]
            result[start:start + len(batch), 3 * q + axis] = np.einsum("ij,ij->i", batch.conj(), (operator @ batch.T).T).real
    return result


def geometries(learner, protocol):
    if learner == "RBF-SVM":
        return [{"gamma_factor": float(g), "alpha": 0.0, "entanglement": "none"} for g in protocol["gamma_factors"]]
    factors = protocol["gamma_factors"] if learner == "PQK" else [1.0]
    return [{"gamma_factor": float(g), "alpha": float(a), "entanglement": e}
            for a, e, g in itertools.product(protocol["alphas"], protocol["entanglement"], factors)]


def canonical(representation, strength):
    return "baseline" if strength == 0 else representation


def split_plan(y, seed):
    train, test, fit, valid = all_indices(y, seed)
    result = {"train": train.tolist(), "test": test.tolist(), "inner_train": fit.tolist(), "inner_valid": valid.tolist()}
    for stage, indices, offset in (("inner", fit, 2007), ("final", train, 3007)):
        folds = list(StratifiedKFold(3, shuffle=True, random_state=seed + offset).split(indices, y[indices]))
        result[f"{stage}_folds"] = [{"fit": indices[a].tolist(), "calibration": indices[b].tolist()} for a, b in folds]
    return result


def prepare_fold(X, y, indices, evaluation, seed, core):
    fit, cal = np.asarray(indices["fit"]), np.asarray(indices["calibration"])
    tick = time.perf_counter()
    features = base.make_features(X) if core else base.make_features(X)[:-1]
    train = features.fit_transform(X.iloc[fit])
    minimum = train.min(0) if core else None
    span = np.maximum(train.max(0) - minimum, 1e-12) if core else None
    gamma = 1 / (7 * max((np.clip((train - minimum) / span, 0, 1) * np.pi).var(), 1e-12)) if core else None
    feature_fit = time.perf_counter() - tick
    risks, weights, risk_model, risk_fit = {}, None, None, 0.0
    if core:
        tick = time.perf_counter()
        risks["train"] = np.zeros(len(fit))
        for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(fit, y[fit])):
            model = base.risk_estimator(X, seed + fold + 1)
            model.fit(X.iloc[fit[a]], y[fit[a]])
            risks["train"][b] = model.predict_proba(X.iloc[fit[b]])[:, 1]
        risk_model = base.risk_estimator(X, seed)
        risk_model.fit(X.iloc[fit], y[fit])
        estimator = risk_model.calibrated_classifiers_[0].estimator
        if not np.allclose(estimator.named_steps["features"].transform(X.iloc[fit]), train):
            raise AssertionError("Risk weights and PCA basis differ")
        coefficients = np.abs(estimator.named_steps["lr"].coef_[0])
        weights = coefficients / coefficients.sum() if coefficients.sum() else np.ones(7) / 7
        risk_fit = time.perf_counter() - tick
    result = {"train": train, "y_train": y[fit], "y_cal": y[cal], "minimum": minimum, "span": span,
              "gamma_reference": gamma, "weights": weights, "seed": seed, "risks": risks,
              "times": {"feature_fit_seconds": feature_fit, "risk_fit_seconds": risk_fit}}
    for part, rows in (("cal", cal), ("evaluation", np.asarray(evaluation))):
        tick = time.perf_counter()
        result[part] = features.transform(X.iloc[rows])
        result["times"][f"{part}_feature_seconds"] = time.perf_counter() - tick
        tick = time.perf_counter()
        if core:
            risks[part] = risk_model.predict_proba(X.iloc[rows])[:, 1]
        result["times"][f"{part}_risk_seconds"] = time.perf_counter() - tick if core else 0.0
    return result


def transform(fold, part, representation, strength):
    risk = fold["risks"][part]
    if representation.endswith("shuffled"):
        offset = {"train": 0, "cal": 1, "evaluation": 2}[part]
        risk = np.random.default_rng(fold["seed"] + 7919 + offset).permutation(risk)
    elif representation.endswith("constant"):
        risk = np.full_like(risk, fold["risks"]["train"].mean())
    weight = fold["weights"] if representation.startswith("feature_specific") else np.ones(7)
    strength = 0.0 if representation == "baseline" else strength
    unit = (fold[part] * (1 + strength * risk[:, None] * weight) - fold["minimum"]) / fold["span"]
    return np.clip(unit, 0, 1) * np.pi, float(np.mean((unit < 0) | (unit > 1)))


def rectangle(left, right, learner, gamma, block):
    result = np.empty((len(left), len(right)))
    for start in range(0, len(left), block):
        result[start:start + block] = kernel_block(left[start:start + block], right, learner, gamma)
    return result


class FrozenResponseEstimator(FrozenEstimator):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # Rows reference an already-fitted kernel basis; CV must not slice columns.
        tags.input_tags.pairwise = False
        return tags


def frozen_calibrate(model, values, y):
    before = model.intercept_.copy() if hasattr(model, "intercept_") else None
    calibrated = CalibratedClassifierCV(FrozenResponseEstimator(model), method="sigmoid", ensemble=False, n_jobs=1)
    calibrated.fit(values, y)
    if before is not None and not np.array_equal(before, model.intercept_):
        raise AssertionError("Calibration refitted the underlying model")
    return calibrated


def ensemble(folds, learner, representation, strength, geometry, cs, options, engines):
    total = np.zeros((len(cs), len(folds[0]["evaluation"])))
    times = [defaultdict(float) for _ in cs]
    members = []
    for fold in folds:
        values, shared = {}, defaultdict(float)
        use_risk = representation != "baseline"
        shared["preprocessing_train_seconds"] = fold["times"]["feature_fit_seconds"] + (fold["times"]["risk_fit_seconds"] if use_risk else 0)
        for part in ("cal", "evaluation"):
            shared[f"preprocessing_{part}_seconds"] = fold["times"][f"{part}_feature_seconds"] + (fold["times"][f"{part}_risk_seconds"] if use_risk else 0)
        for part in ("train", "cal", "evaluation"):
            tick = time.perf_counter()
            values[part], clipped = transform(fold, part, representation, strength)
            shared[f"{part}_angle_seconds"] = time.perf_counter() - tick
            if part == "evaluation":
                shared["angle_clipped_fraction_sum"] += clipped
            if learner != "RBF-SVM":
                engine = engines[(geometry["alpha"], geometry["entanglement"])]
                tick = time.perf_counter()
                values[part] = engine(values[part])
                shared[f"{part}_state_seconds"] = time.perf_counter() - tick
                if learner == "PQK":
                    tick = time.perf_counter()
                    values[part] = local_observables(values[part])
                    shared[f"{part}_projection_seconds"] = time.perf_counter() - tick
        gamma = geometry["gamma_factor"] * (fold["gamma_reference"] if learner == "RBF-SVM" else 1 / 21)
        kernel_learner = "QSVC" if learner == "QSVC" else "RBF-SVM"
        tick = time.perf_counter()
        train_kernel = dense_kernel(values["train"], kernel_learner, gamma, options["kernel_block"], options["memory_limit_gib"], options["reserve_gib"])
        shared["train_kernel_seconds"] = time.perf_counter() - tick
        for part in ("cal", "evaluation"):
            tick = time.perf_counter()
            values[f"{part}_kernel"] = rectangle(values[part], values["train"], kernel_learner, gamma, options["prediction_block"])
            shared[f"{part}_kernel_seconds"] = time.perf_counter() - tick
        for i, c in enumerate(cs):
            tick = time.perf_counter()
            model = SVC(kernel="precomputed", C=c, class_weight="balanced", cache_size=options["svm_cache_mb"])
            model.fit(train_kernel, fold["y_train"])
            fit_seconds = time.perf_counter() - tick
            if model.fit_status_:
                raise RuntimeError("Improved SVC did not converge")
            tick = time.perf_counter()
            calibrated = frozen_calibrate(model, values["cal_kernel"], fold["y_cal"])
            calibration_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            total[i] += calibrated.predict_proba(values["evaluation_kernel"])[:, 1] / len(folds)
            decision_seconds = time.perf_counter() - tick
            times[i].update({k: times[i][k] + v for k, v in shared.items()})
            times[i]["fit_seconds"] += fit_seconds
            times[i]["calibration_seconds"] += calibration_seconds
            times[i]["decision_seconds"] += decision_seconds
            del calibrated, model
        members.append({"fit_rows": len(fold["train"]), "calibration_rows": len(fold["cal"]),
                        "evaluation_rows": len(fold["evaluation"]), "train_kernel_entries": len(fold["train"]) ** 2,
                        "calibration_kernel_entries": len(fold["cal"]) * len(fold["train"]),
                        "evaluation_kernel_entries": len(fold["evaluation"]) * len(fold["train"]),
                        "peak_train_kernel_gib": len(fold["train"]) ** 2 * 8 / 1024 ** 3})
        del train_kernel, values
        gc.collect()
    for timing in times:
        timing["training_seconds"] = sum(v for k, v in timing.items() if k.endswith("_seconds") and not k.startswith("evaluation_") and k not in ("preprocessing_evaluation_seconds", "decision_seconds"))
        timing["inference_seconds"] = sum(timing.get(k, 0.0) for k in ("preprocessing_evaluation_seconds", "evaluation_angle_seconds", "evaluation_state_seconds", "evaluation_projection_seconds", "evaluation_kernel_seconds", "decision_seconds"))
        timing["angle_clipped_fraction"] = timing.pop("angle_clipped_fraction_sum") / len(folds)
    if not np.isfinite(total).all() or (total < 0).any() or (total > 1).any():
        raise AssertionError("Invalid calibrated probabilities")
    return total, [dict(t) for t in times], members


def candidate_rows(scores, y, parameters):
    threshold, cost = base.threshold_for_cost(y, scores, THRESHOLD)
    return {**parameters, "cost_probability_cost": base.metrics(y, scores, THRESHOLD)["cost"],
            "tuned_probability_cost": cost, "probability_threshold": threshold,
            "validation_brier": float(brier_score_loss(y, scores))}


def choose(trials, policy, backbone=None):
    pool = trials
    if backbone is not None:
        pool = [r for r in pool if all(r[k] == backbone[k] for k in ("C", "alpha", "entanglement", "gamma_factor"))]
    target = "tuned_probability_cost" if policy == "tuned_probability" else "cost_probability_cost"
    best = min(pool, key=lambda r: (r[target], r["lambda"], r["validation_brier"], r["C"], r["alpha"], r["entanglement"], r["gamma_factor"]))
    return {**best, "threshold": best["probability_threshold"] if policy == "tuned_probability" else THRESHOLD}


def core_tuning(folds, y, learner, representation, protocol, options, folder, engines):
    trials, seconds = [], 0.0
    strengths = [0.0] if representation == "baseline" else protocol["lambdas"]
    for strength, geometry in itertools.product(strengths, geometries(learner, protocol)):
        rep = canonical(representation, strength)
        identity = {"representation": rep, "lambda": strength, **geometry, "C_grid": protocol["C_grid"]}
        target = folder / "tuning" / key(identity)
        target.mkdir(parents=True, exist_ok=True)
        info, archive = target / "complete.json", target / "probabilities.npz"
        if info.exists():
            saved = json.loads(info.read_text())
            if saved["identity"] != identity or not archive.exists() or sha(archive) != saved["probabilities_sha256"]:
                raise AssertionError("Candidate checkpoint changed")
        else:
            tick = time.perf_counter()
            probabilities, _, _ = ensemble(folds, learner, rep, strength, geometry, protocol["C_grid"], options, engines)
            generated = [candidate_rows(score, y, {"C": c, "lambda": strength, **geometry}) for c, score in zip(protocol["C_grid"], probabilities)]
            np.savez_compressed(archive, score=probabilities)
            saved = {"identity": identity, "trials": generated, "seconds": time.perf_counter() - tick,
                     "probabilities_sha256": sha(archive)}
            write_json(info, saved)
            print(f"candidate learner={learner} representation={rep} lambda={strength} geometry={geometry}", flush=True)
        trials.extend(saved["trials"])
        seconds += saved["seconds"]
    return trials, seconds


def reference_ensemble(folds, learner, index, seed):
    score, timing = np.zeros(len(folds[0]["evaluation"])), defaultdict(float)
    for fold in folds:
        if learner == "risk_only":
            score += fold["risks"]["evaluation"] / len(folds)
            timing["training_seconds"] += fold["times"]["feature_fit_seconds"] + fold["times"]["risk_fit_seconds"]
            timing["inference_seconds"] += fold["times"]["evaluation_feature_seconds"] + fold["times"]["evaluation_risk_seconds"]
        else:
            model = base.reference_models(learner, fold["y_train"], seed)[index]
            tick = time.perf_counter()
            model.fit(fold["train"], fold["y_train"])
            fit_time = time.perf_counter() - tick
            tick = time.perf_counter()
            calibrated = frozen_calibrate(model, fold["cal"], fold["y_cal"])
            cal_time = time.perf_counter() - tick
            tick = time.perf_counter()
            score += calibrated.predict_proba(fold["evaluation"])[:, 1] / len(folds)
            infer_time = time.perf_counter() - tick
            timing["fit_seconds"] += fit_time
            timing["calibration_seconds"] += cal_time
            timing["decision_seconds"] += infer_time
            timing["training_seconds"] += fold["times"]["feature_fit_seconds"] + fold["times"]["cal_feature_seconds"] + fit_time + cal_time
            timing["inference_seconds"] += fold["times"]["evaluation_feature_seconds"] + infer_time
    return score, dict(timing)


def run_job(dataset, seed, learner, options):
    with threadpool_limits(limits=options["blas_threads"]):
        return _run_job(dataset, seed, learner, options)


def _run_job(dataset, seed, learner, options):
    protocol = options["protocol_settings"]
    root = base.ROOT / "code" / dataset / "results" / options["run_id"] / f"seed_{seed:03d}"
    folder = root / learner
    folder.mkdir(parents=True, exist_ok=True)
    with local_lock(folder / "job.lock"):
        X, y, data = base.load_data(dataset, options["data_root"])
        identity = {"options": options, "source_sha256": data["sha256"], "dataset": dataset, "seed": seed, "learner": learner}
        settings = folder / "settings.json"
        if settings.exists() and json.loads(settings.read_text()) != identity:
            raise AssertionError("Job settings changed; use a new run id")
        write_json(settings, identity)
        complete = folder / "complete.json"
        if complete.exists():
            receipt = json.loads(complete.read_text())
            if receipt["identity"] != identity or any(sha(folder / name) != value for name, value in receipt["outputs_sha256"].items()):
                raise AssertionError("Completed job changed")
            return dataset, seed, learner
        started = time.perf_counter()
        split = split_plan(y, seed)
        with local_lock(root / "split.lock", blocking=True):
            split_path = root / "split_indices.json"
            if split_path.exists() and json.loads(split_path.read_text()) != split:
                raise AssertionError("Learners do not share the same splits")
            write_json(split_path, split)
        write_json(folder / "dataset_audit.json", data)
        core = learner in CORE or learner == "risk_only"
        inner = [prepare_fold(X, y, indices, split["inner_valid"], seed + 10000 + i, core)
                 for i, indices in enumerate(split["inner_folds"])]
        engines = {}
        if learner in ("QSVC", "PQK"):
            engines = {(float(a), e): TunedQuantumStates(a, e) for a, e in itertools.product(protocol["alphas"], protocol["entanglement"])}
        trials_by_rep, tuning_seconds = {}, {}
        representations = base.REPRESENTATIONS if learner in CORE else ("risk_score" if learner == "risk_only" else "encoded_full",)
        for representation in representations:
            if learner in CORE:
                trials, seconds = core_tuning(inner, y[split["inner_valid"]], learner, representation, protocol, options, folder, engines)
            else:
                path = folder / "reference_trials.json"
                if path.exists():
                    cached = json.loads(path.read_text())
                    trials, seconds = cached["trials"], cached["seconds"]
                else:
                    tick = time.perf_counter()
                    trials = []
                    for index in range(1 if learner == "risk_only" else 8):
                        score, _ = reference_ensemble(inner, learner, index, seed)
                        trials.append(candidate_rows(score, y[split["inner_valid"]], {"index": index, "C": 0.0, "lambda": 0.0, "alpha": 0.0, "entanglement": "none", "gamma_factor": 0.0}))
                    seconds = time.perf_counter() - tick
                    write_json(path, {"trials": trials, "seconds": seconds})
            trials_by_rep[representation], tuning_seconds[representation] = trials, seconds
        del inner
        gc.collect()
        final = [prepare_fold(X, y, indices, split["test"], seed + 20000 + i, core)
                 for i, indices in enumerate(split["final_folds"])]
        rows, predictions = [], []
        backbone = choose(trials_by_rep["baseline"], "cost_probability") if learner in CORE else None
        for representation in representations:
            for policy in POLICIES:
                parameters = choose(trials_by_rep[representation], policy, backbone if policy == "matched_backbone" and learner in CORE else None)
                numeric = {k: parameters[k] for k in ("C", "lambda", "alpha", "entanglement", "gamma_factor", "index") if k in parameters}
                rep = canonical(representation, parameters["lambda"]) if learner in CORE else representation
                target = folder / "final" / key({"representation": rep, **numeric})
                target.mkdir(parents=True, exist_ok=True)
                archive, receipt = target / "score.npz", target / "complete.json"
                if receipt.exists():
                    saved = json.loads(receipt.read_text())
                    if sha(archive) != saved["score_sha256"]:
                        raise AssertionError("Final probability checkpoint changed")
                    score = np.load(archive)["score"]
                else:
                    if learner in CORE:
                        scores, times, members = ensemble(final, learner, rep, parameters["lambda"], parameters, [parameters["C"]], options, engines)
                        score, timing = scores[0], times[0]
                    else:
                        score, timing = reference_ensemble(final, learner, parameters["index"], seed)
                        members = []
                    np.savez_compressed(archive, score=score)
                    saved = {"timing": timing, "members": members, "score_sha256": sha(archive)}
                    write_json(receipt, saved)
                threshold = parameters["threshold"]
                rows.append({"dataset": dataset, "seed": seed, "learner": learner, "representation": representation, "policy": policy,
                             "n_train": len(split["train"]), "n_test": len(split["test"]), "bad_test": int(y[split["test"]].sum()),
                             **numeric, "threshold": threshold, "validation_cost": parameters["tuned_probability_cost" if policy == "tuned_probability" else "cost_probability_cost"],
                             "validation_brier": parameters["validation_brier"], "tuning_seconds": tuning_seconds[representation],
                             **base.metrics(y[split["test"]], score, threshold), "brier_test": float(brier_score_loss(y[split["test"]], score)),
                             **saved["timing"], "ensemble_members": len(final), "circuit_setup_seconds": sum(e.setup_seconds for e in engines.values()),
                             "train_kernel_entries": sum(m["train_kernel_entries"] for m in saved["members"]),
                             "calibration_kernel_entries": sum(m["calibration_kernel_entries"] for m in saved["members"]),
                             "test_kernel_entries": sum(m["evaluation_kernel_entries"] for m in saved["members"]),
                             "peak_train_kernel_gib": max((m["peak_train_kernel_gib"] for m in saved["members"]), default=0)})
                predictions.append(pd.DataFrame({"dataset": dataset, "seed": seed, "learner": learner, "representation": representation, "policy": policy,
                                                 "row_id": split["test"], "y": y[split["test"]], "score": score,
                                                 "prediction": (score >= threshold).astype(int)}))
        pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(folder / "predictions.csv.gz", index=False, compression="gzip")
        pd.DataFrame([{**r, "representation": rep} for rep, trials in trials_by_rep.items() for r in trials]).to_csv(folder / "validation_trials.csv", index=False)
        write_json(complete, {"identity": identity, "rows": len(rows), "elapsed_seconds": time.perf_counter() - started,
                              "outputs_sha256": {name: sha(folder / name) for name in ("metrics.csv", "predictions.csv.gz", "validation_trials.csv")}})
        return dataset, seed, learner
