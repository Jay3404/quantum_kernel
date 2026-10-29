"""Single full-training SVMs with positive-lambda parallel holdout search."""
from __future__ import annotations

import gc
import itertools
import json
from pathlib import Path
import resource
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from . import controlled as base
from .full_data import decision_blocks, dense_kernel, write_json
from .improved import TunedQuantumStates, geometries, key, local_lock, sha

LEARNERS = ("RBF-SVM", "QSVC")
REPRESENTATIONS = ("baseline", "convex", "feature_specific")
POLICIES = ("validation_cost", "fixed_margin", "matched_backbone")


def split_indices(y, seed):
    train, holdout = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
    valid, test = train_test_split(holdout, test_size=.5, stratify=y[holdout], random_state=seed + 103)
    return {"train": train.tolist(), "validation": valid.tolist(), "test": test.tolist()}


def units(protocol, datasets, seeds):
    result = []
    for dataset, seed, learner in itertools.product(datasets, seeds, LEARNERS):
        result.append((dataset, seed, learner, "baseline", 0.0))
        for representation, strength in itertools.product(REPRESENTATIONS[1:], protocol["lambdas"]):
            result.append((dataset, seed, learner, representation, strength))
    return result


def seed_folder(dataset, seed, options):
    return base.ROOT / "code" / dataset / "results" / options["run_id"] / f"seed_{seed:03d}"


def unit_folder(unit, options):
    dataset, seed, learner, representation, strength = unit
    return seed_folder(dataset, seed, options) / learner / representation / f"lambda_{round(strength * 100):03d}"


def prepare_arrays(X, y, split, seed):
    fit = np.asarray(split["train"])
    tick = time.perf_counter()
    features = base.make_features(X)
    train = features.fit_transform(X.iloc[fit])
    feature_fit = time.perf_counter() - tick
    arrays = {"train": train, "y_train": y[fit], "y_validation": y[split["validation"]]}
    times = {"feature_fit_seconds": feature_fit}
    for part in ("validation", "test"):
        tick = time.perf_counter()
        arrays[part] = features.transform(X.iloc[split[part]])
        times[f"{part}_feature_seconds"] = time.perf_counter() - tick
    tick = time.perf_counter()
    risks = np.zeros(len(fit))
    for index, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(fit, y[fit])):
        model = base.risk_estimator(X, seed + index + 1)
        model.fit(X.iloc[fit[a]], y[fit[a]])
        risks[b] = model.predict_proba(X.iloc[fit[b]])[:, 1]
    model = base.risk_estimator(X, seed)
    model.fit(X.iloc[fit], y[fit])
    estimator = model.calibrated_classifiers_[0].estimator
    if not np.allclose(estimator.named_steps["features"].transform(X.iloc[fit]), train):
        raise AssertionError("Risk weights and PCA basis differ")
    coefficients = np.abs(estimator.named_steps["lr"].coef_[0])
    arrays["weights"] = coefficients / coefficients.sum() if coefficients.sum() else np.ones(7) / 7
    arrays["risk_train"] = risks
    times["risk_fit_seconds"] = time.perf_counter() - tick
    for part in ("validation", "test"):
        tick = time.perf_counter()
        arrays[f"risk_{part}"] = model.predict_proba(X.iloc[split[part]])[:, 1]
        times[f"{part}_risk_seconds"] = time.perf_counter() - tick
    arrays["minimum"] = train.min(0)
    arrays["span"] = np.maximum(train.max(0) - arrays["minimum"], 1e-12)
    original = np.clip((train - arrays["minimum"]) / arrays["span"], 0, 1) * np.pi
    arrays["gamma_reference"] = np.asarray(1 / (7 * max(original.var(), 1e-12)))
    return arrays, times


def prepare_job(dataset, seed, options):
    with threadpool_limits(limits=1):
        folder = seed_folder(dataset, seed, options)
        folder.mkdir(parents=True, exist_ok=True)
        with local_lock(folder / "prepare.lock"):
            X, y, audit = base.load_data(dataset, options["data_root"])
            split = split_indices(y, seed)
            identity = {"options": options, "data_sha256": audit["sha256"], "dataset": dataset, "seed": seed, "split": split}
            receipt, archive = folder / "prepared.json", folder / "prepared.npz"
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if saved["identity"] != identity or sha(archive) != saved["archive_sha256"]:
                    raise AssertionError("Prepared inputs changed")
                return dataset, seed
            arrays, times = prepare_arrays(X, y, split, seed)
            if "y_test" in arrays:
                raise AssertionError("Test labels must never enter candidate cache")
            np.savez_compressed(archive, **arrays)
            write_json(folder / "split_indices.json", split)
            write_json(folder / "dataset_audit.json", audit)
            write_json(receipt, {"identity": identity, "times": times, "archive_sha256": sha(archive)})
            return dataset, seed


def angles(arrays, part, representation, strength):
    if representation == "baseline":
        strength = 0.0
    weight = arrays["weights"] if representation == "feature_specific" else np.ones(7)
    value = arrays[part] * (1 + strength * arrays[f"risk_{part}"][:, None] * weight)
    unit = (value - arrays["minimum"]) / arrays["span"]
    return np.clip(unit, 0, 1) * np.pi, float(np.mean((unit < 0) | (unit > 1)))


def fit_geometry(arrays, times, learner, representation, strength, geometry, protocol, options):
    values, shared = {}, dict(times)
    risk = representation != "baseline"
    if not risk:
        for field in ("risk_fit_seconds", "validation_risk_seconds", "test_risk_seconds"):
            shared[field] = 0.0
    engine = None
    if learner == "QSVC":
        engine = TunedQuantumStates(geometry["alpha"], geometry["entanglement"])
    shared["circuit_setup_seconds"] = engine.setup_seconds if engine else 0.0
    for part in ("train", "validation", "test"):
        tick = time.perf_counter()
        values[part], clipped = angles(arrays, part, representation, strength)
        shared[f"{part}_angle_seconds"] = time.perf_counter() - tick
        shared[f"{part}_clipped_fraction"] = clipped
        shared[f"{part}_state_seconds"] = 0.0
        shared[f"{part}_statevectors"] = len(values[part]) if engine else 0
        if engine:
            tick = time.perf_counter()
            values[part] = engine(values[part])
            shared[f"{part}_state_seconds"] = time.perf_counter() - tick
    gamma = float(arrays["gamma_reference"]) * geometry["gamma_factor"]
    tick = time.perf_counter()
    kernel = dense_kernel(values["train"], learner, gamma, options["kernel_block"], options["memory_limit_gib"], options["reserve_gib"])
    shared["train_kernel_seconds"] = time.perf_counter() - tick
    rows, validation, test = [], [], []
    for c in protocol["C_grid"]:
        tick = time.perf_counter()
        model = SVC(kernel="precomputed", C=c, class_weight="balanced", cache_size=options["svm_cache_mb"])
        model.fit(kernel, arrays["y_train"])
        fit_seconds = time.perf_counter() - tick
        if model.fit_status_:
            raise RuntimeError("Single SVC did not converge")
        # Both margins come from this exact fit; selection never triggers a refit.
        scores, timing = {}, {}
        for part in ("validation", "test"):
            scores[part], kt, dt = decision_blocks(model, values[part], values["train"], learner, gamma, options["prediction_block"])
            if not np.isfinite(scores[part]).all():
                raise AssertionError("Nonfinite SVM margins")
            timing[f"{part}_kernel_seconds"] = kt
            timing[f"{part}_decision_seconds"] = dt
        threshold, cost = base.threshold_for_cost(arrays["y_validation"], scores["validation"], 0.0)
        fixed = base.metrics(arrays["y_validation"], scores["validation"], 0.0)
        training = times["feature_fit_seconds"] + (times["risk_fit_seconds"] if risk else 0.0)
        training += shared["train_angle_seconds"] + shared["train_state_seconds"] + shared["train_kernel_seconds"] + fit_seconds
        inference = times["test_feature_seconds"] + (times["test_risk_seconds"] if risk else 0.0)
        inference += shared["test_angle_seconds"] + shared["test_state_seconds"] + timing["test_kernel_seconds"] + timing["test_decision_seconds"]
        rows.append({"C": c, "lambda": strength, **geometry, "validation_cost": cost, "fixed_validation_cost": fixed["cost"],
                     "validation_threshold": threshold, "validation_auc": fixed["roc_auc"],
                     "fit_rows": len(arrays["y_train"]), "model_fit_calls": 1, "ensemble_members": 1,
                     "n_support": int(model.n_support_.sum()), "fit_seconds": fit_seconds, "training_seconds": training,
                     "inference_seconds": inference, "train_kernel_entries": len(values["train"]) ** 2,
                     "validation_kernel_entries": len(values["validation"]) * len(values["train"]),
                     "test_kernel_entries": len(values["test"]) * len(values["train"]),
                     "peak_train_kernel_gib": kernel.nbytes / 2**30, **shared, **timing})
        validation.append(scores["validation"])
        test.append(scores["test"])
        del model
    del kernel
    gc.collect()
    return rows, validation, test


def run_unit(unit, options):
    with threadpool_limits(limits=1):
        dataset, seed, learner, representation, strength = unit
        if learner not in LEARNERS or representation not in REPRESENTATIONS:
            raise AssertionError("Unknown learner or representation")
        if representation == "baseline" and strength != 0:
            raise AssertionError("Baseline must remain the no-injection control")
        if representation != "baseline" and strength not in options["protocol_settings"]["lambdas"]:
            raise AssertionError("Risk strength must be on the positive grid")
        folder = unit_folder(unit, options)
        folder.mkdir(parents=True, exist_ok=True)
        with local_lock(folder / "unit.lock"):
            prepared = seed_folder(dataset, seed, options)
            info = json.loads((prepared / "prepared.json").read_text())
            identity = {"unit": list(unit), "options": options, "prepared_sha256": sha(prepared / "prepared.json")}
            receipt, archive = folder / "complete.json", folder / "margins.npz"
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if saved["identity"] != identity or sha(archive) != saved["archive_sha256"]:
                    raise AssertionError("Candidate unit changed")
                return unit
            if sha(prepared / "prepared.npz") != info["archive_sha256"]:
                raise AssertionError("Prepared archive changed")
            with np.load(prepared / "prepared.npz", allow_pickle=False) as source:
                arrays = dict(source)
            if "y_test" in arrays or info["identity"]["options"] != options:
                raise AssertionError("Test labels or mismatched inputs in candidate worker")
            tick = time.perf_counter()
            rows, validation, test = [], [], []
            for geometry in geometries(learner, options["protocol_settings"]):
                a, b, c = fit_geometry(arrays, info["times"], learner, representation, strength, geometry, options["protocol_settings"], options)
                rows.extend(a)
                validation.extend(b)
                test.extend(c)
            np.savez_compressed(archive, validation=np.asarray(validation), test=np.asarray(test))
            write_json(receipt, {"identity": identity, "trials": rows, "archive_sha256": sha(archive),
                                 "elapsed_seconds": time.perf_counter() - tick,
                                 "worker_high_water_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if __import__("sys").platform == "darwin" else 2**20)})
            print(f"lambda_completed={dataset}/{seed}/{learner}/{representation}/{strength:.2f}", flush=True)
            return unit


def choose(rows, policy, backbone=None):
    pool = rows
    if backbone is not None:
        pool = [r for r in pool if all(r[k] == backbone[k] for k in ("C", "alpha", "entanglement", "gamma_factor"))]
    field = "fixed_validation_cost" if policy == "fixed_margin" else "validation_cost"
    best = min(pool, key=lambda r: (r[field], r["lambda"], r["C"], r["alpha"], r["entanglement"], r["gamma_factor"]))
    return {**best, "threshold": 0.0 if policy == "fixed_margin" else best["validation_threshold"]}
