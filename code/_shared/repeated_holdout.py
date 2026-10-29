"""Repeated 80/10/10 splits with new teachers; immutable kernel engine reuse."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from . import controlled as base
from . import overnight_risk as old
from . import pca9_followup as engine
from .angle_encoding import AngleEncoding
from .full_data import write_json
from .improved import local_lock, sha
from .single_811 import split_indices

ROOT, PARTS, METHODS = base.ROOT, engine.PARTS, engine.METHODS
PROTOCOL = ROOT / "docs/cmc_pca7_repeated_protocol_20261020.json"
now, peak_gib, release_heap = engine.now, engine.peak_gib, engine.release_heap
jobs, assert_can_fit = engine.jobs, engine.assert_can_fit


class TrainingPCA(PCA):
    def __init__(self, n_components=7, fit_log=None):
        super().__init__(n_components=n_components, svd_solver="full")
        self.fit_log = fit_log

    def _audit(self, values):
        tolerance = self.singular_values_[0] * max(values.shape) * np.finfo(float).eps
        assert len(self.singular_values_) == self.n_components and self.singular_values_[-1] > tolerance
        if self.fit_log:
            with Path(self.fit_log).open("a") as handle:
                handle.write(json.dumps({"rows": len(values), "encoded_features": values.shape[1],
                    "components": self.n_components, "smallest_retained_singular_value": float(self.singular_values_[-1]),
                    "rank_tolerance": float(tolerance), "input_sha256": engine.digest_array(values)}) + "\n")

    def fit(self, X, y=None):
        result = super().fit(X, y)
        self._audit(X)
        return result

    def fit_transform(self, X, y=None):
        result = super().fit_transform(X, y)
        self._audit(X)
        return result


def features(X, dimension, log=None):
    return Pipeline([*base.make_features(X)[:-1].steps, ("pca", TrainingPCA(dimension, log))])


def risk_estimator(X, seed, dimension, log=None):
    model = Pipeline([("features", features(X, dimension, log)),
        ("lr", LogisticRegression(C=1., class_weight="balanced", max_iter=3000, random_state=seed))])
    return CalibratedClassifierCV(model, method="sigmoid", ensemble=False, n_jobs=1,
        cv=StratifiedKFold(3, shuffle=True, random_state=seed))


def prepare(dataset, seed, protocol, output):
    assert_can_fit(output)
    folder = old.prepared_folder(output, dataset, seed)
    folder.mkdir(parents=True, exist_ok=True)
    with local_lock(folder / "prepare.lock"), threadpool_limits(limits=1):
        assert not (folder / "inputs.npz").exists(), "Preserve preparations; no retry"
        X, y, source = base.load_data(dataset, str(ROOT / "data"))
        split = split_indices(y, seed)
        fit = np.asarray(split["train"])
        dimension = protocol["pca_components"]
        log = str(folder / "nested_pca_fits.jsonl")
        pipeline = features(X, dimension, log)
        tick = time.perf_counter()
        train = pipeline.fit_transform(X.iloc[fit])
        times = {"feature_fit_seconds": time.perf_counter() - tick}
        arrays = {"train": train, "y_train": y[fit], "y_tune": y[split["validation"]]}
        for part, name in (("tune", "validation"), ("gate", "test")):
            arrays[part] = pipeline.transform(X.iloc[split[name]])
        risks, folds = np.zeros(len(fit)), []
        tick = time.perf_counter()
        for index, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(fit, y[fit])):
            model = risk_estimator(X, seed + index + 1, dimension, log)
            model.fit(X.iloc[fit[a]], y[fit[a]])
            risks[b] = model.predict_proba(X.iloc[fit[b]])[:, 1]
            folds.append({"train_ids": fit[a].tolist(), "heldout_ids": fit[b].tolist(),
                "calibration_folds": engine.calibration_indices(fit[a], y[fit[a]], seed + index + 1)})
        model = risk_estimator(X, seed, dimension, log)
        model.fit(X.iloc[fit], y[fit])
        estimator = model.calibrated_classifiers_[0].estimator
        np.testing.assert_allclose(estimator.named_steps["features"].transform(X.iloc[fit]), train, atol=1e-10)
        weights = abs(estimator.named_steps["lr"].coef_[0])
        arrays["weights"] = weights / weights.sum() if weights.sum() else np.ones(dimension) / dimension
        arrays["risk_train"] = risks
        for part, name in (("tune", "validation"), ("gate", "test")):
            arrays["risk_" + part] = model.predict_proba(X.iloc[split[name]])[:, 1]
        times["pca_lr_fit_prediction_seconds"] = time.perf_counter() - tick
        teachers = {}
        for prefix, boosted in (("raw_lr", False), ("gbdt", True)):
            tick = time.perf_counter()
            for part, values in old.raw_teacher(X, y, split, seed, boosted).items():
                arrays[prefix + "_" + part] = values
            teachers[prefix] = {"new_outer_oof_fits": 5, "new_full_training_fit": 1,
                "reused_from_prior_campaign": False, "elapsed_seconds": time.perf_counter() - tick}
        for part, name in (("train", "train"), ("tune", "validation"), ("gate", "test")):
            arrays["rank_" + part] = old.midrank(risks, arrays["risk_" + part])
            arrays["raw_" + part] = pipeline[:-1].transform(X.iloc[split[name]])
            for prefix in teachers:
                arrays[prefix + "_rank_" + part] = old.midrank(arrays[prefix + "_train"], arrays[prefix + "_" + part])
        write_json(folder / "split.json", {"outer": split, "actual": split})
        write_json(folder / "oof.json", {"folds": folds,
            "final_calibration_folds": engine.calibration_indices(fit, y[fit], seed),
            "raw_teacher_folds": [{"train_ids": fit[a].tolist(), "heldout_ids": fit[b].tolist(),
                "estimator_seed": seed + i + 31, "raw_lr_calibration_folds": engine.calibration_indices(fit[a], y[fit[a]], seed + i + 31)}
                for i, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(fit, y[fit]))],
            "raw_lr_final_calibration_folds": engine.calibration_indices(fit, y[fit], seed + 37),
            "ensemble": False, "test_labels_used": False})
        np.savez_compressed(folder / "inputs.npz", **arrays)
        logs = [json.loads(line) for line in Path(log).read_text().splitlines()]
        assert len(logs) == 25 and all(row["components"] == dimension for row in logs)
        record = {"dataset": dataset, "seed": seed, "stage": "full", "source": source,
            "pca_components": dimension, "input_sha256": sha(folder / "inputs.npz"),
            "split_sha256": sha(folder / "split.json"), "oof_sha256": sha(folder / "oof.json"),
            "nested_pca_fits_sha256": sha(log), "nested_pca_fit_count": len(logs),
            "encoder": AngleEncoding.fit(train, "linear_no_clip").parameters(), "times": times,
            "train_rows": len(fit), "tune_rows": len(arrays["tune"]), "gate_rows": len(arrays["gate"]),
            "training_row_ids": fit.tolist(), "outer_test_labels_saved": False,
            "new_pca_lr_fit": True, "teachers": teachers, "prior_preparation_reused": False}
        write_json(folder / "audit.json", record)
        return {"dataset": dataset, "seed": seed, "nested_pca_fits": len(logs), "peak_gib": peak_gib()}


def exact_fit(arrays, record, ctx, job, choice, protocol, output, gpu):
    with engine.exact_slot(output, protocol["exact_workers"]):
        from run_pca9_followup import resources
        while True:
            state, _ = resources()
            reserve = (1 - protocol["memory_limit_fraction"]) * state["host_total_gib"]
            safe = state["host_available_gib"] >= reserve + protocol["memory_safety_gib"] + protocol["exact_budget_gib"]
            if state["cgroup_limit_gib"]:
                safe &= state["cgroup_used_gib"] + protocol["exact_budget_gib"] + protocol["memory_safety_gib"] <= protocol["memory_limit_fraction"] * state["cgroup_limit_gib"]
            if safe:
                break
            time.sleep(5)
        # A separate slot is already held, so call the mathematical fit without nesting slots.
        rep = engine.make_representation(arrays, record, ctx, job[3], job[2], job[4], choice["lambda"], protocol["geometries"][choice["geometry"]])
        kernels, timing, _, setup = engine.matrices(rep, arrays, False, None, gpu)
        row, margins = engine.fit_candidate(arrays, kernels, choice, False, job[1])
        row.update(final_exact=True, final_model_fit_calls=1, selection_was_approximate=True,
            exact_fit_seconds=row["fit_seconds"], exact_kernel_seconds=sum(timing.values()),
            kernel_seconds=timing, train_kernel_seconds=timing["train"], test_kernel_seconds=timing["gate"],
            training_seconds=row["fit_seconds"] + timing["train"] + rep["details"]["state_seconds_by_part"]["train"],
            test_inference_seconds=timing["gate"] + row["inference_seconds"]["gate"] + rep["details"]["state_seconds_by_part"]["gate"],
            representation_details=rep["details"], nystrom_setup_seconds=setup)
        del kernels
        release_heap()
        return row, margins, engine.diagnostic(rep, arrays)


def finish(folder, job, record, hashes, rows, choices, started, diagnostics=None, landmarks=None):
    files = {"margins_sha256": "margins.npz", "exact_margins_sha256": "exact_margins.npz", "parameter_lock_sha256": "parameter_lock.json"}
    if diagnostics:
        np.savez_compressed(folder / "geometry.npz", **{k: v["kernel"] for k, v in diagnostics.items()})
        files["geometry_sha256"] = "geometry.npz"
    write_json(folder / "complete.json", {"job": list(job), "stage": "full", "source_sha256": hashes,
        "input_sha256": record["input_sha256"], "trials": rows, "choices": choices,
        **{key: sha(folder / name) for key, name in files.items()},
        "geometry_positions": {key: value["positions"] for key, value in (diagnostics or {}).items()},
        "landmarks": landmarks.tolist() if landmarks is not None else None,
        "held_out_labels_read": False, "elapsed_seconds": time.perf_counter() - started,
        "worker_pid": os.getpid(), "peak_gib": peak_gib(), "execution_origin": "new_repeated_holdout_single_job_process"})


def run_reference(job, arrays, record, folder, hashes):
    started = time.perf_counter()
    rows, scores = [], {"tune": [], "gate": []}
    label = "pca" + str(record["pca_components"])
    for raw in (False, True):
        values = {part: arrays["raw_" + part] if raw else arrays[part] for part in PARTS}
        for learner, depth in (("RF", 6), ("RF", None), ("XGBoost", 3), ("XGBoost", 6)):
            model = engine.RandomForestClassifier(n_estimators=200, max_depth=depth, class_weight="balanced", n_jobs=1,
                random_state=job[1]) if learner == "RF" else engine.XGBClassifier(n_estimators=200, max_depth=depth,
                learning_rate=.05, tree_method="hist", n_jobs=1, random_state=job[1], eval_metric="logloss",
                scale_pos_weight=float(np.sum(arrays["y_train"] == 0) / np.sum(arrays["y_train"] == 1)))
            tick = time.perf_counter()
            model.fit(values["train"], arrays["y_train"])
            fitted = time.perf_counter() - tick
            tick = time.perf_counter()
            margins = {part: model.predict_proba(values[part])[:, 1] for part in scores}
            prediction_seconds = time.perf_counter() - tick
            row = engine.reference_row(job, arrays, learner, raw, depth, margins, len(rows))
            row.update(method="raw" if raw else label, fit_seconds=fitted, training_seconds=fitted,
                all_holdout_inference_seconds=prediction_seconds, model_fit_calls=1)
            rows.append(row)
            for part in scores:
                scores[part].append(margins[part])
    for raw in (False, True):
        margins = {part: arrays[("raw_lr_" if raw else "risk_") + part] for part in scores}
        row = engine.reference_row(job, arrays, "LR", raw, None, margins, len(rows))
        row.update(method="raw" if raw else label, model_fit_calls=0, preparation_score_reuse=True)
        rows.append(row)
        for part in scores:
            scores[part].append(margins[part])
    old.checkpoint(folder, rows, scores)
    choice = old.select_trial(rows)
    np.savez_compressed(folder / "exact_margins.npz", **{f"reference_{part}": scores[part][choice["score_index"]] for part in scores})
    write_json(folder / "parameter_lock.json", {"choices": {"reference": choice}, "rule": "Validation only"})
    finish(folder, job, record, hashes, rows, {"reference": choice}, started)


def run_job(job, protocol, output, hashes):
    assert_can_fit(output)
    folder = old.job_folder(output, job)
    folder.mkdir(parents=True, exist_ok=True)
    with local_lock(folder / "job.lock"), threadpool_limits(limits=1):
        assert not (folder / "complete.json").exists() and not (folder / "progress.json").exists(), "No automatic retry"
        arrays, record = old.load_inputs(output, *job[:2])
        if job[3] == "reference":
            return run_reference(job, arrays, record, folder, hashes)
        started = time.perf_counter()
        ctx = engine.context(arrays, record, job[3], job[2], job[4], job[1], protocol)
        strengths = [0.] if job[3] == "baseline" else [0., *protocol["lambdas"]]
        approximate = len(arrays["train"]) > protocol["exact_search_max_rows"]
        landmarks = old.subset(np.arange(len(arrays["y_train"])), arrays["y_train"], protocol["nystrom_landmarks"], job[1] + 701)
        gpu = protocol["gpu_devices"][(job[1] // 10) % len(protocol["gpu_devices"])] if protocol["gpu_devices"] else None
        rows, scores = [], {"tune": [], "gate": []}
        for gid, strength in itertools.product(range(len(protocol["geometries"])), strengths):
            rep = engine.make_representation(arrays, record, ctx, job[3], job[2], job[4], strength, protocol["geometries"][gid])
            kernels, timing, spectrum, setup = engine.matrices(rep, arrays, approximate, landmarks, gpu)
            for c in protocol["C_grid"]:
                choice = {"dataset": job[0], "seed": job[1], "learner": job[2], "method": job[3], "arm": job[4],
                    "lambda": strength, "C": c, "geometry": gid, "score_index": len(rows)}
                row, margins = engine.fit_candidate(arrays, kernels, choice, approximate, job[1])
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
        write_json(folder / "parameter_lock.json", {"choices": choices, "rule": "Validation parameters locked before exact fit; no test labels"})
        final, exact_scores, diagnostics = {}, {}, {}
        for role, choice in choices.items():
            if approximate:
                row, margins, diag = exact_fit(arrays, record, ctx, job, choice, protocol, output, gpu)
                row["selection_proxy_trial"] = choice["score_index"]
            else:
                row = {**choice, "final_exact": True, "final_model_fit_calls": 1, "selection_was_approximate": False}
                margins = {part: scores[part][choice["score_index"]] for part in scores}
                rep = engine.make_representation(arrays, record, ctx, job[3], job[2], job[4], choice["lambda"], protocol["geometries"][choice["geometry"]])
                diag = engine.diagnostic(rep, arrays)
            final[role] = row
            for part in scores:
                exact_scores[role + "_" + part] = margins[part]
            diagnostics["exact_" + role] = diag
        np.savez_compressed(folder / "exact_margins.npz", **exact_scores)
        finish(folder, job, record, hashes, rows, final, started, diagnostics, landmarks if approximate else None)
