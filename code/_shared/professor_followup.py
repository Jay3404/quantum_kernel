"""Isolated professor follow-up; completed campaign inputs are read-only."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from analyze_pca7_professor_comments import (
    audit_oof, backbone, corrected, holm, kernel_metrics, metrics, probability_metrics, sha)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/cmc_professor_followup_protocol_20261022.json"
NEW_SOURCES = ("code/_shared/professor_followup.py", "code/run_professor_followup.py",
    "code/report_professor_followup.py", "code/tests/test_professor_followup.py",
    "docs/cmc_professor_followup_protocol_20261022.json")
PARTS = ("train", "tune", "gate")


def now():
    return datetime.now(timezone.utc).isoformat()


def software():
    import platform
    import sklearn
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__}


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True,
        default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + "\n"
    if exclusive:
        with path.open("x") as handle:
            handle.write(text)
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def protocol():
    return json.loads(PROTOCOL.read_text())


def parent(p=None):
    return ROOT / "code/results" / (p or protocol())["upstream_run"]


def verify_sources(root, parent_inputs=True):
    meta = json.loads((root / "metadata.json").read_text())
    values = {**meta["source_sha256"], **(meta["parent_sha256"] if parent_inputs else {})}
    for name, digest in values.items():
        assert sha(ROOT / name) == digest, name
    return meta


def freeze(root):
    import tarfile
    p = protocol()
    old = parent(p)
    previous = json.loads((old / "metadata.json").read_text())
    assert previous["status"] == "completed" and previous["completed_jobs"] == 1160 and not previous["failed_jobs"]
    assert json.loads((old / "validation_receipt.json").read_text())["status"] == "passed"
    nine = ROOT / "code/results/cmc_pca9_repeated_20261021"
    nine_meta = json.loads((nine / "metadata.json").read_text())
    assert nine_meta["status"] == "completed" and nine_meta["completed_jobs"] == 870 and not nine_meta["failed_jobs"]
    sources = {**previous["source_sha256"], **nine_meta["source_sha256"], **{name: sha(ROOT / name) for name in NEW_SOURCES},
        "code/analyze_pca7_professor_comments.py": sha(ROOT / "code/analyze_pca7_professor_comments.py")}
    files = ["metadata.json", "validation_receipt.json", "full/seed_metrics.csv", "full/paired_cost.csv",
        "full/selection_lock.json", "full/test_release.json", "full/verification_receipt.json"]
    inputs = {str((old / name).relative_to(ROOT)): sha(old / name) for name in files}
    for folder in (old / "full/prepared").glob("*/seed_*"):
        for name in ("inputs.npz", "audit.json", "split.json", "oof.json"):
            file = folder / name
            inputs[str(file.relative_to(ROOT))] = sha(file)
    inputs[str((old / "full/predictions.csv.gz").relative_to(ROOT))] = sha(old / "full/predictions.csv.gz")
    for dataset in p["datasets"]:
        for seed in p["seeds"]:
            for learner in p["learners"]:
                for method in ("baseline", *p["methods"]):
                    folder = old / "full/candidates" / dataset / f"seed_{seed:04d}" / learner / method / "normal"
                    for name in ("complete.json", "exact_margins.npz", "geometry.npz"):
                        file = folder / name
                        inputs[str(file.relative_to(ROOT))] = sha(file)
    for file in (nine / "metadata.json", nine / "validation_receipt.json",
            ROOT / "docs/cmc_pca7_repeated_final_ko.md", ROOT / "docs/cmc_pca9_repeated_final_ko.md"):
        inputs[str(file.relative_to(ROOT))] = sha(file)
    meta = {"run_id": p["run_id"], "status": "initialized", "started_utc": now(), "protocol": p,
        "source_sha256": sources, "parent_sha256": inputs, "completed_jobs": 0,
        "expected_jobs": 200, "failed_jobs": [], "test_released": False, "initialization_software": software()}
    write(root / "metadata.json", meta, True)
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
        for name in sources:
            archive.add(ROOT / name, arcname=name)
    verify_sources(root)
    return meta


def inputs(dataset, seed, p=None):
    folder = parent(p) / "full/prepared" / dataset / f"seed_{seed:04d}"
    audit = json.loads((folder / "audit.json").read_text())
    assert sha(folder / "inputs.npz") == audit["input_sha256"]
    with np.load(folder / "inputs.npz", allow_pickle=False) as saved:
        arrays = dict(saved)
    assert "y_gate" not in arrays and "y_test" not in arrays
    split = json.loads((folder / "split.json").read_text())["actual"]
    return arrays, audit, split, folder


def cost_cut(y, scores, default=1 / 6, rank_stable=False):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    assert y.shape == scores.shape and y.ndim == 1 and len(y) and np.isfinite(scores).all()
    assert np.isin(y, [0, 1]).all()
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(y) - 1]
    cuts = np.r_[np.nextafter(ordered[0], np.inf), ordered[ends]]
    tp, fp = np.r_[0, np.cumsum(y[order])[ends]], np.r_[0, np.cumsum(1 - y[order])[ends]]
    costs = 5 * (y.sum() - tp) + fp
    best = np.flatnonzero(costs == costs.min())
    index = best[0] if rank_stable else best[np.argmin(abs(cuts[best] - default))]
    return float(cuts[index])


def calibrate(y, raw_train, values, method):
    y, raw_train = np.asarray(y), np.asarray(raw_train, dtype=float)
    assert y.shape == raw_train.shape and np.isfinite(raw_train).all()
    if method == "raw":
        return {key: value.copy() for key, value in values.items()}, None, {"method": method}
    started = time.perf_counter()
    if method == "sigmoid":
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=3000, tol=1e-10)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(raw_train[:, None], y)
        transformed = {key: model.predict_proba(value[:, None])[:, 1] for key, value in values.items()}
        detail = {"slope": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0]),
            "strictly_increasing": bool(model.coef_[0, 0] > 0), "iterations": int(model.n_iter_[0])}
    elif method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(raw_train, y)
        transformed = {key: model.predict(value) for key, value in values.items()}
        detail = {"knots": len(model.X_thresholds_), "may_introduce_ties": True}
    else:
        raise ValueError(method)
    assert all(np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all() for v in transformed.values())
    return transformed, model, {"method": method, "fit_rows": len(y), "fit_scope": "training5OOF_scores_only",
        "fit_and_prediction_seconds": time.perf_counter() - started, **detail}


def calibration_pair(root, dataset, seed):
    p = protocol()
    folder = root / "calibration" / dataset / f"seed_{seed:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "started.json").exists(), "No scientific retry"
    write(folder / "started.json", {"utc": now(), "dataset": dataset, "seed": seed}, True)
    arrays, audit, split, original = inputs(dataset, seed, p)
    folds = audit_oof(original, arrays, lambda path: path)
    folds.pop("test_ids")
    saved, selections, calibration_details = {}, [], []
    for method in p["calibration"]["methods"]:
        scores, model, details = calibrate(arrays["y_train"], arrays["gbdt_train"],
            {part: arrays["gbdt_" + part] for part in ("tune", "gate")}, method)
        calibration_details.append(details)
        if model is not None:
            joblib.dump(model, folder / (method + ".joblib"))
        saved.update({method + "_" + part: value for part, value in scores.items()})
        for policy in p["calibration"]["thresholds"]:
            cut = .5 if policy == "probability_0.5" else 1 / 6 if policy == "bayes_1_6" else cost_cut(
                arrays["y_tune"], scores["tune"], rank_stable=policy == "validation_cost_rank_stable")
            selections.append({"dataset": dataset, "seed": seed, "calibration": method,
                "threshold_policy": policy, "threshold": cut,
                "validation_cost": metrics(arrays["y_tune"], scores["tune"], cut)["cost"]})
    np.savez_compressed(folder / "scores.npz", **saved)
    write(folder / "complete.json", {"status": "completed", "utc": now(), "selections": selections,
        "calibrators": calibration_details, "oof": folds, "scores_sha256": sha(folder / "scores.npz"),
        "parent_input_sha256": audit["input_sha256"], "new_xgb_fits": 0,
        "test_labels_read_for_fitting_or_selection": False, "calibration_software": software()}, True)


def test_labels(p=None):
    p = p or protocol()
    chunks = []
    for chunk in pd.read_csv(parent(p) / "full/predictions.csv.gz", chunksize=100000,
            usecols=["dataset", "seed", "row_id", "y", "kernel", "family", "arm", "role"]):
        keep = chunk.kernel.eq("QSVC") & chunk.family.eq("baseline") & chunk.arm.eq("normal") & chunk.role.eq("baseline")
        chunks.append(chunk.loc[keep, ["dataset", "seed", "row_id", "y"]])
    labels = pd.concat(chunks, ignore_index=True)
    assert not labels.duplicated(["dataset", "seed", "row_id"]).any()
    return labels


def calibration_evaluate(root):
    p = protocol()
    completions = []
    for dataset in p["datasets"]:
        for seed in p["seeds"]:
            path = root / "calibration" / dataset / f"seed_{seed:04d}" / "complete.json"
            value = json.loads(path.read_text())
            assert value["status"] == "completed"
            completions.append({"path": str(path.relative_to(root)), "sha256": sha(path)})
    assert len(completions) == 40
    write(root / "calibration_selection_lock.json", {"utc": now(), "pairs": completions,
        "scope": "All calibration maps and validation-only thresholds locked before test-label evaluation"}, True)
    labels = test_labels(p)
    rows, probability, invariance = [], [], []
    for dataset in p["datasets"]:
        for seed in p["seeds"]:
            folder = root / "calibration" / dataset / f"seed_{seed:04d}"
            complete = json.loads((folder / "complete.json").read_text())
            assert sha(folder / "scores.npz") == complete["scores_sha256"]
            arrays, _, split, _ = inputs(dataset, seed, p)
            y_test = labels[labels.dataset.eq(dataset) & labels.seed.eq(seed)].set_index("row_id").reindex(split["test"]).y.to_numpy(dtype=int)
            with np.load(folder / "scores.npz", allow_pickle=False) as z:
                scores = dict(z)
            for choice in complete["selections"]:
                for part, y in (("validation", arrays["y_tune"]), ("test", y_test)):
                    value = scores[choice["calibration"] + ("_tune" if part == "validation" else "_gate")]
                    rows.append({**choice, "part": part, **metrics(y, value, choice["threshold"]), "n": len(y)})
            for method in p["calibration"]["methods"]:
                value, _ = probability_metrics(y_test, scores[method + "_gate"])
                probability.append({"dataset": dataset, "seed": seed, "calibration": method, **value})
            raw_cut = next(c["threshold"] for c in complete["selections"] if c["calibration"] == "raw" and c["threshold_policy"] == "validation_cost_rank_stable")
            sigmoid_cut = next(c["threshold"] for c in complete["selections"] if c["calibration"] == "sigmoid" and c["threshold_policy"] == "validation_cost_rank_stable")
            mismatch = int(np.sum((scores["raw_gate"] >= raw_cut) != (scores["sigmoid_gate"] >= sigmoid_cut)))
            increasing = next(c["strictly_increasing"] for c in complete["calibrators"] if c["method"] == "sigmoid")
            if increasing:
                assert mismatch == 0, "Strictly monotone rank-stable decisions changed"
            invariance.append({"dataset": dataset, "seed": seed, "strictly_increasing": increasing,
                "rank_stable_test_prediction_mismatches": mismatch})
    pd.DataFrame(rows).to_csv(root / "calibration_metrics.csv", index=False)
    pd.DataFrame(probability).to_csv(root / "calibration_probability_metrics.csv", index=False)
    pd.DataFrame(invariance).to_csv(root / "calibration_rank_invariance.csv", index=False)
    return {"pairs": 40, "threshold_metric_rows": len(rows), "probability_metric_rows": len(probability)}


def random_values(dataset, seed, draw, lengths, p=None):
    p = p or protocol()
    assert dataset in p["random_datasets"] and draw in p["random_draws"]
    return {part: np.random.default_rng(np.random.SeedSequence(
        [20261022, p["datasets"].index(dataset), seed, draw, i])).uniform(size=lengths[part])
        for i, part in enumerate(PARTS)}


def jobs(p=None):
    p = p or protocol()
    random = [("random", d, s, k, draw) for d in p["random_datasets"] for s in p["seeds"]
        for k in p["learners"] for draw in p["random_draws"]]
    matched = [("backbone", d, s, k, m) for d in p["random_datasets"] for s in p["seeds"]
        for k in p["learners"] for m in p["methods"]]
    assert len(random) == 120 and len(matched) == 80
    return random + matched


def job_folder(root, job):
    mode, dataset, seed, kernel, setting = job
    return root / "science" / mode / dataset / f"seed_{seed:04d}" / kernel / str(setting)


def fit_allowed(root):
    assert not (root / "selection_lock.json").exists(), "No fitting after scientific global lock"
    assert not (root / "test_release.json").exists(), "No fitting after test-label evaluation"


def actual_choice(dataset, seed, kernel, method):
    path = parent() / "full/candidates" / dataset / f"seed_{seed:04d}" / kernel / method / "normal/complete.json"
    value = json.loads(path.read_text())
    audit = parent() / "full/prepared" / dataset / f"seed_{seed:04d}" / "audit.json"
    assert value["input_sha256"] == json.loads(audit.read_text())["input_sha256"]
    return value["choices"]["proposed"]


def fit_exact(root, folder, arrays, audit, context, kernel, choice, role, gpu, probe=False):
    from _shared import pca9_followup as engine
    from run_pca9_followup import resources
    from sklearn.svm import SVC
    p = protocol()
    meta = json.loads((root / "metadata.json").read_text())
    count = meta.get("scheduling", {}).get("exact_workers", 1)
    budget = meta.get("scheduling", {}).get("exact_budget_gib", 32.)
    with engine.exact_slot(root / "science", count):
        fit_allowed(root)
        while True:
            state, _ = resources()
            reserve = (1 - p["resources"]["memory_limit_fraction"]) * state["host_total_gib"]
            safe = state["host_available_gib"] >= reserve + p["resources"]["memory_safety_gib"] + budget
            if state["cgroup_limit_gib"]:
                safe &= state["cgroup_used_gib"] + budget + p["resources"]["memory_safety_gib"] <= .8 * state["cgroup_limit_gib"]
            if safe:
                break
            time.sleep(5)
        tick = time.perf_counter()
        method = "baseline" if role == "matched_backbone" else "gbdt_rank_append"
        representation = engine.make_representation(arrays, audit, context, method, kernel,
            "normal", choice["lambda"], p["geometries"][choice["geometry"]])
        matrices, timing, _, _ = engine.matrices(representation, arrays, False, None, gpu)
        model = SVC(kernel="precomputed", C=choice["C"], class_weight="balanced", cache_size=256)
        started = time.perf_counter()
        model.fit(matrices["train"], arrays["y_train"])
        assert model.fit_status_ == 0
        fit_seconds = time.perf_counter() - started
        margins = {part: model.decision_function(matrices[part]) for part in ("tune", "gate")}
        assert all(np.isfinite(v).all() for v in margins.values())
        cut = cost_cut(arrays["y_tune"], margins["tune"], default=0)
        positions = np.linspace(0, len(arrays["gate"]) - 1, min(256, len(arrays["gate"])), dtype=int)
        gram = engine.kernel(representation, ("gate", positions), ("gate", positions), gpu)
        row = {**choice, "role": role, "threshold": cut,
            "tune_cost": metrics(arrays["y_tune"], margins["tune"], cut)["cost"],
            "fit_rows": len(arrays["y_train"]), "model_fit_calls": 1, "ensemble_members": 1,
            "fit_seconds": fit_seconds, "kernel_seconds": timing,
            "training_and_prediction_seconds": time.perf_counter() - tick,
            "representation_details": representation["details"], "geometry_positions": positions.tolist(),
            "test_labels_used": False, "probe": probe}
        if not probe:
            joblib.dump(model, folder / (role + "_model.joblib"))
        del model, matrices, representation
        engine.release_heap()
    return row, margins, gram


def scientific_job(root, job, probe=False):
    import os
    from _shared import pca9_followup as engine, overnight_risk as old
    from threadpoolctl import threadpool_limits
    fit_allowed(root)
    p = protocol()
    mode, dataset, seed, kernel, setting = job
    folder = (root / "preflight" / f"probe_{dataset}_{kernel}") if probe else job_folder(root, job)
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "started.json").exists(), "No scientific retry"
    write(folder / "started.json", {"job": list(job), "pid": os.getpid(), "utc": now()}, True)
    started = time.perf_counter()
    arrays, audit, split, _ = inputs(dataset, seed, p)
    if probe:
        arrays = {key: value.copy() for key, value in arrays.items()}
        for key in list(arrays):
            if key == "gate" or key.endswith("_gate"):
                arrays[key] = arrays["tune" if key == "gate" else key[:-4] + "tune"].copy()
    if mode == "random":
        q = random_values(dataset, seed, setting, {part: len(arrays[part]) for part in PARTS}, p)
        if probe:
            q["gate"] = q["tune"].copy()
        arrays = {**arrays, **{f"gbdt_rank_{part}": q[part] for part in PARTS},
            **{f"gbdt_{part}": q[part] for part in PARTS}}
    gpu = None if probe else p["resources"]["gpu_devices"][(seed // 10) % len(p["resources"]["gpu_devices"])]
    with threadpool_limits(limits=1):
        context = engine.context(arrays, audit, "gbdt_rank_append", kernel, "normal", seed, p)
        if mode == "random":
            np.savez_compressed(folder / "random_feature.npz", **q)
            approximate = len(arrays["train"]) > p["exact_search_max_rows"]
            assert approximate, "Random control scope is full HMEQ/Taiwan"
            landmarks = old.subset(np.arange(len(arrays["y_train"])), arrays["y_train"], p["nystrom_landmarks"], seed + 701)
            rows, saved = [], {}
            strengths = [0., .5] if probe else [0., *p["lambdas"]]
            geometries = p["geometries"][:1] if probe else p["geometries"]
            for gid, geometry in enumerate(geometries):
                for strength in strengths:
                    representation = engine.make_representation(arrays, audit, context, "gbdt_rank_append", kernel,
                        "normal", strength, geometry)
                    matrices, _, _, _ = engine.matrices(representation, arrays, True, landmarks, gpu)
                    for c in p["C_grid"]:
                        choice = {"lambda": strength, "C": c, "geometry": gid, "score_index": len(rows)}
                        row, scores = engine.fit_candidate(arrays, matrices, choice, True, seed)
                        rows.append(row)
                    del matrices, representation
                    engine.release_heap()
                write(folder / "search_checkpoint.json", {"job": list(job), "trials": rows,
                    "completed_geometry": gid, "utc": now()})
            search_peak = engine.peak_gib()
            positive = old.select_trial(rows, True)
            zero = old.select_trial(rows, False)
            matched = actual_choice(dataset, seed, kernel, "gbdt_rank_append")
            choices = {"own_zero": zero, "tuned": positive, "matched": matched}
            if probe:
                choices = {"probe": {"lambda": .5, "C": 1., "geometry": 0}}
        else:
            actual = actual_choice(dataset, seed, kernel, setting)
            choices = {"matched_backbone": {**actual, "lambda": 0.}}
            search_peak = engine.peak_gib()
        choices = {role: {key: value[key] for key in ("lambda", "C", "geometry")} for role, value in choices.items()}
        write(folder / "parameter_lock.json", {"job": list(job), "choices": choices,
            "rule": "Validation-only random search or immutable parent-selected parameters; no test labels", "utc": now()}, True)
        final, margins, grams = {}, {}, {}
        for role, choice in choices.items():
            row, scores, gram = fit_exact(root, folder, arrays, audit, context, kernel, choice, role, gpu, probe)
            final[role] = row
            margins.update({role + "_" + part: value for part, value in scores.items()})
            grams[role] = gram
        np.savez_compressed(folder / "margins.npz", **margins)
        np.savez_compressed(folder / "geometry.npz", **grams)
    result = {"status": "completed", "job": list(job), "worker_pid": os.getpid(), "finished_utc": now(),
        "elapsed_seconds": time.perf_counter() - started, "choices": final, "search_peak_gib": search_peak,
        "peak_gib": engine.peak_gib(), "parent_input_sha256": audit["input_sha256"],
        "margins_sha256": sha(folder / "margins.npz"), "geometry_sha256": sha(folder / "geometry.npz"),
        "parameter_lock_sha256": sha(folder / "parameter_lock.json"), "test_labels_read": False,
        "probes_use_validation_as_test_surrogate": probe, "software": software(),
        "models_sha256": {} if probe else {role: sha(folder / (role + "_model.joblib")) for role in choices}}
    write(folder / "complete.json", result, True)
    return result


def vector(frame, **filters):
    chosen = frame
    for column, value in filters.items():
        chosen = chosen[chosen[column].eq(value)]
    chosen = chosen.sort_values("seed")
    assert chosen.seed.tolist() == list(range(10, 101, 10)), filters
    return chosen


def parent_model(raw, dataset, kernel, method):
    return vector(raw, dataset=dataset, kernel=kernel, family=method, arm="normal",
        role="baseline" if method == "baseline" else "proposed")


def add_contrast(rows, directions, dataset, kernel, method, contrast, delta, family="primary_cost"):
    value = corrected(np.asarray(delta, dtype=float))
    rows.append({"dataset": dataset, "kernel": kernel, "method": method, "contrast": contrast,
        "testing_family": family, **value})
    directions.extend({"dataset": dataset, "kernel": kernel, "method": method, "contrast": contrast,
        "seed": seed, "delta": float(d)} for seed, d in zip(range(10, 101, 10), delta))


def existing_analysis(root):
    p = protocol()
    raw = pd.read_csv(parent(p) / "full/seed_metrics.csv")
    cal = pd.read_csv(root / "calibration_metrics.csv")
    cal = cal[cal.part.eq("test") & cal.threshold_policy.eq("validation_cost")]
    probability = pd.read_csv(root / "calibration_probability_metrics.csv")
    rows, directions = [], []
    for dataset in p["datasets"]:
        teacher = {method: vector(cal, dataset=dataset, calibration=method).cost.to_numpy()
            for method in p["calibration"]["methods"]}
        for method in ("sigmoid", "isotonic"):
            add_contrast(rows, directions, dataset, "XGB", method, "raw_minus_calibrated_cost",
                teacher["raw"] - teacher[method])
        reference = vector(raw, dataset=dataset, role="reference_raw_XGBoost").cost.to_numpy()
        for learner in p["learners"]:
            for method in p["methods"]:
                proposed = parent_model(raw, dataset, learner, method).cost.to_numpy()
                for label, cost in {**teacher, "tuned_reference": reference}.items():
                    add_contrast(rows, directions, dataset, learner, method, "XGB_" + label + "_minus_fusion", cost - proposed)
        for method in p["methods"]:
            gains = {kernel: parent_model(raw, dataset, kernel, "baseline").cost.to_numpy()
                - parent_model(raw, dataset, kernel, method).cost.to_numpy() for kernel in p["learners"]}
            add_contrast(rows, directions, dataset, "interaction", method, "QSVC_minus_RBF_tuned_baseline_gain",
                gains["QSVC"] - gains["RBF-SVM"])
        for method in ("sigmoid", "isotonic"):
            for metric in ("auroc", "ap", "brier", "log_loss", "ece10_uniform"):
                a = vector(probability, dataset=dataset, calibration="raw")[metric].to_numpy()
                b = vector(probability, dataset=dataset, calibration=method)[metric].to_numpy()
                delta = b - a if metric in ("auroc", "ap") else a - b
                add_contrast(rows, directions, dataset, "XGB", method, metric + "_improvement", delta, "secondary_calibration")
    table = pd.DataFrame(rows)
    secondary = table.testing_family.eq("secondary_calibration")
    assert secondary.sum() == 40 and (~secondary).sum() == 80
    table["p_holm"] = np.nan
    table.loc[secondary, "p_holm"] = holm(table.loc[secondary, "p_value"].to_numpy())
    table["family_size"] = np.where(secondary, 40, 100)
    table["complete_family"] = secondary
    table.to_csv(root / "available_paired_statistics.csv", index=False)
    pd.DataFrame(directions).to_csv(root / "available_seed_differences.csv", index=False)
    raw.to_csv(root / "parent_seed_metrics_copy.csv", index=False)
    return {"available_primary_contrasts": 80, "pending_primary_contrasts": 20,
        "secondary_contrasts": 40, "primary_holm_not_applied_until_all100": True}


def evaluate_science(root):
    p = protocol()
    planned = jobs(p)
    completions = []
    for job in planned:
        path = job_folder(root, job) / "complete.json"
        value = json.loads(path.read_text())
        assert value["status"] == "completed" and not value["test_labels_read"]
        completions.append({"job": list(job), "path": str(path.relative_to(root)), "sha256": sha(path)})
    write(root / "selection_lock.json", {"utc": now(), "jobs": completions,
        "scope": "All200 fresh jobs and self-validation thresholds locked; parent tests already known; exploratory only"}, True)
    labels = test_labels(p)
    rows, predictions = [], []
    for job in planned:
        mode, dataset, seed, kernel, setting = job
        folder = job_folder(root, job)
        complete = json.loads((folder / "complete.json").read_text())
        assert sha(folder / "margins.npz") == complete["margins_sha256"]
        _, _, split, _ = inputs(dataset, seed, p)
        y = labels[labels.dataset.eq(dataset) & labels.seed.eq(seed)].set_index("row_id").reindex(split["test"]).y.to_numpy(dtype=int)
        with np.load(folder / "margins.npz", allow_pickle=False) as z:
            scores = dict(z)
        for role, choice in complete["choices"].items():
            score = scores[role + "_gate"]
            assert len(score) == len(y)
            identity = {"mode": mode, "dataset": dataset, "seed": seed, "kernel": kernel,
                "setting": setting, "role": role}
            row = {**identity, **{key: choice[key] for key in ("lambda", "C", "geometry", "threshold", "fit_rows", "tune_cost")},
                **metrics(y, score, choice["threshold"])}
            positions = np.asarray(choice["geometry_positions"])
            row["subset_cost"] = metrics(y[positions], score[positions], choice["threshold"])["cost"]
            rows.append(row)
            predictions.append(pd.DataFrame({**identity, "row_id": split["test"], "y": y, "score": score,
                "threshold": choice["threshold"], "prediction": (score >= choice["threshold"]).astype(int)}))
    pd.DataFrame(rows).to_csv(root / "new_seed_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(root / "new_predictions.csv.gz", index=False)
    write(root / "test_release.json", {"utc": now(), "selection_lock_sha256": sha(root / "selection_lock.json"),
        "new_model_rows": len(rows), "scope": "No new fitting or selection after scientific lock"}, True)


def complete_statistics(root):
    p = protocol()
    raw = pd.read_csv(root / "parent_seed_metrics_copy.csv")
    new = pd.read_csv(root / "new_seed_metrics.csv")
    rows = pd.read_csv(root / "available_paired_statistics.csv").drop(columns=["p_holm", "family_size", "complete_family"]).to_dict("records")
    directions = pd.read_csv(root / "available_seed_differences.csv").to_dict("records")
    for dataset in p["random_datasets"]:
        for kernel in p["learners"]:
            actual = parent_model(raw, dataset, kernel, "gbdt_rank_append").cost.to_numpy()
            baseline = parent_model(raw, dataset, kernel, "baseline").cost.to_numpy()
            selected = new[new["mode"].eq("random") & new.dataset.eq(dataset) & new.kernel.eq(kernel)]
            def average(role):
                group = selected[selected.role.eq(role)].groupby("seed").cost.agg(["count", "mean"])
                assert group.index.tolist() == p["seeds"] and group["count"].eq(3).all()
                return group["mean"].to_numpy()
            tuned, matched = average("tuned"), average("matched")
            add_contrast(rows, directions, dataset, kernel, "random_append", "baseline_minus_random_tuned", baseline - tuned)
            add_contrast(rows, directions, dataset, kernel, "gbdt_rank_append", "random_tuned_minus_actual", tuned - actual)
            add_contrast(rows, directions, dataset, kernel, "gbdt_rank_append", "random_matched_minus_actual", matched - actual)
            for method in p["methods"]:
                control = vector(new, mode="backbone", dataset=dataset, kernel=kernel, setting=method).cost.to_numpy()
                value = parent_model(raw, dataset, kernel, method).cost.to_numpy()
                add_contrast(rows, directions, dataset, kernel, method, "matched_backbone_minus_actual", control - value)
    table = pd.DataFrame(rows)
    for family, size in (("primary_cost", 100), ("secondary_calibration", 40)):
        mask = table.testing_family.eq(family)
        assert mask.sum() == size
        table.loc[mask, "p_holm"] = holm(table.loc[mask, "p_value"].to_numpy())
        table.loc[mask, "family_size"] = size
    table.to_csv(root / "paired_statistics.csv", index=False)
    pd.DataFrame(directions).to_csv(root / "seed_differences.csv", index=False)
    return table


def mechanism_analysis(root):
    from analyze_pca7_professor_comments import risk_factor
    p = protocol()
    labels = test_labels(p)
    raw = pd.read_csv(root / "parent_seed_metrics_copy.csv")
    new = pd.read_csv(root / "new_seed_metrics.csv")
    rows = []
    for dataset in p["random_datasets"]:
        for seed in p["seeds"]:
            arrays, audit, split, _ = inputs(dataset, seed, p)
            y = labels[labels.dataset.eq(dataset) & labels.seed.eq(seed)].set_index("row_id").reindex(split["test"]).y.to_numpy(dtype=int)
            for kernel in p["learners"]:
                for method in p["methods"]:
                    original = parent(p) / "full/candidates" / dataset / f"seed_{seed:04d}" / kernel / method / "normal"
                    chosen = actual_choice(dataset, seed, kernel, method)
                    controlled = job_folder(root, ("backbone", dataset, seed, kernel, method))
                    control = json.loads((controlled / "complete.json").read_text())["choices"]["matched_backbone"]
                    positions = np.asarray(control["geometry_positions"])
                    with np.load(original / "geometry.npz", allow_pickle=False) as z:
                        actual_gram = z["exact_proposed"]
                    with np.load(controlled / "geometry.npz", allow_pickle=False) as z:
                        base_gram = z["matched_backbone"]
                    with np.load(original / "exact_margins.npz", allow_pickle=False) as z:
                        actual_score = z["proposed_gate"]
                    assert chosen["geometry"] == control["geometry"] and chosen["C"] == control["C"]
                    geometry = p["geometries"][chosen["geometry"]]
                    train_angles = np.pi * (arrays["train"] - audit["encoder"]["center"]) / audit["encoder"]["scale"]
                    gamma = geometry["gamma_factor"] / max(7 * train_angles.var(), 1e-12)
                    expected = base_gram * risk_factor(arrays["gbdt_rank_gate"][positions], chosen["lambda"], kernel, method, gamma)
                    error = float(abs(expected - actual_gram).max())
                    assert error < 1e-8, "Matched geometry/kernel product mismatch"
                    actual_metrics = kernel_metrics(actual_gram, y[positions])
                    base_metrics = kernel_metrics(base_gram, y[positions])
                    actual_row = raw[
                        raw.dataset.eq(dataset) & raw.seed.eq(seed) & raw.kernel.eq(kernel) & raw.family.eq(method) & raw.arm.eq("normal") & raw.role.eq("proposed")]
                    base_row = new[new["mode"].eq("backbone") & new.dataset.eq(dataset) & new.seed.eq(seed) & new.kernel.eq(kernel) & new.setting.eq(method)]
                    assert len(actual_row) == len(base_row) == 1
                    rows.append({"dataset": dataset, "seed": seed, "kernel": kernel, "method": method,
                        "C": chosen["C"], "geometry": chosen["geometry"], "lambda": chosen["lambda"],
                        "kernel_product_max_error": error, "n_subset": len(positions),
                        "alignment_before": base_metrics["centered_label_alignment"], "alignment_after": actual_metrics["centered_label_alignment"],
                        "alignment_delta": actual_metrics["centered_label_alignment"] - base_metrics["centered_label_alignment"],
                        "effective_rank_before": base_metrics["entropy_effective_rank"], "effective_rank_after": actual_metrics["entropy_effective_rank"],
                        "effective_rank_delta": actual_metrics["entropy_effective_rank"] - base_metrics["entropy_effective_rank"],
                        "full_cost_reduction": float(base_row.cost.iloc[0] - actual_row.cost.iloc[0]),
                        "subset_cost_reduction": float(base_row.subset_cost.iloc[0] - metrics(y[positions], actual_score[positions], chosen["threshold"])["cost"])})
    frame = pd.DataFrame(rows)
    assert len(frame) == 80
    frame.to_csv(root / "mechanism_seed_metrics.csv", index=False)
    correlations = []
    for key, group in frame.groupby(["dataset", "kernel", "method"]):
        assert len(group) == 10
        for x in ("alignment_delta", "effective_rank_delta"):
            for y in ("full_cost_reduction", "subset_cost_reduction"):
                rho = stats.spearmanr(group[x], group[y]).statistic if group[x].nunique() > 1 and group[y].nunique() > 1 else np.nan
                correlations.append({"dataset": key[0], "kernel": key[1], "method": key[2], "x": x, "y": y,
                    "spearman_rho": rho, "n_outer_seeds": 10, "inference": "Descriptive only; overlapping holdouts; no causal mechanism claim or iid correlation p-value"})
    pd.DataFrame(correlations).to_csv(root / "mechanism_correlations.csv", index=False)
