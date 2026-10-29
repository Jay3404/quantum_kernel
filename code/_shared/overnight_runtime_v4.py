"""Resume verified search prefixes; keep the frozen model and selection algebra."""
import ctypes
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np

from . import overnight_risk as n
from . import overnight_runtime_v3 as v3


def release_heap():
    n.gc.collect()
    trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
    if trim is not None:
        trim(0)


def expected_grid(protocol, method):
    strengths = [0.] if method == "baseline" else [0., *protocol["lambdas"]]
    return [(g, s, c) for g, s, c in itertools.product(range(len(protocol["geometries"])), strengths, protocol["C_grid"])]


def load_checkpoint(folder, arrays, protocol, method):
    folder = Path(folder)
    progress = json.loads((folder / "progress.json").read_text())
    assert not progress["held_out_labels_read"]
    assert n.sha(folder / "margins.npz") == progress["margins_sha256"]
    rows = progress["trials"]
    grid = expected_grid(protocol, method)
    assert len(rows) == progress["candidate_count"] <= len(grid)
    assert len(rows) % len(protocol["C_grid"]) == 0
    assert [(r["geometry"], r["lambda"], r["C"]) for r in rows] == grid[:len(rows)]
    with np.load(folder / "margins.npz", allow_pickle=False) as saved:
        scores = {p: list(saved[p]) for p in ("tune", "gate")}
    for part in scores:
        assert len(scores[part]) == len(rows)
        assert all(v.shape == (len(arrays[part]),) for v in scores[part])
    for index, row in enumerate(rows):
        assert row["score_index"] == index and row["fit_rows"] == len(arrays["train"])
        assert row["approximate_search"] and row["model_fit_calls"] == row["ensemble_members"] == 1
        threshold, _ = n.base.threshold_for_cost(arrays["y_tune"], scores["tune"][index], 0.)
        assert threshold == row["threshold"]
        metrics = n.base.metrics(arrays["y_tune"], scores["tune"][index], threshold)
        for key, value in metrics.items():
            np.testing.assert_allclose(value, row["tune_" + key], atol=1e-12, rtol=1e-12)
    # Old compact checkpoints omit circuit/timing details, not scores or choices.
    for row in rows:
        row["representation_details"] = {"unavailable_from_parent_progress": True}
    lock = folder / "parameter_lock.json"
    if lock.exists():
        assert len(rows) == len(grid)
        for role, original in json.loads(lock.read_text())["choices"].items():
            chosen = n.select_trial(rows, role == "proposed")
            assert all(chosen[k] == original[k] for k in ("score_index", "lambda", "C", "geometry", "threshold", "tune_cost"))
            chosen["representation_details"] = original.get("representation_details", chosen["representation_details"])
    return rows, scores


def run_job(job, stage, protocol, output, hashes):
    folder = n.job_folder(output, job)
    folder.mkdir(parents=True, exist_ok=True)
    arrays, record = n.load_inputs(output, *job[:2])
    parent = folder / "parent_checkpoint"
    if stage != "full" or job[3] == "reference" or len(arrays["train"]) <= protocol["exact_search_max_rows"]:
        assert not parent.exists(), "Resume only declared large-data search checkpoints"
        return v3.run_job(job, stage, protocol, output, hashes)
    with n.local_lock(folder / "job.lock"), n.threadpool_limits(limits=1), v3.implementation():
        assert not (Path(output) / "test_release.json").exists()
        assert not (folder / "complete.json").exists(), "No retry within a continuation run"
        started = time.perf_counter()
        dataset, seed, learner, method, arm = job
        rows, scores = load_checkpoint(parent, arrays, protocol, method) if parent.exists() else ([], {"tune": [], "gate": []})
        inherited = len(rows)
        context = n.representations(arrays, record, method, learner, arm, seed, protocol)
        landmarks = n.subset(np.arange(len(arrays["y_train"])), arrays["y_train"], protocol["nystrom_landmarks"], seed + 701)
        gpu = protocol["gpu_devices"][(seed // 10) % len(protocol["gpu_devices"])] if protocol["gpu_devices"] else None
        strengths = [0.] if method == "baseline" else [0., *protocol["lambdas"]]
        for group, (geometry_id, strength) in enumerate(itertools.product(range(len(protocol["geometries"])), strengths)):
            if group * len(protocol["C_grid"]) < inherited:
                continue
            rep = v3.make_representation(arrays, record, context, method, learner, arm, strength, protocol["geometries"][geometry_id])
            gram = v3.representation_kernel(rep, ("train", landmarks), ("train", landmarks), gpu)
            factor, spectrum = n.nystrom_factor(gram)
            kernels, kernel_times = {}, {}
            for part in n.PARTS:
                tick = time.perf_counter()
                kernels[part] = v3.representation_kernel(rep, (part, slice(None)), ("train", landmarks), gpu) @ factor
                kernel_times[part] = time.perf_counter() - tick
            for c in protocol["C_grid"]:
                model = n.LinearSVC(C=c, class_weight="balanced", dual=False, max_iter=5000, tol=1e-5, random_state=seed)
                tick = time.perf_counter()
                model.fit(kernels["train"], arrays["y_train"])
                fit_seconds = time.perf_counter() - tick
                if model.n_iter_ >= 5000:
                    raise RuntimeError("SVM search candidate did not converge")
                margins, prediction_times = {}, {}
                for part in scores:
                    tick = time.perf_counter()
                    margins[part] = model.decision_function(kernels[part])
                    prediction_times[part] = time.perf_counter() - tick
                threshold, _ = n.base.threshold_for_cost(arrays["y_tune"], margins["tune"], 0.)
                metrics = n.base.metrics(arrays["y_tune"], margins["tune"], threshold)
                rows.append({"dataset": dataset, "seed": seed, "learner": learner, "method": method, "arm": arm,
                    "lambda": strength, "C": c, "geometry": geometry_id, "threshold": threshold,
                    **{"tune_" + k: value for k, value in metrics.items()}, "fit_rows": len(arrays["y_train"]),
                    "approximate_search": True, "fit_seconds": fit_seconds, "kernel_seconds": kernel_times,
                    "inference_seconds": prediction_times,
                    "training_seconds": fit_seconds + kernel_times["train"] + rep["details"]["state_seconds_by_part"]["train"],
                    "test_inference_seconds": kernel_times["gate"] + prediction_times["gate"] + rep["details"]["state_seconds_by_part"]["gate"],
                    "train_kernel_seconds": kernel_times["train"], "test_kernel_seconds": kernel_times["gate"],
                    "representation_details": rep["details"], "nystrom_spectrum": spectrum,
                    "score_index": len(rows), "model_fit_calls": 1, "ensemble_members": 1})
                for part in scores:
                    scores[part].append(margins[part])
                del model
            n.checkpoint(folder, rows, scores)
            n.write_json(folder / "search_details.json", {"trials": rows, "inherited_trial_count": inherited})
            del kernels, rep, gram, factor
            release_heap()
        assert len(rows) == len(expected_grid(protocol, method))
        choices = {"baseline": n.select_trial(rows, False)}
        if method != "baseline":
            choices["proposed"] = n.select_trial(rows, True)
        n.checkpoint(folder, rows, scores)
        n.write_json(folder / "parameter_lock.json", {"choices": choices,
            "rule": "Tune labels only; exact refit parameters locked before exact model fit"})
        exact_scores, diagnostics = {}, {}
        for role, choice in choices.items():
            row, margins, diagnostic = n.exact_fit(arrays, record, context, job, choice, protocol, gpu)
            row["selection_proxy_trial"] = choice["score_index"]
            choices[role] = row
            exact_scores[role + "_tune"], exact_scores[role + "_gate"] = margins["tune"], margins["gate"]
            diagnostics["exact_" + role] = diagnostic
            release_heap()
        np.savez_compressed(folder / "exact_margins.npz", **exact_scores)
        np.savez_compressed(folder / "geometry.npz", **{k: value["kernel"] for k, value in diagnostics.items()})
        n.write_json(folder / "complete.json", {"job": list(job), "stage": stage, "source_sha256": hashes,
            "input_sha256": record["input_sha256"], "trials": rows, "choices": choices,
            "margins_sha256": n.sha(folder / "margins.npz"), "exact_margins_sha256": n.sha(folder / "exact_margins.npz"),
            "parameter_lock_sha256": n.sha(folder / "parameter_lock.json"), "geometry_sha256": n.sha(folder / "geometry.npz"),
            "geometry_positions": {k: value["positions"] for k, value in diagnostics.items()},
            "landmarks": landmarks.tolist(), "held_out_labels_read": False,
            "inherited_search_trials": inherited, "new_search_fit_calls": len(rows) - inherited,
            "elapsed_seconds": time.perf_counter() - started, "worker_pid": os.getpid(),
            "runtime_scope": "New worker elapsed; inherited search trial times retain parent measurements"})
        print("completed", stage, *job, "inherited", inherited, flush=True)
    return job
