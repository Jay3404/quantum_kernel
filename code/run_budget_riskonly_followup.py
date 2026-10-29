"""Frozen, isolated budget control and independently tuned risk-only follow-up."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from analyze_budget_archive_20261026 import choose
from analyze_pca7_professor_comments import corrected, holm, metrics, sha, threshold
from run_risk_only_ablation import features

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "docs/cmc_budget_riskonly_protocol_20261026_v2.json"
RUN = "cmc_budget_riskonly_20261026"
NEW = ["code/run_budget_riskonly_followup.py", "code/analyze_budget_archive_20261026.py",
       "code/monitor_budget_riskonly_followup.py",
       "code/launch_budget_riskonly_followup.py",
       "code/tests/test_budget_archive_20261026.py", "code/tests/test_budget_riskonly_followup.py",
       "docs/cmc_budget_riskonly_protocol_20261026_v2.json"]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + "\n"
    if exclusive:
        with path.open("x") as handle:
            handle.write(payload)
    else:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(payload)
        tmp.replace(path)


def parent():
    return ROOT / "code/results" / read(CONFIG)["parent_run"]


def folder(root, index, probe=False):
    return root / ("preflight" if probe else "jobs") / f"job_{index:04d}"


def verify(root, all_inputs=False):
    meta = read(root / "metadata.json")
    bindings = dict(meta["source_sha256"])
    if all_inputs:
        bindings.update(meta["input_sha256"])
    for name, digest in bindings.items():
        assert sha(ROOT / name) == digest, name
    return meta


def plans(config):
    planned = []
    for d, s in itertools.product(config["datasets"], config["seeds"]):
        prepared = parent() / "full/prepared" / d / f"seed_{s:04d}"
        if d in ("hmeq", "taiwan"):
            for k, m in itertools.product(config["kernels"], config["methods"]):
                source = parent() / "full/candidates" / d / f"seed_{s:04d}" / k / m / "normal/complete.json"
                selected = choose(read(source)["trials"], .5)
                planned.append({"kind": "budget", "dataset": d, "seed": s, "kernel": k, "method": m,
                                "choice": selected, "choice_source": str(source.relative_to(ROOT))})
        for name, k, m in [("sg", "RBF-SVM", "gbdt_soft_gate"),
                           ("qra", "QSVC", "gbdt_rank_append"),
                           ("rra", "RBF-SVM", "gbdt_rank_append")]:
            planned.append({"kind": "risk_only", "context": name, "dataset": d, "seed": s, "kernel": k, "method": m})
    assert len(planned) == 200
    return planned


def initialize(root):
    assert not root.exists(), "Preserve previous attempts; no automatic restart"
    assert ROOT == Path("/workspace/quantum_kernel"), "Scientific fits belong in the approved server container"
    config = read(CONFIG)
    original = read(parent() / "metadata.json")
    assert read(parent() / "validation_receipt.json")["status"] == "passed"
    upstream = read(ROOT / "code/results/cmc_risk_only_ablation_20261023/metadata.json")
    sources = {**original["source_sha256"], **upstream["sources"], **{n: sha(ROOT / n) for n in NEW}}
    inputs = {}
    prior_metadata = ROOT / "code/results/cmc_risk_only_ablation_20261023/metadata.json"
    inputs[str(prior_metadata.relative_to(ROOT))] = sha(prior_metadata)
    for d, s in itertools.product(config["datasets"], config["seeds"]):
        prepared = parent() / "full/prepared" / d / f"seed_{s:04d}"
        for name in ("inputs.npz", "audit.json", "split.json", "oof.json"):
            path = prepared / name
            inputs[str(path.relative_to(ROOT))] = sha(path)
        for k, m in itertools.product(config["kernels"], ["baseline", *config["methods"]]):
            source = parent() / "full/candidates" / d / f"seed_{s:04d}" / k / m / "normal"
            for name in ("complete.json", "margins.npz"):
                path = source / name
                inputs[str(path.relative_to(ROOT))] = sha(path)
    for name in ("metadata.json", "validation_receipt.json", "full/seed_metrics.csv", "full/predictions.csv.gz"):
        path = parent() / name
        inputs[str(path.relative_to(ROOT))] = sha(path)
    root.mkdir(parents=True)
    write(root / "metadata.json", {"run_id": RUN, "status": "initialized", "utc": now(),
          "protocol": config, "source_sha256": sources, "input_sha256": inputs,
          "jobs": plans(config), "completed": 0, "expected": 200, "failures": [],
          "pid": os.getpid(), "python": sys.version, "test_labels_read": False})
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
        for name in sources:
            archive.add(ROOT / name, arcname=name)
    verify(root, True)


def arrays_for(job, probe):
    source = parent() / "full/prepared" / job["dataset"] / f"seed_{job['seed']:04d}"
    record = read(source / "audit.json")
    with np.load(source / "inputs.npz", allow_pickle=False) as saved:
        assert not {"y_gate", "y_test"} & set(saved.files)
        arrays = {key: saved[key] for key in saved.files}
    assert sha(source / "inputs.npz") == record["input_sha256"]
    if probe:
        for key in list(arrays):
            if key == "gate" or key.endswith("_gate"):
                tune_key = "tune" if key == "gate" else key[:-4] + "tune"
                arrays[key] = arrays[tune_key][:2].copy()
    return arrays, record


def gammas(arrays, record):
    protocol = read(ROOT / "docs/cmc_pca7_repeated_protocol_20261020.json")
    angle = np.pi * (arrays["train"] - record["encoder"]["center"]) / record["encoder"]["scale"]
    return [g["gamma_factor"] / max(7 * angle.var(), 1e-12) for g in protocol["geometries"]]


def risk_grid(job, config):
    geometries = [0, 1, 2] if job["context"] == "rra" else [0]
    return [{"C": c, "lambda": strength, "geometry": g}
            for strength, c, g in itertools.product(config["independent_risk_only"]["lambda_grid"],
                config["independent_risk_only"]["C_grid"], geometries)]


def fit_risk(job, arrays, record, out, config, probe):
    q = {p: arrays["gbdt_rank_" + p] for p in ("train", "tune", "gate")}
    bandwidth = gammas(arrays, record)
    grid = [{"C": 100., "lambda": 1., "geometry": 0}] if probe else risk_grid(job, config)
    rows, best, best_model = [], None, None
    for index, choice in enumerate(grid):
        encoded = {p: features(q[p], choice["lambda"], job["kernel"], job["method"])[0] for p in ("train", "tune")}
        kind = features(q["train"][:1], choice["lambda"], job["kernel"], job["method"])[1]
        tick = time.perf_counter()
        model = SVC(C=choice["C"], kernel=kind, gamma=bandwidth[choice["geometry"]] if kind == "rbf" else "scale",
                    class_weight="balanced", cache_size=256, tol=.001)
        model.fit(encoded["train"], arrays["y_train"])
        assert model.fit_status_ == 0 and model.shape_fit_[0] == len(arrays["y_train"])
        score = model.decision_function(encoded["tune"])
        cut = threshold(arrays["y_tune"], score)
        row = {**choice, "score_index": index, "threshold": cut,
               "tune_cost": metrics(arrays["y_tune"], score, cut)["cost"],
               "gamma": bandwidth[choice["geometry"]], "native_kernel": kind,
               "fit_s": time.perf_counter() - tick, "fit_rows": len(arrays["y_train"])}
        rows.append(row)
        key = lambda r: (r["tune_cost"], r["lambda"], r["C"], r["geometry"])
        if best is None or key(row) < key(best):
            best, best_model = row, model
        if index % 10 == 0:
            write(out / "progress.json", {"utc": now(), "trials_completed": len(rows), "expected": len(grid), "best": best}, False)
    assert best_model is not None
    write(out / "validation_selection.json", {"utc": now(), "best": best, "trials": rows,
          "test_labels_read": False, "test_covariates_used_for_selection": False,
          "geometry_duplicates_not_fitted": job["context"] != "rra"})
    encoded = {p: features(q[p], best["lambda"], job["kernel"], job["method"])[0] for p in ("tune", "gate")}
    margins = {p: best_model.decision_function(x) for p, x in encoded.items()}
    joblib.dump(best_model, out / "model.joblib")
    np.savez_compressed(out / "margins.npz", **margins)
    return best, len(rows)


def fit_budget(job, arrays, record, out, probe):
    from _shared import pca9_followup as engine
    protocol = read(ROOT / "docs/cmc_pca7_repeated_protocol_20261020.json")
    choice = dict(job["choice"])
    if probe:
        choice["C"] = 100.
    ctx = engine.context(arrays, record, job["method"], job["kernel"], "normal", job["seed"], protocol)
    rep = engine.make_representation(arrays, record, ctx, job["method"], job["kernel"], "normal", .5, protocol["geometries"][choice["geometry"]])
    gpu = protocol["gpu_devices"][(job["seed"] // 10) % len(protocol["gpu_devices"])]
    kernels, timing, _, _ = engine.matrices(rep, arrays, False, None, gpu)
    model = SVC(C=choice["C"], kernel="precomputed", class_weight="balanced", cache_size=256, tol=.001)
    tick = time.perf_counter()
    model.fit(kernels["train"], arrays["y_train"])
    fit_s = time.perf_counter() - tick
    assert model.fit_status_ == 0 and model.shape_fit_[0] == len(arrays["y_train"])
    margins = {p: model.decision_function(kernels[p]) for p in ("tune", "gate")}
    cut = threshold(arrays["y_tune"], margins["tune"])
    lock = {**{k: choice[k] for k in ("C", "lambda", "geometry")}, "threshold": cut,
            "tune_cost": metrics(arrays["y_tune"], margins["tune"], cut)["cost"],
            "native_kernel": "precomputed", "fit_rows": len(arrays["y_train"]),
            "fit_s": fit_s, "kernel_s": timing, "search_candidates": 9,
            "selection_source": job["choice_source"], "selection_was_approximate": True}
    write(out / "validation_selection.json", {"utc": now(), "best": lock, "test_labels_read": False,
          "no_new_C_or_geometry_search": True})
    joblib.dump(model, out / "model.joblib")
    np.savez_compressed(out / "margins.npz", **margins)
    return lock, 1


def probe_jobs(meta):
    jobs = []
    for d, k, m in itertools.product(("hmeq", "taiwan"), ("RBF-SVM", "QSVC"), ("gbdt_soft_gate", "gbdt_rank_append")):
        jobs.append(next(dict(j) for j in meta["jobs"] if j["kind"] == "budget" and j["dataset"] == d and j["seed"] == 10 and j["kernel"] == k and j["method"] == m))
    for context in ("sg", "qra", "rra"):
        jobs.append(next(dict(j) for j in meta["jobs"] if j["kind"] == "risk_only" and j["dataset"] == "taiwan" and j["seed"] == 10 and j["context"] == context))
    return jobs


def worker(root, index, probe=False):
    meta = verify(root)
    assert not (root / "selection_lock.json").exists() and not (root / "test_release.json").exists()
    job = (probe_jobs(meta) if probe else meta["jobs"])[index]
    out = folder(root, index, probe)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "started.json").exists(), "No automatic retry"
    write(out / "started.json", {"utc": now(), "pid": os.getpid(), "job": job, "probe": probe})
    tick = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            arrays, record = arrays_for(job, probe)
            choice, calls = (fit_budget(job, arrays, record, out, probe) if job["kind"] == "budget"
                             else fit_risk(job, arrays, record, out, meta["protocol"], probe))
        write(out / "complete.json", {"utc": now(), "job": job, "status": "completed", "choice": choice,
              "pid": os.getpid(), "probe": probe, "elapsed_s": time.perf_counter() - tick,
              "model_fit_calls": calls, "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
              "test_covariates_used": not probe, "test_labels_read": False,
              "files_sha256": {n: sha(out / n) for n in ("model.joblib", "margins.npz", "validation_selection.json")}})
    except BaseException:
        write(out / "failure.json", {"utc": now(), "traceback": traceback.format_exc(), "automatic_retry": False})
        raise


def resources():
    info = {line.split(":")[0]: int(line.split()[1]) / 2**20 for line in Path("/proc/meminfo").read_text().splitlines()}
    limit_file, used_file = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
    limit = limit_file.read_text().strip() if limit_file.exists() else "max"
    used = int(used_file.read_text()) / 2**30 if used_file.exists() else 0.
    free = info["MemAvailable"] - .2 * info["MemTotal"] - 8.
    if limit != "max":
        free = min(free, .8 * int(limit) / 2**30 - used - 8.)
    return {"host_total_gib": info["MemTotal"], "host_available_gib": info["MemAvailable"],
            "safe_available_gib": free, "cgroup_limit": limit, "cgroup_used_gib": used,
            "logical_cpus": os.cpu_count(), "load1": os.getloadavg()[0]}


def rss(pid):
    try:
        row = next(line for line in Path(f"/proc/{pid}/status").read_text().splitlines() if line.startswith("VmRSS:"))
        return int(row.split()[1]) / 2**20
    except (OSError, StopIteration):
        return 0.


def call_worker(root, index, probe=False, asynchronous=False):
    out = folder(root, index, probe)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "worker.log").open("x")
    args = [sys.executable, "-B", __file__, "--run-id", root.name, "--operation", "probe" if probe else "worker", "--index", str(index)]
    if asynchronous:
        return subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT), log
    try:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
    finally:
        log.close()


def launch(root):
    meta = verify(root, True)
    assert meta["status"] == "initialized"
    meta.update(status="preparing", pid=os.getpid(), started_utc=now())
    write(root / "metadata.json", meta, False)
    for index in range(11):
        call_worker(root, index, True)
    probes = [read(folder(root, i, True) / "complete.json") for i in range(11)]
    exact_budget = max(2., 1.5 * max(p["peak_gib"] for p in probes[:8]))
    risk_budget = max(2., 1.5 * max(p["peak_gib"] for p in probes[8:]))
    state = resources()
    slots = min(8, max(1, int(state["safe_available_gib"] / exact_budget)))
    workers = min(27, max(1, int(.85 * state["logical_cpus"])))
    write(root / "preflight_receipt.json", {"utc": now(), "status": "passed", "train_validation_only_probes": 11,
          "budget_exact_gib": exact_budget, "budget_risk_gib": risk_budget,
          "exact_slots": slots, "max_workers": workers, "resources": state,
          "probe_sha256": {str(folder(root, i, True).relative_to(root)): sha(folder(root, i, True) / "complete.json") for i in range(11)}})
    meta.update(status="running", scientific_start_utc=now())
    write(root / "metadata.json", meta, False)
    queue, active = list(range(200)), {}
    while queue or active:
        for pid, (process, log, index, budget) in list(active.items()):
            code = process.poll()
            if code is not None:
                log.close()
                del active[pid]
                if code:
                    meta["failures"].append({"index": index, "exit_code": code})
                else:
                    meta["completed"] += 1
        state = resources()
        remaining = sum(max(0., budget - rss(pid)) for pid, (_, _, _, budget) in active.items())
        exact_running = sum(meta["jobs"][i]["kind"] == "budget" for _, _, i, _ in active.values())
        if not meta["failures"] and len(active) < workers and state["load1"] < workers:
            for index in queue:
                exact = meta["jobs"][index]["kind"] == "budget"
                budget = exact_budget if exact else risk_budget
                if exact and exact_running >= slots:
                    continue
                if state["safe_available_gib"] < remaining + budget:
                    continue
                process, log = call_worker(root, index, asynchronous=True)
                active[process.pid] = (process, log, index, budget)
                queue.remove(index)
                break
        exact_running = sum(meta["jobs"][i]["kind"] == "budget" for _, _, i, _ in active.values())
        write(root / "monitor.json", {"utc": now(), "status": "running", "completed": meta["completed"],
              "expected": 200, "queued": len(queue), "failures": meta["failures"],
              "active": [{"pid": pid, "index": value[2], "job": meta["jobs"][value[2]], "rss_gib": rss(pid)} for pid, value in active.items()],
              "resources": state, "exact_slots_in_use": exact_running}, False)
        write(root / "metadata.json", meta, False)
        if meta["failures"] and not active:
            meta["status"] = "failed"
            write(root / "metadata.json", meta, False)
            raise RuntimeError("Scientific failure preserved; queued jobs not retried")
        if queue or active:
            time.sleep(2)
    assert meta["completed"] == 200
    verify(root, True)
    write(root / "selection_lock.json", {"utc": now(), "completed": 200, "test_labels_read": False,
          "jobs_sha256": {str(i): sha(folder(root, i) / "complete.json") for i in range(200)}})
    write(root / "test_release.json", {"utc": now(), "selection_lock_sha256": sha(root / "selection_lock.json"), "new_training_allowed": False})
    evaluate(root)
    validate(root)
    meta.update(status="completed", completed=200, finished_utc=now())
    write(root / "metadata.json", meta, False)
    write(root / "monitor.json", {"utc": now(), "status": "completed", "completed": 200, "expected": 200,
          "active": [], "failures": [], "exact_slots_in_use": 0}, False)


def labels():
    cols = ["dataset", "seed", "row_id", "y", "kernel", "family", "arm", "role"]
    chunks = []
    for chunk in pd.read_csv(parent() / "full/predictions.csv.gz", usecols=cols, chunksize=100000):
        keep = chunk.kernel.eq("QSVC") & chunk.family.eq("baseline") & chunk.arm.eq("normal") & chunk.role.eq("baseline")
        chunks.append(chunk.loc[keep, ["dataset", "seed", "row_id", "y"]])
    frame = pd.concat(chunks, ignore_index=True)
    assert len(frame) == 37960 and not frame.duplicated(["dataset", "seed", "row_id"]).any()
    return frame


def evaluate(root):
    meta = verify(root)
    assert (root / "test_release.json").exists()
    target, old = labels(), pd.read_csv(parent() / "full/seed_metrics.csv")
    records, job_map = [], {}
    for i, j in enumerate(meta["jobs"]):
        job_map[(j["dataset"], j["seed"], j["kernel"] if j["kind"] == "budget" else j["context"], j["method"] if j["kind"] == "budget" else "risk")] = i
    for d, s, k, m in itertools.product(meta["protocol"]["datasets"], meta["protocol"]["seeds"], meta["protocol"]["kernels"], meta["protocol"]["methods"]):
        split = read(parent() / "full/prepared" / d / f"seed_{s:04d}/split.json")["actual"]
        y = target[target.dataset.eq(d) & target.seed.eq(s)].set_index("row_id").reindex(split["test"]).y.to_numpy(dtype=int)
        previous = old[old.dataset.eq(d) & old.seed.eq(s) & old.kernel.eq(k) & old.arm.eq("normal")]
        baseline = previous[previous.family.eq("baseline") & previous.role.eq("baseline")]
        full = previous[previous.family.eq(m) & previous.role.eq("proposed")]
        assert len(baseline) == len(full) == 1
        if d in ("german_numeric", "south_german"):
            path = parent() / "full/candidates" / d / f"seed_{s:04d}" / k / m / "normal"
            choice = choose(read(path / "complete.json")["trials"], .5)
            with np.load(path / "margins.npz", allow_pickle=False) as saved:
                score = saved["gate"][choice["score_index"]]
        else:
            path = folder(root, job_map[(d, s, k, m)])
            choice = read(path / "complete.json")["choice"]
            with np.load(path / "margins.npz", allow_pickle=False) as saved:
                score = saved["gate"]
        budget_result = metrics(y, score, choice["threshold"])
        context = "sg" if m == "gbdt_soft_gate" else ("qra" if k == "QSVC" else "rra")
        path = folder(root, job_map[(d, s, context, "risk")])
        risk_choice = read(path / "complete.json")["choice"]
        with np.load(path / "margins.npz", allow_pickle=False) as saved:
            risk_score = saved["gate"]
        risk_result = metrics(y, risk_score, risk_choice["threshold"])
        records.append({"dataset": d, "seed": s, "kernel": k, "method": m,
                        "baseline9_cost": int(baseline.cost.iloc[0]), "conditioned180_cost": int(full.cost.iloc[0]),
                        "conditioned9_cost": budget_result["cost"], "tuned_risk_only_cost": risk_result["cost"],
                        **{"budget_" + key: value for key, value in budget_result.items()},
                        **{"risk_only_" + key: value for key, value in risk_result.items()}})
    frame = pd.DataFrame(records)
    assert len(frame) == 160
    frame.to_csv(root / "seed_metrics.csv", index=False)
    statistics = []
    contrasts = [("baseline9_minus_conditioned9", "baseline9_cost", "conditioned9_cost"),
                 ("conditioned9_minus_conditioned180", "conditioned9_cost", "conditioned180_cost"),
                 ("tuned_risk_only_minus_conditioned180", "tuned_risk_only_cost", "conditioned180_cost"),
                 ("tuned_risk_only_minus_conditioned9", "tuned_risk_only_cost", "conditioned9_cost")]
    for key, group in frame.groupby(["dataset", "kernel", "method"]):
        group = group.sort_values("seed")
        assert group.seed.tolist() == meta["protocol"]["seeds"]
        for name, left, right in contrasts:
            statistics.append({"dataset": key[0], "kernel": key[1], "method": key[2], "contrast": name,
                               "left_mean": group[left].mean(), "right_mean": group[right].mean(),
                               **corrected(group[left].to_numpy() - group[right].to_numpy())})
    paired = pd.DataFrame(statistics)
    assert len(paired) == 64
    paired["p_holm64"] = holm(paired.p_value)
    paired.to_csv(root / "paired_statistics_Holm64.csv", index=False)


def validate(root):
    meta = verify(root, True)
    assert (root / "selection_lock.json").exists() and (root / "test_release.json").exists()
    assert read(root / "selection_lock.json")["utc"] <= read(root / "test_release.json")["utc"]
    replayed, budget_replayed = 0, 0
    for i, j in enumerate(meta["jobs"]):
        out = folder(root, i)
        complete = read(out / "complete.json")
        assert not (out / "failure.json").exists() and complete["test_labels_read"] is False
        for name, digest in complete["files_sha256"].items():
            assert sha(out / name) == digest
        model = joblib.load(out / "model.joblib")
        arrays, record = arrays_for(j, False)
        choice = complete["choice"]
        selection = read(out / "validation_selection.json")
        if j["kind"] == "risk_only":
            trials = selection["trials"]
            assert len(trials) == len(risk_grid(j, meta["protocol"]))
            assert choice == min(trials, key=lambda r: (r["tune_cost"], r["lambda"], r["C"], r["geometry"]))
        else:
            archived = choose(read(ROOT / j["choice_source"])["trials"], .5)
            assert all(choice[key] == archived[key] for key in ("C", "lambda", "geometry"))
        assert model.shape_fit_[0] == choice["fit_rows"] == len(arrays["y_train"])
        with np.load(out / "margins.npz", allow_pickle=False) as saved:
            cut = threshold(arrays["y_tune"], saved["tune"])
            assert cut == choice["threshold"]
            assert metrics(arrays["y_tune"], saved["tune"], cut)["cost"] == choice["tune_cost"]
            if j["kind"] == "risk_only":
                for part in ("tune", "gate"):
                    x = features(arrays["gbdt_rank_" + part], choice["lambda"], j["kernel"], j["method"])[0]
                    np.testing.assert_allclose(model.decision_function(x), saved[part], rtol=1e-10, atol=1e-10)
                replayed += 1
            else:
                from _shared import pca9_followup as engine
                protocol = read(ROOT / "docs/cmc_pca7_repeated_protocol_20261020.json")
                ctx = engine.context(arrays, record, j["method"], j["kernel"], "normal", j["seed"], protocol)
                rep = engine.make_representation(arrays, record, ctx, j["method"], j["kernel"], "normal", .5,
                                                  protocol["geometries"][choice["geometry"]])
                gpu = protocol["gpu_devices"][(j["seed"] // 10) % len(protocol["gpu_devices"])]
                for part in ("tune", "gate"):
                    cross = engine.kernel(rep, (part, slice(None)), ("train", slice(None)), gpu)
                    np.testing.assert_allclose(model.decision_function(cross), saved[part], rtol=1e-8, atol=1e-8)
                    del cross
                budget_replayed += 1
    old_seed = pd.read_csv(root / "seed_metrics.csv")
    old_stats = pd.read_csv(root / "paired_statistics_Holm64.csv")
    evaluate(root)
    pd.testing.assert_frame_equal(old_seed, pd.read_csv(root / "seed_metrics.csv"))
    pd.testing.assert_frame_equal(old_stats, pd.read_csv(root / "paired_statistics_Holm64.csv"))
    write(root / "validation_receipt.json", {"utc": now(), "status": "passed", "jobs": 200,
          "saved_models": 200, "risk_model_prediction_replays": replayed,
          "budget_model_prediction_replays": budget_replayed,
          "budget_prediction_replay_scope": "Regenerated exact validation/test cross-Grams against all training rows; no new fitting or training-Gram refit",
          "seed_rows": 160, "paired_contrasts": 64, "holm_family": 64,
          "sources_verified": len(meta["source_sha256"]), "inputs_verified": len(meta["input_sha256"]),
          "test_lock_order_passed": True, "full_training_rows_verified": True,
          "risk_tuning_choices_and_thresholds_use_validation_only": True,
          "new_fits_during_validation": 0, "independent_external_confirmation": False}, exclusive=not (root / "validation_receipt.json").exists())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN)
    parser.add_argument("--operation", choices=["initialize", "run", "worker", "probe", "validate"], default="run")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    assert args.run_id == RUN
    root = ROOT / "code/results" / args.run_id
    if args.operation == "initialize":
        initialize(root)
    elif args.operation in ("worker", "probe"):
        worker(root, args.index, args.operation == "probe")
    elif args.operation == "validate":
        validate(root)
    else:
        try:
            launch(root)
        except BaseException:
            if root.exists():
                write(root / ("pipeline_failure_" + str(time.time_ns()) + ".json"),
                      {"utc": now(), "traceback": traceback.format_exc(), "automatic_retry": False})
                meta = read(root / "metadata.json")
                meta.update(status="failed", failure_utc=now())
                write(root / "metadata.json", meta, False)
                write(root / "monitor.json", {"utc": now(), "status": "failed", "completed": meta["completed"],
                      "expected": 200, "automatic_retry": False, "inspect_live_children_before_any_action": True}, False)
            raise


if __name__ == "__main__":
    main()
