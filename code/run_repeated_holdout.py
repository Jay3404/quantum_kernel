"""Frozen repeated-holdout preflight, resource admission and fresh job execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

from _shared import repeated_holdout as p
from _shared import overnight_risk as old
from run_pca9_followup import SOURCES as PRIOR_SOURCES, active, resources

SOURCES = tuple(dict.fromkeys((*PRIOR_SOURCES, "code/_shared/repeated_holdout.py",
    "code/_shared/repeated_reporting.py", "code/run_repeated_holdout.py", "code/validate_repeated_holdout.py",
    "code/monitor_repeated_holdout.py", "code/tests/test_repeated_holdout.py",
    "docs/cmc_pca7_repeated_protocol_20261020.json", "docs/cmc_pca9_repeated_followup_request_20261020.json",
    "code/REPEATED_HOLDOUT_EXPERIMENTS.md")))


def stop_gate():
    root = p.ROOT / "code/results/cmc_pca9_20261020"
    path = root / "operations/authorized_stop_20261020/stop_receipt.json"
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "authorized_stopped" and receipt["remaining_active_workers"] == 0
    assert not active(receipt["experiment_pid"]) and not active(receipt["supervisor_pid"])
    for file in (root / "full/candidates").rglob("worker_operation.json"):
        saved = json.loads(file.read_text())
        assert saved.get("finished_utc") or not active(saved["pid"])
    old_root = p.ROOT / "code/results/cmc_20261019_overnight_risk_v4"
    return {str(path.relative_to(p.ROOT)): p.sha(path), **{str((old_root / name).relative_to(p.ROOT)): p.sha(old_root / name)
        for name in ("validation_receipt.json", "continuation_validation_receipt.json", "full/selection_lock.json", "full/test_release.json")}}


def admitted(state, scheduling, protocol):
    reserve = (1 - protocol["memory_limit_fraction"]) * state["host_total_gib"] + protocol["memory_safety_gib"]
    if state["host_available_gib"] < reserve + scheduling["worker_budget_gib"]:
        return False
    if state["cpu_percent"] >= 100 * protocol["cpu_limit_fraction"]:
        return False
    if state["cgroup_limit_gib"]:
        return state["cgroup_used_gib"] + scheduling["worker_budget_gib"] + protocol["memory_safety_gib"] <= protocol["memory_limit_fraction"] * state["cgroup_limit_gib"]
    return True


def scheduling_for(state, search_peak, exact_peak, protocol):
    budget = max(2., 1.5 * search_peak + .5)
    exact_budget = max(2., 1.5 * exact_peak + .5)
    ceiling = protocol["memory_limit_fraction"] * state["host_total_gib"]
    if state["cgroup_limit_gib"]:
        ceiling = min(ceiling, protocol["memory_limit_fraction"] * state["cgroup_limit_gib"])
    workers = min(protocol["max_workers"], max(1, int(protocol["cpu_limit_fraction"] * state["logical_cpu_limit"])),
        max(1, int((ceiling - protocol["memory_safety_gib"] - exact_budget) / budget)))
    exact = min(protocol["max_exact_workers"], workers,
        max(1, int((ceiling - protocol["memory_safety_gib"] - workers * budget) / exact_budget)))
    assert workers * budget + exact * exact_budget + protocol["memory_safety_gib"] <= ceiling
    return {"workers": workers, "exact_workers": exact, "worker_budget_gib": budget,
        "exact_budget_gib": exact_budget, "measured_search_peak_gib": search_peak, "measured_exact_peak_gib": exact_peak}


def operation(args):
    root = p.ROOT / "code/results" / args.run_id
    meta = json.loads((root / "metadata.json").read_text())
    assert all(p.sha(p.ROOT / name) == value for name, value in meta["source_sha256"].items())
    protocol = meta["protocol"]
    if args.operation == "prepare":
        pairs = [(d, s) for d in protocol["datasets"] for s in protocol["seeds"]]
        result = p.prepare(*pairs[args.index], protocol, root / "full")
        p.write_json(root / "preflight" / f"prepare_{args.index:03d}.json", {"status": "passed", **result, "pid": os.getpid()})
        return
    if args.operation == "probe":
        import numpy as np
        items = [(d, k, m) for d in ("hmeq", "taiwan") for k in protocol["learners"] for m in protocol["methods"]]
        dataset, learner, method = items[args.index]
        job = (dataset, 10, learner, method, "normal")
        arrays, record = old.load_inputs(root / "full", dataset, 10)
        arrays = {key: value.copy() for key, value in arrays.items()}
        for key in list(arrays):
            if key == "gate" or key.endswith("_gate"):
                source = "tune" if key == "gate" else key[:-4] + "tune"
                arrays[key] = arrays[source].copy()
        ctx = p.engine.context(arrays, record, method, learner, "normal", 10, protocol)
        choice = {"lambda": .5, "C": 1., "geometry": 0, "score_index": 0}
        tick = time.perf_counter()
        rep = p.engine.make_representation(arrays, record, ctx, method, learner, "normal", .5, protocol["geometries"][0])
        landmarks = old.subset(np.arange(len(arrays["train"])), arrays["y_train"], 512, 711)
        matrices, _, _, _ = p.engine.matrices(rep, arrays, True, landmarks, None)
        for c in protocol["C_grid"]:
            p.engine.fit_candidate(arrays, matrices, {**choice, "C": c}, True, 10)
        search_peak = p.peak_gib()
        del matrices
        p.release_heap()
        probe_protocol = {**protocol, "exact_workers": 1, "exact_budget_gib": 32.}
        row, margins, _ = p.exact_fit(arrays, record, ctx, job, choice, probe_protocol, root / "full", None)
        assert row["fit_rows"] == len(arrays["train"]) and len(margins["gate"]) == len(arrays["tune"])
        p.write_json(root / "preflight" / f"probe_{args.index:03d}.json", {"status": "passed", "job": list(job),
            "search_peak_gib": search_peak, "full_peak_gib": p.peak_gib(), "elapsed_seconds": time.perf_counter() - tick,
            "full_train_rows": row["fit_rows"], "test_labels_read": False, "test_features_or_scores_used": False,
            "performance_used_for_selection": False, "pid": os.getpid()})
        return
    tasks = p.jobs(protocol)
    job = tuple(tasks[args.index])
    folder = old.job_folder(root / "full", job)
    record = {"job": list(job), "pid": os.getpid(), "started_utc": p.now(), "fresh_process_for_one_job": True,
        "source_sha256": meta["source_sha256"]}
    p.write_json(folder / "worker_operation.json", record)
    try:
        p.run_job(job, protocol, root / "full", meta["source_sha256"])
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(finished_utc=p.now(), peak_gib=p.peak_gib())
        p.write_json(folder / "worker_operation.json", record)


def child(args, mode, index):
    log = p.ROOT / "code/results" / args.run_id / "preflight" / f"{mode}_{index:03d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as handle:
        return subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id,
            "--operation", mode, "--index", str(index)], cwd=p.ROOT, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT).returncode


def preflight(args):
    root = p.ROOT / "code/results" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "metadata.json").exists(), "No preflight retry"
    protocol = json.loads(p.PROTOCOL.read_text())
    assert args.run_id == protocol["run_id"]
    preserved = stop_gate()
    hashes = {name: p.sha(p.ROOT / name) for name in SOURCES}
    meta = {"run_id": args.run_id, "status": "preparing", "phase": "technical_preflight", "started_utc": p.now(),
        "pid": os.getpid(), "protocol": protocol, "source_sha256": hashes, "preserved_inputs_sha256": preserved,
        "failed_jobs": [], "test_released": False, "completed_jobs": 0, "expected_jobs": len(p.jobs(protocol))}
    p.write_json(root / "metadata.json", meta)
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
        for name in SOURCES:
            archive.add(p.ROOT / name, arcname=name)
    try:
        with (root / "preflight_tests.log").open("x") as handle:
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "code/tests", "-p", "test_repeated_holdout.py", "-v"],
                cwd=p.ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = {pool.submit(child, args, "prepare", i): i for i in range(len(protocol["datasets"]) * len(protocol["seeds"]))}
            for future in as_completed(futures):
                assert future.result() == 0, f"Preparation failed:{futures[future]}; no retry"
        for i in range(16):
            assert child(args, "probe", i) == 0, f"Memory probe failed:{i}; no retry"
        import numpy as np
        checks = []
        rng = np.random.default_rng(20261020)
        for device in protocol["gpu_devices"]:
            for kind, dimension in (("rbf", 7), ("fidelity", 128), ("fidelity", 256)):
                values = rng.normal(size=(1001, dimension))
                if kind == "fidelity":
                    values = values + 1j * rng.normal(size=values.shape)
                    values /= np.linalg.norm(values, axis=1, keepdims=True)
                cpu = old.primitive_kernel(values, values, kind, .3)
                gpu = old.primitive_kernel(values, values, kind, .3, device)
                np.testing.assert_allclose(cpu, gpu, atol=1e-11, rtol=1e-11)
                checks.append({"device": device, "dimension": dimension, "kind": kind,
                    "max_absolute_error": float(abs(cpu - gpu).max())})
        probes = [json.loads(file.read_text()) for file in (root / "preflight").glob("probe_*.json")]
        preparations = [json.loads(file.read_text()) for file in (root / "preflight").glob("prepare_*.json")]
        assert len(preparations) == 40 and len(probes) == 16
        state, _ = resources()
        scheduling = scheduling_for(state, max(row["search_peak_gib"] for row in probes), max(row["full_peak_gib"] for row in probes), protocol)
        protocol.update(exact_workers=scheduling["exact_workers"], exact_budget_gib=scheduling["exact_budget_gib"])
        meta["scheduling"] = scheduling
        receipt = {"status": "passed", "finished_utc": p.now(), "prepared_pairs": 40,
            "nested_pca_fits": sum(row["nested_pca_fits"] for row in preparations), "probe_count": len(probes),
            "gpu_gram_checks": checks, "source_sha256": hashes, "scheduling": scheduling,
            "test_labels_used_for_feasibility": False,
            "outputs_sha256": {str(file.relative_to(root)): p.sha(file) for file in (root / "preflight").glob("*")}}
        from validate_repeated_holdout import audit_prepared
        receipt["split_and_oof_audit"] = audit_prepared(root, protocol)
        p.write_json(root / "preflight_receipt.json", receipt)
        meta.update(status="ready", phase="preflight_passed", preflight_sha256=p.sha(root / "preflight_receipt.json"))
    except BaseException as error:
        meta.update(status="preflight_failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        p.write_json(root / "preflight_failure.json", {"status": "failed", "time_utc": p.now(), "error": meta["error"], "automatic_retry": False})
        raise
    finally:
        p.write_json(root / "metadata.json", meta)


def execute(args):
    root = p.ROOT / "code/results" / args.run_id
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["status"] == "ready" and not (root / "full/stage_manifest.json").exists()
    assert all(p.sha(p.ROOT / name) == value for name, value in meta["source_sha256"].items())
    assert p.sha(root / "preflight_receipt.json") == meta["preflight_sha256"]
    stop_gate()
    protocol = meta["protocol"]
    tasks = p.jobs(protocol)
    ordered = sorted(enumerate(tasks), key=lambda pair: (pair[1][4] != "normal", protocol["datasets"].index(pair[1][0]), pair[1][1], pair[1][2], pair[1][3]))
    p.write_json(root / "full/stage_manifest.json", {"jobs": tasks, "source_sha256": meta["source_sha256"], "committed_utc": p.now(),
        "performance_based_task_removal": False, "dispatch_order": [list(job) for _, job in ordered]})
    meta.update(status="running", phase="full_search", pid=os.getpid(), scientific_started_utc=p.now())
    p.write_json(root / "metadata.json", meta)
    queue, running, previous = list(ordered), {}, None
    try:
        while queue or running:
            state, previous = resources(previous)
            for pid, (process, handle, job) in list(running.items()):
                status = process.poll()
                if status is None:
                    continue
                handle.close()
                del running[pid]
                if status:
                    meta["failed_jobs"].append({"job": list(job), "pid": pid, "exit_code": status, "no_retry": True})
                else:
                    meta["completed_jobs"] += 1
            if (root / "operations/stop_requested.json").exists():
                queue.clear()
                meta["stop_requested"] = True
            while queue and not meta["failed_jobs"] and len(running) < meta["scheduling"]["workers"] and admitted(state, meta["scheduling"], protocol):
                index, job = queue.pop(0)
                folder = old.job_folder(root / "full", job)
                folder.mkdir(parents=True, exist_ok=True)
                handle = (folder / "worker.log").open("x")
                process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id,
                    "--operation", "job", "--index", str(index)], cwd=p.ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
                running[process.pid] = process, handle, job
                state["host_available_gib"] -= meta["scheduling"]["worker_budget_gib"]
                if state["cgroup_used_gib"] is not None:
                    state["cgroup_used_gib"] += meta["scheduling"]["worker_budget_gib"]
            p.write_json(root / "monitor.json", {"status": "running", "pid": os.getpid(), "heartbeat_utc": p.now(),
                "completed_jobs": meta["completed_jobs"], "expected_jobs": len(tasks), "failed_jobs": meta["failed_jobs"],
                "active": [{"pid": pid, "job": list(value[2])} for pid, value in running.items()], "queued_jobs": len(queue),
                "resources": state, "scheduling": meta["scheduling"], "automatic_retries": False})
            p.write_json(root / "metadata.json", meta)
            if meta["failed_jobs"] and not running:
                raise RuntimeError("Worker failure; no automatic retry")
            if queue or running:
                time.sleep(10)
        if meta.get("stop_requested"):
            meta.update(status="authorized_stopped", phase="preserved_partial", finished_utc=p.now())
            return
        assert meta["completed_jobs"] == len(tasks) == 1160
        meta["phase"] = "global_selection_and_test_release"
        p.write_json(root / "metadata.json", meta)
        from _shared.repeated_reporting import finalize
        finalize(root, meta)
        meta["test_released"] = True
        p.write_json(root / "metadata.json", meta)
        subprocess.run([sys.executable, "code/validate_repeated_holdout.py", "--run-id", args.run_id], cwd=p.ROOT, check=True)
        meta.update(status="completed", phase="completed_full", finished_utc=p.now())
    except BaseException as error:
        meta.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        p.write_json(root / "metadata.json", meta)
        p.write_json(root / "monitor.json", {"status": meta["status"], "pid": os.getpid(), "heartbeat_utc": p.now(),
            "completed_jobs": meta["completed_jobs"], "expected_jobs": len(tasks), "failed_jobs": meta["failed_jobs"], "error": meta.get("error")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_pca7_repeated_20261020")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--launch-pipeline", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--operation", choices=("prepare", "probe", "job"))
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    assert args.run_id == Path(args.run_id).name and args.run_id.startswith("cmc_pca7_repeated_")
    if args.operation:
        operation(args)
    elif args.launch_pipeline:
        root = p.ROOT / "code/results" / args.run_id
        root.mkdir(parents=True, exist_ok=True)
        assert not (root / "metadata.json").exists()
        with (root / "server.log").open("x") as handle:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id, "--pipeline"],
                cwd=p.ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"run_id": args.run_id, "pid": process.pid, "status": "preflight_pipeline_started"}))
    elif args.preflight or args.pipeline:
        preflight(args)
        if args.pipeline:
            execute(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
