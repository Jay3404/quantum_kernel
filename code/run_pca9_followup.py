"""Preflight and detached, resource-admitted PCA9 sensitivity execution."""
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

from _shared import pca9_followup as p
from _shared import overnight_risk as old
import run_overnight_risk_v4 as upstream

SOURCES = tuple(dict.fromkeys((*upstream.SOURCES, "code/_shared/pca9_followup.py",
    "code/run_pca9_followup.py", "code/_shared/pca9_reporting.py", "code/validate_pca9_followup.py",
    "code/tests/test_pca9_followup.py", "docs/cmc_pca9_protocol_20261020.json",
    "docs/cmc_pca9_followup_request_20261019.json")))


def active(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")


def verify_upstream(protocol):
    assert p.sha(p.REQUEST) == p.REQUEST_SHA
    root = p.ROOT / "code/results" / protocol["upstream_run"]
    metadata = json.loads((root / "metadata.json").read_text())
    a = json.loads((root / "validation_receipt.json").read_text())
    b = json.loads((root / "continuation_validation_receipt.json").read_text())
    monitor = json.loads((root / "monitor.json").read_text())
    extension = json.loads((root / "operations/deadline_extension_20261019/extension_receipt.json").read_text())
    assert metadata["status"] == monitor["status"] == "completed"
    assert metadata["completed_jobs"] == metadata["expected_jobs"] == 1160 and not metadata["failed_jobs"]
    assert a["status"] == b["status"] == "passed" and b["original_audit_sha256"] == p.sha(root / "validation_receipt.json")
    assert a["stages"]["full"]["complete_jobs"] == 1160 and a["stages"]["full"]["missing_jobs"] == 0
    assert all(p.sha(p.ROOT / name) == value for name, value in metadata["source_sha256"].items())
    for key in ("experiment_pid", "controller_pid", "paused_original_supervisor_pid", "paused_guard_lock_holder_pid"):
        assert not active(extension[key]), f"Upstream process still active: {key}"
    inodes = {str(path.stat().st_ino) for path in (p.ROOT / ".overnight-locks/exact_slots").glob("*.lock")}
    for line in Path("/proc/locks").read_text().splitlines():
        assert not any(f":{inode} " in line for inode in inodes), "Upstream exact slot still locked"
    return {str((root / name).relative_to(p.ROOT)): p.sha(root / name) for name in
        ("metadata.json", "validation_receipt.json", "continuation_validation_receipt.json", "source_snapshot.tar.gz",
         "full/selection_lock.json", "full/test_release.json", "full/verification_receipt.json")}


def resources(previous=None):
    info = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()}
    cpu = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
    idle, total = cpu[3] + cpu[4], sum(cpu)
    percent = 0. if previous is None or total == previous[1] else 100 * (1 - (idle - previous[0]) / (total - previous[1]))
    limit, current = None, None
    for cap, used in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                      ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        if Path(cap).exists():
            text = Path(cap).read_text().strip()
            if text != "max" and int(text) < 2**60:
                limit, current = int(text), int(Path(used).read_text())
            break
    cpus = len(os.sched_getaffinity(0))
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        q, period = quota.read_text().split()
        if q != "max":
            cpus = min(cpus, max(1, int(q) // int(period)))
    return {"host_total_gib": info["MemTotal"] / 2**30, "host_available_gib": info["MemAvailable"] / 2**30,
        "host_used_fraction": 1 - info["MemAvailable"] / info["MemTotal"], "cpu_percent": percent,
        "logical_cpu_limit": cpus, "cgroup_limit_gib": limit / 2**30 if limit else None,
        "cgroup_used_gib": current / 2**30 if current is not None else None}, (idle, total)


def admitted(state, scheduling, protocol):
    reserve = .2 * state["host_total_gib"] + protocol["memory_safety_gib"] + scheduling["worker_budget_gib"]
    if state["host_available_gib"] < reserve or state["cpu_percent"] >= 100 * protocol["cpu_limit_fraction"]:
        return False
    if state["cgroup_limit_gib"] is not None:
        return state["cgroup_used_gib"] + scheduling["worker_budget_gib"] + protocol["memory_safety_gib"] <= .8 * state["cgroup_limit_gib"]
    return True


def operation(args, mode, index):
    root = p.ROOT / "code/results" / args.run_id
    meta = json.loads((root / "metadata.json").read_text())
    assert all(p.sha(p.ROOT / name) == value for name, value in meta["source_sha256"].items())
    protocol = meta["protocol"]
    if mode == "prepare":
        pairs = [(d, s) for d in protocol["datasets"] for s in protocol["seeds"]]
        result = p.prepare(*pairs[index], protocol, root / "full")
        p.write_json(root / "preflight" / f"prepare_{index:03d}.json", {"status": "passed", **result, "pid": os.getpid()})
        return
    if mode == "probe":
        learner, method = [(k, m) for k in protocol["learners"] for m in protocol["methods"]][index]
        job = ("hmeq", 2010, learner, method, "normal")
        arrays, record = old.load_inputs(root / "full", *job[:2])
        arrays = {key: value.copy() for key, value in arrays.items()}
        for key in list(arrays):
            if key == "gate" or key.endswith("_gate"):
                source = "tune" if key == "gate" else key[:-4] + "tune"
                arrays[key] = arrays[source].copy()
        ctx = p.context(arrays, record, method, learner, "normal", 2010, protocol)
        choice = {"lambda": .5, "C": 1., "geometry": 0, "score_index": 0}
        tick = time.perf_counter()
        rep = p.make_representation(arrays, record, ctx, method, learner, "normal", .5, protocol["geometries"][0])
        landmarks = old.subset(__import__("numpy").arange(len(arrays["train"])), arrays["y_train"], 512, 2711)
        matrices, _, _, _ = p.matrices(rep, arrays, True, landmarks, None)
        for c in protocol["C_grid"]:
            p.fit_candidate(arrays, matrices, {**choice, "C": c}, True, 2010)
        del matrices
        p.release_heap()
        row, margins, _ = p.exact_fit(arrays, record, ctx, job, choice, protocol, root / "full", None)
        assert row["fit_rows"] == 4768 and len(margins["tune"]) == len(margins["gate"]) == 596
        p.write_json(root / "preflight" / f"probe_{index:03d}.json", {"status": "passed", "job": list(job),
            "full_train_rows": 4768, "peak_gib": p.peak_gib(), "elapsed_seconds": time.perf_counter() - tick,
            "test_labels_read": False, "test_features_or_scores_used": False,
            "probe_evaluation_scope": "Validation rows also stand in for held-out rows of identical size; actual test inputs never reach probe fitting/prediction",
            "performance_used_for_selection": False, "pid": os.getpid()})
        return
    plan = json.loads((root / "full/stage_manifest.json").read_text())
    job = tuple(plan["jobs"][index])
    folder = old.job_folder(root / "full", job)
    record = {"job": list(job), "pid": os.getpid(), "started_utc": p.now(),
        "fresh_process_for_one_job": True, "source_sha256": meta["source_sha256"]}
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


def child(args, mode, index, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    assert not log.exists(), "Do not overwrite failed attempt logs"
    with log.open("x") as handle:
        return subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id,
            "--operation", mode, "--index", str(index)], cwd=p.ROOT, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT).returncode


def preflight(args):
    root = p.ROOT / "code/results" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "metadata.json").exists(), "No automatic preflight retry"
    protocol = json.loads(p.PROTOCOL.read_text())
    upstream_hashes = verify_upstream(protocol)
    local_gate = json.loads((root / "upstream_local_archive_receipt.json").read_text())
    assert local_gate["status"] == "passed" and local_gate["upstream_run"] == protocol["upstream_run"]
    assert all(local_gate["upstream_receipts_sha256"][name] == value for name, value in upstream_hashes.items())
    hashes = {name: p.sha(p.ROOT / name) for name in SOURCES}
    meta = {"run_id": args.run_id, "status": "preparing", "phase": "technical_preflight", "started_utc": p.now(),
        "protocol": protocol, "source_sha256": hashes, "preserved_inputs_sha256": upstream_hashes,
        "upstream_local_archive_receipt_sha256": p.sha(root / "upstream_local_archive_receipt.json"),
        "pid": os.getpid(), "failed_jobs": [], "test_released": False, "completed_jobs": 0,
        "expected_jobs": len(p.jobs(protocol)), "full_families": protocol["methods"]}
    p.write_json(root / "metadata.json", meta)
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
        for name in SOURCES:
            archive.add(p.ROOT / name, arcname=name)
    try:
        with (root / "preflight_tests.log").open("x") as log:
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "code/tests", "-p", "test_pca9_followup.py", "-v"],
                cwd=p.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(child, args, "prepare", i, root / "preflight" / f"prepare_{i:03d}.log"): i
                for i in range(len(protocol["datasets"]) * len(protocol["seeds"]))}
            for future in as_completed(futures):
                assert future.result() == 0, f"Preparation failed at {futures[future]}; no retry"
        for i in range(len(protocol["learners"]) * len(protocol["methods"])):
            assert child(args, "probe", i, root / "preflight" / f"probe_{i:03d}.log") == 0, f"Memory probe failed at {i}; no retry"
        import numpy as np
        import cupy as cp
        rng = np.random.default_rng(20261020)
        checks = []
        for device in protocol["gpu_devices"]:
            properties = cp.cuda.runtime.getDeviceProperties(device)
            for kind, size in (("rbf", 9), ("fidelity", 512), ("fidelity", 1024)):
                values = rng.normal(size=(1001, size))
                if kind == "fidelity":
                    values = values + 1j * rng.normal(size=values.shape)
                    values /= np.linalg.norm(values, axis=1, keepdims=True)
                cpu = old.primitive_kernel(values, values, kind, .3)
                gpu = old.primitive_kernel(values, values, kind, .3, device)
                np.testing.assert_allclose(cpu, gpu, atol=1e-11, rtol=1e-11)
                checks.append({"device": device, "kind": kind, "dimension": size,
                    "max_absolute_error": float(abs(cpu - gpu).max()), "device_bytes": int(properties["totalGlobalMem"])})
        probes = [json.loads(path.read_text()) for path in (root / "preflight").glob("probe_*.json")]
        preparations = [json.loads(path.read_text()) for path in (root / "preflight").glob("prepare_*.json")]
        state, _ = resources()
        peak = max(row["peak_gib"] for row in probes + preparations)
        budget = max(2., peak * 1.5 + .5)
        workers = min(protocol["max_workers"], max(1, int(.8 * state["logical_cpu_limit"]) - 1))
        ceiling = .8 * state["host_total_gib"]
        if state["cgroup_limit_gib"]:
            ceiling = min(ceiling, .8 * state["cgroup_limit_gib"])
        workers = min(workers, max(1, int((ceiling - protocol["memory_safety_gib"]) / budget)))
        exact = min(protocol["max_exact_workers"], workers, max(1, int((ceiling - protocol["memory_safety_gib"] - workers * budget / 2) / budget)))
        meta["protocol"]["exact_workers"] = exact
        meta["scheduling"] = {"workers": workers, "exact_workers": exact, "worker_budget_gib": budget,
            "measured_max_peak_gib": peak, "memory_safety_multiplier": 1.5, "memory_safety_add_gib": .5}
        receipt = {"status": "passed", "finished_utc": p.now(), "source_sha256": hashes,
            "prepared_pairs": len(preparations), "nested_pca_fits": sum(row["nested_pca_fits"] for row in preparations),
            "probe_count": len(probes), "gpu_gram_checks": checks, "resources": state, "scheduling": meta["scheduling"],
            "test_labels_used_for_feasibility_or_selection": False,
            "logs_sha256": {str(path.relative_to(root)): p.sha(path) for path in (root / "preflight").glob("*")}}
        p.write_json(root / "preflight_receipt.json", receipt)
        meta.update(status="ready", phase="preflight_passed", preflight_sha256=p.sha(root / "preflight_receipt.json"))
    except BaseException as error:
        meta.update(status="preflight_failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        p.write_json(root / "preflight_failure.json", {"status": "failed", "time_utc": p.now(),
            "error": meta["error"], "source_sha256": hashes, "no_automatic_retry": True})
        raise
    finally:
        p.write_json(root / "metadata.json", meta)
    print(json.dumps({"status": meta["status"], "scheduling": meta.get("scheduling")}), flush=True)


def execute(args):
    root = p.ROOT / "code/results" / args.run_id
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["status"] == "ready" and not (root / "monitor.json").exists(), "No automatic run retry"
    assert all(p.sha(p.ROOT / name) == value for name, value in meta["source_sha256"].items())
    assert p.sha(root / "preflight_receipt.json") == meta["preflight_sha256"]
    protocol = meta["protocol"]
    verify_upstream(protocol)
    tasks = p.jobs(protocol)
    ordered = sorted(enumerate(tasks), key=lambda pair: (pair[1][4] != "normal",
        protocol["datasets"].index(pair[1][0]), pair[1][1], pair[1][2], pair[1][3], pair[1][4]))
    assert not (root / "full/stage_manifest.json").exists()
    p.write_json(root / "full/stage_manifest.json", {"stage": "full", "jobs": tasks, "methods": protocol["methods"],
        "seeds": protocol["seeds"], "source_sha256": meta["source_sha256"], "committed_utc": p.now(),
        "dispatch_order": [list(job) for _, job in ordered], "performance_based_task_removal": False})
    meta.update(status="running", phase="full_search", pid=os.getpid(), scientific_started_utc=p.now())
    p.write_json(root / "metadata.json", meta)
    queue, running = list(ordered), {}
    previous = None
    peaks = {"cpu_percent": 0., "host_used_fraction": 0., "live_workers": 0}
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
                    meta["failed_jobs"].append({"job": list(job), "exit_code": status, "pid": pid, "no_retry": True})
                else:
                    meta["completed_jobs"] += 1
            while queue and not meta["failed_jobs"] and len(running) < meta["scheduling"]["workers"] and admitted(state, meta["scheduling"], protocol):
                index, job = queue.pop(0)
                folder = old.job_folder(root / "full", job)
                folder.mkdir(parents=True, exist_ok=True)
                log = folder / "worker.log"
                assert not log.exists()
                handle = log.open("x")
                process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id,
                    "--operation", "job", "--index", str(index)], cwd=p.ROOT, stdin=subprocess.DEVNULL,
                    stdout=handle, stderr=subprocess.STDOUT)
                running[process.pid] = process, handle, job
                state["host_available_gib"] -= meta["scheduling"]["worker_budget_gib"]
                if state["cgroup_used_gib"] is not None:
                    state["cgroup_used_gib"] += meta["scheduling"]["worker_budget_gib"]
            if meta["failed_jobs"] and not running:
                raise RuntimeError("A worker failed; remaining tasks retained, no automatic retry")
            observed, _ = resources(previous)
            peaks["cpu_percent"] = max(peaks["cpu_percent"], observed["cpu_percent"])
            peaks["host_used_fraction"] = max(peaks["host_used_fraction"], observed["host_used_fraction"])
            peaks["live_workers"] = max(peaks["live_workers"], len(running))
            p.write_json(root / "monitor.json", {"status": "running", "pid": os.getpid(), "heartbeat_utc": p.now(),
                "phase": meta["phase"], "completed_jobs": meta["completed_jobs"], "expected_jobs": len(tasks),
                "failed_jobs": meta["failed_jobs"], "active": [{"pid": pid, "job": list(value[2])} for pid, value in running.items()],
                "queued_jobs": len(queue), "resources": observed, "observed_peaks": peaks, "scheduling": meta["scheduling"],
                "automatic_retries": False})
            p.write_json(root / "metadata.json", meta)
            if queue or running:
                time.sleep(10)
        assert meta["completed_jobs"] == len(tasks) == 870
        meta["phase"] = "global_selection_and_test_release"
        p.write_json(root / "metadata.json", meta)
        from _shared.pca9_reporting import finalize
        finalize(root, meta)
        meta["test_released"] = True
        p.write_json(root / "metadata.json", meta)
        subprocess.run([sys.executable, str(p.ROOT / "code/validate_pca9_followup.py"), "--run-id", args.run_id], check=True)
        meta.update(status="completed", phase="completed_full", finished_utc=p.now())
    except BaseException as error:
        meta.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        p.write_json(root / "metadata.json", meta)
        p.write_json(root / "monitor.json", {"status": meta["status"], "pid": os.getpid(), "heartbeat_utc": p.now(),
            "completed_jobs": meta["completed_jobs"], "expected_jobs": len(tasks), "failed_jobs": meta["failed_jobs"],
            "observed_peaks": peaks, "automatic_retries": False, "error": meta.get("error")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_pca9_20261020")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--operation", choices=("prepare", "probe", "job"))
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    assert args.run_id == Path(args.run_id).name and args.run_id.startswith("cmc_pca9_")
    if args.operation:
        operation(args, args.operation, args.index)
    elif args.preflight:
        preflight(args)
    elif args.launch:
        root = p.ROOT / "code/results" / args.run_id
        assert json.loads((root / "metadata.json").read_text())["status"] == "ready"
        assert not (root / "monitor.json").exists()
        with (root / "server.log").open("x") as handle:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id],
                cwd=p.ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"run_id": args.run_id, "pid": process.pid}))
    else:
        execute(args)


if __name__ == "__main__":
    main()
