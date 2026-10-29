"""Independent PCA9 repeated-holdout launch after archived PCA7 delivery."""
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

from _shared import repeated_holdout_pca9 as p
from _shared import overnight_risk as old
from run_repeated_holdout import SOURCES as PCA7_SOURCES, admitted, scheduling_for
from run_pca9_followup import active, resources

NEW_SOURCES = ("code/_shared/repeated_holdout_pca9.py", "code/_shared/repeated_reporting_pca9.py",
    "code/_shared/repeated_dimension_comparison.py", "code/run_repeated_holdout_pca9.py",
    "code/validate_repeated_holdout_pca9.py", "code/monitor_repeated_holdout_pca9.py",
    "code/tests/test_repeated_holdout_pca9.py", "docs/cmc_pca9_repeated_protocol_20261021.json",
    "code/REPEATED_HOLDOUT_PCA9_EXPERIMENTS.md")
SOURCES = tuple(dict.fromkeys((*PCA7_SOURCES, *NEW_SOURCES)))


def upstream_gate():
    root = p.UPSTREAM
    meta = json.loads((root / "metadata.json").read_text())
    audit = json.loads((root / "validation_receipt.json").read_text())
    delivery = json.loads((root / "operations/final_report_20261021/final_report_receipt.json").read_text())
    assert meta["status"] == "completed" and meta["completed_jobs"] == 1160 and not meta["failed_jobs"]
    assert audit["status"] == "passed" and audit["jobs"] == 1160
    assert delivery["status"] == "delivered" and delivery["scope"]["new_failures"] == 0
    assert p.sha(root / "validation_receipt.json") == delivery["validation_receipt_sha256"]
    assert p.sha(p.ROOT / delivery["report_path"]) == delivery["report_sha256"]
    assert delivery["live_campaign_processes"] == delivery["held_exact_slots"] == 0
    assert not delivery["test_performance_used_to_choose_pca9_scope"]
    for file in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = [a.decode(errors="replace") for a in file.read_bytes().split(b"\0") if a]
            names = {Path(a).name for a in args}
            if meta["run_id"] in args and names & {"run_repeated_holdout.py", "monitor_repeated_holdout.py"}:
                state = (file.parent / "stat").read_text().rsplit(") ", 1)[1].split()[0]
                assert state in ("Z", "X"), "PCA7 scientific work or supervisor still active"
        except (FileNotFoundError, ProcessLookupError):
            continue
    held_tokens = set(Path("/proc/locks").read_text().split())
    for file in (root / "exact_slots").glob("*.lock"):
        stat = file.stat()
        token = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
        assert token not in held_tokens, "PCA7 exact slot still held"
    old_stop = p.ROOT / "code/results/cmc_pca9_20261020/operations/authorized_stop_20261020/stop_receipt.json"
    stopped = json.loads(old_stop.read_text())
    assert stopped["status"] == "authorized_stopped" and stopped["remaining_active_workers"] == 0
    expected = {**meta["source_sha256"], **meta["preserved_inputs_sha256"]}
    assert all(p.sha(p.ROOT / name) == value for name, value in expected.items())
    files = [root / name for name in ("metadata.json", "validation_receipt.json", "preflight_receipt.json",
        "source_snapshot.tar.gz", "full/seed_metrics.csv", "full/selection_lock.json", "full/test_release.json",
        "full/verification_receipt.json", "operations/final_report_20261021/final_report_receipt.json")]
    files += [p.ROOT / delivery["report_path"], old_stop]
    return {str(file.relative_to(p.ROOT)): p.sha(file) for file in files}


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
        items = [("hmeq", k, m) for k in protocol["learners"] for m in protocol["methods"]]
        dataset, learner, method = items[args.index]
        arrays, record = old.load_inputs(root / "full", dataset, 10)
        arrays = {key: value.copy() for key, value in arrays.items()}
        for key in list(arrays):
            if key == "gate" or key.endswith("_gate"):
                arrays[key] = arrays["tune" if key == "gate" else key[:-4] + "tune"].copy()
        ctx = p.engine.context(arrays, record, method, learner, "normal", 10, protocol)
        choice = {"lambda": .5, "C": 1., "geometry": 0, "score_index": 0}
        started = time.perf_counter()
        rep = p.engine.make_representation(arrays, record, ctx, method, learner, "normal", .5, protocol["geometries"][0])
        landmarks = old.subset(np.arange(len(arrays["train"])), arrays["y_train"], 512, 711)
        matrices, _, _, _ = p.engine.matrices(rep, arrays, True, landmarks, None)
        for c in protocol["C_grid"]:
            p.engine.fit_candidate(arrays, matrices, {**choice, "C": c}, True, 10)
        search_peak = p.peak_gib()
        del matrices
        p.release_heap()
        probe_protocol = {**protocol, "exact_workers": 1, "exact_budget_gib": 32.}
        row, margins, _ = p.exact_fit(arrays, record, ctx, (dataset, 10, learner, method, "normal"), choice, probe_protocol, root / "full", None)
        assert row["fit_rows"] == len(arrays["train"]) and len(margins["gate"]) == len(arrays["tune"])
        p.write_json(root / "preflight" / f"probe_{args.index:03d}.json", {"status": "passed", "dataset": dataset,
            "learner": learner, "method": method, "pca_components": 9, "search_peak_gib": search_peak,
            "full_peak_gib": p.peak_gib(), "elapsed_seconds": time.perf_counter() - started,
            "full_train_rows": row["fit_rows"], "test_labels_read": False, "test_features_or_scores_used": False,
            "performance_used_for_selection": False, "pid": os.getpid()})
        return
    job = tuple(p.jobs(protocol)[args.index])
    folder = old.job_folder(root / "full", job)
    receipt = {"job": list(job), "pid": os.getpid(), "started_utc": p.now(),
        "fresh_process_for_one_job": True, "source_sha256": meta["source_sha256"]}
    p.write_json(folder / "worker_operation.json", receipt)
    try:
        p.run_job(job, protocol, root / "full", meta["source_sha256"])
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt.update(finished_utc=p.now(), peak_gib=p.peak_gib())
        p.write_json(folder / "worker_operation.json", receipt)


def child(args, mode, index):
    file = p.ROOT / "code/results" / args.run_id / "preflight" / f"{mode}_{index:03d}.log"
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("x") as handle:
        return subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id,
            "--operation", mode, "--index", str(index)], cwd=p.ROOT, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT).returncode


def preflight(args):
    root = p.ROOT / "code/results" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "metadata.json").exists(), "No preflight retry"
    protocol = json.loads(p.PROTOCOL.read_text())
    assert args.run_id == protocol["run_id"] and protocol["pca_components"] == 9
    preserved = upstream_gate()
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
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "code/tests", "-p", "test_repeated_holdout_pca9.py", "-v"], cwd=p.ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = {pool.submit(child, args, "prepare", i): i for i in range(30)}
            for future in as_completed(futures):
                assert future.result() == 0, f"Real preparation failed:{futures[future]}; preserve, no retry"
        for i in range(8):
            assert child(args, "probe", i) == 0, f"Real full-size probe failed:{i}; preserve, no retry"
        import numpy as np
        rng, checks = np.random.default_rng(20261021), []
        for device in protocol["gpu_devices"]:
            for kind, dimension in (("rbf", 9), ("fidelity", 512), ("fidelity", 1024)):
                values = rng.normal(size=(1001, dimension))
                if kind == "fidelity":
                    values = values + 1j * rng.normal(size=values.shape)
                    values /= np.linalg.norm(values, axis=1, keepdims=True)
                cpu = old.primitive_kernel(values, values, kind, .3)
                gpu = old.primitive_kernel(values, values, kind, .3, device)
                np.testing.assert_allclose(cpu, gpu, atol=1e-11, rtol=1e-11)
                checks.append({"device": device, "kind": kind, "dimension": dimension, "max_absolute_error": float(abs(cpu - gpu).max())})
        probes = [json.loads(file.read_text()) for file in (root / "preflight").glob("probe_*.json")]
        preparations = [json.loads(file.read_text()) for file in (root / "preflight").glob("prepare_*.json")]
        assert len(preparations) == 30 and len(probes) == 8
        state, _ = resources()
        scheduling = scheduling_for(state, max(row["search_peak_gib"] for row in probes), max(row["full_peak_gib"] for row in probes), protocol)
        protocol.update(exact_workers=scheduling["exact_workers"], exact_budget_gib=scheduling["exact_budget_gib"])
        meta["scheduling"] = scheduling
        from validate_repeated_holdout_pca9 import audit_prepared
        receipt = {"status": "passed", "finished_utc": p.now(), "prepared_pairs": 30,
            "nested_pca_fits": sum(row["nested_pca_fits"] for row in preparations), "probe_count": 8,
            "gpu_gram_checks": checks, "source_sha256": hashes, "scheduling": scheduling,
            "test_labels_used_for_feasibility": False, "split_and_oof_audit": audit_prepared(root, protocol),
            "outputs_sha256": {str(file.relative_to(root)): p.sha(file) for file in (root / "preflight").glob("*")}}
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
    assert upstream_gate() == meta["preserved_inputs_sha256"]
    protocol, tasks = meta["protocol"], p.jobs(meta["protocol"])
    assert len(tasks) == 870
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
                "completed_jobs": meta["completed_jobs"], "expected_jobs": 870, "failed_jobs": meta["failed_jobs"],
                "active": [{"pid": pid, "job": list(value[2])} for pid, value in running.items()], "queued_jobs": len(queue),
                "resources": state, "scheduling": meta["scheduling"], "automatic_retries": False})
            p.write_json(root / "metadata.json", meta)
            if meta["failed_jobs"] and not running:
                raise RuntimeError("Worker failure; preserve, no automatic retry")
            if queue or running:
                time.sleep(10)
        if meta.get("stop_requested"):
            meta.update(status="authorized_stopped", phase="preserved_partial", finished_utc=p.now())
            return
        assert meta["completed_jobs"] == 870
        meta["phase"] = "global_selection_and_test_release"
        p.write_json(root / "metadata.json", meta)
        from _shared.repeated_reporting_pca9 import finalize
        finalize(root, meta)
        meta["test_released"] = True
        p.write_json(root / "metadata.json", meta)
        subprocess.run([sys.executable, "code/validate_repeated_holdout_pca9.py", "--run-id", args.run_id], cwd=p.ROOT, check=True)
        meta.update(status="completed", phase="completed_full", finished_utc=p.now())
    except BaseException as error:
        meta.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        p.write_json(root / "metadata.json", meta)
        p.write_json(root / "monitor.json", {"status": meta["status"], "pid": os.getpid(), "heartbeat_utc": p.now(),
            "completed_jobs": meta["completed_jobs"], "expected_jobs": 870, "failed_jobs": meta["failed_jobs"], "error": meta.get("error")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_pca9_repeated_20261021")
    parser.add_argument("--launch-pipeline", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--operation", choices=("prepare", "probe", "job"))
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    assert args.run_id == Path(args.run_id).name and args.run_id == json.loads(p.PROTOCOL.read_text())["run_id"]
    if args.operation:
        operation(args)
    elif args.launch_pipeline:
        root = p.ROOT / "code/results" / args.run_id
        root.mkdir(parents=True, exist_ok=True)
        assert not (root / "metadata.json").exists() and not (root / "server.log").exists()
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
