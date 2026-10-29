"""Run all second-feedback controls without changing completed campaigns."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from _shared import professor_followup as p


def root_for(args):
    assert args.run_id == p.protocol()["run_id"]
    return p.ROOT / "code/results" / args.run_id


def calibrations(root):
    meta = p.verify_sources(root)
    assert meta["status"] == "initialized"
    meta.update(status="calibrating", phase="training_oof_only_calibration")
    p.write(root / "metadata.json", meta)
    try:
        for dataset in meta["protocol"]["datasets"]:
            for seed in meta["protocol"]["seeds"]:
                p.calibration_pair(root, dataset, seed)
        receipt = p.calibration_evaluate(root)
        receipt.update(p.existing_analysis(root))
        p.verify_sources(root)
        p.write(root / "calibration_receipt.json", {"status": "passed", "utc": p.now(), **receipt,
            "scope": "All40 calibration pairs completed; random/matched controls not yet executed"}, True)
        meta.update(status="calibration_completed", phase="awaiting_scientific_preflight", calibration_pairs=40)
    except BaseException as error:
        meta.update(status="calibration_failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        p.write(root / "calibration_failure.json", {"utc": p.now(), "error": meta["error"], "no_retry": True}, True)
        raise
    finally:
        p.write(root / "metadata.json", meta)


def operation(root, args):
    p.verify_sources(root, parent_inputs=False)
    if args.operation == "probe":
        pairs = [(d, 10, k) for d in p.protocol()["random_datasets"] for k in p.protocol()["learners"]]
        d, s, k = pairs[args.index]
        result = p.scientific_job(root, ("random", d, s, k, 1), probe=True)
        p.write(root / "preflight" / f"probe_{args.index:03d}.json", result, True)
        return
    job = p.jobs()[args.index]
    try:
        p.scientific_job(root, job)
    except BaseException as error:
        p.write(p.job_folder(root, job) / "failure.json", {"utc": p.now(), "job": job,
            "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(), "no_retry": True}, True)
        raise


def preflight(root, args):
    from run_repeated_holdout import scheduling_for
    from run_pca9_followup import resources
    from _shared import overnight_risk as old
    import numpy as np
    meta = p.verify_sources(root)
    assert meta["status"] == "calibration_completed"
    meta.update(status="preparing", phase="full_size_random_control_preflight", pid=os.getpid())
    p.write(root / "metadata.json", meta)
    try:
        for index in range(4):
            file = root / "preflight" / f"probe_{index:03d}.log"
            file.parent.mkdir(exist_ok=True)
            with file.open("x") as handle:
                subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id,
                    "--operation", "probe", "--index", str(index)], cwd=p.ROOT, stdin=subprocess.DEVNULL,
                    stdout=handle, stderr=subprocess.STDOUT, check=True)
        probes = [json.loads((root / "preflight" / f"probe_{i:03d}.json").read_text()) for i in range(4)]
        state, _ = resources()
        scheduling = scheduling_for(state, max(v["search_peak_gib"] for v in probes),
            max(v["peak_gib"] for v in probes), meta["protocol"]["resources"])
        rng, checks = np.random.default_rng(20261022), []
        for device in meta["protocol"]["resources"]["gpu_devices"]:
            for kind, dimension in (("rbf", 8), ("fidelity", 128), ("fidelity", 256)):
                value = rng.normal(size=(1001, dimension))
                if kind == "fidelity":
                    value = value + 1j * rng.normal(size=value.shape)
                    value /= np.linalg.norm(value, axis=1, keepdims=True)
                cpu = old.primitive_kernel(value, value, kind, .3)
                gpu = old.primitive_kernel(value, value, kind, .3, device)
                np.testing.assert_allclose(cpu, gpu, atol=1e-11, rtol=1e-11)
                checks.append({"device": device, "kind": kind, "dimension": dimension,
                    "max_error": float(abs(cpu - gpu).max())})
        p.write(root / "preflight_receipt.json", {"status": "passed", "utc": p.now(), "full_size_probes": 4,
            "probe_hashes": {str((root / "preflight" / f"probe_{i:03d}.json").relative_to(root)):
                p.sha(root / "preflight" / f"probe_{i:03d}.json") for i in range(4)},
            "test_labels_or_test_inputs_used_in_probes": False, "cpu_gpu_checks": checks, "scheduling": scheduling}, True)
        meta.update(status="ready", phase="scientific_preflight_passed", scheduling=scheduling)
    except BaseException as error:
        meta.update(status="preflight_failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        p.write(root / "preflight_failure.json", {"utc": p.now(), "error": meta["error"], "no_retry": True}, True)
        raise
    finally:
        p.write(root / "metadata.json", meta)


def execute(root, args):
    from run_repeated_holdout import admitted
    from run_pca9_followup import resources
    meta = p.verify_sources(root)
    assert meta["status"] == "ready"
    tasks = p.jobs()
    p.write(root / "stage_manifest.json", {"utc": p.now(), "jobs": tasks,
        "source_sha256": meta["source_sha256"], "all_jobs_required": True}, True)
    meta.update(status="running", phase="random_and_matched_full_training", pid=os.getpid(), scientific_started_utc=p.now())
    queue, running, previous = list(enumerate(tasks)), {}, None
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
            while queue and not meta["failed_jobs"] and len(running) < meta["scheduling"]["workers"] and admitted(
                    state, meta["scheduling"], meta["protocol"]["resources"]):
                index, job = queue.pop(0)
                folder = p.job_folder(root, job)
                folder.mkdir(parents=True, exist_ok=True)
                handle = (folder / "worker.log").open("x")
                process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id,
                    "--operation", "job", "--index", str(index)], cwd=p.ROOT, stdin=subprocess.DEVNULL,
                    stdout=handle, stderr=subprocess.STDOUT)
                running[process.pid] = process, handle, job
                state["host_available_gib"] -= meta["scheduling"]["worker_budget_gib"]
                if state["cgroup_used_gib"] is not None:
                    state["cgroup_used_gib"] += meta["scheduling"]["worker_budget_gib"]
            p.write(root / "monitor.json", {"status": "running", "heartbeat_utc": p.now(), "pid": os.getpid(),
                "completed_jobs": meta["completed_jobs"], "expected_jobs": 200, "failed_jobs": meta["failed_jobs"],
                "queued_jobs": len(queue), "active": [{"pid": pid, "job": list(v[2])} for pid, v in running.items()],
                "resources": state, "scheduling": meta["scheduling"], "automatic_retries": False})
            p.write(root / "metadata.json", meta)
            if meta["failed_jobs"] and not running:
                raise RuntimeError("Scientific worker failure; preserved without retry")
            if queue or running:
                time.sleep(10)
        assert meta["completed_jobs"] == 200
        meta["phase"] = "global_lock_evaluation_and_audit"
        p.write(root / "metadata.json", meta)
        p.verify_sources(root)
        p.evaluate_science(root)
        p.complete_statistics(root)
        p.mechanism_analysis(root)
        from report_professor_followup import report
        report(root)
        validate(root)
        meta.update(status="completed", phase="completed_all_requested_controls", test_released=True, finished_utc=p.now())
    except BaseException as error:
        meta.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        p.write(root / "metadata.json", meta)
        p.write(root / "monitor.json", {"status": meta["status"], "heartbeat_utc": p.now(), "pid": os.getpid(),
            "completed_jobs": meta["completed_jobs"], "expected_jobs": 200, "failed_jobs": meta["failed_jobs"], "error": meta.get("error")})


def validate(root):
    import numpy as np
    import pandas as pd
    meta = p.verify_sources(root)
    assert len(p.jobs()) == 200
    test = pd.read_csv(root / "new_seed_metrics.csv")
    assert len(test) == 440
    stats = pd.read_csv(root / "paired_statistics.csv")
    differences = pd.read_csv(root / "seed_differences.csv")
    assert len(stats) == 140 and len(differences) == 1400
    for row in stats.itertuples():
        values = differences[differences.dataset.eq(row.dataset) & differences.kernel.eq(row.kernel)
            & differences.method.eq(row.method) & differences.contrast.eq(row.contrast)].sort_values("seed")
        assert values.seed.tolist() == meta["protocol"]["seeds"]
        result = p.corrected(values.delta.to_numpy())
        for name in ("mean_difference", "sd_difference", "p_value", "ci_low", "ci_high"):
            assert np.isclose(getattr(row, name), result[name], equal_nan=True)
    for family, count in (("primary_cost", 100), ("secondary_calibration", 40)):
        group = stats[stats.testing_family.eq(family)]
        assert len(group) == count and np.allclose(group.p_holm, p.holm(group.p_value.to_numpy()))
    labels = p.test_labels()
    for job in p.jobs():
        folder = p.job_folder(root, job)
        complete = json.loads((folder / "complete.json").read_text())
        assert not complete["test_labels_read"] and complete["job"] == list(job)
        assert p.sha(folder / "margins.npz") == complete["margins_sha256"]
        assert p.sha(folder / "geometry.npz") == complete["geometry_sha256"]
        mode, dataset, seed, kernel, setting = job
        _, _, split, _ = p.inputs(dataset, seed)
        y = labels[labels.dataset.eq(dataset) & labels.seed.eq(seed)].set_index("row_id").reindex(split["test"]).y.to_numpy(dtype=int)
        with np.load(folder / "margins.npz", allow_pickle=False) as saved:
            scores = dict(saved)
        for role, choice in complete["choices"].items():
            assert p.sha(folder / (role + "_model.joblib")) == complete["models_sha256"][role]
            model = p.joblib.load(folder / (role + "_model.joblib"))
            assert model.C == choice["C"] and model.fit_status_ == 0
            assert model.shape_fit_[0] == choice["fit_rows"]
            assert np.min(model.support_) >= 0 and np.max(model.support_) < choice["fit_rows"]
            selected = test[test["mode"].eq(mode) & test.dataset.eq(dataset) & test.seed.eq(seed)
                & test.kernel.eq(kernel) & test.setting.astype(str).eq(str(setting)) & test.role.eq(role)]
            assert len(selected) == 1
            recalculated = p.metrics(y, scores[role + "_gate"], choice["threshold"])
            for name in ("cost", "fn", "fp", "tn", "tp", "f1", "recall", "balanced_accuracy", "auroc", "ap"):
                assert np.isclose(selected[name].iloc[0], recalculated[name]), (job, role, name)
    files = [v for v in root.glob("*.csv*")] + [root / "selection_lock.json", root / "test_release.json"]
    receipt = {"status": "passed", "utc": p.now(), "scientific_jobs": 200, "calibration_pairs": 40,
        "new_model_rows": 440, "paired_contrasts": 140, "source_and_parent_inputs_unchanged": True,
        "output_sha256": {str(v.relative_to(root)): p.sha(v) for v in files},
        "scope": "Integrity, saved SVC parameters/row counts, locked score evaluation and full statistics; not independent retraining or external confirmation"}
    p.write(root / "validation_receipt.json", receipt)
    return receipt


def supervise(root, pid):
    from run_pca9_followup import active, resources
    previous, peak_cpu, peak_memory = None, 0., 0.
    while True:
        meta = json.loads((root / "metadata.json").read_text())
        monitor = json.loads((root / "monitor.json").read_text()) if (root / "monitor.json").exists() else {}
        state, previous = resources(previous)
        peak_cpu, peak_memory = max(peak_cpu, state["cpu_percent"]), max(peak_memory, state["host_used_fraction"])
        mismatches = [name for name, digest in meta["source_sha256"].items() if p.sha(p.ROOT / name) != digest]
        alive = active(pid)
        receipt = {"utc": p.now(), "pid": os.getpid(), "experiment_pid": pid, "experiment_alive": alive,
            "status": "attention_required" if mismatches else meta["status"], "source_mismatches": mismatches,
            "completed_jobs": meta["completed_jobs"], "resources": state, "peak_cpu_percent": peak_cpu,
            "peak_host_memory_fraction": peak_memory, "active_workers": monitor.get("active", []), "automatic_retries": False}
        p.write(root / "operations/supervision.json", receipt)
        if not alive:
            p.write(root / "operations/supervision_outcome.json", receipt)
            return
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_professor_followup_20261022")
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--launch-science", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--supervise", type=int)
    parser.add_argument("--operation", choices=("probe", "job"))
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    root = root_for(args)
    root.mkdir(parents=True, exist_ok=True)
    if args.initialize:
        with (root / "technical_tests.log").open("x") as handle:
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "code/tests", "-p", "test_professor_followup.py", "-v"],
                cwd=p.ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        p.freeze(root)
        p.write(root / "technical_tests_receipt.json", {"status": "passed", "tests": 10,
            "utc": p.now(), "log_sha256": p.sha(root / "technical_tests.log")}, True)
    elif args.calibrate:
        calibrations(root)
    elif args.operation:
        operation(root, args)
    elif args.preflight:
        preflight(root, args)
    elif args.launch_science:
        meta = p.verify_sources(root)
        assert meta["status"] == "calibration_completed"
        with (root / "server.log").open("x") as handle:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id, "--pipeline"],
                cwd=p.ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        with (root / "supervisor.log").open("x") as handle:
            supervisor = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id, "--supervise", str(process.pid)],
                cwd=p.ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        p.write(root / "launch_receipt.json", {"utc": p.now(), "pid": process.pid, "supervisor_pid": supervisor.pid}, True)
        print(json.dumps({"pid": process.pid, "supervisor_pid": supervisor.pid, "status": "preflight_started"}))
    elif args.pipeline:
        preflight(root, args)
        execute(root, args)
    elif args.validate:
        print(json.dumps(validate(root)))
    elif args.supervise:
        supervise(root, args.supervise)
    else:
        parser.error("Select an operation")


if __name__ == "__main__":
    main()
