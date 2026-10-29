"""Authorized memory-only continuation with immutable parent provenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import run_overnight_risk as legacy
import run_overnight_risk_v3 as previous
from _shared import overnight_risk as n
from _shared import overnight_runtime_v4 as runtime
from _shared.overnight_reporting import collect, save_stage

PROTOCOL = n.ROOT / "docs/cmc_overnight_risk_v4_protocol_20261019.json"
SOURCES = tuple(dict.fromkeys((*legacy.SOURCES, *previous.ADDITIONAL_SOURCES,
    "code/run_overnight_risk_v4.py", "code/_shared/overnight_runtime_v4.py",
    "code/validate_overnight_risk_v4.py", "code/tests/test_overnight_runtime_v4.py",
    "docs/cmc_overnight_risk_v4_protocol_20261019.json")))


def job_order(job):
    dataset, seed, learner, method, arm = job
    return (arm != "normal", {"hmeq": 0, "taiwan": 1}.get(dataset, 2),
        method not in ("reference", "baseline"), seed, learner, method, arm)


def copy_artifacts(source, target, frozen):
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if not path.is_file() or path.suffix in (".lock", ".tmp") or ".tmp." in path.name:
            continue
        shutil.copy2(path, target / path.name)
        frozen[str(path.relative_to(n.ROOT))] = n.sha(path)


def adopt_complete(source, target, hashes, frozen):
    copy_artifacts(source, target, frozen)
    old = json.loads((source / "complete.json").read_text())
    shutil.copy2(source / "complete.json", target / "parent_complete.json")
    n.write_json(target / "complete.json", {**old, "source_sha256": hashes,
        "record_type": "adopted_execution_receipt_no_new_fit",
        "parent_complete_path": str((source / "complete.json").relative_to(n.ROOT)),
        "parent_complete_sha256": n.sha(source / "complete.json"),
        "model_execution_source_sha256": old["source_sha256"], "new_model_fit_calls": 0})


def initialize(args):
    protocol = json.loads(PROTOCOL.read_text())
    root = n.ROOT / "code/results" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "metadata.json").exists(), "Separate run, no automatic retries"
    parent = n.ROOT / "code/results" / protocol["continuation"]["parent_run"]
    old = json.loads((parent / "metadata.json").read_text())
    assert not (parent / "full/test_release.json").exists()
    assert not (parent / "full/selection_lock.json").exists()
    assert json.loads((parent / "validation_receipt.json").read_text())["status"] == "passed"
    stop = json.loads((parent / "authorized_stop_receipt.json").read_text())
    assert stop["status"] == "authorized_stop_memory_management"
    assert json.loads((parent / "monitor.json").read_text())["status"] != "running"
    assert not Path(f"/proc/{old['pid']}").exists()
    guard = json.loads((parent / "resource_guard.json").read_text())
    assert guard["status"] == "active" and guard["maximum_concurrent_large_exact_fits"] == 3
    assert time.time() - (parent / "resource_guard.json").stat().st_mtime < 60
    legacy.assert_preserved(old)
    hashes = {name: n.sha(n.ROOT / name) for name in SOURCES}
    frozen = dict(old["preserved_inputs_sha256"])
    for filename in ("metadata.json", "metadata_before_authorized_stop.json", "authorized_stop_receipt.json",
                     "source_snapshot.tar.gz", "validation_receipt.json", "preflight.json",
                     "discovery/shortlist.json", "confirmation/promotion.json", "full/stage_manifest.json"):
        frozen[str((parent / filename).relative_to(n.ROOT))] = n.sha(parent / filename)
    methods = json.loads((parent / "confirmation/promotion.json").read_text())["families"]
    assert methods == old["full_families"]
    tasks = legacy.jobs(protocol, methods, protocol["seeds"], True)
    assert tasks == [tuple(job) for job in json.loads((parent / "full/stage_manifest.json").read_text())["jobs"]]
    output = root / "full"
    output.mkdir(parents=True, exist_ok=True)
    for dataset in protocol["datasets"]:
        for seed in protocol["seeds"]:
            copy_artifacts(n.prepared_folder(parent / "full", dataset, seed),
                n.prepared_folder(output, dataset, seed), frozen)
    groups, _, _ = collect(parent / "full", tasks, old["source_sha256"])
    inherited, resumed, ineligible = [], [], []
    for job in tasks:
        source, target = n.job_folder(parent / "full", job), n.job_folder(output, job)
        if job in groups:
            adopt_complete(source, target, hashes, frozen)
            inherited.append(list(job))
        elif (source / "progress.json").exists() and job[3] != "reference":
            arrays, _ = n.load_inputs(output, *job[:2])
            try:
                rows, _ = runtime.load_checkpoint(source, arrays, protocol, job[3])
            except Exception as error:
                ineligible.append({"job": list(job), "reason": repr(error), "action": "Authorized new search; original checkpoint retained"})
            else:
                checkpoint = target / "parent_checkpoint"
                checkpoint.mkdir(parents=True, exist_ok=True)
                for filename in ("progress.json", "margins.npz", "parameter_lock.json"):
                    if (source / filename).exists():
                        shutil.copy2(source / filename, checkpoint / filename)
                        frozen[str((source / filename).relative_to(n.ROOT))] = n.sha(source / filename)
                resumed.append({"job": list(job), "inherited_trials": len(rows)})
            del arrays
            runtime.release_heap()
    n.write_json(output / "stage_manifest.json", {"stage": "full", "methods": methods,
        "seeds": protocol["seeds"], "jobs": [list(job) for job in tasks],
        "committed_utc": legacy.now(), "source_sha256": hashes,
        "dispatch_order": [list(job) for job in sorted((j for j in tasks if j not in groups), key=job_order)]})
    metadata = {"run_id": args.run_id, "status": "running", "phase": "full_search", "pid": os.getpid(),
        "started_utc": legacy.now(), "protocol": protocol, "source_sha256": hashes,
        "preserved_inputs_sha256": frozen, "failed_jobs": [], "test_released": False,
        "expected_jobs": len(tasks), "completed_jobs": len(inherited), "full_families": methods,
        "parent_run": old["run_id"], "parent_failed_attempts": old["failed_jobs"],
        "inherited_completed_jobs": inherited, "resumed_checkpoints": resumed,
        "ineligible_parent_checkpoints": ineligible, "packages": old["packages"],
        "discovery_summary": old["discovery_summary"], "confirmation_summary": old["confirmation_summary"],
        "stage_provenance": "Discovery/confirmation inherited from independently audited v3; no new family selection",
        "preflight_provenance": "Unchanged CPU/Aer and GPU Gram algebra; parent GPU preflight retained by hash"}
    metadata["resource_guard_snapshot"] = guard
    n.write_json(root / "metadata.json", metadata)
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
        for name in SOURCES:
            archive.add(n.ROOT / name, arcname=name)
    return root, metadata, tasks


def worker(args):
    root = n.ROOT / "code/results" / args.run_id
    metadata = json.loads((root / "metadata.json").read_text())
    tasks = json.loads((root / "full/stage_manifest.json").read_text())["jobs"]
    job = tuple(tasks[args.worker])
    folder = n.job_folder(root / "full", job)
    folder.mkdir(parents=True, exist_ok=True)
    record = {"pid": os.getpid(), "started_utc": legacy.now(), "job": list(job),
        "fresh_process_for_one_job": True, "source_sha256": metadata["source_sha256"]}
    n.write_json(folder / "worker_operation.json", record)
    try:
        runtime.run_job(job, "full", metadata["protocol"], root / "full", metadata["source_sha256"])
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        record["status"] = "completed"
    finally:
        record["finished_utc"] = legacy.now()
        n.write_json(folder / "worker_operation.json", record)


def dispatch(index, job, args, root, protocol):
    while n.available_memory() and n.available_memory() < protocol["continuation"]["launch_reserve_gib"] * 2 ** 30:
        time.sleep(2)
    folder = n.job_folder(root / "full", job)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "worker.log").open("w") as log:
        return subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id,
            "--worker", str(index)], cwd=n.ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT).returncode


def execute(args):
    root = n.ROOT / "code/results" / args.run_id
    with n.local_lock(root / "run.lock"):
        root, metadata, tasks = initialize(args)
        protocol = metadata["protocol"]
        inherited = {tuple(job) for job in metadata["inherited_completed_jobs"]}
        pending = sorted(((i, job) for i, job in enumerate(tasks) if job not in inherited), key=lambda pair: job_order(pair[1]))
        try:
            with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
                futures = {pool.submit(dispatch, index, job, args, root, protocol): job for index, job in pending}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        returncode = future.result()
                        if returncode:
                            raise RuntimeError(f"Worker exited {returncode}; preserved worker.log; no automatic retry")
                    except Exception as error:
                        metadata["failed_jobs"].append({"stage": "full", "job": list(job), "error": repr(error)})
                    else:
                        metadata["completed_jobs"] += 1
                    n.write_json(root / "metadata.json", metadata)
            legacy.assert_preserved(metadata)
            metadata["phase"] = "full_selection"
            n.write_json(root / "metadata.json", metadata)
            save_stage(root / "full", tasks, metadata["source_sha256"], protocol, metadata["full_families"], "full")
            metadata["test_released"] = True
            metadata["full_summary"] = {"expected_jobs": len(tasks), "completed_jobs": metadata["completed_jobs"], "selected_families": metadata["full_families"]}
            n.write_json(root / "metadata.json", metadata)
            subprocess.run([sys.executable, str(n.ROOT / "code/validate_overnight_risk_v4.py"), "--run-id", args.run_id], check=True)
            metadata.update(status="completed_with_failures" if metadata["failed_jobs"] else "completed", phase="completed_full", finished_utc=legacy.now())
        except BaseException as error:
            metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            n.write_json(root / "metadata.json", metadata)
            legacy.root_report(root, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_20261019_overnight_risk_v4")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--supervisor", action="store_true")
    parser.add_argument("--finalize-partial", action="store_true")
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id
    legacy.PROTOCOL = PROTOCOL
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id]
    root = n.ROOT / "code/results" / args.run_id
    if args.worker is not None:
        worker(args)
    elif args.finalize_partial:
        legacy.finalize_partial(args)
        subprocess.run([sys.executable, str(n.ROOT / "code/validate_overnight_risk_v4.py"), "--run-id", args.run_id], check=True)
    elif args.supervisor:
        legacy.supervisor(args, command)
    elif args.launch:
        root.mkdir(parents=True, exist_ok=True)
        assert not (root / "metadata.json").exists() and not (root / "monitor.json").exists()
        with (root / "server.log").open("a") as log:
            process = subprocess.Popen([*command, "--supervisor"], cwd=n.ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print({"run_id": args.run_id, "supervisor_pid": process.pid})
    else:
        root.mkdir(parents=True, exist_ok=True)
        execute(args)


if __name__ == "__main__":
    main()
