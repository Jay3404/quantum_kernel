"""Run or detach the parallel, calibrated full-data revision campaign."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from _shared import controlled as base
from _shared.full_data import write_json
from _shared.improved import LEARNERS, local_lock as run_lock, run_job, sha

ROOT = base.ROOT
DEFAULT_PROTOCOL = ROOT / "docs/cmc_improved_protocol_20261016.json"


def execute(args):
    protocol = json.loads(args.protocol.read_text())
    datasets, seeds, learners = args.datasets or protocol["datasets"], args.seeds or protocol["seeds"], args.learners or list(LEARNERS)
    output = ROOT / "code/results" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    sources = {"improved": ROOT / "code/_shared/improved.py", "runner": Path(__file__),
               "controlled": ROOT / "code/_shared/controlled.py", "full_data": ROOT / "code/_shared/full_data.py",
               "analysis": ROOT / "code/build_improved_results.py", "shared_analysis": ROOT / "code/build_controlled_results.py", "protocol": args.protocol}
    options = {"run_id": args.run_id, "data_root": str(ROOT / "data"), "protocol_settings": protocol, "protocol_path": str(args.protocol.resolve()),
               "kernel_block": 1024, "prediction_block": 256, "memory_limit_gib": 8, "reserve_gib": 16,
               "svm_cache_mb": 256, "blas_threads": args.blas_threads,
               "source_sha256": {name: sha(path) for name, path in sources.items()}}
    with run_lock(output / "run.lock"):
        path = output / "metadata.json"
        previous = json.loads(path.read_text()) if path.exists() else None
        if previous and (previous["options"] != options or previous["datasets"] != datasets or previous["seeds"] != seeds or previous["learners"] != learners):
            raise AssertionError("Campaign settings changed; use another run id")
        metadata = {"run_id": args.run_id, "status": "running", "pid": os.getpid(), "datasets": datasets, "seeds": seeds,
                    "learners": learners, "options": options, "workers": args.workers, "primary_policy": "cost_probability",
                    "full_dataset": True, "expected_model_jobs": len(datasets) * len(seeds) * len(learners),
                    "expected_seed_jobs": len(datasets) * len(seeds), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "host": platform.node(), "python": platform.python_version(),
                    "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "qiskit", "qiskit-aer", "scipy", "xgboost")}}
        write_json(path, metadata)
        completed, failures = {}, []
        try:
            # Large dataset/model jobs enter early so they do not form a serial tail.
            ordered = sorted(datasets, key=lambda d: {"taiwan": 0, "hmeq": 1}.get(d, 2))
            jobs = [(d, s, learner) for d in ordered for s in seeds for learner in learners]
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(run_job, d, s, learner, options): (d, s, learner) for d, s, learner in jobs}
                for count, future in enumerate(as_completed(futures), 1):
                    try:
                        d, s, learner = future.result()
                    except Exception as error:
                        failures.append({"job": futures[future], "error": f"{type(error).__name__}: {error}"})
                        metadata.update(failed_model_jobs=failures, finished_model_jobs=count)
                        write_json(path, metadata)
                        print(f"job_failed={failures[-1]}", flush=True)
                        continue
                    completed.setdefault((d, s), set()).add(learner)
                    metadata.update(completed_model_jobs=count - len(failures), finished_model_jobs=count,
                                    completed_seed_jobs=sum(len(v) == len(learners) for v in completed.values()),
                                    last_completed_dataset=d, last_completed_seed=s, last_completed_learner=learner)
                    write_json(path, metadata)
                    print(f"jobs_completed={count - len(failures)}/{len(jobs)} dataset={d} seed={s} learner={learner}", flush=True)
            if failures:
                raise RuntimeError(f"{len(failures)} model jobs failed; no automatic retry; see metadata.json")
            metadata["status"] = "analyzing"
            write_json(path, metadata)
            if datasets == protocol["datasets"] and seeds == protocol["seeds"] and learners == protocol["learner_jobs"]:
                from build_improved_results import build
                build(args.run_id)
            metadata.update(status="completed", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        except BaseException as error:
            metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write_json(path, metadata)


def supervise(args, command):
    folder = ROOT / "code/results" / args.run_id
    with run_lock(folder / "supervisor.lock"):
        environment = dict(os.environ, OMP_NUM_THREADS=str(args.blas_threads), OPENBLAS_NUM_THREADS=str(args.blas_threads),
                           MKL_NUM_THREADS=str(args.blas_threads), NUMEXPR_NUM_THREADS=str(args.blas_threads))
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, env=environment)
        state = {"status": "running", "supervisor_pid": os.getpid(), "experiment_pid": process.pid, "command": command,
                 "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hard_timeout": None, "automatic_retries": False}
        while process.poll() is None:
            state["heartbeat_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            write_json(folder / "monitor.json", state)
            time.sleep(30)
        state.update(status="completed" if process.returncode == 0 else "failed", returncode=process.returncode,
                     finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write_json(folder / "monitor.json", state)
        metadata_path = folder / "metadata.json"
        if process.returncode and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata["status"] in ("running", "analyzing"):
                metadata.update(status="failed", error=f"Process exited {process.returncode}; see server.log")
                write_json(metadata_path, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_20261016_improved_full")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--datasets", nargs="+", choices=base.DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--learners", nargs="+", choices=LEARNERS)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--supervisor", action="store_true")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.workers < 1 or args.blas_threads < 1:
        parser.error("Use a single run name and positive resources")
    for values in (args.datasets, args.seeds, args.learners):
        if values is not None and len(values) != len(set(values)):
            parser.error("Repeated job dimensions")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id, "--protocol", str(args.protocol.resolve()),
               "--workers", str(args.workers), "--blas-threads", str(args.blas_threads)]
    for flag, values in (("--datasets", args.datasets), ("--seeds", args.seeds), ("--learners", args.learners)):
        if values:
            command.extend([flag, *map(str, values)])
    if args.launch:
        folder = ROOT / "code/results" / args.run_id
        folder.mkdir(parents=True, exist_ok=True)
        monitor = folder / "monitor.json"
        if monitor.exists():
            prior = json.loads(monitor.read_text())
            process_file = Path(f"/proc/{prior['supervisor_pid']}/cmdline")
            if process_file.exists() and "run_improved_experiment.py" in process_file.read_bytes().decode():
                raise RuntimeError("Improved campaign supervisor already running")
        with (folder / "server.log").open("a") as log:
            process = subprocess.Popen([*command, "--supervisor"], cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor_pid": process.pid, "run_id": args.run_id}))
    elif args.supervisor:
        supervise(args, command)
    else:
        execute(args)
