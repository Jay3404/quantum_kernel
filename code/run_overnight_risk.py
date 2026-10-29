"""Deadline-bounded nested risk exploration, controlled full tests, no retries."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

from _shared.full_data import write_json
from _shared.improved import local_lock, sha
from _shared.overnight_risk import ROOT, METRICS, prepare, run_job
from _shared.overnight_reporting import save_stage, collect
from run_german_zz_phase import SOURCES as PRIOR_SOURCES, preserved_inputs

PROTOCOL = ROOT / "docs/cmc_overnight_risk_protocol_20261018.json"
SOURCES = (*PRIOR_SOURCES, "code/run_overnight_risk.py", "code/_shared/overnight_risk.py",
    "code/_shared/overnight_reporting.py", "code/validate_overnight_risk.py", "code/tests/test_overnight_risk.py",
    "docs/cmc_overnight_risk_protocol_20261018.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def stamp(value):
    return datetime.fromisoformat(value).timestamp()


def preservation_protocol(protocol):
    return {**protocol, "seeds": list(range(10, 101, 10))}


def jobs(protocol, methods, seeds, full=False):
    result = []
    for dataset, seed in itertools.product(protocol["datasets"], seeds):
        result.append((dataset, seed, "reference", "reference", "normal"))
        for learner in protocol["learners"]:
            result.append((dataset, seed, learner, "baseline", "normal"))
            for method in methods:
                arms = ["normal", "shuffled"] + ([] if method in METRICS else ["constant"]) + (["separable"] if learner == "QSVC" else []) if full else ["normal"]
                result.extend((dataset, seed, learner, method, arm) for arm in arms)
    return result


def preflight(root, protocol):
    import numpy as np
    from _shared.overnight_risk import primitive_kernel
    result = {"time_utc": now(), "cpu_count": os.cpu_count(), "gpu": [], "gram_checks": []}
    rng = np.random.default_rng(20261018)
    real = rng.normal(size=(1001, 7))
    states = rng.normal(size=(1001, 128)) + 1j * rng.normal(size=(1001, 128))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    import cupy as cp
    for device in protocol["gpu_devices"]:
        properties = cp.cuda.runtime.getDeviceProperties(device)
        result["gpu"].append({"device": device, "name": properties["name"].decode(), "bytes": int(properties["totalGlobalMem"])})
        for kind, value in (("fidelity", states), ("rbf", real)):
            cpu = primitive_kernel(value, value, kind, .3)
            accelerated = primitive_kernel(value, value, kind, .3, device)
            np.testing.assert_allclose(cpu, accelerated, atol=1e-11, rtol=1e-11)
            result["gram_checks"].append({"device": device, "kind": kind, "max_absolute_error": float(abs(cpu - accelerated).max())})
    write_json(root / "preflight.json", result)
    return result


def assert_preserved(metadata):
    for name, value in {**metadata["source_sha256"], **metadata["preserved_inputs_sha256"]}.items():
        if sha(ROOT / name) != value:
            raise AssertionError(f"Frozen input/source changed: {name}")


def execute_stage(stage, methods, seeds, protocol, root, metadata):
    output = root / stage
    output.mkdir(parents=True, exist_ok=True)
    tasks = jobs(protocol, methods, seeds, stage == "full")
    write_json(output / "stage_manifest.json", {"stage": stage, "methods": methods, "seeds": seeds,
        "jobs": [list(job) for job in tasks], "committed_utc": now(), "source_sha256": metadata["source_sha256"]})
    metadata.update(phase=stage + "_prepare", expected_jobs=len(tasks), completed_jobs=0)
    write_json(root / "metadata.json", metadata)
    workers = protocol["workers"]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        preparations = {pool.submit(prepare, dataset, seed, stage, protocol, output): (dataset, seed)
            for dataset, seed in itertools.product(protocol["datasets"], seeds)}
        prepared = set()
        for future in as_completed(preparations):
            try:
                prepared.add(future.result())
            except Exception as error:
                metadata["failed_jobs"].append({"stage": stage, "preparation": preparations[future], "error": repr(error)})
        metadata["phase"] = stage + "_search"
        write_json(root / "metadata.json", metadata)
        pending = {pool.submit(run_job, job, stage, protocol, output, metadata["source_sha256"]): job for job in tasks if job[:2] in prepared}
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as error:
                entry = {"stage": stage, "job": list(pending[future]), "error": f"{type(error).__name__}: {error}"}
                metadata["failed_jobs"].append(entry)
                print("failed", entry, flush=True)
            else:
                metadata["completed_jobs"] += 1
            write_json(root / "metadata.json", metadata)
    assert_preserved(metadata)
    metadata["phase"] = stage + "_selection"
    write_json(root / "metadata.json", metadata)
    promoted = save_stage(output, tasks, metadata["source_sha256"], protocol, methods, stage)
    metadata[stage + "_summary"] = {"expected_jobs": len(tasks), "completed_jobs": metadata["completed_jobs"], "selected_families": promoted}
    write_json(root / "metadata.json", metadata)
    return promoted


def root_report(root, metadata):
    lines = ["# 야간 위험 주입 탐색 실행 상태", "", "## Material Passport", "",
        f"상태: {metadata['status']}. 시작: {metadata['started_utc']}. 마감: 2026년10월19일11:30 KST.", "",
        "32개 사전 고정 방법군을 두 커널에서 탐색한다. 각 후보의 정보/거리/회로 기전은 OVERNIGHT_RISK_EXPERIMENTS.md와 고정 프로토콜에 기록했다. 실험 실패와 미완료 범위는 숨기지 않는다.", ""]
    for stage in ("discovery", "confirmation", "full"):
        if stage + "_summary" in metadata:
            item = metadata[stage + "_summary"]
            lines.append(f"- {stage}: {item['completed_jobs']}/{item['expected_jobs']} 작업. 선정 방법: {', '.join(item['selected_families']) or '없음'}.")
    lines += ["", f"보존된 실패: {len(metadata['failed_jobs'])}개. 후보/시드별 실패는metadata.json에 있다.", "",
        "스모크 통과는 실용적 후보 선정이지 최종 유의성이나 양자 우위의 입증이 아니다. 전체 결과가 있으면 full/결과보고서.md를 확인한다. 정확한 전체10시드가 없는 비교를 전체 결과로 주장하지 않는다."]
    (root / "실행보고서.md").write_text("\n".join(lines) + "\n")


def execute(args):
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["lambdas"] == [i / 20 for i in range(1, 21)]
    assert len(protocol["methods"]) == len(set(protocol["methods"])) == 32
    root = ROOT / "code/results" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    with local_lock(root / "run.lock"):
        if (root / "metadata.json").exists():
            raise AssertionError("Existing run retained; no automatic retry")
        hashes = {name: sha(ROOT / name) for name in SOURCES}
        frozen = preserved_inputs(preservation_protocol(protocol))
        old = ROOT / "code/results/cmc_20261018_german_zz_phase/full"
        for name, value in json.loads((old / "verification_receipt.json").read_text())["outputs_sha256"].items():
            assert sha(old / name) == value
            frozen[str((old / name).relative_to(ROOT))] = value
        metadata = {"run_id": args.run_id, "status": "running", "phase": "preflight", "pid": os.getpid(),
            "started_utc": now(), "protocol": protocol, "source_sha256": hashes,
            "preserved_inputs_sha256": frozen, "failed_jobs": [], "test_released": False,
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "qiskit", "qiskit-aer", "xgboost", "cupy-cuda12x", "metric-learn", "fastrlock")}}
        write_json(root / "metadata.json", metadata)
        with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
            for name in SOURCES:
                archive.add(ROOT / name, arcname=name)
        try:
            metadata["preflight"] = preflight(root, protocol)
            shortlist = execute_stage("discovery", protocol["methods"], protocol["discovery_seeds"], protocol, root, metadata)
            promoted = execute_stage("confirmation", shortlist, protocol["seeds"], protocol, root, metadata) if shortlist else []
            metadata["full_families"] = promoted
            write_json(root / "metadata.json", metadata)
            if promoted and time.time() < stamp(protocol["new_full_cutoff_utc"]):
                execute_stage("full", promoted, protocol["seeds"], protocol, root, metadata)
                metadata["test_released"] = True
            write_json(root / "metadata.json", metadata)
            subprocess.run([sys.executable, str(ROOT / "code/validate_overnight_risk.py"), "--run-id", args.run_id], check=True, cwd=ROOT)
            metadata["validation_receipt_sha256"] = sha(root / "validation_receipt.json")
            metadata.update(status="completed_with_failures" if metadata["failed_jobs"] else "completed",
                phase="completed_full" if metadata["test_released"] else "completed_no_promotion", finished_utc=now())
        except BaseException as error:
            metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write_json(root / "metadata.json", metadata)
            root_report(root, metadata)


def finalize_partial(args):
    root = ROOT / "code/results" / args.run_id
    metadata = json.loads((root / "metadata.json").read_text())
    assert_preserved(metadata)
    plan = root / "full/stage_manifest.json"
    if plan.exists() and not (root / "full/verification_receipt.json").exists():
        stage = json.loads(plan.read_text())
        tasks = [tuple(job) for job in stage["jobs"]]
        groups, _, missing = collect(root / "full", tasks, metadata["source_sha256"])
        if groups:
            save_stage(root / "full", tasks, metadata["source_sha256"], metadata["protocol"], stage["methods"], "full")
            metadata["test_released"] = True
            metadata["full_summary"] = {"expected_jobs": len(tasks), "completed_jobs": len(groups), "selected_families": stage["methods"]}
    metadata.update(status="deadline_reached", phase="finalized_partial", finished_utc=now())
    write_json(root / "metadata.json", metadata)
    root_report(root, metadata)


def supervisor(args, command):
    root = ROOT / "code/results" / args.run_id
    protocol = json.loads(PROTOCOL.read_text())
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", PYTHONPATH=str(ROOT / ".overnight-libs") + os.pathsep + str(ROOT / "code"))
    with local_lock(root / "supervisor.lock"):
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, start_new_session=True)
        monitor = {"status": "running", "supervisor_pid": os.getpid(), "experiment_pid": child.pid,
            "started_utc": now(), "hard_deadline_utc": protocol["compute_deadline_utc"], "automatic_retries": False}
        hit_deadline = False
        while child.poll() is None:
            monitor["heartbeat_utc"] = now()
            write_json(root / "monitor.json", monitor)
            if time.time() >= stamp(protocol["compute_deadline_utc"]):
                hit_deadline = True
                print("AUTHORIZED COMPUTE DEADLINE: preserving completed units; terminating campaign process group", flush=True)
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            time.sleep(20)
        if hit_deadline:
            result = subprocess.run([*command, "--finalize-partial"], cwd=ROOT, env=environment)
            monitor.update(status="deadline_reached", finalize_returncode=result.returncode)
        else:
            monitor.update(status="completed" if child.returncode == 0 else "failed", returncode=child.returncode)
        monitor["finished_utc"] = now()
        write_json(root / "monitor.json", monitor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_20261018_overnight_risk_v2")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--supervisor", action="store_true")
    parser.add_argument("--finalize-partial", action="store_true")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        parser.error("One run name only")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id]
    if args.finalize_partial:
        finalize_partial(args)
    elif args.launch:
        root = ROOT / "code/results" / args.run_id
        root.mkdir(parents=True, exist_ok=True)
        if (root / "metadata.json").exists() or (root / "monitor.json").exists():
            raise AssertionError("Existing experiment retained")
        with (root / "server.log").open("a") as log:
            process = subprocess.Popen([*command, "--supervisor"], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"run_id": args.run_id, "supervisor_pid": process.pid}))
    elif args.supervisor:
        supervisor(args, command)
    else:
        execute(args)
