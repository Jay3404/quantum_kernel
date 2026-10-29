"""User-requested full German ZZ-phase follow-up; does not override smoke gates."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tarfile

from _shared.full_data import write_json
from _shared.improved import local_lock, sha
from _shared.quantum_ten import run_quantum, run_classical, job_folder
from _shared.german_zz_reporting import report
from run_improved_experiment import supervise
from run_quantum_ten import ROOT, SOURCES as TEN_SOURCES, jobs, now, prepare_full, verify_phase

PROTOCOL = ROOT / "docs/cmc_german_zz_phase_protocol_20261018.json"
SOURCES = (*TEN_SOURCES, "code/run_german_zz_phase.py", "code/_shared/german_zz_reporting.py",
    "code/validate_german_zz_phase.py", "code/tests/test_german_zz_phase.py",
    "docs/cmc_german_zz_phase_protocol_20261018.json")


def protocol_check(protocol):
    if protocol["datasets"] != ["german_numeric", "south_german"] or protocol["methods"] != ["zz_phase"]:
        raise AssertionError("Keep the requested German-only ZZ-phase scope")
    if protocol["seeds"] != list(range(10, 101, 10)) or protocol["lambdas"] != [i / 20 for i in range(1, 21)]:
        raise AssertionError("Keep ten paired seeds and every positive 0.05 lambda")
    if protocol["arms"] != ["quantum", "separable", "shuffled"]:
        raise AssertionError("Keep entanglement/risk ablations")


def preserved_inputs(protocol):
    paths = {"data/german-data-numeric", "data/SouthGermanCredit.asc"}
    for dataset, seed in itertools.product(protocol["datasets"], protocol["seeds"]):
        folder = f"code/{dataset}/results/{protocol['source_run']}/seed_{seed:03d}"
        paths.update(f"{folder}/{name}" for name in ("prepared.npz", "prepared.json", "split_indices.json"))
    fico = json.loads((ROOT / "code/results/cmc_20261018_fico_single_811/preflight.json").read_text())["input_hashes"]
    for name, value in fico.items():
        if sha(ROOT / name) != value:
            raise AssertionError("Preserve FICO sources/inputs")
    paths.update(fico)
    for run, suffix in (("cmc_20261018_single_811", ""), ("cmc_20261018_fico_single_811", ""),
                        ("cmc_20261018_separation", "smoke"), ("cmc_20261018_separation_round2", ""),
                        ("cmc_20261018_quantum_ten", "smoke")):
        folder = ROOT / "code/results" / run / suffix
        receipt = folder / "verification_receipt.json"
        paths.add(str(receipt.relative_to(ROOT)))
        for name, value in json.loads(receipt.read_text())["outputs_sha256"].items():
            if sha(folder / name) != value:
                raise AssertionError("Prior result differs from its receipt")
            paths.add(str((folder / name).relative_to(ROOT)))
    return {name: sha(ROOT / name) for name in sorted(paths)}


def check_grids(output, tasks, protocol):
    c_count, gamma_count, n_lambda = len(protocol["C_grid"]), len(protocol["gamma_factors"]), len(protocol["lambdas"])
    for job in tasks:
        receipt = json.loads((job_folder(output, job) / "complete.json").read_text())
        rows = receipt["trials"]
        if job[2] == "classical":
            strengths = sum(1 if v == "baseline" else n_lambda + 1 if v == "aligned_rbf" else n_lambda for v in protocol["classical_variants"])
            expected = strengths * 2 * c_count * gamma_count + 9
        else:
            expected = (1 if job[2] == "standard" else n_lambda + 1) * c_count
            actual = {(r["lambda"], r["C"]) for r in rows}
            grid = {(strength, c) for strength in ([0.] if job[2] == "standard" else [0., *protocol["lambdas"]]) for c in protocol["C_grid"]}
            if actual != grid or any(r["model_fit_calls"] != 1 or r["ensemble_members"] != 1 or r["fit_rows"] != 800 for r in rows):
                raise AssertionError("Incomplete quantum grid or not one full-training SVC")
        if len(rows) != expected or any(r["fit_rows"] != 800 for r in rows):
            raise AssertionError("Incomplete candidate grid or training row count")


def verify(root):
    metadata = json.loads((root / "metadata.json").read_text())
    protocol, hashes = metadata["protocol"], metadata["source_sha256"]
    protocol_check(protocol)
    for name, value in {**hashes, **metadata["preserved_inputs_sha256"]}.items():
        if sha(ROOT / name) != value:
            raise AssertionError(f"Source/input/prior result changed: {name}")
    output = root / "full"
    tasks = jobs(protocol, protocol["methods"], protocol["seeds"], True)
    check_grids(output, tasks, protocol)
    return verify_phase(output, "full", protocol, protocol["methods"], protocol["seeds"], hashes)


def execute(args):
    protocol = json.loads(PROTOCOL.read_text())
    protocol_check(protocol)
    root = ROOT / "code/results" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    path = root / "metadata.json"
    with local_lock(root / "run.lock"):
        if path.exists():
            raise AssertionError("Preserve existing run; no automatic retries")
        hashes = {name: sha(ROOT / name) for name in SOURCES}
        preserved = preserved_inputs(protocol)
        metadata = {"run_id": args.run_id, "status": "running", "phase": "full_prepare", "pid": os.getpid(),
            "started_utc": now(), "protocol": protocol, "source_sha256": hashes,
            "preserved_inputs_sha256": preserved, "workers": args.workers,
            "full_launched": True, "completed_jobs": 0, "failed_jobs": [],
            "packages": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "scikit-learn", "qiskit", "qiskit-aer", "xgboost")}}
        write_json(path, metadata)
        with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
            for name in SOURCES:
                archive.add(ROOT / name, arcname=name)
        output = root / "full"
        output.mkdir(exist_ok=True)
        tasks = jobs(protocol, protocol["methods"], protocol["seeds"], True)
        metadata["expected_jobs"] = len(tasks)
        try:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                for future in as_completed([pool.submit(prepare_full, d, s, protocol, output)
                        for d, s in itertools.product(protocol["datasets"], protocol["seeds"])]):
                    future.result()
                metadata["phase"] = "full_search"
                write_json(path, metadata)
                pending = {pool.submit(run_classical if job[2] == "classical" else run_quantum,
                    job, protocol, output, hashes, True): job for job in tasks}
                for future in as_completed(pending):
                    try:
                        future.result()
                    except Exception as error:
                        metadata["failed_jobs"].append({"job": pending[future], "error": f"{type(error).__name__}: {error}"})
                        print("failed", metadata["failed_jobs"][-1], flush=True)
                    else:
                        metadata["completed_jobs"] += 1
                    write_json(path, metadata)
            if metadata["failed_jobs"]:
                raise RuntimeError("Failed jobs retained; no automatic retry")
            if any(sha(ROOT / name) != value for name, value in {**hashes, **preserved}.items()):
                raise AssertionError("Scientific sources or preserved inputs changed")
            check_grids(output, tasks, protocol)
            metadata["phase"] = "full_reporting"
            write_json(path, metadata)
            metadata["result"] = report(output, tasks, hashes, protocol)
            metadata["audit"] = verify(root)
            metadata.update(status="completed", phase="full_completed", finished_utc=now())
        except BaseException as error:
            metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write_json(path, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_20261018_german_zz_phase")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--supervisor", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or not 1 <= args.workers <= 12:
        parser.error("Use one run name and 1..12 workers")
    root = ROOT / "code/results" / args.run_id
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id, "--workers", str(args.workers)]
    if args.verify:
        print(json.dumps(verify(root), ensure_ascii=False))
    elif args.launch:
        root.mkdir(parents=True, exist_ok=True)
        if (root / "metadata.json").exists() or (root / "monitor.json").exists():
            raise AssertionError("Existing experiment retained")
        with (root / "server.log").open("a") as log:
            process = subprocess.Popen([*command, "--supervisor"], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"run_id": args.run_id, "supervisor_pid": process.pid}))
    elif args.supervisor:
        args.blas_threads = 1
        supervise(args, command)
    else:
        execute(args)
