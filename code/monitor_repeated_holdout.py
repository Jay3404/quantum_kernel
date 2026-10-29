"""Independent non-intervening monitor for preflight and scientific execution."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from _shared import repeated_holdout as p
from run_pca9_followup import active, resources


def supervise(args):
    root = p.ROOT / "code/results" / args.run_id
    previous = None
    peaks = {"cpu_percent": 0., "host_used_fraction": 0., "active_workers": 0}
    while True:
        meta = json.loads((root / "metadata.json").read_text()) if (root / "metadata.json").exists() else {}
        heartbeat = json.loads((root / "monitor.json").read_text()) if (root / "monitor.json").exists() else {}
        state, previous = resources(previous)
        alive = active(args.pid)
        peaks["cpu_percent"] = max(peaks["cpu_percent"], state["cpu_percent"])
        peaks["host_used_fraction"] = max(peaks["host_used_fraction"], state["host_used_fraction"])
        peaks["active_workers"] = max(peaks["active_workers"], len(heartbeat.get("active", [])))
        mismatches = [name for name, value in meta.get("source_sha256", {}).items() if p.sha(p.ROOT / name) != value]
        alert = "Frozen source mismatch" if mismatches else None
        if not alive and meta.get("status") not in ("completed", "failed", "preflight_failed", "authorized_stopped"):
            alert = "Pipeline exited without terminal metadata; preserve workers, no retry"
        if alive and heartbeat.get("status") == "running" and time.time() - (root / "monitor.json").stat().st_mtime > 1800:
            alert = "Scientific heartbeat stale; inspect without killing active fits"
        receipt = {"heartbeat_utc": p.now(), "status": "monitoring" if alive and not alert else "attention_required" if alert else meta.get("status"),
            "supervisor_pid": os.getpid(), "experiment_pid": args.pid, "experiment_alive": alive,
            "phase": meta.get("phase"), "completed_jobs": meta.get("completed_jobs"), "resources": state, "observed_peaks": peaks,
            "active_worker_pids": [w["pid"] for w in heartbeat.get("active", []) if active(w["pid"])],
            "source_mismatches": mismatches, "source_sha256": p.sha(__file__), "alert": alert,
            "automatic_retries": False, "active_fits_terminated_for_resources": False}
        p.write_json(root / "operations/supervision/supervision.json", receipt)
        if not alive:
            p.write_json(root / "operations/supervision/outcome.json", receipt)
            break
        time.sleep(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    assert args.run_id == Path(args.run_id).name
    if args.launch:
        folder = p.ROOT / "code/results" / args.run_id / "operations/supervision"
        folder.mkdir(parents=True, exist_ok=True)
        assert not (folder / "launch_receipt.json").exists()
        with (folder / "supervisor.log").open("x") as handle:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id, "--pid", str(args.pid)],
                cwd=p.ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {"supervisor_pid": child.pid, "experiment_pid": args.pid, "source_sha256": p.sha(__file__), "launched_utc": p.now()}
        p.write_json(folder / "launch_receipt.json", receipt)
        print(json.dumps(receipt))
    else:
        supervise(args)
