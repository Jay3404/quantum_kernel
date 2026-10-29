"""Read-only independent supervision; never restart or cancel scientific fits."""
import argparse
import json
from pathlib import Path
import time

from run_budget_riskonly_followup import ROOT, RUN, now, read, verify, write


def identity(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return {"pid": pid, "state": fields[0], "start_ticks": fields[19],
                "command": Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()}
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    root = ROOT / "code/results" / RUN
    initial = identity(args.pid)
    assert initial and "run_budget_riskonly_followup.py" in initial["command"]
    assert "--operation run" in initial["command"]
    write(root / "operations/supervision_started.json", {"utc": now(), "target": initial, "automatic_retry": False})
    while True:
        current = identity(args.pid)
        same = current is not None and current["start_ticks"] == initial["start_ticks"] and current["state"] != "Z"
        meta = read(root / "metadata.json")
        try:
            verify(root)
            hashes = "passed"
        except Exception as error:
            hashes = str(error)
        monitor = root / "monitor.json"
        modified = monitor.stat().st_mtime if monitor.exists() else (root / "metadata.json").stat().st_mtime
        record = {"utc": now(), "target_alive": same, "target": current,
                  "pipeline_status": meta["status"], "source_hashes": hashes,
                  "monitor_age_s": time.time() - modified,
                  "preparing_age_is_not_a_scientific_timeout": meta["status"] == "preparing",
                  "completed_checkpoints": len(list((root / "jobs").glob("*/complete.json"))),
                  "failures": len(list(root.glob("**/failure.json"))) + len(list(root.glob("pipeline_failure_*.json"))),
                  "automatic_retry": False, "automatic_cancel": False}
        record["requires_inspection"] = hashes != "passed" or not same and meta["status"] != "completed"
        write(root / "operations/supervision.json", record, False)
        if not same:
            write(root / "operations/supervision_final.json", record)
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
