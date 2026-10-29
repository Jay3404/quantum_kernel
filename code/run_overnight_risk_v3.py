"""Run the unchanged scientific protocol with explicit implementation repairs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import run_overnight_risk as original
from _shared.overnight_runtime_v3 import run_job

PROTOCOL = original.ROOT / "docs/cmc_overnight_risk_v3_protocol_20261018.json"
ADDITIONAL_SOURCES = (
    "code/run_overnight_risk_v3.py", "code/_shared/overnight_runtime_v3.py",
    "code/overnight_resource_guard.py", "code/tests/test_overnight_runtime_v3.py",
    "docs/cmc_overnight_risk_v3_protocol_20261018.json",
)


def configure():
    original.PROTOCOL = PROTOCOL
    original.SOURCES = tuple(dict.fromkeys((*original.SOURCES, *ADDITIONAL_SOURCES)))
    original.run_job = run_job


def main():
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_20261018_overnight_risk_v3")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--supervisor", action="store_true")
    parser.add_argument("--finalize-partial", action="store_true")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        parser.error("One run name only")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run-id", args.run_id]
    if args.finalize_partial:
        original.finalize_partial(args)
    elif args.supervisor:
        original.supervisor(args, command)
    elif args.launch:
        root = original.ROOT / "code/results" / args.run_id
        root.mkdir(parents=True, exist_ok=True)
        if (root / "metadata.json").exists() or (root / "monitor.json").exists():
            raise AssertionError("Existing experiment retained")
        guard_path = root / "resource_guard.json"
        guard = json.loads(guard_path.read_text())
        assert guard["status"] == "active" and time.time() - guard_path.stat().st_mtime < 60
        assert guard["maximum_concurrent_large_exact_fits"] == 3
        with (root / "server.log").open("a") as log:
            process = subprocess.Popen([*command, "--supervisor"], cwd=original.ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        print({"run_id": args.run_id, "supervisor_pid": process.pid})
    else:
        original.execute(args)


if __name__ == "__main__":
    main()
