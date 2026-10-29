"""Reserve excess dense-fit slots without changing any campaign source."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

from _shared.full_data import write_json
from _shared.overnight_risk import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--total", type=int, default=8)
    parser.add_argument("--available", type=int, default=3)
    args = parser.parse_args()
    assert 0 < args.available <= args.total
    record_path = Path(args.record)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    lock_root = ROOT / ".overnight-locks/exact_slots"
    lock_root.mkdir(parents=True, exist_ok=True)
    handles = []
    record = {"status": "acquiring", "pid": os.getpid(),
        "maximum_concurrent_large_exact_fits": args.available,
        "configured_upper_bound": args.total, "deadline_utc": args.deadline,
        "reason": "Bound host-memory peaks across simultaneous versioned campaigns; no algorithm/configuration changes"}
    try:
        for index in range(args.available, args.total):
            handle = (lock_root / f"{index}.lock").open("a")
            handles.append(handle)
            while time.time() < deadline:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(2)
            else:
                raise TimeoutError("Deadline reached while reserving dense-fit slots")
        record["status"] = "active"
        while time.time() < deadline:
            record["heartbeat_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(record_path, record)
            time.sleep(min(20, max(0, deadline - time.time())))
        record["status"] = "deadline_released"
    finally:
        for handle in handles:
            handle.close()
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(record_path, record)
        print(json.dumps(record))


if __name__ == "__main__":
    main()
