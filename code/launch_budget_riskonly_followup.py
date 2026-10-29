"""One-shot deployment gate and detached launch for the authorized follow-up."""
import os
from pathlib import Path
import subprocess
import sys
import time

from run_budget_riskonly_followup import ROOT, RUN, initialize, now, read, sha, verify, write


def main():
    assert ROOT == Path("/workspace/quantum_kernel")
    root = ROOT / "code/results" / RUN
    assert not root.exists(), "Never restart or overwrite an existing scientific attempt"
    attempt = 1
    while root.with_name(RUN + f"_deployment_{attempt:02d}").exists():
        attempt += 1
    audit = root.with_name(RUN + f"_deployment_{attempt:02d}")
    audit.mkdir()
    command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "code/tests", "-p", "test_budget*.py"]
    test = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    write(audit / "technical_tests.json", {"utc": now(), "exit_code": test.returncode,
          "stdout": test.stdout, "stderr": test.stderr, "scope": "Synthetic algebra/SVC persistence and archived-plan tests; not scientific test outcomes"})
    assert test.returncode == 0, "Technical failure preserved; no scientific launch"
    initialize(root)
    verify(root, True)
    write(root / "operations/server_technical_tests.json", read(audit / "technical_tests.json"))
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert env.get(key) == "1", key
    assert env.get("PYTHONDONTWRITEBYTECODE") == "1"
    with (root / "pipeline.log").open("x") as log:
        pipeline = subprocess.Popen([sys.executable, "-B", str(ROOT / "code/run_budget_riskonly_followup.py"),
                                     "--operation", "run"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
    time.sleep(1)
    assert pipeline.poll() is None, "Inspect preserved pipeline failure; do not retry"
    with (root / "supervisor.log").open("x") as log:
        supervisor = subprocess.Popen([sys.executable, "-B", str(ROOT / "code/monitor_budget_riskonly_followup.py"),
                                       "--pid", str(pipeline.pid)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, start_new_session=True)
    write(root / "launch_receipt.json", {"utc": now(), "pipeline_pid": pipeline.pid, "supervisor_pid": supervisor.pid,
          "protocol_sha256": sha(ROOT / "docs/cmc_budget_riskonly_protocol_20261026_v2.json"),
          "host_root": "/home/user/workspaces/jaemin/quantum_kernel", "container": "quantum-cmc-a100",
          "container_root": str(ROOT), "technical_tests_passed": True, "automatic_retry": False})
    print({"run": RUN, "pipeline_pid": pipeline.pid, "supervisor_pid": supervisor.pid,
           "status": read(root / "metadata.json")["status"]})


if __name__ == "__main__":
    main()
