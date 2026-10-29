"""Audit inherited executions and exact search-prefix continuation provenance."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _shared import overnight_risk as n
from _shared import overnight_runtime_v4 as runtime
import validate_overnight_risk as original


def main(run_id):
    original.main(run_id)
    root = n.ROOT / "code/results" / run_id
    metadata = json.loads((root / "metadata.json").read_text())
    inherited, resumed, pids = 0, 0, []
    for job in metadata["inherited_completed_jobs"]:
        folder = n.job_folder(root / "full", job)
        current = json.loads((folder / "complete.json").read_text())
        parent_path = n.ROOT / current["parent_complete_path"]
        assert n.sha(parent_path) == current["parent_complete_sha256"] == n.sha(folder / "parent_complete.json")
        parent = json.loads(parent_path.read_text())
        assert current["record_type"] == "adopted_execution_receipt_no_new_fit" and current["new_model_fit_calls"] == 0
        assert current["model_execution_source_sha256"] == parent["source_sha256"]
        assert all(current[key] == value for key, value in parent.items() if key != "source_sha256")
        for name in ("margins.npz", "exact_margins.npz", "parameter_lock.json"):
            assert n.sha(folder / name) == n.sha(parent_path.parent / name)
        inherited += 1
    resume_jobs = {tuple(row["job"]): row for row in metadata["resumed_checkpoints"]}
    for job, info in resume_jobs.items():
        folder = n.job_folder(root / "full", job)
        arrays, _ = n.load_inputs(root / "full", *job[:2])
        rows, scores = runtime.load_checkpoint(folder / "parent_checkpoint", arrays, metadata["protocol"], job[3])
        assert len(rows) == info["inherited_trials"]
        if not (folder / "complete.json").exists():
            continue
        saved = json.loads((folder / "complete.json").read_text())
        assert saved["inherited_search_trials"] == len(rows)
        assert saved["new_search_fit_calls"] == len(saved["trials"]) - len(rows)
        for old, new in zip(rows, saved["trials"]):
            assert all(new[key] == value for key, value in old.items() if key != "representation_details")
        with np.load(folder / "margins.npz", allow_pickle=False) as archive:
            for part in scores:
                np.testing.assert_array_equal(archive[part][:len(rows)], np.asarray(scores[part]))
        resumed += 1
    for path in (root / "full/candidates").rglob("worker_operation.json"):
        operation = json.loads(path.read_text())
        assert operation["fresh_process_for_one_job"]
        assert operation["source_sha256"] == metadata["source_sha256"]
        pids.append(operation["pid"])
    assert len(pids) == len(set(pids)), "Each observed process executed only one job"
    timing = []
    if (root / "full/selection_lock.json").exists():
        lock = json.loads((root / "full/selection_lock.json").read_text())
        for choice in lock["choices"]:
            folder = n.job_folder(root / "full", choice["job"])
            saved = json.loads((folder / "complete.json").read_text())
            adopted = saved.get("record_type") == "adopted_execution_receipt_no_new_fit"
            timing.append({**{k: choice[k] for k in ("dataset", "seed", "kernel", "family", "arm", "role")},
                "execution_origin": "v3_adopted_no_new_fit" if adopted else "v4_fresh_single_job_process",
                "inherited_search_trials": saved.get("inherited_search_trials", 0),
                **{k: choice.get(k) for k in ("training_seconds", "test_inference_seconds", "train_kernel_seconds", "test_kernel_seconds", "exact_fit_seconds")}})
        table = pd.DataFrame(timing)
        table.to_csv(root / "full/runtime_execution_provenance.csv", index=False)
        table.groupby(["dataset", "kernel", "family", "arm", "role", "execution_origin"])[["training_seconds", "test_inference_seconds", "train_kernel_seconds", "test_kernel_seconds", "exact_fit_seconds"]].agg(["mean", "std"]).to_csv(root / "full/runtime_by_execution_origin.csv")
    result = {"status": "passed", "scope": "Original full audit plus byte-preserved adoption, verified prefix scores and distinct one-job process IDs; not independent full retraining",
        "adopted_complete_jobs": inherited, "completed_resumed_jobs": resumed,
        "observed_fresh_worker_processes": len(pids), "parent_failed_attempts_retained": len(metadata["parent_failed_attempts"]),
        "original_audit_sha256": n.sha(root / "validation_receipt.json"), "source_sha256": n.sha(Path(__file__))}
    result["outputs_sha256"] = {name: n.sha(root / "full" / name) for name in
        ("runtime_execution_provenance.csv", "runtime_by_execution_origin.csv") if (root / "full" / name).exists()}
    n.write_json(root / "continuation_validation_receipt.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_20261019_overnight_risk_v4")
    main(parser.parse_args().run_id)
