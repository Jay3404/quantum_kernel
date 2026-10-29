"""Archive gate and integrity audit, not independent full-model retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_gate(run_id):
    upstream = "cmc_20261019_overnight_risk_v4"
    root = ROOT / "code/results" / upstream
    meta = json.loads((root / "metadata.json").read_text())
    a = json.loads((root / "validation_receipt.json").read_text())
    b = json.loads((root / "continuation_validation_receipt.json").read_text())
    assert meta["status"] == "completed" and meta["completed_jobs"] == 1160 and not meta["failed_jobs"]
    assert a["status"] == b["status"] == "passed" and b["original_audit_sha256"] == sha(root / "validation_receipt.json")
    assert all(sha(ROOT / name) == value for name, value in meta["source_sha256"].items())
    with tarfile.open(root / "source_snapshot.tar.gz") as archive:
        assert all(hashlib.sha256(archive.extractfile(name).read()).hexdigest() == value for name, value in meta["source_sha256"].items())
    receipt = json.loads((root / "full/verification_receipt.json").read_text())
    assert all(sha(root / "full" / name) == value for name, value in receipt["outputs_sha256"].items())
    assert all(sha(root / "full" / name) == value for name, value in b["outputs_sha256"].items())
    lock = json.loads((root / "full/selection_lock.json").read_text())
    assert len(lock["candidate_receipts_sha256"]) == 1160
    for name, value in lock["candidate_receipts_sha256"].items():
        assert sha(ROOT / name) == value
        saved = json.loads((ROOT / name).read_text())
        for file, key in (("margins.npz", "margins_sha256"), ("exact_margins.npz", "exact_margins_sha256"),
                          ("parameter_lock.json", "parameter_lock_sha256")):
            assert sha((ROOT / name).parent / file) == saved[key]
    result = {"status": "passed", "upstream_run": upstream, "completed_jobs": 1160,
        "local_candidate_hash_checks": 1160, "source_sha256": sha(__file__),
        "upstream_receipts_sha256": {str((root / name).relative_to(ROOT)): sha(root / name) for name in
            ("metadata.json", "validation_receipt.json", "continuation_validation_receipt.json", "source_snapshot.tar.gz",
             "full/selection_lock.json", "full/test_release.json", "full/verification_receipt.json")}}
    target = ROOT / "code/results" / run_id
    target.mkdir(parents=True, exist_ok=True)
    path = target / "upstream_local_archive_receipt.json"
    assert not path.exists(), "Preserve existing archive receipt"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "passed", "local_candidate_hash_checks": 1160}))


def audit(run_id):
    import numpy as np
    import pandas as pd
    from _shared import pca9_followup as p
    from _shared import pca9_reporting as report
    from _shared import overnight_reporting as common
    from _shared import overnight_risk as old
    root = ROOT / "code/results" / run_id
    output = root / "full"
    meta = json.loads((root / "metadata.json").read_text())
    protocol = meta["protocol"]
    assert all(sha(ROOT / name) == value for name, value in {**meta["source_sha256"], **meta["preserved_inputs_sha256"]}.items())
    with tarfile.open(root / "source_snapshot.tar.gz") as archive:
        assert all(hashlib.sha256(archive.extractfile(name).read()).hexdigest() == value for name, value in meta["source_sha256"].items())
    assert sha(root / "preflight_receipt.json") == meta["preflight_sha256"]
    plan = json.loads((output / "stage_manifest.json").read_text())
    tasks = p.jobs(protocol)
    assert plan["jobs"] == [list(job) for job in tasks] and len(tasks) == 870
    groups, receipts, missing = common.collect(output, tasks, meta["source_sha256"])
    assert not missing
    pids, prepared = [], set()
    for job, saved in groups.items():
        arrays, record = old.load_inputs(output, *job[:2])
        prepared.add(job[:2])
        folder = old.prepared_folder(output, *job[:2])
        assert arrays["train"].shape[1] == record["pca_components"] == 9
        assert sha(folder / "split.json") == record["split_sha256"]
        assert sha(folder / "nested_pca_fits.jsonl") == record["nested_pca_fits_sha256"]
        assert sha(folder / "oof.json") == record["oof_sha256"]
        split = json.loads((folder / "split.json").read_text())["actual"]
        assert record["training_row_ids"] == split["train"] and len(arrays["train"]) == len(split["train"])
        assert not set(split["test"]) & set(split["train"] + split["validation"])
        assert not {"y_test", "y_gate"} & set(arrays)
        oof = json.loads((folder / "oof.json").read_text())
        for fold in oof["folds"]:
            assert not set(fold["train_ids"]) & set(fold["heldout_ids"] + split["test"] + split["validation"])
            assert set(fold["train_ids"] + fold["heldout_ids"]) == set(split["train"])
            for inner in fold["calibration_folds"]:
                assert not set(inner["train_ids"]) & set(inner["heldout_ids"] + fold["heldout_ids"] + split["test"] + split["validation"])
                assert set(inner["train_ids"] + inner["heldout_ids"]) == set(fold["train_ids"])
        for inner in oof["final_calibration_folds"]:
            assert not set(inner["train_ids"]) & set(inner["heldout_ids"] + split["test"] + split["validation"])
            assert set(inner["train_ids"] + inner["heldout_ids"]) == set(split["train"])
        teacher = record["teacher_reuse"]
        assert sha(ROOT / teacher["source_audit_path"]) == teacher["source_audit_sha256"] and teacher["new_fit_calls"] == 0
        assert all(p.digest_array(arrays[key]) == value for key, value in teacher["arrays_sha256"].items())
        operation = json.loads((old.job_folder(output, job) / "worker_operation.json").read_text())
        assert operation["status"] == "completed" and operation["fresh_process_for_one_job"]
        assert operation["source_sha256"] == meta["source_sha256"]
        pids.append(operation["pid"])
        if job[3] == "reference":
            assert {row["method"] for row in saved["trials"]} == {"raw", "pca9"}
            continue
        grid = [(g, s, c) for g in range(3) for s in ([0.] if job[3] == "baseline" else [0., *protocol["lambdas"]]) for c in protocol["C_grid"]]
        assert [(row["geometry"], row["lambda"], row["C"]) for row in saved["trials"]] == grid
        lock = json.loads((old.job_folder(output, job) / "parameter_lock.json").read_text())
        with np.load(old.job_folder(output, job) / "exact_margins.npz", allow_pickle=False) as margins:
            for role, row in saved["choices"].items():
                proxy = old.select_trial(saved["trials"], role == "proposed")
                assert proxy == lock["choices"][role]
                assert all(row[key] == proxy[key] for key in ("lambda", "C", "geometry", "fit_rows"))
                assert row["final_exact"] and row["final_model_fit_calls"] == row["ensemble_members"] == 1
                assert row["fit_rows"] == len(split["train"])
                threshold, _ = p.base.threshold_for_cost(arrays["y_tune"], margins[role + "_tune"], 0.)
                assert threshold == row["threshold"]
                assert len(margins[role + "_gate"]) == len(split["test"])
    assert len(pids) == len(set(pids)) == 870 and len(prepared) == 30
    lock = json.loads((output / "selection_lock.json").read_text())
    choices = common.sensitivity_choices(output, report.selected_choices(groups))
    assert lock["choices"] == choices and lock["candidate_receipts_sha256"] == receipts
    assert lock["lambda_profiles"] == common.sensitivity_choices(output, common.lambda_profiles(groups, protocol))
    release = json.loads((output / "test_release.json").read_text())
    assert release["selection_lock_sha256"] == sha(output / "selection_lock.json")
    expected, predictions = common.evaluate(output, choices, True, protocol)
    actual = pd.read_csv(output / "seed_metrics.csv", float_precision="round_trip")
    for column in ("cost", "fn", "fp", "tp", "tn", "f1", "recall", "balanced_accuracy", "threshold"):
        np.testing.assert_allclose(expected[column], actual[column], atol=1e-11, rtol=1e-11)
    pred = pd.read_csv(output / "predictions.csv.gz", float_precision="round_trip")
    expected_pred = pd.concat(predictions, ignore_index=True)
    for column in ("row_id", "y", "prediction"):
        np.testing.assert_array_equal(pred[column], expected_pred[column])
    np.testing.assert_allclose(pred.score, expected_pred.score, atol=1e-11, rtol=1e-11)
    receipt = json.loads((output / "verification_receipt.json").read_text())
    assert all(sha(output / name) == value for name, value in receipt["outputs_sha256"].items())
    dimension = pd.read_csv(output / "dimension_paired_cost.csv")
    assert len(dimension) == 90 and dimension.complete_ten.all() and dimension.holm_family_size.eq(90).all()
    result = {"status": "passed", "scope": "Frozen source/split/nested-fold records, teacher adoption, complete grid, same-fit scores and metric recomputation; not independent retraining",
        "jobs": 870, "prepared_pairs": 30, "new_worker_processes": 870, "dimension_contrasts": 90,
        "source_sha256": sha(__file__), "verification_receipt_sha256": sha(output / "verification_receipt.json")}
    p.write_json(root / "validation_receipt.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_pca9_20261020")
    parser.add_argument("--archive-upstream", action="store_true")
    args = parser.parse_args()
    assert args.run_id == Path(args.run_id).name
    archive_gate(args.run_id) if args.archive_upstream else audit(args.run_id)
