"""Integrity checks for repeated holdout; never retrains or selects on test."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from _shared import repeated_holdout as p
from _shared import repeated_reporting as report
from _shared import overnight_reporting as common
from _shared import overnight_risk as old


def audit_prepared(root, protocol):
    tests, coverage, overlaps, teacher_hashes = {}, [], [], {}
    for dataset in protocol["datasets"]:
        labels = p.base.load_data(dataset, str(p.ROOT / "data"))[1]
        cohorts = []
        for seed in protocol["seeds"]:
            folder = old.prepared_folder(root / "full", dataset, seed)
            arrays, record = old.load_inputs(root / "full", dataset, seed)
            split = json.loads((folder / "split.json").read_text())["actual"]
            assert split == p.split_indices(labels, seed)
            assert p.sha(folder / "split.json") == record["split_sha256"]
            assert not set(split["test"]) & set(split["train"] + split["validation"])
            assert not set(split["train"]) & set(split["validation"])
            assert set(sum(split.values(), [])) == set(range(len(labels)))
            assert len(arrays["train"]) == len(split["train"]) == round(.8 * len(labels))
            assert arrays["train"].shape[1] == protocol["pca_components"]
            assert record["training_row_ids"] == split["train"]
            assert not {"y_gate", "y_test", "test"} & set(arrays)
            assert not record["prior_preparation_reused"] and record["new_pca_lr_fit"]
            assert all(not row["reused_from_prior_campaign"] and row["new_outer_oof_fits"] == 5 and row["new_full_training_fit"] == 1
                for row in record["teachers"].values())
            assert p.sha(folder / "nested_pca_fits.jsonl") == record["nested_pca_fits_sha256"]
            logs = [json.loads(line) for line in (folder / "nested_pca_fits.jsonl").read_text().splitlines()]
            assert len(logs) == 25 and all(row["components"] == protocol["pca_components"] and row["smallest_retained_singular_value"] > row["rank_tolerance"] for row in logs)
            assert p.sha(folder / "oof.json") == record["oof_sha256"]
            oof = json.loads((folder / "oof.json").read_text())
            assert not oof["ensemble"] and not oof["test_labels_used"]
            for name in ("folds", "raw_teacher_folds"):
                heldouts = []
                for fold in oof[name]:
                    heldouts += fold["heldout_ids"]
                    assert not set(fold["train_ids"]) & set(fold["heldout_ids"] + split["test"] + split["validation"])
                    assert set(fold["train_ids"] + fold["heldout_ids"]) == set(split["train"])
                    key = "calibration_folds" if name == "folds" else "raw_lr_calibration_folds"
                    for inner in fold[key]:
                        assert not set(inner["train_ids"]) & set(inner["heldout_ids"] + fold["heldout_ids"] + split["test"] + split["validation"])
                        assert set(inner["train_ids"] + inner["heldout_ids"]) == set(fold["train_ids"])
                assert sorted(heldouts) == sorted(split["train"])
            for key in ("final_calibration_folds", "raw_lr_final_calibration_folds"):
                for inner in oof[key]:
                    assert not set(inner["train_ids"]) & set(inner["heldout_ids"] + split["test"] + split["validation"])
                    assert set(inner["train_ids"] + inner["heldout_ids"]) == set(split["train"])
            for prefix in ("risk", "gbdt", "raw_lr"):
                for part in p.PARTS:
                    assert np.isfinite(arrays[prefix + "_" + part]).all()
            teacher_hashes[dataset, seed] = p.engine.digest_array(arrays["gbdt_gate"])
            cohorts.append(set(split["test"]))
            coverage += [{"dataset": dataset, "seed": seed, "row_id": i} for i in split["test"]]
        assert len({tuple(sorted(group)) for group in cohorts}) == len(protocol["seeds"])
        assert len({teacher_hashes[dataset, seed] for seed in protocol["seeds"]}) == len(protocol["seeds"])
        for a, b in itertools.combinations(range(len(cohorts)), 2):
            overlaps.append({"dataset": dataset, "seed_a": protocol["seeds"][a], "seed_b": protocol["seeds"][b],
                "intersection": len(cohorts[a] & cohorts[b]), "jaccard": len(cohorts[a] & cohorts[b]) / len(cohorts[a] | cohorts[b])})
        tests[dataset] = {"distinct_test_cohorts": len(cohorts), "unique_test_customers": len(set.union(*cohorts)),
            "total_customer_test_occurrences": sum(map(len, cohorts))}
    pd.DataFrame(coverage).to_csv(root / "test_cohort_coverage.csv", index=False)
    pd.DataFrame(overlaps).to_csv(root / "test_cohort_overlap.csv", index=False)
    return {"status": "passed", "datasets": tests, "prepared_pairs": len(teacher_hashes), "nested_pca_fits": 25 * len(teacher_hashes)}


def audit(run_id):
    root = p.ROOT / "code/results" / run_id
    meta = json.loads((root / "metadata.json").read_text())
    protocol, output = meta["protocol"], root / "full"
    assert all(p.sha(p.ROOT / name) == value for name, value in {**meta["source_sha256"], **meta["preserved_inputs_sha256"]}.items())
    with tarfile.open(root / "source_snapshot.tar.gz") as archive:
        assert all(hashlib.sha256(archive.extractfile(name).read()).hexdigest() == value for name, value in meta["source_sha256"].items())
    assert p.sha(root / "preflight_receipt.json") == meta["preflight_sha256"]
    prepared = audit_prepared(root, protocol)
    tasks = p.jobs(protocol)
    assert len(tasks) == 1160 and json.loads((output / "stage_manifest.json").read_text())["jobs"] == [list(job) for job in tasks]
    groups, receipts, missing = common.collect(output, tasks, meta["source_sha256"])
    assert not missing
    pids = []
    for job, saved in groups.items():
        folder = old.job_folder(output, job)
        arrays, _ = old.load_inputs(output, *job[:2])
        operation = json.loads((folder / "worker_operation.json").read_text())
        assert operation["status"] == "completed" and operation["fresh_process_for_one_job"]
        assert operation["source_sha256"] == meta["source_sha256"]
        pids.append((operation["pid"], operation["started_utc"]))
        assert saved["execution_origin"] == "new_repeated_holdout_single_job_process"
        if job[3] == "reference":
            assert {row["method"] for row in saved["trials"]} == {"raw", "pca7"}
            continue
        grid = [(g, s, c) for g in range(3) for s in ([0.] if job[3] == "baseline" else [0., *protocol["lambdas"]]) for c in protocol["C_grid"]]
        assert [(row["geometry"], row["lambda"], row["C"]) for row in saved["trials"]] == grid
        lock = json.loads((folder / "parameter_lock.json").read_text())
        with np.load(folder / "exact_margins.npz", allow_pickle=False) as margins:
            for role, row in saved["choices"].items():
                proxy = old.select_trial(saved["trials"], role == "proposed")
                assert lock["choices"][role] == proxy
                assert all(row[k] == proxy[k] for k in ("lambda", "C", "geometry"))
                assert row["fit_rows"] == len(arrays["train"])
                assert row["final_exact"] and row["final_model_fit_calls"] == row["ensemble_members"] == 1
                assert row["threshold"] == p.base.threshold_for_cost(arrays["y_tune"], margins[role + "_tune"], 0.)[0]
                assert len(margins[role + "_gate"]) == len(arrays["gate"])
    assert len(pids) == len(set(pids)) == 1160
    lock = json.loads((output / "selection_lock.json").read_text())
    assert lock["candidate_receipts_sha256"] == receipts
    assert lock["choices"] == common.sensitivity_choices(output, report.selected_choices(groups, protocol["pca_components"]))
    assert lock["lambda_profiles"] == common.sensitivity_choices(output, common.lambda_profiles(groups, protocol))
    assert json.loads((output / "test_release.json").read_text())["selection_lock_sha256"] == p.sha(output / "selection_lock.json")
    expected, predictions = report.evaluate(output, lock["choices"])
    actual = pd.read_csv(output / "seed_metrics.csv", float_precision="round_trip")
    for name in ("cost", "fn", "fp", "tp", "tn", "f1", "recall", "balanced_accuracy", "threshold"):
        np.testing.assert_allclose(expected[name], actual[name], atol=1e-11, rtol=1e-11)
    saved_predictions = pd.read_csv(output / "predictions.csv.gz", float_precision="round_trip")
    for name in ("dataset", "seed", "kernel", "family", "arm", "role", "row_id", "y", "prediction"):
        np.testing.assert_array_equal(predictions[name], saved_predictions[name])
    np.testing.assert_allclose(predictions.score, saved_predictions.score, atol=1e-11, rtol=1e-11)
    paired, _ = report.contrasts(expected, protocol)
    saved = pd.read_csv(output / "paired_cost.csv", float_precision="round_trip")
    assert len(saved) == 184 and saved.holm_family_size.eq(184).all()
    for name in ("mean_difference", "sd_difference", "ci_low", "ci_high", "p_value", "p_holm", "variance_multiplier"):
        np.testing.assert_allclose(paired[name], saved[name], atol=1e-11, rtol=1e-11, equal_nan=True)
    receipt = json.loads((output / "verification_receipt.json").read_text())
    assert all(p.sha(output / name) == value for name, value in receipt["outputs_sha256"].items())
    result = {"status": "passed", "jobs": 1160, "prepared_pairs": 40, "nested_pca_fits": 1000,
        "new_worker_processes": 1160, "contrasts": 184, "split_and_oof_audit": prepared,
        "scope": "Frozen sources, fresh teacher records, varying test splits, full grids/rows, locks, same-fit predictions and recomputed metrics/statistics; not independent retraining",
        "source_sha256": p.sha(__file__)}
    p.write_json(root / "validation_receipt.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_pca7_repeated_20261020")
    audit(parser.parse_args().run_id)
