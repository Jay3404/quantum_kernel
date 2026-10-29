"""Read-only model audit and recomputed PCA9/PCA7 statistics; no retraining."""
import argparse
import hashlib
import json
import tarfile

import numpy as np
import pandas as pd

from _shared import repeated_holdout_pca9 as p
from _shared import repeated_reporting_pca9 as report
from _shared import overnight_reporting as common
from _shared import overnight_risk as old
from validate_repeated_holdout import audit_prepared as shared_prepared_audit


def audit_prepared(root, protocol):
    result = shared_prepared_audit(root, protocol)
    for dataset in protocol["datasets"]:
        for seed in protocol["seeds"]:
            folder = old.prepared_folder(root / "full", dataset, seed)
            previous = old.prepared_folder(p.UPSTREAM / "full", dataset, seed)
            comparison = json.loads((folder / "pca7_split_comparison.json").read_text())
            assert comparison["status"] == "passed" and comparison["same_saved_split"]
            assert comparison["pca7_split_sha256"] == p.sha(previous / "split.json") == p.sha(folder / "split.json")
            assert comparison["pca7_input_sha256"] == p.sha(previous / "inputs.npz")
            assert comparison["pca9_input_sha256"] == p.sha(folder / "inputs.npz")
            assert not comparison["prior_scores_models_or_search_prefixes_adopted"]
            arrays, record = old.load_inputs(root / "full", dataset, seed)
            parent, _ = old.load_inputs(p.UPSTREAM / "full", dataset, seed)
            assert record["pca_components"] == 9
            for part in p.PARTS:
                np.testing.assert_allclose(arrays[part][:, :7], parent[part], atol=1e-9, rtol=1e-9)
                np.testing.assert_allclose(arrays["raw_" + part], parent["raw_" + part], atol=1e-12, rtol=1e-12)
    result["same_saved_pca7_splits"] = True
    result["pca_components"] = 9
    return result


def audit(run_id):
    root = p.ROOT / "code/results" / run_id
    meta = json.loads((root / "metadata.json").read_text())
    protocol, output = meta["protocol"], root / "full"
    assert protocol["pca_components"] == 9
    assert all(p.sha(p.ROOT / name) == value for name, value in {**meta["source_sha256"], **meta["preserved_inputs_sha256"]}.items())
    with tarfile.open(root / "source_snapshot.tar.gz") as archive:
        assert all(hashlib.sha256(archive.extractfile(name).read()).hexdigest() == value for name, value in meta["source_sha256"].items())
    assert p.sha(root / "preflight_receipt.json") == meta["preflight_sha256"]
    prepared = audit_prepared(root, protocol)
    tasks = p.jobs(protocol)
    assert len(tasks) == 870 and json.loads((output / "stage_manifest.json").read_text())["jobs"] == [list(job) for job in tasks]
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
            assert {row["method"] for row in saved["trials"]} == {"raw", "pca9"}
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
                if job[2] == "QSVC":
                    expected = 10 if job[3] == "gbdt_rank_append" else 9
                    assert row["representation_details"]["circuits"][0]["qubits"] == expected
    assert len(pids) == len(set(pids)) == 870
    lock = json.loads((output / "selection_lock.json").read_text())
    assert lock["candidate_receipts_sha256"] == receipts
    assert lock["choices"] == common.sensitivity_choices(output, report.selected_choices(groups, 9))
    assert lock["lambda_profiles"] == common.sensitivity_choices(output, common.lambda_profiles(groups, protocol))
    assert lock["within_pca9_holm_family_size"] == 138 and lock["dimension_holm_family_size"] == 90
    assert json.loads((output / "test_release.json").read_text())["selection_lock_sha256"] == p.sha(output / "selection_lock.json")
    expected, predictions = report.evaluate(output, lock["choices"])
    actual = pd.read_csv(output / "seed_metrics.csv", float_precision="round_trip")
    assert actual.pca_components.eq(9).all()
    for name in ("cost", "fn", "fp", "tp", "tn", "f1", "recall", "balanced_accuracy", "threshold"):
        np.testing.assert_allclose(expected[name], actual[name], atol=1e-11, rtol=1e-11)
    saved_predictions = pd.read_csv(output / "predictions.csv.gz", float_precision="round_trip")
    for name in ("dataset", "seed", "kernel", "family", "arm", "role", "row_id", "y", "prediction"):
        np.testing.assert_array_equal(predictions[name], saved_predictions[name])
    np.testing.assert_allclose(predictions.score, saved_predictions.score, atol=1e-11, rtol=1e-11)
    paired, _ = report.contrasts(expected, protocol)
    previous = pd.read_csv(p.UPSTREAM / "full/seed_metrics.csv", float_precision="round_trip")
    dimension, _ = report.compare(expected, previous, protocol)
    for filename, frame, family_size in (("paired_cost.csv", paired, 138), ("pca7_pca9_paired_cost.csv", dimension, 90)):
        saved = pd.read_csv(output / filename, float_precision="round_trip")
        assert len(saved) == family_size and saved.holm_family_size.eq(family_size).all()
        for name in ("mean_difference", "sd_difference", "ci_low", "ci_high", "p_value", "p_holm", "variance_multiplier"):
            np.testing.assert_allclose(frame[name], saved[name], atol=1e-11, rtol=1e-11, equal_nan=True)
    receipt = json.loads((output / "verification_receipt.json").read_text())
    assert all(p.sha(output / name) == value for name, value in receipt["outputs_sha256"].items())
    result = {"status": "passed", "jobs": 870, "prepared_pairs": 30, "nested_pca_fits": 750,
        "new_worker_processes": 870, "contrasts": 138, "pca7_pca9_contrasts": 90, "split_and_oof_audit": prepared,
        "scope": "Frozen sources, fresh teachers, identical saved varying-test splits, all grids/rows, global locks, predictions and recomputed metrics/statistics; not independent retraining",
        "source_sha256": p.sha(__file__)}
    p.write_json(root / "validation_receipt.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_pca9_repeated_20261021")
    audit(parser.parse_args().run_id)
