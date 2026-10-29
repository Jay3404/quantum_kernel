"""Audit frozen overnight inputs, complete grids, selections and same-fit results."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
from _shared import controlled as base
from _shared.improved import sha
from _shared.overnight_risk import ROOT, load_inputs, prepared_folder, job_folder, select_trial
from _shared.overnight_reporting import collect, selected_choices, append_classical_best, sensitivity_choices, lambda_profiles, evaluate
from _shared.full_data import write_json


def main(run_id):
    root = ROOT / "code/results" / run_id
    metadata = json.loads((root / "metadata.json").read_text())
    protocol, hashes = metadata["protocol"], metadata["source_sha256"]
    for name, value in {**hashes, **metadata["preserved_inputs_sha256"]}.items():
        assert sha(ROOT / name) == value, name
    with tarfile.open(root / "source_snapshot.tar.gz") as archive:
        for name, value in hashes.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == value, name
    result = {"status": "passed", "scope": "Frozen inputs, grids, tune-only locks, exact final scores/metrics and previous-result preservation; not independent whole-model retraining", "stages": {}, "source_sha256": sha(Path(__file__))}
    for stage in ("discovery", "confirmation", "full"):
        output = root / stage
        if not (output / "stage_manifest.json").exists():
            continue
        plan = json.loads((output / "stage_manifest.json").read_text())
        tasks = [tuple(job) for job in plan["jobs"]]
        groups, receipts, missing = collect(output, tasks, hashes)
        pairs = sorted({job[:2] for job in groups})
        for dataset, seed in pairs:
            arrays, audit = load_inputs(output, dataset, seed)
            folder = prepared_folder(output, dataset, seed)
            assert sha(folder / "split.json") == audit["split_sha256"]
            split = json.loads((folder / "split.json").read_text())
            actual, outer = split["actual"], split["outer"]
            assert len(actual["train"]) == len(arrays["train"]) == audit["train_rows"]
            assert audit["training_row_ids"] == actual["train"]
            assert not set(actual["train"]) & set(actual["validation"])
            assert not set(actual["train"]) & set(actual["test"])
            assert not set(actual["validation"]) & set(actual["test"])
            if stage == "full":
                assert actual == outer
                assert len(arrays["train"]) == round(.8 * sum(map(len, outer.values())))
                assert not (folder / "confirmation_labels.npz").exists()
            else:
                assert set(sum(actual.values(), [])) <= set(outer["train"])
                assert not set(sum(actual.values(), [])) & set(outer["validation"] + outer["test"])
            for part in ("train", "tune", "gate"):
                reference = np.sort(arrays["risk_train"])
                risk = arrays[f"risk_{part}"]
                rank = (np.searchsorted(reference, risk, "left") + np.searchsorted(reference, risk, "right")) / (2 * len(reference))
                np.testing.assert_array_equal(arrays[f"rank_{part}"], rank)
        trials = 0
        for job, saved in groups.items():
            rows = saved["trials"]
            arrays, _ = load_inputs(output, *job[:2])
            trials += len(rows)
            assert all(r["fit_rows"] == len(arrays["train"]) for r in rows)
            if job[3] != "reference":
                strengths = [0.] if job[3] == "baseline" else [0., *(protocol["lambdas"] if stage == "full" else protocol["smoke_lambdas"])]
                expected = {(s, c, g) for s in strengths for c in protocol["C_grid"] for g in range(len(protocol["geometries"]))}
                assert {(r["lambda"], r["C"], r["geometry"]) for r in rows} == expected
                assert len(rows) == len(expected)
                assert all(r["model_fit_calls"] == r["ensemble_members"] == 1 for r in rows)
                folder = job_folder(output, job)
                lock = json.loads((folder / "parameter_lock.json").read_text())
                with np.load(folder / "exact_margins.npz", allow_pickle=False) as margins:
                    for role, row in saved["choices"].items():
                        proxy = select_trial(rows, role == "proposed")
                        assert proxy == lock["choices"][role]
                        assert all(row[k] == proxy[k] for k in ("lambda", "C", "geometry"))
                        threshold, _ = base.threshold_for_cost(arrays["y_tune"], margins[role + "_tune"], 0.)
                        assert threshold == row["threshold"]
                        assert row["final_exact"] and row["final_model_fit_calls"] == 1
                        assert base.metrics(arrays["y_tune"], margins[role + "_tune"], threshold)["cost"] == row["tune_cost"]
        if (output / "selection_lock.json").exists():
            lock = json.loads((output / "selection_lock.json").read_text())
            choices = append_classical_best(selected_choices(groups))
            profiles = []
            if stage == "full":
                choices = sensitivity_choices(output, choices)
                profiles = sensitivity_choices(output, lambda_profiles(groups, protocol))
            assert choices == lock["choices"] and profiles == lock["lambda_profiles"]
            assert receipts == lock["candidate_receipts_sha256"] and missing == lock["missing_jobs"]
            if stage != "discovery" and (output / "seed_metrics.csv").exists():
                rebuilt, predictions = evaluate(output, choices, stage == "full", protocol)
                stored = pd.read_csv(output / "seed_metrics.csv", float_precision="round_trip")
                for column in ("cost", "fn", "fp", "tp", "tn", "f1", "recall", "balanced_accuracy", "threshold"):
                    np.testing.assert_allclose(rebuilt[column], stored[column], atol=1e-11, rtol=1e-11)
                if stage == "full":
                    actual = pd.read_csv(output / "predictions.csv.gz", float_precision="round_trip")
                    expected = pd.concat(predictions, ignore_index=True)
                    for column in ("row_id", "y", "prediction"):
                        np.testing.assert_array_equal(actual[column], expected[column])
                    np.testing.assert_allclose(actual.score, expected.score, atol=1e-11, rtol=1e-11)
                    receipt = json.loads((output / "verification_receipt.json").read_text())
                    assert all(sha(output / name) == value for name, value in receipt["outputs_sha256"].items())
        result["stages"][stage] = {"jobs": len(tasks), "complete_jobs": len(groups), "missing_jobs": len(missing), "candidate_trials": trials, "prepared_pairs": len(pairs)}
    write_json(root / "validation_receipt.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_20261018_overnight_risk_v2")
    main(parser.parse_args().run_id)
