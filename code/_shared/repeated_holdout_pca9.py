"""PCA9 repeated holdout using immutable dimension-generic mathematics."""
import json
import numpy as np

from . import repeated_holdout as shared
from . import overnight_risk as old

ROOT, PARTS, METHODS = shared.ROOT, shared.PARTS, shared.METHODS
PROTOCOL = ROOT / "docs/cmc_pca9_repeated_protocol_20261021.json"
UPSTREAM = ROOT / "code/results/cmc_pca7_repeated_20261020"
base, engine = shared.base, shared.engine
sha, write_json = shared.sha, shared.write_json
now, peak_gib, release_heap = shared.now, shared.peak_gib, shared.release_heap
jobs, assert_can_fit = shared.jobs, shared.assert_can_fit
TrainingPCA, features, risk_estimator = shared.TrainingPCA, shared.features, shared.risk_estimator
split_indices, run_job, exact_fit = shared.split_indices, shared.run_job, shared.exact_fit


def compare_preparation(dataset, seed, protocol, output, upstream=UPSTREAM):
    previous = old.prepared_folder(upstream / "full", dataset, seed)
    current = old.prepared_folder(output, dataset, seed)
    original = json.loads((previous / "split.json").read_text())
    split = json.loads((current / "split.json").read_text())
    assert split == original
    assert sha(current / "split.json") == sha(previous / "split.json")
    arrays, record = old.load_inputs(output, dataset, seed)
    parent, _ = old.load_inputs(upstream / "full", dataset, seed)
    assert arrays["train"].shape[1] == record["pca_components"] == protocol["pca_components"] == 9
    for part in PARTS:
        np.testing.assert_allclose(arrays[part][:, :7], parent[part], atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(arrays["raw_" + part], parent["raw_" + part], atol=1e-12, rtol=1e-12)
    assert not record["prior_preparation_reused"] and record["new_pca_lr_fit"]
    assert all(not value["reused_from_prior_campaign"] for value in record["teachers"].values())
    result = {"status": "passed", "same_saved_split": True, "pca7_split_sha256": sha(previous / "split.json"),
        "pca9_split_sha256": sha(current / "split.json"), "pca7_input_sha256": sha(previous / "inputs.npz"),
        "pca9_input_sha256": sha(current / "inputs.npz"), "first_seven_components_match": True,
        "same_preprocessed_original_features": True, "new_pca_lr_and_raw_teachers": True,
        "prior_scores_models_or_search_prefixes_adopted": False}
    write_json(current / "pca7_split_comparison.json", result)
    return result


def prepare(dataset, seed, protocol, output, upstream=UPSTREAM):
    assert protocol["pca_components"] == 9
    result = shared.prepare(dataset, seed, protocol, output)
    result["pca7_comparison"] = compare_preparation(dataset, seed, protocol, output, upstream)
    return result
