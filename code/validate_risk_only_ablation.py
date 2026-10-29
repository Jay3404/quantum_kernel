"""Independent saved-model/score/statistics checks for risk-only controls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import t


def sha(path):
    with Path(path).open("rb") as handle:return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    base = Path(__file__).resolve().parents[1]
    root = base / "code/results/cmc_risk_only_ablation_20261023"
    read = lambda path: json.loads(path.read_text())
    meta = read(root / "metadata.json")
    assert meta["status"] == "completed" and meta["jobs_completed"] == 160 and not meta["failures"]
    for name, digest in {**meta["sources"], **meta["inputs"]}.items():assert sha(base / name) == digest, name
    assert read(root / "validation_receipt.json")["status"] == "passed"
    plan = read(root / "parameter_plan.json")
    locked = read(root / "selection_lock.json")
    release = read(root / "test_release.json")
    assert sha(root / "selection_lock.json") == release["selection_lock_sha256"]
    timestamp = lambda value: datetime.fromisoformat(value)
    assert timestamp(plan["utc"]) < timestamp(locked["utc"]) <= timestamp(release["utc"])
    locked_jobs = {tuple(x["job"]): x for x in locked["models_and_validation_thresholds"]}
    assert len(locked_jobs) == len(plan["jobs"]) == 160
    parent = base / "code/results" / meta["config"]["parent_run"] / "full"
    label_parts = []
    for chunk in pd.read_csv(parent / "predictions.csv.gz", chunksize=100000, usecols=["dataset", "seed", "row_id", "y", "kernel", "family", "arm", "role"]):
        selected = chunk.kernel.eq("QSVC") & chunk.family.eq("baseline") & chunk.arm.eq("normal") & chunk.role.eq("baseline")
        label_parts.append(chunk.loc[selected, ["dataset", "seed", "row_id", "y"]])
    labels = pd.concat(label_parts, ignore_index=True)
    assert not labels.duplicated(["dataset", "seed", "row_id"]).any()
    metrics = pd.read_csv(root / "seed_metrics.csv", float_precision="round_trip")
    statistics = pd.read_csv(root / "paired_statistics.csv", float_precision="round_trip")
    assert len(metrics) == 160 and len(statistics) == 16
    for entry in plan["jobs"]:
        d, s, k, m = job = tuple(entry["job"])
        folder = root / "models" / d / f"seed_{s:04d}" / k / m
        complete = read(folder / "complete.json")
        assert sha(folder / "complete.json") == locked_jobs[job]["sha256"]
        assert timestamp(complete["utc"]) < timestamp(locked["utc"])
        assert timestamp(plan["utc"]) < timestamp(read(folder / "started.json")["utc"])
        for name, digest in complete["sha256"].items():assert sha(folder / name) == digest
        chosen = read(folder / "parameter_lock.json")
        actual = parent / "candidates" / d / f"seed_{s:04d}" / k / m / "normal/complete.json"
        assert sha(actual) == entry["parent_complete_sha256"]
        parent_choice = read(actual)["choices"]["proposed"]
        assert all(parent_choice[name] == entry[name] == chosen[name] for name in ("C", "lambda", "geometry"))
        assert chosen["model_fit_calls"] == 1 and chosen["test_labels_read"] is False
        model = joblib.load(folder / "model.joblib")
        prepared = parent / "prepared" / d / f"seed_{s:04d}"
        split = read(prepared / "split.json")["actual"]
        y = labels[labels.dataset.eq(d) & labels.seed.eq(s)].set_index("row_id").reindex(split["test"]).y.to_numpy(dtype=int)
        assert model.fit_status_ == 0 and model.shape_fit_[0] == chosen["fit_rows"] == len(split["train"])
        assert model.C == parent_choice["C"]
        with np.load(prepared / "inputs.npz", allow_pickle=False) as inputs, np.load(folder / "margins.npz", allow_pickle=False) as scores:
            q = inputs["gbdt_rank_gate"]
            strength = parent_choice["lambda"]
            if m == "gbdt_soft_gate":
                norm = np.hypot(1 - q, q)
                x = np.column_stack([np.full(len(q), np.sqrt(1 - strength)), np.sqrt(strength) * (1 - q) / norm, np.sqrt(strength) * q / norm])
            elif k == "QSVC":
                angle = np.pi * strength * (q - .5)
                x = np.column_stack([np.ones(len(q)), np.cos(angle), np.sin(angle)]) / np.sqrt(2)
            else:x = (np.pi * strength * (q - .5))[:, None]
            np.testing.assert_allclose(model.decision_function(x), scores["gate"], atol=1e-7, rtol=1e-7)
            prediction = scores["gate"] >= chosen["threshold"]
            fn = int(((y == 1) & ~prediction).sum())
            fp = int(((y == 0) & prediction).sum())
            row = metrics[metrics.dataset.eq(d) & metrics.seed.eq(s) & metrics.kernel.eq(k) & metrics.method.eq(m)]
            assert len(row) == 1 and row.fn.iloc[0] == fn and row.fp.iloc[0] == fp and row.cost.iloc[0] == 5 * fn + fp
    for row in statistics.itertuples():
        group = metrics[metrics.dataset.eq(row.dataset) & metrics.kernel.eq(row.kernel) & metrics.method.eq(row.method)].sort_values("seed")
        assert group.seed.tolist() == list(range(10, 101, 10))
        delta = (group.cost - group.full_product_cost).to_numpy()
        se = np.sqrt(.225) * delta.std(ddof=1)
        expected = {"mean_difference": delta.mean(), "sd_difference": delta.std(ddof=1),
            "ci_low": delta.mean() - t.ppf(.975, 9) * se if se else np.nan,
            "ci_high": delta.mean() + t.ppf(.975, 9) * se if se else np.nan,
            "p_value": float(2 * t.sf(abs(delta.mean() / se), 9)) if se else 1.}
        for key, value in expected.items():assert np.isclose(value, getattr(row, key), equal_nan=True)
    order = np.argsort(statistics.p_value.to_numpy(), kind="stable")
    adjusted = np.minimum(1., np.maximum.accumulate(statistics.p_value.to_numpy()[order] * np.arange(16, 0, -1)))
    np.testing.assert_allclose(adjusted, statistics.p_holm16_exploratory.to_numpy()[order])
    receipt = {"utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "models_checked": 160,
        "parent_parameter_matches": 160, "independent_test_FN_FP_cost_recomputations": 160,
        "independent_feature_formula_margin_replays": 160, "contrasts_recomputed": 16,
        "all_new_exploratory_holm16_verified": True, "new_model_fits_by_validator": 0,
        "old_frozen_inputs_unchanged": True, "all_models_locked_before_new_test_evaluation": True,
        "validator_sha256": sha(Path(__file__)), "scope": "Saved-model and score/statistics validation, not independent retraining or external confirmation"}
    with (root / "independent_validation_receipt.json").open("x") as handle:json.dump(receipt, handle, indent=2)
    print(json.dumps(receipt))


if __name__ == "__main__":main()
