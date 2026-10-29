"""Simulator-free independent audit of the German ZZ-phase stored experiment."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measurements(y, scores, threshold):
    pred = scores >= threshold
    fn, fp = int(np.sum((y == 1) & ~pred)), int(np.sum((y == 0) & pred))
    return {"cost": 5 * fn + fp, "fn": fn, "fp": fp,
        "tn": int(np.sum((y == 0) & ~pred)), "tp": int(np.sum((y == 1) & pred)),
        "f1": f1_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred), "roc_auc": roc_auc_score(y, scores)}


def choice_key(row):
    return (row["tune_cost"], row["lambda"], row.get("C", 0.), row.get("gamma_factor", 0.),
            row["learner"], row["method"], row.get("scaled", False), str(row.get("depth")))


def main(run_id):
    root = ROOT / "code/results" / run_id
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    protocol = metadata["protocol"]
    assert protocol["datasets"] == ["german_numeric", "south_german"]
    assert protocol["seeds"] == list(range(10, 101, 10))
    assert protocol["lambdas"] == [i / 20 for i in range(1, 21)]
    output = root / "full"
    receipt = json.loads((output / "verification_receipt.json").read_text())
    for name, value in receipt["outputs_sha256"].items():
        assert digest(output / name) == value, name
    for name, value in metadata["preserved_inputs_sha256"].items():
        assert digest(ROOT / name) == value, name
    with tarfile.open(root / "source_snapshot.tar.gz") as archive:
        for name, value in metadata["source_sha256"].items():
            assert digest(ROOT / name) == value, name
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == value, name
    labels, inputs_by_pair, cases = {}, {}, []
    for dataset, seed in itertools.product(protocol["datasets"], protocol["seeds"]):
        source = ROOT / "code" / dataset / "results" / protocol["source_run"] / f"seed_{seed:03d}"
        folder = output / "prepared" / dataset / f"seed_{seed:03d}"
        record = json.loads((folder / "audit.json").read_text())
        assert digest(source / "prepared.npz") == record["source_audit"]["prepared_sha256"]
        assert digest(source / "prepared.json") == record["source_audit"]["source_receipt_sha256"]
        assert digest(source / "split_indices.json") == record["split_indices_sha256"]
        assert digest(folder / "candidate_inputs.npz") == record["candidate_inputs_sha256"]
        data = pd.read_csv(ROOT / "data" / ("german-data-numeric" if dataset == "german_numeric" else "SouthGermanCredit.asc"), sep=r"\s+", header=None if dataset == "german_numeric" else 0)
        y = (data.iloc[:, -1].to_numpy() == (2 if dataset == "german_numeric" else 0)).astype(int)
        train, hold = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
        valid, test = train_test_split(hold, test_size=.5, stratify=y[hold], random_state=seed + 103)
        split = json.loads((source / "split_indices.json").read_text())
        assert split == {"train": train.tolist(), "validation": valid.tolist(), "test": test.tolist()}
        assert not set(train) & set(valid) and not set(train) & set(test) and not set(valid) & set(test)
        with np.load(source / "prepared.npz", allow_pickle=False) as old, np.load(folder / "candidate_inputs.npz", allow_pickle=False) as saved:
            inputs = dict(saved)
            assert not any(k in inputs for k in ("y_test", "y_gate", "test", "risk_test"))
            assert [len(inputs[p]) for p in ("train", "tune", "gate")] == [800, 100, 100]
            for part, original in (("train", "train"), ("tune", "validation"), ("gate", "test")):
                np.testing.assert_array_equal(inputs[part], old[original])
                np.testing.assert_array_equal(inputs[f"risk_{part}"], old[f"risk_{original}"])
                reference = np.sort(old["risk_train"])
                values = inputs[f"risk_{part}"]
                rank = (np.searchsorted(reference, values, "left") + np.searchsorted(reference, values, "right")) / (2 * len(reference))
                np.testing.assert_array_equal(inputs[f"rank_{part}"], rank)
            np.testing.assert_array_equal(inputs["y_train"], y[train])
            np.testing.assert_array_equal(inputs["y_tune"], y[valid])
            np.testing.assert_array_equal(inputs["weights"], old["weights"])
            np.testing.assert_array_equal(record["encoder"]["minimum"], old["train"].min(0))
            np.testing.assert_array_equal(record["encoder"]["span"], np.maximum(np.ptp(old["train"], axis=0), 1e-12))
        inputs_by_pair[dataset, seed] = inputs
        labels[dataset, seed] = y[test], test
        cases.append({"dataset": dataset, "seed": seed, "train": 800, "validation": 100, "test": 100})
    candidates, margins, trial_count = {}, {}, 0
    lock = json.loads((output / "selection_lock.json").read_text())
    for name, value in lock["candidate_receipts_sha256"].items():
        path = ROOT / name
        assert digest(path) == value
        saved = json.loads(path.read_text())
        job = tuple(saved["job"])
        assert saved["source_sha256"] == metadata["source_sha256"] and not saved["held_out_labels_read"]
        archive = path.parent / "margins.npz"
        assert digest(archive) == saved["archive_sha256"]
        with np.load(archive, allow_pickle=False) as scores:
            arrays = dict(scores)
        assert arrays["tune"].shape == arrays["gate"].shape == (len(saved["trials"]), 100)
        assert all(np.isfinite(v).all() for v in arrays.values())
        y = inputs_by_pair[job[:2]]["y_tune"]
        rows = []
        for index, row in enumerate(saved["trials"]):
            assert row["fit_rows"] == 800
            score = arrays["tune"][index]
            prediction = score >= row["threshold"]
            cost = int(5 * np.sum((y == 1) & ~prediction) + np.sum((y == 0) & prediction))
            assert cost == row["tune_cost"]
            thresholds = np.r_[np.nextafter(score.max(), np.inf), np.unique(score)[::-1]]
            candidate_predictions = score[None, :] >= thresholds[:, None]
            costs = 5 * np.sum((y[None, :] == 1) & ~candidate_predictions, axis=1) + np.sum((y[None, :] == 0) & candidate_predictions, axis=1)
            default = 0. if row["learner"] in ("QSVC", "RBF-SVM") else 1 / 6
            optimal = thresholds[costs == costs.min()]
            assert row["threshold"] == optimal[np.argmin(abs(optimal - default))]
            assert cost == costs.min()
            rows.append({**row, "archive": str(archive.relative_to(ROOT)), "archive_sha256": saved["archive_sha256"], "score_index": index})
        candidates[job] = rows
        margins[str(archive.relative_to(ROOT))] = arrays
        trial_count += len(rows)
    assert len(candidates) == 140 and receipt["jobs"] == 140
    raw = pd.read_csv(output / "seed_metrics.csv", float_precision="round_trip")
    profiles = pd.read_csv(output / "lambda_seed_metrics.csv", float_precision="round_trip")
    prediction_table = pd.read_csv(output / "predictions.csv.gz", float_precision="round_trip")
    assert len(lock["models"]) == len(raw) == 260
    assert len(lock["lambda_profiles"]) == len(profiles) == 400
    for is_profile, choices, table in ((False, lock["models"], raw), (True, lock["lambda_profiles"], profiles)):
        for index, row in enumerate(choices):
            dataset, seed, role = row["dataset"], row["seed"], row["role"]
            if is_profile or role in ("baseline", "proposed", "separable", "shuffled"):
                arm = role if role in ("separable", "shuffled") else "quantum"
                pool = candidates[dataset, seed, "zz_phase", arm]
                pool = [r for r in pool if r["lambda"] == row["lambda"]] if is_profile else [r for r in pool if (r["lambda"] == 0 if role == "baseline" else r["lambda"] > 0)]
            elif row["candidate_method"] == "standard_controls" and role.startswith("q_"):
                method = "standard" if role == "q_baseline" else "standard_" + role[2:]
                pool = candidates[dataset, seed, method, "quantum"]
                pool = pool if method == "standard" else [r for r in pool if r["lambda"] > 0]
            else:
                pool = candidates[dataset, seed, "classical", "classical"]
                if role != "classical_best":
                    if row["candidate_method"] == "standard_controls":
                        method = "baseline" if role == "rbf_baseline" else "original_" + role[4:]
                        pool = [r for r in pool if r["method"] == method]
                    else:
                        pool = [r for r in pool if r["learner"] == "RBF-SVM" and (r["lambda"] == 0 if role == "rbf_baseline" else r["lambda"] > 0)]
            best = min(pool, key=choice_key)
            assert all(row[k] == v for k, v in best.items())
            score = margins[row["archive"]]["gate"][row["score_index"]]
            y, ids = labels[dataset, seed]
            for key, value in measurements(y, score, row["threshold"]).items():
                np.testing.assert_allclose(table.iloc[index][key], value, atol=1e-12, rtol=1e-12)
            if not is_profile:
                part = prediction_table[(prediction_table.dataset == dataset) & (prediction_table.seed == seed) & (prediction_table.candidate_method == row["candidate_method"]) & (prediction_table.role == role)]
                np.testing.assert_array_equal(part.row_id, ids)
                np.testing.assert_array_equal(part.y, y)
                np.testing.assert_allclose(part.score, score, atol=1e-12, rtol=1e-12)
                np.testing.assert_array_equal(part.prediction, score >= row["threshold"])
    paired = pd.read_csv(output / "paired_cost.csv")
    assert len(paired) == 12 and paired.holm_family_size.eq(12).all()
    for row in paired.itertuples():
        data = raw[(raw.dataset == row.dataset) & raw.candidate_method.eq("zz_phase")].pivot(index="seed", columns="role", values="cost")
        p, b = data.proposed, data.baseline
        deltas = {"quantum_within": b - p, "quantum_vs_classical": data.classical_best - p,
            "quantum_vs_separable": data.separable - p, "quantum_vs_shuffled": data.shuffled - p,
            "quantum_minus_rbf_gain": (b - p) - (data.rbf_baseline - data.rbf_risk),
            "rbf_within": data.rbf_baseline - data.rbf_risk}
        delta = deltas[row.contrast].to_numpy()
        se = np.sqrt((1 / 10 + 100 / 800) * delta.var(ddof=1))
        half = stats.t.ppf(.975, 9) * se
        expected = [delta.mean(), delta.mean() - half, delta.mean() + half, 2 * stats.t.sf(abs(delta.mean() / se), 9)] if se else [delta.mean(), np.nan, np.nan, np.nan] if delta.mean() else [0., 0., 0., 1.]
        np.testing.assert_allclose([row.mean_difference, row.ci_low, row.ci_high, row.p_value], expected, atol=1e-12, rtol=1e-12, equal_nan=True)
    pvalues = paired.p_value.fillna(1).to_numpy()
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.empty(12)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(pvalues[order] * np.arange(12, 0, -1)))
    adjusted[paired.p_value.isna()] = np.nan
    np.testing.assert_allclose(paired.p_holm, adjusted, atol=1e-12, rtol=1e-12, equal_nan=True)
    result = {"status": "passed", "scope": "Independent full inputs, split/ranks, every validation threshold, locked choices, same-fit predictions/metrics, paired intervals and source/prior integrity; not model retraining",
        "cases": cases, "candidate_jobs": len(candidates), "candidate_trials": trial_count,
        "selected_models": len(raw), "lambda_profiles": len(profiles), "prior_results_unchanged": True,
        "validator_sha256": digest(Path(__file__))}
    (output / "independent_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_20261018_german_zz_phase")
    main(parser.parse_args().run_id)
