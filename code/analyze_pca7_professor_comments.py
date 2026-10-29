"""Read-only, post-hoc protocol/threshold/teacher/kernel audit of locked PCA7."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/cmc_professor_protocol_audit_20261021.json"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_inputs(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.shape != y.shape or not len(y):
        raise ValueError("Expected nonempty aligned vectors")
    if not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected finite scores and binary labels")
    return y.astype(int), scores


def threshold(y, scores, objective="cost"):
    y, scores = binary_inputs(y, scores)
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(y) - 1]
    candidates = np.r_[np.nextafter(ordered[0], np.inf), ordered[ends]]
    tp = np.r_[0, np.cumsum(y[order])[ends]]
    fp = np.r_[0, np.cumsum(1 - y[order])[ends]]
    fn = y.sum() - tp
    if objective == "cost":
        value = 5 * fn + fp
        best = np.flatnonzero(value == value.min())
    elif objective == "f1":
        value = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=2 * tp + fp + fn != 0)
        best = np.flatnonzero(value == value.max())
    else:
        raise ValueError("Unknown threshold objective")
    return float(candidates[best[np.argmin(np.abs(candidates[best]))]])


def metrics(y, scores, cut):
    y, scores = binary_inputs(y, scores)
    predicted = scores >= cut
    tp, fp = int(np.sum((y == 1) & predicted)), int(np.sum((y == 0) & predicted))
    fn, tn = int(np.sum((y == 1) & ~predicted)), int(np.sum((y == 0) & ~predicted))
    recall = tp / (tp + fn) if tp + fn else 0.
    specificity = tn / (tn + fp) if tn + fp else 0.
    return {"cost": 5 * fn + fp, "cost_per_customer": (5 * fn + fp) / len(y),
        "fn": fn, "fp": fp, "tn": tn, "tp": tp,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        "recall": recall, "balanced_accuracy": (recall + specificity) / 2,
        "auroc": float(roc_auc_score(y, scores)), "ap": float(average_precision_score(y, scores))}


def probability_metrics(y, probability, bins=10):
    y, probability = binary_inputs(y, probability)
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Calibration diagnostics need probabilities, not risk ranks")
    index = np.minimum((probability * bins).astype(int), bins - 1)
    rows, ece = [], 0.
    for i in range(bins):
        selected = index == i
        count = int(selected.sum())
        mean = float(probability[selected].mean()) if count else np.nan
        rate = float(y[selected].mean()) if count else np.nan
        ece += count / len(y) * abs(mean - rate) if count else 0.
        rows.append({"bin": i, "lower": i / bins, "upper": (i + 1) / bins,
            "n": count, "mean_prediction": mean, "observed_rate": rate})
    return {"auroc": float(roc_auc_score(y, probability)), "ap": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)), "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "ece10_uniform": float(ece), "mean_prediction": float(probability.mean()), "prevalence": float(y.mean())}, rows


def corrected(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError("Exactly ten finite paired differences are required")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    se = np.sqrt(.225) * sd
    half = stats.t.ppf(.975, 9) * se if se else np.nan
    return {"mean_difference": mean, "sd_difference": sd, "ci_low": mean - half,
        "ci_high": mean + half, "p_value": float(2 * stats.t.sf(abs(mean / se), 9)) if se else 1.,
        "wins": int((values > 0).sum()), "ties": int((values == 0).sum()), "losses": int((values < 0).sum()),
        "variance_multiplier": .225, "df": 9, "n_seeds": 10, "degenerate_variance": se == 0}


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    result = np.empty_like(values)
    result[order] = np.minimum(1., np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return result


def risk_factor(q, strength, learner, method, gamma=1.):
    q = np.asarray(q, dtype=float)
    difference = q[:, None] - q[None, :]
    if method == "gbdt_soft_gate":
        gate = np.column_stack([1 - q, q])
        gate /= np.linalg.norm(gate, axis=1, keepdims=True)
        return 1 - strength + strength * (gate @ gate.T)
    if method != "gbdt_rank_append":
        raise ValueError("Expected one of the two primary methods")
    return np.cos(np.pi * strength * difference / 2) ** 2 if learner == "QSVC" else np.exp(-gamma * (np.pi * strength * difference) ** 2)


def kernel_metrics(kernel, labels):
    kernel = (np.asarray(kernel) + np.asarray(kernel).T) / 2
    labels = np.asarray(labels)
    if kernel.shape != (len(labels), len(labels)) or not np.isfinite(kernel).all():
        raise ValueError("Invalid aligned Gram matrix")
    if not np.allclose(np.diag(kernel), 1., atol=1e-8):
        raise AssertionError("Unit diagonal expected")
    eigenvalues = np.linalg.eigvalsh(kernel)
    if eigenvalues.min() < -1e-7:
        raise AssertionError("Saved kernel is not PSD within tolerance")
    eigenvalues = np.maximum(eigenvalues, 0)
    mass = eigenvalues[eigenvalues > 1e-12] / eigenvalues.sum()
    centered = kernel - kernel.mean(0) - kernel.mean(1)[:, None] + kernel.mean()
    target = np.outer(labels - labels.mean(), labels - labels.mean())
    alignment = np.sum(centered * target) / max(np.linalg.norm(centered) * np.linalg.norm(target), 1e-15)
    off = ~np.eye(len(labels), dtype=bool)
    within = (labels[:, None] == labels[None, :]) & off
    cross = labels[:, None] != labels[None, :]
    return {"n_geometry": len(labels), "centered_label_alignment": float(alignment),
        "entropy_effective_rank": float(np.exp(-np.sum(mass * np.log(mass)))),
        "participation_ratio": float(np.trace(kernel) ** 2 / np.sum(kernel**2)),
        "centered_participation_ratio": float(np.trace(centered) ** 2 / max(np.sum(centered**2), 1e-15)),
        "offdiagonal_mean": float(kernel[off].mean()), "within_class_mean": float(kernel[within].mean()),
        "cross_class_mean": float(kernel[cross].mean()), "minimum_eigenvalue": float(eigenvalues.min())}


def backbone(arrays, audit, positions, learner, geometry):
    encoder = audit["encoder"]
    train = np.pi * (arrays["train"] - encoder["center"]) / encoder["scale"]
    values = np.pi * (arrays["gate"][positions] - encoder["center"]) / encoder["scale"]
    gamma = geometry["gamma_factor"] / max(7 * train.var(), 1e-12)
    if learner == "RBF-SVM":
        squared = np.maximum(np.sum(values**2, 1)[:, None] + np.sum(values**2, 1)[None, :] - 2 * values @ values.T, 0)
        return np.exp(-gamma * squared), gamma
    from qiskit.circuit.library import zz_feature_map
    from qiskit.quantum_info import Statevector
    circuit = zz_feature_map(7, reps=geometry["reps"], entanglement="linear", alpha=geometry["alpha"])
    parameters = tuple(circuit.parameters)
    states = np.asarray([Statevector.from_instruction(circuit.assign_parameters(dict(zip(parameters, row)))).data for row in values])
    return np.clip(abs(states @ states.conj().T) ** 2, 0, 1), gamma


def audit_oof(folder, arrays, consume):
    split = json.loads(consume(folder / "split.json").read_text())["actual"]
    record = json.loads(consume(folder / "oof.json").read_text())
    training, validation, test = (set(split[k]) for k in ("train", "validation", "test"))
    assert not training & (validation | test) and not validation & test
    for key in ("folds", "raw_teacher_folds"):
        seen = []
        for fold in record[key]:
            fit, held = set(fold["train_ids"]), set(fold["heldout_ids"])
            assert not fit & held and fit | held == training
            seen.extend(fold["heldout_ids"])
            calibration = fold.get("calibration_folds", fold.get("raw_lr_calibration_folds"))
            for inner in calibration:
                a, b = set(inner["train_ids"]), set(inner["heldout_ids"])
                assert not a & b and a | b == fit
        assert len(seen) == len(training) and set(seen) == training
    assert len(arrays["gbdt_train"]) == len(training)
    return {"train_rows": len(training), "validation_rows": len(validation), "test_rows": len(test),
        "outer_oof_folds": 5, "nested_lr_calibration_folds": 3, "overlap_violations": 0,
        "test_ids": split["test"]}


def run(output):
    tick = time.perf_counter()
    protocol = json.loads(PROTOCOL.read_text())
    parent = ROOT / "code/results" / protocol["upstream_run"]
    full = parent / "full"
    consumed = {}

    def consume(path):
        path = Path(path)
        key = str(path.relative_to(ROOT))
        digest = sha(path)
        assert key not in consumed or consumed[key] == digest, "Input changed during analysis"
        consumed[key] = digest
        return path

    consume(PROTOCOL)
    consume(Path(__file__))
    meta = json.loads(consume(parent / "metadata.json").read_text())
    receipt = json.loads(consume(parent / "validation_receipt.json").read_text())
    assert meta["status"] == "completed" and meta["completed_jobs"] == 1160 and not meta["failed_jobs"]
    assert receipt["status"] == "passed" and receipt["jobs"] == 1160
    for name, digest in {**meta["source_sha256"], **meta["preserved_inputs_sha256"]}.items():
        assert sha(consume(ROOT / name)) == digest, name
    consume(full / "selection_lock.json")
    consume(full / "test_release.json")
    consume(full / "verification_receipt.json")
    raw = pd.read_csv(consume(full / "seed_metrics.csv"))
    original_paired = pd.read_csv(consume(full / "paired_cost.csv"))
    old_raw = pd.read_csv(consume(ROOT / "code/results/manuscript_main_raw.csv"))
    old_summary = pd.read_csv(consume(ROOT / "code/results/manuscript_main_summary.csv"))
    for name in ("code/_shared/pipeline.py", "code/german_numeric/config.py", "code/south_german/config.py"):
        consume(ROOT / name)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    label_parts = []
    prediction_file = consume(full / "predictions.csv.gz")
    for chunk in pd.read_csv(prediction_file, chunksize=100000):
        chosen = chunk.kernel.eq("QSVC") & chunk.family.eq("baseline") & chunk.arm.eq("normal") & chunk.role.eq("baseline")
        label_parts.append(chunk.loc[chosen, ["dataset", "seed", "row_id", "y"]])
    labels = pd.concat(label_parts, ignore_index=True)
    assert len(labels.groupby(["dataset", "seed"])) == 40
    threshold_rows, teacher_rows, calibration_rows, geometry_rows, fold_rows, identities = [], [], [], [], [], []
    methods = ["baseline", *protocol["primary_methods"], *protocol["ablations"]]
    for dataset in protocol["datasets"]:
        for seed in protocol["seeds"]:
            folder = full / "prepared" / dataset / f"seed_{seed:04d}"
            audit = json.loads(consume(folder / "audit.json").read_text())
            npz = consume(folder / "inputs.npz")
            assert sha(npz) == audit["input_sha256"]
            with np.load(npz, allow_pickle=False) as archive:
                arrays = dict(archive)
            fold = audit_oof(folder, arrays, consume)
            held = labels[labels.dataset.eq(dataset) & labels.seed.eq(seed)].set_index("row_id")
            assert len(held) == len(fold["test_ids"]) and not held.index.has_duplicates
            y_test = held.reindex(fold.pop("test_ids")).y.to_numpy(dtype=int)
            assert len(y_test) == len(arrays["gate"])
            fold_rows.append({"dataset": dataset, "seed": seed, **fold})
            for teacher, prefix in (("PCA7-LR", "risk"), ("Raw-LR", "raw_lr"), ("Raw-XGB", "gbdt")):
                for part, y in (("train_oof", arrays["y_train"]), ("validation", arrays["y_tune"]), ("test", y_test)):
                    suffix = {"train_oof": "train", "validation": "tune", "test": "gate"}[part]
                    record, bins = probability_metrics(y, arrays[prefix + "_" + suffix])
                    identity = {"dataset": dataset, "seed": seed, "teacher": teacher, "part": part}
                    teacher_rows.append({**identity, **record, "n": len(y)})
                    calibration_rows.extend({**identity, **b} for b in bins)
            backbone_cache = {}
            for learner in protocol["learners"]:
                for method in methods:
                    job_folder = full / "candidates" / dataset / f"seed_{seed:04d}" / learner / method / "normal"
                    complete = json.loads(consume(job_folder / "complete.json").read_text())
                    assert complete["source_sha256"] == meta["source_sha256"] and not complete["held_out_labels_read"]
                    assert complete["input_sha256"] == audit["input_sha256"]
                    role = "baseline" if method == "baseline" else "proposed"
                    choice = complete["choices"][role]
                    margins = consume(job_folder / "exact_margins.npz")
                    assert sha(margins) == complete["exact_margins_sha256"]
                    with np.load(margins, allow_pickle=False) as archive:
                        valid_score, test_score = archive[role + "_tune"], archive[role + "_gate"]
                    cuts = {"zero": 0., "validation_f1": threshold(arrays["y_tune"], valid_score, "f1"),
                        "validation_cost": threshold(arrays["y_tune"], valid_score)}
                    assert np.isclose(cuts["validation_cost"], choice["threshold"], atol=1e-12, rtol=0)
                    selected = raw[raw.dataset.eq(dataset) & raw.seed.eq(seed) & raw.kernel.eq(learner) & raw.family.eq(method) & raw.arm.eq("normal") & raw.role.eq(role)]
                    assert len(selected) == 1
                    assert metrics(y_test, test_score, cuts["validation_cost"])["cost"] == int(selected.cost.iloc[0])
                    for policy, cut in cuts.items():
                        for part, y, score in (("validation", arrays["y_tune"], valid_score), ("test", y_test, test_score)):
                            threshold_rows.append({"dataset": dataset, "seed": seed, "learner": learner, "method": method,
                                "threshold_policy": policy, "part": part, "threshold": cut, "n": len(y),
                                "lambda": choice["lambda"], "C": choice["C"], "geometry": choice["geometry"],
                                "fit_rows": choice["fit_rows"], "model_selection_unchanged": True, **metrics(y, score, cut)})
                    if dataset != "hmeq" or method not in ["baseline", *protocol["primary_methods"]]:
                        continue
                    geometry_file = consume(job_folder / "geometry.npz")
                    assert sha(geometry_file) == complete["geometry_sha256"]
                    key = "exact_" + role
                    positions = np.asarray(complete["geometry_positions"][key])
                    with np.load(geometry_file, allow_pickle=False) as archive:
                        gram = archive[key]
                    geometry = meta["protocol"]["geometries"][choice["geometry"]]
                    cache_key = (learner, choice["geometry"], tuple(positions))
                    if cache_key not in backbone_cache:
                        backbone_cache[cache_key] = backbone(arrays, audit, positions, learner, geometry)
                    matched, gamma = backbone_cache[cache_key]
                    factor = np.ones_like(matched) if method == "baseline" else risk_factor(arrays["gbdt_rank_gate"][positions], choice["lambda"], learner, method, gamma)
                    error = float(np.max(abs(gram - matched * factor)))
                    np.testing.assert_allclose(gram, matched * factor, atol=1e-8, rtol=1e-8)
                    identity = {"dataset": dataset, "seed": seed, "learner": learner, "method": method,
                        "lambda": choice["lambda"], "geometry": choice["geometry"], "positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
                        "part": "test_subset", "product_identity_max_error": error}
                    identities.append(identity)
                    geometry_rows.append({**identity, "kernel_variant": "selected_model", **kernel_metrics(gram, y_test[positions])})
                    if method != "baseline":
                        geometry_rows.append({**identity, "kernel_variant": "same_geometry_no_risk", **kernel_metrics(matched, y_test[positions])})
            print(f"analyzed {dataset} seed={seed}", flush=True)
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(output / "threshold_seed_metrics.csv", index=False)
    pd.DataFrame(teacher_rows).to_csv(output / "teacher_seed_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output / "teacher_reliability_bins.csv", index=False)
    pd.DataFrame(geometry_rows).to_csv(output / "hmeq_geometry_seed_metrics.csv", index=False)
    pd.DataFrame(identities).to_csv(output / "kernel_product_identity_checks.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "oof_integrity_checks.csv", index=False)
    test = thresholds[thresholds.part.eq("test")]
    paired, directions = [], []
    for identity, group in test.groupby(["dataset", "learner", "method"]):
        costs = group.pivot(index="seed", columns="threshold_policy", values="cost").reindex(protocol["seeds"])
        for policy in ("zero", "validation_f1"):
            delta = (costs[policy] - costs.validation_cost).to_numpy()
            common = dict(zip(["dataset", "learner", "method"], identity))
            paired.append({**common, "contrast": policy + "_minus_cost", **corrected(delta), "family_size": 80})
            directions.extend({**common, "contrast": policy + "_minus_cost", "seed": seed, "delta": float(value)} for seed, value in zip(protocol["seeds"], delta))
    paired = pd.DataFrame(paired)
    assert len(paired) == 80
    paired["p_holm80"] = holm(paired.p_value)
    paired.to_csv(output / "threshold_paired_cost.csv", index=False)
    pd.DataFrame(directions).to_csv(output / "threshold_seed_differences.csv", index=False)
    original_paired.to_csv(output / "original184_paired_cost_unchanged.csv", index=False)
    bridge = []
    for dataset in ("german_numeric", "south_german"):
        before = old_raw[old_raw.dataset.eq(dataset) & old_raw.experiment.eq("baseline")]
        expected = old_summary[old_summary.dataset.eq(dataset) & old_summary.experiment.eq("baseline")]["Business Cost_mean"].iloc[0]
        assert len(before) == 5 and np.isclose(before["Business Cost"].mean(), expected)
        for version, values, n, seeds in (("manuscript_archived", before["Business Cost"], 200, "10..50"),
            ("new_pca7_all10", test[test.dataset.eq(dataset) & test.learner.eq("QSVC") & test.method.eq("baseline") & test.threshold_policy.eq("validation_cost")].cost, 100, "10..100"),
            ("new_pca7_common5", test[test.dataset.eq(dataset) & test.learner.eq("QSVC") & test.method.eq("baseline") & test.threshold_policy.eq("validation_cost") & test.seed.le(50)].cost, 100, "10..50")):
            bridge.append({"dataset": dataset, "version": version, "n_test": n, "n_seeds": len(values), "seeds": seeds,
                "cost_mean": float(values.mean()), "cost_sd": float(values.std(ddof=1)),
                "per_customer_cost_mean": float(values.mean()) / n, "cost_per100_customers": float(values.mean()) * 100 / n})
    pd.DataFrame(bridge).to_csv(output / "manuscript_baseline_bridge.csv", index=False)
    for name, digest in consumed.items():
        assert sha(ROOT / name) == digest, f"Read-only input changed:{name}"
    import sklearn, scipy
    outputs = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    result = {"status": "passed", "analysis_id": protocol["analysis_id"], "finished_utc": datetime.now(timezone.utc).isoformat(),
        "verification_status": "ANALYZED_NOT_INDEPENDENT_RETRAINING", "new_svm_fits": 0, "new_teacher_fits": 0,
        "upstream_files_modified": False, "active_pca9_sources_modified": False,
        "n_prepared_pairs": len(fold_rows), "n_locked_models": len(thresholds) // 6, "threshold_seed_rows": len(thresholds),
        "teacher_seed_rows": len(teacher_rows), "geometry_seed_rows": len(geometry_rows), "product_identity_checks": len(identities),
        "maximum_product_identity_error": max(r["product_identity_max_error"] for r in identities),
        "threshold_contrasts": 80, "original_risk_contrasts": len(original_paired), "original_risk_holm_family_preserved": 184,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": time.perf_counter() - tick, "input_sha256": consumed, "output_sha256": outputs}
    (output / "analysis_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("sha256")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "code/results/cmc_pca7_professor_audit_20261021")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
