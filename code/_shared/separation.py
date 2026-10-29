"""Isolated, validation-gated risk separation experiments."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from . import controlled as base
from .angle_encoding import AngleEncoding, inject_risk
from .full_data import decision_blocks, dense_kernel, write_json
from .improved import TunedQuantumStates, geometries, local_lock, sha

LEARNERS = ("RBF-SVM", "QSVC")
METHODS = ("baseline", "convex", "feature_specific", "multiply", "phase_shift", "risk_axis")
ROOT = base.ROOT


def load_prepared(dataset, seed, protocol, stage):
    folder = ROOT / "code" / dataset / "results" / protocol["source_run"] / f"seed_{seed:03d}"
    receipt = json.loads((folder / "prepared.json").read_text())
    if sha(folder / "prepared.npz") != receipt["archive_sha256"]:
        raise AssertionError("Source training cache changed")
    parts = ("train", "validation") if stage == "smoke" else ("train", "validation", "test")
    names = [*parts, *(f"risk_{p}" for p in parts), "y_train", "y_validation", "weights"]
    with np.load(folder / "prepared.npz", allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in names}
    tick = time.perf_counter()
    encoder = AngleEncoding.fit(arrays["train"], mode="linear_no_clip")
    encoded = encoder.transform(arrays["train"])
    encoder_fit_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    risk = arrays["risk_train"]
    centered = encoded - encoded.mean(0)
    direction = centered.T @ (risk - risk.mean())
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-12 else np.ones(encoded.shape[1]) / np.sqrt(encoded.shape[1])
    direction_fit_seconds = time.perf_counter() - tick
    context = {"encoder": encoder, "direction": direction, "risk_mean": float(risk.mean()),
               "gamma": 1 / (7 * max(encoded.var(), 1e-12))}
    positions = np.arange(len(risk))
    if stage == "smoke" and len(positions) > protocol["smoke_train_cap"]:
        positions, _ = train_test_split(positions, train_size=protocol["smoke_train_cap"],
                                        stratify=arrays["y_train"], random_state=seed + 919)
        for k in ("train", "risk_train", "y_train"):
            arrays[k] = arrays[k][positions]
    audit = {"source_run": protocol["source_run"], "prepared_sha256": receipt["archive_sha256"],
             "source_receipt_sha256": sha(folder / "prepared.json"), "loaded_keys": names,
             "training_positions": positions.tolist(), "fit_rows": len(positions),
             "full_training_rows": encoder.train_rows, "encoder": encoder.parameters(),
             "direction": direction.tolist(), "risk_mean": context["risk_mean"],
             "test_labels_read": False, "source_times": receipt["times"],
             "encoder_fit_seconds": encoder_fit_seconds, "direction_fit_seconds": direction_fit_seconds}
    return arrays, context, audit


def features(arrays, context, part, method, strength, learner, engine=None):
    if method not in METHODS:
        raise ValueError("Unknown mechanism")
    z, risk = arrays[part], arrays[f"risk_{part}"]
    encoder = context["encoder"]
    if method in ("multiply", "convex", "feature_specific"):
        representation = "feature_specific" if method == "feature_specific" else "convex"
        value = encoder.transform(inject_risk(z, risk, arrays["weights"], representation, strength))
    else:
        value = encoder.transform(z)
        if method == "phase_shift":
            value = value + np.pi * strength * (risk - context["risk_mean"])[:, None] * context["direction"]
    if learner == "QSVC":
        value = engine(value)
        if method == "risk_axis":
            theta = np.pi * strength * risk / 2
            qubit = np.column_stack([np.cos(theta), np.sin(theta)])
            value = np.einsum("ni,nj->nij", value, qubit).reshape(len(value), -1)
    elif method == "risk_axis":
        value = np.column_stack([value, np.pi * strength * risk])
    if not np.isfinite(value).all():
        raise AssertionError("Nonfinite representation")
    return value


def trial_order(row):
    return (row["validation_cost"], row["lambda"], row["C"], row["alpha"], row["entanglement"], row["gamma_factor"])


def choose(rows):
    return min(rows, key=trial_order)


def folder_for(output, unit):
    dataset, seed, learner, method, strength = unit
    return output / dataset / f"seed_{seed:03d}" / learner / method / f"lambda_{round(100 * strength):03d}"


def run_unit(unit, stage, protocol, output, source_hash):
    output = Path(output)
    dataset, seed, learner, method, strength = unit
    folder = folder_for(output, unit)
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"unit": list(unit), "stage": stage, "protocol": protocol, "source_sha256": source_hash}
    with threadpool_limits(limits=1), local_lock(folder / "unit.lock"):
        done = folder / "complete.json"
        if done.exists():
            saved = json.loads(done.read_text())
            if saved["identity"] != identity or saved["archive_sha256"] != sha(folder / "margins.npz"):
                raise AssertionError("Checkpoint identity changed")
            return unit
        arrays, context, audit = load_prepared(dataset, seed, protocol, stage)
        trial_protocol = dict(protocol)
        if stage == "smoke":
            trial_protocol.update(C_grid=protocol["smoke_C_grid"], alphas=protocol["smoke_alphas"], entanglement=["linear"])
        rows, margins = [], {"validation": []}
        if stage == "full":
            margins["test"] = []
        started = time.perf_counter()
        for geometry in geometries(learner, trial_protocol):
            tick = time.perf_counter()
            engine = TunedQuantumStates(geometry["alpha"], geometry["entanglement"]) if learner == "QSVC" else None
            setup = time.perf_counter() - tick
            values, preprocessing = {}, {}
            for part in margins.keys() | {"train"}:
                tick = time.perf_counter()
                values[part] = features(arrays, context, part, method, strength, learner, engine)
                preprocessing[part] = time.perf_counter() - tick
            gamma = context["gamma"] * geometry["gamma_factor"]
            tick = time.perf_counter()
            kernel = dense_kernel(values["train"], learner, gamma, 1024, 10, 24)
            kernel_time = time.perf_counter() - tick
            for c in trial_protocol["C_grid"]:
                tick = time.perf_counter()
                model = SVC(kernel="precomputed", C=c, class_weight="balanced", cache_size=256)
                model.fit(kernel, arrays["y_train"])
                fit_time = time.perf_counter() - tick
                if model.fit_status_:
                    raise RuntimeError("SVC failed to converge")
                prediction_times = {}
                scores = {}
                for part in margins:
                    scores[part], kt, dt = decision_blocks(model, values[part], values["train"], learner, gamma, 256)
                    if not np.isfinite(scores[part]).all():
                        raise AssertionError("Nonfinite margin")
                    margins[part].append(scores[part])
                    prediction_times[f"{part}_kernel_seconds"] = kt
                    prediction_times[f"{part}_decision_seconds"] = dt
                threshold, cost = base.threshold_for_cost(arrays["y_validation"], scores["validation"], 0.)
                metrics = base.metrics(arrays["y_validation"], scores["validation"], threshold)
                yv = arrays["y_validation"]
                tn = int(np.sum((yv == 0) & (scores["validation"] < threshold)))
                source_times = audit["source_times"]
                source_training = source_times["feature_fit_seconds"] + (source_times["risk_fit_seconds"] if method != "baseline" else 0.)
                source_training += audit.get("encoder_fit_seconds", 0.)
                source_training += audit.get("direction_fit_seconds", 0.) if method == "phase_shift" else 0.
                row = {"C": c, "lambda": strength, **geometry, "threshold": threshold, "validation_cost": cost,
                       **{f"validation_{k}": v for k, v in metrics.items()},
                       "validation_specificity": tn / max(int(np.sum(yv == 0)), 1),
                       "fit_rows": len(arrays["y_train"]), "model_fit_calls": 1, "ensemble_members": 1,
                       "n_support": int(model.n_support_.sum()), "fit_seconds": fit_time,
                       "train_kernel_seconds": kernel_time, "state_and_angle_train_seconds": preprocessing["train"],
                       "circuit_setup_seconds": setup,
                       "training_seconds": source_training + setup + preprocessing["train"] + kernel_time + fit_time,
                       "train_kernel_entries": len(values["train"]) ** 2,
                       "quantum_qubits": (8 if method == "risk_axis" else 7) if engine else 0,
                       "simulated_qubits": 7 if engine else 0, "quantum_statevectors": len(values["train"]) if engine else 0,
                       "validation_statevectors": len(values["validation"]) if engine else 0,
                       "peak_train_kernel_gib": kernel.nbytes / 2**30, **prediction_times}
                if stage == "full":
                    row["test_statevectors"] = len(values["test"]) if engine else 0
                    row["test_kernel_entries"] = len(values["test"]) * len(values["train"])
                    row["inference_seconds"] = source_times["test_feature_seconds"] + (source_times["test_risk_seconds"] if method != "baseline" else 0.)
                    row["inference_seconds"] += preprocessing["test"] + prediction_times["test_kernel_seconds"] + prediction_times["test_decision_seconds"]
                rows.append(row)
                del model
            del kernel, values
            gc.collect()
        np.savez_compressed(folder / "margins.npz", **{k: np.asarray(v) for k, v in margins.items()})
        write_json(folder / "audit.json", audit)
        write_json(done, {"identity": identity, "trials": rows, "archive_sha256": sha(folder / "margins.npz"),
                          "audit_sha256": sha(folder / "audit.json"), "elapsed_seconds": time.perf_counter() - started})
        print("completed", stage, *unit, flush=True)
    return unit


def assess_gate(selected, protocol):
    """All choices in selected must have been made with validation labels only."""
    import pandas as pd
    frame = pd.DataFrame(selected)
    qs = frame[frame.learner.eq(protocol["gate"]["learner"])]
    baseline = qs[qs.method.eq("baseline")].set_index(["dataset", "seed"])
    summaries = []
    g = protocol["gate"]
    for method in protocol["candidates"]:
        group = qs[qs.method.eq(method)].set_index(["dataset", "seed"]).reindex(baseline.index)
        if group.validation_cost.isna().any() or len(group) != len(protocol["datasets"]) * len(protocol["smoke_seeds"]):
            raise AssertionError("Incomplete smoke gate")
        if (baseline.validation_cost == 0).any():
            raise AssertionError("Relative cost reduction undefined for zero baseline")
        reduction = (baseline.validation_cost - group.validation_cost) / baseline.validation_cost
        by_dataset = reduction.groupby(level="dataset").mean()
        fn_reduction = float((baseline.validation_fn - group.validation_fn).groupby(level="dataset").mean().mean())
        accuracy_change = float((group.validation_balanced_accuracy - baseline.validation_balanced_accuracy).groupby(level="dataset").mean().mean())
        specificity = group.validation_specificity.groupby(level="dataset").mean()
        row = {"method": method, "macro_cost_reduction": float(by_dataset.mean()),
               "improved_datasets": int((by_dataset > 0).sum()), "mean_fn_reduction": fn_reduction,
               "macro_balanced_accuracy_change": accuracy_change,
               "macro_balanced_accuracy": float(group.validation_balanced_accuracy.groupby(level="dataset").mean().mean()),
               "nonzero_specificity_datasets": int((specificity > 0).sum()),
               "dataset_cost_reduction": by_dataset.to_dict()}
        row["passes"] = bool(row["macro_cost_reduction"] >= g["minimum_macro_cost_reduction"] and
                              row["improved_datasets"] >= g["minimum_improved_datasets"] and
                              fn_reduction > g["minimum_mean_fn_reduction"] and
                              accuracy_change >= -g["maximum_macro_balanced_accuracy_drop"] and
                              row["macro_balanced_accuracy"] >= g["minimum_macro_balanced_accuracy"] and
                              row["nonzero_specificity_datasets"] >= g["minimum_nonzero_specificity_datasets"])
        summaries.append(row)
    passed = [r for r in summaries if r["passes"]]
    best = min(passed, key=lambda r: (-r["macro_cost_reduction"], -r["mean_fn_reduction"], protocol["candidates"].index(r["method"]))) if passed else None
    return {"passed": best is not None, "winner": best["method"] if best else None, "summaries": summaries,
            "test_labels_read": False, "rule": g}
