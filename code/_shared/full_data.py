"""Exact, restartable full-data counterpart of the controlled benchmark."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from . import controlled as base


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def available_memory() -> int:
    memory = Path("/proc/meminfo")
    if not memory.exists():
        return 0
    fields = dict(line.split(":", 1) for line in memory.read_text().splitlines())
    available = int(fields["MemAvailable"].split()[0]) * 1024
    limit_path = Path("/sys/fs/cgroup/memory.max")
    used_path = Path("/sys/fs/cgroup/memory.current")
    if limit_path.exists() and used_path.exists():
        limit = limit_path.read_text().strip()
        if limit != "max":
            available = min(available, int(limit) - int(used_path.read_text()))
    return max(available, 0)


def kernel_block(left: np.ndarray, right: np.ndarray, learner: str, gamma: float) -> np.ndarray:
    return base.fidelity(left, right) if learner == "QSVC" else rbf_kernel(left, right, gamma=gamma)


def dense_kernel(values: np.ndarray, learner: str, gamma: float, block_size: int,
                 memory_limit_gib: float, reserve_gib: float) -> np.ndarray:
    required = len(values) ** 2 * 8
    # Fidelity's complex product and NumPy temporaries are bounded by a tile.
    workspace = block_size ** 2 * 64 + 2 * 1024 ** 3
    if required + workspace > memory_limit_gib * 1024 ** 3:
        raise MemoryError(f"Exact kernel needs {(required + workspace) / 1024 ** 3:.1f} GiB; increase --memory-limit-gib explicitly")
    available = available_memory()
    if available and required + workspace + reserve_gib * 1024 ** 3 > available:
        raise MemoryError("Insufficient available memory for exact kernel plus the reserved host memory")
    result = np.empty((len(values), len(values)), dtype=np.float64, order="C")
    for first in range(0, len(values), block_size):
        end_first = min(first + block_size, len(values))
        for second in range(0, first + 1, block_size):
            end_second = min(second + block_size, len(values))
            block = kernel_block(values[first:end_first], values[second:end_second], learner, gamma)
            result[first:end_first, second:end_second] = block
            if first != second:
                result[second:end_second, first:end_first] = block.T
    return result


def decision_blocks(model: SVC, evaluation: np.ndarray, train: np.ndarray,
                    learner: str, gamma: float, block_size: int) -> tuple[np.ndarray, float, float]:
    result = np.empty(len(evaluation))
    kernel_seconds, prediction_seconds = 0.0, 0.0
    for offset in range(0, len(evaluation), block_size):
        tick = time.perf_counter()
        block = kernel_block(evaluation[offset:offset + block_size], train, learner, gamma)
        kernel_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        result[offset:offset + len(block)] = model.decision_function(block)
        prediction_seconds += time.perf_counter() - tick
        del block
    return result, kernel_seconds, prediction_seconds


def selected_trials(trials: list[dict]) -> dict:
    choices = {}
    for policy in ("fixed", "tuned", "f1"):
        best = min(trials, key=lambda row: (-row["fixed_f1"] if policy == "f1" else row[f"{policy}_cost"], row["lambda"], row["C"], row["gamma_factor"]))
        choices[policy] = {**best, "threshold": best["threshold"] if policy == "tuned" else 0.0, "f1_cost": best["fixed_cost"]}
    return choices


def event(directory: Path, **fields) -> None:
    value = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
    with (directory / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")
    print(json.dumps(value), flush=True)


def inputs(prepared, representation, strength, learner, engine):
    train, evaluation, clipped = base.angles(prepared, representation, strength)
    if learner == "QSVC":
        train, evaluation = engine(train), engine(evaluation)
    return train, evaluation, clipped


def tune(learner, representation, prepared, y, valid_y, engine, seed, folder, options):
    path = folder / f"{learner}_{representation}_trials.json"
    trials = json.loads(path.read_text())["trials"] if path.exists() else []
    grid = base.candidates(learner, representation, seed)
    if len(trials) > len(grid) or any(any(row[key] != parameters[key] for key in parameters) for row, parameters in zip(trials, grid)):
        raise ValueError("Validation checkpoint does not match candidate sequence")
    current_strength = None
    train_kernel = None
    for index in range(len(trials), len(grid)):
        parameters = grid[index]
        event(folder, stage="tuning_started", learner=learner, representation=representation, candidate=index + 1, parameters=parameters)
        started = time.perf_counter()
        changed = parameters["lambda"] != current_strength
        if changed:
            train, valid, _ = inputs(prepared, representation, parameters["lambda"], learner, engine)
            current_strength = parameters["lambda"]
        gamma = prepared.gamma_reference * parameters["gamma_factor"]
        if learner == "RBF-SVM" or changed:
            del train_kernel
            gc.collect()
            train_kernel = dense_kernel(train, learner, gamma, options["kernel_block"], options["memory_limit_gib"], options["reserve_gib"])
        model = SVC(kernel="precomputed", C=parameters["C"], class_weight="balanced", cache_size=options["svm_cache_mb"])
        model.fit(train_kernel, y)
        if model.fit_status_ != 0:
            raise RuntimeError("SVC did not converge")
        score, _, _ = decision_blocks(model, valid, train, learner, gamma, options["prediction_block"])
        threshold, cost = base.threshold_for_cost(valid_y, score, 0.0)
        fixed = base.metrics(valid_y, score, 0.0)
        trials.append({**parameters, "fixed_cost": fixed["cost"], "fixed_f1": fixed["f1"], "tuned_cost": cost, "threshold": threshold, "candidate_seconds": time.perf_counter() - started, "n_support": int(model.n_support_.sum())})
        write_json(path, {"trials": trials})
        event(folder, stage="tuning_completed", learner=learner, representation=representation, candidate=index + 1, seconds=trials[-1]["candidate_seconds"])
        del model
    del train_kernel
    gc.collect()
    return selected_trials(trials), trials, sum(row["candidate_seconds"] for row in trials)


def evaluate(learner, representation, prepared, y, parameters, engine, options):
    started = time.perf_counter()
    train, evaluation, clipped = inputs(prepared, representation, parameters["lambda"], learner, engine)
    gamma = prepared.gamma_reference * parameters["gamma_factor"]
    train_kernel = dense_kernel(train, learner, gamma, options["kernel_block"], options["memory_limit_gib"], options["reserve_gib"])
    train_seconds = time.perf_counter() - started
    description = base.geometry(train_kernel, y)
    started = time.perf_counter()
    model = SVC(kernel="precomputed", C=parameters["C"], class_weight="balanced", cache_size=options["svm_cache_mb"])
    model.fit(train_kernel, y)
    if model.fit_status_ != 0:
        raise RuntimeError("SVC did not converge")
    fit_seconds = time.perf_counter() - started
    # A precomputed SVC retains coefficients/indices, not the training matrix.
    del train_kernel
    gc.collect()
    score, test_seconds, infer_seconds = decision_blocks(model, evaluation, train, learner, gamma, options["prediction_block"])
    preprocessing = prepared.feature_seconds if representation == "baseline" else prepared.total_seconds
    return score, {"preprocessing_seconds": preprocessing, "train_kernel_seconds": train_seconds, "test_kernel_seconds": test_seconds, "fit_seconds": fit_seconds, "inference_seconds": infer_seconds, "final_total_seconds": preprocessing + train_seconds + test_seconds + fit_seconds + infer_seconds, "angle_clipped_fraction": clipped, "n_support": int(model.n_support_.sum()), **description}


def all_indices(y, seed):
    train, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    fit, valid = train_test_split(train, test_size=0.2, stratify=y[train], random_state=seed + 103)
    if set(train) & set(test) or len(train) + len(test) != len(y) or len(set(train) | set(test)) != len(y):
        raise AssertionError("The full source is not partitioned exactly once")
    return train, test, fit, valid


@contextmanager
def run_lock(path):
    import fcntl
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is using this full-data run") from error
        yield


def run_seed(dataset, seed, options):
    folder = base.ROOT / "code" / dataset / "results" / options["run_id"] / f"seed_{seed:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    X, y, audit = base.load_data(dataset, options["data_root"])
    identity = {"options": options, "data_sha256": audit["sha256"]}
    settings = folder / "settings.json"
    if settings.exists() and json.loads(settings.read_text()) != identity:
        raise ValueError("Checkpoint code/data/settings changed; use a new run id")
    write_json(settings, identity)
    if (folder / "complete.json").exists():
        return
    started = time.perf_counter()
    train, test, fit, valid = all_indices(y, seed)
    write_json(folder / "split_indices.json", {"train": train.tolist(), "test": test.tolist(), "inner_train": fit.tolist(), "inner_valid": valid.tolist()})
    write_json(folder / "dataset_audit.json", audit)
    event(folder, stage="prepare_started", dataset=dataset, seed=seed, n_train=len(train), n_test=len(test), source_rows=len(y))
    inner = base.prepare(X.iloc[fit], y[fit], X.iloc[valid], seed)
    final = base.prepare(X.iloc[train], y[train], X.iloc[test], seed)
    engine = base.QuantumStates()
    event(folder, stage="prepare_completed", dataset=dataset, seed=seed)
    rows, predictions, validations = [], [], []

    def record(learner, representation, policy, score, parameters, timing, tuning_seconds):
        threshold = parameters["threshold"]
        rows.append({"dataset": dataset, "seed": seed, "learner": learner, "representation": representation, "policy": policy, "n_train": len(train), "n_test": len(test), "bad_test": int(y[test].sum()), "lambda": parameters.get("lambda", 0.0), "C": parameters.get("C", np.nan), "gamma_factor": parameters.get("gamma_factor", np.nan), "reference_index": parameters.get("index", np.nan), "threshold": threshold, "validation_cost": parameters[f"{policy}_cost"], "tuning_seconds": tuning_seconds, **base.metrics(y[test], score, threshold), **timing})
        predictions.append(pd.DataFrame({"dataset": dataset, "seed": seed, "learner": learner, "representation": representation, "policy": policy, "row_id": test, "y": y[test], "score": score, "prediction": (score >= threshold).astype(int)}))

    def final_score(learner, representation, parameters):
        numeric = {key: parameters[key] for key in ("lambda", "C", "gamma_factor", "index") if key in parameters}
        key = hashlib.sha256(json.dumps(numeric, sort_keys=True).encode()).hexdigest()[:20]
        stem = folder / f"{learner}_{representation}_{key}"
        info_path, scores_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
        if info_path.exists() and scores_path.exists():
            return np.load(scores_path)["score"], json.loads(info_path.read_text())
        event(folder, stage="final_started", learner=learner, representation=representation, parameters=numeric)
        if learner in ("RBF-SVM", "QSVC"):
            score, timing = evaluate(learner, representation, final, y[train], parameters, engine, options)
        else:
            model = base.reference_models(learner, y[train], seed)[parameters["index"]]
            tick = time.perf_counter()
            model.fit(final.reference_train, y[train])
            fit_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            score = model.predict_proba(final.reference_evaluation)[:, 1]
            inference = time.perf_counter() - tick
            timing = {"preprocessing_seconds": final.feature_seconds, "fit_seconds": fit_seconds, "inference_seconds": inference, "train_kernel_seconds": 0.0, "test_kernel_seconds": 0.0, "final_total_seconds": final.feature_seconds + fit_seconds + inference}
        np.savez_compressed(scores_path, score=score)
        write_json(info_path, timing)
        event(folder, stage="final_completed", learner=learner, representation=representation, seconds=timing["final_total_seconds"])
        return score, timing

    # Reference models are recorded first so long kernel jobs do not hide them.
    for learner in ("LR", "RF", "XGBoost"):
        path = folder / f"{learner}_trials.json"
        trials = json.loads(path.read_text())["trials"] if path.exists() else []
        for index, model in enumerate(base.reference_models(learner, y[fit], seed)):
            if index < len(trials):
                continue
            tick = time.perf_counter()
            model.fit(inner.reference_train, y[fit])
            score = model.predict_proba(inner.reference_evaluation)[:, 1]
            threshold, cost = base.threshold_for_cost(y[valid], score, 0.5)
            fixed = base.metrics(y[valid], score, 0.5)
            trials.append({"index": index, "fixed_cost": fixed["cost"], "fixed_f1": fixed["f1"], "tuned_cost": cost, "threshold": threshold, "candidate_seconds": time.perf_counter() - tick})
            write_json(path, {"trials": trials})
            event(folder, stage="reference_tuned", learner=learner, candidate=index + 1)
        validations.extend({"learner": learner, "representation": "encoded_full", **row} for row in trials)
        for policy in ("fixed", "tuned", "f1"):
            best = min(trials, key=lambda row: (-row["fixed_f1"] if policy == "f1" else row[f"{policy}_cost"], row["index"]))
            parameters = {**best, "f1_cost": best["fixed_cost"], "threshold": best["threshold"] if policy == "tuned" else 0.5}
            score, timing = final_score(learner, "encoded_full", parameters)
            record(learner, "encoded_full", policy, score, parameters, timing, sum(row["candidate_seconds"] for row in trials))
        pd.DataFrame(rows).to_csv(folder / "partial_metrics.csv", index=False)

    for learner in ("RBF-SVM", "QSVC"):
        for representation in base.REPRESENTATIONS:
            choices, trials, tuning_seconds = tune(learner, representation, inner, y[fit], y[valid], engine, seed, folder, options)
            validations.extend({"learner": learner, "representation": representation, **row} for row in trials)
            for policy, parameters in choices.items():
                score, timing = final_score(learner, representation, parameters)
                record(learner, representation, policy, score, parameters, timing, tuning_seconds)
            pd.DataFrame(rows).to_csv(folder / "partial_metrics.csv", index=False)

    threshold, cost = base.threshold_for_cost(y[valid], inner.eval_risk, 0.5)
    for policy in ("fixed", "tuned", "f1"):
        fixed_cost = base.metrics(y[valid], inner.eval_risk, 0.5)["cost"]
        parameters = {"threshold": threshold if policy == "tuned" else 0.5, "fixed_cost": fixed_cost, "f1_cost": fixed_cost, "tuned_cost": cost}
        record("risk_only", "risk_score", policy, final.eval_risk, parameters, {"final_total_seconds": final.total_seconds}, 0.0)
    write_json(folder / "risk_diagnostics.json", {"brier_test": float(brier_score_loss(y[test], final.eval_risk)), "risk_auc_test": float(roc_auc_score(y[test], final.eval_risk)), "prevalence_test": float(y[test].mean()), "constant_risk": float(final.train_risk.mean()), "weights": final.weights.tolist(), "risk_estimator": "balanced LR, training-only sigmoid calibration, 5-fold cross-fitting"})
    pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
    pd.DataFrame(validations).to_csv(folder / "validation_trials.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(folder / "predictions.csv.gz", index=False, compression="gzip")
    if len(rows) != 54:
        raise AssertionError("Incomplete model matrix")
    write_json(folder / "complete.json", {**identity, "rows": len(rows), "elapsed_seconds": time.perf_counter() - started})
    event(folder, stage="seed_completed", dataset=dataset, seed=seed, rows=len(rows))


def run_job(dataset, seed, options):
    with threadpool_limits(limits=options["blas_threads"]):
        run_seed(dataset, seed, options)
    return dataset, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=base.AVAILABLE_DATASETS, default=list(base.DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10, 101, 10)))
    parser.add_argument("--run-id", default="cmc_20261016_hmeq_full")
    parser.add_argument("--protocol", type=Path, default=base.ROOT / "docs/cmc_hmeq_experiment_protocol.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-root", type=Path, default=base.ROOT / "data")
    parser.add_argument("--kernel-block", type=int, default=2048)
    parser.add_argument("--prediction-block", type=int, default=256)
    parser.add_argument("--memory-limit-gib", type=float, default=128)
    parser.add_argument("--reserve-gib", type=float, default=24)
    parser.add_argument("--svm-cache-mb", type=float, default=512)
    parser.add_argument("--blas-threads", type=int, default=2)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or len(set(args.datasets)) != len(args.datasets) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use a single directory name and unique dataset/seed values")
    if min(args.kernel_block, args.prediction_block, args.blas_threads, args.memory_limit_gib, args.svm_cache_mb, args.workers) <= 0 or args.reserve_gib < 0:
        parser.error("Resource limits must be positive")
    if "give_me_some_credit" in args.datasets and args.workers > 1:
        parser.error("The archived GMSC benchmark requires --workers 1 to bound memory")
    protocol = json.loads(args.protocol.read_text())
    options = {"run_id": args.run_id, "data_root": str(args.data_root.resolve()), "train_cap": 0, "test_cap": 0, "sampling": "full_stratified_partition", "kernel_block": args.kernel_block, "prediction_block": args.prediction_block, "memory_limit_gib": args.memory_limit_gib, "reserve_gib": args.reserve_gib, "svm_cache_mb": args.svm_cache_mb, "blas_threads": args.blas_threads, "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {"controlled": Path(base.__file__), "full_data": Path(__file__), "full_analysis": base.ROOT / "code" / "build_full_data_results.py", "shared_analysis": base.ROOT / "code" / "build_controlled_results.py", "protocol": args.protocol}.items()}}
    central = base.ROOT / "code" / "results" / args.run_id
    central.mkdir(parents=True, exist_ok=True)
    with run_lock(central / "run.lock"):
        metadata_path = central / "metadata.json"
        previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        if previous and (previous["options"] != options or previous["datasets"] != args.datasets or previous["seeds"] != args.seeds):
            raise ValueError("Run settings changed; use another run id")
        metadata = {"run_id": args.run_id, "options": options, "datasets": args.datasets, "seeds": args.seeds, "host": platform.node(), "python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "qiskit", "qiskit-aer", "scipy", "xgboost")}, "workers": args.workers, "full_dataset": True, "expected_seed_jobs": len(args.datasets) * len(args.seeds), "core_candidate_budget": 20, "reference_candidate_budget": 8, "primary_policy": "tuned", "cost": "5FN+FP", "source_families": 3, "kernel_approximation": False, "feature_map": "7 qubits, ZZ reps=1 full entanglement, Aer CPU double statevector", "protocol": str(args.protocol.resolve()), "status": "running", "pid": os.getpid()}
        write_json(metadata_path, metadata)
        try:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(run_job, dataset, seed, options): (dataset, seed) for seed in args.seeds for dataset in args.datasets}
                for completed, future in enumerate(as_completed(futures), 1):
                    dataset, seed = future.result()
                    metadata.update(last_completed_dataset=dataset, last_completed_seed=seed, completed_seed_jobs=completed)
                    write_json(metadata_path, metadata)
                    print(f"jobs_completed={completed}/{len(futures)} dataset={dataset} seed={seed}", flush=True)
            paths = [base.ROOT / "code" / dataset / "results" / args.run_id / f"seed_{seed:03d}" / "metrics.csv" for dataset in args.datasets for seed in args.seeds]
            raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
            if len(raw) != len(args.datasets) * len(args.seeds) * 54:
                raise AssertionError("Incomplete full-data experiment matrix")
            raw.to_csv(central / "raw_metrics.csv", index=False)
            if args.datasets == protocol["datasets"] and args.seeds == protocol["seeds"]:
                metadata["status"] = "analyzing"
                write_json(metadata_path, metadata)
                from build_full_data_results import build
                build(args.run_id)
            metadata["status"] = "completed"
        except BaseException as error:
            metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write_json(metadata_path, metadata)


if __name__ == "__main__":
    main()
