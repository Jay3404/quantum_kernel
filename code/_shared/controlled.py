"""Paired classical/quantum experiments for the CMC revision."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit.circuit.library import zz_feature_map
from qiskit_aer import AerSimulator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
LEGACY_DATASETS = ("german_numeric", "south_german", "taiwan", "give_me_some_credit")
DATASETS = ("german_numeric", "south_german", "taiwan", "hmeq")
AVAILABLE_DATASETS = (*DATASETS, "give_me_some_credit", "fico_heloc")
FILES = {"german_numeric": "german-data-numeric", "south_german": "SouthGermanCredit.asc", "taiwan": "default of credit card clients.xls", "give_me_some_credit": "cs-training.csv", "hmeq": "hmeq.csv", "fico_heloc": "heloc_dataset_v1.csv"}
FICO_COLUMNS = ("ExternalRiskEstimate", "MSinceOldestTradeOpen", "MSinceMostRecentTradeOpen", "AverageMInFile",
                "NumSatisfactoryTrades", "NumTrades60Ever2DerogPubRec", "NumTrades90Ever2DerogPubRec", "PercentTradesNeverDelq",
                "MSinceMostRecentDelq", "MaxDelq2PublicRecLast12M", "MaxDelqEver", "NumTotalTrades", "NumTradesOpeninLast12M",
                "PercentInstallTrades", "MSinceMostRecentInqexcl7days", "NumInqLast6M", "NumInqLast6Mexcl7days",
                "NetFractionRevolvingBurden", "NetFractionInstallBurden", "NumRevolvingTradesWBalance",
                "NumInstallTradesWBalance", "NumBank2NatlTradesWHighUtilization", "PercentTradesWBalance")
REPRESENTATIONS = ("baseline", "convex", "feature_specific", "convex_shuffled", "feature_specific_shuffled", "convex_constant", "feature_specific_constant")
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


@lru_cache(maxsize=4)
def load_data(dataset: str, data_root: str) -> tuple[pd.DataFrame, np.ndarray, dict]:
    path = Path(data_root) / FILES[dataset]
    if dataset == "german_numeric":
        frame = pd.read_csv(path, sep=r"\s+", header=None)
        y = (frame.iloc[:, -1].to_numpy() == 2).astype(int)
        X = frame.iloc[:, :-1].copy()
        X.columns = [f"f{i}" for i in range(X.shape[1])]
        nominal = []
        expected = 1000
    elif dataset == "south_german":
        frame = pd.read_csv(path, sep=r"\s+")
        y = (frame.kredit.to_numpy() == 0).astype(int)
        X = frame.drop(columns="kredit").copy()
        nominal = ["laufkont", "moral", "verw", "sparkont", "famges", "buerge", "weitkred", "wohn", "pers", "telef", "gastarb"]
        expected = 1000
    elif dataset == "taiwan":
        frame = pd.read_excel(path, header=1)
        y = frame["default payment next month"].to_numpy(dtype=int)
        X = frame.drop(columns=["ID", "default payment next month"]).copy()
        X["EDUCATION"] = X.EDUCATION.replace({5: 0, 6: 0})
        nominal = ["SEX", "EDUCATION", "MARRIAGE"]
        for column in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
            values = X.pop(column)
            X[f"{column}_delay"] = values.clip(lower=0)
            X[f"{column}_status"] = np.where(values > 0, 1, values)
            nominal.append(f"{column}_status")
        expected = 30000
    elif dataset == "hmeq":
        frame = pd.read_csv(path, na_values=["."], skipinitialspace=True)
        expected_columns = {"BAD", "LOAN", "MORTDUE", "VALUE", "REASON", "JOB", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"}
        if set(frame.columns) != expected_columns or not frame.BAD.isin([0, 1]).all():
            raise ValueError("Unexpected HMEQ schema or missing targets")
        y = frame.BAD.to_numpy(dtype=int)
        X = frame.drop(columns="BAD").copy()
        nominal = ["REASON", "JOB"]
        for column in nominal:
            X[column] = X[column].str.strip().replace("", np.nan)
        for column in X.columns.difference(nominal):
            X[column] = pd.to_numeric(X[column], errors="raise")
        expected = 5960
        if y.sum() != 1189:
            raise ValueError("HMEQ target counts differ from the official distribution")
    elif dataset == "give_me_some_credit":
        frame = pd.read_csv(path)
        y = frame.SeriousDlqin2yrs.to_numpy(dtype=int)
        X = frame.drop(columns=["SeriousDlqin2yrs", "Unnamed: 0"], errors="ignore").copy()
        nominal = []
        expected = 150000
    elif dataset == "fico_heloc":
        frame = pd.read_csv(path)
        if set(frame.columns) != {"RiskPerformance", *FICO_COLUMNS} or not frame.RiskPerformance.isin(["Bad", "Good"]).all():
            raise ValueError("Unexpected FICO HELOC schema or missing targets")
        y = frame.RiskPerformance.eq("Bad").to_numpy(dtype=int)
        raw = frame[list(FICO_COLUMNS)].apply(pd.to_numeric, errors="raise")
        if raw.isna().any().any() or ((raw < 0) & ~raw.isin([-7, -8, -9])).any().any():
            raise ValueError("Unexpected FICO missing or special codes")
        X = raw.mask(raw.isin([-7, -8, -9])).copy()
        # Fixed-schema flags retain each code's meaning, including all-special rows.
        for column in FICO_COLUMNS:
            for code in (-7, -8, -9):
                X[f"{column}_special_{abs(code)}"] = raw[column].eq(code).astype(int)
        nominal = []
        expected = 10459
        if y.sum() != 5459:
            raise ValueError("FICO target counts differ from the local v1 inventory")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if len(y) != expected or set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Unexpected source dimensions or targets: {dataset}")
    X = X.reset_index(drop=True)
    X.attrs["nominal"] = nominal
    audit = {"dataset": dataset, "rows": len(y), "bad_count": int(y.sum()), "features": list(X.columns), "nominal_columns": nominal, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "missing": {str(k): int(v) for k, v in X.isna().sum().items()}}
    return X, y, audit


def make_features(X: pd.DataFrame) -> Pipeline:
    nominal = X.attrs["nominal"]
    numeric = [name for name in X.columns if name not in nominal]
    encoders = [("numeric", SimpleImputer(strategy="median"), numeric)]
    if nominal:
        encoders.append(("nominal", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), nominal))
    return Pipeline([("encode", ColumnTransformer(encoders, sparse_threshold=0)), ("scale", StandardScaler()), ("pca", PCA(n_components=7, svd_solver="full"))])


def risk_estimator(X: pd.DataFrame, seed: int) -> CalibratedClassifierCV:
    base = Pipeline([("features", make_features(X)), ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=seed))])
    return CalibratedClassifierCV(base, method="sigmoid", cv=StratifiedKFold(3, shuffle=True, random_state=seed), ensemble=False, n_jobs=1)


@dataclass
class Prepared:
    train: np.ndarray
    evaluation: np.ndarray
    reference_train: np.ndarray
    reference_evaluation: np.ndarray
    train_risk: np.ndarray
    eval_risk: np.ndarray
    weights: np.ndarray
    minimum: np.ndarray
    span: np.ndarray
    gamma_reference: float
    feature_seconds: float
    total_seconds: float
    seed: int


def prepare(X: pd.DataFrame, y: np.ndarray, evaluation: pd.DataFrame, seed: int) -> Prepared:
    started = time.perf_counter()
    features = make_features(X)
    train = features.fit_transform(X)
    evaluated = features.transform(evaluation)
    ref_train = features[:-1].transform(X)
    ref_eval = features[:-1].transform(evaluation)
    feature_seconds = time.perf_counter() - started
    risk = np.zeros(len(y))
    for fold, (fit, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
        model = risk_estimator(X, seed + fold + 1)
        model.fit(X.iloc[fit], y[fit])
        risk[valid] = model.predict_proba(X.iloc[valid])[:, 1]
    model = risk_estimator(X, seed)
    model.fit(X, y)
    final_base = model.calibrated_classifiers_[0].estimator
    if not np.allclose(final_base.named_steps["features"].transform(X), train):
        raise AssertionError("Risk coefficients and shared PCA basis do not agree")
    coefficients = np.abs(final_base.named_steps["lr"].coef_[0])
    weights = coefficients / coefficients.sum() if coefficients.sum() else np.ones(7) / 7
    minimum = train.min(axis=0)
    span = np.maximum(train.max(axis=0) - minimum, 1e-12)
    original_angles = np.clip((train - minimum) / span, 0, 1) * np.pi
    return Prepared(train, evaluated, ref_train, ref_eval, risk, model.predict_proba(evaluation)[:, 1], weights, minimum, span, 1 / (7 * max(original_angles.var(), 1e-12)), feature_seconds, time.perf_counter() - started, seed)


def angles(prepared: Prepared, representation: str, strength: float) -> tuple[np.ndarray, np.ndarray, float]:
    tr, ev = prepared.train_risk, prepared.eval_risk
    if representation.endswith("shuffled"):
        rng = np.random.default_rng(prepared.seed + 7919)
        tr, ev = rng.permutation(tr), rng.permutation(ev)
    elif representation.endswith("constant"):
        tr = np.full_like(tr, prepared.train_risk.mean())
        ev = np.full_like(ev, prepared.train_risk.mean())
    weights = prepared.weights if representation.startswith("feature_specific") else np.ones(7)
    strength = 0.0 if representation == "baseline" else strength
    train = prepared.train * (1 + strength * tr[:, None] * weights)
    evaluation = prepared.evaluation * (1 + strength * ev[:, None] * weights)
    train_unit = (train - prepared.minimum) / prepared.span
    eval_unit = (evaluation - prepared.minimum) / prepared.span
    clipped = float(np.mean((eval_unit < 0) | (eval_unit > 1)))
    return np.clip(train_unit, 0, 1) * np.pi, np.clip(eval_unit, 0, 1) * np.pi, clipped


class QuantumStates:
    def __init__(self) -> None:
        self.backend = AerSimulator(method="statevector", precision="double", max_parallel_threads=1, max_parallel_experiments=1, runtime_parameter_bind_enable=True)
        circuit = zz_feature_map(7, reps=1, entanglement="full")
        circuit.save_statevector(label="state")
        self.circuit = transpile(circuit, self.backend, optimization_level=1, seed_transpiler=17)
        self.parameters = tuple(self.circuit.parameters)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        blocks = []
        for offset in range(0, len(values), 512):
            batch = values[offset:offset + 512]
            bindings = {parameter: batch[:, index].tolist() for index, parameter in enumerate(self.parameters)}
            result = self.backend.run(self.circuit, parameter_binds=[bindings], shots=1).result()
            if not result.success:
                raise RuntimeError(result.status)
            blocks.append(np.asarray([np.asarray(result.data(i)["state"]) for i in range(len(batch))]))
        return np.vstack(blocks)


def fidelity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.clip(np.abs(left @ right.conj().T) ** 2, 0, 1)


def threshold_for_cost(y: np.ndarray, scores: np.ndarray, default: float) -> tuple[float, int]:
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    positives = np.cumsum(y[order])
    negatives = np.cumsum(1 - y[order])
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(y) - 1]
    thresholds = np.r_[np.nextafter(ordered[0], np.inf), ordered[ends]]
    costs = np.r_[5 * y.sum(), 5 * (y.sum() - positives[ends]) + negatives[ends]]
    best = np.flatnonzero(costs == costs.min())
    index = best[np.argmin(np.abs(thresholds[best] - default))]
    return float(thresholds[index]), int(costs[index])


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    prediction = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"cost": int(5 * fn + fp), "fn": int(fn), "fp": int(fp), "tn": int(tn), "tp": int(tp), "f1": float(f1_score(y, prediction, zero_division=0)), "recall": float(recall_score(y, prediction)), "balanced_accuracy": float(balanced_accuracy_score(y, prediction)), "roc_auc": float(roc_auc_score(y, scores))}


def candidates(learner: str, representation: str, seed: int) -> list[dict]:
    if representation == "baseline":
        if learner == "QSVC":
            return [{"lambda": 0.0, "C": float(c), "gamma_factor": 1.0} for c in np.logspace(-3, 3, 20)]
        return [{"lambda": 0.0, "C": float(c), "gamma_factor": float(g)} for c, g in itertools.product(np.logspace(-2, 2, 4), np.logspace(-2, 2, 5))]
    rng = np.random.default_rng(seed + 431)
    rows = []
    for strength in LAMBDAS:
        gammas = rng.permutation(np.logspace(-2, 2, 4))
        rows.extend({"lambda": strength, "C": float(c), "gamma_factor": float(g if learner == "RBF-SVM" else 1.0)} for c, g in zip(np.logspace(-2, 2, 4), gammas))
    return rows


def geometry(kernel: np.ndarray, y: np.ndarray) -> dict:
    ix = np.linspace(0, len(y) - 1, min(256, len(y)), dtype=int)
    K = kernel[np.ix_(ix, ix)]
    centered = K - K.mean(0) - K.mean(1)[:, None] + K.mean()
    labels = (2 * y[ix] - 1).astype(float)
    target = np.outer(labels, labels)
    target = target - target.mean(0) - target.mean(1)[:, None] + target.mean()
    alignment = np.sum(centered * target) / max(np.linalg.norm(centered) * np.linalg.norm(target), 1e-15)
    eigenvalues = np.maximum(np.linalg.eigvalsh(K), 0)
    probabilities = eigenvalues[eigenvalues > 1e-12] / eigenvalues.sum()
    return {"geometry_n": len(ix), "kernel_alignment": float(alignment), "effective_rank": float(np.exp(-np.sum(probabilities * np.log(probabilities))))}


def tune_kernel(learner: str, representation: str, prepared: Prepared, y: np.ndarray, valid_y: np.ndarray, engine: QuantumStates, seed: int) -> tuple[dict, list[dict], float]:
    started = time.perf_counter()
    trials = []
    current_strength = None
    for parameters in candidates(learner, representation, seed):
        if parameters["lambda"] != current_strength:
            train, valid, _ = angles(prepared, representation, parameters["lambda"])
            if learner == "QSVC":
                states, valid_states = engine(train), engine(valid)
                train_kernel, valid_kernel = fidelity(states, states), fidelity(valid_states, states)
            current_strength = parameters["lambda"]
        if learner == "RBF-SVM":
            gamma = prepared.gamma_reference * parameters["gamma_factor"]
            train_kernel, valid_kernel = rbf_kernel(train, gamma=gamma), rbf_kernel(valid, train, gamma=gamma)
        model = SVC(kernel="precomputed", C=parameters["C"], class_weight="balanced")
        model.fit(train_kernel, y)
        score = model.decision_function(valid_kernel)
        threshold, cost = threshold_for_cost(valid_y, score, 0.0)
        fixed = metrics(valid_y, score, 0.0)
        trials.append({**parameters, "fixed_cost": fixed["cost"], "fixed_f1": fixed["f1"], "tuned_cost": cost, "threshold": threshold})
    choices = {}
    for policy in ("fixed", "tuned"):
        best = min(trials, key=lambda row: (row[f"{policy}_cost"], row["lambda"], row["C"], row["gamma_factor"]))
        choices[policy] = {**best, "threshold": 0.0 if policy == "fixed" else best["threshold"]}
    best = min(trials, key=lambda row: (-row["fixed_f1"], row["lambda"], row["C"], row["gamma_factor"]))
    choices["f1"] = {**best, "f1_cost": best["fixed_cost"], "threshold": 0.0}
    return choices, trials, time.perf_counter() - started


def evaluate_kernel(learner: str, representation: str, prepared: Prepared, y: np.ndarray, parameters: dict, engine: QuantumStates) -> tuple[np.ndarray, dict]:
    train, evaluation, clipped = angles(prepared, representation, parameters["lambda"])
    started = time.perf_counter()
    if learner == "QSVC":
        states = engine(train)
        train_kernel = fidelity(states, states)
    else:
        gamma = prepared.gamma_reference * parameters["gamma_factor"]
        train_kernel = rbf_kernel(train, gamma=gamma)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    model = SVC(kernel="precomputed", C=parameters["C"], class_weight="balanced")
    model.fit(train_kernel, y)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    evaluation_kernel = fidelity(engine(evaluation), states) if learner == "QSVC" else rbf_kernel(evaluation, train, gamma=gamma)
    test_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores = model.decision_function(evaluation_kernel)
    infer_seconds = time.perf_counter() - started
    preprocessing = prepared.feature_seconds if representation == "baseline" else prepared.total_seconds
    info = {"preprocessing_seconds": preprocessing, "train_kernel_seconds": train_seconds, "test_kernel_seconds": test_seconds, "fit_seconds": fit_seconds, "inference_seconds": infer_seconds, "final_total_seconds": preprocessing + train_seconds + test_seconds + fit_seconds + infer_seconds, "angle_clipped_fraction": clipped, **geometry(train_kernel, y)}
    return scores, info


def reference_models(name: str, y: np.ndarray, seed: int) -> list:
    if name == "LR":
        return [LogisticRegression(C=float(c), class_weight="balanced", max_iter=3000, random_state=seed) for c in np.logspace(-3, 3, 8)]
    if name == "RF":
        return [RandomForestClassifier(n_estimators=128, max_depth=depth, min_samples_leaf=leaf, max_features=features, class_weight="balanced", random_state=seed, n_jobs=1) for depth, leaf, features in itertools.product((None, 8), (1, 4), ("sqrt", 1.0))]
    from xgboost import XGBClassifier
    return [XGBClassifier(n_estimators=150, max_depth=depth, learning_rate=rate, min_child_weight=child, subsample=0.9, colsample_bytree=0.9, scale_pos_weight=float((y == 0).sum() / y.sum()), objective="binary:logistic", eval_metric="logloss", n_jobs=1, random_state=seed) for depth, rate, child in itertools.product((2, 4), (0.03, 0.1), (1, 5))]


def cap_indices(indices: np.ndarray, y: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or len(indices) <= maximum:
        return indices
    chosen, _ = train_test_split(indices, train_size=maximum, stratify=y[indices], random_state=seed)
    return chosen


def select_training(indices: np.ndarray, X: pd.DataFrame, y: np.ndarray, maximum: int, seed: int, sampling: str) -> np.ndarray:
    random_sample = cap_indices(indices, y, maximum, seed)
    if sampling == "random" or len(random_sample) == len(indices):
        return random_sample
    projection = make_features(X).fit_transform(X.iloc[indices])
    distance = np.square(projection).sum(axis=1)
    chosen = []
    for label in (0, 1):
        count = int((y[random_sample] == label).sum())
        positions = np.flatnonzero(y[indices] == label)
        chosen.extend(indices[positions[np.argsort(-distance[positions], kind="stable")[:count]]])
    return np.asarray(chosen)


def run_seed(dataset: str, seed: int, options: dict) -> str:
    with threadpool_limits(limits=1):
        return _run_seed(dataset, seed, options)


def _run_seed(dataset: str, seed: int, options: dict) -> str:
    directory = ROOT / "code" / dataset / "results" / options["run_id"] / f"seed_{seed:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    complete_path = directory / "complete.json"
    if complete_path.exists():
        receipt = json.loads(complete_path.read_text())
        source_hash = hashlib.sha256((Path(options["data_root"]) / FILES[dataset]).read_bytes()).hexdigest()
        if receipt["options"] != options or receipt["data_sha256"] != source_hash:
            raise ValueError("Code, data, or settings differ from completed run; use a new run id")
        return f"resumed {dataset} seed={seed}"
    started = time.perf_counter()
    X, y, audit = load_data(dataset, options["data_root"])
    train, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    train = select_training(train, X, y, options["train_cap"], seed + 101, options["sampling"])
    test = cap_indices(test, y, options["test_cap"], seed + 102)
    fit, valid = train_test_split(train, test_size=0.2, stratify=y[train], random_state=seed + 103)
    split = {"train": train.tolist(), "test": test.tolist(), "inner_train": fit.tolist(), "inner_valid": valid.tolist()}
    (directory / "split_indices.json").write_text(json.dumps(split), encoding="utf-8")
    (directory / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    inner = prepare(X.iloc[fit], y[fit], X.iloc[valid], seed)
    final = prepare(X.iloc[train], y[train], X.iloc[test], seed)
    engine = QuantumStates()
    rows, predictions, validations = [], [], []

    def record(learner, representation, policy, score, parameters, timing, tuning_seconds):
        threshold = parameters["threshold"]
        row = {"dataset": dataset, "seed": seed, "learner": learner, "representation": representation, "policy": policy, "n_train": len(train), "n_test": len(test), "bad_test": int(y[test].sum()), "lambda": parameters.get("lambda", 0.0), "C": parameters.get("C", np.nan), "gamma_factor": parameters.get("gamma_factor", np.nan), "reference_index": parameters.get("index", np.nan), "threshold": threshold, "validation_cost": parameters[f"{policy}_cost"], "tuning_seconds": tuning_seconds, **metrics(y[test], score, threshold), **timing}
        rows.append(row)
        predictions.append(pd.DataFrame({"dataset": dataset, "seed": seed, "learner": learner, "representation": representation, "policy": policy, "row_id": test, "y": y[test], "score": score, "prediction": (score >= threshold).astype(int)}))

    representations = REPRESENTATIONS[:3] if options["core_only"] else REPRESENTATIONS
    for learner in ("RBF-SVM", "QSVC"):
        for representation in representations:
            choices, trials, tuning_seconds = tune_kernel(learner, representation, inner, y[fit], y[valid], engine, seed)
            validations.extend({"learner": learner, "representation": representation, **row} for row in trials)
            for policy, parameters in choices.items():
                scores, timing = evaluate_kernel(learner, representation, final, y[train], parameters, engine)
                record(learner, representation, policy, scores, parameters, timing, tuning_seconds)
            with (directory / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps({"learner": learner, "representation": representation, "elapsed_seconds": time.perf_counter() - started}) + "\n")

    for learner in (() if options["core_only"] else ("LR", "RF", "XGBoost")):
        tuning_started = time.perf_counter()
        trials = []
        for index, model in enumerate(reference_models(learner, y[fit], seed)):
            model.fit(inner.reference_train, y[fit])
            score = model.predict_proba(inner.reference_evaluation)[:, 1]
            threshold, cost = threshold_for_cost(y[valid], score, 0.5)
            fixed = metrics(y[valid], score, 0.5)
            trials.append({"index": index, "fixed_cost": fixed["cost"], "fixed_f1": fixed["f1"], "tuned_cost": cost, "threshold": threshold})
        tuning_seconds = time.perf_counter() - tuning_started
        validations.extend({"learner": learner, "representation": "encoded_full", **row} for row in trials)
        for policy in ("fixed", "tuned", "f1"):
            best = min(trials, key=lambda row: (-row["fixed_f1"] if policy == "f1" else row[f"{policy}_cost"], row["index"]))
            if policy == "f1":
                best = {**best, "f1_cost": best["fixed_cost"]}
            model = reference_models(learner, y[train], seed)[best["index"]]
            tick = time.perf_counter()
            model.fit(final.reference_train, y[train])
            fit_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            scores = model.predict_proba(final.reference_evaluation)[:, 1]
            infer_seconds = time.perf_counter() - tick
            timing = {"preprocessing_seconds": final.feature_seconds, "fit_seconds": fit_seconds, "inference_seconds": infer_seconds, "train_kernel_seconds": 0.0, "test_kernel_seconds": 0.0, "final_total_seconds": final.feature_seconds + fit_seconds + infer_seconds}
            record(learner, "encoded_full", policy, scores, {**best, "threshold": best["threshold"] if policy == "tuned" else 0.5}, timing, tuning_seconds)
    threshold, cost = threshold_for_cost(y[valid], inner.eval_risk, 0.5)
    for policy in ("fixed", "tuned", "f1"):
        fixed_cost = metrics(y[valid], inner.eval_risk, 0.5)["cost"]
        parameters = {"threshold": threshold if policy == "tuned" else 0.5, "fixed_cost": fixed_cost, "f1_cost": fixed_cost, "tuned_cost": cost}
        record("risk_only", "risk_score", policy, final.eval_risk, parameters, {"final_total_seconds": final.total_seconds}, 0.0)
    diagnostic = {"brier_test": float(brier_score_loss(y[test], final.eval_risk)), "risk_auc_test": float(roc_auc_score(y[test], final.eval_risk)), "prevalence_test": float(y[test].mean()), "constant_risk": float(final.train_risk.mean()), "weights": final.weights.tolist(), "risk_estimator": "balanced LR, training-only sigmoid calibration, 5-fold cross-fitting"}
    (directory / "risk_diagnostics.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
    pd.DataFrame(validations).to_csv(directory / "validation_trials.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(directory / "predictions.csv.gz", index=False, compression="gzip")
    complete_path.write_text(json.dumps({"code_sha256": options["code_sha256"], "data_sha256": audit["sha256"], "rows": len(rows), "elapsed_seconds": time.perf_counter() - started, "options": options}, indent=2), encoding="utf-8")
    return f"completed {dataset} seed={seed} rows={len(rows)} seconds={time.perf_counter() - started:.1f}"


def main(dataset: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=AVAILABLE_DATASETS, default=[dataset] if dataset else list(DATASETS))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--run-id", default="cmc_hmeq_sample_20261016")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10, 101, 10)))
    parser.add_argument("--train-cap", type=int, default=2000)
    parser.add_argument("--test-cap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sampling", choices=("random", "pca_distance"), default="random")
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        parser.error("run-id must be a single directory name")
    if len(args.seeds) != len(set(args.seeds)) or len(args.datasets) != len(set(args.datasets)):
        parser.error("datasets and seeds must not contain duplicates")
    options = {"data_root": str(args.data_root.resolve()), "run_id": args.run_id, "train_cap": args.train_cap, "test_cap": args.test_cap, "sampling": args.sampling, "core_only": args.core_only, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    central = ROOT / "code" / "results" / args.run_id
    central.mkdir(parents=True, exist_ok=True)
    if (central / "metadata.json").exists():
        previous = json.loads((central / "metadata.json").read_text())
        if any(previous.get(key) != value for key, value in options.items()):
            raise ValueError("Existing run has different settings; use a new run id")
    import importlib.metadata
    metadata = {**options, "datasets": args.datasets, "seeds": args.seeds, "workers": args.workers, "python": platform.python_version(), "host": platform.node(), "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "qiskit", "qiskit-aer", "scipy", "xgboost")}, "feature_map": {"name": "ZZFeatureMap", "qubits": 7, "reps": 1, "entanglement": "full", "simulator": "Aer CPU double-precision statevector", "noise": None, "finite_shot_sampling": False}, "primary_policy": "tuned", "cost": "5FN+FP", "core_candidate_budget": 20, "reference_candidate_budget": 8, "lambda_grid": LAMBDAS, "risk_protocol": "5-fold out-of-fold; 3-fold sigmoid calibration inside each fit", "angle_policy": "common training-PCA Min-Max reference then clip to [0,pi]", "source_families": 3}
    (central / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_seed, name, seed, options) for name in args.datasets for seed in args.seeds]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    expected = len(args.datasets) * len(args.seeds)
    paths = [ROOT / "code" / name / "results" / args.run_id / f"seed_{seed:03d}" / "metrics.csv" for name in args.datasets for seed in args.seeds]
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if len(raw) != expected * (21 if args.core_only else 54):
        raise AssertionError("Incomplete experiment matrix")
    raw.to_csv(central / "raw_metrics.csv", index=False)
    print(f"all_completed jobs={expected} rows={len(raw)} output={central}", flush=True)


if __name__ == "__main__":
    main()
