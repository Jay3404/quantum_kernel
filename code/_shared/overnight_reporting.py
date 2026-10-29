"""Precommit candidate choices, then evaluate independent confirmation/test labels."""
import itertools
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import controlled as base
from .full_data import write_json
from .improved import sha
from .overnight_risk import ROOT, METRICS, job_folder, prepared_folder, load_inputs, fixed_test_split
from build_controlled_results import corrected_difference, holm


def collect(output, tasks, hashes):
    groups, receipts, missing = {}, {}, []
    for job in tasks:
        folder = job_folder(output, job)
        path = folder / "complete.json"
        if not path.exists():
            missing.append(list(job))
            continue
        record = json.loads(path.read_text())
        _, inputs = load_inputs(output, *job[:2])
        assert record["job"] == list(job) and record["source_sha256"] == hashes
        assert record["input_sha256"] == inputs["input_sha256"] and not record["held_out_labels_read"]
        for filename, key in (("margins.npz", "margins_sha256"), ("exact_margins.npz", "exact_margins_sha256"),
                              ("parameter_lock.json", "parameter_lock_sha256")):
            assert sha(folder / filename) == record[key]
        if job[3] != "reference":
            assert sha(folder / "geometry.npz") == record["geometry_sha256"]
        groups[job] = record
        receipts[str(path.relative_to(ROOT))] = sha(path)
    return groups, receipts, missing


def selected_choices(groups):
    choices = []
    for job, record in groups.items():
        common = {"dataset": job[0], "seed": job[1], "kernel": job[2], "family": job[3], "arm": job[4], "job": list(job)}
        if job[3] == "reference":
            trials = record["trials"]
            selected = []
            for learner, representation in itertools.product(("LR", "RF", "XGBoost"), ("raw", "pca7")):
                pool = [r for r in trials if r["learner"] == learner and r["method"] == representation]
                row = min(pool, key=lambda r: (r["tune_cost"], str(r.get("depth"))))
                selected.append((f"reference_{representation}_{learner}", row))
            for role, row in selected:
                choices.append({**row, **common, "role": role, "score_source": "margins", "score_key": row["score_index"]})
        else:
            for role, row in record["choices"].items():
                if job[4] != "normal" and role == "baseline":
                    continue
                choices.append({**row, **common, "role": role, "score_source": "exact_margins", "score_key": role})
    return choices


def append_classical_best(choices):
    for dataset, seed in sorted({(r["dataset"], r["seed"]) for r in choices}):
        pool = [r for r in choices if r["dataset"] == dataset and r["seed"] == seed and r["arm"] == "normal"
                and (r["family"] == "reference" or r["kernel"] == "RBF-SVM")]
        if pool:
            row = min(pool, key=lambda r: (r["tune_cost"], r["lambda"], r["C"], r["family"], r["role"]))
            choices.append({**row, "role": "classical_best", "source_family": row["family"], "family": "reference_best"})
    return choices


def geometry(kernel, y):
    off = ~np.eye(len(y), dtype=bool)
    normal = (y[:, None] == 0) & (y[None, :] == 0) & off
    default = (y[:, None] == 1) & (y[None, :] == 1) & off
    cross = y[:, None] != y[None, :]
    if not all(mask.any() for mask in (normal, default, cross)):
        return {"separation_gap": np.nan}
    distances = 2 * (1 - kernel)
    within0, within1, between = [float(distances[m].mean()) for m in (normal, default, cross)]
    return {"distance_normal_normal": within0, "distance_default_default": within1,
        "distance_between_classes": between, "separation_gap": between - .5 * (within0 + within1),
        "offdiagonal_kernel_std": float(kernel[off].std())}


class LossIndex:
    def __init__(self, path):
        frame = pd.read_csv(path, float_precision="round_trip")
        keys = ["dataset", "seed", "kernel", "family", "arm", "role"]
        self.groups = {key: part.sort_values("row_id") for key, part in frame.groupby(keys, sort=False)}
        self.ids = {}

    def matrix(self, models, ratio=5):
        losses, labels = [], None
        for seed, row in models.sort_index().iterrows():
            part = self.groups[row.dataset, int(seed), row.kernel, row.family, row.arm, row.role]
            ids = part.row_id.to_numpy()
            if row.dataset in self.ids:
                np.testing.assert_array_equal(ids, self.ids[row.dataset])
            else:
                self.ids[row.dataset] = ids
            current_labels = part.y.to_numpy()
            if labels is None:
                labels = current_labels
            else:
                np.testing.assert_array_equal(labels, current_labels)
            prediction = part.prediction.to_numpy() if ratio == 5 else (part.score.to_numpy() >= row[f"threshold_ratio_{ratio}"])
            cost = ratio * ((current_labels == 1) & (prediction == 0)) + ((current_labels == 0) & (prediction == 1))
            expected = row.cost if ratio == 5 else row[f"cost_ratio_{ratio}"]
            assert abs(float(cost.sum()) - expected) < 1e-10
            losses.append(cost)
        return np.asarray(losses), labels


def paired_customer_summary(differences, labels, key):
    differences = np.asarray(differences, dtype=float)
    values = differences.mean(axis=0)
    seed_deltas = differences.sum(axis=1)
    observed = float(values.sum())
    seed = int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    boot = np.empty(5000)
    positions = [np.flatnonzero(labels == c) for c in (0, 1)]
    for start in range(0, len(boot), 64):
        n = min(64, len(boot) - start)
        total = np.zeros(n)
        for index in positions:
            if len(index):
                sampled = index[rng.integers(0, len(index), size=(n, len(index)))]
                total += values[sampled].sum(axis=1)
        boot[start:start + n] = total
    permutations, extreme = 19999, 0
    for start in range(0, permutations, 128):
        n = min(128, permutations - start)
        signs = 2 * rng.integers(0, 2, size=(n, len(values)), dtype=np.int8) - 1
        null = signs @ values
        extreme += int(np.sum(abs(null) >= abs(observed) - 1e-12))
    low, high = np.quantile(boot, [.025, .975])
    return {"n": len(seed_deltas), "mean_difference": observed, "sd_difference": float(seed_deltas.std(ddof=1)),
        "ci_low": float(low), "ci_high": float(high), "p_value": (extreme + 1) / (permutations + 1),
        "wins": int((seed_deltas > 0).sum()), "ties": int((seed_deltas == 0).sum()), "losses": int((seed_deltas < 0).sum()),
        "test_customer_clusters": len(values), "bootstrap_replicates": len(boot), "permutations": permutations,
        "test": "Customer-cluster paired sign-flip; stratified customer bootstrap CI; losses averaged across single models, no prediction ensemble"}


def evaluate(output, choices, full, protocol, with_predictions=True):
    labels, raw_data, score_cache, geometry_cache = {}, {}, {}, {}
    rows, predictions = [], []
    for choice in choices:
        pair = choice["dataset"], choice["seed"]
        arrays, record = load_inputs(output, *pair)
        if pair not in labels:
            prepared = prepared_folder(output, *pair)
            if full:
                if pair[0] not in raw_data:
                    raw_data[pair[0]] = base.load_data(pair[0], str(ROOT / "data"))[1]
                y = raw_data[pair[0]]
                split = json.loads((prepared / "split.json").read_text())
                assert split["outer"] == split["actual"] == fixed_test_split(y, pair[1], pair[0])
                assert len(arrays["train"]) == len(split["actual"]["train"])
                labels[pair] = y[split["actual"]["test"]], np.asarray(split["actual"]["test"])
            else:
                assert sha(prepared / "confirmation_labels.npz") == record["confirmation_labels_sha256"]
                with np.load(prepared / "confirmation_labels.npz", allow_pickle=False) as saved:
                    labels[pair] = saved["y_gate"], np.arange(len(saved["y_gate"]))
        folder = job_folder(output, choice["job"])
        key = (str(folder), choice["score_source"])
        if key not in score_cache:
            with np.load(folder / (choice["score_source"] + ".npz"), allow_pickle=False) as saved:
                score_cache[key] = dict(saved)
        archive = score_cache[key]
        if choice["score_source"] == "margins":
            tune, score = archive["tune"][choice["score_key"]], archive["gate"][choice["score_key"]]
        else:
            tune, score = archive[choice["score_key"] + "_tune"], archive[choice["score_key"] + "_gate"]
        selected_tune = base.metrics(arrays["y_tune"], tune, choice["threshold"])
        assert selected_tune["cost"] == choice["tune_cost"] and choice["fit_rows"] == len(arrays["train"])
        y, ids = labels[pair]
        metrics = base.metrics(y, score, choice["threshold"])
        row = {k: v for k, v in choice.items() if k not in ("representation_details", "job", "score_source", "score_key", "kernel_seconds", "inference_seconds", "nystrom_spectrum")}
        row.update(metrics, n_train=len(arrays["train"]), n_validation=len(arrays["tune"]), n_eval=len(y),
            specificity=metrics["tn"] / max(metrics["tn"] + metrics["fp"], 1))
        for ratio, threshold in choice.get("sensitivity_thresholds", {}).items():
            other = base.metrics(y, score, threshold)
            for key, value in other.items():
                row[f"{key}_ratio_{ratio}"] = int(ratio) * other["fn"] + other["fp"] if key == "cost" else value
            row[f"threshold_ratio_{ratio}"] = threshold
        if choice["family"] not in ("reference", "reference_best") and choice["role"] != "lambda_profile":
            if str(folder) not in geometry_cache:
                with np.load(folder / "geometry.npz", allow_pickle=False) as saved:
                    geometry_cache[str(folder)] = dict(saved)
            complete = json.loads((folder / "complete.json").read_text())
            diagnostic_key = "exact_" + choice["score_key"]
            positions = complete["geometry_positions"][diagnostic_key]
            row.update(geometry(geometry_cache[str(folder)][diagnostic_key], y[positions]))
        rows.append(row)
        if full and with_predictions:
            predictions.append(pd.DataFrame({"dataset": pair[0], "seed": pair[1], "kernel": choice["kernel"],
                "family": choice["family"], "arm": choice["arm"], "role": choice["role"], "row_id": ids,
                "y": y, "score": score, "threshold": choice["threshold"], "prediction": (score >= choice["threshold"]).astype(int)}))
    return pd.DataFrame(rows), predictions


def shortlist(groups, protocol):
    rows = []
    for method in protocol["methods"]:
        for learner, dataset, seed in itertools.product(protocol["learners"], protocol["datasets"], protocol["discovery_seeds"]):
            job, baseline = (dataset, seed, learner, method, "normal"), (dataset, seed, learner, "baseline", "normal")
            if job not in groups or baseline not in groups:
                continue
            proposed = groups[job]["choices"]["proposed"]["tune_cost"]
            before = groups[baseline]["choices"]["baseline"]["tune_cost"]
            rows.append({"family": method, "kernel": learner, "dataset": dataset, "seed": seed,
                "rcr": (before - proposed) / before if before else np.nan})
    frame = pd.DataFrame(rows)
    expected = len(protocol["learners"]) * len([d for d in protocol["datasets"] if d != "german_numeric"]) * len(protocol["discovery_seeds"])
    ranking = []
    for family, part in frame.groupby("family"):
        part = part[part.dataset.ne("german_numeric")]
        if len(part) != expected or part.rcr.isna().any():
            continue
        means = part.groupby("dataset").rcr.mean()
        score = float((means.south_german + means.taiwan + means.hmeq) / 3)
        ranking.append({"family": family, "source_balanced_rcr": score, "pure_metric": family in METRICS})
    ranking.sort(key=lambda r: (-r["source_balanced_rcr"], r["family"]))
    chosen = [r["family"] for r in ranking if not r["pure_metric"]][:8] + [r["family"] for r in ranking if r["pure_metric"]][:4]
    for row in ranking:
        if len(chosen) < protocol["shortlist_families"] and row["family"] not in chosen:
            chosen.append(row["family"])
    return chosen[:protocol["shortlist_families"]], ranking, frame


def gate_effects(raw, methods, protocol):
    rows = []
    for family, learner, dataset in itertools.product(methods, protocol["learners"], protocol["datasets"]):
        proposed = raw[(raw.family == family) & (raw.kernel == learner) & (raw.dataset == dataset) & raw.arm.eq("normal") & raw.role.eq("proposed")].set_index("seed")
        baseline = raw[(raw.family == "baseline") & (raw.kernel == learner) & (raw.dataset == dataset) & raw.role.eq("baseline")].set_index("seed")
        complete = sorted(proposed.index) == sorted(baseline.index) == protocol["seeds"]
        result = {"family": family, "kernel": learner, "dataset": dataset, "complete_ten": complete,
            "passes_practical": False, "p_value": np.nan}
        if complete:
            baseline = baseline.reindex(proposed.index)
            delta = baseline.cost - proposed.cost
            result.update(corrected_difference(delta.to_numpy(), float((proposed.n_eval / proposed.n_train).iloc[0])))
            result["mean_rcr"] = float(((baseline.cost - proposed.cost) / baseline.cost.replace(0, np.nan)).mean()) if baseline.cost.gt(0).all() else np.nan
            result["recall_change"] = float((proposed.recall - baseline.recall).mean())
            result["balanced_accuracy"] = float(proposed.balanced_accuracy.mean())
            result["geometry_gap_change"] = float((proposed.separation_gap - baseline.separation_gap).mean())
            rule = protocol["promotion"]
            result["passes_practical"] = bool(result["mean_rcr"] >= rule["mean_rcr"] and result["wins"] >= rule["wins_of_ten"]
                and result["recall_change"] >= rule["minimum_recall_change"] and result["balanced_accuracy"] >= rule["minimum_balanced_accuracy"])
        rows.append(result)
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table.p_value.fillna(1).to_numpy())
    table["holm_family_size"] = len(table)
    primary_mask = table.dataset.ne("german_numeric")
    table["selection_p_holm"] = np.nan
    table.loc[primary_mask, "selection_p_holm"] = holm(table.loc[primary_mask, "p_value"].fillna(1).to_numpy())
    table["statistical_support"] = table.selection_p_holm.lt(.05) & table.get("ci_low", pd.Series(np.nan, index=table.index)).gt(0)
    eligible = []
    for method in methods:
        scopes = table[table.family.eq(method)]
        primary = scopes[scopes.dataset.ne("german_numeric")]
        if not primary.complete_ten.all() or not primary.passes_practical.any():
            continue
        ds = scopes.groupby("dataset").mean_rcr.mean()
        score = float((ds.south_german + ds.taiwan + ds.hmeq) / 3)
        eligible.append({"family": method, "supported_scopes": int(primary.statistical_support.sum()),
            "practical_scopes": int(primary.passes_practical.sum()), "source_balanced_rcr": score})
    eligible.sort(key=lambda r: (-r["supported_scopes"], -r["source_balanced_rcr"], r["family"]))
    return table, [r["family"] for r in eligible[:protocol["full_families"]]], eligible


def threshold_for_ratio(y, scores, ratio, probability=False):
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(y) - 1]
    positives, negatives = np.cumsum(y[order]), np.cumsum(1 - y[order])
    thresholds = np.r_[np.nextafter(ordered[0], np.inf), ordered[ends]]
    costs = np.r_[ratio * y.sum(), ratio * (y.sum() - positives[ends]) + negatives[ends]]
    positions = np.flatnonzero(costs == costs.min())
    default = 1 / (ratio + 1) if probability else 0.
    return float(thresholds[positions[np.argmin(abs(thresholds[positions] - default))]])


def sensitivity_choices(output, choices):
    cache, inputs = {}, {}
    for row in choices:
        pair = row["dataset"], row["seed"]
        if pair not in inputs:
            inputs[pair] = load_inputs(output, *pair)[0]
        path = job_folder(output, row["job"]) / (row["score_source"] + ".npz")
        if str(path) not in cache:
            with np.load(path, allow_pickle=False) as saved:
                cache[str(path)] = dict(saved)
        scores = cache[str(path)]
        tune = scores["tune"][row["score_key"]] if row["score_source"] == "margins" else scores[row["score_key"] + "_tune"]
        row["sensitivity_thresholds"] = {str(r): threshold_for_ratio(inputs[pair]["y_tune"], tune, r,
            row["learner"] in ("LR", "RF", "XGBoost")) for r in (2, 10)}
    return choices


def lambda_profiles(groups, protocol):
    profiles = []
    for job, record in groups.items():
        if job[3] in ("baseline", "reference") or job[4] != "normal":
            continue
        for strength in [0., *protocol["lambdas"]]:
            pool = [r for r in record["trials"] if r["lambda"] == strength]
            if not pool:
                continue
            row = min(pool, key=lambda r: (r["tune_cost"], r["C"], r["geometry"]))
            profiles.append({**row, "kernel": job[2], "family": job[3], "arm": job[4], "role": "lambda_profile",
                "job": list(job), "score_source": "margins", "score_key": row["score_index"],
                "final_exact": not row["approximate_search"], "profile_scope": "approximate-search" if row["approximate_search"] else "exact-search"})
    return profiles


def save_stage(output, tasks, hashes, protocol, methods, stage):
    groups, receipts, missing = collect(output, tasks, hashes)
    choices = append_classical_best(selected_choices(groups))
    profiles = []
    if stage == "full":
        choices = sensitivity_choices(output, choices)
        profiles = sensitivity_choices(output, lambda_profiles(groups, protocol))
    lock = output / "selection_lock.json"
    content = {"choices": choices, "lambda_profiles": profiles, "candidate_receipts_sha256": receipts, "missing_jobs": missing,
        "source_sha256": hashes, "rule": "All choices fixed using tune labels before confirmation/test label access"}
    if lock.exists():
        assert stage == "full" and json.loads(lock.read_text()) == content, "Never alter a stored choice lock"
    else:
        write_json(lock, content)
    if stage == "discovery":
        selected, ranking, trials = shortlist(groups, protocol)
        trials.to_csv(output / "discovery_tune_comparisons.csv", index=False)
        write_json(output / "shortlist.json", {"families": selected, "ranking": ranking, "missing_jobs": missing,
            "confirmation_labels_read": False, "outer_validation_or_test_labels_read": False})
        return selected
    if stage == "full" and not (output / "test_release.json").exists():
        write_json(output / "test_release.json", {"selection_lock_sha256": sha(lock), "released_utc": datetime.now(timezone.utc).isoformat(),
            "no_further_model_fitting_or_selection": True})
    raw, predictions = evaluate(output, choices, stage == "full", protocol)
    raw.to_csv(output / "seed_metrics.csv", index=False)
    raw.groupby(["dataset", "kernel", "family", "arm", "role"])[["cost", "fn", "fp", "f1", "recall", "balanced_accuracy"]].agg(["mean", "std"]).to_csv(output / "summary_metrics.csv")
    timing_columns = [name for name in ("training_seconds", "test_inference_seconds", "train_kernel_seconds", "test_kernel_seconds", "fit_seconds", "exact_fit_seconds") if name in raw]
    raw.groupby(["dataset", "kernel", "family", "arm", "role"])[timing_columns].agg(["mean", "std"]).to_csv(output / "runtime_summary.csv")
    write_json(output / "quantum_settings.json", {"simulator": "CPU Aer double exact statevector; optional GPU Gram computation only; no QPU/shots/noise",
        "models": [{"dataset": c["dataset"], "seed": c["seed"], "family": c["family"], "kernel": c["kernel"], "arm": c["arm"], "role": c["role"],
            "circuits": c.get("representation_details", {}).get("circuits", [])} for c in choices]})
    if stage == "confirmation":
        table, promoted, ranking = gate_effects(raw, methods, protocol)
        table.to_csv(output / "confirmation_comparisons.csv", index=False)
        write_json(output / "promotion.json", {"families": promoted, "ranking": ranking, "all_scopes": len(table),
            "missing_jobs": missing, "rule": protocol["promotion"], "not_final_test_significance": True})
        return promoted
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    profile_metrics, _ = evaluate(output, profiles, True, protocol, with_predictions=False)
    profile_metrics.to_csv(output / "lambda_seed_metrics.csv", index=False)
    losses = LossIndex(output / "predictions.csv.gz")
    supplementary_statistics(output, raw, profile_metrics, methods, protocol, losses)
    full_report(output, raw, choices, groups, methods, protocol, missing, losses)
    return methods


def supplementary_statistics(output, raw, profiles, methods, protocol, losses):
    rows = []
    for family, learner, dataset, strength in itertools.product(methods, protocol["learners"], protocol["datasets"], protocol["lambdas"]):
        part = profiles[(profiles.family == family) & (profiles.kernel == learner) & (profiles.dataset == dataset)]
        proposed = part[part["lambda"].eq(strength)].set_index("seed")
        baseline = part[part["lambda"].eq(0)].set_index("seed")
        complete = sorted(proposed.index) == sorted(baseline.index) == protocol["seeds"]
        row = {"family": family, "kernel": learner, "dataset": dataset, "lambda": strength,
            "complete_ten": complete, "p_value": np.nan, "final_exact_model_comparison": False}
        if complete:
            delta = baseline.cost.reindex(proposed.index) - proposed.cost
            row.update(mean_difference=float(delta.mean()), sd_difference=float(delta.std(ddof=1)),
                wins=int((delta > 0).sum()), ties=int((delta == 0).sum()), losses=int((delta < 0).sum()))
            row["profile_scope"] = str(proposed.profile_scope.iloc[0])
        row["inference"] = "Descriptive lambda sensitivity only; no dependent-seed significance test"
        rows.append(row)
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    table["holm_family_size"] = 0
    table["descriptive_profile_comparisons"] = len(table)
    table.to_csv(output / "lambda_paired_cost.csv", index=False)
    rows = []
    for family, learner, dataset, ratio in itertools.product(methods, protocol["learners"], protocol["datasets"], (2, 10)):
        part = raw[(raw.family == family) & (raw.kernel == learner) & (raw.dataset == dataset) & raw.arm.eq("normal")]
        proposed = part[part.role.eq("proposed")].set_index("seed")
        controls = {"own_no_risk": part[part.role.eq("baseline")].set_index("seed"),
            "tuned_baseline": raw[(raw.family == "baseline") & (raw.kernel == learner) & (raw.dataset == dataset) & raw.role.eq("baseline")].set_index("seed"),
            "best_classical": raw[(raw.family == "reference_best") & (raw.dataset == dataset) & raw.role.eq("classical_best")].set_index("seed")}
        for contrast, control in controls.items():
            complete = sorted(control.index) == sorted(proposed.index) == protocol["seeds"]
            row = {"family": family, "kernel": learner, "dataset": dataset, "fn_cost": ratio, "contrast": contrast,
                "complete_ten": complete, "p_value": np.nan, "models_fixed_at_5_to_1": True}
            if complete:
                name = f"cost_ratio_{ratio}"
                control = control.reindex(proposed.index)
                a, labels = losses.matrix(control, ratio)
                b, other = losses.matrix(proposed, ratio)
                np.testing.assert_array_equal(labels, other)
                row.update(paired_customer_summary(a - b, labels, (family, learner, dataset, ratio, contrast)))
            rows.append(row)
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table.p_value.fillna(1).to_numpy())
    table["holm_family_size"] = len(table)
    table.to_csv(output / "cost_sensitivity_paired.csv", index=False)


def full_report(output, raw, choices, groups, methods, protocol, missing, loss_index=None):
    loss_index = loss_index or LossIndex(output / "predictions.csv.gz")
    stats_rows, directions = [], []
    for family, learner, dataset in itertools.product(methods, protocol["learners"], protocol["datasets"]):
        part = raw[(raw.family == family) & (raw.kernel == learner) & (raw.dataset == dataset)]
        proposed = part[part.arm.eq("normal") & part.role.eq("proposed")].set_index("seed")
        own = part[part.arm.eq("normal") & part.role.eq("baseline")].set_index("seed")
        comparisons = {"own_no_risk": own,
            "tuned_baseline": raw[(raw.family == "baseline") & (raw.kernel == learner) & (raw.dataset == dataset) & raw.role.eq("baseline")].set_index("seed"),
            "best_classical": raw[(raw.family == "reference_best") & (raw.dataset == dataset) & raw.role.eq("classical_best")].set_index("seed")}
        for arm in ("shuffled", "constant", "separable"):
            if arm == "constant" and family in METRICS or arm == "separable" and learner != "QSVC":
                continue
            comparisons[arm] = part[part.arm.eq(arm) & part.role.eq("proposed")].set_index("seed")
        for contrast, control in comparisons.items():
            complete = sorted(control.index) == sorted(proposed.index) == protocol["seeds"]
            result = {"family": family, "kernel": learner, "dataset": dataset, "contrast": contrast,
                "complete_ten": complete, "p_value": np.nan}
            if complete:
                control = control.reindex(proposed.index)
                delta = control.cost - proposed.cost
                a, labels = loss_index.matrix(control)
                b, other = loss_index.matrix(proposed)
                np.testing.assert_array_equal(labels, other)
                result.update(paired_customer_summary(a - b, labels, (family, learner, dataset, contrast)))
                result["primary_source"] = dataset != "german_numeric"
                result["mean_rcr_pct"] = float((100 * delta / control.cost).mean()) if control.cost.gt(0).all() else np.nan
                for seed in proposed.index:
                    d = int(delta.loc[seed])
                    directions.append({"family": family, "kernel": learner, "dataset": dataset, "contrast": contrast,
                        "seed": seed, "control_cost": int(control.loc[seed, "cost"]), "proposed_cost": int(proposed.loc[seed, "cost"]),
                        "delta": d, "direction": "개선" if d > 0 else "악화" if d < 0 else "동률"})
            stats_rows.append(result)
    for family, dataset in itertools.product(methods, protocol["datasets"]):
        delta_by_kernel, matrix_by_kernel, interaction_labels = {}, {}, None
        for learner in protocol["learners"]:
            part = raw[(raw.family == family) & (raw.kernel == learner) & (raw.dataset == dataset) & raw.arm.eq("normal")]
            proposed = part[part.role.eq("proposed")].set_index("seed")
            baseline = part[part.role.eq("baseline")].set_index("seed")
            if sorted(proposed.index) == sorted(baseline.index) == protocol["seeds"]:
                delta_by_kernel[learner] = baseline.cost.reindex(proposed.index) - proposed.cost
                a, labels = loss_index.matrix(baseline.reindex(proposed.index))
                b, other = loss_index.matrix(proposed)
                np.testing.assert_array_equal(labels, other)
                if interaction_labels is not None:
                    np.testing.assert_array_equal(interaction_labels, labels)
                interaction_labels = labels
                matrix_by_kernel[learner] = a - b
        row = {"family": family, "kernel": "interaction", "dataset": dataset, "contrast": "quantum_minus_rbf_gain",
            "complete_ten": len(delta_by_kernel) == 2, "p_value": np.nan}
        if len(delta_by_kernel) == 2:
            delta = delta_by_kernel["QSVC"] - delta_by_kernel["RBF-SVM"]
            row.update(paired_customer_summary(matrix_by_kernel["QSVC"] - matrix_by_kernel["RBF-SVM"], interaction_labels,
                (family, dataset, "quantum_minus_rbf_gain")))
            row["primary_source"] = dataset != "german_numeric"
        stats_rows.append(row)
    stats = pd.DataFrame(stats_rows)
    for column in ("ci_low", "ci_high"):
        if column not in stats:
            stats[column] = np.nan
    stats["p_holm"] = holm(stats.p_value.fillna(1).to_numpy())
    stats["holm_family_size"] = len(stats)
    stats.to_csv(output / "paired_cost.csv", index=False)
    pd.DataFrame(directions).to_csv(output / "seed_directions.csv", index=False)
    lines = ["# 위험 주입 야간 성능 탐색 결과", "", "## Material Passport", "",
        "상태: ANALYZED. 저장 결과 감사이며 독립적인 전체 재학습 재현 또는 양자 우위의 증명이 아니다.", "",
        f"전체 평가 선정 방법군{len(methods)}개, 자료4개·공통10개 새 분할 시드. 미완료 작업{len(missing)}개. 미완료 범위는10시드 결과로 보고하지 않는다.", "",
        "스모크는 새 외부 학습80% 내부에서만 학습/선택/확인을 나누고 전처리·OOF위험을 실제 학습행만으로 다시 적합했다. 최종은 학습80% 전체를 사용한다.4096행 초과의 파라미터 탐색은512랜드마크 Nyström/LinearSVC 대리 탐색이며 최종 예측은 잠긴 파라미터를 사용한 정확 커널 SVC다. 정확 모델의 검증 임계값은 새로 선택한 뒤 같은 적합의 테스트 점수에 적용했다.", "",
        "기존 자료를 이미 반복 탐색했으므로 새 시드는 미사용 외부 자료가 아니다. 두 German 표현은 같은 원자료 계열이다. 실제QPU·유한샷·잡음·계산 복잡도 우위를 입증하지 않는다. 거리학습/위험교사/커널혼합의 고전 기여를 양자 고유 효과와 구분한다.", "",
        "## 같은 커널 기준 대비", "",
        "Δ=대조-제안 비용, 양수가 개선. 테스트 고객은10개 학습 시드에서 공통이다. 고객별10개 단일모델의 손실 차이를 한 묶음으로 두고 층화 고객bootstrap5000회95%구간과 고객 단위paired sign-flip19999회 원p를 계산한다. 예측을 평균하는 앙상블이 아니다. 시드10개를 독립 테스트 표본으로 세지 않는다. 모든 계획 대비에Holm을 적용하며 미완료 대비도제거하지 않는다. German Numeric은 같은 원자료의 민감도 분석이며 독립 확인 자료로 취급하지 않는다.", "",
        "| 자료 | 방법 | 커널 | 비용 Δ | 95% 구간 | 원 p | Holm p | 개선/동률/악화 |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | --- |"]
    for row in stats[stats.contrast.eq("tuned_baseline")].itertuples():
        if row.complete_ten:
            lines.append(f"| {row.dataset} | {row.family} | {row.kernel} | {row.mean_difference:.2f} | [{row.ci_low:.2f},{row.ci_high:.2f}] | {row.p_value:.4f} | {row.p_holm:.4f} | {int(row.wins)}/{int(row.ties)}/{int(row.losses)} |")
        else:
            lines.append(f"| {row.dataset} | {row.family} | {row.kernel} | 미완료 | - | - | - | - |")
    lines += ["", "전체모델의F1/Recall/균형정확도/FN/FP/거리분리는seed_metrics.csv, 모든짝비교는paired_cost.csv, 시드방향은seed_directions.csv다. 모든실패와후보/선택/커널시간/큐비트/깊이는원본complete.json에보존한다.", "",
        "미완료·후보탈락·원p·Holm p를 모두 보존한다. 유의하지 않음을 효과0의 증명으로 해석하지 않는다. 집계반전·개인일반화·선택편향·충돌변수·기저율·평균회귀·완료선택·다중탐색·분석경로·인과과장·역인과11개 주의항목을 해석범위에서 점검했다."]
    (output / "결과보고서.md").write_text("\n".join(lines) + "\n")
    files = ("selection_lock.json", "test_release.json", "seed_metrics.csv", "summary_metrics.csv", "runtime_summary.csv", "quantum_settings.json", "predictions.csv.gz", "paired_cost.csv", "seed_directions.csv", "lambda_seed_metrics.csv", "lambda_paired_cost.csv", "cost_sensitivity_paired.csv", "결과보고서.md")
    write_json(output / "verification_receipt.json", {"scope": "Candidate/source/choice/score integrity; not independent retraining",
        "missing_jobs": missing, "selected_rows": len(raw), "outputs_sha256": {name: sha(output / name) for name in files}})
