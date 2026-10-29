"""Threshold-only sensitivity of archived5:1 models; never trains a model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "docs/cmc_costratio_fixed_protocol_20261029.json"
OUTPUT = ROOT / "code/results/cmc_costratio_fixed_20261029"
PARENT = ROOT / "code/results/cmc_pca7_repeated_20261020/full"
BUDGET = ROOT / "code/results/cmc_budget_riskonly_20261026"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def choose(y, score, ratio, default, ascending=False):
    """Cumulative counts over all distinct score cuts, with explicit tie order."""
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    assert len(y) == len(score) and np.isfinite(score).all() and np.isin(y, [0, 1]).all()
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    ends = np.r_[np.flatnonzero(sorted_score[1:] != sorted_score[:-1]), len(y) - 1]
    cuts = np.r_[np.nextafter(sorted_score[0], np.inf), sorted_score[ends]]
    fn = np.r_[y.sum(), y.sum() - np.cumsum(y[order])[ends]]
    fp = np.r_[0, np.cumsum(1 - y[order])[ends]]
    if ascending:
        cuts, fn, fp = cuts[::-1], fn[::-1], fp[::-1]
    costs = ratio * fn + fp
    best = np.flatnonzero(costs == costs.min())
    index = best[np.argmin(abs(cuts[best] - default))]
    return dict(threshold=float(cuts[index]), validation_cost=int(costs[index]),
                validation_fn=int(fn[index]), validation_fp=int(fp[index]), ties=int(len(best)))


def count(y, score, threshold, ratio):
    predicted = score >= threshold
    fn, fp = int(sum((y == 1) & ~predicted)), int(sum((y == 0) & predicted))
    tp, tn = int(sum((y == 1) & predicted)), int(sum((y == 0) & ~predicted))
    return dict(cost=ratio * fn + fp, fn=fn, fp=fp, tp=tp, tn=tn,
                recall=tp / (tp + fn), f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2)


def specifications():
    config = read(CONFIG)
    raw = pd.read_csv(PARENT / "seed_metrics.csv")
    old = pd.read_csv(BUDGET / "seed_metrics.csv")
    jobs = read(BUDGET / "metadata.json")["jobs"]
    job_index = {(j["dataset"], j["seed"], j.get("context")): i for i, j in enumerate(jobs) if j["kind"] == "risk_only"}
    records = []
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            for kernel in config["kernels"]:
                for method in ("baseline", "gbdt_soft_gate", "gbdt_rank_append"):
                    role = "baseline" if method == "baseline" else "proposed"
                    folder = PARENT / "candidates" / dataset / f"seed_{seed:04d}" / kernel / method / "normal"
                    complete = read(folder / "complete.json")
                    choice = complete["choices"][role]
                    assert choice["final_exact"] and choice["fit_rows"] in (800, 4768, 24000)
                    row = raw[raw.dataset.eq(dataset) & raw.seed.eq(seed) & raw.kernel.eq(kernel) & raw.family.eq(method) & raw.role.eq(role) & raw.arm.eq("normal")].iloc[0]
                    records.append(dict(dataset=dataset, seed=seed, kernel=kernel, method=method, kind="kernel",
                        file=str((folder / "exact_margins.npz").relative_to(ROOT)), tune_key=role + "_tune", test_key=role + "_gate",
                        score_index=None, complete=str((folder / "complete.json").relative_to(ROOT)),
                        choice=choice, archived_cost=int(row.cost), archived_fn=int(row.fn), archived_fp=int(row.fp), ascending=False))
                    if method == "baseline":
                        continue
                    context = "sg" if method == "gbdt_soft_gate" else "qra" if kernel == "QSVC" else "rra"
                    index = job_index[dataset, seed, context]
                    folder = BUDGET / "jobs" / f"job_{index:04d}"
                    choice = read(folder / "complete.json")["choice"]
                    row = old[old.dataset.eq(dataset) & old.seed.eq(seed) & old.kernel.eq(kernel) & old.method.eq(method)].iloc[0]
                    records.append(dict(dataset=dataset, seed=seed, kernel=kernel, method=method, kind="risk_only",
                        file=str((folder / "margins.npz").relative_to(ROOT)), tune_key="tune", test_key="gate", score_index=None,
                        complete=str((folder / "complete.json").relative_to(ROOT)), choice=choice,
                        archived_cost=int(row.tuned_risk_only_cost), archived_fn=int(row.risk_only_fn), archived_fp=int(row.risk_only_fp), ascending=True))
            for learner in ("RF", "XGBoost"):
                folder = PARENT / "candidates" / dataset / f"seed_{seed:04d}/reference/reference/normal"
                row = raw[raw.dataset.eq(dataset) & raw.seed.eq(seed) & raw.role.eq("reference_raw_" + learner)].iloc[0]
                complete = read(folder / "complete.json")
                choice = complete["trials"][int(row.score_index)]
                assert choice["score_index"] == row.score_index
                assert np.isclose(choice["threshold"], row.threshold, rtol=0, atol=1e-14)
                records.append(dict(dataset=dataset, seed=seed, kernel=learner, method="raw", kind="reference",
                    file=str((folder / "margins.npz").relative_to(ROOT)), tune_key="tune", test_key="gate", score_index=int(row.score_index),
                    complete=str((folder / "complete.json").relative_to(ROOT)), choice=choice,
                    archived_cost=int(row.cost), archived_fn=int(row.fn), archived_fp=int(row.fp), ascending=False))
    assert len(records) == 480
    assert len({(r["file"], r["tune_key"], r["score_index"]) for r in records}) == 440
    return records


def run():
    assert not OUTPUT.exists(), "Preserve previous attempts; do not overwrite an analysis"
    OUTPUT.mkdir()
    protocol = read(CONFIG)
    specs = specifications()
    inputs = {str(CONFIG.relative_to(ROOT)): sha(CONFIG), str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    for item in specs:
        for name in (item["file"], item["complete"]):
            inputs.setdefault(name, sha(ROOT / name))
    for dataset in protocol["datasets"]:
        for seed in protocol["seeds"]:
            for name in ("split.json", "inputs.npz"):
                path = PARENT / "prepared" / dataset / f"seed_{seed:04d}" / name
                inputs[str(path.relative_to(ROOT))] = sha(path)
    for path in (PARENT / "seed_metrics.csv", BUDGET / "seed_metrics.csv", BUDGET / "metadata.json"):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    write(OUTPUT / "input_manifest.json", inputs)
    write(OUTPUT / "analysis_specifications.json", specs)
    selected, score_archive = [], {}
    for index, spec in enumerate(specs):
        prepared = PARENT / "prepared" / spec["dataset"] / f"seed_{spec['seed']:04d}"
        with np.load(prepared / "inputs.npz", allow_pickle=False) as saved:
            y = saved["y_tune"]
        with np.load(ROOT / spec["file"], allow_pickle=False) as saved:
            scores = saved[spec["tune_key"]]
            if spec["score_index"] is not None:
                scores = scores[spec["score_index"]]
        score_archive[f"{index:04d}_validation_score"] = scores
        score_archive[f"{index:04d}_validation_y"] = y
        for ratio in protocol["ratios"]:
            default = 1 / (ratio + 1) if spec["kind"] == "reference" else 0.
            choice = choose(y, scores, ratio, default, spec["ascending"])
            if ratio == 5:
                assert choice["threshold"] == spec["choice"]["threshold"], (index, choice, spec["choice"])
                assert choice["validation_cost"] == spec["choice"]["tune_cost"]
            selected.append(dict(context_index=index, ratio=ratio, **choice))
    assert len(selected) == 1440
    write(OUTPUT / "selection_lock.json", dict(scope="All validation-only thresholds fixed before reading test labels for this analysis; parent test results already known",
        protocol_sha256=sha(CONFIG), source_sha256=sha(Path(__file__)), selected=selected, fits=0))
    labels = []
    for chunk in pd.read_csv(PARENT / "predictions.csv.gz", chunksize=150000,
                             usecols=["dataset", "seed", "row_id", "y", "kernel", "family", "role", "arm"]):
        keep = chunk.kernel.eq("QSVC") & chunk.family.eq("baseline") & chunk.role.eq("baseline") & chunk.arm.eq("normal")
        labels.append(chunk.loc[keep, ["dataset", "seed", "row_id", "y"]])
    labels = pd.concat(labels)
    write(OUTPUT / "test_evaluation_receipt.json", dict(selection_lock_sha256=sha(OUTPUT / "selection_lock.json"), new_training_or_selection_after_lock=False))
    rows = []
    for index, spec in enumerate(specs):
        split = read(PARENT / "prepared" / spec["dataset"] / f"seed_{spec['seed']:04d}/split.json")["actual"]["test"]
        y = labels[labels.dataset.eq(spec["dataset"]) & labels.seed.eq(spec["seed"])].set_index("row_id").loc[split].y.to_numpy()
        with np.load(ROOT / spec["file"], allow_pickle=False) as saved:
            score = saved[spec["test_key"]]
            if spec["score_index"] is not None:
                score = score[spec["score_index"]]
        score_archive[f"{index:04d}_test_score"] = score
        score_archive[f"{index:04d}_test_y"] = y
        for choice in selected[index * 3:index * 3 + 3]:
            result = count(y, score, choice["threshold"], choice["ratio"])
            if choice["ratio"] == 5:
                assert (result["cost"], result["fn"], result["fp"]) == (spec["archived_cost"], spec["archived_fn"], spec["archived_fp"])
            fixed = count(y, score, spec["choice"]["threshold"], choice["ratio"])
            rows.append(dict(dataset=spec["dataset"], seed=spec["seed"], kernel=spec["kernel"], method=spec["method"], kind=spec["kind"],
                C=spec["choice"].get("C"), geometry=spec["choice"].get("geometry"), risk_strength=spec["choice"].get("lambda"),
                archived_threshold=spec["choice"]["threshold"], fixed_threshold_cost=fixed["cost"], n_test=len(y), **choice, **result))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "seed_metrics.csv", index=False)
    frame.groupby(["dataset", "kernel", "method", "kind", "ratio"])[["cost", "fn", "fp", "recall", "f1", "balanced_accuracy"]].agg(["mean", "std"]).to_csv(OUTPUT / "summary.csv")
    np.savez_compressed(OUTPUT / "frozen_heldout_scores.npz", **score_archive)
    for name, digest in inputs.items():
        assert sha(ROOT / name) == digest
    write(OUTPUT / "validation_receipt.json", dict(status="passed", context_count=480, unique_frozen_score_sources=440,
        ratio5_archived_thresholds_and_counts_reproduced=480, evaluation_rows=1440, new_fits=0, new_hyperparameter_searches=0,
        new_hypothesis_tests=0, all_validation_thresholds_locked_before_test_evaluation=True,
        inputs_unchanged=len(inputs), protocol_sha256=sha(CONFIG), source_sha256=sha(Path(__file__)),
        scope="Conditional fixed-model descriptive sensitivity; no independently reselected ratio-specific models or external confirmation",
        outputs_sha256={p.name: sha(p) for p in OUTPUT.iterdir() if p.is_file()}))
    print(json.dumps(read(OUTPUT / "validation_receipt.json"), indent=2))


def technical_test():
    y = np.array([0, 1, 0, 1, 1, 0])
    scores = np.array([-.5, .2, .2, .8, -.1, .9])
    checks = 0
    for ratio in (2, 5, 10):
        for ascending in (False, True):
            for default in (0., 1 / (ratio + 1)):
                choice = choose(y, scores, ratio, default, ascending)
                cuts = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
                assert count(y, scores, choice["threshold"], ratio)["cost"] == min(count(y, scores, c, ratio)["cost"] for c in cuts)
                checks += 1
    print(json.dumps(dict(status="passed", synthetic_threshold_checks=checks, model_fits=0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--technical-test", action="store_true")
    args = parser.parse_args()
    technical_test() if args.technical_test else run()
