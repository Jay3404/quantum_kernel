"""Post-lock, no-fit verification and reporting of the completed follow-up."""
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
RUN = "cmc_budget_riskonly_20261026"
SOURCE = ROOT / "code/results" / RUN
OUTPUT = SOURCE.with_name(RUN + "_reporting")
PROJECT = ROOT / "manuscript/cmc_risk_conditioned_en_20261025"
DATASETS = [("german_numeric", "German Numeric"), ("south_german", "South German"),
            ("hmeq", "HMEQ"), ("taiwan", "Taiwan")]
METHODS = [("gbdt_soft_gate", "SG"), ("gbdt_rank_append", "RA")]
CONTRASTS = [("baseline9_minus_conditioned9", "baseline9_cost", "conditioned9_cost"),
             ("conditioned9_minus_conditioned180", "conditioned9_cost", "conditioned180_cost"),
             ("tuned_risk_only_minus_conditioned180", "tuned_risk_only_cost", "conditioned180_cost"),
             ("tuned_risk_only_minus_conditioned9", "tuned_risk_only_cost", "conditioned9_cost")]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def threshold_and_counts(y, scores, cut=None):
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    if cut is None:
        candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
        costs = np.array([5 * np.sum((y == 1) & (scores < z)) + np.sum((y == 0) & (scores >= z)) for z in candidates])
        best = candidates[costs == costs.min()]
        cut = float(best[np.argmin(abs(best))])
    predicted = scores >= cut
    fn = int(np.sum((y == 1) & ~predicted))
    fp = int(np.sum((y == 0) & predicted))
    return cut, {"cost": 5 * fn + fp, "fn": fn, "fp": fp}


def independent_check(frame, pairs, meta):
    parent = ROOT / "code/results" / meta["protocol"]["parent_run"]
    lock = read(SOURCE / "selection_lock.json")
    release = read(SOURCE / "test_release.json")
    assert lock["utc"] <= release["utc"]
    assert sha(SOURCE / "selection_lock.json") == release["selection_lock_sha256"]
    labels = []
    for chunk in pd.read_csv(parent / "full/predictions.csv.gz", chunksize=150000,
                             usecols=["dataset", "seed", "row_id", "y", "kernel", "family", "arm", "role"]):
        rows = chunk[chunk.kernel.eq("QSVC") & chunk.family.eq("baseline")
                     & chunk.arm.eq("normal") & chunk.role.eq("baseline")]
        labels.append(rows[["dataset", "seed", "row_id", "y"]])
    labels = pd.concat(labels, ignore_index=True)
    assert len(labels) == 37960 and not labels.duplicated(["dataset", "seed", "row_id"]).any()
    job_index = {}
    for index, job in enumerate(meta["jobs"]):
        key = (job["dataset"], job["seed"], job["kind"],
               job["kernel"] + "/" + job["method"] if job["kind"] == "budget" else job["context"])
        job_index[key] = index
        out = SOURCE / "jobs" / f"job_{index:04d}"
        complete = read(out / "complete.json")
        assert sha(out / "complete.json") == lock["jobs_sha256"][str(index)]
        selection = read(out / "validation_selection.json")
        started = read(out / "started.json")
        assert started["utc"] <= selection["utc"] <= complete["utc"] <= lock["utc"]
        prepared = parent / "full/prepared" / job["dataset"] / f"seed_{job['seed']:04d}"
        with np.load(prepared / "inputs.npz", allow_pickle=False) as saved:
            y_val = saved["y_tune"]
            assert complete["choice"]["fit_rows"] == len(saved["y_train"])
        with np.load(out / "margins.npz", allow_pickle=False) as saved:
            cut, counts = threshold_and_counts(y_val, saved["tune"])
        assert cut == complete["choice"]["threshold"] and counts["cost"] == complete["choice"]["tune_cost"]
        if job["kind"] == "risk_only":
            trials = selection["trials"]
            assert len(trials) == (180 if job["context"] == "rra" else 60)
            assert complete["choice"] == min(trials, key=lambda x: (x["tune_cost"], x["lambda"], x["C"], x["geometry"]))
    for row in frame.itertuples():
        prepared = parent / "full/prepared" / row.dataset / f"seed_{row.seed:04d}"
        split = read(prepared / "split.json")["actual"]["test"]
        y = labels[labels.dataset.eq(row.dataset) & labels.seed.eq(row.seed)].set_index("row_id").loc[split].y.to_numpy()
        if row.dataset in ("german_numeric", "south_german"):
            out = parent / "full/candidates" / row.dataset / f"seed_{row.seed:04d}" / row.kernel / row.method / "normal"
            choices = [v for v in read(out / "complete.json")["trials"] if v["lambda"] == .5]
            assert len(choices) == 9
            choice = min(choices, key=lambda x: (x["tune_cost"], x["lambda"], x["C"], x["geometry"]))
            with np.load(out / "margins.npz") as saved:
                score = saved["gate"][choice["score_index"]]
        else:
            index = job_index[(row.dataset, row.seed, "budget", row.kernel + "/" + row.method)]
            out = SOURCE / "jobs" / f"job_{index:04d}"
            choice = read(out / "complete.json")["choice"]
            with np.load(out / "margins.npz") as saved:
                score = saved["gate"]
        _, counts = threshold_and_counts(y, score, choice["threshold"])
        assert counts == {key: getattr(row, "budget_" + key) for key in counts}
        assert counts["cost"] == row.conditioned9_cost
        context = "sg" if row.method == "gbdt_soft_gate" else ("qra" if row.kernel == "QSVC" else "rra")
        index = job_index[(row.dataset, row.seed, "risk_only", context)]
        out = SOURCE / "jobs" / f"job_{index:04d}"
        choice = read(out / "complete.json")["choice"]
        with np.load(out / "margins.npz") as saved:
            _, counts = threshold_and_counts(y, saved["gate"], choice["threshold"])
        assert counts == {key: getattr(row, "risk_only_" + key) for key in counts}
        assert counts["cost"] == row.tuned_risk_only_cost
    raw = []
    for row in pairs.itertuples():
        selected = frame[frame.dataset.eq(row.dataset) & frame.kernel.eq(row.kernel) & frame.method.eq(row.method)].sort_values("seed")
        assert selected.seed.tolist() == list(range(10, 101, 10))
        _, left, right = next(v for v in CONTRASTS if v[0] == row.contrast)
        delta = selected[left].to_numpy() - selected[right].to_numpy()
        sd, mean = delta.std(ddof=1), delta.mean()
        error = np.sqrt(.225) * sd
        p = 2 * t.sf(abs(mean / error), 9) if error else 1.
        np.testing.assert_allclose([mean, sd, p], [row.mean_difference, row.sd_difference, row.p_value], atol=1e-12)
        if error:
            np.testing.assert_allclose([mean - t.ppf(.975, 9) * error, mean + t.ppf(.975, 9) * error],
                                       [row.ci_low, row.ci_high], atol=1e-10)
        assert (int((delta > 0).sum()), int((delta == 0).sum()), int((delta < 0).sum())) == (row.wins, row.ties, row.losses)
        raw.append(p)
    ordered = sorted(range(64), key=lambda i: (raw[i], i))
    adjusted, previous = np.empty(64), 0.
    for rank, index in enumerate(ordered):
        previous = max(previous, min(1., (64 - rank) * raw[index]))
        adjusted[index] = previous
    np.testing.assert_allclose(adjusted, pairs.p_holm64, atol=1e-12)
    probes = [read(p) for p in (SOURCE / "preflight").glob("*/complete.json")]
    assert len(probes) == 11 and all(p["utc"] <= meta["scientific_start_utc"] for p in probes)
    assert sum(read(p)["model_fit_calls"] for p in (SOURCE / "jobs").glob("*/complete.json")) == 12080
    return {"validation_thresholds_recomputed": 200, "test_metric_rows_recomputed": 160,
            "paired_statistics_recomputed": 64, "holm64_recomputed": True, "new_fits": 0,
            "all200_complete_hashes_match_global_lock": True,
            "all200_starts_selections_and_completions_precede_global_lock": True,
            "preflight11_precedes_main_start": True, "scientific_candidate_fit_calls_excluding_preflight": 12080,
            "launch_metadata_test_labels_read_field_is_not_used_as_a_live_release_flag": True}


def latex_table(path, caption, label, headings, rows):
    text = ["% Generated from validated completed follow-up; not new fitting.", "\\begin{table}[H]", "\\centering",
            "\\caption{" + caption + "}", "\\label{" + label + "}", "\\begingroup", "\\footnotesize",
            "\\setlength{\\tabcolsep}{3pt}", "\\begin{adjustbox}{max width=\\linewidth}",
            "\\begin{tabular}{" + "l" * len(headings) + "}", "\\toprule", " & ".join(headings) + "\\\\", "\\midrule"]
    text += [" & ".join(row) + "\\\\" for row in rows]
    text += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\endgroup", "\\end{table}", ""]
    path.write_text("\n".join(text))


def main():
    OUTPUT.mkdir(exist_ok=True)
    meta = read(SOURCE / "metadata.json")
    validation = read(SOURCE / "validation_receipt.json")
    assert meta["status"] == "completed" and validation["status"] == "passed"
    assert read(SOURCE / "operations/explicit_validation_20261026.json")["exit_code"] == 0
    manifest_root = SOURCE.with_name(RUN + "_completion_audit")
    manifest = read(manifest_root / "server_content_manifest.json")
    assert all(sha(ROOT / name) == row["sha256"] and (ROOT / name).stat().st_size == row["bytes"] for name, row in manifest.items())
    assert all(sha(ROOT / name) == value for name, value in {**meta["source_sha256"], **meta["input_sha256"]}.items())
    frame = pd.read_csv(SOURCE / "seed_metrics.csv")
    pairs = pd.read_csv(SOURCE / "paired_statistics_Holm64.csv")
    assert len(frame) == 160 and len(pairs) == 64
    independent = independent_check(frame, pairs, meta)
    summaries, costs, budget_pairs, risk_pairs = [], [], [], []
    for (dataset, title), kernel, (method, short) in itertools.product(DATASETS, ("RBF-SVM", "QSVC"), METHODS):
        rows = frame[frame.dataset.eq(dataset) & frame.kernel.eq(kernel) & frame.method.eq(method)].sort_values("seed")
        record = {"dataset": dataset, "kernel": kernel, "method": method}
        cells = []
        for key in ("baseline9_cost", "conditioned9_cost", "conditioned180_cost", "tuned_risk_only_cost"):
            record[key + "_mean"], record[key + "_sd"] = rows[key].mean(), rows[key].std(ddof=1)
            cells.append(f"${record[key + '_mean']:.1f}\\pm{record[key + '_sd']:.1f}$")
        record["budget_improvement_pct_ratio_of_means"] = 100 * (1 - rows.conditioned9_cost.mean() / rows.baseline9_cost.mean())
        record["budget_improvement_pct_mean_seed"] = (100 * (rows.baseline9_cost - rows.conditioned9_cost) / rows.baseline9_cost).mean()
        summaries.append(record)
        costs.append([title, kernel, short, *cells])
        for contrast, table in [("baseline9_minus_conditioned9", budget_pairs),
                                ("tuned_risk_only_minus_conditioned180", risk_pairs)]:
            value = pairs[pairs.dataset.eq(dataset) & pairs.kernel.eq(kernel) & pairs.method.eq(method) & pairs.contrast.eq(contrast)].iloc[0]
            table.append([title, kernel, short, f"{value.mean_difference:.1f}",
                          f"$[{value.ci_low:.2f},{value.ci_high:.2f}]$", f"{value.wins}/{value.ties}/{value.losses}",
                          f"{value.p_value:.4f}", f"{value.p_holm64:.4f}"])
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "cost_summary.csv", index=False)
    latex_table(PROJECT / "tables/budget_tuned_risk_costs.tex",
                "Restricted-search and independently tuned Risk-only test costs (mean $\\pm$ sample SD over 10 seeds). Baseline9 and conditioned9 each search 9 candidates; conditioned9 fixes $\\lambda=.5$. Full180 is the original positive-risk search. Tuned Risk-only uses its own validation selection; SG is shared between parent contexts.",
                "tab:budget-risk-costs", ["Dataset", "Kernel", "Method", "Baseline9", "Conditioned9", "Full180", "Tuned risk-only"], costs)
    latex_table(PROJECT / "tables/budget_paired.tex",
                "Baseline9 cost minus conditioned9 cost. Positive differences favor risk conditioning. Intervals are individual approximate 95\\% intervals; adjusted values use all 64 new contrasts, not this displayed subset. W/T/L denotes lower/equal/higher conditioned cost.",
                "tab:budget-pairs", ["Dataset", "Kernel", "Method", "Difference", "Individual 95\\% CI", "W/T/L", "Raw $p$", "Holm64"], budget_pairs)
    latex_table(PROJECT / "tables/tuned_risk_paired.tex",
                "Independently tuned Risk-only cost minus original Full180 cost. Positive differences favor the full product. Intervals are individual approximate 95\\% intervals and the adjustment includes all 64 new contrasts. These are not the inherited-setting Holm16 component ablations.",
                "tab:tuned-risk-pairs", ["Dataset", "Kernel", "Method", "Difference", "Individual 95\\% CI", "W/T/L", "Raw $p$", "Holm64"], risk_pairs)
    paired_rows = []
    for row in frame.itertuples(index=False):
        values = row._asdict()
        for name, left, right in CONTRASTS:
            paired_rows.append({"dataset": row.dataset, "seed": row.seed, "kernel": row.kernel,
                                "method": row.method, "contrast": name, "left_cost": values[left],
                                "right_cost": values[right], "paired_difference": values[left] - values[right]})
    pd.DataFrame(paired_rows).to_csv(OUTPUT / "paired_seed_differences.csv", index=False)
    classification = []
    for key, values in frame.groupby(["dataset", "kernel", "method"]):
        for prefix in ("budget", "risk_only"):
            item = {"dataset": key[0], "kernel": key[1], "method": key[2], "role": prefix}
            for metric in ("cost", "fn", "fp", "recall", "f1", "balanced_accuracy", "auroc", "ap"):
                item[metric + "_mean"] = values[prefix + "_" + metric].mean()
                item[metric + "_sd"] = values[prefix + "_" + metric].std(ddof=1)
            classification.append(item)
    pd.DataFrame(classification).to_csv(OUTPUT / "classification_summary.csv", index=False)
    hours = (datetime.fromisoformat(meta["finished_utc"]) - datetime.fromisoformat(meta["scientific_start_utc"])).total_seconds() / 3600
    receipt = {"status": "passed", "scope": "Independent no-fit count/threshold/statistic replay and local/server content comparison",
               **independent, "local_server_files": len(manifest), "local_server_bytes": sum(v["bytes"] for v in manifest.values()),
               "server_manifest_sha256": sha(manifest_root / "server_content_manifest.json"),
               "source_sha256": sha(Path(__file__)), "campaign_hours_including_automatic_audit": hours,
               "all_holm64_below_05_count": int((pairs.p_holm64 < .05).sum()), "minimum_holm64": float(pairs.p_holm64.min()),
               "statistical_fallacy_scan": {"coverage": "11/11 considered; not a guarantee of causal validity",
                   "Simpson": "No pooled cross-dataset cost effect; report each representation/kernel/method",
                   "Ecological": "Seed-level comparisons are not customer-level causal claims",
                   "Berkson": "Historical dataset selection and generalization remain unmeasured limitations",
                   "Collider": "No causal adjustment/mediation model is fitted",
                   "Base_rate": "Stratified class prevalence and cohort size retained; costs not pooled across datasets",
                   "Regression_to_mean": "HMEQ was previously observed as favorable; follow-up is explicitly exploratory",
                   "Survivorship": "All200 planned jobs and64 contrasts retained, zero scientific failures",
                   "Look_elsewhere": "All64 contrasts retained; no family reduction after outcomes",
                   "Forking_paths": "Protocol locked before new fits but repeated prior data examination remains",
                   "Correlation_causation": "Geometry associations descriptive only; no causal contribution percentages",
                   "Reverse_causality": "No causal direction inferred from diagnostic correlations"},
               "submission_or_independent_external_confirmation": False,
               "self_reference_exclusion": "Receipt does not include its own hash"}
    (OUTPUT / "independent_verification_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    evidence = read(PROJECT / "evidence/input_manifest.json")
    copies = [(SOURCE / "seed_metrics.csv", "budget_riskonly_seed_metrics.csv"),
              (SOURCE / "paired_statistics_Holm64.csv", "budget_riskonly_pairs_Holm64.csv"),
              (SOURCE / "validation_receipt.json", "budget_riskonly_server_validation.json"),
              (ROOT / "docs/cmc_budget_riskonly_protocol_20261026_v2.json", "budget_riskonly_protocol.json"),
              (OUTPUT / "cost_summary.csv", "budget_riskonly_cost_summary.csv"),
              (OUTPUT / "paired_seed_differences.csv", "budget_riskonly_paired_seed_differences.csv"),
              (OUTPUT / "classification_summary.csv", "budget_riskonly_classification_summary.csv"),
              (OUTPUT / "independent_verification_receipt.json", "budget_riskonly_local_verification.json")]
    for original, name in copies:
        destination = PROJECT / "evidence" / name
        shutil.copy2(original, destination)
        evidence[str(original.relative_to(ROOT))] = {"copy": str(destination.relative_to(PROJECT)), "sha256": sha(original)}
    (PROJECT / "evidence/input_manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
