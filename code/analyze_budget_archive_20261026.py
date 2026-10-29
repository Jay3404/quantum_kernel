"""Scoped archive-only nine-candidate replay; never trains scientific models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_pca7_professor_comments import metrics, sha, threshold

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/cmc_budget_riskonly_protocol_20261026.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def choose(trials, strength):
    rows = [r for r in trials if np.isclose(r["lambda"], strength, atol=1e-12, rtol=0)]
    assert len(rows) == 9
    assert {(r["C"], r["geometry"]) for r in rows} == {(c, g) for c in (.01, 1., 100.) for g in range(3)}
    return min(rows, key=lambda r: (r["tune_cost"], r["lambda"], r["C"], r["geometry"]))


def candidates(parent, dataset, seed, kernel, method):
    return parent / "full/candidates" / dataset / f"seed_{seed:04d}" / kernel / method / "normal"


def target_labels(parent):
    pieces = []
    cols = ["dataset", "seed", "row_id", "y", "kernel", "family", "arm", "role"]
    for chunk in pd.read_csv(parent / "full/predictions.csv.gz", usecols=cols, chunksize=100000):
        keep = chunk.dataset.isin(["german_numeric", "south_german"]) & chunk.kernel.eq("QSVC") & chunk.family.eq("baseline") & chunk.arm.eq("normal") & chunk.role.eq("baseline")
        pieces.append(chunk.loc[keep, ["dataset", "seed", "row_id", "y"]])
    frame = pd.concat(pieces, ignore_index=True)
    assert len(frame) == 2000 and not frame.duplicated(["dataset", "seed", "row_id"]).any()
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "code/results/cmc_budget_archive_20261026")
    args = parser.parse_args()
    output = args.output
    assert not output.exists(), "Preserve previous analysis/failures; no silent retry"
    config = json.loads(PROTOCOL.read_text())
    parent = ROOT / "code/results" / config["parent_run"]
    assert json.loads((parent / "validation_receipt.json").read_text())["status"] == "passed"
    output.mkdir(parents=True)
    bindings = {str(PROTOCOL.relative_to(ROOT)): sha(PROTOCOL),
                str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    plan = []
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            prepared = parent / "full/prepared" / dataset / f"seed_{seed:04d}"
            for name in ("inputs.npz", "split.json", "audit.json"):
                bindings[str((prepared / name).relative_to(ROOT))] = sha(prepared / name)
            for kernel in config["kernels"]:
                for method in ("baseline", *config["methods"]):
                    folder = candidates(parent, dataset, seed, kernel, method)
                    complete = json.loads((folder / "complete.json").read_text())
                    strength = 0. if method == "baseline" else config["budget_control"]["fixed_lambda"]
                    choice = choose(complete["trials"], strength)
                    assert choice["approximate_search"] == (dataset in ("hmeq", "taiwan"))
                    assert sha(folder / "margins.npz") == complete["margins_sha256"]
                    for name in ("complete.json", "margins.npz"):
                        bindings[str((folder / name).relative_to(ROOT))] = sha(folder / name)
                    if dataset in ("german_numeric", "south_german"):
                        with np.load(prepared / "inputs.npz", allow_pickle=False) as stored:
                            y_tune = stored["y_tune"]
                        with np.load(folder / "margins.npz", allow_pickle=False) as stored:
                            tune = stored["tune"][choice["score_index"]]
                        cut = threshold(y_tune, tune)
                        assert cut == choice["threshold"]
                        assert metrics(y_tune, tune, cut)["cost"] == choice["tune_cost"]
                    plan.append({"dataset": dataset, "seed": seed, "kernel": kernel, "method": method,
                                 "folder": str(folder.relative_to(ROOT)), "choice": choice,
                                 "n_candidates": 9,
                                 "final_exact_test_available": dataset in ("german_numeric", "south_german")})
    assert len(plan) == 240
    write(output / "selection_lock.json", {"utc": now(), "choices": plan, "test_labels_read": False,
          "sources_and_inputs_sha256": bindings, "new_fits": 0,
          "large_dataset_choices_need_new_exact_refit": True})
    label_file = parent / "full/predictions.csv.gz"
    write(output / "test_release_german_only.json", {"utc": now(), "labels_sha256": sha(label_file),
          "selection_lock_sha256": sha(output / "selection_lock.json"),
          "scope": "Archived exact German scores only; no large proxy-test evaluation"})
    labels = target_labels(parent)
    rows = []
    for item in plan:
        if not item["final_exact_test_available"]:
            continue
        d, s = item["dataset"], item["seed"]
        split = json.loads((parent / "full/prepared" / d / f"seed_{s:04d}/split.json").read_text())["actual"]
        label_rows = labels[labels.dataset.eq(d) & labels.seed.eq(s)].set_index("row_id")
        y = label_rows.reindex(split["test"]).y.to_numpy(dtype=int)
        with np.load(ROOT / item["folder"] / "margins.npz", allow_pickle=False) as stored:
            margin = stored["gate"][item["choice"]["score_index"]]
        choice = item["choice"]
        rows.append({k: item[k] for k in ("dataset", "seed", "kernel", "method")} |
                    {"n_candidates": 9, "C": choice["C"], "lambda": choice["lambda"], "geometry": choice["geometry"],
                     "threshold": choice["threshold"], "validation_cost": choice["tune_cost"],
                     **metrics(y, margin, choice["threshold"])})
    frame = pd.DataFrame(rows)
    assert len(frame) == 120
    frame.to_csv(output / "german_seed_metrics.csv", index=False)
    original = pd.read_csv(parent / "full/seed_metrics.csv")
    summaries = []
    for (d, k, method), group in frame[~frame.method.eq("baseline")].groupby(["dataset", "kernel", "method"]):
        group = group.sort_values("seed")
        before = frame[frame.dataset.eq(d) & frame.kernel.eq(k) & frame.method.eq("baseline")].sort_values("seed")
        full = original[original.dataset.eq(d) & original.kernel.eq(k) & original.family.eq(method) & original.arm.eq("normal") & original.role.eq("proposed")].sort_values("seed")
        assert group.seed.tolist() == before.seed.tolist() == full.seed.tolist() == config["seeds"]
        diff = before.cost.to_numpy() - group.cost.to_numpy()
        summaries.append({"dataset": d, "kernel": k, "method": method,
                          "baseline9_mean": before.cost.mean(), "baseline9_sd": before.cost.std(ddof=1),
                          "conditioned9_mean": group.cost.mean(), "conditioned9_sd": group.cost.std(ddof=1),
                          "conditioned180_mean": full.cost.mean(), "conditioned180_sd": full.cost.std(ddof=1),
                          "baseline9_minus_conditioned9": diff.mean(),
                          "conditioned9_minus_conditioned180": (group.cost.to_numpy() - full.cost.to_numpy()).mean(),
                          "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()), "losses": int((diff < 0).sum()),
                          "n_seeds": 10, "inference_status": "Holm64 pending full planned follow-up; descriptive only"})
    pd.DataFrame(summaries).to_csv(output / "german_descriptive_summary.csv", index=False)
    assert all(sha(ROOT / name) == digest for name, digest in bindings.items())
    write(output / "verification_receipt.json", {"utc": now(), "status": "passed",
          "scope": "Archive-only exact German replay: 120 evaluated rows,8 conditioned comparison cells; not all16 cells or independent risk-only tuning",
          "new_fits": 0, "validation_choices_locked_before_scoped_label_release": True,
          "test_labels_not_used_for_selection": True, "parent_files_unchanged": True,
          "sources_and_inputs_sha256": bindings,
          "outputs_sha256": {name: sha(output / name) for name in ("german_seed_metrics.csv", "german_descriptive_summary.csv", "selection_lock.json")},
          "new_adjusted_p_values_computed": False, "exploratory": True})
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
