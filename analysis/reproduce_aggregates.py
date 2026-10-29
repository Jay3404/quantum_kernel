"""Recompute supplied model/seed summaries; never fit models or read customer data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"Holm184": 184, "Holm100": 100, "Holm40": 40, "Holm16": 16,
            "Holm64": 64, "Holm138": 138, "Holm90": 90}
SEEDS = list(range(10, 101, 10))


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted


def corrected_summary(differences):
    values = np.asarray(differences, dtype=float)
    assert len(values) == 10 and np.isfinite(values).all()
    mean, sd = values.mean(), values.std(ddof=1)
    se = np.sqrt(.225) * sd
    raw = 2 * t.sf(abs(mean / se), 9) if se else 1.
    interval = (mean - t.ppf(.975, 9) * se, mean + t.ppf(.975, 9) * se) if se else (np.nan, np.nan)
    return dict(mean_difference=mean, sd_difference=sd, raw_p=raw,
                ci_low=interval[0], ci_high=interval[1], wins=int(sum(values > 0)),
                ties=int(sum(values == 0)), losses=int(sum(values < 0)))


def verify_manifest(root):
    manifest = json.loads((root / "PUBLIC_MANIFEST.json").read_text())
    for name, item in manifest["files"].items():
        path = root / name
        assert path.stat().st_size == item["bytes"], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], name
    return len(manifest["files"])


def replay(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    assert not output.is_relative_to(root / "results"), "Do not overwrite supplied results"
    output.mkdir(parents=True, exist_ok=True)
    checked_files = verify_manifest(root)
    expected = pd.read_csv(root / "results/statistics/paired_statistics.csv")
    deltas = pd.read_csv(root / "results/statistics/paired_seed_differences.csv")
    assert len(expected) == 632 and len(deltas) == 6320
    assert expected.comparison_id.is_unique
    assert set(expected.adjustment_family) == set(FAMILIES)
    computed = []
    for row in expected.itertuples():
        group = deltas[deltas.comparison_id.eq(row.comparison_id)].sort_values("seed")
        assert group.seed.tolist() == SEEDS
        summary = corrected_summary(group.delta.to_numpy())
        for name in ("mean_difference", "sd_difference", "raw_p", "ci_low", "ci_high"):
            np.testing.assert_allclose(summary[name], getattr(row, name), atol=1e-9, rtol=1e-7, equal_nan=True)
        for name in ("wins", "ties", "losses"):
            assert summary[name] == getattr(row, name)
        computed.append(dict(comparison_id=row.comparison_id, adjustment_family=row.adjustment_family, **summary))
    computed = pd.DataFrame(computed)
    for family, size in FAMILIES.items():
        selection = computed.adjustment_family.eq(family)
        assert int(selection.sum()) == size
        adjusted = holm(computed.loc[selection, "raw_p"])
        np.testing.assert_allclose(adjusted, expected.loc[selection, "adjusted_p"], atol=1e-9, rtol=1e-9)
        computed.loc[selection, "adjusted_p"] = adjusted
    computed.to_csv(output / "paired_statistics.csv", index=False)

    geometry = pd.read_csv(root / "results/geometry/matched_cases.csv")
    assert len(geometry) == 160
    assert int((geometry.alignment_actual > geometry.alignment_shuffled_matched).sum()) == 151
    assert int((geometry.effective_rank_shuffled_matched > geometry.effective_rank_actual).sum()) == 159
    keys = ["dataset", "kernel", "method"]
    columns = pd.read_csv(root / "results/geometry/matched_summary.csv").columns.tolist()
    summary = geometry.groupby(keys)[[c for c in columns if c not in keys]].mean().reset_index()
    saved = pd.read_csv(root / "results/geometry/matched_summary.csv").sort_values(keys).reset_index(drop=True)
    summary = summary.sort_values(keys).reset_index(drop=True)
    np.testing.assert_allclose(summary.drop(columns=keys), saved.drop(columns=keys), atol=1e-10, rtol=1e-10)
    summary.to_csv(output / "matched_geometry_summary.csv", index=False)
    backbone = pd.read_csv(root / "results/geometry/backbone_cases.csv")
    assert len(backbone) == 160
    assert int((backbone.alignment_delta > 0).sum()) == 158
    assert int((backbone.full_cost_reduction < 0).sum()) == 28
    assert (backbone.loc[backbone.full_cost_reduction < 0, "alignment_delta"] > 0).all()

    sensitivity = pd.read_csv(root / "results/cost_sensitivity/seed_metrics.csv")
    assert len(sensitivity) == 1440 and sensitivity.context_index.nunique() == 480
    assert set(sensitivity.ratio) == {2, 5, 10}
    np.testing.assert_array_equal(sensitivity.cost, sensitivity.ratio * sensitivity.fn + sensitivity.fp)
    groups = ["dataset", "kernel", "method", "kind", "ratio"]
    for _, group in sensitivity.groupby(groups):
        assert group.seed.tolist() == SEEDS
    regenerated = sensitivity.groupby(groups).cost.agg(["mean", "std"]).reset_index()
    supplied = pd.read_csv(root / "results/cost_sensitivity/cost_summary.csv")
    pd.testing.assert_frame_equal(regenerated[groups], supplied[groups])
    np.testing.assert_allclose(regenerated[["mean", "std"]], supplied[["mean", "std"]], atol=1e-9, rtol=1e-9)
    regenerated.to_csv(output / "cost_ratio_summary.csv", index=False)
    receipt = dict(status="passed", public_file_hashes_checked=checked_files,
                   paired_comparisons=632, paired_seed_differences=6320,
                   original_holm_families=FAMILIES, matched_geometry_cases=160,
                   fixed_model_sensitivity_rows=1440, new_fits=0, new_tests=0,
                   customer_scores_read=False, threshold_optimization_replayed=False)
    (output / "replay_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(ROOT, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
