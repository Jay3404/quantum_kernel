"""Reconstruct existing the selected-model component summary components from the supplied runtime CSV."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output_dir.resolve().is_relative_to(ROOT.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (ROOT / "results/runtime/seed_components.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    svm = [row for row in rows if row["kernel"] in {"RBF-SVM", "QSVC"}]
    for row in svm:
        for part, record in (("train", "training"), ("test", "test")):
            actual = float(row[f"cold_{record}_component_sum_s"])
            expected = float(row[f"selected_{record}_recorded_s"]) - float(row[f"state_{part}_incremental_s"]) + float(row[f"state_{part}_cold_s"])
            assert math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12)
    fields = ["fit_s", "state_train_cold_s", "cold_training_component_sum_s", "cold_test_component_sum_s"]
    expected_rows = [
        ["0.328", "0.000", "1.265", "0.128"], ["0.345", "0.000", "0.583", "0.053"],
        ["0.374", "1.403", "2.715", "0.287"], ["0.337", "1.396", "2.133", "0.219"],
        ["9.091", "0.000", "32.711", "3.502"], ["8.657", "0.000", "16.127", "1.619"],
        ["10.092", "10.221", "45.956", "4.710"], ["10.728", "9.898", "30.833", "3.226"]]
    summary = []
    for dataset in ("hmeq", "taiwan"):
        for kernel in ("RBF-SVM", "QSVC"):
            for method in ("gbdt_soft_gate", "gbdt_rank_append"):
                group = [r for r in svm if r["dataset"] == dataset and r["kernel"] == kernel and r["method"] == method]
                assert sorted(int(r["seed"]) for r in group) == list(range(10, 101, 10))
                values = [statistics.mean(float(r[field]) for r in group) for field in fields]
                assert [f"{value:.3f}" for value in values] == expected_rows[len(summary)]
                summary.append(dict(dataset=dataset, kernel=kernel, method=method, **dict(zip(fields, values))))
    with (args.output_dir / "table_a19_reconstructed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    receipt = dict(status="passed", selected_svm_records_checked=len(svm), formula_identities_checked=2*len(svm),
        table_rows_reproduced=8, table_numeric_cells_reproduced=32, new_timing_measurements=0, new_fits=0)
    (args.output_dir / "runtime_replay_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
