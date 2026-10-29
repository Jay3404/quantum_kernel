"""No-fit matched-risk geometry reconstruction; immutable parent inputs."""
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analyze_pca7_professor_comments import backbone, kernel_metrics, risk_factor

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/cmc_matched_geometry_protocol_20261028.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def permute(arrays, seed):
    result = {}
    for index, part in enumerate(("train", "tune", "gate")):
        q = arrays["gbdt_rank_" + part]
        order = np.random.default_rng(seed + 7919 + index).permutation(len(q))
        result[part] = q[order]
        np.testing.assert_array_equal(np.sort(result[part]), np.sort(q))
    return result


def technical_checks():
    q = np.linspace(0, 1, 12)
    labels = np.arange(12) % 2
    base = np.exp(-((q[:, None] - q[None, :]) ** 2))
    for learner in ("RBF-SVM", "QSVC"):
        for method in ("gbdt_soft_gate", "gbdt_rank_append"):
            for strength in (0., .5, 1.):
                factor = risk_factor(q, strength, learner, method)
                metric = kernel_metrics(base * factor, labels)
                assert metric["entropy_effective_rank"] >= kernel_metrics(base, labels)["entropy_effective_rank"] - 1e-9
                if not strength:
                    np.testing.assert_allclose(factor, 1.)
    a = {"gbdt_rank_" + p: q for p in ("train", "tune", "gate")}
    x, y = permute(a, 10), permute(a, 10)
    for p in x:
        np.testing.assert_array_equal(x[p], y[p])
    return {"product_cases": 12, "deterministic_full_split_permutation": True}


def main():
    start = time.perf_counter()
    checks = technical_checks()
    config = json.loads(PROTOCOL.read_text())
    output = ROOT / "code/results" / config["analysis_id"]
    output.mkdir(exist_ok=True)
    assert not (output / "matched_geometry_cases.csv").exists(), "Preserve completed analysis"
    inputs = {}

    def consume(path):
        path = Path(path)
        key = path.relative_to(ROOT).as_posix()
        value = digest(path)
        assert key not in inputs or inputs[key] == value, key
        inputs[key] = value
        return path

    def read(path):
        return json.loads(consume(path).read_text())

    consume(PROTOCOL)
    consume(Path(__file__))
    consume(ROOT / "code/analyze_pca7_professor_comments.py")
    parent = ROOT / "code/results" / config["parent_run"]
    full = parent / "full"
    meta = read(parent / "metadata.json")
    assert meta["status"] == "completed" and meta["completed_jobs"] == 1160 and not meta["failed_jobs"]
    assert read(parent / "validation_receipt.json")["status"] == "passed"
    lock = read(full / "selection_lock.json")
    release = read(full / "test_release.json")
    assert release["no_further_fitting"] and release["selection_lock_sha256"] == digest(full / "selection_lock.json")
    geometries = read(ROOT / "docs/cmc_pca7_repeated_protocol_20261020.json")["geometries"]
    prior = pd.read_csv(consume(ROOT / "code/results/cmc_professor_revision_20261023/geometry_seed_metrics.csv"))
    parts = []
    for chunk in pd.read_csv(consume(full / "predictions.csv.gz"), chunksize=100000):
        keep = chunk.kernel.eq("QSVC") & chunk.family.eq("baseline") & chunk.arm.eq("normal") & chunk.role.eq("baseline")
        parts.append(chunk.loc[keep, ["dataset", "seed", "row_id", "y"]])
    labels = pd.concat(parts, ignore_index=True)
    assert not labels.duplicated(["dataset", "seed", "row_id"]).any()
    rows = []
    grams = {}
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            prepared = full / "prepared" / dataset / f"seed_{seed:04d}"
            audit = read(prepared / "audit.json")
            array_path = consume(prepared / "inputs.npz")
            assert digest(array_path) == audit["input_sha256"]
            with np.load(array_path) as archive:
                arrays = {key: archive[key] for key in archive.files}
            split = read(prepared / "split.json")["actual"]
            y = labels[labels.dataset.eq(dataset) & labels.seed.eq(seed)].set_index("row_id").reindex(split["test"]).y.to_numpy(dtype=int)
            assert len(y) == len(arrays["gate"]) and np.isin(y, [0, 1]).all()
            shuffled = permute(arrays, seed)
            cache = {}
            for kernel in config["kernels"]:
                for method in config["methods"]:
                    folder = full / "candidates" / dataset / f"seed_{seed:04d}" / kernel / method / "normal"
                    complete_path = folder / "complete.json"
                    done = read(complete_path)
                    assert digest(complete_path) == lock["candidate_receipts_sha256"][complete_path.relative_to(ROOT).as_posix()]
                    choice = done["choices"]["proposed"]
                    positions = np.array(done["geometry_positions"]["exact_proposed"])
                    key = (kernel, choice["geometry"], tuple(positions))
                    if key not in cache:
                        cache[key] = backbone(arrays, audit, positions, kernel, geometries[choice["geometry"]])
                    base, gamma = cache[key]
                    geometry_path = consume(folder / "geometry.npz")
                    assert digest(geometry_path) == done["geometry_sha256"]
                    with np.load(geometry_path) as archive:
                        saved = archive["exact_proposed"]
                    actual = base * risk_factor(arrays["gbdt_rank_gate"][positions], choice["lambda"], kernel, method, gamma)
                    error = float(np.max(np.abs(actual - saved)))
                    assert error < 1e-8
                    control = base * risk_factor(shuffled["gate"][positions], choice["lambda"], kernel, method, gamma)
                    metrics = [kernel_metrics(k, y[positions]) for k in (base, actual, control)]
                    previous = prior[prior.dataset.eq(dataset) & prior.seed.eq(seed) & prior.kernel.eq(kernel) & prior.method.eq(method)]
                    assert len(previous) == 1
                    row = {"dataset": dataset, "seed": seed, "kernel": kernel, "method": method,
                           "C": choice["C"], "geometry": choice["geometry"], "lambda": choice["lambda"],
                           "n_subset": len(positions), "actual_gram_max_error": error,
                           "positions_sha256": hashlib.sha256(positions.astype("<i8").tobytes()).hexdigest(),
                           "full_split_rank_multiset_preserved": True}
                    for field, metric in (("alignment", "centered_label_alignment"), ("effective_rank", "entropy_effective_rank")):
                        before, after, permuted = [m[metric] for m in metrics]
                        for name, value in (("base", before), ("actual", after), ("shuffled_matched", permuted),
                                            ("actual_delta", after - before), ("shuffled_delta", permuted - before),
                                            ("actual_minus_shuffled", after - permuted)):
                            row[field + "_" + name] = value
                        assert np.isclose(before, previous.iloc[0][field + "_before"], atol=1e-9, rtol=1e-9)
                        assert np.isclose(after, previous.iloc[0][field + "_after"], atol=1e-9, rtol=1e-9)
                        assert after >= before - 1e-8 if field == "effective_rank" else True
                        assert permuted >= before - 1e-8 if field == "effective_rank" else True
                    case = f"{dataset}__{seed}__{kernel}__{method}"
                    grams[case + "__base"] = base
                    grams[case + "__actual"] = actual
                    grams[case + "__shuffled"] = control
                    rows.append(row)
            print(json.dumps({"dataset": dataset, "seed": seed, "cases": len(rows)}), flush=True)
    frame = pd.DataFrame(rows)
    assert len(frame) == config["expected_cases"] == 160
    frame.to_csv(output / "matched_geometry_cases.csv", index=False)
    numeric = [c for c in frame if c.startswith(("alignment_", "effective_rank_"))]
    summary = frame.groupby(["dataset", "kernel", "method"])[numeric].mean().reset_index()
    summary.to_csv(output / "matched_geometry_summary.csv", index=False)
    np.savez_compressed(output / "reconstructed_grams.npz", **grams)
    for name, value in inputs.items():
        assert digest(ROOT / name) == value, "Input changed: " + name
    (output / "input_manifest.json").write_text(json.dumps(inputs, indent=2) + "\n")
    receipt = {"status": "passed", "cases": len(frame), "cells": len(summary),
               "scientific_fits_started": 0, "new_thresholds_selected": 0, "new_hypothesis_tests": 0,
               "technical_checks": checks, "actual_gram_max_error": float(frame.actual_gram_max_error.max()),
               "actual_alignment_increases": int((frame.alignment_actual_delta > 1e-10).sum()),
               "shuffled_alignment_increases": int((frame.alignment_shuffled_delta > 1e-10).sum()),
               "actual_alignment_above_shuffled": int((frame.alignment_actual_minus_shuffled > 1e-10).sum()),
               "actual_effective_rank_increases": int((frame.effective_rank_actual_delta > 1e-10).sum()),
               "shuffled_effective_rank_increases": int((frame.effective_rank_shuffled_delta > 1e-10).sum()),
               "actual_effective_rank_above_shuffled": int((frame.effective_rank_actual_minus_shuffled > 1e-10).sum()),
               "original_actual_metrics_reproduced": 160, "inputs_unchanged": len(inputs),
               "python": platform.python_version(), "elapsed_seconds": time.perf_counter() - start,
               "caveat": config["interpretation"], "cost_scope": config["cost_scope"]}
    receipt["outputs_sha256"] = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    (output / "validation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
