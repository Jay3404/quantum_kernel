"""Predeclared paired PCA9-minus-PCA7 repeated-holdout sensitivity."""
import itertools
import numpy as np
import pandas as pd

from .repeated_reporting import corrected_summary
from build_controlled_results import holm


def compare(current, previous, protocol):
    seeds = protocol["seeds"]
    rows, directions = [], []

    def costs(frame, dataset, kernel, family, role):
        selected = frame[frame.dataset.eq(dataset) & frame.kernel.eq(kernel) & frame.family.eq(family)
            & frame.arm.eq("normal") & frame.role.eq(role)].set_index("seed")
        assert len(selected) == len(seeds) == 10
        selected = selected.reindex(seeds)
        assert not selected.cost.isna().any()
        return selected

    def add(dataset, kernel, family, contrast, values, model):
        values = np.asarray(values, dtype=float)
        ratio = float((model.n_eval / model.n_train).max())
        rows.append({"dataset": dataset, "kernel": kernel, "family": family, "contrast": contrast,
            "positive_favors": "PCA9", "primary_source": dataset != "german_numeric",
            **corrected_summary(values, ratio)})
        directions.extend({"dataset": dataset, "kernel": kernel, "family": family, "contrast": contrast,
            "seed": seed, "delta": value, "direction": "improvement" if value > 0 else "worse" if value < 0 else "tie"}
            for seed, value in zip(seeds, values))

    gains = {}
    for dataset, kernel in itertools.product(protocol["datasets"], protocol["learners"]):
        for family in ("baseline", *protocol["methods"]):
            role = "baseline" if family == "baseline" else "proposed"
            a, b = [costs(frame, dataset, kernel, family, role) for frame in (current, previous)]
            np.testing.assert_array_equal(a.n_eval, b.n_eval)
            np.testing.assert_array_equal(a.n_train, b.n_train)
            add(dataset, kernel, family, "PCA7_cost_minus_PCA9_cost", b.cost - a.cost, a)
            if family == "baseline":
                continue
            for control, baseline_family in (("tuned", "baseline"), ("own_zero", family)):
                ca, cb = [costs(frame, dataset, kernel, baseline_family, "baseline") for frame in (current, previous)]
                gain9, gain7 = ca.cost - a.cost, cb.cost - b.cost
                add(dataset, kernel, family, f"PCA9_minus_PCA7_{control}_gain", gain9 - gain7, a)
                if control == "own_zero":
                    gains[dataset, kernel, family] = gain9 - gain7
    for dataset, family in itertools.product(protocol["datasets"], protocol["methods"]):
        model = costs(current, dataset, "QSVC", family, "proposed")
        add(dataset, "interaction", family, "PCA9_minus_PCA7_quantum_minus_rbf_own_zero_gain",
            gains[dataset, "QSVC", family] - gains[dataset, "RBF-SVM", family], model)
    result = pd.DataFrame(rows)
    assert len(result) == 30 * len(protocol["datasets"]) == 90
    result["p_holm"] = holm(result.p_value.to_numpy())
    result["holm_family_size"] = len(result)
    return result, pd.DataFrame(directions)
