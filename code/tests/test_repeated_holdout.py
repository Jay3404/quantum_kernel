import copy
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import stats

from _shared import repeated_holdout as p
from _shared import repeated_reporting as report
from _shared import overnight_reporting as common
from _shared import overnight_risk as old
from _shared import overnight_runtime_v3 as isolated_previous
from _shared.angle_encoding import AngleEncoding
import run_repeated_holdout as runner


def fixture():
    rng = np.random.default_rng(123)
    arrays = {part: rng.normal(size=(n, 7)) for part, n in (("train", 30), ("tune", 12), ("gate", 10))}
    arrays.update(y_train=np.tile([0, 1], 15), y_tune=np.tile([0, 1], 6), weights=np.ones(7) / 7)
    for part in p.PARTS:
        arrays["raw_" + part] = rng.normal(size=(len(arrays[part]), 12))
        arrays["risk_" + part] = rng.uniform(.05, .95, len(arrays[part]))
        arrays["rank_" + part] = old.midrank(arrays["risk_train"], arrays["risk_" + part])
        for prefix in ("raw_lr", "gbdt"):
            arrays[prefix + "_" + part] = arrays["risk_" + part].copy()
            arrays[prefix + "_rank_" + part] = arrays["rank_" + part].copy()
    record = {"pca_components": 7, "encoder": AngleEncoding.fit(arrays["train"], "linear_no_clip").parameters()}
    return arrays, record


class RepeatedHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(p.PROTOCOL.read_text())

    def test_full_fixed_scope_and_seeds(self):
        self.assertEqual(self.protocol["seeds"], list(range(10, 101, 10)))
        self.assertEqual(self.protocol["pca_components"], 7)
        self.assertEqual(len(set(p.jobs(self.protocol))), 1160)
        self.assertEqual(self.protocol["methods"], list(p.METHODS))
        previous = json.loads((p.ROOT / "docs/cmc_pca9_protocol_20261020.json").read_text())
        for name in ("lambdas", "C_grid", "geometries", "nystrom_landmarks", "exact_search_max_rows"):
            self.assertEqual(self.protocol[name], previous[name])

    def test_test_customers_change_and_are_disjoint_per_split(self):
        y = np.tile([0, 0, 0, 1], 250)
        cohorts = []
        for seed in self.protocol["seeds"]:
            split = p.split_indices(y, seed)
            self.assertEqual([len(split[k]) for k in ("train", "validation", "test")], [800, 100, 100])
            self.assertFalse(set(split["test"]) & set(split["train"] + split["validation"]))
            self.assertEqual(split, p.split_indices(y, seed))
            cohorts.append(tuple(sorted(split["test"])))
        self.assertEqual(len(set(cohorts)), 10)

    def test_fold_specific_preprocessing_and_sigmoid(self):
        rng = np.random.default_rng(9)
        X = pd.DataFrame(rng.normal(size=(90, 12)))
        X.attrs["nominal"] = []
        with tempfile.TemporaryDirectory() as temporary:
            log = str(Path(temporary) / "fit.jsonl")
            model = p.risk_estimator(X, 10, 7, log).fit(X, np.tile([0, 1], 45))
            rows = [json.loads(line) for line in Path(log).read_text().splitlines()]
            self.assertEqual(sorted(row["rows"] for row in rows), [60, 60, 60, 90])
            self.assertTrue(all(row["components"] == 7 for row in rows))
            self.assertFalse(model.ensemble)

    def test_rank_deficiency_fails_without_reducing_dimension(self):
        with self.assertRaises(AssertionError):
            p.TrainingPCA(7).fit_transform(np.ones((20, 12)))

    def test_seven_eight_qubits_psd_zero_and_array_preservation(self):
        arrays, record = fixture()
        original = {k: v.copy() for k, v in arrays.items()}
        for learner in self.protocol["learners"]:
            ctx = p.engine.context(arrays, record, "baseline", learner, "normal", 10, self.protocol)
            zero = p.engine.make_representation(arrays, record, ctx, "baseline", learner, "normal", 0., self.protocol["geometries"][0])
            expected = p.engine.kernel(zero, ("train", slice(None)), ("train", slice(None)))
            for method in p.METHODS:
                ctx = p.engine.context(arrays, record, method, learner, "normal", 10, self.protocol)
                for strength in (0., .5, 1.):
                    rep = p.engine.make_representation(arrays, record, ctx, method, learner, "normal", strength, self.protocol["geometries"][0])
                    gram = p.engine.kernel(rep, ("train", slice(None)), ("train", slice(None)))
                    np.testing.assert_allclose(gram, gram.T, atol=1e-12)
                    np.testing.assert_allclose(np.diag(gram), 1, atol=1e-12)
                    self.assertGreaterEqual(np.linalg.eigvalsh(gram).min(), -1e-9)
                    if strength == 0:
                        np.testing.assert_allclose(gram, expected, atol=1e-12)
                    if learner == "QSVC":
                        self.assertEqual(rep["values"]["train"].shape[1], 256 if method == "gbdt_rank_append" else 128)
                    else:
                        self.assertAlmostEqual(rep["gamma"], self.protocol["geometries"][0]["gamma_factor"] / (7 * ctx["angles"]["train"].var()))
        for key in original:
            np.testing.assert_array_equal(arrays[key], original[key])

    def test_mathematical_pca7_equivalence_to_prior_engine(self):
        arrays, record = fixture()
        record["direction"] = (np.ones(7) / np.sqrt(7)).tolist()
        for learner, method in itertools.product(self.protocol["learners"], p.METHODS):
            old_context = old.representations(arrays, record, method, learner, "normal", 10, self.protocol)
            fresh_context = p.engine.context(arrays, record, method, learner, "normal", 10, self.protocol)
            for strength in (0., .35, 1.):
                legacy = isolated_previous.make_representation(arrays, record, old_context, method, learner, "normal", strength, self.protocol["geometries"][0])
                fresh = p.engine.make_representation(arrays, record, fresh_context, method, learner, "normal", strength, self.protocol["geometries"][0])
                np.testing.assert_allclose(old.representation_kernel(legacy, ("tune", slice(None)), ("train", slice(None))),
                    p.engine.kernel(fresh, ("tune", slice(None)), ("train", slice(None))), atol=1e-10, rtol=1e-10)

    def test_corrected_interval_uses_overlap_correction(self):
        values = np.arange(-4, 6, dtype=float)
        actual = report.corrected_summary(values, .125)
        se = np.sqrt((.1 + .125) * values.var(ddof=1))
        self.assertAlmostEqual(actual["corrected_se"], se)
        self.assertAlmostEqual(actual["ci_high"], values.mean() + stats.t.ppf(.975, 9) * se)
        self.assertGreater(se, values.std(ddof=1) / np.sqrt(10))
        self.assertEqual(report.corrected_summary(np.ones(10), .125)["p_value"], 1)

    def test_all184_contrasts_include_unfavorable_and_related_scopes(self):
        rows = []
        for dataset, seed in itertools.product(self.protocol["datasets"], self.protocol["seeds"]):
            for kernel in self.protocol["learners"]:
                rows.append(dict(dataset=dataset, seed=seed, kernel=kernel, family="baseline", arm="normal", role="baseline", cost=60 + seed / 10, n_eval=100, n_train=800))
                for method in p.METHODS:
                    for arm, role in [("normal", "baseline"), ("normal", "proposed"), ("shuffled", "proposed")] + ([] if method == "pls_metric" else [("constant", "proposed")]) + ([("separable", "proposed")] if kernel == "QSVC" else []):
                        rows.append(dict(dataset=dataset, seed=seed, kernel=kernel, family=method, arm=arm, role=role,
                            cost=65 + (seed % 30) / 10, n_eval=100, n_train=800))
            rows.append(dict(dataset=dataset, seed=seed, kernel="reference", family="reference_best", arm="normal", role="classical_best", cost=55 + seed / 10, n_eval=100, n_train=800))
        table, directions = report.contrasts(pd.DataFrame(rows), self.protocol)
        self.assertEqual(len(table), 184)
        self.assertTrue(table.holm_family_size.eq(184).all())
        self.assertTrue(table.p_holm.ge(table.p_value).all())
        self.assertEqual(len(directions), 1840)

    def test_85percent_cpu_and_memory_admission(self):
        state = dict(host_total_gib=188., host_available_gib=180., cpu_percent=84., logical_cpu_limit=32,
            cgroup_limit_gib=None, cgroup_used_gib=None)
        scheduling = runner.scheduling_for(state, .8, 10., self.protocol)
        self.assertEqual(scheduling["workers"], 27)
        self.assertTrue(runner.admitted(state, scheduling, self.protocol))
        self.assertFalse(runner.admitted({**state, "cpu_percent": 85.}, scheduling, self.protocol))
        self.assertFalse(runner.admitted({**state, "host_available_gib": 40.}, scheduling, self.protocol))
        self.assertFalse(runner.admitted({**state, "cgroup_limit_gib": 30., "cgroup_used_gib": 23.}, scheduling, self.protocol))

    def test_fresh_teachers_preparation_and_locked_prediction_pipeline(self):
        rng = np.random.default_rng(100)
        X = pd.DataFrame(rng.normal(size=(120, 12)))
        X.attrs["nominal"] = []
        y = np.tile([0, 1], 60)
        protocol = {**self.protocol, "gpu_devices": [], "exact_search_max_rows": 40, "nystrom_landmarks": 16,
            "C_grid": [1.], "lambdas": [.05, .5], "geometries": [self.protocol["geometries"][0]],
            "exact_workers": 1, "exact_budget_gib": 2.}
        with tempfile.TemporaryDirectory(dir=p.ROOT / "code/results") as temporary, patch.object(p.base, "load_data", return_value=(X, y, {"synthetic": True})):
            output = Path(temporary) / "full"
            p.prepare("synthetic", 10, protocol, output)
            arrays, record = old.load_inputs(output, "synthetic", 10)
            self.assertFalse(record["prior_preparation_reused"])
            self.assertFalse(record["teachers"]["gbdt"]["reused_from_prior_campaign"])
            self.assertEqual(record["nested_pca_fit_count"], 25)
            for method, learner in (("rank_mixture", "RBF-SVM"), ("reference", "reference")):
                job = ("synthetic", 10, learner, method, "normal")
                p.run_job(job, protocol, output, {})
            tasks = [("synthetic", 10, "RBF-SVM", "rank_mixture", "normal"), ("synthetic", 10, "reference", "reference", "normal")]
            groups, _, missing = common.collect(output, tasks, {})
            self.assertFalse(missing)
            choices = common.sensitivity_choices(output, report.selected_choices(groups, 7))
            with self.assertRaises(AssertionError):
                report.evaluate(output, choices)
            p.write_json(output / "selection_lock.json", {"choices": choices})
            p.write_json(output / "test_release.json", {})
            raw, predictions = report.evaluate(output, choices)
            self.assertTrue(raw.n_train.eq(96).all())
            self.assertTrue(raw.pca_components.eq(7).all())
            self.assertNotIn("pca9", set(raw.method))
            self.assertEqual(set(predictions.row_id), set(p.split_indices(y, 10)["test"]))
            saved = groups[tasks[0]]
            self.assertEqual(len(saved["trials"]), 3)
            for choice in saved["choices"].values():
                self.assertEqual(choice["final_model_fit_calls"], 1)
                self.assertEqual(choice["fit_rows"], 96)
            with self.assertRaises(AssertionError):
                p.run_job(tasks[0], protocol, output, {})

    def test_global_lock_or_test_release_blocks_training(self):
        for name in ("selection_lock.json", "test_release.json"):
            with tempfile.TemporaryDirectory() as temporary:
                Path(temporary, name).write_text("{}")
                with self.assertRaises(AssertionError):
                    p.assert_can_fit(temporary)


if __name__ == "__main__":
    unittest.main()
