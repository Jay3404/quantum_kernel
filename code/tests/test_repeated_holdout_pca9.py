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

from _shared import repeated_holdout_pca9 as p
from _shared import repeated_reporting_pca9 as report
from _shared import overnight_reporting as common
from _shared import overnight_risk as old
from _shared.angle_encoding import AngleEncoding
import run_repeated_holdout_pca9 as runner


def fixture(dimension=9):
    rng = np.random.default_rng(123)
    arrays = {part: rng.normal(size=(n, dimension)) for part, n in (("train", 30), ("tune", 12), ("gate", 10))}
    arrays.update(y_train=np.tile([0, 1], 15), y_tune=np.tile([0, 1], 6), weights=np.ones(dimension) / dimension)
    for part in p.PARTS:
        arrays["raw_" + part] = rng.normal(size=(len(arrays[part]), 12))
        arrays["risk_" + part] = rng.uniform(.05, .95, len(arrays[part]))
        arrays["rank_" + part] = old.midrank(arrays["risk_train"], arrays["risk_" + part])
        for prefix in ("raw_lr", "gbdt"):
            arrays[prefix + "_" + part] = arrays["risk_" + part].copy()
            arrays[prefix + "_rank_" + part] = arrays["rank_" + part].copy()
    return arrays, {"pca_components": dimension, "encoder": AngleEncoding.fit(arrays["train"], "linear_no_clip").parameters()}


def statistics_fixture(protocol):
    rows = []
    for dataset, seed in itertools.product(protocol["datasets"], protocol["seeds"]):
        for kernel in protocol["learners"]:
            rows.append(dict(dataset=dataset, seed=seed, kernel=kernel, family="baseline", arm="normal", role="baseline", cost=60 + seed / 10, n_eval=100, n_train=800))
            for method in p.METHODS:
                roles = [("normal", "baseline"), ("normal", "proposed"), ("shuffled", "proposed")]
                roles += [] if method == "pls_metric" else [("constant", "proposed")]
                roles += [("separable", "proposed")] if kernel == "QSVC" else []
                for arm, role in roles:
                    rows.append(dict(dataset=dataset, seed=seed, kernel=kernel, family=method, arm=arm, role=role,
                        cost=65 + (seed % 30) / 10, n_eval=100, n_train=800))
        rows.append(dict(dataset=dataset, seed=seed, kernel="reference", family="reference_best", arm="normal", role="classical_best", cost=55 + seed / 10, n_eval=100, n_train=800))
    return pd.DataFrame(rows)


class PCA9RepeatedTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(p.PROTOCOL.read_text())

    def test_fixed_870_scope_matches_pca7_without_taiwan(self):
        previous = json.loads((p.ROOT / "docs/cmc_pca7_repeated_protocol_20261020.json").read_text())
        self.assertEqual(self.protocol["datasets"], ["german_numeric", "south_german", "hmeq"])
        self.assertEqual(self.protocol["pca_components"], 9)
        self.assertEqual(len(set(p.jobs(self.protocol))), 870)
        for key in ("seeds", "methods", "learners", "lambdas", "C_grid", "geometries", "nystrom_landmarks", "exact_search_max_rows"):
            self.assertEqual(self.protocol[key], previous[key])

    def test_nine_component_nested_calibration(self):
        rng = np.random.default_rng(9)
        X = pd.DataFrame(rng.normal(size=(90, 12)))
        X.attrs["nominal"] = []
        with tempfile.TemporaryDirectory() as temporary:
            log = str(Path(temporary) / "fit.jsonl")
            model = p.risk_estimator(X, 10, 9, log).fit(X, np.tile([0, 1], 45))
            rows = [json.loads(line) for line in Path(log).read_text().splitlines()]
            self.assertEqual(sorted(row["rows"] for row in rows), [60, 60, 60, 90])
            self.assertTrue(all(row["components"] == 9 for row in rows))
            self.assertFalse(model.ensemble)

    def test_rank_deficiency_not_silently_reduced(self):
        with self.assertRaises(AssertionError):
            p.TrainingPCA(9).fit_transform(np.ones((20, 12)))

    def test_nine_ten_qubits_psd_zero_and_no_mutation(self):
        arrays, record = fixture()
        snapshot = {key: value.copy() for key, value in arrays.items()}
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
                        dimension = 10 if method == "gbdt_rank_append" else 9
                        self.assertEqual(rep["values"]["train"].shape[1], 2 ** dimension)
                        self.assertEqual(rep["details"]["circuits"][0]["qubits"], dimension)
                    else:
                        self.assertAlmostEqual(rep["gamma"], self.protocol["geometries"][0]["gamma_factor"] / (9 * ctx["angles"]["train"].var()))
        for key in snapshot:
            np.testing.assert_array_equal(arrays[key], snapshot[key])

    def test_pca7_and_pca9_dimensions_remain_distinct(self):
        grams = []
        for dimension in (7, 9):
            arrays, record = fixture(dimension)
            protocol = {**self.protocol, "pca_components": dimension}
            ctx = p.engine.context(arrays, record, "baseline", "QSVC", "normal", 10, protocol)
            rep = p.engine.make_representation(arrays, record, ctx, "baseline", "QSVC", "normal", 0., protocol["geometries"][0])
            self.assertEqual(rep["values"]["train"].shape[1], 2 ** dimension)
            grams.append(p.engine.kernel(rep, ("train", slice(None)), ("train", slice(None))))
        self.assertGreater(float(abs(grams[0] - grams[1]).max()), .01)

    def test_append_is_independent_auxiliary(self):
        arrays, record = fixture()
        geometry = self.protocol["geometries"][0]
        ctx = p.engine.context(arrays, record, "gbdt_rank_append", "QSVC", "normal", 10, self.protocol)
        base = p.engine.make_representation(arrays, record, ctx, "gbdt_rank_append", "QSVC", "normal", 0., geometry)
        rep = p.engine.make_representation(arrays, record, ctx, "gbdt_rank_append", "QSVC", "normal", .5, geometry)
        left = ("train", slice(None))
        angles = np.pi * .5 * (ctx["q"]["train"] - .5)
        np.testing.assert_allclose(p.engine.kernel(rep, left, left), p.engine.kernel(base, left, left)
            * np.cos((angles[:, None] - angles[None, :]) / 2) ** 2, atol=1e-12)

    def test_corrected_statistics_not_independent_t(self):
        values = np.arange(-4, 6, dtype=float)
        result = report.corrected_summary(values, .125)
        expected = np.sqrt(.225 * values.var(ddof=1))
        self.assertAlmostEqual(result["corrected_se"], expected)
        self.assertAlmostEqual(result["ci_high"], values.mean() + stats.t.ppf(.975, 9) * expected)
        self.assertEqual(report.corrected_summary(np.ones(10), .125)["p_value"], 1)

    def test_frozen_138_and_90_comparison_families(self):
        current = statistics_fixture(self.protocol)
        previous = current.copy()
        previous["cost"] += (previous.seed % 30) / 10
        within, _ = report.contrasts(current, self.protocol)
        dimension, directions = report.compare(current, previous, self.protocol)
        self.assertEqual(len(within), 138)
        self.assertTrue(within.holm_family_size.eq(138).all())
        self.assertEqual(len(dimension), 90)
        self.assertTrue(dimension.holm_family_size.eq(90).all())
        self.assertEqual(len(directions), 900)
        self.assertTrue(dimension.p_holm.ge(dimension.p_value).all())
        self.assertTrue(dimension.variance_multiplier.eq(.225).all())

    def test_cpu_and_memory_admission(self):
        state = dict(host_total_gib=188., host_available_gib=180., cpu_percent=84., logical_cpu_limit=32,
            cgroup_limit_gib=None, cgroup_used_gib=None)
        scheduling = runner.scheduling_for(state, .8, 10., self.protocol)
        self.assertEqual(scheduling["workers"], 27)
        self.assertFalse(runner.admitted({**state, "cpu_percent": 85.}, scheduling, self.protocol))
        self.assertFalse(runner.admitted({**state, "host_available_gib": 40.}, scheduling, self.protocol))

    def test_fresh_preparation_exact_rows_and_test_blocking(self):
        rng = np.random.default_rng(100)
        X = pd.DataFrame(rng.normal(size=(120, 12)))
        X.attrs["nominal"] = []
        y = np.tile([0, 1], 60)
        protocol = {**self.protocol, "gpu_devices": [], "exact_search_max_rows": 40, "nystrom_landmarks": 16,
            "C_grid": [1.], "lambdas": [.5], "geometries": [self.protocol["geometries"][0]],
            "exact_workers": 1, "exact_budget_gib": 2.}
        with tempfile.TemporaryDirectory(dir=p.ROOT / "code/results") as temporary, patch.object(p.base, "load_data", return_value=(X, y, {"synthetic": True})):
            upstream = Path(temporary) / "upstream"
            p.shared.prepare("synthetic", 10, {**protocol, "pca_components": 7}, upstream / "full")
            output = Path(temporary) / "new/full"
            result = p.prepare("synthetic", 10, protocol, output, upstream=upstream)
            self.assertTrue(result["pca7_comparison"]["same_saved_split"])
            arrays, record = old.load_inputs(output, "synthetic", 10)
            self.assertEqual(arrays["train"].shape, (96, 9))
            self.assertTrue(record["new_pca_lr_fit"])
            self.assertTrue(all(not value["reused_from_prior_campaign"] for value in record["teachers"].values()))
            logs = [json.loads(line) for line in (old.prepared_folder(output, "synthetic", 10) / "nested_pca_fits.jsonl").read_text().splitlines()]
            self.assertEqual(len(logs), 25)
            self.assertTrue(all(row["components"] == 9 for row in logs))
            tasks = [("synthetic", 10, "RBF-SVM", "rank_mixture", "normal"),
                ("synthetic", 10, "QSVC", "gbdt_rank_append", "normal"), ("synthetic", 10, "reference", "reference", "normal")]
            for job in tasks:
                p.run_job(job, protocol, output, {})
            groups, _, missing = common.collect(output, tasks, {})
            self.assertFalse(missing)
            choices = common.sensitivity_choices(output, report.selected_choices(groups, 9))
            with self.assertRaises(AssertionError):
                report.evaluate(output, choices)
            p.write_json(output / "selection_lock.json", {"choices": choices})
            p.write_json(output / "test_release.json", {})
            raw, predictions = report.evaluate(output, choices)
            self.assertTrue(raw.n_train.eq(96).all())
            self.assertTrue(raw.pca_components.eq(9).all())
            self.assertNotIn("pca7", set(raw.method))
            self.assertEqual(set(predictions.row_id), set(p.split_indices(y, 10)["test"]))
            for job in tasks[:2]:
                for choice in groups[job]["choices"].values():
                    self.assertEqual(choice["final_model_fit_calls"], 1)
                    self.assertEqual(choice["fit_rows"], 96)
            with self.assertRaises(AssertionError):
                p.run_job(tasks[0], protocol, output, {})

    def test_global_lock_or_release_blocks_new_training(self):
        for name in ("selection_lock.json", "test_release.json"):
            with tempfile.TemporaryDirectory() as temporary:
                Path(temporary, name).write_text("{}")
                with self.assertRaises(AssertionError):
                    p.assert_can_fit(temporary)

    def test_upstream_pca7_report_audit_stop_and_source_gate(self):
        files = runner.upstream_gate()
        self.assertIn("docs/cmc_pca7_repeated_final_ko.md", files)
        self.assertIn("code/results/cmc_pca7_repeated_20261020/validation_receipt.json", files)
        self.assertIn("code/results/cmc_pca9_20261020/operations/authorized_stop_20261020/stop_receipt.json", files)


if __name__ == "__main__":
    unittest.main()
