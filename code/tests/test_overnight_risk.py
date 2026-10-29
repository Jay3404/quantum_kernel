import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import rbf_kernel
from _shared.angle_encoding import AngleEncoding
from _shared.improved import sha
from _shared import overnight_risk as night
from _shared import overnight_reporting as report
import run_overnight_risk as run

P = json.loads(run.PROTOCOL.read_text())


def fixture():
    rng = np.random.default_rng(778)
    arrays = {p: rng.normal(size=(n, 7)) for p, n in (("train", 32), ("tune", 12), ("gate", 12))}
    arrays.update(y_train=np.tile([0, 1], 16), y_tune=np.tile([0, 1], 6), weights=np.arange(1, 8) / 28)
    for p in night.PARTS:
        arrays[f"risk_{p}"] = rng.uniform(.05, .95, len(arrays[p]))
        arrays[f"rank_{p}"] = night.midrank(arrays["risk_train"], arrays[f"risk_{p}"])
        arrays[f"raw_{p}"] = np.column_stack([arrays[p], rng.normal(size=(len(arrays[p]), 3))])
        for prefix in ("raw_lr", "gbdt"):
            arrays[f"{prefix}_{p}"] = arrays[f"risk_{p}"] ** 2
            arrays[f"{prefix}_rank_{p}"] = night.midrank(arrays[f"{prefix}_train"], arrays[f"{prefix}_{p}"])
    record = {"encoder": AngleEncoding.fit(arrays["train"], "linear_no_clip").parameters(),
        "direction": (np.ones(7) / np.sqrt(7)).tolist(), "input_sha256": "fixture"}
    return arrays, record


class OvernightTests(unittest.TestCase):
    def test_scope_deadline_and_all_methods(self):
        self.assertEqual(len(set(P["methods"])), 32)
        self.assertEqual(P["lambdas"], [i / 20 for i in range(1, 21)])
        self.assertEqual(len(P["seeds"]), 10)
        self.assertEqual(len(run.jobs(P, P["methods"], P["discovery_seeds"])), 804)
        self.assertLess(run.stamp(P["new_full_cutoff_utc"]), run.stamp(P["compute_deadline_utc"]))
        self.assertLess(run.stamp(P["compute_deadline_utc"]), run.stamp(P["deadline_utc"]))
        self.assertEqual(run.preservation_protocol(P)["seeds"], list(range(10, 101, 10)))
        self.assertEqual(P["seeds"], list(range(2010, 2101, 10)))

    def test_nested_split_has_no_outer_validation_or_test_in_smoke(self):
        y = np.tile([0, 0, 1, 0, 1], 200)
        for stage, cap in (("discovery", 200), ("confirmation", 2048), ("full", 200)):
            outer, split = night.make_splits(y, 2010, stage, cap)
            self.assertFalse(set(split["train"]) & set(split["validation"]))
            self.assertFalse(set(split["train"]) & set(split["test"]))
            if stage != "full":
                used = set(sum(split.values(), []))
                self.assertTrue(used <= set(outer["train"]))
                self.assertFalse(used & set(outer["test"]))
                self.assertFalse(used & set(outer["validation"]))
            else:
                self.assertEqual(split, outer)
                self.assertEqual(len(split["train"]), 800)
        a = night.fixed_test_split(y, 2010, "german_numeric")
        b = night.fixed_test_split(y, 2020, "german_numeric")
        self.assertEqual(a["test"], b["test"])
        self.assertNotEqual(a["train"], b["train"])

    def test_every_method_has_psd_unit_diagonal_and_finite_values(self):
        arrays, record = fixture()
        protocol = {**P, "metric_train_cap": 24, "alignment": {**P["alignment"], "sample_size": 24, "maxfev": 8}}
        for method in P["methods"]:
            for learner in P["learners"]:
                with self.subTest(method=method, learner=learner):
                    context = night.representations(arrays, record, method, learner, "normal", 2010, protocol)
                    rep = night.make_representation(arrays, record, context, method, learner, "normal", .5, P["geometries"][0])
                    k = night.representation_kernel(rep, ("train", slice(None)), ("train", slice(None)))
                    np.testing.assert_allclose(k, k.T, atol=1e-11)
                    np.testing.assert_allclose(np.diag(k), 1., atol=1e-10)
                    self.assertGreater(np.linalg.eigvalsh(k).min(), -1e-8)

    def test_non_circuit_zero_strength_is_common_backbone(self):
        arrays, record = fixture()
        for learner in P["learners"]:
            reference = night.representations(arrays, record, "baseline", learner, "normal", 2010, P)
            rep = night.make_representation(arrays, record, reference, "baseline", learner, "normal", 0., P["geometries"][0])
            expected = night.representation_kernel(rep, ("train", slice(None)), ("train", slice(None)))
            for method in ("multiply", "rank_append", "rank_mixture", "soft_gate", "hellinger_product", "lda_stretch", "residual_risk"):
                context = night.representations(arrays, record, method, learner, "normal", 2010, P)
                current = night.make_representation(arrays, record, context, method, learner, "normal", 0., P["geometries"][0])
                kernel = night.representation_kernel(current, ("train", slice(None)), ("train", slice(None)))
                np.testing.assert_allclose(kernel, expected, atol=1e-10)

    def test_nystrom_matches_existing_sklearn_engine(self):
        x = np.random.default_rng(7).normal(size=(30, 7))
        model = Nystroem(kernel="rbf", gamma=.3, n_components=12, random_state=13)
        expected = model.fit_transform(x)
        indices = model.component_indices_
        factor, info = night.nystrom_factor(rbf_kernel(x[indices], gamma=.3))
        actual = rbf_kernel(x, x[indices], gamma=.3) @ factor
        np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, atol=1e-10)
        self.assertEqual(info["rank"], 12)

    def test_uncertainty_and_rank_mixture_are_distinct(self):
        arrays, record = fixture()
        kernels = []
        for method in ("rank_mixture", "uncertainty_mixture"):
            context = night.representations(arrays, record, method, "RBF-SVM", "normal", 2010, P)
            rep = night.make_representation(arrays, record, context, method, "RBF-SVM", "normal", .5, P["geometries"][0])
            kernels.append(night.representation_kernel(rep, ("train", slice(None)), ("train", slice(None))))
        self.assertGreater(abs(kernels[0] - kernels[1]).max(), .01)

    def test_no_entangling_control_matches_repetition_count(self):
        one = night.BackboneStates(.5, 1, False)
        two = night.BackboneStates(.5, 2, False)
        self.assertNotEqual(one.circuit.depth(), two.circuit.depth())
        self.assertTrue(all(len(v.qubits) <= 1 or v.operation.name == "save_statevector" for v in two.circuit.data))

    def test_worker_never_reads_poisoned_confirmation_labels(self):
        arrays, record = fixture()
        protocol = {**P, "gpu_devices": [], "C_grid": [.1], "smoke_lambdas": [.05], "geometries": [P["geometries"][0]]}
        with tempfile.TemporaryDirectory() as temporary:
            folder = night.prepared_folder(temporary, "synthetic", 2010)
            folder.mkdir(parents=True)
            np.savez_compressed(folder / "inputs.npz", **arrays)
            np.savez_compressed(folder / "confirmation_labels.npz", y_gate=np.array([object()], dtype=object))
            record["input_sha256"] = sha(folder / "inputs.npz")
            night.write_json(folder / "audit.json", record)
            job = ("synthetic", 2010, "RBF-SVM", "rank_product", "normal")
            night.run_job(job, "confirmation", protocol, temporary, {})
            saved = json.loads((night.job_folder(temporary, job) / "complete.json").read_text())
            self.assertFalse(saved["held_out_labels_read"])
            self.assertEqual(len(saved["trials"]), 2)
            self.assertTrue(all(r["fit_rows"] == 32 and r["model_fit_calls"] == r["ensemble_members"] == 1 for r in saved["trials"]))

    def test_approximate_selection_ends_in_one_exact_fit_and_own_threshold(self):
        arrays, record = fixture()
        protocol = {**P, "gpu_devices": [], "exact_search_max_rows": 16, "C_grid": [.1],
            "lambdas": [.05], "geometries": [P["geometries"][0]], "nystrom_landmarks": 16}
        with tempfile.TemporaryDirectory() as temporary:
            folder = night.prepared_folder(temporary, "synthetic", 2010)
            folder.mkdir(parents=True)
            np.savez_compressed(folder / "inputs.npz", **arrays)
            record["input_sha256"] = sha(folder / "inputs.npz")
            night.write_json(folder / "audit.json", record)
            job = ("synthetic", 2010, "RBF-SVM", "rank_product", "normal")
            night.run_job(job, "full", protocol, temporary, {})
            output = night.job_folder(temporary, job)
            saved = json.loads((output / "complete.json").read_text())
            lock = json.loads((output / "parameter_lock.json").read_text())
            with np.load(output / "exact_margins.npz") as scores:
                for role, choice in saved["choices"].items():
                    self.assertTrue(choice["final_exact"])
                    self.assertEqual(choice["final_model_fit_calls"], 1)
                    self.assertTrue(all(choice[k] == lock["choices"][role][k] for k in ("lambda", "C", "geometry")))
                    threshold, _ = night.base.threshold_for_cost(arrays["y_tune"], scores[role + "_tune"], 0.)
                    self.assertEqual(threshold, choice["threshold"])

    def test_cost_ratio_threshold_and_distance_orientation(self):
        y, score = np.array([0, 1, 0, 1]), np.array([.1, .2, .8, .9])
        for ratio in (2, 5, 10):
            threshold = report.threshold_for_ratio(y, score, ratio)
            candidates = np.r_[np.nextafter(score.max(), np.inf), np.unique(score)]
            costs = [ratio * np.sum((y == 1) & (score < t)) + np.sum((y == 0) & (score >= t)) for t in candidates]
            actual = ratio * np.sum((y == 1) & (score < threshold)) + np.sum((y == 0) & (score >= threshold))
            self.assertEqual(actual, min(costs))
        kernel = np.equal.outer(y, y).astype(float)
        self.assertAlmostEqual(report.geometry(kernel, y)["separation_gap"], 2.)

    def test_promotion_is_practical_not_fabricated_statistical_support(self):
        rows = []
        for d in P["datasets"]:
            for seed in P["seeds"]:
                for learner in P["learners"]:
                    for family, role, cost in (("baseline", "baseline", 100), ("rank_append", "proposed", 80 + (seed // 10) % 3)):
                        rows.append({"dataset": d, "seed": seed, "kernel": learner, "family": family, "arm": "normal", "role": role,
                            "cost": cost, "n_eval": 100, "n_train": 600, "recall": .8, "balanced_accuracy": .7, "separation_gap": .1})
        table, methods, _ = report.gate_effects(pd.DataFrame(rows), ["rank_append"], P)
        self.assertEqual(methods, ["rank_append"])
        self.assertTrue(table.passes_practical.all())
        self.assertTrue(table.holm_family_size.eq(8).all())
        altered = pd.DataFrame(rows)
        altered.loc[altered.dataset.eq("german_numeric") & altered.role.eq("proposed"), "cost"] = 1000
        changed, chosen, _ = report.gate_effects(altered, ["rank_append"], P)
        self.assertEqual(chosen, methods)
        np.testing.assert_allclose(table.loc[table.dataset.ne("german_numeric"), "selection_p_holm"],
            changed.loc[changed.dataset.ne("german_numeric"), "selection_p_holm"])

    def test_customer_inference_does_not_treat_duplicate_training_seeds_as_new_customers(self):
        delta = np.array([[5., -1., 0., 1.], [3., 0., 1., -1.]])
        labels = np.array([1, 0, 0, 1])
        first = report.paired_customer_summary(delta, labels, "same-inference-key")
        repeated = report.paired_customer_summary(np.tile(delta, (5, 1)), labels, "same-inference-key")
        self.assertEqual(first["test_customer_clusters"], 4)
        for key in ("mean_difference", "ci_low", "ci_high", "p_value"):
            self.assertEqual(first[key], repeated[key])

    def test_full_report_preserves_planned_family_and_interaction(self):
        protocol = {**P, "datasets": ["german_numeric"]}
        rows = []
        for i, seed in enumerate(P["seeds"]):
            common = {"dataset": "german_numeric", "seed": seed, "n_train": 800, "n_eval": 100}
            for learner in P["learners"]:
                rows.append({**common, "kernel": learner, "family": "baseline", "arm": "normal", "role": "baseline", "cost": 100})
                for arm, role, cost in (("normal", "baseline", 100), ("normal", "proposed", 80 + i),
                                        ("shuffled", "proposed", 95 + i % 3), ("constant", "proposed", 101)):
                    rows.append({**common, "kernel": learner, "family": "rank_append", "arm": arm, "role": role, "cost": cost})
                if learner == "QSVC":
                    rows.append({**common, "kernel": learner, "family": "rank_append", "arm": "separable", "role": "proposed", "cost": 93})
            rows.append({**common, "kernel": "reference", "family": "reference_best", "arm": "normal", "role": "classical_best", "cost": 98})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for name in ("selection_lock.json", "test_release.json", "seed_metrics.csv", "summary_metrics.csv", "runtime_summary.csv", "quantum_settings.json", "predictions.csv.gz", "lambda_seed_metrics.csv", "lambda_paired_cost.csv", "cost_sensitivity_paired.csv"):
                (output / name).write_text("explicit synthetic test fixture")
            class SyntheticLosses:
                def matrix(self, models, ratio=5):
                    value = models.sort_index().cost.to_numpy() / 2
                    return np.column_stack([value, value]), np.array([0, 1])
            report.full_report(output, pd.DataFrame(rows), [], {}, ["rank_append"], protocol, [], SyntheticLosses())
            table = pd.read_csv(output / "paired_cost.csv")
            self.assertEqual(len(table), 12)
            self.assertTrue(table.holm_family_size.eq(12).all())
            self.assertEqual(len(table[table.contrast.eq("quantum_minus_rbf_gain")]), 1)
            self.assertTrue((output / "verification_receipt.json").exists())

    def test_full_pipeline_locks_before_first_test_label_read(self):
        arrays, record = fixture()
        protocol = {**P, "datasets": ["synthetic"], "methods": ["rank_product"], "lambdas": [.05],
            "C_grid": [.1], "geometries": [P["geometries"][0]], "gpu_devices": []}
        split = {"train": list(range(32)), "validation": list(range(32, 44)), "test": list(range(44, 56))}
        rf, xgb = night.RandomForestClassifier, night.XGBClassifier
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "code/results/explicit_synthetic_full/full"
            for seed in P["seeds"]:
                prepared = night.prepared_folder(output, "synthetic", seed)
                prepared.mkdir(parents=True)
                np.savez_compressed(prepared / "inputs.npz", **arrays)
                night.write_json(prepared / "audit.json", {**record, "input_sha256": sha(prepared / "inputs.npz")})
                night.write_json(prepared / "split.json", {"outer": split, "actual": split})
            tasks = run.jobs(protocol, protocol["methods"], protocol["seeds"], True)
            with patch.object(night, "RandomForestClassifier", side_effect=lambda **kw: rf(**{**kw, "n_estimators": 3})), patch.object(night, "XGBClassifier", side_effect=lambda **kw: xgb(**{**kw, "n_estimators": 3})):
                for job in tasks:
                    night.run_job(job, "full", protocol, output, {})
            def guarded_labels(*args):
                self.assertTrue((output / "selection_lock.json").exists())
                self.assertTrue((output / "test_release.json").exists())
                return None, np.tile([0, 1], 28), {"synthetic": True}
            with patch.object(report, "ROOT", root), patch.object(report.base, "load_data", side_effect=guarded_labels), patch.object(report, "fixed_test_split", return_value=split):
                report.save_stage(output, tasks, {}, protocol, protocol["methods"], "full")
            self.assertEqual(len(pd.read_csv(output / "paired_cost.csv")), 12)
            self.assertEqual(len(pd.read_csv(output / "lambda_paired_cost.csv")), 2)
            self.assertTrue((output / "cost_sensitivity_paired.csv").exists())
            with self.assertRaises(AssertionError):
                night.run_job(tasks[0], "full", protocol, output, {})


if __name__ == "__main__":
    unittest.main()
