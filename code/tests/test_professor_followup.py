import ast
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared import professor_followup as p


class FollowupTests(unittest.TestCase):
    def test_manifest_counts_and_scope(self):
        protocol = p.protocol()
        tasks = p.jobs(protocol)
        self.assertEqual(len(tasks), 200)
        self.assertEqual(len(set(tasks)), 200)
        self.assertEqual(sum(t[0] == "random" for t in tasks), 120)
        self.assertEqual(protocol["statistics"]["primary_cost_family"], 100)
        self.assertEqual(protocol["statistics"]["secondary_calibration_family"], 40)

    def test_random_control_fixed_shared_and_independent(self):
        lengths = {"train": 200, "tune": 30, "gate": 30}
        first = p.random_values("hmeq", 10, 1, lengths)
        again = p.random_values("hmeq", 10, 1, lengths)
        second = p.random_values("hmeq", 10, 2, lengths)
        for part in lengths:
            np.testing.assert_array_equal(first[part], again[part])
            self.assertFalse(np.array_equal(first[part], second[part]))
            self.assertTrue(((first[part] >= 0) & (first[part] < 1)).all())
        self.assertFalse(np.array_equal(first["tune"], first["gate"]))

    def test_threshold_reference_and_scale_stable_ties(self):
        rng = np.random.default_rng(133)
        for _ in range(20):
            y = rng.integers(0, 2, 50)
            score = rng.integers(0, 30, 50) / 30
            candidates = np.r_[np.nextafter(score.max(), np.inf), np.unique(score)[::-1]]
            costs = np.array([p.metrics(y, score, cut)["cost"] for cut in candidates])
            best = np.flatnonzero(costs == costs.min())
            expected = candidates[best[np.argmin(abs(candidates[best] - 1 / 6))]]
            self.assertEqual(p.cost_cut(y, score), expected)
            self.assertEqual(p.cost_cut(y, score, rank_stable=True), candidates[best[0]])
            raw = p.cost_cut(y, score, rank_stable=True)
            shifted = np.exp(score)
            transformed = p.cost_cut(y, shifted, rank_stable=True)
            np.testing.assert_array_equal(score >= raw, shifted >= transformed)

    def test_sigmoid_train_only_and_heldout_invariance(self):
        train = np.linspace(.01, .99, 200)
        y = np.array([int(i % 7 < 1 + 5 * train[i]) for i in range(200)])
        values = {"tune": np.linspace(.02, .98, 30), "gate": np.linspace(.03, .97, 25)}
        transformed, model, details = p.calibrate(y, train, values, "sigmoid")
        self.assertGreater(details["slope"], 0)
        for part in values:
            np.testing.assert_array_equal(np.argsort(values[part]), np.argsort(transformed[part]))
        changed, _, changed_details = p.calibrate(y, train, {"tune": values["tune"], "gate": values["gate"][::-1]}, "sigmoid")
        self.assertEqual(details["slope"], changed_details["slope"])
        np.testing.assert_array_equal(transformed["tune"], changed["tune"])
        self.assertEqual(model.classes_.tolist(), [0, 1])

    def test_isotonic_bounded_and_raw_identity(self):
        train = np.linspace(.1, .9, 50)
        y = np.arange(50) % 3 == 0
        values = {"tune": np.array([0., .2, .8, 1.]), "gate": np.array([.1, .5, .9])}
        calibrated, _, _ = p.calibrate(y, train, values, "isotonic")
        self.assertTrue(((calibrated["tune"] >= 0) & (calibrated["tune"] <= 1)).all())
        raw, model, _ = p.calibrate(y, train, values, "raw")
        self.assertIsNone(model)
        np.testing.assert_array_equal(raw["gate"], values["gate"])

    def test_global_lock_blocks_fit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p.fit_allowed(root)
            (root / "selection_lock.json").touch()
            with self.assertRaises(AssertionError):
                p.fit_allowed(root)

    def test_corrected_standard_error_and_degenerate_case(self):
        value = np.arange(10.) - 2
        result = p.corrected(value)
        half = p.stats.t.ppf(.975, 9) * np.sqrt(.225) * value.std(ddof=1)
        self.assertAlmostEqual(result["ci_high"], value.mean() + half)
        self.assertEqual(p.corrected(np.zeros(10))["p_value"], 1)

    def test_independent_eighth_qubit_product(self):
        rng = np.random.default_rng(71)
        state = rng.normal(size=(15, 128)) + 1j * rng.normal(size=(15, 128))
        state /= np.linalg.norm(state, axis=1, keepdims=True)
        q = rng.uniform(size=15)
        angle = np.pi * .6 * (q - .5)
        aux = np.column_stack([np.cos(angle / 2), np.sin(angle / 2)])
        eight = np.einsum("ni,nj->nij", state, aux).reshape(15, 256)
        expected = abs(state @ state.conj().T)**2 * np.cos(np.pi * .6 * (q[:, None] - q[None, :]) / 2)**2
        np.testing.assert_allclose(abs(eight @ eight.conj().T)**2, expected, atol=1e-12)

    def test_scientific_fit_does_not_read_test_labels(self):
        tree = ast.parse(Path(p.__file__).read_text())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in ("scientific_job", "fit_exact", "calibration_pair"):
            calls = [n for n in ast.walk(functions[name]) if isinstance(n, ast.Call)]
            self.assertFalse([n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "test_labels"])
            literals = [n.value for n in ast.walk(functions[name]) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            self.assertNotIn("y_test", literals)
            self.assertNotIn("y_gate", literals)

    def test_constant_dimension_and_holm(self):
        np.testing.assert_allclose(p.holm([.03, .001, .02]), [.04, .003, .04])
        result = p.kernel_metrics(np.eye(8), np.arange(8) % 2)
        self.assertAlmostEqual(result["entropy_effective_rank"], 8)


if __name__ == "__main__":
    unittest.main()
