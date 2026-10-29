import ast
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_professor_revision as r


class RevisionTests(unittest.TestCase):
    def test_permutation_preserves_distribution_and_pairing(self):
        arrays = {}
        for part, count in (("train", 100), ("tune", 30), ("gate", 25)):
            arrays["gbdt_" + part] = np.arange(count) / count
            arrays["gbdt_rank_" + part] = arrays["gbdt_" + part] ** 2
        result = r.permuted_risk(arrays, 10)
        again = r.permuted_risk(arrays, 10)
        for part, (p, q) in result.items():
            np.testing.assert_array_equal(np.sort(p), arrays["gbdt_" + part])
            np.testing.assert_array_equal(q, p ** 2)
            np.testing.assert_array_equal(p, again[part][0])
            self.assertFalse(np.array_equal(p, arrays["gbdt_" + part]))

    def test_matched_zero_rejects_approximate_or_ambiguous(self):
        choice = {"C": 1., "geometry": 2}
        good = {"C": 1., "geometry": 2, "lambda": 0., "approximate_search": False}
        self.assertIs(r.exact_zero_trial([good], choice), good)
        for rows in ([], [good, good], [{**good, "approximate_search": True}]):
            with self.assertRaises(ValueError):r.exact_zero_trial(rows, choice)

    def test_cold_accounting_replaces_not_adds_incremental_state(self):
        choice = {"training_seconds": 5., "test_inference_seconds": 2., "fit_seconds": 1.,
            "kernel_seconds": dict(train=2., tune=.2, gate=.3), "inference_seconds": dict(tune=.1, gate=.2),
            "representation_details": {"measured_cold_state_seconds_by_part": dict(train=7., tune=2., gate=3.),
                "state_seconds_by_part": dict(train=2., tune=1., gate=1.), "state_cache_reused": False}}
        got = r.cold_accounting(choice)
        self.assertEqual(got["cold_training_component_sum_s"], 10.)
        self.assertEqual(got["cold_test_component_sum_s"], 4.)

    def test_soft_gate_psd_and_unit_diagonal(self):
        q = np.linspace(0, 1, 17)
        factor = r.risk_factor(q, .7, "QSVC", "gbdt_soft_gate")
        np.testing.assert_allclose(np.diag(factor), 1.)
        self.assertGreater(np.linalg.eigvalsh(factor).min(), -1e-12)
        self.assertLess(factor[0, -1], factor[0, 1])

    def test_independent_qubit_factorization(self):
        q = np.array([.1, .4, .8])
        angle = np.pi * .6 * (q - .5)
        aux = np.column_stack([np.cos(angle / 2), np.sin(angle / 2)])
        np.testing.assert_allclose(abs(aux @ aux.T) ** 2, r.risk_factor(q, .6, "QSVC", "gbdt_rank_append"))

    def test_no_model_fit_calls_or_old_output_writes(self):
        tree = ast.parse(Path(r.__file__).read_text())
        fit_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("fit", "fit_transform", "partial_fit")]
        self.assertEqual(fit_calls, [])
        self.assertEqual(r.read(r.PROTOCOL)["status"], "retrospective_analysis_plan_not_preregistration")


if __name__ == "__main__":unittest.main()
