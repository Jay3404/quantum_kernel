import ast
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_risk_only_ablation as r


class RiskOnlyTests(unittest.TestCase):
    def test_all_exact_risk_gram_identities(self):
        q = np.random.default_rng(333).uniform(size=45)
        for method in ("gbdt_soft_gate", "gbdt_rank_append"):
            for kernel in ("RBF-SVM", "QSVC"):
                for strength in (.05, .4, 1.):
                    values, kind = r.features(q, strength, kernel, method)
                    actual = values @ values.T if kind == "linear" else rbf_kernel(values, gamma=.37)
                    np.testing.assert_allclose(actual, r.risk_factor(q, strength, kernel, method, .37), atol=1e-12)

    def test_native_libsvm_matches_precomputed_factor(self):
        rng = np.random.default_rng(10)
        q = rng.uniform(size=80)
        y = ((q + rng.normal(0, .2, 80)) > .65).astype(int)
        for method in ("gbdt_soft_gate", "gbdt_rank_append"):
            for kernel in ("RBF-SVM", "QSVC"):
                values, kind = r.features(q, .7, kernel, method)
                native = SVC(C=1., kernel=kind, gamma=.3 if kind == "rbf" else "scale", class_weight="balanced", tol=1e-8).fit(values, y)
                matrix = r.risk_factor(q, .7, kernel, method, .3)
                gram = SVC(C=1., kernel="precomputed", class_weight="balanced", tol=1e-8).fit(matrix, y)
                np.testing.assert_allclose(native.decision_function(values), gram.decision_function(matrix), atol=1e-5)

    def test_complete_universe_and_resource_bound(self):
        self.assertEqual(len(r.jobs()), 160)
        self.assertEqual(len(set(r.jobs())), 160)
        self.assertEqual(r.configuration()["expected_jobs"], 160)

    def test_no_test_labels_in_worker(self):
        tree = ast.parse(Path(r.__file__).read_text())
        functions = {x.name: x for x in tree.body if isinstance(x, ast.FunctionDef)}
        for name in ("fit_job", "load"):
            calls = [x for x in ast.walk(functions[name]) if isinstance(x, ast.Call)]
            self.assertFalse(any(isinstance(x.func, ast.Name) and x.func.id == "labels" for x in calls))
        self.assertIn("selection_lock.json", ast.unparse(functions["fit_job"]))

    def test_feature_input_validation(self):
        for q in (np.array([-.1, .5]), np.array([.5, np.nan]), np.ones((3, 2))):
            with self.assertRaises(AssertionError):r.features(q, .5, "QSVC", "gbdt_soft_gate")


if __name__ == "__main__":unittest.main()
