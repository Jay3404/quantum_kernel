import unittest
import tempfile
from pathlib import Path
import joblib
import numpy as np

from run_budget_riskonly_followup import risk_grid, plans, probe_jobs, read, CONFIG, fit_risk
from run_risk_only_ablation import features
from analyze_pca7_professor_comments import risk_factor


class RiskOnlyGridTests(unittest.TestCase):
    def config(self):
        return {"independent_risk_only": {"lambda_grid": [round(.05 * i, 2) for i in range(1, 21)], "C_grid": [.01, 1., 100.]}}

    def test_distinct_grid_counts(self):
        self.assertEqual(len(risk_grid({"context": "sg"}, self.config())), 60)
        self.assertEqual(len(risk_grid({"context": "qra"}, self.config())), 60)
        self.assertEqual(len(risk_grid({"context": "rra"}, self.config())), 180)

    def test_exact_finite_feature_identity(self):
        q = np.array([0., .12, .5, .77, 1.])
        for method in ("gbdt_soft_gate", "gbdt_rank_append"):
            for strength in (.05, .5, 1.):
                x, kind = features(q, strength, "QSVC", method)
                self.assertEqual(kind, "linear")
                np.testing.assert_allclose(x @ x.T, risk_factor(q, strength, "QSVC", method), atol=1e-14)

    def test_no_zero_in_positive_search(self):
        self.assertTrue(all(r["lambda"] > 0 for r in risk_grid({"context": "sg"}, self.config())))

    def test_complete_job_plan_and_memory_probes(self):
        jobs = plans(read(CONFIG))
        self.assertEqual(len(jobs), 200)
        self.assertEqual(sum(j["kind"] == "budget" for j in jobs), 80)
        probes = probe_jobs({"jobs": jobs})
        self.assertEqual(len(probes), 11)
        self.assertEqual(len({(j["dataset"], j["kernel"], j["method"]) for j in probes[:8]}), 8)

    def test_synthetic_validation_selection_and_saved_prediction(self):
        rng = np.random.default_rng(3)
        arrays = {"train": rng.normal(size=(30, 7)), "y_train": np.tile([0, 1], 15),
                  "y_tune": np.tile([0, 1], 5)}
        for part, size in [("train", 30), ("tune", 10), ("gate", 6)]:
            arrays["gbdt_rank_" + part] = rng.uniform(size=size)
        record = {"encoder": {"center": np.zeros(7), "scale": np.ones(7)}}
        config = {"independent_risk_only": {"lambda_grid": [.1, .5, 1.], "C_grid": [.01, 1.]}}
        for context, kernel, method in [("sg", "RBF-SVM", "gbdt_soft_gate"),
                                       ("qra", "QSVC", "gbdt_rank_append"),
                                       ("rra", "RBF-SVM", "gbdt_rank_append")]:
            job = {"context": context, "kernel": kernel, "method": method}
            with tempfile.TemporaryDirectory() as temporary:
                out = Path(temporary)
                best, calls = fit_risk(job, arrays, record, out, config, False)
                self.assertEqual(calls, len(risk_grid(job, config)))
                trials = read(out / "validation_selection.json")["trials"]
                self.assertEqual(best, min(trials, key=lambda r: (r["tune_cost"], r["lambda"], r["C"], r["geometry"])))
                model = joblib.load(out / "model.joblib")
                x, _ = features(arrays["gbdt_rank_gate"], best["lambda"], kernel, method)
                with np.load(out / "margins.npz") as saved:
                    np.testing.assert_allclose(model.decision_function(x), saved["gate"])


if __name__ == "__main__":
    unittest.main()
