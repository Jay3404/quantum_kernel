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

from _shared import controlled as base
from _shared.quantum_ten import RiskStates
from _shared.german_zz_reporting import seed_differences
from _shared import german_zz_reporting as reporting
import run_german_zz_phase as run

PROTOCOL = json.loads(run.PROTOCOL.read_text())


class GermanZZTests(unittest.TestCase):
    def test_fixed_full_scope_and_job_grid(self):
        run.protocol_check(PROTOCOL)
        jobs = run.jobs(PROTOCOL, PROTOCOL["methods"], PROTOCOL["seeds"], True)
        self.assertEqual(len(jobs), 140)
        self.assertEqual(len(set(jobs)), 140)
        self.assertEqual({j[0] for j in jobs}, {"german_numeric", "south_german"})

    def test_scope_and_lambda_changes_rejected(self):
        for key, value in (("datasets", ["german_numeric"]), ("methods", ["reupload"]),
                           ("lambdas", [0., .5]), ("seeds", [10]), ("arms", ["quantum"])):
            protocol = copy.deepcopy(PROTOCOL)
            protocol[key] = value
            with self.assertRaises(AssertionError):
                run.protocol_check(protocol)

    def test_same_risk_phase_preserves_pair_fidelity(self):
        rng = np.random.default_rng(91)
        angles = rng.uniform(0, np.pi, (5, 7))
        weights = np.arange(1, 8) / 28
        engine = RiskStates("zz_phase", weights, np.ones(7) / np.sqrt(7))
        clean = engine(np.column_stack([angles, np.zeros(5)]))
        shared = engine(np.column_stack([angles, np.full(5, .7)]))
        np.testing.assert_allclose(base.fidelity(clean, clean), base.fidelity(shared, shared), atol=1e-12)
        distinct = engine(np.column_stack([angles, np.linspace(-.7, .7, 5)]))
        self.assertGreater(np.max(abs(base.fidelity(clean, clean) - base.fidelity(distinct, distinct))), 1e-5)
        self.assertEqual(clean.shape, (5, 128))

    def test_rz_changes_phase_not_basis_probabilities(self):
        rng = np.random.default_rng(92)
        angles = rng.uniform(0, np.pi, (3, 7))
        engine = RiskStates("zz_phase", np.ones(7) / 7, np.ones(7) / np.sqrt(7))
        clean = engine(np.column_stack([angles, np.zeros(3)]))
        risk = engine(np.column_stack([angles, [-.5, .1, .8]]))
        np.testing.assert_allclose(abs(clean) ** 2, abs(risk) ** 2, atol=1e-12)

    def test_seed_direction_is_control_minus_proposed(self):
        rows = []
        for role, cost in (("baseline", 100), ("proposed", 90), ("classical_best", 80), ("separable", 90), ("shuffled", 110)):
            rows.append({"dataset": "german_numeric", "candidate_method": "zz_phase", "seed": 10,
                "role": role, "cost": cost, "lambda": .25})
        result = seed_differences(pd.DataFrame(rows)).set_index("control")
        self.assertEqual(result.loc["baseline", "paired_cost_difference"], 10)
        self.assertEqual(result.loc["classical_best", "paired_cost_difference"], -10)
        self.assertEqual(result.loc["separable", "direction"], "동률")
        self.assertAlmostEqual(result.loc["baseline", "relative_cost_reduction_pct"], 10.)

    def test_full_report_uses_two_datasets_and_separate_multiplicity(self):
        groups = {}
        for dataset in PROTOCOL["datasets"]:
            for seed in PROTOCOL["seeds"]:
                common = {"dataset": dataset, "seed": seed, "C": 1., "gamma_factor": 1.,
                    "threshold": 0., "fit_rows": 800, "tune_cost": 50, "learner": "QSVC", "fit_seconds": .01}
                for method in ("zz_phase", "standard", "standard_convex", "standard_feature_specific"):
                    for arm in PROTOCOL["arms"] if method == "zz_phase" else ["quantum"]:
                        groups[dataset, seed, method, arm] = [{**common, "method": method, "arm": arm, "lambda": strength}
                            for strength in ([0.] if method == "standard" else [0., *PROTOCOL["lambdas"]])]
                groups[dataset, seed, "classical", "classical"] = [{**common, "learner": "RBF-SVM", "arm": "classical",
                    "method": method, "lambda": 0. if method == "baseline" else .05}
                    for method in ("baseline", "append", "original_convex", "original_feature_specific")]

        def synthetic_evaluate(output, choices, full, protocol, with_predictions=True):
            rows = []
            for choice in choices:
                cost = 100 - choice["seed"] / 10 if choice["role"] in ("proposed", "lambda_profile") else 110
                rows.append({**choice, "cost": cost, "fn": 5, "fp": cost - 25, "f1": .7, "recall": .8,
                    "balanced_accuracy": .7, "n_train": 800, "n_eval": 100, "n_tune": 100})
            return pd.DataFrame(rows), [pd.DataFrame({"explicitly_synthetic": [True]})] if with_predictions else []

        with tempfile.TemporaryDirectory() as temporary, patch.object(reporting, "collect", return_value=(groups, {}, [])), patch.object(reporting, "evaluate", side_effect=synthetic_evaluate):
            output = Path(temporary)
            receipt = reporting.report(output, [], {}, PROTOCOL)
            self.assertEqual(receipt["models"], 260)
            self.assertEqual(receipt["lambda_models"], 400)
            self.assertTrue(pd.read_csv(output / "paired_cost.csv").holm_family_size.eq(12).all())
            self.assertTrue(pd.read_csv(output / "lambda_paired_cost.csv").holm_family_size.eq(240).all())
            self.assertEqual(len(pd.read_csv(output / "seed_directions.csv")), 80)
            self.assertIn("German 계열", (output / "결과보고서.md").read_text())


if __name__ == "__main__":
    unittest.main()
