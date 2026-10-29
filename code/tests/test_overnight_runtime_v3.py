import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from _shared import overnight_risk as night
from _shared import overnight_runtime_v3 as runtime
from _shared.improved import sha
import test_overnight_risk as original_tests
import run_overnight_risk_v3 as run_v3

P, fixture = original_tests.P, original_tests.fixture


class AdaptedOriginalTests(original_tests.OvernightTests):
    def setUp(self):
        self.adapter = runtime.implementation()
        self.adapter.__enter__()
        self.addCleanup(self.adapter.__exit__, None, None, None)


class RuntimeRegressionTests(unittest.TestCase):
    def test_scientific_settings_are_unchanged(self):
        revised = json.loads(run_v3.PROTOCOL.read_text())
        implementation_keys = {"run_id", "preflight_history", "implementation_revision", "resources"}
        self.assertEqual({k: v for k, v in P.items() if k not in implementation_keys},
            {k: v for k, v in revised.items() if k not in implementation_keys})

    def test_context_survives_all_strengths_and_geometries(self):
        arrays, record = fixture()
        protocol = {**P, "metric_train_cap": 24,
            "alignment": {**P["alignment"], "sample_size": 24, "maxfev": 8}}
        for method in P["methods"]:
            for learner in P["learners"]:
                with self.subTest(method=method, learner=learner):
                    context = night.representations(arrays, record, method, learner, "normal", 2010, protocol)
                    before = copy.deepcopy(context)
                    for geometry in P["geometries"]:
                        for strength in (0., .05, .5, 1.):
                            rep = runtime.make_representation(arrays, record, context, method, learner,
                                "normal", strength, geometry)
                            k = runtime.representation_kernel(rep, ("train", slice(None)), ("train", slice(None)))
                            self.assertTrue(np.isfinite(k).all())
                            np.testing.assert_allclose(np.diag(k), 1., atol=1e-10)
                            self.assertGreater(np.linalg.eigvalsh(k).min(), -1e-8)
                            for key in ("angles", "metric"):
                                if before[key] is not None:
                                    for part in night.PARTS:
                                        np.testing.assert_array_equal(context[key][part], before[key][part])
                                        self.assertTrue(np.isrealobj(context[key][part]))

    def test_blocked_composites_match_frozen_algebra(self):
        arrays, record = fixture()
        for method in ("baseline", "lda_stretch", "rank_product", "logit_product",
                       "soft_gate", "hellinger_product", "uncertainty_mixture", "rank_mixture"):
            for learner in P["learners"]:
                context = night.representations(arrays, record, method, learner, "normal", 2010, P)
                rep = runtime.make_representation(arrays, record, context, method, learner,
                    "normal", .5, P["geometries"][0])
                for left, right in ((('train', slice(None)), ('train', slice(None))),
                                    (('tune', np.array([7, 1, 5, 0])), ('train', slice(None, None, 2)))):
                    expected = runtime._KERNEL(rep, left, right)
                    with patch.object(runtime, "BLOCK_ROWS", 3):
                        actual = runtime.representation_kernel(rep, left, right)
                    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_two_formerly_failing_workers_complete_multiple_lambdas(self):
        arrays, record = fixture()
        protocol = {**P, "gpu_devices": [], "C_grid": [.1],
            "smoke_lambdas": [.05, .5, 1.], "geometries": [P["geometries"][0]]}
        with tempfile.TemporaryDirectory() as temporary:
            folder = night.prepared_folder(temporary, "synthetic", 2010)
            folder.mkdir(parents=True)
            np.savez_compressed(folder / "inputs.npz", **arrays)
            night.write_json(folder / "audit.json", {**record, "input_sha256": sha(folder / "inputs.npz")})
            for method in ("residual_risk", "lda_stretch"):
                job = ("synthetic", 2010, "QSVC", method, "normal")
                runtime.run_job(job, "confirmation", protocol, temporary, {})
                saved = json.loads((night.job_folder(temporary, job) / "complete.json").read_text())
                self.assertEqual(len(saved["trials"]), 4)
                self.assertFalse(saved["held_out_labels_read"])

    def test_adapter_restores_module_after_exception(self):
        before = night.make_representation, night.representation_kernel
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with runtime.implementation():
                raise RuntimeError("sentinel")
        self.assertEqual(before, (night.make_representation, night.representation_kernel))


if __name__ == "__main__":
    unittest.main()
