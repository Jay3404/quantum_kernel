import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis.reproduce_aggregates import ROOT, corrected_summary, holm, replay


class PublicReplayTests(unittest.TestCase):
    def test_holm_monotonic_adjustment(self):
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])

    def test_zero_variance_preserves_original_convention(self):
        result = corrected_summary(np.ones(10))
        self.assertEqual(result["raw_p"], 1.)
        self.assertTrue(np.isnan(result["ci_low"]))
        self.assertTrue(np.isnan(result["ci_high"]))

    def test_all_released_statistics_and_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = replay(ROOT, Path(temporary))
            self.assertEqual(receipt["paired_comparisons"], 632)
            self.assertEqual(receipt["fixed_model_sensitivity_rows"], 1440)
            self.assertFalse(receipt["customer_scores_read"])

    def test_release_contains_only_text_code_and_aggregates(self):
        manifest = json.loads((ROOT / "PUBLIC_MANIFEST.json").read_text())
        forbidden = {".pdf", ".tex", ".zip", ".npz", ".npy", ".pkl", ".joblib", ".xls", ".xlsx"}
        for name in manifest["files"]:
            self.assertNotIn(Path(name).suffix.lower(), forbidden)
            self.assertFalse(name.startswith(("manuscript/", "local_archive/", ".ssh/")))


if __name__ == "__main__":
    unittest.main()
