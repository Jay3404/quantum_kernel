import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from _shared import overnight_risk as n
from _shared import overnight_runtime_v3 as v3
from _shared import overnight_runtime_v4 as v4
import run_overnight_risk_v4 as runner
from test_overnight_risk import P, fixture


def prepared(output, arrays, record):
    folder = n.prepared_folder(output, "synthetic", 2010)
    folder.mkdir(parents=True)
    np.savez_compressed(folder / "inputs.npz", **arrays)
    n.write_json(folder / "audit.json", {**record, "input_sha256": n.sha(folder / "inputs.npz")})


class ContinuationTests(unittest.TestCase):
    def protocol(self):
        return {**P, "gpu_devices": [], "exact_search_max_rows": 16,
            "C_grid": [.1], "lambdas": [.05, .5], "geometries": [P["geometries"][0]], "nystrom_landmarks": 16}

    def test_scientific_settings_unchanged(self):
        new = json.loads(runner.PROTOCOL.read_text())
        previous = json.loads((runner.PROTOCOL.parent / "cmc_overnight_risk_v3_protocol_20261018.json").read_text())
        runtime_keys = {"run_id", "preflight_history", "implementation_revision", "resources", "workers", "continuation"}
        self.assertEqual({k: v for k, v in new.items() if k not in runtime_keys},
            {k: v for k, v in previous.items() if k not in runtime_keys})

    def test_complete_prefix_skips_all_search_fits_and_matches_exact_scores(self):
        self.check_resume(None)

    def test_partial_prefix_only_fits_missing_trials_and_matches(self):
        self.check_resume(1)

    def test_quantum_metric_prefix_matches_uninterrupted(self):
        self.check_resume(1, "QSVC", "pls_metric")

    def test_quantum_risk_prefix_matches_uninterrupted(self):
        self.check_resume(None, "QSVC", "rank_mixture")

    def check_resume(self, prefix, learner="RBF-SVM", method="rank_product"):
        arrays, record = fixture()
        protocol = self.protocol()
        job = ("synthetic", 2010, learner, method, "normal")
        with tempfile.TemporaryDirectory() as temporary:
            parent, new = Path(temporary) / "parent", Path(temporary) / "new"
            prepared(parent, arrays, record)
            prepared(new, arrays, record)
            np.savez_compressed(n.prepared_folder(new, "synthetic", 2010) / "confirmation_labels.npz",
                y_gate=np.array([object()], dtype=object))
            v3.run_job(job, "full", protocol, parent, {})
            source = n.job_folder(parent, job)
            target = n.job_folder(new, job) / "parent_checkpoint"
            target.mkdir(parents=True)
            progress = json.loads((source / "progress.json").read_text())
            count = len(progress["trials"]) if prefix is None else prefix
            with np.load(source / "margins.npz") as old:
                np.savez_compressed(target / "margins.npz", **{p: old[p][:count] for p in ("tune", "gate")})
            n.write_json(target / "progress.json", {**progress, "trials": progress["trials"][:count],
                "candidate_count": count, "margins_sha256": n.sha(target / "margins.npz")})
            if prefix is None:
                shutil.copy2(source / "parameter_lock.json", target / "parameter_lock.json")
            original = n.LinearSVC
            with patch.object(n, "LinearSVC", side_effect=lambda **kw: original(**kw)) as model:
                v4.run_job(job, "full", protocol, new, {})
            actual = n.job_folder(new, job)
            saved = json.loads((actual / "complete.json").read_text())
            self.assertEqual(model.call_count, len(progress["trials"]) - count)
            self.assertEqual(saved["inherited_search_trials"], count)
            self.assertFalse(saved["held_out_labels_read"])
            before = json.loads((source / "complete.json").read_text())
            for role in before["choices"]:
                self.assertEqual([before["choices"][role][k] for k in ("lambda", "C", "geometry", "threshold")],
                    [saved["choices"][role][k] for k in ("lambda", "C", "geometry", "threshold")])
            with np.load(source / "exact_margins.npz") as expected, np.load(actual / "exact_margins.npz") as result:
                for key in expected.files:
                    np.testing.assert_allclose(result[key], expected[key], atol=1e-12, rtol=1e-12)

    def test_checkpoint_rejects_corrupted_scores(self):
        arrays, record = fixture()
        protocol = self.protocol()
        job = ("synthetic", 2010, "RBF-SVM", "baseline", "normal")
        with tempfile.TemporaryDirectory() as temporary:
            prepared(temporary, arrays, record)
            v3.run_job(job, "full", protocol, temporary, {})
            folder = n.job_folder(temporary, job)
            content = json.loads((folder / "progress.json").read_text())
            content["margins_sha256"] = "invalid"
            n.write_json(folder / "progress.json", content)
            with self.assertRaises(AssertionError):
                v4.load_checkpoint(folder, arrays, protocol, "baseline")

    def test_adoption_preserves_parent_receipt_and_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "parent", root / "new"
            source.mkdir()
            n.write_json(source / "complete.json", {"source_sha256": {"old": "hash"}, "trials": [{"cost": 7}]})
            (source / "margins.npz").write_bytes(b"immutable fixture")
            before = n.sha(source / "complete.json")
            frozen = {}
            with patch.object(n, "ROOT", root):
                runner.adopt_complete(source, target, {"new": "hash"}, frozen)
            saved = json.loads((target / "complete.json").read_text())
            self.assertEqual(n.sha(source / "complete.json"), before)
            self.assertEqual(saved["parent_complete_sha256"], before)
            self.assertEqual(saved["new_model_fit_calls"], 0)
            self.assertEqual(saved["model_execution_source_sha256"], {"old": "hash"})
            self.assertEqual(n.sha(source / "margins.npz"), n.sha(target / "margins.npz"))

    def test_schedule_prioritizes_main_scopes_without_removing_controls(self):
        methods = ["rank_mixture", "pls_metric"]
        tasks = runner.legacy.jobs(P, methods, P["seeds"], True)
        ordered = sorted(tasks, key=runner.job_order)
        self.assertEqual(set(tasks), set(ordered))
        self.assertTrue(all(j[0] == "hmeq" and j[4] == "normal" for j in ordered[:70]))
        normal_end = max(i for i, j in enumerate(ordered) if j[4] == "normal")
        self.assertTrue(all(j[4] != "normal" for j in ordered[normal_end + 1:]))


if __name__ == "__main__":
    unittest.main()
