import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector

from _shared import pca9_followup as p
from _shared import overnight_risk as old
from _shared.angle_encoding import AngleEncoding
import run_pca9_followup as runner


def fixture():
    rng = np.random.default_rng(1117)
    arrays = {part: rng.normal(size=(n, 9)) for part, n in (("train", 30), ("tune", 12), ("gate", 10))}
    arrays.update(y_train=np.tile([0, 1], 15), y_tune=np.tile([0, 1], 6), weights=np.ones(9) / 9)
    for part in p.PARTS:
        arrays["raw_" + part] = rng.normal(size=(len(arrays[part]), 12))
        arrays["risk_" + part] = rng.uniform(.05, .95, len(arrays[part]))
        arrays["rank_" + part] = old.midrank(arrays["risk_train"], arrays["risk_" + part])
        for prefix in ("raw_lr", "gbdt"):
            arrays[prefix + "_" + part] = arrays["risk_" + part].copy()
            arrays[prefix + "_rank_" + part] = arrays["rank_" + part].copy()
    record = {"pca_components": 9, "encoder": AngleEncoding.fit(arrays["train"], "linear_no_clip").parameters()}
    return arrays, record


class PCA9Tests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(p.PROTOCOL.read_text())

    def test_pinned_request_and_fixed_complete_scope(self):
        self.assertEqual(p.sha(p.REQUEST), p.REQUEST_SHA)
        self.assertEqual(self.protocol["pca_components"], 9)
        self.assertEqual(self.protocol["methods"], list(p.METHODS))
        self.assertEqual(len(p.jobs(self.protocol)), 870)
        self.assertEqual(len(set(p.jobs(self.protocol))), 870)
        self.assertNotIn("taiwan", self.protocol["datasets"])
        before = json.loads((p.ROOT / "docs/cmc_overnight_risk_v4_protocol_20261019.json").read_text())
        for field in ("seeds", "lambdas", "C_grid", "geometries", "nystrom_landmarks", "exact_search_max_rows"):
            self.assertEqual(self.protocol[field], before[field])

    def test_nine_qubit_binding_matches_individual_state(self):
        values = np.random.default_rng(7).uniform(0, np.pi, (3, 9))
        engine = p.DimensionStates(9, .5, 1)
        actual = engine(values)
        self.assertEqual(actual.shape, (3, 512))
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-12)
        individual = engine.backend.run(engine.circuit.assign_parameters(dict(zip(engine.parameters, values[0]))), shots=1).result()
        np.testing.assert_allclose(actual[0], np.asarray(individual.data(0)["state"]), atol=1e-12)

    def test_ten_qubit_auxiliary_matches_explicit_circuit_and_product(self):
        values = np.random.default_rng(8).uniform(0, np.pi, (3, 9))
        risks = np.array([.1, .5, .9])
        strength = .65
        engine = p.DimensionStates(9, .5, 1)
        states = engine(values)
        angles = np.pi * strength * (risks - .5)
        aux = np.column_stack([np.cos(angles / 2), np.sin(angles / 2)])
        combined = np.einsum("ni,nj->nij", states, aux).reshape(3, 1024)
        gram = old.primitive_kernel(combined, combined, "fidelity", 1.)
        factor = np.cos((angles[:, None] - angles[None, :]) / 2)**2
        np.testing.assert_allclose(gram, old.primitive_kernel(states, states, "fidelity", 1.) * factor, atol=1e-12)
        direct = []
        for row, angle in zip(values, angles):
            circuit = QuantumCircuit(10)
            fmap = zz_feature_map(9, reps=1, entanglement="linear", alpha=.5)
            circuit.compose(fmap.assign_parameters(dict(zip(fmap.parameters, row))), qubits=range(9), inplace=True)
            circuit.ry(angle, 9)
            direct.append(Statevector.from_instruction(circuit).data)
        np.testing.assert_allclose(gram, old.primitive_kernel(np.asarray(direct), np.asarray(direct), "fidelity", 1.), atol=1e-12)

    def test_nested_calibration_refits_nine_dimensional_preprocessing(self):
        rng = np.random.default_rng(9)
        X = pd.DataFrame(rng.normal(size=(90, 12)))
        X.attrs["nominal"] = []
        y = np.tile([0, 1], 45)
        with tempfile.TemporaryDirectory() as temporary:
            log = str(Path(temporary) / "fits.jsonl")
            model = p.risk_estimator(X, 2010, 9, log)
            model.fit(X, y)
            records = [json.loads(line) for line in Path(log).read_text().splitlines()]
            self.assertEqual(sorted(row["rows"] for row in records), [60, 60, 60, 90])
            self.assertTrue(all(row["components"] == 9 for row in records))
            self.assertFalse(model.ensemble)
            self.assertEqual(len(model.calibrated_classifiers_), 1)
            self.assertEqual(model.predict_proba(X.iloc[:2]).shape, (2, 2))

    def test_rank_deficiency_is_not_silently_reduced(self):
        with self.assertRaises(AssertionError):
            p.CheckedPCA(9).fit_transform(np.ones((20, 12)))

    def test_all_fixed_kernels_psd_diagonal_immutable_and_zero_identity(self):
        arrays, record = fixture()
        original = {key: value.copy() for key, value in arrays.items()}
        for learner in self.protocol["learners"]:
            baseline = p.context(arrays, record, "baseline", learner, "normal", 2010, self.protocol)
            reference = p.make_representation(arrays, record, baseline, "baseline", learner, "normal", 0., self.protocol["geometries"][0])
            expected = p.kernel(reference, ("train", slice(None)), ("train", slice(None)))
            for method in p.METHODS:
                ctx = p.context(arrays, record, method, learner, "normal", 2010, self.protocol)
                before = {key: value.copy() for key, value in ctx["angles"].items()}
                for strength in (0., .5, 1.):
                    rep = p.make_representation(arrays, record, ctx, method, learner, "normal", strength, self.protocol["geometries"][0])
                    gram = p.kernel(rep, ("train", slice(None)), ("train", slice(None)))
                    np.testing.assert_allclose(gram, gram.T, atol=1e-12)
                    np.testing.assert_allclose(np.diag(gram), 1, atol=1e-12)
                    self.assertGreaterEqual(np.linalg.eigvalsh(gram).min(), -1e-9)
                    if strength == 0:
                        np.testing.assert_allclose(gram, expected, atol=1e-12)
                    if learner == "RBF-SVM":
                        self.assertAlmostEqual(rep["gamma"], self.protocol["geometries"][0]["gamma_factor"] / (9 * ctx["angles"]["train"].var()))
                for key in before:
                    np.testing.assert_array_equal(before[key], ctx["angles"][key])
            for key in arrays:
                np.testing.assert_array_equal(original[key], arrays[key])

    def test_constant_gate_and_append_reduce_to_unmodified_kernel(self):
        arrays, record = fixture()
        for learner, method in __import__("itertools").product(self.protocol["learners"], p.METHODS[:2]):
            ctx = p.context(arrays, record, method, learner, "constant", 2010, self.protocol)
            zero = p.make_representation(arrays, record, ctx, method, learner, "constant", 0., self.protocol["geometries"][0])
            positive = p.make_representation(arrays, record, ctx, method, learner, "constant", .9, self.protocol["geometries"][0])
            np.testing.assert_allclose(p.kernel(zero, ("tune", slice(None)), ("train", slice(None))),
                p.kernel(positive, ("tune", slice(None)), ("train", slice(None))), atol=1e-12)

    def test_separable_map_is_nine_qubits(self):
        arrays, record = fixture()
        ctx = p.context(arrays, record, "rank_mixture", "QSVC", "separable", 2010, self.protocol)
        rep = p.make_representation(arrays, record, ctx, "rank_mixture", "QSVC", "separable", .5, self.protocol["geometries"][0])
        self.assertEqual(rep["values"]["train"].shape[1], 512)
        self.assertFalse(rep["details"]["circuits"][0]["entangled"])

    def test_full_row_exact_final_and_fresh_threshold_after_surrogate(self):
        arrays, record = fixture()
        protocol = {**self.protocol, "gpu_devices": [], "exact_search_max_rows": 16, "nystrom_landmarks": 16,
            "C_grid": [1.], "lambdas": [.05, .5], "geometries": [self.protocol["geometries"][0]], "exact_workers": 1}
        before = old.make_representation
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "full"
            folder = old.prepared_folder(output, "synthetic", 2010)
            folder.mkdir(parents=True)
            np.savez_compressed(folder / "inputs.npz", **arrays)
            p.write_json(folder / "audit.json", {**record, "input_sha256": p.sha(folder / "inputs.npz")})
            job = ("synthetic", 2010, "RBF-SVM", "rank_mixture", "normal")
            p.run_job(job, protocol, output, {})
            saved = json.loads((old.job_folder(output, job) / "complete.json").read_text())
            self.assertEqual(len(saved["trials"]), 3)
            with np.load(old.job_folder(output, job) / "exact_margins.npz") as scores:
                for role, row in saved["choices"].items():
                    self.assertEqual(row["fit_rows"], 30)
                    self.assertTrue(row["final_exact"])
                    self.assertEqual(row["final_model_fit_calls"], 1)
                    self.assertEqual(row["threshold"], p.base.threshold_for_cost(arrays["y_tune"], scores[role + "_tune"], 0.)[0])
            with self.assertRaises(AssertionError):
                p.run_job(job, protocol, output, {})
        self.assertIs(old.make_representation, before)

    def test_global_selection_lock_and_test_release_both_block_fitting(self):
        for name in ("selection_lock.json", "test_release.json"):
            with tempfile.TemporaryDirectory() as temporary:
                Path(temporary, name).write_text("{}")
                with self.assertRaises(AssertionError):
                    p.assert_can_fit(temporary)

    def test_memory_cpu_admission_preserves_twenty_percent_and_cgroup(self):
        state = {"host_total_gib": 188., "host_available_gib": 180., "cpu_percent": 50.,
            "cgroup_limit_gib": None, "cgroup_used_gib": None}
        scheduling = {"worker_budget_gib": 3.}
        self.assertTrue(runner.admitted(state, scheduling, self.protocol))
        self.assertFalse(runner.admitted({**state, "host_available_gib": 45.}, scheduling, self.protocol))
        self.assertFalse(runner.admitted({**state, "cpu_percent": 81.}, scheduling, self.protocol))
        self.assertFalse(runner.admitted({**state, "cgroup_limit_gib": 30., "cgroup_used_gib": 22.}, scheduling, self.protocol))


if __name__ == "__main__":
    unittest.main()
