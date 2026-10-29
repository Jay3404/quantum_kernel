"""Read-only third-feedback analysis; no model fitting or new selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import tarfile
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from analyze_pca7_professor_comments import (
    audit_oof, backbone, corrected, holm, kernel_metrics, metrics, risk_factor, sha, threshold)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/cmc_professor_revision_protocol_20261023.json"
LABELS = {"german_numeric": "German Numeric", "south_german": "South German", "hmeq": "HMEQ", "taiwan": "Taiwan"}
METHODS = {"gbdt_soft_gate": "Soft Gate", "gbdt_rank_append": "Rank Append"}


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
        handle.write("\n")


def read(path):
    return json.loads(Path(path).read_text())


def permuted_risk(arrays, seed):
    result = {}
    for index, part in enumerate(("train", "tune", "gate")):
        order = np.random.default_rng(seed + 7919 + index).permutation(len(arrays["gbdt_" + part]))
        result[part] = (arrays["gbdt_" + part][order], arrays["gbdt_rank_" + part][order])
    return result


def exact_zero_trial(trials, chosen):
    candidates = [row for row in trials if row["lambda"] == 0 and row["C"] == chosen["C"] and row["geometry"] == chosen["geometry"]]
    if len(candidates) != 1 or candidates[0]["approximate_search"]:
        raise ValueError("One full-row exact lambda0 candidate is required")
    return candidates[0]


def cold_accounting(choice):
    detail = choice["representation_details"]
    cold = detail["measured_cold_state_seconds_by_part"]
    incremental = detail["state_seconds_by_part"]
    return {"selected_training_recorded_s": choice["training_seconds"],
        "selected_test_recorded_s": choice["test_inference_seconds"],
        "cold_training_component_sum_s": choice["training_seconds"] - incremental["train"] + cold["train"],
        "cold_test_component_sum_s": choice["test_inference_seconds"] - incremental["gate"] + cold["gate"],
        "state_train_cold_s": cold["train"], "state_test_cold_s": cold["gate"],
        "state_train_incremental_s": incremental["train"], "state_test_incremental_s": incremental["gate"],
        "state_cache_reused": detail["state_cache_reused"], "fit_s": choice["fit_seconds"],
        **{"gram_" + part + "_s": choice["kernel_seconds"][part] for part in ("train", "tune", "gate")},
        **{"decision_" + part + "_s": choice["inference_seconds"][part] for part in ("tune", "gate")}}


class Inputs:
    def __init__(self):
        self.hashes = {}

    def consume(self, path):
        path = Path(path)
        name = str(path.relative_to(ROOT))
        digest = sha(path)
        assert name not in self.hashes or self.hashes[name] == digest, name
        self.hashes[name] = digest
        return path

    def json(self, path):
        return read(self.consume(path))

    def csv(self, path):
        return pd.read_csv(self.consume(path), float_precision="round_trip")

    def npz(self, path):
        with np.load(self.consume(path), allow_pickle=False) as saved:
            return dict(saved)

    def verify(self):
        for name, digest in self.hashes.items():
            assert sha(ROOT / name) == digest, name


def only(frame, **filters):
    for key, value in filters.items():
        frame = frame[frame[key].eq(value)]
    assert len(frame) == 1, filters
    return frame.iloc[0]


def analyze(output):
    tick = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    consumed = Inputs()
    config = consumed.json(PROTOCOL)
    for name in (Path(__file__), ROOT / "code/tests/test_professor_revision.py"):
        consumed.consume(name)
    with tarfile.open(output / "analysis_source_snapshot.tar.gz", "w:gz") as archive:
        for name in list(consumed.hashes):
            archive.add(ROOT / name, arcname=name)
    parent = ROOT / "code/results" / config["parent_run"]
    full = parent / "full"
    follow = ROOT / "code/results" / config["followup_run"]
    for folder, count in ((parent, 1160), (follow, 200)):
        meta = consumed.json(folder / "metadata.json")
        assert meta["status"] == "completed" and meta["completed_jobs"] == count and not meta["failed_jobs"]
        assert consumed.json(folder / "validation_receipt.json")["status"] == "passed"
        for name, digest in meta["source_sha256"].items():
            assert sha(consumed.consume(ROOT / name)) == digest, name
        for name, digest in meta.get("parent_sha256", {}).items():
            assert sha(consumed.consume(ROOT / name)) == digest, name
    lock = consumed.json(full / "selection_lock.json")
    release = consumed.json(full / "test_release.json")
    assert release["selection_lock_sha256"] == sha(full / "selection_lock.json") and release["no_further_fitting"]
    receipts = lock["candidate_receipts_sha256"]
    raw = consumed.csv(full / "seed_metrics.csv")
    pairs = consumed.csv(full / "paired_cost.csv")
    assert len(pairs) == 184
    np.testing.assert_allclose(pairs.p_holm, holm(pairs.p_value.to_numpy()), rtol=1e-12)
    prior_geometry = consumed.csv(follow / "mechanism_seed_metrics.csv")
    follow_metrics = consumed.csv(follow / "new_seed_metrics.csv")
    new_pairs = consumed.csv(follow / "paired_statistics.csv")
    for family, count in (("primary_cost", 100), ("secondary_calibration", 40)):
        rows = new_pairs[new_pairs.testing_family.eq(family)]
        assert len(rows) == count
        np.testing.assert_allclose(rows.p_holm, holm(rows.p_value.to_numpy()), rtol=1e-12)
    predictions = consumed.csv(full / "predictions.csv.gz")
    labels = predictions[predictions.kernel.eq("QSVC") & predictions.family.eq("baseline") & predictions.arm.eq("normal") & predictions.role.eq("baseline")]
    assert not labels.duplicated(["dataset", "seed", "row_id"]).any()
    del predictions
    geometry_rows, shuffle_rows, shuffle_audits, runtime, fairness, preparation = [], [], [], [], [], []
    gram_cache = {}

    def complete(folder):
        path = folder / "complete.json"
        result = consumed.json(path)
        assert sha(path) == receipts[str(path.relative_to(ROOT))]
        assert result["held_out_labels_read"] is False
        for name, key in (("margins.npz", "margins_sha256"), ("exact_margins.npz", "exact_margins_sha256"), ("parameter_lock.json", "parameter_lock_sha256")):
            assert sha(consumed.consume(folder / name)) == result[key]
        if "geometry_sha256" in result:
            assert sha(consumed.consume(folder / "geometry.npz")) == result["geometry_sha256"]
        return result

    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            prepared = full / "prepared" / dataset / f"seed_{seed:04d}"
            audit = consumed.json(prepared / "audit.json")
            arrays = consumed.npz(prepared / "inputs.npz")
            assert sha(prepared / "inputs.npz") == audit["input_sha256"]
            split = consumed.json(prepared / "split.json")["actual"]
            fold_audit = audit_oof(prepared, arrays, consumed.consume)
            assert fold_audit["overlap_violations"] == 0
            y = labels[labels.dataset.eq(dataset) & labels.seed.eq(seed)].set_index("row_id").reindex(split["test"]).y.to_numpy(dtype=int)
            assert len(y) == len(arrays["gate"]) and np.isin(y, [0, 1]).all()
            shuffled = permuted_risk(arrays, seed)
            for part, (probability, rank) in shuffled.items():
                np.testing.assert_array_equal(np.sort(probability), np.sort(arrays["gbdt_" + part]))
                np.testing.assert_array_equal(np.sort(rank), np.sort(arrays["gbdt_rank_" + part]))
            preparation.append({"dataset": dataset, "seed": seed,
                "feature_fit_s": audit["times"]["feature_fit_seconds"],
                "pca_lr_prepare_s": audit["times"]["pca_lr_fit_prediction_seconds"],
                "xgb_oof_full_prepare_s": audit["teachers"]["gbdt"]["elapsed_seconds"],
                "raw_lr_prepare_s": audit["teachers"]["raw_lr"]["elapsed_seconds"]})
            reference_folder = full / "candidates" / dataset / f"seed_{seed:04d}" / "reference/reference/normal"
            reference = complete(reference_folder)
            candidates = [r for r in reference["trials"] if r["learner"] == "XGBoost" and r["method"] == "raw"]
            assert len(candidates) == 2 and {r["depth"] for r in candidates} == {3, 6}
            chosen_reference = min(candidates, key=lambda r: (r["tune_cost"], str(r["depth"])))
            ref_row = only(raw, dataset=dataset, seed=seed, role="reference_raw_XGBoost")
            assert int(ref_row.score_index) == chosen_reference["score_index"]
            saved_reference = consumed.npz(reference_folder / "margins.npz")
            rtune = saved_reference["tune"][chosen_reference["score_index"]]
            rgate = saved_reference["gate"][chosen_reference["score_index"]]
            assert chosen_reference["fit_rows"] == len(split["train"])
            assert metrics(arrays["y_tune"], rtune, chosen_reference["threshold"])["cost"] == chosen_reference["tune_cost"]
            assert metrics(y, rgate, chosen_reference["threshold"])["cost"] == ref_row.cost
            # Probability thresholds use a different tie anchor from SVM margins.
            from _shared.professor_followup import cost_cut
            assert cost_cut(arrays["y_tune"], rtune) == chosen_reference["threshold"]
            fairness.append({"dataset": dataset, "seed": seed, "n_train": len(split["train"]),
                "n_validation": len(split["validation"]), "n_test": len(split["test"]),
                "split_sha256": sha(prepared / "split.json"), "xgb_depth": chosen_reference["depth"],
                "xgb_candidates": 2, "svm_positive_candidates": 180, "svm_zero_candidates": 9,
                "xgb_raw_features": arrays["raw_train"].shape[1], "svm_backbone_features": arrays["train"].shape[1],
                "same_split_fit_rows_cost_objective": True, "equal_search_budget": False,
                "equal_input_information": False, "svm_proxy_search": len(split["train"]) > 4096})
            runtime.append({"dataset": dataset, "seed": seed, "kernel": "XGB", "method": "standalone_depth3_or6",
                "fit_s": chosen_reference["fit_seconds"],
                "validation_plus_test_inference_s": chosen_reference["all_holdout_inference_seconds"],
                "selected_training_recorded_s": chosen_reference["training_seconds"]})
            for kernel in config["kernels"]:
                for method in config["methods"]:
                    folder = full / "candidates" / dataset / f"seed_{seed:04d}" / kernel / method / "normal"
                    actual = complete(folder)
                    choice = actual["choices"]["proposed"]
                    assert choice["fit_rows"] == len(split["train"]) and choice["final_model_fit_calls"] == 1
                    margins = consumed.npz(folder / "exact_margins.npz")
                    actual_metric = metrics(y, margins["proposed_gate"], choice["threshold"])
                    expected_row = only(raw, dataset=dataset, seed=seed, kernel=kernel, family=method, arm="normal", role="proposed")
                    assert actual_metric["cost"] == expected_row.cost
                    assert threshold(arrays["y_tune"], margins["proposed_tune"]) == choice["threshold"]
                    runtime.append({"dataset": dataset, "seed": seed, "kernel": kernel, "method": method, **cold_accounting(choice)})
                    positions = np.array(actual["geometry_positions"]["exact_proposed"])
                    gid = choice["geometry"]
                    key = (dataset, seed, kernel, gid)
                    if key not in gram_cache:
                        gram_cache[key] = backbone(arrays, audit, positions, kernel, read(ROOT / "docs/cmc_pca7_repeated_protocol_20261020.json")["geometries"][gid])
                    base_gram, gamma = gram_cache[key]
                    grams = consumed.npz(folder / "geometry.npz")
                    proposed_gram = grams["exact_proposed"]
                    expected_gram = base_gram * risk_factor(arrays["gbdt_rank_gate"][positions], choice["lambda"], kernel, method, gamma)
                    error = float(np.max(abs(expected_gram - proposed_gram)))
                    assert error < 1e-8, (dataset, seed, kernel, method, error)
                    if dataset in ("german_numeric", "south_german"):
                        zero = exact_zero_trial(actual["trials"], choice)
                        scores = consumed.npz(folder / "margins.npz")
                        zero_score = scores["gate"][zero["score_index"]]
                        assert zero["fit_rows"] == len(split["train"])
                        assert threshold(arrays["y_tune"], scores["tune"][zero["score_index"]]) == zero["threshold"]
                        source = "existing_exact_search_lambda0_at_actual_C_geometry"
                    else:
                        controlled = follow / "science/backbone" / dataset / f"seed_{seed:04d}" / kernel / method
                        done = consumed.json(controlled / "complete.json")
                        assert sha(consumed.consume(controlled / "margins.npz")) == done["margins_sha256"]
                        zero = done["choices"]["matched_backbone"]
                        assert zero["C"] == choice["C"] and zero["geometry"] == gid and zero["lambda"] == 0
                        zero_score = consumed.npz(controlled / "margins.npz")["matched_backbone_gate"]
                        source = "previous_full_row_matched_backbone_refit"
                    before, after = kernel_metrics(base_gram, y[positions]), kernel_metrics(proposed_gram, y[positions])
                    row = {"dataset": dataset, "seed": seed, "kernel": kernel, "method": method,
                        "C": choice["C"], "geometry": gid, "lambda": choice["lambda"], "n_subset": len(positions),
                        "matched_backbone_source": source, "kernel_product_max_error": error,
                        "full_cost_reduction": metrics(y, zero_score, zero["threshold"])["cost"] - actual_metric["cost"],
                        "subset_cost_reduction": metrics(y[positions], zero_score[positions], zero["threshold"])["cost"] - metrics(y[positions], margins["proposed_gate"][positions], choice["threshold"])["cost"]}
                    for field, metric in (("alignment", "centered_label_alignment"), ("effective_rank", "entropy_effective_rank")):
                        row.update({field + "_before": before[metric], field + "_after": after[metric], field + "_delta": after[metric] - before[metric]})
                    if dataset in ("hmeq", "taiwan"):
                        previous = only(prior_geometry, dataset=dataset, seed=seed, kernel=kernel, method=method)
                        for field in ("full_cost_reduction", "subset_cost_reduction", "alignment_delta", "effective_rank_delta"):
                            assert np.isclose(row[field], previous[field], atol=1e-9, rtol=1e-9)
                    geometry_rows.append(row)
                    control_folder = folder.parent / "shuffled"
                    control = complete(control_folder)
                    selected = control["choices"]["proposed"]
                    assert selected["fit_rows"] == len(split["train"]) and selected["final_model_fit_calls"] == 1
                    saved = consumed.npz(control_folder / "exact_margins.npz")
                    assert threshold(arrays["y_tune"], saved["proposed_tune"]) == selected["threshold"]
                    control_metric = metrics(y, saved["proposed_gate"], selected["threshold"])
                    archived = only(raw, dataset=dataset, seed=seed, kernel=kernel, family=method, arm="shuffled", role="proposed")
                    assert control_metric["cost"] == archived.cost
                    shuffle_rows.append({"dataset": dataset, "seed": seed, "kernel": kernel, "method": method,
                        "actual_cost": actual_metric["cost"], "shuffled_cost": control_metric["cost"],
                        "delta": control_metric["cost"] - actual_metric["cost"],
                        "actual_validation_cost": choice["tune_cost"], "shuffled_validation_cost": selected["tune_cost"],
                        "actual_fn": actual_metric["fn"], "actual_fp": actual_metric["fp"],
                        "shuffled_fn": control_metric["fn"], "shuffled_fp": control_metric["fp"]})
                    shuffle_positions = np.array(control["geometry_positions"]["exact_proposed"])
                    np.testing.assert_array_equal(shuffle_positions, positions)
                    key = (dataset, seed, kernel, selected["geometry"])
                    if key not in gram_cache:
                        gram_cache[key] = backbone(arrays, audit, positions, kernel, read(ROOT / "docs/cmc_pca7_repeated_protocol_20261020.json")["geometries"][selected["geometry"]])
                    bg, g = gram_cache[key]
                    sg = consumed.npz(control_folder / "geometry.npz")["exact_proposed"]
                    permutation_error = float(np.max(abs(sg - bg * risk_factor(shuffled["gate"][1][positions], selected["lambda"], kernel, method, g))))
                    assert permutation_error < 1e-8
                    shuffle_audits.append({"dataset": dataset, "seed": seed, "kernel": kernel, "method": method,
                        "raw_score_and_rank_multisets_preserved_all_parts": True,
                        "permutes_within_each_split_only": True, "labels_used_to_generate_permutation": False,
                        "same_permutation_across_kernels_and_methods": True, "kernel_product_max_error": permutation_error,
                        "permutation_draws_per_seed": 1, "independently_validation_tuned_control": True})
            # Keep the geometry cache bounded to one dataset/seed.
            gram_cache.clear()
            print(json.dumps({"dataset": dataset, "seed": seed, "geometry_rows": len(geometry_rows), "shuffled_jobs_verified": len(shuffle_audits)}), flush=True)
    assert len(geometry_rows) == len(shuffle_rows) == len(shuffle_audits) == 160
    geometry = pd.DataFrame(geometry_rows)
    shuffled = pd.DataFrame(shuffle_rows)
    contrasts = []
    for (dataset, kernel, method), group in shuffled.groupby(["dataset", "kernel", "method"], sort=False):
        group = group.sort_values("seed")
        assert group.seed.tolist() == config["seeds"]
        result = corrected(group.delta.to_numpy())
        previous = only(pairs, dataset=dataset, kernel=kernel, family=method, contrast="shuffled")
        for field in ("mean_difference", "ci_low", "ci_high", "p_value"):
            assert np.isclose(result[field], previous[field], equal_nan=True)
        contrasts.append({"dataset": dataset, "kernel": kernel, "method": method,
            "actual_mean": group.actual_cost.mean(), "actual_sd": group.actual_cost.std(ddof=1),
            "shuffled_mean": group.shuffled_cost.mean(), "shuffled_sd": group.shuffled_cost.std(ddof=1),
            **result, "p_holm_original184": previous.p_holm, "holm_family_size": 184})
    correlations = []
    for (dataset, kernel, method), group in geometry.groupby(["dataset", "kernel", "method"], sort=False):
        for x in ("alignment_delta", "effective_rank_delta"):
            for y in ("full_cost_reduction", "subset_cost_reduction"):
                rho = float(spearmanr(group[x], group[y]).statistic) if group[x].nunique() > 1 and group[y].nunique() > 1 else np.nan
                correlations.append({"dataset": dataset, "kernel": kernel, "method": method, "x": x, "y": y,
                    "spearman_rho": rho, "n_seeds": 10, "scope": "descriptive_overlapping_holdouts_no_iid_p_no_causal_claim"})
    for name, frame in (("geometry_seed_metrics.csv", geometry), ("geometry_correlations.csv", pd.DataFrame(correlations)),
            ("shuffled_seed_metrics.csv", shuffled), ("shuffled_contrasts.csv", pd.DataFrame(contrasts)),
            ("shuffled_distribution_audit.csv", pd.DataFrame(shuffle_audits)), ("fairness_seed_audit.csv", pd.DataFrame(fairness)),
            ("runtime_seed_components.csv", pd.DataFrame(runtime)), ("preparation_runtime.csv", pd.DataFrame(preparation))):
        frame.to_csv(output / name, index=False)
    figure(output, geometry, pd.DataFrame(correlations), config)
    report(output, config, pd.DataFrame(contrasts), pd.DataFrame(correlations), pd.DataFrame(runtime), pd.DataFrame(preparation))
    consumed.verify()
    write(output / "input_manifest.json", {"utc": now(), "sha256": consumed.hashes})
    write(output / "validation_receipt.json", {"utc": now(), "status": "passed", "verification_status": "ANALYZED",
        "scope": "Independent saved-score/threshold/statistic recalculation, full-subset statevector Gram reconstruction and frozen-code/input integrity; not independent retraining or external confirmation",
        "shuffled_jobs_verified": 160, "dataset_seed_split_OOF_audits": 40, "fairness_audits": 40,
        "geometry_seed_rows": 160, "descriptive_correlations": 64, "shuffled_contrasts_reproduced": 16,
        "original_holm184_preserved": True, "followup_holm100_and40_preserved": True,
        "new_fits": 0, "new_parameter_or_threshold_selections": 0, "old_files_modified": False,
        "elapsed_seconds": time.perf_counter() - tick, "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "statistical_fallacy_scan": {"simpson": "dataset/kernel-stratified; no pooled independent-dataset claim",
            "ecological": "no individual causal claims", "berkson": "data-acquisition selection unresolved",
            "base_rate": "FN/FP and same cohort retained", "regression_to_mean": "favorable HMEQ selection remains exploratory",
            "survivorship": "all160 shuffled and160 geometry cases retained", "multiple_testing": "all original families preserved; no retrospective smaller-family significance",
            "garden_of_forking_paths": "post-hoc status and future-versus-current hypotheses separated",
            "correlation_causation": "rho descriptive only", "reverse_causation": "no causal interpretation",
            "selection_bias": "previously examined datasets/seeds/tests; no external confirmation"}})
    outputs = {str(path.relative_to(output)): sha(path) for path in output.rglob("*") if path.is_file()}
    write(output / "output_manifest.json", {"utc": now(), "sha256": outputs,
        "exclusions": ["output_manifest.json", "independent_validation_receipt.json"], "self_reference_excluded": True})
    return {"status": "passed", "output": str(output), "geometry_rows": 160, "shuffled_jobs": 160}


def figure(output, rows, correlations, config):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharey="col")
    colors = {"gbdt_soft_gate": "#18765d", "gbdt_rank_append": "#b43c73"}
    markers = {"gbdt_soft_gate": "o", "gbdt_rank_append": "^"}
    for col, dataset in enumerate(config["datasets"]):
        for line, kernel in enumerate(config["kernels"]):
            ax = axes[line, col]
            for index, method in enumerate(config["methods"]):
                group = rows[rows.dataset.eq(dataset) & rows.kernel.eq(kernel) & rows.method.eq(method)]
                ax.scatter(group.alignment_delta, group.full_cost_reduction, s=40, alpha=.8, color=colors[method], marker=markers[method], label=METHODS[method])
                rho = only(correlations, dataset=dataset, kernel=kernel, method=method, x="alignment_delta", y="full_cost_reduction").spearman_rho
                ax.text(.03, .96 - .075 * index, f"{METHODS[method]} rho={rho:.2f}", color=colors[method], transform=ax.transAxes, va="top", fontsize=9)
            ax.axhline(0, color="#777777", linewidth=.7)
            ax.axvline(0, color="#777777", linewidth=.7)
            ax.grid(alpha=.15)
            ax.set_title(f"{LABELS[dataset]} | {kernel}", fontsize=11)
            if line == 1:ax.set_xlabel("Alignment change", fontsize=10)
            if col == 0:ax.set_ylabel("Matched backbone cost - risk cost", fontsize=10)
            ax.tick_params(labelsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, .995))
    fig.text(.5, .012, "10 overlapping holdouts per method; matched C/geometry; descriptive associations, not causal. German representations are related.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .945))
    fig.savefig(output / "kernel_alignment_cost_all_datasets.png", dpi=180)
    fig.savefig(output / "kernel_alignment_cost_all_datasets.pdf")
    plt.close(fig)


def report(output, config, contrasts, correlations, runtime, preparation):
    lines = ["# 교수님 3차 피드백: 핵심 통제 실험 및 원고 개정안", "", "## Material Passport", "",
        "상태: ANALYZED. 이미 완료·감사·보관한 PCA7 및2차 후속실험을 읽기 전용으로 재검증했다. 새 모델 학습·파라미터 선정·임계값 선정은0건이다. 기존160개 Shuffled Risk 대조를 요청 조건과 대조하고, 동일C/기하의 커널 분석을4자료160행으로 확장했다. 기존에 본 테스트·시드의 사후 분석이며 독립 확인 실험이 아니다.", "",
        "## 1. 연구 방향과 핵심 결론", "",
        "가제: Risk-Conditioned Kernel Learning for Cost-Sensitive Credit Classification: A Comparative Study of Classical and Quantum Models", "",
        "Quantum-specific Advantage나 실무적 우월성은 주장하지 않는다. HMEQ에서 검증 튜닝 XGB 단독109.5±13.7, RBF Soft Gate141.4±20.8, QSVC Rank Append140.7±21.0이다. XGB 단독이 더 낮은 관측 평균비용을 보였다는 결과를 본문에 유지한다. 위험점수용160-tree/depth3 XGB와 단독 참조200-tree/depth3·6 XGB는 서로 다른 모델이다.", "",
        "## 2. Shuffled Risk Control", "",
        "기존 PCA7에4자료×10시드×2커널×2방법=160개가 이미 포함돼 있었다. 새로 수행했다고 표현하지 않는다. 학습·검증·테스트 각각 안에서 XGB 점수와 대응 midrank를 같은 순열로 섞었다. 원점수/순위의 각 구간 주변분포는 정확히 보존하며 고객과의 대응을 제거한다. 레이블로 순열을 만들지 않고 두 커널/방법이 같은 순열을 공유한다. 훈련은5OOF 점수, 검증·테스트는 전체학습 교사의 점수라는 원래 경계를 유지한다. 시드당 순열1회라는 한계가 있다.", "",
        "실제/셔플은 같은 격자에서 각각 검증 선정했으므로 정보가 있는 파이프라인과 없는 파이프라인의 비교다. C/기하/lambda를 모두 고정한 순수 단일요인 비교라고 부르지는 않는다. 저장된 셔플 Gram을 같은 backbone×순열 위험커널로 재구성해 일치함도 확인했다. 비용=5FN+FP, 평균±표본표준편차. 양수 차이는 실제 위험정보가 유리함을 뜻한다.", "",
        "| 자료 | 커널 | 방법 | 실제 비용 | Shuffled 비용 | 차이 | 개별 근사95% CI | 원p | Holm184 p | 개선/동률/악화 |", "|---|---|---|---:|---:|---:|---|---:|---:|---|"]
    for r in contrasts.itertuples():
        lines.append(f"| {LABELS[r.dataset]} | {r.kernel} | {METHODS[r.method]} | {r.actual_mean:.1f}±{r.actual_sd:.1f} | {r.shuffled_mean:.1f}±{r.shuffled_sd:.1f} | {r.mean_difference:.1f} | [{r.ci_low:.2f}, {r.ci_high:.2f}] | {r.p_value:.6f} | {r.p_holm_original184:.6f} | {r.wins}/{r.ties}/{r.losses} |")
    lines += ["", "HMEQ QSVC Rank Append는 분포를 보존한 셔플 대비 차이83.5,95% CI[52.68,114.32],Holm184 p=.031358이다. HMEQ QSVC Soft Gate는 차이93.9이나Holm184 p=.083596으로 같은 결론이 아니다. German 두 표현은 작거나 음의 차이가 섞이고, Taiwan도 해당 가족보정 후 유의성을 확보하지 못했다. 무작위8큐비트 대조의Holm100 결과와 셔플 대조의Holm184 결과를 혼합하지 않는다.", "",
        "## 3. 공정 비교: 일치한 조건과 남은 차이", "",
        "| 항목 | 확인 결과 |", "|---|---|",
        "| 자료 분할/레이블 | 모든 모델은 시드별 동일 고객의80/10/10,양성=부도/불량,비용5FN+FP |",
        "| 전처리/학습행 | 학습80%에서만 전처리 적합,선택모델 전체 학습행 사용. 위험교사5OOF는 각 폴드 내부 전처리 |",
        "| 검증/테스트 | 검증 비용으로 선정하고 전체 선정 잠금 뒤 테스트 평가. 임계값 선정 후 train+validation 재적합 없음 |",
        "| XGB 단독 튜닝 | 200trees,depth3/6 두 후보,learning_rate=.05,hist,훈련 불균형 비율 가중치. 검증 비용으로depth 선정 |",
        "| SVM 튜닝 | 양수20lambda×3C×3기하=180후보,별도lambda0의9후보. 큰 자료는512랜드마크 근사 탐색 후 전체행 정확SVC |",
        "| 입력 차이 | 단독XGB는 전처리 원특성; SVM은PCA7와 원특성XGB에서 추정한 위험정보. 동일 입력 비교가 아님 |",
        "| 임계값 동률 | XGB확률은1/6,SVM결정함수는0에 가장 가까운 최소비용 절단값. 같은 비용목적이지만 동률기준/점수척도는 다름 |",
        "| 최종 판정 | 분할·목적함수·테스트 사용 통제는 맞지만,동일 탐색예산/입력/최적화라고 주장하면 안 됨 |", "",
        "XGB의 제한된2후보 탐색이 SVM보다 넓은 예산을 받았다고 볼 수는 없다. 그렇다고 후보 수만으로 모델 간 공정성을 인증할 수도 없다. 본 비교는 동일 의사결정 프로토콜 아래 사전에 고정한 모델별 제한 격자의 결과이며,최적의모든XGB/모든양자모델에 대한 서열은 아니다. 같은 검증세트를 파라미터/임계값 선택에 함께 사용한 선택편향과 큰 자료 근사 탐색의 한계도 남는다.", "",
        "## 4. Dataset별 Kernel Geometry", "",
        "동일C/기하에서 위험 없는 backbone과 실제 위험형의 중심화 레이블 Alignment,entropy effective rank를 같은 테스트 고객 부분집합에서 비교했다. German은100명 전체,큰 자료는최대256명이다. German matched backbone은 이미 계산된 정확lambda0 검색후보의 저장점수/검증임계값을 추출했다. HMEQ/Taiwan은이전 후속실험의전체행 matched refit을 사용했다. 새 모델은학습하지 않았다. 비용감소는 독립 튜닝baseline 대비가 아니라 이 matched backbone 대비다.", "",
        "| 자료 | 커널 | 방법 | Alignment 변화-전체비용 감소 rho | Effective Rank 변화-전체비용 감소 rho |", "|---|---|---|---:|---:|"]
    for dataset in config["datasets"]:
        for kernel in config["kernels"]:
            for method in config["methods"]:
                a = only(correlations, dataset=dataset, kernel=kernel, method=method, x="alignment_delta", y="full_cost_reduction").spearman_rho
                e = only(correlations, dataset=dataset, kernel=kernel, method=method, x="effective_rank_delta", y="full_cost_reduction").spearman_rho
                lines.append(f"| {LABELS[dataset]} | {kernel} | {METHODS[method]} | {a:.3f} | {e:.3f} |")
    lines += ["", "![자료별 정렬도 변화와 비용 감소](kernel_alignment_cost_all_datasets.png)", "",
        "각rho는겹치는10분할의기술적연관이며독립표본p나인과메커니즘을보고하지않는다. 모든64개계수와같은부분집합비용분석도CSV에보존했다. 한자료의큰상관이나유효순위증가를보편적성능법칙으로해석하지않는다. 조건별차이를발견의후보로제시하되원인규명으로쓰지않는다.", "",
        "## 5. Runtime", "",
        "새 속도 벤치마크가 아니라 기존 같은 서버의 측정 기록을 분해했다. 아래cold값은 실제 선택기록의증분상태시간을저장된최초상태생성시간으로교체한구성요소합이다. 전체파이프라인의새cold실행측정이나가속비가아니다. CPU Aer정확배정밀도상태벡터와장치별잠금GPU Gram을사용했고실제양자장치의샷/통신/대기시간은포함되지않는다.", "",
        "| 자료 | 모델 | SVC/XGB 적합(s) | 최초 상태생성 학습/테스트(s) | cold 학습 구성합(s) | cold 테스트 구성합(s) |", "|---|---|---:|---:|---:|---:|"]
    for dataset in config["datasets"]:
        group = runtime[runtime.dataset.eq(dataset)]
        ref = group[group.kernel.eq("XGB")]
        lines.append(f"| {LABELS[dataset]} | XGB 단독 | {ref.fit_s.mean():.4f} | 해당없음 | {ref.fit_s.mean():.4f} | 별도측정없음 |")
        for kernel in config["kernels"]:
            for method in config["methods"]:
                rows = group[group.kernel.eq(kernel) & group.method.eq(method)]
                lines.append(f"| {LABELS[dataset]} | {kernel} {METHODS[method]} | {rows.fit_s.mean():.4f} | {rows.state_train_cold_s.mean():.4f}/{rows.state_test_cold_s.mean():.4f} | {rows.cold_training_component_sum_s.mean():.4f} | {rows.cold_test_component_sum_s.mean():.4f} |")
    lines += ["", "위표는XGB위험교사준비,전처리,전체격자탐색을제외한다. XGB단독은검증+테스트추론합만저장되어SVM테스트전용시간과직접비교하지않는다. 위험교사5OOF+전체학습준비시간과선택비용을반드시따로더고려해야한다.", "",
        "| 자료 | 전처리/PCA 적합 평균(s) | XGB5OOF+전체 위험 준비 평균(s) | XGB 단독 검증+테스트 추론 평균(s) |", "|---|---:|---:|---:|"]
    for dataset in config["datasets"]:
        prep = preparation[preparation.dataset.eq(dataset)]
        ref = runtime[runtime.dataset.eq(dataset) & runtime.kernel.eq("XGB")]
        lines.append(f"| {LABELS[dataset]} | {prep.feature_fit_s.mean():.4f} | {prep.xgb_oof_full_prepare_s.mean():.4f} | {ref.validation_plus_test_inference_s.mean():.4f} |")
    lines += ["", "모든시드별Gram/상태/적합/추론과캐시기록은runtime_seed_components.csv에있다. 준비시간은모든방법에공통인PCA-LR/raw-LR실험준비를포함한전체캠페인시간과구분하며,XGB위험준비시간에는OOF용반복전처리와예측도포함된다. 동시실행·GPU잠금·캐시·환경차이가있어모델별end-to-end배속우위를주장하지않는다.", "",
        "## 6. 주 가설과 탐색 분석의 경계", "",
        "이번논문의핵심질문은(1)분포보존셔플대비실제위험정보가기여하는가,(2)그기여가자료와커널에따라어떻게달라지는가로정리한다. 그러나이미결과를본현재분석을확인적사전등록으로바꿀수는없다. 기존Holm184,후속비용Holm100,보정품질Holm40은그대로둔다.95%CI는각대비의개별근사구간이고Holm은여러대비의가족오류율보정이므로CI가0을제외해도보정p가.05보다클수있다.", "",
        "향후정말새로운독립확인을한다면사전에두과학가설을고정할수있다. 예를들어정보효과는3원자료(South German,HMEQ,Taiwan)×2방법×2커널=12대비,커널개선량차이는3자료×2방법=6대비이며총18대비전체에Holm18을적용하도록설계할수있다. 가설이2개라는것이자동으로p값도2개라는뜻은아니다. German Numeric은관련표현민감도로별도표시한다. 이는차기독립자료프로토콜제안이지현재p값재계산이나신규실험실행승인이아니다. 같은기존자료에시드만추가해독립외부확인이라고부르지않는다. Geometry/보정/런타임은탐색·기술분석으로분리한다.", "",
        "## 7. 방법의 직관과 Contribution", "",
        "q(x)는훈련OOF위험점수의훈련기준midrank다. Soft Gate는단순특성스케일링이아니라K_base(x,x')×[(1-lambda)+lambda*g(q(x))^Tg(q(x'))]이며g(q)=(1-q,q)/||(1-q,q)||다. 비슷한위험수준의고객쌍은기존유사도를유지하고다른위험수준의쌍은약화한다. lambda0에서기본커널로돌아간다. 이위험친화도는PSD이고그람의원소별곱도PSD다. 데이터에서학습한위험축으로유사도를조건화한다는의미이며커널곱자체가새롭다는뜻은아니다.", "",
        "Rank Append도QSVC에서는독립보조Ry큐비트로K_base×cos²(pi*lambda*(q-q')/2),RBF에서는K_base×exp(-gamma*pi²*lambda²*(q-q')²)로분해된다. 추가큐비트는backbone과얽히지않으며위험요인은고전적으로계산할수있다. 두방법을공통의위험조건화제품커널관점에서설명할수있지만일반텐서곱커널의새로운발명이나양자고유위험통합이라고부르지않는다.", "",
        "기여후보1: 고정된두위험통합방식에서임계값·차원증가·분포보존셔플을분리한통제근거. 특히HMEQ QSVC Rank Append의정보효과가무작위대조뿐아니라분포보존셔플에서도남는다. 모든자료/방법으로일반화하지않는다.",
        "기여후보2: 동일C/기하의제품커널재구성과자료별정렬도/유효순위/비용변화의상호일치·불일치를정리한경험적조건지도. 단순정렬도증가=비용감소라는보편법칙은지지하지않는다.",
        "기여후보3: 위험정보로취약한커널을개선하는것과강한원특성XGB를넘는것의차이를보여주는실용적한계. 양자우위나XGB대체를주장하지않는다. 이기여들의선행연구대비신규성은별도문헌대조가필요하며현재결과만으로인과발견/CMC게재가능성을확정하지않는다.", "",
        "PLS Metric/LR Rank Mixture는기존결과를삭제하지않고Appendix/Ablation으로옮기는구성안을권한다. 본문은두방법·분할/OOF/선정프로토콜·통제실험·자료별Geometry·Runtime·한계순서로재정리한다. 새로운방법추가는하지않는다. 다음원고수정에서는기존manuscript파일과도표위치를확인한뒤반영하며,이번보고서는그원고자체를수정한것은아니다.", "",
        "## 8. 한계 및 검증 범위", "",
        "겹치는반복분할,n10,관련German표현,이미본테스트,HMEQ사후주목,큰자료근사탐색,시드당셔플1회,모델별탐색/입력차이,기록기반시간합산이남는다. 통계오류11유형을검토했고자료수집/collider선택은자료생성과정미확보로미확인이다. 생존한유리한대조만선택하지않고16셔플대비/64상관계수를전부보존했다. 해시/저장점수/커널수식의재검증은독립재학습이나독립외부확인이아니다.", ""]
    (output / "report_ko.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "code/results/cmc_professor_revision_20261023"))
    args = parser.parse_args()
    out = Path(args.output)
    try:
        with threadpool_limits(limits=1):
            print(json.dumps(analyze(out)))
    except BaseException:
        if out.is_dir() and not (out / "failure.json").exists():
            write(out / "failure.json", {"utc": now(), "scope": "read_only_analysis_no_scientific_fitting", "traceback": traceback.format_exc()})
        raise
