"""Locked PCA9 evaluation and predeclared paired dimension sensitivity."""
import itertools
import json

import numpy as np
import pandas as pd

from . import pca9_followup as p
from . import overnight_reporting as report
from build_controlled_results import holm


def selected_choices(groups):
    choices = []
    for job, record in groups.items():
        common = {"dataset": job[0], "seed": job[1], "kernel": job[2], "family": job[3], "arm": job[4], "job": list(job)}
        if job[3] == "reference":
            for learner, representation in itertools.product(("LR", "RF", "XGBoost"), ("raw", "pca9")):
                pool = [r for r in record["trials"] if r["learner"] == learner and r["method"] == representation]
                row = min(pool, key=lambda r: (r["tune_cost"], str(r.get("depth"))))
                choices.append({**row, **common, "role": f"reference_{representation}_{learner}",
                    "score_source": "margins", "score_key": row["score_index"]})
        else:
            for role, row in record["choices"].items():
                if job[4] != "normal" and role == "baseline":
                    continue
                choices.append({**row, **common, "role": role, "score_source": "exact_margins", "score_key": role})
    return report.append_classical_best(choices)


def dimension_sensitivity(root, protocol, raw):
    previous = p.ROOT / "code/results" / protocol["upstream_run"] / "full"
    before = pd.read_csv(previous / "seed_metrics.csv", float_precision="round_trip")
    indices = {7: report.LossIndex(previous / "predictions.csv.gz"), 9: report.LossIndex(root / "full/predictions.csv.gz")}
    tables = {7: before, 9: raw}
    metrics, rows = [], []

    def matrix(dimension, dataset, kernel, family, role):
        table = tables[dimension]
        models = table[table.dataset.eq(dataset) & table.kernel.eq(kernel) & table.family.eq(family)
            & table.arm.eq("normal") & table.role.eq(role)].set_index("seed")
        assert sorted(models.index) == protocol["seeds"]
        losses, labels = indices[dimension].matrix(models)
        np.testing.assert_array_equal(indices[7].ids.get(dataset, indices[dimension].ids[dataset]), indices[dimension].ids[dataset])
        return losses, labels, models

    def add(dataset, learner, family, contrast, values, labels):
        rows.append({"dataset": dataset, "kernel": learner, "family": family, "contrast": contrast,
            "primary_source": dataset != "german_numeric", "complete_ten": True,
            **report.paired_customer_summary(values, labels, ("pca9_dimension", dataset, learner, family, contrast))})

    gains = {}
    for dataset, learner in itertools.product(protocol["datasets"], protocol["learners"]):
        b7, labels, baseline7 = matrix(7, dataset, learner, "baseline", "baseline")
        b9, other, baseline9 = matrix(9, dataset, learner, "baseline", "baseline")
        np.testing.assert_array_equal(labels, other)
        add(dataset, learner, "baseline", "absolute_cost_pca7_minus_pca9", b7 - b9, labels)
        for family in protocol["methods"]:
            a7, _, models7 = matrix(7, dataset, learner, family, "proposed")
            a9, _, models9 = matrix(9, dataset, learner, family, "proposed")
            own7, _, _ = matrix(7, dataset, learner, family, "baseline")
            own9, _, _ = matrix(9, dataset, learner, family, "baseline")
            add(dataset, learner, family, "absolute_cost_pca7_minus_pca9", a7 - a9, labels)
            add(dataset, learner, family, "tuned_baseline_gain_pca9_minus_pca7", (b9 - a9) - (b7 - a7), labels)
            add(dataset, learner, family, "own_zero_gain_pca9_minus_pca7", (own9 - a9) - (own7 - a7), labels)
            gains[dataset, learner, family] = (own9 - a9) - (own7 - a7), labels
            for seed in protocol["seeds"]:
                old_cost, new_cost = float(models7.loc[seed, "cost"]), float(models9.loc[seed, "cost"])
                b_old, b_new = float(baseline7.loc[seed, "cost"]), float(baseline9.loc[seed, "cost"])
                metrics.append({"dataset": dataset, "kernel": learner, "family": family, "seed": seed,
                    "pca7_cost": old_cost, "pca9_cost": new_cost, "pca7_baseline_cost": b_old, "pca9_baseline_cost": b_new,
                    "absolute_pca9_minus_pca7": new_cost - old_cost,
                    "pca7_gain": b_old - old_cost, "pca9_gain": b_new - new_cost,
                    "pca7_rcr_pct": 100 * (b_old - old_cost) / b_old if b_old else np.nan,
                    "pca9_rcr_pct": 100 * (b_new - new_cost) / b_new if b_new else np.nan,
                    "inference_limit": "Reused fixed test customers; exploratory sensitivity, not an independent replication"})
    for dataset, family in itertools.product(protocol["datasets"], protocol["methods"]):
        q, labels = gains[dataset, "QSVC", family]
        r, other = gains[dataset, "RBF-SVM", family]
        np.testing.assert_array_equal(labels, other)
        add(dataset, "interaction", family, "quantum_minus_rbf_gain_change_pca9_minus_pca7", q - r, labels)
    table = pd.DataFrame(rows)
    assert len(table) == 90
    table["p_holm"] = holm(table.p_value.to_numpy())
    table["holm_family_size"] = len(table)
    table.to_csv(root / "full/dimension_paired_cost.csv", index=False)
    seed_metrics = pd.DataFrame(metrics)
    seed_metrics["rcr_change_pp"] = seed_metrics.pca9_rcr_pct - seed_metrics.pca7_rcr_pct
    seed_metrics.to_csv(root / "full/dimension_seed_metrics.csv", index=False)
    seed_metrics.groupby(["dataset", "kernel", "family"]).mean(numeric_only=True).to_csv(root / "full/dimension_summary.csv")


def finalize(root, meta):
    output, protocol = root / "full", meta["protocol"]
    tasks = p.jobs(protocol)
    groups, receipts, missing = report.collect(output, tasks, meta["source_sha256"])
    assert not missing and len(groups) == 870, "Never release partial PCA9 tests"
    choices = report.sensitivity_choices(output, selected_choices(groups))
    profiles = report.sensitivity_choices(output, report.lambda_profiles(groups, protocol))
    content = {"choices": choices, "lambda_profiles": profiles, "candidate_receipts_sha256": receipts,
        "missing_jobs": [], "source_sha256": meta["source_sha256"], "dimension": 9,
        "rule": "All planned choices and exact validation thresholds fixed before test labels; exploratory reused cohort"}
    assert not (output / "selection_lock.json").exists() and not (output / "test_release.json").exists()
    p.write_json(output / "selection_lock.json", content)
    p.write_json(output / "test_release.json", {"selection_lock_sha256": p.sha(output / "selection_lock.json"),
        "released_utc": p.now(), "no_further_model_fitting_or_selection": True, "reused_upstream_test_customers": True})
    raw, predictions = report.evaluate(output, choices, True, protocol)
    raw["pca_components"] = 9
    raw["execution_origin"] = "new_pca9_single_job_process_with_verified_teacher_reuse"
    raw.to_csv(output / "seed_metrics.csv", index=False)
    columns = ["cost", "fn", "fp", "f1", "recall", "balanced_accuracy"]
    raw.groupby(["dataset", "kernel", "family", "arm", "role"])[columns].agg(["mean", "std"]).to_csv(output / "summary_metrics.csv")
    timing = [name for name in ("training_seconds", "test_inference_seconds", "train_kernel_seconds", "test_kernel_seconds", "fit_seconds", "exact_fit_seconds") if name in raw]
    raw.groupby(["dataset", "kernel", "family", "arm", "role"])[timing].agg(["mean", "std"]).to_csv(output / "runtime_summary.csv")
    p.write_json(output / "quantum_settings.json", {"simulator": "CPU Aer double exact noiseless9/10-qubit statevector; GPU Gram only",
        "state_cache_timing": "Within-job cache reuse is explicitly measured; cached trial times are not cold-start state simulation times",
        "models": [{"dataset": c["dataset"], "seed": c["seed"], "family": c["family"], "kernel": c["kernel"],
            "arm": c["arm"], "role": c["role"], "circuits": c.get("representation_details", {}).get("circuits", [])} for c in choices]})
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    profile_metrics, _ = report.evaluate(output, profiles, True, protocol, with_predictions=False)
    profile_metrics.to_csv(output / "lambda_seed_metrics.csv", index=False)
    losses = report.LossIndex(output / "predictions.csv.gz")
    report.supplementary_statistics(output, raw, profile_metrics, protocol["methods"], protocol, losses)
    report.full_report(output, raw, choices, groups, protocol["methods"], protocol, [], losses)
    dimension_sensitivity(root, protocol, raw)
    runtimes = [{"job": list(job), "execution_origin": record["execution_origin"],
        "elapsed_seconds": record["elapsed_seconds"], "peak_gib": record["peak_gib"],
        "pid": record["worker_pid"], "state_cache_scope": "One job, across lambda strengths only"} for job, record in groups.items()]
    pd.DataFrame(runtimes).to_csv(output / "runtime_execution_provenance.csv", index=False)
    write_report(root, protocol, raw)
    files = [path for path in output.iterdir() if path.is_file() and path.suffix not in (".lock", ".tmp")
        and path.name != "verification_receipt.json"]
    p.write_json(output / "verification_receipt.json", {"scope": "Frozen artifacts, selection, scores and recomputed metrics; no independent retraining",
        "missing_jobs": [], "planned_jobs": 870, "selected_rows": len(raw),
        "outputs_sha256": {path.name: p.sha(path) for path in files}})


def write_report(root, protocol, raw):
    table = pd.read_csv(root / "full/paired_cost.csv")
    lines = ["# PCA 9차원 후속 비교", "", "## Material Passport", "",
        "상태: ANALYZED. German Numeric·South German·HMEQ, 공통10시드·870개 계획 작업의 저장 결과 분석이다. 독립 재학습 재현이나 양자 우위의 증명이 아니다.", "",
        "PCA7과 동일한 저장 분할과 최종 테스트 고객을 재사용한 탐색적 차원 민감도 분석이다. 방법 네 개와 격자는 PCA7 테스트 공개 전에 고정했다. German 두 표현은 같은 원자료 계열이며 독립 자료 둘로 세지 않는다.", "",
        "전처리/PCA9는 LR의5폴드OOF와 내부3폴드sigmoid 보정의 실제 학습 폴드마다 다시 적합했다. 원특성GBDT/LR 점수는 입력/분할/소스가 정확히 일치하는 원본만 검증 재사용했다. PLS9는 지도 표현 대조군이며 직접적인 위험점수 주입과 다르다.", "",
        "최종SVM은 학습80% 전체의 단일 모델이다. HMEQ의512랜드마크 Nyström 탐색과 잠긴 파라미터의 정확 최종SVC를 구분하고, 정확 모델 자체의 검증 임계값을 새로 선택했다. 모든 선택을 잠근 뒤 테스트를 공개했으며 공개 후 학습은 없다.", "",
        "CPU Aer의9큐비트 정확 배정밀도 상태벡터이며 Rank Append는 독립 보조큐비트로10큐비트다. GPU는 장치 잠금을 사용하는Gram 계산이다. 동일 작업 안의 상태벡터 캐시 시간과 전체 작업 경과시간을 별도로 남겼고 이전 세대 시간과 합산해 속도 우위를 주장하지 않는다.", "",
        "## 자체 기본형 대비", "", "비용=5FN+FP, Δ=대조−제안으로 양수가 개선이다. 고객별10개 단일모델 손실차이를 묶어bootstrap5000회·sign-flip19999회로 계산했다. 예측 앙상블이나 독립 시드 검정이 아니다.", "",
        "| 자료 | 방법 | 커널 | 비용 Δ | 95% 구간 | 원 p | Holm p | 개선/동률/악화 |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | --- |"]
    for row in table[table.contrast.eq("tuned_baseline")].itertuples():
        lines.append(f"| {row.dataset} | {row.family} | {row.kernel} | {row.mean_difference:.2f} | [{row.ci_low:.2f}, {row.ci_high:.2f}] | {row.p_value:.4f} | {row.p_holm:.4f} | {int(row.wins)}/{int(row.ties)}/{int(row.losses)} |")
    lines += ["", "## 7차원 대비", "",
        "dimension_summary.csv는 절대 최종비용, 각 차원의 자체 기본형 대비 개선량, 상대 개선율과 그 차이를 분리한다. dimension_paired_cost.csv의90개 사전고정 대비는 별도Holm군이다. 상대 개선율 변화는 기술통계이며 나쁜 기본형에서 개선폭이 크다는 사실을 더 좋은 최종모델 또는 양자 고유 이점으로 해석하지 않는다.", "",
        "고객bootstrap은 개발자료·적합 모델·층화 테스트 비율에 조건부이며 모든 학습 불확실성을 포함하지 않는다. 재사용·선택편향·기저율·다중탐색·완료선택·인과과장 등11개 해석 위험은 기존 프로토콜과 동일하게 제한한다. 모든 제어군과 실패·체크포인트·소스 해시를 보존한다."]
    (root / "full/결과보고서.md").write_text("\n".join(lines) + "\n")
