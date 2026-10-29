"""Audit every revised-model job and build the Korean paired-results report."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _shared import controlled as base
from _shared.improved import CORE, LEARNERS, POLICIES, THRESHOLD, sha
from _shared.full_data import all_indices, write_json
from build_controlled_results import LABELS, corrected_difference, holm

ROOT = base.ROOT
NAMES = {"baseline": "기본형", "convex": "볼록형", "feature_specific": "특성별"}


def audit_split(split, y, seed):
    train, test, fit, valid = all_indices(y, seed)
    for name, expected in zip(("train", "test", "inner_train", "inner_valid"), (train, test, fit, valid)):
        if split[name] != expected.tolist():
            raise AssertionError("Full paired partition differs from the planned split")
    for stage, rows in (("inner", fit), ("final", train)):
        fit_counts, cal_counts = Counter(), Counter()
        for fold in split[f"{stage}_folds"]:
            a, b = fold["fit"], fold["calibration"]
            if len(a) != len(set(a)) or len(b) != len(set(b)) or set(a) & set(b) or set(a) | set(b) != set(rows):
                raise AssertionError("Fit/calibration leak or incomplete partition")
            fit_counts.update(a)
            cal_counts.update(b)
        if any(fit_counts[i] != 2 or cal_counts[i] != 1 for i in rows):
            raise AssertionError("Full training rows not used exactly twice for fitting and once for calibration")


def audit_job(folder, group, split, source_y, metadata):
    receipt = json.loads((folder / "complete.json").read_text())
    if receipt["identity"]["options"] != metadata["options"]:
        raise AssertionError("Job source/settings differ from the campaign")
    for name, expected in receipt["outputs_sha256"].items():
        if sha(folder / name) != expected:
            raise AssertionError("Job output checksum differs")
    predicted = pd.read_csv(folder / "predictions.csv.gz", float_precision="round_trip")
    keys = ["representation", "policy"]
    if predicted.duplicated(keys + ["row_id"]).any() or group.duplicated(keys).any():
        raise AssertionError("Duplicate model predictions or metrics")
    expected = {(r.representation, r.policy) for r in group.itertuples()}
    if set(predicted[keys].itertuples(index=False, name=None)) != expected:
        raise AssertionError("Prediction model matrix is incomplete")
    for row in group.itertuples():
        saved = predicted[predicted.representation.eq(row.representation) & predicted.policy.eq(row.policy)]
        if len(saved) != len(split["test"]) or set(saved.row_id) != set(split["test"]):
            raise AssertionError("Unequal held-out rows across models")
        saved = saved.set_index("row_id").loc[split["test"]]
        if not np.array_equal(saved.y, source_y[split["test"]]) or not np.array_equal(saved.prediction, (saved.score >= row.threshold).astype(int)):
            raise AssertionError("Saved source targets or decisions do not reproduce")
        if not saved.score.between(0, 1).all():
            raise AssertionError("Reported score is not a probability")
        for metric, value in base.metrics(saved.y.to_numpy(), saved.score.to_numpy(), row.threshold).items():
            if not np.isclose(value, getattr(row, metric), rtol=1e-12, atol=1e-12):
                raise AssertionError(f"Saved {metric} differs from predictions")
        if row.policy != "tuned_probability" and row.threshold != THRESHOLD:
            raise AssertionError("Primary or matched threshold changed")
    return len(predicted)


def contrasts(raw):
    main, projected, controls = [], [], []
    for (dataset, policy), group in raw.groupby(["dataset", "policy"]):
        costs = group.pivot(index="seed", columns=["learner", "representation"], values="cost")
        ratio = float((group.n_test / group.n_train).iloc[0])
        gains = {}
        for learner in CORE:
            baseline = costs[learner, "baseline"]
            for representation in ("convex", "feature_specific"):
                delta = baseline - costs[learner, representation]
                gains[learner, representation] = delta
                row = {"dataset": dataset, "policy": policy, "learner": learner, "representation": representation,
                       "contrast": "within_kernel", "mean_rcr_pct": float((100 * delta / baseline.replace(0, np.nan)).mean()),
                       **corrected_difference(delta.to_numpy(), ratio)}
                (projected if learner == "PQK" else main).append(row)
                for control in ("shuffled", "constant"):
                    delta_control = costs[learner, f"{representation}_{control}"] - costs[learner, representation]
                    controls.append({"dataset": dataset, "policy": policy, "learner": learner, "representation": representation,
                                     "control": control, **corrected_difference(delta_control.to_numpy(), ratio)})
        for representation in ("convex", "feature_specific"):
            difference = gains["QSVC", representation] - gains["RBF-SVM", representation]
            main.append({"dataset": dataset, "policy": policy, "learner": "interaction", "representation": representation,
                         "contrast": "quantum_minus_classical", **corrected_difference(difference.to_numpy(), ratio)})
    frames = [pd.DataFrame(rows) for rows in (main, projected, controls)]
    for frame in frames:
        bad = frame.sd_difference.eq(0) & frame.mean_difference.ne(0)
        frame.loc[bad, ["p_value", "ci_low", "ci_high"]] = np.nan
        frame["p_holm"] = holm(frame.p_value.fillna(1).to_numpy())
        frame.loc[bad, "p_holm"] = np.nan
        frame["holm_family_size"] = len(frame)
    return frames


def f(value, places=3):
    return "해당 없음" if pd.isna(value) else f"{value:.{places}f}"


def build(run_id):
    output = ROOT / "code/results" / run_id
    metadata = json.loads((output / "metadata.json").read_text())
    protocol = metadata["options"]["protocol_settings"]
    if metadata["datasets"] != list(base.DATASETS) or metadata["seeds"] != protocol["seeds"] or metadata["learners"] != list(LEARNERS):
        raise AssertionError("Full planned campaign is required")
    sources = {"improved": ROOT / "code/_shared/improved.py", "runner": ROOT / "code/run_improved_experiment.py",
               "controlled": ROOT / "code/_shared/controlled.py", "full_data": ROOT / "code/_shared/full_data.py",
               "analysis": Path(__file__), "shared_analysis": ROOT / "code/build_controlled_results.py",
               "protocol": Path(metadata["options"]["protocol_path"])}
    if any(sha(path) != metadata["options"]["source_sha256"][name] for name, path in sources.items()):
        raise AssertionError("Campaign sources changed")
    rows, audits, accounting = [], [], []
    for dataset in metadata["datasets"]:
        _, y, source = base.load_data(dataset, str(ROOT / "data"))
        for seed in metadata["seeds"]:
            root = ROOT / "code" / dataset / "results" / run_id / f"seed_{seed:03d}"
            split = json.loads((root / "split_indices.json").read_text())
            audit_split(split, y, seed)
            for learner in LEARNERS:
                folder = root / learner
                metrics = pd.read_csv(folder / "metrics.csv", float_precision="round_trip")
                representations = base.REPRESENTATIONS if learner in CORE else ("risk_score" if learner == "risk_only" else "encoded_full",)
                expected = {(r, p) for r in representations for p in POLICIES}
                if set(metrics[["representation", "policy"]].itertuples(index=False, name=None)) != expected or len(metrics) != len(expected):
                    raise AssertionError("Incomplete per-job model matrix")
                if not (metrics.dataset.eq(dataset) & metrics.seed.eq(seed) & metrics.learner.eq(learner)
                        & metrics.n_train.eq(len(split["train"])) & metrics.n_test.eq(len(split["test"]))).all():
                    raise AssertionError("Metric identity or denominator differs")
                receipt = json.loads((folder / "complete.json").read_text())
                if receipt["identity"]["source_sha256"] != source["sha256"]:
                    raise AssertionError("Data changed after training")
                count = audit_job(folder, metrics, split, y, metadata)
                audits.append({"dataset": dataset, "seed": seed, "learner": learner, "prediction_rows": count,
                               "split_sha256": sha(root / "split_indices.json"), "predictions_sha256": sha(folder / "predictions.csv.gz")})
                rows.append(metrics)
                candidate_files = list((folder / "tuning").glob("*/complete.json"))
                accounting.append({"dataset": dataset, "seed": seed, "learner": learner,
                                   "job_elapsed_seconds": receipt["elapsed_seconds"], "unique_kernel_candidate_groups": len(candidate_files),
                                   "unique_candidate_group_seconds": sum(json.loads(p.read_text())["seconds"] for p in candidate_files),
                                   "unique_final_configurations": len(list((folder / "final").glob("*/complete.json")))})
    raw = pd.concat(rows, ignore_index=True)
    if len(raw) != protocol["expected_metric_rows"] or len(audits) != protocol["expected_model_jobs"]:
        raise AssertionError("Full campaign row count differs from the protocol")
    raw.to_csv(output / "raw_metrics.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "prediction_audit.csv", index=False)
    pd.DataFrame(accounting).to_csv(output / "compute_accounting.csv", index=False)
    main, projected, controls = contrasts(raw)
    for name, frame in (("main_contrasts", main), ("projected_contrasts", projected), ("ablation_contrasts", controls)):
        frame.to_csv(output / f"{name}.csv", index=False)
    primary = raw[raw.policy.eq("cost_probability") & raw.learner.isin(["RBF-SVM", "QSVC"]) & raw.representation.isin(NAMES)].copy()
    baseline = primary[primary.representation.eq("baseline")][["dataset", "seed", "learner", "cost"]].rename(columns={"cost": "baseline_cost"})
    primary = primary.merge(baseline, on=["dataset", "seed", "learner"], validate="many_to_one")
    primary["paired_cost_difference"] = primary.baseline_cost - primary.cost
    primary["relative_cost_reduction_pct"] = 100 * primary.paired_cost_difference / primary.baseline_cost.replace(0, np.nan)
    primary["direction"] = np.select([primary.paired_cost_difference.gt(0), primary.paired_cost_difference.lt(0)], ["개선", "악화"], default="동률")
    primary.to_csv(output / "primary_seed_metrics.csv", index=False)
    primary[primary.representation.ne("baseline")].to_csv(output / "primary_seed_directions.csv", index=False)
    metrics = ["f1", "recall", "balanced_accuracy", "fn", "fp", "cost", "brier_test", "relative_cost_reduction_pct"]
    summary = primary.groupby(["dataset", "learner", "representation"])[metrics].agg(["mean", "std"])
    summary.to_csv(output / "primary_summary_metrics.csv")
    time_fields = [c for c in raw if c.endswith("_seconds")]
    raw.groupby(["dataset", "learner", "representation", "policy"])[time_fields].agg(["mean", "std"]).to_csv(output / "runtime_summary.csv")
    raw[raw.policy.eq("cost_probability") & raw.learner.isin(["QSVC", "PQK"])].groupby(
        ["dataset", "learner", "representation", "C", "lambda", "alpha", "entanglement", "gamma_factor"]).size().rename("seeds").to_csv(output / "feature_map_selection_counts.csv")
    raw[["dataset", "seed", "learner", "representation", "policy", "train_kernel_entries", "calibration_kernel_entries", "test_kernel_entries", "peak_train_kernel_gib"]].to_csv(output / "kernel_workload.csv", index=False)
    main[main.policy.eq("cost_probability") & main.contrast.eq("within_kernel")].to_csv(output / "primary_paired_cost.csv", index=False)
    write_json(output / "experiment_protocol.json", protocol)
    acquired = json.loads((ROOT / "data/acquisition_manifest.json").read_text())
    write_json(output / "data_manifest.json", {d: acquired[d] for d in metadata["datasets"]})
    lines = ["# 개선된 전체 데이터 실험 결과", "", f"실행명: `{run_id}`. 네 자료의 공통 10개 시드, 280개 모델 작업과 3,000개 평가 조건을 모두 검증했다. 주 비교 6개 모델의 주 정책은 240개 조건이다.",
             "", "## 변경과 해석 범위", "", "기존 결과와 큰 개선 사례를 본 뒤 비교 설계를 수정한 탐색적 후속 연구다. 같은 원자료·외부 분할을 재사용하므로 독립적인 확인 연구나 사전등록 연구로 표현하지 않는다. German 두 자료는 같은 원자료 계열이다.",
             "", "모든 원본 행을 외부 학습 80%·테스트 20%로 층화 분할했다. 외부 학습의 모든 행은 3개 앙상블 구성원 중 두 개의 모델 적합, 하나의 확률 보정에 사용된다. 구성원 하나는 학습의 2/3에서 적합한다. 전체 외부 학습 행에 단일 SVC를 적합한 이전 실행과는 모델 구조가 다르며, 행을 일부 추출하거나 커널을 근사하지 않았다.",
             "", "각 구성원은 전처리·PCA·위험점수를 자신의 적합 행에서만 학습한다. SVC를 별도 행에서 sigmoid 보정한 뒤 모델과 보정기를 고정한다. 세 확률을 평균하고 주 판정 기준은 1/6로 고정한다. 하이퍼파라미터는 내부 검증의 5FN+FP로 선택한다. 원시 margin 임계값을 다른 재학습 모델에 전달하지 않는다.",
             "", "공통 C·커널 설정 후보를 사용하되 위험형은 추가 λ 후보가 있어 탐색 후보 수가 5배다. 후보 수가 같다고 주장하지 않는다. λ=0의 계산은 기본형과 공유한다. 공통 기본형 설정을 고정한 matched_backbone 분석을 별도 보고해 위험 변환과 설정 선택의 효과를 구분한다.",
             "", "## 주 6개 모델", "", "시드 평균 ± 표본 표준편차다. 상대 감소율은 각 시드의 같은 커널 기본형 대비 감소율을 먼저 구한 뒤 평균했다.",
             "", "| 자료 | 모델 | F1 | 재현율 | 균형 정확도 | FN | FP | 비용 | 상대 감소율(%) |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for d in metadata["datasets"]:
        for learner, rep in [(k, r) for k in ("RBF-SVM", "QSVC") for r in NAMES]:
            row = summary.loc[(d, learner, rep)]
            cells = [f"{f(row[(m, 'mean')])} ± {f(row[(m, 'std')])}" for m in ("f1", "recall", "balanced_accuracy", "fn", "fp", "cost", "relative_cost_reduction_pct")]
            lines.append(f"| {LABELS[d]} | {learner} {NAMES[rep]} | " + " | ".join(cells) + " |")
    for policy, title in (("cost_probability", "주 정책의 짝비교"), ("matched_backbone", "기본형 설정 고정 기전 비교")):
        lines.extend(["", f"## {title}", "", "Δ=기본형 비용−위험형 비용. 양수는 개선이다. 반복 분할 중복을 고려한 근사 보정 재표본 양측 t 검정과 개별 95% 신뢰구간을 사용한다. Holm 보정군은 세 정책과 두 상호작용을 포함한 기존 범위 72개이며 구간은 동시신뢰구간이 아니다.",
                      "", "| 자료 | 커널 | 변환 | 평균 Δ | 보정 95% 구간 | 원 p | Holm p | 개선/동률/악화 |", "| --- | --- | --- | ---: | --- | ---: | ---: | --- |"])
        for r in main[main.policy.eq(policy) & main.contrast.eq("within_kernel")].itertuples():
            lines.append(f"| {LABELS[r.dataset]} | {r.learner} | {NAMES[r.representation]} | {f(r.mean_difference)} | [{f(r.ci_low)}, {f(r.ci_high)}] | {f(r.p_value)} | {f(r.p_holm)} | {r.wins}/{r.ties}/{r.losses} |")
    lines.extend(["", "## 시드별 방향", "", "모든 시드의 두 비용·Δ·상대 감소율·방향은 `primary_seed_directions.csv`에 있다. 누락·실패·악화 시드를 제외하지 않는다. 시드 10개는 독립된 10개 원자료가 아니며 신뢰구간과 검정은 근사다. 유의하지 않음을 효과가 정확히 0이라는 증거로 해석하지 않는다.",
                  "", "## 보충 비교", "", "투영 양자 커널(PQK)은 같은 ZZ 회로의 21개 로컬 X/Y/Z 기대값에 Gaussian 커널을 적용한다. 원래 6모델의 QSVC를 대체하지 않는다. 아래 값은 주 정책의 평균 사업 비용이며 고전 참조는 PCA 이전 인코딩 특성을 사용하므로 입력 조건이 다르다.",
                  "", "| 자료 | PQK 기본 | PQK 볼록형 | PQK 특성별 | LR | RF | XGBoost | 위험점수만 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    p = raw[raw.policy.eq("cost_probability")]
    for d in metadata["datasets"]:
        values = [p[p.dataset.eq(d) & p.learner.eq("PQK") & p.representation.eq(r)].cost.mean() for r in NAMES]
        values += [p[p.dataset.eq(d) & p.learner.eq(k)].cost.mean() for k in ("LR", "RF", "XGBoost", "risk_only")]
        lines.append(f"| {LABELS[d]} | " + " | ".join(f(v) for v in values) + " |")
    lines.extend(["", "PQK 비교 24개와 세 커널의 섞기/상수 대조군 비교 144개는 각각 별도 보정군이다. 결과는 `projected_contrasts.csv`, `ablation_contrasts.csv`에 있다.",
                  "", "## 학습·추론 시간", "", "아래는 병렬 부하 환경의 구성요소 측정 시간이며 전용 CPU의 격리 성능 시험이 아니다. 구성원 3개의 단계 시간을 합산한 단독 앙상블 비용으로 공유 전처리·커널 비용을 귀속한다. 다른 정책이나 같은 λ=0 행의 시간을 합산해 실제 벽시계 시간이라고 해석하지 않는다. 탐색 실제 고유 계산은 `compute_accounting.csv`, 모든 세부 단계는 `raw_metrics.csv`와 `runtime_summary.csv`에 있다.",
                  "", "| 자료 | 모델 | 학습 평균(초) | 추론 평균(초) |", "| --- | --- | ---: | ---: |"])
    for d in metadata["datasets"]:
        for learner, rep in [(k, r) for k in ("RBF-SVM", "QSVC") for r in NAMES]:
            group = p[p.dataset.eq(d) & p.learner.eq(learner) & p.representation.eq(rep)]
            lines.append(f"| {LABELS[d]} | {learner} {NAMES[rep]} | {f(group.training_seconds.mean())} | {f(group.inference_seconds.mean())} |")
    lines.extend(["", "학습은 적합·보정 행의 전처리, 위험점수 교차 적합, 상태/투영 생성, 학습/보정 커널, SVC 적합과 sigmoid 보정을 포함한다. 추론은 테스트 전처리·상태/투영 생성·테스트 커널·확률 예측을 포함한다. 회로 초기화·탐색은 별도 기록하며 파일 입출력·검증·가비지 수집은 단계 합계에서 제외한다.",
                  "", "## 양자 설정과 계산량", "", "7큐비트·ZZ·반복 1·alpha 0.5/1/2·linear/full을 내부 검증에서 선택했다. Aer CPU 배정밀도 이상적 상태벡터이며 잡음·유한 샷 추정·실제 QPU를 사용하지 않았다. 선택 횟수는 `feature_map_selection_counts.csv`, 3개 구성원의 적합/보정/테스트 커널 원소와 최대 학습 행렬 메모리는 `kernel_workload.csv`에 있다. 논리 원소 수는 QPU 실행 횟수나 측정 샷 수가 아니다.",
                  "", "## 검증과 한계", "", f"원본 분할·폴드 내부 적합/보정 분리·원본 라벨·같은 테스트 행·확률·임계값·혼동행렬·모든 지표를 검증했다. 저장 예측은 {sum(a['prediction_rows'] for a in audits):,}행이다. 전체 탐색을 독립적으로 다시 수행한 재현 검증은 아니다.",
                  "", "자료별 분석, 개인·인과효과로의 확대 금지, 원자료 선택 편향, 충돌변수·역인과 추정 아님, 기저율·FN/FP 동시 보고, 극단 시드 사후 선택 금지, 전 시드 보존, 다중비교 보정, 기존 결과 이후 방법 변경 공개 등 통계적 주의 11개 항목을 검토했다. 하위집단 대표성이나 모든 편향의 배제를 확인한 것은 아니다."])
    (output / "결과보고서.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "verification_receipt.json", {"scope": "all source partitions, nested fit/calibration boundaries, predictions and paired summaries; not independent campaign rerun",
               "seed_jobs": 40, "model_jobs": len(audits), "metric_rows": len(raw), "primary_rows": len(primary),
               "prediction_rows": sum(a["prediction_rows"] for a in audits), "analysis_source_sha256": sha(__file__),
               "outputs_sha256": {name: sha(output / name) for name in ("raw_metrics.csv", "primary_paired_cost.csv", "primary_seed_directions.csv", "prediction_audit.csv", "결과보고서.md")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_20261016_improved_full")
    build(parser.parse_args().run_id)
