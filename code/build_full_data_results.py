"""Audit and report only a complete four-dataset full-data experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from _shared.controlled import DATASETS
from build_controlled_results import LABELS, audit_predictions, figures, paired_contrasts

ROOT = Path(__file__).resolve().parents[1]
SIZES = {"german_numeric": (1000, 800, 200), "south_german": (1000, 800, 200), "taiwan": (30000, 24000, 6000), "give_me_some_credit": (150000, 120000, 30000), "hmeq": (5960, 4768, 1192)}


def build(run_id):
    output = ROOT / "code" / "results" / run_id
    metadata = json.loads((output / "metadata.json").read_text())
    protocol_path = ROOT / metadata.get("protocol", "docs/cmc_full_experiment_protocol.json")
    protocol = json.loads(protocol_path.read_text())
    sources = {"controlled": ROOT / "code" / "_shared" / "controlled.py", "full_data": ROOT / "code" / "_shared" / "full_data.py", "full_analysis": Path(__file__), "shared_analysis": ROOT / "code" / "build_controlled_results.py", "protocol": protocol_path}
    expected_sources = {**metadata["options"]["source_sha256"], **metadata.get("analysis_source_sha256", {})}
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != expected_sources[name] for name, path in sources.items()):
        raise AssertionError("Code or protocol differs from the recorded full-data run")
    raw = pd.read_csv(output / "raw_metrics.csv", float_precision="round_trip")
    seeds = metadata["seeds"]
    amendment = None
    if metadata.get("seed_plan"):
        plan_path = output / metadata["seed_plan"]
        if hashlib.sha256(plan_path.read_bytes()).hexdigest() != metadata["seed_plan_sha256"]:
            raise AssertionError("Recorded seed amendment changed")
        amendment = json.loads(plan_path.read_text())
        from run_full_campaign import validate_seed_plan
        if validate_seed_plan(amendment, run_id) != seeds:
            raise AssertionError("Results differ from the approved ten-seed amendment")
    elif seeds != protocol["seeds"]:
        raise AssertionError("A changed seed scope requires a documented amendment")
    if not metadata["full_dataset"] or set(metadata["datasets"]) != set(DATASETS):
        raise AssertionError("The full four-dataset run is required")
    expected = {(dataset, seed) for dataset in DATASETS for seed in seeds}
    if set(raw[["dataset", "seed"]].itertuples(index=False, name=None)) != expected or (raw.groupby(["dataset", "seed"]).size() != 54).any():
        raise AssertionError("Incomplete full-data result matrix")
    for dataset, seed in sorted(expected):
        folder = ROOT / "code" / dataset / "results" / run_id / f"seed_{seed:03d}"
        receipt = json.loads((folder / "complete.json").read_text())
        if receipt["options"] != metadata["options"]:
            raise AssertionError("Per-seed code/settings differ from run metadata")
        split = json.loads((folder / "split_indices.json").read_text())
        total, train, test = SIZES[dataset]
        if len(split["train"]) != train or len(split["test"]) != test or set(split["train"]) | set(split["test"]) != set(range(total)):
            raise AssertionError("Source rows are missing from full-data split")
        selected = raw[(raw.dataset == dataset) & (raw.seed == seed)]
        if not (selected.n_train.eq(train) & selected.n_test.eq(test)).all():
            raise AssertionError("Metric denominators differ from full-data split")
    audit, risk_rows = audit_predictions(raw, run_id)
    audit.to_csv(output / "prediction_audit.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(output / "risk_diagnostics.csv", index=False)
    numeric = [name for name in ("cost", "fn", "fp", "f1", "recall", "balanced_accuracy", "roc_auc", "lambda", "preprocessing_seconds", "train_kernel_seconds", "test_kernel_seconds", "fit_seconds", "inference_seconds", "tuning_seconds", "final_total_seconds", "angle_clipped_fraction", "kernel_alignment", "effective_rank") if name in raw]
    raw.groupby(["dataset", "learner", "representation", "policy"])[numeric].agg(["mean", "std"]).to_csv(output / "summary_metrics.csv")
    main, ablations = paired_contrasts(raw)
    main.to_csv(output / "main_contrasts.csv", index=False)
    ablations.to_csv(output / "ablation_contrasts.csv", index=False)
    sensitivity = pd.concat([raw.assign(cost_ratio=ratio, sensitivity_cost=ratio * raw.fn + raw.fp) for ratio in (2, 5, 10)])
    sensitivity.groupby(["dataset", "learner", "representation", "policy", "cost_ratio"]).sensitivity_cost.agg(["mean", "std"]).to_csv(output / "cost_ratio_sensitivity.csv")
    figures(raw, main, ablations, output, datasets=DATASETS)
    acquired = json.loads((ROOT / "data/acquisition_manifest.json").read_text())
    (output / "data_manifest.json").write_text(json.dumps({name: acquired[name] for name in DATASETS}, indent=2), encoding="utf-8")
    shutil.copyfile(protocol_path, output / "experiment_protocol.json")
    if amendment:
        shutil.copyfile(output / metadata["seed_plan"], output / "seed_amendment.json")
    primary = raw[raw.policy.eq("tuned")]
    interactions = main[main.policy.eq("tuned") & main.contrast.eq("quantum_minus_classical")]
    supported = interactions[interactions.mean_difference.gt(0) & interactions.p_holm.lt(.05)]
    lines = ["# 전체 데이터 통제 실험 결과", "", "## 자료 이력", "", f"- 실행명: `{run_id}`", "- 유형: 전체 데이터 코드 실험과 저장된 예측 검증", f"- 상태: 4개 데이터셋의 {len(seeds)}개 시드 모두 완료 및 일치 검증", f"- 실행 서버: `{metadata['host']}`", f"- 검증한 조합: {len(audit)}개, 예측 {int(audit.prediction_rows.sum()):,}행", "- 기존 표본 실험 결과를 확인한 후 사용자의 요청으로 확장했다. 독립적인 사전등록 확인 연구로 표현하지 않는다."]
    if amendment:
        lines.append("- 실행 도중 2026년 10월 16일 사용자의 요청에 따라 기존 시드 목록의 앞 10개로 줄였다. 성능에 따른 시드 선택은 하지 않았고, 원래 20개 시드 계획과 변경 기록을 함께 보존했다.")
    if "hmeq" in DATASETS:
        lines.append("- 사용자 요청으로 GMSC를 HMEQ로 교체한 별도 실행이다. HMEQ의 BAD는 부도 또는 심각한 연체를 뜻하며 결측 행을 삭제하지 않았다. REASON·JOB은 범주형으로 인코딩하고 수치형 중앙값·범주형 최빈값 대체는 각 학습 경계 안에서만 적합했다.")
        lines.append("- HMEQ는 [SAS 공식 저장소의 고정 커밋](https://github.com/sassoftware/sas-viya-dmml-pipelines/blob/cf9f13a0ce15c8968c879e9e6dcba11deeff200b/data/hmeq.csv)에서 받았다. 원래 금융기관의 이름과 수집 시점까지 확인했다는 뜻은 아니다.")
    lines.extend(["", "## 데이터 범위", "", "모든 원본 행을 시드별 학습 80%·테스트 20%로 층화 분할했다. 학습과 평가에 같은 행을 중복 사용하지 않았다. 학습 자료 내부 20%를 검증에 사용한 뒤 선택한 설정으로 외부 학습 전체를 재학습했다.", "", "| 데이터셋 | 전체 | 학습 | 테스트 |", "| --- | ---: | ---: | ---: |"])
    for dataset in DATASETS:
        lines.append(f"| {LABELS[dataset]} | " + " | ".join(f"{n:,}" for n in SIZES[dataset]) + " |")
    lines.extend(["", "German 두 자료는 같은 원자료 계열이므로 독립된 원자료는 세 계열이다. 부분집합을 선택하지 않았으므로 PCA 거리 표본 선택 민감도는 이 실행에 적용하지 않는다. 이전 표본 실험은 별도 결과로 유지한다.", "", "## 핵심 평균 비용", "", "주 선택 기준은 내부 검증의 `5FN+FP`이며, 임계값도 검증 자료에서 결정했다. 아래는 테스트 전체의 평균 비용이다. 자료별 테스트 수가 다르므로 서로 다른 자료 간 비용 크기를 직접 비교하지 않는다.", "", "| 데이터셋 | RBF 기본 | RBF 볼록형 | RBF 특성별 | QSVC 기본 | QSVC 볼록형 | QSVC 특성별 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for dataset in DATASETS:
        values = [primary[(primary.dataset == dataset) & primary.learner.eq(learner) & primary.representation.eq(representation)].cost.mean() for learner in ("RBF-SVM", "QSVC") for representation in ("baseline", "convex", "feature_specific")]
        lines.append(f"| {LABELS[dataset]} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines.extend(["", "![핵심 비교](controlled_main_cost.png)", "", "## 양자 커널의 추가 개선량", "", "차이는 `(QSVC 기본 - QSVC 위험반영) - (RBF 기본 - RBF 위험반영)`으로 정의한다. 양수라도 최종 성능이나 계산시간의 양자 우위를 뜻하지 않는다.", "", "| 자료 | 변환 | 평균 차이 | 보정 95% 신뢰구간 | Holm p |", "| --- | --- | ---: | --- | ---: |"])
    for row in interactions.itertuples():
        lines.append(f"| {LABELS[row.dataset]} | {row.representation} | {row.mean_difference:.2f} | [{row.ci_low:.2f}, {row.ci_high:.2f}] | {row.p_holm:.4f} |")
    lines.extend(["", f"양의 추가 개선과 Holm 보정 p<0.05를 함께 만족한 조건은 {len(supported)}개다. 유의하지 않은 결과는 효과가 정확히 없다는 증명이 아니다.", "", f"반복 분할의 중복을 고려한 근사 보정 재표본 t 신뢰구간을 사용했다. {len(seeds)}개 시드를 독립된 {len(seeds)}개 모집단으로 간주하지 않는다. p에는 주요 비교 전체의 Holm 보정을 적용했으나 신뢰구간은 동시신뢰구간이 아니다. [원 방법](https://proceedings.neurips.cc/paper_files/paper/1999/file/7d12b66d3df6af8d429c1a357d8b9e1a-Paper.pdf)", "", "![추가 개선량](controlled_incremental_effect.png)", "", "## 고전 비교 모델", "", "비교 모델은 PCA 이전의 인코딩된 전체 특성을 사용하므로 핵심 모델의 공통 7차원 입력 비교와 구분한다.", "", "| 자료 | LR | RF | XGBoost | 위험점수만 사용 |", "| --- | ---: | ---: | ---: | ---: |"])
    for dataset in DATASETS:
        values = [primary[primary.dataset.eq(dataset) & primary.learner.eq(learner)].cost.mean() for learner in ("LR", "RF", "XGBoost", "risk_only")]
        lines.append(f"| {LABELS[dataset]} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines.extend([
        "", "## 계산 및 민감도", "",
        "수식·특성맵·검증 후보 예산은 기존 통제 실험과 같다. 전체 학습 커널은 배정밀도로 대칭 블록 계산하며, 근사 커널이나 일부 행만으로 학습하지 않는다. 평가 커널을 작은 행 묶음으로 계산하는 것은 메모리 사용만 바꾼다. 7큐비트의 이상적 CPU 상태벡터 시뮬레이션이므로 실제 양자 하드웨어 속도·잡음·계산 이득을 검증하지 않았다.",
        "", "위험점수를 섞은 대조군과 상수 위험점수 대조군의 비교는 `ablation_contrasts.csv`에 있다. 비용비율 2:1·10:1 민감도는 고정된 예측의 재평가이며 비율별 재튜닝이 아니다. 전체 데이터 사용은 원자료 자체의 대표성 문제를 해소하지 않는다.",
        "", "![대조군](controlled_risk_controls.png)", "", "## 해석상 주의점 검토", "",
        "| 항목 | 처리와 한계 |", "| --- | --- |",
        "| 집계 방향 역전 | 자료별 분리 보고, 하위집단 역전은 별도 미검증 |",
        "| 집단 결과의 개인 적용 | 평균 비용을 개별 차주의 효과로 해석하지 않음 |",
        "| 선택 편향 | 모든 원본 행 사용, 원자료 수집 편향은 남음 |",
        "| 충돌변수 편향 | 인과 조정 효과를 추정하지 않음 |",
        "| 기저율 무시 | 테스트 양성 수 및 위험점수 진단을 기록 |",
        "| 평균으로의 회귀 | 극단적 성능 사례를 골라 분석하지 않음 |",
        f"| 생존자 편향 | 예정된 모든 {len(expected)}개 조합과 54개 결과 조건을 검증 |",
        "| 다중 탐색 | 조건 전체 기록, 주요·대조군 비교군별 Holm 보정 |",
        "| 분석 선택의 자유도 | 기존 표본 결과 이후 확장임을 명시 |",
        "| 인과 과장 | 입력 변환의 예측 비교로 범위를 한정 |",
        "| 역인과 | 예측 결과를 인과 방향의 근거로 사용하지 않음 |", "",
        "11개 항목을 위 범위에서 검토했다. 시드 간 원자료 중복과 같은 원자료 계열 두 자료를 고려해 해석해야 한다.",
        "", "## 검증과 재현", "",
        "원본 행의 전체 분할, 학습·평가 분리, 원본 라벨, 저장된 점수·임계값·예측 및 FN·FP·비용 일치를 검증했다. 이는 전체 실험의 독립적인 재실행 검증과는 다르다. 코드·환경·데이터 해시는 `metadata.json`, `data_manifest.json`, `verification_receipt.json`에 기록한다. 세부 분할·탐색·예측은 각 자료의 `results/<실행명>/seed_*/`에 있다.",
    ])
    (output / "결과보고서.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {"scope": "full source partitions, original labels, held-out predictions, thresholds, confusion counts and paired summaries; not an independent rerun", "seeds_checked": seeds, "seed_jobs_checked": len(audit), "prediction_rows_checked": int(audit.prediction_rows.sum()), "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(), "seed_amendment_sha256": metadata.get("seed_plan_sha256")}
    (output / "verification_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_20261016_hmeq_full")
    build(parser.parse_args().run_id)
