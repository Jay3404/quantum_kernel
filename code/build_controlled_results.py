"""Validate and summarize the completed CMC controlled experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _shared.controlled import LEGACY_DATASETS as DATASETS, load_data

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"german_numeric": "German Numeric", "south_german": "South German", "taiwan": "Taiwan Default", "give_me_some_credit": "Give Me Some Credit", "hmeq": "HMEQ", "fico_heloc": "FICO HELOC"}
COLORS = {"baseline": "#525252", "convex": "#0072b2", "feature_specific": "#d55e00"}


def corrected_difference(values: np.ndarray, ratio: float) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    variance = float(values.var(ddof=1))
    se = np.sqrt((1 / n + ratio) * variance)
    half = float(stats.t.ppf(0.975, n - 1) * se)
    p = float(2 * stats.t.sf(abs(mean / se), n - 1)) if se else (1.0 if mean == 0 else 0.0)
    return {"n": n, "mean_difference": mean, "sd_difference": np.sqrt(variance), "ci_low": mean - half, "ci_high": mean + half, "p_value": p, "wins": int((values > 0).sum()), "ties": int((values == 0).sum()), "losses": int((values < 0).sum()), "test_train_ratio": ratio}


def holm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted


def paired_contrasts(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    main, ablations = [], []
    for (dataset, policy), group in raw.groupby(["dataset", "policy"]):
        costs = group.pivot(index="seed", columns=["learner", "representation"], values="cost")
        ratio = float((group.n_test / group.n_train).mean())
        n_test = int(group.n_test.iloc[0])
        improvements = {}
        for learner in ("RBF-SVM", "QSVC"):
            baseline = costs[learner, "baseline"]
            for representation in ("convex", "feature_specific"):
                delta = baseline - costs[learner, representation]
                improvements[learner, representation] = delta
                result = corrected_difference(delta.to_numpy(), ratio)
                rcr = 100 * delta / baseline
                main.append({"dataset": dataset, "policy": policy, "contrast": "within_kernel", "learner": learner, "representation": representation, "n_test": n_test, "mean_rcr_pct": float(rcr.mean()), **result})
                for control in ("shuffled", "constant"):
                    key = (learner, f"{representation}_{control}")
                    if key in costs:
                        difference = costs[key] - costs[learner, representation]
                        ablations.append({"dataset": dataset, "policy": policy, "learner": learner, "representation": representation, "control": control, "n_test": n_test, **corrected_difference(difference.to_numpy(), ratio)})
        for representation in ("convex", "feature_specific"):
            delta = improvements["QSVC", representation] - improvements["RBF-SVM", representation]
            main.append({"dataset": dataset, "policy": policy, "contrast": "quantum_minus_classical", "learner": "interaction", "representation": representation, "n_test": n_test, "mean_rcr_pct": np.nan, **corrected_difference(delta.to_numpy(), ratio)})
    main, ablations = pd.DataFrame(main), pd.DataFrame(ablations)
    for frame in (main, ablations):
        frame["p_holm"] = holm(frame.p_value.to_numpy())
    return main, ablations


def audit_predictions(raw: pd.DataFrame, run_id: str) -> tuple[pd.DataFrame, list[dict]]:
    keys = ["learner", "representation", "policy"]
    audit_rows, risk_rows = [], []
    for (dataset, seed), group in raw.groupby(["dataset", "seed"]):
        folder = ROOT / "code" / dataset / "results" / run_id / f"seed_{seed:03d}"
        split = json.loads((folder / "split_indices.json").read_text())
        if set(split["train"]) & set(split["test"]):
            raise AssertionError("Train/test overlap")
        if set(split["inner_train"]) & set(split["inner_valid"]):
            raise AssertionError("Inner train/validation overlap")
        if set(split["inner_train"]) | set(split["inner_valid"]) != set(split["train"]):
            raise AssertionError("Inner split is not a partition of training")
        predicted = pd.read_csv(folder / "predictions.csv.gz", float_precision="round_trip")
        if predicted.duplicated(keys + ["row_id"]).any():
            raise AssertionError("Duplicate predictions")
        if set(predicted.row_id) != set(split["test"]):
            raise AssertionError("Unexpected held-out cases")
        source_y = load_data(dataset, str(ROOT / "data"))[1]
        if not np.array_equal(predicted.y.to_numpy(), source_y[predicted.row_id.to_numpy()]):
            raise AssertionError("Saved labels differ from source targets")
        counted = predicted.assign(fn=(predicted.y.eq(1) & predicted.prediction.eq(0)).astype(int), fp=(predicted.y.eq(0) & predicted.prediction.eq(1)).astype(int))
        counted = counted.groupby(keys).agg(fn=("fn", "sum"), fp=("fp", "sum"), count=("row_id", "size")).reset_index()
        matched = group.merge(counted, on=keys, suffixes=("", "_saved"), validate="one_to_one")
        if len(matched) != len(group) or len(counted) != len(group):
            raise AssertionError("Prediction model matrix differs from metric matrix")
        if not ((matched.fn == matched.fn_saved) & (matched.fp == matched.fp_saved) & (matched.n_test == matched["count"]) & (matched.cost == 5 * matched.fn + matched.fp)).all():
            raise AssertionError("Saved metrics do not agree with predictions")
        scored = predicted.merge(group[keys + ["threshold"]], on=keys, validate="many_to_one")
        if not np.array_equal((scored.score >= scored.threshold).astype(int), scored.prediction):
            raise AssertionError("Scores and saved thresholds do not reproduce predictions")
        risk = predicted[predicted.learner.eq("risk_only") & predicted.policy.eq("tuned")].copy()
        risk["bin"] = np.minimum((risk.score * 10).astype(int), 9)
        reliability = risk.groupby("bin").agg(predicted=("score", "mean"), observed=("y", "mean"), count=("y", "size"))
        ece = float((abs(reliability.predicted - reliability.observed) * reliability["count"]).sum() / len(risk))
        diagnostic = json.loads((folder / "risk_diagnostics.json").read_text())
        risk_rows.append({"dataset": dataset, "seed": seed, "ece_10_bins": ece, **{k: v for k, v in diagnostic.items() if k not in ("weights", "risk_estimator")}})
        audit_rows.append({"dataset": dataset, "seed": seed, "models_and_policies": len(group), "held_out_cases": len(split["test"]), "prediction_rows": len(predicted), "status": "PREDICTION_CONSISTENCY_CHECKED"})
    return pd.DataFrame(audit_rows), risk_rows


def save_figure(figure, folder: Path, name: str) -> None:
    for extension in ("png", "pdf", "svg"):
        figure.savefig(folder / f"{name}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def figures(raw: pd.DataFrame, main: pd.DataFrame, ablations: pd.DataFrame, output: Path, datasets=DATASETS) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    primary = raw[raw.policy.eq("tuned")]
    figure, axes = plt.subplots(2, 2, figsize=(10, 6.3), constrained_layout=True)
    figure.suptitle(f"Mean cost and SD across {raw.seed.nunique()} paired splits", fontsize=10)
    for ax, dataset in zip(axes.flat, datasets):
        for offset, representation in zip((-0.16, 0, 0.16), ("baseline", "convex", "feature_specific")):
            for x, learner in enumerate(("RBF-SVM", "QSVC")):
                values = primary[(primary.dataset == dataset) & (primary.learner == learner) & (primary.representation == representation)]
                normalized = values.cost / values.n_test * 1000
                ax.errorbar(x + offset, normalized.mean(), yerr=normalized.std(ddof=1), fmt="o", color=COLORS[representation], capsize=3, label=representation.replace("_", " ") if x == 0 else None)
        ax.set(title=LABELS[dataset], xticks=[0, 1], xticklabels=["RBF-SVM", "QSVC"], xlim=(-0.5, 1.5), ylabel="Cost per 1,000 test cases")
        ax.grid(axis="y", alpha=0.2)
    axes.flat[0].legend(frameon=False, fontsize=8)
    save_figure(figure, output, "controlled_main_cost")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, representation in zip(axes, ("convex", "feature_specific")):
        selected = main[(main.policy == "tuned") & (main.contrast == "quantum_minus_classical") & (main.representation == representation)].set_index("dataset")
        for index, dataset in enumerate(datasets):
            row = selected.loc[dataset]
            scale = 1000 / row.n_test
            ax.errorbar(row.mean_difference * scale, index, xerr=np.array([[row.mean_difference - row.ci_low], [row.ci_high - row.mean_difference]]) * scale, fmt="o", color=COLORS[representation], capsize=3)
        ax.axvline(0, color="#666666", linestyle="--", linewidth=0.8)
        ax.set(yticks=range(len(datasets)), yticklabels=[LABELS[d] for d in datasets], title=representation.replace("_", " "), xlabel="Quantum minus classical improvement\n(cost per 1,000; corrected 95% CI)")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    save_figure(figure, output, "controlled_incremental_effect")

    figure, axes = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)
    for ax, dataset in zip(axes.flat, datasets):
        selected = ablations[(ablations.policy == "tuned") & (ablations.dataset == dataset)]
        for index, (learner, representation) in enumerate(itertools_pairs()):
            for offset, control, marker in ((-0.09, "shuffled", "o"), (0.09, "constant", "s")):
                row = selected[(selected.learner == learner) & (selected.representation == representation) & (selected.control == control)].iloc[0]
                scale = 1000 / row.n_test
                ax.errorbar(index + offset, row.mean_difference * scale, yerr=np.array([[row.mean_difference - row.ci_low], [row.ci_high - row.mean_difference]]) * scale, fmt=marker, color=COLORS[representation], capsize=2, label=control if index == 0 else None)
        ax.axhline(0, color="#666666", linestyle="--", linewidth=0.8)
        ax.set(title=LABELS[dataset], xticks=range(4), xticklabels=["RBF\nconvex", "RBF\nfeature", "QSVC\nconvex", "QSVC\nfeature"], ylabel="Control cost minus real-risk cost\n(per 1,000; corrected 95% CI)")
    axes.flat[0].legend(frameon=False, fontsize=8)
    save_figure(figure, output, "controlled_risk_controls")


def itertools_pairs():
    return [(learner, representation) for learner in ("RBF-SVM", "QSVC") for representation in ("convex", "feature_specific")]


def sampling_comparison(raw: pd.DataFrame, run_id: str, distance_id: str, output: Path) -> pd.DataFrame:
    distance = pd.read_csv(ROOT / "code" / "results" / distance_id / "raw_metrics.csv")
    distance_audit, _ = audit_predictions(distance, distance_id)
    distance_audit.to_csv(output / "distance_prediction_audit.csv", index=False)
    keys = ["dataset", "seed", "learner", "representation", "policy"]
    joined = distance.merge(raw, on=keys, suffixes=("_distance", "_random"), validate="one_to_one")
    for dataset, seed in joined[["dataset", "seed"]].drop_duplicates().itertuples(index=False):
        a = json.loads((ROOT / "code" / dataset / "results" / run_id / f"seed_{seed:03d}" / "split_indices.json").read_text())
        b = json.loads((ROOT / "code" / dataset / "results" / distance_id / f"seed_{seed:03d}" / "split_indices.json").read_text())
        if a["test"] != b["test"]:
            raise AssertionError("Sampling sensitivity changed held-out cases")
    rows = []
    for (dataset, learner, representation, policy), group in joined.groupby(["dataset", "learner", "representation", "policy"]):
        delta = group.cost_distance - group.cost_random
        rows.append({"dataset": dataset, "learner": learner, "representation": representation, "policy": policy, "random_cost": group.cost_random.mean(), "distance_cost": group.cost_distance.mean(), **corrected_difference(delta.to_numpy(), float((group.n_test_random / group.n_train_random).mean()))})
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm(frame.p_value.to_numpy())
    frame.to_csv(output / "sampling_sensitivity.csv", index=False)
    return frame


def write_report(raw, main, ablations, audit, risk, sampling, metadata, output):
    primary = raw[raw.policy.eq("tuned")]
    interactions = main[(main.policy == "tuned") & (main.contrast == "quantum_minus_classical")]
    supported = interactions[(interactions.mean_difference > 0) & (interactions.p_holm < 0.05)]
    lines = ["# 서버 통제 실험 결과", "", "## 자료 이력", "", f"- 실행명: `{metadata['run_id']}`", "- 유형: 코드 실험 및 예측 결과 검증", "- 상태: 전체 실행 완료, 저장된 예측으로 지표 재계산 검증 완료", f"- 실행 서버: `{metadata['host']}`", f"- 완료한 데이터셋·시드 조합: {len(audit)}개", f"- 전체 모델·평가 조건 결과: {len(raw):,}행", "- 데이터셋은 4개, 독립된 원자료 계열은 3개다.", "", "## 실험 조건", "", "German Numeric과 South German은 학습 800개·테스트 200개를 사용했다. Taiwan과 Give Me Some Credit은 원자료를 먼저 학습·테스트로 분리한 뒤 학습 2,000개·테스트 1,000개를 각각 층화 무작위 추출했다. 결과는 이 표본 규모에 한정한다. 각 데이터셋에서 20개 시드를 사용했고, 모든 모델이 같은 테스트 표본을 공유한다.", "", "주 결과는 내부 검증의 `5FN + FP`로 하이퍼파라미터와 임계값을 선택한 조건이다. 고정 임계값의 비용 기준 선택과 고정 임계값의 F1 기준 선택은 보조 조건으로 기록했다. 실제 부도손실 금액을 추정한 결과가 아니라 5:1 오분류 비용 가정에 따른 비교다.", "", "## 핵심 모델의 평균 비용", "", "아래 값은 테스트 오분류 비용의 평균이다. 평균·표준편차와 F1·재현율·균형 정확도·FN·FP는 `summary_metrics.csv`에 함께 기록했다.", "", "| 데이터셋 | RBF 기본 | RBF 볼록형 | RBF 특성별 | QSVC 기본 | QSVC 볼록형 | QSVC 특성별 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        values = []
        for learner, representation in [(l, r) for l in ("RBF-SVM", "QSVC") for r in ("baseline", "convex", "feature_specific")]:
            subset = primary[(primary.dataset == dataset) & (primary.learner == learner) & (primary.representation == representation)]
            values.append(f"{subset.cost.mean():.1f}")
        lines.append(f"| {LABELS[dataset]} | " + " | ".join(values) + " |")
    lines.extend(["", "![핵심 비교](controlled_main_cost.png)", "", "## 양자 커널의 추가 개선량", "", "추가 개선량은 `(양자 기본 비용 - 양자 위험정보 반영 비용) - (고전 기본 비용 - 고전 위험정보 반영 비용)`이다. 양수이면 양자 커널 내부의 개선폭이 더 크다는 뜻이며, 최종 비용이나 계산 효율에서 고전 모델보다 우수하다는 뜻은 아니다.", "", "| 데이터셋 | 변환 | 추가 개선량 | 보정 95% 신뢰구간 | Holm 보정 p | 양의 방향 시드 |", "| --- | --- | ---: | --- | ---: | ---: |"])
    for row in interactions.itertuples():
        name = "볼록형" if row.representation == "convex" else "특성별"
        lines.append(f"| {LABELS[row.dataset]} | {name} | {row.mean_difference:.2f} | [{row.ci_low:.2f}, {row.ci_high:.2f}] | {row.p_holm:.4f} | {row.wins}/{row.n} |")
    lines.extend(["", f"이 설정에서 양의 추가 개선량과 Holm 보정 p < 0.05를 함께 만족한 조건은 {len(supported)}개다. 유의한 양의 결과가 없더라도 효과가 정확히 0이라고 증명한 것은 아니다.", "", "![추가 개선량](controlled_incremental_effect.png)", "", "반복 분할의 표본 중복을 고려해 표준오차를 `sqrt((1/n + n_test/n_train) * 표본분산)`으로 보정하고 t 분포를 이용한 근사 신뢰구간과 양측 검정을 계산했다. 신뢰구간은 다중 비교 동시신뢰구간이 아니며, 검정 p에는 주요 변환·데이터셋·평가 조건 전체에 Holm 보정을 적용했다. 이 근사법의 가정과 3개 원자료 계열이라는 범위를 유지해 해석해야 한다. [통계 방법의 원 논문](https://proceedings.neurips.cc/paper_files/paper/1999/file/7d12b66d3df6af8d429c1a357d8b9e1a-Paper.pdf)", "", "## 위험정보 대조군", "", "위험점수를 섞은 조건과 학습 위험점수 평균으로 대체한 조건을 동일한 예산으로 튜닝했다. `ablation_contrasts.csv`의 양수는 실제 위험정보가 해당 대조군보다 낮은 비용을 보였다는 뜻이다. 두 대조군에서도 특성별 가중치는 유지하므로 표본별 위험정보의 추가 효과를 비교할 수 있다. 각도 정규화는 모든 조건이 같은 학습 기준을 공유하고, 범위 밖 입력을 자른 비율도 기록했다.", "", "![위험정보 대조군](controlled_risk_controls.png)", "", "## 고전 비교 모델과 계산시간", "", "로지스틱 회귀·랜덤 포리스트·XGBoost는 PCA 이전의 인코딩된 전체 특성을 사용한다. 핵심 여섯 모델의 입력 통제 비교와 구분해, 실제로 고려할 만한 비교 모델의 비용을 보여준다.", "", "| 데이터셋 | 로지스틱 회귀 | 랜덤 포리스트 | XGBoost | 위험점수만 사용 |", "| --- | ---: | ---: | ---: | ---: |"])
    for dataset in DATASETS:
        values = [primary[(primary.dataset == dataset) & (primary.learner == learner)].cost.mean() for learner in ("LR", "RF", "XGBoost", "risk_only")]
        lines.append(f"| {LABELS[dataset]} | " + " | ".join(f"{v:.1f}" for v in values) + " |")
    lines.extend(["", "전처리·커널 구성·분류기 학습·추론·탐색 경과시간은 `summary_metrics.csv`에 구분해 기록했다. 위험정보를 쓰는 모델의 전처리 시간에는 교차 적합과 확률 보정을 포함한다. 탐색 시간은 공통 내부 전처리를 제외한 모델별 탐색 시간이다. 병렬 작업 환경에서 측정한 경과시간이므로 단독 실행 성능이나 양자 하드웨어 실행시간으로 해석하지 않는다.", "", "양자 커널은 7큐비트, 1회 반복, 완전 연결 ZZ 특성맵과 CPU의 배정밀도 상태벡터 시뮬레이션을 사용했다. 유한 샷 잡음 및 장치 잡음은 포함하지 않았다.", "", "## 민감도와 확률 보정 점검", "", "`cost_ratio_sensitivity.csv`는 기존 예측을 2:1·5:1·10:1 비용비율로 재평가한 결과다. 해당 비용비율에 맞춰 재튜닝한 결과가 아니다. `selection_policy_comparison.csv`에는 비용 기준 선택과 F1 기준 선택, 고정 임계값과 조정 임계값 결과를 함께 남겼다.", ""])
    if len(sampling):
        lines.append("`sampling_sensitivity.csv`에는 Taiwan과 Give Me Some Credit의 학습 부분집합을 PCA 거리 기준으로 선택한 비교가 있다. 표본 선택은 학습 자료 안에서만 수행했고, 40개 데이터셋·시드 조합 모두 주 실험과 같은 테스트 행을 사용했음을 확인했다. 양수 차이는 무작위 표본 조건의 비용이 더 낮다는 뜻이다.")
    lines.extend(["", "위험점수는 학습 데이터 내부의 시그모이드 보정과 교차 적합을 사용했다. 아래 Brier 점수와 10구간 보정오차는 테스트 자료에서의 기술통계이며, 실제 부도확률이 완벽히 보정되었다는 보장은 아니다.", "", "| 데이터셋 | 평균 Brier 점수 | 평균 10구간 보정오차 | 평균 위험점수 AUC |", "| --- | ---: | ---: | ---: |"])
    for dataset, frame in risk.groupby("dataset", sort=False):
        lines.append(f"| {LABELS[dataset]} | {frame.brier_test.mean():.4f} | {frame.ece_10_bins.mean():.4f} | {frame.risk_auc_test.mean():.4f} |")
    lines.extend(["", "## 해석상 주의점 검토", "", "| 점검 항목 | 이번 분석에서의 처리와 범위 |", "| --- | --- |", "| 집계에 따른 방향 역전 | 데이터셋별 결과를 분리했다. 성별 등 하위집단의 역전 여부는 별도 검토하지 않았다. |", "| 집단 결과의 개인 적용 | 평균 비용을 개별 차주의 효과로 해석하지 않는다. |", "| 선택 편향 | 주 실험은 층화 무작위 표본이다. 원자료 수집 당시의 금융 표본 선택 편향은 해소하지 못한다. |", "| 충돌변수 편향 | 인과 조정변수의 효과를 추정한 연구가 아니다. |", "| 기저율 무시 | 시드별 테스트 양성 수와 위험점수 진단에 기저율을 기록했다. |", "| 평균으로의 회귀 | 주 표본은 성능이 극단적인 사례를 골라 구성하지 않았다. 거리 표본은 별도 민감도 분석이다. |", "| 생존자 편향 | 모든 예정 시드의 완료와 예측 행 수를 검증했다. 결과가 나쁜 시드를 제외하지 않았다. |", "| 다중 탐색 | 모든 변환·대조군을 기록하고 검정군별 Holm 보정을 적용했다. |", "| 분석 선택의 자유도 | 본 실행 전에 별도 설정 파일을 기록했다. 공개 사전등록을 했다는 뜻은 아니다. |", "| 인과 과장 | 입력 변환에 대한 계산 실험으로 한정하며 실제 차주의 금융 결과에 대한 인과효과를 주장하지 않는다. |", "| 역인과 해석 | 원자료의 변수 간 인과 방향을 추정하지 않았으며, 예측 성능을 원인과 결과의 방향에 대한 증거로 사용하지 않는다. |", "", "11개 항목을 위 범위에서 검토했다. 원자료의 대표성·하위집단 성능·실제 하드웨어 결과까지 검증했다는 의미는 아니다.", "", "## 재현과 파일 안내", "", "- `metadata.json`: 실행 조건, 코드 해시, 패키지 버전", "- `data_manifest.json`: 네 공식 원본의 출처와 파일 해시", "- `raw_metrics.csv`: 시드별 모든 평가 조건", "- `summary_metrics.csv`: 평균·표준편차와 실행시간", "- `main_contrasts.csv`: 짝지어진 비용 개선 및 추가 개선량", "- `ablation_contrasts.csv`: 위험정보 대조군 비교", "- `prediction_audit.csv`: 원본 라벨·분할·예측·임계값·혼동행렬의 일치 검증", "- `risk_diagnostics.csv`: 확률 보정 기술통계", "- 데이터셋별 `results/<실행명>/seed_*/`: 분할 행 번호, 검증 후보, 압축된 테스트 예측, 완료 기록", "", "수식은 기존 원고와 동일하게 유지했다. 클래스 가중 LR의 학습 내부 확률 보정, 공통 각도 정규화, 공정한 커널 비교는 개정된 실험 절차이므로 이전 5개 시드 결과와 같은 프로토콜의 반복으로 표현하지 않는다."])
    lines.extend(["", "## 최종 평가 단계의 평균 경과시간", "", "단위는 초이며, 전처리·최종 커널 계산·분류기 학습·추론을 합산한 값이다. 튜닝 시간은 제외한다. 위험정보 모델에는 위험점수의 교차 적합과 확률 보정 시간이 포함된다. 병렬 실행 환경에서의 참고 수치다.", "", "| 데이터셋 | RBF 기본 | QSVC 기본 | QSVC 볼록형 | XGBoost |", "| --- | ---: | ---: | ---: | ---: |"])
    for dataset in DATASETS:
        values = []
        for learner, representation in (("RBF-SVM", "baseline"), ("QSVC", "baseline"), ("QSVC", "convex"), ("XGBoost", "encoded_full")):
            values.append(primary[(primary.dataset == dataset) & (primary.learner == learner) & (primary.representation == representation)].final_total_seconds.mean())
        lines.append(f"| {LABELS[dataset]} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    receipt_path = output / "reproducibility_check.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        lines.extend(["", "## 예비 실행 재현 확인", "", f"학습 400개·테스트 200개의 예비 실행을 네 자료에서 별도 실행명으로 다시 수행했다. {receipt['datasets_checked']}개 자료 모두 저장된 테스트 점수·예측 및 비용 지표가 정확히 일치했다. 본 실험의 모든 20개 시드를 독립 재실행한 검증과는 범위를 구분한다."])
    lines.extend(["", "## 양자 커널 계산량", "", "`quantum_kernel_cost.csv`에는 최종 모델 평가마다 구성한 학습·테스트 커널의 원소 수, 상태벡터 수, 계산 경과시간을 기록했다. 행렬 원소 수는 실제 양자 하드웨어에서 실행한 회로 횟수의 계측값이 아니다."])
    (output / "결과보고서.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cmc_20261015")
    parser.add_argument("--distance-id", default="cmc_distance_20261015")
    args = parser.parse_args()
    output = ROOT / "code" / "results" / args.run_id
    metadata = json.loads((output / "metadata.json").read_text())
    raw = pd.read_csv(output / "raw_metrics.csv", float_precision="round_trip")
    if set(raw.dataset) != set(DATASETS) or (raw.groupby("dataset").seed.nunique() != 20).any():
        raise AssertionError("The agreed four-dataset, 20-seed matrix is incomplete")
    audit, risk_rows = audit_predictions(raw, args.run_id)
    audit.to_csv(output / "prediction_audit.csv", index=False)
    risk = pd.DataFrame(risk_rows)
    risk.to_csv(output / "risk_diagnostics.csv", index=False)
    numeric = ["cost", "fn", "fp", "f1", "recall", "balanced_accuracy", "roc_auc", "lambda", "preprocessing_seconds", "train_kernel_seconds", "test_kernel_seconds", "fit_seconds", "inference_seconds", "tuning_seconds", "final_total_seconds", "angle_clipped_fraction", "kernel_alignment", "effective_rank"]
    raw.groupby(["dataset", "learner", "representation", "policy"])[numeric].agg(["mean", "std"]).to_csv(output / "summary_metrics.csv")
    quantum = raw[raw.learner.eq("QSVC")].copy()
    quantum["train_kernel_entries"] = quantum.n_train ** 2
    quantum["test_kernel_entries"] = quantum.n_train * quantum.n_test
    quantum["statevectors_per_final_evaluation"] = quantum.n_train + quantum.n_test
    quantum[["dataset", "seed", "representation", "policy", "n_train", "n_test", "train_kernel_entries", "test_kernel_entries", "statevectors_per_final_evaluation", "train_kernel_seconds", "test_kernel_seconds"]].to_csv(output / "quantum_kernel_cost.csv", index=False)
    main_table, ablations = paired_contrasts(raw)
    main_table.to_csv(output / "main_contrasts.csv", index=False)
    ablations.to_csv(output / "ablation_contrasts.csv", index=False)
    sensitivity = pd.concat([raw.assign(cost_ratio=ratio, sensitivity_cost=ratio * raw.fn + raw.fp) for ratio in (2, 5, 10)], ignore_index=True)
    sensitivity.groupby(["dataset", "learner", "representation", "policy", "cost_ratio"]).sensitivity_cost.agg(["mean", "std"]).to_csv(output / "cost_ratio_sensitivity.csv")
    raw[raw.representation.isin(["baseline", "convex", "feature_specific", "encoded_full", "risk_score"])].groupby(["dataset", "learner", "representation", "policy"]).cost.agg(["mean", "std"]).to_csv(output / "selection_policy_comparison.csv")
    sampling = sampling_comparison(raw, args.run_id, args.distance_id, output)
    figures(raw, main_table, ablations, output)
    source = ROOT / "data" / "acquisition_manifest.json"
    shutil.copyfile(source, output / "data_manifest.json")
    shutil.copyfile(ROOT / "docs" / "cmc_experiment_protocol.json", output / "experiment_protocol.json")
    write_report(raw, main_table, ablations, audit, risk, sampling, metadata, output)
    receipt = {"analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "protocol_sha256": hashlib.sha256((ROOT / "docs" / "cmc_experiment_protocol.json").read_bytes()).hexdigest(), "main_seed_jobs_checked": len(audit), "main_prediction_rows_checked": int(audit.prediction_rows.sum()), "distance_seed_jobs_checked": 40, "scope": "saved predictions, original labels, split partitions, thresholds, confusion counts, paired summaries"}
    (output / "verification_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"verified seeds={len(audit)} prediction_rows={audit.prediction_rows.sum()} metrics={len(raw)}")
    print(output / "결과보고서.md")


if __name__ == "__main__":
    main()
