"""Publish the completed risk-only component control without changing old reports."""
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analyze_pca7_professor_comments import sha


def main():
    base = Path(__file__).resolve().parents[1]
    root = base / "code/results/cmc_risk_only_ablation_20261023"
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["status"] == "completed" and meta["jobs_completed"] == 160 and not meta["failures"]
    assert json.loads((root / "independent_validation_receipt.json").read_text())["status"] == "passed"
    data = pd.read_csv(root / "paired_statistics.csv", float_precision="round_trip")
    names = {"german_numeric": "German Numeric", "south_german": "South German", "hmeq": "HMEQ", "taiwan": "Taiwan"}
    methods = {"gbdt_soft_gate": "Soft Gate", "gbdt_rank_append": "Rank Append"}
    profiles = [("RBF-SVM", "gbdt_soft_gate"), ("RBF-SVM", "gbdt_rank_append"), ("QSVC", "gbdt_soft_gate"), ("QSVC", "gbdt_rank_append")]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (dataset, title) in zip(axes.flat, names.items()):
        group = data[data.dataset.eq(dataset)]
        for index, (kernel, method) in enumerate(profiles):
            row = group[group.kernel.eq(kernel) & group.method.eq(method)].iloc[0]
            color = "#18765d" if method == "gbdt_soft_gate" else "#b43c73"
            ax.errorbar(row.mean_difference, 3 - index,
                xerr=[[row.mean_difference - row.ci_low], [row.ci_high - row.mean_difference]],
                marker="o" if kernel == "RBF-SVM" else "^", color=color, capsize=4, linestyle="none", markersize=7)
        ax.axvline(0, color="#666666", linewidth=1)
        ax.set_yticks(range(4), [f"{k} / {methods[m]}" for k, m in profiles[::-1]], fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(-.5, 3.5)
        ax.grid(axis="x", alpha=.2)
        ax.set_xlabel("Risk-only cost - full product cost", fontsize=10)
    figure.suptitle("Does the customer-level backbone add cost benefit at matched parameters?", fontsize=13)
    figure.text(.5, .015, "Individual corrected-resampling 95% CIs; 10 overlapping holdouts. No Holm16-adjusted p < .05. Exploratory, not equivalence tests.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .05, 1, .95))
    artifacts = []
    for suffix in ("png", "pdf"):
        file = base / "docs/assets" / ("cmc_risk_only_ablation_20261023." + suffix)
        assert not file.exists()
        figure.savefig(file, dpi=180)
        artifacts.append(file)
    plt.close(figure)
    lines = ["# 추가 필수 대조: 위험요인만으로 충분한가?", "", "## Material Passport", "",
        "작성일: 2026-10-23. 상태: ANALYZED. 사용자의 추가 실험 승인에 따라 기존 두 방법의 구성요소 대조를 새로 수행했습니다. 네 자료×10시드×부모 커널 2개×방법 2개, 총160개 전체 학습행 SVC를 완료했고 실패는0건입니다. 기존 보고서·동결 소스·모델·분할은 수정하지 않았습니다.", "",
        "## 1. 실험 목적과 통제", "",
        "지금까지는 위험정보가 없는 모델과 결합모델을 비교했습니다. 이번에는 반대로 **고객별 PCA7 backbone을 제거하고, 기존과 정확히 같은 위험 커널만 남겼습니다.** 이 대조로 위험점수 자체의 예측력과, matched 설정에서 backbone을 더했을 때의 추가 효용을 구분합니다.", "",
        "C·lambda·기하는 결합모델이 검증에서 선택한 값을 그대로 고정했습니다. 새 파라미터 탐색은 없고, 각 대조모델의 임계값만 같은 검증5FN+FP 기준으로 정했습니다. 모든160개 모델·임계값을 잠근 뒤 테스트 레이블을 읽었습니다. 위험 교사와 위험점수는 새로 학습하지 않았고, 훈련5OOF·같은80/10/10 고객 분할을 유지했습니다.", "",
        "Soft Gate와 QSVC Rank Append의 위험요인은 정확한3차원 고전 특징지도로, RBF Rank Append는1차원 Gaussian 커널로 구현했습니다. 따라서 표의 RBF/QSVC는 **어느 결합모델의 파라미터와 위험요인을 맞췄는지**를 나타냅니다. 위험요인 단독에 양자 장치를 사용했다는 뜻은 아닙니다. RBF 폭은 부모의 훈련 분산 기반 값을 상속하지만, 고객별 PCA 좌표는 단독 모델 입력에 넣지 않았습니다.", "",
        "기술검사5개와24,000학습행 사전 측정3개가 통과했습니다. 새 모델160개 모두 전체 학습행·수렴·부모 파라미터 일치를 확인했고, 저장 모델의 예측 재실행 및 별도 FN/FP·통계 재계산도 통과했습니다. 로컬 새 적합의 최대 관측 메모리는 약0.70GiB였습니다.", "",
        "## 2. 전체 결과", "",
        "비용=5FN+FP,10시드 평균±표본표준편차입니다. 차이=위험요인 단독−전체 제품 커널이므로 양수는 전체 결합모델의 관측 비용이 낮다는 뜻입니다. 새 진단대비16개 전부에 탐색적 Holm16을 적용했고, 기존184/100/40가족은 변경하지 않았습니다.", "",
        "| 자료 | 부모 커널 | 방법 | 위험요인 단독 | 전체 결합 평균 | 차이 | 근사 개별95% CI | 원p | 탐색 Holm16 p |", "|---|---|---|---:|---:|---:|---|---:|---:|"]
    for row in data.itertuples():
        lines.append(f"| {names[row.dataset]} | {row.kernel} | {methods[row.method]} | {row.risk_only_mean:.1f}±{row.risk_only_sd:.1f} | {row.full_product_mean:.1f} | {row.mean_difference:.1f} | [{row.ci_low:.2f}, {row.ci_high:.2f}] | {row.p_value:.5f} | {row.p_holm16_exploratory:.5f} |")
    lines += ["", "![위험요인 단독 대비 결합모델의 차이](assets/cmc_risk_only_ablation_20261023.png)", "",
        "## 3. 해석", "",
        "**HMEQ:** 위험요인 단독 평균 비용은 네 matched 설정 모두154.8이었습니다. 전체 결합은RBF Soft Gate141.4, RBF Rank Append146.4, QSVC Soft Gate142.5, QSVC Rank Append140.7입니다. QSVC Rank Append의 추가 차이14.1은 개별 근사95% CI[1.54,26.66],원p=.03173이지만 새16대비 전체 보정 후p=.49625입니다. QSVC Soft Gate도 같은 보정 후p=.49625입니다.", "",
        "따라서 위험점수만 남긴 단순 대조에서도 상당히 낮은 비용이 관찰됐고, **그 위에 고객별 backbone을 더하는 추가 효용의 확실한 근거는 이번 가족보정 아래 확보하지 못했습니다.** 이것은 두 모델의 동등성이나 backbone이 불필요함을 입증한 것은 아닙니다. 차이의 CI가 넓고, 이 대조는 full-model용 파라미터를 상속한 조건부 비교입니다.", "",
        "**Taiwan:** 위험요인 단독은1669.2~1669.4였고, 전체 결합은1665.1~1682.6이었습니다. RBF Soft Gate를 제외한 세 matched 설정에서 결합모델의 관측 평균비용이 더 높았지만, 새16대비 전체 보정 후 유의한 차이는 없었습니다. German 두 표현에서도 작거나 방향이 섞인 차이가 관찰됐습니다.", "",
        "기본 QSVC, 위험요인 단독, 전체 결합의 비용을 단계적으로 나열할 수는 있지만, 이를 인과적 비율 분해로 해석하지 않습니다. 기본형과 결합형은 독립적으로 검증 선택됐고, 각 모델은 자체 검증 임계값을 사용합니다. 점수 척도·동률 처리·커널 곡률의 영향도 함께 남습니다.", "",
        "**Tuned XGB 단독:** HMEQ109.5±13.7이라는 기존 결과는 유지됩니다. 이번 대조에서도 Quantum Kernel을 결합해야 할 실무적 필요성은 입증되지 않았습니다. 반대로 더 강한 별도 XGB와 약한 위험 교사의 차이를 같은 모델의 보정 효과로 혼동하지 않습니다.", "",
        "## 4. 통계 및 재현 범위", "",
        "근사 corrected resampled t(df9,분산계수.225),10개 짝 차이와 개별95% CI를 사용했습니다. 새16개 위험요인 대조 전체를 고정한 뒤 적합을 시작했지만, 이미 본 자료·시드·테스트에 이어진 탐색적 후속이며 독립 확인이나 깨끗한 사전등록 연구는 아닙니다. 이Holm16은 새 대조의 가족이며, 기존 유리한 대비만 골라184/100을 줄인 것이 아닙니다.", "",
        "독립 검증은 저장 모델·예측·FN/FP·수식·통계의 별도 재계산입니다. 모든 모델을 독립적으로 다시 학습하거나 새 외부 자료에서 확인한 것은 아닙니다. 새 SVC는 로컬sklearn1.8.0, 부모는 서버sklearn1.7.2와 전계산 Gram을 사용했습니다. 정확한 커널 항등식과 native/precomputed SVC의 합성 등가 검사를 통과했지만, 바이트 동일 모델이나 환경 간 속도 우위는 주장하지 않습니다.", "",
        "## 5. 원고에 반영할 메시지", "",
        "이번 결과는 기존Contribution을 더 좁혀야 한다는 근거입니다. 단순히 결합 후 비용이 낮아졌다는 사실만으로 새로운 커널 구조의 필요성을 주장할 수 없습니다. 위험정보 자체를 남긴 대조, 분포 보존 셔플, 차원 증가 대조를 함께 제시해야 합니다.", "",
        "권장 중심 메시지: **위험정보는 커널모델의 관측 성능을 크게 바꿀 수 있지만, 그 효과가 새 커널 구조의 독자적 가치나 양자 고유 이점으로 이어지는지는 별도 통제가 필요하다.** 여기에 제품 구조에서 기대되는Effective Rank 증가와 실제 비용 효용의 불일치를 연결합니다. 이 방향의 문헌 대비 신규성은 여전히 별도로 검토해야 합니다.", "",
        "추가 모델·튜닝 격자를 늘리지는 않았습니다. 위험요인 단독의 추가 최적화나 결과를 본 뒤의 모델 확장은 이번 범위에 넣지 않았습니다. 현재 결과는 조건부 대조이며, 더 좋은 결과가 나올 때까지 실험을 늘리는 방식으로 해석하지 않습니다.", "",
        "[이전 3차 피드백 보고서](cmc_professor_revision_20261023_ko.md) · [전체 시드별 결과와 저장 모델](../code/results/cmc_risk_only_ablation_20261023/report_ko.md)", ""]
    report = base / "docs/cmc_risk_only_ablation_20261023_ko.md"
    with report.open("x") as handle:handle.write("\n".join(lines))
    artifacts.append(report)
    receipt = {"utc": datetime.now(timezone.utc).isoformat(), "status": "ready_for_delivery", "scope": "Report/figure rendering only, no new fits",
        "renderer_sha256": sha(Path(__file__)), "paired_statistics_sha256": sha(root / "paired_statistics.csv"),
        "artifacts": {str(path.relative_to(base)): sha(path) for path in artifacts}}
    with (root / "report_artifacts_receipt.json").open("x") as handle:json.dump(receipt, handle, indent=2)
    print(json.dumps(receipt))


if __name__ == "__main__":main()
