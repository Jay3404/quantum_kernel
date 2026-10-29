"""Korean report for all completed second-feedback controls, never partial-as-full."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from _shared import professor_followup as p

DN = {"german_numeric": "German Numeric", "south_german": "South German", "hmeq": "HMEQ", "taiwan": "Taiwan"}
MN = {"gbdt_soft_gate": "Soft Gate", "gbdt_rank_append": "Rank Append"}


def ms(series, digits=1):
    return f"{series.mean():,.{digits}f} ± {series.std(ddof=1):,.{digits}f}"


def report(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p.verify_sources(root)
    assert (root / "test_release.json").exists()
    cal = pd.read_csv(root / "calibration_metrics.csv")
    prob = pd.read_csv(root / "calibration_probability_metrics.csv")
    raw = pd.read_csv(root / "parent_seed_metrics_copy.csv")
    new = pd.read_csv(root / "new_seed_metrics.csv")
    paired = pd.read_csv(root / "paired_statistics.csv")
    geometry = pd.read_csv(root / "mechanism_seed_metrics.csv")
    correlation = pd.read_csv(root / "mechanism_correlations.csv")
    assert len(paired) == 140 and len(new) == 440 and len(geometry) == 80
    lines = ["# 교수님 2차 피드백: 전체 추가 실험 결과", "", "## Material Passport", "",
        "상태: ANALYZED. 고정 PCA7의 동일 분할/OOF 점수를 이용한 보정 실험과200개 새 대조 작업을 수행했다. 이미 본 테스트·자료·시드를 이용한 탐색적 후속 분석이며 외부 독립 확인이나 전체 파이프라인 독립 재현은 아니다.",
        "기존 PCA7/PCA9 소스와 결과는 읽기 전용으로 유지했다. 무작위120작업(2자료×10시드×2커널×3난수반복)과 동일기하 backbone80작업을 모두 완료한 뒤 전역 잠금하고 테스트를 평가했다. 3난수반복은 시드 내부에서 평균내고 추론의 n은10이다.", "",
        "## 1. XGB 단독과 보정", "",
        "XGB160-tree/depth3 위험교사의 가중치와 점수는 그대로 유지했다. sigmoid/isotonic 보정기는 학습5OOF 점수/라벨에만 적합해 검증/테스트에 적용했다. 보정된 훈련 점수를 OOF라고 부르거나 새 커널에 주입하지 않았다. 모델 자체를 바꾸는 효과와 보정 효과를 혼동하지 않는다.",
        "각 모드의 검증 비용최적 임계값으로 평가했다. 아래 값은10시드 평균 ± 표본표준편차, AUROC/AP는 높을수록 나머지 확률 지표와 비용은 낮을수록 좋다.", "",
        "| 자료 | 보정 | 비용 | AUROC | AP | Brier | Log loss | ECE10 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in p.protocol()["datasets"]:
        for method in p.protocol()["calibration"]["methods"]:
            costs = cal[cal.dataset.eq(dataset) & cal.calibration.eq(method) & cal.part.eq("test") & cal.threshold_policy.eq("validation_cost")]
            values = prob[prob.dataset.eq(dataset) & prob.calibration.eq(method)]
            assert len(costs) == len(values) == 10
            lines.append(f"| {DN[dataset]} | {method} | {ms(costs.cost)} | " + " | ".join(ms(values[c], 3) for c in ("auroc", "ap", "brier", "log_loss", "ece10_uniform")) + " |")
    lines += ["", "### 강한 XGB 참조와 비교", "",
        "아래 XGB 참조는 별도로 검증 튜닝한200-tree/depth3·6 모델이며 AUROC.929의 고정 위험교사와 다르다. 본문에서 반드시 구분한다. 고전 참조가 더 좋은 결과도 숨기지 않는다.", "",
        "| 자료 | XGB 참조 | RBF Soft Gate | RBF Rank Append | QSVC Soft Gate | QSVC Rank Append |", "|---|---:|---:|---:|---:|---:|"]
    for dataset in p.protocol()["datasets"]:
        cells = [ms(p.vector(raw, dataset=dataset, role="reference_raw_XGBoost").cost)]
        cells += [ms(p.parent_model(raw, dataset, kernel, method).cost) for kernel in p.protocol()["learners"] for method in p.protocol()["methods"]]
        lines.append(f"| {DN[dataset]} | " + " | ".join(cells) + " |")
    lines += ["", "확률.5, 비용비 기준1/6, 검증 비용최적, 순위 기반 동률 처리의4정책을 모두 저장했다. 단조 sigmoid는 순위를 보존하므로 확률 보정이 좋아져도 순위 기반 정책이나 비용이 반드시 개선되지는 않는다. 동률 처리로 달라진 결정과 순위 변화도 구분한다.", ""]
    lines += ["## 2. 커널별 짝 개선량", "",
        "각 시드에서 (QSVC기본형-QSVC위험형)-(RBF기본형-RBF위험형)을 계산했다. 양수는 QSVC 감소량이 더 큼을 뜻하며 최종 절대 비용 우위와 다르다. 개별95% 근사구간과 새 전체100대비 Holm을 함께 표시한다."]
    interaction = paired[paired.contrast.eq("QSVC_minus_RBF_tuned_baseline_gain")]
    lines += stat_table(interaction)
    lines += ["", "## 3. 무작위8큐비트/8좌표 대조", "",
        "동일7차원 backbone에 독립 Uniform(0,1) 좌표 하나를 추가했다. QSVC는 독립 보조Ry1개로8큐비트, RBF는8좌표다. 같은 난수값을 두 커널에 적용했다. 실제위험·난수·위험순위셔플·상수 대조를 구분한다. tuned는 같은 양수20lambda×3C×3기하로 검증 선정, matched는 실제위험의 lambda/C/기하를 고정하고 자체 검증 임계값만 정했다. 모든 큰 자료는512-landmark 검색 후 전체행 단일 정확SVC다.", "",
        "| 자료 | 커널 | 실제위험 비용 | 난수 tuned 비용 | 난수 matched 비용 |", "|---|---|---:|---:|---:|"]
    for dataset in p.protocol()["random_datasets"]:
        for kernel in p.protocol()["learners"]:
            selected = new[new["mode"].eq("random") & new.dataset.eq(dataset) & new.kernel.eq(kernel)]
            means = {role: selected[selected.role.eq(role)].groupby("seed").cost.mean() for role in ("tuned", "matched")}
            lines.append(f"| {DN[dataset]} | {kernel} | {ms(p.parent_model(raw, dataset, kernel, 'gbdt_rank_append').cost)} | {ms(means['tuned'])} | {ms(means['matched'])} |")
    lines += stat_table(paired[paired.contrast.str.startswith("random_") | paired.contrast.eq("baseline_minus_random_tuned")])
    lines += ["", "## 4. OOF·통계 범위", "",
        "기존40개 분할의 학습/검증/테스트 및5OOF/내부3fold LR보정 기록을 다시 확인했다. OOF는학습80%내부이며 검증10%/테스트10%를 제외한다. 전체nestedCV가 아니라 holdout모델선정+학습내부OOF다. 보정기는학습OOF만 사용하며 훈련 보정값 자체의 교차적합을 주장하지 않는다.",
        "기존184가족의 Holm값은 그대로 유지했다. 새 비용 관련100대비에 Holm100, 별도 확률지표40대비에 Holm40을 적용했다. corrected resampled t의df9·분산계수.225이며 표준오차는sqrt(.225)×짝차이SD다.95%구간은 개별 구간이지 동시구간이 아니다.0분산은p1/구간정의불가다. 이미 본 테스트에 대한 탐색적 분석이라 새 확인적 유의성으로 주장하지 않는다.", "",
        "## 5. 표현 변화와 비용 감소", "",
        "동일C/기하의 위험 없는 backbone을 전체 학습행으로 새로 적합하고 그 모델의 자체 검증 임계값을 사용했다. 실제위험과 같은256명 테스트 부분 Gram에서 정렬도/entropy effective rank를 비교했다. 전체 테스트 비용과 같은256명 부분 비용의 감소를 각각 연결해10시드 Spearman rho만 기술한다. 겹치는 분할의 독립성이나 인과 메커니즘·유효순위의 단조 우수성을 주장하지 않는다.", "",
        "| 자료 | 커널 | 방법 | 표현 변화 | 비용 범위 | Spearman rho |", "|---|---|---|---|---|---:|"]
    for row in correlation.itertuples():
        lines.append(f"| {DN[row.dataset]} | {row.kernel} | {MN[row.method]} | {row.x} | {row.y} | {row.spearman_rho:.3f} |")
    assets = p.ROOT / "docs/assets"
    assets.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for i, dataset in enumerate(p.protocol()["random_datasets"]):
        for j, kernel in enumerate(p.protocol()["learners"]):
            ax = axes[i, j]
            for method, color in zip(p.protocol()["methods"], ("#137f78", "#b84a55")):
                values = geometry[geometry.dataset.eq(dataset) & geometry.kernel.eq(kernel) & geometry.method.eq(method)]
                ax.scatter(values.alignment_delta, values.full_cost_reduction, label=MN[method], color=color, s=25)
            ax.axhline(0, color="#888888", lw=.7)
            ax.axvline(0, color="#888888", lw=.7)
            ax.set_title(f"{DN[dataset]} | {kernel}")
            ax.set_xlabel("Centered alignment change")
            ax.set_ylabel("Matched-backbone cost minus risk cost")
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    image = assets / "cmc_professor_followup_mechanism_20261022.png"
    fig.savefig(image, dpi=150)
    plt.close(fig)
    lines += ["", "![정렬도 변화와 비용 감소](assets/cmc_professor_followup_mechanism_20261022.png)", "",
        "## 전체 사전 대비", "", "유리하거나 불리한 결과를 모두 표시한다. 양수는contrast에 정의된 개선 방향이다."]
    lines += stat_table(paired)
    lines += ["", "## 선택 모델 실행시간", "",
        "부모 실험의 선택 모델 학습/테스트 추론 기록 평균(초)이다. 전체 격자 탐색이나 OOF교사 학습을 포함한 비용이 아니며, 캐시·CPU/GPU 혼합 실행 때문에 속도 우위의 공정한 벤치마크로 해석하지 않는다.", "",
        "| 자료 | 커널 | 방법 | 선택 학습(s) | 테스트 추론(s) |", "|---|---|---|---:|---:|"]
    for dataset in p.protocol()["datasets"]:
        for kernel in p.protocol()["learners"]:
            for method in p.protocol()["methods"]:
                values = p.parent_model(raw, dataset, kernel, method)
                lines.append(f"| {DN[dataset]} | {kernel} | {MN[method]} | {values.training_seconds.mean():.4f} | {values.test_inference_seconds.mean():.4f} |")
    lines += ["", "## 해석과 한계", "",
        "Quantum 결합의 실질적 이유는 강한XGB 단독 대비 비용·안정성·계산비용까지 비교해 판단해야 한다. 약한 기본형 대비 상대개선만으로 필요성이나양자우위를 주장하지 않는다. 두 독일 표현은 독립 원자료 둘이 아니다. 같은 자료/시드/이미본 결과 재사용, 선택편향, 겹친고객, 소표본10반복, 큰자료 근사검색이라는 한계가 남는다.",
        "통계 오류11유형 점검: 자료/커널별 분리로Simpson집계왜곡을 제한했고, 개인인과추론을 하지 않았다. Berkson/collider/자료수집선택은자료생성과정미확보로미확인이다. 기저율을확률지표에보존하고,극단HMEQ선택/평균회귀는사후탐색한계로남겼다. 전체200작업/40보정쌍을유지하며생존자편향을피하고,전체100/40가족보정과forking-paths공개를했다. 상관·역인과·개인인과·양자고유메커니즘은입증하지않았다. 점검은편향부재의인증이아니다.",
        "확률 보정과 임계값 선정은 다른 절차다. Brier/log loss는보정만의지표가아니며ECE는bin정의에민감하다. 새SVM의전체행·단일모델·독립보조큐비트·테스트차단과최종감사범위는validation_receipt에기록한다.", ""]
    path = p.ROOT / "docs/cmc_professor_followup_final_20261022_ko.md"
    assert not path.exists(), "Preserve delivered reports"
    path.write_text("\n".join(lines) + "\n")
    p.write(root / "report_receipt.json", {"report": str(path.relative_to(p.ROOT)), "sha256": p.sha(path),
        "figure": str(image.relative_to(p.ROOT)), "figure_sha256": p.sha(image), "primary_cost_contrasts": 100,
        "secondary_calibration_contrasts": 40, "scientific_jobs": 200, "calibration_pairs": 40}, True)


def stat_table(frame):
    lines = ["", "| 자료 | 커널 | 방법 | 대비 | 평균차 | 근사95% CI | 원p | Holm p | 개선/동률/악화 |",
        "|---|---|---|---|---:|---|---:|---:|---|"]
    for row in frame.itertuples():
        lines.append(f"| {DN[row.dataset]} | {row.kernel} | {MN.get(row.method, row.method)} | {row.contrast} | {row.mean_difference:.3f} | [{row.ci_low:.3f}, {row.ci_high:.3f}] | {row.p_value:.6f} | {row.p_holm:.6f} | {row.wins}/{row.ties}/{row.losses} |")
    return lines
