"""Full German ZZ-phase follow-up, using the frozen candidate/selection APIs."""
import json

import numpy as np
import pandas as pd

from .full_data import write_json
from .improved import sha
from .quantum_ten_reporting import collect, select, evaluate, paired
from build_controlled_results import holm


CONTRASTS = {
    "quantum_within": "동일 ZZ 무주입 대비",
    "quantum_vs_classical": "검증 선택 최선 고전 대비",
    "quantum_vs_separable": "얽힘 제거 대비",
    "quantum_vs_shuffled": "위험 순위 셔플 대비",
    "quantum_minus_rbf_gain": "양자 위험 이득 - RBF 위험 이득",
    "rbf_within": "RBF 위험형의 무주입 대비",
}


def seed_differences(raw):
    rows = []
    for dataset, group in raw[raw.candidate_method.eq("zz_phase")].groupby("dataset"):
        models = {role: part.set_index("seed") for role, part in group.groupby("role")}
        proposed = models["proposed"]
        for role in ("baseline", "classical_best", "separable", "shuffled"):
            control = models[role].reindex(proposed.index)
            for seed in proposed.index:
                before, after = int(control.loc[seed, "cost"]), int(proposed.loc[seed, "cost"])
                delta = before - after
                rows.append({"dataset": dataset, "seed": seed, "control": role,
                    "control_cost": before, "proposed_cost": after,
                    "paired_cost_difference": delta,
                    "relative_cost_reduction_pct": 100 * delta / before if before else np.nan,
                    "direction": "개선" if delta > 0 else "악화" if delta < 0 else "동률",
                    "selected_lambda": proposed.loc[seed, "lambda"]})
    return pd.DataFrame(rows)


def report(output, tasks, hashes, protocol):
    groups, receipts, accounting = collect(output, tasks, hashes)
    choices, profiles = select(groups, protocol["datasets"], protocol["seeds"],
                               protocol["methods"], protocol["lambdas"])
    lock = output / "selection_lock.json"
    if lock.exists():
        raise AssertionError("Preserve prior selection lock; no automatic retry")
    write_json(lock, {"models": choices, "lambda_profiles": profiles,
        "candidate_receipts_sha256": receipts, "source_sha256": hashes,
        "rule": "Validation 5FN+FP only; commit every choice before reading test labels"})
    raw, predictions = evaluate(output, choices, True, protocol)
    profile, _ = evaluate(output, profiles, True, protocol, with_predictions=False)
    stats = paired(raw)
    directions = seed_differences(raw)
    raw.to_csv(output / "seed_metrics.csv", index=False)
    profile.to_csv(output / "lambda_seed_metrics.csv", index=False)
    stats.to_csv(output / "paired_cost.csv", index=False)
    directions.to_csv(output / "seed_directions.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(accounting).to_csv(output / "compute_accounting.csv", index=False)
    pd.DataFrame([r for part in groups.values() for r in part]).to_csv(output / "all_tune_trials.csv", index=False)
    raw.groupby(["dataset", "candidate_method", "role"])[["cost", "fn", "fp", "f1", "recall", "balanced_accuracy"]].agg(["mean", "std"]).to_csv(output / "summary_metrics.csv")
    raw[raw.candidate_method.eq("standard_controls")].to_csv(output / "standard_reference_models.csv", index=False)
    settings = [{"dataset": r["dataset"], "seed": r["seed"], "method": r["candidate_method"],
        "role": r["role"], "lambda": r["lambda"], "details": r.get("representation_details", {})} for r in choices]
    write_json(output / "quantum_settings.json", {"selected_model_settings": settings,
        "simulator": "7-qubit CPU Aer double exact statevector; no QPU/shots/noise/GPU acceleration"})
    profile_stats = []
    for strength in protocol["lambdas"]:
        proposed = profile[profile["lambda"].eq(strength)].copy()
        proposed["role"] = "proposed"
        controls = raw[raw.role.ne("proposed") & raw.candidate_method.eq("zz_phase")]
        table = paired(pd.concat([proposed, controls], ignore_index=True))
        table["lambda"] = strength
        profile_stats.append(table)
    ps = pd.concat(profile_stats, ignore_index=True)
    ps["p_holm"] = holm(ps.p_value.fillna(1).to_numpy())
    ps.loc[ps.p_value.isna(), "p_holm"] = np.nan
    ps["holm_family_size"] = len(ps)
    ps.to_csv(output / "lambda_paired_cost.csv", index=False)
    main = raw[raw.candidate_method.eq("zz_phase")]
    times = [c for c in raw.columns if c.endswith("seconds") or c.endswith("kernel_entries")]
    main[main.learner.isin(["QSVC", "RBF-SVM"])].groupby(["dataset", "role"])[times].agg(["mean", "std"]).to_csv(output / "kernel_runtime_summary.csv")
    comparisons = []
    for (dataset, control), part in directions.groupby(["dataset", "control"]):
        comparisons.append({"dataset": dataset, "control": control,
            "mean_relative_cost_reduction_pct": part.relative_cost_reduction_pct.mean(),
            "mean_control_cost": part.control_cost.mean(), "mean_proposed_cost": part.proposed_cost.mean()})
    pd.DataFrame(comparisons).to_csv(output / "relative_cost_summary.csv", index=False)
    lines = ["# German 계열 ZZ 위험 위상 후속 실험", "", "## Material Passport", "",
        "상태: ANALYZED. 저장된 입력·선택·점수 검증이며 독립적인 종단간 재학습 재현은 아니다.", "",
        "German Numeric과 South German, 공통10시드, 각1000행 전체의 800/100/100 분할이다. 각 후보 SVM은 학습800행 모두에 단 한 번 적합했다. 양수 lambda는0.05~1.00의0.05간격이며 무주입0은 별도 기준이다. lambda/C/임계값/고전 대조는 검증100행의5FN+FP로 선택하고 잠근 뒤 테스트100행에 적용했다.", "",
        "이전 스모크의 승격 기준을 통과한 실행이 아니라 사용자 요청에 따른 별도 탐색 후속이다. 이전 결과를 본 설계이며 시드와 행이 겹치므로 미사용 확증 테스트가 아니다. 두 German 자료는 같은 원자료 계열이다.", "",
        "## 비용과 짝비교", "",
        "Δ=대조 비용-제안 비용으로 양수가 개선이다. 95%구간과 원p는 반복 분할 중복을 근사 보정한 짝비교 t 방식이다. 주12개 비교에 Holm을 적용한다. 원p도 함께 보존하며 보정이 실제 비용을 바꾸지 않는다.", "",
        "| 자료 | 비교 | 비용 Δ | 95% 구간 | 원 p | Holm p | 개선/동률/악화 |",
        "| --- | --- | ---: | --- | ---: | ---: | --- |"]
    for row in stats.itertuples():
        lines.append(f"| {row.dataset} | {CONTRASTS[row.contrast]} | {row.mean_difference:.2f} | [{row.ci_low:.2f}, {row.ci_high:.2f}] | {row.p_value:.4f} | {row.p_holm:.4f} | {row.wins}/{row.ties}/{row.losses} |")
    lines += ["", "## 회로와 해석", "",
        "7큐비트, ZZ feature map reps2/linear/alpha0.5 뒤 LR 위험 순위에 따른 국소RZ를 추가한다. 입력 특성은 위험점수로 곱하지 않으며 위험 조건ZZ 결합이나 추가 차원이 아니다. 동일 reps2 무주입과 비교하고, 별도의 이전 reps1 여섯 모델 참조와 혼동하지 않는다.", "",
        "CPU Aer double 정확 상태벡터이며 A100 GPU 가속, 실제 QPU, 유한 샷, 잡음, 계산 복잡도 우위를 측정하지 않았다. 고전 모델은 같은 PCA7/LR 정보를 쓰되 더 넓게 탐색하며 원래 특성 전체를 사용하는 최강 고전 모델의 검증은 아니다. 학습형 RBF 정렬은 학습128행/Powell24평가 예산이며 수렴 보장이 아니다.", "",
        "모든 양수 lambda별 결과는 lambda_seed_metrics.csv와 lambda_paired_cost.csv에 있다. 테스트 비용이 좋은 lambda를 사후 선택한 최종 모델로 보고하지 않는다. 고정 lambda 프로파일240개 비교는 별도 탐색 Holm군이다.", "",
        "F1/재현율/균형 정확도/FN/FP는 seed_metrics.csv에, 시간은 kernel_runtime_summary.csv에, 회로·깊이·게이트 수는 quantum_settings.json에 있다. 커널 시간은 행렬/상태 준비/적합/추론으로 구분한다. 공유 전처리 캐시 시간은 새 벽시계 실행 시간과 구분한다. RF/XGBoost 원본 추론 시간은 검증+테스트 합이므로 커널 모델의 테스트 단독 시간과 직접 비교하지 않는다.", "",
        "집계 반전·개인 일반화·선택 편향·충돌변수·기저율·평균 회귀·완료 선택·다중 탐색·사후 경로·인과 과장·역인과의11개 항목을 해석 범위에서 점검했다. 편향 제거·인과 효과·양자 고유 이점을 인증하지 않는다."]
    (output / "결과보고서.md").write_text("\n".join(lines) + "\n")
    files = ["selection_lock.json", "seed_metrics.csv", "lambda_seed_metrics.csv", "paired_cost.csv",
        "seed_directions.csv", "predictions.csv.gz", "compute_accounting.csv", "all_tune_trials.csv",
        "summary_metrics.csv", "standard_reference_models.csv", "quantum_settings.json", "lambda_paired_cost.csv",
        "kernel_runtime_summary.csv", "relative_cost_summary.csv", "결과보고서.md"]
    receipt = {"scope": "Complete candidates, validation-only locks and same-fit stored metrics; not independent retraining",
        "jobs": len(tasks), "models": len(choices), "lambda_models": len(profiles),
        "candidate_fits": sum(a["fits"] for a in accounting),
        "outputs_sha256": {name: sha(output / name) for name in files}}
    write_json(output / "verification_receipt.json", receipt)
    return receipt
