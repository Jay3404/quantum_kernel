"""Globally locked PCA9 evaluation and frozen 7/9 comparison families."""
import json
import pandas as pd

from . import repeated_holdout_pca9 as p
from . import overnight_reporting as common
from .repeated_reporting import selected_choices, evaluate, contrasts, corrected_summary
from .repeated_dimension_comparison import compare


def finalize(root, meta):
    output, protocol = root / "full", meta["protocol"]
    groups, receipts, missing = common.collect(output, p.jobs(protocol), meta["source_sha256"])
    assert not missing and len(groups) == meta["expected_jobs"] == 870
    choices = common.sensitivity_choices(output, selected_choices(groups, 9))
    profiles = common.sensitivity_choices(output, common.lambda_profiles(groups, protocol))
    assert not (output / "selection_lock.json").exists() and not (output / "test_release.json").exists()
    p.write_json(output / "selection_lock.json", {"choices": choices, "lambda_profiles": profiles,
        "candidate_receipts_sha256": receipts, "source_sha256": meta["source_sha256"], "missing_jobs": [],
        "within_pca9_holm_family_size": 138, "dimension_holm_family_size": 90,
        "rule": "All870 planned PCA9 models/validation thresholds frozen before test release; no test-adaptive selection"})
    p.write_json(output / "test_release.json", {"selection_lock_sha256": p.sha(output / "selection_lock.json"),
        "released_utc": p.now(), "no_further_fitting": True, "same_saved_varying_test_splits_as_pca7": True})
    raw, predictions = evaluate(output, choices)
    raw.to_csv(output / "seed_metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    raw.groupby(["dataset", "kernel", "family", "arm", "role"])[["cost", "fn", "fp", "f1", "recall", "balanced_accuracy"]].agg(["mean", "std"]).to_csv(output / "summary_metrics.csv")
    paired, directions = contrasts(raw, protocol)
    assert len(paired) == 138
    paired.to_csv(output / "paired_cost.csv", index=False)
    directions.to_csv(output / "seed_directions.csv", index=False)
    previous = pd.read_csv(p.UPSTREAM / "full/seed_metrics.csv", float_precision="round_trip")
    dimension, dimension_directions = compare(raw, previous, protocol)
    dimension.to_csv(output / "pca7_pca9_paired_cost.csv", index=False)
    dimension_directions.to_csv(output / "pca7_pca9_seed_directions.csv", index=False)
    profile_raw, _ = evaluate(output, profiles)
    profile_raw.to_csv(output / "lambda_seed_metrics.csv", index=False)
    timing = [k for k in ("training_seconds", "test_inference_seconds", "train_kernel_seconds", "test_kernel_seconds", "fit_seconds", "exact_fit_seconds") if k in raw]
    raw.groupby(["dataset", "kernel", "family", "arm", "role"])[timing].agg(["mean", "std"]).to_csv(output / "runtime_summary.csv")
    pd.DataFrame([{"job": list(job), "elapsed_seconds": saved["elapsed_seconds"], "peak_gib": saved["peak_gib"],
        "pid": saved["worker_pid"], "execution_origin": saved["execution_origin"]} for job, saved in groups.items()]).to_csv(output / "runtime_execution_provenance.csv", index=False)
    p.write_json(output / "quantum_settings.json", {"pca_components": 9,
        "backend": "CPU Aer exact double9/10-qubit states; locked GPU Gram only",
        "models": [{k: row[k] for k in ("dataset", "seed", "kernel", "family", "arm", "role")} |
            {"circuits": row.get("representation_details", {}).get("circuits", [])} for row in choices]})
    lines = ["# PCA9 전체 재분할 반복 실험", "",
        "PCA7과 같은 저장된 시드10~100의80/10/10 가변 테스트 고객 분할이다. 모든 PCA9 전처리·중첩 PCA·위험 교사를 새로 학습했다. 이전 점수·모델·탐색 prefix는 상속하지 않았다.", "",
        "German 두 표현은 관련 자료다. 기존 자료와 공개된 PCA7/이전 탐색의 분할을 재사용한 차원 민감도이며 독립 외부 확인 실험이 아니다. 모든870작업과 검증 임계값을 잠근 뒤 테스트를 공개했다. 전체 학습80% 단일 정확SVC, 예측 앙상블 없음.", "",
        "차원 내138대비와 별도 사전 고정7/9비교90대비에 각각 Holm을 적용했다. df9·분산계수.225의 근사 corrected resampled t다. 중첩 고객을 독립 표본으로 세지 않으며 보정이 연구 선택 편향을 없애지는 않는다.", "",
        "| 자료 | 커널 | 방법 | 기본형 대비 평균 비용차 | 근사95% CI | 원p | Holm p(138) | 개선/동률/악화 |",
        "|---|---|---|---:|---|---:|---:|---|"]
    for row in paired[paired.contrast.eq("tuned_baseline")].itertuples():
        lines.append(f"| {row.dataset} | {row.kernel} | {row.family} | {row.mean_difference:.2f} | [{row.ci_low:.2f},{row.ci_high:.2f}] | {row.p_value:.6f} | {row.p_holm:.6f} | {row.wins}/{row.ties}/{row.losses} |")
    lines += ["", "전체 학습행 정확SVC와512랜드마크 근사lambda 기술곡선을 구분한다.",
        "양자 회로는 CPU Aer9/10큐비트 exact double이며 GPU는 Gram 대수만 담당한다. 상대 개선폭이나 캐시 재사용 시간이 양자 고유 이점을 증명하지 않는다."]
    (output / "결과보고서.md").write_text("\n".join(lines) + "\n")
    files = [file for file in output.iterdir() if file.is_file() and file.name != "verification_receipt.json"]
    p.write_json(output / "verification_receipt.json", {"planned_jobs": len(groups), "selected_rows": len(raw),
        "outputs_sha256": {file.name: p.sha(file) for file in files}})
