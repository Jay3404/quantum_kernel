"""Locked varying-test evaluation and approximate corrected resampled inference."""
import itertools
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from . import repeated_holdout as p
from . import overnight_reporting as common
from . import overnight_risk as old
from build_controlled_results import holm


def selected_choices(groups, dimension):
    choices = []
    for job, saved in groups.items():
        identity = dict(dataset=job[0], seed=job[1], kernel=job[2], family=job[3], arm=job[4], job=list(job))
        if job[3] == "reference":
            for learner, representation in itertools.product(("LR", "RF", "XGBoost"), ("raw", f"pca{dimension}")):
                row = min((r for r in saved["trials"] if r["learner"] == learner and r["method"] == representation),
                    key=lambda r: (r["tune_cost"], str(r.get("depth"))))
                choices.append({**row, **identity, "role": f"reference_{representation}_{learner}",
                    "score_source": "margins", "score_key": row["score_index"]})
        else:
            for role, row in saved["choices"].items():
                if role == "baseline" and job[4] != "normal":
                    continue
                choices.append({**row, **identity, "role": role, "score_source": "exact_margins", "score_key": role})
    return common.append_classical_best(choices)


def evaluate(output, choices):
    assert (output / "selection_lock.json").exists() and (output / "test_release.json").exists()

    @lru_cache(maxsize=2)
    def inputs(dataset, seed):
        arrays, record = old.load_inputs(output, dataset, seed)
        y = p.base.load_data(dataset, str(p.ROOT / "data"))[1]
        split = json.loads((old.prepared_folder(output, dataset, seed) / "split.json").read_text())
        assert split["actual"] == split["outer"] == p.split_indices(y, seed)
        return arrays, y[np.asarray(split["actual"]["test"])], np.asarray(split["actual"]["test"])

    @lru_cache(maxsize=4)
    def archive(path):
        with np.load(path, allow_pickle=False) as data:
            return dict(data)

    rows, predictions = [], []
    for choice in choices:
        arrays, labels, ids = inputs(choice["dataset"], choice["seed"])
        path = old.job_folder(output, choice["job"]) / (choice["score_source"] + ".npz")
        saved = archive(str(path))
        key = choice["score_key"]
        if choice["score_source"] == "margins":
            tune, scores = saved["tune"][key], saved["gate"][key]
        else:
            tune, scores = saved[key + "_tune"], saved[key + "_gate"]
        assert p.base.metrics(arrays["y_tune"], tune, choice["threshold"])["cost"] == choice["tune_cost"]
        assert choice["fit_rows"] == len(arrays["train"])
        row = {k: v for k, v in choice.items() if not isinstance(v, (dict, list))}
        row.update(p.base.metrics(labels, scores, choice["threshold"]), n_train=len(arrays["train"]),
            n_validation=len(arrays["tune"]), n_eval=len(labels), pca_components=arrays["train"].shape[1])
        for ratio, threshold in choice.get("sensitivity_thresholds", {}).items():
            metric = p.base.metrics(labels, scores, threshold)
            row[f"cost_ratio_{ratio}"] = int(ratio) * metric["fn"] + metric["fp"]
            row[f"threshold_ratio_{ratio}"] = threshold
        rows.append(row)
        predictions.append(pd.DataFrame({**{k: choice[k] for k in ("dataset", "seed", "kernel", "family", "arm", "role")},
            "row_id": ids, "y": labels, "score": scores, "threshold": choice["threshold"],
            "prediction": (scores >= choice["threshold"]).astype(int)}))
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def corrected_summary(values, ratio):
    values = np.asarray(values, dtype=float)
    assert len(values) == 10 and np.isfinite(values).all()
    variance = float(values.var(ddof=1))
    mean = float(values.mean())
    se = np.sqrt((1 / len(values) + ratio) * variance)
    half = stats.t.ppf(.975, len(values) - 1) * se if se else np.nan
    return {"n": len(values), "mean_difference": mean, "sd_difference": np.sqrt(variance),
        "corrected_se": se, "ci_low": mean - half, "ci_high": mean + half,
        "p_value": float(2 * stats.t.sf(abs(mean / se), len(values) - 1)) if se else 1.,
        "degenerate_variance": not bool(se), "test_train_ratio": ratio,
        "variance_multiplier": 1 / len(values) + ratio, "df": len(values) - 1,
        "wins": int((values > 0).sum()), "ties": int((values == 0).sum()), "losses": int((values < 0).sum()),
        "inference": "Approximate corrected resampled t; overlapping repetitions are not independent test samples"}


def contrasts(raw, protocol):
    rows, directions = [], []
    seeds = protocol["seeds"]

    def models(dataset, kernel, family, arm, role):
        frame = raw[raw.dataset.eq(dataset) & raw.family.eq(family) & raw.arm.eq(arm) & raw.role.eq(role)]
        if kernel is not None:
            frame = frame[frame.kernel.eq(kernel)]
        frame = frame.set_index("seed").reindex(seeds)
        assert not frame.cost.isna().any()
        return frame

    def add(dataset, learner, family, contrast, delta, proposed):
        ratio = float((proposed.n_eval / proposed.n_train).max())
        rows.append(dict(dataset=dataset, kernel=learner, family=family, contrast=contrast,
            primary_source=dataset != "german_numeric", complete_ten=True,
            **corrected_summary(delta, ratio)))
        directions.extend(dict(dataset=dataset, kernel=learner, family=family, contrast=contrast,
            seed=seed, delta=float(value), direction="improvement" if value > 0 else "worse" if value < 0 else "tie")
            for seed, value in zip(seeds, delta))

    for dataset, learner, family in itertools.product(protocol["datasets"], protocol["learners"], protocol["methods"]):
        proposed = models(dataset, learner, family, "normal", "proposed")
        controls = {"own_no_risk": models(dataset, learner, family, "normal", "baseline"),
            "tuned_baseline": models(dataset, learner, "baseline", "normal", "baseline"),
            "best_classical": models(dataset, None, "reference_best", "normal", "classical_best")}
        for arm in ("shuffled", "constant", "separable"):
            if arm == "constant" and family == "pls_metric" or arm == "separable" and learner != "QSVC":
                continue
            controls[arm] = models(dataset, learner, family, arm, "proposed")
        for name, control in controls.items():
            delta = (control.cost - proposed.cost).to_numpy()
            add(dataset, learner, family, name, delta, proposed)
            rows[-1]["mean_seed_rcr_pct"] = float((100 * (control.cost - proposed.cost) / control.cost).mean()) if control.cost.gt(0).all() else np.nan
            rows[-1]["ratio_of_means_rcr_pct"] = 100 * float(control.cost.mean() - proposed.cost.mean()) / control.cost.mean() if control.cost.mean() else np.nan
    for dataset, family in itertools.product(protocol["datasets"], protocol["methods"]):
        gain = {}
        for learner in protocol["learners"]:
            proposed = models(dataset, learner, family, "normal", "proposed")
            gain[learner] = models(dataset, learner, family, "normal", "baseline").cost - proposed.cost
        add(dataset, "interaction", family, "quantum_minus_rbf_own_zero_gain",
            (gain["QSVC"] - gain["RBF-SVM"]).to_numpy(), proposed)
    frame = pd.DataFrame(rows)
    assert len(frame) == 46 * len(protocol["datasets"])
    frame["p_holm"] = holm(frame.p_value.to_numpy())
    frame["holm_family_size"] = len(frame)
    return frame, pd.DataFrame(directions)


def finalize(root, meta):
    output, protocol = root / "full", meta["protocol"]
    groups, receipts, missing = common.collect(output, p.jobs(protocol), meta["source_sha256"])
    assert not missing and len(groups) == meta["expected_jobs"]
    choices = common.sensitivity_choices(output, selected_choices(groups, protocol["pca_components"]))
    profiles = common.sensitivity_choices(output, common.lambda_profiles(groups, protocol))
    assert not (output / "selection_lock.json").exists() and not (output / "test_release.json").exists()
    p.write_json(output / "selection_lock.json", {"choices": choices, "lambda_profiles": profiles,
        "candidate_receipts_sha256": receipts, "source_sha256": meta["source_sha256"], "missing_jobs": [],
        "rule": "All dataset/seed choices globally frozen using validation only; no test-adaptive method selection"})
    p.write_json(output / "test_release.json", {"selection_lock_sha256": p.sha(output / "selection_lock.json"),
        "released_utc": p.now(), "no_further_fitting": True, "test_customers_vary_between_seeds": True})
    raw, predictions = evaluate(output, choices)
    raw.to_csv(output / "seed_metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    raw.groupby(["dataset", "kernel", "family", "arm", "role"])[["cost", "fn", "fp", "f1", "recall", "balanced_accuracy"]].agg(["mean", "std"]).to_csv(output / "summary_metrics.csv")
    paired, directions = contrasts(raw, protocol)
    paired.to_csv(output / "paired_cost.csv", index=False)
    directions.to_csv(output / "seed_directions.csv", index=False)
    profiles_raw, _ = evaluate(output, profiles)
    profiles_raw.to_csv(output / "lambda_seed_metrics.csv", index=False)
    timing = [k for k in ("training_seconds", "test_inference_seconds", "train_kernel_seconds", "test_kernel_seconds", "fit_seconds", "exact_fit_seconds") if k in raw]
    raw.groupby(["dataset", "kernel", "family", "arm", "role"])[timing].agg(["mean", "std"]).to_csv(output / "runtime_summary.csv")
    pd.DataFrame([{"job": list(job), "elapsed_seconds": saved["elapsed_seconds"], "peak_gib": saved["peak_gib"],
        "pid": saved["worker_pid"], "execution_origin": saved["execution_origin"]} for job, saved in groups.items()]).to_csv(output / "runtime_execution_provenance.csv", index=False)
    p.write_json(output / "quantum_settings.json", {"pca_components": protocol["pca_components"],
        "backend": "CPU Aer exact double7/8-qubit states; locked GPU Gram only",
        "models": [{k: row[k] for k in ("dataset", "seed", "kernel", "family", "arm", "role")} |
            {"circuits": row.get("representation_details", {}).get("circuits", [])} for row in choices]})
    lines = ["# PCA7 전체 재분할 반복 실험", "", "시드10,20,...,100마다 학습80%·검증10%·테스트10%를 모두 다시 분할했다. 모든 방법은 같은 시드의 같은 고객 분할을 사용한다. 전처리·PCA·LR/XGB 위험 추정부터 새로 학습했고 이전 점수나 모델을 상속하지 않았다.", "",
        "기존에 탐색한 자료와 시드의 재사용이므로 독립 외부 확인 실험이 아니다. German 두 표현은 같은 원자료 계열이다. PLS는 지도 표현 대조이며, 원특성 XGB와 PCA-LR는 교사와 입력이 동시에 달라진다.", "",
        "비용=5FN+FP. 검증 선택과 최종 테스트를 구분한다. 모든 계획 작업과 임계값을 잠근 뒤 테스트를 공개했다. 큰 자료는512랜드마크 근사 탐색 후 전체 학습80% 단일 정확SVC와 자기 검증 임계값을 사용했다. 예측 앙상블은 없다.", "",
        "반복 분할의 의존성을 반영한 근사 corrected resampled t(df9, 분산계수1/10+1/8)와 전체184대비 Holm을 적용했다. 시드를 독립 고객으로 세지 않는다. 이 근사 검정은 학습자료 중첩과 반복적인 연구 선택 편향을 완전히 제거하지 않는다. 상대 개선폭만으로 양자 고유 이점을 주장하지 않는다.", "",
        "| 자료 | 커널 | 방법 | 평균 비용 감소 | 95% 구간 | 원p | Holm p | 개선/동률/악화 |",
        "|---|---|---|---:|---|---:|---:|---|"]
    for row in paired[paired.contrast.eq("tuned_baseline")].itertuples():
        lines.append(f"| {row.dataset} | {row.kernel} | {row.family} | {row.mean_difference:.2f} | [{row.ci_low:.2f}, {row.ci_high:.2f}] | {row.p_value:.4f} | {row.p_holm:.4f} | {row.wins}/{row.ties}/{row.losses} |")
    lines += ["", "lambda 곡선은 기술통계이며 큰 자료의 곡선은 근사 탐색 결과다. 비용비율2/10은5:1로 선택한 모델을 고정한 임계값 민감도다. 전체 작업 경과시간과 캐시 재사용/최종 적합 시간, 준비 단계 교사 학습시간은 분리 기록했다.", "",
        "통계 근거: " + protocol["statistics_source"]]
    (output / "결과보고서.md").write_text("\n".join(lines) + "\n")
    files = [file for file in output.iterdir() if file.is_file() and file.name != "verification_receipt.json"]
    p.write_json(output / "verification_receipt.json", {"planned_jobs": len(groups), "selected_rows": len(raw),
        "outputs_sha256": {file.name: p.sha(file) for file in files}})
