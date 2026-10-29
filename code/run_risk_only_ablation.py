"""Exact risk-factor-only SVC component ablation, isolated from frozen runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import tarfile
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from analyze_pca7_professor_comments import corrected, holm, metrics, risk_factor, sha, threshold

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/cmc_risk_only_ablation_protocol_20261023.json"
NEW_SOURCES = ("code/run_risk_only_ablation.py", "code/tests/test_risk_only_ablation.py", "docs/cmc_risk_only_ablation_protocol_20261023.json")


def now():return datetime.now(timezone.utc).isoformat()


def read(path):return json.loads(Path(path).read_text())


def write(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = json.dumps(value, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n"
    if exclusive:
        with path.open("x") as handle:handle.write(value)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value)
        temporary.replace(path)


def configuration():return read(PROTOCOL)


def jobs(config=None):
    p = config or configuration()
    result = [(d, s, k, m) for d in p["datasets"] for s in p["seeds"] for k in p["kernels"] for m in p["methods"]]
    assert len(result) == p["expected_jobs"] == 160
    return result


def job_path(root, job):
    d, s, k, m = job
    return root / "models" / d / f"seed_{s:04d}" / k / m


def features(q, strength, kernel, method):
    q = np.asarray(q, dtype=float)
    assert q.ndim == 1 and np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
    assert 0 < strength <= 1
    if method == "gbdt_soft_gate":
        gate = np.column_stack([1 - q, q])
        gate /= np.linalg.norm(gate, axis=1, keepdims=True)
        return np.column_stack([np.full(len(q), np.sqrt(1 - strength)), np.sqrt(strength) * gate]), "linear"
    assert method == "gbdt_rank_append"
    angle = np.pi * strength * (q - .5)
    if kernel == "QSVC":
        return np.column_stack([np.ones(len(q)), np.cos(angle), np.sin(angle)]) / np.sqrt(2), "linear"
    assert kernel == "RBF-SVM"
    return angle[:, None], "rbf"


def peak_gib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 2**30 if sys.platform == "darwin" else value / 2**20


def available_gib():
    result = subprocess.run(["vm_stat"], text=True, capture_output=True, check=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", result).group(1))
    fields = {key.strip(): int(value) for key, value in re.findall(r"^([^:\n]+):\s+(\d+)\.", result, re.M)}
    return sum(fields[key] for key in ("Pages free", "Pages inactive", "Pages speculative")) * page / 2**30


def verify(root, inputs=False):
    meta = read(root / "metadata.json")
    for name, digest in meta["sources"].items():assert sha(ROOT / name) == digest, name
    if inputs:
        for name, digest in meta["inputs"].items():assert sha(ROOT / name) == digest, name
    return meta


def initialize(root):
    assert not root.exists(), "Preserve existing runs and failures; no automatic restart"
    p = configuration()
    analysis = ROOT / "code/results" / p["analysis_run"]
    assert read(analysis / "independent_validation_receipt.json")["status"] == "passed"
    bindings = read(analysis / "input_manifest.json")["sha256"]
    for name in ("geometry_seed_metrics.csv", "input_manifest.json", "independent_validation_receipt.json"):
        path = analysis / name
        bindings[str(path.relative_to(ROOT))] = sha(path)
    sources = read(ROOT / "code/results/cmc_professor_followup_20261022/metadata.json")["source_sha256"]
    sources = {**sources, **{name: sha(ROOT / name) for name in NEW_SOURCES}}
    parent = ROOT / "code/results" / p["parent_run"]
    plan = []
    for job in jobs(p):
        d, s, k, m = job
        file = parent / "full/candidates" / d / f"seed_{s:04d}" / k / m / "normal/complete.json"
        assert sha(file) == bindings[str(file.relative_to(ROOT))]
        choice = read(file)["choices"]["proposed"]
        plan.append({"job": job, **{key: choice[key] for key in ("C", "lambda", "geometry")}, "parent_complete_sha256": sha(file)})
    root.mkdir(parents=True)
    write(root / "metadata.json", {"run_id": p["run_id"], "status": "initialized", "utc": now(), "config": p,
        "sources": sources, "inputs": bindings, "jobs_completed": 0, "failures": [],
        "local_environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "logical_cpus": os.cpu_count(), "host_memory_gib": 64., "parent_sklearn": "1.7.2"}})
    write(root / "parameter_plan.json", {"utc": now(), "jobs": plan, "scope": "All160parent validation-selected parameters fixed before new SVC fits; no new search"})
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
        for name in sources:archive.add(ROOT / name, arcname=name)
    verify(root, True)


def load(root, job, probe=False):
    p = configuration()
    folder = ROOT / "code/results" / p["parent_run"] / "full/prepared" / job[0] / f"seed_{job[1]:04d}"
    meta = read(root / "metadata.json")
    for name in ("audit.json", "inputs.npz", "split.json", "oof.json"):
        file = folder / name
        assert sha(file) == meta["inputs"][str(file.relative_to(ROOT))]
    audit = read(folder / "audit.json")
    parts = ("train", "tune") if probe else ("train", "tune", "gate")
    with np.load(folder / "inputs.npz", allow_pickle=False) as saved:
        assert not {"y_test", "y_gate"} & set(saved.files)
        values = {part: saved["gbdt_rank_" + part] for part in parts}
        ytrain, ytune = saved["y_train"], saved["y_tune"]
        train = saved["train"]
    geometry = read(ROOT / "docs/cmc_pca7_repeated_protocol_20261020.json")["geometries"]
    angles = np.pi * (train - audit["encoder"]["center"]) / audit["encoder"]["scale"]
    return values, ytrain, ytune, [row["gamma_factor"] / max(7 * angles.var(), 1e-12) for row in geometry]


def fit_job(root, index, probe=False):
    assert not (root / "selection_lock.json").exists() and not (root / "test_release.json").exists()
    verify(root)
    if probe:
        probe_settings = [("RBF-SVM", "gbdt_soft_gate"), ("QSVC", "gbdt_rank_append"), ("RBF-SVM", "gbdt_rank_append")]
        kernel, method = probe_settings[index]
        job = ("taiwan", 10, kernel, method)
        choice = {"job": job, "C": 100., "lambda": 1., "geometry": 0}
        folder = root / "preflight" / f"probe_{index:02d}"
    else:
        choice = read(root / "parameter_plan.json")["jobs"][index]
        job = tuple(choice["job"])
        folder = job_path(root, job)
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / "started.json", {"utc": now(), "pid": os.getpid(), "job": job, "probe": probe})
    tick = time.perf_counter()
    try:
        values, ytrain, ytune, gammas = load(root, job, probe)
        encoded = {part: features(q, choice["lambda"], job[2], job[3])[0] for part, q in values.items()}
        kind = features(values["train"][:1], choice["lambda"], job[2], job[3])[1]
        gamma = gammas[choice["geometry"]]
        model = SVC(C=choice["C"], kernel=kind, gamma=gamma if kind == "rbf" else "scale", class_weight="balanced", cache_size=256, tol=.001)
        start = time.perf_counter()
        model.fit(encoded["train"], ytrain)
        fit_seconds = time.perf_counter() - start
        assert model.fit_status_ == 0 and model.shape_fit_[0] == len(ytrain)
        scores = {part: model.decision_function(encoded[part]) for part in encoded if part != "train"}
        cut = threshold(ytune, scores["tune"])
        write(folder / "parameter_lock.json", {"utc": now(), "job": job, **{key: choice[key] for key in ("C", "lambda", "geometry")},
            "gamma": gamma, "native_kernel": kind, "threshold": cut, "tune_cost": metrics(ytune, scores["tune"], cut)["cost"],
            "fit_rows": len(ytrain), "feature_dimensions": encoded["train"].shape[1], "model_fit_calls": 1,
            "parent_parameters_not_retuned": True, "test_labels_read": False})
        joblib.dump(model, folder / "model.joblib")
        np.savez_compressed(folder / "margins.npz", **scores)
        result = {"utc": now(), "status": "completed", "job": job, "pid": os.getpid(), "probe": probe,
            "elapsed_s": time.perf_counter() - tick, "fit_s": fit_seconds, "peak_gib": peak_gib(), "support_vectors": len(model.support_),
            "test_inputs_used": not probe, "test_labels_read": False,
            "sha256": {name: sha(folder / name) for name in ("model.joblib", "margins.npz", "parameter_lock.json")}}
        write(folder / "complete.json", result)
        return result
    except BaseException:
        write(folder / "failure.json", {"utc": now(), "traceback": traceback.format_exc(), "automatic_retry": False})
        raise


def launch(root):
    meta = verify(root, True)
    assert meta["status"] == "initialized"
    meta["status"] = "preparing"
    write(root / "metadata.json", meta, False)
    for index in range(3):
        folder = root / "preflight"
        folder.mkdir(exist_ok=True)
        with (folder / f"probe_{index:02d}.log").open("x") as handle:
            subprocess.run([sys.executable, "-B", __file__, "--operation", "probe", "--index", str(index)], stdout=handle, stderr=subprocess.STDOUT, check=True)
    probes = [read(root / "preflight" / f"probe_{i:02d}/complete.json") for i in range(3)]
    budget = max(1.5, 1.5 * max(x["peak_gib"] for x in probes))
    write(root / "preflight_receipt.json", {"utc": now(), "status": "passed", "full_training_rows": 24000,
        "train_validation_only_probes": 3, "test_inputs_used": False, "worker_budget_gib": budget, "max_workers": 4,
        "probe_hashes": {str((root / "preflight" / f"probe_{i:02d}/complete.json").relative_to(root)): sha(root / "preflight" / f"probe_{i:02d}/complete.json") for i in range(3)}})
    meta.update(status="running", scientific_start_utc=now(), worker_budget_gib=budget)
    write(root / "metadata.json", meta, False)
    queue, running = list(range(160)), {}
    while queue or running:
        for pid, (process, log, index) in list(running.items()):
            code = process.poll()
            if code is not None:
                log.close()
                del running[pid]
                if code:meta["failures"].append({"job_index": index, "exit_code": code})
                else:meta["jobs_completed"] += 1
        free = available_gib()
        while queue and not meta["failures"] and len(running) < 4 and free >= .2 * 64 + 8 + budget:
            index = queue.pop(0)
            folder = job_path(root, jobs()[index])
            folder.mkdir(parents=True, exist_ok=True)
            log = (folder / "worker.log").open("x")
            process = subprocess.Popen([sys.executable, "-B", __file__, "--operation", "job", "--index", str(index)], stdout=log, stderr=subprocess.STDOUT)
            running[process.pid] = process, log, index
            free -= budget
        write(root / "monitor.json", {"utc": now(), "status": "running", "completed": meta["jobs_completed"], "expected": 160,
            "failures": meta["failures"], "queued": len(queue), "active": [{"pid": pid, "index": value[2]} for pid, value in running.items()],
            "available_gib": free, "max_workers": 4, "automatic_retries": False}, False)
        write(root / "metadata.json", meta, False)
        if meta["failures"] and not running:raise RuntimeError("Worker failure preserved; no automatic scientific retry")
        if queue or running:time.sleep(3)
    assert meta["jobs_completed"] == 160
    lock = []
    for job in jobs():
        file = job_path(root, job) / "complete.json"
        assert read(file)["status"] == "completed"
        lock.append({"job": job, "path": str(file.relative_to(root)), "sha256": sha(file)})
    write(root / "selection_lock.json", {"utc": now(), "models_and_validation_thresholds": lock, "all_models": 160})
    write(root / "test_release.json", {"utc": now(), "selection_lock_sha256": sha(root / "selection_lock.json"), "no_further_fitting": True})
    evaluate(root)
    validate(root)
    meta.update(status="completed", finished_utc=now())
    write(root / "metadata.json", meta, False)
    write(root / "monitor.json", {"utc": now(), "status": "completed", "completed": 160, "expected": 160, "failures": [], "active": []}, False)


def labels():
    p = configuration()
    file = ROOT / "code/results" / p["parent_run"] / "full/predictions.csv.gz"
    rows = []
    for chunk in pd.read_csv(file, chunksize=100000, usecols=["dataset", "seed", "row_id", "y", "kernel", "family", "arm", "role"]):
        keep = chunk.kernel.eq("QSVC") & chunk.family.eq("baseline") & chunk.arm.eq("normal") & chunk.role.eq("baseline")
        rows.append(chunk.loc[keep, ["dataset", "seed", "row_id", "y"]])
    result = pd.concat(rows, ignore_index=True)
    assert not result.duplicated(["dataset", "seed", "row_id"]).any()
    return result


def evaluate(root):
    verify(root, True)
    assert (root / "selection_lock.json").exists() and (root / "test_release.json").exists()
    p = configuration()
    parent = ROOT / "code/results" / p["parent_run"]
    raw = pd.read_csv(parent / "full/seed_metrics.csv", float_precision="round_trip")
    geometry = pd.read_csv(ROOT / "code/results" / p["analysis_run"] / "geometry_seed_metrics.csv", float_precision="round_trip")
    target = labels()
    rows = []
    for job in jobs():
        d, s, k, m = job
        folder = job_path(root, job)
        choice = read(folder / "parameter_lock.json")
        split = read(parent / "full/prepared" / d / f"seed_{s:04d}/split.json")["actual"]
        y = target[target.dataset.eq(d) & target.seed.eq(s)].set_index("row_id").reindex(split["test"]).y.to_numpy(dtype=int)
        with np.load(folder / "margins.npz", allow_pickle=False) as saved:score = saved["gate"]
        result = metrics(y, score, choice["threshold"])
        full = raw[raw.dataset.eq(d) & raw.seed.eq(s) & raw.kernel.eq(k) & raw.family.eq(m) & raw.arm.eq("normal") & raw.role.eq("proposed")]
        before = geometry[geometry.dataset.eq(d) & geometry.seed.eq(s) & geometry.kernel.eq(k) & geometry.method.eq(m)]
        assert len(full) == len(before) == 1
        rows.append({"dataset": d, "seed": s, "kernel": k, "method": m, **result,
            "full_product_cost": full.cost.iloc[0], "matched_backbone_cost": full.cost.iloc[0] + before.full_cost_reduction.iloc[0],
            "delta_risk_only_minus_full": result["cost"] - full.cost.iloc[0],
            "C": choice["C"], "lambda": choice["lambda"], "geometry": choice["geometry"], "threshold": choice["threshold"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "seed_metrics.csv", index=False)
    statistics = []
    for key, group in frame.groupby(["dataset", "kernel", "method"], sort=False):
        group = group.sort_values("seed")
        assert group.seed.tolist() == p["seeds"]
        statistics.append({"dataset": key[0], "kernel": key[1], "method": key[2], **corrected(group.delta_risk_only_minus_full.to_numpy()),
            "risk_only_mean": group.cost.mean(), "risk_only_sd": group.cost.std(ddof=1), "full_product_mean": group.full_product_cost.mean(),
            "matched_backbone_mean": group.matched_backbone_cost.mean()})
    stats = pd.DataFrame(statistics)
    assert len(stats) == 16
    stats["p_holm16_exploratory"] = holm(stats.p_value.to_numpy())
    stats.to_csv(root / "paired_statistics.csv", index=False)
    report = ["# 위험요인 단독 대조 결과", "", "## Material Passport", "",
        "상태: ANALYZED. 기존 두 방법에서 고객별 PCA7 backbone만 제거한 구성요소 대조다. 네 자료×10시드×두 커널×두 방법의160개 SVC를 전체 학습행으로 새로 적합했다. C/lambda/기하는 부모의 검증선정 값을 고정했고, 각 대조모델의 임계값만 검증 비용으로 선택했다. 이미 본 자료·시드·테스트를 이용하므로 독립 확인은 아니다.", "",
        "Soft Gate 및 QSVC Rank Append의 위험 커널은 정확한3차원 고전 특징지도로, RBF Rank Append는1차원 Gaussian 커널로 계산했다. 이는 양자 위험요인 자체가 별도 양자 계산을 요구하지 않음을 이용한 정확 구현이며, 전체 양자 backbone의 고전적 단순성을 주장하지 않는다.", "",
        "위험요인 단독은 별도로 최적화한 최강 단독모델이 아니다. 결합모델용으로 선택된 파라미터에서 backbone을 제거한 조건부 비교다. 양수 차이는 같은 조건에서 전체 제품 커널이 더 낮은 비용을 보였다는 뜻이며, 새16개 진단대비 전체 Holm16을 적용했다. 기존184/100/40가족을 축소하거나 재계산하지 않았다. 근사 corrected resampled t(df9,분산계수.225),개별95%CI,10개 중첩분할이다.", "",
        "| 자료 | 부모 커널 | 방법 | matched backbone | 위험요인 단독 평균±SD | 전체 제품 커널 | 단독−제품 | 근사95% CI | 원p | 탐색 Holm16 p | 개선/동률/악화 |", "|---|---|---|---:|---:|---:|---:|---|---:|---:|---|"]
    for r in stats.itertuples():
        report.append(f"| {r.dataset} | {r.kernel} | {r.method} | {r.matched_backbone_mean:.1f} | {r.risk_only_mean:.1f}±{r.risk_only_sd:.1f} | {r.full_product_mean:.1f} | {r.mean_difference:.1f} | [{r.ci_low:.2f},{r.ci_high:.2f}] | {r.p_value:.6f} | {r.p_holm16_exploratory:.6f} | {r.wins}/{r.ties}/{r.losses} |")
    report += ["", "## 해석과 한계", "",
        "새 결과는 위험점수 자체로 설명되는 부분과 matched 설정에서 backbone이 추가하는 부분을 구분하기 위한 진단이다. 차이가 작거나 유의하지 않다는 것만으로 동등성을 입증하지 않는다. 반대로 제품 커널이 유리해도 강한 XGB 단독 대비 실용적 우위나 양자 고유 이점을 뜻하지 않는다. 고정 파라미터 대조라는 제한, 이미 본 테스트,겹친분할,n10,관련German표현의 한계를 유지한다.", "",
        "새 SVC는 로컬 sklearn1.8.0에서 적합했다. 부모는 서버 sklearn1.7.2와 전계산 Gram을 사용했다. 위험 Gram의 정확한 특징지도 항등식을 검증했지만 서로 다른 커널 구현/환경의 바이트 동일성이나 속도 비교는 주장하지 않는다. 원특성 교사점수는 새로 적합하지 않았고, 새 학습에 테스트 레이블을 사용하지 않았다.", "",
        "통계 오류11유형 점검: 자료/커널별 결과를 유지해Simpson집계왜곡을 제한,개인 인과 추론 없음,자료수집/ collider선택 미확인,동일 고객/FN/FP/기저율 유지,HMEQ사후주목·평균회귀 한계 유지,160개 전부 보존해생존자편향 제한,새16대비 전체 보정,사후탐색 지위 공개,상관/역인과 주장 없음. 점검은편향 부재의 인증이 아니다.", ""]
    (root / "report_ko.md").write_text("\n".join(report))


def validate(root):
    meta = verify(root, True)
    assert not meta["failures"]
    lock = read(root / "selection_lock.json")
    release = read(root / "test_release.json")
    assert release["selection_lock_sha256"] == sha(root / "selection_lock.json")
    assert datetime.fromisoformat(lock["utc"]) <= datetime.fromisoformat(release["utc"])
    locked = {tuple(x["job"]): x for x in lock["models_and_validation_thresholds"]}
    plans = {tuple(x["job"]): x for x in read(root / "parameter_plan.json")["jobs"]}
    count = 0
    for job in jobs():
        folder = job_path(root, job)
        complete = read(folder / "complete.json")
        assert sha(folder / "complete.json") == locked[job]["sha256"]
        assert datetime.fromisoformat(complete["utc"]) < datetime.fromisoformat(lock["utc"])
        for name, digest in complete["sha256"].items():assert sha(folder / name) == digest
        choice = read(folder / "parameter_lock.json")
        assert all(choice[key] == plans[job][key] for key in ("C", "lambda", "geometry"))
        values, ytrain, ytune, _ = load(root, job)
        model = joblib.load(folder / "model.joblib")
        assert model.fit_status_ == 0 and model.shape_fit_[0] == choice["fit_rows"] == len(ytrain)
        assert model.C == choice["C"] and model.kernel == choice["native_kernel"]
        if model.kernel == "rbf":assert model._gamma == choice["gamma"]
        with np.load(folder / "margins.npz", allow_pickle=False) as saved:
            for part in ("tune", "gate"):
                x, _ = features(values[part], choice["lambda"], job[2], job[3])
                np.testing.assert_array_equal(model.decision_function(x), saved[part])
            assert threshold(ytune, saved["tune"]) == choice["threshold"]
        count += 1
    frame = pd.read_csv(root / "seed_metrics.csv", float_precision="round_trip")
    stats = pd.read_csv(root / "paired_statistics.csv", float_precision="round_trip")
    assert len(frame) == 160 and len(stats) == 16
    for row in stats.itertuples():
        values = frame[frame.dataset.eq(row.dataset) & frame.kernel.eq(row.kernel) & frame.method.eq(row.method)].sort_values("seed")
        result = corrected(values.delta_risk_only_minus_full.to_numpy())
        for field in ("mean_difference", "ci_low", "ci_high", "p_value"):
            assert np.isclose(getattr(row, field), result[field], equal_nan=True)
    np.testing.assert_allclose(stats.p_holm16_exploratory, holm(stats.p_value.to_numpy()))
    write(root / "validation_receipt.json", {"utc": now(), "status": "passed", "models_verified": count, "full_training_rows": True,
        "fit_status_zero": True, "saved_model_margin_replays": count * 2, "parent_parameters_preserved": True,
        "global_lock_before_test_labels": True, "contrasts": 16, "holm_family": "new_exploratory16_not_independent_confirmation",
        "new_teacher_fits": 0, "automatic_scientific_retries": False, "environment": {"sklearn": sklearn.__version__, "python": platform.python_version()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("run", "job", "probe"), default="run")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    root = ROOT / "code/results" / configuration()["run_id"]
    with threadpool_limits(limits=1):
        if args.operation == "run":
            initialize(root)
            try:launch(root)
            except BaseException:
                meta = read(root / "metadata.json")
                meta.update(status="failed", error=traceback.format_exc())
                write(root / "metadata.json", meta, False)
                write(root / "failure.json", {"utc": now(), "traceback": traceback.format_exc(), "automatic_retry": False})
                raise
            print(json.dumps({"status": "completed", "jobs": 160, "root": str(root)}))
        else:print(json.dumps(fit_job(root, args.index, args.operation == "probe")))
