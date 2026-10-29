"""Precommit choices, compare all ten methods, and audit saved predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import controlled as base
from .full_data import write_json
from .improved import sha
from .quantum_ten import LABELS, job_folder, load_inputs
from .separation_round2 import prepared_folder
from build_controlled_results import corrected_difference, holm

ROOT = base.ROOT


def choose(rows):
    if not rows:
        raise AssertionError("Missing planned model candidates")
    return min(rows,key=lambda r:(r["tune_cost"],r["lambda"],r.get("C",0.),r.get("gamma_factor",0.),r["learner"],r["method"],r.get("scaled",False),str(r.get("depth"))))


def collect(output,jobs,hashes):
    groups,receipts,accounting = {},{},[]
    for job in jobs:
        folder = job_folder(output,job)
        saved = json.loads((folder/"complete.json").read_text())
        _,record = load_inputs(output,*job[:2])
        if saved["job"]!=list(job) or saved["source_sha256"]!=hashes or saved["prepared_input_sha256"]!=record["candidate_inputs_sha256"] or saved["prepared_audit_sha256"]!=record["audit_sha256"]:
            raise AssertionError("Candidate identity differs")
        if sha(folder/"margins.npz")!=saved["archive_sha256"] or saved["held_out_labels_read"]:
            raise AssertionError("Margin integrity or label isolation differs")
        with np.load(folder/"margins.npz",allow_pickle=False) as margins:
            if margins["tune"].shape[0]!=len(saved["trials"]) or margins["gate"].shape[0]!=len(saved["trials"]):
                raise AssertionError("Incomplete candidate margin archive")
        rows = [{**r,"archive":str((folder/"margins.npz").relative_to(ROOT)),"archive_sha256":saved["archive_sha256"],"score_index":i} for i,r in enumerate(saved["trials"])]
        groups[job] = rows
        receipts[str((folder/"complete.json").relative_to(ROOT))] = sha(folder/"complete.json")
        accounting.append({"dataset":job[0],"seed":job[1],"method":job[2],"arm":job[3],"fits":sum(r.get("model_fit_calls",0) for r in rows),"candidates":len(rows),"elapsed_seconds":saved["elapsed_seconds"]})
    return groups,receipts,accounting


def select(groups,datasets,seeds,methods,strengths):
    choices,profiles = [],[]
    for dataset in datasets:
        for seed in seeds:
            classic = groups[dataset,seed,"classical","classical"]
            shared = {"classical_best":choose(classic),
                      "rbf_baseline":choose([r for r in classic if r["learner"]=="RBF-SVM" and r["lambda"]==0]),
                      "rbf_risk":choose([r for r in classic if r["learner"]=="RBF-SVM" and r["lambda"]>0])}
            for method in methods:
                rows = groups[dataset,seed,method,"quantum"]
                selected = {"baseline":choose([r for r in rows if r["lambda"]==0]),
                            "proposed":choose([r for r in rows if r["lambda"]>0]),
                            "separable":choose([r for r in groups[dataset,seed,method,"separable"] if r["lambda"]>0]),
                            "shuffled":choose([r for r in groups[dataset,seed,method,"shuffled"] if r["lambda"]>0]),**shared}
                for role,row in selected.items():
                    choices.append({**row,"candidate_method":method,"role":role})
                for strength in strengths:
                    row = choose([r for r in rows if r["lambda"]==strength])
                    profiles.append({**row,"candidate_method":method,"role":"lambda_profile"})
            # The original six categories are retained as separately named unclipped controls.
            if (dataset,seed,"standard","quantum") in groups:
                controls = {"q_baseline":choose(groups[dataset,seed,"standard","quantum"]),
                            "rbf_baseline":choose([r for r in classic if r["method"]=="baseline"])}
                for rep in ("convex","feature_specific"):
                    key = dataset,seed,"standard_"+rep,"quantum"
                    if key in groups:
                        controls["q_"+rep] = choose([r for r in groups[key] if r["lambda"]>0])
                    controls["rbf_"+rep] = choose([r for r in classic if r["method"]=="original_"+rep])
                for role,row in controls.items():
                    choices.append({**row,"candidate_method":"standard_controls","role":role})
    return choices,profiles


def evaluate(output,choices,full,protocol,with_predictions=True):
    labels,cache,rows,predictions = {},{},[],[]
    source_data = {}
    for row in choices:
        pair = row["dataset"],row["seed"]
        folder = prepared_folder(output,*pair)
        arrays,record = load_inputs(output,*pair)
        if pair not in labels:
            if full:
                if pair[0] not in source_data:
                    _,y,source = base.load_data(pair[0],ROOT/"data")
                    source_data[pair[0]] = y,source
                y,source = source_data[pair[0]]
                source_folder = ROOT/"code"/pair[0]/"results"/protocol["source_run"]/f"seed_{pair[1]:03d}"
                receipt = json.loads((source_folder/"prepared.json").read_text())
                split = json.loads((source_folder/"split_indices.json").read_text())
                from .single_811 import split_indices
                if split!=split_indices(y,pair[1]) or source["sha256"]!=receipt["identity"]["data_sha256"] or len(arrays["y_train"])!=len(split["train"]):
                    raise AssertionError("Full data/split/training count differs")
                labels[pair] = y[split["test"]],np.asarray(split["test"])
            else:
                if sha(folder/"gate_labels.npz")!=record["gate_labels_sha256"]:
                    raise AssertionError("Gate labels changed")
                with np.load(folder/"gate_labels.npz",allow_pickle=False) as saved:
                    labels[pair] = saved["y_gate"],np.asarray(record["gate_positions"])
        archive = ROOT/row["archive"]
        if row["archive"] not in cache:
            if sha(archive)!=row["archive_sha256"]:
                raise AssertionError("Locked margins changed")
            with np.load(archive,allow_pickle=False) as saved:
                cache[row["archive"]] = {k:saved[k] for k in ("tune","gate")}
        scores = cache[row["archive"]]
        index = row["score_index"]
        tune = base.metrics(arrays["y_tune"],scores["tune"][index],row["threshold"])
        if tune["cost"]!=row["tune_cost"] or row["fit_rows"]!=len(arrays["y_train"]):
            raise AssertionError("Same-fit tune selection or training rows differ")
        y,row_ids = labels[pair]
        score = scores["gate"][index]
        metrics = base.metrics(y,score,row["threshold"])
        result = {**{k:v for k,v in row.items() if k not in ("archive","archive_sha256","score_index","representation_details")},
                  "n_train":len(arrays["y_train"]),"n_tune":len(arrays["y_tune"]),"n_eval":len(y),**metrics,
                  "specificity":metrics["tn"]/max(metrics["tn"]+metrics["fp"],1)}
        source_times = record.get("source_audit",{}).get("source_times",{})
        result["cached_preprocessing_training_seconds"] = source_times.get("feature_fit_seconds",0.)+(source_times.get("risk_fit_seconds",0.) if row["lambda"]>0 else 0.)
        rows.append(result)
        if full and with_predictions:
            predictions.append(pd.DataFrame({"dataset":pair[0],"seed":pair[1],"candidate_method":row["candidate_method"],"role":row["role"],"lambda":row["lambda"],"row_id":row_ids,"y":y,"score":score,"threshold":row["threshold"],"prediction":(score>=row["threshold"]).astype(int)}))
    return pd.DataFrame(rows),predictions


def effect(group):
    model = {role:rows.set_index("seed") for role,rows in group.groupby("role")}
    q,b = model["proposed"],model["baseline"]
    c,s,h,rb,rq = [model[k].reindex(q.index) for k in ("classical_best","separable","shuffled","rbf_baseline","rbf_risk")]
    b = b.reindex(q.index)
    denominators = [x.cost for x in (b,c,s,h,rb)]
    zero = any(x.eq(0).any() for x in denominators)
    div = lambda n,d: n/d.replace(0,np.nan)
    own = div(b.cost-q.cost,b.cost)
    metrics = {"own_rcr":own,"classical_rcr":div(c.cost-q.cost,c.cost),"interaction":own-div(rb.cost-rq.cost,rb.cost),
               "separable_rcr":div(s.cost-q.cost,s.cost),"shuffled_rcr":div(h.cost-q.cost,h.cost),"recall_change":q.recall-b.recall}
    summary = {k:None if v.isna().any() else float(v.mean()) for k,v in metrics.items()}
    summary.update(zero_denominator=zero,own_wins=int((b.cost>q.cost).sum()),classical_wins=int((c.cost>q.cost).sum()),
                   n=len(q),balanced_accuracy=float(q.balanced_accuracy.mean()),specificity_positive=bool(q.specificity.gt(0).all()))
    return summary


def passes(row,g,dataset_scope):
    if row["zero_denominator"] or any(row[k] is None for k in ("own_rcr","classical_rcr","interaction","separable_rcr","shuffled_rcr")):
        return False
    ok = row["own_rcr"]>=g["own_baseline_rcr"] and row["classical_rcr"]>=g["best_classical_rcr"] and row["interaction"]>=g["interaction_rcr_points"] and row["separable_rcr"]>=g["separable_rcr"] and row["shuffled_rcr"]>=g["shuffled_rcr"] and row["recall_change"]>=g["minimum_recall_change"] and row["balanced_accuracy"]>=g["minimum_balanced_accuracy"] and row["specificity_positive"]
    if dataset_scope:
        ok = ok and row["own_wins"]>=g["dataset_wins_of_five"] and row["classical_wins"]>=g["dataset_wins_of_five"]
    else:
        ok = ok and row["classical_improved_datasets"]>=g["global_improved_datasets"]
    return bool(ok)


def promotion(raw,protocol):
    data_rows,global_rows,promoted = [],[],[]
    current = raw[raw.candidate_method.ne("standard_controls")]
    for method in protocol["methods"]:
        rows = []
        for dataset in protocol["datasets"]:
            part = current[current.candidate_method.eq(method)&current.dataset.eq(dataset)]
            row = {"method":method,"dataset":dataset,**effect(part)}
            if row["n"]!=len(protocol["smoke_seeds"]):
                raise AssertionError("Incomplete five-seed smoke")
            row["passes"] = passes(row,protocol["promotion"],True)
            rows.append(row)
            data_rows.append(row)
        aggregate = {"method":method,"scope":"global","zero_denominator":any(r["zero_denominator"] for r in rows),
                     "specificity_positive":all(r["specificity_positive"] for r in rows),
                     "classical_improved_datasets":sum(r["classical_rcr"] is not None and r["classical_rcr"]>0 for r in rows)}
        for k in ("own_rcr","classical_rcr","interaction","separable_rcr","shuffled_rcr","recall_change","balanced_accuracy"):
            aggregate[k] = None if any(r[k] is None for r in rows) else float(np.mean([r[k] for r in rows]))
        aggregate["passes"] = passes(aggregate,protocol["promotion"],False)
        global_rows.append(aggregate)
        if aggregate["passes"] or any(r["passes"] for r in rows):
            promoted.append(method)
    return {"promoted_methods":promoted,"global_scopes":global_rows,"dataset_scopes":data_rows,
            "rule":protocol["promotion"],"not_a_significance_or_quantum_advantage_proof":True}


def paired(raw,profiles=False):
    rows = []
    current = raw[raw.candidate_method.ne("standard_controls")]
    for (dataset,method),part in current.groupby(["dataset","candidate_method"]):
        model = {role:group.set_index("seed") for role,group in part.groupby("role")}
        q,b,c,s,h,rb,rq = [model[k] for k in ("proposed","baseline","classical_best","separable","shuffled","rbf_baseline","rbf_risk")]
        index = q.index
        b,c,s,h,rb,rq = [x.reindex(index) for x in (b,c,s,h,rb,rq)]
        ratio = float((q.n_eval/q.n_train).iloc[0])
        contrasts = {"quantum_within":b.cost-q.cost,"quantum_vs_classical":c.cost-q.cost,"quantum_vs_separable":s.cost-q.cost,
                     "quantum_vs_shuffled":h.cost-q.cost,"quantum_minus_rbf_gain":(b.cost-q.cost)-(rb.cost-rq.cost),"rbf_within":rb.cost-rq.cost}
        for name,delta in contrasts.items():
            if delta.isna().any() or len(delta)!=10:
                raise AssertionError("Ten complete common paired seeds required")
            rows.append({"dataset":dataset,"method":method,"contrast":name,**corrected_difference(delta.to_numpy(),ratio)})
    result = pd.DataFrame(rows)
    singular = result.sd_difference.eq(0)&result.mean_difference.ne(0)
    result.loc[singular,["p_value","ci_low","ci_high"]] = np.nan
    result["p_holm"] = holm(result.p_value.fillna(1).to_numpy())
    result.loc[result.p_value.isna(),"p_holm"] = np.nan
    result["holm_family_size"] = len(result)
    return result


def report(output,jobs,hashes,protocol,methods,seeds,full=False):
    groups,receipts,accounting = collect(output,jobs,hashes)
    strengths = protocol["lambdas"] if full else protocol["smoke_lambdas"]
    choices,profiles = select(groups,protocol["datasets"],seeds,methods,strengths)
    lock = {"models":choices,"lambda_profiles":profiles,"candidate_receipts_sha256":receipts,"source_sha256":hashes,
            "rule":"Every method/ablation/classical/lambda/threshold chosen on tune labels only; committed before gate/test labels"}
    path = output/"selection_lock.json"
    if path.exists():
        raise AssertionError("Never overwrite selection lock")
    write_json(path,lock)
    # No evaluation label read occurs above this boundary.
    raw,predictions = evaluate(output,choices,full,protocol)
    profile,_ = evaluate(output,profiles,full,protocol,with_predictions=False)
    raw.to_csv(output/"seed_metrics.csv",index=False)
    profile.to_csv(output/"lambda_seed_metrics.csv",index=False)
    pd.DataFrame(accounting).to_csv(output/"compute_accounting.csv",index=False)
    pd.DataFrame([r for group in groups.values() for r in group]).to_csv(output/"all_tune_trials.csv",index=False)
    raw.groupby(["dataset","candidate_method","role"])[["cost","fn","fp","f1","recall","balanced_accuracy"]].agg(["mean","std"]).to_csv(output/"summary_metrics.csv")
    raw[raw.candidate_method.eq("standard_controls")].to_csv(output/"standard_reference_models.csv",index=False)
    settings = [{"dataset":r["dataset"],"seed":r["seed"],"method":r["candidate_method"],"role":r["role"],"lambda":r["lambda"],"details":r.get("representation_details",{})} for r in choices]
    write_json(output/"quantum_settings.json",{"selected_model_settings":settings,"simulator":"CPU Aer double exact statevector; no QPU/no finite-shot noise/no GPU acceleration"})
    if full:
        stats = paired(raw)
        stats.to_csv(output/"paired_cost.csv",index=False)
        pd.concat(predictions,ignore_index=True).to_csv(output/"predictions.csv.gz",index=False,compression="gzip")
        directional = []
        for (dataset,method),group in raw[raw.candidate_method.ne("standard_controls")].groupby(["dataset","candidate_method"]):
            q = group[group.role.eq("proposed")].set_index("seed")
            b = group[group.role.eq("baseline")].set_index("seed").reindex(q.index)
            for seed in q.index:
                delta = int(b.loc[seed,"cost"]-q.loc[seed,"cost"])
                directional.append({"dataset":dataset,"method":method,"seed":seed,"baseline_cost":int(b.loc[seed,"cost"]),"proposed_cost":int(q.loc[seed,"cost"]),"paired_cost_difference":delta,"relative_cost_reduction_pct":100*delta/b.loc[seed,"cost"] if b.loc[seed,"cost"] else np.nan,"direction":"개선" if delta>0 else "악화" if delta<0 else "동률"})
        pd.DataFrame(directional).to_csv(output/"seed_directions.csv",index=False)
        profile_stats = []
        for strength in strengths:
            q = profile[profile["lambda"].eq(strength)].copy()
            q["role"] = "proposed"
            part = pd.concat([q,raw[raw.role.ne("proposed")&raw.candidate_method.ne("standard_controls")]],ignore_index=True)
            table = paired(part)
            table["lambda"] = strength
            profile_stats.append(table)
        ps = pd.concat(profile_stats,ignore_index=True)
        ps["p_holm"] = holm(ps.p_value.fillna(1).to_numpy())
        ps.loc[ps.p_value.isna(),"p_holm"] = np.nan
        ps["holm_family_size"] = len(ps)
        ps.to_csv(output/"lambda_paired_cost.csv",index=False)
        outcome = {"completed_full_methods":methods,"seed_jobs":len(seeds)*len(protocol["datasets"])}
    else:
        outcome = promotion(raw,protocol)
        write_json(output/"promotion.json",outcome)
        pd.DataFrame(outcome["global_scopes"]).to_csv(output/"global_smoke_comparisons.csv",index=False)
        pd.DataFrame(outcome["dataset_scopes"]).to_csv(output/"dataset_smoke_comparisons.csv",index=False)
    lines = ["# 양자 위험 커널 10개 방법 비교", "", "## Material Passport", "", "상태: ANALYZED. 저장 점수/분할/선택 검증이며 독립적 종단간 재학습 재현이나 이론적 양자 우위의 증명이 아니다.", "",
             f"단계: {'전체' if full else '스모크'}. 자료4개, 시드{len(seeds)}개, 후보 방법{len(methods)}개. 모든 C/lambda/고전 대조/판정 기준을 선택용 비용으로 잠근 뒤 확인용 정답을 사용했다.", "",
             "## 결과", ""]
    if full:
        lines += ["| 자료 | 방법 | 기준 대비 비용 Δ | 95% 구간 | 원 p | Holm p | 개선/동률/악화 |","| --- | --- | ---: | --- | ---: | ---: | --- |"]
        for r in stats[stats.contrast.eq("quantum_within")].itertuples():
            lines.append(f"| {r.dataset} | {LABELS[r.method]} | {r.mean_difference:.2f} | [{r.ci_low:.2f}, {r.ci_high:.2f}] | {r.p_value:.4f} | {r.p_holm:.4f} | {r.wins}/{r.ties}/{r.losses} |")
        lines += ["", "모든 고전·얽힘 제거·셔플·상호작용 짝비교는 paired_cost.csv, lambda 전체는 lambda_paired_cost.csv에 있다. Δ는 대조−양자 비용으로 양수가 개선이다. 반복 분할 중복 보정 t 구간/검정은 근사이며 사후 선택과 데이터 재사용을 없애지 않는다."]
    else:
        lines += ["양수는 비용 감소, 상호작용은 양자 위험 주입 감소율−RBF 위험 주입 감소율이다. 전 방법의 실패도 모두 제시한다.", "", "| 방법 | 자기 기준 대비 | 최선 고전 대비 | 상호작용 | 얽힘 제거 대비 | 셔플 대비 | 전체/자료 한정 통과 |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        fmt = lambda x:"정의 불가" if x is None else f"{100*x:.2f}%"
        for r in outcome["global_scopes"]:
            ds = [x["dataset"] for x in outcome["dataset_scopes"] if x["method"]==r["method"] and x["passes"]]
            lines.append(f"| {LABELS[r['method']]} | {fmt(r['own_rcr'])} | {fmt(r['classical_rcr'])} | {fmt(r['interaction'])}p | {fmt(r['separable_rcr'])} | {fmt(r['shuffled_rcr'])} | {'전체' if r['passes'] else ', '.join(ds) if ds else '없음'} |")
        lines += ["", "전체 실험 승격 방법: "+(", ".join(outcome["promoted_methods"]) or "없음")+". 유리한 자료만 골라 전체 결과를 보고하지 않고 승격된 방법은 네 자료 모두에서 전체 행·10개 시드로 실행한다."]
    lines += ["", "## 자원과 해석 범위", "", "7/8큐비트 CPU Aer double 정확 상태벡터다. 실제 QPU·잡음·유한 샷·GPU 가속·계산 복잡도 이점을 측정하지 않았다. 관측량·국소 HS·다중 척도는 고전 후처리 또는 커널 혼합을 포함한다. 정렬형은 학습128개에서 Powell 최대24회 평가로 맞추며 Qiskit의 SVCLoss 학습기와 동일한 알고리즘이라고 주장하지 않는다. optimizer 성공 여부·깊이·게이트 수·상태벡터 행 수는 quantum_settings.json에 있다.", "",
        "RBF에는 더 넓은 탐색을 허용했고 RF/XGBoost/LR도 같은 PCA7+LR 정보로 비교했다. 원래 특성을 모두 쓰는 최강의 상용 신용평가 모델을 이겼다는 검증이 아니다. 모든 방법은 작은 회로를 고전적으로 시뮬레이션했으므로 경험적 대조 이득을 일반적인 양자 우위로 부르지 않는다.", "",
        "스모크는 최대2048 학습 행·시드60~100 다섯 개·검증10%의 선택/확인 절반이다. 전체는 모든 학습80%·검증10%·테스트10%, 양수0.05 간격 lambda, 같은 적합 점수이며 선택 후 재학습하지 않는다. 원래 여섯 모델의 무절단 대조 버전은 standard_reference_models.csv에 있다. 캐시 PCA/LR 시간과 새 커널/학습 시간을 구분하고 공유 시간을 합해 벽시계 시간이라고 주장하지 않는다.", "",
        "10개 전체와40개 자료 한정 스모크 비교는 사전 고정 실용 기준이지 유의성 검정이 아니다. 후속 전체는 탐색 후 선택·겹치는 시드/행·이전 결과를 본 설계라는 한계가 있다. 두 German 자료는 같은 원자료 계열이다.", "",
        "집계 반전·개인 일반화·선택 편향·충돌변수·기저율·평균 회귀·완료 선택·다중 탐색·사후 경로·인과 과장·역인과의 11개 항목을 이 범위에서 점검했다. 모든 편향 제거·인과적 금융 이익을 인증하지 않는다.", "",
        "근거: [projected quantum kernel 원논문](https://www.nature.com/articles/s41467-021-22539-9), [Qiskit Pauli feature map](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.pauli_feature_map), [학습형 양자 커널 예제](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/08_quantum_kernel_trainer.html)."]
    (output/"결과보고서.md").write_text("\n".join(lines)+"\n")
    files = ["selection_lock.json","seed_metrics.csv","lambda_seed_metrics.csv","결과보고서.md"]
    if full:
        files.extend(["paired_cost.csv","seed_directions.csv","predictions.csv.gz","lambda_paired_cost.csv"])
    else:
        files.append("promotion.json")
    write_json(output/"verification_receipt.json",{"scope":"Complete jobs, source identity, tune-only locks and same-fit stored metrics; not independent retraining",
        "jobs":len(jobs),"models":len(choices),"lambda_models":len(profiles),"candidate_fits":sum(a["fits"] for a in accounting),
        "outputs_sha256":{name:sha(output/name) for name in files}})
    return outcome
