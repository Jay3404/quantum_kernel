"""Ten quantum-risk mechanisms and matched, leakage-contained controls."""
from __future__ import annotations

from functools import lru_cache
import gc
import json
from pathlib import Path
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import pauli_feature_map, zz_feature_map, z_feature_map
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from . import controlled as base
from .full_data import available_memory, write_json
from .improved import local_lock, sha
from .separation_round2 import encoder_from, midrank, prepared_folder

ROOT = base.ROOT
METHODS = ("zz_phase", "reupload", "pauli_xy", "iqp_coupling", "risk_ancilla", "aligned_kernel",
           "projected_single", "projected_pairs", "local_hs", "multiscale")
LABELS = {"zz_phase":"ZZ 위험 위상", "reupload":"비가환 위험 재업로딩", "pauli_xy":"Pauli XX/YY",
          "iqp_coupling":"위험 조건 IQP 결합", "risk_ancilla":"얽힌 위험 보조 큐비트", "aligned_kernel":"학습형 커널 정렬",
          "projected_single":"단일 큐비트 관측량", "projected_pairs":"이중 큐비트 상관 관측량", "local_hs":"국소 밀도행렬",
          "multiscale":"다중 척도 fidelity"}


class RiskStates(base.QuantumStates):
    def __init__(self, method, weights, direction, entangled=True, alpha=.5):
        started = time.perf_counter()
        x = ParameterVector("x", 8)
        u = x[7]
        nq = 8 if method == "risk_ancilla" else 7
        circuit = QuantumCircuit(nq)
        weight = 7*np.asarray(weights)
        def layer(reps=1):
            fm = zz_feature_map(7,reps=reps,entanglement="linear",alpha=alpha,parameter_prefix="z") if entangled else z_feature_map(7,reps=reps,alpha=alpha,parameter_prefix="z")
            return fm.assign_parameters(dict(zip(fm.parameters,x[:7])))
        if method.startswith("standard"):
            circuit.compose(layer(),qubits=range(7),inplace=True)
        elif method == "zz_phase":
            circuit.compose(layer(2),inplace=True)
            for j in range(7):
                circuit.rz(u*weight[j],j)
        elif method == "pauli_xy":
            fm = pauli_feature_map(7,reps=2,entanglement="linear",alpha=alpha,paulis=["Z","XX","YY"] if entangled else ["Z"],parameter_prefix="z")
            fm = fm.assign_parameters({p:x[j]+u*direction[j] for j,p in enumerate(fm.parameters)})
            circuit.compose(fm,inplace=True)
        elif method == "iqp_coupling":
            circuit.h(range(7))
            for j in range(7):
                circuit.rz(alpha*x[j]+u*weight[j],j)
            if entangled:
                for j in range(6):
                    circuit.rzz(alpha*x[j]*x[j+1]*(1+u*weight[j]*weight[j+1]/np.pi),j,j+1)
        elif method == "risk_ancilla":
            circuit.compose(layer(),qubits=range(7),inplace=True)
            circuit.ry(u,7)
            if entangled:
                for j in range(7):
                    circuit.rzz(u*weight[j]/np.sqrt(7),j,7)
        else:
            circuit.compose(layer(),inplace=True)
            for j in range(7):
                circuit.ry(u*weight[j],j)
            circuit.compose(layer(),inplace=True)
        self.backend = AerSimulator(method="statevector",precision="double",max_parallel_threads=1,max_parallel_experiments=1,runtime_parameter_bind_enable=True)
        circuit.save_statevector(label="state")
        self.circuit = transpile(circuit,self.backend,optimization_level=1,seed_transpiler=17)
        self.parameters = tuple(self.circuit.parameters)
        self.setup_seconds = time.perf_counter()-started
        self.qubits = nq
        self.depth = self.circuit.depth()
        self.gates = dict(self.circuit.count_ops())


@lru_cache(maxsize=100)
def observable(label):
    return SparsePauliOp.from_list([(label,1)]).to_matrix(sparse=True)


def observables(states, pairs=False):
    nq = int(round(np.log2(states.shape[1])))
    labels = []
    for j in range(nq):
        for axis in "XYZ":
            value = ["I"]*nq
            value[nq-j-1] = axis
            labels.append("".join(value))
    if pairs:
        for j in range(nq-1):
            for axis in "XYZ":
                value = ["I"]*nq
                value[nq-j-1] = value[nq-j-2] = axis
                labels.append("".join(value))
    features = np.empty((len(states),len(labels)))
    for j,label in enumerate(labels):
        op = observable(label)
        for start in range(0,len(states),512):
            block = states[start:start+512]
            features[start:start+len(block),j] = np.einsum("ij,ij->i",block.conj(),(op@block.T).T).real
    return features


def centered_alignment(kernel,y):
    k = kernel-kernel.mean(0)-kernel.mean(1)[:,None]+kernel.mean()
    v = (2*np.asarray(y)-1).astype(float)
    v -= v.mean()
    target = np.outer(v,v)
    return float(np.sum(k*target)/max(np.linalg.norm(k)*np.linalg.norm(target),1e-15))


def ranks(arrays,arm,seed):
    result = {p:arrays[f"rank_{p}"].copy() for p in ("train","tune","gate")}
    if arm == "shuffled":
        for index,p in enumerate(result):
            result[p] = np.random.default_rng(seed+7919+index).permutation(result[p])
    return result


def fit_alignment(arrays,record,q,strength,seed,protocol,entangled=True,classical=False,scaled=False):
    positions = np.arange(len(arrays["y_train"]))
    n = protocol["alignment"]["sample_size"]
    if len(positions)>n:
        positions,_ = train_test_split(positions,train_size=n,stratify=arrays["y_train"],random_state=seed+4011)
    all_angles = encoder_from(record).transform(arrays["train"])
    if classical and scaled:
        all_angles = StandardScaler().fit_transform(all_angles)
    angles = all_angles[positions]
    risk = q["train"][positions]-.5
    y = arrays["y_train"][positions]
    start = time.perf_counter()
    evaluations = []
    def objective(point):
        alpha,gain = np.exp(point[0]),np.exp(point[1])
        if classical:
            all_risk = q["train"]-.5
            all_value = np.column_stack([all_angles+np.pi*strength*gain*all_risk[:,None]*np.asarray(record["direction"]),np.pi*strength*gain*all_risk])
            value = all_value[positions]
            difference = np.sum((value[:,None,:]-value[None,:,:])**2,axis=2)
            gamma = alpha/(all_value.shape[1]*max(all_value.var(),1e-12))
            k = np.exp(-gamma*difference)
        else:
            engine = RiskStates("reupload",arrays["weights"],record["direction"],entangled,alpha)
            state = engine(np.column_stack([angles,np.pi*strength*gain*risk]))
            k = base.fidelity(state,state)
        score = centered_alignment(k,y)
        evaluations.append({"alpha_or_gamma_factor":float(alpha),"risk_gain":float(gain),"alignment":score})
        return -score
    bounds = protocol["alignment"]["classical_gamma_bounds" if classical else "quantum_alpha_bounds"]
    result = minimize(objective,np.log([1. if classical else .5,1.]),method="Powell",
        bounds=[np.log(bounds),np.log(protocol["alignment"]["risk_gain_bounds"])],
        options={"maxfev":protocol["alignment"]["maxfev"],"xtol":.05,"ftol":.001})
    best = max(evaluations,key=lambda row:row["alignment"])
    return best["alpha_or_gamma_factor"],best["risk_gain"],{"evaluations":evaluations,"selected":best,"optimizer_success":bool(result.success),
        "optimizer_message":str(result.message),"training_positions":positions.tolist(),"seconds":time.perf_counter()-start}


def quantum_representation(arrays,record,method,strength,arm,seed,protocol):
    q = ranks(arrays,arm,seed)
    entangled = arm != "separable"
    alpha,gain,alignment = .5,1.,None
    if method == "aligned_kernel":
        alpha,gain,alignment = fit_alignment(arrays,record,q,strength,seed,protocol,entangled)
    scales = [.25,.5,1.] if method == "multiscale" else [alpha]
    values = {p:[] for p in ("train","tune","gate")}
    state_times = {p:0. for p in values}
    setups = []
    for scale in scales:
        engine = RiskStates(method,arrays["weights"],record["direction"],entangled,scale)
        setups.append({"qubits":engine.qubits,"alpha":scale,"risk_gain":gain,"entangled":entangled,"depth":engine.depth,"gates":engine.gates,"setup_seconds":engine.setup_seconds})
        for part in values:
            tick = time.perf_counter()
            z = arrays[part]
            if method in ("standard_convex","standard_feature_specific"):
                weight = arrays["weights"] if method.endswith("feature_specific") else np.ones(7)
                z = z*(1+strength*arrays[f"risk_{part}"][:,None]*weight)
            angle = encoder_from(record).transform(z)
            state = engine(np.column_stack([angle,np.pi*strength*gain*(q[part]-.5)]))
            if method in ("projected_single","projected_pairs","local_hs"):
                value = observables(state,pairs=method=="projected_pairs")
                if method == "local_hs":
                    bloch = value.reshape(len(value),7,3)
                    local = np.concatenate([np.ones((len(value),7,1)),bloch],axis=2)
                    value = (local/np.linalg.norm(local,axis=2,keepdims=True)).reshape(len(value),-1)/np.sqrt(7)
            else:
                value = state
            values[part].append(value)
            state_times[part] += time.perf_counter()-tick
    kind = "mixture" if method == "multiscale" else "linear" if method == "local_hs" else "rbf" if method.startswith("projected") else "fidelity"
    if kind != "mixture":
        values = {p:v[0] for p,v in values.items()}
    gamma = 1.
    if kind == "rbf":
        scaler = StandardScaler().fit(values["train"])
        values = {p:scaler.transform(v) for p,v in values.items()}
        gamma = 1/(values["train"].shape[1]*max(values["train"].var(),1e-12))
    return {"kind":kind,"values":values,"gamma":gamma}, {"circuits":setups,"representation_seconds":state_times,"alignment":alignment,
        "statevector_rows":{p:len(arrays[p])*len(scales) for p in values},
        "feature_dimension":values["train"][0].shape[1] if kind=="mixture" else values["train"].shape[1]}


def matrix(rep,part):
    train = rep["values"]["train"]
    left = rep["values"][part]
    n = len(left[0]) if rep["kind"]=="mixture" else len(left)
    m = len(train[0]) if rep["kind"]=="mixture" else len(train)
    required = n*m*8
    available = available_memory()
    if available and required+32*2**30 > available:
        raise MemoryError("Exact kernel and reserved memory unavailable")
    result = np.empty((n,m))
    for i in range(0,n,512):
        for j in range(0,m,512):
            if part=="train" and j>i:
                continue
            if rep["kind"]=="mixture":
                value = sum(base.fidelity(x[i:i+512],z[j:j+512]) for x,z in zip(left,train))/len(left)
            elif rep["kind"]=="fidelity":
                value = base.fidelity(left[i:i+512],train[j:j+512])
            elif rep["kind"]=="linear":
                value = left[i:i+512]@train[j:j+512].T
            else:
                x,z = left[i:i+512],train[j:j+512]
                distance = np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(z*z,axis=1)[None,:]-2*x@z.T,0.)
                value = np.exp(-rep["gamma"]*distance)
            result[i:i+len(value),j:j+value.shape[1]] = value
            if part=="train" and i!=j:
                result[j:j+value.shape[1],i:i+len(value)] = value.T
    if not np.isfinite(result).all():
        raise AssertionError("Nonfinite kernel")
    return result


def load_inputs(output,dataset,seed):
    folder = prepared_folder(output,dataset,seed)
    record = json.loads((folder/"audit.json").read_text())
    if sha(folder/"candidate_inputs.npz")!=record["candidate_inputs_sha256"]:
        raise AssertionError("Candidate inputs changed")
    with np.load(folder/"candidate_inputs.npz",allow_pickle=False) as saved:
        arrays = dict(saved)
    if any(k in arrays for k in ("y_gate","y_test","test")):
        raise AssertionError("Held-out labels reached worker")
    record["audit_sha256"] = sha(folder/"audit.json")
    return arrays,record


def job_folder(output,job):
    dataset,seed,method,arm = job
    return Path(output)/"candidates"/dataset/f"seed_{seed:03d}"/method/arm


def fit_kernel(rep,arrays,protocol,fields,details):
    rows,scores = [],{"tune":[],"gate":[]}
    factors = protocol["gamma_factors"] if rep["kind"]=="rbf" else [1.]
    for factor in factors:
        current = {**rep,"gamma":rep["gamma"]*factor}
        kernels,kernel_times = {},{}
        for part in ("train","tune","gate"):
            tick = time.perf_counter()
            kernels[part] = matrix(current,part)
            kernel_times[part] = time.perf_counter()-tick
        for c in protocol["C_grid"]:
            model = SVC(kernel="precomputed",C=c,class_weight="balanced",cache_size=256)
            tick = time.perf_counter()
            model.fit(kernels["train"],arrays["y_train"])
            fit_time = time.perf_counter()-tick
            if model.fit_status_:
                raise RuntimeError("SVC did not converge")
            margins,decision_times = {},{}
            for part in scores:
                tick = time.perf_counter()
                margins[part] = model.decision_function(kernels[part])
                decision_times[part] = time.perf_counter()-tick
                if not np.isfinite(margins[part]).all():
                    raise AssertionError("Nonfinite SVC margin")
            threshold,_ = base.threshold_for_cost(arrays["y_tune"],margins["tune"],0.)
            metrics = base.metrics(arrays["y_tune"],margins["tune"],threshold)
            rows.append({**fields,"C":c,"gamma_factor":factor,"effective_gamma":current["gamma"],"threshold":threshold,**{f"tune_{k}":v for k,v in metrics.items()},
                "fit_rows":len(arrays["y_train"]),"model_fit_calls":1,"ensemble_members":1,
                "fit_seconds":fit_time,"inference_seconds":decision_times["gate"]+kernel_times["gate"]+details.get("representation_seconds",{}).get("gate",0.),
                "training_seconds":fit_time+kernel_times["train"]+details.get("representation_seconds",{}).get("train",0.)+sum(c["setup_seconds"] for c in details.get("circuits",[]))+(details.get("alignment") or {}).get("seconds",0.),
                **{f"kernel_{p}_seconds":v for p,v in kernel_times.items()},**{f"decision_{p}_seconds":v for p,v in decision_times.items()},
                "train_kernel_entries":len(arrays["y_train"])**2,"tune_kernel_entries":len(arrays["y_tune"])*len(arrays["y_train"]),
                "gate_kernel_entries":len(arrays["gate"])*len(arrays["y_train"]),"representation_details":details})
            for p in scores:
                scores[p].append(margins[p])
            del model
        del kernels
        gc.collect()
    return rows,scores


def run_quantum(job,protocol,output,hashes,full=False):
    dataset,seed,method,arm = job
    folder = job_folder(output,job)
    folder.mkdir(parents=True,exist_ok=True)
    with local_lock(folder/"job.lock"),threadpool_limits(limits=1):
        if (folder/"complete.json").exists():
            raise AssertionError("Preserve completed job, no automatic retry")
        arrays,record = load_inputs(output,dataset,seed)
        strengths = [0.,*(protocol["lambdas"] if full else protocol["smoke_lambdas"])]
        if method=="standard":
            strengths = [0.]
        rows,scores = [],{"tune":[],"gate":[]}
        started = time.perf_counter()
        for strength in strengths:
            rep,details = quantum_representation(arrays,record,method,strength,arm,seed,protocol)
            fields = {"dataset":dataset,"seed":seed,"method":method,"arm":arm,"lambda":strength,"learner":"QSVC"}
            current,margins = fit_kernel(rep,arrays,protocol,fields,details)
            rows.extend(current)
            for p in scores:
                scores[p].extend(margins[p])
            del rep
            gc.collect()
        save_job(folder,job,rows,scores,record,hashes,time.perf_counter()-started)
    return job


def save_job(folder,job,rows,scores,record,hashes,elapsed):
    np.savez_compressed(folder/"margins.npz",**{p:np.asarray(v) for p,v in scores.items()})
    write_json(folder/"complete.json",{"job":list(job),"source_sha256":hashes,"trials":rows,
        "prepared_input_sha256":record["candidate_inputs_sha256"],"archive_sha256":sha(folder/"margins.npz"),
        "prepared_audit_sha256":record["audit_sha256"],
        "elapsed_seconds":elapsed,"held_out_labels_read":False})
    print("completed",*job,flush=True)


def classical_representation(arrays,record,variant,strength,scaled,seed,protocol):
    encoder = encoder_from(record)
    values = {}
    factor,gain,alignment = 1.,1.,None
    q = {p:arrays[f"rank_{p}"] for p in ("train","tune","gate")}
    if variant=="aligned_rbf":
        factor,gain,alignment = fit_alignment(arrays,record,q,strength,seed,protocol,classical=True,scaled=scaled)
    angle_scaler = StandardScaler().fit(encoder.transform(arrays["train"])) if scaled else None
    for p in q:
        z = arrays[p]
        if variant.startswith("original_"):
            weights = arrays["weights"] if variant.endswith("feature_specific") else np.ones(7)
            z = z*(1+strength*arrays[f"risk_{p}"][:,None]*weights)
        angle = encoder.transform(z)
        if angle_scaler is not None:
            angle = angle_scaler.transform(angle)
        risk = np.pi*strength*gain*(q[p]-.5)
        if variant=="append":
            value = np.column_stack([angle,risk])
        elif variant in ("shift","aligned_rbf"):
            value = np.column_stack([angle+risk[:,None]*np.asarray(record["direction"]),risk])
        elif variant=="polynomial":
            pairs = np.column_stack([angle[:,j]*angle[:,k] for j in range(7) for k in range(j+1,7)])
            value = np.column_stack([angle,risk,risk[:,None]*angle,risk[:,None]*pairs])
        else:
            value = angle
        values[p] = value
    gamma = factor/(values["train"].shape[1]*max(values["train"].var(),1e-12))
    return {"kind":"rbf","values":values,"gamma":gamma},{"scaled":scaled,"alignment":alignment,"feature_dimension":values["train"].shape[1]}


def run_classical(job,protocol,output,hashes,full=False):
    dataset,seed,_,_ = job
    folder = job_folder(output,job)
    folder.mkdir(parents=True,exist_ok=True)
    with local_lock(folder/"job.lock"),threadpool_limits(limits=1):
        if (folder/"complete.json").exists():
            raise AssertionError("Preserve completed classical job")
        arrays,record = load_inputs(output,dataset,seed)
        strengths = [0.,*(protocol["lambdas"] if full else protocol["smoke_lambdas"])]
        rows,scores = [],{"tune":[],"gate":[]}
        started = time.perf_counter()
        for variant in protocol["classical_variants"]:
            for strength in strengths if variant=="aligned_rbf" else [0.] if variant=="baseline" else strengths[1:]:
                for scaled in (False,True):
                    rep,details = classical_representation(arrays,record,variant,strength,scaled,seed,protocol)
                    fields = {"dataset":dataset,"seed":seed,"method":variant,"arm":"classical","lambda":strength,"learner":"RBF-SVM","scaled":scaled}
                    current,margins = fit_kernel(rep,arrays,protocol,fields,details)
                    rows.extend(current)
                    for p in scores:
                        scores[p].extend(margins[p])
                    del rep
        for appended in (False,True):
            x = {p:np.column_stack([arrays[p],arrays[f"rank_{p}"]]) if appended else arrays[p] for p in ("train","tune","gate")}
            for learner,depth in [("RF",6),("RF",None),("XGBoost",3),("XGBoost",5)]:
                model = RandomForestClassifier(n_estimators=200,max_depth=depth,class_weight="balanced",n_jobs=1,random_state=seed) if learner=="RF" else XGBClassifier(n_estimators=200,max_depth=depth,learning_rate=.05,subsample=1.,colsample_bytree=1.,n_jobs=1,random_state=seed,tree_method="hist",eval_metric="logloss",scale_pos_weight=float(np.sum(arrays["y_train"]==0)/np.sum(arrays["y_train"]==1)))
                tick = time.perf_counter()
                model.fit(x["train"],arrays["y_train"])
                fit_time = time.perf_counter()-tick
                tick = time.perf_counter()
                margins = {p:model.predict_proba(x[p])[:,1] for p in scores}
                infer = time.perf_counter()-tick
                threshold,_ = base.threshold_for_cost(arrays["y_tune"],margins["tune"],1/6)
                metrics = base.metrics(arrays["y_tune"],margins["tune"],threshold)
                rows.append({"dataset":dataset,"seed":seed,"method":"tree_rank" if appended else "tree_base","arm":"classical","learner":learner,"depth":depth,"lambda":1. if appended else 0.,"C":0.,"gamma_factor":0.,"threshold":threshold,**{f"tune_{k}":v for k,v in metrics.items()},"fit_rows":len(arrays["y_train"]),"model_fit_calls":1,"fit_seconds":fit_time,"inference_seconds":infer})
                for p in scores:
                    scores[p].append(margins[p])
        threshold,_ = base.threshold_for_cost(arrays["y_tune"],arrays["risk_tune"],1/6)
        rows.append({"dataset":dataset,"seed":seed,"method":"LR_score","arm":"classical","learner":"LR","lambda":0.,"C":0.,"gamma_factor":0.,"threshold":threshold,**{f"tune_{k}":v for k,v in base.metrics(arrays["y_tune"],arrays["risk_tune"],threshold).items()},"fit_rows":len(arrays["y_train"]),"model_fit_calls":0})
        for p in scores:
            scores[p].append(arrays[f"risk_{p}"])
        save_job(folder,job,rows,scores,record,hashes,time.perf_counter()-started)
    return job
