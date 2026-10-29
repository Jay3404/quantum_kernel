"""Ten quantum mechanisms: smoke gate followed only by eligible full runs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np
import pandas as pd

from _shared import controlled as base
from _shared.angle_encoding import AngleEncoding
from _shared.full_data import write_json
from _shared.improved import local_lock, sha
from _shared.quantum_ten import METHODS, run_quantum, run_classical
from _shared.quantum_ten_reporting import collect, select, evaluate, report
from _shared.separation_round2 import midrank, prepare, prepared_folder
from run_improved_experiment import supervise

ROOT = base.ROOT
PROTOCOL = ROOT/"docs/cmc_quantum_ten_protocol_20261018.json"
SOURCES = ("code/run_quantum_ten.py","code/_shared/quantum_ten.py","code/_shared/quantum_ten_reporting.py",
    "code/_shared/separation_round2.py","code/_shared/separation.py","code/_shared/angle_encoding.py",
    "code/_shared/single_811.py","code/_shared/controlled.py","code/_shared/full_data.py","code/_shared/improved.py",
    "code/build_controlled_results.py","code/run_improved_experiment.py","docs/cmc_quantum_ten_protocol_20261018.json")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())


def jobs(protocol,methods,seeds,full):
    result = []
    for dataset,seed in itertools.product(protocol["datasets"],seeds):
        result.append((dataset,seed,"classical","classical"))
        for method in methods:
            result.extend((dataset,seed,method,arm) for arm in protocol["arms"])
        result.append((dataset,seed,"standard","quantum"))
        if full:
            result.extend((dataset,seed,"standard_"+rep,"quantum") for rep in ("convex","feature_specific"))
    return result


def prepare_full(dataset,seed,protocol,output):
    folder = prepared_folder(output,dataset,seed)
    folder.mkdir(parents=True,exist_ok=True)
    source = ROOT/"code"/dataset/"results"/protocol["source_run"]/f"seed_{seed:03d}"
    receipt = json.loads((source/"prepared.json").read_text())
    if sha(source/"prepared.npz")!=receipt["archive_sha256"]:
        raise AssertionError("Full training source cache changed")
    with np.load(source/"prepared.npz",allow_pickle=False) as old:
        inputs = {"train":old["train"],"tune":old["validation"],"gate":old["test"],"y_train":old["y_train"],"y_tune":old["y_validation"],"weights":old["weights"]}
        for new,original in (("train","train"),("tune","validation"),("gate","test")):
            inputs[f"risk_{new}"] = old[f"risk_{original}"]
    encoder = AngleEncoding.fit(inputs["train"],"linear_no_clip")
    angles = encoder.transform(inputs["train"])
    risk = inputs["risk_train"]
    direction = (angles-angles.mean(0)).T@(risk-risk.mean())
    norm = np.linalg.norm(direction)
    direction = direction/norm if norm>1e-12 else np.ones(7)/np.sqrt(7)
    for p in ("train","tune","gate"):
        inputs[f"rank_{p}"] = midrank(risk,inputs[f"risk_{p}"])
    np.savez_compressed(folder/"candidate_inputs.npz",**inputs)
    audit = {"dataset":dataset,"seed":seed,"encoder":encoder.parameters(),"direction":direction.tolist(),"risk_mean":float(risk.mean()),
             "gamma":1/(7*max(angles.var(),1e-12)),"candidate_inputs_sha256":sha(folder/"candidate_inputs.npz"),
             "candidate_loaded_keys":list(inputs),"source_audit":{"prepared_sha256":receipt["archive_sha256"],"source_receipt_sha256":sha(source/"prepared.json"),"source_times":receipt["times"]},
             "split_indices_sha256":sha(source/"split_indices.json"),"full_training_rows":len(risk),"test_labels_read":False}
    write_json(folder/"audit.json",audit)
    return dataset,seed


def verify_phase(output,stage,protocol,methods,seeds,hashes):
    tasks = jobs(protocol,methods,seeds,stage=="full")
    groups,receipts,_ = collect(output,tasks,hashes)
    choices,profiles = select(groups,protocol["datasets"],seeds,methods,protocol["lambdas"] if stage=="full" else protocol["smoke_lambdas"])
    lock = json.loads((output/"selection_lock.json").read_text())
    if choices!=lock["models"] or profiles!=lock["lambda_profiles"] or receipts!=lock["candidate_receipts_sha256"]:
        raise AssertionError("Stored tune-only choice lock does not reproduce")
    rebuilt,_ = evaluate(output,choices,stage=="full",protocol,with_predictions=False)
    saved = pd.read_csv(output/"seed_metrics.csv",float_precision="round_trip")
    for key in ("cost","fn","fp","tn","tp","f1","recall","balanced_accuracy","roc_auc","threshold","fit_rows"):
        np.testing.assert_allclose(rebuilt[key],saved[key],atol=1e-12,rtol=1e-12)
    receipt = json.loads((output/"verification_receipt.json").read_text())
    for name,value in receipt["outputs_sha256"].items():
        if sha(output/name)!=value:
            raise AssertionError("Reported output hash differs")
    return {"stage":stage,"jobs":len(tasks),"selected_models":len(choices),"profile_models":len(profiles),"scope":"Source/lock, complete grid, same-fit stored metrics; not independent retraining"}


def execute(args):
    protocol = json.loads(PROTOCOL.read_text())
    if tuple(protocol["methods"])!=METHODS or protocol["lambdas"]!=[i/20 for i in range(1,21)]:
        raise AssertionError("Keep all ten predeclared methods and full positive grid")
    output = ROOT/"code/results"/args.run_id
    output.mkdir(parents=True,exist_ok=True)
    hashes = {name:sha(ROOT/name) for name in SOURCES}
    frozen = json.loads((ROOT/"code/results/cmc_20261018_fico_single_811/preflight.json").read_text())["input_hashes"]
    if any(sha(ROOT/name)!=value for name,value in frozen.items()):
        raise AssertionError("FICO source/input changed; leave prior run intact")
    metadata = {"run_id":args.run_id,"status":"running","phase":"smoke_prepare","pid":os.getpid(),"started_utc":now(),
                "protocol":protocol,"source_sha256":hashes,"preserved_fico_inputs":frozen,"smoke_workers":args.workers,"full_workers":4,
                "full_launched":False,"packages":{n:importlib.metadata.version(n) for n in ("numpy","scipy","scikit-learn","qiskit","qiskit-aer","xgboost")}}
    path = output/"metadata.json"
    with local_lock(output/"run.lock"):
        if path.exists():
            raise AssertionError("Preserve existing run; no automatic retries")
        write_json(path,metadata)
        with tarfile.open(output/"source_snapshot.tar.gz","w:gz") as archive:
            for name in SOURCES:
                archive.add(ROOT/name,arcname=name)
        try:
            promoted = []
            for stage in ("smoke","full"):
                if stage=="full" and not promoted:
                    break
                full = stage=="full"
                methods = promoted if full else protocol["methods"]
                seeds = protocol["seeds"] if full else protocol["smoke_seeds"]
                stage_output = output/stage
                stage_output.mkdir(exist_ok=True)
                metadata.update(phase=stage+"_prepare",full_launched=full,active_methods=methods,active_seeds=seeds,completed_jobs=0,failed_jobs=[])
                write_json(path,metadata)
                tasks = jobs(protocol,methods,seeds,full)
                with ProcessPoolExecutor(max_workers=4 if full else args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
                    preparations = [pool.submit(prepare_full if full else prepare,d,s,protocol,stage_output) for d,s in itertools.product(protocol["datasets"],seeds)]
                    for future in as_completed(preparations):
                        future.result()
                    metadata.update(phase=stage+"_search",expected_jobs=len(tasks))
                    write_json(path,metadata)
                    pending = {pool.submit(run_classical if job[2]=="classical" else run_quantum,job,protocol,stage_output,hashes,full):job for job in tasks}
                    for future in as_completed(pending):
                        try:
                            future.result()
                        except Exception as error:
                            metadata["failed_jobs"].append({"job":pending[future],"error":f"{type(error).__name__}: {error}"})
                            print("failed",metadata["failed_jobs"][-1],flush=True)
                        else:
                            metadata["completed_jobs"]+=1
                        write_json(path,metadata)
                if metadata["failed_jobs"]:
                    raise RuntimeError("Failures preserved; no automatic retry or full promotion")
                if hashes!={name:sha(ROOT/name) for name in SOURCES}:
                    raise AssertionError("Scientific code or protocol changed during run")
                metadata["phase"] = stage+"_reporting"
                write_json(path,metadata)
                outcome = report(stage_output,tasks,hashes,protocol,methods,seeds,full)
                audit = verify_phase(stage_output,stage,protocol,methods,seeds,hashes)
                metadata[stage+"_audit"] = audit
                if not full:
                    promoted = outcome["promoted_methods"]
                    metadata["promoted_methods"] = promoted
                    metadata["smoke_finished_utc"] = now()
                    write_json(path,metadata)
            metadata.update(status="completed",phase="full_completed" if promoted else "smoke_completed_no_promotion",finished_utc=now())
        except BaseException as error:
            metadata.update(status="failed",error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write_json(path,metadata)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",default="cmc_20261018_quantum_ten")
    parser.add_argument("--workers",type=int,default=12)
    parser.add_argument("--launch",action="store_true")
    parser.add_argument("--supervisor",action="store_true")
    parser.add_argument("--verify",action="store_true")
    args = parser.parse_args()
    if Path(args.run_id).name!=args.run_id or not 1<=args.workers<=12:
        parser.error("Use one run name and 1..12 workers")
    command = [sys.executable,"-u",str(Path(__file__).resolve()),"--run-id",args.run_id,"--workers",str(args.workers)]
    if args.verify:
        root = ROOT/"code/results"/args.run_id
        meta = json.loads((root/"metadata.json").read_text())
        print(json.dumps(verify_phase(root/"smoke","smoke",meta["protocol"],meta["protocol"]["methods"],meta["protocol"]["smoke_seeds"],meta["source_sha256"]),ensure_ascii=False))
        if meta["full_launched"] and meta["status"]=="completed":
            print(json.dumps(verify_phase(root/"full","full",meta["protocol"],meta["promoted_methods"],meta["protocol"]["seeds"],meta["source_sha256"]),ensure_ascii=False))
    elif args.launch:
        folder = ROOT/"code/results"/args.run_id
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/"metadata.json").exists() or (folder/"monitor.json").exists():
            raise RuntimeError("Existing experiment retained")
        with (folder/"server.log").open("a") as log:
            process = subprocess.Popen([*command,"--supervisor"],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({"run_id":args.run_id,"supervisor_pid":process.pid}))
    elif args.supervisor:
        args.blas_threads = 1
        supervise(args,command)
    else:
        execute(args)
