from pathlib import Path
import json,hashlib,sys,datetime
import numpy as np,pandas as pd,joblib
sys.path.insert(0,"code")
from _shared import professor_followup as p
root=p.ROOT/"code/results/cmc_professor_followup_20261022"
meta=p.verify_sources(root)
assert meta["status"]=="calibration_completed" and meta["completed_jobs"]==0
labels=p.test_labels()
stored=pd.read_csv(root/"calibration_metrics.csv")
probability=pd.read_csv(root/"calibration_probability_metrics.csv")
map_checks=metric_checks=fold_checks=0
for dataset in meta["protocol"]["datasets"]:
 for seed in meta["protocol"]["seeds"]:
  arrays,audit,split,original=p.inputs(dataset,seed)
  p.audit_oof(original,arrays,lambda v:v)
  fold_checks+=1
  folder=root/"calibration"/dataset/f"seed_{seed:04d}"
  complete=json.loads((folder/"complete.json").read_text())
  assert p.sha(folder/"scores.npz")==complete["scores_sha256"]
  with np.load(folder/"scores.npz",allow_pickle=False) as archive:scores=dict(archive)
  y=labels[labels.dataset.eq(dataset)&labels.seed.eq(seed)].set_index("row_id").reindex(split["test"]).y.to_numpy(dtype=int)
  for method in ("raw","sigmoid","isotonic"):
   model=None if method=="raw" else joblib.load(folder/(method+".joblib"))
   for part in ("tune","gate"):
    base=arrays["gbdt_"+part]
    expected=base if method=="raw" else model.predict_proba(base[:,None])[:,1] if method=="sigmoid" else model.predict(base)
    np.testing.assert_array_equal(scores[method+"_"+part],expected)
    map_checks+=1
   actual,_=p.probability_metrics(y,scores[method+"_gate"])
   row=probability[probability.dataset.eq(dataset)&probability.seed.eq(seed)&probability.calibration.eq(method)]
   assert len(row)==1
   for key,value in actual.items():assert np.isclose(row[key].iloc[0],value)
  for choice in complete["selections"]:
   for part,yy,suffix in (("validation",arrays["y_tune"],"tune"),("test",y,"gate")):
    actual=p.metrics(yy,scores[choice["calibration"]+"_"+suffix],choice["threshold"])
    row=stored[stored.dataset.eq(dataset)&stored.seed.eq(seed)&stored.calibration.eq(choice["calibration"])&stored.threshold_policy.eq(choice["threshold_policy"])&stored.part.eq(part)]
    assert len(row)==1
    for key,value in actual.items():assert np.isclose(row[key].iloc[0],value),(dataset,seed,key)
    metric_checks+=1
stats=pd.read_csv(root/"available_paired_statistics.csv")
diffs=pd.read_csv(root/"available_seed_differences.csv")
for row in stats.itertuples():
 selected=diffs[diffs.dataset.eq(row.dataset)&diffs.kernel.eq(row.kernel)&diffs.method.eq(row.method)&diffs.contrast.eq(row.contrast)].sort_values("seed")
 assert selected.seed.tolist()==meta["protocol"]["seeds"]
 check=p.corrected(selected.delta.to_numpy())
 for key in ("mean_difference","sd_difference","ci_low","ci_high","p_value"):
  assert np.isclose(getattr(row,key),check[key],equal_nan=True)
secondary=stats[stats.testing_family.eq("secondary_calibration")]
assert len(secondary)==40 and np.allclose(secondary.p_holm,p.holm(secondary.p_value.to_numpy()))
primary=stats[stats.testing_family.eq("primary_cost")]
assert len(primary)==80 and primary.p_holm.isna().all() and not primary.complete_family.any()
receipt={"status":"passed_calibration_and_existing_results_only","utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"fold_audits":fold_checks,"heldout_map_predictions_checked":map_checks,"threshold_metric_rows_recalculated":metric_checks,"probability_metric_rows_recalculated":120,"available_contrasts_recalculated":120,"primary_cost_family_pending":20,"sources_checked":len(meta["source_sha256"]),"parent_inputs_checked":len(meta["parent_sha256"]),"new_remote_jobs_completed":0,"automatic_scientific_retries":False,"scope":"All40 fitted calibration maps replayed on saved heldout scores; all stored metrics/statistics independently recalculated, not independent XGB/SVM retraining."}
p.write(root/"calibration_independent_validation_receipt.json",receipt,True)
print(json.dumps(receipt))

