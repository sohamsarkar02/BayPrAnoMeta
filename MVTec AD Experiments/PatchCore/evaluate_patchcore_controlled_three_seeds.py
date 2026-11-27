#!/usr/bin/env python3

import argparse,csv,time
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score,f1_score,roc_auc_score
from patchcore_controlled_common import *
FIELDS=["method","category","held_out_subtype","split_seed","run_seed","coreset_fraction","candidate_memory_size","requested_coreset_fraction","actual_coreset_size","actual_fraction","memory_bank_source","memory_bank_test_overlap","AUROC","AUPRC","F1","AUROC_se","AUPRC_se","F1_se","original_AUROC","reversed_score_AUROC","below_chance_diagnostic_triggered","diagnosis"]
def csvwrite(path,rows,fields):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
def image_score(patches,bank,chunk=2048):
 nearest=[]
 for start in range(0,len(bank),chunk):nearest.append(np.linalg.norm(patches[:,None]-bank[None,start:start+chunk],axis=2).min(axis=1))
 return float(np.min(np.stack(nearest),axis=0).max())
def primary(y,s):
 ts=np.unique(s);return float(roc_auc_score(y,s)),float(average_precision_score(y,s)),float(max(f1_score(y,s>=t,zero_division=0) for t in ts))
def bootstrap(y,s,seed,n=200):
 rng=np.random.default_rng(seed);vals=[]
 for _ in range(n):
  idx=rng.choice(len(y),len(y),replace=True)
  if len(np.unique(y[idx]))==2:vals.append(primary(y[idx],s[idx]))
 a=np.asarray(vals);return [float(a[:,i].std(ddof=1)/np.sqrt(len(a))) for i in range(3)]
def desc(x):return {"mean":float(np.mean(x)),"median":float(np.median(x)),"std":float(np.std(x))}
def evaluate_seed(seed,a):
 set_seed(seed);model=SharedEmbeddingNet().to(DEVICE);state=Path(a.result_root)/"seed_models"/f"seed_{seed}_encoder.pth";model.load_state_dict(torch.load(state,map_location=DEVICE));model.eval();mon=Monitor();mon.start();started=time.perf_counter();per_fraction={f:[] for f in a.fractions}
 for ci,c in enumerate(CLIENTS):
  d=load_category(Path(a.base_root)/c,a.split_seed)
  if d is None:continue
  paths=d["test_normal"]+d["test_anomaly"];y=np.asarray([0]*len(d["test_normal"])+[1]*len(d["test_anomaly"]));patches=[]
  with torch.no_grad():
   for p in paths:patches.append(model(load_tensor(p)).cpu().numpy())
  for fraction in a.fractions:
   root=Path(a.result_root)/f"fraction_{FRACTION_KEYS[fraction]}"/f"seed_{seed}";bank=np.load(root/f"{c}_memory.npy")
   with (root/f"{c}_memory_metadata.json").open() as f:meta=json.load(f)
   source=set(meta["memory_bank_source_paths"]);overlap=len(source&set(paths))
   if overlap or meta["memory_bank_source"]!="train/good":raise RuntimeError(f"{c} fraction={fraction}: invalid memory bank source/overlap")
   scores=np.asarray([image_score(x,bank) for x in patches]);
   if not np.isfinite(scores).all():raise RuntimeError(f"{c} fraction={fraction}: non-finite scores")
   au,ap,f1=primary(y,scores);ause,apse,f1se=bootstrap(y,scores,seed*100000+ci*1000+int(fraction*100));trigger=bool(au<a.below_chance_threshold);diagnosis="BELOW_CHANCE_REQUIRES_PERFORMANCE_REVIEW_NOT_AUTOMATIC_SCORE_FLIP" if trigger else None
   row={"method":"Controlled PatchCore","category":c,"held_out_subtype":d["held_out_subtype"],"split_seed":a.split_seed,"run_seed":seed,"coreset_fraction":fraction,"candidate_memory_size":meta["candidate_memory_size"],"requested_coreset_fraction":fraction,"actual_coreset_size":len(bank),"actual_fraction":len(bank)/meta["candidate_memory_size"],"memory_bank_source":"train/good","memory_bank_test_overlap":overlap,"AUROC":au,"AUPRC":ap,"F1":f1,"AUROC_se":ause,"AUPRC_se":apse,"F1_se":f1se,"original_AUROC":au,"reversed_score_AUROC":float(roc_auc_score(y,-scores)),"below_chance_diagnostic_triggered":trigger,"diagnosis":diagnosis};per_fraction[fraction].append(row)
   out=root/"evaluation"/c;records=[{"image_path":p,"label":int(label),"anomaly_score":float(score)} for p,label,score in zip(paths,y,scores)];csvwrite(out/"query_scores.csv",records,["image_path","label","anomaly_score"]);write(out/"metrics.json",row);write(out/"test_manifest.json",{"category":c,"held_out_subtype":d["held_out_subtype"],"normal_paths":d["test_normal"],"anomaly_paths":d["test_anomaly"]})
   if trigger:write(out/"below_chance_diagnostic.json",{**row,"label_convention":{"normal":0,"anomaly":1},"score_direction":"larger nearest-normal-patch distance means more anomalous","primary_score_was_not_modified":True,"normal_score":desc(scores[y==0]),"anomaly_score":desc(scores[y==1])})
 for fraction,rows in per_fraction.items():
  root=Path(a.result_root)/f"fraction_{FRACTION_KEYS[fraction]}"/f"seed_{seed}"/"evaluation";csvwrite(root/"results.csv",rows,FIELDS);write(root/"results.json",rows)
 mon.stop();write(Path(a.result_root)/f"seed_{seed}_evaluation_runtime_memory.json",{"run_seed":seed,"evaluation_sec":time.perf_counter()-started,"peak_rss_mb":mon.peak});return per_fraction
def aggregate(allr,seeds,a):
 for fraction in a.fractions:
  root=Path(a.result_root)/f"fraction_{FRACTION_KEYS[fraction]}";combined=[r for s in seeds for r in allr[s][fraction]];csvwrite(root/"three_seed_results.csv",combined,FIELDS);by={(r["run_seed"],r["category"]):r for r in combined};cats=set.intersection(*(set(r["category"] for r in allr[s][fraction]) for s in seeds));rows=[]
  for c in sorted(cats):
   held={by[s,c]["held_out_subtype"] for s in seeds}
   if len(held)!=1:raise RuntimeError(f"Held-out subtype mismatch: {c}")
   row={"method":"Controlled PatchCore","category":c,"held_out_subtype":held.pop(),"coreset_fraction":fraction}
   for metric in ("AUROC","AUPRC","F1"):
    vals=np.asarray([by[s,c][metric] for s in seeds]);row.update({f"seed{s}_{metric}":float(by[s,c][metric]) for s in seeds});row[f"mean_{metric}"]=float(vals.mean());row[f"std_{metric}"]=float(vals.std(ddof=1));
   vals=np.asarray([by[s,c]["AUROC"] for s in seeds]);row["any_seed_below_0_50"]=bool(np.any(vals<.5));row["mean_below_0_50"]=bool(vals.mean()<.5);rows.append(row)
  fields=["method","category","held_out_subtype","coreset_fraction"]+sum(([f"seed{s}_{m}" for s in seeds]+[f"mean_{m}",f"std_{m}"] for m in ("AUROC","AUPRC","F1")),[])+["any_seed_below_0_50","mean_below_0_50"];csvwrite(root/"three_seed_summary.csv",rows,fields);write(root/"three_seed_summary.json",rows)
def main():
 p=argparse.ArgumentParser();p.add_argument("--seeds",nargs="+",type=int,default=[1,2,3]);p.add_argument("--fractions",nargs="+",type=float,default=list(FRACTIONS));p.add_argument("--split-seed",type=int,default=SPLIT_SEED);p.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None,default=BASE_ROOT);p.add_argument("--result-root",default=RESULT_ROOT);p.add_argument("--below-chance-threshold",type=float,default=.50);p.add_argument("--debug-protocol",action="store_true");a=p.parse_args();a.fractions=tuple(a.fractions)
 if any(f not in FRACTIONS for f in a.fractions):raise ValueError(f"Fractions must be selected from {FRACTIONS}")
 allr={}
 for seed in a.seeds:print(f"\n=== Controlled PatchCore evaluation seed {seed} ===");allr[seed]=evaluate_seed(seed,a)
 aggregate(allr,a.seeds,a)
if __name__=="__main__":main()
