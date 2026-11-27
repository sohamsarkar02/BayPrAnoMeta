#!/usr/bin/env python3
"""Evaluate Fixed-Lambda, Proto-MAML with Bayesian scoring, and the full model."""
import argparse,random,time
from pathlib import Path
import higher,numpy as np,torch
from scipy.stats import kendalltau,spearmanr
from sklearn.metrics import roc_auc_score
from ablation_common import *

def adapt_and_score(model,mode,xs,xq,debug):
 opt=torch.optim.SGD(model.parameters(),lr=INNER_LR)
 with higher.innerloop_ctx(model,opt,copy_initial_weights=True,track_higher_grads=False) as (fm,diff):
  for _ in range(INNER_STEPS):
   zs=fm(xs)
   if mode=="proto":proto=zs.mean(0,keepdim=True);inner=((zs-proto)**2).sum(1).mean()
   else:
    posterior=fixed_lambda_posterior if mode=="fixed_lambda" else niw_posterior;mu,sigma,dof=posterior(zs);inner=-log_student_t(zs,mu,sigma,dof).mean()
   diff.step(inner)
  zs=fm(xs);zq=fm(xq);proto=zs.mean(0,keepdim=True);posterior=fixed_lambda_posterior if mode=="fixed_lambda" else niw_posterior;mu,sigma,dof=posterior(zs);logp0=log_student_t(zq,mu,sigma,dof);logp1=reference_log_probability(zq);bayes=logp1-logp0;euclid=((zq-proto)**2).sum(1)
 if debug:
  assert torch.isfinite(torch.cat([logp0,logp1,bayes,euclid])).all()
 return {"embedding":zq.detach().cpu().numpy(),"norm_sq":(zq**2).sum(1).detach().cpu().numpy(),"logp0":logp0.detach().cpu().numpy(),"logp1":logp1.detach().cpu().numpy(),"bayesian_score":bayes.detach().cpu().numpy(),"euclidean_score":euclid.detach().cpu().numpy()}

def evaluate_method(model,mode,category,data,seed,a):
 rng=random.Random(seed+2);episode_au=[];episode_pr=[];episode_f1=[];episode_reverse=[];episode_eu=[];records=[];manifest=[]
 for eid in range(a.test_episodes):
  support,qn,qa=sample_episode(data,"test",rng);paths=qn+qa;y=np.asarray([0]*Q_N+[1]*Q_A);xs=load_stack(support,transform_eval);xq=load_stack(paths,transform_eval);v=adapt_and_score(model,mode,xs,xq,a.debug_protocol);au,pr,f1=episode_metrics(v["bayesian_score"],y);episode_au.append(au);episode_pr.append(pr);episode_f1.append(f1);episode_reverse.append(float(roc_auc_score(y,-v["bayesian_score"])));episode_eu.append(float(roc_auc_score(y,v["euclidean_score"])));manifest.append({"episode_id":eid,"support":support,"normal_query":qn,"anomaly_query":qa})
  records.extend({"method":{"full":"BayPrAnoMeta","fixed_lambda":"BayPrAnoMeta-FixedLambda","proto":"Proto-MAML+Bayesian-score"}[mode],"run_seed":seed,"split_seed":a.split_seed,"category":category,"held_out_subtype":data["held_out_subtype"],"episode_id":eid,"image_path":p,"label":int(label),"embedding_norm_sq":float(n2),"logp0":float(p0),"logp1":float(p1),"final_score":float(bs),"euclidean_score":float(es)} for p,label,n2,p0,p1,bs,es in zip(paths,y,v["norm_sq"],v["logp0"],v["logp1"],v["bayesian_score"],v["euclidean_score"]))
 labels=np.asarray([r["label"] for r in records]);bs=np.asarray([r["final_score"] for r in records]);es=np.asarray([r["euclidean_score"] for r in records]);rho=spearmanr(bs,es).statistic;tau=kendalltau(bs,es).statistic
 triggered=bool(np.mean(episode_au)<.5);result={"method":records[0]["method"],"category":category,"held_out_subtype":data["held_out_subtype"],"run_seed":seed,"split_seed":a.split_seed,"AUROC":float(np.mean(episode_au)),"AUROC_se":float(np.std(episode_au,ddof=1)/np.sqrt(len(episode_au))),"AUPRC":float(np.mean(episode_pr)),"AUPRC_se":float(np.std(episode_pr,ddof=1)/np.sqrt(len(episode_pr))),"F1":float(np.mean(episode_f1)),"F1_se":float(np.std(episode_f1,ddof=1)/np.sqrt(len(episode_f1))),"reversed_score_AUROC":float(np.mean(episode_reverse)),"below_chance_diagnostic_triggered":triggered,"diagnosis":"BELOW_CHANCE_REQUIRES_PERFORMANCE_REVIEW_NOT_AUTOMATIC_SCORE_FLIP" if triggered else None,"euclidean_AUROC":float(np.mean(episode_eu)),"pooled_bayesian_AUROC":float(roc_auc_score(labels,bs)),"pooled_euclidean_AUROC":float(roc_auc_score(labels,es)),"spearman_rho":float(rho),"kendall_tau":float(tau),"test_episodes":a.test_episodes,"support_query_disjoint":all(set(m["support"]).isdisjoint(m["normal_query"]) for m in manifest)}
 return result,records,manifest

def spread_rows(records,seed,split_seed,category,held):
 rows=[];labels=np.asarray([r["label"] for r in records])
 for group,mask in (("normal",labels==0),("anomaly",labels==1),("all",np.ones(len(labels),dtype=bool))):
  row={"method":"BayPrAnoMeta","run_seed":seed,"split_seed":split_seed,"category":category,"held_out_subtype":held,"query_group":group,"n_query_records":int(mask.sum())}
  for source,prefix in (("embedding_norm_sq","norm_sq"),("logp0","logp0"),("logp1","logp1"),("final_score","final_score")):
   d=descriptive([r[source] for r,m in zip(records,mask) if m]);row.update({f"{prefix}_{k}":v for k,v in d.items()})
  rows.append(row)
 return rows

def evaluate_seed(seed,a):
 set_seed(seed);clients=load_clients(a.base_root,a.split_seed,a.validation_fraction);cks=checkpoints(seed,a.full_run_root,a.proto_run_root,a.fixed_run_root)
 for name,path in cks.items():
  if not path.exists():raise FileNotFoundError(f"Missing {name} checkpoint: {path}")
 models={name:EmbeddingNet().to(DEVICE) for name in cks}
 for name,model in models.items():model.load_state_dict(torch.load(cks[name],map_location=DEVICE));model.eval()
 out=Path(a.output_root)/f"seed_{seed}";results=[];spread=[];rank=[];started=time.perf_counter();monitor=PeakMemoryMonitor();monitor.start()
 for category,data in clients.items():
  method_outputs={}
  for mode in ("full","fixed_lambda","proto"):
   result,records,manifest=evaluate_method(models[mode],mode,category,data,seed,a);method_outputs[mode]=result;results.append(result);method_dir=out/mode/category;csvwrite(method_dir/"query_values.csv",records);write_json(method_dir/"episode_manifest.json",manifest);write_json(method_dir/"metrics.json",result)
   if mode=="full":spread.extend(spread_rows(records,seed,a.split_seed,category,data["held_out_subtype"]));rank.append({"method":"BayPrAnoMeta","category":category,"held_out_subtype":data["held_out_subtype"],"run_seed":seed,"split_seed":a.split_seed,"bayesian_AUROC":result["AUROC"],"euclidean_AUROC":result["euclidean_AUROC"],"spearman_rho":result["spearman_rho"],"kendall_tau":result["kendall_tau"]})
   if mode=="proto":rank.append({"method":"Proto-MAML encoder","category":category,"held_out_subtype":data["held_out_subtype"],"run_seed":seed,"split_seed":a.split_seed,"bayesian_AUROC":result["AUROC"],"euclidean_AUROC":result["euclidean_AUROC"],"spearman_rho":result["spearman_rho"],"kendall_tau":result["kendall_tau"]})
 monitor.stop();csvwrite(out/"method_results.csv",results);csvwrite(out/"p0_p1_embedding_spread.csv",spread);csvwrite(out/"bayesian_vs_euclidean_rank.csv",rank);write_json(out/"runtime_memory.json",{"evaluation_sec":time.perf_counter()-started,"peak_rss_mb":monitor.peak_mb});return results,spread,rank

def aggregate(all_results,all_spread,all_rank,seeds,a):
 out=Path(a.output_root);csvwrite(out/"p0_p1_embedding_spread.csv",[r for s in seeds for r in all_spread[s]]);rank_rows=[r for s in seeds for r in all_rank[s]];summary_rank=[]
 for method in sorted(set(r["method"] for r in rank_rows)):
  for c in sorted(set(r["category"] for r in rank_rows if r["method"]==method)):
   rows=[r for r in rank_rows if r["method"]==method and r["category"]==c];held={r["held_out_subtype"] for r in rows}
   if len(held)!=1:raise RuntimeError("Held-out subtype mismatch in rank aggregation")
   summary_rank.append({"method":method,"category":c,"held_out_subtype":held.pop(),"bayesian_AUROC_mean":float(np.mean([r["bayesian_AUROC"] for r in rows])),"bayesian_AUROC_std":float(np.std([r["bayesian_AUROC"] for r in rows],ddof=1)),"euclidean_AUROC_mean":float(np.mean([r["euclidean_AUROC"] for r in rows])),"euclidean_AUROC_std":float(np.std([r["euclidean_AUROC"] for r in rows],ddof=1)),"spearman_rho_mean":float(np.mean([r["spearman_rho"] for r in rows])),"spearman_rho_std":float(np.std([r["spearman_rho"] for r in rows],ddof=1)),"kendall_tau_mean":float(np.mean([r["kendall_tau"] for r in rows])),"kendall_tau_std":float(np.std([r["kendall_tau"] for r in rows],ddof=1))})
 csvwrite(out/"bayesian_vs_euclidean_rank.csv",rank_rows);csvwrite(out/"bayesian_vs_euclidean_rank_three_seed_summary.csv",summary_rank)
 flat={s:{(r["method"],r["category"]):r for r in all_results[s]} for s in seeds};cats=set.intersection(*(set(r["category"] for r in all_results[s]) for s in seeds));rows=[];mapping={"full":"BayPrAnoMeta","fixed_lambda":"BayPrAnoMeta-FixedLambda","proto_bayes":"Proto-MAML+Bayesian-score"}
 for c in sorted(cats):
  held={flat[s][("BayPrAnoMeta",c)]["held_out_subtype"] for s in seeds};held|={flat[s][("BayPrAnoMeta-FixedLambda",c)]["held_out_subtype"] for s in seeds};held|={flat[s][("Proto-MAML+Bayesian-score",c)]["held_out_subtype"] for s in seeds}
  if len(held)!=1:raise RuntimeError(f"Held-out subtype mismatch for {c}")
  row={"category":c,"held_out_subtype":held.pop()}
  for prefix,method in mapping.items():
   vals=np.asarray([flat[s][(method,c)]["AUROC"] for s in seeds]);row.update({f"{prefix}_seed{s}_AUROC":float(flat[s][(method,c)]["AUROC"]) for s in seeds});row[f"{prefix}_mean_AUROC"]=float(vals.mean());row[f"{prefix}_std_AUROC"]=float(vals.std(ddof=1))
  rows.append(row)
 csvwrite(out/"bayesian_ablation_summary.csv",rows);write_json(out/"bayesian_ablation_summary.json",rows)

def main():
 p=argparse.ArgumentParser();p.add_argument("--seeds",nargs="+",type=int,default=[1,2,3]);p.add_argument("--split-seed",type=int,default=SPLIT_SEED);p.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None,default=BASE_ROOT);p.add_argument("--output-root",required=True);p.add_argument("--full-run-root",default=str(FULL_RUN_ROOT));p.add_argument("--proto-run-root",default=str(PROTO_RUN_ROOT));p.add_argument("--fixed-run-root",default=str(FIXED_RUN_ROOT));p.add_argument("--validation-fraction",type=float,default=.30);p.add_argument("--test-episodes",type=int,default=TEST_EPISODES);p.add_argument("--debug-protocol",action="store_true");a=p.parse_args();ar={};asp={};ark={}
 for seed in a.seeds:
  print(f"\n=== Bayesian ablation evaluation seed {seed} ===");ar[seed],asp[seed],ark[seed]=evaluate_seed(seed,a)
 aggregate(ar,asp,ark,a.seeds,a)
if __name__=="__main__":main()
