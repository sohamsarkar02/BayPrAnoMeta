#!/usr/bin/env python3
"""Evaluate corrected Classical MAML and aggregate seeds without score flipping."""
import argparse, csv, random, time
from pathlib import Path
import higher, matplotlib.pyplot as plt, numpy as np, seaborn as sns, torch, torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from maml_three_seed_common import *
from utils.utils_data_visa_leave1out import discover_categories, load_client_dataset_visa, load_images, transform_eval
from utils.utils_eval_stats import aggregate_with_se
from utils.utils_plotting import plot_pr_curve, plot_roc_curve, plot_score_histogram

def csv_write(path, rows, fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)

def episode_metrics(labels, scores):
    thresholds=np.linspace(float(scores.min()),float(scores.max()),200)
    return (roc_auc_score(labels,scores),average_precision_score(labels,scores),
            max(f1_score(labels,scores>=t,zero_division=0) for t in thresholds))

def describe(values):
    values=np.asarray(values,float)
    return {"mean":float(values.mean()),"median":float(np.median(values)),"std":float(values.std())}

def evaluate_category(model, category, data, seed, split_seed, out_root, debug_protocol=False, below_chance_threshold=.50, test_episodes=TEST_EPISODES):
    rng=random.Random(seed+4000); aucs=[];prs=[];f1s=[];records=[];manifest=[]
    for episode_id in range(test_episodes):
        support,qn,qa=sample_episode(data,"test",rng); paths=qn+qa
        if debug_protocol:
            assert set(support).isdisjoint(qn), "Support/query overlap detected."
            assert set(qa).issubset(set(data["test"]["anomaly"])), "Non-held-out anomaly entered test query."
        xs=stack(support,transform_eval,load_images);xq=stack(paths,transform_eval,load_images)
        labels=np.asarray([0]*Q_N+[1]*Q_A);inner_opt=torch.optim.SGD(model.parameters(),lr=INNER_LR)
        with higher.innerloop_ctx(model,inner_opt,copy_initial_weights=True,track_higher_grads=False) as (fmodel,diffopt):
            for _ in range(INNER_STEPS):
                logits=fmodel(xs);diffopt.step(nn.BCEWithLogitsLoss()(logits,torch.zeros_like(logits)))
            with torch.no_grad(): scores=fmodel(xq).cpu().numpy()
        if not np.isfinite(scores).all(): raise RuntimeError(f"Non-finite scores for {category}.")
        auc,pr,f1=episode_metrics(labels,scores);aucs.append(float(auc));prs.append(float(pr));f1s.append(float(f1))
        records.extend({"episode_id":episode_id,"image_path":p,"label":int(y),"score":float(s),"logit":float(s)} for p,y,s in zip(paths,labels,scores))
        manifest.append({"episode_id":episode_id,"support":support,"normal_query":qn,"anomaly_query":qa})
    out=Path(out_root)/"evaluation"/category;out.mkdir(parents=True,exist_ok=True)
    labels=np.asarray([r["label"] for r in records]);scores=np.asarray([r["score"] for r in records])
    auc_m,auc_se=aggregate_with_se(aucs);pr_m,pr_se=aggregate_with_se(prs);f1_m,f1_se=aggregate_with_se(f1s)
    plot_roc_curve(labels,scores,out/"roc.png");plot_pr_curve(labels,scores,out/"pr.png");plot_score_histogram(labels,scores,out/"score_hist.png")
    threshold=float(np.percentile(scores[labels==0],95));preds=(scores>=threshold).astype(int);cm=confusion_matrix(labels,preds,labels=[0,1])
    plt.figure(figsize=(5,4));sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Normal","Anomaly"],yticklabels=["Normal","Anomaly"]);plt.tight_layout();plt.savefig(out/"confusion_matrix.png");plt.close()
    errors=[{"path":r["image_path"],"ground_truth":int(y),"prediction":int(p),"score":float(s),"threshold":threshold} for r,y,p,s in zip(records,labels,preds,scores)]
    write_json(out/"false_positives.json",[e for e in errors if e["ground_truth"]==0 and e["prediction"]==1]);write_json(out/"false_negatives.json",[e for e in errors if e["ground_truth"]==1 and e["prediction"]==0])
    csv_write(out/"query_scores.csv",records,["episode_id","image_path","label","score","logit"]);write_json(out/"episode_manifest.json",manifest)
    triggered=bool(auc_m < below_chance_threshold)
    diagnosis="BELOW_CHANCE_REQUIRES_PERFORMANCE_REVIEW_NOT_AUTOMATIC_SCORE_FLIP" if triggered else None
    result={"method":"Classical MAML normal-only","category":category,"held_out_subtype":data["held_out_subtype"],"run_seed":seed,"seed":seed,"split_seed":split_seed,
        "AUROC":float(auc_m),"AUROC_se":float(auc_se),"AUPRC":float(pr_m),"AUPRC_se":float(pr_se),"F1":float(f1_m),"F1_se":float(f1_se),
        "auc_mean":float(auc_m),"auc_se":float(auc_se),"pr_mean":float(pr_m),"pr_se":float(pr_se),"f1_mean":float(f1_m),"f1_se":float(f1_se),
        "pooled_AUROC":float(roc_auc_score(labels,scores)),"reversed_score_AUROC":float(roc_auc_score(labels,-scores)),"threshold_95_normal":threshold,"confusion_matrix":cm.tolist(),
        "below_chance_diagnostic_triggered":triggered,"diagnosis":diagnosis,
        "label_convention":{"normal":0,"anomaly":1},"score_convention":"larger adapted classifier logit means more anomalous",
        "support_query_disjoint":all(set(x["support"]).isdisjoint(x["normal_query"]) for x in manifest),"scores_finite":bool(np.isfinite(scores).all())}
    if triggered:
        diagnostic={**result,"primary_score_was_not_modified":True,"normal_score":describe(scores[labels==0]),"anomaly_score":describe(scores[labels==1]),
                    "diagnosis":"BELOW_CHANCE_REQUIRES_PERFORMANCE_REVIEW_NOT_AUTOMATIC_SCORE_FLIP"}
        write_json(out/"below_chance_diagnostic.json",diagnostic)
    write_json(out/"metrics.json",result);return result

RESULT_FIELDS=["method","seed","run_seed","split_seed","category","held_out_subtype","auc_mean","auc_se","pr_mean","pr_se","f1_mean","f1_se","AUROC","AUROC_se","AUPRC","AUPRC_se","F1","F1_se","pooled_AUROC","reversed_score_AUROC","below_chance_diagnostic_triggered","diagnosis"]
def evaluate_seed(seed,args):
    set_seed(seed);root=Path(args.run_root)/PROTOCOL/f"seed_{seed}";checkpoint=root/"checkpoints"/"final_model.pth"; root = Path(args.output_root)/PROTOCOL/f"seed_{seed}"
    if not checkpoint.exists(): raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    model=MAMLNet().to(DEVICE);model.load_state_dict(torch.load(checkpoint,map_location=DEVICE));model.eval()
    monitor=PeakMemoryMonitor();monitor.start();started=time.perf_counter();results={}
    categories=discover_categories(args.base_root,args.split_csv, subset_file=args.subset_file)
    if args.category:categories=[c for c in categories if c==args.category]
    print("Discovered VisA categories:",categories)
    for category in categories:
        data=load_client_dataset_visa(args.base_root,category,seed=args.split_seed,validation_fraction=args.validation_fraction,split_csv=args.split_csv, subset_file=args.subset_file,audit_dir=root/"evaluation"/"split_audit")
        if data is not None: results[category]=evaluate_category(model,category,data,seed,args.split_seed,root,args.debug_protocol,args.below_chance_threshold,args.test_episodes)
    monitor.stop();csv_write(root/"evaluation"/"results.csv",[results[c] for c in sorted(results)],RESULT_FIELDS)
    write_json(root/"evaluation"/"results.json",results);np.save(root/"evaluation"/"results.npy",results)
    write_json(root/"evaluation"/"metadata.json",{"method":"Classical MAML corrected","run_seed":seed,"split_seed":args.split_seed,
        "dataset":"VisA","test_episodes_per_category":args.test_episodes,"score":"adapted anomaly logit; higher is anomalous","checkpoint":str(checkpoint),"split_source":args.split_csv})
    write_json(root/"evaluation"/"runtime_memory.json",{"evaluation_sec":time.perf_counter()-started,"peak_rss_mb":monitor.peak_mb})
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int);p.add_argument("--seeds",nargs="+",type=int);p.add_argument("--split-seed",type=int,default=SPLIT_SEED)
    p.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None,default=BASE_ROOT);p.add_argument("--checkpoint-root", "--run-root", dest="run_root",default=RUN_ROOT); p.add_argument("--output-root", required=True, help="Directory for evaluation artifacts.");p.add_argument("--validation-fraction",type=float,default=.30);p.add_argument("--below-chance-threshold",type=float,default=.50);p.add_argument("--debug-protocol",action="store_true");p.add_argument("--split-csv",default="split_csv/2cls_highshot.csv"); p.add_argument("--subset-file", help='Paper image allowlist. Required for exact reproduction of the paper experiments.');p.add_argument("--category");p.add_argument("--test-episodes",type=int,default=TEST_EPISODES);p.add_argument("--smoke-test",action="store_true")
    args=p.parse_args();args.seeds=[args.seed] if args.seed is not None else (args.seeds or [1,2,3])
    if args.smoke_test:args.seeds=[args.seed or 1];args.test_episodes=3;args.category=args.category or discover_categories(args.base_root,args.split_csv, subset_file=args.subset_file)[0]
    all_results={}
    for seed in args.seeds:
        print(f"\n=== Starting Classical MAML evaluation: run seed {seed} ===")
        all_results[seed]=evaluate_seed(seed,args)
        print(f"=== Completed Classical MAML evaluation: run seed {seed} ===")
    shared=set.intersection(*(set(r) for r in all_results.values()));summary=[]
    for category in sorted(shared):
        held={all_results[s][category]["held_out_subtype"] for s in args.seeds}
        if len(held)!=1: raise RuntimeError(f"Held-out subtype differs across seeds for {category}; refusing aggregation.")
        values=np.asarray([all_results[s][category]["AUROC"] for s in args.seeds]);row={"method":"Classical MAML normal-only","category":category,"held_out_subtype":held.pop()}
        row.update({f"seed{s}_AUROC":float(all_results[s][category]["AUROC"]) for s in args.seeds});row.update({"mean_AUROC":float(values.mean()),"std_AUROC":float(values.std(ddof=1)) if len(values)>1 else 0.0,"any_seed_below_0_50":bool(np.any(values<.5)),"mean_below_0_50":bool(values.mean()<.5)});summary.append(row)
    base=Path(args.output_root)/PROTOCOL;fields=["method","category","held_out_subtype"]+[f"seed{s}_AUROC" for s in args.seeds]+["mean_AUROC","std_AUROC","any_seed_below_0_50","mean_below_0_50"]
    csv_write(base/"three_seed_auroc_summary.csv",summary,fields);write_json(base/"three_seed_auroc_summary.json",summary)
    csv_write(base/"three_seed_results.csv",[v for s in args.seeds for v in all_results[s].values()],RESULT_FIELDS)
    write_json(base/"evaluation_split_manifest.json",{"split_seed":args.split_seed,"run_seeds":args.seeds,
        "held_out_subtypes":{c:all_results[args.seeds[0]][c]["held_out_subtype"] for c in sorted(shared)}})
if __name__=="__main__":main()
