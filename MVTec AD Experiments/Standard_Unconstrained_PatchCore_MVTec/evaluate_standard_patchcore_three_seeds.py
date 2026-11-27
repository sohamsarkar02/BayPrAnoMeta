#!/usr/bin/env python3
"""Evaluate saved standard PatchCore banks on corrected paired episode manifests."""

import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from standard_patchcore_common import *
from utils.utils_eval_stats import aggregate_with_se,compute_episode_metrics
from utils.utils_plotting import plot_pr_curve,plot_roc_curve,plot_score_histogram


def csv_write(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(rows[0])
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)


def score_paths(model,paths,batch_size):
    scores={}
    ordered=list(dict.fromkeys(paths))
    for start in range(0,len(ordered),batch_size):
        batch_paths=ordered[start:start+batch_size]
        images=torch.stack([PathDataset([path])[0] for path in batch_paths])
        batch_scores,_=model.predict(images)
        scores.update({path:float(score) for path,score in zip(batch_paths,batch_scores)})
    return scores


def evaluate_seed(seed,args):
    categories=CLIENTS[:1] if args.smoke_test else CLIENTS
    monitor=PeakMemoryMonitor();monitor.start();started=time.perf_counter();rows=[]
    for category_index,category in enumerate(categories):
        data=load_protocol(args.base_root,category,args.split_seed,args.validation_fraction)
        if data is None:continue
        bank_dir=Path(args.bank_root or args.result_root)/f"seed_{seed}"/"memory_banks"/category
        bank=np.load(bank_dir/"memory_bank.npy")
        metadata=json.loads((bank_dir/"metadata.json").read_text())
        manifest_path=Path(args.reference_manifest_root)/f"seed_{seed}"/"eval"/category/"test_episodes.json"
        if not manifest_path.is_file():raise FileNotFoundError(f"Missing paired manifest: {manifest_path}")
        episodes=json.loads(manifest_path.read_text()); episodes=episodes[:2] if args.smoke_test else episodes
        if len(episodes)!=(2 if args.smoke_test else 100):raise RuntimeError(f"Unexpected episode count for {category}")
        query_paths=[path for episode in episodes for path in episode["normal_query"]+episode["anomaly_query"]]
        allowed=set(data["test"]["normal"])|set(data["test"]["anomaly"])
        if not set(query_paths)<=allowed:raise RuntimeError(f"{category}: paired manifest does not match corrected test pools")
        model,_=build_patchcore(seed*1000+category_index)
        model.anomaly_scorer.fit(detection_features=[bank])
        scoring_start=time.perf_counter();score_map=score_paths(model,query_paths,args.batch_size);scoring_seconds=time.perf_counter()-scoring_start
        aucs,aps,f1s,records=[],[],[],[];labels_all=[];scores_all=[]
        for episode in episodes:
            paths=episode["normal_query"]+episode["anomaly_query"]
            labels=[0]*len(episode["normal_query"])+[1]*len(episode["anomaly_query"])
            scores=[score_map[path] for path in paths]
            auc,ap,f1=compute_episode_metrics(scores,labels);aucs.append(auc);aps.append(ap);f1s.append(f1)
            labels_all.extend(labels);scores_all.extend(scores)
            records.extend({"method":"PatchCore-Standard","image_path":path,"label":label,"anomaly_score":score,
                            "category":category,"held_out_subtype":data["held_out_subtype"],"episode":episode["episode"],
                            "run_seed":seed,"split_seed":args.split_seed}
                           for path,label,score in zip(paths,labels,scores))
        auc_mean,auc_se=aggregate_with_se(aucs);ap_mean,ap_se=aggregate_with_se(aps);f1_mean,f1_se=aggregate_with_se(f1s)
        out=Path(args.result_root)/f"seed_{seed}"/"evaluation"/category;out.mkdir(parents=True,exist_ok=True)
        csv_write(out/"query_results.csv",records);write_json(out/"test_episodes.json",episodes)
        plot_roc_curve(labels_all,scores_all,out/"roc.png");plot_pr_curve(labels_all,scores_all,out/"pr.png");plot_score_histogram(labels_all,scores_all,out/"score_hist.png")
        labels_np=np.asarray(labels_all);scores_np=np.asarray(scores_all);threshold=float(np.percentile(scores_np[labels_np==0],95));preds=(scores_np>=threshold).astype(int)
        plt.figure(figsize=(5,4));sns.heatmap(confusion_matrix(labels_np,preds),annot=True,fmt="d",cmap="Blues",xticklabels=["Normal","Anomaly"],yticklabels=["Normal","Anomaly"]);plt.tight_layout();plt.savefig(out/"confusion_matrix.png");plt.close()
        row={"method":"PatchCore-Standard","category":category,"held_out_subtype":data["held_out_subtype"],"run_seed":seed,"split_seed":args.split_seed,
             "AUROC":float(auc_mean),"AUROC_se":float(auc_se),"AUPRC":float(ap_mean),"AUPRC_se":float(ap_se),"F1":float(f1_mean),"F1_se":float(f1_se),
             "coreset_fraction":CORESET_FRACTION,"memory_bank_size_before_coreset":metadata["memory_bank_size_before_coreset"],"memory_bank_size_after_coreset":len(bank),
             "backbone":metadata["backbone"],"feature_layers":"layer2+layer3","projection_head_128d":False,"memory_bank_rebuilt_per_episode":False,
             "paired_manifest_path":str(manifest_path),"query_scoring_seconds":scoring_seconds}
        rows.append(row);write_json(out/"metrics.json",row);print(f"seed={seed} {category}: AUROC={auc_mean:.4f} ± {auc_se:.4f}")
        del model
    csv_write(Path(args.result_root)/f"seed_{seed}"/"evaluation"/"results.csv",rows);write_json(Path(args.result_root)/f"seed_{seed}"/"evaluation"/"results.json",rows)
    monitor.stop();write_json(Path(args.result_root)/f"seed_{seed}"/"evaluation"/"runtime_memory.json",{"run_seed":seed,"evaluation_seconds":time.perf_counter()-started,"peak_rss_mb":monitor.peak_mb,"evaluation_seconds_per_episode":(time.perf_counter()-started)/max(1,len(rows)*(2 if args.smoke_test else 100))})
    return rows


def aggregate(all_results,seeds,args):
    lookup={(r["run_seed"],r["category"]):r for rows in all_results.values() for r in rows};categories=sorted(set.intersection(*(set(r["category"] for r in all_results[s]) for s in seeds)));summary=[]
    for category in categories:
        row={"method":"PatchCore-Standard","category":category,"held_out_subtype":lookup[seeds[0],category]["held_out_subtype"]}
        for metric in ("AUROC","AUPRC","F1"):
            values=np.asarray([lookup[s,category][metric] for s in seeds]);row.update({f"seed_{s}_{metric}":float(lookup[s,category][metric]) for s in seeds});row[f"mean_{metric}"]=float(values.mean());row[f"std_{metric}"]=float(values.std(ddof=1)) if len(values)>1 else 0.
        summary.append(row)
    root=Path(args.result_root);csv_write(root/"three_seed_summary.csv",summary);write_json(root/"three_seed_summary.json",summary)
    write_json(root/"overall_summary.json",{"method":"PatchCore-Standard","run_seeds":seeds,"number_of_categories":len(summary),"overall_mean_category_AUROC":float(np.mean([r["mean_AUROC"] for r in summary]))})


def main():
    p=argparse.ArgumentParser();p.add_argument("--seeds",nargs="+",type=int,default=[1,2,3]);p.add_argument("--split-seed",type=int,default=SPLIT_SEED);p.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None,default=BASE_ROOT);p.add_argument("--output-root","--result-root",dest="result_root",required=True);p.add_argument("--bank-root",required=True,help="Output root of the standard PatchCore fitting command.");p.add_argument("--reference-manifest-root",default=str(REFERENCE_MANIFEST_ROOT));p.add_argument("--validation-fraction",type=float,default=.30);p.add_argument("--batch-size",type=int,default=2);p.add_argument("--debug-protocol",action="store_true");p.add_argument("--smoke-test",action="store_true");args=p.parse_args()
    if args.smoke_test:args.seeds=[1];args.result_root+="_smoke"
    all_results={seed:evaluate_seed(seed,args) for seed in args.seeds};aggregate(all_results,args.seeds,args)
if __name__=="__main__":main()
