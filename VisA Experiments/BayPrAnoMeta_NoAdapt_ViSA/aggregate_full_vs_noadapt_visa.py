#!/usr/bin/env python3
"""Aggregate completed VisA NoAdapt seeds and compare with Full BayPrAnoMeta."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

PROTOCOL = "revised_clean_official_2cls_highshot"
NOADAPT_ROOT = Path("runs/baypranometa_noadapt_visa_three_seed_fixed_split42_25ep_30epochs_100test")
FULL_ROOT = Path("../Bayesian_ProtoMAML/runs/baypranometa_visa_three_seed_fixed_split42_25ep_30epochs_100test")


def read_csv(path):
    with path.open(newline="") as stream: return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def bootstrap_ci(values, draws=100000):
    values = np.asarray(values); rng = np.random.default_rng(42); means = np.empty(draws)
    for start in range(0, draws, 10000):
        stop = min(start + 10000, draws)
        means[start:stop] = values[rng.integers(0, len(values), size=(stop-start, len(values)))].mean(1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seeds", nargs="+", type=int, default=[1,2,3]); parser.add_argument("--noadapt-root", type=Path, default=NOADAPT_ROOT); parser.add_argument("--full-root", type=Path, default=FULL_ROOT); args = parser.parse_args()
    noadapt, full, seed_macro = {}, {}, []
    for seed in args.seeds:
        npath = args.noadapt_root / PROTOCOL / f"seed_{seed}" / "eval" / "results.csv"; fpath = args.full_root / PROTOCOL / f"seed_{seed}" / "eval" / "results.csv"
        if not npath.is_file(): raise FileNotFoundError(npath)
        if not fpath.is_file(): raise FileNotFoundError(fpath)
        noadapt[seed] = {r["category"]:r for r in read_csv(npath)}; full[seed] = {r["category"]:r for r in read_csv(fpath)}
        seed_macro.append({"run_seed":seed, "full_macro_AUROC":float(np.mean([float(r["auc_mean"]) for r in full[seed].values()])), "noadapt_macro_AUROC":float(np.mean([float(r["auc_mean"]) for r in noadapt[seed].values()]))})
    categories = sorted(set.intersection(*(set(noadapt[s]) & set(full[s]) for s in args.seeds))); rows=[]
    for category in categories:
        groups={noadapt[s][category]["held_out_subtype"] for s in args.seeds}|{full[s][category]["held_out_subtype"] for s in args.seeds}
        if len(groups)!=1: raise AssertionError(category)
        row={"category":category,"held_out_group":next(iter(groups))}
        for field,metric in (("auc_mean","AUROC"),("pr_mean","AUPRC"),("f1_mean","F1")):
            fv=np.asarray([float(full[s][category][field]) for s in args.seeds]); nv=np.asarray([float(noadapt[s][category][field]) for s in args.seeds])
            for s,v in zip(args.seeds,fv): row[f"full_seed{s}_{metric}"]=float(v)
            for s,v in zip(args.seeds,nv): row[f"noadapt_seed{s}_{metric}"]=float(v)
            row[f"full_mean_{metric}"]=float(fv.mean()); row[f"full_sd_{metric}"]=float(fv.std(ddof=1)); row[f"noadapt_mean_{metric}"]=float(nv.mean()); row[f"noadapt_sd_{metric}"]=float(nv.std(ddof=1)); row[f"full_minus_noadapt_{metric}"]=float(fv.mean()-nv.mean())
        rows.append(row)
    differences=np.asarray([r["full_minus_noadapt_AUROC"] for r in rows]); tol=1e-12; w=wilcoxon(differences,alternative="two-sided"); t=ttest_rel([r["full_mean_AUROC"] for r in rows],[r["noadapt_mean_AUROC"] for r in rows]); ci=bootstrap_ci(differences)
    statistics=[{"paired_categories":len(rows),"full_overall_three_seed_macro_AUROC":float(np.mean([r["full_mean_AUROC"] for r in rows])),"noadapt_overall_three_seed_macro_AUROC":float(np.mean([r["noadapt_mean_AUROC"] for r in rows])),"mean_difference_full_minus_noadapt":float(differences.mean()),"median_difference_full_minus_noadapt":float(np.median(differences)),"full_wins":int(np.sum(differences>tol)),"ties":int(np.sum(abs(differences)<=tol)),"noadapt_wins":int(np.sum(differences < -tol)),"wilcoxon_statistic":float(w.statistic),"wilcoxon_two_sided_p":float(w.pvalue),"paired_t_statistic":float(t.statistic),"paired_t_two_sided_p":float(t.pvalue),"bootstrap_95_ci_low":ci[0],"bootstrap_95_ci_high":ci[1],"bootstrap_unit":"category","bootstrap_draws":100000}]
    out=args.noadapt_root/PROTOCOL; write_csv(out/"per_category_per_seed_comparison.csv",rows); write_csv(out/"three_seed_category_summary.csv",rows); write_csv(out/"seedwise_macro_auroc.csv",seed_macro); write_csv(out/"paired_statistical_test_summary.csv",statistics); (out/"overall_summary.json").write_text(json.dumps(statistics[0],indent=2)); print(json.dumps(statistics[0],indent=2))


if __name__ == "__main__": main()
