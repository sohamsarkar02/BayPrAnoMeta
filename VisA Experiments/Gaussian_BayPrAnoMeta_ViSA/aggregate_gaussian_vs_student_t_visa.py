#!/usr/bin/env python3
"""Aggregate completed Gaussian VisA seeds and pair them with Full Student-t results."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

PROTOCOL = "revised_clean_official_2cls_highshot"
GAUSSIAN_ROOT = Path("runs/gaussian_baypranometa_visa_three_seed_fixed_split42_25ep_30epochs_100test")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--gaussian-root", type=Path, default=GAUSSIAN_ROOT)
    parser.add_argument("--full-root", type=Path, default=FULL_ROOT)
    args = parser.parse_args()
    gaussian, full, per_seed = {}, {}, []
    for seed in args.seeds:
        gpath = args.gaussian_root / PROTOCOL / f"seed_{seed}" / "eval" / "results.csv"
        fpath = args.full_root / PROTOCOL / f"seed_{seed}" / "eval" / "results.csv"
        if not gpath.is_file(): raise FileNotFoundError(gpath)
        if not fpath.is_file(): raise FileNotFoundError(fpath)
        gaussian[seed] = {row["category"]: row for row in read_csv(gpath)}
        full[seed] = {row["category"]: row for row in read_csv(fpath)}
        per_seed.append({"run_seed": seed,
            "student_t_macro_AUROC": float(np.mean([float(row["auc_mean"]) for row in full[seed].values()])),
            "gaussian_macro_AUROC": float(np.mean([float(row["auc_mean"]) for row in gaussian[seed].values()]))})
    categories = sorted(set.intersection(*(set(gaussian[s]) & set(full[s]) for s in args.seeds)))
    rows = []
    for category in categories:
        groups = {gaussian[s][category]["held_out_subtype"] for s in args.seeds} | {full[s][category]["held_out_subtype"] for s in args.seeds}
        if len(groups) != 1: raise AssertionError(f"Held-out group mismatch: {category}: {groups}")
        row = {"category": category, "held_out_group": next(iter(groups))}
        for field, metric in (("auc_mean", "AUROC"), ("pr_mean", "AUPRC"), ("f1_mean", "F1")):
            tv = np.asarray([float(full[s][category][field]) for s in args.seeds]); gv = np.asarray([float(gaussian[s][category][field]) for s in args.seeds])
            for seed, value in zip(args.seeds, tv): row[f"student_t_seed{seed}_{metric}"] = float(value)
            for seed, value in zip(args.seeds, gv): row[f"gaussian_seed{seed}_{metric}"] = float(value)
            row[f"student_t_mean_{metric}"] = float(tv.mean()); row[f"student_t_sd_{metric}"] = float(tv.std(ddof=1))
            row[f"gaussian_mean_{metric}"] = float(gv.mean()); row[f"gaussian_sd_{metric}"] = float(gv.std(ddof=1))
            row[f"student_t_minus_gaussian_{metric}"] = float(tv.mean() - gv.mean())
        rows.append(row)
    differences = np.asarray([row["student_t_minus_gaussian_AUROC"] for row in rows]); tol = 1e-12
    w = wilcoxon(differences, alternative="two-sided"); t = ttest_rel([r["student_t_mean_AUROC"] for r in rows], [r["gaussian_mean_AUROC"] for r in rows])
    ci = bootstrap_ci(differences)
    statistics = [{"paired_categories": len(rows),
        "student_t_overall_three_seed_macro_AUROC": float(np.mean([r["student_t_mean_AUROC"] for r in rows])),
        "gaussian_overall_three_seed_macro_AUROC": float(np.mean([r["gaussian_mean_AUROC"] for r in rows])),
        "mean_paired_difference_student_t_minus_gaussian": float(differences.mean()),
        "median_paired_difference_student_t_minus_gaussian": float(np.median(differences)),
        "student_t_wins": int(np.sum(differences > tol)), "ties": int(np.sum(abs(differences) <= tol)), "gaussian_wins": int(np.sum(differences < -tol)),
        "wilcoxon_statistic": float(w.statistic), "wilcoxon_two_sided_p": float(w.pvalue),
        "paired_t_statistic": float(t.statistic), "paired_t_two_sided_p": float(t.pvalue),
        "bootstrap_95_ci_low": ci[0], "bootstrap_95_ci_high": ci[1], "bootstrap_unit": "category", "bootstrap_draws": 100000}]
    out = args.gaussian_root / PROTOCOL
    write_csv(out / "per_category_per_seed_comparison.csv", rows)
    write_csv(out / "three_seed_category_summary.csv", rows)
    write_csv(out / "per_seed_macro_auroc.csv", per_seed)
    write_csv(out / "paired_statistical_test_summary.csv", statistics)
    (out / "overall_summary.json").write_text(json.dumps(statistics[0], indent=2))
    print(json.dumps(statistics[0], indent=2))


if __name__ == "__main__": main()
