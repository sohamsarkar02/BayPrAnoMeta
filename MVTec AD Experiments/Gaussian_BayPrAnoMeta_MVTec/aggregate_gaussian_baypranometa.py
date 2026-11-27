#!/usr/bin/env python3
"""Aggregate completed Gaussian seeds and pair categories with Full BayPrAnoMeta."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

PROTOCOL = "revised_clean"
DEFAULT_GAUSSIAN_ROOT = Path("runs/gaussian_baypranometa_three_seed_fixed_split42_25ep_30epochs_100test")
DEFAULT_FULL_ROOT = Path("../Bayesian_ProtoMAML/runs/baypranometa_avgpool_three_seed")


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--gaussian-root", type=Path, default=DEFAULT_GAUSSIAN_ROOT)
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    args = parser.parse_args()
    gaussian, full = {}, {}
    for seed in args.seeds:
        gpath = args.gaussian_root / PROTOCOL / f"seed_{seed}" / "eval" / "results.csv"
        fpath = args.full_root / PROTOCOL / f"seed_{seed}" / "eval" / "results.csv"
        if not gpath.is_file(): raise FileNotFoundError(gpath)
        if not fpath.is_file(): raise FileNotFoundError(fpath)
        gaussian[seed] = {r["category"]: r for r in read_rows(gpath)}
        full[seed] = {r["category"]: r for r in read_rows(fpath)}
    categories = sorted(set.intersection(*(set(gaussian[s]) & set(full[s]) for s in args.seeds)))
    rows = []
    for category in categories:
        subtypes = {gaussian[s][category]["held_out_subtype"] for s in args.seeds} | {full[s][category]["held_out_subtype"] for s in args.seeds}
        if len(subtypes) != 1: raise AssertionError(f"Held-out subtype mismatch for {category}: {subtypes}")
        row = {"category": category, "held_out_subtype": next(iter(subtypes))}
        for metric in ("auc_mean", "pr_mean", "f1_mean"):
            short = {"auc_mean": "AUROC", "pr_mean": "AUPRC", "f1_mean": "F1"}[metric]
            gv = np.asarray([float(gaussian[s][category][metric]) for s in args.seeds])
            fv = np.asarray([float(full[s][category][metric]) for s in args.seeds])
            for seed, value in zip(args.seeds, gv): row[f"gaussian_seed{seed}_{short}"] = float(value)
            row[f"gaussian_mean_{short}"] = float(gv.mean()); row[f"gaussian_sd_{short}"] = float(gv.std(ddof=1)) if len(gv) > 1 else 0.0
            row[f"full_mean_{short}"] = float(fv.mean()); row[f"full_sd_{short}"] = float(fv.std(ddof=1)) if len(fv) > 1 else 0.0
            row[f"gaussian_minus_full_{short}"] = float(gv.mean() - fv.mean())
        rows.append(row)
    out = args.gaussian_root / PROTOCOL
    write_rows(out / "three_seed_category_summary.csv", rows)
    write_rows(out / "paired_gaussian_vs_full_baypranometa.csv", rows)
    summary = {"method": "Gaussian_BayPrAnoMeta_MVTec", "seeds": args.seeds, "paired_categories": len(rows)}
    for metric in ("AUROC", "AUPRC", "F1"):
        summary[f"gaussian_overall_macro_mean_{metric}"] = float(np.mean([r[f"gaussian_mean_{metric}"] for r in rows]))
        summary[f"full_overall_macro_mean_{metric}"] = float(np.mean([r[f"full_mean_{metric}"] for r in rows]))
        summary[f"mean_paired_difference_{metric}"] = float(np.mean([r[f"gaussian_minus_full_{metric}"] for r in rows]))
    (out / "overall_macro_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
