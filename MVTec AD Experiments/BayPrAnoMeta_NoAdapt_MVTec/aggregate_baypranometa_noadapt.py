#!/usr/bin/env python3
"""Aggregate NoAdapt seeds and pair category metrics with Full BayPrAnoMeta."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

PROTOCOL = "revised_clean"
DEFAULT_NOADAPT_ROOT = Path("runs/baypranometa_noadapt_three_seed_fixed_split42_25ep_30epochs_100test")
DEFAULT_FULL_ROOT = Path("../Bayesian_ProtoMAML/runs/baypranometa_avgpool_three_seed")


def read_rows(path):
    with path.open(newline="") as handle: return list(csv.DictReader(handle))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--noadapt-root", type=Path, default=DEFAULT_NOADAPT_ROOT)
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    args = parser.parse_args()
    noadapt, full = {}, {}
    for seed in args.seeds:
        npath = args.noadapt_root / PROTOCOL / f"seed_{seed}" / "eval" / "results.csv"
        fpath = args.full_root / PROTOCOL / f"seed_{seed}" / "eval" / "results.csv"
        if not npath.is_file(): raise FileNotFoundError(npath)
        if not fpath.is_file(): raise FileNotFoundError(fpath)
        noadapt[seed] = {r["category"]: r for r in read_rows(npath)}
        full[seed] = {r["category"]: r for r in read_rows(fpath)}
    categories = sorted(set.intersection(*(set(noadapt[s]) & set(full[s]) for s in args.seeds)))
    rows = []
    for category in categories:
        subtypes = {noadapt[s][category]["held_out_subtype"] for s in args.seeds} | {full[s][category]["held_out_subtype"] for s in args.seeds}
        if len(subtypes) != 1: raise AssertionError(f"Held-out subtype mismatch for {category}: {subtypes}")
        row = {"category": category, "held_out_subtype": next(iter(subtypes))}
        for field, metric in (("auc_mean", "AUROC"), ("pr_mean", "AUPRC"), ("f1_mean", "F1")):
            nv = np.asarray([float(noadapt[s][category][field]) for s in args.seeds])
            fv = np.asarray([float(full[s][category][field]) for s in args.seeds])
            for seed, value in zip(args.seeds, nv): row[f"noadapt_seed{seed}_{metric}"] = float(value)
            row[f"noadapt_mean_{metric}"] = float(nv.mean()); row[f"noadapt_sd_{metric}"] = float(nv.std(ddof=1)) if len(nv) > 1 else 0.0
            row[f"full_mean_{metric}"] = float(fv.mean()); row[f"full_sd_{metric}"] = float(fv.std(ddof=1)) if len(fv) > 1 else 0.0
            row[f"full_minus_noadapt_{metric}"] = float(fv.mean() - nv.mean())
        rows.append(row)
    out = args.noadapt_root / PROTOCOL
    write_rows(out / "three_seed_category_summary.csv", rows)
    write_rows(out / "paired_full_vs_noadapt.csv", rows)
    summary = {"method": "BayPrAnoMeta_NoAdapt_MVTec", "seeds": args.seeds, "paired_categories": len(rows)}
    for metric in ("AUROC", "AUPRC", "F1"):
        summary[f"noadapt_overall_macro_mean_{metric}"] = float(np.mean([r[f"noadapt_mean_{metric}"] for r in rows]))
        summary[f"full_overall_macro_mean_{metric}"] = float(np.mean([r[f"full_mean_{metric}"] for r in rows]))
        summary[f"mean_full_minus_noadapt_{metric}"] = float(np.mean([r[f"full_minus_noadapt_{metric}"] for r in rows]))
    (out / "overall_macro_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
