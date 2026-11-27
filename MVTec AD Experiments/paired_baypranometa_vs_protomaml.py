#!/usr/bin/env python3
"""Paired category-level test: BayPrAnoMeta versus Proto_MAML."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
BAYES_ROOT = ROOT / "Bayesian_ProtoMAML/runs/baypranometa_avgpool_three_seed/revised_clean"
PROTO_ROOT = ROOT / "CLassical_ProtoMAML/runs/protomaml_corrected_avgpool_three_seed/revised_clean"
SEEDS = (1, 2, 3)


def load_seed(path, seed):
    if not path.is_file():
        raise FileNotFoundError(f"Final corrected result file not found: {path}")
    output = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"category", "held_out_subtype", "auc_mean"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for row in reader:
            category = row["category"].strip()
            if category in output:
                raise ValueError(f"Duplicate category {category!r} in {path}")
            seed_text = row.get("seed") or row.get("run_seed")
            if seed_text and int(seed_text) != seed:
                raise ValueError(f"Seed mismatch in {path}: expected {seed}")
            output[category] = {
                "subtype": row["held_out_subtype"].strip(),
                "auc": float(row["auc_mean"]),
            }
    return output


def load_results(bayes_root=BAYES_ROOT, proto_root=PROTO_ROOT):
    bayes = {s: load_seed(bayes_root / f"seed_{s}/eval/results.csv", s) for s in SEEDS}
    proto = {s: load_seed(proto_root / f"seed_{s}/evaluation/results.csv", s) for s in SEEDS}
    return bayes, proto


def verify_pairing(bayes, proto):
    reference = set(bayes[1])
    for method, data in (("BayPrAnoMeta", bayes), ("Proto-MAML", proto)):
        for seed, rows in data.items():
            if set(rows) != reference:
                raise ValueError(f"Category mismatch for {method}, seed {seed}")
    mismatches = []
    for category in sorted(reference):
        values = [bayes[s][category]["subtype"] for s in SEEDS]
        values += [proto[s][category]["subtype"] for s in SEEDS]
        if len(set(values)) != 1:
            mismatches.append(f"{category}: {values}")
    if mismatches:
        raise ValueError("Held-out subtype mismatch; test NOT run:\n" + "\n".join(mismatches))
    print(f"Confirmed before testing: all {len(reference)} categories have matching held-out subtypes.")
    return sorted(reference)


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-run-root", type=Path, required=True)
    parser.add_argument("--proto-run-root", type=Path, required=True)
    parser.add_argument("--output-root", "--output-dir", dest="output_dir", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    bayes, proto = load_results(args.full_run_root, args.proto_run_root)
    categories = verify_pairing(bayes, proto)

    paired = []
    for category in categories:
        b = np.array([bayes[s][category]["auc"] for s in SEEDS])
        p = np.array([proto[s][category]["auc"] for s in SEEDS])
        paired.append({
            "category": category,
            "held_out_subtype": bayes[1][category]["subtype"],
            "bayes_seed1": b[0], "bayes_seed2": b[1], "bayes_seed3": b[2],
            "bayes_mean": b.mean(), "bayes_sd": b.std(ddof=1),
            "proto_seed1": p[0], "proto_seed2": p[1], "proto_seed3": p[2],
            "proto_mean": p.mean(), "proto_sd": p.std(ddof=1),
            "paired_difference": b.mean() - p.mean(),
        })

    differences = np.array([row["paired_difference"] for row in paired])
    test = wilcoxon(differences, alternative="two-sided", zero_method="wilcox",
                    correction=False, method="auto")
    rng = np.random.default_rng(args.bootstrap_seed)
    indices = rng.integers(0, len(differences),
                           size=(args.bootstrap_resamples, len(differences)))
    bootstrap_means = differences[indices].mean(axis=1)
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    summary = {
        "number_of_paired_categories": len(paired),
        "baypranometa_overall_mean_category_AUROC": np.mean([r["bayes_mean"] for r in paired]),
        "protomaml_overall_mean_category_AUROC": np.mean([r["proto_mean"] for r in paired]),
        "mean_paired_difference": differences.mean(),
        "median_paired_difference": np.median(differences),
        "baypranometa_wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "protomaml_wins": int((differences < 0).sum()),
        "wilcoxon_statistic": float(test.statistic),
        "two_sided_p_value": float(test.pvalue),
        "bootstrap_95_ci_lower": ci_low, "bootstrap_95_ci_upper": ci_high,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "wilcoxon_alternative": "two-sided", "wilcoxon_zero_method": "wilcox",
    }
    paired_path = args.output_dir / "paired_baypranometa_vs_protomaml.csv"
    summary_path = args.output_dir / "paired_test_summary.csv"
    write_csv(paired_path, list(paired[0]), paired)
    write_csv(summary_path, list(summary), [summary])
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Wrote: {paired_path}\nWrote: {summary_path}")


if __name__ == "__main__":
    main()
