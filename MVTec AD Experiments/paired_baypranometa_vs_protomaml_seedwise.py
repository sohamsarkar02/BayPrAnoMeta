#!/usr/bin/env python3
"""Seedwise paired tests across MVTec categories.

Runs one two-sided Wilcoxon signed-rank test per seed. Within each seed, the
paired observations are the matched MVTec categories (n=14), not individual
evaluation samples.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from paired_baypranometa_vs_protomaml import (
    ROOT,
    SEEDS,
    load_results,
    verify_pairing,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-run-root", type=Path, required=True)
    parser.add_argument("--proto-run-root", type=Path, required=True)
    parser.add_argument("--output-root", "--output-dir", dest="output_dir", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_resamples < 1:
        parser.error("--bootstrap-resamples must be positive")

    bayes, proto = load_results(args.full_run_root, args.proto_run_root)
    categories = verify_pairing(bayes, proto)  # Validate paired categories and held-out subtypes before testing.
    detailed_rows = []
    summary_rows = []

    for seed in SEEDS:
        differences = []
        bayes_values = []
        proto_values = []
        for category in categories:
            bayes_auc = float(bayes[seed][category]["auc"])
            proto_auc = float(proto[seed][category]["auc"])
            difference = bayes_auc - proto_auc
            bayes_values.append(bayes_auc)
            proto_values.append(proto_auc)
            differences.append(difference)
            detailed_rows.append({
                "seed": seed,
                "category": category,
                "held_out_subtype": bayes[seed][category]["subtype"],
                "bayes_auroc": bayes_auc,
                "proto_auroc": proto_auc,
                "paired_difference": difference,
            })

        d = np.asarray(differences)
        test = wilcoxon(
            d,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="auto",
        )

        # A reproducible but distinct bootstrap stream for each seed.
        rng = np.random.default_rng(args.bootstrap_seed + seed - 1)
        indices = rng.integers(
            0, len(d), size=(args.bootstrap_resamples, len(d))
        )
        bootstrap_means = d[indices].mean(axis=1)
        ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])

        summary_rows.append({
            "seed": seed,
            "number_of_paired_categories": len(d),
            "baypranometa_mean_category_AUROC": np.mean(bayes_values),
            "protomaml_mean_category_AUROC": np.mean(proto_values),
            "mean_paired_difference": d.mean(),
            "median_paired_difference": np.median(d),
            "baypranometa_wins": int((d > 0).sum()),
            "ties": int((d == 0).sum()),
            "protomaml_wins": int((d < 0).sum()),
            "wilcoxon_statistic": float(test.statistic),
            "two_sided_p_value": float(test.pvalue),
            "bootstrap_95_ci_lower": ci_low,
            "bootstrap_95_ci_upper": ci_high,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed + seed - 1,
            "wilcoxon_alternative": "two-sided",
            "wilcoxon_zero_method": "wilcox",
        })

    detail_path = args.output_dir / "paired_baypranometa_vs_protomaml_seedwise.csv"
    summary_path = args.output_dir / "paired_test_summary_seedwise.csv"
    write_csv(detail_path, list(detailed_rows[0]), detailed_rows)
    write_csv(summary_path, list(summary_rows[0]), summary_rows)

    for row in summary_rows:
        print(f"\nSeed {row['seed']} (paired categories={row['number_of_paired_categories']})")
        print(f"BayPrAnoMeta mean AUROC: {row['baypranometa_mean_category_AUROC']:.6f}")
        print(f"Proto-MAML mean AUROC: {row['protomaml_mean_category_AUROC']:.6f}")
        print(f"Mean difference: {row['mean_paired_difference']:.6f}")
        print(f"Median difference: {row['median_paired_difference']:.6f}")
        print(
            "Wins / ties / losses: "
            f"{row['baypranometa_wins']} / {row['ties']} / {row['protomaml_wins']}"
        )
        print(f"Wilcoxon statistic: {row['wilcoxon_statistic']}")
        print(f"Two-sided p-value: {row['two_sided_p_value']:.12g}")
        print(
            "95% bootstrap CI: "
            f"[{row['bootstrap_95_ci_lower']:.6f}, "
            f"{row['bootstrap_95_ci_upper']:.6f}]"
        )
    print(f"\nWrote:\n{detail_path}\n{summary_path}")


if __name__ == "__main__":
    main()
