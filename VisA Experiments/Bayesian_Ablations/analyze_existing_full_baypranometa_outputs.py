#!/usr/bin/env python3
"""Analyze completed Full VisA BayPrAnoMeta CSVs without model execution."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from ablation_common_visa import *


REQUIRED = {
    "embedding_norm_sq": "||z||^2", "logp_normal": "log p0",
    "logp_anomaly": "log p1", "final_score": "Bayesian score",
    "label": "label", "category": "category/client",
    "held_out_subtype": "held-out group", "episode_id": "episode",
    "run_seed": "seed", "image_path": "image path",
    "euclidean_score": "Euclidean prototype-distance score",
}


def audit_and_analyze(args):
    output = Path(args.output_root)
    audit_rows, spread_rows, score_checks = [], [], []
    categories = discover_categories(args.base_root, args.split_csv, subset_file=args.subset_file)
    for seed in args.seeds:
        for category in categories:
            path = Path(args.full_run_root) / PROTOCOL / f"seed_{seed}" / "eval" / category / "query_results.csv"
            records = read_csv(path)
            expected = args.test_episodes * (Q_N + Q_A)
            fields = set(records[0]) if records else set()
            if len(records) != expected:
                raise RuntimeError(f"{path}: {len(records)} rows, expected {expected}")
            for column, label in REQUIRED.items():
                audit_rows.append({
                    "field": label, "saved": column in fields, "file": str(path),
                    "column_name": column, "run_seed": seed, "category": category,
                    "complete": column in fields and len(records) == expected,
                })
            differences = np.asarray([
                float(row["final_score"]) -
                (float(row["logp_anomaly"]) - float(row["logp_normal"]))
                for row in records])
            score_checks.append({
                "run_seed": seed, "category": category,
                "max_abs_saved_vs_reconstructed_score_difference": float(np.max(np.abs(differences))),
                "saved_score_matches_logp1_minus_logp0_within_5e-5": bool(
                    np.max(np.abs(differences)) <= 5e-5),
            })
            labels = np.asarray([int(row["label"]) for row in records])
            for group, mask in (("normal", labels == 0), ("anomaly", labels == 1),
                                ("all", np.ones(len(labels), dtype=bool))):
                summary = {
                    "method": "BayPrAnoMeta", "run_seed": seed, "split_seed": args.split_seed,
                    "category": category, "held_out_subtype": records[0]["held_out_subtype"],
                    "query_group": group, "n_query_records": int(mask.sum()),
                    "source": "existing completed query_results.csv",
                }
                for column, prefix in (
                        ("embedding_norm_sq", "norm_sq"), ("logp_normal", "logp0"),
                        ("logp_anomaly", "logp1"), ("final_score", "bayesian_score")):
                    stats = descriptive([float(row[column]) for row, keep in zip(records, mask) if keep])
                    summary.update({f"{prefix}_{key}": value for key, value in stats.items()})
                spread_rows.append(summary)

    write_csv(output / "existing_full_query_field_audit.csv", audit_rows)
    write_csv(output / "existing_full_score_reconstruction_check.csv", score_checks)
    write_csv(output / "p0_p1_embedding_spread_from_existing.csv", spread_rows)
    write_json(output / "existing_full_reuse_manifest.json", {
        "full_training_rerun_required": False,
        "full_evaluation_required_only_for": ["euclidean_prototype_distance"],
        "records_reused": len(args.seeds) * len(categories) * args.test_episodes * (Q_N + Q_A),
        "seeds": args.seeds, "categories": categories,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None, default=BASE_ROOT)
    parser.add_argument("--split-csv", default="split_csv/2cls_highshot.csv"); parser.add_argument("--subset-file", help='Paper image allowlist. Required for exact reproduction of the paper experiments.')
    parser.add_argument("--full-run-root", default=str(FULL_RUN_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--test-episodes", type=int, default=TEST_EPISODES)
    args = parser.parse_args()
    if args.split_seed != 42: parser.error("SPLIT_SEED is fixed at 42")
    audit_and_analyze(args)


if __name__ == "__main__":
    main()
