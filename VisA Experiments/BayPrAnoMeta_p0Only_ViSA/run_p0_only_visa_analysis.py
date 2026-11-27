#!/usr/bin/env python3
"""Evaluation-only p0-score ablation for completed Full BayPrAnoMeta VisA runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "protocols" / "visa"))
from visa_protocol import add_manifest_protocol_arguments, manifest_index, verify_test_episodes

import numpy as np
from scipy.stats import kendalltau, spearmanr, ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
FULL_ROOT = (HERE.parent / "Bayesian_ProtoMAML" / "runs" /
             "baypranometa_visa_three_seed_fixed_split42_25ep_30epochs_100test" /
             "revised_clean_official_2cls_highshot")
OUTPUT_ROOT = HERE / "results" / "p0_only_ablation"
SEEDS = (1, 2, 3)
SPLIT_SEED = 42
CATEGORIES = ("candle", "capsules", "cashew", "chewinggum", "fryum",
              "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def expected_queries(manifest: list[dict]) -> list[tuple[int, str, int, list[str]]]:
    expected = []
    for episode in manifest:
        episode_id = int(episode["episode"])
        support = list(episode["support"])
        queries = list(episode["normal_query"]) + list(episode["anomaly_query"])
        if len(support) != 5 or len(episode["normal_query"]) != 12 or len(episode["anomaly_query"]) != 4:
            raise AssertionError(f"Unexpected K/QN/QA in episode {episode_id}")
        if set(support) & set(queries):
            raise AssertionError(f"Support/query overlap in episode {episode_id}")
        expected.extend((episode_id, path, 0, support) for path in episode["normal_query"])
        expected.extend((episode_id, path, 1, support) for path in episode["anomaly_query"])
    return expected


def normalize_saved(rows: list[dict], manifest: list[dict], seed: int, category: str,
                    full_root: Path, source_path: Path | None = None) -> tuple[list[dict], float]:
    expected = expected_queries(manifest)
    actual = [(int(row["episode_id"]), row["image_path"], int(row["label"])) for row in rows]
    expected_keys = [(episode, path, label) for episode, path, label, _ in expected]
    if actual != expected_keys:
        raise AssertionError(f"Query identities/order differ from Full manifest: seed={seed}, category={category}")
    if len(rows) != 1600 or len(manifest) != 100:
        raise AssertionError(f"Expected 100 episodes/1600 queries: seed={seed}, category={category}")

    normalized, maximum_saved_score_error = [], 0.0
    for row, (episode, path, label, support) in zip(rows, expected):
        if int(row["run_seed"]) != seed or int(row["split_seed"]) != SPLIT_SEED:
            raise AssertionError(f"Seed metadata mismatch: seed={seed}, category={category}")
        if row["category"] != category or int(row["true_label"]) != label:
            raise AssertionError(f"Category/label metadata mismatch: seed={seed}, category={category}")
        log_p0, log_p1 = float(row["logp_normal"]), float(row["logp_anomaly"])
        # Preserve the Full evaluator's float32 subtraction exactly for ranking.
        # The separately serialized log densities reconstruct this value within
        # at most 2e-5, but re-subtracting them in Python float64 can break rare
        # near-ties and change an episode AUROC by one pair.
        formula_score = log_p1 - log_p0
        score_full, score_p0 = float(row["score"]), -log_p0
        if not all(math.isfinite(value) for value in (log_p0, log_p1, score_full, score_p0)):
            raise AssertionError(f"Non-finite saved likelihood: seed={seed}, category={category}")
        maximum_saved_score_error = max(maximum_saved_score_error, abs(formula_score - score_full))
        normalized.append({
            "run_seed": seed, "split_seed": SPLIT_SEED, "category": category,
            "held_out_subtype": row["held_out_subtype"], "episode_id": episode,
            "image_path": path, "label": label, "log_p0": log_p0, "log_p1": log_p1,
            "score_full": score_full, "score_p0": score_p0,
            "support_image_paths": json.dumps(support),
            "source_file": str(source_path or (full_root / f"seed_{seed}" / "eval" / category / "query_results.csv")),
        })
    if maximum_saved_score_error > 2e-5:
        raise AssertionError(f"Saved score formula mismatch: seed={seed}, category={category}")
    return normalized, maximum_saved_score_error


def oracle_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    thresholds = np.linspace(np.min(scores), np.max(scores), 200)
    predictions = scores[:, None] >= thresholds[None, :]
    positive = labels[:, None] == 1
    tp = np.sum(predictions & positive, axis=0)
    fp = np.sum(predictions & ~positive, axis=0)
    fn = np.sum(~predictions & positive, axis=0)
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator != 0)
    return float(np.max(f1))


def metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores)), oracle_f1(labels, scores)


def mean_se(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def category_seed_metrics(records: list[dict]) -> dict:
    labels = np.asarray([row["label"] for row in records])
    full = np.asarray([row["score_full"] for row in records])
    p0 = np.asarray([row["score_p0"] for row in records])
    log_p0 = np.asarray([row["log_p0"] for row in records])
    log_p1 = np.asarray([row["log_p1"] for row in records])
    episode_ids = np.asarray([row["episode_id"] for row in records])
    episode_metrics = {"full": [], "p0": []}
    for episode in sorted(set(episode_ids)):
        mask = episode_ids == episode
        episode_metrics["full"].append(metrics(labels[mask], full[mask]))
        episode_metrics["p0"].append(metrics(labels[mask], p0[mask]))
    full_ep, p0_ep = np.asarray(episode_metrics["full"]), np.asarray(episode_metrics["p0"])
    result = {"episodes": len(set(episode_ids)), "queries": len(records)}
    for method, values in (("full", full_ep), ("p0", p0_ep)):
        for index, name in enumerate(("AUROC", "AUPRC", "F1")):
            mean, se = mean_se(values[:, index])
            result[f"{method}_{name}"], result[f"{method}_{name}_se"] = mean, se
    rho = spearmanr(full, p0).statistic
    tau = kendalltau(full, p0).statistic
    sd0, sd1 = float(log_p0.std(ddof=1)), float(log_p1.std(ddof=1))
    result.update({
        "full_minus_p0_AUROC": result["full_AUROC"] - result["p0_AUROC"],
        "absolute_AUROC_difference": abs(result["full_AUROC"] - result["p0_AUROC"]),
        "spearman_full_vs_p0": float(rho), "kendall_full_vs_p0": float(tau),
        "sd_log_p0": sd0, "sd_log_p1": sd1,
        "sd_log_p0_div_sd_log_p1": sd0 / sd1 if sd1 else float("inf"),
    })
    return result


def bootstrap_ci(values: np.ndarray, draws: int = 100000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for start in range(0, draws, 10000):
        stop = min(draws, start + 10000)
        sampled = values[rng.integers(0, len(values), size=(stop - start, len(values)))]
        means[start:stop] = sampled.mean(axis=1)
    return tuple(float(value) for value in np.percentile(means, (2.5, 97.5)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-diagnostics-root", type=Path, help="Protocol directory of a Full evaluation run using --save-query-diagnostics.")
    parser.add_argument("--full-root", type=Path, default=FULL_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    add_manifest_protocol_arguments(parser)
    args = parser.parse_args()
    protocol_index = manifest_index(args, parser)
    if args.query_diagnostics_root is not None and args.data_root is None:
        parser.error("--data-root is required with --query-diagnostics-root")
    full_root = args.full_root.resolve()

    all_queries, metric_rows, audit_rows = [], [], []
    for seed in SEEDS:
        results_path = full_root / f"seed_{seed}" / "eval" / "results.csv"
        saved_results = {row["category"]: row for row in read_csv(results_path)}
        if set(saved_results) != set(CATEGORIES):
            raise AssertionError(f"Unexpected categories in {results_path}")
        for category in CATEGORIES:
            category_root = full_root / f"seed_{seed}" / "eval" / category
            source_path = category_root / "query_results.csv"
            manifest_path = category_root / "test_episodes.json"
            if args.query_diagnostics_root is not None:
                source_path = args.query_diagnostics_root / f"seed_{seed}" / "eval" / category / "query_diagnostics.csv"
            if not source_path.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"Missing saved Full query data: {source_path} or {manifest_path}")
            episodes = json.loads(manifest_path.read_text())
            verify_test_episodes(episodes, category, protocol_index)
            rows = read_csv(source_path)
            if args.query_diagnostics_root is not None:
                for row in rows:
                    row["image_path"] = str((args.data_root / row["image_path"]).resolve())
                    row["held_out_subtype"] = saved_results[category]["held_out_subtype"]
                    row["true_label"] = row["label"]
            records, score_error = normalize_saved(rows, episodes,
                                                    seed, category, full_root, source_path)
            values = category_seed_metrics(records)
            saved_auc = float(saved_results[category]["auc_mean"])
            derived_difference = values["full_AUROC"] - saved_auc
            if abs(derived_difference) > 1e-12:
                raise AssertionError(f"Full AUROC mismatch: seed={seed}, category={category}")
            metric_rows.append({
                "category": category, "held_out_subtype": records[0]["held_out_subtype"],
                "run_seed": seed, "split_seed": SPLIT_SEED, "query_source": "saved_full_query_results",
                "saved_full_AUROC": saved_auc, "query_derived_minus_saved_full_AUROC": derived_difference,
                **values,
            })
            audit_rows.append({
                "category": category, "run_seed": seed, "queries": len(records), "episodes": values["episodes"],
                "identities_match_full_manifest": True, "support_query_disjoint": True,
                "maximum_saved_score_rounding_error": score_error, "query_recomputation": False,
                "episode_resampling": False, "audit_passed": True,
            })
            all_queries.extend(records)
            print(f"seed={seed} category={category} full={values['full_AUROC']:.6f} p0={values['p0_AUROC']:.6f}", flush=True)

    output = args.output_root.resolve()
    write_csv(output / "per_query_comparison.csv", all_queries)
    write_csv(output / "per_category_per_seed_metrics.csv", metric_rows)
    write_csv(output / "protocol_audit.csv", audit_rows)

    category_summary, ranking_summary, spread_summary = [], [], []
    for category in CATEGORIES:
        rows = sorted((row for row in metric_rows if row["category"] == category), key=lambda row: row["run_seed"])
        if len(rows) != 3 or len({row["held_out_subtype"] for row in rows}) != 1:
            raise AssertionError(f"Incomplete/mismatched category: {category}")
        summary = {"category": category, "held_out_subtype": rows[0]["held_out_subtype"]}
        for metric in ("AUROC", "AUPRC", "F1"):
            for method in ("full", "p0"):
                values = np.asarray([row[f"{method}_{metric}"] for row in rows])
                for row, value in zip(rows, values):
                    summary[f"{method}_seed{row['run_seed']}_{metric}"] = float(value)
                summary[f"{method}_mean_{metric}"] = float(values.mean())
                summary[f"{method}_sd_{metric}"] = float(values.std(ddof=1))
        summary["full_minus_p0_AUROC"] = summary["full_mean_AUROC"] - summary["p0_mean_AUROC"]
        category_summary.append(summary)
        ranking_summary.append({
            "category": category, "held_out_subtype": rows[0]["held_out_subtype"],
            "mean_spearman": float(np.mean([row["spearman_full_vs_p0"] for row in rows])),
            "sd_spearman": float(np.std([row["spearman_full_vs_p0"] for row in rows], ddof=1)),
            "mean_kendall_tau": float(np.mean([row["kendall_full_vs_p0"] for row in rows])),
            "sd_kendall_tau": float(np.std([row["kendall_full_vs_p0"] for row in rows], ddof=1)),
            "mean_absolute_AUROC_difference": float(np.mean([row["absolute_AUROC_difference"] for row in rows])),
        })
        for row in rows:
            spread_summary.append({
                "category": category, "run_seed": row["run_seed"], "split_seed": SPLIT_SEED,
                "sd_log_p0": row["sd_log_p0"], "sd_log_p1": row["sd_log_p1"],
                "sd_log_p0_div_sd_log_p1": row["sd_log_p0_div_sd_log_p1"], "queries": row["queries"],
            })
    write_csv(output / "three_seed_category_summary.csv", category_summary)
    write_csv(output / "ranking_correlation_summary.csv", ranking_summary)
    write_csv(output / "log_p0_log_p1_spread_summary.csv", spread_summary)

    seed_macro = []
    for seed in SEEDS:
        rows = [row for row in metric_rows if row["run_seed"] == seed]
        seed_macro.append({
            "run_seed": seed, "categories": len(rows),
            "full_macro_AUROC": float(np.mean([row["full_AUROC"] for row in rows])),
            "p0_macro_AUROC": float(np.mean([row["p0_AUROC"] for row in rows])),
            "full_minus_p0_macro_AUROC": float(np.mean([row["full_minus_p0_AUROC"] for row in rows])),
        })
    write_csv(output / "seed_wise_macro_summary.csv", seed_macro)

    differences = np.asarray([row["full_minus_p0_AUROC"] for row in category_summary])
    tolerance = 1e-12
    try:
        w_result = wilcoxon(differences, alternative="two-sided")
    except ValueError:
        w_result = type("Result", (), {"statistic": float("nan"), "pvalue": float("nan")})()
    t_result = ttest_rel([row["full_mean_AUROC"] for row in category_summary],
                         [row["p0_mean_AUROC"] for row in category_summary])
    ci_low, ci_high = bootstrap_ci(differences)
    statistical = [{
        "paired_categories": len(category_summary),
        "full_overall_three_seed_macro_AUROC": float(np.mean([row["full_mean_AUROC"] for row in category_summary])),
        "p0_overall_three_seed_macro_AUROC": float(np.mean([row["p0_mean_AUROC"] for row in category_summary])),
        "mean_paired_difference_full_minus_p0": float(differences.mean()),
        "median_paired_difference_full_minus_p0": float(np.median(differences)),
        "full_wins": int(np.sum(differences > tolerance)), "ties": int(np.sum(np.abs(differences) <= tolerance)),
        "p0_wins": int(np.sum(differences < -tolerance)),
        "wilcoxon_statistic": float(w_result.statistic), "wilcoxon_two_sided_p": float(w_result.pvalue),
        "paired_t_statistic": float(t_result.statistic), "paired_t_two_sided_p": float(t_result.pvalue),
        "bootstrap_95_ci_mean_difference_low": ci_low, "bootstrap_95_ci_mean_difference_high": ci_high,
        "bootstrap_unit": "category", "bootstrap_draws": 100000,
        "macro_mean_spearman": float(np.mean([row["mean_spearman"] for row in ranking_summary])),
        "macro_mean_kendall_tau": float(np.mean([row["mean_kendall_tau"] for row in ranking_summary])),
        "macro_mean_absolute_AUROC_difference": float(np.mean([row["mean_absolute_AUROC_difference"] for row in ranking_summary])),
        "macro_mean_sd_log_p0": float(np.mean([row["sd_log_p0"] for row in spread_summary])),
        "macro_mean_sd_log_p1": float(np.mean([row["sd_log_p1"] for row in spread_summary])),
        "macro_mean_sd_ratio_p0_div_p1": float(np.mean([row["sd_log_p0_div_sd_log_p1"] for row in spread_summary])),
    }]
    write_csv(output / "paired_statistical_test_summary.csv", statistical)

    audit = {
        "retraining_performed": False, "checkpoints_modified": False,
        "same_queries_as_full_baypranometa": True, "score_full_formula": "log_p1 - log_p0",
        "score_p0_formula": "-log_p0", "higher_score_means_more_anomalous_full": True,
        "higher_score_means_more_anomalous_p0": True, "episode_resampling": False,
        "same_three_seeds": True, "run_seeds": list(SEEDS), "split_seed": SPLIT_SEED,
        "query_rows": len(all_queries), "category_seed_pairs": len(metric_rows),
        "query_files_reused": 36, "query_files_reevaluated": 0,
        "all_audit_checks_passed": True,
    }
    (output / "analysis_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({**audit, **statistical[0]}, indent=2))


if __name__ == "__main__":
    main()
