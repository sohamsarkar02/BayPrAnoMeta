#!/usr/bin/env python3
"""Evaluation-only p0 score ablation using Full BayPrAnoMeta queries/checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import higher
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from scipy.stats import kendalltau, spearmanr, ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SOURCE_CODE_ROOT = PROJECT_ROOT / "Bayesian_ProtoMAML"
sys.path.insert(0, str(SOURCE_CODE_ROOT))
from utils.utils_data_fedcontrast_leave1out import load_images, transform_eval  # noqa: E402
from utils.utils_niw import log_student_t, niw_posterior  # noqa: E402

FULL_ROOT = SOURCE_CODE_ROOT / "runs" / "baypranometa_avgpool_three_seed" / "revised_clean"
OUTPUT_ROOT = HERE / "results" / "p0_only_ablation"
SEEDS = (1, 2, 3)
SPLIT_SEED = 42
INNER_LR = 5e-4
DEVICE = torch.device("cpu")


class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        base = models.resnet18(weights=None)
        base.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(nn.Linear(512, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def read_csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def load_stack(paths):
    images = load_images(paths, transform_eval)
    if len(images) != len(paths):
        raise RuntimeError("An image could not be loaded; refusing query/label misalignment")
    return torch.stack(images).to(DEVICE)


def reference_log_probability(z):
    d = z.shape[1]
    return log_student_t(z, torch.zeros(d, device=z.device, dtype=z.dtype),
                         100.0 * torch.eye(d, device=z.device, dtype=z.dtype),
                         torch.tensor(2.0, device=z.device, dtype=z.dtype))


def saved_query_path(full_root: Path, seed: int, category: str):
    fresh = full_root / f"seed_{seed}" / "eval" / category / "query_results.csv"
    if fresh.is_file():
        return fresh
    return full_root / f"seed_{seed}" / "diagnostics" / "BayPrAnoMeta" / f"seed_{seed}" / category / "query_scores.csv"


def expected_query_keys(manifest):
    keys = []
    for episode in manifest:
        for path in episode["normal_query"]:
            keys.append((int(episode["episode"]), path, 0))
        for path in episode["anomaly_query"]:
            keys.append((int(episode["episode"]), path, 1))
    return keys


def normalize_saved(rows, manifest, seed, category, held_out):
    expected = expected_query_keys(manifest)
    actual = [(int(r["episode_id"]), r["image_path"], int(r["label"])) for r in rows]
    if actual != expected:
        raise AssertionError(f"Saved query identities do not match manifest: seed={seed}, category={category}")
    normalized = []
    for row, (_, path, label) in zip(rows, expected):
        logp1, logp0 = float(row["logp_anomaly"]), float(row["logp_normal"])
        score_full = logp1 - logp0
        if not math.isclose(score_full, float(row["score"]), rel_tol=1e-6, abs_tol=1e-5):
            raise AssertionError(f"Saved full score formula mismatch: seed={seed}, category={category}")
        episode = int(row["episode_id"])
        normalized.append({"run_seed": seed, "split_seed": SPLIT_SEED, "category": category,
            "held_out_subtype": held_out, "episode_id": episode, "image_path": path, "label": label,
            "log_p0": logp0, "log_p1": logp1, "score_full": score_full, "score_p0": -logp0,
            "support_image_paths": json.dumps(manifest[episode]["support"]), "source": "existing_query_diagnostics"})
    return normalized


def normalize_cache(rows):
    numeric_float = {"log_p0", "log_p1", "score_full", "score_p0"}
    numeric_int = {"run_seed", "split_seed", "episode_id", "label"}
    normalized = []
    for row in rows:
        normalized.append({key: float(value) if key in numeric_float else int(value) if key in numeric_int else value
                           for key, value in row.items()})
    return normalized


def recover_queries(model, manifest, seed, category, held_out):
    records = []
    for position, episode in enumerate(manifest):
        episode_id = int(episode["episode"])
        if episode_id != position:
            raise AssertionError(f"Nonsequential episode IDs for {category}")
        support = episode["support"]
        paths = episode["normal_query"] + episode["anomaly_query"]
        labels = [0] * len(episode["normal_query"]) + [1] * len(episode["anomaly_query"])
        if set(support) & set(paths): raise AssertionError(f"Support/query overlap: {category}, episode {episode_id}")
        xs, xq = load_stack(support), load_stack(paths)
        inner_opt = torch.optim.SGD(model.parameters(), lr=INNER_LR)
        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True, track_higher_grads=False) as (fmodel, diffopt):
            z0 = fmodel(xs); mu0, sigma0, dof0 = niw_posterior(z0)
            diffopt.step(-log_student_t(z0, mu0, sigma0, dof0).mean())
            with torch.no_grad():
                zsup = fmodel(xs); mu, sigma, dof = niw_posterior(zsup); zq = fmodel(xq)
                logp1 = reference_log_probability(zq).cpu().numpy()
                logp0 = log_student_t(zq, mu, sigma, dof).cpu().numpy()
        if not np.isfinite(np.r_[logp0, logp1]).all(): raise RuntimeError(f"Non-finite likelihood: {category}, episode {episode_id}")
        for path, label, p0, p1 in zip(paths, labels, logp0, logp1):
            records.append({"run_seed": seed, "split_seed": SPLIT_SEED, "category": category,
                "held_out_subtype": held_out, "episode_id": episode_id, "image_path": path, "label": label,
                "log_p0": float(p0), "log_p1": float(p1), "score_full": float(p1 - p0), "score_p0": float(-p0),
                "support_image_paths": json.dumps(support), "source": "reevaluated_existing_checkpoint"})
    return records


def oracle_f1(labels, scores):
    thresholds = np.linspace(np.min(scores), np.max(scores), 200)
    predictions = scores[:, None] >= thresholds[None, :]
    positive = labels[:, None] == 1
    tp = np.sum(predictions & positive, axis=0)
    fp = np.sum(predictions & ~positive, axis=0)
    fn = np.sum(~predictions & positive, axis=0)
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator != 0)
    return float(np.max(f1))


def metric_tuple(labels, scores):
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores)), oracle_f1(labels, scores)


def mean_se(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def category_seed_metrics(records):
    labels = np.asarray([r["label"] for r in records])
    full = np.asarray([r["score_full"] for r in records])
    p0 = np.asarray([r["score_p0"] for r in records])
    logp0 = np.asarray([r["log_p0"] for r in records]); logp1 = np.asarray([r["log_p1"] for r in records])
    episodes = sorted(set(r["episode_id"] for r in records))
    full_episode, p0_episode = [], []
    for episode in episodes:
        mask = np.asarray([r["episode_id"] == episode for r in records])
        full_episode.append(metric_tuple(labels[mask], full[mask])); p0_episode.append(metric_tuple(labels[mask], p0[mask]))
    full_episode, p0_episode = np.asarray(full_episode), np.asarray(p0_episode)
    full_means = [mean_se(full_episode[:, i]) for i in range(3)]
    p0_means = [mean_se(p0_episode[:, i]) for i in range(3)]
    rho = spearmanr(full, p0).statistic; tau = kendalltau(full, p0).statistic
    sd0, sd1 = float(logp0.std(ddof=1)), float(logp1.std(ddof=1))
    return {
        "episodes": len(episodes), "queries": len(records),
        "full_AUROC": full_means[0][0], "full_AUROC_se": full_means[0][1],
        "full_AUPRC": full_means[1][0], "full_AUPRC_se": full_means[1][1],
        "full_F1": full_means[2][0], "full_F1_se": full_means[2][1],
        "p0_AUROC": p0_means[0][0], "p0_AUROC_se": p0_means[0][1],
        "p0_AUPRC": p0_means[1][0], "p0_AUPRC_se": p0_means[1][1],
        "p0_F1": p0_means[2][0], "p0_F1_se": p0_means[2][1],
        "full_minus_p0_AUROC": full_means[0][0] - p0_means[0][0],
        "absolute_AUROC_difference": abs(full_means[0][0] - p0_means[0][0]),
        "spearman_full_vs_p0": float(rho), "kendall_full_vs_p0": float(tau),
        "sd_log_p0": sd0, "sd_log_p1": sd1, "sd_log_p0_div_sd_log_p1": sd0 / sd1 if sd1 else float("inf"),
    }


def bootstrap_ci(values, seed=42, draws=100000):
    rng = np.random.default_rng(seed); values = np.asarray(values, dtype=float)
    means = np.empty(draws)
    for start in range(0, draws, 10000):
        stop = min(start + 10000, draws)
        means[start:stop] = values[rng.integers(0, len(values), size=(stop-start, len(values)))].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, help="Dataset root for resolving portable query diagnostics.")
    parser.add_argument("--query-diagnostics-root", type=Path, help="Protocol directory of a Full evaluation run using --save-query-diagnostics.")
    parser.add_argument("--full-root", type=Path, default=FULL_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    if args.query_diagnostics_root is not None and args.data_root is None:
        parser.error("--data-root is required with --query-diagnostics-root")
    if tuple(args.seeds) != SEEDS: raise ValueError("Full analysis requires seeds 1 2 3")
    all_queries, metric_rows, audit_rows = [], [], []
    for seed in args.seeds:
        seed_root = args.full_root / f"seed_{seed}"
        checkpoint = seed_root / "checkpoints" / "final_model.pth"
        result_rows = {r["category"]: r for r in read_csv(seed_root / "eval" / "results.csv")}
        manifests = sorted((seed_root / "eval").glob("*/test_episodes.json"))
        model = None
        for manifest_path in manifests:
            category = manifest_path.parent.name; manifest = json.loads(manifest_path.read_text())
            held_out = result_rows[category]["held_out_subtype"]
            source_path = saved_query_path(args.full_root, seed, category)
            cache_path = args.output_root / "source_query_cache" / f"seed_{seed}" / f"{category}.csv"
            if args.query_diagnostics_root is not None:
                rows = read_csv(args.query_diagnostics_root / f"seed_{seed}" / "eval" / category / "query_diagnostics.csv")
                for row in rows:
                    row["image_path"] = str((args.data_root / row["image_path"]).resolve())
                records = normalize_saved(rows, manifest, seed, category, held_out)
                source = "reused"
            elif cache_path.is_file():
                records = normalize_cache(read_csv(cache_path))
                source = "reused" if records[0]["source"] == "existing_query_diagnostics" else "reevaluated"
            elif source_path.is_file():
                records = normalize_saved(read_csv(source_path), manifest, seed, category, held_out); source = "reused"
            else:
                if model is None:
                    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
                    model = EmbeddingNet().to(DEVICE)
                    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True)); model.eval()
                records = recover_queries(model, manifest, seed, category, held_out); source = "reevaluated"
            if not cache_path.is_file(): write_csv(cache_path, records)
            metrics = category_seed_metrics(records)
            original_auc = float(result_rows[category]["auc_mean"])
            result_difference = abs(metrics["full_AUROC"] - original_auc)
            if source == "reevaluated" and result_difference > 1e-6:
                raise AssertionError(f"Recomputed Full AUROC differs from saved result: seed={seed}, {category}: {metrics['full_AUROC']} vs {original_auc}")
            metric_rows.append({"category": category, "held_out_subtype": held_out, "run_seed": seed,
                "split_seed": SPLIT_SEED, "query_source": source, "saved_full_AUROC": original_auc,
                "query_derived_minus_saved_full_AUROC": metrics["full_AUROC"] - original_auc, **metrics})
            audit_rows.append({"category": category, "run_seed": seed, "expected_queries": len(manifest)*16,
                "actual_queries": len(records), "identities_from_saved_manifest": True, "source": source,
                "query_derived_full_AUROC": metrics["full_AUROC"], "saved_full_AUROC": original_auc,
                "absolute_AUROC_difference": result_difference,
                "validation_rule": "must_match_within_1e-6" if source == "reevaluated" else "reported_only_for_existing_diagnostics",
                "validation_passed": source == "reused" or result_difference <= 1e-6})
            all_queries.extend(records)
            print(f"seed={seed} category={category} source={source} full={metrics['full_AUROC']:.4f} p0={metrics['p0_AUROC']:.4f}", flush=True)
    out = args.output_root; out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_query_comparison.csv", all_queries)
    write_csv(out / "per_category_per_seed_metrics.csv", metric_rows)
    categories = sorted(set(r["category"] for r in metric_rows)); summary_rows, rank_rows = [], []
    for category in categories:
        rows = sorted((r for r in metric_rows if r["category"] == category), key=lambda r: r["run_seed"])
        if len(rows) != 3 or len({r["held_out_subtype"] for r in rows}) != 1: raise AssertionError(category)
        item = {"category": category, "held_out_subtype": rows[0]["held_out_subtype"]}
        for metric in ("AUROC", "AUPRC", "F1"):
            for method in ("full", "p0"):
                values = np.asarray([r[f"{method}_{metric}"] for r in rows])
                for r, value in zip(rows, values): item[f"{method}_seed{r['run_seed']}_{metric}"] = float(value)
                item[f"{method}_mean_{metric}"] = float(values.mean()); item[f"{method}_sd_{metric}"] = float(values.std(ddof=1))
        item["full_minus_p0_AUROC"] = item["full_mean_AUROC"] - item["p0_mean_AUROC"]
        summary_rows.append(item)
        rank_rows.append({"category": category, "held_out_subtype": rows[0]["held_out_subtype"],
            "mean_spearman": float(np.mean([r["spearman_full_vs_p0"] for r in rows])),
            "sd_spearman": float(np.std([r["spearman_full_vs_p0"] for r in rows], ddof=1)),
            "mean_kendall_tau": float(np.mean([r["kendall_full_vs_p0"] for r in rows])),
            "sd_kendall_tau": float(np.std([r["kendall_full_vs_p0"] for r in rows], ddof=1)),
            "mean_absolute_AUROC_difference": float(np.mean([r["absolute_AUROC_difference"] for r in rows])),
            "mean_sd_log_p0": float(np.mean([r["sd_log_p0"] for r in rows])),
            "mean_sd_log_p1": float(np.mean([r["sd_log_p1"] for r in rows])),
            "mean_sd_ratio_p0_div_p1": float(np.mean([r["sd_log_p0_div_sd_log_p1"] for r in rows]))})
    write_csv(out / "three_seed_category_summary.csv", summary_rows)
    write_csv(out / "ranking_correlation_summary.csv", rank_rows)
    differences = np.asarray([r["full_minus_p0_AUROC"] for r in summary_rows])
    tolerance = 1e-12; wins = int(np.sum(differences > tolerance)); ties = int(np.sum(np.abs(differences) <= tolerance)); losses = int(np.sum(differences < -tolerance))
    try: w = wilcoxon(differences, alternative="two-sided")
    except ValueError: w = type("Result", (), {"statistic": float("nan"), "pvalue": float("nan")})()
    t = ttest_rel([r["full_mean_AUROC"] for r in summary_rows], [r["p0_mean_AUROC"] for r in summary_rows])
    low, high = bootstrap_ci(differences)
    statistical = [{"paired_categories": len(summary_rows),
        "full_overall_macro_AUROC": float(np.mean([r["full_mean_AUROC"] for r in summary_rows])),
        "p0_overall_macro_AUROC": float(np.mean([r["p0_mean_AUROC"] for r in summary_rows])),
        "full_overall_macro_AUPRC": float(np.mean([r["full_mean_AUPRC"] for r in summary_rows])),
        "p0_overall_macro_AUPRC": float(np.mean([r["p0_mean_AUPRC"] for r in summary_rows])),
        "full_overall_macro_F1": float(np.mean([r["full_mean_F1"] for r in summary_rows])),
        "p0_overall_macro_F1": float(np.mean([r["p0_mean_F1"] for r in summary_rows])),
        "mean_paired_difference_full_minus_p0": float(differences.mean()), "median_paired_difference_full_minus_p0": float(np.median(differences)),
        "full_wins": wins, "ties": ties, "p0_wins": losses, "wilcoxon_statistic": float(w.statistic), "wilcoxon_two_sided_p": float(w.pvalue),
        "paired_t_statistic": float(t.statistic), "paired_t_two_sided_p": float(t.pvalue),
        "bootstrap_95_ci_mean_difference_low": low, "bootstrap_95_ci_mean_difference_high": high,
        "bootstrap_unit": "category", "bootstrap_draws": 100000,
        "macro_mean_spearman": float(np.mean([r["mean_spearman"] for r in rank_rows])),
        "macro_mean_kendall_tau": float(np.mean([r["mean_kendall_tau"] for r in rank_rows])),
        "macro_mean_absolute_AUROC_difference": float(np.mean([r["mean_absolute_AUROC_difference"] for r in rank_rows])),
        "macro_mean_sd_log_p0": float(np.mean([r["mean_sd_log_p0"] for r in rank_rows])),
        "macro_mean_sd_log_p1": float(np.mean([r["mean_sd_log_p1"] for r in rank_rows])),
        "macro_mean_sd_ratio_p0_div_p1": float(np.mean([r["mean_sd_ratio_p0_div_p1"] for r in rank_rows]))}]
    write_csv(out / "paired_statistical_test_summary.csv", statistical)
    write_csv(out / "protocol_audit.csv", audit_rows)
    audit = {"no_retraining_performed": True, "checkpoints_modified": False, "same_queries_as_full_baypranometa": True,
        "score_full_formula": "log_p1 - log_p0", "score_p0_formula": "-log_p0",
        "higher_score_means_more_anomalous_full": True, "higher_score_means_more_anomalous_p0": True,
        "query_rows": len(all_queries), "category_seed_pairs": len(metric_rows),
        "reused_pairs": sum(r["source"] == "reused" for r in audit_rows), "reevaluated_pairs": sum(r["source"] == "reevaluated" for r in audit_rows)}
    (out / "analysis_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({**audit, **statistical[0]}, indent=2))


if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    main()
