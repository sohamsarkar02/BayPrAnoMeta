#!/usr/bin/env python3
"""Evaluate the three MVTec-designed Bayesian ablations on identical saved VisA episodes."""
import argparse, time
from pathlib import Path

import higher
import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from ablation_common_visa import *


def episode_metrics(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    return (float(roc_auc_score(labels, scores)),
            float(average_precision_score(labels, scores)),
            float(max(f1_score(labels, scores >= t, zero_division=0) for t in thresholds)))


def adapt_and_score(model, mode, xs, xq, debug=False):
    inner_optimizer = torch.optim.SGD(model.parameters(), lr=INNER_LR)
    with higher.innerloop_ctx(
            model, inner_optimizer, copy_initial_weights=True,
            track_higher_grads=False) as (fmodel, diffopt):
        for _ in range(INNER_STEPS):
            z_support = fmodel(xs)
            if mode == "proto":
                prototype = z_support.mean(0, keepdim=True)
                inner_loss = ((z_support - prototype) ** 2).sum(1).mean()
            else:
                posterior = fixed_lambda_posterior if mode == "fixed_lambda" else niw_posterior
                mu, sigma, dof = posterior(z_support, validate=debug)
                inner_loss = -log_student_t(z_support, mu, sigma, dof).mean()
            diffopt.step(inner_loss)

        # All modes recompute support embeddings after their own adaptation.
        z_support = fmodel(xs)
        z_query = fmodel(xq)
        prototype = z_support.mean(0, keepdim=True)
        posterior = fixed_lambda_posterior if mode == "fixed_lambda" else niw_posterior
        mu, sigma, dof = posterior(z_support, validate=debug)
        logp0 = log_student_t(z_query, mu, sigma, dof)
        logp1 = reference_log_probability(z_query)
        bayesian_score = logp1 - logp0
        euclidean_score = ((z_query - prototype) ** 2).sum(1)

    values = {
        "embedding_norm_sq": (z_query ** 2).sum(1),
        "logp0": logp0, "logp1": logp1,
        "bayesian_score": bayesian_score,
        "euclidean_score": euclidean_score,
    }
    if debug:
        assert all(torch.isfinite(value).all() for value in values.values())
    return {key: value.detach().cpu().numpy() for key, value in values.items()}


def saved_full_records(full_root, seed, category):
    path = Path(full_root) / PROTOCOL / f"seed_{seed}" / "eval" / category / "query_results.csv"
    rows = read_csv(path)
    expected = TEST_EPISODES * (Q_N + Q_A)
    if len(rows) != expected:
        raise RuntimeError(f"Incomplete Full BayPrAnoMeta records: {path} has {len(rows)}, expected {expected}")
    return rows


def saved_manifest(full_root, seed, category):
    path = Path(full_root) / PROTOCOL / f"seed_{seed}" / "eval" / category / "test_episodes.json"
    import json
    with path.open() as handle:
        return json.load(handle)


def evaluate_mode(model, mode, category, data, seed, args, manifest, saved_full=None):
    episode_auroc, episode_auprc, episode_f1, episode_euclidean = [], [], [], []
    records = []
    episode_count = 2 if args.smoke_test else args.test_episodes
    saved_index = {}
    if saved_full:
        saved_index = {(int(r["episode_id"]), r["image_path"]): r for r in saved_full}

    for episode_id, episode in enumerate(manifest[:episode_count]):
        support = episode["support"]
        normal_query = episode["normal_query"]
        anomaly_query = episode["anomaly_query"]
        if not set(support).isdisjoint(normal_query):
            raise RuntimeError("Support/query normal leakage in saved manifest.")
        paths = normal_query + anomaly_query
        labels = np.asarray([0] * Q_N + [1] * Q_A)
        values = adapt_and_score(model, mode, load_stack(support), load_stack(paths), args.debug_protocol)

        bayesian = values["bayesian_score"]
        if mode == "full":
            # Reuse completed Full outputs. Reevaluation supplies only the missing Euclidean score.
            output_values = []
            for index, path in enumerate(paths):
                old = saved_index[(episode_id, path)]
                checks = {
                    "embedding_norm_sq": float(old["embedding_norm_sq"]),
                    "logp0": float(old["logp_normal"]),
                    "logp1": float(old["logp_anomaly"]),
                    "bayesian_score": float(old["final_score"]),
                }
                for key, expected in checks.items():
                    if not np.isclose(values[key][index], expected, rtol=2e-4, atol=2e-4):
                        raise RuntimeError(
                            f"Full reevaluation mismatch for seed={seed}, {category}, episode={episode_id}, "
                            f"{path}, {key}: saved={expected}, reevaluated={values[key][index]}")
                output_values.append(checks)
            bayesian = np.asarray([row["bayesian_score"] for row in output_values])

        auroc, auprc, f1 = episode_metrics(bayesian, labels)
        episode_auroc.append(auroc); episode_auprc.append(auprc); episode_f1.append(f1)
        episode_euclidean.append(float(roc_auc_score(labels, values["euclidean_score"])))

        for index, (path, label) in enumerate(zip(paths, labels)):
            if mode == "full":
                source = saved_index[(episode_id, path)]
                norm_sq = float(source["embedding_norm_sq"])
                logp0 = float(source["logp_normal"])
                logp1 = float(source["logp_anomaly"])
                score = float(source["final_score"])
                provenance = "existing_full_query_results_plus_manifest_reevaluated_euclidean"
            else:
                norm_sq = float(values["embedding_norm_sq"][index])
                logp0 = float(values["logp0"][index])
                logp1 = float(values["logp1"][index])
                score = float(values["bayesian_score"][index])
                provenance = "checkpoint_evaluation"
            records.append({
                "method": {"full": "BayPrAnoMeta", "fixed_lambda": "BayPrAnoMeta-FixedLambda",
                           "proto": "Proto-MAML+Bayesian-score"}[mode],
                "run_seed": seed, "split_seed": args.split_seed,
                "category": category, "held_out_subtype": data["held_out_subtype"],
                "episode_id": episode_id, "image_path": path, "label": int(label),
                "embedding_norm_sq": norm_sq, "logp0": logp0, "logp1": logp1,
                "bayesian_score": score,
                "euclidean_score": float(values["euclidean_score"][index]),
                "provenance": provenance,
            })

    labels = np.asarray([r["label"] for r in records])
    bayesian = np.asarray([r["bayesian_score"] for r in records])
    euclidean = np.asarray([r["euclidean_score"] for r in records])
    rho = float(spearmanr(bayesian, euclidean).statistic)
    tau = float(kendalltau(bayesian, euclidean).statistic)
    bayes_mean, euclidean_mean = float(np.mean(episode_auroc)), float(np.mean(episode_euclidean))
    result = {
        "method": records[0]["method"], "category": category,
        "held_out_subtype": data["held_out_subtype"], "run_seed": seed,
        "split_seed": args.split_seed, "AUROC": bayes_mean,
        "AUROC_se": float(np.std(episode_auroc, ddof=1) / np.sqrt(len(episode_auroc))) if len(episode_auroc) > 1 else 0.0,
        "AUPRC": float(np.mean(episode_auprc)),
        "F1": float(np.mean(episode_f1)),
        "euclidean_AUROC": euclidean_mean,
        "absolute_AUROC_difference": abs(bayes_mean - euclidean_mean),
        "pooled_bayesian_AUROC": float(roc_auc_score(labels, bayesian)),
        "pooled_euclidean_AUROC": float(roc_auc_score(labels, euclidean)),
        "spearman_rho": rho, "kendall_tau": tau,
        "test_episodes": episode_count,
        "score_orientation": "higher logp1-logp0 is anomalous; no inversion",
    }
    return result, records


def spread_rows(records):
    rows = []
    labels = np.asarray([r["label"] for r in records])
    for group, mask in (("normal", labels == 0), ("anomaly", labels == 1),
                        ("all", np.ones(len(labels), dtype=bool))):
        row = {
            "method": "BayPrAnoMeta", "run_seed": records[0]["run_seed"],
            "split_seed": records[0]["split_seed"], "category": records[0]["category"],
            "held_out_subtype": records[0]["held_out_subtype"],
            "query_group": group, "n_query_records": int(mask.sum()),
        }
        for field, prefix in (
                ("embedding_norm_sq", "norm_sq"), ("logp0", "logp0"),
                ("logp1", "logp1"), ("bayesian_score", "bayesian_score")):
            summary = descriptive([record[field] for record, keep in zip(records, mask) if keep])
            row.update({f"{prefix}_{name}": value for name, value in summary.items()})
        rows.append(row)
    return rows


def existing_proto_auroc(proto_root, seed, category):
    path = Path(proto_root) / PROTOCOL / f"seed_{seed}" / "evaluation" / "results.csv"
    rows = read_csv(path)
    match = [row for row in rows if row["category"] == category]
    if len(match) != 1: raise RuntimeError(f"Missing existing Proto-MAML result for {category}, seed {seed}")
    return float(match[0]["auc_mean"])


def evaluate_seed(seed, args):
    set_seed(seed)
    clients = load_clients(
        args.base_root, args.split_seed, args.validation_fraction, args.split_csv,
        args.category, Path(args.output_root) / f"seed_{seed}" / "split_audit",
        subset_file=args.subset_file)
    paths = checkpoint_paths(seed, args.full_run_root, args.proto_run_root, args.fixed_run_root)
    for mode, path in paths.items():
        if not path.exists(): raise FileNotFoundError(f"Missing {mode} checkpoint: {path}")
    models = {mode: EmbeddingNet().to(DEVICE) for mode in paths}
    for mode, model in models.items():
        model.load_state_dict(torch.load(paths[mode], map_location=DEVICE)); model.eval()

    output = Path(args.output_root) / f"seed_{seed}"
    results, spread, rank = [], [], []
    monitor = PeakMemoryMonitor(); monitor.start(); started = time.perf_counter()
    for category, data in clients.items():
        manifest = saved_manifest(args.full_evaluation_root or args.full_run_root, seed, category)
        if args.subset_file is not None:
            normal_paths = set(data["test"]["normal"])
            anomaly_paths = set(data["test"]["anomaly"])
            for episode in manifest:
                if (not normal_paths.issuperset(episode["support"] + episode["normal_query"])
                        or not anomaly_paths.issuperset(episode["anomaly_query"])):
                    raise ValueError(f"Saved VisA manifest contains images outside the test pool: {category}")
        full_saved = saved_full_records(args.full_evaluation_root or args.full_run_root, seed, category)
        for mode in ("full", "fixed_lambda", "proto"):
            result, records = evaluate_mode(
                models[mode], mode, category, data, seed, args, manifest,
                full_saved if mode == "full" else None)
            results.append(result)
            method_dir = output / mode / category
            write_csv(method_dir / "query_values.csv", records)
            write_json(method_dir / "metrics.json", result)
            if mode == "full":
                spread.extend(spread_rows(records))
            if mode in ("full", "proto"):
                rank.append({
                    "method": result["method"], "category": category,
                    "held_out_subtype": data["held_out_subtype"], "run_seed": seed,
                    "split_seed": args.split_seed,
                    "bayesian_AUROC": result["AUROC"],
                    "euclidean_AUROC": result["euclidean_AUROC"],
                    "absolute_AUROC_difference": result["absolute_AUROC_difference"],
                    "spearman_rho": result["spearman_rho"], "kendall_tau": result["kendall_tau"],
                })
        results.append({
            "method": "Classical Proto-MAML", "category": category,
            "held_out_subtype": data["held_out_subtype"], "run_seed": seed,
            "split_seed": args.split_seed,
            "AUROC": existing_proto_auroc(args.proto_evaluation_root or args.proto_run_root, seed, category),
            "provenance": "existing Proto_MAML results.csv",
        })

    elapsed = time.perf_counter() - started; monitor.stop()
    write_csv(output / "method_results.csv", results)
    write_csv(output / "p0_p1_embedding_spread.csv", spread)
    write_csv(output / "bayesian_vs_euclidean_rank.csv", rank)
    write_json(output / "runtime_memory.json", {
        "evaluation_sec": elapsed,
        "evaluation_seconds_per_episode": elapsed / max(1, len(clients) * 3 * (2 if args.smoke_test else args.test_episodes)),
        "peak_rss_mb": monitor.peak_mb, "peak_gpu_memory_mb": 0.0,
        "num_parameters": sum(p.numel() for p in models["full"].parameters()),
        "full_training_reused": True, "proto_training_reused": True,
    })
    return results, spread, rank


def aggregate(by_seed, seeds, args):
    output = Path(args.output_root)
    all_results = [row for seed in seeds for row in by_seed[seed][0]]
    all_spread = [row for seed in seeds for row in by_seed[seed][1]]
    all_rank = [row for seed in seeds for row in by_seed[seed][2]]
    write_csv(output / "p0_p1_embedding_spread.csv", all_spread)
    write_csv(output / "bayesian_vs_euclidean_rank.csv", all_rank)

    rank_summary = []
    for method in sorted({row["method"] for row in all_rank}):
        for category in sorted({row["category"] for row in all_rank if row["method"] == method}):
            rows = [row for row in all_rank if row["method"] == method and row["category"] == category]
            summary = {"method": method, "category": category,
                       "held_out_subtype": rows[0]["held_out_subtype"], "n_seeds": len(rows)}
            for field in ("bayesian_AUROC", "euclidean_AUROC", "absolute_AUROC_difference",
                          "spearman_rho", "kendall_tau"):
                values = np.asarray([row[field] for row in rows])
                summary[f"{field}_mean"] = float(values.mean())
                summary[f"{field}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            rank_summary.append(summary)
    write_csv(output / "bayesian_vs_euclidean_rank_three_seed_summary.csv", rank_summary)

    lookup = {(row["run_seed"], row["category"], row["method"]): row for row in all_results}
    categories = sorted({row["category"] for row in all_results})
    summary_rows = []
    method_columns = [
        ("protomaml", "Classical Proto-MAML"),
        ("proto_bayesian", "Proto-MAML+Bayesian-score"),
        ("fixed_lambda", "BayPrAnoMeta-FixedLambda"),
        ("full", "BayPrAnoMeta"),
    ]
    for category in categories:
        row = {"category": category}
        held = {lookup[(seed, category, method)]["held_out_subtype"]
                for seed in seeds for _, method in method_columns}
        if len(held) != 1: raise RuntimeError(f"Held-out group mismatch for {category}")
        row["held_out_subtype"] = held.pop()
        for prefix, method in method_columns:
            values = np.asarray([lookup[(seed, category, method)]["AUROC"] for seed in seeds])
            for seed, value in zip(seeds, values): row[f"{prefix}_seed{seed}_AUROC"] = float(value)
            row[f"{prefix}_mean_AUROC"] = float(values.mean())
            row[f"{prefix}_sample_sd_AUROC"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    write_csv(output / "bayesian_ablation_summary.csv", summary_rows)
    write_json(output / "bayesian_ablation_summary.json", summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[1, 2, 3])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None, default=BASE_ROOT)
    parser.add_argument("--split-csv", default="split_csv/2cls_highshot.csv"); parser.add_argument("--subset-file", help='Paper image allowlist. Required for exact reproduction of the paper experiments.')
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--full-evaluation-root", help="Output root of the fresh Full evaluation.")
    parser.add_argument("--proto-evaluation-root", help="Output root of the fresh Proto-MAML evaluation.")
    parser.add_argument("--full-run-root", default=str(FULL_RUN_ROOT))
    parser.add_argument("--proto-run-root", default=str(PROTO_RUN_ROOT))
    parser.add_argument("--fixed-run-root", default=str(FIXED_RUN_ROOT))
    parser.add_argument("--validation-fraction", type=float, default=0.30)
    parser.add_argument("--test-episodes", type=int, default=TEST_EPISODES)
    parser.add_argument("--category")
    parser.add_argument("--debug-protocol", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.split_seed != 42: parser.error("SPLIT_SEED is fixed at 42")
    if args.seed is not None and args.seeds is not None: parser.error("Use either --seed or --seeds")
    args.seeds = [args.seed] if args.seed is not None else (args.seeds or [1, 2, 3])
    if args.smoke_test:
        args.seeds = [args.seed or 1]
        args.category = args.category or discover_categories(args.base_root, args.split_csv, subset_file=args.subset_file)[0]
        args.output_root += "_smoke"
        args.fixed_run_root += "_smoke"

    by_seed = {}
    for seed in args.seeds:
        by_seed[seed] = evaluate_seed(seed, args)
    aggregate(by_seed, args.seeds, args)


if __name__ == "__main__":
    main()

