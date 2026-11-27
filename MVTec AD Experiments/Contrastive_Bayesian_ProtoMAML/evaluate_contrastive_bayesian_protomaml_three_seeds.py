#!/usr/bin/env python3
"""Standalone evaluation for clean Contrastive BayPrAnoMeta, seeds 1/2/3.

The split seed fixes the held-out anomaly subtype. Each run seed controls
evaluation sampling.
"""
import argparse
import csv
import json
import os
import random
import threading
import time
from pathlib import Path

import higher
import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score

from utils.utils_below_chance_diagnostics import save_below_chance_diagnostics
from utils.utils_data_fedcontrast_leave1out import (
    load_client_dataset_leave1out, load_images, transform_eval,
)
from utils.utils_eval_stats import aggregate_with_se, compute_episode_metrics
from utils.utils_niw import log_student_t, niw_posterior
from utils.utils_plotting import plot_pr_curve, plot_roc_curve, plot_score_histogram


BASE_ROOT = None
CLIENTS = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
           "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
RUN_ROOT = "runs/contrastive_baypranometa_three_seed_fixed_split42_25ep_30epochs_100test"
PROTOCOL = "revised_clean"
SPLIT_SEED = 42
TEST_EPISODES, INNER_STEPS, INNER_LR = 100, 1, 5e-4
K_SHOT, Q_N, Q_A = 5, 12, 4
DEVICE = torch.device("cpu")


class Monitor:
    def __init__(self):
        self.process, self.peak_mb, self.live = psutil.Process(os.getpid()), 0.0, False

    def start(self):
        self.live = True
        def poll():
            while self.live:
                self.peak_mb = max(self.peak_mb, self.process.memory_info().rss / 1024 ** 2)
                time.sleep(0.1)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()

    def stop(self):
        self.live = False
        self.thread.join()


class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # These names intentionally match the standalone training script so
        # strict checkpoint loading verifies architecture compatibility.
        self.f = nn.Sequential(*list(backbone.children())[:-1])
        self.h = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.LayerNorm(128))

    def forward(self, images):
        return self.h(self.f(images).flatten(1))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))


def set_run_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def tensor(paths):
    images = load_images(paths, transform_eval)
    if len(images) != len(paths):
        raise RuntimeError("An evaluation image could not be loaded.")
    return torch.stack(images).to(DEVICE)


def anomaly_reference(z):
    dim = z.shape[1]
    return log_student_t(z, torch.zeros(dim, device=DEVICE),
                         100.0 * torch.eye(dim, device=DEVICE), torch.tensor(2.0, device=DEVICE))


def sample_test_episode(data, rng, debug_protocol):
    normals = rng.sample(data["test"]["normal"], K_SHOT + Q_N)
    support, normal_query = normals[:K_SHOT], normals[K_SHOT:]
    anomaly_query = rng.sample(data["test"]["anomaly"], Q_A)
    if debug_protocol:
        assert set(support).isdisjoint(normal_query)
        assert all(Path(p).parent.name == data["held_out_subtype"] for p in anomaly_query)
    return support, normal_query, anomaly_query


def save_confusion_matrix(labels, scores, names, out, category, held_out):
    labels, scores = np.asarray(labels), np.asarray(scores)
    threshold = float(np.percentile(scores[labels == 0], 95))
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    plt.title(f"{category} | Held-out: {held_out}"); plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png"); plt.close()
    records = []
    for index in np.where(labels != predictions)[0]:
        records.append({"path": names[index], "filename": Path(names[index]).name,
                        "ground_truth": int(labels[index]), "prediction": int(predictions[index]),
                        "score": float(scores[index]), "threshold": threshold,
                        "error_type": "false_positive" if labels[index] == 0 else "false_negative"})
    write_json(out / "false_positives.json", [r for r in records if r["error_type"] == "false_positive"])
    write_json(out / "false_negatives.json", [r for r in records if r["error_type"] == "false_negative"])
    return threshold, matrix, records


def save_tsne(embeddings, labels, out, category, run_seed):
    embeddings, labels = np.vstack(embeddings), np.asarray(labels)
    # t-SNE needs a perplexity strictly below the number of samples.
    perplexity = min(30, max(5, (len(embeddings) - 1) // 3))
    points = TSNE(n_components=2, perplexity=perplexity, init="random", random_state=run_seed).fit_transform(embeddings)
    plt.figure(figsize=(6, 5))
    plt.scatter(points[labels == 0, 0], points[labels == 0, 1], s=5, label="Normal")
    plt.scatter(points[labels == 1, 0], points[labels == 1, 1], s=5, label="Anomaly")
    plt.legend(); plt.title(f"{category} t-SNE"); plt.tight_layout(); plt.savefig(out / "tsne.png"); plt.close()


def save_all_clients_pr(evaluation_root, categories):
    """Preserve the legacy combined PR-curve output without resampling episodes."""
    plt.figure(figsize=(7, 5))
    for category in categories:
        csv_path = Path(evaluation_root) / category / "query_scores.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as handle:
            rows = list(csv.DictReader(handle))
        labels = np.asarray([int(row["label"]) for row in rows])
        scores = np.asarray([float(row["final_score"]) for row in rows])
        precision, recall, _ = precision_recall_curve(labels, scores)
        plt.plot(recall, precision, linewidth=1, label=category)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curves Across Categories")
    plt.legend(ncol=2, fontsize=7); plt.tight_layout()
    plt.savefig(Path(evaluation_root) / "all_clients_pr_curves.png"); plt.close()


def evaluate_client(model, category, data, run_seed, split_seed, output_root, debug_protocol):
    print(f"Evaluating run_seed={run_seed}: {category}; held-out={data['held_out_subtype']}")
    rng = random.Random(run_seed + 2)  # Persistent: never reset inside an episode.
    aucs, prs, f1s, reverse_aucs = [], [], [], []
    scores, labels, names, embeddings, query_records, manifest = [], [], [], [], [], []
    overlap_violations = 0
    model.eval()
    for episode_id in range(TEST_EPISODES):
        support, normal_query, anomaly_query = sample_test_episode(data, rng, debug_protocol)
        if not set(support).isdisjoint(normal_query): overlap_violations += 1
        query_paths, episode_labels = normal_query + anomaly_query, [0] * Q_N + [1] * Q_A
        xs, xq = tensor(support), tensor(query_paths)
        inner_optimizer = torch.optim.SGD(model.parameters(), lr=INNER_LR)
        with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=True,
                                  track_higher_grads=False) as (fmodel, diffopt):
            z_support = fmodel(xs)
            mu, sigma, dof = niw_posterior(z_support, validate=debug_protocol)
            diffopt.step(-log_student_t(z_support, mu, sigma, dof).mean())
            # The posterior and queries must be in the same adapted representation.
            z_support_adapted = fmodel(xs)
            mu, sigma, dof = niw_posterior(z_support_adapted, validate=debug_protocol)
            z_query = fmodel(xq)
            logp0 = log_student_t(z_query, mu, sigma, dof)
            logp1 = anomaly_reference(z_query)
            episode_scores = (logp1 - logp0).detach().cpu().numpy()
            episode_logp0, episode_logp1 = logp0.detach().cpu().numpy(), logp1.detach().cpu().numpy()
            episode_embeddings = z_query.detach().cpu().numpy()
        auc, pr, f1 = compute_episode_metrics(episode_scores, episode_labels)
        aucs.append(float(auc)); prs.append(float(pr)); f1s.append(float(f1))
        reverse_aucs.append(float(roc_auc_score(episode_labels, -episode_scores)))
        scores.extend(episode_scores); labels.extend(episode_labels); names.extend(query_paths); embeddings.append(episode_embeddings)
        manifest.append({"episode_id": episode_id, "support": support, "normal_query": normal_query, "anomaly_query": anomaly_query})
        for path, label, p0, p1, score in zip(query_paths, episode_labels, episode_logp0, episode_logp1, episode_scores):
            query_records.append({"episode_id": episode_id, "path": path, "label": int(label),
                                  "logp_normal": float(p0), "logp_anomaly": float(p1), "final_score": float(score)})
    out = Path(output_root) / category; out.mkdir(parents=True, exist_ok=True)
    auc_mean, auc_se = aggregate_with_se(aucs); pr_mean, pr_se = aggregate_with_se(prs); f1_mean, f1_se = aggregate_with_se(f1s)
    score_array, label_array = np.asarray(scores), np.asarray(labels)
    plot_roc_curve(label_array, score_array, out / "roc.png")
    plot_pr_curve(label_array, score_array, out / "pr.png")
    plot_score_histogram(label_array, score_array, out / "score_hist.png")
    threshold, matrix, errors = save_confusion_matrix(label_array, score_array, names, out, category, data["held_out_subtype"])
    save_tsne(embeddings, label_array, out, category, run_seed)
    write_json(out / "episode_manifest.json", manifest)
    with (out / "query_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(query_records[0])); writer.writeheader(); writer.writerows(query_records)
    protocol = {"train_test_normal_disjoint": set(data["train"]["normal"]).isdisjoint(data["test"]["normal"]),
                "episode_overlap_violations": overlap_violations, "labels_match_expected": all(x in (0, 1) for x in label_array),
                "test_anomalies_are_held_out_subtype": all(Path(p).parent.name == data["held_out_subtype"] for p in data["test"]["anomaly"])}
    metric = {"category": category, "held_out_subtype": data["held_out_subtype"], "run_seed": run_seed, "split_seed": split_seed,
              "auc_mean": float(auc_mean), "auc_se": float(auc_se), "pr_mean": float(pr_mean), "pr_se": float(pr_se),
              "f1_mean": float(f1_mean), "f1_se": float(f1_se), "pooled_auc": float(roc_auc_score(label_array, score_array)),
              "pooled_reversed_score_auc": float(roc_auc_score(label_array, -score_array)), "threshold_95_normal": threshold,
              "confusion_matrix": matrix.tolist(), "n_false_positives": sum(x["error_type"] == "false_positive" for x in errors),
              "n_false_negatives": sum(x["error_type"] == "false_negative" for x in errors), "protocol": protocol}
    write_json(out / "metrics.json", metric)
    if auc_mean < 0.5:
        save_below_chance_diagnostics(Path(output_root).parent / "diagnostics" / "Contrastive_BayPrAnoMeta" /
                                      f"seed_{run_seed}" / category, query_records, aucs, reverse_aucs, protocol,
                                      {"post_adaptation": True, "support_size": K_SHOT, "predictive_dof": 8},
                                      "Contrastive BayPrAnoMeta", run_seed, split_seed, category, data["held_out_subtype"])
    print(f"  AUC={auc_mean:.4f} +/- {auc_se:.4f} | PR={pr_mean:.4f} +/- {pr_se:.4f} | F1={f1_mean:.4f} +/- {f1_se:.4f}")
    return metric


def evaluate_seed(run_seed, args):
    set_run_seed(run_seed)
    root = Path(args.run_root) / PROTOCOL / f"seed_{run_seed}"
    checkpoint = root / "checkpoints" / "final_model.pth"
    root = Path(args.output_root) / PROTOCOL / f"seed_{run_seed}"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}. Train this seed before evaluating it.")
    model = EmbeddingNet().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    started = time.perf_counter(); monitor = Monitor(); monitor.start(); results = {}
    for category in CLIENTS:
        data = load_client_dataset_leave1out(Path(args.base_root) / category, seed=args.split_seed,
                                             protocol=PROTOCOL, validation_fraction=args.validation_fraction)
        if data is not None:
            results[category] = evaluate_client(model, category, data, run_seed, args.split_seed, root / "evaluation",
                                                args.debug_protocol)
    monitor.stop()
    write_json(root / "evaluation" / "results.json", results)
    np.save(root / "evaluation" / "results.npy", results)
    save_all_clients_pr(root / "evaluation", results.keys())
    write_json(root / "evaluation" / "runtime_memory.json", {"run_seed": run_seed, "split_seed": args.split_seed,
                                                                "evaluation_sec": time.perf_counter() - started,
                                                                "peak_rss_mb": monitor.peak_mb, "test_episodes_per_category": TEST_EPISODES})
    return {category: value["held_out_subtype"] for category, value in results.items()}


CSV_COLUMNS = ["category", "held_out_subtype", "run_seed", "split_seed", "auc_mean", "auc_se",
               "pr_mean", "pr_se", "f1_mean", "f1_se", "pooled_auc", "pooled_reversed_score_auc",
               "threshold_95_normal", "n_false_positives", "n_false_negatives",
               "diagnosis", "diagnostic_original_auc", "diagnostic_reversed_score_auc",
               "normal_logp0_mean", "normal_logp0_median", "normal_logp0_std",
               "anomaly_logp0_mean", "anomaly_logp0_median", "anomaly_logp0_std",
               "normal_logp1_mean", "normal_logp1_median", "normal_logp1_std",
               "anomaly_logp1_mean", "anomaly_logp1_median", "anomaly_logp1_std",
               "normal_score_mean", "normal_score_median", "normal_score_std",
               "anomaly_score_mean", "anomaly_score_median", "anomaly_score_std"]


def write_csv(path, rows, columns):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def add_saved_diagnosis(row, protocol_root):
    """Attach saved below-chance diagnostics; never recompute scores or AUROC."""
    diagnostic_path = (Path(protocol_root) / f"seed_{row['run_seed']}" / "diagnostics" /
                       "Contrastive_BayPrAnoMeta" / f"seed_{row['run_seed']}" /
                       row["category"] / "summary.json")
    if not diagnostic_path.exists():
        row["diagnosis"] = "NOT_TRIGGERED_AUROC_GE_0_5"
        return row
    with diagnostic_path.open() as handle:
        diagnostic = json.load(handle)
    row["diagnosis"] = diagnostic["diagnosis"]
    row["diagnostic_original_auc"] = diagnostic["original_AUROC"]
    row["diagnostic_reversed_score_auc"] = diagnostic["reversed_score_AUROC"]
    for population in ("normal", "anomaly"):
        for quantity, key in (("logp0", f"{population}_logp0"),
                              ("logp1", f"{population}_logp1"),
                              ("score", f"{population}_score")):
            for statistic in ("mean", "median", "std"):
                row[f"{population}_{quantity}_{statistic}"] = diagnostic[key][statistic]
    return row


def export_saved_results(args):
    """Write CSV artifacts from completed JSON results without loading models or data."""
    protocol_root = Path(args.output_root) / PROTOCOL
    by_seed, all_rows = {}, []
    for run_seed in args.seeds:
        results_path = protocol_root / f"seed_{run_seed}" / "evaluation" / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Cannot export seed {run_seed}: missing {results_path}")
        with results_path.open() as handle:
            results = json.load(handle)
        rows = [add_saved_diagnosis(dict(results[category]), protocol_root) for category in sorted(results)]
        write_csv(results_path.parent / "results.csv", rows, CSV_COLUMNS)
        by_seed[run_seed] = {row["category"]: row for row in rows}; all_rows.extend(rows)
    write_csv(protocol_root / "three_seed_results.csv", all_rows, CSV_COLUMNS)

    shared_categories = set.intersection(*(set(results) for results in by_seed.values()))
    summary_rows = []
    for category in sorted(shared_categories):
        held_out = {by_seed[seed][category]["held_out_subtype"] for seed in args.seeds}
        split_seeds = {by_seed[seed][category]["split_seed"] for seed in args.seeds}
        if len(held_out) != 1 or len(split_seeds) != 1:
            raise RuntimeError(f"Refusing AUROC summary for {category}: split/held-out subtype differs across seeds.")
        aucs = np.asarray([by_seed[seed][category]["auc_mean"] for seed in args.seeds], dtype=float)
        row = {"category": category, "held_out_subtype": held_out.pop(), "split_seed": split_seeds.pop(),
               "n_seeds": len(args.seeds), "auroc_mean": float(aucs.mean()),
               "auroc_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0}
        row.update({f"seed_{seed}_auroc": float(by_seed[seed][category]["auc_mean"]) for seed in args.seeds})
        row.update({f"seed_{seed}_diagnosis": by_seed[seed][category]["diagnosis"] for seed in args.seeds})
        row["n_below_chance_diagnoses"] = sum(
            by_seed[seed][category]["diagnosis"] != "NOT_TRIGGERED_AUROC_GE_0_5" for seed in args.seeds)
        summary_rows.append(row)
    summary_columns = ["category", "held_out_subtype", "split_seed", "n_seeds"] + \
                      [f"seed_{seed}_auroc" for seed in args.seeds] + \
                      [f"seed_{seed}_diagnosis" for seed in args.seeds] + \
                      ["n_below_chance_diagnoses", "auroc_mean", "auroc_std"]
    write_csv(protocol_root / "three_seed_auroc_summary.csv", summary_rows, summary_columns)
    print(f"Exported saved results only: {protocol_root / 'three_seed_auroc_summary.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None, default=BASE_ROOT)
    parser.add_argument("--checkpoint-root", "--run-root", dest="run_root", default=RUN_ROOT); parser.add_argument("--output-root", required=True, help="Directory for evaluation artifacts.")
    parser.add_argument("--validation-fraction", type=float, default=.30)
    parser.add_argument("--debug-protocol", action="store_true")
    parser.add_argument("--export-existing-results", action="store_true",
                        help="Create CSVs from completed results.json files; does not load models or evaluate episodes.")
    args = parser.parse_args(); reference = None
    if args.export_existing_results:
        export_saved_results(args)
        return
    for run_seed in args.seeds:
        held_out = evaluate_seed(run_seed, args)
        if reference is None: reference = held_out
        elif held_out != reference: raise RuntimeError("Refusing aggregation: held-out subtypes differ across run seeds.")
    write_json(Path(args.output_root) / PROTOCOL / "evaluation_split_manifest.json",
               {"split_seed": args.split_seed, "run_seeds": args.seeds, "held_out_subtypes": reference})
    export_saved_results(args)


if __name__ == "__main__":
    main()
