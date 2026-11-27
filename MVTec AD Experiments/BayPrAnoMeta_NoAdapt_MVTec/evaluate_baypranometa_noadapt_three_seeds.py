#!/usr/bin/env python3
"""Support-conditioned BayPrAnoMeta evaluation with zero inner learning rate."""

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
from sklearn.metrics import confusion_matrix, precision_recall_curve

from utils.utils_below_chance_diagnostics import save_below_chance_diagnostics
from utils.utils_data_fedcontrast_leave1out import load_client_dataset_leave1out, load_images, transform_eval
from utils.utils_eval_stats import aggregate_with_se, compute_episode_metrics
from utils.utils_niw import KAPPA0, NU0_OFFSET, log_student_t, niw_posterior
from utils.utils_plotting import plot_pr_curve, plot_roc_curve, plot_score_histogram

BASE_ROOT = None
CLIENTS = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
RUN_ROOT = "runs/baypranometa_noadapt_three_seed_fixed_split42_25ep_30epochs_100test"
SMOKE_RUN_ROOT = "runs/baypranometa_noadapt_smoke"
FULL_MANIFEST_ROOT = Path("../Bayesian_ProtoMAML/runs/baypranometa_avgpool_three_seed/revised_clean")
PROTOCOL = "revised_clean"
SPLIT_SEED = 42
DEVICE = torch.device("cpu")
TEST_EPISODES, INNER_STEPS, INNER_LR = 100, 1, 0.0
K_SHOT, Q_N, Q_A = 5, 12, 4


class PeakMemoryMonitor:
    def __init__(self, interval=.10):
        self.process, self.interval, self.peak_mb = psutil.Process(os.getpid()), interval, 0.0
        self.running, self.thread = False, None
    def _monitor(self):
        while self.running:
            self.peak_mb = max(self.peak_mb, self.process.memory_info().rss / 1024 ** 2); time.sleep(self.interval)
    def start(self):
        self.running = True; self.thread = threading.Thread(target=self._monitor, daemon=True); self.thread.start()
    def stop(self):
        self.running = False
        if self.thread: self.thread.join()


class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__(); base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT); base.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.features = nn.Sequential(*list(base.children())[:-1]); self.head = nn.Sequential(nn.Linear(512, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
    def forward(self, x): return self.head(self.features(x).flatten(1))


def set_seed(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle: json.dump(payload, handle, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value))
def load_stack(paths):
    images = load_images(paths, transform_eval)
    if len(images) != len(paths): raise RuntimeError("Could not load an evaluation image; refusing label/image misalignment.")
    return torch.stack(images).to(DEVICE)
def reference_log_probability(z):
    d = z.shape[1]
    return log_student_t(z, torch.zeros(d, device=DEVICE, dtype=z.dtype), 100.0 * torch.eye(d, device=DEVICE, dtype=z.dtype), torch.tensor(2.0, device=DEVICE, dtype=z.dtype))


def sample_test_episode(data, rng, debug=False):
    normal, anomaly = data["test"]["normal"], data["test"]["anomaly"]
    if len(normal) < K_SHOT + Q_N or len(anomaly) < Q_A: raise ValueError("Insufficient test images for one episode.")
    selected = rng.sample(normal, K_SHOT + Q_N); support, qn, qa = selected[:K_SHOT], selected[K_SHOT:], rng.sample(anomaly, Q_A)
    if debug: assert set(support).isdisjoint(qn)
    return support, qn, qa


def evaluate_category(model, category, data, run_seed, run_dir, args, pr_curves):
    rng = random.Random(run_seed + 2); aucs, reverse_aucs, prs, f1s = [], [], [], []
    scores_all, labels_all, paths_all, embeddings, manifest, records, sigmas, violations = [], [], [], [], [], [], [], []
    n_test_episodes = 1 if args.smoke_test else TEST_EPISODES
    full_manifest_path = args.full_manifest_root / f"seed_{run_seed}" / "eval" / category / "test_episodes.json"
    full_manifest = json.loads(full_manifest_path.read_text()) if full_manifest_path.is_file() else None
    for episode_id in range(n_test_episodes):
        support, qn, qa = sample_test_episode(data, rng, args.debug_protocol); paths, labels = qn + qa, [0] * Q_N + [1] * Q_A
        if full_manifest is None:
            raise FileNotFoundError(f"Required Full BayPrAnoMeta manifest missing: {full_manifest_path}")
        expected = full_manifest[episode_id]
        if support != expected["support"] or qn != expected["normal_query"] or qa != expected["anomaly_query"]:
            raise AssertionError(f"Episode identities differ from Full BayPrAnoMeta: {category}, episode {episode_id}")
        overlap = {"support_normal_query": sorted(set(support) & set(qn)), "support_anomaly_query": sorted(set(support) & set(qa)), "normal_anomaly_query": sorted(set(qn) & set(qa))}
        if any(overlap.values()): violations.append({"episode_id": episode_id, **overlap})
        xs, xq = load_stack(support), load_stack(paths); inner_opt = torch.optim.SGD(model.parameters(), lr=INNER_LR)
        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True, track_higher_grads=False) as (fmodel, diffopt):
            for _ in range(INNER_STEPS):
                z0 = fmodel(xs); mu0, sigma0, dof0 = niw_posterior(z0, validate=args.debug_protocol)
                before_update = [parameter.clone() for parameter in fmodel.parameters()] if args.debug_protocol else None
                diffopt.step(-log_student_t(z0, mu0, sigma0, dof0).mean())
                if args.debug_protocol and not all(torch.equal(before, after) for before, after in zip(before_update, fmodel.parameters())):
                    raise AssertionError("NoAdapt fast parameters changed despite INNER_LR=0")
            with torch.no_grad():
                zsup = fmodel(xs); mu, sigma, dof = niw_posterior(zsup, validate=args.debug_protocol); zq = fmodel(xq)
                logp_a, logp_n = reference_log_probability(zq), log_student_t(zq, mu, sigma, dof)
                scores, z = (logp_a - logp_n).cpu().numpy(), zq.cpu().numpy()
                p1, p0 = logp_a.cpu().numpy(), logp_n.cpu().numpy(); sigmas.append(sigma.cpu().numpy())
        auc, pr, f1 = compute_episode_metrics(scores, labels); reverse_auc, _, _ = compute_episode_metrics(-scores, labels)
        aucs.append(auc); reverse_aucs.append(reverse_auc); prs.append(pr); f1s.append(f1)
        scores_all.extend(scores.tolist()); labels_all.extend(labels); paths_all.extend(paths); embeddings.append(z)
        manifest.append({"episode": episode_id, "support": support, "normal_query": qn, "anomaly_query": qa})
        support_json = json.dumps(support)
        records.extend({"run_seed": run_seed, "split_seed": args.split_seed, "method": "BayPrAnoMeta_NoAdapt_MVTec", "category": category, "held_out_subtype": data["held_out_subtype"], "episode_id": episode_id,
                        "image_path": path, "label": int(label), "score": float(score), "logp1": float(a), "logp0": float(n),
                        "logp_anomaly": float(a), "logp_normal": float(n), "final_score": float(score), "support_image_paths": support_json}
                       for path, label, score, a, n in zip(paths, labels, scores, p1, p0))
    auc_m, auc_se = aggregate_with_se(aucs); pr_m, pr_se = aggregate_with_se(prs); f1_m, f1_se = aggregate_with_se(f1s)
    out = run_dir / "eval" / category; out.mkdir(parents=True, exist_ok=True); write_json(out / "test_episodes.json", manifest)
    with (out / "query_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    plot_roc_curve(labels_all, scores_all, out / "roc.png"); plot_pr_curve(labels_all, scores_all, out / "pr.png"); plot_score_histogram(labels_all, scores_all, out / "score_hist.png")
    precision, recall, _ = precision_recall_curve(labels_all, scores_all); pr_curves[category] = (recall, precision)
    labels_np, scores_np = np.asarray(labels_all), np.asarray(scores_all); threshold = float(np.percentile(scores_np[labels_np == 0], 95)); preds = (scores_np >= threshold).astype(int)
    plt.figure(figsize=(5, 4)); sns.heatmap(confusion_matrix(labels_np, preds), annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"{category} Confusion Matrix (95% thr)"); plt.tight_layout(); plt.savefig(out / "confusion_matrix.png"); plt.close()
    def errors(indices): return [{"path": paths_all[i], "filename": Path(paths_all[i]).name, "ground_truth": int(labels_np[i]), "prediction": int(preds[i]), "score": float(scores_np[i]), "threshold": threshold} for i in indices]
    write_json(out / "false_positives.json", errors(np.where((labels_np == 0) & (preds == 1))[0])); write_json(out / "false_negatives.json", errors(np.where((labels_np == 1) & (preds == 0))[0]))
    Z = np.vstack(embeddings); Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-8)
    Z2 = TSNE(n_components=2, random_state=run_seed, perplexity=min(30, len(Z) - 1), init="random").fit_transform(Z)
    plt.figure(figsize=(6, 5)); plt.scatter(Z2[labels_np == 0, 0], Z2[labels_np == 0, 1], s=5, label="Normal"); plt.scatter(Z2[labels_np == 1, 0], Z2[labels_np == 1, 1], s=5, label="Anomaly")
    plt.legend(); plt.tight_layout(); plt.savefig(out / "tsne.png"); plt.close()
    report = {"train_normal_count": len(data["train"]["normal"]), "test_normal_count": len(data["test"]["normal"]), "train_anomaly_count": len(data["train"]["anomaly"]), "test_anomaly_count": len(data["test"]["anomaly"]),
              "train_test_normal_disjoint": set(data["train"]["normal"]).isdisjoint(data["test"]["normal"]), "held_out_absent_from_training": data["held_out_subtype"] not in data["seen_subtypes"], "episode_overlap_violations": violations,
              "labels_match_expected": all([record["label"] for record in records[i:i + Q_N + Q_A]] == [0] * Q_N + [1] * Q_A for i in range(0, len(records), Q_N + Q_A))}
    triggered = bool(auc_m < args.below_chance_threshold); diagnosis = None
    if triggered:
        d = sigmas[0].shape[0]; diagnosis = save_below_chance_diagnostics(out_dir=run_dir / "diagnostics" / "BayPrAnoMeta" / f"seed_{run_seed}" / category, method="BayPrAnoMeta", seed=run_seed, split_seed=args.split_seed, category=category, held_out_subtype=data["held_out_subtype"], records=records, episode_aucs=aucs, reversed_episode_aucs=reverse_aucs, protocol_report=report, posterior_matrices=sigmas, kappa_n=KAPPA0 + K_SHOT, nu_n=d + NU0_OFFSET + K_SHOT, dof=NU0_OFFSET + K_SHOT + 1)
    result = {"auc_mean": float(auc_m), "auc_se": float(auc_se), "pr_mean": float(pr_m), "pr_se": float(pr_se), "f1_mean": float(f1_m), "f1_se": float(f1_se), "held_out_subtype": data["held_out_subtype"],
              "checkpoint_used": str(run_dir / "checkpoints" / "final_model.pth"), "full_manifest_path": str(full_manifest_path),
              "same_episode_identities_as_full": True, "inner_lr": INNER_LR, "support_conditioning_retained": True,
              "all_scores_finite": bool(np.isfinite(scores_all).all()), "threshold_95pct_normal": threshold, "below_chance_diagnostic_triggered": triggered, "diagnosis": diagnosis["diagnosis"] if diagnosis else None}
    write_json(out / "metrics.json", result); print(f"run_seed={run_seed} split_seed={args.split_seed} {category}: AUC={auc_m:.4f} ± {auc_se:.4f}"); return result


def evaluate_seed(run_seed, args):
    set_seed(run_seed); run_dir = Path(args.run_root) / PROTOCOL / f"seed_{run_seed}"; checkpoint = run_dir / "checkpoints" / "final_model.pth"; run_dir = Path(args.output_root) / PROTOCOL / f"seed_{run_seed}"
    if not checkpoint.is_file(): raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    model_start = time.perf_counter(); model = EmbeddingNet().to(DEVICE); model.load_state_dict(torch.load(checkpoint, map_location=DEVICE)); model.eval(); model_load_sec = time.perf_counter() - model_start
    process, monitor = psutil.Process(os.getpid()), PeakMemoryMonitor(); monitor.start(); started = time.perf_counter(); results, curves = {}, {}
    for category in (args.categories or CLIENTS):
        data = load_client_dataset_leave1out(Path(args.base_root) / category, seed=args.split_seed, protocol=PROTOCOL, validation_fraction=args.validation_fraction)
        if data is not None: results[category] = evaluate_category(model, category, data, run_seed, run_dir, args, curves)
    eval_dir = run_dir / "eval"; plt.figure(figsize=(7, 6))
    for category, (recall, precision) in curves.items(): plt.plot(recall, precision, label=category)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(eval_dir / "all_clients_pr_curves.png"); plt.close()
    np.save(eval_dir / "results.npy", results); write_json(eval_dir / "results.json", results)
    if args.smoke_test:
        training_audit_path = run_dir / "training_gradient_audit.json"
        training_audit = json.loads(training_audit_path.read_text()) if training_audit_path.is_file() else {}
        write_json(eval_dir / "smoke_test_audit.json", {
            "alpha_full": 5e-4, "alpha_noadapt": INNER_LR,
            "support_niw_posterior_still_computed": True,
            "support_dependent_posterior_still_used": True,
            "student_t_p0_unchanged": True, "fixed_student_t_p1_unchanged": True,
            "outer_loss_unchanged": True, "architecture_unchanged": True,
            "data_manifests_matched_to_full": all(r["same_episode_identities_as_full"] for r in results.values()),
            "only_inner_encoder_parameter_adaptation_removed": True,
            "outer_loss_updated_theta_during_smoke_training": bool(training_audit.get("outer_optimizer_updated_theta", False)),
            "all_scores_finite": all(r["all_scores_finite"] for r in results.values()),
            "smoke_test_passed": bool(training_audit.get("outer_optimizer_updated_theta", False)) and all(r["same_episode_identities_as_full"] and r["all_scores_finite"] for r in results.values()),
        })
    with (eval_dir / "results.csv").open("w", newline="") as handle:
        fields = ["method", "seed", "category", "held_out_subtype", "auc_mean", "auc_se", "pr_mean", "pr_se", "f1_mean", "f1_se", "same_episode_identities_as_full", "inner_lr", "support_conditioning_retained", "checkpoint_used", "below_chance_diagnostic_triggered", "diagnosis"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for category, result in results.items():
            writer.writerow({"method": "BayPrAnoMeta_NoAdapt_MVTec", "seed": run_seed, "category": category,
                             **{field: result.get(field) for field in fields if field not in {"method", "seed", "category"}}})
    write_json(eval_dir / "metadata.json", {"method": "BayPrAnoMeta_NoAdapt_MVTec", "run_seed": run_seed, "split_seed": args.split_seed, "protocol": PROTOCOL, "test_episodes": 1 if args.smoke_test else TEST_EPISODES,
               "inner_lr": INNER_LR, "support_conditioning_retained": True, "checkpoint_path": str(checkpoint)})
    monitor.stop(); write_json(eval_dir / "runtime_memory.json", {"run_seed": run_seed, "split_seed": args.split_seed, "model_load_sec": model_load_sec, "evaluation_loop_sec": time.perf_counter() - started, "peak_rss_mb": monitor.peak_mb, "end_rss_mb": process.memory_info().rss / 1024 ** 2,
                        "timing_scope": "Includes data loading, image preprocessing, adaptation, scoring, diagnostics, and plots."})
    return results, {category: result["held_out_subtype"] for category, result in results.items()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3]); parser.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None, default=BASE_ROOT); parser.add_argument("--checkpoint-root", "--run-root", dest="run_root", default=RUN_ROOT); parser.add_argument("--output-root", required=True, help="Directory for evaluation artifacts."); parser.add_argument("--split-seed", type=int, default=SPLIT_SEED); parser.add_argument("--validation-fraction", type=float, default=.30); parser.add_argument("--below-chance-threshold", type=float, default=.50); parser.add_argument("--debug-protocol", action="store_true"); parser.add_argument("--categories", nargs="+", choices=CLIENTS); parser.add_argument("--smoke-test", action="store_true"); parser.add_argument("--full-manifest-root", type=Path, default=FULL_MANIFEST_ROOT); args = parser.parse_args()
    if args.split_seed != SPLIT_SEED: raise ValueError("This ablation is locked to SPLIT_SEED=42")
    if args.smoke_test and args.run_root == RUN_ROOT: args.run_root = SMOKE_RUN_ROOT
    all_results, split_reference = {}, None
    for seed in args.seeds:
        results, held_out = evaluate_seed(seed, args)
        if split_reference is None:
            split_reference = held_out
        elif held_out != split_reference:
            raise RuntimeError("Held-out subtype differs across run seeds; refusing three-seed aggregation.")
        all_results[seed] = results
    summary_root = Path(args.output_root) / PROTOCOL
    write_json(summary_root / "split_manifest.json", {"split_seed": args.split_seed, "held_out_subtypes": split_reference, "run_seeds": args.seeds})
    rows = []
    for category in CLIENTS:
        values = [all_results[seed][category]["auc_mean"] for seed in args.seeds if category in all_results[seed]]
        if not values: continue
        row = {"method": "BayPrAnoMeta_NoAdapt_MVTec", "category": category, **{f"seed_{seed}_AUROC": all_results[seed].get(category, {}).get("auc_mean") for seed in args.seeds}, "mean_AUROC": float(np.mean(values)), "std_AUROC": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "any_seed_below_0_50": bool(any(v < .5 for v in values)), "mean_below_0_50": bool(np.mean(values) < .5)}; rows.append(row)
    with (summary_root / "three_seed_auroc_summary.csv").open("w", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(summary_root / "three_seed_auroc_summary.json", rows)


if __name__ == "__main__": main()
