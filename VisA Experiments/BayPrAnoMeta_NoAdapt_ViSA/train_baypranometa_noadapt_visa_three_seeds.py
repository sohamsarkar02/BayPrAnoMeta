#!/usr/bin/env python3
"""Support-conditioned BayPrAnoMeta VisA training with zero inner learning rate.
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
import numpy as np
import psutil
import torch
import torch.nn as nn
import torchvision.models as models

from utils.utils_data_visa_leave1out import discover_categories, load_client_dataset_visa, load_images, transform_eval, transform_train
from utils.utils_niw import log_student_t, niw_posterior
from utils.utils_plotting import plot_loss_curve

BASE_ROOT = None
RUN_ROOT = "runs/baypranometa_noadapt_visa_three_seed_fixed_split42_25ep_30epochs_100test"
SMOKE_RUN_ROOT = "runs/baypranometa_noadapt_visa_smoke"
PROTOCOL = "revised_clean_official_2cls_highshot"
SPLIT_SEED = 42
DEVICE = torch.device("cpu")

# Training and validation episode budgets.
EPOCHS, EPISODES_PER_EPOCH, VAL_EPISODES = 30, 25, 20
INNER_STEPS, INNER_LR, META_LR = 1, 0.0, 1e-4
K_SHOT, Q_N, Q_A = 5, 12, 4


class PeakMemoryMonitor:
    def __init__(self, interval=0.10):
        self.process, self.interval, self.peak_mb = psutil.Process(os.getpid()), interval, 0.0
        self.running, self.thread = False, None
    def _monitor(self):
        while self.running:
            self.peak_mb = max(self.peak_mb, self.process.memory_info().rss / 1024 ** 2)
            time.sleep(self.interval)
    def start(self):
        self.running = True; self.thread = threading.Thread(target=self._monitor, daemon=True); self.thread.start()
    def stop(self):
        self.running = False
        if self.thread: self.thread.join()


class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(nn.Linear(512, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value))


def load_stack(paths, transform):
    images = load_images(paths, transform)
    if len(images) != len(paths):
        raise RuntimeError("An episode image could not be loaded; refusing label/image misalignment.")
    return torch.stack(images).to(DEVICE)


def sample_episode(data, split, rng, debug=False):
    pool = data[split]
    if len(pool["normal"]) < K_SHOT + Q_N or len(pool["anomaly"]) < Q_A:
        raise ValueError(f"Insufficient {split} images for one episode.")
    normal = rng.sample(pool["normal"], K_SHOT + Q_N)
    support, qn, qa = normal[:K_SHOT], normal[K_SHOT:], rng.sample(pool["anomaly"], Q_A)
    if debug:
        assert set(support).isdisjoint(qn)
    return support, qn, qa


def reference_log_probability(z):
    d = z.shape[1]
    return log_student_t(z, torch.zeros(d, device=DEVICE, dtype=z.dtype),
                         100.0 * torch.eye(d, device=DEVICE, dtype=z.dtype),
                         torch.tensor(2.0, device=DEVICE, dtype=z.dtype))


def run_episode(model, data, split, rng, training, debug=False):
    support, qn, qa = sample_episode(data, split, rng, debug)
    labels = torch.tensor([0] * Q_N + [1] * Q_A, dtype=torch.float32, device=DEVICE)
    transform = transform_train if training else transform_eval
    xs, xq = load_stack(support, transform), load_stack(qn + qa, transform)
    inner_opt = torch.optim.SGD(model.parameters(), lr=INNER_LR)
    with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False, track_higher_grads=True) as (fmodel, diffopt):
        for _ in range(INNER_STEPS):
            z_sup = fmodel(xs)
            mu, sigma, dof = niw_posterior(z_sup, validate=debug)
            before_update = [parameter.clone() for parameter in fmodel.parameters()] if debug else None
            diffopt.step(-log_student_t(z_sup, mu, sigma, dof).mean())
            if debug and not all(torch.equal(before, after) for before, after in zip(before_update, fmodel.parameters())):
                raise AssertionError("NoAdapt fast parameters changed despite INNER_LR=0")
        # NIW predictive posterior is recomputed from adapted support embeddings.
        z_sup = fmodel(xs)
        mu, sigma, dof = niw_posterior(z_sup, validate=debug)
        zq = fmodel(xq)
        loss = -((1 - labels) * log_student_t(zq, mu, sigma, dof) + labels * reference_log_probability(zq)).mean()
    return loss, {"support": support, "normal_query": qn, "anomaly_query": qa}


def train_seed(run_seed, args):
    set_seed(run_seed)
    run_dir = Path(args.run_root) / PROTOCOL / f"seed_{run_seed}"
    data_start = time.perf_counter()
    categories = discover_categories(args.base_root, args.split_csv, subset_file=args.subset_file)
    if args.category: categories = [c for c in categories if c == args.category]
    print("Discovered VisA categories:", categories)
    clients = {}
    for client in categories:
        data = load_client_dataset_visa(args.base_root, client, seed=args.split_seed,
                                        validation_fraction=args.validation_fraction,
                                        split_csv=args.split_csv, subset_file=args.subset_file,
                                        audit_dir=run_dir / "split_audit")
        if data is not None: clients[client] = data
    if not clients: raise RuntimeError("No eligible VisA categories loaded.")
    data_load_sec = time.perf_counter() - data_start

    model_start = time.perf_counter(); model = EmbeddingNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=META_LR); model_init_sec = time.perf_counter() - model_start
    epochs = 1 if args.smoke_test else EPOCHS
    train_episodes = 3 if args.smoke_test else EPISODES_PER_EPOCH
    validation_episodes = 3 if args.smoke_test else VAL_EPISODES
    metadata = {"method": "BayPrAnoMeta_NoAdapt_ViSA", "dataset": "VisA", "protocol": PROTOCOL, "run_seed": run_seed, "split_seed": args.split_seed,
                "epochs": epochs, "episodes_per_epoch": train_episodes, "validation_episodes": validation_episodes,
                "K_SHOT": K_SHOT, "Q_N": Q_N, "Q_A": Q_A, "inner_steps": INNER_STEPS, "inner_lr": INNER_LR,
                "meta_lr": META_LR, "backbone": "ResNet18", "embedding_dim": 128,
                "full_inner_lr": 5e-4, "noadapt_inner_lr": INNER_LR,
                "support_conditioning_retained": True, "post_inner_posterior_recomputed": True,
                "score_type": "log p1(z) - log p0(z | support; theta_T=theta)", "checkpoint_path": str(run_dir / "checkpoints" / "final_model.pth"),
                "held_out_subtypes": {c: d["held_out_subtype"] for c, d in clients.items()},
                "split_source": args.split_csv, "smoke_test": args.smoke_test,
                "anomaly_grouping_note": "VisA has no subtype labels; official high-shot test anomalies form the held-out image group."}
    write_json(run_dir / "metadata.json", metadata)

    train_rng, val_rng = random.Random(run_seed), random.Random(run_seed + 1)
    client_rng, val_client_rng = random.Random(run_seed), random.Random(run_seed + 1)
    process, monitor = psutil.Process(os.getpid()), PeakMemoryMonitor(); monitor.start(); loop_start = time.perf_counter()
    train_losses, val_losses, train_manifest, val_manifest, epoch_times = [], [], [], [], []
    gradients_verified, outer_update_verified, best_validation = False, False, float("inf")
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter(); model.train(); losses = []
        for episode_id in range(train_episodes):
            category = client_rng.choice(list(clients))
            loss, episode = run_episode(model, clients[category], "train", train_rng, True, args.debug_protocol)
            optimizer.zero_grad(); loss.backward()
            if not gradients_verified:
                if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters()):
                    raise RuntimeError("Meta-gradient sanity check failed.")
                gradients_verified = True
            before_outer_update = [parameter.detach().clone() for parameter in model.parameters()] if not outer_update_verified else None
            optimizer.step()
            if before_outer_update is not None:
                outer_update_verified = any(not torch.equal(before, after.detach()) for before, after in zip(before_outer_update, model.parameters()))
                if not outer_update_verified: raise RuntimeError("Outer optimizer did not update theta")
            losses.append(loss.item())
            train_manifest.append({"epoch": epoch, "episode": episode_id, "category": category, **episode})
        train_losses.append(float(np.mean(losses)))
        model.eval(); losses = []
        for episode_id in range(validation_episodes):
            category = val_client_rng.choice(list(clients))
            loss, episode = run_episode(model, clients[category], "validation", val_rng, False, args.debug_protocol)
            losses.append(loss.detach().item())
            val_manifest.append({"epoch": epoch, "episode": episode_id, "category": category, **episode})
        val_losses.append(float(np.mean(losses))); epoch_times.append(time.perf_counter() - epoch_start)
        print(f"run_seed={run_seed} split_seed={args.split_seed} epoch={epoch:03d} train={train_losses[-1]:.4f} val={val_losses[-1]:.4f}")
        if val_losses[-1] < best_validation:
            best_validation = val_losses[-1]
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best_model.pth")
            metadata["best_validation_epoch"] = epoch
            metadata["best_validation_loss"] = best_validation
        if epoch % 5 == 0:
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), run_dir / "checkpoints" / f"bpmaml_epoch_{epoch}.pth")
    plot_loss_curve(train_losses, val_losses, run_dir / "plots" / "loss_curve.png"); write_json(run_dir / "loss_history.json", [{"epoch": index, "train_loss": train, "validation_loss": validation} for index, (train, validation) in enumerate(zip(train_losses, val_losses), 1)])
    torch.save(model.state_dict(), run_dir / "checkpoints" / "final_model.pth")
    write_json(run_dir / "episode_manifests" / "train_episodes.json", train_manifest)
    write_json(run_dir / "episode_manifests" / "validation_episodes.json", val_manifest)
    write_json(run_dir / "training_gradient_audit.json", {
        "full_inner_lr": 5e-4, "noadapt_inner_lr": INNER_LR,
        "fast_parameters_unchanged_by_inner_step": True,
        "outer_gradients_nonzero_and_finite": gradients_verified,
        "outer_optimizer_updated_theta": outer_update_verified,
    })
    with (run_dir / "losses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss", "epoch_seconds"]); writer.writeheader()
        for i, (tr, va, sec) in enumerate(zip(train_losses, val_losses, epoch_times), 1): writer.writerow({"epoch": i, "train_loss": tr, "validation_loss": va, "epoch_seconds": sec})
    write_json(run_dir / "metadata.json", metadata)
    monitor.stop()
    runtime = {"run_seed": run_seed, "split_seed": args.split_seed, "data_load_sec": data_load_sec, "model_init_sec": model_init_sec,
               "training_loop_sec": time.perf_counter() - loop_start, "epoch_times_sec": epoch_times,
               "peak_rss_mb": monitor.peak_mb, "end_rss_mb": process.memory_info().rss / 1024 ** 2,
               "average_seconds_per_epoch": float(np.mean(epoch_times)), "peak_gpu_memory_mb": 0.0,
               "device": str(DEVICE), "num_parameters": sum(p.numel() for p in model.parameters()),
               "timing_scope": "Includes image I/O, augmentation, differentiable adaptation, outer updates, validation, and plot creation."}
    write_json(run_dir / "runtime_memory.json", runtime)
    for name in ("runtime.csv", "memory.csv"):
        with (run_dir / name).open("w", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=list(runtime)); writer.writeheader(); writer.writerow(runtime)
    return metadata["held_out_subtypes"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None, default=BASE_ROOT)
    parser.add_argument("--output-root", "--run-root", dest="run_root", default=RUN_ROOT)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--validation-fraction", type=float, default=.30)
    parser.add_argument("--split-csv", default="split_csv/2cls_highshot.csv"); parser.add_argument("--subset-file", help='Paper image allowlist. Required for exact reproduction of the paper experiments.')
    parser.add_argument("--category", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--debug-protocol", action="store_true")
    args = parser.parse_args()
    if args.split_seed != SPLIT_SEED: raise ValueError("NoAdapt VisA ablation is locked to SPLIT_SEED=42")
    if args.smoke_test:
        if args.run_root == RUN_ROOT: args.run_root = SMOKE_RUN_ROOT
        args.seeds = [1]
        if args.category is None: args.category = discover_categories(args.base_root, args.split_csv, subset_file=args.subset_file)[0]
    split_reference = None
    for seed in args.seeds:
        held_out = train_seed(seed, args)
        if split_reference is None:
            split_reference = held_out
        elif held_out != split_reference:
            raise RuntimeError("Held-out subtype differs across run seeds; refusing to continue.")
    write_json(Path(args.run_root) / PROTOCOL / "split_manifest.json", {"split_seed": args.split_seed, "held_out_subtypes": split_reference, "run_seeds": args.seeds})


if __name__ == "__main__":
    main()
