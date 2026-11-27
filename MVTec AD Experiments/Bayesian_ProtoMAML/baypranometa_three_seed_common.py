"""Standalone shared implementation for the revised reduced-budget 3-seed run."""

import csv
import json
import math
import os
import random
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

DEVICE = torch.device("cpu")
CLIENTS = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
K_SHOT, Q_N, Q_A = 5, 12, 4
INNER_STEPS, INNER_LR, META_LR = 1, 5e-4, 1e-4
KAPPA0, NU0_OFFSET = 0.01, 2

transform_train = T.Compose([T.Resize(256), T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ColorJitter(.1, .1, .1, .1), T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])
transform_eval = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def json_default(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    return str(value)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle: json.dump(value, handle, indent=2, default=json_default)


class PeakMemoryMonitor:
    def __init__(self, interval=.10):
        self.process, self.interval, self.peak_mb = psutil.Process(os.getpid()), interval, 0.0
        self.running, self.thread = False, None
    def _loop(self):
        while self.running:
            self.peak_mb = max(self.peak_mb, self.process.memory_info().rss / 1024 ** 2); time.sleep(self.interval)
    def start(self):
        self.running = True; self.thread = threading.Thread(target=self._loop, daemon=True); self.thread.start()
    def stop(self):
        self.running = False
        if self.thread: self.thread.join()


def list_images(path): return sorted(str(p) for p in Path(path).glob("*.png"))


def load_stack(paths, transform):
    tensors = []
    for path in paths:
        try:
            with Image.open(path) as image: tensors.append(transform(image.convert("RGB")))
        except Exception as exc:
            raise RuntimeError(f"Could not load {path}") from exc
    return torch.stack(tensors).to(DEVICE)


def split_pool(paths, rng, fraction):
    shuffled = list(paths); rng.shuffle(shuffled)
    n_val = min(max(1, round(len(shuffled) * fraction)), len(shuffled) - 1)
    return shuffled[n_val:], shuffled[:n_val]


def load_revised_client(client_root, seed, validation_fraction=.30):
    """Leakage-free MVTec split: train/validation from train/good only; test untouched."""
    root, rng = Path(client_root), random.Random(seed)
    train_good, test_good = list_images(root / "train" / "good"), list_images(root / "test" / "good")
    anom_dirs = sorted((p for p in (root / "test").iterdir() if p.is_dir() and p.name != "good"), key=lambda p: p.name)
    if len(train_good) < 2 or not test_good or len(anom_dirs) < 2: return None
    rng.shuffle(anom_dirs); held_out, seen = anom_dirs[0], anom_dirs[1:]
    train_anom, val_anom = [], []
    for directory in seen:
        paths = list_images(directory)
        if len(paths) >= 2:
            tr, va = split_pool(paths, rng, validation_fraction); train_anom.extend(tr); val_anom.extend(va)
        else: train_anom.extend(paths)
    train_normal, val_normal = split_pool(train_good, rng, validation_fraction)
    test_anom = list_images(held_out)
    if not train_anom or not val_anom or not test_anom: return None
    data = {"train": {"normal": train_normal, "anomaly": train_anom}, "validation": {"normal": val_normal, "anomaly": val_anom}, "test": {"normal": test_good, "anomaly": test_anom}, "held_out_subtype": held_out.name, "seen_subtypes": [p.name for p in seen]}
    assert set(train_normal).isdisjoint(test_good) and set(val_normal).isdisjoint(test_good)
    assert set(train_anom).isdisjoint(test_anom) and held_out.name not in data["seen_subtypes"]
    return data


class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__(); base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT); base.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.features = nn.Sequential(*list(base.children())[:-1]); self.head = nn.Sequential(nn.Linear(512, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
    def forward(self, x): return self.head(self.features(x).flatten(1))


def niw_posterior(z):
    K, d = z.shape; mu0 = torch.zeros(d, device=z.device, dtype=z.dtype); lambda0 = torch.eye(d, device=z.device, dtype=z.dtype)
    z_bar = z.mean(0); scatter = (z - z_bar).T @ (z - z_bar); kappa_n, nu_n = KAPPA0 + K, d + NU0_OFFSET + K
    diff = (z_bar - mu0).unsqueeze(1); lambda_n = lambda0 + scatter + (KAPPA0 * K / kappa_n) * (diff @ diff.T)
    dof = nu_n - d + 1; sigma_n = ((kappa_n + 1) / (kappa_n * dof)) * lambda_n
    return (KAPPA0 * mu0 + K * z_bar) / kappa_n, sigma_n, dof


def log_student_t(x, mu, sigma, dof):
    d = x.shape[1]; dof = torch.as_tensor(dof, device=x.device, dtype=x.dtype); xm = x - mu.unsqueeze(0)
    L = torch.linalg.cholesky(sigma + 1e-6 * torch.eye(d, device=x.device, dtype=x.dtype)); inv = torch.cholesky_inverse(L)
    delta = ((xm @ inv) * xm).sum(1); logdet = 2 * torch.log(torch.diag(L)).sum(); dt = torch.tensor(float(d), device=x.device, dtype=x.dtype)
    return torch.lgamma((dof + dt) / 2) - torch.lgamma(dof / 2) - .5 * dt * torch.log(dof * torch.tensor(math.pi, device=x.device)) - .5 * logdet - ((dof + dt) / 2) * torch.log1p(delta / dof)


def reference_log_probability(z):
    d = z.shape[1]
    return log_student_t(z, torch.zeros(d, device=z.device, dtype=z.dtype), 100 * torch.eye(d, device=z.device, dtype=z.dtype), torch.tensor(2., device=z.device, dtype=z.dtype))


def sample_episode(data, split, rng):
    pool = data[split]; normals, anomalies = pool["normal"], pool["anomaly"]
    if len(normals) < K_SHOT + Q_N or len(anomalies) < Q_A: raise ValueError(f"Insufficient {split} images for one episode.")
    chosen = rng.sample(normals, K_SHOT + Q_N); support, qn, qa = chosen[:K_SHOT], chosen[K_SHOT:], rng.sample(anomalies, Q_A)
    assert set(support).isdisjoint(qn)
    return support, qn, qa


def episode_metrics(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    auc, pr = roc_auc_score(labels, scores), average_precision_score(labels, scores)
    thresholds = np.linspace(scores.min(), scores.max(), 200); f1 = max(f1_score(labels, scores >= t, zero_division=0) for t in thresholds)
    return float(auc), float(pr), float(f1)


def descriptive(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std()), "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())}


def save_curve(losses_train, losses_val, path):
    plt.figure(figsize=(6, 4)); plt.plot(losses_train, label="Train Loss"); plt.plot(losses_val, label="Val Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()


def save_below_chance(out_dir, *, seed, category, held_out, records, episode_aucs, reversed_aucs, data, matrices):
    """BayPrAnoMeta-specific diagnostic; it never changes the primary scores."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([r["label"] for r in records]); scores = np.asarray([r["final_score"] for r in records]); p1 = np.asarray([r["logp_anomaly"] for r in records]); p0 = np.asarray([r["logp_normal"] for r in records])
    normal, anomaly = labels == 0, labels == 1; original, reversed_score = float(np.mean(episode_aucs)), float(np.mean(reversed_aucs))
    overlap = [r["overlap"] for r in records if any(r["overlap"].values())]
    diagnosis = "DATA_PROTOCOL_ERROR" if overlap else "NUMERICAL_INSTABILITY" if not np.isfinite(np.r_[scores, p1, p0]).all() else "SCORE_DIRECTION_ERROR" if reversed_score - original >= .20 else "HIGH_EPISODE_VARIANCE" if np.std(episode_aucs) >= .20 else "GENUINE_BELOW_CHANCE_PERFORMANCE"
    posterior = []
    scale = (KAPPA0 + K_SHOT) * 8 / (KAPPA0 + K_SHOT + 1)
    for i, sigma in enumerate(matrices):
        se, le = np.linalg.eigvalsh(sigma), np.linalg.eigvalsh(sigma * scale)
        posterior.append({"episode_id": i, "lambda_n_eigenvalues": le.tolist(), "sigma_n_eigenvalues": se.tolist(), "sigma_min_eigenvalue": float(se.min()), "sigma_max_eigenvalue": float(se.max()), "sigma_condition_number": float(se.max() / se.min())})
    summary = {"method": "BayPrAnoMeta", "seed": seed, "category": category, "held_out_subtype": held_out, "original_AUROC": original, "reversed_score_AUROC": reversed_score, "primary_score_was_not_modified": True, "severity": "AUROC_LT_0_30" if original < .3 else "AUROC_LT_0_40" if original < .4 else "AUROC_LT_0_50", "label_convention": {"normal": 0, "anomaly": 1}, "score_direction": "log p1 - log p0; higher is anomalous", "train_test_normal_disjoint": set(data["train"]["normal"]).isdisjoint(data["test"]["normal"]), "held_out_absent_from_training": held_out not in data["seen_subtypes"], "episode_overlap_violations": overlap, "normal_score": descriptive(scores[normal]), "anomaly_score": descriptive(scores[anomaly]), "normal_logp_anomaly": descriptive(p1[normal]), "anomaly_logp_anomaly": descriptive(p1[anomaly]), "normal_logp_normal": descriptive(p0[normal]), "anomaly_logp_normal": descriptive(p0[anomaly]), "episode_AUROC": {**descriptive(episode_aucs), "p5": float(np.percentile(episode_aucs, 5)), "p95": float(np.percentile(episode_aucs, 95)), "fraction_below_0_50": float(np.mean(np.asarray(episode_aucs) < .5))}, "posterior_recomputed_after_adaptation": True, "posterior": {"kappa_n": KAPPA0 + K_SHOT, "nu_n": 128 + NU0_OFFSET + K_SHOT, "degrees_of_freedom": 8, "episodes": posterior}, "diagnosis": diagnosis, "note": "Repeated episode queries are traceability records, not independent image-level samples for McNemar or inferential tests."}
    write_json(out_dir / "summary.json", summary)
    fields = list(records[0]);
    for name in ("query_scores.csv", "below_chance_diagnostics.csv"):
        with (out_dir / name).open("w", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    for name, arrays, labels_plot in (("score_distribution.png", [scores[normal], scores[anomaly]], ["Normal", "Anomaly"]), ("log_probability_distribution.png", [p1, p0, scores], ["log p1", "log p0", "score"])):
        plt.figure(figsize=(6, 4)); [plt.hist(a, bins=30, alpha=.55, label=l) for a, l in zip(arrays, labels_plot)]; plt.legend(); plt.tight_layout(); plt.savefig(out_dir / name); plt.close()
    fpr, tpr, _ = roc_curve(labels, scores); plt.figure(figsize=(5, 4)); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],"--",color="gray"); plt.tight_layout(); plt.savefig(out_dir / "roc_curve.png"); plt.close()
    plt.figure(figsize=(6, 4)); plt.hist(episode_aucs, bins=min(20, len(episode_aucs))); plt.axvline(.5, color="red", linestyle="--"); plt.tight_layout(); plt.savefig(out_dir / "episode_auroc_distribution.png"); plt.close()
    return summary
