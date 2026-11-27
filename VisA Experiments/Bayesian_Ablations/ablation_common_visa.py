"""Shared corrected VisA components for the MVTec-designed Bayesian ablations."""
import csv, json, math, os, random, sys, threading, time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torchvision.models as models

HERE = Path(__file__).resolve().parent
VISA_ROOT = HERE.parent
BAYES_ROOT = VISA_ROOT / "Bayesian_ProtoMAML"
PROTO_ROOT = VISA_ROOT / "Classical_ProtoMAML"
sys.path.insert(0, str(BAYES_ROOT))

from utils.utils_data_visa_leave1out import (  # noqa: E402
    discover_categories, load_client_dataset_visa, load_images, transform_eval,
)
from utils.utils_niw import log_student_t, niw_posterior  # noqa: E402

BASE_ROOT = None
PROTOCOL = "revised_clean_official_2cls_highshot"
SPLIT_SEED = 42
K_SHOT, Q_N, Q_A = 5, 12, 4
INNER_STEPS, INNER_LR = 1, 5e-4
TEST_EPISODES = 100
DEVICE = torch.device("cpu")
FULL_RUN_ROOT = BAYES_ROOT / "runs" / "baypranometa_visa_three_seed_fixed_split42_25ep_30epochs_100test"
PROTO_RUN_ROOT = PROTO_ROOT / "runs" / "protomaml_visa_corrected_avgpool_three_seed"
FIXED_RUN_ROOT = HERE / "results" / "fixed_lambda"
OUTPUT_ROOT = HERE / "results" / "evaluation"


class PeakMemoryMonitor:
    def __init__(self):
        self.process, self.peak_mb, self.running = psutil.Process(os.getpid()), 0.0, False

    def start(self):
        self.running = True
        def poll():
            while self.running:
                self.peak_mb = max(self.peak_mb, self.process.memory_info().rss / 1024 ** 2)
                time.sleep(0.1)
        self.thread = threading.Thread(target=poll, daemon=True); self.thread.start()

    def stop(self):
        self.running = False; self.thread.join()


class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.LayerNorm(128))

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_stack(paths):
    images = load_images(paths, transform_eval)
    if len(images) != len(paths):
        raise RuntimeError("Image loading failed; refusing label/path misalignment.")
    return torch.stack(images).to(DEVICE)


def load_clients(base_root, split_seed, validation_fraction, split_csv, category=None, audit_dir=None, subset_file=None):
    categories = discover_categories(base_root, split_csv, subset_file=subset_file)
    if category:
        if category not in categories: raise ValueError(f"Unknown VisA category: {category}")
        categories = [category]
    clients = {}
    for name in categories:
        data = load_client_dataset_visa(
            base_root, name, seed=split_seed, validation_fraction=validation_fraction,
            split_csv=split_csv, subset_file=subset_file, audit_dir=audit_dir, print_audit=False)
        if data is not None: clients[name] = data
    if not clients: raise RuntimeError("No eligible VisA categories.")
    return clients


def fixed_lambda_posterior(z, validate=False):
    K, d = z.shape
    mu0 = torch.zeros(d, device=z.device, dtype=z.dtype)
    lambda0 = torch.eye(d, device=z.device, dtype=z.dtype)
    zbar = z.mean(0); kappa_n = 0.01 + K; nu_n = d + 2 + K
    mu_n = (0.01 * mu0 + K * zbar) / kappa_n
    dof = nu_n - d + 1
    sigma_n = ((kappa_n + 1) / (kappa_n * dof)) * lambda0
    if validate:
        expected = ((kappa_n + 1) / (kappa_n * dof)) * lambda0
        assert torch.allclose(sigma_n, expected)
        torch.linalg.cholesky(sigma_n + 1e-6 * lambda0)
    return mu_n, sigma_n, dof


def reference_log_probability(z):
    d = z.shape[1]
    return log_student_t(
        z, torch.zeros(d, device=z.device, dtype=z.dtype),
        100.0 * torch.eye(d, device=z.device, dtype=z.dtype),
        torch.tensor(2.0, device=z.device, dtype=z.dtype))


def checkpoint_paths(seed, full_root=FULL_RUN_ROOT, proto_root=PROTO_RUN_ROOT, fixed_root=FIXED_RUN_ROOT):
    return {
        "full": Path(full_root) / PROTOCOL / f"seed_{seed}" / "checkpoints" / "final_model.pth",
        "proto": Path(proto_root) / PROTOCOL / f"seed_{seed}" / "checkpoints" / "final_model.pth",
        "fixed_lambda": Path(fixed_root) / PROTOCOL / f"seed_{seed}" / "checkpoints" / "final_model.pth",
    }


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def descriptive(values):
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "min": float(x.min()), "max": float(x.max())}

