"""Shared definitions for corrected Classical MAML three-seed runs."""
import json, os, random, threading, time
from pathlib import Path
import numpy as np
import psutil
import torch
import torch.nn as nn
import torchvision.models as models

BASE_ROOT = None
CLIENTS = ["bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]
RUN_ROOT = "runs/maml_corrected_avgpool_three_seed"
PROTOCOL, SPLIT_SEED = "revised_clean", 42
DEVICE = torch.device("cpu")
EPOCHS, EPISODES_PER_EPOCH, VAL_EPISODES, TEST_EPISODES = 30, 25, 20, 100
INNER_STEPS, INNER_LR, META_LR = 1, 5e-4, 1e-4
K_SHOT, Q_N, Q_A = 5, 12, 4

class MAMLNet(nn.Module):
    """ResNet18 -> 128D embedding -> one anomaly logit."""
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.embedding = nn.Sequential(nn.Linear(512,128), nn.ReLU(), nn.LayerNorm(128))
        self.classifier = nn.Linear(128,1)
    def forward(self, x):
        z = self.features(x).flatten(1)
        return self.classifier(self.embedding(z)).squeeze(-1)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=2, default=lambda x: x.item() if isinstance(x,np.generic) else str(x))

def stack(paths, transform, loader):
    tensors = loader(paths, transform)
    if len(tensors) != len(paths):
        raise RuntimeError("At least one episode image failed to load.")
    return torch.stack(tensors).to(DEVICE)

def sample_episode(data, split, rng):
    normal = rng.sample(data[split]["normal"], K_SHOT + Q_N)
    support, normal_query = normal[:K_SHOT], normal[K_SHOT:]
    anomaly_query = rng.sample(data[split]["anomaly"], Q_A)
    assert set(support).isdisjoint(normal_query)
    return support, normal_query, anomaly_query

class PeakMemoryMonitor:
    def __init__(self):
        self.process, self.peak_mb, self.live = psutil.Process(os.getpid()), 0.0, False
    def start(self):
        self.live = True
        def poll():
            while self.live:
                self.peak_mb = max(self.peak_mb, self.process.memory_info().rss / 1024**2)
                time.sleep(.1)
        self.thread = threading.Thread(target=poll, daemon=True); self.thread.start()
    def stop(self):
        self.live = False; self.thread.join()
