"""Unchanged representation-controlled PatchCore components used by MVTec."""
import json
import os
import random
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

SPLIT_SEED = 42
DEVICE = torch.device("cpu")
TRANSFORM = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([.485, .456, .406], [.229, .224, .225]),
])


class SharedEmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.maxpool = nn.AvgPool2d(3, 2, 1)
        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.projector = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True), nn.LayerNorm(128))

    def forward(self, images):
        features = self.backbone(images)
        return self.projector(features.permute(0, 2, 3, 1).reshape(-1, 512))


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_tensor(path):
    with Image.open(path) as image:
        return TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)


def coreset_size(total, fraction):
    return total if fraction == 1.0 else min(total, max(1, round(fraction * total)))


def greedy_order(features, maximum, seed):
    rng = np.random.default_rng(seed)
    index = int(rng.integers(len(features)))
    selected = [index]
    distances = np.linalg.norm(features - features[index], axis=1)
    for _ in range(1, maximum):
        index = int(np.argmax(distances)); selected.append(index)
        distances = np.minimum(distances, np.linalg.norm(features - features[index], axis=1))
    return np.asarray(selected, dtype=np.int64)


def image_score(patches, memory_bank, chunk=2048):
    nearest = []
    for start in range(0, len(memory_bank), chunk):
        distances = np.linalg.norm(patches[:, None] - memory_bank[None, start:start + chunk], axis=2)
        nearest.append(distances.min(axis=1))
    return float(np.min(np.stack(nearest), axis=0).max())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, default=lambda item: item.item() if isinstance(item, np.generic) else str(item))


class Monitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid()); self.peak = 0.0; self.live = False

    def start(self):
        self.live = True
        def poll():
            while self.live:
                self.peak = max(self.peak, self.process.memory_info().rss / 1024 ** 2)
                time.sleep(.1)
        self.thread = threading.Thread(target=poll, daemon=True); self.thread.start()

    def stop(self):
        self.live = False; self.thread.join()
