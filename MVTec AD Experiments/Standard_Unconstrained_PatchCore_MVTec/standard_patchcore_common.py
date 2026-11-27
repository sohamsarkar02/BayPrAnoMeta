"""Protocol adapter around the vendored official PatchCore implementation."""

import json, os, random, sys, threading, time
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

OFFICIAL_ROOT = Path(__file__).resolve().parent / "official_patchcore"
sys.path.insert(0, str(OFFICIAL_ROOT))
import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler

from utils.utils_data_fedcontrast_leave1out import load_client_dataset_leave1out, transform_eval

BASE_ROOT = None
REFERENCE_MANIFEST_ROOT = Path(__file__).resolve().parents[1] / "Bayesian_ProtoMAML" / "runs" / "baypranometa_avgpool_three_seed" / "revised_clean"
RESULT_ROOT = "results/standard_unconstrained_patchcore_mvtec"
CLIENTS = ["bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]
SPLIT_SEED = 42
DEVICE = torch.device("cpu")
CORESET_FRACTION = 0.10
BACKBONE = "wideresnet50"
FEATURE_LAYERS = ["layer2", "layer3"]
PRETRAIN_DIM = TARGET_DIM = 1024
PATCH_SIZE = 3
NUM_NN = 1


class PathDataset(Dataset):
    def __init__(self, paths): self.paths = list(paths)
    def __len__(self): return len(self.paths)
    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return transform_eval(image.convert("RGB"))


class RecordingApproximateGreedyCoresetSampler(patchcore.sampler.ApproximateGreedyCoresetSampler):
    def run(self, features):
        self.before_count = len(features)
        sampled = super().run(features)
        self.after_count = len(sampled)
        return sampled


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_patchcore(seed):
    set_seed(seed)
    backbone = patchcore.backbones.load(BACKBONE)
    backbone.name, backbone.seed = BACKBONE, seed
    sampler = RecordingApproximateGreedyCoresetSampler(CORESET_FRACTION, DEVICE)
    model = patchcore.patchcore.PatchCore(DEVICE)
    model.load(backbone=backbone, layers_to_extract_from=FEATURE_LAYERS, device=DEVICE,
               input_shape=(3,224,224), pretrain_embed_dimension=PRETRAIN_DIM,
               target_embed_dimension=TARGET_DIM, patchsize=PATCH_SIZE,
               featuresampler=sampler, anomaly_score_num_nn=NUM_NN,
               nn_method=patchcore.common.FaissNN(False, 4))
    return model, sampler


def loader(paths, batch_size=2):
    return DataLoader(PathDataset(paths), batch_size=batch_size, shuffle=False, num_workers=0)


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w") as handle: json.dump(value,handle,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))


class PeakMemoryMonitor:
    def __init__(self): self.process=psutil.Process(os.getpid()); self.peak_mb=0.; self.running=False
    def start(self):
        self.running=True
        def poll():
            while self.running:
                self.peak_mb=max(self.peak_mb,self.process.memory_info().rss/1024**2); time.sleep(.1)
        self.thread=threading.Thread(target=poll,daemon=True); self.thread.start()
    def stop(self): self.running=False; self.thread.join()


def load_protocol(base_root, category, split_seed, validation_fraction=.30):
    return load_client_dataset_leave1out(Path(base_root)/category, seed=split_seed,
                                         protocol="revised_clean", validation_fraction=validation_fraction)
