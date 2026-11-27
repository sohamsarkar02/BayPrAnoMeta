import random
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

transform_train = T.Compose([T.Resize(256), T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ColorJitter(.1, .1, .1, .1), T.ToTensor(), T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
transform_eval = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])

VALID_PROTOCOLS = {"legacy_leaky", "revised_clean"}


def list_images(path): return sorted(str(p) for p in Path(path).glob("*.png"))
def load_images(paths, transform):
    images = []
    for path in paths:
        try: images.append(transform(Image.open(path).convert("RGB")))
        except Exception: continue
    return images


def _split(paths, rng, fraction):
    paths = list(paths)
    if len(paths) < 2: return paths, []
    rng.shuffle(paths); n_val = min(max(1, int(round(len(paths) * fraction))), len(paths) - 1)
    return paths[n_val:], paths[:n_val]


def _assert_protocol(data, protocol):
    train_n, val_n, test_n = map(set, (data["train"]["normal"], data["validation"]["normal"], data["test"]["normal"]))
    train_a, val_a, test_a = map(set, (data["train"]["anomaly"], data["validation"]["anomaly"], data["test"]["anomaly"]))
    assert train_a.isdisjoint(test_a) and val_a.isdisjoint(test_a)
    assert data["held_out_subtype"] not in data["seen_subtypes"]
    assert all(Path(path).parent.name == data["held_out_subtype"] for path in test_a)
    if protocol == "revised_clean":
        assert train_n.isdisjoint(val_n) and train_n.isdisjoint(test_n) and val_n.isdisjoint(test_n)
        assert train_a.isdisjoint(val_a)


def load_client_dataset_leave1out(client_root, seed=42, protocol="revised_clean", validation_fraction=.30):
    """Clean leave-one-subtype-out split; seed fixes subtype and train/validation split."""
    if protocol not in VALID_PROTOCOLS: raise ValueError(f"Unknown protocol: {protocol}")
    rng, root = random.Random(seed), Path(client_root)
    train_good, test_good = list_images(root / "train" / "good"), list_images(root / "test" / "good")
    if not train_good or not test_good: return None
    anom_dirs = sorted((d for d in (root / "test").iterdir() if d.is_dir() and d.name != "good"), key=lambda d: d.name)
    if len(anom_dirs) < 2: return None
    rng.shuffle(anom_dirs); held_out, seen = anom_dirs[0], anom_dirs[1:]
    anom_train, anom_val, seen_all = [], [], []
    for directory in seen:
        images = list_images(directory); seen_all.extend(images); train, val = _split(images, rng, validation_fraction); anom_train.extend(train); anom_val.extend(val)
    anom_test = list_images(held_out)
    if not anom_train or not anom_test: return None
    if protocol == "legacy_leaky":
        train_normal, val_normal, anom_train, val_anom = train_good + test_good, train_good + test_good, seen_all, seen_all
    else:
        train_normal, val_normal = _split(train_good, rng, validation_fraction); val_anom = anom_val
        if not val_normal or not val_anom: raise ValueError(f"{root.name}: empty clean validation pool")
    data = {"train": {"normal": train_normal, "anomaly": anom_train}, "validation": {"normal": val_normal, "anomaly": val_anom}, "test": {"normal": test_good, "anomaly": anom_test}, "protocol": protocol, "held_out_subtype": held_out.name, "seen_subtypes": [d.name for d in seen]}
    _assert_protocol(data, protocol); return data
