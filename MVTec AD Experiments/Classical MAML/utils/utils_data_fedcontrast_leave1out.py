import random
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# ------------------------------
# Transforms (CPU-safe)
# ------------------------------

transform_train = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.1, 0.1, 0.1, 0.1),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

transform_eval = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------
# Utilities (CPU-only)
# ------------------------------

def list_images(path):
    path = Path(path)
    return sorted([str(p) for p in path.glob("*.png")])

def load_images(paths, transform):
    imgs = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))
        except Exception:
            continue
    return imgs

# ------------------------------
# Leave-One-Anomaly-Subtype-Out
# ------------------------------

def _split_for_validation(paths, rng, fraction):
    shuffled = list(paths)
    rng.shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, []
    n_val = min(max(1, int(round(len(shuffled) * fraction))), len(shuffled) - 1)
    return shuffled[n_val:], shuffled[:n_val]


def load_client_dataset_leave1out(client_root, seed=42, protocol="revised_clean", validation_fraction=0.30):
    """
    Leave-one-anomaly-subtype-out protocol.

    Training (revised_clean):
      - Normal  : a subset of train/good only
      - Anomaly : all anomaly subtypes EXCEPT one

    Validation:
      - A genuinely disjoint remainder of train/good and seen anomalies

    Testing:
      - Normal  : test/good
      - Anomaly : ONLY the held-out subtype

    Clients with < 2 anomaly subtypes (e.g., toothbrush)
    are skipped by returning None.
    """

    if protocol != "revised_clean":
        raise ValueError("Classical MAML corrected runs require protocol='revised_clean'.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    rng = random.Random(seed)
    root = Path(client_root)

    # -------- Normal images --------
    train_good = list_images(root / "train" / "good")
    test_good = list_images(root / "test" / "good")

    if len(train_good) == 0 or len(test_good) == 0:
        return None

    # -------- Anomaly subtypes --------
    anom_dirs = sorted([
        d for d in (root / "test").iterdir()
        if d.is_dir() and d.name != "good"
    ], key=lambda d: d.name)

    # Skip clients with only one anomaly subtype
    if len(anom_dirs) < 2:
        return None

    rng.shuffle(anom_dirs)

    held_out_dir = anom_dirs[0]
    seen_dirs = anom_dirs[1:]

    # -------- Build anomaly pools --------
    anom_train, anom_val = [], []
    for d in seen_dirs:
        subtype_train, subtype_val = _split_for_validation(list_images(d), rng, validation_fraction)
        anom_train.extend(subtype_train)
        anom_val.extend(subtype_val)

    anom_test = list_images(held_out_dir)

    if len(anom_train) == 0 or len(anom_test) == 0:
        return None

    train_normal, val_normal = _split_for_validation(train_good, rng, validation_fraction)
    if not train_normal or not val_normal or not anom_train or not anom_val:
        return None

    data = {
        "train": {
            "normal": train_normal,
            "anomaly": anom_train
        },
        "validation": {
            "normal": val_normal,
            "anomaly": anom_val
        },
        "test": {
            "normal": test_good,
            "anomaly": anom_test
        },
        "held_out_subtype": held_out_dir.name,
        "seen_subtypes": [d.name for d in seen_dirs],
        "protocol": protocol,
        "validation_fraction": validation_fraction
    }
    assert set(train_normal).isdisjoint(val_normal)
    assert set(train_normal).isdisjoint(test_good)
    assert set(val_normal).isdisjoint(test_good)
    assert set(anom_train).isdisjoint(anom_val)
    assert set(anom_train).isdisjoint(anom_test)
    assert all(Path(p).parent.name == held_out_dir.name for p in anom_test)
    return data
