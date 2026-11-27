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

VALID_PROTOCOLS = {"legacy_leaky", "revised_clean"}


def _split_for_validation(paths, rng, validation_fraction):
    """Deterministically split one pool without changing its image identities."""
    paths = list(paths)
    if len(paths) < 2:
        return paths, []
    shuffled = list(paths)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * validation_fraction)))
    n_val = min(n_val, len(shuffled) - 1)
    return shuffled[n_val:], shuffled[:n_val]


def _assert_protocol(data, protocol):
    """Cheap, one-time path-level checks; no image tensors are loaded here."""
    train_n = set(data["train"]["normal"])
    val_n = set(data["validation"]["normal"])
    test_n = set(data["test"]["normal"])
    train_a = set(data["train"]["anomaly"])
    val_a = set(data["validation"]["anomaly"])
    test_a = set(data["test"]["anomaly"])

    assert train_a.isdisjoint(test_a), "Held-out anomaly images entered training."
    assert val_a.isdisjoint(test_a), "Held-out anomaly images entered validation."
    assert data["held_out_subtype"] not in data["seen_subtypes"]
    assert all(Path(p).parent.name == data["held_out_subtype"] for p in test_a)
    if protocol == "revised_clean":
        assert train_n.isdisjoint(val_n), "Train and validation normal pools overlap."
        assert train_n.isdisjoint(test_n), "Test/good entered revised training."
        assert val_n.isdisjoint(test_n), "Test/good entered revised validation."
        assert train_a.isdisjoint(val_a), "Train and validation anomaly pools overlap."


def load_client_dataset_leave1out(
    client_root,
    seed=42,
    protocol="revised_clean",
    validation_fraction=0.2,
):
    """
    Leave-one-anomaly-subtype-out protocol.

    Training (revised_clean):
      - Normal  : a deterministic subset of train/good
      - Anomaly : all anomaly subtypes EXCEPT one

    Validation (revised_clean):
      - Normal  : disjoint remainder of train/good
      - Anomaly : disjoint images from seen anomaly subtypes

    Testing (both protocols):
      - Normal  : test/good
      - Anomaly : ONLY the held-out subtype

    Clients with < 2 anomaly subtypes are skipped.
    """

    if protocol not in VALID_PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}; choose from {sorted(VALID_PROTOCOLS)}")
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
    anom_dirs = sorted(
        (d for d in (root / "test").iterdir() if d.is_dir() and d.name != "good"),
        key=lambda d: d.name,
    )

    if len(anom_dirs) < 2:
        return None

    rng.shuffle(anom_dirs)

    held_out_dir = anom_dirs[0]
    seen_dirs = anom_dirs[1:]

    # -------- Build anomaly pools and a true validation pool --------
    anom_train, anom_val, anom_seen_all = [], [], []
    for d in seen_dirs:
        subtype_images = list_images(d)
        anom_seen_all.extend(subtype_images)
        subtype_train, subtype_val = _split_for_validation(subtype_images, rng, validation_fraction)
        anom_train.extend(subtype_train)
        anom_val.extend(subtype_val)

    anom_test = list_images(held_out_dir)

    if len(anom_train) == 0 or len(anom_test) == 0:
        return None

    if protocol == "legacy_leaky":
        # Historical compatibility only: deliberately reproduces the paper's
        # former train/good + test/good normal-training pool.
        train_normal = train_good + test_good
        validation_normal = list(train_normal)
        anom_train = list(anom_seen_all)
        validation_anomaly = list(anom_seen_all)
    else:
        train_normal, validation_normal = _split_for_validation(
            train_good, rng, validation_fraction
        )
        validation_anomaly = anom_val
        if not validation_normal or not validation_anomaly:
            raise ValueError(
                f"{root.name}: revised_clean validation split is empty; "
                "adjust validation_fraction or provide more training images."
            )

    data = {
        "train": {
            "normal": train_normal,
            "anomaly": anom_train
        },
        "validation": {
            "normal": validation_normal,
            "anomaly": validation_anomaly,
        },
        "test": {
            "normal": test_good,
            "anomaly": anom_test
        },
        "protocol": protocol,
        "validation_fraction": validation_fraction,
        "held_out_subtype": held_out_dir.name,
        "seen_subtypes": [d.name for d in seen_dirs]
    }
    _assert_protocol(data, protocol)
    return data
