"""Leakage-free VisA loader using the official 2-class few-shot split."""

import csv
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "protocols" / "visa"))
from visa_protocol import read_split

import torchvision.transforms as T
from PIL import Image

transform_train = T.Compose([
    T.Resize(256), T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
    T.ColorJitter(.1, .1, .1, .1), T.ToTensor(),
    T.Normalize([.485, .456, .406], [.229, .224, .225]),
])
transform_eval = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([.485, .456, .406], [.229, .224, .225]),
])

HELD_OUT_GROUP = "official_2cls_highshot_test_anomalies"
SEEN_GROUP = "official_2cls_highshot_train_anomalies"


def load_images(paths, transform):
    tensors = []
    for path in paths:
        try:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        except Exception:
            continue
    return tensors


def discover_categories(dataset_root, split_csv="split_csv/2cls_highshot.csv", subset_file=None):
    return sorted({row["object"].strip() for row in read_split(dataset_root, split_csv, subset_file)})


def _split_validation(paths, rng, fraction, minimum_each=1):
    paths = list(paths); rng.shuffle(paths)
    if len(paths) < 2 * minimum_each:
        return [], []
    n_val = min(max(minimum_each, round(len(paths) * fraction)), len(paths) - minimum_each)
    return paths[n_val:], paths[:n_val]


def _read_category(dataset_root, category, split_csv, subset_file):
    root, rows, missing = Path(dataset_root), [], []
    for row in read_split(root, split_csv, subset_file):
        if row["object"].strip() != category:
            continue
        image = (root / row["image"].strip()).resolve()
        if not image.is_file():
            if subset_file is not None:
                raise FileNotFoundError(f"Allowlisted VisA image is missing: {row['image']}")
            missing.append(str(image))
            continue
        rows.append({"split": row["split"].strip().lower(),
                     "label": row["label"].strip().lower(), "path": str(image)})
    return rows, missing


def load_client_dataset_visa(dataset_root, category, seed=42,
                             validation_fraction=.30,
                             split_csv="split_csv/2cls_highshot.csv",
                             audit_dir=None, print_audit=True, subset_file=None):
    """Build fixed official train/test groups and validation from train only.

    VisA supplies no anomaly subtype labels. Therefore the official few-shot
    train/test anomaly partition is used as the closest valid held-out grouping.
    ``seed`` affects only the train/validation split and never the official test set.
    """
    rows, missing = _read_category(dataset_root, category, split_csv, subset_file)
    if not rows:
        return None
    pools = {(split, label): sorted(r["path"] for r in rows
                                    if r["split"] == split and r["label"] == label)
             for split in ("train", "test") for label in ("normal", "anomaly")}
    rng = random.Random(seed)
    train_n, val_n = _split_validation(pools[("train", "normal")], rng, validation_fraction, 17)
    train_a, val_a = _split_validation(pools[("train", "anomaly")], rng, validation_fraction, 4)
    test_n, test_a = pools[("test", "normal")], pools[("test", "anomaly")]
    if min(len(train_n), len(val_n), len(train_a), len(val_a), len(test_n), len(test_a)) == 0:
        print(f"WARNING: {category} cannot support train/validation/test episodic pools.")
        return None
    data = {
        "train": {"normal": train_n, "anomaly": train_a},
        "validation": {"normal": val_n, "anomaly": val_a},
        "test": {"normal": test_n, "anomaly": test_a},
        "held_out_subtype": HELD_OUT_GROUP,
        "held_out_anomaly_group": HELD_OUT_GROUP,
        "seen_subtypes": [SEEN_GROUP],
        "split_source": split_csv,
        "anomaly_subtypes_available": False, "missing_official_split_images": missing,
    }
    sets = {"train_n": set(train_n), "val_n": set(val_n), "test_n": set(test_n),
            "train_a": set(train_a), "val_a": set(val_a), "test_a": set(test_a)}
    checks = {
        "train_normal_intersection_test_normal": len(sets["train_n"] & sets["test_n"]),
        "train_normal_intersection_validation_normal": len(sets["train_n"] & sets["val_n"]),
        "validation_normal_intersection_test_normal": len(sets["val_n"] & sets["test_n"]),
        "training_anomaly_intersection_heldout_anomaly": len(sets["train_a"] & sets["test_a"]),
        "validation_anomaly_intersection_heldout_anomaly": len(sets["val_a"] & sets["test_a"]),
    }
    if any(checks.values()):
        raise RuntimeError(f"Data leakage detected for {category}: {checks}")
    audit = {
        "category": category, "split_seed": seed, "split_source": split_csv,
        "train_normals": len(train_n), "validation_normals": len(val_n),
        "test_normals": len(test_n), "train_anomalies": len(train_a),
        "validation_anomalies": len(val_a), "held_out_anomaly_group": HELD_OUT_GROUP,
        "test_anomalies": len(test_a), "overlap_checks": checks,
        "missing_official_split_image_count": len(missing),
        "warning": "VisA has no anomaly subtype labels; official test anomalies are an image-group holdout. The local copy is reduced; only physically present paths listed by the official split are used.",
    }
    if print_audit:
        print(f"Category: {category}\n  Train normals: {len(train_n)}\n  Validation normals: {len(val_n)}"
              f"\n  Test normals: {len(test_n)}\n  Train anomalies: {len(train_a)}"
              f"\n  Held-out anomaly group: {HELD_OUT_GROUP}\n  Test anomalies: {len(test_a)}"
              f"\n  Overlap checks: {checks}")
    if audit_dir:
        path = Path(audit_dir) / f"{category}.json"; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle: json.dump(audit, handle, indent=2)
    data["audit"] = audit
    return data
