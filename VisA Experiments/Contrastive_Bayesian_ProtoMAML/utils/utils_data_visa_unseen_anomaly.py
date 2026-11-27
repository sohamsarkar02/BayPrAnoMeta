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
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.JPG", "*.JPEG", "*.PNG"]
    imgs = []
    for ext in exts:
        imgs.extend(path.rglob(ext))
    return sorted([str(p) for p in imgs])

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
# Unseen-Anomaly-Images Split
# ------------------------------

def load_client_dataset_visa(client_root, seed=42, anomaly_test_ratio=0.2):
    """
    VisA protocol (no anomaly subtypes, no train/test folders).

    Expected structure:
        client_root/
            Data/
                Images/
                    Normal/
                    Anomaly/

    Training:
      - Normal  : all normal images
      - Anomaly : random subset of anomaly images

    Testing:
      - Normal  : all normal images
      - Anomaly : held-out anomaly images only

    This simulates evaluation on unseen anomaly images.
    """

    rng = random.Random(seed)
    root = Path(client_root)

    # -------- Normal images --------
    normal_imgs = list_images(root / "Data" / "Images" / "Normal")

    # -------- Anomaly images --------
    anomaly_imgs = list_images(root / "Data" / "Images" / "Anomaly")

    if len(normal_imgs) == 0 or len(anomaly_imgs) < 8:
        return None

    rng.shuffle(anomaly_imgs)

    n_test = max(4, int(anomaly_test_ratio * len(anomaly_imgs)))
    n_test = min(n_test, len(anomaly_imgs) - 4)

    anom_test = anomaly_imgs[:n_test]
    anom_train = anomaly_imgs[n_test:]

    if len(anom_train) < 4 or len(anom_test) < 4:
        return None

    return {
        "train": {
            "normal": normal_imgs,
            "anomaly": anom_train
        },
        "test": {
            "normal": normal_imgs,
            "anomaly": anom_test
        },
        "held_out_anomaly_count": len(anom_test),
        "seen_anomaly_count": len(anom_train)
    }