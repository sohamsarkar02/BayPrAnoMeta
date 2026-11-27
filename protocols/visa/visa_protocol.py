"""VisA official split and exact paper image-subset validation."""

import csv
from collections import Counter
from functools import lru_cache
from pathlib import Path, PurePosixPath

DEFAULT_SPLIT_CSV = "split_csv/2cls_highshot.csv"
SUBSET_HELP = "Paper image allowlist. Required for exact reproduction of the paper experiments."


def split_path(data_root, split_csv):
    path = Path(split_csv)
    if path.is_absolute() or (str(path) != DEFAULT_SPLIT_CSV and path.is_file()):
        return path.resolve()
    return (Path(data_root) / path).resolve()


def read_split(data_root, split_csv=DEFAULT_SPLIT_CSV, subset_file=None):
    """Return official rows in CSV order, optionally restricted to the paper subset."""
    root = Path(data_root).resolve()
    path = split_path(root, split_csv)
    if subset_file is not None:
        return _paper_rows(root, path, Path(subset_file).resolve())
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=4)
def _paper_rows(root, split_csv, subset_file):
    paths = subset_file.read_text(encoding="utf-8").splitlines()
    allowed = set(paths)
    if len(paths) != 2173 or len(allowed) != 2173:
        raise ValueError(f"VisA subset must contain exactly 2,173 unique image paths: {subset_file}")
    for image in paths:
        relative = PurePosixPath(image)
        if relative.is_absolute() or ".." in relative.parts or "\\" in image or ":" in image:
            raise ValueError(f"Expected a dataset-relative VisA subset path: {image!r}")
        resolved = (root / image).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise FileNotFoundError(f"Allowlisted VisA image is missing under --data-root: {image}")
    with split_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = tuple(row for row in csv.DictReader(handle) if row["image"] in allowed)
    if len(rows) != 2173 or {row["image"] for row in rows} != allowed:
        raise ValueError("VisA subset paths must each appear exactly once in the official split CSV.")
    counts = Counter(row["split"] for row in rows)
    if counts != {"train": 1302, "test": 871}:
        raise ValueError(f"Unexpected VisA subset partition counts: {dict(counts)}")
    return rows


def add_manifest_protocol_arguments(parser):
    parser.add_argument("--data-root", type=Path, help="VisA image root, required with --subset-file.")
    parser.add_argument("--split-csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--subset-file", type=Path, help=SUBSET_HELP)


def manifest_index(args, parser):
    if args.subset_file is None and args.data_root is None:
        return None
    if args.data_root is None:
        parser.error("--data-root is required when --subset-file is supplied")
    root = args.data_root.resolve()
    rows = read_split(root, args.split_csv, args.subset_file)
    return root, {
        (root / row["image"]).resolve(): (row["object"], row["label"])
        for row in rows if row["split"] == "test"
    }


def verify_test_episodes(episodes, category, protocol):
    if protocol is None:
        return
    root, test_images = protocol
    for episode in episodes:
        for field, label in (("support", "normal"), ("normal_query", "normal"),
                             ("anomaly_query", "anomaly")):
            for image in episode[field]:
                if test_images.get((root / image).resolve()) != (category, label):
                    raise ValueError(f"VisA episode image is outside its official test pool: {image}")
