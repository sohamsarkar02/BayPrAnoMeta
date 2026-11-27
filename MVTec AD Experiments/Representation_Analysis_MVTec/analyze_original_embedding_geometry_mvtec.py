#!/usr/bin/env python3
"""Evaluation-only original-space analysis of corrected MVTec representations."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import higher
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.stats import ttest_rel, wilcoxon
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BAYES_CODE = PROJECT / "Bayesian_ProtoMAML"
PROTO_CODE = PROJECT / "CLassical_ProtoMAML"
BAYES_ROOT = BAYES_CODE / "runs" / "baypranometa_avgpool_three_seed" / "revised_clean"
PROTO_ROOT = PROTO_CODE / "runs" / "protomaml_corrected_avgpool_three_seed" / "revised_clean"
OUTPUT_ROOT = HERE / "results"
SEEDS = (1, 2, 3)
INNER_LR = 5e-4
EPSILON = 1e-12
EXPECTED_EPISODES = 100
EXPECTED_CATEGORY_COUNT = 14
SMOKE_CATEGORY = "bottle"

sys.path.insert(0, str(BAYES_CODE))
from utils.utils_niw import log_student_t, niw_posterior  # noqa: E402

TRANSFORM = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([.485, .456, .406], [.229, .224, .225]),
])


class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        base = models.resnet18(weights=None)
        base.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(nn.Linear(512, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))

    def forward(self, images):
        return self.head(self.features(images).flatten(1))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda item: item.item() if isinstance(item, np.generic) else str(item)))


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_image(path: str) -> torch.Tensor:
    with Image.open(path) as image:
        return TRANSFORM(image.convert("RGB"))


def reference_log_probability(z: torch.Tensor) -> torch.Tensor:
    dimension = z.shape[1]
    return log_student_t(z, torch.zeros(dimension, dtype=z.dtype, device=z.device),
                         100.0 * torch.eye(dimension, dtype=z.dtype, device=z.device),
                         torch.tensor(2.0, dtype=z.dtype, device=z.device))


def load_manifest(seed: int, category: str) -> list[dict]:
    path = BAYES_ROOT / f"seed_{seed}" / "eval" / category / "test_episodes.json"
    episodes = json.loads(path.read_text())
    if len(episodes) != EXPECTED_EPISODES: raise AssertionError(path)
    for index, episode in enumerate(episodes):
        if int(episode["episode"]) != index: raise AssertionError(f"Nonsequential episode IDs: {path}")
        if len(episode["support"]) != 5 or len(episode["normal_query"]) != 12 or len(episode["anomaly_query"]) != 4:
            raise AssertionError(f"Unexpected K/QN/QA: {path}")
        if set(episode["support"]) & set(episode["normal_query"] + episode["anomaly_query"]):
            raise AssertionError(f"Support/query overlap: {path}")
    proto_path = PROTO_ROOT / f"seed_{seed}" / "evaluation" / category / "episode_manifest.json"
    proto = json.loads(proto_path.read_text())
    proto_normalized = [{"episode": int(item["episode_id"]), "support": item["support"],
                         "normal_query": item["normal_query"], "anomaly_query": item["anomaly_query"]}
                        for item in proto]
    if episodes != proto_normalized: raise AssertionError(f"Method manifests differ: seed={seed}, category={category}")
    return episodes


def save_state(output: Path, metric_rows, rank_rows, episode_rows):
    """Atomically persist progress after every completed method/category/seed."""
    state = output / "intermediate_state.json"
    temporary = state.with_suffix(".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps({"metric_rows": metric_rows, "rank_rows": rank_rows,
                                     "episode_rows": episode_rows}, indent=2))
    temporary.replace(state)
    if metric_rows: write_csv(output / "original_embedding_metrics_per_category_seed.partial.csv", metric_rows)
    if rank_rows: write_csv(output / "effective_rank_details.partial.csv", rank_rows)


def load_state(output: Path):
    state = output / "intermediate_state.json"
    if not state.is_file(): return [], [], []
    payload = json.loads(state.read_text())
    return payload.get("metric_rows", []), payload.get("rank_rows", []), payload.get("episode_rows", [])


def categories() -> list[str]:
    sets = []
    for seed in SEEDS:
        bayes = {path.parent.name for path in (BAYES_ROOT / f"seed_{seed}" / "eval").glob("*/test_episodes.json")}
        proto = {path.parent.name for path in (PROTO_ROOT / f"seed_{seed}" / "evaluation").glob("*/episode_manifest.json")}
        sets.append(bayes & proto)
    shared = sorted(set.intersection(*sets))
    if len(shared) != EXPECTED_CATEGORY_COUNT:
        raise AssertionError(f"Expected {EXPECTED_CATEGORY_COUNT} paired corrected categories, found {shared}")
    return shared


BAYES_CHECKPOINT_ROOT = PROTO_CHECKPOINT_ROOT = None


def extract_occurrences(method: str, seed: int, category: str, episodes: list[dict], cache_dir: Path):
    cache_path = cache_dir / method / f"seed_{seed}" / f"{category}.npz"
    if cache_path.is_file():
        saved = np.load(cache_path, allow_pickle=True)
        return (saved["embeddings"], saved["paths"].astype(str), saved["labels"].astype(int),
                saved["episode_ids"].astype(int), float(saved["derived_mean_episode_auroc"]))

    checkpoint_root = (BAYES_CHECKPOINT_ROOT or BAYES_ROOT) if method == "BayPrAnoMeta" else (PROTO_CHECKPOINT_ROOT or PROTO_ROOT)
    checkpoint = checkpoint_root / f"seed_{seed}" / "checkpoints" / "final_model.pth"
    model = EmbeddingNet().cpu(); model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)); model.eval()
    unique_paths = sorted({path for episode in episodes for key in ("support", "normal_query", "anomaly_query") for path in episode[key]})
    image_cache = {path: load_image(path) for path in unique_paths}
    embeddings, paths, labels, episode_ids, episode_aucs = [], [], [], [], []

    for episode in episodes:
        support = episode["support"]; query_paths = episode["normal_query"] + episode["anomaly_query"]
        y = np.asarray([0] * 12 + [1] * 4)
        xs = torch.stack([image_cache[path] for path in support])
        xq = torch.stack([image_cache[path] for path in query_paths])
        optimizer = torch.optim.SGD(model.parameters(), lr=INNER_LR)
        with higher.innerloop_ctx(model, optimizer, copy_initial_weights=True, track_higher_grads=False) as (adapted, diffopt):
            zs = adapted(xs)
            if method == "BayPrAnoMeta":
                mu0, sigma0, dof0 = niw_posterior(zs)
                diffopt.step(-log_student_t(zs, mu0, sigma0, dof0).mean())
                with torch.no_grad():
                    zsup = adapted(xs); mu, sigma, dof = niw_posterior(zsup); zq = adapted(xq)
                    scores = (reference_log_probability(zq) - log_student_t(zq, mu, sigma, dof)).cpu().numpy()
            else:
                prototype = zs.mean(0, keepdim=True)
                diffopt.step(((zs - prototype) ** 2).sum(1).mean())
                with torch.no_grad():
                    zsup = adapted(xs); prototype = zsup.mean(0, keepdim=True); zq = adapted(xq)
                    scores = ((zq - prototype) ** 2).sum(1).cpu().numpy()
        array = zq.cpu().numpy()
        if array.shape != (16, 128) or not np.isfinite(array).all(): raise RuntimeError(f"Invalid embeddings: {method}, {seed}, {category}")
        embeddings.append(array); paths.extend(query_paths); labels.extend(y.tolist())
        episode_ids.extend([int(episode["episode"])] * 16); episode_aucs.append(float(roc_auc_score(y, scores)))

    embeddings_array = np.vstack(embeddings).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings_array, paths=np.asarray(paths), labels=np.asarray(labels),
                        episode_ids=np.asarray(episode_ids), derived_mean_episode_auroc=np.mean(episode_aucs),
                        checkpoint=str(checkpoint), method=method, seed=seed, category=category)
    return embeddings_array, np.asarray(paths), np.asarray(labels), np.asarray(episode_ids), float(np.mean(episode_aucs))


def episode_geometry(embeddings, labels, episode_ids):
    rows = []; active = 0; minimum_dispersion = float("inf")
    for episode in sorted(set(episode_ids)):
        mask = episode_ids == episode; z = embeddings[mask]; y = labels[mask]
        normal, anomaly = z[y == 0].astype(np.float64), z[y == 1].astype(np.float64)
        normal_centroid, anomaly_centroid = normal.mean(0), anomaly.mean(0)
        v_n = float(np.mean(np.sum((normal - normal_centroid) ** 2, axis=1)))
        v_a = float(np.mean(np.sum((anomaly - anomaly_centroid) ** 2, axis=1)))
        distance = float(np.linalg.norm(normal_centroid - anomaly_centroid))
        dispersion = v_n + v_a; minimum_dispersion = min(minimum_dispersion, dispersion)
        if EPSILON > 1e-6 * dispersion: active += 1
        rows.append({"episode": int(episode), "within_normal_variance": v_n,
                     "within_anomaly_variance": v_a, "centroid_distance": distance,
                     "normalized_separation": distance / math.sqrt(dispersion + EPSILON),
                     "dispersion_sum": dispersion})
    return rows, active, minimum_dispersion


def unique_image_means(embeddings, paths, labels):
    grouped, grouped_labels = defaultdict(list), {}
    for embedding, path, label in zip(embeddings, paths, labels):
        grouped[str(path)].append(embedding.astype(np.float64))
        if path in grouped_labels and grouped_labels[path] != int(label): raise AssertionError(path)
        grouped_labels[str(path)] = int(label)
    ordered = sorted(grouped)
    means = np.vstack([np.mean(grouped[path], axis=0) for path in ordered])
    unique_labels = np.asarray([grouped_labels[path] for path in ordered])
    counts = np.asarray([len(grouped[path]) for path in ordered])
    return means, np.asarray(ordered), unique_labels, counts


def effective_rank(matrix):
    covariance = np.cov(matrix, rowvar=False, ddof=1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    maximum = float(max(eigenvalues[-1], 0.0)); tolerance = max(1.0, maximum) * 1e-10
    materially_negative = eigenvalues[eigenvalues < -tolerance]
    if len(materially_negative): raise RuntimeError(f"Materially negative covariance eigenvalue: {materially_negative.min()}")
    clipped_count = int(np.sum((eigenvalues < 0) & (eigenvalues >= -tolerance)))
    eigenvalues = np.clip(eigenvalues, 0.0, None); total = float(eigenvalues.sum())
    if total <= 0: raise RuntimeError("Zero covariance spectrum")
    probabilities = eigenvalues[eigenvalues > 0] / total
    rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    numerical_tolerance = max(covariance.shape) * np.finfo(np.float64).eps * maximum
    numerical_rank = int(np.sum(eigenvalues > numerical_tolerance))
    return rank, numerical_rank, clipped_count, float(eigenvalues.min()), maximum


def save_corrected_tsne(category, seed, per_method, output):
    perplexity = min(30, len(next(iter(per_method.values()))[0]) - 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    coordinate_rows = []
    for axis, method in zip(axes, ("BayPrAnoMeta", "Proto-MAML")):
        matrix, paths, labels = per_method[method]
        points = TSNE(n_components=2, perplexity=perplexity, init="random", random_state=seed).fit_transform(matrix)
        axis.scatter(points[labels == 0, 0], points[labels == 0, 1], s=18, alpha=.8, label="Normal")
        axis.scatter(points[labels == 1, 0], points[labels == 1, 1], s=18, alpha=.8, label="Anomaly")
        axis.set_title(method); axis.legend(); axis.set_xticks([]); axis.set_yticks([])
        coordinate_rows.extend({"method": method, "category": category, "run_seed": seed,
                                "image_path": path, "label": int(label), "tsne_x": float(point[0]),
                                "tsne_y": float(point[1]), "perplexity": perplexity,
                                "preprocessing": "raw_128d_per_image_mean_no_standardization_no_pca"}
                               for path, label, point in zip(paths, labels, points))
    figure.suptitle(f"{category}, seed {seed} — corrected matched t-SNE")
    figure.tight_layout(); figure_path = output / "corrected_tSNE" / category / f"seed_{seed}_comparison.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(figure_path, dpi=180); plt.close(figure)
    write_csv(figure_path.with_suffix(".csv"), coordinate_rows)


def bootstrap_ci(values, draws=100000, seed=42):
    values = np.asarray(values, dtype=float); rng = np.random.default_rng(seed); means = np.empty(draws)
    for start in range(0, draws, 10000):
        stop = min(draws, start + 10000)
        means[start:stop] = values[rng.integers(0, len(values), size=(stop - start, len(values)))].mean(1)
    return tuple(float(value) for value in np.percentile(means, [2.5, 97.5]))


def main(configure_parser=None, configure_run=None):
    global BAYES_ROOT, PROTO_ROOT, BAYES_CHECKPOINT_ROOT, PROTO_CHECKPOINT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--full-checkpoint-root", type=Path, help="Protocol subdirectory of the Full training output.")
    parser.add_argument("--proto-checkpoint-root", type=Path, help="Protocol subdirectory of the Proto-MAML training output.")
    parser.add_argument("--full-run-root", type=Path, default=BAYES_ROOT)
    parser.add_argument("--proto-run-root", type=Path, default=PROTO_ROOT)
    if configure_parser is not None:
        configure_parser(parser)
    args = parser.parse_args()
    if configure_run is not None:
        configure_run(args, parser)
    BAYES_ROOT, PROTO_ROOT = args.full_run_root, args.proto_run_root
    BAYES_CHECKPOINT_ROOT, PROTO_CHECKPOINT_ROOT = args.full_checkpoint_root, args.proto_checkpoint_root
    if any(seed not in SEEDS for seed in args.seeds): raise ValueError("Seeds must be selected from 1, 2, 3")
    available_categories = categories()
    selected_categories = args.categories or available_categories
    unknown = sorted(set(selected_categories) - set(available_categories))
    if unknown: raise ValueError(f"Categories are not available in both corrected runs: {unknown}")
    if args.max_episodes is not None and not 1 <= args.max_episodes <= EXPECTED_EPISODES:
        raise ValueError("--max-episodes must be between 1 and 100")
    if args.smoke_test:
        args.seeds, selected_categories, args.max_episodes = [1], [SMOKE_CATEGORY], 3
        if args.output_root == OUTPUT_ROOT: args.output_root = HERE / "results_smoke"
    output = args.output_root.resolve(); set_seed(42)
    metric_rows, rank_rows, episode_rows = load_state(output)
    completed = {(row["method"], int(row["run_seed"]), row["category"]) for row in metric_rows}
    unique_for_tsne = {}

    for seed in args.seeds:
        for category in selected_categories:
            episodes = load_manifest(seed, category)
            if args.max_episodes is not None: episodes = episodes[:args.max_episodes]
            unique_for_tsne[(seed, category)] = {}
            for method in ("BayPrAnoMeta", "Proto-MAML"):
                key = (method, seed, category)
                if key in completed:
                    print(f"SKIP completed method={method} seed={seed} category={category}", flush=True)
                    cache_path = output / "embedding_cache" / method / f"seed_{seed}" / f"{category}.npz"
                    if not cache_path.is_file(): raise FileNotFoundError(f"State is complete but embedding cache is missing: {cache_path}")
                    saved = np.load(cache_path, allow_pickle=True)
                    means, unique_paths, unique_labels, _ = unique_image_means(
                        saved["embeddings"], saved["paths"].astype(str), saved["labels"].astype(int))
                    unique_for_tsne[(seed, category)][method] = (means, unique_paths, unique_labels)
                    continue
                print(f"START method={method} seed={seed} category={category} episodes={len(episodes)}", flush=True)
                embeddings, paths, labels, episode_ids, derived_auc = extract_occurrences(
                    method, seed, category, episodes, output / "embedding_cache")
                geometry, epsilon_active, minimum_dispersion = episode_geometry(embeddings, labels, episode_ids)
                means, unique_paths, unique_labels, occurrence_counts = unique_image_means(embeddings, paths, labels)
                rank, numerical_rank, clipped, minimum_eigenvalue, maximum_eigenvalue = effective_rank(means)
                unique_for_tsne[(seed, category)][method] = (means, unique_paths, unique_labels)
                row = {"method": method, "category": category, "run_seed": seed, "split_seed": 42,
                    "within_normal_variance": float(np.mean([item["within_normal_variance"] for item in geometry])),
                    "within_anomaly_variance": float(np.mean([item["within_anomaly_variance"] for item in geometry])),
                    "centroid_distance": float(np.mean([item["centroid_distance"] for item in geometry])),
                    "normalized_separation": float(np.mean([item["normalized_separation"] for item in geometry])),
                    "effective_rank": rank, "episodes": len(geometry),
                    "epsilon": EPSILON, "epsilon_materially_active_episodes": epsilon_active,
                    "minimum_vn_plus_va": minimum_dispersion, "derived_mean_episode_AUROC": derived_auc,
                    "embedding_dimension": 128, "model_head_normalization": "LayerNorm; no explicit L2 normalization"}
                metric_rows.append(row)
                rank_rows.append({"method": method, "category": category, "run_seed": seed,
                    "effective_rank": rank, "unique_normal_images": int(np.sum(unique_labels == 0)),
                    "unique_anomaly_images": int(np.sum(unique_labels == 1)),
                    "total_unique_images": len(unique_paths),
                    "maximum_possible_empirical_covariance_rank": min(127, len(unique_paths) - 1),
                    "numerical_rank": numerical_rank, "tiny_negative_eigenvalues_clipped": clipped,
                    "minimum_eigenvalue_after_clipping": minimum_eigenvalue,
                    "maximum_eigenvalue": maximum_eigenvalue,
                    "minimum_occurrences_per_image": int(occurrence_counts.min()),
                    "maximum_occurrences_per_image": int(occurrence_counts.max()),
                    "deduplication_rule": "mean all task-adapted occurrences for each physical image path"})
                episode_rows.extend({"method": method, "category": category, "run_seed": seed, **item} for item in geometry)
                completed.add(key); save_state(output, metric_rows, rank_rows, episode_rows)
                print(f"extracted method={method} seed={seed} category={category} "
                      f"VN={row['within_normal_variance']:.6g} Dnorm={row['normalized_separation']:.6g} rank={rank:.4f}", flush=True)
            if not args.skip_tsne and len(unique_for_tsne[(seed, category)]) == 2:
                save_corrected_tsne(category, seed, unique_for_tsne[(seed, category)], output)

    write_csv(output / "original_embedding_metrics_per_category_seed.csv", metric_rows)
    write_csv(output / "effective_rank_details.csv", rank_rows)
    write_csv(output / "episode_level_geometry.csv", episode_rows)

    summary_rows = []
    metric_names = ("within_normal_variance", "within_anomaly_variance", "centroid_distance",
                    "normalized_separation", "effective_rank")
    expected_pairs = len(args.seeds) * len(selected_categories) * 2
    selected_metric_rows = [row for row in metric_rows if int(row["run_seed"]) in args.seeds and row["category"] in selected_categories]
    if len(selected_metric_rows) < expected_pairs: raise RuntimeError("Selected run did not complete; intermediate state was saved for resume")
    for method in ("BayPrAnoMeta", "Proto-MAML"):
        for category in selected_categories:
            rows = sorted((row for row in selected_metric_rows if row["method"] == method and row["category"] == category), key=lambda row: row["run_seed"])
            item = {"method": method, "category": category, "n_seeds": len(rows)}
            for name in metric_names:
                values = np.asarray([row[name] for row in rows])
                for row, value in zip(rows, values): item[f"seed_{row['run_seed']}_{name}"] = float(value)
                item[f"mean_{name}"] = float(values.mean()); item[f"sd_{name}"] = float(values.std(ddof=1))
            summary_rows.append(item)
    write_csv(output / "original_embedding_three_seed_summary.csv", summary_rows)

    directions = {"within_normal_variance": "lower", "within_anomaly_variance": "none",
                  "centroid_distance": "higher", "normalized_separation": "higher", "effective_rank": "none"}
    paired_rows = []
    for name in metric_names:
        bayes = np.asarray([next(row for row in summary_rows if row["method"] == "BayPrAnoMeta" and row["category"] == category)[f"mean_{name}"] for category in selected_categories])
        proto = np.asarray([next(row for row in summary_rows if row["method"] == "Proto-MAML" and row["category"] == category)[f"mean_{name}"] for category in selected_categories])
        differences = bayes - proto; low, high = bootstrap_ci(differences)
        try: w = wilcoxon(differences, alternative="two-sided")
        except ValueError: w = type("Result", (), {"statistic": float("nan"), "pvalue": float("nan")})()
        t = ttest_rel(bayes, proto); direction = directions[name]; tolerance = 1e-12
        if direction == "lower": wins, losses = int(np.sum(differences < -tolerance)), int(np.sum(differences > tolerance))
        elif direction == "higher": wins, losses = int(np.sum(differences > tolerance)), int(np.sum(differences < -tolerance))
        else: wins = losses = "NA"
        paired_rows.append({"metric": name, "difference_definition": "BayPrAnoMeta - Proto-MAML",
            "preferred_direction_for_baypranometa_win": direction, "paired_categories": len(selected_categories),
            "baypranometa_macro_mean": float(bayes.mean()), "protomaml_macro_mean": float(proto.mean()),
            "mean_paired_difference": float(differences.mean()), "median_paired_difference": float(np.median(differences)),
            "baypranometa_wins": wins, "ties": int(np.sum(np.abs(differences) <= tolerance)) if direction != "none" else "NA",
            "protomaml_wins": losses, "wilcoxon_statistic": float(w.statistic),
            "wilcoxon_two_sided_p": float(w.pvalue), "paired_t_statistic": float(t.statistic),
            "paired_t_two_sided_p": float(t.pvalue), "bootstrap_95_ci_low": low,
            "bootstrap_95_ci_high": high, "bootstrap_draws": 100000, "bootstrap_unit": "category"})
    write_csv(output / "paired_embedding_statistics.csv", paired_rows)

    macro_rows = []
    for method in ("BayPrAnoMeta", "Proto-MAML"):
        rows = [row for row in summary_rows if row["method"] == method]
        macro_rows.append({"method": method, "categories": len(rows),
                           **{f"macro_mean_{name}": float(np.mean([row[f'mean_{name}'] for row in rows])) for name in metric_names}})
    write_csv(output / "macro_summary.csv", macro_rows)
    audit = {"retraining_performed": False, "checkpoints_modified": False,
        "methods": ["BayPrAnoMeta", "Proto-MAML"], "run_seeds": list(args.seeds), "split_seed": 42,
        "paired_categories": selected_categories, "paired_category_count": len(selected_categories),
        "episodes_per_category_seed": len(episodes), "query_embedding_dimension": 128,
        "embedding_stage": "post-adaptation query embedding", "explicit_l2_normalization": False,
        "model_head_layernorm": True, "epsilon": EPSILON,
        "epsilon_materially_active_total_episodes": int(sum(row["epsilon_materially_active_episodes"] for row in metric_rows)),
        "effective_rank_rule": "covariance of per-image means of repeated task-adapted embeddings",
        "corrected_tsne_preprocessing": "raw 128D per-image means for both methods; no standardization; no PCA",
        "corrected_tsne_perplexity_rule": "min(30, total_unique_images - 1)", "corrected_tsne_initialization": "random",
        "corrected_tsne_random_seed": "matched run seed for both methods",
        "raw_embedding_cache_created": True, "smoke_test": bool(args.smoke_test),
        "resume_support": True, "intermediate_save_after_each_method_category_seed": True}
    write_json(output / "analysis_audit.json", audit)
    print(json.dumps({"macro": macro_rows, "paired": paired_rows, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
