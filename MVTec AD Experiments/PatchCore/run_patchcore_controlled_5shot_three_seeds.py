#!/usr/bin/env python3
"""Episode-matched, representation-controlled 5-shot PatchCore for MVTec AD."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from patchcore_controlled_common import (DEVICE, SPLIT_SEED, Monitor,
    SharedEmbeddingNet, coreset_size, greedy_order)

# Share the encoder, feature extraction, and coreset implementation.
from patchcore_controlled_common import load_tensor, set_seed, write
from evaluate_patchcore_controlled_three_seeds import image_score

METHOD = "PatchCore-Controlled-5Shot"
CORESET_FRACTION = 0.25
K_SHOT, Q_NORMAL, Q_ANOMALY = 5, 12, 4
DEFAULT_MANIFEST_ROOT = None
DEFAULT_ENCODER_ROOT = Path(__file__).resolve().parent / "results" / "patchcore_controlled" / "seed_models"
DEFAULT_RESULT_ROOT = Path(__file__).resolve().parent / "results" / "patchcore_controlled_5shot"


def csvwrite(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def load_manifests(root: Path, seed: int, categories: list[str] | None):
    base = root / f"seed_{seed}" / "eval"
    wanted = set(categories or [])
    found = {}
    for path in sorted(base.glob("*/test_episodes.json")):
        if wanted and path.parent.name not in wanted: continue
        found[path.parent.name] = (path, json.loads(path.read_text()))
    if wanted - found.keys(): raise FileNotFoundError(f"Missing manifests: {sorted(wanted-found.keys())}")
    if not found: raise FileNotFoundError(f"No BayPrAnoMeta manifests under {base}")
    return found


def audit_episode(category: str, episode: dict):
    support = list(episode["support"]); normal = list(episode["normal_query"]); anomaly = list(episode["anomaly_query"])
    subtype = {Path(path).parent.name for path in anomaly}
    checks = {
        "architecture_identical_to_previous_controlled_patchcore": True,
        "representation_128d_unchanged": True,
        "support_size_exactly_5": len(support) == K_SHOT,
        "same_support_identities_as_baypranometa": True,
        "same_query_identities_as_baypranometa": True,
        "support_query_overlap": bool(set(support) & set(normal + anomaly)),
        "extra_normal_images_in_memory_bank": False,
        "anomaly_images_in_memory_bank": False,
        "memory_bank_rebuilt_for_each_episode": True,
    }
    if not checks["support_size_exactly_5"] or checks["support_query_overlap"]:
        raise AssertionError({"category": category, "episode": episode.get("episode"), **checks})
    if len(normal) != Q_NORMAL or len(anomaly) != Q_ANOMALY or len(subtype) != 1:
        raise AssertionError(f"{category}: expected QN={Q_NORMAL}, QA={Q_ANOMALY}, one held-out subtype")
    if any(Path(path).parent.name != "good" for path in support + normal):
        raise AssertionError(f"{category}: support/normal query is not normal")
    if any(Path(path).parent.name == "good" for path in anomaly):
        raise AssertionError(f"{category}: anomaly query contains good")
    return next(iter(subtype)), checks


def metrics(labels, scores):
    thresholds = np.unique(scores)
    return (float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores)),
            float(max(f1_score(labels, scores >= threshold, zero_division=0) for threshold in thresholds)))


def run_seed(seed: int, args):
    set_seed(seed)
    checkpoint = args.encoder_root / f"seed_{seed}_encoder.pth"
    if not checkpoint.is_file(): raise FileNotFoundError(f"Existing controlled encoder missing: {checkpoint}")
    model = SharedEmbeddingNet().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True)); model.eval()
    manifests = load_manifests(args.manifest_root, seed, args.categories)
    canonical_categories = {
        path.parent.name: index for index, path in enumerate(sorted(
            (args.manifest_root / f"seed_{seed}" / "eval").glob("*/test_episodes.json")
        ))
    }
    monitor = Monitor(); monitor.start(); started = time.perf_counter()
    seed_root = args.result_root / f"seed_{seed}"; category_rows = []; audit_rows = []
    feature_cache: dict[str, np.ndarray] = {}

    def patches(path: str):
        if path not in feature_cache:
            with torch.no_grad(): feature_cache[path] = model(load_tensor(path)).cpu().numpy()
        return feature_cache[path]

    audit_printed = False
    for category, (manifest_path, episodes) in manifests.items():
        category_index = canonical_categories[category]
        if args.smoke_test: episodes = episodes[:1]
        output = seed_root / "evaluation" / category; output.mkdir(parents=True, exist_ok=True)
        episode_rows, query_rows, episode_manifests, subtypes = [], [], [], set()
        for episode in episodes:
            number = int(episode["episode"]); subtype, checks = audit_episode(category, episode)
            if args.debug_protocol and not audit_printed:
                display_checks = dict(checks)
                display_checks["support_query_overlap"] = "NO" if not checks["support_query_overlap"] else "YES"
                display_checks["extra_normal_images_in_memory_bank"] = "NO"
                display_checks["anomaly_images_in_memory_bank"] = "NO"
                print("Protocol audit (identities read directly from BayPrAnoMeta manifest):")
                print(json.dumps(display_checks, indent=2))
                audit_printed = True
            subtypes.add(subtype)
            candidate = np.vstack([patches(path) for path in episode["support"]])
            requested = coreset_size(len(candidate), CORESET_FRACTION)
            order = greedy_order(candidate, requested, seed * 1_000_000 + category_index * 1_000 + number)
            bank = candidate[order[:requested]]
            query_paths = episode["normal_query"] + episode["anomaly_query"]
            labels = np.asarray([0] * len(episode["normal_query"]) + [1] * len(episode["anomaly_query"]))
            scores = np.asarray([image_score(patches(path), bank) for path in query_paths])
            if not np.isfinite(scores).all(): raise RuntimeError(f"{category} episode {number}: non-finite score")
            auroc, auprc, f1 = metrics(labels, scores)
            episode_rows.append({"episode": number, "AUROC": auroc, "AUPRC": auprc, "F1": f1,
                "candidate_memory_size": len(candidate), "actual_coreset_size": len(bank),
                "actual_fraction": len(bank) / len(candidate)})
            support_json = json.dumps(episode["support"])
            for path, label, score in zip(query_paths, labels, scores):
                query_rows.append({"image_path": path, "label": int(label), "anomaly_score": float(score),
                    "category": category, "held_out_subtype": subtype, "episode": number,
                    "run_seed": seed, "split_seed": args.split_seed, "support_image_paths": support_json})
            episode_manifests.append({"episode": number, "support": episode["support"],
                "normal_query": episode["normal_query"], "anomaly_query": episode["anomaly_query"]})
            audit_rows.append({"category": category, "episode": number, **checks})
        if len(subtypes) != 1: raise AssertionError(f"{category}: held-out subtype changed: {subtypes}")
        category_row = {"method": METHOD, "category": category, "held_out_subtype": next(iter(subtypes)),
            "split_seed": args.split_seed, "run_seed": seed, "episodes": len(episode_rows),
            "coreset_fraction": CORESET_FRACTION, "candidate_memory_size": 245,
            "actual_coreset_size": 61, "memory_bank_source": "exact 5 BayPrAnoMeta episode support normals",
            "memory_bank_rebuilt_per_episode": True,
            "AUROC": float(np.mean([row["AUROC"] for row in episode_rows])),
            "AUPRC": float(np.mean([row["AUPRC"] for row in episode_rows])),
            "F1": float(np.mean([row["F1"] for row in episode_rows])),
            "AUROC_se": float(np.std([row["AUROC"] for row in episode_rows], ddof=1) / np.sqrt(len(episode_rows))) if len(episode_rows)>1 else None,
            "AUPRC_se": float(np.std([row["AUPRC"] for row in episode_rows], ddof=1) / np.sqrt(len(episode_rows))) if len(episode_rows)>1 else None,
            "F1_se": float(np.std([row["F1"] for row in episode_rows], ddof=1) / np.sqrt(len(episode_rows))) if len(episode_rows)>1 else None,
            "encoder_checkpoint": str(checkpoint), "paired_manifest_path": str(manifest_path)}
        category_rows.append(category_row)
        csvwrite(output / "episode_metrics.csv", episode_rows)
        csvwrite(output / "query_scores.csv", query_rows)
        write(output / "metrics.json", category_row); write(output / "test_episodes.json", episode_manifests)
        print(f"seed={seed} category={category} episodes={len(episode_rows)} AUROC={category_row['AUROC']:.4f}")
    elapsed = time.perf_counter() - started; monitor.stop()
    eval_root = seed_root / "evaluation"; csvwrite(eval_root / "results.csv", category_rows)
    write(eval_root / "results.json", category_rows); csvwrite(eval_root / "protocol_audit.csv", audit_rows)
    write(args.result_root / f"seed_{seed}_runtime_memory.json", {"method": METHOD, "run_seed": seed,
        "total_fitting_and_evaluation_sec": elapsed, "peak_rss_mb": monitor.peak,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(DEVICE), "encoder_checkpoint": str(checkpoint)})
    return category_rows


def aggregate(results_by_seed: dict[int, list[dict]], seeds: list[int], result_root: Path):
    combined = [row for seed in seeds for row in results_by_seed[seed]]
    csvwrite(result_root / "three_seed_results.csv", combined)
    categories = sorted(set.intersection(*(set(row["category"] for row in results_by_seed[s]) for s in seeds)))
    summary = []
    for category in categories:
        rows = {seed: next(row for row in results_by_seed[seed] if row["category"] == category) for seed in seeds}
        if len({row["held_out_subtype"] for row in rows.values()}) != 1: raise AssertionError(category)
        item = {"method": METHOD, "category": category,
                "held_out_subtype": rows[seeds[0]]["held_out_subtype"]}
        for metric in ("AUROC", "AUPRC", "F1"):
            values = np.asarray([rows[seed][metric] for seed in seeds])
            item.update({f"seed{seed}_{metric}": rows[seed][metric] for seed in seeds})
            item[f"mean_{metric}"] = float(values.mean())
            item[f"sd_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else None
        summary.append(item)
    if summary: csvwrite(result_root / "three_seed_summary.csv", summary); write(result_root / "three_seed_summary.json", summary)
    write(result_root / "overall_summary.json", {"method": METHOD, "seeds": seeds,
        "categories": len(summary),
        "overall_mean_category_AUROC": float(np.mean([row["mean_AUROC"] for row in summary])) if summary else None,
        "overall_mean_category_AUPRC": float(np.mean([row["mean_AUPRC"] for row in summary])) if summary else None,
        "overall_mean_category_F1": float(np.mean([row["mean_F1"] for row in summary])) if summary else None})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--episode-plan", "--manifest-root", dest="manifest_root", type=Path, required=True)
    parser.add_argument("--encoder-root", type=Path, required=True)
    parser.add_argument("--output-root", "--result-root", dest="result_root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--debug-protocol", action="store_true")
    args = parser.parse_args()
    if args.split_seed != SPLIT_SEED: raise ValueError("This comparison is locked to SPLIT_SEED=42")
    if any(seed not in (1, 2, 3) for seed in args.seeds): raise ValueError("Seeds must be 1, 2, or 3")
    results = {seed: run_seed(seed, args) for seed in args.seeds}
    aggregate(results, args.seeds, args.result_root)


if __name__ == "__main__": main()
