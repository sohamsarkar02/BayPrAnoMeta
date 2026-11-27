#!/usr/bin/env python3
"""Episode-matched controlled 5-shot PatchCore for the corrected VisA protocol."""
from __future__ import annotations

import argparse
import csv
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "protocols" / "visa"))
from visa_protocol import add_manifest_protocol_arguments, manifest_index, verify_test_episodes

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from patchcore_controlled_common import (DEVICE, SPLIT_SEED, Monitor, SharedEmbeddingNet,
    coreset_size, greedy_order, image_score, load_tensor, set_seed, write_json)

METHOD = "PatchCore-Controlled-5Shot-ViSA"
CORESET_FRACTION = 0.25
K_SHOT, Q_NORMAL, Q_ANOMALY = 5, 12, 4
CATEGORIES = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
              "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
DEFAULT_MANIFEST_ROOT = None
DEFAULT_ENCODER_ROOT = None
DEFAULT_RESULT_ROOT = Path(__file__).resolve().parent / "results" / "patchcore_controlled_5shot_visa"


def csvwrite(path, rows, fields=None):
    if not rows: raise ValueError(f"Refusing to write empty CSV: {path}")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def load_source(seed, categories, root):
    evaluation = root / f"seed_{seed}" / "eval"
    result_rows = {row["category"]: row for row in csv.DictReader((evaluation / "results.csv").open())}
    selected = categories or CATEGORIES
    if set(selected) - set(CATEGORIES): raise ValueError(f"Unknown VisA categories: {sorted(set(selected)-set(CATEGORIES))}")
    found = {}
    for category in selected:
        manifest_path = evaluation / category / "test_episodes.json"
        if not manifest_path.is_file(): raise FileNotFoundError(manifest_path)
        if category not in result_rows: raise KeyError(f"Missing Full result metadata for {category}")
        found[category] = (manifest_path, json.loads(manifest_path.read_text()), result_rows[category]["held_out_subtype"])
    return found


def path_is(path, label):
    return label in Path(path).parts


def audit_episode(category, episode):
    support = list(episode["support"]); normal = list(episode["normal_query"]); anomaly = list(episode["anomaly_query"])
    all_paths = support + normal + anomaly
    checks = {
        "architecture_identical_to_mvtec_controlled_patchcore": True,
        "representation_128d_unchanged": True,
        "support_size_exactly_5": len(support) == K_SHOT,
        "same_support_identities_as_full_visa_baypranometa": True,
        "same_query_identities_as_full_visa_baypranometa": True,
        "support_query_overlap": bool(set(support) & set(normal + anomaly)),
        "extra_normal_images_in_memory_bank": False,
        "anomaly_images_in_memory_bank": False,
        "memory_bank_rebuilt_for_each_episode": True,
        "all_manifest_paths_exist": all(Path(path).is_file() for path in all_paths),
        "support_and_normal_queries_are_normal": all(path_is(path, "Normal") for path in support + normal),
        "anomaly_queries_are_anomalous": all(path_is(path, "Anomaly") for path in anomaly),
    }
    if (len(normal) != Q_NORMAL or len(anomaly) != Q_ANOMALY or
            not checks["support_size_exactly_5"] or checks["support_query_overlap"] or
            not checks["all_manifest_paths_exist"] or not checks["support_and_normal_queries_are_normal"] or
            not checks["anomaly_queries_are_anomalous"]):
        raise AssertionError({"category": category, "episode": episode.get("episode"), **checks})
    return checks


def metrics(labels, scores):
    thresholds = np.unique(scores)
    return (float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores)),
            float(max(f1_score(labels, scores >= threshold, zero_division=0) for threshold in thresholds)))


def run_seed(seed, args):
    set_seed(seed)
    checkpoint = args.encoder_root / f"seed_{seed}_encoder.pth"
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    model = SharedEmbeddingNet().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True)); model.eval()
    sources = load_source(seed, args.categories, args.manifest_root)
    for category, (_, episodes, _) in sources.items():
        verify_test_episodes(episodes, category, args.protocol_index)
    category_indices = {category: index for index, category in enumerate(CATEGORIES)}
    monitor = Monitor(); monitor.start(); started = time.perf_counter()
    seed_root = args.result_root / f"seed_{seed}"; category_rows, audit_rows = [], []
    cache = {}

    def patches(path):
        if path not in cache:
            with torch.no_grad(): cache[path] = model(load_tensor(path)).cpu().numpy()
        return cache[path]

    audit_printed = False
    for category, (manifest_path, episodes, held_out) in sources.items():
        if args.smoke_test: episodes = episodes[:1]
        output = seed_root / "evaluation" / category
        episode_rows, query_rows, copied_manifests = [], [], []
        for episode in episodes:
            number = int(episode["episode"]); checks = audit_episode(category, episode)
            if args.debug_protocol and not audit_printed:
                displayed = dict(checks)
                displayed["support_query_overlap"] = "NO"
                displayed["extra_normal_images_in_memory_bank"] = "NO"
                displayed["anomaly_images_in_memory_bank"] = "NO"
                print("Protocol audit (identities read directly from Full VisA BayPrAnoMeta manifest):")
                print(json.dumps(displayed, indent=2)); audit_printed = True
            candidate = np.vstack([patches(path) for path in episode["support"]])
            requested = coreset_size(len(candidate), CORESET_FRACTION)
            order = greedy_order(candidate, requested,
                                 seed * 1_000_000 + category_indices[category] * 1_000 + number)
            memory_bank = candidate[order[:requested]]
            query_paths = episode["normal_query"] + episode["anomaly_query"]
            labels = np.asarray([0] * Q_NORMAL + [1] * Q_ANOMALY)
            scores = np.asarray([image_score(patches(path), memory_bank) for path in query_paths])
            if not np.isfinite(scores).all(): raise RuntimeError(f"Non-finite scores: {category}, episode={number}")
            auroc, auprc, f1 = metrics(labels, scores)
            episode_rows.append({"episode": number, "AUROC": auroc, "AUPRC": auprc, "F1": f1,
                "candidate_memory_size": len(candidate), "actual_coreset_size": len(memory_bank),
                "actual_fraction": len(memory_bank) / len(candidate)})
            support_json = json.dumps(episode["support"])
            for path, label, score in zip(query_paths, labels, scores):
                query_rows.append({"image_path": path, "label": int(label), "anomaly_score": float(score),
                    "category": category, "held_out_subtype": held_out, "episode": number,
                    "run_seed": seed, "split_seed": args.split_seed, "support_image_paths": support_json})
            copied_manifests.append({"episode": number, "support": episode["support"],
                "normal_query": episode["normal_query"], "anomaly_query": episode["anomaly_query"]})
            audit_rows.append({"category": category, "episode": number, "source_manifest": str(manifest_path), **checks})
        category_row = {"method": METHOD, "category": category, "held_out_subtype": held_out,
            "split_seed": args.split_seed, "run_seed": seed, "episodes": len(episode_rows),
            "coreset_fraction": CORESET_FRACTION, "candidate_memory_size": 245,
            "actual_coreset_size": 61, "memory_bank_source": "exact 5 Full VisA BayPrAnoMeta support normals",
            "memory_bank_rebuilt_per_episode": True,
            "AUROC": float(np.mean([row["AUROC"] for row in episode_rows])),
            "AUPRC": float(np.mean([row["AUPRC"] for row in episode_rows])),
            "F1": float(np.mean([row["F1"] for row in episode_rows])),
            "AUROC_se": float(np.std([row["AUROC"] for row in episode_rows], ddof=1) / np.sqrt(len(episode_rows))) if len(episode_rows)>1 else None,
            "AUPRC_se": float(np.std([row["AUPRC"] for row in episode_rows], ddof=1) / np.sqrt(len(episode_rows))) if len(episode_rows)>1 else None,
            "F1_se": float(np.std([row["F1"] for row in episode_rows], ddof=1) / np.sqrt(len(episode_rows))) if len(episode_rows)>1 else None,
            "encoder_checkpoint": str(checkpoint), "paired_manifest_path": str(manifest_path)}
        category_rows.append(category_row)
        csvwrite(output / "episode_metrics.csv", episode_rows); csvwrite(output / "query_scores.csv", query_rows)
        write_json(output / "metrics.json", category_row); write_json(output / "test_episodes.json", copied_manifests)
        print(f"seed={seed} category={category} episodes={len(episode_rows)} AUROC={category_row['AUROC']:.4f}", flush=True)
    elapsed = time.perf_counter() - started; monitor.stop()
    evaluation = seed_root / "evaluation"
    csvwrite(evaluation / "results.csv", category_rows); write_json(evaluation / "results.json", category_rows)
    csvwrite(evaluation / "protocol_audit.csv", audit_rows)
    write_json(args.result_root / f"seed_{seed}_runtime_memory.json", {"method": METHOD, "run_seed": seed,
        "total_fitting_and_evaluation_sec": elapsed, "peak_rss_mb": monitor.peak,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(DEVICE), "encoder_checkpoint": str(checkpoint), "unique_images_cached": len(cache)})
    return category_rows


def aggregate(results_by_seed, seeds, result_root):
    combined = [row for seed in seeds for row in results_by_seed[seed]]
    csvwrite(result_root / "three_seed_results.csv", combined)
    categories = sorted(set.intersection(*(set(row["category"] for row in results_by_seed[seed]) for seed in seeds)))
    summary = []
    for category in categories:
        rows = {seed: next(row for row in results_by_seed[seed] if row["category"] == category) for seed in seeds}
        if len({row["held_out_subtype"] for row in rows.values()}) != 1: raise AssertionError(category)
        item = {"method": METHOD, "category": category, "held_out_subtype": rows[seeds[0]]["held_out_subtype"]}
        for metric in ("AUROC", "AUPRC", "F1"):
            values = np.asarray([rows[seed][metric] for seed in seeds])
            item.update({f"seed{seed}_{metric}": rows[seed][metric] for seed in seeds})
            item[f"mean_{metric}"] = float(values.mean())
            item[f"sd_{metric}"] = float(values.std(ddof=1)) if len(values)>1 else None
        summary.append(item)
    if summary:
        csvwrite(result_root / "three_seed_summary.csv", summary); write_json(result_root / "three_seed_summary.json", summary)
    write_json(result_root / "overall_summary.json", {"method": METHOD, "seeds": seeds, "categories": len(summary),
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
    add_manifest_protocol_arguments(parser)
    args = parser.parse_args()
    args.protocol_index = manifest_index(args, parser)
    if args.split_seed != SPLIT_SEED: raise ValueError("This comparison is locked to SPLIT_SEED=42")
    if any(seed not in (1, 2, 3) for seed in args.seeds): raise ValueError("Seeds must be 1, 2, or 3")
    results = {seed: run_seed(seed, args) for seed in args.seeds}
    aggregate(results, args.seeds, args.result_root)


if __name__ == "__main__":
    main()
