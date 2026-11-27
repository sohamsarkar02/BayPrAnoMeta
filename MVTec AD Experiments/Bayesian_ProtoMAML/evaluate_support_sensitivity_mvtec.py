#!/usr/bin/env python3
"""Test-time support sensitivity using the saved corrected MVTec episodes."""
import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
RUN = PROJECT / "runs/baypranometa_avgpool_three_seed/revised_clean"
OUTPUT = PROJECT / "results/support_sensitivity_mvtec"
SIZES = (1, 3, 7)
SEEDS = (1, 2, 3)
CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "transistor", "wood", "zipper",
)
SPLIT_SEED, EPISODES, QN, QA = 42, 100, 12, 4
METRICS = ("AUROC", "AUPRC", "oracle_F1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def write_csv(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def preflight(root):
    split = read_json(RUN / "split_manifest.json")
    held = split["held_out_subtypes"]
    require(split["split_seed"] == SPLIT_SEED and set(held) == set(CATEGORIES)
            and sorted(split["run_seeds"]) == list(SEEDS),
            "Corrected split manifest differs from the requested protocol.")
    normals = {
        c: sorted(str(p.resolve()) for p in (root / c / "test/good").glob("*.png")
                  if p.is_file())
        for c in CATEGORIES
    }
    needed = max(SIZES) + QN
    short = [f"{c}: {len(paths)} available, {needed} required"
             for c, paths in normals.items() if len(set(paths)) < needed]
    require(not short, "Impossible distinct-normal budget: " + "; ".join(short))
    return held


def prepare(main, dataset, held):
    for category in CATEGORIES:
        data = main.load_client_dataset_leave1out(
            dataset / category, seed=SPLIT_SEED, protocol=main.PROTOCOL,
            validation_fraction=0.30,
        )
        require(data is not None and data["held_out_subtype"] == held[category],
                f"Corrected held-out subtype differs for {category}.")
        normal_pool = {str(Path(p).resolve()) for p in data["test"]["normal"]}
        anomaly_pool = {str(Path(p).resolve()) for p in data["test"]["anomaly"]}
        for seed in SEEDS:
            manifest = RUN / f"seed_{seed}/eval" / category / "test_episodes.json"
            portable = manifest.with_name("episode_plan.json")
            saved = read_json(portable if portable.is_file() else manifest)
            if portable.is_file():
                for episode in saved:
                    for key in ("support", "normal_query", "anomaly_query"):
                        episode[key] = [str((Path(dataset) / path).resolve()) for path in episode[key]]
            require([r["episode"] for r in saved] == list(range(EPISODES)),
                    f"Unexpected episode IDs in {manifest}.")
            for episode in saved:
                support, normal, anomaly = [
                    list(episode[key])
                    for key in ("support", "normal_query", "anomaly_query")
                ]
                paths = support + normal + anomaly
                require(
                    [len(support), len(normal), len(anomaly)] == [5, QN, QA]
                    and len(set(paths)) == len(paths)
                    and set(support + normal) <= normal_pool
                    and set(anomaly) <= anomaly_pool,
                    f"Invalid corrected episode in {manifest}.",
                )
                candidates = sorted(normal_pool.difference(support, normal, anomaly))
                episode_id = episode["episode"]
                require(len(candidates) >= 2,
                        f"Cannot extend seed {seed}, {category}, episode {episode_id} to K=7.")
                extension_seed = f"{SPLIT_SEED}:{seed}:{category}:{episode_id}"
                added = random.Random(extension_seed).sample(candidates, 2)
                yield {
                    "seed": seed, "category": category, "episode": episode_id,
                    "held_out_subtype": held[category], "split_seed": SPLIT_SEED,
                    "support": support + added, "original_K5_support": episode["support"],
                    "added_K7_support": added,
                    "normal_query": normal, "anomaly_query": anomaly,
                    "original_K5_normal_query": episode["normal_query"],
                    "original_K5_anomaly_query": episode["anomaly_query"],
                }


def audit_group(rows):
    one, three, seven = rows
    normal, anomaly = seven["normal_query"], seven["anomaly_query"]
    five = seven["original_K5_support"]
    added = seven["added_K7_support"]
    overlap = {
        f"support_query_overlap_K{r['K']}": len(set(r["support"]) & set(normal + anomaly))
        for r in rows
    }
    checks = {
        "original_K5_support_preserved": seven["support"][:5] == five,
        "queries_match_original_K5": all(
            r["normal_query"] == seven["original_K5_normal_query"]
            and r["anomaly_query"] == seven["original_K5_anomaly_query"] for r in rows
        ),
        "queries_match_across_K1_K3_K7": all(
            r["normal_query"] == normal and r["anomaly_query"] == anomaly for r in rows
        ),
        "S1_subset_S3": one["support"] == three["support"][:1],
        "S3_subset_S5": three["support"] == five[:3],
        "S5_subset_S7": set(five) < set(seven["support"]),
        "K7_extension_feasible": (
            len(set(seven["support"])) == 7 and len(added) == 2
            and seven["support"][5:] == added
        ),
    }
    require(all(checks.values()) and not any(overlap.values()), "Cross-K audit failed.")
    return {
        "seed": seven["seed"], "category": seven["category"], "episode": seven["episode"],
        **checks, **overlap, "K7_extension_count": len(added),
    }


def evaluate_seed(main, seed, checkpoint, digest, episodes, k):
    main.set_seed(seed)
    model = main.load_evaluation_model(checkpoint)
    labels = [0] * QN + [1] * QA
    results, manifest = [], []
    for category in CATEGORIES:
        values, diagnostics = [], []
        selected = [r for r in episodes if r["seed"] == seed and r["category"] == category]
        require(len(selected) == EPISODES, f"Missing episodes for seed {seed}, {category}.")
        for episode in selected:
            row = {**episode, "K": k, "support": episode["support"][:k]}
            manifest.append({**row, "RUN_SEED": seed, "SPLIT_SEED": SPLIT_SEED})
            queries = row["normal_query"] + row["anomaly_query"]
            scores, _, p1, p0, _ = main.score_episode(model, row["support"], queries, True)
            require(all(main.np.isfinite(values).all() for values in (scores, p1, p0)),
                    f"Non-finite scores or log densities for seed {seed}, {category}, "
                    f"episode {row['episode']}, K={k}.")
            values.append(main.compute_episode_metrics(scores, labels))
            records = main.query_records(
                seed, SPLIT_SEED, category, row["held_out_subtype"], row["episode"],
                queries, labels, scores, p1, p0,
            )
            diagnostics.extend(
                {"episode": record["episode_id"], "image_path": record["image_path"],
                 "label": record["label"], "anomaly_score": record["final_score"],
                 "logp0": record["logp_normal"], "logp1": record["logp_anomaly"], "K": k}
                for record in records
            )
        category_dir = OUTPUT / f"K_{k}/seed_{seed}" / category
        category_dir.mkdir(parents=True)
        write_csv(category_dir / "query_values.csv", diagnostics)
        results.append({
            "K": k, "seed": seed, "category": category,
            **{m: statistics.mean(float(x) for x in v)
               for m, v in zip(METRICS, zip(*values))},
        })
    from utils.utils_niw import KAPPA0, LAMBDA0_SCALE, NU0_OFFSET
    out = OUTPUT / f"K_{k}/seed_{seed}"
    require(len(manifest) == len(CATEGORIES) * EPISODES, "Incomplete K manifest.")
    write_csv(out / "results.csv", results)
    write_json(out / "episode_manifest.json", manifest)
    write_json(out / "metadata.json", {
        "K": k, "QN": QN, "QA": QA, "test_episodes_per_category": EPISODES,
        "RUN_SEED": seed, "SPLIT_SEED": SPLIT_SEED, "device": str(main.DEVICE),
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
        "corrected_K5_evaluator": str(Path(main.__file__).resolve()),
        "corrected_K5_evaluator_sha256": sha256(Path(main.__file__)),
        "inner_steps": main.INNER_STEPS, "inner_lr": main.INNER_LR,
        "NIW": {"mu0": "zero vector", "kappa0": KAPPA0, "nu0": f"d + {NU0_OFFSET}",
                "Lambda0": f"{LAMBDA0_SCALE} * I", "predictive_dof": NU0_OFFSET + k + 1},
        "anomaly_reference": {"family": "Student-t", "location": "zero vector",
                              "scale": "100 * I", "dof": 2},
        "episode_design": "Use saved corrected K=5 query and ordered support manifests "
                          "as the pairing reference. Evaluate only K=1,3,7.",
        "K5_reevaluated": False,
        "support_extension_rule": "Sort eligible test-normal paths after excluding both "
                                  "query classes and saved K=5 support. Draw two without "
                                  "replacement with random.Random seeded by the string "
                                  "SPLIT_SEED:RUN_SEED:category:episode_id; append in draw order.",
        "oracle_F1": "Corrected evaluator's 200-threshold test-score sweep.",
        "validation_selected_F1": None,
        "validation_selected_F1_reason": "Corrected evaluator lacks validation calibration.",
    })


def aggregate(rows):
    macros = []
    final = []
    for k in SIZES:
        per_k = []
        for seed in SEEDS:
            selected = [row for row in rows if row["K"] == k and row["seed"] == seed]
            require(len(selected) == len(CATEGORIES)
                    and {r["category"] for r in selected} == set(CATEGORIES),
                    f"Incomplete results for K={k}, seed={seed}.")
            per_k.append({
                "K": k, "seed": seed,
                **{f"macro_{m}": statistics.mean(row[m] for row in selected) for m in METRICS},
            })
        macros.extend(per_k)
        summary = {"K": k}
        for metric in METRICS:
            values = [row[f"macro_{metric}"] for row in per_k]
            summary[f"mean_macro_{metric}"] = statistics.mean(values)
            summary[f"sample_SD_macro_{metric}"] = statistics.stdev(values)
        final.append(summary)
    return macros, final


def combine(episodes):
    results, manifests = [], {}
    for k in SIZES:
        for seed in SEEDS:
            directory = OUTPUT / f"K_{k}/seed_{seed}"
            with (directory / "results.csv").open(newline="") as handle:
                for row in csv.DictReader(handle):
                    require(int(row["K"]) == k and int(row["seed"]) == seed,
                            f"Unexpected result identity in {directory}.")
                    results.append({**row, "K": k, "seed": seed,
                                    **{m: float(row[m]) for m in METRICS}})
            for row in read_json(directory / "episode_manifest.json"):
                key = (k, seed, row["category"], row["episode"])
                require(key not in manifests, f"Duplicate saved episode: {key}")
                manifests[key] = row
    require(len(manifests) == len(episodes) * len(SIZES), "Incomplete saved manifests.")
    audit = []
    for episode in episodes:
        group = []
        for k in SIZES:
            key = (k, episode["seed"], episode["category"], episode["episode"])
            saved = manifests[key]
            expected = {**episode, "K": k, "support": episode["support"][:k],
                        "RUN_SEED": episode["seed"], "SPLIT_SEED": SPLIT_SEED}
            require(saved == expected, f"Saved manifest differs from paired design: {key}")
            group.append(saved)
        audit.append(audit_group(group))
    macros, final = aggregate(results)
    write_csv(OUTPUT / "cross_k_audit.csv", audit)
    write_csv(OUTPUT / "support_sensitivity_summary.csv", results)
    write_csv(OUTPUT / "support_sensitivity_macro_summary.csv", macros)
    write_csv(OUTPUT / "support_sensitivity_three_seed_summary.csv", final)
    return final


def main():
    global RUN, OUTPUT, CHECKPOINT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", "--dataset-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--support-size", type=int, choices=SIZES, required=True)
    parser.add_argument("--episode-plan", dest="run_root", type=Path, default=RUN, help="Protocol subdirectory of the fresh Full evaluation output.")
    parser.add_argument("--checkpoint-root", type=Path, help="Protocol subdirectory of the Full training output. Defaults to --episode-plan.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    RUN, OUTPUT = args.run_root, args.output_root
    CHECKPOINT_ROOT = args.checkpoint_root or RUN
    k = args.support_size
    selected_output = OUTPUT / f"K_{k}"
    require(not selected_output.exists(), f"Output directory already exists: {selected_output}")
    held = preflight(args.dataset_root)
    # Import only after the feasibility check, with project bytecode writes disabled.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(PROJECT))
    import evaluate_bayesian_protomaml_three_seeds_reduced as corrected

    require((corrected.SPLIT_SEED, corrected.TEST_EPISODES, corrected.Q_N, corrected.Q_A,
             corrected.INNER_STEPS, corrected.INNER_LR, corrected.PROTOCOL)
            == (42, 100, 12, 4, 1, 5e-4, "revised_clean"),
            "Corrected evaluator constants differ from the requested protocol.")
    episodes = list(prepare(corrected, args.dataset_root, held))
    require(len(episodes) == len(SEEDS) * len(CATEGORIES) * EPISODES, "Incomplete manifest.")
    checkpoints = {}
    for seed in SEEDS:
        folder = RUN / f"seed_{seed}"
        expected = CHECKPOINT_ROOT / f"seed_{seed}" / "checkpoints/final_model.pth"
        meta = read_json(folder / "eval/metadata.json")
        require(Path(meta["checkpoint_path"]).resolve(strict=True) == expected.resolve(strict=True)
                and meta["run_seed"] == seed and meta["split_seed"] == SPLIT_SEED
                and meta["protocol"] == corrected.PROTOCOL,
                f"Corrected evaluation checkpoint metadata differs for seed {seed}.")
        checkpoints[seed] = (expected, sha256(expected))
    selected_output.mkdir(parents=True, exist_ok=False)
    for seed, (checkpoint, digest) in checkpoints.items():
        evaluate_seed(corrected, seed, checkpoint, digest, episodes, k)
        require(sha256(checkpoint) == digest, f"Checkpoint changed: {checkpoint}")
    write_json(selected_output / "complete.json", {"K": k, "seeds": list(SEEDS)})
    if not all((OUTPUT / f"K_{size}/complete.json").is_file() for size in SIZES):
        print("Corrected K=5 evaluator reused as implementation basis: YES")
        print(f"K={k} evaluation complete: YES")
        return
    final = combine(episodes)

    print("\n".join([
        "Corrected K=5 evaluator reused as implementation basis: YES",
        *(f"K={k} evaluation complete: YES" for k in SIZES), "",
        "All 14 categories present: YES", "All 3 seeds present: YES",
        "100 episodes per category/seed/K: YES", "Original K=5 queries preserved: YES",
        "Original K=5 support preserved as S5: YES", "Queries identical across K=1,3,7: YES",
        "S1 subset S3: YES", "S3 subset S5: YES", "S5 subset S7: YES",
        "Support-query overlap detected: NO",
        "Corrected checkpoints reused: YES", "Any retraining performed: NO",
        "Any existing main evaluation files modified: NO", "",
        "K,mean_macro_AUROC,sample_SD_macro_AUROC",
        *(f"{r['K']},{r['mean_macro_AUROC']:.9f},{r['sample_SD_macro_AUROC']:.9f}"
          for r in final),
    ]))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
