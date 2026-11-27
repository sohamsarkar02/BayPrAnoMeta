#!/usr/bin/env python3
"""Test-time support sensitivity using the saved corrected VisA episodes."""
import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
RUN = PROJECT / "runs/baypranometa_visa_three_seed_fixed_split42_25ep_30epochs_100test/revised_clean_official_2cls_highshot"
OUTPUT = PROJECT / "results/support_sensitivity_visa"
SIZES = (1, 3, 7)
SEEDS = (1, 2, 3)
CATEGORIES = (
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
)
SPLIT_CSV = "split_csv/2cls_highshot.csv"
OFFICIAL_SPLIT = "2cls_highshot"
TEST_GROUP = "official_2cls_highshot_test_anomalies"
EXPECTED_K5 = (0.673758681, 0.669123264, 0.673949653)
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


def preflight(main, root, subset_file, verify_reference):
    split = read_json(RUN / "split_manifest.json")
    held = split["held_out_subtypes"]
    require(split["split_seed"] == SPLIT_SEED and set(held) == set(CATEGORIES)
            and set(held.values()) == {TEST_GROUP}
            and sorted(split["run_seeds"]) == list(SEEDS)
            and set(main.discover_categories(root, SPLIT_CSV, subset_file=subset_file)) == set(CATEGORIES),
            "Corrected split manifest differs from the requested VisA protocol.")
    seed_macros = []
    for seed in SEEDS:
        results_path = RUN / f"seed_{seed}/eval/results.csv"
        with results_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == len(CATEGORIES) and {r["category"] for r in rows} == set(CATEGORIES)
                and all(r["method"] == "BayPrAnoMeta" and int(r["seed"]) == seed
                        and int(r["split_seed"]) == SPLIT_SEED
                        and r["held_out_subtype"] == TEST_GROUP
                        and int(r["num_test_episodes"]) == EPISODES for r in rows),
                f"Unexpected corrected K=5 results in {results_path}.")
        seed_macros.append(statistics.mean(float(r["auc_mean"]) for r in rows))
    require(not verify_reference or (tuple(round(x, 9) for x in seed_macros) == EXPECTED_K5
            and round(statistics.mean(seed_macros), 9) == 0.672277199
            and round(statistics.stdev(seed_macros), 9) == 0.002733057),
            "Saved corrected VisA K=5 AUROCs do not match the reference.")
    return held


def prepare(main, dataset, held, subset_file):
    for category in CATEGORIES:
        data = main.load_client_dataset_visa(
            dataset, category, seed=SPLIT_SEED, validation_fraction=0.30,
            split_csv=SPLIT_CSV, subset_file=subset_file, audit_dir=None, print_audit=False,
        )
        require(data is not None and data["held_out_subtype"] == held[category],
                f"Corrected official test anomaly group differs for {category}.")
        normal_pool = {str(Path(p).resolve()) for p in data["test"]["normal"]}
        anomaly_pool = {str(Path(p).resolve()) for p in data["test"]["anomaly"]}
        require(len(normal_pool) >= max(SIZES) + QN and len(anomaly_pool) >= QA,
                f"{category}: insufficient official test images for the paired K=7 design "
                f"({len(normal_pool)} normals, {len(anomaly_pool)} anomalies).")
        for seed in SEEDS:
            audit_path = RUN / f"seed_{seed}/eval/split_audit" / f"{category}.json"
            require(read_json(audit_path) == data["audit"],
                    f"Reconstructed split audit differs from {audit_path}.")
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
                    "official_test_anomaly_group": held[category], "split_seed": SPLIT_SEED,
                    "official_split": OFFICIAL_SPLIT,
                    "all_images_from_official_test_pool": (
                        set(support + added + normal) <= normal_pool and set(anomaly) <= anomaly_pool
                    ),
                    "support": support + added, "original_K5_support": episode["support"],
                    "added_K7_support": added,
                    "normal_query": normal, "anomaly_query": anomaly,
                    "original_K5_normal_query": episode["normal_query"],
                    "original_K5_anomaly_query": episode["anomaly_query"],
                }


def score_episode(main, model, support, paths, debug_protocol=True):
    """Adapt the encoder and score queries with the VisA Bayesian evaluator protocol."""
    load_stack, torch, higher = main.load_stack, main.torch, main.higher
    INNER_LR, INNER_STEPS = main.INNER_LR, main.INNER_STEPS
    niw_posterior, log_student_t = main.niw_posterior, main.log_student_t
    reference_log_probability = main.reference_log_probability
    xs, xq = load_stack(support), load_stack(paths); inner_opt = torch.optim.SGD(model.parameters(), lr=INNER_LR)
    with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True, track_higher_grads=False) as (fmodel, diffopt):
        for _ in range(INNER_STEPS):
            z0 = fmodel(xs); mu0, sigma0, dof0 = niw_posterior(z0, validate=debug_protocol)
            diffopt.step(-log_student_t(z0, mu0, sigma0, dof0).mean())
        with torch.no_grad():
            zsup = fmodel(xs); mu, sigma, dof = niw_posterior(zsup, validate=debug_protocol); zq = fmodel(xq)
            logp_a, logp_n = reference_log_probability(zq), log_student_t(zq, mu, sigma, dof)
            scores, z = (logp_a - logp_n).cpu().numpy(), zq.cpu().numpy()
            p1, p0 = logp_a.cpu().numpy(), logp_n.cpu().numpy()
    return scores, p1, p0


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
        "all_images_from_official_test_pool": all(r["all_images_from_official_test_pool"] for r in rows),
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
    model = main.EmbeddingNet().to(main.DEVICE)
    model.load_state_dict(main.torch.load(checkpoint, map_location=main.DEVICE))
    model.eval()
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
            scores, p1, p0 = score_episode(main, model, row["support"], queries)
            require(all(main.np.isfinite(values).all() for values in (scores, p1, p0)),
                    f"Non-finite scores or log densities for seed {seed}, {category}, "
                    f"episode {row['episode']}, K={k}.")
            values.append(main.compute_episode_metrics(scores, labels))
            diagnostics.extend(
                {"episode": row["episode"], "image_path": path, "label": label,
                 "anomaly_score": float(score), "logp0": float(n),
                 "logp1": float(a), "K": k}
                for path, label, score, a, n in zip(queries, labels, scores, p1, p0)
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
        "official_split": OFFICIAL_SPLIT, "split_source": SPLIT_CSV,
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


def combine(episodes, checkpoints, evaluator_digest):
    results, manifests = [], {}
    for k in SIZES:
        for seed in SEEDS:
            directory = OUTPUT / f"K_{k}/seed_{seed}"
            meta = read_json(directory / "metadata.json")
            checkpoint, digest = checkpoints[seed]
            require(meta["checkpoint"] == str(checkpoint)
                    and meta["checkpoint_sha256"] == digest
                    and meta["corrected_K5_evaluator_sha256"] == evaluator_digest,
                    f"Incompatible checkpoint or evaluator in {directory}.")
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
    global RUN, OUTPUT, CHECKPOINT_ROOT, SPLIT_CSV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", "--dataset-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--support-size", type=int, choices=SIZES, required=True)
    parser.add_argument("--episode-plan", dest="run_root", type=Path, default=RUN, help="Protocol subdirectory of the fresh Full evaluation output.")
    parser.add_argument("--checkpoint-root", type=Path, help="Protocol subdirectory of the Full training output. Defaults to --episode-plan.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--split-csv", default=SPLIT_CSV); parser.add_argument("--subset-file", help='Paper image allowlist. Required for exact reproduction of the paper experiments.')
    parser.add_argument("--verify-paper-auroc", action="store_true", help="Also check K=5 AUROC against the published run.")
    args = parser.parse_args()
    SPLIT_CSV = args.split_csv
    RUN, OUTPUT = args.run_root, args.output_root
    CHECKPOINT_ROOT = args.checkpoint_root or RUN
    k = args.support_size
    selected_output = OUTPUT / f"K_{k}"
    require(not selected_output.exists(), f"Output directory already exists: {selected_output}")
    # Import the corrected evaluator with project bytecode writes disabled.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(PROJECT))
    import evaluate_baypranometa_visa_three_seeds as corrected

    require((corrected.SPLIT_SEED, corrected.TEST_EPISODES, corrected.Q_N, corrected.Q_A,
             corrected.INNER_STEPS, corrected.INNER_LR, corrected.PROTOCOL)
            == (42, 100, 12, 4, 1, 5e-4, "revised_clean_official_2cls_highshot"),
            "Corrected evaluator constants differ from the requested protocol.")
    held = preflight(corrected, args.dataset_root, args.subset_file, args.verify_paper_auroc)
    episodes = list(prepare(corrected, args.dataset_root, held, args.subset_file))
    require(len(episodes) == len(SEEDS) * len(CATEGORIES) * EPISODES, "Incomplete manifest.")
    checkpoints = {}
    for seed in SEEDS:
        folder = RUN / f"seed_{seed}"
        expected = CHECKPOINT_ROOT / f"seed_{seed}" / "checkpoints/final_model.pth"
        meta = read_json(folder / "eval/metadata.json")
        require(Path(meta["checkpoint_path"]).resolve(strict=True) == expected.resolve(strict=True)
                and meta["run_seed"] == seed and meta["split_seed"] == SPLIT_SEED
                and meta["protocol"] == corrected.PROTOCOL
                and meta["split_source"] == SPLIT_CSV and meta["test_episodes"] == EPISODES,
                f"Corrected evaluation checkpoint metadata differs for seed {seed}.")
        checkpoints[seed] = (expected, sha256(expected))
    for episode in episodes:
        audit_group([{**episode, "K": size, "support": episode["support"][:size]} for size in SIZES])
    selected_output.mkdir(parents=True, exist_ok=False)
    for seed, (checkpoint, digest) in checkpoints.items():
        evaluate_seed(corrected, seed, checkpoint, digest, episodes, k)
        require(sha256(checkpoint) == digest, f"Checkpoint changed: {checkpoint}")
    write_json(selected_output / "complete.json", {"K": k, "seeds": list(SEEDS)})
    if not all((OUTPUT / f"K_{size}/complete.json").is_file() for size in SIZES):
        print("Corrected K=5 evaluator reused as implementation basis: YES")
        print(f"K={k} evaluation complete: YES")
        return
    final = combine(episodes, checkpoints, sha256(Path(corrected.__file__)))

    print("\n".join([
        "Corrected K=5 evaluator reused as implementation basis: YES",
        *(f"K={k} evaluation complete: YES" for k in SIZES), "",
        "All 12 categories present: YES", "All 3 seeds present: YES",
        "100 episodes per category/seed/K: YES", "Original K=5 queries preserved: YES",
        "Original K=5 support preserved as S5: YES", "Queries identical across K=1,3,7: YES",
        "S1 subset S3: YES", "S3 subset S5: YES", "S5 subset S7: YES",
        "Support-query overlap detected: NO",
        "All episode images respect official 2cls_highshot test pools: YES",
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
