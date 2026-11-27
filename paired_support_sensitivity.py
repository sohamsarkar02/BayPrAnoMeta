"""Compare support sizes using the three-seed category means as paired units."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def read_scores(sensitivity_csv, full_root):
    scores = {}

    def add(size, seed, category, value):
        key = (size, seed, category)
        score = float(value)
        if key in scores or seed not in (1, 2, 3) or not 0 <= score <= 1:
            raise ValueError(f"Invalid or duplicate category result: {key}")
        scores[key] = score

    with sensitivity_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            add(int(row["K"]), int(row["seed"]), row["category"], row["AUROC"])
    for seed in (1, 2, 3):
        with (full_root / f"seed_{seed}" / "eval" / "results.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                add(5, seed, row["category"], row["auc_mean"])
    return scores


def compare(scores, baseline, candidate):
    categories = sorted({category for size, _, category in scores if size in (baseline, candidate)})
    if not categories:
        raise ValueError("No categories available for the requested comparison.")
    differences = []
    for category in categories:
        try:
            differences.append(np.mean([scores[candidate, s, category] - scores[baseline, s, category]
                                        for s in (1, 2, 3)]))
        except KeyError as error:
            raise ValueError(f"Missing paired category result: {error}") from error
    differences = np.asarray(differences)
    rng = np.random.default_rng(42)
    bootstrap = rng.choice(differences, size=(100_000, len(categories)), replace=True).mean(axis=1)
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    result = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
    return dict(baseline=baseline, candidate=candidate, categories=len(categories),
                mean_difference=float(differences.mean()), ci_low=float(low), ci_high=float(high),
                wilcoxon_statistic=float(result.statistic), wilcoxon_p=float(result.pvalue))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity-csv", type=Path, required=True)
    parser.add_argument("--full-run-root", type=Path, required=True,
                        help="Protocol subdirectory of the Full evaluation output for K=5.")
    parser.add_argument("--compare", type=int, nargs=2, action="append", required=True,
                        metavar=("BASELINE", "CANDIDATE"), choices=(1, 3, 5, 7))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = read_scores(args.sensitivity_csv, args.full_run_root)
    rows = [compare(scores, baseline, candidate) for baseline, candidate in args.compare]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
