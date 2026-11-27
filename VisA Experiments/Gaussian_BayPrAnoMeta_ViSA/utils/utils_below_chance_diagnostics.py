"""Generic, non-corrective diagnostics for below-chance anomaly results."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)),
            "median": float(np.median(values)), "min": float(np.min(values)), "max": float(np.max(values))}


def _episode_stats(values):
    values = np.asarray(values, dtype=float)
    return {**_stats(values), "p5": float(np.percentile(values, 5)), "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)), "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)), "fraction_below_0_50": float(np.mean(values < 0.50))}


def _write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _posterior_report(matrices, kappa_n, dof):
    episodes = []
    scale = kappa_n * dof / (kappa_n + 1)
    for episode_id, sigma in enumerate(matrices):
        sigma_eigs = np.linalg.eigvalsh(sigma)
        lambda_eigs = np.linalg.eigvalsh(sigma * scale)
        episodes.append({"episode_id": episode_id, "lambda_n_eigenvalues": lambda_eigs.tolist(),
                         "sigma_n_eigenvalues": sigma_eigs.tolist(), "sigma_min_eigenvalue": float(sigma_eigs.min()),
                         "sigma_max_eigenvalue": float(sigma_eigs.max()),
                         "sigma_condition_number": float(sigma_eigs.max() / sigma_eigs.min())})
    return {"kappa_n": float(kappa_n), "degrees_of_freedom": float(dof), "episodes": episodes,
            "all_sigma_positive_definite": bool(all(item["sigma_min_eigenvalue"] > 0 for item in episodes))}


def save_below_chance_diagnostics(*, out_dir, method, seed, category, held_out_subtype, records,
                                  episode_aucs, reversed_episode_aucs, protocol_report,
                                  posterior_matrices=None, kappa_n=None, nu_n=None, dof=None, split_seed=None):
    """Persist diagnostic evidence without altering scores, labels, or predictions."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([record["label"] for record in records], dtype=int)
    scores = np.asarray([record["final_score"] for record in records], dtype=float)
    has_log_probabilities = all("logp_anomaly" in record and "logp_normal" in record for record in records)
    logp_a = np.asarray([record["logp_anomaly"] for record in records], dtype=float) if has_log_probabilities else None
    logp_n = np.asarray([record["logp_normal"] for record in records], dtype=float) if has_log_probabilities else None
    finite = bool(np.isfinite(scores).all() and (not has_log_probabilities or (np.isfinite(logp_a).all() and np.isfinite(logp_n).all())))
    original = float(np.mean(episode_aucs))
    reversed_score = float(np.mean(reversed_episode_aucs))
    pooled_original = float(roc_auc_score(labels, scores))
    pooled_reversed = float(roc_auc_score(labels, -scores))
    severity = "AUROC_LT_0_30" if original < 0.30 else "AUROC_LT_0_40" if original < 0.40 else "AUROC_LT_0_50"
    violations = protocol_report["episode_overlap_violations"]
    if violations or not protocol_report["train_test_normal_disjoint"] or not protocol_report["held_out_absent_from_training"]:
        diagnosis = "DATA_PROTOCOL_ERROR"
    elif not finite:
        diagnosis = "NUMERICAL_INSTABILITY"
    elif reversed_score - original >= 0.20:
        # A higher reversed AUROC is diagnostic evidence only.  The score and
        # labels are not changed, and no sign error is asserted automatically.
        diagnosis = "INCONCLUSIVE_REQUIRES_MANUAL_INSPECTION"
    elif np.std(episode_aucs) >= 0.20:
        diagnosis = "HIGH_EPISODE_VARIANCE"
    else:
        diagnosis = "GENUINE_BELOW_CHANCE_PERFORMANCE"

    normal, anomaly = labels == 0, labels == 1
    summary = {
        "method": method, "run_seed": seed, "split_seed": split_seed, "category": category, "held_out_subtype": held_out_subtype,
        "trigger": "reported mean episode AUROC < 0.50", "severity": severity,
        "reported_mean_episode_AUROC": original, "reported_mean_episode_reversed_score_AUROC": reversed_score,
        "original_AUROC": pooled_original, "reversed_score_AUROC": pooled_reversed,
        "pooled_repeated_query_AUROC_diagnostic_only": pooled_original,
        "pooled_reversed_score_AUROC_diagnostic_only": pooled_reversed, "primary_score_was_not_modified": True,
        "label_convention": {"normal": 0, "anomaly": 1, "labels_match_expected_qn_qa_order": protocol_report["labels_match_expected"]},
        "score_direction": "higher log p1(z) - log p0(z | adapted support) means more anomalous",
        "protocol": protocol_report, "posterior_recomputed_after_adaptation": True,
        "finite_log_probabilities_and_scores": finite, "normal_score": _stats(scores[normal]),
        "anomaly_score": _stats(scores[anomaly]), "episode_AUROC": _episode_stats(episode_aucs),
        "diagnosis": diagnosis,
        "note": "Repeated episode queries are preserved for traceability only and are not independent observations for McNemar or inferential image-level tests.",
    }
    if posterior_matrices is not None:
        summary["posterior"] = {"nu_n": float(nu_n), **_posterior_report(posterior_matrices, kappa_n, dof)}
    if has_log_probabilities:
        summary.update({"normal_logp_anomaly": _stats(logp_a[normal]), "anomaly_logp_anomaly": _stats(logp_a[anomaly]),
                        "normal_logp_normal": _stats(logp_n[normal]), "anomaly_logp_normal": _stats(logp_n[anomaly])})
    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    _write_csv(out_dir / "query_scores.csv", records)
    _write_csv(out_dir / "below_chance_diagnostics.csv", records)

    plt.figure(figsize=(6, 4)); plt.hist(scores[normal], bins=30, alpha=.7, label="Normal"); plt.hist(scores[anomaly], bins=30, alpha=.7, label="Anomaly")
    plt.xlabel("Original anomaly score"); plt.ylabel("Count"); plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "score_distribution.png"); plt.close()
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(5, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.tight_layout(); plt.savefig(out_dir / "roc_curve.png"); plt.close()
    if has_log_probabilities:
        plt.figure(figsize=(6, 4)); plt.hist(logp_a, bins=30, alpha=.55, label="log p1"); plt.hist(logp_n, bins=30, alpha=.55, label="log p0"); plt.hist(scores, bins=30, alpha=.55, label="score")
        plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "log_probability_distribution.png"); plt.close()
    plt.figure(figsize=(6, 4)); plt.hist(episode_aucs, bins=min(20, len(episode_aucs))); plt.axvline(.5, color="red", linestyle="--")
    plt.xlabel("Episode AUROC"); plt.ylabel("Episodes"); plt.tight_layout(); plt.savefig(out_dir / "episode_auroc_distribution.png"); plt.close()
    return summary
