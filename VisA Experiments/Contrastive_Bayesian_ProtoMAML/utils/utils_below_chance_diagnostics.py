import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _stats(x):
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "std": float(x.std()), "min": float(x.min()), "max": float(x.max())}


def save_below_chance_diagnostics(out_dir, records, episode_aucs, reverse_aucs, protocol, posterior, method, run_seed, split_seed, category, held_out):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    labels, scores = np.asarray([r["label"] for r in records]), np.asarray([r["final_score"] for r in records])
    p0, p1 = np.asarray([r["logp_normal"] for r in records]), np.asarray([r["logp_anomaly"] for r in records])
    normal, anomaly = labels == 0, labels == 1
    finite = bool(np.isfinite(scores).all() and np.isfinite(p0).all() and np.isfinite(p1).all())
    if protocol["episode_overlap_violations"] or not protocol["train_test_normal_disjoint"]: diagnosis = "DATA_PROTOCOL_ERROR"
    elif not finite: diagnosis = "NUMERICAL_INSTABILITY"
    elif np.std(episode_aucs) >= .20: diagnosis = "HIGH_EPISODE_VARIANCE"
    else: diagnosis = "GENUINE_BELOW_CHANCE_PERFORMANCE"
    summary = {"method": method, "run_seed": run_seed, "split_seed": split_seed, "category": category, "held_out_subtype": held_out,
               "reported_mean_episode_AUROC": float(np.mean(episode_aucs)), "reported_mean_episode_reversed_score_AUROC": float(np.mean(reverse_aucs)),
               "original_AUROC": float(roc_auc_score(labels, scores)), "reversed_score_AUROC": float(roc_auc_score(labels, -scores)),
               "primary_score_was_not_modified": True, "label_convention": {"normal": 0, "anomaly": 1, "verified": protocol["labels_match_expected"]},
               "score_direction": "higher log p1(z) - log p0(z | adapted support) means more anomalous", "protocol": protocol,
               "posterior_recomputed_after_adaptation": True, "finite_scores": finite, "normal_logp0": _stats(p0[normal]), "anomaly_logp0": _stats(p0[anomaly]),
               "normal_logp1": _stats(p1[normal]), "anomaly_logp1": _stats(p1[anomaly]), "normal_score": _stats(scores[normal]), "anomaly_score": _stats(scores[anomaly]),
               "episode_AUROC": _stats(episode_aucs), "posterior": posterior, "diagnosis": diagnosis,
               "note": "Reversed AUROC is diagnostic only; no score or label was changed. Repeated episode queries are not independent image-level observations."}
    with (out_dir / "summary.json").open("w") as handle: json.dump(summary, handle, indent=2)
    with (out_dir / "query_scores.csv").open("w", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    plt.figure(figsize=(6,4)); plt.hist(scores[normal], bins=30, alpha=.7, label="Normal"); plt.hist(scores[anomaly], bins=30, alpha=.7, label="Anomaly"); plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "score_distribution.png"); plt.close()
    fpr, tpr, _ = roc_curve(labels, scores); plt.figure(figsize=(5,4)); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],"--",color="gray"); plt.tight_layout(); plt.savefig(out_dir / "roc_curve.png"); plt.close()
    plt.figure(figsize=(6,4)); plt.hist(p0,bins=30,alpha=.5,label="logp0"); plt.hist(p1,bins=30,alpha=.5,label="logp1"); plt.hist(scores,bins=30,alpha=.5,label="score"); plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "log_probability_distribution.png"); plt.close()
    return summary

