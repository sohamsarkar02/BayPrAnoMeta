#!/usr/bin/env python3
"""Fast mathematical and protocol checks; no training."""
import argparse, random
from pathlib import Path

import numpy as np
import torch

from ablation_common_visa import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None, default=BASE_ROOT)
    parser.add_argument("--split-csv", default="split_csv/2cls_highshot.csv"); parser.add_argument("--subset-file", help='Paper image allowlist. Required for exact reproduction of the paper experiments.')
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--category", default="candle")
    parser.add_argument("--output", default=str(HERE / "results" / "sanity_check.json"))
    args = parser.parse_args()
    if args.split_seed != 42: parser.error("SPLIT_SEED is fixed at 42")
    data = load_clients(args.base_root, args.split_seed, 0.30, args.split_csv, args.category, subset_file=args.subset_file)[args.category]
    rng = random.Random(1)
    selected = rng.sample(data["test"]["normal"], K_SHOT + Q_N)
    support, normal_query = selected[:K_SHOT], selected[K_SHOT:]
    anomaly_query = rng.sample(data["test"]["anomaly"], Q_A)
    z = torch.randn(K_SHOT, 128, requires_grad=True)
    mu, sigma, dof = fixed_lambda_posterior(z, validate=True)
    expected = ((0.01 + K_SHOT + 1) / ((0.01 + K_SHOT) * dof)) * torch.eye(128)
    loss = -log_student_t(z, mu, sigma, dof).mean(); loss.backward()
    checks = {
        "category": args.category,
        "split_seed": args.split_seed,
        "held_out_subtype": data["held_out_subtype"],
        "support_query_disjoint": set(support).isdisjoint(normal_query),
        "held_out_anomaly_disjoint_from_train": set(anomaly_query).isdisjoint(data["train"]["anomaly"]),
        "held_out_anomaly_disjoint_from_validation": set(anomaly_query).isdisjoint(data["validation"]["anomaly"]),
        "fixed_lambda_sigma_exact": bool(torch.allclose(sigma.detach(), expected)),
        "fixed_lambda_dof": int(dof),
        "niw_gradient_finite": bool(torch.isfinite(z.grad).all()),
        "label_convention": {"normal": 0, "anomaly": 1},
        "score_direction": "log p1 - log p0; larger is anomalous",
    }
    if not all(value for key, value in checks.items()
               if key in ("support_query_disjoint", "held_out_anomaly_disjoint_from_train",
                          "held_out_anomaly_disjoint_from_validation", "fixed_lambda_sigma_exact",
                          "niw_gradient_finite")):
        raise RuntimeError(checks)
    write_json(args.output, checks)
    print(checks)


if __name__ == "__main__":
    main()

