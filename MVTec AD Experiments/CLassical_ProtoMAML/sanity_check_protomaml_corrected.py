#!/usr/bin/env python3
"""One-category, one-episode gradient/adaptation sanity check; writes no results."""
import argparse
import random
from pathlib import Path

import higher
import torch
import torch.nn as nn

from train_protomaml_three_seeds_corrected import (
    BASE_ROOT, DEVICE, INNER_LR, K_SHOT, Q_A, Q_N, EmbeddingNet,
    load_stack, sample_episode, set_seed,
)
from utils.utils_data_fedcontrast_leave1out import load_client_dataset_leave1out, transform_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None, default=BASE_ROOT)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--run-seed", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=.30)
    args = parser.parse_args()

    set_seed(args.run_seed)
    data = load_client_dataset_leave1out(
        Path(args.base_root) / args.category, seed=args.split_seed,
        protocol="revised_clean", validation_fraction=args.validation_fraction,
    )
    if data is None:
        raise RuntimeError(f"Category {args.category!r} is not eligible.")
    support, qn, qa = sample_episode(data, "train", random.Random(args.run_seed), True)
    xs = load_stack(support, transform_train)
    xq = load_stack(qn + qa, transform_train)
    labels = torch.tensor([0] * Q_N + [1] * Q_A, dtype=torch.float32, device=DEVICE)

    model = EmbeddingNet().to(DEVICE)
    original_before = [p.detach().clone() for p in model.parameters()]
    inner_opt = torch.optim.SGD(model.parameters(), lr=INNER_LR)
    with higher.innerloop_ctx(
        model, inner_opt, copy_initial_weights=False, track_higher_grads=True
    ) as (fmodel, diffopt):
        fast_before = [p.detach().clone() for p in fmodel.parameters()]
        z_support = fmodel(xs)
        prototype = z_support.mean(0, keepdim=True)
        inner_loss = ((z_support - prototype) ** 2).sum(1).mean()
        diffopt.step(inner_loss)
        fast_changed = any(not torch.equal(a, b.detach()) for a, b in zip(fast_before, fmodel.parameters()))
        original_unchanged_during_inner = all(
            torch.equal(a, b.detach()) for a, b in zip(original_before, model.parameters())
        )
        z_support = fmodel(xs)
        prototype = z_support.mean(0, keepdim=True)
        logits = ((fmodel(xq) - prototype) ** 2).sum(1)
        outer_loss = nn.BCEWithLogitsLoss()(logits, labels)
        gradients = torch.autograd.grad(outer_loss, tuple(model.parameters()), allow_unused=True)

    original_unchanged_after_inner = all(
        torch.equal(a, b.detach()) for a, b in zip(original_before, model.parameters())
    )
    nonzero_meta_gradient = any(
        g is not None and torch.isfinite(g).all() and g.abs().sum().item() > 0 for g in gradients
    )
    assert set(support).isdisjoint(qn)
    assert fast_changed, "Higher fast weights did not change."
    assert original_unchanged_during_inner and original_unchanged_after_inner, "Inner step modified the meta-model."
    assert nonzero_meta_gradient, "Outer loss did not reach the original meta-parameters."
    assert torch.equal(labels.long().cpu(), torch.tensor([0] * Q_N + [1] * Q_A))
    print({
        "category": args.category,
        "held_out_subtype": data["held_out_subtype"],
        "support_query_disjoint": True,
        "fast_weights_changed": fast_changed,
        "original_meta_model_unchanged": True,
        "nonzero_meta_gradient": nonzero_meta_gradient,
        "label_convention": {"normal": 0, "anomaly": 1},
        "query_logit": "positive squared distance",
        "inner_loss": float(inner_loss.detach()),
        "outer_loss": float(outer_loss.detach()),
    })


if __name__ == "__main__":
    main()
