#!/usr/bin/env python3
"""Leakage-free Classical MAML training for run seeds 1, 2, and 3."""
import argparse, random, time
from pathlib import Path
import higher, numpy as np, torch, torch.nn as nn
from maml_three_seed_common import *
from utils.utils_data_fedcontrast_leave1out import load_client_dataset_leave1out, load_images, transform_eval, transform_train
from utils.utils_plotting import plot_loss_curve

def run_episode(model, data, split, rng, training):
    support, qn, qa = sample_episode(data, split, rng)
    transform = transform_train if training else transform_eval
    xs = stack(support, transform, load_images)
    xq = stack(qn + qa, transform, load_images)
    labels = torch.tensor([0]*Q_N + [1]*Q_A, dtype=torch.float32, device=DEVICE)
    inner_opt = torch.optim.SGD(model.parameters(), lr=INNER_LR)
    bce = nn.BCEWithLogitsLoss()
    with higher.innerloop_ctx(
        model, inner_opt,
        copy_initial_weights=not training,
        track_higher_grads=training,
    ) as (fmodel, diffopt):
        for _ in range(INNER_STEPS):
            support_logits = fmodel(xs)
            diffopt.step(bce(support_logits, torch.zeros_like(support_logits)))
        query_logits = fmodel(xq)
        outer_loss = bce(query_logits, labels)
    manifest = {"support": support, "normal_query": qn, "anomaly_query": qa}
    return outer_loss, manifest

def train_seed(run_seed, args):
    set_seed(run_seed)
    run_dir = Path(args.run_root) / PROTOCOL / f"seed_{run_seed}"
    load_started = time.perf_counter()
    datasets = {}
    for category in CLIENTS:
        data = load_client_dataset_leave1out(Path(args.base_root)/category, seed=args.split_seed,
                                             protocol=PROTOCOL, validation_fraction=args.validation_fraction)
        if data is not None: datasets[category] = data
    if not datasets: raise RuntimeError("No eligible MVTec categories loaded.")
    data_load_sec = time.perf_counter() - load_started
    model = MAMLNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=META_LR)
    train_rng, val_rng = random.Random(run_seed), random.Random(run_seed + 1000)
    client_rng, val_client_rng = random.Random(run_seed + 2000), random.Random(run_seed + 3000)
    train_losses, val_losses, train_manifest, val_manifest, epoch_times = [], [], [], [], []
    monitor = PeakMemoryMonitor(); monitor.start(); loop_started = time.perf_counter()
    gradient_checked = False
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter(); model.train(); losses = []
        for episode_id in range(EPISODES_PER_EPOCH):
            category = client_rng.choice(list(datasets))
            loss, episode = run_episode(model, datasets[category], "train", train_rng, True)
            optimizer.zero_grad(); loss.backward()
            if not gradient_checked:
                valid = [p.grad for p in model.parameters() if p.grad is not None]
                if not valid or not all(torch.isfinite(g).all() for g in valid) or not any(g.abs().sum() > 0 for g in valid):
                    raise RuntimeError("MAML meta-gradient sanity check failed.")
                gradient_checked = True
            optimizer.step(); losses.append(float(loss.detach()))
            train_manifest.append({"epoch":epoch,"episode_id":episode_id,"category":category,**episode})
        train_losses.append(float(np.mean(losses)))
        model.eval(); losses = []
        for episode_id in range(VAL_EPISODES):
            category = val_client_rng.choice(list(datasets))
            loss, episode = run_episode(model, datasets[category], "validation", val_rng, False)
            losses.append(float(loss.detach()))
            val_manifest.append({"epoch":epoch,"episode_id":episode_id,"category":category,**episode})
        val_losses.append(float(np.mean(losses))); epoch_times.append(time.perf_counter()-epoch_started)
        print(f"seed={run_seed} epoch={epoch:03d} train={train_losses[-1]:.4f} val={val_losses[-1]:.4f}")
        if epoch % 5 == 0:
            (run_dir/"checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), run_dir/"checkpoints"/f"maml_epoch_{epoch}.pth")
    monitor.stop(); (run_dir/"checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir/"checkpoints"/"final_model.pth")
    plot_loss_curve(train_losses, val_losses, run_dir/"plots"/"loss_curve.png"); write_json(run_dir / "loss_history.json", [{"epoch": index, "train_loss": train, "validation_loss": validation} for index, (train, validation) in enumerate(zip(train_losses, val_losses), 1)])
    write_json(run_dir/"episode_manifests"/"train_episodes.json", train_manifest)
    write_json(run_dir/"episode_manifests"/"validation_episodes.json", val_manifest)
    held = {c:d["held_out_subtype"] for c,d in datasets.items()}
    write_json(run_dir/"metadata.json", {"method":"Classical MAML corrected","protocol":PROTOCOL,
        "run_seed":run_seed,"split_seed":args.split_seed,"held_out_subtypes":held,
        "epochs":EPOCHS,"episodes_per_epoch":EPISODES_PER_EPOCH,"validation_episodes":VAL_EPISODES,
        "K_SHOT":K_SHOT,"Q_N":Q_N,"Q_A":Q_A,"inner_steps":INNER_STEPS,"inner_lr":INNER_LR,"meta_lr":META_LR,
        "backbone":"ImageNet-pretrained ResNet18; first max-pool replaced by 3x3 stride-2 average pooling",
        "embedding_dim":128,"classifier":"Linear(128,1)","label_convention":{"normal":0,"anomaly":1},
        "support_definition":"normal-only","score":"adapted anomaly logit; higher is anomalous"})
    write_json(run_dir/"runtime_memory.json", {"data_load_sec":data_load_sec,"training_loop_sec":time.perf_counter()-loop_started,
        "epoch_times_sec":epoch_times,"peak_rss_mb":monitor.peak_mb})
    return held

def main():
    p=argparse.ArgumentParser();p.add_argument("--seeds",nargs="+",type=int,default=[1,2,3]);p.add_argument("--split-seed",type=int,default=SPLIT_SEED)
    p.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None,default=BASE_ROOT);p.add_argument("--output-root", "--run-root", dest="run_root",default=RUN_ROOT);p.add_argument("--validation-fraction",type=float,default=.30);p.add_argument("--debug-protocol",action="store_true")
    args=p.parse_args(); reference=None
    for seed in args.seeds:
        print(f"\n=== Starting Classical MAML training: run seed {seed} ===")
        held=train_seed(seed,args)
        if reference is None: reference=held
        elif held != reference: raise RuntimeError("Held-out subtypes differ across run seeds.")
        print(f"=== Completed Classical MAML training: run seed {seed} ===")
    write_json(Path(args.run_root)/PROTOCOL/"split_manifest.json",{"split_seed":args.split_seed,"run_seeds":args.seeds,"held_out_subtypes":reference})
if __name__=="__main__": main()
