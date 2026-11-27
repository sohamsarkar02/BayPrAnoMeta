#!/usr/bin/env python3
"""Fit one standard official PatchCore memory bank per category and seed."""

import argparse, time
from pathlib import Path
import numpy as np
from standard_patchcore_common import *


def train_seed(seed,args):
    categories=CLIENTS[:1] if args.smoke_test else CLIENTS
    held={}; monitor=PeakMemoryMonitor(); monitor.start(); started=time.perf_counter()
    for category_index,category in enumerate(categories):
        data=load_protocol(args.base_root,category,args.split_seed,args.validation_fraction)
        if data is None: continue
        fit_paths=data["train"]["normal"][:2] if args.smoke_test else data["train"]["normal"]
        source=set(fit_paths)
        forbidden=set(data["validation"]["normal"])|set(data["test"]["normal"])|set(data["train"]["anomaly"])|set(data["validation"]["anomaly"])|set(data["test"]["anomaly"])
        if source & forbidden: raise RuntimeError(f"{category}: memory-bank leakage detected")
        set_seed(seed*1000+category_index)
        model,sampler=build_patchcore(seed*1000+category_index)
        fit_start=time.perf_counter(); model.fit(loader(fit_paths,args.batch_size)); fit_seconds=time.perf_counter()-fit_start
        bank=np.asarray(model.anomaly_scorer.detection_features,dtype=np.float32)
        out=Path(args.result_root)/f"seed_{seed}"/"memory_banks"/category; out.mkdir(parents=True,exist_ok=True)
        np.save(out/"memory_bank.npy",bank)
        metadata={"method":"PatchCore-Standard","category":category,"run_seed":seed,"split_seed":args.split_seed,
                  "held_out_subtype":data["held_out_subtype"],"memory_bank_source":"corrected logical training-normal pool",
                  "memory_bank_source_paths":fit_paths,"eligible_training_normal_count":len(data["train"]["normal"]),"smoke_subset":bool(args.smoke_test),"validation_normal_overlap":len(source&set(data["validation"]["normal"])),
                  "test_normal_overlap":len(source&set(data["test"]["normal"])),"anomaly_overlap":len(source&forbidden),
                  "held_out_subtype_used_for_fitting":False,"backbone":"torchvision wide_resnet50_2 (official name wideresnet50)",
                  "feature_layers":FEATURE_LAYERS,"pretrain_embed_dimension":PRETRAIN_DIM,"target_embed_dimension":TARGET_DIM,
                  "custom_128d_projection_head":False,"patch_size":PATCH_SIZE,"nearest_neighbours":NUM_NN,
                  "sampler":"official ApproximateGreedyCoresetSampler","coreset_fraction":CORESET_FRACTION,
                  "memory_bank_size_before_coreset":sampler.before_count,"memory_bank_size_after_coreset":sampler.after_count,
                  "actual_coreset_fraction":sampler.after_count/sampler.before_count,"fit_seconds":fit_seconds,
                  "memory_bank_rebuilt_per_test_episode":False}
        write_json(out/"metadata.json",metadata); held[category]=data["held_out_subtype"]
        print(f"seed={seed} {category}: patches {sampler.before_count} -> {sampler.after_count}")
        del model
    monitor.stop(); write_json(Path(args.result_root)/f"seed_{seed}"/"training_runtime_memory.json",
        {"run_seed":seed,"split_seed":args.split_seed,"training_seconds":time.perf_counter()-started,"peak_rss_mb":monitor.peak_mb})
    return held


def main():
    p=argparse.ArgumentParser();p.add_argument("--seeds",nargs="+",type=int,default=[1,2,3]);p.add_argument("--split-seed",type=int,default=SPLIT_SEED);p.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None,default=BASE_ROOT);p.add_argument("--output-root", "--result-root", dest="result_root",default=RESULT_ROOT);p.add_argument("--validation-fraction",type=float,default=.30);p.add_argument("--batch-size",type=int,default=2);p.add_argument("--debug-protocol",action="store_true");p.add_argument("--smoke-test",action="store_true");args=p.parse_args()
    if args.smoke_test: args.seeds=[1]; args.result_root += "_smoke"
    reference=None
    for seed in args.seeds:
        held=train_seed(seed,args)
        if reference is None: reference=held
        elif held!=reference: raise RuntimeError("Held-out subtype mismatch across seeds")
    write_json(Path(args.result_root)/"split_manifest.json",{"split_seed":args.split_seed,"run_seeds":args.seeds,"held_out_subtypes":reference})
if __name__=="__main__":main()
