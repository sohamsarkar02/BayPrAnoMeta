#!/usr/bin/env python3

import argparse,time
from pathlib import Path
import numpy as np,torch
from patchcore_controlled_common import *
def train_seed(seed,a):
 set_seed(seed);model=SharedEmbeddingNet().to(DEVICE);model.eval();model_dir=Path(a.result_root)/"seed_models";model_dir.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),model_dir/f"seed_{seed}_encoder.pth")
 if a.encoder_only:return {}
 mon=Monitor();mon.start();started=time.perf_counter();held={}
 for ci,c in enumerate(CLIENTS):
  d=load_category(Path(a.base_root)/c,a.split_seed)
  if d is None:continue
  source=set(d["train_normal"]);test_n=set(d["test_normal"]);test_a=set(d["test_anomaly"]);overlap_n=source&test_n;overlap_a=source&test_a
  if overlap_n or overlap_a:raise RuntimeError(f"{c}: memory-bank leakage detected: test_normal={len(overlap_n)}, test_anomaly={len(overlap_a)}")
  candidate=extract(model,d["train_normal"]);n=len(candidate);subsampled=[f for f in a.fractions if f<1.0];max_k=max((coreset_size(n,f) for f in subsampled),default=0);order=greedy_order(candidate,max_k,seed*1000+ci) if max_k else np.asarray([],dtype=np.int64);held[c]=d["held_out_subtype"]
  for fraction in a.fractions:
   k=coreset_size(n,fraction);bank=candidate if fraction==1.0 else candidate[order[:k]];root=Path(a.result_root)/f"fraction_{FRACTION_KEYS[fraction]}"/f"seed_{seed}";root.mkdir(parents=True,exist_ok=True);np.save(root/f"{c}_memory.npy",bank)
   write(root/f"{c}_memory_metadata.json",{"category":c,"held_out_subtype":d["held_out_subtype"],"split_seed":a.split_seed,"run_seed":seed,"memory_bank_source":"train/good","memory_bank_source_paths":d["train_normal"],"memory_bank_test_overlap":0,"memory_bank_test_normal_overlap":0,"memory_bank_test_anomaly_overlap":0,"candidate_memory_size":n,"requested_coreset_fraction":fraction,"actual_coreset_size":len(bank),"actual_fraction":len(bank)/n,"full_bank_used_without_subsampling":fraction==1.0,"encoder_state":str(model_dir/f"seed_{seed}_encoder.pth")})
  print(f"seed={seed} {c}: candidate patches={n}; all fractions saved")
 mon.stop();write(Path(a.result_root)/f"seed_{seed}_training_runtime_memory.json",{"run_seed":seed,"training_sec":time.perf_counter()-started,"peak_rss_mb":mon.peak});return held
def main():
 p=argparse.ArgumentParser();p.add_argument("--seeds",nargs="+",type=int,default=[1,2,3]);p.add_argument("--fractions",nargs="+",type=float,default=list(FRACTIONS));p.add_argument("--split-seed",type=int,default=SPLIT_SEED);p.add_argument("--data-root", "--base-root", dest="base_root", default=BASE_ROOT);p.add_argument("--output-root", "--result-root", dest="result_root",default=RESULT_ROOT);p.add_argument("--debug-protocol",action="store_true");p.add_argument("--encoder-only", action="store_true", help="Write only the seed encoders used by the 5-shot runners.");a=p.parse_args();a.fractions=tuple(a.fractions)
 if a.base_root is None and not a.encoder_only:p.error("--data-root is required unless --encoder-only is used")
 if any(f not in FRACTIONS for f in a.fractions):raise ValueError(f"Fractions must be selected from {FRACTIONS}")
 ref=None
 for seed in a.seeds:
  print(f"\n=== Controlled PatchCore training seed {seed} ===");held=train_seed(seed,a)
  if ref is None:ref=held
  elif held!=ref:raise RuntimeError("Held-out subtype differs across run seeds.")
 write(Path(a.result_root)/"split_manifest.json",{"split_seed":a.split_seed,"run_seeds":a.seeds,"held_out_subtypes":ref})
if __name__=="__main__":main()
