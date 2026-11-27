"""Shared components for the Bayesian ablations."""
import csv,random,sys,time
from pathlib import Path
import numpy as np,torch

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/"Bayesian_ProtoMAML"))
from baypranometa_three_seed_common import *  # noqa: E402,F401,F403

BASE_ROOT=None
OUTPUT_ROOT=Path(__file__).resolve().parent/"results"
FULL_RUN_ROOT=PROJECT_ROOT/"Bayesian_ProtoMAML"/"runs"/"baypranometa_avgpool_three_seed"
PROTO_RUN_ROOT=PROJECT_ROOT/"CLassical_ProtoMAML"/"runs"/"protomaml_corrected_avgpool_three_seed"
FIXED_RUN_ROOT=OUTPUT_ROOT/"fixed_lambda"
PROTOCOL="revised_clean";SPLIT_SEED=42;EPOCHS=30;EPISODES_PER_EPOCH=25;VAL_EPISODES=20;TEST_EPISODES=100

def fixed_lambda_posterior(z,validate=False):
 K,d=z.shape;mu0=torch.zeros(d,device=z.device,dtype=z.dtype);lambda0=torch.eye(d,device=z.device,dtype=z.dtype);zbar=z.mean(0);kappa_n=KAPPA0+K;nu_n=d+NU0_OFFSET+K
 mu_n=(KAPPA0*mu0+K*zbar)/kappa_n;dof=nu_n-d+1;sigma_n=((kappa_n+1)/(kappa_n*dof))*lambda0
 if validate:
  assert dof==NU0_OFFSET+K+1 and torch.allclose(sigma_n,((kappa_n+1)/(kappa_n*dof))*lambda0)
  assert torch.linalg.cholesky(sigma_n+1e-6*lambda0).shape==(d,d)
 return mu_n,sigma_n,dof

def csvwrite(path,rows,fields=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);rows=list(rows)
 if fields is None:fields=list(rows[0]) if rows else []
 with path.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)

def checkpoints(seed,full_root=FULL_RUN_ROOT,proto_root=PROTO_RUN_ROOT,fixed_root=FIXED_RUN_ROOT):
 return {"full":Path(full_root)/PROTOCOL/f"seed_{seed}"/"checkpoints"/"final_model.pth","proto":Path(proto_root)/PROTOCOL/f"seed_{seed}"/"checkpoints"/"final_model.pth","fixed_lambda":Path(fixed_root)/PROTOCOL/f"seed_{seed}"/"checkpoints"/"final_model.pth"}

def load_clients(base_root,split_seed,validation_fraction):
 out={}
 for c in CLIENTS:
  d=load_revised_client(Path(base_root)/c,split_seed,validation_fraction)
  if d is not None:out[c]=d
 if not out:raise RuntimeError("No eligible MVTec categories loaded.")
 return out
