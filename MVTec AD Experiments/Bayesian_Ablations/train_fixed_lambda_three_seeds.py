#!/usr/bin/env python3
"""Train BayPrAnoMeta-FixedLambda from scratch for seeds 1,2,3."""
import argparse,random,time
from pathlib import Path
import higher,numpy as np,torch
from ablation_common import *

def run_episode(model,data,split,rng,training,debug):
 support,qn,qa=sample_episode(data,split,rng);labels=torch.tensor([0]*Q_N+[1]*Q_A,dtype=torch.float32,device=DEVICE);transform=transform_train if training else transform_eval;xs=load_stack(support,transform);xq=load_stack(qn+qa,transform);opt=torch.optim.SGD(model.parameters(),lr=INNER_LR)
 with higher.innerloop_ctx(model,opt,copy_initial_weights=not training,track_higher_grads=training) as (fm,diff):
  for _ in range(INNER_STEPS):
   zs=fm(xs);mu,sigma,dof=fixed_lambda_posterior(zs,debug);diff.step(-log_student_t(zs,mu,sigma,dof).mean())
  zs=fm(xs);mu,sigma,dof=fixed_lambda_posterior(zs,debug);zq=fm(xq);loss=-((1-labels)*log_student_t(zq,mu,sigma,dof)+labels*reference_log_probability(zq)).mean()
 return loss,{"support":support,"normal_query":qn,"anomaly_query":qa}
def train_seed(seed,a):
 set_seed(seed);root=Path(a.run_root)/PROTOCOL/f"seed_{seed}"
 # Create directories required by checkpoint and plot writers.
 for directory in (root/"checkpoints",root/"plots",root/"episode_manifests"):
  directory.mkdir(parents=True,exist_ok=True)
 clients=load_clients(a.base_root,a.split_seed,a.validation_fraction);model=EmbeddingNet().to(DEVICE);outer=torch.optim.Adam(model.parameters(),lr=META_LR);tr=random.Random(seed);vr=random.Random(seed+1);cr=random.Random(seed);vcr=random.Random(seed+1);tl=[];vl=[];tm=[];vm=[];monitor=PeakMemoryMonitor();monitor.start();started=time.perf_counter();checked=False
 for epoch in range(1,EPOCHS+1):
  model.train();losses=[]
  for eid in range(EPISODES_PER_EPOCH):
   c=cr.choice(list(clients));loss,ep=run_episode(model,clients[c],"train",tr,True,a.debug_protocol);outer.zero_grad();loss.backward()
   if not checked:
    if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters()):raise RuntimeError("FixedLambda meta-gradient check failed")
    checked=True
   outer.step();losses.append(float(loss.detach()));tm.append({"epoch":epoch,"episode_id":eid,"category":c,**ep})
  tl.append(float(np.mean(losses)));model.eval();losses=[]
  for eid in range(VAL_EPISODES):
   c=vcr.choice(list(clients));loss,ep=run_episode(model,clients[c],"validation",vr,False,a.debug_protocol);losses.append(float(loss.detach()));vm.append({"epoch":epoch,"episode_id":eid,"category":c,**ep})
  vl.append(float(np.mean(losses)));print(f"FixedLambda seed={seed} epoch={epoch:03d} train={tl[-1]:.4f} val={vl[-1]:.4f}")
  if epoch%5==0:(root/"checkpoints").mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),root/"checkpoints"/f"fixed_lambda_epoch_{epoch}.pth")
 monitor.stop();torch.save(model.state_dict(),root/"checkpoints"/"final_model.pth");write_json(root/"episode_manifests"/"train.json",tm);write_json(root/"episode_manifests"/"validation.json",vm);held={c:d["held_out_subtype"] for c,d in clients.items()};write_json(root/"metadata.json",{"method":"BayPrAnoMeta-FixedLambda","protocol":PROTOCOL,"run_seed":seed,"split_seed":a.split_seed,"held_out_subtypes":held,"epochs":EPOCHS,"episodes_per_epoch":EPISODES_PER_EPOCH,"validation_episodes":VAL_EPISODES,"K_SHOT":K_SHOT,"Q_N":Q_N,"Q_A":Q_A,"inner_steps":INNER_STEPS,"inner_lr":INNER_LR,"meta_lr":META_LR,"Lambda_n":"Lambda_0 exactly","Sigma_n":"((kappa_n+1)/(kappa_n*dof))*Lambda_0","mu_n_kappa_n_nu_n_dof":"unchanged from full BayPrAnoMeta"});write_json(root/"runtime_memory.json",{"training_sec":time.perf_counter()-started,"peak_rss_mb":monitor.peak_mb});save_curve(tl,vl,root/"plots"/"loss_curve.png"); write_json(root / "loss_history.json", [{"epoch": index, "train_loss": train, "validation_loss": validation} for index, (train, validation) in enumerate(zip(tl, vl), 1)]);return held
def main():
 p=argparse.ArgumentParser();p.add_argument("--seeds",nargs="+",type=int,default=[1,2,3]);p.add_argument("--split-seed",type=int,default=SPLIT_SEED);p.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None,default=BASE_ROOT);p.add_argument("--output-root", "--run-root", dest="run_root",default=str(FIXED_RUN_ROOT));p.add_argument("--validation-fraction",type=float,default=.30);p.add_argument("--debug-protocol",action="store_true");a=p.parse_args();ref=None
 for seed in a.seeds:
  print(f"\n=== Training BayPrAnoMeta-FixedLambda seed {seed} ===");held=train_seed(seed,a)
  if ref is None:ref=held
  elif held!=ref:raise RuntimeError("Held-out subtype differs across run seeds")
 write_json(Path(a.run_root)/PROTOCOL/"split_manifest.json",{"split_seed":a.split_seed,"run_seeds":a.seeds,"held_out_subtypes":ref})
if __name__=="__main__":main()
