#!/usr/bin/env python3
"""Standalone clean Contrastive BayPrAnoMeta training, sequential run seeds 1/2/3."""
import argparse, json, os, random, threading, time
from pathlib import Path
import higher, numpy as np, psutil, torch, torch.nn as nn, torchvision.models as models
from utils.utils_data_fedcontrast_leave1out import load_client_dataset_leave1out, load_images, transform_eval, transform_train
from utils.utils_niw import log_student_t, niw_posterior
from utils.utils_plotting import plot_loss_curve

BASE_ROOT = None
CLIENTS=["bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]
RUN_ROOT="runs/contrastive_baypranometa_three_seed_fixed_split42_25ep_30epochs_100test"; PROTOCOL="revised_clean"; SPLIT_SEED=42; DEVICE=torch.device("cpu")
EPOCHS,EPISODES_PER_EPOCH,VAL_EPISODES=30,25,20; K_SHOT,Q_N,Q_A=5,12,4; INNER_STEPS,INNER_LR,META_LR=1,5e-4,1e-4; LAMBDA_CON,TEMPERATURE=.1,.07

class Monitor:
 def __init__(self): self.p=psutil.Process(os.getpid());self.peak=0.;self.live=False
 def start(self):
  self.live=True
  def f():
   while self.live: self.peak=max(self.peak,self.p.memory_info().rss/1024**2);time.sleep(.1)
  self.t=threading.Thread(target=f,daemon=True);self.t.start()
 def stop(self): self.live=False;self.t.join()
class Net(nn.Module):
 def __init__(self):
  super().__init__();b=models.resnet18(weights=models.ResNet18_Weights.DEFAULT);b.maxpool=nn.AvgPool2d(kernel_size=3,stride=2,padding=1);self.f=nn.Sequential(*list(b.children())[:-1]);self.h=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.LayerNorm(128))
 def forward(self,x): return self.h(self.f(x).flatten(1))
def write(path,x):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w") as f: json.dump(x,f,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))
def set_seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)
def stack(paths,t):
 x=load_images(paths,t)
 if len(x)!=len(paths): raise RuntimeError("Image loading failed")
 return torch.stack(x).to(DEVICE)
def supcon(z,y):
 z=nn.functional.normalize(z,dim=1); y=y.view(-1); sim=z@z.T/TEMPERATURE
 self_mask=torch.eye(len(z),device=z.device,dtype=torch.bool)
 positive_mask=y[:,None].eq(y[None,:]) & ~self_mask
 positive_counts=positive_mask.sum(1); valid=positive_counts>0
 if len(y)==Q_N+Q_A and torch.equal(torch.bincount(y,minlength=2),torch.tensor([Q_N,Q_A],device=y.device)):
  assert torch.all(positive_counts[y==0]==Q_N-1) and torch.all(positive_counts[y==1]==Q_A-1)
 if not valid.any(): return z.sum()*0.0
 logits=sim-sim.max(1,keepdim=True)[0]
 log_denom=torch.logsumexp(logits.masked_fill(self_mask,float("-inf")),dim=1,keepdim=True)
 log_prob=logits-log_denom
 return -((positive_mask[valid].float()*log_prob[valid]).sum(1)/positive_counts[valid]).mean()
def ref(z):
 d=z.shape[1];return log_student_t(z,torch.zeros(d,device=DEVICE),100*torch.eye(d,device=DEVICE),torch.tensor(2.,device=DEVICE))
def episode(data,split,rng,train,debug):
 p=data[split];n=rng.sample(p["normal"],K_SHOT+Q_N);s,qn,qa=n[:K_SHOT],n[K_SHOT:],rng.sample(p["anomaly"],Q_A)
 if debug: assert set(s).isdisjoint(qn)
 y=torch.tensor([0]*Q_N+[1]*Q_A,device=DEVICE);t=transform_train if train else transform_eval;xs,xq=stack(s,t),stack(qn+qa,t);return s,qn,qa,y,xs,xq
def loss(model,data,split,rng,train,debug):
 s,qn,qa,y,xs,xq=episode(data,split,rng,train,debug);o=torch.optim.SGD(model.parameters(),lr=INNER_LR)
 with higher.innerloop_ctx(model,o,copy_initial_weights=False,track_higher_grads=True) as (fm,do):
  for _ in range(INNER_STEPS):
   zs=fm(xs);mu,sg,df=niw_posterior(zs,validate=debug);do.step(-log_student_t(zs,mu,sg,df).mean())
  zs=fm(xs);mu,sg,df=niw_posterior(zs,validate=debug);zq=fm(xq);bayes=-((1-y)*log_student_t(zq,mu,sg,df)+y*ref(zq)).mean();total=bayes+LAMBDA_CON*supcon(zq,y)
 return total,{"support":s,"normal_query":qn,"anomaly_query":qa}
def train(run,args):
 set_seed(run);rd=Path(args.run_root)/PROTOCOL/f"seed_{run}";start=time.perf_counter();ds={}
 for c in CLIENTS:
  d=load_client_dataset_leave1out(Path(args.base_root)/c,seed=args.split_seed,protocol=PROTOCOL,validation_fraction=args.validation_fraction)
  if d: ds[c]=d
 if not ds: raise RuntimeError("No clients")
 data_t=time.perf_counter()-start;m=Net().to(DEVICE);opt=torch.optim.Adam(m.parameters(),lr=META_LR);tr,vr,cr,vc=random.Random(run),random.Random(run+1),random.Random(run),random.Random(run+1);mon=Monitor();mon.start();start=time.perf_counter();tl,vl,tm,vm=[],[],[],[]
 for e in range(1,EPOCHS+1):
  m.train();a=[]
  for i in range(EPISODES_PER_EPOCH):
   c=cr.choice(list(ds));l,ep=loss(m,ds[c],"train",tr,True,args.debug_protocol);opt.zero_grad();l.backward();opt.step();a.append(l.item());tm.append({"epoch":e,"episode":i,"category":c,**ep})
  tl.append(float(np.mean(a)));m.eval();a=[]
  for i in range(VAL_EPISODES):
   c=vc.choice(list(ds));l,ep=loss(m,ds[c],"validation",vr,False,args.debug_protocol);a.append(l.item());vm.append({"epoch":e,"episode":i,"category":c,**ep})
  vl.append(float(np.mean(a)));print(f"run_seed={run} split_seed={args.split_seed} epoch={e:03d} train={tl[-1]:.4f} val={vl[-1]:.4f}")
  if e%5==0: (rd/"checkpoints").mkdir(parents=True,exist_ok=True);torch.save(m.state_dict(),rd/"checkpoints"/f"epoch_{e}.pth")
 plot_loss_curve(tl,vl,rd/"plots"/"loss_curve.png"); write(rd / "loss_history.json", [{"epoch": index, "train_loss": train, "validation_loss": validation} for index, (train, validation) in enumerate(zip(tl, vl), 1)]);torch.save(m.state_dict(),rd/"checkpoints"/"final_model.pth");mon.stop();held={c:d["held_out_subtype"] for c,d in ds.items()};write(rd/"metadata.json",{"method":"Contrastive BayPrAnoMeta","run_seed":run,"split_seed":args.split_seed,"held_out_subtypes":held,"epochs":EPOCHS,"episodes_per_epoch":EPISODES_PER_EPOCH,"validation_episodes":VAL_EPISODES,"lambda_con":LAMBDA_CON,"temperature":TEMPERATURE});write(rd/"episode_manifests"/"train.json",tm);write(rd/"episode_manifests"/"validation.json",vm);write(rd/"runtime_memory.json",{"run_seed":run,"split_seed":args.split_seed,"data_load_sec":data_t,"training_loop_sec":time.perf_counter()-start,"peak_rss_mb":mon.peak});return held
def main():
 p=argparse.ArgumentParser();p.add_argument("--seeds",nargs="+",type=int,default=[1,2,3]);p.add_argument("--split-seed",type=int,default=SPLIT_SEED);p.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None,default=BASE_ROOT);p.add_argument("--output-root", "--run-root", dest="run_root",default=RUN_ROOT);p.add_argument("--validation-fraction",type=float,default=.30);p.add_argument("--debug-protocol",action="store_true");a=p.parse_args();refmap=None
 for s in a.seeds:
  h=train(s,a)
  if refmap is None: refmap=h
  elif h!=refmap: raise RuntimeError("Held-out subtype differs across run seeds")
 write(Path(a.run_root)/PROTOCOL/"split_manifest.json",{"split_seed":a.split_seed,"held_out_subtypes":refmap,"run_seeds":a.seeds})
if __name__=="__main__": main()
