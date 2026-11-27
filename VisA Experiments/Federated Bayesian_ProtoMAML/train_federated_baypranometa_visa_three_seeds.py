#!/usr/bin/env python3
"""Standalone revised federated BayPrAnoMeta training for run seeds 1, 2, 3."""
import argparse, json, os, random, threading, time
from copy import deepcopy
from pathlib import Path
import higher, numpy as np, psutil, torch, torch.nn as nn, torchvision.models as models
from utils.utils_data_visa_leave1out import discover_categories, load_client_dataset_visa, load_images, transform_eval, transform_train
from utils.utils_niw import log_student_t, niw_posterior
from utils.utils_plotting import plot_loss_curve

BASE_ROOT = None
RUN_ROOT="runs/federated_baypranometa_visa_three_seed_fixed_split42_25epclient_30rounds_100test"; PROTOCOL="revised_clean_official_2cls_highshot"; SPLIT_SEED=42; DEVICE=torch.device("cpu")
ROUNDS,EPISODES_PER_CLIENT,VAL_EPISODES=30,25,20; K_SHOT,Q_N,Q_A=5,12,4; INNER_STEPS,INNER_LR,SERVER_LR=1,5e-4,1e-4

class Monitor:
 def __init__(self): self.p=psutil.Process(os.getpid());self.peak=0.;self.live=False
 def start(self):
  self.live=True
  def poll():
   while self.live: self.peak=max(self.peak,self.p.memory_info().rss/1024**2);time.sleep(.1)
  self.t=threading.Thread(target=poll,daemon=True);self.t.start()
 def stop(self): self.live=False;self.t.join()
class Net(nn.Module):
 def __init__(self):
  super().__init__();b=models.resnet18(weights=models.ResNet18_Weights.DEFAULT);b.maxpool=nn.AvgPool2d(kernel_size=3,stride=2,padding=1);self.features=nn.Sequential(*list(b.children())[:-1]);self.head=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.LayerNorm(128))
 def forward(self,x): return self.head(self.features(x).flatten(1))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("w") as f: json.dump(x,f,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))
def set_seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)
def stack(paths,t):
 x=load_images(paths,t)
 if len(x)!=len(paths): raise RuntimeError("Image loading failed")
 return torch.stack(x).to(DEVICE)
def ref(z):
 d=z.shape[1];return log_student_t(z,torch.zeros(d,device=DEVICE),100*torch.eye(d,device=DEVICE),torch.tensor(2.,device=DEVICE))
def sample(data,split,rng,debug):
 pool=data[split];normal=rng.sample(pool["normal"],K_SHOT+Q_N);s,qn,qa=normal[:K_SHOT],normal[K_SHOT:],rng.sample(pool["anomaly"],Q_A)
 if debug: assert set(s).isdisjoint(qn)
 return s,qn,qa
def client_gradient(global_model,data,rng,debug,episode_count=EPISODES_PER_CLIENT):
 """Average client meta-gradients; server aggregation remains gradient averaging."""
 grads_acc=None; losses=[];manifest=[]
 for episode_id in range(episode_count):
  s,qn,qa=sample(data,"train",rng,debug);xs,xq=stack(s,transform_train),stack(qn+qa,transform_eval);y=torch.tensor([0]*Q_N+[1]*Q_A,device=DEVICE)
  local=deepcopy(global_model);opt=torch.optim.SGD(local.parameters(),lr=INNER_LR)
  with higher.innerloop_ctx(local,opt,copy_initial_weights=False,track_higher_grads=True) as (fm,diff):
   for _ in range(INNER_STEPS):
    zs=fm(xs);mu,sg,df=niw_posterior(zs,validate=debug);diff.step(-log_student_t(zs,mu,sg,df).mean())
   zs=fm(xs);mu,sg,df=niw_posterior(zs,validate=debug);zq=fm(xq);outer=-((1-y)*log_student_t(zq,mu,sg,df)+y*ref(zq)).mean()
   grads=torch.autograd.grad(outer,local.parameters(),allow_unused=False)
  grads=[g.detach() for g in grads];losses.append(float(outer.detach()));manifest.append({"episode":episode_id,"support":s,"normal_query":qn,"anomaly_query":qa})
  if grads_acc is None: grads_acc=grads
  else:
   for i,g in enumerate(grads): grads_acc[i]+=g
 return [g/episode_count for g in grads_acc],float(np.mean(losses)),manifest
def validation(model,ds,rng,debug,episode_count=VAL_EPISODES):
 vals=[];manifest=[];clients=list(ds)
 model.eval()
 for i in range(episode_count):
  c=rng.choice(clients);s,qn,qa=sample(ds[c],"validation",rng,debug);xs,xq=stack(s,transform_eval),stack(qn+qa,transform_eval);y=torch.tensor([0]*Q_N+[1]*Q_A,device=DEVICE);opt=torch.optim.SGD(model.parameters(),lr=INNER_LR)
  with higher.innerloop_ctx(model,opt,copy_initial_weights=True,track_higher_grads=False) as (fm,diff):
   zs=fm(xs);mu,sg,df=niw_posterior(zs,validate=debug);diff.step(-log_student_t(zs,mu,sg,df).mean());zs=fm(xs);mu,sg,df=niw_posterior(zs,validate=debug);zq=fm(xq)
   with torch.no_grad(): vals.append(float(-((1-y)*log_student_t(zq,mu,sg,df)+y*ref(zq)).mean()))
  manifest.append({"episode":i,"category":c,"support":s,"normal_query":qn,"anomaly_query":qa})
 return float(np.mean(vals)),manifest
def train(run,args):
 set_seed(run);root=Path(args.run_root)/PROTOCOL/f"seed_{run}";started=time.perf_counter();ds={}
 clients=discover_categories(args.base_root,args.split_csv, subset_file=args.subset_file)
 if args.category:
  if args.category not in clients: raise ValueError(f"Unknown VisA client: {args.category}")
  clients=[args.category]
 if args.smoke_test and not args.category: clients=clients[:2]
 for c in clients:
  data=load_client_dataset_visa(args.base_root,c,seed=args.split_seed,validation_fraction=args.validation_fraction,split_csv=args.split_csv, subset_file=args.subset_file,audit_dir=root/"protocol_audit")
  if data: ds[c]=data
 if not ds: raise RuntimeError("No clients loaded")
 rounds=1 if args.smoke_test else ROUNDS;client_episodes=1 if args.smoke_test else EPISODES_PER_CLIENT;val_episodes=1 if args.smoke_test else VAL_EPISODES
 data_sec=time.perf_counter()-started;m=Net().to(DEVICE);parameter_count=sum(p.numel() for p in m.parameters());server=torch.optim.Adam(m.parameters(),lr=SERVER_LR);trng,vrng=random.Random(run),random.Random(run+1);mon=Monitor();mon.start();started=time.perf_counter();tl=[];vl=[];train_manifest=[];val_manifest=[];round_seconds=[]
 for round_id in range(1,rounds+1):
  round_started=time.perf_counter()
  m.train();client_grads=[];losses=[]
  for category,data in ds.items():
   grad,loss,episodes=client_gradient(m,data,trng,args.debug_protocol,client_episodes);client_grads.append(grad);losses.append(loss);train_manifest.extend({"round":round_id,"category":category,**e} for e in episodes)
  server.zero_grad()
  for idx,p in enumerate(m.parameters()): p.grad=torch.stack([g[idx] for g in client_grads]).mean(0)
  server.step();train_loss=float(np.mean(losses));val_loss,episodes=validation(m,ds,vrng,args.debug_protocol,val_episodes);tl.append(train_loss);vl.append(val_loss);val_manifest.extend({"round":round_id,**e} for e in episodes);round_seconds.append(time.perf_counter()-round_started)
  print(f"run_seed={run} split_seed={args.split_seed} round={round_id:03d} train={train_loss:.4f} val={val_loss:.4f}")
  if round_id%5==0: (root/"checkpoints").mkdir(parents=True,exist_ok=True);torch.save(m.state_dict(),root/"checkpoints"/f"round_{round_id}.pth")
 (root/"checkpoints").mkdir(parents=True,exist_ok=True);torch.save(m.state_dict(),root/"checkpoints"/"final_model.pth");plot_loss_curve(tl,vl,root/"plots"/"loss_curve.png"); write(root / "loss_history.json", [{"round": index, "train_loss": train, "validation_loss": validation} for index, (train, validation) in enumerate(zip(tl, vl), 1)]);mon.stop();held={c:d["held_out_subtype"] for c,d in ds.items()}
 write(root/"metadata.json",{"method":"Federated BayPrAnoMeta","dataset":"VisA","protocol":PROTOCOL,"run_seed":run,"split_seed":args.split_seed,"clients":list(ds),"held_out_subtypes":held,"rounds":rounds,"episodes_per_client_per_round":client_episodes,"validation_episodes_per_round":val_episodes,"federated_aggregation":"existing server-side unweighted average of client meta-gradients with Adam update","backbone":"ImageNet-pretrained ResNet-18 with first max-pool replaced by 3x3 average pooling","embedding_dim":128,"inner_steps":INNER_STEPS,"inner_lr":INNER_LR,"server_lr":SERVER_LR,"k_shot":K_SHOT,"q_normal":Q_N,"q_anomaly":Q_A,"contrastive_loss":False,"score":"log p1(z) - log p0(z | S)","normal_label":0,"anomaly_label":1,"num_parameters":parameter_count})
 write(root/"episode_manifests"/"train.json",train_manifest);write(root/"episode_manifests"/"validation.json",val_manifest);write(root/"client_participation.json",[{"round":r,"clients":list(ds)} for r in range(1,rounds+1)]);write(root/"runtime_memory.json",{"run_seed":run,"split_seed":args.split_seed,"data_load_sec":data_sec,"training_sec":time.perf_counter()-started,"round_seconds":round_seconds,"average_seconds_per_communication_round":float(np.mean(round_seconds)),"peak_rss_mb":mon.peak,"peak_gpu_memory_mb":0.0,"device":str(DEVICE),"num_parameters":parameter_count});return held
def main():
 p=argparse.ArgumentParser();p.add_argument("--seed",type=int,choices=[1,2,3]);p.add_argument("--seeds",nargs="+",type=int);p.add_argument("--split-seed",type=int,default=SPLIT_SEED);p.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None,default=BASE_ROOT);p.add_argument("--output-root", "--run-root", dest="run_root",default=RUN_ROOT);p.add_argument("--split-csv",default="split_csv/2cls_highshot.csv"); p.add_argument("--subset-file", help='Paper image allowlist. Required for exact reproduction of the paper experiments.');p.add_argument("--category");p.add_argument("--validation-fraction",type=float,default=.30);p.add_argument("--debug-protocol",action="store_true");p.add_argument("--smoke-test",action="store_true");a=p.parse_args()
 if a.split_seed!=42:p.error("SPLIT_SEED is fixed at 42")
 if a.seed is not None and a.seeds is not None:p.error("Use either --seed or --seeds")
 a.seeds=[a.seed] if a.seed is not None else (a.seeds or [1,2,3])
 if a.smoke_test and a.seed is None and a.seeds==[1,2,3]:a.seeds=[1]
 if a.smoke_test:a.run_root=a.run_root+"_smoke"
 reference=None
 for seed in a.seeds:
  held=train(seed,a)
  if reference is None: reference=held
  elif reference!=held: raise RuntimeError("Held-out subtype differs across run seeds")
 write(Path(a.run_root)/PROTOCOL/"split_manifest.json",{"split_seed":a.split_seed,"run_seeds":a.seeds,"held_out_subtypes":reference})
if __name__=="__main__": main()
