#!/usr/bin/env python3
"""Standalone revised federated BayPrAnoMeta evaluation for run seeds 1, 2, 3."""
import argparse,csv,json,os,random,threading,time
from pathlib import Path
import higher,matplotlib.pyplot as plt,numpy as np,psutil,seaborn as sns,torch,torch.nn as nn,torchvision.models as models
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix,precision_recall_curve,roc_auc_score
from utils.utils_data_visa_leave1out import discover_categories,load_client_dataset_visa,load_images,transform_eval
from utils.utils_eval_stats import aggregate_with_se,compute_episode_metrics
from utils.utils_niw import log_student_t,niw_posterior
from utils.utils_plotting import plot_pr_curve,plot_roc_curve,plot_score_histogram

BASE_ROOT = None
RUN_ROOT="runs/fedcon_baypranometa_visa_three_seed_fixed_split42_25epclient_30rounds_100test"; PROTOCOL="revised_clean_official_2cls_highshot";SPLIT_SEED=42;DEVICE=torch.device("cpu")
TEST_EPISODES,INNER_LR,K_SHOT,Q_N,Q_A=100,5e-4,5,12,4
class Monitor:
 def __init__(self):self.p=psutil.Process(os.getpid());self.peak=0.;self.live=False
 def start(self):
  self.live=True
  def poll():
   while self.live:self.peak=max(self.peak,self.p.memory_info().rss/1024**2);time.sleep(.1)
  self.t=threading.Thread(target=poll,daemon=True);self.t.start()
 def stop(self):self.live=False;self.t.join()
class Net(nn.Module):
 def __init__(self):
  super().__init__();b=models.resnet18(weights=models.ResNet18_Weights.DEFAULT);b.maxpool=nn.AvgPool2d(kernel_size=3,stride=2,padding=1);self.features=nn.Sequential(*list(b.children())[:-1]);self.head=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.LayerNorm(128))
 def forward(self,x):return self.head(self.features(x).flatten(1))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("w") as f:json.dump(x,f,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))
def set_seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def stack(paths):
 x=load_images(paths,transform_eval)
 if len(x)!=len(paths):raise RuntimeError("Image loading failed")
 return torch.stack(x).to(DEVICE)
def ref(z):
 d=z.shape[1];return log_student_t(z,torch.zeros(d,device=DEVICE),100*torch.eye(d,device=DEVICE),torch.tensor(2.,device=DEVICE))
def episode(data,rng,debug):
 normal=rng.sample(data["test"]["normal"],K_SHOT+Q_N);s,qn,qa=normal[:K_SHOT],normal[K_SHOT:],rng.sample(data["test"]["anomaly"],Q_A)
 if debug:assert set(s).isdisjoint(qn) and all(p in set(data["test"]["anomaly"]) for p in qa)
 return s,qn,qa
def tsne(emb,y,out,category,seed):
 e=np.vstack(emb);y=np.asarray(y);p=min(30,max(5,(len(e)-1)//3));z=TSNE(n_components=2,perplexity=p,init="random",random_state=seed).fit_transform(e);plt.figure(figsize=(6,5));plt.scatter(z[y==0,0],z[y==0,1],s=5,label="Normal");plt.scatter(z[y==1,0],z[y==1,1],s=5,label="Anomaly");plt.legend();plt.title(f"{category} t-SNE");plt.tight_layout();plt.savefig(out/"tsne.png");plt.close()
def save_diag(root,records,aucs,reverse,protocol,seed,category,held):
 labels=np.asarray([r["label"] for r in records]);score=np.asarray([r["final_score"] for r in records]);p0=np.asarray([r["logp_normal"] for r in records]);p1=np.asarray([r["logp_anomaly"] for r in records]);n=labels==0;a=labels==1;stat=lambda x:{"mean":float(x.mean()),"median":float(np.median(x)),"std":float(x.std())}
 out=Path(root)/"diagnostics"/"FedCon_BayPrAnoMeta"/f"seed_{seed}"/category;out.mkdir(parents=True,exist_ok=True)
 summary={"method":"FedCon BayPrAnoMeta","run_seed":seed,"split_seed":SPLIT_SEED,"category":category,"held_out_subtype":held,"reported_mean_episode_AUROC":float(np.mean(aucs)),"reported_mean_episode_reversed_score_AUROC":float(np.mean(reverse)),"original_AUROC":float(roc_auc_score(labels,score)),"reversed_score_AUROC":float(roc_auc_score(labels,-score)),"primary_score_was_not_modified":True,"score_direction":"higher log p1(z) - log p0(z | adapted support) means more anomalous","protocol":protocol,"posterior_recomputed_after_adaptation":True,"normal_logp0":stat(p0[n]),"anomaly_logp0":stat(p0[a]),"normal_logp1":stat(p1[n]),"anomaly_logp1":stat(p1[a]),"normal_score":stat(score[n]),"anomaly_score":stat(score[a]),"diagnosis":"GENUINE_BELOW_CHANCE_PERFORMANCE","note":"Reversed AUROC is diagnostic only; repeated episode queries are not independent image observations."}
 write(out/"summary.json",summary)
 with (out/"query_scores.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 return summary
def evaluate_client(model,category,data,seed,root,debug,test_episodes):
 rng=random.Random(seed+2);aucs=[];prs=[];f1s=[];rev=[];scores=[];labels=[];paths=[];emb=[];records=[];manifest=[];overlap=0;model.eval()
 for eid in range(test_episodes):
  s,qn,qa=episode(data,rng,debug);overlap+=not set(s).isdisjoint(qn);names=qn+qa;y=[0]*Q_N+[1]*Q_A;xs,xq=stack(s),stack(names);opt=torch.optim.SGD(model.parameters(),lr=INNER_LR)
  with higher.innerloop_ctx(model,opt,copy_initial_weights=True,track_higher_grads=False) as (fm,diff):
   zs=fm(xs);mu,sg,df=niw_posterior(zs,validate=debug);diff.step(-log_student_t(zs,mu,sg,df).mean());zs=fm(xs);mu,sg,df=niw_posterior(zs,validate=debug);zq=fm(xq);p0=log_student_t(zq,mu,sg,df);p1=ref(zq);sc=(p1-p0).detach().cpu().numpy();p0=p0.detach().cpu().numpy();p1=p1.detach().cpu().numpy();zq=zq.detach().cpu().numpy()
  au,pr,f1=compute_episode_metrics(sc,y);aucs.append(float(au));prs.append(float(pr));f1s.append(float(f1));rev.append(float(roc_auc_score(y,-sc)));scores.extend(sc);labels.extend(y);paths.extend(names);emb.append(zq);manifest.append({"episode_id":eid,"support":s,"normal_query":qn,"anomaly_query":qa})
  records.extend({"episode_id":eid,"path":p,"label":int(l),"logp_normal":float(x0),"logp_anomaly":float(x1),"final_score":float(x)} for p,l,x0,x1,x in zip(names,y,p0,p1,sc))
 out=Path(root)/"evaluation"/category;out.mkdir(parents=True,exist_ok=True);lab=np.asarray(labels);sc=np.asarray(scores);am,ase=aggregate_with_se(aucs);pm,pse=aggregate_with_se(prs);fm,fse=aggregate_with_se(f1s);plot_roc_curve(lab,sc,out/"roc.png");plot_pr_curve(lab,sc,out/"pr.png");plot_score_histogram(lab,sc,out/"score_hist.png");threshold=float(np.percentile(sc[lab==0],95));pred=(sc>=threshold).astype(int);cm=confusion_matrix(lab,pred,labels=[0,1]);plt.figure(figsize=(5,4));sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Normal","Anomaly"],yticklabels=["Normal","Anomaly"]);plt.title(f"{category} | Held-out: {data['held_out_subtype']}");plt.tight_layout();plt.savefig(out/"confusion_matrix.png");plt.close();tsne(emb,lab,out,category,seed)
 errors=[{"path":paths[i],"filename":Path(paths[i]).name,"ground_truth":int(lab[i]),"prediction":int(pred[i]),"score":float(sc[i]),"threshold":threshold,"error_type":"false_positive" if lab[i]==0 else "false_negative"} for i in np.where(lab!=pred)[0]];write(out/"false_positives.json",[r for r in errors if r["error_type"]=="false_positive"]);write(out/"false_negatives.json",[r for r in errors if r["error_type"]=="false_negative"]);write(out/"episode_manifest.json",manifest)
 with (out/"query_scores.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 protocol={"train_test_normal_disjoint":set(data["train"]["normal"]).isdisjoint(data["test"]["normal"]),"episode_overlap_violations":int(overlap),"labels_match_expected":all(r["label"] in (0,1) for r in records),"test_anomalies_are_held_out_group":not (set(data["train"]["anomaly"])|set(data["validation"]["anomaly"])) & set(data["test"]["anomaly"])}
 metric={"category":category,"held_out_subtype":data["held_out_subtype"],"run_seed":seed,"split_seed":SPLIT_SEED,"auc_mean":float(am),"auc_se":float(ase),"pr_mean":float(pm),"pr_se":float(pse),"f1_mean":float(fm),"f1_se":float(fse),"pooled_auc":float(roc_auc_score(lab,sc)),"pooled_reversed_score_auc":float(roc_auc_score(lab,-sc)),"threshold_95_normal":threshold,"confusion_matrix":cm.tolist(),"n_false_positives":sum(r["error_type"]=="false_positive" for r in errors),"n_false_negatives":sum(r["error_type"]=="false_negative" for r in errors),"protocol":protocol,"diagnosis":"NOT_TRIGGERED_AUROC_GE_0_5"}
 if am<.5:
  metric["diagnosis"]="GENUINE_BELOW_CHANCE_PERFORMANCE";save_diag(Path(root),records,aucs,rev,protocol,seed,category,data["held_out_subtype"])
 write(out/"metrics.json",metric)
 print(f"seed={seed} {category}: AUC={am:.4f} +/- {ase:.4f} | PR={pm:.4f} +/- {pse:.4f}");return metric
CSV=["category","held_out_subtype","run_seed","split_seed","auc_mean","auc_se","pr_mean","pr_se","f1_mean","f1_se","pooled_auc","pooled_reversed_score_auc","threshold_95_normal","n_false_positives","n_false_negatives","diagnosis"]
def csvwrite(path,rows,cols):
 with Path(path).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore");w.writeheader();w.writerows(rows)
def combined_pr(root,categories):
 plt.figure(figsize=(7,5))
 for category in categories:
  with (Path(root)/"evaluation"/category/"query_scores.csv").open() as f: rows=list(csv.DictReader(f))
  labels=np.asarray([int(r["label"]) for r in rows]);scores=np.asarray([float(r["final_score"]) for r in rows]);precision,recall,_=precision_recall_curve(labels,scores);plt.plot(recall,precision,label=category,linewidth=1)
 plt.xlabel("Recall");plt.ylabel("Precision");plt.title("PR Curves Across Categories");plt.legend(ncol=2,fontsize=7);plt.tight_layout();plt.savefig(Path(root)/"evaluation"/"all_clients_pr_curves.png");plt.close()
def evaluate_seed(seed,args):
 set_seed(seed);root=Path(args.run_root)/PROTOCOL/f"seed_{seed}";ck=root/"checkpoints"/"final_model.pth"; root = Path(args.output_root)/PROTOCOL/f"seed_{seed}"
 if not ck.exists():raise FileNotFoundError(f"Missing checkpoint: {ck}")
 model=Net().to(DEVICE);model.load_state_dict(torch.load(ck,map_location=DEVICE));parameter_count=sum(p.numel() for p in model.parameters());started=time.perf_counter();mon=Monitor();mon.start();results={}
 clients=discover_categories(args.base_root,args.split_csv, subset_file=args.subset_file)
 if args.category:
  if args.category not in clients: raise ValueError(f"Unknown VisA client: {args.category}")
  clients=[args.category]
 if args.smoke_test and not args.category: clients=clients[:2]
 test_episodes=2 if args.smoke_test else args.test_episodes
 for category in clients:
  data=load_client_dataset_visa(args.base_root,category,seed=args.split_seed,validation_fraction=args.validation_fraction,split_csv=args.split_csv, subset_file=args.subset_file,audit_dir=root/"evaluation"/"protocol_audit")
  if data:results[category]=evaluate_client(model,category,data,seed,root,args.debug_protocol,test_episodes)
 mon.stop();elapsed=time.perf_counter()-started;write(root/"evaluation"/"results.json",results);np.save(root/"evaluation"/"results.npy",results);csvwrite(root/"evaluation"/"results.csv",[results[c] for c in sorted(results)],CSV);combined_pr(root,results);write(root/"evaluation"/"metadata.json",{"method":"FedCon BayPrAnoMeta","dataset":"VisA","clients":list(results),"run_seed":seed,"split_seed":args.split_seed,"protocol":PROTOCOL,"test_episodes":test_episodes,"checkpoint_path":str(ck),"backbone":"ImageNet-pretrained ResNet-18 with first max-pool replaced by 3x3 average pooling","embedding_dim":128,"k_shot":K_SHOT,"q_normal":Q_N,"q_anomaly":Q_A,"score":"log p1(z) - log p0(z | S)","normal_label":0,"anomaly_label":1,"num_parameters":parameter_count});write(root/"evaluation"/"runtime_memory.json",{"run_seed":seed,"split_seed":args.split_seed,"evaluation_sec":elapsed,"evaluation_seconds_per_episode":elapsed/max(1,test_episodes*len(results)),"peak_rss_mb":mon.peak,"peak_gpu_memory_mb":0.0,"device":str(DEVICE),"num_parameters":parameter_count,"test_episodes_per_category":test_episodes});return results
def main():
 p=argparse.ArgumentParser();p.add_argument("--seed",type=int,choices=[1,2,3]);p.add_argument("--seeds",nargs="+",type=int);p.add_argument("--split-seed",type=int,default=SPLIT_SEED);p.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None,default=BASE_ROOT);p.add_argument("--checkpoint-root", "--run-root", dest="run_root",default=RUN_ROOT); p.add_argument("--output-root", required=True, help="Directory for evaluation artifacts.");p.add_argument("--split-csv",default="split_csv/2cls_highshot.csv"); p.add_argument("--subset-file", help='Paper image allowlist. Required for exact reproduction of the paper experiments.');p.add_argument("--category");p.add_argument("--test-episodes",type=int,default=TEST_EPISODES);p.add_argument("--validation-fraction",type=float,default=.30);p.add_argument("--debug-protocol",action="store_true");p.add_argument("--smoke-test",action="store_true");a=p.parse_args()
 if a.split_seed!=42:p.error("SPLIT_SEED is fixed at 42")
 if a.seed is not None and a.seeds is not None:p.error("Use either --seed or --seeds")
 a.seeds=[a.seed] if a.seed is not None else (a.seeds or [1,2,3])
 if a.smoke_test and a.seed is None and a.seeds==[1,2,3]:a.seeds=[1]
 if a.smoke_test:a.run_root=a.run_root+"_smoke"
 allr={seed:evaluate_seed(seed,a) for seed in a.seeds};shared=set.intersection(*(set(x) for x in allr.values()));rows=[]
 for c in sorted(shared):
  held={allr[s][c]["held_out_subtype"] for s in a.seeds}
  if len(held)!=1:raise RuntimeError("Held-out subtype differs across seeds")
  vals=np.asarray([allr[s][c]["auc_mean"] for s in a.seeds]);row={"category":c,"held_out_subtype":held.pop(),"split_seed":a.split_seed,"n_seeds":len(a.seeds),"auroc_mean":float(vals.mean()),"auroc_std":float(vals.std(ddof=1)) if len(vals)>1 else 0.0};row.update({f"seed_{s}_auroc":float(allr[s][c]["auc_mean"]) for s in a.seeds});rows.append(row)
 cols=["category","held_out_subtype","split_seed","n_seeds"]+[f"seed_{s}_auroc" for s in a.seeds]+["auroc_mean","auroc_std"];base=Path(a.output_root)/PROTOCOL;csvwrite(base/"three_seed_results.csv",[r for s in a.seeds for r in allr[s].values()],CSV);csvwrite(base/"three_seed_auroc_summary.csv",rows,cols);write(base/"three_seed_auroc_summary.json",rows);write(base/"evaluation_split_manifest.json",{"split_seed":a.split_seed,"run_seeds":a.seeds,"held_out_subtypes":{c:allr[a.seeds[0]][c]["held_out_subtype"] for c in shared}})
if __name__=="__main__":main()
