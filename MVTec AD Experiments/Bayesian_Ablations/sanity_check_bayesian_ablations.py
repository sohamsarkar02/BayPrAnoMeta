#!/usr/bin/env python3
"""One-category/one-episode sanity checks; no experiment outputs are written."""
import argparse,copy,random
from pathlib import Path
import higher,torch
from ablation_common import *
from evaluate_bayesian_ablations_three_seeds import adapt_and_score
def main():
 p=argparse.ArgumentParser();p.add_argument("--category",default="bottle");p.add_argument("--run-seed",type=int,default=1);p.add_argument("--split-seed",type=int,default=42);p.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None,default=BASE_ROOT);a=p.parse_args();set_seed(a.run_seed);d=load_revised_client(Path(a.base_root)/a.category,a.split_seed,.30);rng1=random.Random(a.run_seed+2);rng2=random.Random(a.run_seed+2);ep1=sample_episode(d,"test",rng1);ep2=sample_episode(d,"test",rng2);support,qn,qa=ep1;xs=load_stack(support,transform_eval);xq=load_stack(qn+qa,transform_eval);base=EmbeddingNet().to(DEVICE);fixed=copy.deepcopy(base);proto=copy.deepcopy(base);z=base(xs);mu,sigma,dof=fixed_lambda_posterior(z,True);kappa=KAPPA0+K_SHOT;expected=((kappa+1)/(kappa*(NU0_OFFSET+K_SHOT+1)))*torch.eye(128);values={m:adapt_and_score(model,m,xs,xq,True) for m,model in (("full",base),("fixed_lambda",fixed),("proto",proto))}
 checks={"category":a.category,"held_out_subtype":d["held_out_subtype"],"same_episode_reproducible":ep1==ep2,"support_query_disjoint":set(support).isdisjoint(qn),"held_out_anomalies_only":all(Path(p).parent.name==d["held_out_subtype"] for p in qa),"fixed_lambda_sigma_exact":bool(torch.allclose(sigma,expected)),"fixed_lambda_dof":int(dof),"all_scores_finite":all(np.isfinite(v["bayesian_score"]).all() and np.isfinite(v["euclidean_score"]).all() for v in values.values()),"records_per_method":len(qn)+len(qa),"label_convention":{"normal":0,"anomaly":1}}
 if not all(v for v in checks.values() if isinstance(v,bool)):raise RuntimeError(checks)
 print(checks)
if __name__=="__main__":main()
