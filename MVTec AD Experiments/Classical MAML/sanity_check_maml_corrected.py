#!/usr/bin/env python3
"""One real episode: protocol, Higher, gradient, label, and score checks only."""
import argparse, random
from pathlib import Path
import higher, torch, torch.nn as nn
from maml_three_seed_common import *
from utils.utils_data_fedcontrast_leave1out import load_client_dataset_leave1out, load_images, transform_eval

def main():
    p=argparse.ArgumentParser();p.add_argument("--category",default="bottle");p.add_argument("--run-seed",type=int,default=1)
    p.add_argument("--split-seed",type=int,default=42);p.add_argument("--data-root", "--base-root", dest="base_root", required=BASE_ROOT is None,default=BASE_ROOT);p.add_argument("--validation-fraction",type=float,default=.30);args=p.parse_args()
    set_seed(args.run_seed);data=load_client_dataset_leave1out(Path(args.base_root)/args.category,seed=args.split_seed,protocol=PROTOCOL,validation_fraction=args.validation_fraction)
    if data is None:raise RuntimeError("Category cannot support the requested protocol.")
    support,qn,qa=sample_episode(data,"test",random.Random(args.run_seed+4000));xs=stack(support,transform_eval,load_images);xq=stack(qn+qa,transform_eval,load_images)
    labels=torch.tensor([0]*Q_N+[1]*Q_A,dtype=torch.float32);model=MAMLNet().to(DEVICE);before={n:p.detach().clone() for n,p in model.named_parameters()};opt=torch.optim.SGD(model.parameters(),lr=INNER_LR)
    with higher.innerloop_ctx(model,opt,copy_initial_weights=False,track_higher_grads=True) as (fmodel,diffopt):
        fast_before=[p.detach().clone() for p in fmodel.parameters()];support_logits=fmodel(xs);inner=nn.BCEWithLogitsLoss()(support_logits,torch.zeros_like(support_logits));diffopt.step(inner)
        fast_changed=any(not torch.equal(a,b.detach()) for a,b in zip(fast_before,fmodel.parameters()));scores=fmodel(xq);outer=nn.BCEWithLogitsLoss()(scores,labels)
    original_unchanged=all(torch.equal(before[n],p.detach()) for n,p in model.named_parameters());model.zero_grad();outer.backward();nonzero=any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters())
    checks={"category":args.category,"held_out_subtype":data["held_out_subtype"],"split_seed":args.split_seed,"run_seed":args.run_seed,
        "train_test_normal_disjoint":set(data["train"]["normal"]).isdisjoint(data["test"]["normal"]),"held_out_absent_from_training":data["held_out_subtype"] not in data["seen_subtypes"],
        "support_query_disjoint":set(support).isdisjoint(qn),"labels":{"normal":0,"anomaly":1},"fast_weights_changed":fast_changed,
        "original_model_unchanged_by_inner_step":original_unchanged,"outer_meta_gradient_nonzero":nonzero,"scores_finite":bool(torch.isfinite(scores).all()),
        "score_convention":"larger adapted binary logit means more anomalous"}
    if not all(v for k,v in checks.items() if isinstance(v,bool)):raise RuntimeError(checks)
    print(checks)
if __name__=="__main__":main()
