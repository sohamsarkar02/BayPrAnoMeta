# BayPrAnoMeta NoAdapt — MVTec AD

The NoAdapt ablation sets the inner-loop learning rate to zero. The Full
BayPrAnoMeta baseline uses `INNER_LR = 5e-4`.

The inner loss and differentiable optimizer call remain in place, but the fast
parameters are unchanged (`theta_T = theta`). The five support normals still
construct the NIW posterior before the zero-rate inner step and again afterward.
The post-step support-dependent Student-t `p0`, fixed Student-t `p1`, outer loss,
architecture, optimizer, and data protocol are unchanged. The outer loss still
backpropagates into and updates `theta`.

## Bottle/Seed-1 smoke test

```bash
python train_baypranometa_noadapt_three_seeds.py --seeds 1 --split-seed 42 --categories bottle --smoke-test --debug-protocol
python evaluate_baypranometa_noadapt_three_seeds.py --seeds 1 --split-seed 42 --categories bottle --smoke-test --debug-protocol
```

## Full Seed 1

```bash
caffeinate -i python train_baypranometa_noadapt_three_seeds.py --seeds 1 --split-seed 42 --debug-protocol
caffeinate -i python evaluate_baypranometa_noadapt_three_seeds.py --seeds 1 --split-seed 42 --debug-protocol
```

## Full Seeds 2 and 3 sequentially

```bash
caffeinate -i python train_baypranometa_noadapt_three_seeds.py --seeds 2 3 --split-seed 42 --debug-protocol
caffeinate -i python evaluate_baypranometa_noadapt_three_seeds.py --seeds 2 3 --split-seed 42 --debug-protocol
python aggregate_baypranometa_noadapt.py --seeds 1 2 3
```
