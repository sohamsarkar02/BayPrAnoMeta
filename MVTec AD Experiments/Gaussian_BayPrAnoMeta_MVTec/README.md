# Gaussian BayPrAnoMeta — MVTec AD

This ablation uses the Full BayPrAnoMeta implementation as its baseline. The NIW
update, architecture, preprocessing,
fixed Student-t anomaly reference `p1`, differentiable inner adaptation,
post-adaptation posterior recomputation, optimization, and data protocol remain
unchanged.

The sole methodological change is the normal predictive density:

`Sigma_G = dof/(dof-2) * Sigma_n`

`log p0_G(z) = -0.5 * [d log(2 pi) + logdet(Sigma_G) + ||L^-1(z-mu_n)||^2]`

where `L L^T = Sigma_G`. The implementation uses a Cholesky triangular solve,
not an explicit inverse.

## Bottle/Seed-1 smoke test

```bash
python train_gaussian_baypranometa_three_seeds.py --seeds 1 --split-seed 42 --categories bottle --smoke-test --debug-protocol
python evaluate_gaussian_baypranometa_three_seeds.py --seeds 1 --split-seed 42 --categories bottle --smoke-test --debug-protocol
```

## Full Seed 1

```bash
caffeinate -i python train_gaussian_baypranometa_three_seeds.py --seeds 1 --split-seed 42 --debug-protocol
caffeinate -i python evaluate_gaussian_baypranometa_three_seeds.py --seeds 1 --split-seed 42 --debug-protocol
```

## Full Seeds 2 and 3 sequentially

```bash
caffeinate -i python train_gaussian_baypranometa_three_seeds.py --seeds 2 3 --split-seed 42 --debug-protocol
caffeinate -i python evaluate_gaussian_baypranometa_three_seeds.py --seeds 2 3 --split-seed 42 --debug-protocol
python aggregate_gaussian_baypranometa.py --seeds 1 2 3
```
