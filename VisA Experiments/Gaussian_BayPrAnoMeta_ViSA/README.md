# Gaussian BayPrAnoMeta — VisA

The VisA ablation replaces the normal Student-t predictive density with a
covariance-matched Gaussian:

`Sigma_G = dof/(dof-2) * Sigma_n`

The Gaussian log-density uses a differentiable Cholesky triangular solve and
does not form an explicit inverse. NIW updates, `p1`, architecture, data split,
optimization, adaptation and post-adaptation posterior recomputation are
unchanged.

## Candle/Seed-1 smoke test

```bash
python train_gaussian_baypranometa_visa_three_seeds.py --smoke-test --category candle --debug-protocol
python evaluate_gaussian_baypranometa_visa_three_seeds.py --smoke-test --category candle --debug-protocol
```

## Full Seed 1

```bash
caffeinate -i python train_gaussian_baypranometa_visa_three_seeds.py --seeds 1 --split-seed 42 --debug-protocol
caffeinate -i python evaluate_gaussian_baypranometa_visa_three_seeds.py --seeds 1 --split-seed 42 --debug-protocol
```

## Full Seeds 2 and 3 sequentially

```bash
caffeinate -i python train_gaussian_baypranometa_visa_three_seeds.py --seeds 2 3 --split-seed 42 --debug-protocol
caffeinate -i python evaluate_gaussian_baypranometa_visa_three_seeds.py --seeds 2 3 --split-seed 42 --debug-protocol
python aggregate_gaussian_vs_student_t_visa.py --seeds 1 2 3
```
