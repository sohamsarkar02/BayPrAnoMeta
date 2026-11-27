# BayPrAnoMeta NoAdapt — VisA

The NoAdapt ablation sets the inner-loop learning rate to zero. The Full
BayPrAnoMeta baseline uses `INNER_LR = 5e-4`.

The five support normals still construct the NIW posterior before the zero-rate
inner step and after it. Student-t `p0`, fixed Student-t `p1`, outer loss,
architecture, optimizer, official VisA split and loss normalization are
unchanged. The outer loss still trains the encoder.

## Smoke test

```bash
python train_baypranometa_noadapt_visa_three_seeds.py --smoke-test --category candle --debug-protocol
python evaluate_baypranometa_noadapt_visa_three_seeds.py --smoke-test --category candle --debug-protocol
```

## Full Seed 1

```bash
caffeinate -i python train_baypranometa_noadapt_visa_three_seeds.py --seeds 1 --split-seed 42 --debug-protocol
caffeinate -i python evaluate_baypranometa_noadapt_visa_three_seeds.py --seeds 1 --split-seed 42 --debug-protocol
```

## Full Seeds 2 and 3 sequentially

```bash
caffeinate -i python train_baypranometa_noadapt_visa_three_seeds.py --seeds 2 3 --split-seed 42 --debug-protocol
caffeinate -i python evaluate_baypranometa_noadapt_visa_three_seeds.py --seeds 2 3 --split-seed 42 --debug-protocol
python aggregate_full_vs_noadapt_visa.py --seeds 1 2 3
```
