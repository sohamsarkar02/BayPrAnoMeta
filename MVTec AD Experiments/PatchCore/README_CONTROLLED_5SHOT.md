# PatchCore-Controlled-5Shot

This is a separate episodic baseline. It loads the exact existing controlled
PatchCore encoder checkpoint for each seed and preserves its ResNet18 backbone,
3x3 stride-2 average pool, dense layer4 features, 512-to-128 projection with
ReLU and LayerNorm, 224-pixel preprocessing, nearest-normal-patch image score,
and greedy 25% coreset implementation.

The only experimental change is the normal-data budget. Each corrected
BayPrAnoMeta `test_episodes.json` supplies exactly five normal support images and
the paired 12 normal plus four held-out-subtype anomaly queries. A 245-descriptor
candidate bank is built from those five images, reduced to 61 descriptors by the
existing 25% rule, used for that episode, and then discarded.

Smoke test:

```bash
python run_patchcore_controlled_5shot_three_seeds.py \
  --seeds 1 --split-seed 42 --categories bottle --smoke-test --debug-protocol
```

Full sequential run:

```bash
caffeinate -i python run_patchcore_controlled_5shot_three_seeds.py \
  --seeds 1 2 3 --split-seed 42 --debug-protocol
```
