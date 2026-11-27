# PatchCore-Controlled-5Shot-ViSA

This is the VisA counterpart of the MVTec representation-controlled 5-shot
PatchCore baseline. It preserves the exact frozen controlled encoder checkpoint,
ResNet18 architecture, average pooling, dense layer4 descriptors, 512-to-128
projection, preprocessing, nearest-neighbour image score, and 25% greedy coreset.

The dataset-specific change is limited to reading the corrected Full VisA
BayPrAnoMeta test manifests. Each episode uses exactly five matched normal
support images, 12 matched normal queries, and four matched anomaly queries.
The 245-descriptor support bank is reduced to 61 descriptors and rebuilt for
every episode. No extra normal or anomaly image enters the bank.

Smoke test:

```bash
# Run from the copied PatchCore_Controlled_5Shot_ViSA folder.
python run_patchcore_controlled_5shot_visa_three_seeds.py \
  --seeds 1 --split-seed 42 --categories candle --smoke-test --debug-protocol
```

Full sequential three-seed run:

```bash
caffeinate -i python run_patchcore_controlled_5shot_visa_three_seeds.py \
  --seeds 1 2 3 --split-seed 42 --debug-protocol
```
