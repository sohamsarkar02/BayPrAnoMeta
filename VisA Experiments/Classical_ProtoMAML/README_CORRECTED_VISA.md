# Proto_MAML for VisA

Source of truth: corrected MVTec `CLassical_ProtoMAML` implementation.

- VisA split: `split_csv/2cls_highshot.csv`
- Protocol: `revised_clean_official_2cls_highshot`
- Split seed: 42; run seeds: 1, 2, 3
- Labels: normal=0, anomaly=1
- Training query logit and evaluation score: positive squared distance to the prototype recomputed from adapted support embeddings; higher is anomalous
- The formerly incorrect negative-distance sign is not used.
- No per-category score inversion is performed.

Smoke test:

```bash
python train_protomaml_visa_three_seeds.py --seed 1 --smoke-test --debug-protocol
python evaluate_protomaml_visa_three_seeds.py --seed 1 --smoke-test --debug-protocol
```

Full sequential three-seed run:

```bash
python train_protomaml_visa_three_seeds.py --seeds 1 2 3 --split-seed 42 --debug-protocol
python evaluate_protomaml_visa_three_seeds.py --seeds 1 2 3 --split-seed 42 --debug-protocol
```

Output root: `runs/protomaml_visa_corrected_avgpool_three_seed/revised_clean_official_2cls_highshot/`.
