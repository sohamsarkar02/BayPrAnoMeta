# Corrected Classical MAML for VisA

Source of truth: corrected MVTec `Classical MAML` normal-only binary-classifier implementation.

- VisA split: `split_csv/2cls_highshot.csv`
- Protocol: `revised_clean_official_2cls_highshot`
- Split seed: 42; run seeds: 1, 2, 3
- Labels: normal=0, anomaly=1
- Training/evaluation score: adapted binary anomaly logit; higher is anomalous
- No per-category score inversion is performed.

Smoke test:

```bash
python train_maml_visa_three_seeds.py --seed 1 --smoke-test --debug-protocol
python evaluate_maml_visa_three_seeds.py --seed 1 --smoke-test --debug-protocol
```

Full sequential three-seed run:

```bash
python train_maml_visa_three_seeds.py --seeds 1 2 3 --split-seed 42 --debug-protocol
python evaluate_maml_visa_three_seeds.py --seeds 1 2 3 --split-seed 42 --debug-protocol
```

Output root: `runs/maml_visa_corrected_avgpool_three_seed/revised_clean_official_2cls_highshot/`.
