# BayPrAnoMeta p0-only VisA ablation

This is an evaluation-only analysis of the completed corrected Full Student-t
BayPrAnoMeta VisA results. It does not load or modify checkpoints and does not
generate new episodes.

For each saved Full query it compares:

- Full score: `log_p1 - log_p0`
- p0-only score: `-log_p0`

Both scores use the convention that larger values indicate greater anomaly
evidence. Metrics follow the Full evaluation convention: compute AUROC, AUPRC,
and oracle F1 for each of the 100 episodes, then report the episode mean and SE.

Run:

```bash
# Run from the copied BayPrAnoMeta_p0Only_ViSA folder.
python run_p0_only_visa_analysis.py
```

Derived files are written only under `results/p0_only_ablation/` in this folder.
