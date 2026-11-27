# BayPrAnoMeta p0-only — MVTec AD

Evaluation-only ablation. It never trains or updates a checkpoint. It reuses
existing query diagnostics when complete and reevaluates existing checkpoints
against the already saved Full BayPrAnoMeta episode manifests only when query
likelihoods were not saved.

Scores:

- Full: `log_p1 - log_p0`
- p0-only: `-log_p0`

Run:

```bash
python run_p0_only_analysis.py
```

Outputs are written under `results/p0_only_ablation/`.
