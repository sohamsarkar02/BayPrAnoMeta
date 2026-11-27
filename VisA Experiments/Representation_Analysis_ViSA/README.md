# Corrected VisA representation analysis

Evaluation-only counterpart of the corrected MVTec representation analysis.
It uses the current corrected Full Student-t BayPrAnoMeta and Proto_MAML VisA checkpoints for seeds 1, 2, and 3, together with their 36
verified-identical test manifests.

The script extracts post-adaptation 128D query embeddings without additional
normalization. It computes episode-level within-class variance and centroid
separation, and category-level effective rank from per-image means of repeated
task-adapted embeddings. Corrected t-SNE uses raw per-image mean embeddings for
both methods, no PCA, no coordinate standardization, matched random seeds, and
perplexity `min(30, n_unique - 1)`.

The shared engine provides deterministic progress reporting, compressed
embedding caches, atomic intermediate saves, and resume support.

Smoke test only:

```bash
# Run from the copied Representation_Analysis_ViSA folder.
python analyze_original_embedding_geometry_visa.py --smoke-test
```

Full local run:

```bash
# Run from the copied Representation_Analysis_ViSA folder.
caffeinate -i python analyze_original_embedding_geometry_visa.py --seeds 1 2 3
```
