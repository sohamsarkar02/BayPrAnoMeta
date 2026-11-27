# Corrected MVTec representation analysis

Evaluation-only comparison of the corrected Full BayPrAnoMeta and Proto_MAML post-adaptation 128D query embeddings for seeds 1, 2, and 3.

No model is retrained. Exact matched BayPrAnoMeta test manifests are used for
both methods. The analysis computes episode-level within-class variance and
centroid separation, plus category-level effective rank after averaging all
task-adapted occurrences of each physical query image.

Corrected t-SNE uses raw per-image mean 128D embeddings for both methods, with
no coordinate standardization and no PCA. Both panels use sklearn t-SNE with
perplexity 30, random initialization, and the same run seed.

Run:

```bash
# Run from the copied Representation_Analysis_MVTec folder.
caffeinate -i python analyze_original_embedding_geometry_mvtec.py
```

Smoke test:

```bash
python analyze_original_embedding_geometry_mvtec.py --smoke-test
```
