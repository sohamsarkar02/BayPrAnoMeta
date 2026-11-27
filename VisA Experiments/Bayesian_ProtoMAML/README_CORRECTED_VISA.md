# Corrected BayPrAnoMeta for VisA

This implementation ports the corrected MVTec BayPrAnoMeta logic without
changing the model, NIW/Student-t mathematics, anomaly reference, episode
sizes, transforms, or optimization hyperparameters.

## VisA protocol

- Source: `split_csv/2cls_highshot.csv`.
- Only paths both listed in that official split and physically present locally
  are used. Missing official paths are counted in each split-audit JSON.
- Official train normals/anomalies are split deterministically into train and
  validation pools with `SPLIT_SEED=42`.
- Official test normals and anomalies remain untouched by training/validation.
- VisA has no anomaly subtype labels in the supplied metadata. The recorded
  held-out group is `official_2cls_highshot_test_anomalies`; this is an official
  image-group holdout, not leave-one-anomaly-subtype-out.
- The high-shot split is required for this reduced local dataset because the
  locally available few-shot training-anomaly pools cannot support disjoint
  train/validation episodes while retaining `Q_A=4`.

## Smoke test (not run automatically)

```bash
python train_baypranometa_visa_three_seeds.py --smoke-test --debug-protocol
python evaluate_baypranometa_visa_three_seeds.py --smoke-test --debug-protocol
```

The smoke trainer uses the first discovered category, seed 1, one epoch, three
training episodes, and three validation episodes. The evaluator uses the same
category/seed and three test episodes.

## Full three-seed run

```bash
python train_baypranometa_visa_three_seeds.py --seeds 1 2 3 --split-seed 42 --debug-protocol
python evaluate_baypranometa_visa_three_seeds.py --seeds 1 2 3 --split-seed 42 --debug-protocol
```

Individual seeds can be run with `--seeds 1`, `--seeds 2`, or `--seeds 3`.

## Output root

`runs/baypranometa_visa_three_seed_fixed_split42_25ep_30epochs_100test/`

Each `revised_clean_official_2cls_highshot/seed_N/` contains metadata,
split-audit files, checkpoints (`best_model.pth`, periodic checkpoints, and
`final_model.pth`), loss CSV/curve, episode manifests, runtime/memory JSON and
CSV, category evaluation plots, query CSVs, error JSON, metrics, and results.
The protocol root also contains the fixed split manifest and three-seed AUROC
summary CSV/JSON.
