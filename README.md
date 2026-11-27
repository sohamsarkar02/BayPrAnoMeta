# BayPrAnoMeta: Bayesian Proto-MAML for Few-Shot Industrial Image Anomaly Detection

This repository accompanies the anonymous BayPrAnoMeta manuscript.

BayPrAnoMeta uses a small support set of normal images to adapt an embedding model to an industrial inspection task. The model is meta-trained on episodes. A Normal–Inverse-Wishart (NIW) prior and the adapted support embeddings define a Student-t posterior predictive model for normality. Query anomalies are ranked by the log-likelihood ratio `log p1(z) - log p0(z | S)`, where `p1` is a fixed anomaly reference and `p0` is the support-conditioned normal predictive density. The release also includes supervised-contrastive and federated extensions.

## Repository contents

| Location | Contents |
| --- | --- |
| `MVTec AD Experiments/` | MVTec AD implementations of Classical MAML, Proto_MAML, Full BayPrAnoMeta, Controlled PatchCore-25%, and standard unconstrained PatchCore |
| `VisA Experiments/` | VisA implementations of Classical MAML, Proto_MAML, Full BayPrAnoMeta, and Controlled PatchCore-25% |
| Both experiment trees | Gaussian, p0-only, NoAdapt, Proto-MAML + Bayesian Score, and Fixed-Lambda ablations. Contrastive, federated, and federated contrastive variants. Support-size sensitivity, anomaly-score diagnostics, representation analysis, and training-loss and runtime instrumentation and analysis code. |
| `protocols/visa/` | Official VisA split, paper image allowlist, and shared protocol validator |
| Analysis entry points | Paired category-level tests for BayPrAnoMeta versus Proto_MAML and for support-size comparisons |

## Installation

Use Python 3.11 or newer and install compatible PyTorch and torchvision builds for your platform. [requirements.txt](requirements.txt) records the versions used for release verification.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Standard PatchCore can use the verified scikit-learn brute-force squared-L2 fallback without FAISS. FAISS is optional.

## Datasets and experiment settings

Download MVTec AD and VisA separately. Supply their extracted roots through `--data-root`.

**MVTec AD.** In the corrected protocol, normal training and validation images come from `train/good`. The `test/good` images are reserved for evaluation. Split seed 42 selects one held-out anomaly subtype per eligible category. That subtype is excluded from training and validation. Within each test episode, support and query images are disjoint.

**VisA.** Exact paper reproduction requires both the [official two-class high-shot CSV](protocols/visa/2cls_highshot.csv) and the [paper image allowlist](protocols/visa/visa_2cls_highshot_available_images.txt). The CSV lists 10,821 images. The paper used the 2,173 images available in the audited experimental installation, comprising 1,302 official training images and 871 official test images. The loader preserves the official labels and split assignments. Split seed 42 divides the eligible official training images into meta-training and validation pools. The audit verified all 871 historical test identities path by path. The 1,302 training-side identities come from the audited installation and match historical counts by partition and label. Historical records did not independently verify each training-side path.

The VisA allowlist SHA-256 is `0471d94109ed6716cfd53ca05899b9d1374b210f504bf3e70e9629535ea25055`.

The main episodic experiments use `SPLIT_SEED=42` and run seeds `1, 2, 3`. Each test episode uses five normal support images, 12 normal queries, and four anomalous queries. The centralized budget is 30 epochs, with 25 training episodes and 20 validation episodes per epoch. Federated runs use all eligible clients, 25 episodes per client per round, and 30 communication rounds.

## Command setup

Run the examples from the repository root in Bash. Replace the three placeholder paths and keep the variables and functions in the same shell. Trainers write a protocol subdirectory under their output root. Main evaluators take the parent training root and write to a separate evaluation root. Several downstream commands take the protocol subdirectory itself, as shown below.

```bash
M="MVTec AD Experiments"
V="VisA Experiments"
MVTEC_DATA=/path/to/mvtec
VISA_DATA=/path/to/visa
OUT=/path/to/output
MO="$OUT/mvtec"
VO="$OUT/visa"
MP=revised_clean
VP=revised_clean_official_2cls_highshot
VISA_ARGS=(--data-root "$VISA_DATA" \
  --split-csv protocols/visa/2cls_highshot.csv \
  --subset-file protocols/visa/visa_2cls_highshot_available_images.txt)

mvtec_train() {
  local script="$1" name="$2"; shift 2
  python "$M/$script" --data-root "$MVTEC_DATA" --output-root "$MO/$name/train" "$@"
}
mvtec_eval() {
  local script="$1" name="$2"; shift 2
  python "$M/$script" --data-root "$MVTEC_DATA" \
    --checkpoint-root "$MO/$name/train" --output-root "$MO/$name/eval" "$@"
}
visa_train() {
  local script="$1" name="$2"; shift 2
  python "$V/$script" "${VISA_ARGS[@]}" --output-root "$VO/$name/train" "$@"
}
visa_eval() {
  local script="$1" name="$2"; shift 2
  python "$V/$script" "${VISA_ARGS[@]}" \
    --checkpoint-root "$VO/$name/train" --output-root "$VO/$name/eval" "$@"
}
```

## Main MVTec AD experiments

Train and evaluate Full BayPrAnoMeta first. The evaluation exports the episode plan and query diagnostics used in subsequent examples.

```bash
mvtec_train Bayesian_ProtoMAML/train_bayesian_protomaml_three_seeds_reduced.py full
mvtec_eval Bayesian_ProtoMAML/evaluate_bayesian_protomaml_three_seeds_reduced.py full \
  --save-episode-plan --save-query-diagnostics

mvtec_train CLassical_ProtoMAML/train_protomaml_three_seeds_corrected.py proto
mvtec_eval CLassical_ProtoMAML/evaluate_protomaml_three_seeds_corrected.py proto

mvtec_train 'Classical MAML/train_maml_three_seeds_corrected.py' maml
mvtec_eval 'Classical MAML/evaluate_maml_three_seeds_corrected.py' maml
```

Controlled PatchCore builds a bank from each episode's five normal support images and uses a 25% coreset. Generate the matching seed encoders once. The VisA controlled runner uses these same encoders.

```bash
python "$M/PatchCore/train_patchcore_controlled_three_seeds.py" \
  --encoder-only --output-root "$MO/patchcore-encoders"
python "$M/PatchCore/run_patchcore_controlled_5shot_three_seeds.py" \
  --encoder-root "$MO/patchcore-encoders/seed_models" \
  --episode-plan "$MO/full/eval/$MP" --output-root "$MO/patchcore-controlled"
```

Standard unconstrained PatchCore fits category-level banks from eligible training normals with pretrained WideResNet-50-2 features and a 10% coreset. The evaluator uses the Full model's query manifests for matched evaluation.

```bash
python "$M/Standard_Unconstrained_PatchCore_MVTec/train_standard_patchcore_three_seeds.py" \
  --data-root "$MVTEC_DATA" --output-root "$MO/patchcore-standard/fit"
python "$M/Standard_Unconstrained_PatchCore_MVTec/evaluate_standard_patchcore_three_seeds.py" \
  --data-root "$MVTEC_DATA" --bank-root "$MO/patchcore-standard/fit" \
  --reference-manifest-root "$MO/full/eval/$MP" --output-root "$MO/patchcore-standard/eval"
```

## Main VisA experiments

The VisA helpers pass both protocol files on every invocation. Complete the Full evaluation and the MVTec encoder generation before running Controlled PatchCore on VisA.

```bash
visa_train Bayesian_ProtoMAML/train_baypranometa_visa_three_seeds.py full
visa_eval Bayesian_ProtoMAML/evaluate_baypranometa_visa_three_seeds.py full \
  --save-episode-plan --save-query-diagnostics

visa_train Classical_ProtoMAML/train_protomaml_visa_three_seeds.py proto
visa_eval Classical_ProtoMAML/evaluate_protomaml_visa_three_seeds.py proto

visa_train 'Classical MAML/train_maml_visa_three_seeds.py' maml
visa_eval 'Classical MAML/evaluate_maml_visa_three_seeds.py' maml

python "$V/PatchCore_Controlled_5Shot_ViSA/run_patchcore_controlled_5shot_visa_three_seeds.py" \
  "${VISA_ARGS[@]}" --encoder-root "$MO/patchcore-encoders/seed_models" \
  --episode-plan "$VO/full/eval/$VP" --output-root "$VO/patchcore-controlled"
```

## Ablations

- **Gaussian** replaces the Student-t normal predictive density with a covariance-matched Gaussian.
- **NoAdapt** sets the inner-loop learning rate to zero while retaining the support-conditioned normal predictive model.
- **Proto-MAML + Bayesian Score** applies Bayesian scoring to the Proto_MAML encoder.
- **Fixed-Lambda** sets the posterior scale-update matrix to the prior matrix, `Lambda_n = Lambda_0`.
- **p0-only** ranks queries by `-log p0(z | S)`. See [Query diagnostics and p0-only](#query-diagnostics-and-p0-only) for the commands.

The Gaussian and NoAdapt evaluators use Full test manifests. The joint Bayesian ablation evaluators consume the Full, Proto_MAML, and Fixed-Lambda checkpoints. The VisA joint evaluator also reads fresh Full and Proto-MAML evaluation outputs.

```bash
mvtec_train Gaussian_BayPrAnoMeta_MVTec/train_gaussian_baypranometa_three_seeds.py gaussian
mvtec_eval Gaussian_BayPrAnoMeta_MVTec/evaluate_gaussian_baypranometa_three_seeds.py gaussian \
  --full-manifest-root "$MO/full/eval/$MP"
mvtec_train BayPrAnoMeta_NoAdapt_MVTec/train_baypranometa_noadapt_three_seeds.py noadapt
mvtec_eval BayPrAnoMeta_NoAdapt_MVTec/evaluate_baypranometa_noadapt_three_seeds.py noadapt \
  --full-manifest-root "$MO/full/eval/$MP"
mvtec_train Bayesian_Ablations/train_fixed_lambda_three_seeds.py fixed
python "$M/Bayesian_Ablations/evaluate_bayesian_ablations_three_seeds.py" \
  --data-root "$MVTEC_DATA" --full-run-root "$MO/full/train" \
  --proto-run-root "$MO/proto/train" --fixed-run-root "$MO/fixed/train" \
  --output-root "$MO/bayesian-ablations"

visa_train Gaussian_BayPrAnoMeta_ViSA/train_gaussian_baypranometa_visa_three_seeds.py gaussian
visa_eval Gaussian_BayPrAnoMeta_ViSA/evaluate_gaussian_baypranometa_visa_three_seeds.py gaussian \
  --full-run-root "$VO/full/eval"
visa_train BayPrAnoMeta_NoAdapt_ViSA/train_baypranometa_noadapt_visa_three_seeds.py noadapt
visa_eval BayPrAnoMeta_NoAdapt_ViSA/evaluate_baypranometa_noadapt_visa_three_seeds.py noadapt \
  --full-run-root "$VO/full/eval"
visa_train Bayesian_Ablations/train_fixed_lambda_visa_three_seeds.py fixed
python "$V/Bayesian_Ablations/evaluate_bayesian_ablations_visa_three_seeds.py" "${VISA_ARGS[@]}" \
  --full-run-root "$VO/full/train" --proto-run-root "$VO/proto/train" \
  --fixed-run-root "$VO/fixed/train" --full-evaluation-root "$VO/full/eval" \
  --proto-evaluation-root "$VO/proto/eval" --output-root "$VO/bayesian-ablations"
```

## Contrastive and federated variants

The contrastive variants add a supervised contrastive objective during training. In the federated simulation, object categories act as clients. Client examples remain within their client computations and the server aggregates only model/meta-gradient updates. This setup provides no formal privacy guarantee. Evaluation uses the trained global model.

```bash
mvtec_train Contrastive_Bayesian_ProtoMAML/train_contrastive_bayesian_protomaml_three_seeds.py contrastive
mvtec_eval Contrastive_Bayesian_ProtoMAML/evaluate_contrastive_bayesian_protomaml_three_seeds.py contrastive
mvtec_train 'Federated Bayesian_ProtoMAML/train_fed_bayesian_protomaml_three_seeds_reduced.py' federated
mvtec_eval 'Federated Bayesian_ProtoMAML/evaluate_fed_bayesian_protomaml_three_seeds_reduced.py' federated
mvtec_train FedCon_Bayesian_ProtoMAML/train_fedcon_bayesian_protomaml_three_seeds.py fedcon
mvtec_eval FedCon_Bayesian_ProtoMAML/evaluate_fedcon_bayesian_protomaml_three_seeds.py fedcon

visa_train Contrastive_Bayesian_ProtoMAML/train_contrastive_baypranometa_visa_three_seeds.py contrastive
visa_eval Contrastive_Bayesian_ProtoMAML/evaluate_contrastive_baypranometa_visa_three_seeds.py contrastive
visa_train 'Federated Bayesian_ProtoMAML/train_federated_baypranometa_visa_three_seeds.py' federated
visa_eval 'Federated Bayesian_ProtoMAML/evaluate_federated_baypranometa_visa_three_seeds.py' federated
visa_train FedCon_Bayesian_ProtoMAML/train_fedcon_baypranometa_visa_three_seeds.py fedcon
visa_eval FedCon_Bayesian_ProtoMAML/evaluate_fedcon_baypranometa_visa_three_seeds.py fedcon
```

## Support-size sensitivity

The main Full evaluations above use `--save-episode-plan` to export dataset-relative episode identities. They also supply the K=5 results. Run the sensitivity script once per remaining size. The support sets are nested, **S₁ ⊂ S₃ ⊂ S₅ ⊂ S₇**, and the queries are fixed across sizes. Historical episode manifests are not included.

```bash
for k in 1 3 7; do
  python "$M/Bayesian_ProtoMAML/evaluate_support_sensitivity_mvtec.py" \
    --data-root "$MVTEC_DATA" --support-size "$k" --episode-plan "$MO/full/eval/$MP" \
    --checkpoint-root "$MO/full/train/$MP" --output-root "$MO/support-sensitivity"
  python "$V/Bayesian_ProtoMAML/evaluate_support_sensitivity_visa.py" "${VISA_ARGS[@]}" \
    --support-size "$k" --episode-plan "$VO/full/eval/$VP" \
    --checkpoint-root "$VO/full/train/$VP" --output-root "$VO/support-sensitivity"
done
```

## Query diagnostics and p0-only

The Full evaluation examples use `--save-query-diagnostics`. The export records the squared embedding norm, `log p0(z | S)`, `log p1(z)`, and the likelihood-ratio score for each query. The p0-only scripts use these exported diagnostics together with the associated Full metrics and manifests.

```bash
python "$M/BayPrAnoMeta_p0Only_MVTec/run_p0_only_analysis.py" \
  --data-root "$MVTEC_DATA" --full-root "$MO/full/eval/$MP" \
  --query-diagnostics-root "$MO/full/eval/$MP" --output-root "$MO/p0-only"
python "$V/BayPrAnoMeta_p0Only_ViSA/run_p0_only_visa_analysis.py" "${VISA_ARGS[@]}" \
  --full-root "$VO/full/eval/$VP" --query-diagnostics-root "$VO/full/eval/$VP" \
  --output-root "$VO/p0-only"
```

## Representation analysis

Run these commands after the Full and Proto_MAML evaluations. They use matched test manifests and the corresponding checkpoints. The commands generate matched t-SNE visualizations and compute within-normal variance, within-anomaly variance, centroid distance, normalized separation, and effective rank in the original **128-dimensional** adapted embedding space. Effective rank uses covariance over per-image means of repeated adapted embeddings. t-SNE coordinates are used for visualization.

```bash
python "$M/Representation_Analysis_MVTec/analyze_original_embedding_geometry_mvtec.py" \
  --full-run-root "$MO/full/eval/$MP" --proto-run-root "$MO/proto/eval/$MP" \
  --full-checkpoint-root "$MO/full/train/$MP" --proto-checkpoint-root "$MO/proto/train/$MP" \
  --output-root "$MO/representation"
python "$V/Representation_Analysis_ViSA/analyze_original_embedding_geometry_visa.py" "${VISA_ARGS[@]}" \
  --full-run-root "$VO/full/eval/$VP" --proto-run-root "$VO/proto/eval/$VP" \
  --full-checkpoint-root "$VO/full/train/$VP" --proto-checkpoint-root "$VO/proto/train/$VP" \
  --output-root "$VO/representation"
```

## Paired statistical analyses

The primary paired unit is the object category. AUROC is averaged across the three seeds within each category before comparison. The analysis uses a two-sided Wilcoxon signed-rank test and a paired bootstrap confidence interval with 100,000 draws and seed 42. For the primary comparison, differences are Full BayPrAnoMeta minus Proto_MAML. For support-size comparisons, `--compare BASELINE CANDIDATE` reports candidate minus baseline. The support-size command initializes `default_rng(42)` separately for each comparison.

```bash
python "$M/paired_baypranometa_vs_protomaml.py" \
  --full-run-root "$MO/full/eval/$MP" --proto-run-root "$MO/proto/eval/$MP" \
  --output-root "$MO/paired" --bootstrap-resamples 100000 --bootstrap-seed 42

python "$M/paired_baypranometa_vs_protomaml.py" \
  --full-run-root "$VO/full/eval/$VP" --proto-run-root "$VO/proto/eval/$VP" \
  --output-root "$VO/paired" --bootstrap-resamples 100000 --bootstrap-seed 42

python paired_support_sensitivity.py \
  --sensitivity-csv "$MO/support-sensitivity/support_sensitivity_summary.csv" \
  --full-run-root "$MO/full/eval/$MP" --compare 1 3 --compare 3 5 --compare 5 7 --compare 1 7 \
  --output "$MO/paired/support-size.csv"
python paired_support_sensitivity.py \
  --sensitivity-csv "$VO/support-sensitivity/support_sensitivity_summary.csv" \
  --full-run-root "$VO/full/eval/$VP" --compare 1 3 --compare 3 5 --compare 5 7 --compare 1 7 \
  --output "$VO/paired/support-size.csv"
```

## Loss, runtime, and outputs

The retained neural trainers emit `loss_history.json` with the unsmoothed training and validation losses used for plotting, per epoch for centralized methods and per round for federated methods. The release excludes historical exact epoch-loss JSON files. Runtime and peak process RSS instrumentation is retained. Paper-scale timing was not rerun after packaging.

Fresh runs can write checkpoints, metric files, episode plans, query diagnostics, figures, loss histories, and runtime records. Generated outputs are intentionally excluded from the public repository. [`.gitignore`](.gitignore) excludes common experiment artifacts by default.

## Reproducibility status

All Python files passed syntax and import checks, and the main command-line scripts were checked using `--help`. We also tested checkpoint loading, forward passes, a single inner-loop adaptation step, and the Controlled PatchCore pipeline. These checks verify the main interfaces of the released code, but they are not a full reproduction of the experiments. Trained checkpoints and result CSV files from the original experiments are not included, and the full training and evaluation runs were not repeated for this release.

## PatchCore attribution

Standard PatchCore derives from Amazon Science's [patchcore-inspection](https://github.com/amazon-science/patchcore-inspection), licensed under Apache-2.0. The upstream [LICENSE](third_party/patchcore/LICENSE), [NOTICE](third_party/patchcore/NOTICE), and [local attribution](third_party/patchcore/ATTRIBUTION.txt) are under `third_party/patchcore/`. The exact historical upstream commit is unavailable from the retained source snapshot. The BayPrAnoMeta release includes a local scikit-learn brute-force squared-L2 fallback.

## License

BayPrAnoMeta-owned source code in this repository is released under the MIT License. See [LICENSE](LICENSE). Third-party PatchCore-derived code retains its upstream Apache-2.0 licensing and attribution under [third_party/patchcore/](third_party/patchcore/).
