# Standard Unconstrained PatchCore - MVTec AD

Official PatchCore detection configuration: pretrained WideResNet-50-2
(`wideresnet50`), `layer2` + `layer3`, official multi-scale patch embedding,
1024-to-1024 aggregation, patch size 3, one nearest neighbour, and official
approximate greedy coreset sampling at 10%.

Only the data/evaluation adapter is project-specific. Memory banks use the
corrected logical training-normal subset and evaluation reuses the saved
BayPrAnoMeta query manifests while ignoring episode support images.

The available FAISS wheel conflicts with PyTorch's OpenMP runtime on this Mac,
so the sole official-source compatibility patch uses an exact squared-L2
scikit-learn backend. It preserves FAISS `IndexFlatL2` distances and one-nearest-
neighbour scoring; only the search backend differs.
