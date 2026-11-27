# Normal-only MAML versus supervised binary-class MAML

## Shared predictor

Let the shared encoder and projection network be `f_theta(x) in R^128`, and let the binary anomaly logit be

`a_theta(x) = w^T f_theta(x) + b`,

where label 0 denotes normal and label 1 denotes anomaly. Larger logits indicate greater anomaly likelihood.

## Case 1: Classical MAML with normal-only adaptation (main fair baseline)

The support set contains only `K=5` normal images:

`S_T^N = {x_i^N}_{i=1}^K`.

Its inner loss is

`L_inner^N(theta; S_T^N) = (1/K) sum_i BCEWithLogits(a_theta(x_i^N), 0)`.

The adapted parameters are

`theta_T^N = theta - alpha grad_theta L_inner^N(theta; S_T^N)`.

The query contains `Q_N=12` normals and `Q_A=4` anomalies, and the outer loss is

`L_query^N(theta_T^N) = (1/|Q_T|) sum_(x_j,y_j in Q_T) BCEWithLogits(a_(theta_T^N)(x_j), y_j)`.

At test time the anomaly score is

`s_N(x*) = a_(theta_T^N)(x*)`.

This formulation evaluates adaptation when only normal support is available. It is the fair comparison with Proto-MAML and BayPrAnoMeta under the manuscript's normal-only support protocol.

## Case 2: Supervised binary-class MAML (separate auxiliary experiment)

The support set contains labeled examples from both classes:

`S_T^B = {(x_i^N,0)}_{i=1}^{K_N} union {(x_i^A,1)}_{i=1}^{K_A}`,

with `K_N=5` and `K_A=5` in the separate implementation. Its inner loss is

`L_inner^B(theta; S_T^B) = (1/(K_N+K_A)) [sum_i BCEWithLogits(a_theta(x_i^N),0) + sum_i BCEWithLogits(a_theta(x_i^A),1)]`.

The adapted parameters are

`theta_T^B = theta - alpha grad_theta L_inner^B(theta; S_T^B)`.

The outer loss retains the same labeled query objective:

`L_query^B(theta_T^B) = (1/|Q_T|) sum_(x_j,y_j in Q_T) BCEWithLogits(a_(theta_T^B)(x_j), y_j)`.

At test time the anomaly score is

`s_B(x*) = a_(theta_T^B)(x*)`.

## Critical experimental distinction

Normal-only MAML receives no labeled anomaly support at adaptation time. Supervised binary-class MAML receives five labeled anomalies, including five images from the held-out anomaly subtype during evaluation. The anomaly support and anomaly query are disjoint, but the anomaly subtype is no longer unseen at adaptation time. Consequently, supervised binary MAML is not a like-for-like replacement for the main normal-only baseline and its results must be reported separately as an auxiliary supervised few-shot experiment.
