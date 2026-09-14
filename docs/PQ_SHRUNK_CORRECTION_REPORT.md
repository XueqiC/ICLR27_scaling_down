# A4 — shrunk source correction for pruning and quantization

**This branch is CLOSED.**

Positive paired gains with 95% intervals excluding zero: **0/9** leave-one-source-state-out pairs (primary; requires at least 5/9), **2/9** leave-one-size-out pairs, and **0/9** on both splits.

Maximum shrinkage (α = ∞) was selected in **14/108** outer folds. **The data selects finite corrections in most folds; this alone does not establish held-out gain.**

Within individual comparisons, the data prefers the median curve (∞ in most folds) for: pruning/code (size); pruning/qa (size); grouped_quantization/qa (size); per_channel_quantization/qa (size).

## Fixed predictor and selection

`prediction = unchanged V92 median(configuration) + X beta`

The correction minimizes `mean((target − median − X beta)^2) + α ||beta||²`. It is fitted only to the frozen median's residual. X is V92's unchanged standardized raw linear design: N0, D0, compression configuration, arm-local L0, and the existing B, V, W dense descriptors, plus intercept. This is one fixed dense_statistics-budget candidate. There are no additional input statistics, transformations, or interactions. Configuration and intercept coefficients belong only to the penalized additive correction; the saved median anchors never change. All coefficients, including the intercept, are penalized.

The grid was fixed before A4 evaluation: **1, 10, 100, 1,000, 10,000, 1,000,000, ∞**. Only grouped inner-training CV MAE selects α, separately per outer fold; exact ties favor stronger shrinkage. Source holdouts use inner source folds; size holdouts use inner size folds. Each inner fold computes its own median once from original inner-training responses, then freezes it across the grid. No median is ever fitted to residuals. Inner validation outcomes cannot enter the median, scaler, or residual fit. The outer median is obtained before selecting/fitting the correction and remains unchanged.

The decision rule was fixed before A4 outcomes: fewer than five of nine primary arm/capability pairs with positive gains and intervals excluding zero closes this branch. The primary split remains V92's leave-one-source-state-out split; size results are reported separately and never substituted to choose a favorable split. Maximum shrinkage in most folds also ends the branch with the median.

## Identical panel, published comparator, and scoring

**Publication discrepancy:** the current saved V92 numeric artifact has **459 primary rows, nine states in every arm, and 4/9 qualifying same-form statistics pairs**. A4 found stale coverage prose and a loader assertion from the pre-gap-fill panel; the historical count in its task differed from the current saved verdicts. The working-tree V92 Markdown and loader test now match the completed panel, with a separate legacy fixture case. The pinned V92 JSON retains its historical protocol note unchanged. Both whole-response candidates still lose to the median in **9/9** pairs. A4 compares directly with the current saved numeric panel and scores, pinned by SHA256 `29383f1b0a12dc2bd7c28479e8dd0257b72333f56b029812d9c3a0bec1d79b39`. It does not reconstruct a different historical six-state baseline. This discrepancy changes the coverage description, not A4's fixed predictor, grid, scoring, or decision rule.

Primary/size evaluation uses the common measured configuration grid: pruning densities .6/.7/.8/.9; grouped bits 3/4/5 × group sizes 64/128/256; per-channel bits 3/4/6/8. Budgets use exactly the same rows within each arm.

459 existing primary response rows are reused (108 pruning, 243 grouped quantization, 108 per-channel quantization), across math, code, and QA. No extra-strength rows are scored. The V91 descriptors and historical responses are read from existing JSON only. This analysis uses CPU NumPy with one BLAS thread; it loads no model, trains no model, accesses no GPU, and performs no new measurement or dense-statistic computation.

The comparator MAEs, predictions, and outer median anchors are **read directly** from `results/v92-input-comparison/summary.json`. The run checks all 54 input hashes, the V92 implementation hash, primary observations, ordered folds, and exact median predictions. It verifies the saved anchors using V92's original fitter on original responses; it does not replace the published comparison scores with a different recomputation.

MAE is in nats/reference token on signed compressed-minus-dense CE response. Positive paired gain means median MAE minus correction MAE. Configurations receive equal weight within a state, then states receive equal weight. V92's unchanged paired bootstrap resamples whole source states 20,000 times (seed 9201): nine clusters in every arm. **Size-split intervals also cluster by source state**, exactly as V92. These percentile intervals condition on fitted out-of-fold predictions; they do not include refitting uncertainty. This remains a small development panel within Pythia, not architecture-transfer evidence.

## Summary table

`source` = leave one source state out; `size` = leave one size out. Selected α distributions show every outer fold; exact fold assignments, inner scores, residual targets, coefficients, and prediction vectors are in `summary.json`.

| Split | Arm | Capability | V92 median MAE | Shrunk MAE | Paired gain | 95% CI | Selected α × folds |
|---|---|---|---:|---:|---:|---|---|
| source | pruning | math | 0.37594 | 0.37862 | -0.00267923 | [-0.0824295, 0.0999079] | 1×4, 100×5 |
| source | pruning | code | 0.47609 | 0.47422 | 0.00187382 | [-0.0152679, 0.0207492] | 10×6, 100×3 |
| source | pruning | qa | 0.52545 | 0.53089 | -0.00544291 | [-0.0251945, 0.0125634] | 10×8, 100×1 |
| source | grouped_quantization | math | 0.83278 | 0.83056 | 0.00222011 | [-0.00120637, 0.00641554] | 100×1, 1000×8 |
| source | grouped_quantization | code | 0.91010 | 0.91119 | -0.00108671 | [-0.0145134, 0.0127721] | 10×1, 100×2, 1000×6 |
| source | grouped_quantization | qa | 0.97957 | 0.95670 | 0.0228672 | [-0.000994702, 0.064114] | 10×1, 100×8 |
| source | per_channel_quantization | math | 1.19239 | 1.19251 | -0.000124739 | [-0.000886159, 0.000456844] | 1000×2, 10000×7 |
| source | per_channel_quantization | code | 1.36677 | 1.36729 | -0.000522324 | [-0.00147433, 0.000240076] | 1000×3, 10000×4, ∞×2 |
| source | per_channel_quantization | qa | 1.31277 | 1.31276 | 1.3978e-05 | [-0.000549667, 0.000595472] | 1000×6, 10000×2, ∞×1 |
| size | pruning | math | 0.37939 | 0.37974 | -0.000350405 | [-0.00308688, 0.00189125] | 100×2, ∞×1 |
| size | pruning | code | 0.54320 | 0.54239 | 0.000811283 | [0, 0.00166791] | 100×1, ∞×2 |
| size | pruning | qa | 0.59788 | 0.59753 | 0.000356584 | [-8.25192e-05, 0.00116472] | 100×1, ∞×2 |
| size | grouped_quantization | math | 0.82773 | 0.82707 | 0.000665645 | [0.000170599, 0.00117436] | 1000×2, 10000×1 |
| size | grouped_quantization | code | 0.91815 | 0.91768 | 0.000464003 | [0.000163023, 0.00079282] | 1000×1, 10000×1, ∞×1 |
| size | grouped_quantization | qa | 1.06468 | 1.06449 | 0.000183354 | [-0.000223037, 0.000815934] | 1000×1, ∞×2 |
| size | per_channel_quantization | math | 1.12514 | 1.12514 | -2.63948e-08 | [-9.65863e-07, 9.72195e-07] | 1e+06×2, ∞×1 |
| size | per_channel_quantization | code | 1.26931 | 1.26946 | -0.000146544 | [-0.000388807, 3.81776e-05] | 10000×2, 1e+06×1 |
| size | per_channel_quantization | qa | 1.20301 | 1.20287 | 0.000139656 | [-0.000116931, 0.00050044] | 1000×1, ∞×2 |

## Degenerate limits

At **α = ∞**, every correction coefficient is exactly zero. The predictor **is the published median curve**, its MAE is identical, and paired gain is **exactly 0 with interval [0, 0] by construction**. This is an identity, not evidence of improvement. The implementation returns the median prediction directly at this endpoint. At α = 0, the correction would be an unpenalized least-squares fit to the frozen median residual; that limit is documented but excluded from this strongly shrunk grid.

For finite α, the amount of departure also depends on the design and residual scale. The JSON records each coefficient vector and the source-weighted mean absolute departure from the median, alongside the selected α and its inner CV scores.

## Published whole-response candidates

These V92 dense_statistics MAEs are also read, without refitting, to contextualize the original whole-response OLS/ridge candidates.

| Split | Arm | Capability | V92 OLS MAE | V92 ridge MAE |
|---|---|---|---:|---:|
| source | pruning | math | 0.45718 | 0.42178 |
| source | pruning | code | 0.62085 | 0.53399 |
| source | pruning | qa | 0.87536 | 0.56917 |
| source | grouped_quantization | math | 1.25514 | 1.04438 |
| source | grouped_quantization | code | 1.36483 | 1.14031 |
| source | grouped_quantization | qa | 1.89662 | 1.18485 |
| source | per_channel_quantization | math | 1.99650 | 1.74369 |
| source | per_channel_quantization | code | 2.33925 | 2.03427 |
| source | per_channel_quantization | qa | 3.02246 | 1.77866 |
| size | pruning | math | 0.53750 | 0.51029 |
| size | pruning | code | 0.80617 | 0.73002 |
| size | pruning | qa | 0.74022 | 0.74223 |
| size | grouped_quantization | math | 1.71333 | 0.97925 |
| size | grouped_quantization | code | 1.47167 | 1.29525 |
| size | grouped_quantization | qa | 2.60624 | 1.63267 |
| size | per_channel_quantization | math | 2.05493 | 1.85700 |
| size | per_channel_quantization | code | 3.58101 | 2.31455 |
| size | per_channel_quantization | qa | 2.42867 | 2.25494 |

## Reuse and validation

Imported V92 code paths: `load_data`, `fields`, `budget_inputs (via design/predict)`, `Standardizer`, `design`, `fit (median_curve only)`, `predict`, `make_folds`, `state_mean`, `clustered`, `paired`, `bootstrap_weights`. **No V92 functions were copied.** V92.linear_fit leaves the intercept unpenalized, so it cannot give the required zero-correction limit; the A4 solver penalizes the whole residual coefficient vector.

Tests cover exact infinite-shrinkage predictions, all-coefficient shrinkage, training-only nested selection, frozen medians fitted only to original responses, published baseline read-through, matching panel/folds/scoring, and a non-zero CLI exit on failed validation. A negative scientific result is a successful run; execution or integrity failures raise errors and exit non-zero.

A4 test command (20 passed):

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest -q tests/test_a4_shrunk_source_correction.py
```

A4's additional V92 test run exposed a stale coverage assertion in `test_real_loader_uses_only_requested_sources_and_common_development_grid`. The follow-up documentation/test fix parameterizes that test against the current numeric artifact and the older paper-mirror fixture, preserving missing-state handling, common-grid selection, and the input-file allowlist. A4 tests verify exact equality to the actual published observations, hashes, folds, scores, and cluster counts.

## Interpretation

The orthogonal-residual variance finding is not a demonstration that the residual is random noise. It is not a pre-compression predictor because computing it needs the compressed model. The scalar response residual fitted here is a different quantity; predictors consume only V92's pre-compression inputs.

The shrunk correction has lower point MAE than both published whole-response OLS and ridge in 18/18 split/arm/capability comparisons. Keeping the median largely removes their excess error, but the incremental source correction fails the fixed majority criterion against that median.

This check tests the specified residual correction on the existing information budget. Its result does not establish that source information is worthless or that the unexplained response is random noise.

**This branch is CLOSED.**
