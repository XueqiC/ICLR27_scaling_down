# V59 shared structure

CPU/NumPy; deterministic seed 59053; MAE and widths in nats. M/C/Q = math/code/QA. Part 1 and sharing use every registered held-out dev cell: pruning 17 sources, quantization six states, distillation 12 runs (48 checkpoints). All delivered-form MAEs are recomputed, with historical numbers used only for reproduction assertions.

| Part 1: family / comparator | Shared MAE M/C/Q | Specific MAE M/C/Q | Macro shared → specific | Verdict |
|---|---:|---:|---:|---|
| Pruning power / v53 power | 0.328/0.355/0.798 | 0.328/0.355/0.798 | 0.494 → 0.494 | within 0.02 |
| Quant separable / v55 2D | 1.953/2.185/4.749 | 0.914/1.910/4.371 | 2.962 → 2.398 | worse |
| Distill log(1+E) / v56 F2:L0 | 0.057/0.052/1.518 | 0.062/0.057/0.706 | 0.542 → 0.275 | worse |
| Distill log(1+E) / F2 same phi | 0.057/0.052/1.518 | 0.062/0.057/0.706 | 0.542 → 0.275 | worse |

Shared family means the same functional template, with parameters fitted separately by method and capability; it does not mean equal exponents across methods. Pruning's shared and delivered forms coincide. Quantization uses qmax^(-p)(g/128)^q with v55's original ridge parameterization; u is centered for the 2D comparator. Distillation uses log(1+E) with v53-style phi. Its native delivered comparator is F2:L0; all ten v56 forms are refitted and stored in JSON. The extra F2 same-phi row holds features and ridge preprocessing fixed. D0 is unavailable for Gemma: its standardized column is zero, so its effect cannot be estimated. N0 uses the counts already recorded by v56. The native F2 comparison also changes feature/scaling conventions; use the same-phi row to isolate shape. Verdicts use macro MAE and the inclusive 0.02 threshold; capability verdicts are in JSON.

| Part 2: quantity | Math | Code | QA |
|---|---:|---:|---:|
| LOSO MAE: shared gamma → per-cap gamma | 0.351 → 0.328 | 0.338 → 0.355 | 0.785 → 0.798 |
| LOSO MAE: shared terms → per-cap terms | 3.457 → 0.914 | 3.494 → 1.910 | 5.046 → 4.371 |
| Per-source gamma: min / median / max | 3.00/4.60/6.00 | 2.30/5.95/6.00 | 0.50/3.40/6.00 |
| Full-dev gamma: shared → per-cap | 3.30 → 3.75 | 3.30 → 3.05 | 3.30 → 3.05 |
| beta_intercept: estimate [95% CI] | 3.275 [0.325, 4.583] | 4.499 [1.030, 5.686] | -7.409 [-15.499, 5.032] |
| beta_logN0: estimate [95% CI] | 0.319 [-0.030, 0.700] | 0.273 [0.026, 0.502] | -0.481 [-1.052, 0.006] |
| beta_L0c: estimate [95% CI] | 3.850 [0.114, 5.554] | 5.357 [1.084, 6.920] | 5.486 [-3.535, 11.277] |
| beta_logD0: estimate [95% CI] | 0.250 [0.018, 0.380] | 0.347 [0.086, 0.462] | 0.106 [-0.279, 0.391] |
| gamma: estimate [95% CI] | 3.750 [3.200, 6.000] | 3.050 [2.500, 6.000] | 3.050 [2.450, 6.000] |

Pruning shares one gamma with separate four-vector amplitudes (13 vs 15 coefficients including exponents). Quantization shares the four nonconstant term coefficient vectors, retaining capability-specific four-vector offsets (28 vs 60). At two dev bit levels u^2 aliases the intercept; ridge identifies a solution but not unique unregularized term effects.

'Distribution of parameters across models' = separately fitted scalar amplitude and gamma for each source/capability, using all its dev densities. These describe observed models, not predictions for an unseen gamma. Grid-boundary counts M/C/Q: 3/8/9 of 17. 'Parameter confidence interval' = 95% percentile intervals from 1000 cluster-bootstrap draws of 17 sources with replacement; all densities/capabilities travel together. Standardizer and gamma are refitted; beta is mapped back to the full-dev standardized basis before taking percentiles. The intervals are conditional on the delivered power family and gamma grid, not model-selection uncertainty.

| Part 3: remaining cells | n/cap | K0 MAE M/C/Q | K1 MAE M/C/Q | Median MAE M/C/Q | Strength MAE M/C/Q | 80% PI: coverage/width K0 → K1 | 95% PI: coverage/width K0 → K1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prune dev LOSO | 67 | 0.406/0.435/0.972 | 17.634/32.592/43.992 | 0.510/0.564/0.659 | 0.610/0.741/0.840 | — | — |
| Prune confirmation | 6 | 0.359/0.330/0.990 | 0.160/0.508/0.579 | 0.410/0.315/0.317 | 0.669/0.729/0.849 | 77.8%/1.621 → 100.0%/3.629 | 100.0%/4.594 → 100.0%/210.610 |
| Prune pairs: 17-state OOF | 24 | 0.509/0.374/0.668 | 0.418/0.579/1.291 | 0.297/0.179/0.267 | 0.350/0.458/0.609 | — | — |
| Prune pairs: independent 13 | 24 | 0.495/0.394/1.231 | 0.570/0.594/1.152 | 0.303/0.206/0.257 | 0.383/0.649/0.803 | 66.7%/1.523 → 87.5%/4.277 | 86.1%/3.806 → 100.0%/103.209 |
| Quant 2D dev LOSO | 18 | 1.181/2.499/5.748 | 1.736/6.199/70.595 | 2.225/2.308/2.335 | — | — | — |
| Quant 2D bit_test | 12 | 0.263/0.641/1.337 | 0.351/2.880/16.096 | 0.590/0.767/0.556 | — | — | — |
| Quant 2D granularity_test | 18 | 0.592/1.389/3.272 | 0.868/3.490/40.561 | 1.206/1.322/1.266 | — | — | — |
| Quant 2D joint_test | 2 | 0.503/0.140/0.389 | 0.166/0.193/0.219 | 0.226/0.401/0.447 | — | — | — |
| Quant sep bit_test | 12 | 0.850/1.133/2.638 | 0.133/0.124/0.293 | 0.590/0.767/0.556 | — | — | — |
| Quant sep granularity_test | 18 | 1.522/1.778/3.969 | 0.660/0.440/0.769 | 1.206/1.322/1.266 | — | — | — |
| Quant sep joint_test | 2 | 0.774/0.485/0.915 | 0.368/0.431/0.120 | 0.226/0.401/0.447 | — | — | — |

Part 3 scores identical remaining cells for K0, K1 and baselines. K1 pruning fixes training-only gamma and sets A_c = observed_dL_c(d_cal)/[((1-d_cal)/0.3)^gamma_c]. The mildest density is 0.9 for 16 dev sources and 0.65 for 1B@96k; confirmation uses 0.85. Median and strength-only curves are refitted on each training fold, with v53 interpolation and unregularized strength-only fitting.

The confirmation panel is 410M@48k, 1.4B@112k, 6.9B@80k at 0.675/0.575 after calibration. Pairs are 1B and 6.9B at 32k/112k, scoring 0.8/0.75/0.7/0.65/0.6/0.55 after calibration at 0.9. Pairs belong to the registered dev panel. Their 17-source OOF results are descriptive; independent PI evaluation excludes all four together and refits from the other 13 sources.

Prediction intervals use per-capability absolute LOSO error quantiles (NumPy method='higher'), prediction +/- q80 or q95, evaluated without clipping. The compact table pools coverage and mean width over capabilities; JSON supplies per-capability coverage, widths, counts, radii, all residuals and per-source MAEs. K0 and K1 interval banks use the same post-calibration cells. No confirmation or pair source enters its own interval bank or any fit used to construct that bank. These are descriptive empirical intervals, not guaranteed nominal coverage; cells within sources are dependent.

Calibration cost: one compressed measurement per source, evaluated on three capabilities. Pruning: 17 for the dev LOSO, 3 for confirmation, 4 for each pair evaluation (the same four measured anchors are reused); 20 unique pruning sources total. Quantization: 6 b5_g256 measurements reused for dev/bit/granularity, plus one b5_g128 for the joint source. Dense reference losses are inputs for both K0 and K1 and are not counted as extra compressed measurements.

Fixed b5_g256 for six dev sources (5-bit endpoint, not the unique mildest cell: b5_g64 also exists). Joint source has no b5_g256; use available highest-bit b5_g128, score b3_g128/b4_g128 only.

Multiply the frozen source-specific 2D curve by measured_anchor/predicted_anchor. This is a one-scalar amplitude correction conditional on x; it does not identify the 2D shape. No clipping; fail if denominator <=1e-12 in magnitude.

Quant sep uses exact amplitude calibration of the separable law, with p and q from training. Small signed responses at the 5-bit anchor can make amplitude calibration unstable, especially when extrapolating toward 3 bits. Primary calibration tests exclude all target-source dev rows (five-source fits for bit/granularity, six for joint). The delivered full-six-source configuration split is additionally reported in JSON, with calibration anchors already used in its bit/granularity training.

Input SHA256 values, source membership, every fold fit/prediction, bootstrap samples, calibration records, and all per-capability statistics are in summary.json. Outputs are created exclusively; existing files are never overwritten.
