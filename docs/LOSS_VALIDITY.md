# Capability-loss measurement validity

Endpoint: capability-conditioned teacher-forced LOSS. Primary/secondary pairs: MATH-500/GSM8K, MBPP/HumanEval, 2WikiMultihopQA/HotpotQA. No accuracy targets are fitted.

Preview with `python analysis/v23_loss_validity.py --model gemma3-270m --prune-density 0.9 0.8 0.7 0.6 --quant-bits 8 6 4 3 --dry-run`. For a separately launched measurement worker, replace `--dry-run` with `--device cuda:0`. An uncompressed reference is included automatically. Use `--adapter PATH` for V12/PEFT artifacts or `--checkpoint PATH_OR_HF_ID` for a full checkpoint. Rebuild this report with `python analysis/v23_loss_validity.py --summarize`.

Each L_c is the mean of per-example target CE divided by the scored target length. JSON also retains the legacy corpus-token-weighted mean. Both use the V6 measurement half, zero-shot prompt/target tokenization and fixed truncation; primary values are freshly measured, not imported from incompatible historical means. GSM8K keeps worked solutions, HumanEval keeps the canonical continuation, and HotpotQA keeps short answers.

Deltas subtract each benchmark's own uncompressed source baseline. Correlations use compression cells only, allowing different baselines/scales. Pruning and quantization are separate trajectories; dense anchors are not scored. Pearson/Spearman/Kendall need >=3 nonconstant paired responses. Ordering agreement excludes pairs tied on either benchmark; sign agreement retains negative deltas. N/A means undefined or insufficient data.

Law transfer: fit ΔL_primary = a (s/s_max)^γ on all eligible primary cells (γ grid 0.1–4); calibrate a nonnegative secondary scale on its two mildest cells, then score the stronger held-out secondary cells. This is benchmark transfer with target calibration, not zero-shot transfer or source-severity extrapolation. s=1-density for pruning, s=4^-bits for quantization. At least four cells are needed. Exclude the first ΔL>1 nat on either benchmark and all stronger cells from the law diagnostic; correlations retain the full sweep. MAE is in secondary nats/token; relative MAE divides by held-out secondary RMS ΔL. Zero MAE is the no-response baseline. This power-shape check is a validity diagnostic, not a validation of every paper law.

Groups keep model/checkpoint, tokenizer, probe protocol and baseline hashes separate. Correlation alone does not establish capability validity. These are descriptive point estimates; no seed/item uncertainty is claimed. Test/validation splits are used for secondary benchmarks; checkpoint training overlap must still be audited before claiming held-out benchmark validity.

Measurement files: 36. Compression trajectories: 6.

## gemma3-12b / dense / prune

Protocol `efa71a9c0156`; baseline `46a2d39a4c04`.

| Capability | Cells | Pearson | Spearman | Kendall | Order agreement (pairs) | Sign agreement |
|---|---:|---:|---:|---:|---:|---:|
| math | 5 | 0.9945 | 1 | 1 | 1 (10) | 1 |
| code | 5 | 0.9969 | 1 | 1 | 1 (10) | 1 |
| qa | 5 | 0.9626 | 0.9 | 0.8 | 0.9 (10) | 1 |

| Capability | Law transfer | Pre-cliff / held-out | γ | Scale | MAE | Relative MAE | Zero MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| math | ok | 5 / 3 | 3.82 | 1.344 | 0.1865 | 0.5014 | 0.3339 |
| code | ok | 5 / 3 | 3.909 | 1.522 | 0.09348 | 0.2793 | 0.2899 |
| qa | ok | 5 / 3 | 0.1 | 2.863 | 0.4133 | 1.403 | 0.288 |

| Checkpoint | Capability | Primary L_c | Primary ΔL | Secondary L_c | Secondary ΔL |
|---|---|---:|---:|---:|---:|
| dense_prune-d0.9 | math | 0.7167 | 0.01726 | 0.846 | 0.02161 |
| dense_prune-d0.9 | code | 0.9592 | 0.01955 | 0.4263 | 0.0165 |
| dense_prune-d0.9 | qa | 8.725 | 0.08396 | 8.996 | 0.1954 |
| dense_prune-d0.85 | math | 0.7501 | 0.05068 | 0.8874 | 0.06295 |
| dense_prune-d0.85 | code | 0.9867 | 0.04705 | 0.4535 | 0.04369 |
| dense_prune-d0.85 | qa | 9.027 | 0.3858 | 9.273 | 0.4721 |
| dense_prune-d0.8 | math | 0.8122 | 0.1127 | 0.9548 | 0.1303 |
| dense_prune-d0.8 | code | 1.037 | 0.09752 | 0.5114 | 0.1016 |
| dense_prune-d0.8 | qa | 9.027 | 0.3863 | 9.159 | 0.3583 |
| dense_prune-d0.75 | math | 1.061 | 0.3614 | 1.164 | 0.3401 |
| dense_prune-d0.75 | code | 1.143 | 0.2035 | 0.67 | 0.2602 |
| dense_prune-d0.75 | qa | 8.556 | -0.08472 | 8.593 | -0.2083 |
| dense_prune-d0.7 | math | 1.373 | 0.6736 | 1.356 | 0.5314 |
| dense_prune-d0.7 | code | 1.39 | 0.4506 | 0.9179 | 0.5081 |
| dense_prune-d0.7 | qa | 8.504 | -0.1372 | 8.504 | -0.2974 |

## gemma3-27b / dense / prune

Protocol `efa71a9c0156`; baseline `3c52d326e060`.

| Capability | Cells | Pearson | Spearman | Kendall | Order agreement (pairs) | Sign agreement |
|---|---:|---:|---:|---:|---:|---:|
| math | 5 | 0.9771 | 1 | 1 | 1 (10) | 1 |
| code | 5 | 0.9318 | 1 | 1 | 1 (10) | 1 |
| qa | 5 | 0.9628 | 0.7 | 0.6 | 0.8 (10) | 1 |

| Capability | Law transfer | Pre-cliff / held-out | γ | Scale | MAE | Relative MAE | Zero MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| math | insufficient: need >=4 pre-cliff compression cells | 3 / 0 | N/A | N/A | N/A | N/A | N/A |
| code | insufficient: need >=4 pre-cliff compression cells | 3 / 0 | N/A | N/A | N/A | N/A | N/A |
| qa | ok | 4 / 2 | 4 | 12.67 | 3.068 | 5.123 | 0.4532 |

| Checkpoint | Capability | Primary L_c | Primary ΔL | Secondary L_c | Secondary ΔL |
|---|---|---:|---:|---:|---:|
| dense_prune-d0.9 | math | 0.664 | 0.007141 | 0.822 | 0.01911 |
| dense_prune-d0.9 | code | 0.9478 | 0.01713 | 0.3991 | 0.007269 |
| dense_prune-d0.9 | qa | 9.525 | 0.04843 | 9.965 | 0.2767 |
| dense_prune-d0.85 | math | 0.6931 | 0.03631 | 0.8627 | 0.0598 |
| dense_prune-d0.85 | code | 0.9707 | 0.0401 | 0.4234 | 0.03165 |
| dense_prune-d0.85 | qa | 9.578 | 0.1016 | 10.14 | 0.4568 |
| dense_prune-d0.8 | math | 1.632 | 0.9747 | 1.407 | 0.6045 |
| dense_prune-d0.8 | code | 1.63 | 0.6997 | 1.136 | 0.744 |
| dense_prune-d0.8 | qa | 8.688 | -0.7878 | 8.843 | -0.8448 |
| dense_prune-d0.75 | math | 2.773 | 2.116 | 2.576 | 1.773 |
| dense_prune-d0.75 | code | 2.619 | 1.689 | 1.342 | 0.9497 |
| dense_prune-d0.75 | qa | 10.14 | 0.6634 | 9.75 | 0.06151 |
| dense_prune-d0.7 | math | 5.336 | 4.679 | 3.449 | 2.646 |
| dense_prune-d0.7 | code | 6.44 | 5.51 | 2.091 | 1.699 |
| dense_prune-d0.7 | qa | 12.58 | 3.099 | 12.28 | 2.596 |

## gemma3-4b / dense / prune

Protocol `efa71a9c0156`; baseline `bc76960fc061`.

| Capability | Cells | Pearson | Spearman | Kendall | Order agreement (pairs) | Sign agreement |
|---|---:|---:|---:|---:|---:|---:|
| math | 5 | 0.9957 | 1 | 1 | 1 (10) | 1 |
| code | 5 | 0.9961 | 1 | 1 | 1 (10) | 1 |
| qa | 5 | 0.9844 | 0.9 | 0.8 | 0.9 (10) | 1 |

| Capability | Law transfer | Pre-cliff / held-out | γ | Scale | MAE | Relative MAE | Zero MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| math | ok | 5 / 3 | 4 | 1.176 | 0.2142 | 0.5513 | 0.3321 |
| code | ok | 4 / 2 | 3.564 | 1.04 | 0.04651 | 0.2777 | 0.1588 |
| qa | ok | 5 / 3 | 0.1 | 1.377 | 0.1245 | 0.465 | 0.2304 |

| Checkpoint | Capability | Primary L_c | Primary ΔL | Secondary L_c | Secondary ΔL |
|---|---|---:|---:|---:|---:|
| dense_prune-d0.9 | math | 0.8171 | 0.02105 | 0.9486 | 0.01291 |
| dense_prune-d0.9 | code | 0.9761 | 0.002274 | 0.4596 | 0.007526 |
| dense_prune-d0.9 | qa | 8.423 | 0.08526 | 8.874 | 0.09058 |
| dense_prune-d0.85 | math | 0.8738 | 0.07773 | 0.9965 | 0.06081 |
| dense_prune-d0.85 | code | 1.017 | 0.04304 | 0.4986 | 0.04657 |
| dense_prune-d0.85 | qa | 8.581 | 0.243 | 9.114 | 0.331 |
| dense_prune-d0.8 | math | 0.9544 | 0.1583 | 1.077 | 0.1412 |
| dense_prune-d0.8 | code | 1.101 | 0.1271 | 0.5577 | 0.1056 |
| dense_prune-d0.8 | qa | 8.578 | 0.2402 | 9.162 | 0.3785 |
| dense_prune-d0.75 | math | 1.078 | 0.2816 | 1.18 | 0.2439 |
| dense_prune-d0.75 | code | 1.245 | 0.2707 | 0.664 | 0.212 |
| dense_prune-d0.75 | qa | 8.543 | 0.2046 | 9.047 | 0.2637 |
| dense_prune-d0.7 | math | 1.683 | 0.8865 | 1.547 | 0.6112 |
| dense_prune-d0.7 | code | 2.561 | 1.587 | 1.242 | 0.7902 |
| dense_prune-d0.7 | qa | 8.375 | 0.03711 | 8.832 | 0.04893 |

## muse-30b / dense / prune

Protocol `efa71a9c0156`; baseline `df2ba2180903`.

| Capability | Cells | Pearson | Spearman | Kendall | Order agreement (pairs) | Sign agreement |
|---|---:|---:|---:|---:|---:|---:|
| math | 5 | 0.9946 | 0.9 | 0.8 | 0.9 (10) | 1 |
| code | 5 | 0.9971 | 1 | 1 | 1 (10) | 0.8 |
| qa | 5 | 0.8772 | 0.9 | 0.8 | 0.9 (10) | 1 |

| Capability | Law transfer | Pre-cliff / held-out | γ | Scale | MAE | Relative MAE | Zero MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| math | ok | 5 / 3 | 4 | 0.402 | 0.05773 | 0.4703 | 0.09899 |
| code | ok | 5 / 3 | 4 | 0.8896 | 0.001835 | 0.0208 | 0.07581 |
| qa | ok | 5 / 3 | 1.827 | 1.102 | 0.07422 | 0.4876 | 0.151 |

| Checkpoint | Capability | Primary L_c | Primary ΔL | Secondary L_c | Secondary ΔL |
|---|---|---:|---:|---:|---:|
| dense_prune-d0.9 | math | 0.7654 | 0.00574 | 0.7249 | 0.004646 |
| dense_prune-d0.9 | code | 1.219 | -0.008475 | 0.4741 | 0.0002337 |
| dense_prune-d0.9 | qa | 8.584 | 0.02131 | 8.751 | 0.03464 |
| dense_prune-d0.85 | math | 0.7767 | 0.01712 | 0.7243 | 0.003991 |
| dense_prune-d0.85 | code | 1.239 | 0.01199 | 0.4824 | 0.008551 |
| dense_prune-d0.85 | qa | 8.654 | 0.09116 | 8.806 | 0.09008 |
| dense_prune-d0.8 | math | 0.7914 | 0.03176 | 0.7336 | 0.01334 |
| dense_prune-d0.8 | code | 1.254 | 0.02689 | 0.5016 | 0.02771 |
| dense_prune-d0.8 | qa | 8.687 | 0.1237 | 8.847 | 0.1309 |
| dense_prune-d0.75 | math | 0.8476 | 0.08802 | 0.8131 | 0.09285 |
| dense_prune-d0.75 | code | 1.301 | 0.07363 | 0.5375 | 0.06361 |
| dense_prune-d0.75 | qa | 8.779 | 0.2162 | 8.892 | 0.1766 |
| dense_prune-d0.7 | math | 0.9496 | 0.19 | 0.911 | 0.1908 |
| dense_prune-d0.7 | code | 1.376 | 0.1484 | 0.61 | 0.1361 |
| dense_prune-d0.7 | qa | 8.834 | 0.2714 | 8.861 | 0.1456 |

## olmo3-32b / dense / prune

Protocol `efa71a9c0156`; baseline `afab19017354`.

| Capability | Cells | Pearson | Spearman | Kendall | Order agreement (pairs) | Sign agreement |
|---|---:|---:|---:|---:|---:|---:|
| math | 5 | -0.5838 | -0.9 | -0.8 | 0.1 (10) | 0.8 |
| code | 5 | 0.9886 | 1 | 1 | 1 (10) | 1 |
| qa | 5 | -0.5718 | -0.6 | -0.4 | 0.3 (10) | 0.4 |

| Capability | Law transfer | Pre-cliff / held-out | γ | Scale | MAE | Relative MAE | Zero MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| math | ok | 5 / 3 | 3.033 | 2.663 | 0.02826 | 11.1 | 0.00229 |
| code | ok | 5 / 3 | 2.248 | 0.3854 | 0.001425 | 0.1183 | 0.01142 |
| qa | ok | 5 / 3 | 2.003 | 0 | 0.09438 | 0.9898 | 0.09438 |

| Checkpoint | Capability | Primary L_c | Primary ΔL | Secondary L_c | Secondary ΔL |
|---|---|---:|---:|---:|---:|
| dense_prune-d0.9 | math | 0.6384 | 0.0005302 | 0.7272 | 0.005497 |
| dense_prune-d0.9 | code | 1.294 | 0.0005986 | 0.5521 | 0.0005088 |
| dense_prune-d0.9 | qa | 7.843 | -0.004291 | 8.199 | -0.01965 |
| dense_prune-d0.85 | math | 0.6407 | 0.002843 | 0.7261 | 0.004386 |
| dense_prune-d0.85 | code | 1.304 | 0.01093 | 0.555 | 0.003344 |
| dense_prune-d0.85 | qa | 7.839 | -0.008869 | 8.186 | -0.03216 |
| dense_prune-d0.8 | math | 0.6432 | 0.005359 | 0.724 | 0.00235 |
| dense_prune-d0.8 | code | 1.309 | 0.01613 | 0.5586 | 0.007011 |
| dense_prune-d0.8 | qa | 7.868 | 0.02014 | 8.14 | -0.07823 |
| dense_prune-d0.75 | math | 0.6467 | 0.008783 | 0.7181 | -0.003619 |
| dense_prune-d0.75 | code | 1.315 | 0.02232 | 0.5625 | 0.01087 |
| dense_prune-d0.75 | qa | 7.848 | 0.0001571 | 8.125 | -0.09339 |
| dense_prune-d0.7 | math | 0.6552 | 0.01732 | 0.7226 | 0.0009017 |
| dense_prune-d0.7 | code | 1.332 | 0.03861 | 0.568 | 0.01637 |
| dense_prune-d0.7 | qa | 7.854 | 0.006163 | 8.107 | -0.1115 |

## olmo3-7b / dense / prune

Protocol `efa71a9c0156`; baseline `9f9ac5f59df4`.

| Capability | Cells | Pearson | Spearman | Kendall | Order agreement (pairs) | Sign agreement |
|---|---:|---:|---:|---:|---:|---:|
| math | 5 | 0.1586 | 0.4 | 0.4 | 0.7 (10) | 0.6 |
| code | 5 | 0.845 | 0.9 | 0.8 | 0.9 (10) | 0.8 |
| qa | 5 | 0.2745 | 0.6 | 0.4 | 0.7 (10) | 0.6 |

| Capability | Law transfer | Pre-cliff / held-out | γ | Scale | MAE | Relative MAE | Zero MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| math | ok | 5 / 3 | 3.404 | 1.075 | 0.01555 | 2.495 | 0.006195 |
| code | ok | 5 / 3 | 1.869 | 0.2485 | 0.01418 | 0.6058 | 0.01982 |
| qa | ok | 5 / 3 | 1.003 | 0 | 0.049 | 0.8454 | 0.049 |

| Checkpoint | Capability | Primary L_c | Primary ΔL | Secondary L_c | Secondary ΔL |
|---|---|---:|---:|---:|---:|
| dense_prune-d0.9 | math | 0.6872 | -0.0003472 | 0.8021 | 0.0008822 |
| dense_prune-d0.9 | code | 1.395 | -0.0006465 | 0.5811 | 0.001027 |
| dense_prune-d0.9 | qa | 7.568 | 0.01685 | 7.959 | -0.02033 |
| dense_prune-d0.85 | math | 0.6894 | 0.001908 | 0.804 | 0.002754 |
| dense_prune-d0.85 | code | 1.402 | 0.006378 | 0.5822 | 0.00211 |
| dense_prune-d0.85 | qa | 7.637 | 0.0862 | 7.988 | 0.009157 |
| dense_prune-d0.8 | math | 0.6943 | 0.006839 | 0.8067 | 0.005415 |
| dense_prune-d0.8 | code | 1.421 | 0.02493 | 0.5891 | 0.008969 |
| dense_prune-d0.8 | qa | 7.671 | 0.1197 | 8.016 | 0.03653 |
| dense_prune-d0.75 | math | 0.7031 | 0.01558 | 0.7951 | -0.006115 |
| dense_prune-d0.75 | code | 1.411 | 0.01527 | 0.5933 | 0.01324 |
| dense_prune-d0.75 | qa | 7.779 | 0.228 | 7.998 | 0.0189 |
| dense_prune-d0.7 | math | 0.7148 | 0.02727 | 0.8083 | 0.007056 |
| dense_prune-d0.7 | code | 1.428 | 0.03255 | 0.6173 | 0.03724 |
| dense_prune-d0.7 | qa | 7.663 | 0.1118 | 7.887 | -0.09158 |

