# V14 Stage-A refitting report

This is an artifact-only, CPU refit. No model inference or new measurement was run.

Held-out pruning errors use cell-bootstrap 95% percentile intervals from 1,000 resamples. MAE is in CE nats; relative error is MAE divided by the mean absolute held-out damage and is dimensionless.

## Part 1 — pruning law battery

Complete models: 12; measured cells: 360; pre-cliff cells: 180.

Deleted mass is normalized by each capability spectrum's total mass and linearly interpolated within the boundary count bin.

### Shared-gamma in-sample parameter audit

These parameter fits are descriptive; the validation tables below determine claim status.

| model | family | cells | gamma | log-space R² | B by capability |
|---|---|---:|---:|---:|---|
| Qwen3-0.6B | qwen3 | 18 | 1.4389 | 0.9790 | code=164.1, math=124.4, qa=95.25 |
| Qwen3-1.7B | qwen3 | 21 | 1.0992 | 0.9619 | code=55.93, math=41.9, qa=83.02 |
| Qwen3-4B | qwen3 | 18 | 2.2981 | 0.9605 | code=797.1, math=787.6, qa=539.8 |
| gemma3-12b | gemma3 | 8 | 1.8153 | 0.9850 | code=755.7, math=1119, qa=710.8 |
| gemma3-1b | gemma3 | 12 | 1.5753 | 0.8721 | code=316.4, math=373.7, qa=59.64 |
| gemma3-270m | gemma3 | 11 | 1.4005 | 0.9853 | code=460, math=343.3, qa=171.4 |
| gemma3-27b | gemma3 | 7 | 1.4273 | 0.9725 | code=1314, math=1158, qa=692.6 |
| gemma3-4b | gemma3 | 12 | 1.7499 | 0.9187 | code=903.3, math=644.2, qa=108.5 |
| gemma4-31b | gemma4 | 11 | 1.3544 | 0.9051 | code=65, math=66.73, qa=124.3 |
| muse-30b | muse_glimmer | 16 | 2.1394 | 0.9760 | code=491.5, math=365.9, qa=278.5 |
| olmo3-32b | olmo3 | 24 | 0.8359 | 0.7912 | code=4.066, math=1.264, qa=1.788 |
| olmo3-7b | olmo3 | 22 | 1.4346 | 0.8901 | code=21.19, math=15.68, qa=28.01 |

### V-a

fit density >= 0.55; predict deeper pre-cliff cells

| form | cells | MAE nats [95% CI] | relative error [95% CI] |
|---|---:|---:|---:|
| P1_raw_sparsity | 56 | 3.1796 [1.6126, 5.2267] | 0.9172 [0.5406, 1.3191] |
| P2_deleted_mass | 56 | 3.1836 [1.5174, 5.2559] | 0.9184 [0.5392, 1.3260] |
| P3_shared_gamma | 63 | 2.5260 [1.0935, 4.5754] | 0.7718 [0.4111, 1.1771] |
| baseline_P1_direct | 45 | 3.9031 [0.8893, 7.4852] | 1.0977 [0.3366, 1.8051] |
| baseline_P2_direct | 45 | 4.2359 [1.1694, 8.3232] | 1.1912 [0.4348, 1.9072] |

Paired comparisons use only cells predicted by both forms.

| comparison | paired cells | candidate MAE | baseline MAE | candidate - baseline MAE [95% CI] |
|---|---:|---:|---:|---:|
| P2_vs_direct_P2 | 45 | 3.0687 | 4.2359 | -1.1672 [-3.6071, 1.0865] |
| P3_vs_direct_P2 | 45 | 3.1122 | 4.2359 | -1.1237 [-2.5125, -0.1088] |
| P2_vs_P1 | 56 | 3.1836 | 3.1796 | 0.0040 [-0.2514, 0.2762] |
| P3_vs_P1 | 56 | 2.6760 | 3.1796 | -0.5036 [-1.9309, 0.7545] |

### V-b

leave largest model out; transfer exponent and calibrate B on one shallow target cell

| form | cells | MAE nats [95% CI] | relative error [95% CI] |
|---|---:|---:|---:|
| P1_raw_sparsity | 40 | 2.1505 [1.2276, 3.2148] | 1.4880 [0.8143, 2.8672] |
| P2_deleted_mass | 40 | 2.2233 [1.2539, 3.3747] | 1.5383 [0.7494, 3.2569] |
| P3_shared_gamma | 40 | 1.4080 [0.8792, 2.0051] | 0.9742 [0.5717, 1.7285] |
| baseline_P1_direct | 40 | 1.1448 [0.7448, 1.5877] | 0.8044 [0.5900, 1.1316] |
| baseline_P2_direct | 40 | 1.1285 [0.7402, 1.5530] | 0.7930 [0.5657, 1.1703] |

Paired comparisons use only cells predicted by both forms.

| comparison | paired cells | candidate MAE | baseline MAE | candidate - baseline MAE [95% CI] |
|---|---:|---:|---:|---:|
| P2_vs_direct_P2 | 37 | 2.2641 | 1.1938 | 1.0703 [0.1828, 2.2697] |
| P3_vs_direct_P2 | 37 | 1.3785 | 1.1938 | 0.1847 [-0.1877, 0.6388] |
| P2_vs_P1 | 40 | 2.2233 | 2.1505 | 0.0727 [-0.0872, 0.2796] |
| P3_vs_P1 | 40 | 1.4080 | 2.1505 | -0.7425 [-1.3321, -0.2658] |

### V-c

leave one family out; transfer exponent and calibrate B on one shallow target cell

| form | cells | MAE nats [95% CI] | relative error [95% CI] |
|---|---:|---:|---:|
| P1_raw_sparsity | 144 | 10.3235 [5.5754, 16.3862] | 3.5869 [1.9272, 6.0221] |
| P2_deleted_mass | 144 | 4.2252 [2.7446, 6.2135] | 1.4681 [0.9410, 2.2379] |
| P3_shared_gamma | 144 | 2.3105 [1.6559, 3.2145] | 0.8028 [0.5495, 1.1667] |
| baseline_P1_direct | 144 | 10.4920 [5.1843, 17.0705] | 3.6752 [1.9220, 5.8942] |
| baseline_P2_direct | 144 | 7.7401 [4.5090, 11.6050] | 2.7112 [1.6469, 4.0246] |

Paired comparisons use only cells predicted by both forms.

| comparison | paired cells | candidate MAE | baseline MAE | candidate - baseline MAE [95% CI] |
|---|---:|---:|---:|---:|
| P2_vs_direct_P2 | 136 | 3.8913 | 8.1855 | -4.2941 [-8.3788, -0.6292] |
| P3_vs_direct_P2 | 136 | 2.1671 | 8.1855 | -6.0184 [-10.1499, -2.4520] |
| P2_vs_P1 | 144 | 4.2252 | 10.3235 | -6.0983 [-9.9959, -3.0489] |
| P3_vs_P1 | 144 | 2.3105 | 10.3235 | -8.0130 [-12.7550, -4.0369] |

### V-d sign prediction

| group | correct / cells | accuracy [95% Wilson CI] |
|---|---:|---:|
| pooled | 73 / 108 | 0.6759 [0.5829, 0.7568] |
| gemma3 | 32 / 45 | 0.7111 [0.5663, 0.8227] |
| gemma4 | 4 / 9 | 0.4444 [0.1888, 0.7333] |
| muse_glimmer | 9 / 9 | 1.0000 [0.7009, 1.0000] |
| olmo3 | 11 / 18 | 0.6111 [0.3862, 0.7969] |
| qwen3 | 17 / 27 | 0.6296 [0.4423, 0.7847] |

## Part 2 — quantization forms

Eligible positive-damage model/capability cells: 16 across 9 artifact models. Bit 8 is retained as its observed near-zero anchor, including small negative values.

| family | fixed 4^-b MAE | learned exponential MAE | quadratic MAE | winner |
|---|---:|---:|---:|---|
| gemma3 | 9.8854 | 7.9493 | 13.6959 | learned_exponential |
| muse_glimmer | 2.4346 | 6.7670 | 5.9488 | fixed_4^-b |
| olmo3 | 2.3744 | 1.5100 | 3.2543 | learned_exponential |
| qwen3 | 5.3313 | 4.3666 | 7.3619 | learned_exponential |
| pooled | 7.6307 | 6.7273 | 10.8873 | learned_exponential |

## Part 3 — recovery refit

R² is computed only over nonzero-budget points; the D_R=0 damaged anchor is exact by construction.

| model / run / capability | r | beta | D0 tokens | R² | bound? | largest observed | largest predicted from smaller two | abs. error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma3-270m / prune_0.6_c4 / code | 0.2261 | 1.2189 | 150296 | 1.0000 | no | 1.9954 | 2.1304 | 0.1350 |
| gemma3-270m / prune_0.6_c4 / math | 0.1753 | 1.5631 | 227222 | 1.0000 | no | 1.2450 | 1.3177 | 0.0727 |
| gemma3-270m / prune_0.6_c4 / qa | 0.0000 | 1111.9846 | 137518907 | 0.0596 | yes | -0.1657 | 0.0000 | 0.1657 |
| gemma3-270m / prune_0.6_traces / code | 0.1734 | 8.0336 | 2126 | 0.0000 | no | 1.6722 | 1.3447 | 0.3275 |
| gemma3-270m / prune_0.6_traces / math | 0.1888 | 644.6289 | 84019705 | 0.7005 | no | 1.3649 | 1.2687 | 0.0962 |
| gemma3-270m / prune_0.6_traces / qa | 0.2311 | 46.0726 | 6643 | 0.0000 | no | 4.4763 | 0.4184 | 4.0578 |
| gemma3-270m / quant_3_traces / code | 0.1015 | 1.3400 | 108935 | 1.0000 | no | 2.9144 | 3.2121 | 0.2976 |
| gemma3-270m / quant_3_traces / math | 0.1012 | 2.5222 | 335296 | 1.0000 | no | 2.6520 | 2.6998 | 0.0478 |
| gemma3-270m / quant_3_traces / qa | 0.0762 | 795608.2563 | 144188645701 | 0.3082 | no | 3.2900 | 0.4677 | 2.8223 |

Caveat: fitting r, D0, and beta to only two nonzero budgets is underidentified even with the exact zero-budget anchor. The numeric extrapolation uses a weak, documented deterministic tiebreak and is a diagnostic check, not an identified parameter estimate. Boundary fits and extremely large beta values indicate that the monotone form is not identifying a stable saturation timescale.

## Part 4 — noise-scale screen

This is **not a significance test**. Per-sample losses were not stored, so the requested honest yardstick is used instead of a fabricated bootstrap.

Median |Delta L| at pruning d=0.9: 0.012647 nats; screen threshold (2x median): 0.025293 nats. 29 of 62 negative-damage cells exceed it.

| method | model | capability | setting | Delta L | clears screen |
|---|---|---|---|---:|---:|
| pruning | Qwen3-0.6B | math | density=0.9 | -0.003790 | no |
| pruning | Qwen3-0.6B | qa | density=0.9 | -0.134340 | yes |
| pruning | Qwen3-0.6B | qa | density=0.8 | -0.136935 | yes |
| pruning | Qwen3-0.6B | qa | density=0.7 | -0.257841 | yes |
| pruning | Qwen3-0.6B | qa | density=0.6 | -0.053044 | yes |
| pruning | Qwen3-1.7B | math | density=0.9 | -0.010706 | no |
| pruning | Qwen3-1.7B | code | density=0.9 | -0.023187 | no |
| pruning | Qwen3-1.7B | code | density=0.8 | -0.005995 | no |
| pruning | Qwen3-1.7B | qa | density=0.9 | -0.049700 | yes |
| pruning | Qwen3-1.7B | qa | density=0.8 | -0.180639 | yes |
| pruning | Qwen3-1.7B | qa | density=0.7 | -0.508764 | yes |
| pruning | Qwen3-1.7B | qa | density=0.6 | -0.793358 | yes |
| pruning | Qwen3-1.7B | qa | density=0.55 | -0.531481 | yes |
| pruning | Qwen3-4B | math | density=0.7 | -0.003146 | no |
| pruning | Qwen3-4B | code | density=0.8 | -0.009824 | no |
| pruning | Qwen3-4B | code | density=0.7 | -0.088197 | yes |
| pruning | Qwen3-4B | code | density=0.6 | -0.001300 | no |
| pruning | Qwen3-4B | qa | density=0.9 | -0.002106 | no |
| pruning | Qwen3-4B | qa | density=0.8 | -0.292713 | yes |
| pruning | Qwen3-4B | qa | density=0.7 | -0.793109 | yes |
| pruning | Qwen3-4B | qa | density=0.6 | -1.760461 | yes |
| pruning | Qwen3-4B | qa | density=0.55 | -1.957993 | yes |
| pruning | Qwen3-4B | qa | density=0.5 | -1.476444 | yes |
| pruning | Qwen3-4B | qa | density=0.45 | -0.552666 | yes |
| pruning | muse-30b | code | density=0.9 | -0.005771 | no |
| pruning | olmo3-32b | qa | density=0.9 | -0.006806 | no |
| pruning | olmo3-32b | qa | density=0.55 | -0.007730 | no |
| pruning | olmo3-32b | qa | density=0.45 | -0.077387 | yes |
| pruning | olmo3-32b | qa | density=0.4 | -0.100049 | yes |
| pruning | olmo3-32b | qa | density=0.35 | -0.131776 | yes |
| pruning | olmo3-32b | qa | density=0.3 | -0.239362 | yes |
| pruning | olmo3-7b | math | density=0.9 | -0.000507 | no |
| pruning | olmo3-7b | code | density=0.9 | -0.001049 | no |
| pruning | olmo3-7b | qa | density=0.55 | -0.020651 | no |
| pruning | olmo3-7b | qa | density=0.5 | -0.011432 | no |
| quantization | Qwen3-0.6B | code | bits=8 | -0.000578 | no |
| quantization | Qwen3-0.6B | qa | bits=6 | -0.018220 | no |
| quantization | Qwen3-1.7B | code | bits=8 | -0.009679 | no |
| quantization | Qwen3-1.7B | math | bits=8 | -0.006114 | no |
| quantization | Qwen3-1.7B | qa | bits=8 | -0.058464 | yes |
| quantization | Qwen3-1.7B | code | bits=6 | -0.020081 | no |
| quantization | Qwen3-1.7B | math | bits=6 | -0.005218 | no |
| quantization | Qwen3-1.7B | qa | bits=4 | -0.213878 | yes |
| quantization | Qwen3-4B | math | bits=6 | -0.001839 | no |
| quantization | Qwen3-4B | qa | bits=6 | -0.082986 | yes |
| quantization | gemma3-1b | code | bits=8 | -0.000803 | no |
| quantization | gemma3-27b | code | bits=8 | -0.001827 | no |
| quantization | gemma3-27b | qa | bits=8 | -0.019843 | no |
| quantization | gemma3-27b | qa | bits=6 | -0.041444 | yes |
| quantization | gemma3-27b | qa | bits=4 | -0.665105 | yes |
| quantization | gemma3-4b | qa | bits=8 | -0.010707 | no |
| quantization | gemma3-4b | qa | bits=6 | -0.020605 | no |
| quantization | gemma3-4b | qa | bits=4 | -0.265314 | yes |
| quantization | muse-30b | code | bits=8 | -0.002370 | no |
| quantization | muse-30b | math | bits=8 | -0.000763 | no |
| quantization | muse-30b | qa | bits=6 | -0.101784 | yes |
| quantization | muse-30b | qa | bits=4 | -0.106781 | yes |
| quantization | olmo3-7b | math | bits=8 | -0.000183 | no |
| quantization | olmo3-7b | qa | bits=8 | -0.005248 | no |
| quantization | olmo3-7b | math | bits=6 | -0.000574 | no |
| quantization | olmo3-7b | qa | bits=6 | -0.004372 | no |
| quantization | olmo3-7b | qa | bits=4 | -0.071095 | yes |

## VERDICT

For pruning comparisons, PASS requires the paired 95% cell-bootstrap interval for candidate-minus-baseline MAE to lie fully below zero. Sign prediction requires a Wilson lower bound above 0.5. Quantization uses the pooled held-out winner. A recovery run passes only if its diagnostic extrapolation beats carry-forward in every capability; the underidentification caveat still applies.

| status | claim | exact comparison |
|---|---|---|
| PASS | V-a: shared-gamma first-order + deleted-mass law beats the direct-damage deleted-mass baseline | `{"baseline": "baseline_P2_direct", "baseline_mae_nats": 4.235923783254633, "bootstrap_resamples": 1000, "candidate": "P3_shared_gamma", "candidate_mae_nats": 3.1122297572152213, "candidate_wins": true, "mae_difference_candidate_minus_baseline": -1.1236940260394124, "mae_difference_ci95": [-2.5125365880347483, -0.1087620047897005], "n_paired_cells": 45}` |
| FAIL | V-a: shared-gamma deleted mass is more portable than per-capability raw sparsity | `{"baseline": "P1_raw_sparsity", "baseline_mae_nats": 3.1796399742113715, "bootstrap_resamples": 1000, "candidate": "P3_shared_gamma", "candidate_mae_nats": 2.6760061510146795, "candidate_wins": true, "mae_difference_candidate_minus_baseline": -0.5036338231966914, "mae_difference_ci95": [-1.9308685751893204, 0.7545136912862227], "n_paired_cells": 56}` |
| FAIL | V-b: shared-gamma first-order + deleted-mass law beats the direct-damage deleted-mass baseline | `{"baseline": "baseline_P2_direct", "baseline_mae_nats": 1.1937722724167719, "bootstrap_resamples": 1000, "candidate": "P3_shared_gamma", "candidate_mae_nats": 1.378512954057465, "candidate_wins": false, "mae_difference_candidate_minus_baseline": 0.18474068164069393, "mae_difference_ci95": [-0.1877418017724038, 0.6387828467929169], "n_paired_cells": 37}` |
| PASS | V-b: shared-gamma deleted mass is more portable than per-capability raw sparsity | `{"baseline": "P1_raw_sparsity", "baseline_mae_nats": 2.150538592873445, "bootstrap_resamples": 1000, "candidate": "P3_shared_gamma", "candidate_mae_nats": 1.4080184498089707, "candidate_wins": true, "mae_difference_candidate_minus_baseline": -0.7425201430644746, "mae_difference_ci95": [-1.3320658908585996, -0.265801773222478], "n_paired_cells": 40}` |
| PASS | V-c: shared-gamma first-order + deleted-mass law beats the direct-damage deleted-mass baseline | `{"baseline": "baseline_P2_direct", "baseline_mae_nats": 8.185450734676527, "bootstrap_resamples": 1000, "candidate": "P3_shared_gamma", "candidate_mae_nats": 2.167091293763739, "candidate_wins": true, "mae_difference_candidate_minus_baseline": -6.0183594409127865, "mae_difference_ci95": [-10.149877002787266, -2.4520149901014654], "n_paired_cells": 136}` |
| PASS | V-c: shared-gamma deleted mass is more portable than per-capability raw sparsity | `{"baseline": "P1_raw_sparsity", "baseline_mae_nats": 10.323529172996318, "bootstrap_resamples": 1000, "candidate": "P3_shared_gamma", "candidate_mae_nats": 2.3104889660364107, "candidate_wins": true, "mae_difference_candidate_minus_baseline": -8.013040206959907, "mae_difference_ci95": [-12.754982036669972, -4.036901470971442], "n_paired_cells": 144}` |
| PASS | V-d: signed first-order term predicts the sign of mild-density measured damage above chance | `{"accuracy": 0.6759259259259259, "ci95": [0.5829333238469441, 0.7568333458718599], "n_cells": 108, "n_correct": 73}` |
| FAIL | The fixed q*4^(-b) quantization form wins pooled leave-one-bit-out validation | `{"mae_nats": {"fixed_4^-b": 7.630708067603073, "learned_exponential": 6.727273390076048, "quadratic_b": 10.887305234041337}, "winner": "learned_exponential"}` |
| FAIL | Saturating recovery extrapolation beats carrying forward the second budget | `{"n_fits": 9, "wins": 4}` |
| FAIL | Recovery ladder prune_0.6_c4: saturating-form largest-budget prediction beats carry-forward in every capability | `{"n_capabilities": 3, "wins": 2}` |
| FAIL | Recovery ladder prune_0.6_traces: saturating-form largest-budget prediction beats carry-forward in every capability | `{"n_capabilities": 3, "wins": 0}` |
| FAIL | Recovery ladder quant_3_traces: saturating-form largest-budget prediction beats carry-forward in every capability | `{"n_capabilities": 3, "wins": 2}` |
| SCREEN_ONLY | Reported negative-damage cells clear the descriptive noise-scale screen | `{"n_exceeding": 29, "n_improvement_cells": 62, "threshold": 0.025293042433487622}` |
