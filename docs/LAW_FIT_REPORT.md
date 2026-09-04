# Unified held-out law-fit report

Generated 2026-09-03 by `analysis/v18_law_fit.py` from committed/local result artifacts only. This was a CPU-only refit: no inference, training, GPU use, or result-input mutation occurred.

## Decision rule and scope

Headline metrics below are pooled predictions for observations excluded from their fold's fit. MAE is in CE nats; relative error is MAE divided by mean absolute held-out damage. Brackets on MAE/relative error are deterministic 1,000-resample held-out-cell bootstrap 95% intervals; sign brackets and calibration brackets are Wilson 95% intervals. Calibration is empirical coverage of a nominal 95% absolute-residual interval calibrated on that fold's training residuals. These intervals quantify variation across available cells, not benchmark-item or seed uncertainty.

Every compression-damage fit uses the V17 cliff filter: scan each `(model, method, capability)` trajectory toward stronger compression; the first `Delta L_c > 1.0` nat cell and all deeper cells are excluded. Dense anchors are never scored. Recovery has no varying compression coordinate, so each recovery observation is admitted only when its own `Delta L_c <= 1.0` nat. Negative deltas remain eligible; censoring them would invalidate sign evaluation.

`r_storage=b/16` for quantization is a nominal bit budget, not serialized size or runtime memory. Unstructured pruning density is likewise nominal and does not imply sparse-kernel speedup.

Status `OK` means the declared fold produced predictions for every held-out eligible cell. `PARTIAL` means a requested axis is underidentified or some held-out cells are not estimable. `PLANNED` names the missing data. `NON-DECISIONAL` is an in-sample diagnostic and never a headline.

## Held-out findings

- **Pruning remains on probation.** Best candidate-versus-baseline held-out MAEs are `fit d>=0.55 -> deeper pre-cliff`: candidate `density_only` 0.333 vs baseline `baseline_raw_sparsity` 0.326; `leave-largest-model-out`: candidate `hierarchical_capability_family` 0.283 vs baseline `baseline_anchor_only` 0.289; `leave-one-family-out`: candidate `density_only` 0.339 vs baseline `baseline_removed_weight_norm` 0.304. No candidate clears the directive's no-held-out-sign-error rule.
- **Quantization remains on probation.** Best held-out comparisons are `leave-one-bit-out (pre-cliff only)`: candidate `family_conditioned_smooth_cliff` 0.121 vs categorical baseline not estimable; `leave-largest-model-out`: candidate `family_conditioned_smooth_cliff` 0.091 vs baseline `bit_width_categorical_baseline` 0.106; `leave-one-family-out`: candidate `fixed_4^-b` 0.133 vs baseline `bit_width_categorical_baseline` 0.109. No candidate clears the no-held-out-sign-error rule.
- **Distillation is PARTIAL:** the real four-rotation size-held-out MAE is 0.165 nats, but only one family/teacher/data budget is available; cross-family and D-ladder rows remain PLANNED.
- **Recovery is PARTIAL:** C4 has a diagnostic QA-only held-out-budget score after filtering; aligned traces have no estimable held-out law because only one budget per capability remains below the 1-nat cap.
- **Cliffs are separately held out:** pruning cliff-density MAE is 0.078 for largest-model holdout and 0.146 for family holdout; quantization bit-cliff MAE is 0.238 bits for model holdout and 0.422 bits for family holdout.

## 1. Pruning

Available: 12 models in 4 directive families, 174 pre-cliff cells from 393 non-anchor cells. There are 34 observed cliff trajectories and 2 right-censored trajectories.

The mechanism row is intentionally an upper-bound comparator: it consumes measured `g·delta_w` and capability Fisher mass `m_c`, so it is not the observable-only reduced law. The removed-weight-norm baseline is derived from the committed V6 magnitude histogram and is explicitly binned/approximate.
The mechanism comparator has 153 rather than 174 eligible cells because the later Qwen-1.7B/Gemma-1B density infill measured loss but did not re-measure `g·delta_w`; those 21 cells remain in every observable-input candidate.

| Method | Inputs | Candidate law | Fit points | Held-out split | MAE | Relative error | Sign accuracy | Cliff error | Calibration |
|---|---|---|---|---|---|---|---|---|---|
| Pruning [OK] | L_c0, N0, f, d | A(1-d)^gamma | 146/fold; 28 held-out | fit d>=0.55 -> deeper pre-cliff | 0.333 [0.216, 0.469] | 0.859 [0.708, 1.020] | 0.750 [0.566, 0.873] | n/a | 95% PI cover 0.964 [0.823, 0.994], width 1.898 |
| Pruning [OK] | L_c0, N0, f, d | A(1-d)^gamma | 145-166/fold; 55 held-out | leave-largest-model-out | 0.342 [0.240, 0.475] | 1.184 [1.031, 1.397] | 0.691 [0.560, 0.797] | n/a | 95% PI cover 0.909 [0.804, 0.961], width 1.811 |
| Pruning [OK] | L_c0, N0, f, d | A(1-d)^gamma | 119-161/fold; 174 held-out | leave-one-family-out | 0.339 [0.289, 0.398] | 1.081 [1.000, 1.162] | 0.776 [0.708, 0.831] | n/a | 95% PI cover 0.908 [0.856, 0.943], width 1.723 |
| Pruning [NON-DECISIONAL] | L_c0, N0, f, d | A(1-d)^gamma | 174 | NON-DECISIONAL in-sample | 0.302 [0.252, 0.359] | 0.964 [0.901, 1.035] | 0.776 [0.708, 0.831] | n/a | n/a |
| Pruning [OK] | L_c0, N0, f, d | [A+B log(N0)](1-d)^gamma | 146/fold; 28 held-out | fit d>=0.55 -> deeper pre-cliff | 0.337 [0.236, 0.484] | 0.870 [0.741, 0.994] | 0.750 [0.566, 0.873] | n/a | 95% PI cover 0.964 [0.823, 0.994], width 1.905 |
| Pruning [OK] | L_c0, N0, f, d | [A+B log(N0)](1-d)^gamma | 145-166/fold; 55 held-out | leave-largest-model-out | 0.341 [0.228, 0.466] | 1.180 [1.027, 1.405] | 0.691 [0.560, 0.797] | n/a | 95% PI cover 0.909 [0.804, 0.961], width 1.817 |
| Pruning [OK] | L_c0, N0, f, d | [A+B log(N0)](1-d)^gamma | 119-161/fold; 174 held-out | leave-one-family-out | 0.364 [0.307, 0.427] | 1.160 [1.053, 1.283] | 0.741 [0.672, 0.801] | n/a | 95% PI cover 0.891 [0.836, 0.929], width 1.708 |
| Pruning [NON-DECISIONAL] | L_c0, N0, f, d | [A+B log(N0)](1-d)^gamma | 174 | NON-DECISIONAL in-sample | 0.300 [0.250, 0.350] | 0.955 [0.891, 1.018] | 0.776 [0.708, 0.831] | n/a | n/a |
| Pruning [OK] | L_c0, N0, f, d | [A+B L_c0](1-d)^gamma | 146/fold; 28 held-out | fit d>=0.55 -> deeper pre-cliff | 0.335 [0.239, 0.433] | 0.864 [0.600, 1.249] | 0.821 [0.644, 0.921] | n/a | 95% PI cover 0.964 [0.823, 0.994], width 1.864 |
| Pruning [OK] | L_c0, N0, f, d | [A+B L_c0](1-d)^gamma | 145-166/fold; 55 held-out | leave-largest-model-out | 0.341 [0.250, 0.452] | 1.182 [0.950, 1.594] | 0.818 [0.697, 0.898] | n/a | 95% PI cover 0.945 [0.851, 0.981], width 1.758 |
| Pruning [OK] | L_c0, N0, f, d | [A+B L_c0](1-d)^gamma | 119-161/fold; 174 held-out | leave-one-family-out | 0.442 [0.383, 0.506] | 1.410 [1.248, 1.591] | 0.615 [0.541, 0.684] | n/a | 95% PI cover 0.810 [0.746, 0.862], width 1.599 |
| Pruning [NON-DECISIONAL] | L_c0, N0, f, d | [A+B L_c0](1-d)^gamma | 174 | NON-DECISIONAL in-sample | 0.287 [0.248, 0.330] | 0.914 [0.830, 1.009] | 0.718 [0.647, 0.780] | n/a | n/a |
| Pruning [OK] | L_c0, N0, f, d | [A+u_c+v_f](1-d)^gamma (unseen effects=0) | 146/fold; 28 held-out | fit d>=0.55 -> deeper pre-cliff | 0.420 [0.300, 0.550] | 1.084 [0.734, 1.668] | 0.821 [0.644, 0.921] | n/a | 95% PI cover 0.857 [0.685, 0.943], width 1.378 |
| Pruning [OK] | L_c0, N0, f, d | [A+u_c+v_f](1-d)^gamma (unseen effects=0) | 145-166/fold; 55 held-out | leave-largest-model-out | 0.283 [0.199, 0.388] | 0.979 [0.765, 1.258] | 0.855 [0.738, 0.924] | n/a | 95% PI cover 0.927 [0.827, 0.971], width 1.528 |
| Pruning [OK] | L_c0, N0, f, d | [A+u_c+v_f](1-d)^gamma (unseen effects=0) | 119-161/fold; 174 held-out | leave-one-family-out | 0.483 [0.405, 0.559] | 1.539 [1.324, 1.802] | 0.661 [0.588, 0.727] | n/a | 95% PI cover 0.741 [0.672, 0.801], width 1.375 |
| Pruning [NON-DECISIONAL] | L_c0, N0, f, d | [A+u_c+v_f](1-d)^gamma (unseen effects=0) | 174 | NON-DECISIONAL in-sample | 0.233 [0.198, 0.271] | 0.744 [0.653, 0.841] | 0.862 [0.803, 0.906] | n/a | n/a |
| Pruning [OK] | L_c0, N0, f, d, measured g·delta_w, m_c | g·delta_w + B m_c(d)^gamma (measured-mechanism upper bound) | 126/fold; 27 held-out | fit d>=0.55 -> deeper pre-cliff | 0.855 [0.630, 1.120] | 2.208 [1.422, 3.404] | 0.815 [0.633, 0.918] | n/a | 95% PI cover 0.815 [0.633, 0.918], width 2.547 |
| Pruning [OK] | L_c0, N0, f, d, measured g·delta_w, m_c | g·delta_w + B m_c(d)^gamma (measured-mechanism upper bound) | 124-145/fold; 55 held-out | leave-largest-model-out | 0.380 [0.257, 0.525] | 1.317 [0.867, 2.061] | 0.800 [0.676, 0.884] | n/a | 95% PI cover 0.909 [0.804, 0.961], width 2.513 |
| Pruning [OK] | L_c0, N0, f, d, measured g·delta_w, m_c | g·delta_w + B m_c(d)^gamma (measured-mechanism upper bound) | 99-140/fold; 153 held-out | leave-one-family-out | 0.532 [0.410, 0.675] | 2.006 [1.551, 2.576] | 0.837 [0.770, 0.887] | n/a | 95% PI cover 0.882 [0.822, 0.924], width 2.457 |
| Pruning [NON-DECISIONAL] | L_c0, N0, f, d, measured g·delta_w, m_c | g·delta_w + B m_c(d)^gamma (measured-mechanism upper bound) | 153 | NON-DECISIONAL in-sample | 0.314 [0.255, 0.376] | 1.184 [0.970, 1.471] | 0.830 [0.763, 0.881] | n/a | n/a |
| Pruning [OK] | L_c0, N0, f, d | A(1-d)^gamma+C{sigmoid[(d*-d)/w]-sigmoid[(d*-1)/w]} | 146/fold; 28 held-out | fit d>=0.55 -> deeper pre-cliff | 1.799 [1.208, 2.451] | 4.644 [3.067, 7.318] | 0.750 [0.566, 0.873] | n/a | 95% PI cover 0.393 [0.236, 0.576], width 1.879 |
| Pruning [OK] | L_c0, N0, f, d | A(1-d)^gamma+C{sigmoid[(d*-d)/w]-sigmoid[(d*-1)/w]} | 145-166/fold; 55 held-out | leave-largest-model-out | 0.370 [0.253, 0.497] | 1.284 [1.027, 1.675] | 0.618 [0.486, 0.735] | n/a | 95% PI cover 0.909 [0.804, 0.961], width 1.854 |
| Pruning [OK] | L_c0, N0, f, d | A(1-d)^gamma+C{sigmoid[(d*-d)/w]-sigmoid[(d*-1)/w]} | 119-161/fold; 174 held-out | leave-one-family-out | 0.394 [0.339, 0.448] | 1.255 [1.167, 1.381] | 0.661 [0.588, 0.727] | n/a | 95% PI cover 0.879 [0.823, 0.920], width 1.718 |
| Pruning [NON-DECISIONAL] | L_c0, N0, f, d | A(1-d)^gamma+C{sigmoid[(d*-d)/w]-sigmoid[(d*-1)/w]} | 174 | NON-DECISIONAL in-sample | 0.296 [0.250, 0.343] | 0.944 [0.868, 1.033] | 0.759 [0.690, 0.816] | n/a | n/a |
| Pruning [OK] | L_c0, N0, f, d | A(1-d) | 146/fold; 28 held-out | fit d>=0.55 -> deeper pre-cliff | 0.326 [0.222, 0.453] | 0.841 [0.637, 1.085] | 0.750 [0.566, 0.873] | n/a | 95% PI cover 0.964 [0.823, 0.994], width 1.874 |
| Pruning [OK] | L_c0, N0, f, d | A(1-d) | 145-166/fold; 55 held-out | leave-largest-model-out | 0.325 [0.215, 0.465] | 1.125 [0.949, 1.348] | 0.691 [0.560, 0.797] | n/a | 95% PI cover 0.909 [0.804, 0.961], width 1.834 |
| Pruning [OK] | L_c0, N0, f, d | A(1-d) | 119-161/fold; 174 held-out | leave-one-family-out | 0.310 [0.257, 0.367] | 0.989 [0.915, 1.067] | 0.776 [0.708, 0.831] | n/a | 95% PI cover 0.931 [0.883, 0.960], width 1.789 |
| Pruning [NON-DECISIONAL] | L_c0, N0, f, d | A(1-d) | 174 | NON-DECISIONAL in-sample | 0.291 [0.240, 0.345] | 0.927 [0.869, 0.989] | 0.776 [0.708, 0.831] | n/a | n/a |
| Pruning [OK] | L_c0, N0, f, d | A{(N0 d)^(-alpha)-N0^(-alpha)} | 146/fold; 28 held-out | fit d>=0.55 -> deeper pre-cliff | 0.376 [0.265, 0.504] | 0.970 [0.924, 1.003] | 0.750 [0.566, 0.873] | n/a | 95% PI cover 0.964 [0.823, 0.994], width 1.832 |
| Pruning [OK] | L_c0, N0, f, d | A{(N0 d)^(-alpha)-N0^(-alpha)} | 145-166/fold; 55 held-out | leave-largest-model-out | 0.316 [0.197, 0.458] | 1.095 [0.982, 1.204] | 0.691 [0.560, 0.797] | n/a | 95% PI cover 0.909 [0.804, 0.961], width 1.819 |
| Pruning [OK] | L_c0, N0, f, d | A{(N0 d)^(-alpha)-N0^(-alpha)} | 119-161/fold; 174 held-out | leave-one-family-out | 0.371 [0.308, 0.441] | 1.181 [1.053, 1.321] | 0.776 [0.708, 0.831] | n/a | 95% PI cover 0.897 [0.842, 0.934], width 1.671 |
| Pruning [NON-DECISIONAL] | L_c0, N0, f, d | A{(N0 d)^(-alpha)-N0^(-alpha)} | 174 | NON-DECISIONAL in-sample | 0.284 [0.233, 0.340] | 0.905 [0.843, 0.969] | 0.776 [0.708, 0.831] | n/a | n/a |
| Pruning [OK] | L_c0, N0, f, d, binned removed-weight L2 | A times binned removed-weight L2 fraction | 146/fold; 28 held-out | fit d>=0.55 -> deeper pre-cliff | 0.328 [0.219, 0.465] | 0.847 [0.662, 1.044] | 0.750 [0.566, 0.873] | n/a | 95% PI cover 0.964 [0.823, 0.994], width 1.827 |
| Pruning [OK] | L_c0, N0, f, d, binned removed-weight L2 | A times binned removed-weight L2 fraction | 145-166/fold; 55 held-out | leave-largest-model-out | 0.304 [0.196, 0.445] | 1.052 [0.901, 1.216] | 0.691 [0.560, 0.797] | n/a | 95% PI cover 0.909 [0.804, 0.961], width 1.829 |
| Pruning [OK] | L_c0, N0, f, d, binned removed-weight L2 | A times binned removed-weight L2 fraction | 119-161/fold; 174 held-out | leave-one-family-out | 0.304 [0.249, 0.366] | 0.968 [0.901, 1.032] | 0.776 [0.708, 0.831] | n/a | 95% PI cover 0.931 [0.883, 0.960], width 1.788 |
| Pruning [NON-DECISIONAL] | L_c0, N0, f, d, binned removed-weight L2 | A times binned removed-weight L2 fraction | 174 | NON-DECISIONAL in-sample | 0.292 [0.238, 0.345] | 0.932 [0.884, 0.983] | 0.776 [0.708, 0.831] | n/a | n/a |
| Pruning [OK] | L_c0, N0, f, d | Delta L=0 (dense anchor) | 146/fold; 28 held-out | fit d>=0.55 -> deeper pre-cliff | 0.387 [0.273, 0.513] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.121] | n/a | 95% PI cover 0.964 [0.823, 0.994], width 1.827 |
| Pruning [OK] | L_c0, N0, f, d | Delta L=0 (dense anchor) | 145-166/fold; 55 held-out | leave-largest-model-out | 0.289 [0.187, 0.402] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.065] | n/a | 95% PI cover 0.927 [0.827, 0.971], width 1.820 |
| Pruning [OK] | L_c0, N0, f, d | Delta L=0 (dense anchor) | 119-161/fold; 174 held-out | leave-one-family-out | 0.314 [0.262, 0.372] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.022] | n/a | 95% PI cover 0.937 [0.890, 0.964], width 1.805 |
| Pruning [NON-DECISIONAL] | L_c0, N0, f, d | Delta L=0 (dense anchor) | 174 | NON-DECISIONAL in-sample | 0.314 [0.260, 0.369] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.022] | n/a | n/a |
| Pruning [OK] | L_c0, N0, f, d | A(1-d)^gamma+C{sigmoid[(d*-d)/w]-sigmoid[(d*-1)/w]}; location submodel | 31-33/fold; 7 held-out | held-out cliff density: largest model | n/a | n/a | n/a | 0.078 [0.048, 0.114] | 95% PI cover 0.714 [0.359, 0.918], width 0.199 |
| Pruning [OK] | L_c0, N0, f, d | A(1-d)^gamma+C{sigmoid[(d*-d)/w]-sigmoid[(d*-1)/w]}; location submodel | 16-31/fold; 34 held-out | held-out cliff density: family | n/a | n/a | n/a | 0.146 [0.115, 0.178] | 95% PI cover 0.294 [0.168, 0.462], width 0.140 |

Held-out sign audit (leave-one-family-out): math 50/54, code 47/54, qa 18/66. The two cliff-censored trajectories are olmo3-32b/math (below 0.3), olmo3-32b/qa (below 0.3); they are not converted to invented cliff locations.

## 2. Quantization

Available now: 12 model artifacts and 105 pre-cliff cells from 144 non-anchor cells. The earlier v14 positive-damage screen retained 9 models; v18 uses all 12 current artifacts because negative damage is required for the requested sign test. There are 34 observed bit-cliff trajectories and 2 right-censored trajectories.

| Method | Inputs | Candidate law | Fit points | Held-out split | MAE | Relative error | Sign accuracy | Cliff error | Calibration |
|---|---|---|---|---|---|---|---|---|---|
| Quantization [OK] | L_c0, N0, f, b; r_storage=b/16 nominal | q[4^(4-b)-4^(4-16)] | 69-103/fold; 105 held-out | leave-one-bit-out (pre-cliff only) | 0.121 [0.081, 0.161] | 0.875 [0.784, 0.986] | 0.667 [0.572, 0.750] | n/a | 95% PI cover 0.790 [0.703, 0.857], width 1.090 |
| Quantization [OK] | L_c0, N0, f, b; r_storage=b/16 nominal | q[4^(4-b)-4^(4-16)] | 94-97/fold; 28 held-out | leave-largest-model-out | 0.140 [0.060, 0.232] | 1.081 [0.756, 1.761] | 0.750 [0.566, 0.873] | n/a | 95% PI cover 0.929 [0.774, 0.980], width 1.322 |
| Quantization [OK] | L_c0, N0, f, b; r_storage=b/16 nominal | q[4^(4-b)-4^(4-16)] | 54-98/fold; 105 held-out | leave-one-family-out | 0.133 [0.092, 0.179] | 0.957 [0.829, 1.152] | 0.667 [0.572, 0.750] | n/a | 95% PI cover 0.886 [0.811, 0.933], width 1.102 |
| Quantization [NON-DECISIONAL] | L_c0, N0, f, b; r_storage=b/16 nominal | q[4^(4-b)-4^(4-16)] | 105 | NON-DECISIONAL in-sample | 0.110 [0.073, 0.147] | 0.791 [0.673, 0.931] | 0.667 [0.572, 0.750] | n/a | n/a |
| Quantization [OK] | L_c0, N0, f, b; r_storage=b/16 nominal | q{exp[k(4-b)]-exp[k(4-16)]} | 69-103/fold; 105 held-out | leave-one-bit-out (pre-cliff only) | 0.159 [0.104, 0.218] | 1.149 [0.939, 1.438] | 0.667 [0.572, 0.750] | n/a | 95% PI cover 0.771 [0.682, 0.841], width 1.014 |
| Quantization [OK] | L_c0, N0, f, b; r_storage=b/16 nominal | q{exp[k(4-b)]-exp[k(4-16)]} | 94-97/fold; 28 held-out | leave-largest-model-out | 0.224 [0.080, 0.426] | 1.731 [0.838, 3.139] | 0.750 [0.566, 0.873] | n/a | 95% PI cover 0.893 [0.728, 0.963], width 1.296 |
| Quantization [OK] | L_c0, N0, f, b; r_storage=b/16 nominal | q{exp[k(4-b)]-exp[k(4-16)]} | 54-98/fold; 105 held-out | leave-one-family-out | 0.157 [0.102, 0.223] | 1.131 [0.858, 1.497] | 0.667 [0.572, 0.750] | n/a | 95% PI cover 0.886 [0.811, 0.933], width 1.047 |
| Quantization [NON-DECISIONAL] | L_c0, N0, f, b; r_storage=b/16 nominal | q{exp[k(4-b)]-exp[k(4-16)]} | 105 | NON-DECISIONAL in-sample | 0.115 [0.080, 0.152] | 0.830 [0.678, 1.033] | 0.667 [0.572, 0.750] | n/a | n/a |
| Quantization [OK] | L_c0, N0, f, b; r_storage=b/16 nominal | A{(N0 b/16)^(-alpha)-N0^(-alpha)} | 69-103/fold; 105 held-out | leave-one-bit-out (pre-cliff only) | 0.206 [0.167, 0.246] | 1.484 [1.279, 1.825] | 0.667 [0.572, 0.750] | n/a | 95% PI cover 0.762 [0.672, 0.833], width 0.937 |
| Quantization [OK] | L_c0, N0, f, b; r_storage=b/16 nominal | A{(N0 b/16)^(-alpha)-N0^(-alpha)} | 94-97/fold; 28 held-out | leave-largest-model-out | 0.129 [0.063, 0.210] | 1.000 [0.818, 1.423] | 0.750 [0.566, 0.873] | n/a | 95% PI cover 0.964 [0.823, 0.994], width 1.242 |
| Quantization [OK] | L_c0, N0, f, b; r_storage=b/16 nominal | A{(N0 b/16)^(-alpha)-N0^(-alpha)} | 54-98/fold; 105 held-out | leave-one-family-out | 0.161 [0.129, 0.199] | 1.164 [0.991, 1.439] | 0.667 [0.572, 0.750] | n/a | 95% PI cover 0.924 [0.857, 0.961], width 1.056 |
| Quantization [NON-DECISIONAL] | L_c0, N0, f, b; r_storage=b/16 nominal | A{(N0 b/16)^(-alpha)-N0^(-alpha)} | 105 | NON-DECISIONAL in-sample | 0.162 [0.132, 0.198] | 1.172 [0.989, 1.476] | 0.667 [0.572, 0.750] | n/a | n/a |
| Quantization [OK] | L_c0, N0, f, b; r_storage=b/16 nominal | smooth exponential + family-conditioned sigmoid bit cliff | 69-103/fold; 105 held-out | leave-one-bit-out (pre-cliff only) | 0.121 [0.084, 0.161] | 0.874 [0.807, 0.946] | 0.533 [0.438, 0.626] | n/a | 95% PI cover 0.790 [0.703, 0.857], width 0.993 |
| Quantization [OK] | L_c0, N0, f, b; r_storage=b/16 nominal | smooth exponential + family-conditioned sigmoid bit cliff | 94-97/fold; 28 held-out | leave-largest-model-out | 0.091 [0.038, 0.164] | 0.705 [0.537, 0.934] | 0.607 [0.424, 0.764] | n/a | 95% PI cover 0.964 [0.823, 0.994], width 1.130 |
| Quantization [OK] | L_c0, N0, f, b; r_storage=b/16 nominal | smooth exponential + family-conditioned sigmoid bit cliff | 54-98/fold; 105 held-out | leave-one-family-out | 0.154 [0.107, 0.206] | 1.111 [1.001, 1.246] | 0.514 [0.420, 0.608] | n/a | 95% PI cover 0.886 [0.811, 0.933], width 1.055 |
| Quantization [NON-DECISIONAL] | L_c0, N0, f, b; r_storage=b/16 nominal | smooth exponential + family-conditioned sigmoid bit cliff | 105 | NON-DECISIONAL in-sample | 0.091 [0.060, 0.131] | 0.653 [0.511, 0.846] | 0.629 [0.533, 0.715] | n/a | n/a |
| Quantization [PARTIAL] | L_c0, N0, f, b; r_storage=b/16 nominal | mean Delta L by (capability, bit) | 69-103/fold; 105 held-out | leave-one-bit-out (pre-cliff only) | n/a | n/a | n/a | n/a | n/a |
| Quantization [PARTIAL] | L_c0, N0, f, b; r_storage=b/16 nominal | mean Delta L by (capability, bit) | 94-97/fold; 28 held-out | leave-largest-model-out | 0.106 [0.046, 0.173] | 1.073 [0.657, 2.436] | 0.769 [0.579, 0.890] | n/a | 95% PI cover 1.000 [0.871, 1.000], width 0.988 |
| Quantization [PARTIAL] | L_c0, N0, f, b; r_storage=b/16 nominal | mean Delta L by (capability, bit) | 54-98/fold; 105 held-out | leave-one-family-out | 0.109 [0.075, 0.146] | 0.828 [0.687, 0.993] | 0.670 [0.574, 0.753] | n/a | 95% PI cover 0.922 [0.854, 0.960], width 0.959 |
| Quantization [NON-DECISIONAL] | L_c0, N0, f, b; r_storage=b/16 nominal | mean Delta L by (capability, bit) | 105 | NON-DECISIONAL in-sample | 0.094 [0.065, 0.127] | 0.677 [0.530, 0.864] | 0.724 [0.632, 0.800] | n/a | n/a |
| Quantization [OK] | L_c0, N0, f, b; r_storage=b/16 nominal | smooth exponential + family-conditioned sigmoid bit cliff; location submodel | 31-33/fold; 34 held-out | int3-collapse prediction: leave-model-out | n/a | n/a | n/a | 0.238 [0.133, 0.347] | 95% PI cover 0.853 [0.699, 0.936], width 1.727 |
| Quantization [OK] | L_c0, N0, f, b; r_storage=b/16 nominal | smooth exponential + family-conditioned sigmoid bit cliff; location submodel | 16-31/fold; 34 held-out | int3-collapse prediction: leave-family-out | n/a | n/a | n/a | 0.422 [0.311, 0.522] | 95% PI cover 0.735 [0.569, 0.854], width 1.368 |

A pure bit-category baseline is deliberately non-estimable when that bit is globally held out; those folds are `PARTIAL`, not silently imputed. Int3 damage magnitudes beyond the 1-nat cap are not used. The separate int3-collapse rows evaluate held-out cliff onset in bits. Capability-specific held-out sign audit for the family-conditioned candidate (leave-one-family-out): math 17/35, code 17/34, qa 20/36. Censored: olmo3-32b/math (below 3), olmo3-32b/qa (below 3).

## 3. Distillation

Only the registered clean source-referenced ladder is fitted: Gemma-3 students 270M/1B/4B/12B distilled with the same `gpt-5.6-luna`, `full`, 600-trace configuration, referenced to the tested Gemma-3 27B dense source. This is one family, one teacher, one data budget and one recorded seed, so the law remains `PARTIAL` even though every size rotation is genuinely held out.

| Method | Inputs | Candidate law | Fit points | Held-out split | MAE | Relative error | Sign accuracy | Cliff error | Calibration |
|---|---|---|---|---|---|---|---|---|---|
| Distillation [PARTIAL] | source L_c0 (Gemma-3 27B), N0, f, r_storage | Delta L_c=floor_c-alpha_c log(r_storage) | 9/fold; 12 held-out | fit 3 student sizes -> predict 4th (all four rotations) | 0.165 [0.100, 0.244] | 0.232 [0.136, 0.371] | 0.917 [0.646, 0.985] | n/a | 95% PI cover 0.667 [0.391, 0.862], width 0.346 |
| Distillation [NON-DECISIONAL] | source L_c0 (Gemma-3 27B), N0, f, r_storage | Delta L_c=floor_c-alpha_c log(r_storage) | 12 | NON-DECISIONAL in-sample | 0.082 [0.047, 0.119] | 0.116 [0.066, 0.196] | 1.000 [0.758, 1.000] | n/a | n/a |
| Distillation [PLANNED] | L_c0, N0, f, r_storage | same source-referenced log-size law | 0; missing a clean Qwen source-referenced student ladder | Qwen/Gemma cross-family | n/a | n/a | n/a | n/a | n/a |
| Distillation [PLANNED] | L_c0, N0, f, r_storage, D | capacity floor plus D-ladder term | 0; missing completed multi-D runs for >=3 training budgets plus a held-out budget | held-out D-ladder | n/a | n/a | n/a | n/a | n/a |

Full-ladder parameter diagnostic (NON-DECISIONAL; student-size cluster bootstrap):

| Capability | floor_c | bootstrap 95% CI | alpha_c | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| math | -0.046 | [-0.434, 0.554] | 0.197 | [0.055, 0.354] |
| code | -0.012 | [-0.242, 0.155] | 0.137 | [-0.001, 0.213] |
| qa | -1.649 | [-2.224, -0.433] | 0.131 | [-0.161, 0.380] |

For QA: **distilled students achieve lower QA probe loss than the tested 27B source under this evaluation**.

## 4. Recovery

The same Gemma-3 1B prune-0.6 anchor feeds both sources. Recovery is fitted to normalized damage `z_c(D_R)=Delta L_c(D_R)/Delta L_c(0)`, making the displayed saturation law dimensionless and anchored at one. C4 is assessed only with the monotonic saturation form. Traces compare monotonic saturation, change-point, and early-recovery-plus-late-penalty, but the comparison is underdetermined. In particular, enforcing the common pre-cliff rule excludes the 4M/16M aligned points that exhibit late re-damage; fitting a penalty to them would violate the report's own scope.

| Method | Inputs | Candidate law | Fit points | Held-out split | MAE | Relative error | Sign accuracy | Cliff error | Calibration |
|---|---|---|---|---|---|---|---|---|---|
| Recovery [PARTIAL] | Delta L_c(0), D_R, source, capability | r_c+(1-r_c)(1+D_R/D0)^(-beta) | 2/fold; 3 held-out | leave-one-budget-out (c4); PARTIAL: only QA retains 3 pre-cliff budgets; math/code retain 2/1 | 0.168 [0.014, 0.245] | 0.996 [0.876, 1.000] | 0.333 [0.061, 0.792] | n/a | 95% PI cover 0.667 [0.208, 0.939], width 0.490 |
| Recovery [NON-DECISIONAL] | Delta L_c(0), D_R, source, capability | r_c+(1-r_c)(1+D_R/D0)^(-beta) | 6 | NON-DECISIONAL in-sample (c4; underidentified) | 0.082 [0.000, 0.163] | 0.166 [0.000, 0.603] | 0.667 [0.300, 0.903] | n/a | n/a |
| Recovery [PARTIAL] | Delta L_c(0), D_R, source, capability | r_c+(1-r_c)(1+D_R/D0)^(-beta) | 0 | leave-one-budget-out (traces); PARTIAL: only the 1M point is pre-cliff in each capability; 4M/16M exceed 1 nat | n/a | n/a | n/a | n/a | n/a |
| Recovery [NON-DECISIONAL] | Delta L_c(0), D_R, source, capability | r_c+(1-r_c)(1+D_R/D0)^(-beta) | 3 | NON-DECISIONAL in-sample (traces; underidentified) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [0.439, 1.000] | n/a | n/a |
| Recovery [PARTIAL] | Delta L_c(0), D_R, source, capability | 1+a log(1+D_R/1M)+c[log(1+D_R/1M)-tau]_+ | 0 | leave-one-budget-out (traces); PARTIAL: only the 1M point is pre-cliff in each capability; 4M/16M exceed 1 nat | n/a | n/a | n/a | n/a | n/a |
| Recovery [NON-DECISIONAL] | Delta L_c(0), D_R, source, capability | 1+a log(1+D_R/1M)+c[log(1+D_R/1M)-tau]_+ | 3 | NON-DECISIONAL in-sample (traces; underidentified) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [0.439, 1.000] | n/a | n/a |
| Recovery [PARTIAL] | Delta L_c(0), D_R, source, capability | (1+D_R/D0)^(-beta)+kappa(D_R/1M)^p | 0 | leave-one-budget-out (traces); PARTIAL: only the 1M point is pre-cliff in each capability; 4M/16M exceed 1 nat | n/a | n/a | n/a | n/a | n/a |
| Recovery [NON-DECISIONAL] | Delta L_c(0), D_R, source, capability | (1+D_R/D0)^(-beta)+kappa(D_R/1M)^p | 3 | NON-DECISIONAL in-sample (traces; underidentified) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [0.439, 1.000] | n/a | n/a |

Aligned availability audit (not fitted): all three requested budgets exist for math/code/QA; at 1M all three are pre-cliff, while every 4M/16M capability cell exceeds the 1-nat cap. Thus each aligned candidate row is emitted as `PARTIAL` with no held-out estimate, rather than fitting forbidden points or presenting a three-point interpolation as prediction.

Eligible recovery counts after the 1-nat cap: c4: math=2, code=1, qa=3; traces: math=1, code=1, qa=1. A defensible aligned held-out comparison needs at least three pre-cliff training budgets plus one held-out budget per capability, denser sampling around the early optimum, and another seed/model. The existing late points remain descriptive evidence of non-monotonicity, not fit points.

## Provenance and limitations

- Pruning: `results/v6-capability-geometry/*/{prune_losses.json,alignment.json,spectrum_bins.npz,fisher_meta.json}`; V6 artifacts do not record a run seed in these files.
- Quantization: `results/v10-quantization/*/quant_losses.json`; nominal `b/16`; these compact loss files do not record a seed.
- Distillation: selected V16 `gpt-5.6-luna_full_600/residual.json` files plus the V6 Gemma-3 27B dense reference; the selected V16 records use seed 0.
- Recovery: `results/v13-recovery/gemma3-1b/prune_0.6_{c4,traces}/recovery.json`; traces records seed 0, while the salvaged C4 compact artifact does not record a seed.
- There are no repeated measurement seeds for these law cells. Consequently the calibration/MAE intervals are cell-sampling diagnostics and cannot estimate seed or benchmark-item uncertainty.
- No post-cliff loss magnitude contributes to any law coefficient or headline damage metric. Cliff-location models use only observed threshold-crossing coordinates; censored trajectories stay censored.
