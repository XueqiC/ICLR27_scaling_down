# Unified held-out law-fit report

Pruning/recovery historical analyses: `analysis/v18_law_fit.py`. Quantization and distillation refreshed by `analysis/v24_quant_baselines.py` and `analysis/v25_distill_delta.py`; all are CPU-only existing-data analyses. Historical V21/V22 candidates are retained with their original scope.

## Decision rule and scope

V24/V25 supersede the quantization/distillation headline comparisons below. They retain the entire declared held-out panels without V18's outcome-based cliff filter and report each capability separately. Their paired bootstrap units and calibration requirements are defined in their sections; the older V18 conventions in the following paragraphs apply only to historical tables.

Headline metrics below are pooled predictions for observations excluded from their fold's fit. MAE is in CE nats; relative error is MAE divided by mean absolute held-out damage. Brackets on MAE/relative error are deterministic 1,000-resample held-out-cell bootstrap 95% intervals; sign brackets and calibration brackets are Wilson 95% intervals. Calibration is empirical coverage of a nominal 95% absolute-residual interval calibrated on that fold's training residuals. These intervals quantify variation across available cells, not benchmark-item or seed uncertainty.

Every compression-damage fit uses the V17 cliff filter: scan each `(model, method, capability)` trajectory toward stronger compression; the first `Delta L_c > 1.0` nat cell and all deeper cells are excluded. Dense anchors are never scored. Recovery has no varying compression coordinate, so each recovery observation is admitted only when its own `Delta L_c <= 1.0` nat. Negative deltas remain eligible; censoring them would invalidate sign evaluation.

`r_storage=b/16` for quantization is a nominal bit budget, not serialized size or runtime memory. Unstructured pruning density is likewise nominal and does not imply sparse-kernel speedup.

Status `OK` means the declared fold produced predictions for every held-out eligible cell. `PARTIAL` means a requested axis is underidentified or some held-out cells are not estimable. `PLANNED` names the missing data. `NON-DECISIONAL` is an in-sample diagnostic and never a headline.

## Held-out findings

- **Pruning remains on probation.** Best candidate-versus-baseline held-out MAEs are `fit d>=0.55 -> deeper pre-cliff`: candidate `density_only` 0.333 vs baseline `baseline_raw_sparsity` 0.326; `leave-largest-model-out`: candidate `hierarchical_capability_family` 0.283 vs baseline `baseline_anchor_only` 0.289; `leave-one-family-out`: candidate `density_only` 0.339 vs baseline `baseline_removed_weight_norm` 0.304. No candidate clears the directive's no-held-out-sign-error rule.
- **Quantization: V24 frozen 5-bit law versus best simple baseline.** math 0.07918 vs zero_change 0.09263 (advantage unresolved); code 0.06197 vs zero_change 0.11025 (advantage unresolved); qa 0.11979 vs zero_change 0.12229 (advantage unresolved). The law uses three own-model compressed calibration points; the dense-only LOMO audit quantifies their cost.
- **Distillation: V25 evaluates signed own-dense changes against zero and mean corrections.** See per-capability size/budget holdouts, paired complexity contrasts and the LoRA-only sensitivity below; low total-loss MAE alone is not evidence of transfer. The main formulation is dense baseline plus transfer response; source-referenced size gaps are historical.
- **Recovery is PARTIAL:** C4 has a diagnostic QA-only held-out-budget score after filtering; aligned traces have no estimable held-out law because only one budget per capability remains below the 1-nat cap.
- **Cliffs are separately held out:** pruning cliff-density MAE is 0.078 for largest-model holdout and 0.146 for family holdout; quantization bit-cliff MAE is 0.238 bits for model holdout and 0.422 bits for family holdout.

## Paired law-versus-baseline bootstrap

The comparison below resamples common held-out cells and recomputes `candidate MAE - baseline MAE` in each resample. Negative favors the candidate. It is conditional on the displayed candidates having been selected by pooled MAE and therefore does not correct selection bias. A CI containing zero supplies no evidence that added form complexity improves prediction.

| Method | Protocol | Candidate | Baseline | Paired cells | Paired bootstrap ΔMAE candidate−baseline (95% CI) | Simplicity decision |
|---|---|---|---|---|---|---|
| Pruning | fit d>=0.55 -> deeper pre-cliff | density_only (0.333) | baseline_raw_sparsity (0.326) | 28 | 0.007 [-0.032, 0.043] | unresolved; prefer density_only |
| Pruning | leave-largest-model-out | hierarchical_capability_family (0.283) | baseline_anchor_only (0.289) | 55 | -0.006 [-0.074, 0.062] | unresolved; prefer density_only |
| Pruning | leave-one-family-out | density_only (0.339) | baseline_removed_weight_norm (0.304) | 174 | 0.036 [0.026, 0.046] | baseline lower error; prefer density_only |
| Quantization | leave-one-bit-out (pre-cliff only) | family_conditioned_smooth_cliff (0.121) | categorical baseline not estimable | 0 | n/a | historical comparison; use V24 simple baselines below |
| Quantization | leave-largest-model-out | family_conditioned_smooth_cliff (0.091) | baseline `bit_width_categorical_baseline` 0.106 | 26 | -0.043 [-0.087, -0.007] | candidate lower error |
| Quantization | leave-one-family-out | fixed_4^-b (0.133) | baseline `bit_width_categorical_baseline` 0.109 | 103 | 0.011 [-0.006, 0.027] | unresolved; prefer fixed_4^-b |

**Simplicity verdict.** The historical pruning comparisons support keeping the reduced density-only form on probation. Quantization's decisive comparison is the V24 frozen 5-bit test against zero-change, nearest-bit and interpolation below. The absence of an unseen category in a categorical estimator is not evidence that a continuous law beats simple predictive baselines; use the paired best-simple gains reported in V24.

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

### V24: frozen 5-bit predictions versus simple baselines

CPU-only analysis of all 12 preregistered models × 3 capabilities. All candidates score the SAME 5-bit cells; no observed-loss/cliff filter, no sign censoring, and no 5-bit refit. The frozen JSON predictions are used as recorded (four decimal places). MAE units are CE nats.

**Calibration audit:** q_c was fitted separately for each model and capability using that model's OWN 8-, 6-, and 4-bit ΔL values. The law is ΔL=q_c(4^-b−4^-16), with a dense anchor, one fitted coefficient, and **three compressed calibration points**. This is a CALIBRATED prediction, not a basic-parameter-only law. Frozen values reconstruct to rounding precision.

Nearest-bit uses ΔL(4); interpolation uses [ΔL(4)+ΔL(6)]/2. The dense-only LOMO variant regresses signed q_c from the other 11 models on log(N0/1e9), family, and dense L_c, separately per capability, with fixed ridge λ=1 and training-only standardization. N0 is consistently the V6 fisher_meta 2-D language-weight count, including embeddings; it is not total checkpoint storage. Unseen-family correction is zero. At inference it uses no compressed measurements of the held-out model, but it still requires that model's measured dense L_c.

Intervals use 10,000 paired model-bootstrap draws (12 models per capability). They describe panel variation conditional on these fits, not item/seed uncertainty or retraining uncertainty. Positive candidate−law favors 4^-b.

| Capability | Candidate | MAE | Candidate MAE − frozen 4^-b (95% CI) |
|---|---|---:|---:|
| math | zero_change | 0.09263 | 0.01344 [-0.05734, 0.09043] |
| math | nearest_4bit | 0.59099 | 0.51181 [0.28933, 0.77432] |
| math | linear_4_6bit | 0.25741 | 0.17823 [0.09852, 0.27268] |
| math | law_frozen | 0.07918 | 0.00000 [0.00000, 0.00000] |
| math | law_reconstructed | 0.07918 | -0.00000 [-0.00002, 0.00001] |
| math | dense_only_lomo | 0.09289 | 0.01370 [-0.06044, 0.07124] |
| code | zero_change | 0.11025 | 0.04829 [-0.00355, 0.11360] |
| code | nearest_4bit | 0.55635 | 0.49438 [0.23071, 0.83327] |
| code | linear_4_6bit | 0.22978 | 0.16781 [0.07615, 0.28500] |
| code | law_frozen | 0.06197 | 0.00000 [0.00000, 0.00000] |
| code | law_reconstructed | 0.06197 | 0.00001 [-0.00001, 0.00002] |
| code | dense_only_lomo | 0.13408 | 0.07211 [0.02020, 0.13184] |
| qa | zero_change | 0.12229 | 0.00249 [-0.06054, 0.06122] |
| qa | nearest_4bit | 0.39782 | 0.27803 [0.11039, 0.46431] |
| qa | linear_4_6bit | 0.19530 | 0.07551 [0.00971, 0.14759] |
| qa | law_frozen | 0.11979 | 0.00000 [0.00000, 0.00000] |
| qa | law_reconstructed | 0.11979 | -0.00000 [-0.00001, 0.00001] |
| qa | dense_only_lomo | 0.16859 | 0.04880 [-0.02472, 0.11743] |

| Capability | Best simple baseline | Improvement of 4^-b over best simple (95% CI) | Dense-only minus exact calibrated MAE (95% CI) |
|---|---|---:|---:|
| math | zero_change | 0.01344 [-0.05734, 0.09043] | 0.01370 [-0.06044, 0.07125] |
| code | zero_change | 0.04829 [-0.00355, 0.11360] | 0.07211 [0.02019, 0.13184] |
| qa | zero_change | 0.00249 [-0.06054, 0.05344] | 0.04880 [-0.02471, 0.11743] |

The best-simple column takes the minimum simple-baseline MAE within each bootstrap draw; per-candidate intervals keep the named comparator fixed. These exploratory comparisons do not supply a new independent validation set. A positive calibration-cost column quantifies the error reduction bought by measuring three compressed calibration points, for this specified dense-only estimator; it is not an optimal-estimator bound.

- math: 4^-b has no resolved advantage over the best simple baseline.
- code: 4^-b has no resolved advantage over the best simple baseline.
- qa: 4^-b has no resolved advantage over the best simple baseline.

Qwen-0.6B's input records a 5-bit dense-protocol discrepancy (QA 0.0133 nats); the canonical dense anchor is retained for every candidate. This is a measurement caveat, not silently corrected. The numerical preregistration audit does not independently verify its historical freeze timestamp.

Reproduce: `python analysis/v24_quant_baselines.py --dry-run`, then the same command without `--dry-run`. Cell predictions, fold coefficients and SHA-256 input hashes: `results/v24-quant-baselines/summary.json`.

### Historical V18 quantization candidates

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

### V25: dense baseline plus signed transfer response

The main formulation is **δ_c=L_c(S_KD)−L_c(S0)** and **L̂_c=L_c(S0)+δ̂_c**. S0 is the student's own dense model; negative δ means lower loss. Each cell uses its paired eval.json dense anchor, so MAE in predicted L equals MAE in predicted δ. No reference-model gap enters the fit. Math, code and QA are fitted and scored separately, without pooled MAE.

Grid: Gemma-3 270M/1B/4B × D=75/150/300/600 traces PER DOMAIN, teacher gpt-5.6-luna, recipe full. Fixed units: D_*=150 traces/domain and N_*=10^9 nominal parameters. u=log(1+D/D_*), v=log(N_S/N_*); units and forms are not selected by test error. The low-parameter response is δ̂_c=a_c u+b_c u²+k_c uv (OLS, no intercept). All response variants satisfy δ̂(D=0)=0. The training-mean comparator is intentionally a constant correction and need not satisfy this boundary.

Four main candidates: zero (dense student), training mean δ, data-only u+u², and size×data u+u²+uv. The u and u+uv ablations isolate curvature and interaction. Coefficients and means are fitted only on each fold's development rows. Entire-size and entire-budget holdouts are the main tests; leave-cell-out is an interpolation diagnostic. Every candidate scores the same cells within each protocol.

**Recipe confound:** Gemma-3 4B at D=600 uses full training; the other 11 cells use LoRA. The requested full grid is retained and a LoRA-only sensitivity excludes that cell from both training and testing. D effects in the full grid cannot be assigned solely to data volume. Small dense-anchor differences across runs are recorded, not silently replaced.

#### full_grid: held-out MAE in nats

| Protocol | Capability | Cells | Dense δ=0 | Mean δ | u | Data-only u+u² | u+uv | Size×data u+u²+uv |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| leave_size_out | math | 12 | 0.08952 | 0.02969 | 0.03358 | 0.03574 | 0.03146 | 0.03189 |
| leave_size_out | code | 12 | 0.10125 | 0.05002 | 0.03312 | 0.03687 | 0.03410 | 0.03716 |
| leave_size_out | qa | 12 | 1.14667 | 0.24763 | 0.57360 | 0.25080 | 0.61448 | 0.19609 |
| leave_budget_out | math | 12 | 0.08952 | 0.03134 | 0.04674 | 0.02686 | 0.04638 | 0.02908 |
| leave_budget_out | code | 12 | 0.10125 | 0.06294 | 0.04761 | 0.07878 | 0.04761 | 0.08031 |
| leave_budget_out | qa | 12 | 1.14667 | 0.24194 | 0.83700 | 0.47316 | 0.87193 | 0.46116 |
| leave_cell_out | math | 12 | 0.08952 | 0.02979 | 0.03568 | 0.02832 | 0.04096 | 0.03283 |
| leave_cell_out | code | 12 | 0.10125 | 0.05457 | 0.03498 | 0.03667 | 0.04070 | 0.04181 |
| leave_cell_out | qa | 12 | 1.14667 | 0.23052 | 0.63463 | 0.24237 | 0.74091 | 0.23139 |
#### lora_only: held-out MAE in nats

| Protocol | Capability | Cells | Dense δ=0 | Mean δ | u | Data-only u+u² | u+uv | Size×data u+u²+uv |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| leave_size_out | math | 11 | 0.09026 | 0.03309 | 0.03829 | 0.03793 | 0.05853 | 0.05020 |
| leave_size_out | code | 11 | 0.10266 | 0.05327 | 0.03166 | 0.03287 | 0.06404 | 0.06542 |
| leave_size_out | qa | 11 | 1.14018 | 0.26262 | 0.60468 | 0.22611 | 0.62199 | 0.38939 |
| leave_budget_out | math | 11 | 0.09026 | 0.03451 | 0.03908 | 0.02528 | 0.02680 | 0.01759 |
| leave_budget_out | code | 11 | 0.10266 | 0.07001 | 0.03553 | 0.06058 | 0.03352 | 0.05675 |
| leave_budget_out | qa | 11 | 1.14018 | 0.26887 | 0.79730 | 0.35606 | 0.87105 | 0.37782 |
| leave_cell_out | math | 11 | 0.09026 | 0.03203 | 0.03519 | 0.02934 | 0.02558 | 0.02180 |
| leave_cell_out | code | 11 | 0.10266 | 0.05833 | 0.02850 | 0.03111 | 0.03231 | 0.03815 |
| leave_cell_out | qa | 11 | 1.14018 | 0.24839 | 0.67450 | 0.20881 | 0.77306 | 0.25090 |

#### Paired gains and complexity tests, full grid

Differences below are candidate MAE minus comparator MAE; negative favors the added response. 95% intervals use 10,000 paired model-bootstrap draws, only THREE model clusters. They are descriptive, conditional on overlapping fitted CV folds, and do not measure independent seed/item or retraining uncertainty. No multiplicity correction is applied.

| Protocol | Capability | Contrast | ΔMAE (95% CI) |
|---|---|---|---:|
| leave_size_out | math | u_u2_minus_zero | -0.05378 [-0.06452, -0.04097] |
| leave_size_out | math | u_u2_minus_mean | 0.00604 [0.00208, 0.01205] |
| leave_size_out | math | u_u2_uv_minus_zero | -0.05763 [-0.06645, -0.04103] |
| leave_size_out | math | u_u2_uv_minus_mean | 0.00219 [-0.00748, 0.01012] |
| leave_size_out | math | u_u2_minus_u | 0.00216 [-0.01271, 0.02325] |
| leave_size_out | math | u_uv_minus_u | -0.00212 [-0.00441, -0.00002] |
| leave_size_out | math | u_u2_uv_minus_u_u2 | -0.00385 [-0.00955, -0.00006] |
| leave_size_out | code | u_u2_minus_zero | -0.06438 [-0.07945, -0.05015] |
| leave_size_out | code | u_u2_minus_mean | -0.01316 [-0.02333, 0.00338] |
| leave_size_out | code | u_u2_uv_minus_zero | -0.06409 [-0.07678, -0.05220] |
| leave_size_out | code | u_u2_uv_minus_mean | -0.01286 [-0.02066, 0.00134] |
| leave_size_out | code | u_u2_minus_u | 0.00375 [-0.00255, 0.00914] |
| leave_size_out | code | u_uv_minus_u | 0.00098 [-0.00205, 0.00497] |
| leave_size_out | code | u_u2_uv_minus_u_u2 | 0.00029 [-0.00205, 0.00267] |
| leave_size_out | qa | u_u2_minus_zero | -0.89586 [-1.03804, -0.76781] |
| leave_size_out | qa | u_u2_minus_mean | 0.00318 [-0.10826, 0.15112] |
| leave_size_out | qa | u_u2_uv_minus_zero | -0.95058 [-1.03754, -0.85332] |
| leave_size_out | qa | u_u2_uv_minus_mean | -0.05154 [-0.11884, 0.07198] |
| leave_size_out | qa | u_u2_minus_u | -0.32280 [-0.53048, -0.14342] |
| leave_size_out | qa | u_uv_minus_u | 0.04088 [0.00050, 0.10895] |
| leave_size_out | qa | u_u2_uv_minus_u_u2 | -0.05472 [-0.08550, 0.00050] |
| leave_budget_out | math | u_u2_minus_zero | -0.06267 [-0.07422, -0.05636] |
| leave_budget_out | math | u_u2_minus_mean | -0.00448 [-0.01605, 0.00296] |
| leave_budget_out | math | u_u2_uv_minus_zero | -0.06044 [-0.06274, -0.05674] |
| leave_budget_out | math | u_u2_uv_minus_mean | -0.00226 [-0.01644, 0.01113] |
| leave_budget_out | math | u_u2_minus_u | -0.01989 [-0.03714, -0.00760] |
| leave_budget_out | math | u_uv_minus_u | -0.00036 [-0.01231, 0.01148] |
| leave_budget_out | math | u_u2_uv_minus_u_u2 | 0.00223 [-0.00442, 0.01148] |
| leave_budget_out | code | u_u2_minus_zero | -0.02247 [-0.04631, -0.00430] |
| leave_budget_out | code | u_u2_minus_mean | 0.01584 [-0.00549, 0.04109] |
| leave_budget_out | code | u_u2_uv_minus_zero | -0.02093 [-0.05458, 0.00873] |
| leave_budget_out | code | u_u2_uv_minus_mean | 0.01737 [-0.01376, 0.05412] |
| leave_budget_out | code | u_u2_minus_u | 0.03117 [0.02606, 0.03428] |
| leave_budget_out | code | u_uv_minus_u | -0.00000 [-0.01278, 0.01303] |
| leave_budget_out | code | u_u2_uv_minus_u_u2 | 0.00153 [-0.00827, 0.01303] |
| leave_budget_out | qa | u_u2_minus_zero | -0.67351 [-0.82079, -0.58517] |
| leave_budget_out | qa | u_u2_minus_mean | 0.23122 [0.08057, 0.46950] |
| leave_budget_out | qa | u_u2_uv_minus_zero | -0.68551 [-0.82101, -0.57354] |
| leave_budget_out | qa | u_u2_uv_minus_mean | 0.21922 [0.08035, 0.48113] |
| leave_budget_out | qa | u_u2_minus_u | -0.36385 [-0.57390, -0.10294] |
| leave_budget_out | qa | u_uv_minus_u | 0.03493 [0.00131, 0.06833] |
| leave_budget_out | qa | u_u2_uv_minus_u_u2 | -0.01200 [-0.04740, 0.01163] |

#### Interpretation

- math: observed δ range 0.01853 to 0.16589 nats.
  leave_size_out: neither main response beats both simple baselines in point MAE.
  leave_budget_out: u_u2, u_u2_uv beat both simple baselines in point MAE.
- code: observed δ range 0.01019 to 0.19771 nats.
  leave_size_out: u_u2, u_u2_uv beat both simple baselines in point MAE.
  leave_budget_out: neither main response beats both simple baselines in point MAE.
- qa: observed δ range -1.48590 to -0.65329 nats.
  leave_size_out: u_u2_uv beat both simple baselines in point MAE.
  leave_budget_out: neither main response beats both simple baselines in point MAE.

Low total-loss MAE alone does not demonstrate learned transfer. Curvature/interaction claims require improvement over the constant correction and simpler response on these held-out axes, with uncertainty and the recipe sensitivity considered. QA is loss-space only; lower CE does not establish better QA accuracy. One family, one teacher and effectively one training realization per cell limit generalization.

#### Full-grid coefficients (NON-DECISIONAL)

All fits below use all 12 cells per capability, solely to specify the response; held-out comparisons above use freshly fitted development-only coefficients.

| Capability | Form | a (u) | b (u²) | k (uv) |
|---|---|---:|---:|---:|
| math | u | 0.08332 | 0.00000 | 0.00000 |
| math | u_u2 | 0.18166 | -0.07411 | 0.00000 |
| math | u_uv | 0.08305 | 0.00000 | 0.01030 |
| math | u_u2_uv | 0.18140 | -0.07411 | 0.01030 |
| code | u | 0.10391 | 0.00000 | 0.00000 |
| code | u_u2 | 0.16489 | -0.04596 | 0.00000 |
| code | u_uv | 0.10409 | 0.00000 | -0.00693 |
| code | u_u2_uv | 0.16507 | -0.04596 | -0.00693 |
| qa | u | -0.92981 | 0.00000 | 0.00000 |
| qa | u_u2 | -2.98776 | 1.55086 | 0.00000 |
| qa | u_uv | -0.92757 | 0.00000 | -0.08716 |
| qa | u_u2_uv | -2.98553 | 1.55086 | -0.08716 |

#### Simpler response and recipe sensitivity

- math: leave_size_out: u does not beat both simple baselines in point MAE; adding u² changes MAE by 0.00216; leave_budget_out: u does not beat both simple baselines in point MAE; adding u² changes MAE by -0.01989; LoRA-only size holdout: size×data MAE 0.05020 versus constant 0.03309.
- code: leave_size_out: u beats both simple baselines in point MAE; adding u² changes MAE by 0.00375; leave_budget_out: u beats both simple baselines in point MAE; adding u² changes MAE by 0.03117; LoRA-only size holdout: size×data MAE 0.06542 versus constant 0.05327.
- qa: leave_size_out: u does not beat both simple baselines in point MAE; adding u² changes MAE by -0.32280; leave_budget_out: u does not beat both simple baselines in point MAE; adding u² changes MAE by -0.36385; LoRA-only size holdout: size×data MAE 0.38939 versus constant 0.26262.

Full-data coefficients are NON-DECISIONAL diagnostics in summary.json. The JSON also includes every fold, δ̂ and L̂, simple-baseline contrasts for all ablations, and LoRA-only contrasts. Reproduce: `python analysis/v25_distill_delta.py --dry-run`, then without `--dry-run`. Outputs: `results/v25-distill-delta/`.

### HISTORICAL candidates: source-referenced size gaps (V21)

Updated 2026-09-04 by `analysis/v21_distill_law.py`, CPU-only, from existing V6/V12 JSON artifacts. The size-law response is source-referenced; the D-ladder response is explicitly student-own-dense-referenced.

#### Held-out rows and complexity

| Test | Law / fit-to-test protocol | Added parameters per capability | Held-out cells | MAE (nats) | MAE by capability | Sign accuracy | Status |
|---|---|---:|---:|---:|---|---|---|
| Within Gemma size rotation | `floor_c-alpha_c log r`; fit 3 sizes, predict 4th | n/a | 12 | 0.164 | math 0.153, code 0.090, qa† 0.249 | 11/12 (0.917) | PARTIAL: one family |
| Shared cross-family, no refit | fit all 4 Gemma sizes, predict both Qwen sizes | 0 | 6 | 0.475 | math 0.195, code 0.167, qa† 1.062 | 4/6 (0.667) | HELD OUT |
| Hierarchical family intercept | keep Gemma exponent; fit Qwen intercept on one size, predict the other, both rotations | +1 | 6 | 0.106 | math 0.079, code 0.082, qa† 0.157 | 6/6 (1.000) | HELD OUT |
| Family-specific Qwen exponent | fit Qwen intercept + exponent on both Qwen sizes | +2 | 0 | n/a | n/a | n/a | NON-DECISIONAL: two parameters saturate two sizes |
| Gemma-3 4B D-only | `a_c+b_c(D/600)^(-beta_c)`; leave one D out | 0 | 12 | 0.615 | math 0.073, code 0.114, qa† 1.657 | 12/12 (1.000) | PARTIAL: one fixed-size D ladder |
| Separable Gemma size + D | add `-alpha_c log r`; leave the same 4B D out and train on remaining D plus other D=600 sizes | +1 | 12 | 0.632 | math 0.059, code 0.073, qa† 1.762 | 12/12 (1.000) | PARTIAL: one fixed-size D ladder |

#### Cross-family source-referenced size test

The no-refit row is the strict family holdout: no Qwen outcome participates in its Gemma fit. The hierarchical row spends exactly one Qwen calibration parameter per capability in each fold. The Qwen-specific exponent row cannot be honestly held out with only two sizes, so its zero in-sample residual is not used for selection.

| Qwen held-out size | r_storage | Capability | Observed source-referenced ΔL | Shared prediction | Hierarchical held-out prediction |
|---|---:|---|---:|---:|---:|
| Qwen3-0.6B | 0.150 | math | 0.093 | 0.328 | 0.173 |
| Qwen3-0.6B | 0.150 | code | 0.121 | 0.248 | 0.039 |
| Qwen3-0.6B | 0.150 | qa† | -2.385 | -1.402 | -2.543 |
| Qwen3-1.7B | 0.425 | math | -0.034 | 0.122 | -0.113 |
| Qwen3-1.7B | 0.425 | code | -0.104 | 0.105 | -0.022 |
| Qwen3-1.7B | 0.425 | qa† | -2.683 | -1.542 | -2.526 |

Family-specific coefficients below are descriptive only because the Qwen line has exactly two points.

| Capability | Gemma floor | Gemma alpha | Qwen saturated floor | Qwen saturated alpha | Trend sign matches? |
|---|---:|---:|---:|---:|---|
| math | -0.048 | 0.198 | -0.138 | 0.122 | yes |
| code | -0.013 | 0.137 | -0.289 | 0.216 | yes |
| qa† | -1.657 | 0.135 | -2.928 | 0.286 | yes |

**V21 recorded verdict: HIERARCHICAL; superseded in interpretation by V22 below.** The 0.475→0.106 held-out MAE and 4/6→6/6 signs remain reproducible for the source-referenced total gap. These do not isolate the distillation effect or establish an intrinsic family offset. The family-specific exponent remains untested.

This cross-family claim is limited: Gemma uses a 27B dense source whereas Qwen uses a 4B dense source. Their absolute source capabilities and source sizes differ, so equal nominal `r_storage` does not represent a perfectly matched capacity gap.

#### Gemma-3 4B data-ladder separability

The observed own-dense loss deltas are:

| D per domain | math | code | qa† | recorded training mode |
|---:|---:|---:|---:|---|
| 75 | 0.077 | 0.021 | -1.327 | lora |
| 150 | 0.137 | 0.110 | -1.459 | lora |
| 300 | 0.166 | 0.198 | -0.967 | lora |
| 600 | 0.081 | 0.086 | -1.218 | full |

All three observed sequences are non-monotone, so neither candidate can reproduce their ordering at fixed size. The fitted beta values below are constrained-predictor diagnostics, not clean scaling exponents.

| Capability | Observed monotone? | D-only beta (at bound?) | Separable beta (at bound?) |
|---|---|---:|---:|
| math | no | 4.000 (yes) | 4.000 (yes) |
| code | no | 3.089 (no) | 2.935 (no) |
| qa† | no | 0.391 (no) | 0.087 (no) |

**D-ladder verdict: no clean monotone data exponent is supported.** The D-only held-out MAE is 0.615 nats; adding the separable size term gives 0.632 on the same 12 held-out cells. Regardless of which error is smaller, both impose monotonicity contradicted by every observed capability sequence (math 0.077→0.137→0.166→0.081; code and QA also reverse direction). At one fixed 4B size, `a_c` and `-alpha_c log r_storage` are collinear; the separable row identifies alpha only by borrowing the D=600 Gemma size ladder, so this remains a partial separability test.

† **QA loss-space caveat:** V19 found that QA loss gains do not reliably track accuracy. Every negative QA delta and every QA prediction above is loss-space only and must not be described as a QA accuracy or behavioral improvement.

Additional design caveat: the V12 metadata records LoRA for D=75/150/300 and `training_mode=full` for D=600 at Gemma-3 4B. `recipe=full` is constant, but training mode is not; this can confound the apparent D effect. The size ladders likewise mix recorded training modes. There is one recorded probe seed, so these cell-level comparisons do not estimate seed or benchmark-item uncertainty.

Provenance: `results/v12-distill/{gemma3-270m,gemma3-1b,gemma3-4b,gemma3-12b,Qwen3-0.6B,Qwen3-1.7B}/gpt-5.6-luna_full_*/eval.json` and dense key `1.0` from `results/v6-capability-geometry/{gemma3-27b,Qwen3-4B}/prune_losses.json`. Machine-readable records and every fold's train/test row IDs are in `results/v21-distill-law/summary.json`.

## 4. Recovery

The same Gemma-3 1B prune-0.6 anchor feeds both sources. Recovery is fitted to normalized damage `z_c(D_R)=Delta L_c(D_R)/Delta L_c(0)`, making the displayed saturation law dimensionless and anchored at one. C4 is assessed only with the monotonic saturation form. Traces compare monotonic saturation, change-point, and early-recovery-plus-late-penalty, but the comparison is underdetermined. In particular, enforcing the common pre-cliff rule excludes the 4M/16M aligned points that exhibit late re-damage; fitting a penalty to them would violate the report's own scope.

| Method | Inputs | Candidate law | Fit points | Held-out split | MAE | Relative error | Sign accuracy | Cliff error | Calibration |
|---|---|---|---|---|---|---|---|---|---|
| Recovery [PARTIAL] | Delta L_c(0), D_R, source, capability | r_c+(1-r_c)(1+D_R/D0)^(-beta) | 2/fold; 3 held-out | leave-one-budget-out (c4); PARTIAL: only QA retains 3 pre-cliff budgets; math/code retain 2/1 | 0.232 [0.007, 0.621] | 1.018 [0.999, 1.239] | 0.667 [0.208, 0.939] | n/a | 95% PI cover 0.333 [0.061, 0.792], width 0.075 |
| Recovery [NON-DECISIONAL] | Delta L_c(0), D_R, source, capability | r_c+(1-r_c)(1+D_R/D0)^(-beta) | 3 | NON-DECISIONAL in-sample (c4; underidentified) | 0.021 [0.000, 0.056] | 0.093 [0.000, 1.016] | 0.667 [0.208, 0.939] | n/a | n/a |
| Recovery [PARTIAL] | Delta L_c(0), D_R, source, capability | r_c+(1-r_c)(1+D_R/D0)^(-beta) | 0 | leave-one-budget-out (traces); PARTIAL: only the 1M point is pre-cliff in each capability; 4M/16M exceed 1 nat | n/a | n/a | n/a | n/a | n/a |
| Recovery [NON-DECISIONAL] | Delta L_c(0), D_R, source, capability | r_c+(1-r_c)(1+D_R/D0)^(-beta) | 3 | NON-DECISIONAL in-sample (traces; underidentified) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [0.439, 1.000] | n/a | n/a |
| Recovery [PARTIAL] | Delta L_c(0), D_R, source, capability | 1+a log(1+D_R/1M)+c[log(1+D_R/1M)-tau]_+ | 0 | leave-one-budget-out (traces); PARTIAL: only the 1M point is pre-cliff in each capability; 4M/16M exceed 1 nat | n/a | n/a | n/a | n/a | n/a |
| Recovery [NON-DECISIONAL] | Delta L_c(0), D_R, source, capability | 1+a log(1+D_R/1M)+c[log(1+D_R/1M)-tau]_+ | 3 | NON-DECISIONAL in-sample (traces; underidentified) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [0.439, 1.000] | n/a | n/a |
| Recovery [PARTIAL] | Delta L_c(0), D_R, source, capability | (1+D_R/D0)^(-beta)+kappa(D_R/1M)^p | 0 | leave-one-budget-out (traces); PARTIAL: only the 1M point is pre-cliff in each capability; 4M/16M exceed 1 nat | n/a | n/a | n/a | n/a | n/a |
| Recovery [NON-DECISIONAL] | Delta L_c(0), D_R, source, capability | (1+D_R/D0)^(-beta)+kappa(D_R/1M)^p | 3 | NON-DECISIONAL in-sample (traces; underidentified) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [0.439, 1.000] | n/a | n/a |

Aligned availability audit (not fitted): all three requested budgets exist for math/code/QA; at 1M all three are pre-cliff, while every 4M/16M capability cell exceeds the 1-nat cap. Thus each aligned candidate row is emitted as `PARTIAL` with no held-out estimate, rather than fitting forbidden points or presenting a three-point interpolation as prediction.

Eligible recovery counts after the 1-nat cap: c4: math=0, code=0, qa=3; traces: math=1, code=1, qa=1. A defensible aligned held-out comparison needs at least three pre-cliff training budgets plus one held-out budget per capability, denser sampling around the early optimum, and another seed/model. The existing late points remain descriptive evidence of non-monotonicity, not fit points.

## Provenance and limitations

- Pruning: `results/v6-capability-geometry/*/{prune_losses.json,alignment.json,spectrum_bins.npz,fisher_meta.json}`; V6 artifacts do not record a run seed in these files.
- Quantization: `results/v10-quantization/*/quant_losses.json`; nominal `b/16`; these compact loss files do not record a seed.
- Distillation: selected V16 `gpt-5.6-luna_full_600/residual.json` files plus the V6 Gemma-3 27B dense reference; the selected V16 records use seed 0.
- Recovery: `results/v13-recovery/gemma3-1b/prune_0.6_{c4,traces}/recovery.json`; traces records seed 0, while the salvaged C4 compact artifact does not record a seed.
- There are no repeated measurement seeds for these law cells. Consequently the calibration/MAE intervals are cell-sampling diagnostics and cannot estimate seed or benchmark-item uncertainty.
- No post-cliff loss magnitude contributes to any law coefficient or headline damage metric. Cliff-location models use only observed threshold-crossing coordinates; censored trajectories stay censored.

<!-- V22_DISTILL_DECOMP_START -->
## Stage A / V22: distillation decomposition and C21 reassessment

**Verdict: PARTIAL / EXPLORATORY.** The held-out 0.106 calculation is reproducible, but it supports a calibrated predictor of the combined baseline-plus-training gap. It does not establish a distillation-specific hierarchical family law.

### Paired loss accounting

`L(S_KD)-L(B_ref) = [L(S0)-L(B_ref)] + [L(S_KD)-L(S0)]`.

The first term is the student's own dense baseline gap: pre-existing size/quality, not distillation. The signed DISTILLATION GAIN is negative when training lowers loss and positive when it damages loss. B_ref is a same-family comparison checkpoint, not the API teacher. References are Gemma-3 27B and Qwen-3 4B; the single OLMo student is included separately against OLMo-3 32B and excluded from C21 fits.

Coverage: 23 V12 runs and 16 V16 remeasurements; all three capabilities for each. `decomposition.csv` and `summary.json` contain every term and its artifact paths. V12 reproduces C21; V16 pairs are retained as separate measurements. No V16 generic residual is substituted for capability loss.

The following table is the C21 GPT/full/600 slice; all other teachers, recipes and budgets are in the complete table below.

| Student | Capability | BASELINE GAP | DISTILLATION GAIN | Total source gap |
|---|---|---:|---:|---:|
| Qwen3-0.6B | math | +0.092546 | +0.000569 | +0.093115 |
| Qwen3-0.6B | code | +0.044785 | +0.076495 | +0.121280 |
| Qwen3-0.6B | qa | -1.103231 | -1.281884 | -2.385115 |
| Qwen3-1.7B | math | +0.096167 | -0.130148 | -0.033980 |
| Qwen3-1.7B | code | +0.056342 | -0.160322 | -0.103980 |
| Qwen3-1.7B | qa | -0.337837 | -2.345076 | -2.682913 |
| gemma3-12b | math | +0.042478 | +0.099220 | +0.141698 |
| gemma3-12b | code | +0.016580 | +0.138424 | +0.155004 |
| gemma3-12b | qa | -0.354971 | -1.186239 | -1.541210 |
| gemma3-1b | math | +0.618626 | +0.119412 | +0.738038 |
| gemma3-1b | code | +0.326146 | +0.138314 | +0.464460 |
| gemma3-1b | qa | -0.218641 | -0.747323 | -0.965964 |
| gemma3-270m | math | +0.714114 | +0.094332 | +0.808446 |
| gemma3-270m | code | +0.456350 | +0.173134 | +0.629484 |
| gemma3-270m | qa | -0.510692 | -0.653293 | -1.163985 |
| gemma3-4b | math | +0.138818 | +0.081423 | +0.220242 |
| gemma3-4b | code | +0.067842 | +0.085695 | +0.153537 |
| gemma3-4b | qa | -0.312578 | -1.218003 | -1.530581 |

### Where the fitted size slope comes from

Identical OLS designs imply `alpha_total = alpha_baseline + alpha_gain` exactly. These are descriptive fits on the four Gemma sizes, not new held-out exponent estimates.

| Capability | Total alpha | Baseline alpha | Gain alpha | Baseline / total |
|---|---:|---:|---:|---:|
| math | 0.198289 | 0.196089 | 0.002200 | 98.9% |
| code | 0.137244 | 0.124011 | 0.013232 | 90.4% |
| qa | 0.134813 | -0.029674 | 0.164487 | -22.0% |

The negative QA baseline fraction means opposing components, not an explained-variance share. Math/code slopes largely reflect dense size/quality. The QA total slope hides cancellation.

| Response fitted and tested | Gemma→Qwen no-refit MAE | Rotating per-capability intercept MAE |
|---|---:|---:|
| source_referenced_gap | 0.475053 | 0.106379 |
| baseline_gap | 0.194437 | 0.361013 |
| distillation_gain | 0.324245 | 0.414450 |

Baseline and gain prediction errors can cancel in their sum; success on the total therefore does not validate prediction of the training effect.

### Reference and coordinate accounting before a family interpretation

Let `x=-log(r_storage)`. Moving Qwen to the Gemma 27B coordinate gives `x_common=x_Q+log(27/4)`; moving its loss gap to the Gemma source gives `y_common=y_Q+L_ref,Q-L_ref,G`. Thus a reference-corrected prediction in the original Qwen units is `p_Q + k_c`, where `k_c=(L_ref,G-L_ref,Q)+alpha_c*log(27/4)`. This uses only dense reference losses and Gemma-fitted alpha, no Qwen student outcomes. It retains V21's registered ratios (.010/.036/.157/.445 and .150/.425); these are not exactly nominal student B/27 (implied Gemma sizes .270/.972/4.239/12.015B). This is a coordinate-origin sensitivity, not a harmonized parameter-count refit.

| Capability | Ref-loss shift | Log-coordinate shift | Known correction k | Mean original offset* | Mean remaining offset* | No-refit MAE after correction |
|---|---:|---:|---:|---:|---:|---:|
| math | -0.309941 | +0.378641 | +0.068699 | -0.195463 | -0.264162 | 0.264162 |
| code | -0.811850 | +0.262073 | -0.549777 | -0.167500 | +0.382278 | 0.382278 |
| qa | -0.943125 | +0.257431 | -0.685693 | -1.062197 | -0.376504 | 0.376504 |

*Means use both Qwen sizes and are descriptive calibration summaries, not held-out estimates. Actual fold offsets are listed below. Reference-loss differences include size, quality, and tokenizer/evaluation effects; the residual cannot be identified as an intrinsic family effect from these data.

The uncalibrated pooled MAE changes 0.475053→0.340981, a 28.2% reduction in error, not a causal percentage of a family effect. The correction overshoots code and moves math in the wrong direction; it accounts for part of the large QA offset. The three capabilities do not support one scalar explained fraction. With a freely calibrated intercept this constant correction is absorbed, so the rotated total-gap MAE remains 0.106.

| Calibration → held-out size | Capability | Total offset | Baseline contribution | Gain contribution | Known correction | Remaining offset |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B → Qwen3-0.6B | math | -0.155756 | +0.069046 | -0.224802 | +0.068699 | -0.224455 |
| Qwen3-1.7B → Qwen3-0.6B | code | -0.208663 | +0.061839 | -0.270502 | -0.549777 | +0.341114 |
| Qwen3-1.7B → Qwen3-0.6B | qa | -1.140895 | -0.041791 | -1.099104 | -0.685693 | -0.455202 |
| Qwen3-0.6B → Qwen3-1.7B | math | -0.235169 | -0.138793 | -0.096376 | +0.068699 | -0.303869 |
| Qwen3-0.6B → Qwen3-1.7B | code | -0.126336 | -0.078871 | -0.047465 | -0.549777 | +0.423441 |
| Qwen3-0.6B → Qwen3-1.7B | qa | -0.983499 | -0.776282 | -0.207217 | -0.685693 | -0.297806 |

### What 0.106 does and does not test

V21 fits the Gemma floor/exponent on four sizes, calibrates Qwen offsets on 0.6B and predicts 1.7B, then reverses those roles. 0.106 is pooled **held-out prediction MAE**, not the zero calibration-fit residual. Each fold adds **one offset per capability**, three parameters total, not one shared offset. The exponent is also capability-specific. There are only two Qwen sizes: each capability supplies one independent size contrast. Its two rotated absolute errors are identical; six scores are not six independent size tests. Shared Qwen-4B reference and probe items remain in both folds. This is within-Qwen size transfer after calibration, not a second strict family holdout. Two Qwen points saturate a separate slope/intercept fit; they cannot validate that alternative.

Using complete V16 Gemma pairs where available and the V12 Qwen pairs gives total-gap rotated MAE 0.107302 as a labeled measurement-version sensitivity. V16 and V12 discrepancies are recorded in `measurement_comparison.csv`; missing checkpoint/revision hashes prevent attributing them to numerical error alone. V12 records mixed LoRA/full training modes, including a mode switch at D=600 for Gemma 4B. D is examples per domain, not supervised tokens. Neither size nor D is an isolated training intervention here.

All values are completion-token-weighted CE nats per model-token. Cross-family token units are not invariant. Per-byte/per-character NLL on frozen common text is TO-DO; none was measured. QA loss changes are not claims of QA accuracy improvement.

### Suggested C21 replacement (proposal only; ledger claim unchanged)

C21 — PARTIAL / EXPLORATORY. On the six GPT full-600 students, a Gemma fit of the source-referenced post-training loss gap predicts Qwen with MAE 0.475 nats/token (4/6 signs). Calibrating one Qwen offset per capability on one size and predicting the other, in both rotations, gives MAE 0.106 (6/6 signs). This is a calibrated two-size transfer diagnostic, with three offsets per fold and only one independent size contrast per capability. About 99%/90% of the Gemma math/code log-size slopes are dense baseline-gap effects; they are not distillation gains. Unequal reference losses and 27B-versus-4B coordinates partly account for the offsets in recorded token units. A distillation-specific shared exponent or intrinsic family intercept is not established. A family-specific exponent has no held-out test with two Qwen sizes. D ladders are non-monotone and the 4B ladder changes training mode; no clean D exponent is supported. QA findings are loss-space only. Evidence: results/v22-distill-decomp/report.md and V21 recorded folds.

Complete decomposition: `results/v22-distill-decomp/decomposition.csv`; provenance and folds: `summary.json` in that directory.
<!-- V22_DISTILL_DECOMP_END -->
