# V76: Value of capability conditioning

MAE in nats on the exact frozen confirmation panels. Positive gain is baseline MAE minus per-capability MAE. V76 is a retrospective ablation; its new shared variants were fitted only to historical development data.

## Primary comparison

| Arm | Capability | A: per-cap | B: shared + scale | C: shared + offset | D: shared | B − A [95% CI] | D − A [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|
| pruning | math | 0.198322 | 0.217299 | 0.203786 | 0.168110 | 0.018977 [-0.029150, 0.068998] | -0.030212 [-0.135014, 0.076484] |
| pruning | code | 0.149631 | 0.158850 | 0.163465 | 0.107219 | 0.009219 [-0.011634, 0.027267] | -0.042412 [-0.110851, 0.028850] |
| pruning | qa | 0.158475 | 0.348352 | 0.411190 | 0.568121 | 0.189877 [0.125212, 0.254542] | 0.409645 [0.333226, 0.486065] |
| pruning | macro | 0.168809 | 0.241500 | 0.259480 | 0.281150 | 0.072691 [0.029958, 0.115424] | 0.112341 [0.068803, 0.152955] |
| quantization | math | 0.323413 | 0.334346 | 0.337959 | 0.343919 | 0.010933 [0.002377, 0.016037] | 0.020507 [-0.018889, 0.036856] |
| quantization | code | 0.383784 | 0.374420 | 0.448536 | 0.431217 | -0.009364 [-0.023434, 0.022648] | 0.047433 [-0.151729, 0.168199] |
| quantization | qa | 0.322192 | 0.330591 | 0.312649 | 0.335631 | 0.008399 [-0.017702, 0.025436] | 0.013438 [-0.138518, 0.131340] |
| quantization | macro | 0.343130 | 0.346452 | 0.366381 | 0.370256 | 0.003323 [-0.011865, 0.014475] | 0.027126 [-0.013093, 0.057565] |
| distillation | math | 0.065576 | 0.065576 | 0.065357 | 0.131278 | 0.000000 [0.000000, 0.000000] | 0.065703 [0.065423, 0.066088] |
| distillation | code | 0.033888 | 0.033888 | 0.043650 | 0.094188 | 0.000000 [0.000000, 0.000000] | 0.060300 [0.058963, 0.061342] |
| distillation | qa | 1.235334 | 1.235334 | 0.631627 | 1.119562 | 0.000000 [0.000000, 0.000000] | -0.115771 [-0.127697, -0.095478] |
| distillation | macro | 0.444933 | 0.444933 | 0.246878 | 0.448343 | 0.000000 [0.000000, 0.000000] | 0.003410 [-0.000818, 0.010238] |

## Interpretation

- Pruning: A reduces macro MAE relative to B by 0.072691 nats (30.10%); signed gain 0.072691 [0.029958, 0.115424]. The panel has 5 state clusters, so its interval is descriptive.
- Quantization: A reduces macro MAE relative to B by 0.003323 nats (0.96%); signed gain 0.003323 [-0.011865, 0.014475]. The panel has 3 state clusters, so its interval is descriptive.
- Distillation: A and B are algebraically equivalent, with zero gain up to floating-point rounding. The comparison cannot distinguish per-capability shapes from constant capability scales when every curve is a single coefficient times log(1+E).

Per-capability curves improve pooled MAE over shared + scale under every reported development fitting convention. QA drives the macro improvement; math/code gains are small and their primary state-bootstrap intervals include zero. Shared-only D has lower pooled math/code MAE than A.

The primary gain over shared + scale is small and its state-bootstrap interval includes zero. Its sign changes under the development-row LAD sensitivity; these cells do not establish an advantage for separate capability shapes.

## Data and fitting

- Pruning: exact V53 register, 17 development states / 84 state-density cells / 252 capability rows. Confirmation: 410M@48k, 1.4B@112k, 6.9B@80k at d=0.85,0.675,0.575; 2.8B@16k and @143k at d=0.85,0.75,0.65 (15 cells total).
- Grouped quantization: 6 development states × 9 configurations = 54 cells; 21 frozen V69 confirmation cells. Two confirmation states appeared at different development configurations; the new 1.4B@112k state did not. All 21 cells are scored, with 6,6,9 cells per state.
- Distillation: 25 development trajectories × 4 checkpoints; 12 confirmation trajectories × 3 budgets. Two students share six sampled pools; six pools are the independent bootstrap units. The development fits pool all students, including the four 4B trajectories.

- **equal_information**: Every variant has the same exact development labels and configuration inputs; zero confirmation-label fitting, calibration, selection, or tuning. No source descriptors.
- **A**: V53/V69 median over development states separately for each capability and anchor; V70 OLS a_c log(1+E). Reproduces frozen predictions.
- **shared_curve**: At each pruning density or quantization configuration: median over development states of their arithmetic mean signed response over the three capabilities. Same interpolation as A.
- **B**: Hold shared curve fixed; fit a separate unrestricted signed scale by unregularized least squares against each capability's development median anchors, equal weight per anchor.
- **C**: Hold shared curve fixed; offset is the mean capability-median-anchor residual, equal weight per anchor.
- **correction_formulas**: With m_kc the development capability median and h_k the shared anchor: s_c=sum_k(h_k*m_kc)/sum_k(h_k^2); b_c=mean_k(m_kc-h_k). No scale sign constraint, ridge, intercept in B, or slope refit in C.
- **D**: Shared curve only; no capability correction.
- **distillation**: All 25 trajectories x 4 checkpoints fit jointly with equal checkpoint weight, as in V70. Shared a is OLS on the capability-mean signed response; scales are OLS on the same 100 points. C and D are supplemental; the requested primary contrast is A versus B.
- **pruning_interpolation**: Adjacent linear interpolation/boundary-pair extension, no clipping, no synthetic dense training anchor.
- **quantization_interpolation**: V69 adjacent interpolation/extrapolation in log2(g/128) at each exact b=3,4,5; apply the V69 zero floor after extrapolating each variant's corrected anchors at g=32/512. Interior remains signed.
- **confirmation_weighting**: Equal cells/checkpoints within capability; macro is the equal mean of the three capability MAEs. Quantization states contribute 6,6,9 cells; equal-state sensitivity is also reported.
- **distillation_prediction_inputs**: Frozen planned E=T_planned/D_U; actual exposure overshoot does not replace planned predictions.
- **sensitivity**: Also report mean of capability medians (aggregation order), corrections fitted to individual development rows, and LAD corrections on anchors or development rows. No test-based choice among these conventions.

Full anchors, correction constants, predictions, per-state MAEs, and input SHA256 hashes are in summary.json. B and C use the same full development information as A; parameter counts and label aggregation differ.

## Frozen subpanels

| Arm / panel | Clusters | Capability | A MAE | B MAE | C MAE | D MAE | B − A [95% CI] |
|---|---:|---|---:|---:|---:|---:|---:|
| pruning / v53_confirmation | 3 | math | 0.277260 | 0.258510 | 0.259988 | 0.167148 | -0.018750 [-0.054343, 0.042720] |
| pruning / v53_confirmation | 3 | code | 0.213521 | 0.208733 | 0.194249 | 0.113981 | -0.004788 [-0.029311, 0.015416] |
| pruning / v53_confirmation | 3 | qa | 0.220642 | 0.352859 | 0.548391 | 0.697232 | 0.132217 [0.084838, 0.167241] |
| pruning / v53_confirmation | 3 | macro | 0.237141 | 0.273367 | 0.334209 | 0.326120 | 0.036226 [0.013247, 0.067569] |
| pruning / v72_repeat | 2 | math | 0.079916 | 0.155483 | 0.119482 | 0.169553 | 0.075567 [0.075567, 0.075567] |
| pruning / v72_repeat | 2 | code | 0.053794 | 0.084024 | 0.117289 | 0.097076 | 0.030230 [0.030230, 0.030230] |
| pruning / v72_repeat | 2 | qa | 0.065225 | 0.341593 | 0.205388 | 0.374455 | 0.276367 [0.276367, 0.276367] |
| pruning / v72_repeat | 2 | macro | 0.066312 | 0.193700 | 0.147386 | 0.213695 | 0.127388 [0.127388, 0.127388] |
| quantization / development_state_boundary | 2 | math | 0.499839 | 0.506944 | 0.501599 | 0.508084 | 0.007105 [0.002377, 0.011833] |
| quantization / development_state_boundary | 2 | code | 0.556736 | 0.557925 | 0.581332 | 0.564971 | 0.001189 [-0.020270, 0.022648] |
| quantization / development_state_boundary | 2 | qa | 0.457940 | 0.453561 | 0.456972 | 0.454350 | -0.004379 [-0.017702, 0.008944] |
| quantization / development_state_boundary | 2 | macro | 0.504838 | 0.506143 | 0.513301 | 0.509135 | 0.001305 [-0.011865, 0.014475] |
| quantization / new_state_boundary | 1 | math | 0.129466 | 0.147291 | 0.160228 | 0.169291 | 0.017825 [CI unavailable] |
| quantization / new_state_boundary | 1 | code | 0.216078 | 0.188890 | 0.344580 | 0.324891 | -0.027188 [CI unavailable] |
| quantization / new_state_boundary | 1 | qa | 0.132765 | 0.167457 | 0.063708 | 0.126153 | 0.034692 [CI unavailable] |
| quantization / new_state_boundary | 1 | macro | 0.159436 | 0.167879 | 0.189505 | 0.206778 | 0.008443 [CI unavailable] |
| quantization / new_state_interior | 1 | math | 0.005603 | 0.018065 | 0.038863 | 0.036521 | 0.012462 [CI unavailable] |
| quantization / new_state_interior | 1 | code | 0.027385 | 0.011459 | 0.125265 | 0.108854 | -0.015926 [CI unavailable] |
| quantization / new_state_interior | 1 | qa | 0.158057 | 0.164982 | 0.233241 | 0.279708 | 0.006925 [CI unavailable] |
| quantization / new_state_interior | 1 | macro | 0.063682 | 0.064835 | 0.132456 | 0.141694 | 0.001154 [CI unavailable] |
| distillation / gemma3-1b | 6 | math | 0.056897 | 0.056897 | 0.056627 | 0.122600 | 0.000000 [0.000000, 0.000000] |
| distillation / gemma3-1b | 6 | code | 0.048525 | 0.048525 | 0.057902 | 0.109822 | 0.000000 [0.000000, 0.000000] |
| distillation / gemma3-1b | 6 | qa | 1.207714 | 1.207714 | 0.788724 | 1.103400 | 0.000000 [0.000000, 0.000000] |
| distillation / gemma3-1b | 6 | macro | 0.437712 | 0.437712 | 0.301084 | 0.445274 | 0.000000 [0.000000, 0.000000] |
| distillation / gemma3-270m | 6 | math | 0.074254 | 0.074254 | 0.074086 | 0.139957 | 0.000000 [0.000000, 0.000000] |
| distillation / gemma3-270m | 6 | code | 0.019252 | 0.019252 | 0.029398 | 0.078554 | 0.000000 [0.000000, 0.000000] |
| distillation / gemma3-270m | 6 | qa | 1.262953 | 1.262953 | 0.474530 | 1.135724 | 0.000000 [0.000000, 0.000000] |
| distillation / gemma3-270m | 6 | macro | 0.452153 | 0.452153 | 0.192672 | 0.451412 | 0.000000 [0.000000, 0.000000] |

## Sensitivity to development fitting convention

These are fixed alternative estimators, not selected using confirmation MAE. All reproduce the same frozen A predictions.

| Arm | Convention | Math B − A | Code B − A | QA B − A | Macro B − A [95% CI] |
|---|---|---:|---:|---:|---:|
| pruning | primary | 0.018977 | 0.009219 | 0.189877 | 0.072691 [0.029958, 0.115424] |
| pruning | mean_of_capability_medians | -0.010255 | -0.019088 | 0.193821 | 0.054826 [0.010268, 0.099384] |
| pruning | development_rows_ols | 0.108223 | 0.106310 | 0.324631 | 0.179721 [0.136564, 0.224077] |
| pruning | median_anchors_lad | 0.007457 | -0.001053 | 0.216816 | 0.074407 [0.030968, 0.117845] |
| pruning | development_rows_lad | 0.037321 | 0.022886 | 0.206732 | 0.088980 [0.057947, 0.120012] |
| quantization | primary | 0.010933 | -0.009364 | 0.008399 | 0.003323 [-0.011865, 0.014475] |
| quantization | mean_of_capability_medians | 0.003507 | -0.011616 | 0.011915 | 0.001269 [-0.008960, 0.005465] |
| quantization | development_rows_ols | 0.511580 | 0.481296 | 0.637256 | 0.543378 [-0.113465, 0.948073] |
| quantization | median_anchors_lad | 0.016953 | -0.004213 | 0.006913 | 0.006551 [-0.017294, 0.020952] |
| quantization | development_rows_lad | -0.004210 | -0.025180 | 0.012784 | -0.005535 [-0.026772, 0.027289] |

## Repeated V72 outcome vector

The two V72 state labels have exactly identical recorded signed response vectors at all three densities and capabilities, despite distinct weight hashes in freeze.json. Their source-free predictions and paired gains are therefore also identical. The cause is not established. The collapsed V72 intervals do not establish repeatability across distinct measured responses; a sensitivity counting this outcome vector once is reported.

Sensitivity removes pythia-2.8b@step16000; the other V72 state remains. This leaves 12 cells across 4 states. The primary analysis above retains both requested state labels.

| Capability | A MAE | B MAE | B − A [95% CI] |
|---|---:|---:|---:|
| math | 0.227924 | 0.232753 | 0.004829 [-0.049485, 0.059144] |
| code | 0.173590 | 0.177556 | 0.003967 [-0.018129, 0.022823] |
| qa | 0.181788 | 0.350042 | 0.168254 [0.105439, 0.243418] |
| macro | 0.194434 | 0.253450 | 0.059017 [0.020555, 0.102507] |

Quantization equal-state macro MAEs (instead of equal-cell): A=0.379065, B=0.381939, C=0.399030, D=0.401118.

## Distillation equivalence

a_c=(w.T y_c)/(w.T w); a=(w.T mean_c(y_c))/(w.T w); s_c=((a*w).T y_c)/((a*w).T(a*w))=a_c/a, for a != 0. Thus a*s_c*w=a_c*w; arbitrary signed scales are allowed.

Maximum absolute A/B confirmation prediction difference: 1.39e-16 nats. B has no additional expressive restriction here; a nonzero shared amplitude and three signed scales reparameterize the three capability amplitudes.

## Uncertainty and validation

- resamples: 5000
- seed: 0
- rng: NumPy PCG64
- interval: 95% paired percentile
- gain_sign: baseline MAE minus A MAE; positive favors per-capability relations
- unit: Pruning/quantization: whole source state. Distillation: whole sampled pool, retaining both students and every budget.
- pairing: Same draws across variants and capabilities within each panel; macro uses the same draws.
- weighting: Resample clusters, sum their absolute errors and divide by resampled row counts (preserves cell-weighted estimand).
- scope: Conditional on fixed development fits, probes and training seed. Does not include development-fit or probe-item uncertainty. No interval for one-state panels; 2-5-state intervals are descriptive and cannot establish population generalization. No multiple-comparison adjustment.
- The two V72 state labels have exactly identical recorded signed response vectors at all three densities and capabilities, despite distinct weight hashes in freeze.json. Their source-free predictions and paired gains are therefore also identical. The cause is not established. The collapsed V72 intervals do not establish repeatability across distinct measured responses; a sensitivity counting this outcome vector once is reported.
- Verified exact development membership, confirmation membership, disjointness at the relevant unit, raw signed responses, historical hashes and frozen prediction chains. Independently reconstructed every A prediction across all 216 capability rows.
- Checks use only NumPy and Python standard-library operations on CPU; no models are loaded.

## Outputs

Paper outputs are staged to respect the requested write-only scope:

- `paper/paper/tables/cap_conditioning.tex` under this results directory (uses `[H]`).
- `paper/analysis/v76_cap_conditioning.py` under this results directory (byte-identical code mirror).
- Reproduce: `python -B analysis/v76_cap_conditioning.py`; verify without writes: append `--check`.
