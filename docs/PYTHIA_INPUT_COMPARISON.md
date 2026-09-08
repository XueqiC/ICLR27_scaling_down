# Pythia four-input comparison (V36b)

The KEY comparison asks whether either training scale or dense measurement predicts compression loss increments well enough to beat a simple baseline, before asking whether combining them helps. Failure to gain from adding D0 alone cannot distinguish information redundancy from both predictors failing.

Endpoint: **ΔL_c = L_c(config) − L_c(dense)**, in **nats per native token**. This is signed capability loss damage, not an a_c label or a task-accuracy endpoint. The available clean panel is **410m + 1.4b × step16k/64k/143k**; **2.8b is excluded**. The 12 existing V6/V10 JSON files supply six cells per arm, 24 observations per arm/capability, and 144 total. Pruning (density 0.9/0.8/0.7/0.6, dense key `1.0`) and quantization (8/6/4/3 bits, dense key `dense`) use their own dense references and are never pooled. No model runs.

The [V36 analysis](PYTHIA_CONTROLLED.md) supplies the loader, architecture/token accounting, direct OLS machinery, whole-step folds and paired interval computation. V36b computes all counts from the clean panel, with no dependency on the earlier report's historical three-size prose.

| Step | D0 processed tokens |
|---|---:|
| 16000 | 33,554,432,000 |
| 64000 | 134,217,728,000 |
| 143000 | 299,892,736,000 |

N0 transformer-matrix counts: 410m: 301,989,888; 1.4b: 1,207,959,552.

**Same held-out set:** three leave-one-step-out folds, each with 16 training and 8 test observations per arm/capability. All configurations and both sizes at the held-out step travel together; every one of the 24 observations is scored once by every model. The middle step is interpolation and endpoints are extrapolation. There is no leave-one-size-out analysis on this two-size panel.

For known compression setting q, the input models are:

```text
simple:        0, or training-only mean/median Delta L at q
{N0,D0}:       alpha_q + beta_q log(N0/1e9) + delta_q log(D0/1e9)
{N0,L0}:       alpha_q + beta_q log(N0/1e9) + gamma_q L0
{N0,D0,L0}:    alpha_q + beta_q log(N0/1e9) + gamma_q L0 + delta_q log(D0/1e9)
```

All predictor models directly fit the raw signed configuration-level ΔL by the same unweighted V36 OLS solve. Continuous features are standardized on training rows only. No a_c-label regression, response transform, clipping, tuning or regularization is used. The compression setting is always known; this is prediction at the four measured settings. The standalone models have 12 coefficients and the combined model has 16. The combined fit is **saturated: four coefficients per config, four training cells**. Full rank does not ensure stable extrapolation; fold condition numbers are shown below. All negative, near-zero, int3 and large-damage observations remain in the primary scores.

**Reference and intervals:** strongest simple means the lowest overall OOF MAE among zero change, training-only config mean and training-only config median, selected once per arm/capability. The selection is an empirical comparison reference, not a deployable selector; there is no per-row or per-fold oracle. All candidates are disclosed below. Positive improvement = reference MAE − model MAE. Brackets on improvements are **paired-model 95% intervals** from V36's exact 27 resamples of the three whole held-out steps. Fits, OOF predictions and reference selection are held fixed; selection uncertainty is not covered. With three groups and overlapping training sets/shared trajectories, these are coarse descriptive, panel-conditional intervals, not seed/probe/population/causal inference. There is no multiplicity correction or equivalence test; an interval containing zero does not prove redundancy.

## KEY four-input table: pruning

| Capability | Inputs | MAE [95%] | Improvement over strongest simple [paired 95%] |
|---|---|---:|---:|
| math | Strongest simple (config_median) | 0.19545 [0.03692, 0.32248] | 0.00000 [0.00000, 0.00000] |
| math | {N0, D0} | 0.16765 [0.11236, 0.23884] | 0.02780 [-0.07544, 0.17074] |
| math | {N0, L0} | 0.16534 [0.11483, 0.19752] | 0.03012 [-0.07790, 0.13882] |
| math | {N0, D0, L0} | 0.17962 [0.10409, 0.23344] | 0.01583 [-0.06717, 0.12115] |
| code | Strongest simple (config_median) | 0.22203 [0.09559, 0.37481] | 0.00000 [0.00000, 0.00000] |
| code | {N0, D0} | 0.24098 [0.14699, 0.30582] | -0.01895 [-0.11012, 0.10466] |
| code | {N0, L0} | 0.23842 [0.17444, 0.30378] | -0.01639 [-0.07885, 0.07103] |
| code | {N0, D0, L0} | 0.21689 [0.09708, 0.30071] | 0.00514 [-0.10501, 0.12192] |
| qa | Strongest simple (config_median) | 0.16345 [0.07923, 0.31567] | 0.00000 [0.00000, 0.00000] |
| qa | {N0, D0} | 0.33179 [0.19190, 0.49534] | -0.16833 [-0.41611, 0.00754] |
| qa | {N0, L0} | 0.90475 [0.17295, 2.26890] | -0.74129 [-2.18967, 0.04328] |
| qa | {N0, D0, L0} | 0.78448 [0.21911, 1.85351] | -0.62102 [-1.77428, 0.03485] |

Zero-change MAE is the mean |ΔL|: it shows the response magnitude available to model. The reference row has zero improvement by definition; its gain over zero is shown here.

| Capability | Zero MAE | Config mean MAE | Config median MAE | Selected reference | Reference gain over zero [paired 95%] |
|---|---:|---:|---:|---|---:|
| math | 0.29955 | 0.21321 | 0.19545 | config_median | 0.10410 [-0.10736, 0.23824] |
| code | 0.32352 | 0.24732 | 0.22203 | config_median | 0.10148 [-0.08549, 0.19603] |
| qa | 0.22494 | 0.20589 | 0.16345 | config_median | 0.06148 [0.01157, 0.11517] |

| Capability | L0 model gain over D0 model [paired 95%] | Combined gain over D0 model [paired 95%] | Combined gain over L0 model [paired 95%] |
|---|---:|---:|---:|
| math | 0.00231 [-0.03192, 0.04132] | -0.01197 [-0.04959, 0.00827] | -0.01429 [-0.03592, 0.01073] |
| code | 0.00256 [-0.03363, 0.06878] | 0.02409 [0.00511, 0.04990] | 0.02153 [-0.06367, 0.07736] |
| qa | -0.57296 [-1.77356, 0.03573] | -0.45269 [-1.35817, 0.02731] | 0.12027 [-0.04615, 0.41539] |

**math:** Both standalone predictors beat the simple reference in point MAE; this provides a predictive signal against which to assess their overlap. Read the paired intervals alongside these point comparisons; a non-significant combined gain alone establishes neither redundancy nor absence of signal.

**code:** Both standalone predictors fail to beat the simple reference in point MAE; their comparison cannot demonstrate information redundancy. Read the paired intervals alongside these point comparisons; a non-significant combined gain alone establishes neither redundancy nor absence of signal.

**qa:** Both standalone predictors fail to beat the simple reference in point MAE; their comparison cannot demonstrate information redundancy. Read the paired intervals alongside these point comparisons; a non-significant combined gain alone establishes neither redundancy nor absence of signal.

## KEY four-input table: quantization

| Capability | Inputs | MAE [95%] | Improvement over strongest simple [paired 95%] |
|---|---|---:|---:|
| math | Strongest simple (config_mean) | 0.90812 [0.22367, 1.32426] | 0.00000 [0.00000, 0.00000] |
| math | {N0, D0} | 0.65348 [0.37000, 1.00585] | 0.25464 [-0.14633, 0.73968] |
| math | {N0, L0} | 0.78871 [0.46844, 1.22732] | 0.11941 [-0.24477, 0.65389] |
| math | {N0, D0, L0} | 0.62262 [0.38925, 0.78874] | 0.28551 [-0.16558, 0.53552] |
| code | Strongest simple (config_mean) | 0.99411 [0.31063, 1.35016] | 0.00000 [0.00000, 0.00000] |
| code | {N0, D0} | 0.37451 [0.22751, 0.56711] | 0.61961 [0.08312, 0.99266] |
| code | {N0, L0} | 0.61390 [0.40428, 0.89691] | 0.38022 [-0.09365, 0.78105] |
| code | {N0, D0, L0} | 1.35934 [0.35735, 3.35359] | -0.36523 [-2.03205, 0.99281] |
| qa | Strongest simple (config_mean) | 0.84993 [0.37508, 1.10438] | 0.00000 [0.00000, 0.00000] |
| qa | {N0, D0} | 0.43557 [0.33893, 0.57593] | 0.41436 [-0.01675, 0.76545] |
| qa | {N0, L0} | 2.12044 [0.21989, 5.14867] | -1.27051 [-4.07835, 0.15519] |
| qa | {N0, D0, L0} | 1.48716 [0.42306, 3.55695] | -0.63724 [-2.48663, 0.62290] |

Zero-change MAE is the mean |ΔL|: it shows the response magnitude available to model. The reference row has zero improvement by definition; its gain over zero is shown here.

| Capability | Zero MAE | Config mean MAE | Config median MAE | Selected reference | Reference gain over zero [paired 95%] |
|---|---:|---:|---:|---|---:|
| math | 1.04287 | 0.90812 | 0.94382 | config_mean | 0.13474 [-0.91794, 0.72091] |
| code | 1.20640 | 0.99411 | 1.02752 | config_mean | 0.21229 [-1.04223, 0.91423] |
| qa | 0.89433 | 0.84993 | 0.94759 | config_mean | 0.04440 [-0.90361, 0.52926] |

| Capability | L0 model gain over D0 model [paired 95%] | Combined gain over D0 model [paired 95%] | Combined gain over L0 model [paired 95%] |
|---|---:|---:|---:|
| math | -0.13523 [-0.22147, -0.08578] | 0.03086 [-0.20415, 0.31599] | 0.16609 [-0.11837, 0.53746] |
| code | -0.23939 [-0.32980, -0.17677] | -0.98483 [-3.02470, 0.20976] | -0.74544 [-2.81310, 0.53956] |
| qa | -1.68487 [-4.57274, 0.17194] | -1.05160 [-2.98102, -0.03123] | 0.63327 [-0.20317, 1.59172] |

**math:** Both standalone predictors beat the simple reference in point MAE; this provides a predictive signal against which to assess their overlap. Read the paired intervals alongside these point comparisons; a non-significant combined gain alone establishes neither redundancy nor absence of signal.

**code:** Both standalone predictors beat the simple reference in point MAE; this provides a predictive signal against which to assess their overlap. Read the paired intervals alongside these point comparisons; a non-significant combined gain alone establishes neither redundancy nor absence of signal.

**qa:** Only {N0, D0} beats the simple reference in point MAE; the inputs are not interchangeable in this fitted model class. Read the paired intervals alongside these point comparisons; a non-significant combined gain alone establishes neither redundancy nor absence of signal.

## Fold stability

All values are native-token MAE. The simple reference is fixed across the three folds. These are the same held-out predictions as the KEY table.

| Arm | Capability | Held-out step | Simple | N0,D0 | N0,L0 | Combined | Combined condition number |
|---|---|---:|---:|---:|---:|---:|---:|
| pruning | math | 16000 | 0.22695 | 0.23884 | 0.19752 | 0.23344 | 24.00 |
| pruning | math | 64000 | 0.03692 | 0.11236 | 0.11483 | 0.10409 | 13.45 |
| pruning | math | 143000 | 0.32248 | 0.15174 | 0.18367 | 0.20134 | 19.38 |
| pruning | code | 16000 | 0.19570 | 0.30582 | 0.23704 | 0.30071 | 11.82 |
| pruning | code | 64000 | 0.09559 | 0.14699 | 0.17444 | 0.09708 | 16.97 |
| pruning | code | 143000 | 0.37481 | 0.27015 | 0.30378 | 0.25289 | 109.45 |
| pruning | qa | 16000 | 0.07923 | 0.49534 | 2.26890 | 1.85351 | 12.98 |
| pruning | qa | 64000 | 0.09547 | 0.19190 | 0.17295 | 0.21911 | 1.54 |
| pruning | qa | 143000 | 0.31567 | 0.30812 | 0.27239 | 0.28081 | 1.80 |
| quantization | math | 16000 | 1.17644 | 1.00585 | 1.22732 | 0.68986 | 24.00 |
| quantization | math | 64000 | 0.22367 | 0.37000 | 0.46844 | 0.38925 | 13.45 |
| quantization | math | 143000 | 1.32426 | 0.58458 | 0.67037 | 0.78874 | 19.38 |
| quantization | code | 16000 | 1.35016 | 0.56711 | 0.89691 | 0.35735 | 11.82 |
| quantization | code | 64000 | 0.31063 | 0.22751 | 0.40428 | 0.36707 | 16.97 |
| quantization | code | 143000 | 1.32155 | 0.32889 | 0.54050 | 3.35359 | 109.45 |
| quantization | qa | 16000 | 1.07032 | 0.57593 | 5.14867 | 3.55695 | 12.98 |
| quantization | qa | 64000 | 0.37508 | 0.39183 | 0.21989 | 0.42306 | 1.54 |
| quantization | qa | 143000 | 1.10438 | 0.33893 | 0.99275 | 0.48148 | 1.80 |

## Training trajectory correlation

Pearson correlations use unique dense cells (not four repetitions per compression setting). Both raw D0 and the fitted log(D0) scale are shown. Within-size correlations describe the training trajectory; the six-cell correlation also includes size variation. Three checkpoints per size make these descriptive. Correlated inputs and unstable saturated fits preclude a causal coefficient interpretation.

| Arm | Capability | Cells | n | corr(D0,L0) | corr(log D0,L0) |
|---|---|---|---:|---:|---:|
| pruning | math | all sizes | 6 | -0.61288 | -0.65449 |
| pruning | math | 410m | 3 | -0.91862 | -0.99352 |
| pruning | math | 1.4b | 3 | -0.94328 | -0.99894 |
| pruning | code | all sizes | 6 | -0.72559 | -0.78684 |
| pruning | code | 410m | 3 | -0.86669 | -0.97368 |
| pruning | code | 1.4b | 3 | -0.94629 | -0.99932 |
| pruning | qa | all sizes | 6 | 0.31106 | 0.38634 |
| pruning | qa | 410m | 3 | -0.93971 | -0.99839 |
| pruning | qa | 1.4b | 3 | 0.80875 | 0.94393 |
| quantization | math | all sizes | 6 | -0.61288 | -0.65449 |
| quantization | math | 410m | 3 | -0.91862 | -0.99352 |
| quantization | math | 1.4b | 3 | -0.94328 | -0.99894 |
| quantization | code | all sizes | 6 | -0.72559 | -0.78684 |
| quantization | code | 410m | 3 | -0.86669 | -0.97368 |
| quantization | code | 1.4b | 3 | -0.94629 | -0.99932 |
| quantization | qa | all sizes | 6 | 0.31106 | 0.38634 |
| quantization | qa | 410m | 3 | -0.93971 | -0.99839 |
| quantization | qa | 1.4b | 3 | 0.80875 | 0.94393 |

## The three loss curves

**L_c,0(D0)** is dense capability loss; **ΔL_m,c(D0)** is its compression increment; **L_m,c(D0) = L_c,0(D0) + ΔL_m,c(D0)** is absolute compressed loss. Absolute here means the actual compressed loss, not |ΔL|. Lower loss is better. A larger compression penalty with more training need not erase the improvement in dense loss. Each figure has one row per size and these three columns; no averaging across sizes/configs/arms. Lines join measured checkpoints, with linear token and native-token-loss axes. Every table sequence follows **step16000 → step64000 → step143000**. Endpoint changes describe late versus early only; the trajectory shapes use both adjacent transitions with V36's 1e-12 numerical tolerance.

### pruning: math

![pruning math: dense, increment and absolute compressed loss](../../results/v36b-input-comparison/curves_pruning_math.png)

[Full precision curve table (CSV)](../../results/v36b-input-comparison/curves_pruning_math.csv). The increment rises while absolute compressed loss falls from early to late in **4/8** fixed size/config trajectories.

| Size | Dense L0 at the three steps | Dense late−early | Dense shape |
|---|---|---:|---|
| 410m | 1.61204 → 1.49905 → 1.45855 | -0.15348 | decreasing |
| 1.4b | 1.46994 → 1.31300 → 1.23713 | -0.23282 | decreasing |

| Size | Config | ΔL at the three steps | Absolute compressed L at the three steps | ΔL late−early | L late−early | ΔL / L shape |
|---|---:|---|---|---:|---:|---|
| 410m | 0.9 | 0.00225 → 0.00825 → 0.03411 | 1.61429 → 1.50730 → 1.49266 | +0.03186 | -0.12163 | increasing / decreasing |
| 410m | 0.8 | 0.02064 → 0.04742 → 0.19665 | 1.63268 → 1.54647 → 1.65520 | +0.17601 | +0.02252 | increasing / nonmonotonic |
| 410m | 0.7 | 0.10061 → 0.19637 → 0.61281 | 1.71265 → 1.69542 → 2.07136 | +0.51220 | +0.35872 | increasing / nonmonotonic |
| 410m | 0.6 | 0.37365 → 0.71237 → 1.66246 | 1.98568 → 2.21142 → 3.12102 | +1.28882 | +1.13533 | increasing / increasing |
| 1.4b | 0.9 | 0.00105 → 0.00708 → 0.01010 | 1.47099 → 1.32008 → 1.24723 | +0.00906 | -0.22376 | increasing / decreasing |
| 1.4b | 0.8 | 0.01505 → 0.03838 → 0.07063 | 1.48499 → 1.35138 → 1.30776 | +0.05558 | -0.17723 | increasing / decreasing |
| 1.4b | 0.7 | 0.07811 → 0.19738 → 0.29481 | 1.54805 → 1.51038 → 1.53193 | +0.21670 | -0.01612 | increasing / nonmonotonic |
| 1.4b | 0.6 | 0.36538 → 0.99409 → 1.14965 | 1.83532 → 2.30709 → 2.38678 | +0.78427 | +0.55145 | increasing / increasing |

For example, 410m at config 0.9: the increment rises by 0.03186, while dense loss changes by -0.15348 and absolute compressed loss changes by -0.12163 nats/native token. This is increased fragility alongside improved absolute compressed loss.

### pruning: code

![pruning code: dense, increment and absolute compressed loss](../../results/v36b-input-comparison/curves_pruning_code.png)

[Full precision curve table (CSV)](../../results/v36b-input-comparison/curves_pruning_code.csv). The increment rises while absolute compressed loss falls from early to late in **4/8** fixed size/config trajectories.

| Size | Dense L0 at the three steps | Dense late−early | Dense shape |
|---|---|---:|---|
| 410m | 1.54582 → 1.43594 → 1.41627 | -0.12955 | decreasing |
| 1.4b | 1.46524 → 1.35025 → 1.29255 | -0.17269 | decreasing |

| Size | Config | ΔL at the three steps | Absolute compressed L at the three steps | ΔL late−early | L late−early | ΔL / L shape |
|---|---:|---|---|---:|---:|---|
| 410m | 0.9 | 0.00184 → 0.00618 → 0.00957 | 1.54766 → 1.44212 → 1.42584 | +0.00773 | -0.12182 | increasing / decreasing |
| 410m | 0.8 | 0.00499 → 0.06216 → 0.21613 | 1.55081 → 1.49810 → 1.63240 | +0.21114 | +0.08159 | increasing / nonmonotonic |
| 410m | 0.7 | 0.11320 → 0.16817 → 0.80604 | 1.65902 → 1.60411 → 2.22231 | +0.69283 | +0.56329 | increasing / nonmonotonic |
| 410m | 0.6 | 0.34550 → 0.99287 → 2.30550 | 1.89131 → 2.42881 → 3.72177 | +1.96001 | +1.83046 | increasing / increasing |
| 1.4b | 0.9 | 0.00006 → 0.00398 → 0.02110 | 1.46530 → 1.35423 → 1.31364 | +0.02104 | -0.15165 | increasing / decreasing |
| 1.4b | 0.8 | 0.01254 → 0.01795 → 0.05152 | 1.47778 → 1.36820 → 1.34407 | +0.03898 | -0.13371 | increasing / decreasing |
| 1.4b | 0.7 | 0.04956 → 0.08533 → 0.17406 | 1.51480 → 1.43558 → 1.46660 | +0.12449 | -0.04819 | increasing / nonmonotonic |
| 1.4b | 0.6 | 0.35399 → 0.97938 → 0.98277 | 1.81923 → 2.32963 → 2.27531 | +0.62877 | +0.45609 | increasing / nonmonotonic |

For example, 410m at config 0.9: the increment rises by 0.00773, while dense loss changes by -0.12955 and absolute compressed loss changes by -0.12182 nats/native token. This is increased fragility alongside improved absolute compressed loss.

### pruning: qa

![pruning qa: dense, increment and absolute compressed loss](../../results/v36b-input-comparison/curves_pruning_qa.png)

[Full precision curve table (CSV)](../../results/v36b-input-comparison/curves_pruning_qa.csv). The increment rises while absolute compressed loss falls from early to late in **0/8** fixed size/config trajectories.

| Size | Dense L0 at the three steps | Dense late−early | Dense shape |
|---|---|---:|---|
| 410m | 5.07539 → 5.01889 → 4.99275 | -0.08264 | decreasing |
| 1.4b | 4.91249 → 5.11214 → 5.12034 | +0.20785 | increasing |

| Size | Config | ΔL at the three steps | Absolute compressed L at the three steps | ΔL late−early | L late−early | ΔL / L shape |
|---|---:|---|---|---:|---:|---|
| 410m | 0.9 | -0.01646 → -0.01444 → -0.11538 | 5.05894 → 5.00446 → 4.87738 | -0.09892 | -0.18156 | nonmonotonic / decreasing |
| 410m | 0.8 | -0.03274 → -0.11811 → -0.16683 | 5.04266 → 4.90078 → 4.82593 | -0.13409 | -0.21673 | decreasing / decreasing |
| 410m | 0.7 | -0.18102 → -0.29004 → 0.12773 | 4.89437 → 4.72885 → 5.12048 | +0.30876 | +0.22612 | nonmonotonic / nonmonotonic |
| 410m | 0.6 | -0.39134 → -0.24774 → 1.25677 | 4.68405 → 4.77115 → 6.24952 | +1.64811 | +1.56547 | increasing / increasing |
| 1.4b | 0.9 | 0.00986 → -0.02665 → 0.02822 | 4.92235 → 5.08549 → 5.14856 | +0.01836 | +0.22621 | nonmonotonic / increasing |
| 1.4b | 0.8 | -0.00250 → -0.05980 → -0.06381 | 4.90999 → 5.05234 → 5.05653 | -0.06131 | +0.14654 | decreasing / increasing |
| 1.4b | 0.7 | -0.14169 → -0.42666 → -0.41394 | 4.77079 → 4.68548 → 4.70639 | -0.27225 | -0.06440 | nonmonotonic / nonmonotonic |
| 1.4b | 0.6 | -0.31993 → -0.50163 → -0.44519 | 4.59256 → 4.61050 → 4.67514 | -0.12527 | +0.08258 | nonmonotonic / increasing |

### quantization: math

![quantization math: dense, increment and absolute compressed loss](../../results/v36b-input-comparison/curves_quantization_math.png)

[Full precision curve table (CSV)](../../results/v36b-input-comparison/curves_quantization_math.csv). The increment rises while absolute compressed loss falls from early to late in **4/8** fixed size/config trajectories.

| Size | Dense L0 at the three steps | Dense late−early | Dense shape |
|---|---|---:|---|
| 410m | 1.61204 → 1.49905 → 1.45855 | -0.15348 | decreasing |
| 1.4b | 1.46994 → 1.31300 → 1.23713 | -0.23282 | decreasing |

| Size | Config | ΔL at the three steps | Absolute compressed L at the three steps | ΔL late−early | L late−early | ΔL / L shape |
|---|---:|---|---|---:|---:|---|
| 410m | 8 | 0.00063 → 0.00119 → 0.00275 | 1.61267 → 1.50024 → 1.46130 | +0.00212 | -0.15137 | increasing / decreasing |
| 410m | 6 | 0.00431 → 0.00799 → 0.02343 | 1.61635 → 1.50704 → 1.48199 | +0.01912 | -0.13436 | increasing / decreasing |
| 410m | 4 | 0.07884 → 0.11957 → 0.55946 | 1.69088 → 1.61862 → 2.01801 | +0.48062 | +0.32714 | increasing / nonmonotonic |
| 410m | 3 | 1.07684 → 2.88361 → 7.01246 | 2.68888 → 4.38266 → 8.47101 | +5.93562 | +5.78213 | increasing / increasing |
| 1.4b | 8 | -0.00018 → 0.00146 → -0.00127 | 1.46977 → 1.31446 → 1.23586 | -0.00109 | -0.23390 | nonmonotonic / decreasing |
| 1.4b | 6 | 0.00075 → 0.00686 → 0.00473 | 1.47070 → 1.31986 → 1.24185 | +0.00397 | -0.22884 | nonmonotonic / decreasing |
| 1.4b | 4 | 0.07544 → 0.11843 → 0.26627 | 1.54538 → 1.43142 → 1.50340 | +0.19084 | -0.04198 | increasing / nonmonotonic |
| 1.4b | 3 | 0.83101 → 4.41752 → 7.53381 | 2.30096 → 5.73052 → 8.77094 | +6.70280 | +6.46998 | increasing / increasing |

For example, 410m at config 8: the increment rises by 0.00212, while dense loss changes by -0.15348 and absolute compressed loss changes by -0.15137 nats/native token. This is increased fragility alongside improved absolute compressed loss.

### quantization: code

![quantization code: dense, increment and absolute compressed loss](../../results/v36b-input-comparison/curves_quantization_code.png)

[Full precision curve table (CSV)](../../results/v36b-input-comparison/curves_quantization_code.csv). The increment rises while absolute compressed loss falls from early to late in **4/8** fixed size/config trajectories.

| Size | Dense L0 at the three steps | Dense late−early | Dense shape |
|---|---|---:|---|
| 410m | 1.54582 → 1.43594 → 1.41627 | -0.12955 | decreasing |
| 1.4b | 1.46524 → 1.35025 → 1.29255 | -0.17269 | decreasing |

| Size | Config | ΔL at the three steps | Absolute compressed L at the three steps | ΔL late−early | L late−early | ΔL / L shape |
|---|---:|---|---|---:|---:|---|
| 410m | 8 | 0.00053 → -0.00101 → 0.00808 | 1.54635 → 1.43493 → 1.42435 | +0.00755 | -0.12200 | nonmonotonic / decreasing |
| 410m | 6 | 0.00256 → 0.00053 → 0.00939 | 1.54837 → 1.43647 → 1.42566 | +0.00683 | -0.12271 | nonmonotonic / decreasing |
| 410m | 4 | 0.04914 → 0.12081 → 0.67471 | 1.59496 → 1.55675 → 2.09098 | +0.62556 | +0.49602 | increasing / nonmonotonic |
| 410m | 3 | 1.28019 → 3.72296 → 6.94788 | 2.82600 → 5.15890 → 8.36415 | +5.66770 | +5.53815 | increasing / increasing |
| 1.4b | 8 | -0.00166 → -0.00059 → 0.00256 | 1.46357 → 1.34966 → 1.29510 | +0.00422 | -0.16847 | increasing / decreasing |
| 1.4b | 6 | -0.00897 → 0.00178 → 0.00909 | 1.45626 → 1.35203 → 1.30164 | +0.01807 | -0.15462 | increasing / decreasing |
| 1.4b | 4 | 0.05473 → 0.08700 → 0.30455 | 1.51997 → 1.43725 → 1.59710 | +0.24982 | +0.07713 | increasing / nonmonotonic |
| 1.4b | 3 | 1.06566 → 5.86415 → 8.73502 | 2.53090 → 7.21440 → 10.02757 | +7.66936 | +7.49667 | increasing / increasing |

For example, 410m at config 8: the increment rises by 0.00755, while dense loss changes by -0.12955 and absolute compressed loss changes by -0.12200 nats/native token. This is increased fragility alongside improved absolute compressed loss.

### quantization: qa

![quantization qa: dense, increment and absolute compressed loss](../../results/v36b-input-comparison/curves_quantization_qa.png)

[Full precision curve table (CSV)](../../results/v36b-input-comparison/curves_quantization_qa.csv). The increment rises while absolute compressed loss falls from early to late in **1/8** fixed size/config trajectories.

| Size | Dense L0 at the three steps | Dense late−early | Dense shape |
|---|---|---:|---|
| 410m | 5.07539 → 5.01889 → 4.99275 | -0.08264 | decreasing |
| 1.4b | 4.91249 → 5.11214 → 5.12034 | +0.20785 | increasing |

| Size | Config | ΔL at the three steps | Absolute compressed L at the three steps | ΔL late−early | L late−early | ΔL / L shape |
|---|---:|---|---|---:|---:|---|
| 410m | 8 | -0.00796 → 0.01046 → 0.01248 | 5.06743 → 5.02935 → 5.00523 | +0.02044 | -0.06220 | increasing / decreasing |
| 410m | 6 | 0.01972 → -0.00974 → -0.00291 | 5.09512 → 5.00915 → 4.98984 | -0.02264 | -0.10528 | nonmonotonic / decreasing |
| 410m | 4 | -0.06197 → -0.13563 → 0.53024 | 5.01343 → 4.88326 → 5.52299 | +0.59221 | +0.50957 | nonmonotonic / nonmonotonic |
| 410m | 3 | 0.60355 → 2.20901 → 6.14175 | 5.67894 → 7.22790 → 11.13451 | +5.53820 | +5.45556 | increasing / increasing |
| 1.4b | 8 | -0.00588 → -0.02142 → 0.03654 | 4.90661 → 5.09072 → 5.15687 | +0.04242 | +0.25027 | nonmonotonic / increasing |
| 1.4b | 6 | -0.01497 → -0.01830 → 0.00662 | 4.89752 → 5.09384 → 5.12696 | +0.02160 | +0.22944 | nonmonotonic / increasing |
| 1.4b | 4 | 0.03000 → -0.07512 → -0.13881 | 4.94249 → 5.03701 → 4.98152 | -0.16882 | +0.03903 | decreasing / nonmonotonic |
| 1.4b | 3 | 0.58965 → 4.75502 → 6.02605 | 5.50214 → 9.86716 → 11.14639 | +5.43640 | +5.64425 | increasing / increasing |

For example, 410m at config 8: the increment rises by 0.02044, while dense loss changes by -0.08264 and absolute compressed loss changes by -0.06220 nats/native token. This is increased fragility alongside improved absolute compressed loss.

## Reproduction and audit trail

```bash
SDL_V36_SIZES=410m,1.4b python3 analysis/v36b_input_comparison.py --dry-run
SDL_V36_SIZES=410m,1.4b python3 analysis/v36b_input_comparison.py
python3 -m pytest -q tests/test_v36b_inputs.py tests/test_v36.py
```

With V36's default size environment, V36b also excludes 2.8b automatically and requires both clean sizes. Explicit loader subsets avoid changing V36's global SIZES. Analysis uses NumPy and the standard library; PNG rendering uses matplotlib's Agg CPU backend. The dry run prints the report without writing or importing matplotlib. `results/v36b-input-comparison/summary.json` retains source SHA256 hashes, architecture metadata, all train/test memberships, coefficients, training standardizers, ranks/condition numbers, every held-out prediction, baseline selection, paired metrics (including per-config slices), correlations and all raw curves. Inputs are rehashed before writing. This report, six full-precision CSV curve tables and six PNG figures are generated from that summary. The existing aggregate losses cannot establish independent probe/seed uncertainty. These are fixed-recipe-series predictive comparisons, not isolated causal effects of training tokens.
