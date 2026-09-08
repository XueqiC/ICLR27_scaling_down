# Pythia four-input comparison (V36b)

The KEY comparison asks whether either training scale or dense measurement predicts compression loss increments well enough to beat a simple baseline, before asking whether combining them helps. Failure to gain from adding D0 alone cannot distinguish information redundancy from both predictors failing.

Endpoint: **ΔL_c = L_c(config) − L_c(dense)**, in **nats per native token**. This is signed capability loss damage, not an a_c label or a task-accuracy endpoint. The available clean panel is **410m + 1.4b × step16k/64k/143k**; **2.8b is excluded**. The 12 existing V6/V10 JSON files supply six cells per arm, 24 observations per arm/capability, and 144 total. Pruning (density 0.9/0.8/0.7/0.6, dense key `1.0`) and quantization (8/6/4/3 bits, dense key `dense`) use their own dense references and are never pooled. No model runs.

The [V36 analysis](PYTHIA_CONTROLLED.md) supplies the loader, architecture/token accounting, direct OLS machinery, whole-step folds and paired interval computation. V36b computes all counts from the clean panel, with no dependency on the earlier report's historical three-size prose.

| Step | D0 processed tokens |
|---|---:|
| 16000 | 33,554,432,000 |
| 64000 | 134,217,728,000 |
| 143000 | 299,892,736,000 |

N0 transformer-matrix counts: 160m: 84,934,656; 410m: 301,989,888; 1.4b: 1,207,959,552.

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
| math | Strongest simple (config_median) | 0.41309 [0.09629, 0.88515] | 0.00000 [0.00000, 0.00000] |
| math | {N0, D0} | 0.79981 [0.49476, 1.20553] | -0.38671 [-0.94769, 0.18602] |
| math | {N0, L0} | 0.61277 [0.10373, 0.93197] | -0.19968 [-0.54478, -0.00744] |
| math | {N0, D0, L0} | 0.25307 [0.14294, 0.41064] | 0.16003 [-0.04665, 0.47452] |
| code | Strongest simple (config_median) | 0.51586 [0.15502, 1.08374] | 0.00000 [0.00000, 0.00000] |
| code | {N0, D0} | 0.94717 [0.56645, 1.40522] | -0.43131 [-1.09639, 0.21389] |
| code | {N0, L0} | 0.69287 [0.09902, 1.11728] | -0.17701 [-0.55349, 0.05600] |
| code | {N0, D0, L0} | 0.29159 [0.10973, 0.48824] | 0.22428 [0.03204, 0.59551] |
| qa | Strongest simple (zero) | 0.49968 [0.13808, 1.05897] | 0.00000 [0.00000, 0.00000] |
| qa | {N0, D0} | 0.94412 [0.52432, 1.41240] | -0.44445 [-1.27432, 0.16331] |
| qa | {N0, L0} | 0.77419 [0.39469, 0.99373] | -0.27451 [-0.79606, 0.06524] |
| qa | {N0, D0, L0} | 0.80567 [0.43360, 1.03952] | -0.30600 [-0.90144, 0.11507] |

Zero-change MAE is the mean |ΔL|: it shows the response magnitude available to model. The reference row has zero improvement by definition; its gain over zero is shown here.

| Capability | Zero MAE | Config mean MAE | Config median MAE | Selected reference | Reference gain over zero [paired 95%] |
|---|---:|---:|---:|---|---:|
| math | 0.55789 | 0.58679 | 0.41309 | config_median | 0.14479 [-0.07717, 0.25952] |
| code | 0.66769 | 0.71155 | 0.51586 | config_median | 0.15182 [-0.11028, 0.29257] |
| qa | 0.49968 | 0.67771 | 0.52651 | zero | 0.00000 [0.00000, 0.00000] |

| Capability | L0 model gain over D0 model [paired 95%] | Combined gain over D0 model [paired 95%] | Combined gain over L0 model [paired 95%] |
|---|---:|---:|---:|
| math | 0.18703 [-0.23284, 0.40291] | 0.54674 [0.28849, 0.99990] | 0.35970 [-0.03921, 0.59699] |
| code | 0.25431 [-0.24742, 0.54291] | 0.65559 [0.38162, 1.12843] | 0.40128 [-0.01072, 0.62904] |
| qa | 0.16994 [-0.09807, 0.47826] | 0.13845 [-0.04824, 0.37288] | -0.03149 [-0.10538, 0.04983] |

**math:** Both standalone predictors fail to beat the simple reference in point MAE; their comparison cannot demonstrate information redundancy. Read the paired intervals alongside these point comparisons; a non-significant combined gain alone establishes neither redundancy nor absence of signal.

**code:** Both standalone predictors fail to beat the simple reference in point MAE; their comparison cannot demonstrate information redundancy. Read the paired intervals alongside these point comparisons; a non-significant combined gain alone establishes neither redundancy nor absence of signal.

**qa:** Both standalone predictors fail to beat the simple reference in point MAE; their comparison cannot demonstrate information redundancy. Read the paired intervals alongside these point comparisons; a non-significant combined gain alone establishes neither redundancy nor absence of signal.

## KEY four-input table: quantization

| Capability | Inputs | MAE [95%] | Improvement over strongest simple [paired 95%] |
|---|---|---:|---:|
| math | Strongest simple (config_median) | 1.41002 [0.17899, 2.87191] | 0.00000 [0.00000, 0.00000] |
| math | {N0, D0} | 2.36337 [1.35328, 3.63021] | -0.95335 [-2.45104, 0.76528] |
| math | {N0, L0} | 1.98931 [0.33542, 3.06177] | -0.57929 [-1.39158, -0.15643] |
| math | {N0, D0, L0} | 1.14448 [0.46978, 2.06359] | 0.26554 [-0.29079, 0.80832] |
| code | Strongest simple (config_median) | 1.55326 [0.33163, 3.02429] | 0.00000 [0.00000, 0.00000] |
| code | {N0, D0} | 2.19509 [1.37136, 3.25745] | -0.64183 [-1.95359, 1.06782] |
| code | {N0, L0} | 2.07693 [0.37423, 3.24288] | -0.52368 [-1.30983, -0.04260] |
| code | {N0, D0, L0} | 0.62600 [0.22094, 1.14772] | 0.92726 [0.11069, 1.87657] |
| qa | Strongest simple (config_median) | 1.47288 [0.27836, 2.96807] | 0.00000 [0.00000, 0.00000] |
| qa | {N0, D0} | 2.45683 [1.59639, 3.65270] | -0.98395 [-2.48049, 0.84666] |
| qa | {N0, L0} | 1.96411 [0.92058, 2.62249] | -0.49123 [-1.17706, 0.34558] |
| qa | {N0, D0, L0} | 1.98938 [1.12520, 2.70581] | -0.51650 [-1.53360, 0.83093] |

Zero-change MAE is the mean |ΔL|: it shows the response magnitude available to model. The reference row has zero improvement by definition; its gain over zero is shown here.

| Capability | Zero MAE | Config mean MAE | Config median MAE | Selected reference | Reference gain over zero [paired 95%] |
|---|---:|---:|---:|---|---:|
| math | 1.58035 | 1.87723 | 1.41002 | config_median | 0.17032 [-0.86631, 0.79161] |
| code | 1.84075 | 1.94157 | 1.55326 | config_median | 0.28749 [-0.90664, 1.01185] |
| qa | 1.52474 | 1.89536 | 1.47288 | config_median | 0.05187 [-0.94195, 0.66101] |

| Capability | L0 model gain over D0 model [paired 95%] | Combined gain over D0 model [paired 95%] | Combined gain over L0 model [paired 95%] |
|---|---:|---:|---:|
| math | 0.37406 [-0.95514, 1.05945] | 1.21889 [0.04304, 2.73014] | 0.84483 [-0.13436, 1.67069] |
| code | 0.11816 [-1.28641, 0.99713] | 1.56909 [0.80876, 2.74811] | 1.45094 [0.15329, 2.10435] |
| qa | 0.49272 [-0.50109, 1.30343] | 0.46745 [-0.01573, 0.94689] | -0.02527 [-0.35654, 0.48535] |

**math:** Both standalone predictors fail to beat the simple reference in point MAE; their comparison cannot demonstrate information redundancy. Read the paired intervals alongside these point comparisons; a non-significant combined gain alone establishes neither redundancy nor absence of signal.

**code:** Both standalone predictors fail to beat the simple reference in point MAE; their comparison cannot demonstrate information redundancy. Read the paired intervals alongside these point comparisons; a non-significant combined gain alone establishes neither redundancy nor absence of signal.

**qa:** Both standalone predictors fail to beat the simple reference in point MAE; their comparison cannot demonstrate information redundancy. Read the paired intervals alongside these point comparisons; a non-significant combined gain alone establishes neither redundancy nor absence of signal.

## Fold stability

All values are native-token MAE. The simple reference is fixed across the three folds. These are the same held-out predictions as the KEY table.

| Arm | Capability | Held-out step | Simple | N0,D0 | N0,L0 | Combined | Combined condition number |
|---|---|---:|---:|---:|---:|---:|---:|
| pruning | math | 16000 | 0.25784 | 1.20553 | 0.80262 | 0.20563 | 4.32 |
| pruning | math | 64000 | 0.09629 | 0.49476 | 0.10373 | 0.14294 | 3.68 |
| pruning | math | 143000 | 0.88515 | 0.69913 | 0.93197 | 0.41064 | 14.21 |
| pruning | code | 16000 | 0.30883 | 1.40522 | 0.86231 | 0.27679 | 3.53 |
| pruning | code | 64000 | 0.15502 | 0.56645 | 0.09902 | 0.10973 | 3.26 |
| pruning | code | 143000 | 1.08374 | 0.86985 | 1.11728 | 0.48824 | 7.40 |
| pruning | qa | 16000 | 0.13808 | 1.41240 | 0.93414 | 1.03952 | 1.79 |
| pruning | qa | 64000 | 0.30199 | 0.52432 | 0.39469 | 0.43360 | 2.21 |
| pruning | qa | 143000 | 1.05897 | 0.89566 | 0.99373 | 0.94390 | 2.48 |
| quantization | math | 16000 | 1.17917 | 3.63021 | 2.57075 | 0.90007 | 4.33 |
| quantization | math | 64000 | 0.17899 | 1.35328 | 0.33542 | 0.46978 | 3.69 |
| quantization | math | 143000 | 2.87191 | 2.10663 | 3.06177 | 2.06359 | 14.19 |
| quantization | code | 16000 | 1.30386 | 3.25745 | 2.61369 | 0.50934 | 3.56 |
| quantization | code | 64000 | 0.33163 | 1.37136 | 0.37423 | 0.22094 | 3.29 |
| quantization | code | 143000 | 3.02429 | 1.95647 | 3.24288 | 1.14772 | 7.40 |
| quantization | qa | 16000 | 1.17221 | 3.65270 | 2.34927 | 2.70581 | 1.78 |
| quantization | qa | 64000 | 0.27836 | 1.59639 | 0.92058 | 1.12520 | 2.21 |
| quantization | qa | 143000 | 2.96807 | 2.12140 | 2.62249 | 2.13714 | 2.48 |

## Training trajectory correlation

Pearson correlations use unique dense cells (not four repetitions per compression setting). Both raw D0 and the fitted log(D0) scale are shown. Within-size correlations describe the training trajectory; the six-cell correlation also includes size variation. Three checkpoints per size make these descriptive. Correlated inputs and unstable saturated fits preclude a causal coefficient interpretation.

| Arm | Capability | Cells | n | corr(D0,L0) | corr(log D0,L0) |
|---|---|---|---:|---:|---:|
| pruning | math | all sizes | 9 | 0.01723 | -0.03938 |
| pruning | math | 160m | 3 | 0.86091 | 0.67777 |
| pruning | math | 410m | 3 | -0.91862 | -0.99352 |
| pruning | math | 1.4b | 3 | -0.94450 | -0.99910 |
| pruning | code | all sizes | 9 | 0.07177 | 0.00762 |
| pruning | code | 160m | 3 | 0.85416 | 0.66806 |
| pruning | code | 410m | 3 | -0.86669 | -0.97368 |
| pruning | code | 1.4b | 3 | -0.94617 | -0.99930 |
| pruning | qa | all sizes | 9 | 0.60702 | 0.60605 |
| pruning | qa | 160m | 3 | 0.99809 | 0.93793 |
| pruning | qa | 410m | 3 | -0.93971 | -0.99839 |
| pruning | qa | 1.4b | 3 | 0.80913 | 0.94414 |
| quantization | math | all sizes | 9 | 0.01617 | -0.04048 |
| quantization | math | 160m | 3 | 0.86051 | 0.67720 |
| quantization | math | 410m | 3 | -0.91862 | -0.99352 |
| quantization | math | 1.4b | 3 | -0.94328 | -0.99894 |
| quantization | code | all sizes | 9 | 0.06478 | 0.00100 |
| quantization | code | 160m | 3 | 0.85142 | 0.66415 |
| quantization | code | 410m | 3 | -0.86669 | -0.97368 |
| quantization | code | 1.4b | 3 | -0.94629 | -0.99932 |
| quantization | qa | all sizes | 9 | 0.60771 | 0.60631 |
| quantization | qa | 160m | 3 | 0.99795 | 0.93718 |
| quantization | qa | 410m | 3 | -0.93971 | -0.99839 |
| quantization | qa | 1.4b | 3 | 0.80875 | 0.94393 |

## The three loss curves

**L_c,0(D0)** is dense capability loss; **ΔL_m,c(D0)** is its compression increment; **L_m,c(D0) = L_c,0(D0) + ΔL_m,c(D0)** is absolute compressed loss. Absolute here means the actual compressed loss, not |ΔL|. Lower loss is better. A larger compression penalty with more training need not erase the improvement in dense loss. Each figure has one row per size and these three columns; no averaging across sizes/configs/arms. Lines join measured checkpoints, with linear token and native-token-loss axes. Every table sequence follows **step16000 → step64000 → step143000**. Endpoint changes describe late versus early only; the trajectory shapes use both adjacent transitions with V36's 1e-12 numerical tolerance.

### pruning: math

![pruning math: dense, increment and absolute compressed loss](../../results/v36b-input-comparison/curves_pruning_math.png)

[Full precision curve table (CSV)](../../results/v36b-input-comparison/curves_pruning_math.csv). The increment rises while absolute compressed loss falls from early to late in **4/8** fixed size/config trajectories.

| Size | Dense L0 at the three steps | Dense late−early | Dense shape |
|---|---|---:|---|
| 160m | 1.80064 → 1.73044 → 2.16737 | +0.36672 | nonmonotonic |
| 410m | 1.61204 → 1.49905 → 1.45855 | -0.15348 | decreasing |
| 1.4b | 1.46994 → 1.31377 → 1.23713 | -0.23282 | decreasing |

| Size | Config | ΔL at the three steps | Absolute compressed L at the three steps | ΔL late−early | L late−early | ΔL / L shape |
|---|---:|---|---|---:|---:|---|
| 160m | 0.9 | 0.00888 → 0.00890 → 0.11880 | 1.80952 → 1.73934 → 2.28617 | +0.10992 | +0.47665 | increasing / nonmonotonic |
| 160m | 0.8 | 0.07326 → 0.10789 → 0.73384 | 1.87390 → 1.83833 → 2.90121 | +0.66058 | +1.02731 | increasing / nonmonotonic |
| 160m | 0.7 | 0.25706 → 0.44436 → 2.38808 | 2.05770 → 2.17480 → 4.55545 | +2.13102 | +2.49775 | increasing / increasing |
| 160m | 0.6 | 0.87210 → 1.42177 → 6.46413 | 2.67274 → 3.15222 → 8.63150 | +5.59203 | +5.95875 | increasing / increasing |
| 410m | 0.9 | 0.00225 → 0.00825 → 0.03411 | 1.61429 → 1.50730 → 1.49266 | +0.03186 | -0.12163 | increasing / decreasing |
| 410m | 0.8 | 0.02064 → 0.04742 → 0.19665 | 1.63268 → 1.54647 → 1.65520 | +0.17601 | +0.02252 | increasing / nonmonotonic |
| 410m | 0.7 | 0.10061 → 0.19637 → 0.61281 | 1.71265 → 1.69542 → 2.07136 | +0.51220 | +0.35872 | increasing / nonmonotonic |
| 410m | 0.6 | 0.37365 → 0.71237 → 1.66246 | 1.98568 → 2.21142 → 3.12102 | +1.28882 | +1.13533 | increasing / increasing |
| 1.4b | 0.9 | 0.00105 → 0.00643 → 0.01010 | 1.47099 → 1.32020 → 1.24723 | +0.00906 | -0.22376 | increasing / decreasing |
| 1.4b | 0.8 | 0.01505 → 0.03775 → 0.07063 | 1.48499 → 1.35152 → 1.30776 | +0.05558 | -0.17723 | increasing / decreasing |
| 1.4b | 0.7 | 0.07811 → 0.19548 → 0.29481 | 1.54805 → 1.50925 → 1.53193 | +0.21670 | -0.01612 | increasing / nonmonotonic |
| 1.4b | 0.6 | 0.36538 → 0.99284 → 1.14965 | 1.83532 → 2.30661 → 2.38678 | +0.78427 | +0.55145 | increasing / increasing |

For example, 410m at config 0.9: the increment rises by 0.03186, while dense loss changes by -0.15348 and absolute compressed loss changes by -0.12163 nats/native token. This is increased fragility alongside improved absolute compressed loss.

### pruning: code

![pruning code: dense, increment and absolute compressed loss](../../results/v36b-input-comparison/curves_pruning_code.png)

[Full precision curve table (CSV)](../../results/v36b-input-comparison/curves_pruning_code.csv). The increment rises while absolute compressed loss falls from early to late in **4/8** fixed size/config trajectories.

| Size | Dense L0 at the three steps | Dense late−early | Dense shape |
|---|---|---:|---|
| 160m | 1.75261 → 1.67346 → 2.12913 | +0.37652 | nonmonotonic |
| 410m | 1.54582 → 1.43594 → 1.41627 | -0.12955 | decreasing |
| 1.4b | 1.46524 → 1.35019 → 1.29255 | -0.17269 | decreasing |

| Size | Config | ΔL at the three steps | Absolute compressed L at the three steps | ΔL late−early | L late−early | ΔL / L shape |
|---|---:|---|---|---:|---:|---|
| 160m | 0.9 | 0.01361 → 0.01652 → 0.23794 | 1.76622 → 1.68998 → 2.36707 | +0.22433 | +0.60084 | increasing / nonmonotonic |
| 160m | 0.8 | 0.07565 → 0.18481 → 1.08444 | 1.82826 → 1.85827 → 3.21357 | +1.00879 | +1.38531 | increasing / increasing |
| 160m | 0.7 | 0.27799 → 0.61380 → 3.38026 | 2.03060 → 2.28726 → 5.50939 | +3.10227 | +3.47879 | increasing / increasing |
| 160m | 0.6 | 1.13365 → 2.00446 → 7.24643 | 2.88626 → 3.67792 → 9.37556 | +6.11279 | +6.48930 | increasing / increasing |
| 410m | 0.9 | 0.00184 → 0.00618 → 0.00957 | 1.54766 → 1.44212 → 1.42584 | +0.00773 | -0.12182 | increasing / decreasing |
| 410m | 0.8 | 0.00499 → 0.06216 → 0.21613 | 1.55081 → 1.49810 → 1.63240 | +0.21114 | +0.08159 | increasing / nonmonotonic |
| 410m | 0.7 | 0.11320 → 0.16817 → 0.80604 | 1.65902 → 1.60411 → 2.22231 | +0.69283 | +0.56329 | increasing / nonmonotonic |
| 410m | 0.6 | 0.34550 → 0.99287 → 2.30550 | 1.89131 → 2.42881 → 3.72177 | +1.96001 | +1.83046 | increasing / increasing |
| 1.4b | 0.9 | 0.00006 → 0.00505 → 0.02110 | 1.46530 → 1.35524 → 1.31364 | +0.02104 | -0.15165 | increasing / decreasing |
| 1.4b | 0.8 | 0.01254 → 0.01860 → 0.05152 | 1.47778 → 1.36879 → 1.34407 | +0.03898 | -0.13371 | increasing / decreasing |
| 1.4b | 0.7 | 0.04956 → 0.08807 → 0.17406 | 1.51480 → 1.43826 → 1.46660 | +0.12449 | -0.04819 | increasing / nonmonotonic |
| 1.4b | 0.6 | 0.35399 → 0.97766 → 0.98277 | 1.81923 → 2.32785 → 2.27531 | +0.62877 | +0.45609 | increasing / nonmonotonic |

For example, 410m at config 0.9: the increment rises by 0.00773, while dense loss changes by -0.12955 and absolute compressed loss changes by -0.12182 nats/native token. This is increased fragility alongside improved absolute compressed loss.

### pruning: qa

![pruning qa: dense, increment and absolute compressed loss](../../results/v36b-input-comparison/curves_pruning_qa.png)

[Full precision curve table (CSV)](../../results/v36b-input-comparison/curves_pruning_qa.csv). The increment rises while absolute compressed loss falls from early to late in **0/8** fixed size/config trajectories.

| Size | Dense L0 at the three steps | Dense late−early | Dense shape |
|---|---|---:|---|
| 160m | 4.77070 → 4.91647 → 5.22231 | +0.45161 | increasing |
| 410m | 5.07539 → 5.01889 → 4.99275 | -0.08264 | decreasing |
| 1.4b | 4.91249 → 5.11199 → 5.12034 | +0.20785 | increasing |

| Size | Config | ΔL at the three steps | Absolute compressed L at the three steps | ΔL late−early | L late−early | ΔL / L shape |
|---|---:|---|---|---:|---:|---|
| 160m | 0.9 | 0.01209 → -0.04396 → 0.22065 | 4.78279 → 4.87250 → 5.44297 | +0.20856 | +0.66017 | nonmonotonic / increasing |
| 160m | 0.8 | -0.02718 → -0.11122 → 0.95247 | 4.74352 → 4.80525 → 6.17479 | +0.97965 | +1.43126 | nonmonotonic / increasing |
| 160m | 0.7 | -0.17363 → 0.26675 → 2.81000 | 4.59708 → 5.18322 → 8.03232 | +2.98363 | +3.43524 | increasing / increasing |
| 160m | 0.6 | 0.34847 → 1.52519 → 6.10658 | 5.11918 → 6.44166 → 11.32890 | +5.75811 | +6.20972 | increasing / increasing |
| 410m | 0.9 | -0.01646 → -0.01444 → -0.11538 | 5.05894 → 5.00446 → 4.87738 | -0.09892 | -0.18156 | nonmonotonic / decreasing |
| 410m | 0.8 | -0.03274 → -0.11811 → -0.16683 | 5.04266 → 4.90078 → 4.82593 | -0.13409 | -0.21673 | decreasing / decreasing |
| 410m | 0.7 | -0.18102 → -0.29004 → 0.12773 | 4.89437 → 4.72885 → 5.12048 | +0.30876 | +0.22612 | nonmonotonic / nonmonotonic |
| 410m | 0.6 | -0.39134 → -0.24774 → 1.25677 | 4.68405 → 4.77115 → 6.24952 | +1.64811 | +1.56547 | increasing / increasing |
| 1.4b | 0.9 | 0.00986 → -0.02368 → 0.02822 | 4.92235 → 5.08831 → 5.14856 | +0.01836 | +0.22621 | nonmonotonic / increasing |
| 1.4b | 0.8 | -0.00250 → -0.06675 → -0.06381 | 4.90999 → 5.04524 → 5.05653 | -0.06131 | +0.14654 | nonmonotonic / increasing |
| 1.4b | 0.7 | -0.14169 → -0.41641 → -0.41394 | 4.77079 → 4.69558 → 4.70639 | -0.27225 | -0.06440 | nonmonotonic / nonmonotonic |
| 1.4b | 0.6 | -0.31993 → -0.49958 → -0.44519 | 4.59256 → 4.61240 → 4.67514 | -0.12527 | +0.08258 | nonmonotonic / increasing |

### quantization: math

![quantization math: dense, increment and absolute compressed loss](../../results/v36b-input-comparison/curves_quantization_math.png)

[Full precision curve table (CSV)](../../results/v36b-input-comparison/curves_quantization_math.csv). The increment rises while absolute compressed loss falls from early to late in **4/8** fixed size/config trajectories.

| Size | Dense L0 at the three steps | Dense late−early | Dense shape |
|---|---|---:|---|
| 160m | 1.80064 → 1.73044 → 2.16527 | +0.36463 | nonmonotonic |
| 410m | 1.61204 → 1.49905 → 1.45855 | -0.15348 | decreasing |
| 1.4b | 1.46994 → 1.31300 → 1.23713 | -0.23282 | decreasing |

| Size | Config | ΔL at the three steps | Absolute compressed L at the three steps | ΔL late−early | L late−early | ΔL / L shape |
|---|---:|---|---|---:|---:|---|
| 160m | 8 | -0.00028 → 0.00000 → 0.03164 | 1.80036 → 1.73044 → 2.19691 | +0.03191 | +0.39654 | increasing / nonmonotonic |
| 160m | 6 | 0.00210 → 0.00716 → 0.18777 | 1.80274 → 1.73760 → 2.35304 | +0.18568 | +0.55030 | increasing / nonmonotonic |
| 160m | 4 | 0.13039 → 0.22157 → 2.83742 | 1.93103 → 1.95201 → 5.00269 | +2.70703 | +3.07166 | increasing / increasing |
| 160m | 3 | 1.55355 → 3.86182 → 23.03002 | 3.35419 → 5.59226 → 25.19529 | +21.47647 | +21.84110 | increasing / increasing |
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
| 160m | 1.75261 → 1.67346 → 2.11594 | +0.36332 | nonmonotonic |
| 410m | 1.54582 → 1.43594 → 1.41627 | -0.12955 | decreasing |
| 1.4b | 1.46524 → 1.35025 → 1.29255 | -0.17269 | decreasing |

| Size | Config | ΔL at the three steps | Absolute compressed L at the three steps | ΔL late−early | L late−early | ΔL / L shape |
|---|---:|---|---|---:|---:|---|
| 160m | 8 | 0.00095 → -0.00155 → 0.05408 | 1.75357 → 1.67192 → 2.17001 | +0.05313 | +0.41645 | nonmonotonic / nonmonotonic |
| 160m | 6 | 0.01349 → 0.00351 → 0.26866 | 1.76610 → 1.67697 → 2.38460 | +0.25517 | +0.61849 | nonmonotonic / nonmonotonic |
| 160m | 4 | 0.16407 → 0.31953 → 3.52359 | 1.91669 → 1.99299 → 5.63953 | +3.35952 | +3.72284 | increasing / increasing |
| 160m | 3 | 2.12461 → 5.99828 → 24.84104 | 3.87723 → 7.67174 → 26.95698 | +22.71643 | +23.07975 | increasing / increasing |
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
| 160m | 4.77070 → 4.91647 → 5.22505 | +0.45434 | increasing |
| 410m | 5.07539 → 5.01889 → 4.99275 | -0.08264 | decreasing |
| 1.4b | 4.91249 → 5.11214 → 5.12034 | +0.20785 | increasing |

| Size | Config | ΔL at the three steps | Absolute compressed L at the three steps | ΔL late−early | L late−early | ΔL / L shape |
|---|---:|---|---|---:|---:|---|
| 160m | 8 | -0.00478 → 0.00737 → -0.00927 | 4.76592 → 4.92384 → 5.21578 | -0.00449 | +0.44986 | nonmonotonic / increasing |
| 160m | 6 | 0.03357 → 0.00440 → 0.35789 | 4.80427 → 4.92087 → 5.58294 | +0.32432 | +0.77867 | nonmonotonic / increasing |
| 160m | 4 | 0.05665 → -0.11478 → 3.63926 | 4.82735 → 4.80169 → 8.86431 | +3.58261 | +4.03695 | nonmonotonic / nonmonotonic |
| 160m | 3 | 1.33433 → 3.91124 → 23.95342 | 6.10504 → 8.82771 → 29.17847 | +22.61909 | +23.07343 | increasing / increasing |
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
