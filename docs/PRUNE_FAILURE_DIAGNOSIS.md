# Pruning prediction failure diagnosis

The old density-only headline reproduces as 0.33288 [0.22319, 0.83225] nats on 28 pre-cliff rows from four models. The reported 1.3–1.9 values are v28 development LOMO errors, not Qwen3-8B prospective errors. Frozen Qwen3-8B Mode A MAE is 0.61993 [0.61993, 0.61993], versus zero-change 0.24075 [0.24075, 0.24075]. Range selection, coefficient transfer, and calibration conditioning are distinguishable failure sources; these data do not support a blanket unpredictability claim.

All intervals below are 95% paired MODEL-bootstrap intervals, using 10000 resamples. Whole models retain their capability/density cells; means weight cells equally. Fits are held fixed, so intervals describe this small panel, not retraining or measurement uncertainty. Cross-range contrasts resample the same models and recompute each slice's denominator. Counts, chosen hyperparameters, and coordinate brackets are design facts, not estimated metrics. Qwen3-8B has one model: its [x,x] intervals are explicitly degenerate and cannot establish population uncertainty.

![Qwen curve decomposition and development calibration errors](../../results/v29-prune-diagnosis/diagnosis.png)

Figure: top, Qwen3-8B observed curves, frozen Mode A, and an oracle scalar correction on the scored grid (one model; no nondegenerate model CI). Bottom, development LOMO MAEs with model-bootstrap intervals, using a logarithmic vertical scale. Oracle correction is never used in the LOMO results.

## Inputs and matched protocol

Only JSON observations were loaded. Twelve development models come from the v28 metadata whitelist; Qwen3-8B is held out completely. All v6 grids are complete, and v9 contains no alternate pruning-loss JSON. The schema is `{density: {math, code, qa}}`; `1.0` supplies each source's own dense CE. Underscore metadata and archive copies are excluded. Saved v28 development inputs and its frozen formula seal were verified. Historical reproduction alone includes the two models' infill densities.

Mode A permits N0, family, and dense capability loss, with zero target compression points. Mode B adds exactly density 0.9 per capability; 0.9 is excluded from every matched test. Main training uses 0.9/0.8/0.7/0.6 on other models; testing uses the same target rows at 0.8/0.7/0.6. All gamma, coefficient, family, and feature standardization fits exclude the target model. Mode B shrinkage is separate and never replaces Mode A.

The exact v18 `density_only` form is a single signed `A(1-d)^gamma`, pooled across capabilities, fitted by original-scale least squares with gamma in [0.05,8]. We import its original fitter and predictor. This differs from v14's mechanism-corrected, positive-residual log fits, which are not the C5 headline estimator. `old_cap_A` keeps the same form but fits each capability separately. v28 uses `a_c(model)((1-d)/0.3)^gamma_c`, profiles a shared shape with per-model labels, then maps labels from log(N0/1e9), family, and dense L_c using its unchanged ridge helper. `v28_mean_A` replaces that mapping with the training-population coefficient mean. Thus the density exponent family is shared by both approaches; a remaining difference can arise from pooling, shape estimation, or amplitude mapping rather than a new density function.

## Historical score and calibration access

Reconstruction yields 174 pre-cliff rows, 146 training rows at d≥0.55, and 28 test rows below 0.55. The test models are Qwen3-1.7B, Qwen3-4B, olmo3-32b, and olmo3-7b. The historical fit includes their shallow compressed observations in its pooled coefficients; it does not fit a separate amplitude to each target. Removing each target's contribution while retaining exactly its test cells isolates that access.

| Capability | Models / rows | Mean absolute response | historical_own_points | historical_transfer_A | zero |
|---|---:|---:|---:|---:|---:|
| math | 2 / 8 | 0.34335 [0.29444, 0.42487] | 0.19614 [0.16173, 0.25350] | 0.18719 [0.15691, 0.23764] | 0.34335 [0.29444, 0.42487] |
| code | 3 / 8 | 0.53546 [0.38400, 0.90386] | 0.36431 [0.21392, 0.73494] | 0.33921 [0.20834, 0.63207] | 0.53546 [0.38400, 0.90386] |
| qa | 4 / 12 | 0.31817 [0.13145, 0.80710] | 0.40309 [0.21056, 0.86469] | 0.42586 [0.20185, 0.91596] | 0.31817 [0.13145, 0.80710] |
| pooled | 4 / 28 | 0.38745 [0.30816, 0.83129] | 0.33288 [0.22319, 0.83225] | 0.33291 [0.21338, 0.84499] | 0.38745 [0.30816, 0.83129] |

Target-compression access benefit (transfer-only MAE minus original MAE): 0.00003 [-0.01152, 0.01500]. This measures the actual historical use of target points; the one-point B experiment below is a different calibration protocol.

| Historical target | Compressed configurations contributing to pooled training |
|---|---|
| Qwen3-1.7B | 0.9, 0.8, 0.7, 0.675, 0.65, 0.625, 0.6, 0.575, 0.55 |
| Qwen3-4B | 0.9, 0.8, 0.7, 0.6, 0.55 |
| olmo3-32b | 0.9, 0.8, 0.7, 0.6, 0.55 |
| olmo3-7b | 0.9, 0.8, 0.7, 0.6, 0.55 |

The historical leave-one-family-out density-only score is 0.33928 [0.23733, 0.45217]; it has no held-out-family compression access. This excludes target calibration as a universal explanation for the old ~0.33 headline.

## Matched-condition LOMO

Entries are MAEs; response amplitudes and zero-change are reported on identical rows.

| Capability | Models / rows | Mean absolute response | old_A | old_cap_A | v28_mean_A | v28_A | zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| math | 12 / 36 | 1.56887 [0.73220, 2.51960] | 1.54198 [1.02122, 2.10633] | 1.54609 [1.02555, 2.09453] | 1.53428 [0.99857, 2.10917] | 1.41473 [0.84500, 2.06268] | 1.56887 [0.73220, 2.51960] |
| code | 12 / 36 | 1.57652 [0.72986, 2.54230] | 1.56519 [1.07008, 2.10213] | 1.59184 [1.10233, 2.11419] | 1.60451 [1.11044, 2.12783] | 1.29818 [0.71641, 1.92430] | 1.57652 [0.72986, 2.54230] |
| qa | 12 / 36 | 1.69267 [0.64673, 2.87452] | 2.08642 [1.51828, 2.75822] | 2.08359 [1.48367, 2.80512] | 2.07631 [1.47796, 2.79466] | 1.91500 [1.19195, 2.77552] | 1.69267 [0.64673, 2.87452] |
| pooled | 12 / 108 | 1.61269 [0.73011, 2.60592] | 1.73120 [1.23034, 2.27170] | 1.74051 [1.23669, 2.29137] | 1.73836 [1.22663, 2.29887] | 1.54263 [0.94164, 2.22112] | 1.61269 [0.73011, 2.60592] |

| Capability | old A gain over zero | v28 A gain over zero | v28 A minus old A | Mapping effect: v28 A minus mean |
|---|---:|---:|---:|---:|
| math | 0.02689 [-0.65211, 0.69134] | 0.15414 [-0.36182, 0.67097] | -0.12725 [-0.44133, 0.20272] | -0.11956 [-0.43544, 0.21173] |
| code | 0.01134 [-0.68785, 0.69991] | 0.27835 [-0.26963, 0.83457] | -0.26701 [-0.69821, 0.21066] | -0.30633 [-0.74213, 0.17243] |
| qa | -0.39376 [-1.09780, 0.32347] | -0.22234 [-0.90992, 0.48238] | -0.17142 [-0.69981, 0.38435] | -0.16130 [-0.67561, 0.37932] |
| pooled | -0.11851 [-0.77936, 0.53734] | 0.07005 [-0.42729, 0.56574] | -0.18856 [-0.56394, 0.20284] | -0.19573 [-0.56387, 0.19078] |

Positive gain over zero is better; negative candidate-minus-candidate MAE is better for the first candidate. Zero-change is the required simple comparator. Training-population mean is also shown; zero is not assumed to win every capability. Matched Mode A excludes unequal target calibration and unequal test-row difficulty as explanations for its remaining predictor gap.

On this matched grid v28 is lower-error than the old form in all three capabilities, although the paired intervals include zero. The apparent jump from the historical score is therefore not a demonstrated deterioration in v28's functional form. Neither Mode A method establishes an improvement over zero-change across this small model panel.

## Damage range and three-way attribution

The historical pre-cliff rule removes the first positive ΔL>1 nat and every deeper point, separately per model/capability. It is an outcome-selected diagnostic, not an available Mode A deployment filter. Negative responses are retained, including large QA decreases. Density≥0.6 alone is not smooth: several models cross the historical damage threshold by 0.8 or 0.7. The next table holds both fitted predictors fixed and restricts scoring to d≥0.6 AND pre-cliff.

| Capability | Models / rows | Mean absolute response | old_A | v28_A | old_B | v28_B | zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| math | 12 / 25 | 0.27329 [0.15880, 0.42581] | 0.91671 [0.48217, 1.26465] | 0.68979 [0.40815, 0.96075] | 0.54063 [0.33076, 0.77545] | 0.91242 [0.53147, 1.33669] | 0.27329 [0.15880, 0.42581] |
| code | 12 / 25 | 0.26664 [0.15241, 0.40767] | 1.02439 [0.59898, 1.33971] | 0.53742 [0.32759, 0.74821] | 1.18073 [0.30408, 2.09798] | 0.70906 [0.22617, 1.21540] | 0.26664 [0.15241, 0.40767] |
| qa | 12 / 28 | 0.36115 [0.19373, 0.56849] | 1.44460 [1.00135, 1.81054] | 1.09984 [0.61681, 1.57790] | 6.39225 [2.86391, 10.82216] | 13.23228 [5.42234, 23.15399] | 0.36115 [0.19373, 0.56849] |
| pooled | 12 / 78 | 0.30270 [0.20356, 0.41693] | 1.14072 [0.76304, 1.44016] | 0.78815 [0.50736, 1.05915] | 2.84637 [1.42781, 4.68506] | 5.26976 [2.27075, 9.40066] | 0.30270 [0.20356, 0.41693] |

The following controlled bridge is additive: old B on smooth rows → old A on smooth rows (calibration access) → old A on all v28 rows (damage range) → v28 A on those same rows (residual form/estimation). It diagnoses matched conditions; it is not a causal decomposition of two historical scores with different test sets. Changing the order changes components when range and form interact.

| Capability | Calibration: old A−B, smooth | Range: old A full−smooth | Residual form: v28 A−old A, full | Sum / controlled gap |
|---|---:|---:|---:|---:|
| math | 0.37608 [-0.18586, 0.84493] | 0.62527 [0.09868, 1.32461] | -0.12725 [-0.44133, 0.20272] | 0.87410 [0.29790, 1.52267] |
| code | -0.15634 [-1.01435, 0.65077] | 0.54080 [0.03207, 1.19843] | -0.26701 [-0.69821, 0.21066] | 0.11744 [-1.11846, 1.32989] |
| qa | -4.94765 [-9.45020, -1.32436] | 0.64183 [0.05984, 1.46707] | -0.17142 [-0.69981, 0.38435] | -4.47725 [-9.23338, -0.64845] |
| pooled | -1.70565 [-3.72558, -0.16475] | 0.59047 [0.05271, 1.30432] | -0.18856 [-0.56394, 0.20284] | -1.30374 [-3.44183, 0.46684] |

| Capability | v28 range penalty full−smooth | Residual v28−old on smooth | Range × form interaction |
|---|---:|---:|---:|
| math | 0.72493 [0.15947, 1.43692] | -0.22692 [-0.46800, 0.03851] | 0.09967 [-0.10169, 0.29393] |
| code | 0.76076 [0.22031, 1.37550] | -0.48698 [-0.82836, -0.05890] | 0.21996 [-0.01959, 0.45391] |
| qa | 0.81516 [0.12436, 1.72964] | -0.34475 [-0.84759, 0.20900] | 0.17333 [-0.12116, 0.53443] |
| pooled | 0.75448 [0.17133, 1.48897] | -0.35257 [-0.66135, 0.01623] | 0.16401 [0.02967, 0.30378] |

Matched one-point calibration on the full original v28 test grid (exactly d=0.9; no target shape fitting):

| Capability | Models / rows | Mean absolute response | old_A | old_B | v28_A | v28_B | zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| math | 12 / 36 | 1.56887 [0.73220, 2.51960] | 1.54198 [1.02122, 2.10633] | 4.06510 [1.10286, 8.97895] | 1.41473 [0.84500, 2.06268] | 9.29039 [2.09719, 21.74722] | 1.56887 [0.73220, 2.51960] |
| code | 12 / 36 | 1.57652 [0.72986, 2.54230] | 1.56519 [1.07008, 2.10213] | 3.01161 [1.10917, 6.01271] | 1.29818 [0.71641, 1.92430] | 1.80860 [0.87896, 3.09608] | 1.57652 [0.72986, 2.54230] |
| qa | 12 / 36 | 1.69267 [0.64673, 2.87452] | 2.08642 [1.51828, 2.75822] | 17.37414 [7.13749, 29.68038] | 1.91500 [1.19195, 2.77552] | 34.49935 [15.84184, 55.88683] | 1.69267 [0.64673, 2.87452] |
| pooled | 12 / 108 | 1.61269 [0.73011, 2.60592] | 1.73120 [1.23034, 2.27170] | 8.15028 [3.38985, 14.51726] | 1.54263 [0.94164, 2.22112] | 15.19945 [6.49823, 26.01949] | 1.61269 [0.73011, 2.60592] |

Deep/collapse extension: train both on the original grid, then score identical common-grid rows from d=0.8 through 0.3. This is extrapolation beyond v28's frozen domain and is labelled as such; it cannot explain the original v28 errors at d≥0.6 by itself.

| Capability | Models / rows | Mean absolute response | old_A | v28_A | v28_B | zero |
|---|---:|---:|---:|---:|---:|---:|
| math | 12 / 108 | 7.63144 [4.98551, 10.16984] | 7.91851 [6.21188, 9.80000] | 9.84052 [6.93799, 12.68429] | 137.96676 [21.60802, 338.59700] | 7.63144 [4.98551, 10.16984] |
| code | 12 / 108 | 7.40303 [4.94111, 9.76806] | 8.10573 [6.55417, 9.85333] | 5.04114 [3.54695, 6.56473] | 15.80169 [5.80291, 31.54456] | 7.40303 [4.94111, 9.76806] |
| qa | 12 / 108 | 6.09978 [3.77060, 8.41618] | 8.94281 [7.11342, 10.84563] | 12.71748 [9.39983, 16.16855] | 440.12856 [183.40842, 742.89441] | 6.09978 [3.77060, 8.41618] |
| pooled | 12 / 324 | 7.04475 [4.56618, 9.43385] | 8.32235 [6.65106, 10.12091] | 9.19971 [6.87634, 11.54247] | 197.96567 [71.96426, 358.19217] | 7.04475 [4.56618, 9.43385] |

To distinguish training-range contamination from scoring-range selection, both forms are also refitted on other models' d≥0.6 pre-cliff rows and evaluated on the same pre-cliff target rows:

| Capability | Models / rows | Mean absolute response | old_A | v28_A | zero |
|---|---:|---:|---:|---:|---:|
| math | 12 / 25 | 0.27329 [0.15880, 0.42581] | 0.25660 [0.19084, 0.35577] | 0.76420 [0.47602, 1.05823] | 0.27329 [0.15880, 0.42581] |
| code | 12 / 25 | 0.26664 [0.15241, 0.40767] | 0.25912 [0.18746, 0.35059] | 0.86464 [0.47947, 1.29001] | 0.26664 [0.15241, 0.40767] |
| qa | 12 / 28 | 0.36115 [0.19373, 0.56849] | 0.38819 [0.19071, 0.63778] | 0.33411 [0.18878, 0.48635] | 0.36115 [0.19373, 0.56849] |
| pooled | 12 / 78 | 0.30270 [0.20356, 0.41693] | 0.30464 [0.21690, 0.40979] | 0.64200 [0.46272, 0.82886] | 0.30270 [0.20356, 0.41693] |

Matching the training range matters as well as restricting test rows. The next bridge keeps Mode A throughout: old smooth-fit/smooth-test → old original-fit/smooth-test → old original-fit/full-test → v28 original-fit/full-test. The first two columns isolate inclusion of rapid-damage training observations and test observations. Add the residual full-grid form difference above to obtain the total controlled A gap.

| Capability | Training-range effect, old | Scoring-range effect, old | Combined range effect, old | Residual form with BOTH smooth refits | Total controlled A gap |
|---|---:|---:|---:|---:|---:|
| math | 0.66011 [0.17532, 1.04206] | 0.62527 [0.09868, 1.32461] | 1.28538 [0.78049, 1.80647] | 0.50761 [0.21718, 0.78343] | 1.15813 [0.63447, 1.74359] |
| code | 0.76528 [0.31123, 1.10106] | 0.54080 [0.03207, 1.19843] | 1.30607 [0.85728, 1.76922] | 0.60552 [0.19899, 1.06243] | 1.03906 [0.48950, 1.60500] |
| qa | 1.05641 [0.63066, 1.34740] | 0.64183 [0.05984, 1.46707] | 1.69823 [1.15606, 2.34962] | -0.05408 [-0.33534, 0.20929] | 1.52681 [0.77782, 2.39236] |
| pooled | 0.83608 [0.43445, 1.14012] | 0.59047 [0.05271, 1.30432] | 1.42655 [0.95650, 1.92805] | 0.33736 [0.09560, 0.58231] | 1.23799 [0.65291, 1.89055] |

On this controlled Mode A bridge, combined training/scoring range contributes 1.42655 [0.95650, 1.92805] nats, whereas switching to v28 on matched full rows changes error by -0.18856 [-0.56394, 0.20284]. The actual historical calibration-access effect is negligible within its interval. Range selection is therefore the main supported explanation for the headline discrepancy; it is not evidence that the new form universally predicts less accurately.

With smooth training and smooth scoring, old density-only returns close to the historical error scale. Here a residual v28 disadvantage remains for math/code after matching range and calibration. This is a distinct prediction-recipe error: outcome censoring leaves unequal per-model support, and v28's profiled model coefficients differ from a pooled fit. The following ablation separates capability pooling, profiled shape/coefficient estimation, and metadata mapping on those exact rows; it does not attribute all error to metadata.

| Capability | Capability-specific old minus pooled old | v28 population mean minus capability-specific old | v28 mapping minus population mean |
|---|---:|---:|---:|
| math | 0.02808 [-0.03006, 0.07972] | 1.22265 [0.52612, 1.77391] | -0.74312 [-1.34931, -0.02647] |
| code | 0.03212 [-0.02236, 0.08277] | 2.09441 [1.07918, 2.85123] | -1.52100 [-2.38047, -0.44093] |
| qa | 0.04773 [-0.05545, 0.14578] | 0.08642 [-0.05072, 0.21638] | -0.18823 [-0.46610, 0.06946] |
| pooled | 0.03643 [-0.01382, 0.08392] | 1.09418 [0.51552, 1.54087] | -0.79325 [-1.31970, -0.17437] |

Both forms refitted on other models' full 0.9–0.3 grid, scored on 0.8–0.3 (includes saturation and nonmonotonic collapse):

| Capability | Models / rows | Mean absolute response | old_A | v28_A | zero |
|---|---:|---:|---:|---:|---:|
| math | 12 / 108 | 7.63144 [4.98551, 10.16984] | 4.72246 [3.44992, 5.94328] | 3.58514 [2.95377, 4.34294] | 7.63144 [4.98551, 10.16984] |
| code | 12 / 108 | 7.40303 [4.94111, 9.76806] | 4.41968 [3.27005, 5.55701] | 3.07183 [2.54478, 3.72233] | 7.40303 [4.94111, 9.76806] |
| qa | 12 / 108 | 6.09978 [3.77060, 8.41618] | 4.74259 [3.71958, 5.77978] | 3.66943 [2.94840, 4.50935] | 6.09978 [3.77060, 8.41618] |
| pooled | 12 / 324 | 7.04475 [4.56618, 9.43385] | 4.62824 [3.55612, 5.70072] | 3.44213 [2.86328, 4.15036] | 7.04475 [4.56618, 9.43385] |

The response amplitudes reveal how much the historical ≤1-nat selection changes task difficulty. The matched residual difference isolates the prediction recipe, while the per-capability and population-mean ablations distinguish pooling from coefficient mapping. The two laws share a power-density family; a residual gap is not evidence that an entirely different density function was tested.

## Qwen3-8B: amplitude, sign, shape, and damage onset

Frozen formulas were bound directly to saved source-dense losses and, for Mode B only, its d=0.9 loss. No Qwen3-8B curve entered fitting or shrinkage selection. Freeze names Qwen/Qwen3-8B-Base; actual pruning JSON contains no checkpoint or revision identity. Directory Qwen--Qwen3-8B alone cannot verify Base versus post-trained identity. This limits checkpoint-level attribution; no identity difference is assumed.

| Capability | Models / rows | Mean absolute response | old_A | v28_A | v28_B | v28_B_shrink | zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| math | 1 / 3 | 0.06378 [0.06378, 0.06378] | 1.45106 [1.45106, 1.45106] | 0.73463 [0.73463, 0.73463] | 0.98747 [0.98747, 0.98747] | 1.45650 [1.45650, 1.45650] | 0.06378 [0.06378, 0.06378] |
| code | 1 / 3 | 0.12701 [0.12701, 0.12701] | 1.38784 [1.38784, 1.38784] | 0.38938 [0.38938, 0.38938] | 0.29623 [0.29623, 0.29623] | 0.62058 [0.62058, 0.62058] | 0.12701 [0.12701, 0.12701] |
| qa | 1 / 3 | 0.53146 [0.53146, 0.53146] | 2.04631 [2.04631, 2.04631] | 0.73577 [0.73577, 0.73577] | 22.68855 [22.68855, 22.68855] | 1.87922 [1.87922, 1.87922] | 0.53146 [0.53146, 0.53146] |
| pooled | 1 / 9 | 0.24075 [0.24075, 0.24075] | 1.62840 [1.62840, 1.62840] | 0.61993 [0.61993, 0.61993] | 7.99075 [7.99075, 7.99075] | 1.31877 [1.31877, 1.31877] | 0.24075 [0.24075, 0.24075] |

| Capability | Actual calibration ΔL at 0.9 | Frozen A improvement over zero | Shrink B improvement over zero |
|---|---:|---:|---:|
| math | 0.00370 [0.00370, 0.00370] | -0.67085 [-0.67085, -0.67085] | -1.39271 [-1.39271, -1.39271] |
| code | 0.00491 [0.00491, 0.00491] | -0.26237 [-0.26237, -0.26237] | -0.49357 [-0.49357, -0.49357] |
| qa | -0.06360 [-0.06360, -0.06360] | -0.20431 [-0.20431, -0.20431] | -1.34775 [-1.34775, -1.34775] |

| Capability / density | Actual signed ΔL | Frozen A ΔL | Frozen B ΔL |
|---|---:|---:|---:|
| math / 0.8 | 0.01083 [0.01083, 0.01083] | 0.07194 [0.07194, 0.07194] | 0.09472 [0.09472, 0.09472] |
| math / 0.7 | 0.03204 [0.03204, 0.03204] | 0.47973 [0.47973, 0.47973] | 0.63166 [0.63166, 0.63166] |
| math / 0.6 | 0.14848 [0.14848, 0.14848] | 1.84357 [1.84357, 1.84357] | 2.42740 [2.42740, 2.42740] |
| code / 0.8 | 0.02311 [0.02311, 0.02311] | 0.08105 [0.08105, 0.08105] | 0.06643 [0.06643, 0.06643] |
| code / 0.7 | 0.10120 [0.10120, 0.10120] | 0.37191 [0.37191, 0.37191] | 0.30482 [0.30482, 0.30482] |
| code / 0.6 | 0.25672 [0.25672, 0.25672] | 1.09621 [1.09621, 1.09621] | 0.89846 [0.89846, 0.89846] |
| qa / 0.8 | -0.16438 [-0.16438, -0.16438] | 0.01637 [0.01637, 0.01637] | -1.85999 [-1.85999, -1.85999] |
| qa / 0.7 | -0.41490 [-0.41490, -0.41490] | 0.11791 [0.11791, 0.11791] | -13.40035 [-13.40035, -13.40035] |
| qa / 0.6 | -1.01511 [-1.01511, -1.01511] | 0.47866 [0.47866, 0.47866] | -54.39969 [-54.39969, -54.39969] |

For each capability, fit a single scalar s by least squares on the THREE scored points (an oracle diagnostic). The exact orthogonal decomposition is MSE(p−y) = MSE((1−s)p) + MSE(sp−y). The first term is amplitude **or sign** mismatch; negative s means a sign reversal, which positive amplitude correction cannot repair. MAE before/after is also reported, but only the squared errors have this orthogonal decomposition. A pooled row fits ONE scalar across all capabilities.

| Capability | Signed scale | Raw MAE | Scaled MAE | Nonnegative-scale MAE | Amplitude/sign MSE fraction | Shape MSE fraction |
|---|---:|---:|---:|---:|---:|---:|
| math | 0.07977 [0.07977, 0.07977] | 0.73463 [0.73463, 0.73463] | 0.00425 [0.00425, 0.00425] | 0.00425 [0.00425, 0.00425] | 0.99998 [0.99998, 0.99998] | 0.00002 [0.00002, 0.00002] |
| code | 0.23833 [0.23833, 0.23833] | 0.38938 [0.38938, 0.38938] | 0.00697 [0.00697, 0.00697] | 0.00697 [0.00697, 0.00697] | 0.99975 [0.99975, 0.99975] | 0.00025 [0.00025, 0.00025] |
| qa | -2.20935 [-2.20935, -2.20935] | 0.73577 [0.73577, 0.73577] | 0.10835 [0.10835, 0.10835] | 0.53146 [0.53146, 0.53146] | 0.98348 [0.98348, 0.98348] | 0.01652 [0.01652, 0.01652] |
| pooled_single_scale | 0.01404 [0.01404, 0.01404] | 0.61993 [0.61993, 0.61993] | 0.23556 [0.23556, 0.23556] | 0.23556 [0.23556, 0.23556] | 0.79265 [0.79265, 0.79265] | 0.20735 [0.20735, 0.20735] |

**Amplitude transfer dominates math/code; QA is primarily a sign error.** Their best positive scales are far below one and the residual curve MAEs are small. QA requires a negative scale: shrinking a positive curve alone cannot predict its observed loss reduction. The single pooled scale mostly suppresses all predictions toward zero; capability-specific scales explain much more. These are diagnostic oracle corrections, not validated predictors.

Positive-damage onset uses the historical ΔL>1 nat threshold; it denotes departure from the perturbative region, not proven catastrophic collapse. Actual locations are grid brackets or censored below d=0.6. A power law has no explicit collapse parameter. Its continuous threshold crossing is reported with that limitation. Thresholds 0.5, 2, and 8 nats plus absolute-response crossings are retained in summary.json to expose threshold and QA-sign sensitivity.

| Capability | Predicted continuous crossing | Predicted measured-grid crossing | Actual crossing | After signed amplitude correction |
|---|---:|---|---|---|
| math | 0.64901 [0.64901, 0.64901] | [0.6, 0.7] | not observed through 0.6; if present, below 0.6 | not observed through 0.6; if present, below 0.6 |
| code | 0.60966 [0.60966, 0.60966] | [0.6, 0.7] | not observed through 0.6; if present, below 0.6 | not observed through 0.6; if present, below 0.6 |
| qa | 0.53467 [0.53467, 0.53467] | not observed through 0.6; if present, below 0.6 | not observed through 0.6; if present, below 0.6 | not observed through 0.6; if present, below 0.6 |

Premature threshold entry overlaps with amplitude error: changing amplitude alone can move or remove the predicted crossing. It must not be added as a third independent squared-error component. Math/code overestimation and QA sign mismatch are assessed separately from the density-shape residual; the measured source grid cannot identify its eventual collapse density.

## One-point calibration instability

Direct calibration uses a=y(0.9)/x(0.9). Shrinkage minimizes `(y0−a*x0)^2 + lambda*(a−mean_dev)^2`. Lambda is selected separately for each capability by inner LOMO on the eleven outer-training models, minimizing model-equal MAE at 0.8/0.7/0.6. Fixed grid: [0.0, 1e-08, 1e-06, 0.0001, 0.01, 1.0, 100.0]. Gamma, population mean, and mapping are refitted in every inner fold. Every mode still receives exactly one target compressed point. Shrinkage targets the dev-population mean, not the target curve or an oracle amplitude.

| Capability | Models / rows | Mean absolute response | v28_A | v28_B | v28_B_shrink | zero |
|---|---:|---:|---:|---:|---:|---:|
| math | 12 / 36 | 1.56887 [0.73220, 2.51960] | 1.41473 [0.84500, 2.06268] | 9.29039 [2.09719, 21.74722] | 1.72296 [1.16891, 2.33629] | 1.56887 [0.73220, 2.51960] |
| code | 12 / 36 | 1.57652 [0.72986, 2.54230] | 1.29818 [0.71641, 1.92430] | 1.80860 [0.87896, 3.09608] | 1.27246 [0.79210, 1.80711] | 1.57652 [0.72986, 2.54230] |
| qa | 12 / 36 | 1.69267 [0.64673, 2.87452] | 1.91500 [1.19195, 2.77552] | 34.49935 [15.84184, 55.88683] | 2.02994 [1.48179, 2.68957] | 1.69267 [0.64673, 2.87452] |
| pooled | 12 / 108 | 1.61269 [0.73011, 2.60592] | 1.54263 [0.94164, 2.22112] | 15.19945 [6.49823, 26.01949] | 1.67512 [1.20432, 2.24707] | 1.61269 [0.73011, 2.60592] |

| Capability | Direct−shrink MAE (positive favors shrink) | Fraction of model/capability curves improved | Mean absolute ΔL(0.9) | Mean absolute ΔL(test) |
|---|---:|---:|---:|---:|
| math | 7.56743 [0.49358, 19.90933] | 0.75000 [0.50000, 1.00000] | 0.01830 [0.00606, 0.03568] | 1.56887 [0.73220, 2.51960] |
| code | 0.53614 [-0.18476, 1.62388] | 0.58333 [0.33333, 0.83333] | 0.01387 [0.00557, 0.02522] | 1.57652 [0.72986, 2.54230] |
| qa | 32.46941 [13.88228, 53.78343] | 0.91667 [0.75000, 1.00000] | 0.09284 [0.04907, 0.13823] | 1.69267 [0.64673, 2.87452] |
| pooled | 13.52433 [4.99231, 24.14189] | 0.75000 [0.55556, 0.91667] | 0.04167 [0.02272, 0.06317] | 1.61269 [0.73011, 2.60592] |

Nested shrinkage is more stable on development models: the paired MAE reduction excludes zero for math, QA, and pooled errors; code remains uncertain. It does not establish a gain over zero-change, and it does not improve on Mode A for Qwen3-8B. The near-zero QA weight on direct calibration shows how strongly the shallow observation must be discounted under this power shape.

| Capability | Shrink B improvement over zero |
|---|---:|
| math | -0.15409 [-0.77624, 0.47730] |
| code | 0.30406 [-0.29039, 0.93694] |
| qa | -0.33727 [-1.01348, 0.36880] |
| pooled | -0.06243 [-0.63739, 0.54613] |

Sensitivity is the exact multiplier from an additive error in the calibration RESPONSE to the predicted response at d=0.6. For example, multiplying the gain by 0.001 gives the prediction change caused by a 0.001-nat perturbation; this is a sensitivity experiment, not an estimated noise level.

| Capability | Direct gain at 0.6 | Shrink gain at 0.6 | Weight on direct amplitude | Direct amplitude error | Shrink amplitude error |
|---|---:|---:|---:|---:|---:|
| math | 933.99911 [558.31372, 1433.05311] | 27.43589 [2.23865, 59.45477] | 0.10149 [0.00340, 0.25821] | 5.15163 [1.33773, 11.56617] | 1.07131 [0.63305, 1.72997] |
| code | 221.35478 [158.31394, 302.81152] | 130.75927 [113.90838, 144.99033] | 0.68481 [0.58887, 0.76573] | 1.32069 [0.56836, 2.27547] | 0.96506 [0.55971, 1.50194] |
| qa | 897.06511 [728.52934, 1082.67227] | 1.99241 [1.75492, 2.29243] | 0.00287 [0.00185, 0.00439] | 19.37894 [9.27151, 30.84658] | 1.16837 [0.81512, 1.66736] |
| pooled | 684.13967 [485.47585, 942.57769] | 53.39585 [44.83814, 64.11068] | 0.26306 [0.20616, 0.33930] | 8.61709 [3.93328, 14.40884] | 1.06824 [0.72493, 1.60988] |

Amplitude errors here compare with the diagnostic best scalar fitted to target test points; these oracle labels never enter prediction. The exact per-model responses, amplitudes, selected lambda, and 0.3 extrapolation sensitivities are in `calibration.model_diagnostics`. As a separate one-point-location sensitivity, d=0.8 calibration is compared with d=0.9 on common test points 0.7/0.6:

| Capability | Models / rows | Mean absolute response | v28_B | v28_B80 | v28_B_shrink | zero |
|---|---:|---:|---:|---:|---:|---:|
| math | 12 / 24 | 2.24406 [1.04403, 3.61439] | 13.67881 [3.06908, 32.07892] | 4.54810 [0.74616, 9.46855] | 2.48238 [1.68555, 3.38429] | 2.24406 [1.04403, 3.61439] |
| code | 12 / 24 | 2.26337 [1.05659, 3.63343] | 2.63082 [1.25852, 4.53825] | 1.81277 [0.65431, 3.20442] | 1.83854 [1.14223, 2.62503] | 2.26337 [1.05659, 3.63343] |
| qa | 12 / 24 | 2.38483 [0.86452, 4.12365] | 50.51676 [23.08953, 81.86096] | 4.44152 [2.39208, 7.01161] | 2.90372 [2.08987, 3.87183] | 2.38483 [0.86452, 4.12365] |
| pooled | 12 / 72 | 2.29742 [1.03038, 3.73502] | 22.27547 [9.49469, 38.22965] | 3.60079 [1.42650, 6.25665] | 2.40821 [1.73356, 3.24534] | 2.29742 [1.03038, 3.73502] |

Only aggregate losses are saved. Gains are exact sensitivity to perturbations of delta at 0.9; residuals also contain shape error. No measurement variance or SNR can be estimated. Small response and large extrapolation gain establish poor conditioning. They do not establish that stochastic measurement error, rather than local-versus-deep shape mismatch, caused the observed failures. One-point calibration locks in shallow sign and shape deviations; shrinkage limits this amplification but may bias robust or negative-response sources toward a damaging population mean.

## Distinguishable explanations

| Candidate explanation | Isolating evidence / exclusions |
|---|---|
| Historical task was easier | Exact score reproduction, pre-cliff versus full fixed-fit scoring, and response amplitudes isolate range selection. Deep-grid extrapolation is a separate stress test. |
| Historical predictor used target compression | Same historical test rows with and without the target's shallow contributions isolate the actual benefit. Historical family holdout already forbids that access. Matched A/B isolates the separate one-point protocol. |
| Basic-input amplitude transfer is misspecified | Matched A comparison plus per-capability pooling and dev-mean ablations isolate estimation/mapping effects. Qwen oracle scaling tests whether a correct density shape could survive an amplitude correction. |
| Qwen enters rapid damage later | Predicted versus censored actual threshold crossings test this in the observed range. Timing and amplitude overlap; no unmeasured collapse density is invented. |
| Calibration measurement noise is amplified | Small saved shallow responses and exact sensitivity gains support poor conditioning; nested shrinkage tests stabilization. Aggregate JSON cannot distinguish measurement noise from deterministic shape error. |

Artifacts: `results/v29-prune-diagnosis/summary.json` contains every fold, held-out row, paired contrast, amplitude, and CI; `predictions.csv` provides matched and Qwen predictions. This report is also copied to that directory. No existing predictor, source result, paper ledger, or frozen prediction file was modified.
