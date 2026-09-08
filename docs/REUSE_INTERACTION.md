# V37b: reuse interaction and training-exposure confound

**Matched reuse does not match training exposure.** The train logs verify Gemma U75 uxseenE: 28 optimizer steps / 39,134 supervised completion tokens, versus U600 uxseen: 224 / 316,782, full-run E=2 / 2. These full-run totals are not substituted for snapshot exposure. The exact matched-E tables below quantify the same confound at each evaluated comparison.

**The prespecified one-coefficient interaction has no supported primary matched-level MAE gain for any student/capability.**

This Priority-3 CPU-only audit reuses 15 runs and 36 nonbaseline snapshots, fits signed capability loss changes separately for each student and math/code/QA, and introduces no latent variable. Positive delta means worse loss; negative means improvement. Pruning and quantization are outside this distillation panel.

## Fixed protocol and provenance

The retrospective specification was fixed before V37b fitting, with previously observed V37 residuals already known. It is not a prospective preregistration. Practical tolerance is **0.05 nat per target token**, unchanged across capabilities. All logarithms are natural; D_ref=100,000 input tokens.

**Correction to the premise:** the checked-in V37 model already used `x*z` and `x^2*z`, where x=log1p(E), z=log(D_U/100000). It was an interaction, not an additive log-D model. V37b tests a stricter one-coefficient interaction; its additive comparator is newly evaluated on identical V37b folds. Different student conditioning and folds prevent treating a change from V37 scores as an interaction-only improvement.

1. E-only: delta_hat = b1*x + b2*x² (2 parameters).
2. Interaction: delta_hat = b1*x + b2*x² + gamma*E*z (3 parameters), so k(E)=gamma*E and h(0)=k(0)=0.
3. Additive diagnostic: delta_hat = b1*x + b2*x² + a*z (3 parameters). This comparator violates the zero-reuse anchor and is not an eligible anchored response law.

Separate student/capability weighted least squares gives equal pools, then seeds, then checkpoints per pool/seed. There is no tuning, fitted offset, sign restriction, or target calibration. Four nominal E-level folds hold out all seeds and pools at matched-low (~1), matched-high (~2), U75 ~8, or U75 ~16 reuse. Actual T/D_U enters the basis. The primary score uses the 24 raw matched-level snapshots; all 36 snapshots form a secondary score. High-reuse U75 observations support the three-parameter fit when one matched level is held out. The same run can occur at another training level; this estimates held-level prediction, not new-run or new-student transfer.

Gain means MAE(E-only) minus MAE(interaction). A gain is supported when its paired 95% interval is entirely positive; a mean gain of at least 0.05 nat is practically material. Intervals are Student-t over three seed blocks (df=2), averaging levels/pools within each seed first. They condition on fixed OOF predictions, not model-refitting uncertainty. They are pointwise and do not treat checkpoints as independent replicates.

## Training-exposure confound

Input T includes prompt and completion tokens after truncation, with repetitions. D_U is one-pass input-token pool volume. Supervised S is `completion_tokens_seen`, not T. Every snapshot is checked against the matching train-log milestone and the per-update cumulative curve. No final-run total is scaled down to invent snapshot counts.

At each U600 snapshot E, U75 is evaluated at T*=E*D_U. Steps and S are linearly interpolated between adjacent logged updates, while delta is interpolated between dense/low/high saved checkpoints without extrapolation. Fractional steps describe interpolation only. The JSON retains original E/T/steps/S, interpolation brackets, signed nominal gaps, exact-E gaps and source paths for every pair.

| Student | Level | Seed | Matched E | Nominal U75 E | D_U 75 / 600 | Aligned T 75 / 600 | Steps 75 / 600 | Supervised S 75 / 600 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | matched_low | 0 | 0.96884 | 1.00000 | 65,476 / 521,533 | 63,436.0 / 505,284 | 13.52 / 108 | 17,542.6 / 143,729 |
| Qwen3-4B | matched_low | 1 | 0.96304 | 1.00000 | 65,476 / 521,533 | 63,056.1 / 502,258 | 13.41 / 107 | 17,499.3 / 141,462 |
| Qwen3-4B | matched_low | 2 | 0.96110 | 1.00000 | 65,476 / 521,533 | 62,928.7 / 501,243 | 13.55 / 107 | 17,742.4 / 142,924 |
| Qwen3-4B | matched_high | 0 | 1.91936 | 1.92900 | 65,476 / 521,533 | 125,672.1 / 1,001,010 | 26.85 / 214 | 35,347.0 / 283,935 |
| Qwen3-4B | matched_high | 1 | 1.92627 | 1.94677 | 65,476 / 521,533 | 126,124.2 / 1,004,611 | 26.77 / 215 | 35,091.0 / 284,466 |
| Qwen3-4B | matched_high | 2 | 1.92486 | 1.94703 | 65,476 / 521,533 | 126,032.0 / 1,003,877 | 26.75 / 215 | 34,877.2 / 285,752 |
| gemma3-1b | matched_low | 0 | 0.93883 | 1.00000 | 67,027 / 533,869 | 62,926.9 / 501,212 | 13.06 / 105 | 18,064.3 / 149,800 |
| gemma3-1b | matched_low | 1 | 0.93896 | 0.93710 | 67,027 / 533,869 | 62,936.0 / 501,284 | 13.03 / 104 | 18,249.3 / 146,887 |
| gemma3-1b | matched_low | 2 | 0.94051 | 1.00000 | 67,027 / 533,869 | 63,039.7 / 502,110 | 13.32 / 105 | 18,695.7 / 150,805 |
| gemma3-1b | matched_high | 0 | 1.87489 | 1.93003 | 67,027 / 533,869 | 125,668.5 / 1,000,948 | 26.16 / 209 | 36,737.1 / 295,457 |
| gemma3-1b | matched_high | 1 | 1.87817 | 1.94638 | 67,027 / 533,869 | 125,888.3 / 1,002,698 | 26.21 / 210 | 37,235.7 / 298,539 |
| gemma3-1b | matched_high | 2 | 1.87904 | 1.94572 | 67,027 / 533,869 | 125,946.6 / 1,003,163 | 26.24 / 210 | 36,631.8 / 299,120 |

| U600 / U75 at matched E | Minimum | Maximum |
|---|---:|---:|
| D_U | 7.9650 | 7.9653 |
| processed_tokens | 7.9650 | 7.9653 |
| optimizer_steps_so_far | 7.8830 | 8.0384 |
| completion_tokens_seen | 8.0176 | 8.2926 |

The maximum numerical error in **log T = log E + log D_U** is 1.78e-15. After centering within each student/seed/E pair, log T and log D_U are identical. The table gives correlations of these centered log exposures across all pairs; correlations near one measure a shared pool contrast, not independent identifying variation.

| | log D_U | log T | log steps | log S |
|---|---:|---:|---:|---:|
| log_D_U | 1.000000 | 1.000000 | 0.999996 | 0.999989 |
| log_T | 1.000000 | 1.000000 | 0.999996 | 0.999989 |
| log_steps | 0.999996 | 0.999996 | 1.000000 | 0.999991 |
| log_supervised | 0.999989 | 0.999989 | 0.999991 | 1.000000 |

The four-column centered design has rank 3; it is singular. A joint regression cannot disentangle D_U from T at fixed E, and steps/S add little independent variation.

**Schedule check (full-run settings, identical across the three seeds in each row).** All runs use AdamW, learning rate 1e-4, cosine scheduling, warmup ratio 0.03 and effective batch size 16 sequences. Warmup counts are integer-truncated. Snapshot schedule fractions and logged next-update rates are retained in JSON.

| Student | Pool / variant | Planned steps | Warmup steps | Final E | Final supervised S |
|---|---|---:|---:|---:|---:|
| gemma3-1b | 75 / uxseen | 238 | 7 | 17.0 | 332,639 |
| gemma3-1b | 600 / uxseen | 224 | 6 | 2.0 | 316,782 |
| gemma3-1b | 75 / uxseenE | 28 | 0 | 2.0 | 39,134 |
| Qwen3-4B | 75 / uxseen | 224 | 6 | 16.0 | 292,064 |
| Qwen3-4B | 600 / uxseen | 224 | 6 | 2.0 | 295,740 |

Gemma's matched-E U75 run has a 28-step horizon and zero warmup versus U600's 224 and six. Qwen shares the 224-step horizon across pools but reaches matched E at very different schedule fractions. Thus equal base learning rates do not match optimization histories. The logged rate follows scheduler.step(); it is not the rate applied to the preceding update or an integrated training dose.

## Held-out interaction results

Primary score: raw matched-level checkpoints, nats per target token. All model comparisons use the same folds and weights.

| Student | Capability | E-only MAE | Interaction MAE | Additive MAE | Gain over E-only [95% CI] | Gain over additive [95% CI] |
|---|---|---:|---:|---:|---|---|
| gemma3-1b | math | 0.0535 | 0.1061 | 0.0539 | -0.0526 [-0.0638, -0.0414] | -0.0522 [-0.0524, -0.0520] |
| gemma3-1b | code | 0.0576 | 0.0719 | 0.0270 | -0.0143 [-0.0249, -0.0037] | -0.0449 [-0.0451, -0.0447] |
| gemma3-1b | qa | 0.2885 | 0.4900 | 0.4281 | -0.2015 [-0.2600, -0.1429] | -0.0619 [-0.0628, -0.0610] |
| Qwen3-4B | math | 0.0195 | 0.0165 | 0.0276 | +0.0030 [-0.0006, +0.0066] | +0.0111 [+0.0037, +0.0184] |
| Qwen3-4B | code | 0.0571 | 0.0708 | 0.0731 | -0.0137 [-0.0207, -0.0068] | +0.0023 [+0.0022, +0.0023] |
| Qwen3-4B | qa | 0.4415 | 0.9382 | 0.7338 | -0.4967 [-0.4990, -0.4943] | -0.2044 [-0.2066, -0.2023] |

Secondary, all-level panel (includes high-E extrapolation; not substituted for the primary endpoint):

| Student | Capability | E-only MAE | Interaction MAE | Additive MAE | Gain over E-only [95% CI] |
|---|---|---:|---:|---:|---|
| gemma3-1b | math | 0.1196 | 0.1620 | 0.1008 | -0.0424 [-0.0519, -0.0329] |
| gemma3-1b | code | 0.0705 | 0.0839 | 0.0298 | -0.0134 [-0.0241, -0.0028] |
| gemma3-1b | qa | 0.4419 | 0.6405 | 0.5732 | -0.1986 [-0.2279, -0.1693] |
| Qwen3-4B | math | 0.0365 | 0.0283 | 0.0385 | +0.0081 [+0.0049, +0.0114] |
| Qwen3-4B | code | 0.0671 | 0.0870 | 0.0901 | -0.0199 [-0.0234, -0.0164] |
| Qwen3-4B | qa | 0.9727 | 1.4144 | 1.1730 | -0.4417 [-0.4442, -0.4393] |

Fold coefficients, ranks, condition numbers, per-level errors and each raw prediction are in the summary. A supported gain does not establish 0.05-nat predictive sufficiency or isolate optimizer exposure from diversity.

## Matched-E residual versus exposure

Define r_c = delta75(E) − delta600(E). The common E-only prediction cancels in this signed paired residual. Fit separately **r=beta*log(S75/S600)** and **r=beta*(steps75−steps600)/100**, each one slope with no intercept. These are differences of regressions on log cumulative supervised tokens and on optimizer steps. They predict zero gap at equal exposure. No simultaneous collinear exposure coefficients are fitted.

For each student/capability, train on one matched E level (all three seeds) and predict the other; reverse for the second fold. A one-parameter constant gap is trained on the same folds because log(S75/S600) is almost constant. Full-data uncentered R² is descriptive; held-level SSE reduction can be negative. Call a residual **largely explained** only if held-level SSE falls by at least 50% versus zero gap and both corrected-gap paired intervals lie entirely within ±0.05 nat. This is a predictive compatibility rule, not causal attribution.

Exact-E outcome interpolation itself uses observed target checkpoints, sometimes the other E level. Therefore this regression validation is a descriptive diagnostic; the raw nominal-gap sensitivity below avoids cross-level outcome interpolation but retains E mismatch. Neither is the primary raw-snapshot interaction CV.

| Student | Cap | Exposure | Full-data slope | Descriptive R² (uncentered) | Held-level SSE reduction | MAE zero → exposure | MAE gain over constant [95% CI] | Largely explained? |
|---|---|---|---:|---:|---:|---|---|---|
| gemma3-1b | math | log_supervised_ratio | +0.0413 | 0.984 | +0.978 | 0.0864 → 0.0097 | +0.0000 [-0.0006, +0.0007] | yes |
| gemma3-1b | math | optimizer_step_difference_per_100 | +0.0573 | 0.909 | +0.506 | 0.0864 → 0.0572 | -0.0475 [-0.0635, -0.0314] | no |
| gemma3-1b | code | log_supervised_ratio | +0.0535 | 0.975 | +0.914 | 0.1120 → 0.0323 | -0.0004 [-0.0019, +0.0011] | no |
| gemma3-1b | code | optimizer_step_difference_per_100 | +0.0768 | 0.963 | +0.793 | 0.1120 → 0.0484 | -0.0165 [-0.0273, -0.0057] | no |
| gemma3-1b | qa | log_supervised_ratio | +0.0744 | 0.095 | -2.240 | 0.4454 → 0.8918 | -0.0009 [-0.0066, +0.0048] | no |
| gemma3-1b | qa | optimizer_step_difference_per_100 | +0.2002 | 0.329 | -2.537 | 0.4454 → 0.8853 | +0.0055 [+0.0040, +0.0071] | no |
| Qwen3-4B | math | log_supervised_ratio | +0.0113 | 0.521 | -0.523 | 0.0286 → 0.0386 | +0.0000 [-0.0002, +0.0003] | no |
| Qwen3-4B | math | optimizer_step_difference_per_100 | +0.0192 | 0.760 | +0.185 | 0.0286 → 0.0257 | +0.0129 [+0.0128, +0.0131] | no |
| Qwen3-4B | code | log_supervised_ratio | +0.0259 | 0.328 | -1.683 | 0.0774 → 0.1548 | -0.0000 [-0.0006, +0.0006] | no |
| Qwen3-4B | code | optimizer_step_difference_per_100 | +0.0511 | 0.644 | -1.219 | 0.0774 → 0.1335 | +0.0213 [+0.0210, +0.0216] | no |
| Qwen3-4B | qa | log_supervised_ratio | +0.0361 | 0.011 | -2.693 | 0.6803 → 1.3602 | +0.0004 [-0.0011, +0.0019] | no |
| Qwen3-4B | qa | optimizer_step_difference_per_100 | +0.1934 | 0.161 | -3.793 | 0.6803 → 1.4738 | -0.1133 [-0.1156, -0.1110] | no |

Signed gaps before and after held-level exposure correction (paired 95% intervals):

| Student | Cap | Level | Raw nominal gap | Exact-E gap | Corrected: log S | Corrected: steps |
|---|---|---|---|---|---|---|
| gemma3-1b | math | matched_low | -0.0826 [-0.1147, -0.0504] | -0.0829 [-0.1147, -0.0512] | +0.0073 [-0.0237, +0.0382] | -0.0381 [-0.0700, -0.0062] |
| gemma3-1b | math | matched_high | -0.0890 [-0.1221, -0.0560] | -0.0899 [-0.1223, -0.0575] | -0.0072 [-0.0405, +0.0260] | +0.0763 [+0.0430, +0.1096] |
| gemma3-1b | code | matched_low | -0.0960 [-0.1193, -0.0727] | -0.0961 [-0.1193, -0.0729] | +0.0323 [+0.0090, +0.0557] | -0.0322 [-0.0560, -0.0084] |
| gemma3-1b | code | matched_high | -0.1275 [-0.1513, -0.1036] | -0.1280 [-0.1517, -0.1043] | -0.0323 [-0.0571, -0.0075] | +0.0645 [+0.0402, +0.0888] |
| gemma3-1b | qa | matched_low | +0.2498 [-0.1826, +0.6821] | +0.2884 [-0.1831, +0.7599] | +0.8927 [+0.4094, +1.3761] | +0.5891 [+0.1148, +1.0635] |
| gemma3-1b | qa | matched_high | -0.6190 [-1.2442, +0.0061] | -0.6025 [-1.2205, +0.0156] | -0.8908 [-1.5077, -0.2740] | -1.1815 [-1.8028, -0.5602] |
| Qwen3-4B | math | matched_low | -0.0092 [-0.0526, +0.0342] | -0.0043 [-0.0490, +0.0405] | +0.0386 [-0.0057, +0.0830] | +0.0172 [-0.0273, +0.0617] |
| Qwen3-4B | math | matched_high | -0.0432 [-0.0673, -0.0190] | -0.0429 [-0.0670, -0.0189] | -0.0386 [-0.0627, -0.0145] | -0.0343 [-0.0583, -0.0102] |
| Qwen3-4B | code | matched_low | +0.0154 [+0.0079, +0.0229] | +0.0232 [+0.0164, +0.0301] | +0.1548 [+0.1473, +0.1624] | +0.0890 [+0.0816, +0.0963] |
| Qwen3-4B | code | matched_high | -0.1337 [-0.1503, -0.1171] | -0.1315 [-0.1461, -0.1170] | -0.1548 [-0.1696, -0.1400] | -0.1781 [-0.1930, -0.1632] |
| Qwen3-4B | qa | matched_low | +0.5309 [-0.1903, +1.2520] | +0.6051 [-0.1372, +1.3473] | +1.3605 [+0.6243, +2.0966] | +0.9825 [+0.2442, +1.7208] |
| Qwen3-4B | qa | matched_high | -0.7733 [-1.3012, -0.2454] | -0.7555 [-1.2758, -0.2353] | -1.3599 [-1.8790, -0.8408] | -1.9652 [-2.4893, -1.4411] |

Raw nominal-E sensitivity (uses nominal gaps and nominal exposures consistently):

| Student | Cap | Exposure | Held-level SSE reduction | Largely explained? |
|---|---|---|---:|---|
| gemma3-1b | math | log_supervised_ratio | +0.980 | yes |
| gemma3-1b | math | optimizer_step_difference_per_100 | +0.493 | no |
| gemma3-1b | code | log_supervised_ratio | +0.920 | no |
| gemma3-1b | code | optimizer_step_difference_per_100 | +0.787 | no |
| gemma3-1b | qa | log_supervised_ratio | -2.089 | no |
| gemma3-1b | qa | optimizer_step_difference_per_100 | -2.208 | no |
| Qwen3-4B | math | log_supervised_ratio | -0.139 | no |
| Qwen3-4B | math | optimizer_step_difference_per_100 | +0.540 | no |
| Qwen3-4B | code | log_supervised_ratio | -1.426 | no |
| Qwen3-4B | code | optimizer_step_difference_per_100 | -0.868 | no |
| Qwen3-4B | qa | log_supervised_ratio | -2.602 | no |
| Qwen3-4B | qa | optimizer_step_difference_per_100 | -3.458 | no |

## Interpretation at the fixed tolerance

| Student | Capability | Interaction gain supported / material? | Exposure models meeting rule |
|---|---|---|---|
| gemma3-1b | math | no / no | log_supervised_ratio |
| gemma3-1b | code | no / no | none |
| gemma3-1b | qa | no / no | none |
| Qwen3-4B | math | no / no | none |
| Qwen3-4B | code | no / no | none |
| Qwen3-4B | qa | no / no | none |

The measured-exposure rule is met for gemma3-1b/math.
The data establish the advisor's confound: matched E varies pool volume, processed tokens, optimizer steps and supervised tokens together by about eightfold. They do not identify which of those changes causes the residual. Predictive improvement is capability-specific, and a high descriptive R² alone is insufficient. There is **no evidence here that requires a new latent capability variable**; equally, exposure-only sufficiency must not be claimed where the fixed rule fails. Measured optimization history remains an unresolved explanation.

For gemma3-1b/math, log supervised exposure removes 97.8% of held-level squared gap, but its MAE gain over a constant gap is +0.0000 [-0.0006, +0.0007] nat. This supports compatibility with an observed exposure difference, without evidence that exposure explains more than a persistent pool gap. The almost constant exposure ratio cannot distinguish those accounts.

- Only two students, two fixed nested pools, one teacher/recipe, and three shuffle/training seeds. No data-subset or evaluation-sample resampling; QA has only 251/271 measured target tokens.
- At fixed E, log T = log E + log D_U exactly. Steps and supervised tokens track T closely; regression cannot separate unique-data diversity from cumulative exposure or schedule effects.
- All E-level CV predictions condition on other saved checkpoints of the same runs. These are held-level curve checks, not independent-run, held-pool, or new-student transfer estimates.
- Matched-level interaction fits use high-reuse U75 checkpoints (E about 7.5 and 15). In Gemma these come from a different schedule horizon than uxseenE; no schedule covariate is fitted. All-level secondary scores include substantial extrapolation.
- Only two matched E levels identify the exposure diagnostic. Log supervised ratios are nearly constant; comparison with a constant gap is necessary. No ordinary six-independent-observation slope p-values are reported.
- Interpolated signed outcomes use observed target checkpoints, sometimes including the other matched level or the dense origin. Exposure cross-level validation is therefore descriptive interpolation sensitivity, not independent held-out-outcome prediction; raw nominal sensitivity is also reported.
- A low-DOF model failing does not rule out nonlinear optimization/exposure effects. A model succeeding does not establish causation; neither result demonstrates a new latent capability variable.

A discriminating follow-up would cross pool size with controlled optimizer/supervised-token budgets and schedule horizons. Because T=E*D_U, E, T and D_U cannot all be independently held fixed; the experiment must explicitly vary the chosen exposure axis, for example through batch size or supervision density. No new training is launched here.

Regenerate with NumPy and SciPy on CPU:

```bash
python analysis/v37b_reuse_interaction.py
python -m pytest -q tests/test_v37b.py tests/test_v37.py
```

Artifacts: [summary.json](../../results/v37b-reuse-interaction/summary.json), [implementation](../../analysis/v37b_reuse_interaction.py), [tests](../../tests/test_v37b.py). Prior: [V37](REUSE_SUFFICIENCY.md).
