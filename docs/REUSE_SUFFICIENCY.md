# Distillation reuse-count sufficiency (V37)

**E-only sufficiency is not established for math, code, QA at the fixed 0.05-nat tolerance.** The matched-E residual intervals fail the equivalence criterion. Whether adding unique-data volume helps prediction is assessed separately below; a smaller matched-T gap does not establish sufficiency.

This CPU-only analysis reuses 15 existing trajectories, 36 nonbaseline snapshots and 108 signed capability observations from Gemma3-1B and Qwen3-4B. No training or GPU evaluation is performed. The protocol was fixed for this retrospective analysis before fitting; the earlier residuals were already known, so this is not a prospective preregistration.

**Endpoint and fixed decision rule.** signed delta_c = post_training[c] - dense[c], legacy V12 capability loss, nats/token. Positive means loss worsens; negative means loss improves. The practical tolerance is **0.05 nat per target token** for every capability and student, the maximum accepted systematic discrepancy at matched reuse. Select E-only per capability only if (a) all four nominal matched-E **95% paired intervals lie entirely inside [-0.05,+0.05] nat**, and (b) adding log D_U does **not lower macro held-out MAE in either specified quadratic-model holdout**. Any positive reduction above 1e-12 counts as improvement; this is a conservative point-error rule, not a significance test. A CI that includes zero does not establish equivalence. The interpolation and basis checks are declared sensitivities, not tuned selection rules.

**Data and actual reuse accounting.** E = T / D_U uses snapshot `processed_tokens` (input tokens, including repetitions) and that run's `train_log.json:unique_data_pool_tokens` (one-pass selected pool input tokens). U is the requested examples **per domain**, not token volume. T=0 snapshots validate the known delta=0 anchor and are excluded from fitting/scoring. Final run-level evals are excluded. Milestones are selected by their recorded requested labels; actual T, never nominal T, enters the model. Each input JSON is hashed in the summary.

| Student | Run set | D_U | Requested T | Actual E range | Trajectories / snapshots |
|---|---|---:|---|---|---:|
| gemma3-1b | U75 uxseen | 67027 | 500000, 1000000 | 7.4996–14.9353 | 3 / 6 |
| gemma3-1b | U75 uxseenE | 67027 | 62775, 125550 | 0.9371–1.9464 | 3 / 6 |
| gemma3-1b | U600 uxseen | 533869 | 500000, 1000000 | 0.9388–1.8790 | 3 / 6 |
| Qwen3-4B | U75 uxseen | 65476 | 62800, 125500, 500000, 1000000 | 1.0000–15.3153 | 3 / 12 |
| Qwen3-4B | U600 uxseen | 521533 | 500000, 1000000 | 0.9611–1.9263 | 3 / 6 |

**Models and folds.** Write x = ln(1+E), z = ln(D_U/100000). The fixed nested candidates are:

1. E-only: δ̂_c = b₁c x + b₂c x².
2. E + log D_U: δ̂_c = b₁c x + b₂c x² + b₃c xz + b₄c x²z.

Each capability is fitted separately by weighted least squares of the **signed observed delta**, with no positivity or monotonicity constraint. Both obey h(0,D_U)=0 by the dense-baseline identity. There are no student labels, dense-loss descriptors, target offsets, target amplitude fits, hyperparameter searches or outcome-dependent knots. The same training observations/weights are used for both candidates. Cells (student,pool) have equal weight, then seeds within a cell, then checkpoints within a cell/seed. Gemma's two U75 variants form one seed block; Qwen's four U75 checkpoints form one seed block. Scoring uses this same macro weighting.

Primary holdout: four leave-one-(student,pool)-out folds: Gemma U75, Gemma U600, Qwen U75, Qwen U600. Secondary: two leave-one-student-out folds: Gemma and Qwen. Every seed, checkpoint and run variant of the held-out group is excluded from fitting. The first scheme has other-pool observations of the held-out student; only the second holds out the entire student. No target response is available to either prediction function. Fold membership, coefficients, conditioning, input-range extrapolation flags and every out-of-fold prediction are in the JSON.

**Held-out signed-response prediction.** MAE is |observed signed δ − predicted signed δ|. Reduction = MAE(E-only) − MAE(E+log D_U); positive favors including volume. Zero predicts δ=0 and has no sign. All errors and intervals are in nats/token.

| Holdout | Capability | E-only MAE | E+log D_U MAE | Zero MAE | Reduction [paired 95% CI] | Volume lowers MAE? |
|---|---|---:|---:|---:|---|---|
| student_pool | math | 0.2402 | 0.2880 | 0.1897 | -0.0478 [-0.0479, -0.0477] | no |
| student_pool | code | 0.3317 | 0.4094 | 0.2035 | -0.0776 [-0.0778, -0.0775] | no |
| student_pool | qa | 1.6218 | 1.9669 | 1.8326 | -0.3451 [-0.3458, -0.3445] | no |
| student | math | 0.2835 | 0.2863 | 0.1897 | -0.0028 [-0.0028, -0.0028] | no |
| student | code | 0.4046 | 0.4072 | 0.2035 | -0.0026 [-0.0026, -0.0026] | no |
| student | qa | 1.9563 | 1.9556 | 1.8326 | +0.0006 [+0.0003, +0.0010] | yes |

Adding log D_U worsens the primary macro MAE for all three capabilities. This lack of predictive improvement does not satisfy the separate residual-equivalence requirement.

The leave-one-student-out QA reduction is only 0.000649 nat, far below the 0.05-nat residual tolerance. It counts under the specified point-error rule, but is not evidence of practically useful prediction; opposite student-level effects are visible in the fold table below.

Both candidates have higher held-out math/code MAE than the zero-change baseline under both holdout schemes. Neither candidate is validated as a useful cross-student math/code predictor by this panel.

The paired prediction intervals enumerate the 729 stratified bootstrap resamples of three seed blocks within each of two fixed students. A block keeps both pools, all checkpoints, and both candidate errors together. The intervals describe seed variation **conditional on the fitted out-of-fold predictions**, not refitting uncertainty or generalization to new students/pools. Per-fold/per-student n=3 paired t intervals are also stored. They should not be interpreted as independent folds or an n=36 uncertainty estimate.

| Holdout | Held-out group | Capability | E-only MAE | E+log D_U MAE | Reduction |
|---|---|---|---:|---:|---:|
| student_pool | Qwen3-4B / 75 | math | 0.3542 | 0.3496 | +0.0046 |
| student_pool | Qwen3-4B / 75 | code | 0.4633 | 0.4831 | -0.0198 |
| student_pool | Qwen3-4B / 75 | qa | 2.2066 | 2.4462 | -0.2396 |
| student_pool | Qwen3-4B / 600 | math | 0.1124 | 0.2176 | -0.1052 |
| student_pool | Qwen3-4B / 600 | code | 0.1977 | 0.3317 | -0.1340 |
| student_pool | Qwen3-4B / 600 | qa | 1.1873 | 1.5210 | -0.3337 |
| student_pool | gemma3-1b / 75 | math | 0.2900 | 0.3562 | -0.0663 |
| student_pool | gemma3-1b / 75 | code | 0.3855 | 0.4777 | -0.0922 |
| student_pool | gemma3-1b / 75 | qa | 2.1233 | 2.3242 | -0.2010 |
| student_pool | gemma3-1b / 600 | math | 0.2042 | 0.2287 | -0.0245 |
| student_pool | gemma3-1b / 600 | code | 0.2804 | 0.3449 | -0.0645 |
| student_pool | gemma3-1b / 600 | qa | 0.9699 | 1.5760 | -0.6061 |
| student | Qwen3-4B | math | 0.2796 | 0.2853 | -0.0057 |
| student | Qwen3-4B | code | 0.4058 | 0.4098 | -0.0040 |
| student | Qwen3-4B | qa | 2.0171 | 1.9899 | +0.0273 |
| student | gemma3-1b | math | 0.2873 | 0.2872 | +0.0001 |
| student | gemma3-1b | code | 0.4034 | 0.4046 | -0.0012 |
| student | gemma3-1b | qa | 1.8954 | 1.9214 | -0.0260 |

**Matched-E residuals in the same signed-loss frame.** Residual = δ(U75) − δ(U600), paired by training seed. Low/high refer to U600 T≈500k/1000k. Each interval is a two-sided 95% t interval with n=3 seeds, df=2. The two checkpoints are paired observations of each trajectory.

| Student | Pair | Math residual [95% CI] | Code residual [95% CI] | QA residual [95% CI] |
|---|---|---|---|---|
| gemma3-1b | low | -0.083 [-0.115, -0.050] | -0.096 [-0.119, -0.073] | +0.250 [-0.183, +0.682] |
| gemma3-1b | high | -0.089 [-0.122, -0.056] | -0.127 [-0.151, -0.104] | -0.619 [-1.244, +0.006] |
| Qwen3-4B | low | -0.009 [-0.053, +0.034] | +0.015 [+0.008, +0.023] | +0.531 [-0.190, +1.252] |
| Qwen3-4B | high | -0.043 [-0.067, -0.019] | -0.134 [-0.150, -0.117] | -0.773 [-1.301, -0.245] |

This recovers the previously observed Gemma v31b math/code residuals of roughly −0.08 to −0.13 nat and Qwen n=3 residuals: math −0.009/−0.043, code +0.015/−0.13, QA +0.53/−0.77 (low/high). Residual shrinkage relative to matched-T is compatible with E being an important coordinate; these magnitudes and intervals do not meet the stipulated sufficiency criterion. QA is noisy; an interval crossing zero cannot be described as a vanished effect.

| Student | Capability | Per-seed average residual [95% CI] | Per-seed mean absolute residual [95% CI] | Paired high−low [95% CI] |
|---|---|---|---|---|
| gemma3-1b | math | -0.086 [-0.118, -0.054] | +0.086 [+0.054, +0.118] | -0.006 [-0.017, +0.005] |
| gemma3-1b | code | -0.112 [-0.135, -0.089] | +0.112 [+0.089, +0.135] | -0.031 [-0.042, -0.021] |
| gemma3-1b | qa | -0.185 [-0.665, +0.295] | +0.434 [+0.192, +0.676] | -0.869 [-1.353, -0.385] |
| Qwen3-4B | math | -0.026 [-0.057, +0.005] | +0.029 [+0.012, +0.047] | -0.034 [-0.067, -0.001] |
| Qwen3-4B | code | -0.059 [-0.071, -0.048] | +0.075 [+0.069, +0.080] | -0.149 [-0.160, -0.138] |
| Qwen3-4B | qa | -0.121 [-0.658, +0.415] | +0.652 [+0.318, +0.986] | -1.304 [-1.971, -0.637] |

Every combined interval above still uses **three** seed values. The mean absolute residual prevents opposite low/high signs (especially QA) from cancelling. Symmetric t intervals for magnitudes can extend below zero at n=3; these intervals are not clipped.

**Actual-E mismatch and interpolation sensitivity.** Optimizer-update overshoot means the nominal pairs do not have exactly equal E. No snapshot is relabelled with a nominal processed-token value.

| Student | Pair | U75 actual E range | U600 actual E range | Maximum relative E mismatch |
|---|---|---|---|---:|
| gemma3-1b | low | 0.9371–1.0000 | 0.9388–0.9405 | 6.52% |
| gemma3-1b | high | 1.9300–1.9464 | 1.8749–1.8790 | 3.63% |
| Qwen3-4B | low | 1.0000–1.0000 | 0.9611–0.9688 | 4.05% |
| Qwen3-4B | high | 1.9290–1.9470 | 1.9194–1.9263 | 1.15% |

For a declared sensitivity, U75 signed responses are linearly interpolated to each seed's actual U600 E using the same U75 trajectory's dense (E=0,δ=0), low and high checkpoints. There is no extrapolation or vertical shift. This is a descriptive comparison using measured target responses, **not a held-out prediction result**; interpolation imposes local linearity.

| Student | Pair | Interpolated math residual [95% CI] | Interpolated code residual [95% CI] | Interpolated QA residual [95% CI] |
|---|---|---|---|---|
| gemma3-1b | low | -0.083 [-0.115, -0.051] | -0.096 [-0.119, -0.073] | +0.288 [-0.183, +0.760] |
| gemma3-1b | high | -0.090 [-0.122, -0.058] | -0.128 [-0.152, -0.104] | -0.602 [-1.221, +0.016] |
| Qwen3-4B | low | -0.004 [-0.049, +0.040] | +0.023 [+0.016, +0.030] | +0.605 [-0.137, +1.347] |
| Qwen3-4B | high | -0.043 [-0.067, -0.019] | -0.132 [-0.146, -0.117] | -0.756 [-1.276, -0.235] |

**Fixed basis sensitivity.** Replace [x,x²] with the continuous linear-spline basis [x,max(x−ln2,0),max(x−ln3,0)] (fixed E knots 1 and 2), and append the same basis times z for the volume model. This gives 3 versus 6 coefficients, keeps the zero anchor and signed endpoint, and changes no folds or weighting. These results are not used to pick a favorable basis.

| Holdout | Capability | E-only MAE | E+log D_U MAE | Reduction |
|---|---|---:|---:|---:|
| student_pool | all | unavailable | unavailable | rank-deficient design |
| student | all | unavailable | unavailable | rank-deficient design |

A rank-deficient spline design is reported as unavailable for the entire holdout scheme, without a partial-fold average or arbitrary minimum-norm predictions. In particular, U600 has no E>2 measurements to identify a separate high-reuse spline slope in a one-student training fold. The two quadratic candidates above remain identifiable; the data do not support unrestricted response-surface comparisons.

**Decisions and scope.**

| Capability | All raw matched-E intervals within ±0.05? | All interpolated intervals within ±0.05? | No volume improvement in either primary-model holdout? | Select E-only? |
|---|---|---|---|---|
| math | no | no | yes | no |
| code | no | no | yes | no |
| qa | no | no | no | no |

Failure to select E-only does not establish that the volume model is sufficient. The models predict sign directly; low-E Gemma math/code responses are generally positive while Qwen responses are negative, and high reuse can change sign. Both students' U600 QA responses are negative. A curve alignment using a target-derived offset would incur a **calibration cost** and could not be called zero-calibration transfer; no such offset is used here.

- Only two students and two pools, one teacher/recipe. D_U is nearly collinear with pool and differs slightly by student tokenizer; no causal isolation of volume or broad student-generalization CI.
- All seeds use first n_per_domain rows of the same fixed pool. Intervals reflect training/shuffle randomness, not independent data-subset resampling or new evaluation samples.
- Nominal matched-E checkpoints overshoot requested tokens; actual E and interpolation sensitivity are reported. Interpolation assumes local linear response; no exact matched-E experiment is invented.
- Cosine schedule horizons differ: Gemma U75 uxseen/uxseenE plans 238/28 updates vs U600 224; Qwen U75/U600 both plan 224 but matched-E snapshots are at different schedule fractions. Matched-E residuals can include optimization-schedule effects.
- QA is noisy, with only 251 (Gemma) / 271 (Qwen) measured target tokens in the legacy loss. Crossing zero is not evidence of equivalence within 0.05 nat.
- Held-out gains depend on the specified response family and may mask opposite fold-level effects. Neither a worse volume model nor a shrinking pool gap proves E sufficiency; failure to select E-only does not validate the volume model.

Regenerate from the repository root (NumPy and SciPy, CPU only):

```bash
python analysis/v37_reuse_sufficiency.py
python -m pytest -q tests/test_v37.py
```

Machine-readable results: [summary.json](../../results/v37-reuse-sufficiency/summary.json). Implementation: [v37_reuse_sufficiency.py](../../analysis/v37_reuse_sufficiency.py). Earlier context: [UXSEEN_CONTROL.md](UXSEEN_CONTROL.md).
