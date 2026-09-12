Retrospective development-set analysis on already unblinded data. This is not an independent test.

**code (MBPP:e80dbf3c3adc): signal absent on leave-one-student-size-out.**
I1: zero = constant MAE 0.068729 [0.049633, 0.090800] (n=40 trajectories); 134 pairs. T_only: MAE 0.055529, gain 0.013200, width 0.032868 (fails); reuse_only: MAE 0.042318, gain 0.026411, width 0.029654 (fails); two_dimensional: MAE 0.054040, gain 0.014689, width 0.030432 (fails); saturation_p1: MAE 0.042724, gain 0.026005, width 0.029571 (fails); saturation_p2: MAE 0.034910, gain 0.033819, width 0.030580 (clears).
I2: zero = constant MAE 0.119520 [0.075791, 0.161529] (n=41 trajectories); 1163 pairs. T_only: MAE 0.119682, gain -0.000161, width 0.002229 (fails); reuse_only: MAE 0.066719, gain 0.052801, width 0.041968 (clears); two_dimensional: MAE 0.084208, gain 0.035313, width 0.022438 (clears); saturation_p1: MAE 0.072095, gain 0.047426, width 0.040577 (clears); saturation_p2: MAE 0.062941, gain 0.056579, width 0.057124 (fails).
Same-candidate, both-target passes: none.

**math (MATH-500:3816ece4b994): signal absent on leave-one-student-size-out.**
I1: zero = constant MAE 0.068018 [0.039571, 0.101299] (n=40 trajectories); 134 pairs. T_only: MAE 0.076455, gain -0.008436, width 0.031734 (fails); reuse_only: MAE 0.067577, gain 0.000442, width 0.026232 (fails); two_dimensional: MAE 0.074474, gain -0.006456, width 0.030265 (fails); saturation_p1: MAE 0.066808, gain 0.001210, width 0.025237 (fails); saturation_p2: MAE 0.066257, gain 0.001761, width 0.035360 (fails).
I2: zero = constant MAE 0.135263 [0.075493, 0.200719] (n=41 trajectories); 1163 pairs. T_only: MAE 0.135600, gain -0.000337, width 0.001991 (fails); reuse_only: MAE 0.099276, gain 0.035986, width 0.043820 (fails); two_dimensional: MAE 0.109509, gain 0.025753, width 0.034171 (fails); saturation_p1: MAE 0.117923, gain 0.017340, width 0.081841 (fails); saturation_p2: MAE 0.108888, gain 0.026375, width 0.103405 (fails).
Same-candidate, both-target passes: none.

**qa (2WikiMultihopQA:075005b82489): signal present on leave-one-student-size-out.**
I1: zero = constant MAE 0.856061 [0.606933, 1.124245] (n=40 trajectories); 134 pairs. T_only: MAE 0.787390, gain 0.068671, width 0.033272 (clears); reuse_only: MAE 0.725494, gain 0.130567, width 0.050314 (clears); two_dimensional: MAE 0.655652, gain 0.200409, width 0.143715 (clears); saturation_p1: MAE 0.557511, gain 0.298550, width 0.387835 (fails); saturation_p2: MAE 0.440145, gain 0.415916, width 0.397704 (clears).
I2: zero = constant MAE 1.706543 [1.119982, 2.248909] (n=41 trajectories); 1163 pairs. T_only: MAE 1.707224, gain -0.000681, width 0.003031 (fails); reuse_only: MAE 1.485954, gain 0.220589, width 0.149666 (clears); two_dimensional: MAE 1.179629, gain 0.526913, width 0.392352 (clears); saturation_p1: MAE 0.996798, gain 0.709745, width 0.796442 (fails); saturation_p2: MAE 0.855159, gain 0.851384, width 1.083827 (fails).
Same-candidate, both-target passes: reuse_only, two_dimensional.

## Retrospective development-set tables

A capability has signal only if the SAME fixed candidate beats BOTH zero and the per-capability constant for BOTH I1 and I2 on leave-one-student-size-out. For each baseline and target, improvement = baseline MAE - candidate MAE must be strictly greater than the full width (upper minus lower) of its paired 95% trajectory-bootstrap improvement interval. Otherwise signal is absent. Target-specific verdicts are also reported. No candidate is selected or refit after comparison.

This is a retrospective development-set analysis on already unblinded data, not an independent test. There are only three observed student sizes; trajectory intervals do not establish uncertainty over a population of sizes.
Each fit is per capability and evaluation distribution. Ordinary least squares fits delta, giving each observed checkpoint equal weight, including update 0 when observed. No intervention difference is used for fitting, and no average delta-fit score is evaluated. All seven candidates are retained. Coefficients are unconstrained real numbers; tau alone is positive and profiled on the observed training T range. T_ref = D_ref = 100000.
Delta = checkpoint loss minus that trajectory's own update-0 loss, or its own dense evaluation if update 0 is absent. T comes from actual cumulative supervised completion counts in train_log.json, checked against per-update counts where available. D_U is read from the matching U and data_seed in the V47 register; E = T/D_U. Planned counts are never substituted.
I1 retains every ordered positive-budget checkpoint pair with T2/T1 in [1.7,2.3]. I2 retains every unordered trajectory/checkpoint pair of the same student and distribution with different D_U and max(T1,T2)/min(T1,T2) <= 1.10, oriented from smaller to larger D_U. Both predictions use the actual endpoint T (no interpolation). Zero-budget pairs have no defined relative T match and are excluded from both targets. All matches count; nearby final and milestone checkpoints can both participate.
I2 includes changes in completion-token pool size caused by changing seeds even at the same nominal U. Different schedules, training seeds, and registered run generations are retained as requested; these retrospective contrasts can include their effects and the allowed budget mismatch, so they do not isolate a randomized causal effect of pool size.
Leave-one-student-size-out fits to the other students and predicts both endpoints with the same fitted function. Leave-one-pool-seed-out fits only the same student's other data seeds; all runs and pool sizes with the held seed are withheld together. I2 is scored there only when both trajectories share the held-out data seed. Cross-seed I2 pairs are explicitly counted as unevaluable on that split. A separately cross-fitted endpoint from another seed fold is never substituted. The zero and per-capability constant therefore have identical intervention predictions.
Reported overall MAE and bias weight each usable intervention pair equally, with per-student results alongside. Metric intervals resample whole contributing trajectory directories with replacement and retain all their checkpoints. I2 pairs are reconstructed using both endpoint multiplicities; they are never assigned to just one endpoint cluster. The same draws are shared across candidates and baseline differences. Any graph-empty resample is redrawn and counted.
Each per-trajectory row describes all incident pairs; I2 appears in both endpoint rows but only once in overall metrics. Trajectories with no pair on a split/target have zero counts in trajectory_pair_coverage in JSON. Repeated data/training seeds and shared pools can leave dependence between trajectories beyond the requested bootstrap unit; 95% intervals are conditional descriptive uncertainty, especially limited below six trajectories.

Inventory: 89 run directories; 41 included; 48 dropped. All previously named development, test, confirmation, repeat, and throughput runs are already unblinded here.

### Fixed candidates fitted to delta

| Candidate | Formula |
|---|---|
| zero | 0 |
| constant | a (per capability and distribution) |
| T_only | a*log(1+T/T_ref) |
| reuse_only | a*log(1+E) |
| two_dimensional | a+b*log(1+T/T_ref)+q*log(D_U/D_ref) |
| saturation_p1 | -a*(1-exp(-T/tau))+b*log(1+E) |
| saturation_p2 | -a*(1-exp(-T/tau))+b*log(1+E)**2 |

### Usable intervention pairs

Counts are per evaluation distribution and capability. Clusters are distinct trajectory directories contributing at least one pair.

| Split | Student | Capability / distribution | Target | Pairs | Clusters | Unevaluable pairs |
|---|---|---|---|---:|---:|---:|
| all_usable_pairs | gemma3-270m | code / MBPP:e80dbf3c3adc | I1 | 52 | 16 | 0 |
| leave_one_student_size_out | gemma3-270m | code / MBPP:e80dbf3c3adc | I1 | 52 | 16 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-270m | code / MBPP:e80dbf3c3adc | I1 | 52 | 16 | 0 |
| all_usable_pairs | gemma3-270m | code / MBPP:e80dbf3c3adc | I2 | 480 | 16 | 0 |
| leave_one_student_size_out | gemma3-270m | code / MBPP:e80dbf3c3adc | I2 | 480 | 16 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-270m | code / MBPP:e80dbf3c3adc | I2 | 28 | 7 | 452 |
| all_usable_pairs | gemma3-270m | math / MATH-500:3816ece4b994 | I1 | 52 | 16 | 0 |
| leave_one_student_size_out | gemma3-270m | math / MATH-500:3816ece4b994 | I1 | 52 | 16 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-270m | math / MATH-500:3816ece4b994 | I1 | 52 | 16 | 0 |
| all_usable_pairs | gemma3-270m | math / MATH-500:3816ece4b994 | I2 | 480 | 16 | 0 |
| leave_one_student_size_out | gemma3-270m | math / MATH-500:3816ece4b994 | I2 | 480 | 16 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-270m | math / MATH-500:3816ece4b994 | I2 | 28 | 7 | 452 |
| all_usable_pairs | gemma3-270m | qa / 2WikiMultihopQA:075005b82489 | I1 | 52 | 16 | 0 |
| leave_one_student_size_out | gemma3-270m | qa / 2WikiMultihopQA:075005b82489 | I1 | 52 | 16 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-270m | qa / 2WikiMultihopQA:075005b82489 | I1 | 52 | 16 | 0 |
| all_usable_pairs | gemma3-270m | qa / 2WikiMultihopQA:075005b82489 | I2 | 480 | 16 | 0 |
| leave_one_student_size_out | gemma3-270m | qa / 2WikiMultihopQA:075005b82489 | I2 | 480 | 16 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-270m | qa / 2WikiMultihopQA:075005b82489 | I2 | 28 | 7 | 452 |
| all_usable_pairs | gemma3-1b | code / MBPP:e80dbf3c3adc | I1 | 54 | 17 | 0 |
| leave_one_student_size_out | gemma3-1b | code / MBPP:e80dbf3c3adc | I1 | 54 | 17 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-1b | code / MBPP:e80dbf3c3adc | I1 | 54 | 17 | 0 |
| all_usable_pairs | gemma3-1b | code / MBPP:e80dbf3c3adc | I2 | 542 | 18 | 0 |
| leave_one_student_size_out | gemma3-1b | code / MBPP:e80dbf3c3adc | I2 | 542 | 18 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-1b | code / MBPP:e80dbf3c3adc | I2 | 38 | 8 | 504 |
| all_usable_pairs | gemma3-1b | math / MATH-500:3816ece4b994 | I1 | 54 | 17 | 0 |
| leave_one_student_size_out | gemma3-1b | math / MATH-500:3816ece4b994 | I1 | 54 | 17 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-1b | math / MATH-500:3816ece4b994 | I1 | 54 | 17 | 0 |
| all_usable_pairs | gemma3-1b | math / MATH-500:3816ece4b994 | I2 | 542 | 18 | 0 |
| leave_one_student_size_out | gemma3-1b | math / MATH-500:3816ece4b994 | I2 | 542 | 18 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-1b | math / MATH-500:3816ece4b994 | I2 | 38 | 8 | 504 |
| all_usable_pairs | gemma3-1b | qa / 2WikiMultihopQA:075005b82489 | I1 | 54 | 17 | 0 |
| leave_one_student_size_out | gemma3-1b | qa / 2WikiMultihopQA:075005b82489 | I1 | 54 | 17 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-1b | qa / 2WikiMultihopQA:075005b82489 | I1 | 54 | 17 | 0 |
| all_usable_pairs | gemma3-1b | qa / 2WikiMultihopQA:075005b82489 | I2 | 542 | 18 | 0 |
| leave_one_student_size_out | gemma3-1b | qa / 2WikiMultihopQA:075005b82489 | I2 | 542 | 18 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-1b | qa / 2WikiMultihopQA:075005b82489 | I2 | 38 | 8 | 504 |
| all_usable_pairs | gemma3-4b | code / MBPP:e80dbf3c3adc | I1 | 28 | 7 | 0 |
| leave_one_student_size_out | gemma3-4b | code / MBPP:e80dbf3c3adc | I1 | 28 | 7 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-4b | code / MBPP:e80dbf3c3adc | I1 | 28 | 7 | 0 |
| all_usable_pairs | gemma3-4b | code / MBPP:e80dbf3c3adc | I2 | 141 | 7 | 0 |
| leave_one_student_size_out | gemma3-4b | code / MBPP:e80dbf3c3adc | I2 | 141 | 7 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-4b | code / MBPP:e80dbf3c3adc | I2 | 14 | 4 | 127 |
| all_usable_pairs | gemma3-4b | math / MATH-500:3816ece4b994 | I1 | 28 | 7 | 0 |
| leave_one_student_size_out | gemma3-4b | math / MATH-500:3816ece4b994 | I1 | 28 | 7 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-4b | math / MATH-500:3816ece4b994 | I1 | 28 | 7 | 0 |
| all_usable_pairs | gemma3-4b | math / MATH-500:3816ece4b994 | I2 | 141 | 7 | 0 |
| leave_one_student_size_out | gemma3-4b | math / MATH-500:3816ece4b994 | I2 | 141 | 7 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-4b | math / MATH-500:3816ece4b994 | I2 | 14 | 4 | 127 |
| all_usable_pairs | gemma3-4b | qa / 2WikiMultihopQA:075005b82489 | I1 | 28 | 7 | 0 |
| leave_one_student_size_out | gemma3-4b | qa / 2WikiMultihopQA:075005b82489 | I1 | 28 | 7 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-4b | qa / 2WikiMultihopQA:075005b82489 | I1 | 28 | 7 | 0 |
| all_usable_pairs | gemma3-4b | qa / 2WikiMultihopQA:075005b82489 | I2 | 141 | 7 | 0 |
| leave_one_student_size_out | gemma3-4b | qa / 2WikiMultihopQA:075005b82489 | I2 | 141 | 7 | 0 |
| leave_one_pool_seed_out_within_student | gemma3-4b | qa / 2WikiMultihopQA:075005b82489 | I2 | 14 | 4 | 127 |

### Intervention MAE and signed bias

All intervals below are 95% percentile intervals with 5,000 trajectory resamples, seed 0. Bias = predicted intervention minus measured intervention. All losses and errors are nats/token. A positive paired gain means improvement. Every interval states its trajectory count. Zero and constant comparisons are both included, even though they agree algebraically.

#### leave_one_student_size_out: code / MBPP:e80dbf3c3adc — retrospective development-set

| Student | Target | Candidate | Pairs | MAE [95% CI] | Bias [95% CI] | Gain over zero [95% CI] | Gain over constant [95% CI] | Clears target rule |
|---|---|---|---:|---|---|---|---|---|
| all | I1 | zero | 134 | 0.068729 [0.049633, 0.090800] (n=40 trajectories) | -0.068106 [-0.090165, -0.048874] (n=40 trajectories) | 0.000000 [0.000000, 0.000000] (n=40 trajectories) | 0.000000 [-0.000000, 0.000000] (n=40 trajectories) | no |
| all | I1 | constant | 134 | 0.068729 [0.049633, 0.090800] (n=40 trajectories) | -0.068106 [-0.090165, -0.048874] (n=40 trajectories) | 0.000000 [-0.000000, 0.000000] (n=40 trajectories) | 0.000000 [0.000000, 0.000000] (n=40 trajectories) | no |
| all | I1 | T_only | 134 | 0.055529 [0.046803, 0.065592] (n=40 trajectories) | 0.010465 [-0.011159, 0.029455] (n=40 trajectories) | 0.013200 [-0.002766, 0.030102] (n=40 trajectories) | 0.013200 [-0.002766, 0.030102] (n=40 trajectories) | no |
| all | I1 | reuse_only | 134 | 0.042318 [0.032952, 0.053526] (n=40 trajectories) | -0.000170 [-0.017512, 0.014630] (n=40 trajectories) | 0.026411 [0.012011, 0.041664] (n=40 trajectories) | 0.026411 [0.012011, 0.041664] (n=40 trajectories) | no |
| all | I1 | two_dimensional | 134 | 0.054040 [0.044776, 0.065039] (n=40 trajectories) | 0.005739 [-0.016165, 0.024838] (n=40 trajectories) | 0.014689 [-0.000067, 0.030365] (n=40 trajectories) | 0.014689 [-0.000067, 0.030365] (n=40 trajectories) | no |
| all | I1 | saturation_p1 | 134 | 0.042724 [0.032902, 0.054516] (n=40 trajectories) | -0.000632 [-0.018169, 0.014103] (n=40 trajectories) | 0.026005 [0.011536, 0.041107] (n=40 trajectories) | 0.026005 [0.011536, 0.041107] (n=40 trajectories) | no |
| all | I1 | saturation_p2 | 134 | 0.034910 [0.027521, 0.043874] (n=40 trajectories) | 0.003224 [-0.009588, 0.013768] (n=40 trajectories) | 0.033819 [0.019138, 0.049718] (n=40 trajectories) | 0.033819 [0.019138, 0.049718] (n=40 trajectories) | yes |
| all | I2 | zero | 1163 | 0.119520 [0.075791, 0.161529] (n=41 trajectories) | 0.112726 [0.067444, 0.155261] (n=41 trajectories) | 0.000000 [0.000000, 0.000000] (n=41 trajectories) | 0.000000 [-0.000000, 0.000000] (n=41 trajectories) | no |
| all | I2 | constant | 1163 | 0.119520 [0.075791, 0.161529] (n=41 trajectories) | 0.112726 [0.067444, 0.155261] (n=41 trajectories) | 0.000000 [-0.000000, 0.000000] (n=41 trajectories) | 0.000000 [0.000000, 0.000000] (n=41 trajectories) | no |
| all | I2 | T_only | 1163 | 0.119682 [0.075468, 0.162289] (n=41 trajectories) | 0.112758 [0.066746, 0.155765] (n=41 trajectories) | -0.000161 [-0.001280, 0.000949] (n=41 trajectories) | -0.000161 [-0.001280, 0.000949] (n=41 trajectories) | no |
| all | I2 | reuse_only | 1163 | 0.066719 [0.043104, 0.093167] (n=41 trajectories) | 0.017963 [-0.009185, 0.048931] (n=41 trajectories) | 0.052801 [0.030031, 0.071999] (n=41 trajectories) | 0.052801 [0.030031, 0.071999] (n=41 trajectories) | yes |
| all | I2 | two_dimensional | 1163 | 0.084208 [0.050045, 0.120001] (n=41 trajectories) | 0.047939 [0.012646, 0.087391] (n=41 trajectories) | 0.035313 [0.023061, 0.045499] (n=41 trajectories) | 0.035313 [0.023061, 0.045499] (n=41 trajectories) | yes |
| all | I2 | saturation_p1 | 1163 | 0.072095 [0.046696, 0.099050] (n=41 trajectories) | 0.015810 [-0.018680, 0.054765] (n=41 trajectories) | 0.047426 [0.025609, 0.066186] (n=41 trajectories) | 0.047426 [0.025609, 0.066186] (n=41 trajectories) | yes |
| all | I2 | saturation_p2 | 1163 | 0.062941 [0.042219, 0.083308] (n=41 trajectories) | 0.004605 [-0.029062, 0.041228] (n=41 trajectories) | 0.056579 [0.027050, 0.084174] (n=41 trajectories) | 0.056579 [0.027050, 0.084174] (n=41 trajectories) | no |
| gemma3-1b | I1 | zero | 54 | 0.074555 [0.038912, 0.114661] (n=17 trajectories) | -0.074184 [-0.114420, -0.038466] (n=17 trajectories) | 0.000000 [0.000000, 0.000000] (n=17 trajectories) | 0.000000 [0.000000, 0.000000] (n=17 trajectories) | no |
| gemma3-1b | I1 | constant | 54 | 0.074555 [0.038912, 0.114661] (n=17 trajectories) | -0.074184 [-0.114420, -0.038466] (n=17 trajectories) | 0.000000 [0.000000, 0.000000] (n=17 trajectories) | 0.000000 [0.000000, 0.000000] (n=17 trajectories) | no |
| gemma3-1b | I1 | T_only | 54 | 0.068204 [0.052331, 0.086378] (n=17 trajectories) | 0.006808 [-0.032862, 0.041882] (n=17 trajectories) | 0.006351 [-0.018108, 0.033739] (n=17 trajectories) | 0.006351 [-0.018108, 0.033739] (n=17 trajectories) | no |
| gemma3-1b | I1 | reuse_only | 54 | 0.054497 [0.036906, 0.075319] (n=17 trajectories) | -0.006486 [-0.039948, 0.022267] (n=17 trajectories) | 0.020058 [-0.000813, 0.042884] (n=17 trajectories) | 0.020058 [-0.000813, 0.042884] (n=17 trajectories) | no |
| gemma3-1b | I1 | two_dimensional | 54 | 0.065530 [0.047509, 0.086091] (n=17 trajectories) | -0.000328 [-0.040116, 0.034819] (n=17 trajectories) | 0.009025 [-0.012754, 0.033298] (n=17 trajectories) | 0.009025 [-0.012754, 0.033298] (n=17 trajectories) | no |
| gemma3-1b | I1 | saturation_p1 | 54 | 0.054469 [0.035046, 0.077048] (n=17 trajectories) | -0.009988 [-0.044691, 0.019832] (n=17 trajectories) | 0.020086 [0.001396, 0.040559] (n=17 trajectories) | 0.020086 [0.001396, 0.040559] (n=17 trajectories) | no |
| gemma3-1b | I1 | saturation_p2 | 54 | 0.039194 [0.023446, 0.058238] (n=17 trajectories) | -0.009435 [-0.034406, 0.011649] (n=17 trajectories) | 0.035361 [0.013319, 0.059622] (n=17 trajectories) | 0.035361 [0.013319, 0.059622] (n=17 trajectories) | no |
| gemma3-1b | I2 | zero | 542 | 0.147352 [0.066077, 0.207693] (n=18 trajectories) | 0.139365 [0.053876, 0.202910] (n=18 trajectories) | 0.000000 [0.000000, 0.000000] (n=18 trajectories) | 0.000000 [-0.000000, 0.000000] (n=18 trajectories) | no |
| gemma3-1b | I2 | constant | 542 | 0.147352 [0.066077, 0.207693] (n=18 trajectories) | 0.139365 [0.053876, 0.202910] (n=18 trajectories) | 0.000000 [-0.000000, 0.000000] (n=18 trajectories) | 0.000000 [0.000000, 0.000000] (n=18 trajectories) | no |
| gemma3-1b | I2 | T_only | 542 | 0.146943 [0.064968, 0.207547] (n=18 trajectories) | 0.138942 [0.052606, 0.202527] (n=18 trajectories) | 0.000409 [-0.001261, 0.002022] (n=18 trajectories) | 0.000409 [-0.001261, 0.002022] (n=18 trajectories) | no |
| gemma3-1b | I2 | reuse_only | 542 | 0.089649 [0.044577, 0.125221] (n=18 trajectories) | 0.045562 [-0.001912, 0.089664] (n=18 trajectories) | 0.057703 [0.017888, 0.086759] (n=18 trajectories) | 0.057703 [0.017888, 0.086759] (n=18 trajectories) | no |
| gemma3-1b | I2 | two_dimensional | 542 | 0.114169 [0.050378, 0.162526] (n=18 trajectories) | 0.085661 [0.021542, 0.138909] (n=18 trajectories) | 0.033184 [0.015084, 0.048108] (n=18 trajectories) | 0.033184 [0.015084, 0.048108] (n=18 trajectories) | yes |
| gemma3-1b | I2 | saturation_p1 | 542 | 0.094346 [0.044834, 0.133712] (n=18 trajectories) | 0.059505 [0.007468, 0.105399] (n=18 trajectories) | 0.053006 [0.018850, 0.077568] (n=18 trajectories) | 0.053006 [0.018850, 0.077568] (n=18 trajectories) | no |
| gemma3-1b | I2 | saturation_p2 | 542 | 0.074839 [0.036612, 0.108025] (n=18 trajectories) | 0.048123 [0.003480, 0.087553] (n=18 trajectories) | 0.072513 [0.025665, 0.104063] (n=18 trajectories) | 0.072513 [0.025665, 0.104063] (n=18 trajectories) | no |
| gemma3-270m | I1 | zero | 52 | 0.061473 [0.043543, 0.080420] (n=16 trajectories) | -0.060274 [-0.079750, -0.041684] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | 0.000000 [-0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I1 | constant | 52 | 0.061473 [0.043543, 0.080420] (n=16 trajectories) | -0.060274 [-0.079750, -0.041684] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I1 | T_only | 52 | 0.041535 [0.033172, 0.049208] (n=16 trajectories) | 0.017691 [-0.002017, 0.036175] (n=16 trajectories) | 0.019938 [-0.003886, 0.044725] (n=16 trajectories) | 0.019938 [-0.003886, 0.044725] (n=16 trajectories) | no |
| gemma3-270m | I1 | reuse_only | 52 | 0.030858 [0.023948, 0.037582] (n=16 trajectories) | 0.011326 [-0.000495, 0.022410] (n=16 trajectories) | 0.030614 [0.007740, 0.054873] (n=16 trajectories) | 0.030614 [0.007740, 0.054873] (n=16 trajectories) | no |
| gemma3-270m | I1 | two_dimensional | 52 | 0.041070 [0.033192, 0.048203] (n=16 trajectories) | 0.014454 [-0.005248, 0.032946] (n=16 trajectories) | 0.020403 [-0.002326, 0.044123] (n=16 trajectories) | 0.020403 [-0.002326, 0.044123] (n=16 trajectories) | no |
| gemma3-270m | I1 | saturation_p1 | 52 | 0.031699 [0.023866, 0.039360] (n=16 trajectories) | 0.012542 [0.002067, 0.022283] (n=16 trajectories) | 0.029773 [0.005377, 0.055575] (n=16 trajectories) | 0.029773 [0.005377, 0.055575] (n=16 trajectories) | no |
| gemma3-270m | I1 | saturation_p2 | 52 | 0.036342 [0.030495, 0.041201] (n=16 trajectories) | 0.021255 [0.014917, 0.027468] (n=16 trajectories) | 0.025130 [0.006467, 0.045160] (n=16 trajectories) | 0.025130 [0.006467, 0.045160] (n=16 trajectories) | no |
| gemma3-270m | I2 | zero | 480 | 0.084923 [0.041556, 0.111446] (n=16 trajectories) | 0.081837 [0.036316, 0.109263] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | 0.000000 [-0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I2 | constant | 480 | 0.084923 [0.041556, 0.111446] (n=16 trajectories) | 0.081837 [0.036316, 0.109263] (n=16 trajectories) | 0.000000 [-0.000000, 0.000000] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I2 | T_only | 480 | 0.085652 [0.041041, 0.113108] (n=16 trajectories) | 0.082259 [0.035372, 0.110768] (n=16 trajectories) | -0.000729 [-0.002266, 0.000964] (n=16 trajectories) | -0.000729 [-0.002266, 0.000964] (n=16 trajectories) | no |
| gemma3-270m | I2 | reuse_only | 480 | 0.040496 [0.029318, 0.049678] (n=16 trajectories) | -0.015532 [-0.029724, 0.001983] (n=16 trajectories) | 0.044427 [0.010911, 0.067698] (n=16 trajectories) | 0.044427 [0.010911, 0.067698] (n=16 trajectories) | no |
| gemma3-270m | I2 | two_dimensional | 480 | 0.048973 [0.027260, 0.063473] (n=16 trajectories) | 0.005878 [-0.011172, 0.025308] (n=16 trajectories) | 0.035950 [0.014055, 0.049392] (n=16 trajectories) | 0.035950 [0.014055, 0.049392] (n=16 trajectories) | yes |
| gemma3-270m | I2 | saturation_p1 | 480 | 0.049329 [0.034747, 0.063114] (n=16 trajectories) | -0.033503 [-0.049174, -0.014494] (n=16 trajectories) | 0.035594 [0.000753, 0.061691] (n=16 trajectories) | 0.035594 [0.000753, 0.061691] (n=16 trajectories) | no |
| gemma3-270m | I2 | saturation_p2 | 480 | 0.056198 [0.035908, 0.074638] (n=16 trajectories) | -0.045802 [-0.062470, -0.026300] (n=16 trajectories) | 0.028725 [-0.005564, 0.057683] (n=16 trajectories) | 0.028725 [-0.005564, 0.057683] (n=16 trajectories) | no |
| gemma3-4b | I1 | zero | 28 | 0.070970 [0.030878, 0.118322] (n=7 trajectories) | -0.070930 [-0.118322, -0.030799] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I1 | constant | 28 | 0.070970 [0.030878, 0.118322] (n=7 trajectories) | -0.070930 [-0.118322, -0.030799] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I1 | T_only | 28 | 0.057073 [0.040975, 0.076750] (n=7 trajectories) | 0.004097 [-0.043350, 0.044402] (n=7 trajectories) | 0.013897 [-0.015175, 0.044618] (n=7 trajectories) | 0.013897 [-0.015175, 0.044618] (n=7 trajectories) | no |
| gemma3-4b | I1 | reuse_only | 28 | 0.040110 [0.020846, 0.063607] (n=7 trajectories) | -0.009339 [-0.047373, 0.021802] (n=7 trajectories) | 0.030860 [0.005179, 0.058690] (n=7 trajectories) | 0.030860 [0.005179, 0.058690] (n=7 trajectories) | no |
| gemma3-4b | I1 | two_dimensional | 28 | 0.055965 [0.038251, 0.077238] (n=7 trajectories) | 0.001257 [-0.046183, 0.041555] (n=7 trajectories) | 0.015005 [-0.012560, 0.044069] (n=7 trajectories) | 0.015005 [-0.012560, 0.044069] (n=7 trajectories) | no |
| gemma3-4b | I1 | saturation_p1 | 28 | 0.040548 [0.023367, 0.061422] (n=7 trajectories) | -0.007053 [-0.044004, 0.023094] (n=7 trajectories) | 0.030422 [0.002839, 0.060587] (n=7 trajectories) | 0.030422 [0.002839, 0.060587] (n=7 trajectories) | no |
| gemma3-4b | I1 | saturation_p2 | 28 | 0.023989 [0.015963, 0.035994] (n=7 trajectories) | -0.005850 [-0.025740, 0.009222] (n=7 trajectories) | 0.046980 [0.011926, 0.084462] (n=7 trajectories) | 0.046980 [0.011926, 0.084462] (n=7 trajectories) | no |
| gemma3-4b | I2 | zero | 141 | 0.130313 [0.028727, 0.202351] (n=7 trajectories) | 0.115477 [0.009236, 0.193320] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I2 | constant | 141 | 0.130313 [0.028727, 0.202351] (n=7 trajectories) | 0.115477 [0.009236, 0.193320] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I2 | T_only | 141 | 0.130735 [0.027634, 0.202948] (n=7 trajectories) | 0.115933 [0.007589, 0.193321] (n=7 trajectories) | -0.000422 [-0.002289, 0.001414] (n=7 trajectories) | -0.000422 [-0.002289, 0.001414] (n=7 trajectories) | no |
| gemma3-4b | I2 | reuse_only | 141 | 0.067845 [0.022323, 0.103727] (n=7 trajectories) | 0.025899 [-0.003200, 0.062843] (n=7 trajectories) | 0.062468 [0.005614, 0.103299] (n=7 trajectories) | 0.062468 [0.005614, 0.103299] (n=7 trajectories) | no |
| gemma3-4b | I2 | two_dimensional | 141 | 0.088989 [0.022910, 0.136825] (n=7 trajectories) | 0.046119 [-0.001242, 0.091710] (n=7 trajectories) | 0.041325 [0.005006, 0.068826] (n=7 trajectories) | 0.041325 [0.005006, 0.068826] (n=7 trajectories) | no |
| gemma3-4b | I2 | saturation_p1 | 141 | 0.064061 [0.022011, 0.097830] (n=7 trajectories) | 0.015720 [-0.008620, 0.047711] (n=7 trajectories) | 0.066252 [0.005670, 0.109679] (n=7 trajectories) | 0.066252 [0.005670, 0.109679] (n=7 trajectories) | no |
| gemma3-4b | I2 | saturation_p2 | 141 | 0.040161 [0.020270, 0.059818] (n=7 trajectories) | 0.008920 [-0.017466, 0.037689] (n=7 trajectories) | 0.090153 [0.005469, 0.148412] (n=7 trajectories) | 0.090153 [0.005469, 0.148412] (n=7 trajectories) | no |

#### leave_one_student_size_out: math / MATH-500:3816ece4b994 — retrospective development-set

| Student | Target | Candidate | Pairs | MAE [95% CI] | Bias [95% CI] | Gain over zero [95% CI] | Gain over constant [95% CI] | Clears target rule |
|---|---|---|---:|---|---|---|---|---|
| all | I1 | zero | 134 | 0.068018 [0.039571, 0.101299] (n=40 trajectories) | -0.063012 [-0.097100, -0.033662] (n=40 trajectories) | 0.000000 [0.000000, 0.000000] (n=40 trajectories) | 0.000000 [-0.000000, 0.000000] (n=40 trajectories) | no |
| all | I1 | constant | 134 | 0.068018 [0.039571, 0.101299] (n=40 trajectories) | -0.063012 [-0.097100, -0.033662] (n=40 trajectories) | 0.000000 [-0.000000, 0.000000] (n=40 trajectories) | 0.000000 [0.000000, 0.000000] (n=40 trajectories) | no |
| all | I1 | T_only | 134 | 0.076455 [0.056740, 0.099433] (n=40 trajectories) | 0.008180 [-0.026914, 0.038027] (n=40 trajectories) | -0.008436 [-0.023952, 0.007783] (n=40 trajectories) | -0.008436 [-0.023952, 0.007783] (n=40 trajectories) | no |
| all | I1 | reuse_only | 134 | 0.067577 [0.048158, 0.090847] (n=40 trajectories) | 0.002134 [-0.029972, 0.029249] (n=40 trajectories) | 0.000442 [-0.012431, 0.013800] (n=40 trajectories) | 0.000442 [-0.012431, 0.013800] (n=40 trajectories) | no |
| all | I1 | two_dimensional | 134 | 0.074474 [0.053884, 0.098241] (n=40 trajectories) | 0.004149 [-0.031183, 0.034086] (n=40 trajectories) | -0.006456 [-0.021306, 0.008960] (n=40 trajectories) | -0.006456 [-0.021306, 0.008960] (n=40 trajectories) | no |
| all | I1 | saturation_p1 | 134 | 0.066808 [0.047306, 0.089831] (n=40 trajectories) | 0.002669 [-0.029131, 0.029281] (n=40 trajectories) | 0.001210 [-0.011118, 0.014119] (n=40 trajectories) | 0.001210 [-0.011118, 0.014119] (n=40 trajectories) | no |
| all | I1 | saturation_p2 | 134 | 0.066257 [0.048059, 0.086533] (n=40 trajectories) | 0.017674 [-0.010583, 0.042523] (n=40 trajectories) | 0.001761 [-0.014919, 0.020440] (n=40 trajectories) | 0.001761 [-0.014919, 0.020440] (n=40 trajectories) | no |
| all | I2 | zero | 1163 | 0.135263 [0.075493, 0.200719] (n=41 trajectories) | 0.129470 [0.070660, 0.192880] (n=41 trajectories) | 0.000000 [0.000000, 0.000000] (n=41 trajectories) | 0.000000 [-0.000000, 0.000000] (n=41 trajectories) | no |
| all | I2 | constant | 1163 | 0.135263 [0.075493, 0.200719] (n=41 trajectories) | 0.129470 [0.070660, 0.192880] (n=41 trajectories) | 0.000000 [-0.000000, 0.000000] (n=41 trajectories) | 0.000000 [0.000000, 0.000000] (n=41 trajectories) | no |
| all | I2 | T_only | 1163 | 0.135600 [0.075472, 0.201331] (n=41 trajectories) | 0.129535 [0.070131, 0.193251] (n=41 trajectories) | -0.000337 [-0.001344, 0.000647] (n=41 trajectories) | -0.000337 [-0.001344, 0.000647] (n=41 trajectories) | no |
| all | I2 | reuse_only | 1163 | 0.099276 [0.057227, 0.147338] (n=41 trajectories) | 0.037818 [-0.016049, 0.098701] (n=41 trajectories) | 0.035986 [0.013960, 0.057780] (n=41 trajectories) | 0.035986 [0.013960, 0.057780] (n=41 trajectories) | no |
| all | I2 | two_dimensional | 1163 | 0.109509 [0.061872, 0.162782] (n=41 trajectories) | 0.045135 [-0.014189, 0.110197] (n=41 trajectories) | 0.025753 [0.008517, 0.042689] (n=41 trajectories) | 0.025753 [0.008517, 0.042689] (n=41 trajectories) | no |
| all | I2 | saturation_p1 | 1163 | 0.117923 [0.076547, 0.156379] (n=41 trajectories) | -0.000793 [-0.065902, 0.066528] (n=41 trajectories) | 0.017340 [-0.024261, 0.057579] (n=41 trajectories) | 0.017340 [-0.024261, 0.057579] (n=41 trajectories) | no |
| all | I2 | saturation_p2 | 1163 | 0.108888 [0.070152, 0.145404] (n=41 trajectories) | -0.008946 [-0.075195, 0.057416] (n=41 trajectories) | 0.026375 [-0.025799, 0.077606] (n=41 trajectories) | 0.026375 [-0.025799, 0.077606] (n=41 trajectories) | no |
| gemma3-1b | I1 | zero | 54 | 0.078280 [0.034358, 0.127393] (n=17 trajectories) | -0.074757 [-0.125014, -0.029748] (n=17 trajectories) | 0.000000 [0.000000, 0.000000] (n=17 trajectories) | 0.000000 [0.000000, 0.000000] (n=17 trajectories) | no |
| gemma3-1b | I1 | constant | 54 | 0.078280 [0.034358, 0.127393] (n=17 trajectories) | -0.074757 [-0.125014, -0.029748] (n=17 trajectories) | 0.000000 [-0.000000, 0.000000] (n=17 trajectories) | 0.000000 [0.000000, 0.000000] (n=17 trajectories) | no |
| gemma3-1b | I1 | T_only | 54 | 0.079938 [0.052657, 0.110284] (n=17 trajectories) | -0.007878 [-0.057531, 0.036742] (n=17 trajectories) | -0.001658 [-0.023049, 0.021927] (n=17 trajectories) | -0.001658 [-0.023049, 0.021927] (n=17 trajectories) | no |
| gemma3-1b | I1 | reuse_only | 54 | 0.070320 [0.041049, 0.103831] (n=17 trajectories) | -0.015540 [-0.059745, 0.023672] (n=17 trajectories) | 0.007960 [-0.009249, 0.027148] (n=17 trajectories) | 0.007960 [-0.009249, 0.027148] (n=17 trajectories) | no |
| gemma3-1b | I1 | two_dimensional | 54 | 0.076998 [0.047903, 0.109730] (n=17 trajectories) | -0.014502 [-0.064202, 0.030146] (n=17 trajectories) | 0.001282 [-0.017960, 0.022621] (n=17 trajectories) | 0.001282 [-0.017960, 0.022621] (n=17 trajectories) | no |
| gemma3-1b | I1 | saturation_p1 | 54 | 0.066360 [0.036435, 0.100620] (n=17 trajectories) | -0.018718 [-0.061177, 0.018854] (n=17 trajectories) | 0.011921 [-0.004406, 0.029992] (n=17 trajectories) | 0.011921 [-0.004406, 0.029992] (n=17 trajectories) | no |
| gemma3-1b | I1 | saturation_p2 | 54 | 0.051777 [0.030588, 0.076865] (n=17 trajectories) | -0.007108 [-0.038297, 0.019501] (n=17 trajectories) | 0.026504 [0.002112, 0.053695] (n=17 trajectories) | 0.026504 [0.002112, 0.053695] (n=17 trajectories) | no |
| gemma3-1b | I2 | zero | 542 | 0.168267 [0.069273, 0.238058] (n=18 trajectories) | 0.161930 [0.062807, 0.233060] (n=18 trajectories) | 0.000000 [0.000000, 0.000000] (n=18 trajectories) | 0.000000 [-0.000000, 0.000000] (n=18 trajectories) | no |
| gemma3-1b | I2 | constant | 542 | 0.168267 [0.069273, 0.238058] (n=18 trajectories) | 0.161930 [0.062807, 0.233060] (n=18 trajectories) | 0.000000 [-0.000000, 0.000000] (n=18 trajectories) | 0.000000 [0.000000, 0.000000] (n=18 trajectories) | no |
| gemma3-1b | I2 | T_only | 542 | 0.168056 [0.068533, 0.238828] (n=18 trajectories) | 0.161581 [0.061853, 0.233312] (n=18 trajectories) | 0.000211 [-0.001107, 0.001487] (n=18 trajectories) | 0.000211 [-0.001107, 0.001487] (n=18 trajectories) | no |
| gemma3-1b | I2 | reuse_only | 542 | 0.111225 [0.040569, 0.165767] (n=18 trajectories) | 0.079879 [0.015371, 0.135874] (n=18 trajectories) | 0.057041 [0.026405, 0.079772] (n=18 trajectories) | 0.057041 [0.026405, 0.079772] (n=18 trajectories) | yes |
| gemma3-1b | I2 | two_dimensional | 542 | 0.126304 [0.044742, 0.186500] (n=18 trajectories) | 0.095048 [0.023648, 0.154559] (n=18 trajectories) | 0.041963 [0.021582, 0.059650] (n=18 trajectories) | 0.041963 [0.021582, 0.059650] (n=18 trajectories) | yes |
| gemma3-1b | I2 | saturation_p1 | 542 | 0.104134 [0.042229, 0.153223] (n=18 trajectories) | 0.056781 [0.002264, 0.109073] (n=18 trajectories) | 0.064133 [0.024542, 0.092678] (n=18 trajectories) | 0.064133 [0.024542, 0.092678] (n=18 trajectories) | no |
| gemma3-1b | I2 | saturation_p2 | 542 | 0.080530 [0.032771, 0.123980] (n=18 trajectories) | 0.048746 [0.000683, 0.095908] (n=18 trajectories) | 0.087737 [0.034353, 0.123575] (n=18 trajectories) | 0.087737 [0.034353, 0.123575] (n=18 trajectories) | no |
| gemma3-270m | I1 | zero | 52 | 0.035416 [0.022230, 0.049120] (n=16 trajectories) | -0.026576 [-0.043386, -0.010746] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I1 | constant | 52 | 0.035416 [0.022230, 0.049120] (n=16 trajectories) | -0.026576 [-0.043386, -0.010746] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I1 | T_only | 52 | 0.059182 [0.045511, 0.072228] (n=16 trajectories) | 0.055537 [0.038722, 0.071408] (n=16 trajectories) | -0.023766 [-0.049221, 0.002824] (n=16 trajectories) | -0.023766 [-0.049221, 0.002824] (n=16 trajectories) | no |
| gemma3-270m | I1 | reuse_only | 52 | 0.053909 [0.047057, 0.060580] (n=16 trajectories) | 0.052882 [0.044430, 0.060475] (n=16 trajectories) | -0.018492 [-0.038087, 0.001807] (n=16 trajectories) | -0.018492 [-0.038087, 0.001807] (n=16 trajectories) | no |
| gemma3-270m | I1 | two_dimensional | 52 | 0.058004 [0.044463, 0.070841] (n=16 trajectories) | 0.054053 [0.037263, 0.069932] (n=16 trajectories) | -0.022588 [-0.047798, 0.003737] (n=16 trajectories) | -0.022588 [-0.047798, 0.003737] (n=16 trajectories) | no |
| gemma3-270m | I1 | saturation_p1 | 52 | 0.057625 [0.053010, 0.061013] (n=16 trajectories) | 0.057625 [0.053010, 0.061013] (n=16 trajectories) | -0.022209 [-0.038035, -0.005081] (n=16 trajectories) | -0.022209 [-0.038035, -0.005081] (n=16 trajectories) | no |
| gemma3-270m | I1 | saturation_p2 | 52 | 0.079779 [0.063759, 0.096855] (n=16 trajectories) | 0.079779 [0.063759, 0.096855] (n=16 trajectories) | -0.044363 [-0.050759, -0.037993] (n=16 trajectories) | -0.044363 [-0.050759, -0.037993] (n=16 trajectories) | no |
| gemma3-270m | I2 | zero | 480 | 0.063597 [0.030568, 0.084939] (n=16 trajectories) | 0.060506 [0.027973, 0.081883] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | 0.000000 [-0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I2 | constant | 480 | 0.063597 [0.030568, 0.084939] (n=16 trajectories) | 0.060506 [0.027973, 0.081883] (n=16 trajectories) | 0.000000 [-0.000000, 0.000000] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I2 | T_only | 480 | 0.064469 [0.030024, 0.086792] (n=16 trajectories) | 0.060950 [0.026755, 0.083675] (n=16 trajectories) | -0.000872 [-0.002434, 0.000856] (n=16 trajectories) | -0.000872 [-0.002434, 0.000856] (n=16 trajectories) | no |
| gemma3-270m | I2 | reuse_only | 480 | 0.056234 [0.035234, 0.071832] (n=16 trajectories) | -0.047549 [-0.066201, -0.023190] (n=16 trajectories) | 0.007363 [-0.012156, 0.029393] (n=16 trajectories) | 0.007363 [-0.012156, 0.029393] (n=16 trajectories) | no |
| gemma3-270m | I2 | two_dimensional | 480 | 0.059575 [0.035890, 0.076997] (n=16 trajectories) | -0.049090 [-0.070016, -0.021663] (n=16 trajectories) | 0.004022 [-0.014866, 0.026411] (n=16 trajectories) | 0.004022 [-0.014866, 0.026411] (n=16 trajectories) | no |
| gemma3-270m | I2 | saturation_p1 | 480 | 0.113871 [0.067863, 0.144952] (n=16 trajectories) | -0.107492 [-0.139528, -0.060219] (n=16 trajectories) | -0.050274 [-0.073928, -0.019077] (n=16 trajectories) | -0.050274 [-0.073928, -0.019077] (n=16 trajectories) | no |
| gemma3-270m | I2 | saturation_p2 | 480 | 0.124456 [0.072352, 0.156404] (n=16 trajectories) | -0.117794 [-0.150342, -0.066242] (n=16 trajectories) | -0.060859 [-0.085098, -0.032130] (n=16 trajectories) | -0.060859 [-0.085098, -0.032130] (n=16 trajectories) | no |
| gemma3-4b | I1 | zero | 28 | 0.108774 [0.021846, 0.214535] (n=7 trajectories) | -0.108025 [-0.213985, -0.020679] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I1 | constant | 28 | 0.108774 [0.021846, 0.214535] (n=7 trajectories) | -0.108025 [-0.213985, -0.020679] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I1 | T_only | 28 | 0.101814 [0.035485, 0.182604] (n=7 trajectories) | -0.048798 [-0.154795, 0.038671] (n=7 trajectories) | 0.006961 [-0.017279, 0.032301] (n=7 trajectories) | 0.006961 [-0.017279, 0.032301] (n=7 trajectories) | no |
| gemma3-4b | I1 | reuse_only | 28 | 0.087670 [0.019429, 0.171254] (n=7 trajectories) | -0.058027 [-0.157415, 0.021860] (n=7 trajectories) | 0.021104 [-0.000627, 0.044041] (n=7 trajectories) | 0.021104 [-0.000627, 0.044041] (n=7 trajectories) | no |
| gemma3-4b | I1 | two_dimensional | 28 | 0.100194 [0.031724, 0.184202] (n=7 trajectories) | -0.052558 [-0.158553, 0.034908] (n=7 trajectories) | 0.008580 [-0.013510, 0.031759] (n=7 trajectories) | 0.008580 [-0.013510, 0.031759] (n=7 trajectories) | no |
| gemma3-4b | I1 | saturation_p1 | 28 | 0.084729 [0.017453, 0.166121] (n=7 trajectories) | -0.058148 [-0.155015, 0.018992] (n=7 trajectories) | 0.024045 [0.000952, 0.048372] (n=7 trajectories) | 0.024045 [0.000952, 0.048372] (n=7 trajectories) | no |
| gemma3-4b | I1 | saturation_p2 | 28 | 0.069071 [0.012690, 0.136631] (n=7 trajectories) | -0.049870 [-0.127498, 0.013671] (n=7 trajectories) | 0.039703 [0.006652, 0.074187] (n=7 trajectories) | 0.039703 [0.006652, 0.074187] (n=7 trajectories) | no |
| gemma3-4b | I2 | zero | 141 | 0.252364 [0.016391, 0.402493] (n=7 trajectories) | 0.239470 [0.001091, 0.391623] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I2 | constant | 141 | 0.252364 [0.016391, 0.402493] (n=7 trajectories) | 0.239470 [0.001091, 0.391623] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I2 | T_only | 141 | 0.252990 [0.016101, 0.403665] (n=7 trajectories) | 0.239830 [0.000185, 0.392721] (n=7 trajectories) | -0.000625 [-0.001933, 0.000487] (n=7 trajectories) | -0.000625 [-0.001933, 0.000487] (n=7 trajectories) | no |
| gemma3-4b | I2 | reuse_only | 141 | 0.199870 [0.014060, 0.315936] (n=7 trajectories) | 0.166753 [-0.007155, 0.280135] (n=7 trajectories) | 0.052494 [0.001488, 0.086684] (n=7 trajectories) | 0.052494 [0.001488, 0.086684] (n=7 trajectories) | no |
| gemma3-4b | I2 | two_dimensional | 141 | 0.214941 [0.014986, 0.340654] (n=7 trajectories) | 0.174038 [-0.008081, 0.293357] (n=7 trajectories) | 0.037423 [0.000779, 0.063310] (n=7 trajectories) | 0.037423 [0.000779, 0.063310] (n=7 trajectories) | no |
| gemma3-4b | I2 | saturation_p1 | 141 | 0.184718 [0.013436, 0.291840] (n=7 trajectories) | 0.141121 [-0.009660, 0.240556] (n=7 trajectories) | 0.067646 [0.001243, 0.111611] (n=7 trajectories) | 0.067646 [0.001243, 0.111611] (n=7 trajectories) | no |
| gemma3-4b | I2 | saturation_p2 | 141 | 0.164895 [0.013679, 0.262272] (n=7 trajectories) | 0.139833 [-0.006392, 0.235757] (n=7 trajectories) | 0.087469 [0.001979, 0.141654] (n=7 trajectories) | 0.087469 [0.001979, 0.141654] (n=7 trajectories) | no |

#### leave_one_student_size_out: qa / 2WikiMultihopQA:075005b82489 — retrospective development-set

| Student | Target | Candidate | Pairs | MAE [95% CI] | Bias [95% CI] | Gain over zero [95% CI] | Gain over constant [95% CI] | Clears target rule |
|---|---|---|---:|---|---|---|---|---|
| all | I1 | zero | 134 | 0.856061 [0.606933, 1.124245] (n=40 trajectories) | -0.826850 [-1.103091, -0.569582] (n=40 trajectories) | 0.000000 [0.000000, 0.000000] (n=40 trajectories) | 0.000000 [-0.000000, 0.000000] (n=40 trajectories) | no |
| all | I1 | constant | 134 | 0.856061 [0.606933, 1.124245] (n=40 trajectories) | -0.826850 [-1.103091, -0.569582] (n=40 trajectories) | 0.000000 [-0.000000, 0.000000] (n=40 trajectories) | 0.000000 [0.000000, 0.000000] (n=40 trajectories) | no |
| all | I1 | T_only | 134 | 0.787390 [0.541381, 1.054731] (n=40 trajectories) | -0.734836 [-1.014853, -0.477455] (n=40 trajectories) | 0.068671 [0.052769, 0.086041] (n=40 trajectories) | 0.068671 [0.052769, 0.086041] (n=40 trajectories) | yes |
| all | I1 | reuse_only | 134 | 0.725494 [0.494873, 0.979657] (n=40 trajectories) | -0.635001 [-0.903834, -0.389316] (n=40 trajectories) | 0.130567 [0.106450, 0.156764] (n=40 trajectories) | 0.130567 [0.106450, 0.156764] (n=40 trajectories) | yes |
| all | I1 | two_dimensional | 134 | 0.655652 [0.453016, 0.886089] (n=40 trajectories) | -0.383565 [-0.664175, -0.124531] (n=40 trajectories) | 0.200409 [0.124767, 0.268481] (n=40 trajectories) | 0.200409 [0.124767, 0.268481] (n=40 trajectories) | yes |
| all | I1 | saturation_p1 | 134 | 0.557511 [0.441400, 0.693126] (n=40 trajectories) | -0.019101 [-0.236363, 0.178171] (n=40 trajectories) | 0.298550 [0.103338, 0.491173] (n=40 trajectories) | 0.298550 [0.103338, 0.491173] (n=40 trajectories) | no |
| all | I1 | saturation_p2 | 134 | 0.440145 [0.365577, 0.526807] (n=40 trajectories) | -0.011829 [-0.165143, 0.136238] (n=40 trajectories) | 0.415916 [0.223560, 0.621263] (n=40 trajectories) | 0.415916 [0.223560, 0.621263] (n=40 trajectories) | yes |
| all | I2 | zero | 1163 | 1.706543 [1.119982, 2.248909] (n=41 trajectories) | 1.611298 [1.022181, 2.128627] (n=41 trajectories) | 0.000000 [0.000000, 0.000000] (n=41 trajectories) | 0.000000 [-0.000000, 0.000000] (n=41 trajectories) | no |
| all | I2 | constant | 1163 | 1.706543 [1.119982, 2.248909] (n=41 trajectories) | 1.611298 [1.022181, 2.128627] (n=41 trajectories) | 0.000000 [-0.000000, 0.000000] (n=41 trajectories) | 0.000000 [0.000000, 0.000000] (n=41 trajectories) | no |
| all | I2 | T_only | 1163 | 1.707224 [1.119751, 2.249571] (n=41 trajectories) | 1.611660 [1.021977, 2.129154] (n=41 trajectories) | -0.000681 [-0.002328, 0.000703] (n=41 trajectories) | -0.000681 [-0.002328, 0.000703] (n=41 trajectories) | no |
| all | I2 | reuse_only | 1163 | 1.485954 [0.929996, 2.022336] (n=41 trajectories) | 1.343926 [0.789157, 1.871305] (n=41 trajectories) | 0.220589 [0.143235, 0.292901] (n=41 trajectories) | 0.220589 [0.143235, 0.292901] (n=41 trajectories) | yes |
| all | I2 | two_dimensional | 1163 | 1.179629 [0.782726, 1.570970] (n=41 trajectories) | 0.536044 [0.076083, 1.013148] (n=41 trajectories) | 0.526913 [0.310568, 0.702919] (n=41 trajectories) | 0.526913 [0.310568, 0.702919] (n=41 trajectories) | yes |
| all | I2 | saturation_p1 | 1163 | 0.996798 [0.710876, 1.227807] (n=41 trajectories) | -0.017893 [-0.495325, 0.432138] (n=41 trajectories) | 0.709745 [0.296290, 1.092732] (n=41 trajectories) | 0.709745 [0.296290, 1.092732] (n=41 trajectories) | no |
| all | I2 | saturation_p2 | 1163 | 0.855159 [0.601015, 1.105719] (n=41 trajectories) | -0.204130 [-0.686454, 0.219094] (n=41 trajectories) | 0.851384 [0.310341, 1.394168] (n=41 trajectories) | 0.851384 [0.310341, 1.394168] (n=41 trajectories) | no |
| gemma3-1b | I1 | zero | 54 | 0.937554 [0.577239, 1.343379] (n=17 trajectories) | -0.925268 [-1.336060, -0.559609] (n=17 trajectories) | 0.000000 [0.000000, 0.000000] (n=17 trajectories) | 0.000000 [-0.000000, 0.000000] (n=17 trajectories) | no |
| gemma3-1b | I1 | constant | 54 | 0.937554 [0.577239, 1.343379] (n=17 trajectories) | -0.925268 [-1.336060, -0.559609] (n=17 trajectories) | 0.000000 [-0.000000, 0.000000] (n=17 trajectories) | 0.000000 [0.000000, 0.000000] (n=17 trajectories) | no |
| gemma3-1b | I1 | T_only | 54 | 0.911392 [0.553402, 1.316389] (n=17 trajectories) | -0.896635 [-1.307058, -0.531215] (n=17 trajectories) | 0.026162 [0.024199, 0.028207] (n=17 trajectories) | 0.026162 [0.024199, 0.028207] (n=17 trajectories) | yes |
| gemma3-1b | I1 | reuse_only | 54 | 0.825135 [0.481227, 1.214196] (n=17 trajectories) | -0.801546 [-1.199304, -0.448045] (n=17 trajectories) | 0.112420 [0.095262, 0.131295] (n=17 trajectories) | 0.112420 [0.095262, 0.131295] (n=17 trajectories) | yes |
| gemma3-1b | I1 | two_dimensional | 54 | 0.662935 [0.326048, 1.038759] (n=17 trajectories) | -0.573764 [-0.982408, -0.209707] (n=17 trajectories) | 0.274619 [0.238471, 0.309978] (n=17 trajectories) | 0.274619 [0.238471, 0.309978] (n=17 trajectories) | yes |
| gemma3-1b | I1 | saturation_p1 | 54 | 0.536709 [0.352575, 0.730609] (n=17 trajectories) | -0.253099 [-0.553427, 0.009154] (n=17 trajectories) | 0.400846 [0.190559, 0.636614] (n=17 trajectories) | 0.400846 [0.190559, 0.636614] (n=17 trajectories) | no |
| gemma3-1b | I1 | saturation_p2 | 54 | 0.387114 [0.299144, 0.488433] (n=17 trajectories) | -0.249538 [-0.400410, -0.114297] (n=17 trajectories) | 0.550441 [0.272086, 0.861710] (n=17 trajectories) | 0.550441 [0.272086, 0.861710] (n=17 trajectories) | no |
| gemma3-1b | I2 | zero | 542 | 2.021787 [1.059780, 2.658941] (n=18 trajectories) | 1.911497 [0.941121, 2.554019] (n=18 trajectories) | 0.000000 [0.000000, 0.000000] (n=18 trajectories) | 0.000000 [-0.000000, 0.000000] (n=18 trajectories) | no |
| gemma3-1b | I2 | constant | 542 | 2.021787 [1.059780, 2.658941] (n=18 trajectories) | 1.911497 [0.941121, 2.554019] (n=18 trajectories) | 0.000000 [-0.000000, 0.000000] (n=18 trajectories) | 0.000000 [0.000000, 0.000000] (n=18 trajectories) | no |
| gemma3-1b | I2 | T_only | 542 | 2.021470 [1.059643, 2.658498] (n=18 trajectories) | 1.911347 [0.940731, 2.553982] (n=18 trajectories) | 0.000317 [-0.000230, 0.000809] (n=18 trajectories) | 0.000317 [-0.000230, 0.000809] (n=18 trajectories) | no |
| gemma3-1b | I2 | reuse_only | 542 | 1.864071 [0.967992, 2.455082] (n=18 trajectories) | 1.740067 [0.843741, 2.335674] (n=18 trajectories) | 0.157716 [0.087241, 0.206965] (n=18 trajectories) | 0.157716 [0.087241, 0.206965] (n=18 trajectories) | yes |
| gemma3-1b | I2 | two_dimensional | 542 | 1.403193 [0.692530, 1.844045] (n=18 trajectories) | 1.025587 [0.425563, 1.480186] (n=18 trajectories) | 0.618594 [0.343537, 0.843879] (n=18 trajectories) | 0.618594 [0.343537, 0.843879] (n=18 trajectories) | yes |
| gemma3-1b | I2 | saturation_p1 | 542 | 0.968772 [0.498082, 1.263621] (n=18 trajectories) | 0.527743 [0.144450, 0.842050] (n=18 trajectories) | 1.053015 [0.540122, 1.439326] (n=18 trajectories) | 1.053015 [0.540122, 1.439326] (n=18 trajectories) | yes |
| gemma3-1b | I2 | saturation_p2 | 542 | 0.639318 [0.371131, 0.831012] (n=18 trajectories) | 0.338981 [0.092823, 0.537385] (n=18 trajectories) | 1.382469 [0.672072, 1.876212] (n=18 trajectories) | 1.382469 [0.672072, 1.876212] (n=18 trajectories) | yes |
| gemma3-270m | I1 | zero | 52 | 0.634605 [0.343807, 0.935531] (n=16 trajectories) | -0.599639 [-0.915598, -0.296186] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | 0.000000 [-0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I1 | constant | 52 | 0.634605 [0.343807, 0.935531] (n=16 trajectories) | -0.599639 [-0.915598, -0.296186] (n=16 trajectories) | 0.000000 [-0.000000, 0.000000] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I1 | T_only | 52 | 0.513659 [0.241971, 0.798543] (n=16 trajectories) | -0.428640 [-0.743740, -0.126680] (n=16 trajectories) | 0.120946 [0.097308, 0.141127] (n=16 trajectories) | 0.120946 [0.097308, 0.141127] (n=16 trajectories) | yes |
| gemma3-270m | I1 | reuse_only | 52 | 0.478060 [0.236437, 0.732482] (n=16 trajectories) | -0.310246 [-0.595013, -0.036231] (n=16 trajectories) | 0.156545 [0.101760, 0.209866] (n=16 trajectories) | 0.156545 [0.101760, 0.209866] (n=16 trajectories) | yes |
| gemma3-270m | I1 | two_dimensional | 52 | 0.549287 [0.381084, 0.717293] (n=16 trajectories) | -0.040229 [-0.353802, 0.258676] (n=16 trajectories) | 0.085318 [-0.064400, 0.243564] (n=16 trajectories) | 0.085318 [-0.064400, 0.243564] (n=16 trajectories) | no |
| gemma3-270m | I1 | saturation_p1 | 52 | 0.549272 [0.459115, 0.632285] (n=16 trajectories) | 0.403569 [0.232163, 0.563475] (n=16 trajectories) | 0.085333 [-0.277318, 0.462226] (n=16 trajectories) | 0.085333 [-0.277318, 0.462226] (n=16 trajectories) | no |
| gemma3-270m | I1 | saturation_p2 | 52 | 0.484464 [0.406621, 0.575220] (n=16 trajectories) | 0.428426 [0.328848, 0.540887] (n=16 trajectories) | 0.150140 [-0.098352, 0.421806] (n=16 trajectories) | 0.150140 [-0.098352, 0.421806] (n=16 trajectories) | no |
| gemma3-270m | I2 | zero | 480 | 1.184684 [0.521498, 1.594360] (n=16 trajectories) | 1.131687 [0.430920, 1.564574] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | 0.000000 [-0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I2 | constant | 480 | 1.184684 [0.521498, 1.594360] (n=16 trajectories) | 1.131687 [0.430920, 1.564574] (n=16 trajectories) | 0.000000 [-0.000000, 0.000000] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I2 | T_only | 480 | 1.186459 [0.521942, 1.598030] (n=16 trajectories) | 1.132612 [0.427987, 1.569238] (n=16 trajectories) | -0.001775 [-0.004839, 0.001742] (n=16 trajectories) | -0.001775 [-0.004839, 0.001742] (n=16 trajectories) | no |
| gemma3-270m | I2 | reuse_only | 480 | 0.882025 [0.358125, 1.226929] (n=16 trajectories) | 0.738142 [0.200291, 1.104611] (n=16 trajectories) | 0.302659 [0.155730, 0.387390] (n=16 trajectories) | 0.302659 [0.155730, 0.387390] (n=16 trajectories) | yes |
| gemma3-270m | I2 | two_dimensional | 480 | 0.788180 [0.467300, 1.005637] (n=16 trajectories) | -0.174676 [-0.415314, 0.187889] (n=16 trajectories) | 0.396504 [0.021415, 0.634537] (n=16 trajectories) | 0.396504 [0.021415, 0.634537] (n=16 trajectories) | no |
| gemma3-270m | I2 | saturation_p1 | 480 | 0.965202 [0.629404, 1.217650] (n=16 trajectories) | -0.808826 [-1.093236, -0.414050] (n=16 trajectories) | 0.219482 [-0.257300, 0.669399] (n=16 trajectories) | 0.219482 [-0.257300, 0.669399] (n=16 trajectories) | no |
| gemma3-270m | I2 | saturation_p2 | 480 | 1.103528 [0.685409, 1.407705] (n=16 trajectories) | -1.019046 [-1.305642, -0.624980] (n=16 trajectories) | 0.081156 [-0.330129, 0.520565] (n=16 trajectories) | 0.081156 [-0.330129, 0.520565] (n=16 trajectories) | no |
| gemma3-4b | I1 | zero | 28 | 1.110172 [0.489851, 1.846079] (n=7 trajectories) | -1.059011 [-1.822080, -0.414526] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I1 | constant | 28 | 1.110172 [0.489851, 1.846079] (n=7 trajectories) | -1.059011 [-1.822080, -0.414526] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I1 | T_only | 28 | 1.056603 [0.441231, 1.784402] (n=7 trajectories) | -0.991448 [-1.754687, -0.346808] (n=7 trajectories) | 0.053569 [0.047925, 0.059353] (n=7 trajectories) | 0.053569 [0.047925, 0.059353] (n=7 trajectories) | yes |
| gemma3-4b | I1 | reuse_only | 28 | 0.992852 [0.402604, 1.695842] (n=7 trajectories) | -0.916923 [-1.660643, -0.293488] (n=7 trajectories) | 0.117320 [0.086502, 0.148736] (n=7 trajectories) | 0.117320 [0.086502, 0.148736] (n=7 trajectories) | yes |
| gemma3-4b | I1 | two_dimensional | 28 | 0.839143 [0.274187, 1.521660] (n=7 trajectories) | -0.654379 [-1.418792, -0.008627] (n=7 trajectories) | 0.271029 [0.197095, 0.343836] (n=7 trajectories) | 0.271029 [0.197095, 0.343836] (n=7 trajectories) | yes |
| gemma3-4b | I1 | saturation_p1 | 28 | 0.612931 [0.211642, 1.098168] (n=7 trajectories) | -0.352777 [-0.957135, 0.142076] (n=7 trajectories) | 0.497242 [0.193361, 0.806899] (n=7 trajectories) | 0.497242 [0.193361, 0.806899] (n=7 trajectories) | no |
| gemma3-4b | I1 | saturation_p2 | 28 | 0.460114 [0.231753, 0.772150] (n=7 trajectories) | -0.371008 [-0.727654, -0.096722] (n=7 trajectories) | 0.650058 [0.228295, 1.090430] (n=7 trajectories) | 0.650058 [0.228295, 1.090430] (n=7 trajectories) | no |
| gemma3-4b | I2 | zero | 141 | 2.271291 [0.205746, 3.692607] (n=7 trajectories) | 2.090063 [0.073503, 3.604623] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I2 | constant | 141 | 2.271291 [0.205746, 3.692607] (n=7 trajectories) | 2.090063 [0.073503, 3.604623] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I2 | T_only | 141 | 2.272088 [0.205886, 3.693148] (n=7 trajectories) | 2.090474 [0.071920, 3.608639] (n=7 trajectories) | -0.000798 [-0.002333, 0.000511] (n=7 trajectories) | -0.000798 [-0.002333, 0.000511] (n=7 trajectories) | no |
| gemma3-4b | I2 | reuse_only | 141 | 2.088410 [0.205647, 3.423758] (n=7 trajectories) | 1.883411 [0.049802, 3.293876] (n=7 trajectories) | 0.182881 [0.000629, 0.298620] (n=7 trajectories) | 0.182881 [0.000629, 0.298620] (n=7 trajectories) | no |
| gemma3-4b | I2 | two_dimensional | 141 | 1.652849 [0.226563, 2.761965] (n=7 trajectories) | 1.073727 [-0.061797, 2.121306] (n=7 trajectories) | 0.618442 [-0.030223, 1.044471] (n=7 trajectories) | 0.618442 [-0.030223, 1.044471] (n=7 trajectories) | no |
| gemma3-4b | I2 | saturation_p1 | 141 | 1.212086 [0.228029, 2.092536] (n=7 trajectories) | 0.577240 [-0.092964, 1.411946] (n=7 trajectories) | 1.059204 [-0.047039, 1.765858] (n=7 trajectories) | 1.059204 [-0.047039, 1.765858] (n=7 trajectories) | no |
| gemma3-4b | I2 | saturation_p2 | 141 | 0.839332 [0.212589, 1.557323] (n=7 trajectories) | 0.482352 [-0.042844, 1.267668] (n=7 trajectories) | 1.431959 [-0.024791, 2.330549] (n=7 trajectories) | 1.431959 [-0.024791, 2.330549] (n=7 trajectories) | no |

#### leave_one_pool_seed_out_within_student: code / MBPP:e80dbf3c3adc — retrospective development-set

| Student | Target | Candidate | Pairs | MAE [95% CI] | Bias [95% CI] | Gain over zero [95% CI] | Gain over constant [95% CI] | Clears target rule |
|---|---|---|---:|---|---|---|---|---|
| all | I1 | zero | 134 | 0.068729 [0.049633, 0.090800] (n=40 trajectories) | -0.068106 [-0.090165, -0.048874] (n=40 trajectories) | 0.000000 [0.000000, 0.000000] (n=40 trajectories) | 0.000000 [-0.000000, 0.000000] (n=40 trajectories) | no |
| all | I1 | constant | 134 | 0.068729 [0.049633, 0.090800] (n=40 trajectories) | -0.068106 [-0.090165, -0.048874] (n=40 trajectories) | 0.000000 [-0.000000, 0.000000] (n=40 trajectories) | 0.000000 [0.000000, 0.000000] (n=40 trajectories) | no |
| all | I1 | T_only | 134 | 0.056585 [0.047394, 0.067340] (n=40 trajectories) | 0.007983 [-0.014428, 0.027600] (n=40 trajectories) | 0.012144 [-0.002898, 0.028346] (n=40 trajectories) | 0.012144 [-0.002898, 0.028346] (n=40 trajectories) | no |
| all | I1 | reuse_only | 134 | 0.043521 [0.034637, 0.054578] (n=40 trajectories) | -0.001491 [-0.019147, 0.013442] (n=40 trajectories) | 0.025208 [0.010868, 0.040687] (n=40 trajectories) | 0.025208 [0.010868, 0.040687] (n=40 trajectories) | no |
| all | I1 | two_dimensional | 134 | 0.055670 [0.046276, 0.067045] (n=40 trajectories) | 0.003134 [-0.019300, 0.022759] (n=40 trajectories) | 0.013059 [-0.000522, 0.027677] (n=40 trajectories) | 0.013059 [-0.000522, 0.027677] (n=40 trajectories) | no |
| all | I1 | saturation_p1 | 134 | 0.043926 [0.035508, 0.054123] (n=40 trajectories) | -0.002487 [-0.019547, 0.012348] (n=40 trajectories) | 0.024804 [0.010882, 0.039910] (n=40 trajectories) | 0.024804 [0.010882, 0.039910] (n=40 trajectories) | no |
| all | I1 | saturation_p2 | 134 | 0.029364 [0.023947, 0.035862] (n=40 trajectories) | 0.002426 [-0.007524, 0.010502] (n=40 trajectories) | 0.039366 [0.022091, 0.058063] (n=40 trajectories) | 0.039366 [0.022091, 0.058063] (n=40 trajectories) | yes |
| all | I2 | zero | 80 | 0.243439 [0.166097, 0.328564] (n=19 trajectories) | 0.241682 [0.164121, 0.328286] (n=19 trajectories) | 0.000000 [0.000000, 0.000000] (n=19 trajectories) | 0.000000 [-0.000000, 0.000000] (n=19 trajectories) | no |
| all | I2 | constant | 80 | 0.243439 [0.166097, 0.328564] (n=19 trajectories) | 0.241682 [0.164121, 0.328286] (n=19 trajectories) | 0.000000 [-0.000000, 0.000000] (n=19 trajectories) | 0.000000 [0.000000, 0.000000] (n=19 trajectories) | no |
| all | I2 | T_only | 80 | 0.242718 [0.165473, 0.326642] (n=19 trajectories) | 0.241085 [0.163632, 0.325952] (n=19 trajectories) | 0.000722 [-0.000499, 0.002541] (n=19 trajectories) | 0.000722 [-0.000499, 0.002541] (n=19 trajectories) | no |
| all | I2 | reuse_only | 80 | 0.117881 [0.061797, 0.178550] (n=19 trajectories) | 0.048267 [-0.025827, 0.133675] (n=19 trajectories) | 0.125558 [0.098571, 0.160866] (n=19 trajectories) | 0.125558 [0.098571, 0.160866] (n=19 trajectories) | yes |
| all | I2 | two_dimensional | 80 | 0.159231 [0.101596, 0.216955] (n=19 trajectories) | 0.096988 [0.032233, 0.170826] (n=19 trajectories) | 0.084209 [0.062724, 0.119683] (n=19 trajectories) | 0.084209 [0.062724, 0.119683] (n=19 trajectories) | yes |
| all | I2 | saturation_p1 | 80 | 0.109002 [0.064617, 0.153962] (n=19 trajectories) | 0.034548 [-0.026970, 0.091316] (n=19 trajectories) | 0.134438 [0.099299, 0.185914] (n=19 trajectories) | 0.134438 [0.099299, 0.185914] (n=19 trajectories) | yes |
| all | I2 | saturation_p2 | 80 | 0.066490 [0.036383, 0.097417] (n=19 trajectories) | 0.017877 [-0.038263, 0.063914] (n=19 trajectories) | 0.176949 [0.125651, 0.238288] (n=19 trajectories) | 0.176949 [0.125651, 0.238288] (n=19 trajectories) | yes |
| gemma3-1b | I1 | zero | 54 | 0.074555 [0.038912, 0.114661] (n=17 trajectories) | -0.074184 [-0.114420, -0.038466] (n=17 trajectories) | 0.000000 [0.000000, 0.000000] (n=17 trajectories) | 0.000000 [-0.000000, 0.000000] (n=17 trajectories) | no |
| gemma3-1b | I1 | constant | 54 | 0.074555 [0.038912, 0.114661] (n=17 trajectories) | -0.074184 [-0.114420, -0.038466] (n=17 trajectories) | 0.000000 [-0.000000, 0.000000] (n=17 trajectories) | 0.000000 [0.000000, 0.000000] (n=17 trajectories) | no |
| gemma3-1b | I1 | T_only | 54 | 0.065803 [0.046634, 0.088059] (n=17 trajectories) | -0.003624 [-0.044377, 0.032056] (n=17 trajectories) | 0.008751 [-0.011099, 0.031101] (n=17 trajectories) | 0.008751 [-0.011099, 0.031101] (n=17 trajectories) | no |
| gemma3-1b | I1 | reuse_only | 54 | 0.054254 [0.035001, 0.076985] (n=17 trajectories) | -0.011135 [-0.045665, 0.018654] (n=17 trajectories) | 0.020300 [0.001320, 0.041147] (n=17 trajectories) | 0.020300 [0.001320, 0.041147] (n=17 trajectories) | no |
| gemma3-1b | I1 | two_dimensional | 54 | 0.065785 [0.046272, 0.088361] (n=17 trajectories) | -0.004317 [-0.045258, 0.031411] (n=17 trajectories) | 0.008770 [-0.010732, 0.030752] (n=17 trajectories) | 0.008770 [-0.010732, 0.030752] (n=17 trajectories) | no |
| gemma3-1b | I1 | saturation_p1 | 54 | 0.054416 [0.037514, 0.074409] (n=17 trajectories) | -0.007329 [-0.039717, 0.020387] (n=17 trajectories) | 0.020139 [-0.001938, 0.044506] (n=17 trajectories) | 0.020139 [-0.001938, 0.044506] (n=17 trajectories) | no |
| gemma3-1b | I1 | saturation_p2 | 54 | 0.037022 [0.026960, 0.049177] (n=17 trajectories) | 0.002271 [-0.017609, 0.018794] (n=17 trajectories) | 0.037533 [0.008234, 0.069540] (n=17 trajectories) | 0.037533 [0.008234, 0.069540] (n=17 trajectories) | no |
| gemma3-1b | I2 | zero | 38 | 0.305401 [0.209762, 0.415491] (n=8 trajectories) | 0.303798 [0.201276, 0.415491] (n=8 trajectories) | 0.000000 [0.000000, 0.000000] (n=8 trajectories) | 0.000000 [-0.000000, 0.000000] (n=8 trajectories) | no |
| gemma3-1b | I2 | constant | 38 | 0.305401 [0.209762, 0.415491] (n=8 trajectories) | 0.303798 [0.201276, 0.415491] (n=8 trajectories) | 0.000000 [-0.000000, 0.000000] (n=8 trajectories) | 0.000000 [0.000000, 0.000000] (n=8 trajectories) | no |
| gemma3-1b | I2 | T_only | 38 | 0.304004 [0.207958, 0.410938] (n=8 trajectories) | 0.302460 [0.199573, 0.410938] (n=8 trajectories) | 0.001397 [-0.000797, 0.004554] (n=8 trajectories) | 0.001397 [-0.000797, 0.004554] (n=8 trajectories) | no |
| gemma3-1b | I2 | reuse_only | 38 | 0.170514 [0.097780, 0.216568] (n=8 trajectories) | 0.118809 [0.011204, 0.216568] (n=8 trajectories) | 0.134887 [0.107120, 0.198923] (n=8 trajectories) | 0.134887 [0.107120, 0.198923] (n=8 trajectories) | yes |
| gemma3-1b | I2 | two_dimensional | 38 | 0.205591 [0.130031, 0.252318] (n=8 trajectories) | 0.134665 [0.006068, 0.252318] (n=8 trajectories) | 0.099810 [0.071289, 0.163173] (n=8 trajectories) | 0.099810 [0.071289, 0.163173] (n=8 trajectories) | yes |
| gemma3-1b | I2 | saturation_p1 | 38 | 0.142845 [0.080041, 0.177801] (n=8 trajectories) | 0.048866 [-0.080041, 0.147078] (n=8 trajectories) | 0.162556 [0.126782, 0.249353] (n=8 trajectories) | 0.162556 [0.126782, 0.249353] (n=8 trajectories) | yes |
| gemma3-1b | I2 | saturation_p2 | 38 | 0.092586 [0.077362, 0.115771] (n=8 trajectories) | 0.030091 [-0.080678, 0.095124] (n=8 trajectories) | 0.212815 [0.129084, 0.309403] (n=8 trajectories) | 0.212815 [0.129084, 0.309403] (n=8 trajectories) | yes |
| gemma3-270m | I1 | zero | 52 | 0.061473 [0.043543, 0.080420] (n=16 trajectories) | -0.060274 [-0.079750, -0.041684] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | 0.000000 [-0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I1 | constant | 52 | 0.061473 [0.043543, 0.080420] (n=16 trajectories) | -0.060274 [-0.079750, -0.041684] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I1 | T_only | 52 | 0.041942 [0.034188, 0.048570] (n=16 trajectories) | 0.013444 [-0.006766, 0.032303] (n=16 trajectories) | 0.019531 [-0.002437, 0.042649] (n=16 trajectories) | 0.019531 [-0.002437, 0.042649] (n=16 trajectories) | no |
| gemma3-270m | I1 | reuse_only | 52 | 0.032289 [0.026252, 0.037860] (n=16 trajectories) | 0.002313 [-0.010460, 0.014447] (n=16 trajectories) | 0.029183 [0.008991, 0.050576] (n=16 trajectories) | 0.029183 [0.008991, 0.050576] (n=16 trajectories) | no |
| gemma3-270m | I1 | two_dimensional | 52 | 0.041579 [0.034122, 0.047642] (n=16 trajectories) | 0.007678 [-0.012554, 0.026665] (n=16 trajectories) | 0.019893 [-0.000118, 0.040551] (n=16 trajectories) | 0.019893 [-0.000118, 0.040551] (n=16 trajectories) | no |
| gemma3-270m | I1 | saturation_p1 | 52 | 0.033181 [0.027122, 0.038613] (n=16 trajectories) | -0.000466 [-0.015055, 0.013327] (n=16 trajectories) | 0.028292 [0.010530, 0.046937] (n=16 trajectories) | 0.028292 [0.010530, 0.046937] (n=16 trajectories) | no |
| gemma3-270m | I1 | saturation_p2 | 52 | 0.024763 [0.020054, 0.029529] (n=16 trajectories) | 0.001958 [-0.005378, 0.008267] (n=16 trajectories) | 0.036709 [0.016155, 0.058491] (n=16 trajectories) | 0.036709 [0.016155, 0.058491] (n=16 trajectories) | no |
| gemma3-270m | I2 | zero | 28 | 0.156437 [0.131157, 0.179026] (n=7 trajectories) | 0.154847 [0.129504, 0.176314] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-270m | I2 | constant | 28 | 0.156437 [0.131157, 0.179026] (n=7 trajectories) | 0.154847 [0.129504, 0.176314] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-270m | I2 | T_only | 28 | 0.156112 [0.129237, 0.179589] (n=7 trajectories) | 0.154644 [0.127692, 0.177389] (n=7 trajectories) | 0.000325 [-0.000563, 0.001919] (n=7 trajectories) | 0.000325 [-0.000563, 0.001919] (n=7 trajectories) | no |
| gemma3-270m | I2 | reuse_only | 28 | 0.056373 [0.047819, 0.063871] (n=7 trajectories) | -0.024272 [-0.053824, -0.002272] (n=7 trajectories) | 0.100064 [0.077333, 0.115155] (n=7 trajectories) | 0.100064 [0.077333, 0.115155] (n=7 trajectories) | yes |
| gemma3-270m | I2 | two_dimensional | 28 | 0.096199 [0.072924, 0.114237] (n=7 trajectories) | 0.051300 [0.015040, 0.074103] (n=7 trajectories) | 0.060238 [0.055237, 0.064789] (n=7 trajectories) | 0.060238 [0.055237, 0.064789] (n=7 trajectories) | yes |
| gemma3-270m | I2 | saturation_p1 | 28 | 0.062210 [0.039909, 0.075583] (n=7 trajectories) | 0.011838 [-0.026966, 0.034924] (n=7 trajectories) | 0.094226 [0.086944, 0.103443] (n=7 trajectories) | 0.094226 [0.086944, 0.103443] (n=7 trajectories) | yes |
| gemma3-270m | I2 | saturation_p2 | 28 | 0.032734 [0.022213, 0.038546] (n=7 trajectories) | 0.002646 [-0.035804, 0.026733] (n=7 trajectories) | 0.123703 [0.095353, 0.140481] (n=7 trajectories) | 0.123703 [0.095353, 0.140481] (n=7 trajectories) | yes |
| gemma3-4b | I1 | zero | 28 | 0.070970 [0.030878, 0.118322] (n=7 trajectories) | -0.070930 [-0.118322, -0.030799] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I1 | constant | 28 | 0.070970 [0.030878, 0.118322] (n=7 trajectories) | -0.070930 [-0.118322, -0.030799] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I1 | T_only | 28 | 0.066003 [0.057430, 0.078354] (n=7 trajectories) | 0.020227 [-0.029480, 0.061624] (n=7 trajectories) | 0.004967 [-0.031461, 0.043911] (n=7 trajectories) | 0.004967 [-0.031461, 0.043911] (n=7 trajectories) | no |
| gemma3-4b | I1 | reuse_only | 28 | 0.043679 [0.033563, 0.056699] (n=7 trajectories) | 0.010042 [-0.025175, 0.038301] (n=7 trajectories) | 0.027291 [-0.009630, 0.067078] (n=7 trajectories) | 0.027291 [-0.009630, 0.067078] (n=7 trajectories) | no |
| gemma3-4b | I1 | two_dimensional | 28 | 0.062330 [0.049562, 0.080798] (n=7 trajectories) | 0.009066 [-0.041067, 0.050810] (n=7 trajectories) | 0.008639 [-0.021288, 0.041013] (n=7 trajectories) | 0.008639 [-0.021288, 0.041013] (n=7 trajectories) | no |
| gemma3-4b | I1 | saturation_p1 | 28 | 0.043649 [0.030647, 0.062288] (n=7 trajectories) | 0.003098 [-0.035455, 0.033782] (n=7 trajectories) | 0.027321 [-0.004509, 0.062594] (n=7 trajectories) | 0.027321 [-0.004509, 0.062594] (n=7 trajectories) | no |
| gemma3-4b | I1 | saturation_p2 | 28 | 0.023137 [0.013615, 0.036233] (n=7 trajectories) | 0.003595 [-0.018962, 0.018271] (n=7 trajectories) | 0.047833 [0.010736, 0.088176] (n=7 trajectories) | 0.047833 [0.010736, 0.088176] (n=7 trajectories) | no |
| gemma3-4b | I2 | zero | 14 | 0.249263 [0.225543, 0.272982] (n=4 trajectories; BELOW SIX) | 0.246752 [0.220521, 0.272982] (n=4 trajectories; BELOW SIX) | 0.000000 [0.000000, 0.000000] (n=4 trajectories; BELOW SIX) | 0.000000 [0.000000, 0.000000] (n=4 trajectories; BELOW SIX) | no |
| gemma3-4b | I2 | constant | 14 | 0.249263 [0.225543, 0.272982] (n=4 trajectories; BELOW SIX) | 0.246752 [0.220521, 0.272982] (n=4 trajectories; BELOW SIX) | 0.000000 [0.000000, 0.000000] (n=4 trajectories; BELOW SIX) | 0.000000 [0.000000, 0.000000] (n=4 trajectories; BELOW SIX) | no |
| gemma3-4b | I2 | T_only | 14 | 0.249579 [0.226221, 0.272938] (n=4 trajectories; BELOW SIX) | 0.247376 [0.221815, 0.272938] (n=4 trajectories; BELOW SIX) | -0.000317 [-0.000677, 0.000044] (n=4 trajectories; BELOW SIX) | -0.000317 [-0.000677, 0.000044] (n=4 trajectories; BELOW SIX) | no |
| gemma3-4b | I2 | reuse_only | 14 | 0.098037 [0.073763, 0.122311] (n=4 trajectories; BELOW SIX) | 0.001873 [-0.035589, 0.039335] (n=4 trajectories; BELOW SIX) | 0.151226 [0.150671, 0.151780] (n=4 trajectories; BELOW SIX) | 0.151226 [0.150671, 0.151780] (n=4 trajectories; BELOW SIX) | yes |
| gemma3-4b | I2 | two_dimensional | 14 | 0.159458 [0.129666, 0.189251] (n=4 trajectories; BELOW SIX) | 0.086096 [0.038876, 0.133317] (n=4 trajectories; BELOW SIX) | 0.089804 [0.083731, 0.095878] (n=4 trajectories; BELOW SIX) | 0.089804 [0.083731, 0.095878] (n=4 trajectories; BELOW SIX) | yes |
| gemma3-4b | I2 | saturation_p1 | 14 | 0.110722 [0.082231, 0.139213] (n=4 trajectories; BELOW SIX) | 0.041105 [-0.008630, 0.090840] (n=4 trajectories; BELOW SIX) | 0.138540 [0.133769, 0.143312] (n=4 trajectories; BELOW SIX) | 0.138540 [0.133769, 0.143312] (n=4 trajectories; BELOW SIX) | yes |
| gemma3-4b | I2 | saturation_p2 | 14 | 0.063171 [0.039726, 0.086616] (n=4 trajectories; BELOW SIX) | 0.015186 [-0.039726, 0.070098] (n=4 trajectories; BELOW SIX) | 0.186092 [0.185818, 0.186366] (n=4 trajectories; BELOW SIX) | 0.186092 [0.185818, 0.186366] (n=4 trajectories; BELOW SIX) | yes |

#### leave_one_pool_seed_out_within_student: math / MATH-500:3816ece4b994 — retrospective development-set

| Student | Target | Candidate | Pairs | MAE [95% CI] | Bias [95% CI] | Gain over zero [95% CI] | Gain over constant [95% CI] | Clears target rule |
|---|---|---|---:|---|---|---|---|---|
| all | I1 | zero | 134 | 0.068018 [0.039571, 0.101299] (n=40 trajectories) | -0.063012 [-0.097100, -0.033662] (n=40 trajectories) | 0.000000 [0.000000, 0.000000] (n=40 trajectories) | 0.000000 [-0.000000, 0.000000] (n=40 trajectories) | no |
| all | I1 | constant | 134 | 0.068018 [0.039571, 0.101299] (n=40 trajectories) | -0.063012 [-0.097100, -0.033662] (n=40 trajectories) | 0.000000 [-0.000000, 0.000000] (n=40 trajectories) | 0.000000 [0.000000, 0.000000] (n=40 trajectories) | no |
| all | I1 | T_only | 134 | 0.074749 [0.057234, 0.095017] (n=40 trajectories) | 0.004140 [-0.029595, 0.033194] (n=40 trajectories) | -0.006730 [-0.022638, 0.009829] (n=40 trajectories) | -0.006730 [-0.022638, 0.009829] (n=40 trajectories) | no |
| all | I1 | reuse_only | 134 | 0.063939 [0.046829, 0.084118] (n=40 trajectories) | -0.000988 [-0.029251, 0.023299] (n=40 trajectories) | 0.004079 [-0.010257, 0.019348] (n=40 trajectories) | 0.004079 [-0.010257, 0.019348] (n=40 trajectories) | no |
| all | I1 | two_dimensional | 134 | 0.073251 [0.055238, 0.094397] (n=40 trajectories) | -0.000339 [-0.034139, 0.028710] (n=40 trajectories) | -0.005233 [-0.020056, 0.010134] (n=40 trajectories) | -0.005233 [-0.020056, 0.010134] (n=40 trajectories) | no |
| all | I1 | saturation_p1 | 134 | 0.060966 [0.044815, 0.080114] (n=40 trajectories) | -0.000948 [-0.027187, 0.021373] (n=40 trajectories) | 0.007052 [-0.007277, 0.022644] (n=40 trajectories) | 0.007052 [-0.007277, 0.022644] (n=40 trajectories) | no |
| all | I1 | saturation_p2 | 134 | 0.043167 [0.033977, 0.054263] (n=40 trajectories) | 0.012731 [-0.003237, 0.026033] (n=40 trajectories) | 0.024851 [0.002532, 0.050500] (n=40 trajectories) | 0.024851 [0.002532, 0.050500] (n=40 trajectories) | no |
| all | I2 | zero | 80 | 0.285991 [0.148494, 0.435423] (n=19 trajectories) | 0.280951 [0.142048, 0.432735] (n=19 trajectories) | 0.000000 [0.000000, 0.000000] (n=19 trajectories) | 0.000000 [-0.000000, 0.000000] (n=19 trajectories) | no |
| all | I2 | constant | 80 | 0.285991 [0.148494, 0.435423] (n=19 trajectories) | 0.280951 [0.142048, 0.432735] (n=19 trajectories) | 0.000000 [-0.000000, 0.000000] (n=19 trajectories) | 0.000000 [0.000000, 0.000000] (n=19 trajectories) | no |
| all | I2 | T_only | 80 | 0.285334 [0.148283, 0.435714] (n=19 trajectories) | 0.280400 [0.142113, 0.432651] (n=19 trajectories) | 0.000657 [-0.000391, 0.002458] (n=19 trajectories) | 0.000657 [-0.000391, 0.002458] (n=19 trajectories) | no |
| all | I2 | reuse_only | 80 | 0.169783 [0.078145, 0.273034] (n=19 trajectories) | 0.100963 [0.016628, 0.190242] (n=19 trajectories) | 0.116208 [0.068918, 0.173805] (n=19 trajectories) | 0.116208 [0.068918, 0.173805] (n=19 trajectories) | yes |
| all | I2 | two_dimensional | 80 | 0.196621 [0.103715, 0.308175] (n=19 trajectories) | 0.098200 [0.039991, 0.174425] (n=19 trajectories) | 0.089370 [0.041294, 0.147816] (n=19 trajectories) | 0.089370 [0.041294, 0.147816] (n=19 trajectories) | no |
| all | I2 | saturation_p1 | 80 | 0.132879 [0.071232, 0.220345] (n=19 trajectories) | 0.013462 [-0.031937, 0.073292] (n=19 trajectories) | 0.153112 [0.069996, 0.244837] (n=19 trajectories) | 0.153112 [0.069996, 0.244837] (n=19 trajectories) | no |
| all | I2 | saturation_p2 | 80 | 0.072251 [0.031686, 0.143512] (n=19 trajectories) | 0.005451 [-0.038080, 0.072665] (n=19 trajectories) | 0.213740 [0.104563, 0.331134] (n=19 trajectories) | 0.213740 [0.104563, 0.331134] (n=19 trajectories) | no |
| gemma3-1b | I1 | zero | 54 | 0.078280 [0.034358, 0.127393] (n=17 trajectories) | -0.074757 [-0.125014, -0.029748] (n=17 trajectories) | 0.000000 [0.000000, 0.000000] (n=17 trajectories) | 0.000000 [-0.000000, 0.000000] (n=17 trajectories) | no |
| gemma3-1b | I1 | constant | 54 | 0.078280 [0.034358, 0.127393] (n=17 trajectories) | -0.074757 [-0.125014, -0.029748] (n=17 trajectories) | 0.000000 [-0.000000, 0.000000] (n=17 trajectories) | 0.000000 [0.000000, 0.000000] (n=17 trajectories) | no |
| gemma3-1b | I1 | T_only | 54 | 0.081717 [0.054919, 0.111910] (n=17 trajectories) | -0.006473 [-0.056955, 0.038698] (n=17 trajectories) | -0.003436 [-0.024638, 0.020177] (n=17 trajectories) | -0.003436 [-0.024638, 0.020177] (n=17 trajectories) | no |
| gemma3-1b | I1 | reuse_only | 54 | 0.072278 [0.043987, 0.104720] (n=17 trajectories) | -0.012042 [-0.056589, 0.027255] (n=17 trajectories) | 0.006003 [-0.011888, 0.026112] (n=17 trajectories) | 0.006003 [-0.011888, 0.026112] (n=17 trajectories) | no |
| gemma3-1b | I1 | two_dimensional | 54 | 0.081714 [0.054809, 0.112148] (n=17 trajectories) | -0.006805 [-0.057543, 0.038553] (n=17 trajectories) | -0.003433 [-0.024607, 0.019906] (n=17 trajectories) | -0.003433 [-0.024607, 0.019906] (n=17 trajectories) | no |
| gemma3-1b | I1 | saturation_p1 | 54 | 0.067452 [0.042270, 0.096803] (n=17 trajectories) | -0.007061 [-0.047828, 0.028315] (n=17 trajectories) | 0.010828 [-0.010158, 0.034095] (n=17 trajectories) | 0.010828 [-0.010158, 0.034095] (n=17 trajectories) | no |
| gemma3-1b | I1 | saturation_p2 | 54 | 0.049393 [0.035644, 0.068887] (n=17 trajectories) | 0.009621 [-0.017733, 0.030961] (n=17 trajectories) | 0.028888 [-0.003631, 0.064621] (n=17 trajectories) | 0.028888 [-0.003631, 0.064621] (n=17 trajectories) | no |
| gemma3-1b | I2 | zero | 38 | 0.331513 [0.266636, 0.405852] (n=8 trajectories) | 0.326613 [0.257610, 0.405007] (n=8 trajectories) | 0.000000 [0.000000, 0.000000] (n=8 trajectories) | 0.000000 [-0.000000, 0.000000] (n=8 trajectories) | no |
| gemma3-1b | I2 | constant | 38 | 0.331513 [0.266636, 0.405852] (n=8 trajectories) | 0.326613 [0.257610, 0.405007] (n=8 trajectories) | 0.000000 [-0.000000, 0.000000] (n=8 trajectories) | 0.000000 [0.000000, 0.000000] (n=8 trajectories) | no |
| gemma3-1b | I2 | T_only | 38 | 0.330118 [0.266679, 0.406348] (n=8 trajectories) | 0.325292 [0.257576, 0.405956] (n=8 trajectories) | 0.001394 [-0.000497, 0.004514] (n=8 trajectories) | 0.001394 [-0.000497, 0.004514] (n=8 trajectories) | no |
| gemma3-1b | I2 | reuse_only | 38 | 0.203261 [0.169402, 0.290701] (n=8 trajectories) | 0.140391 [0.078582, 0.229080] (n=8 trajectories) | 0.128252 [0.097234, 0.193532] (n=8 trajectories) | 0.128252 [0.097234, 0.193532] (n=8 trajectories) | yes |
| gemma3-1b | I2 | two_dimensional | 38 | 0.224202 [0.196238, 0.318302] (n=8 trajectories) | 0.120097 [0.046507, 0.220984] (n=8 trajectories) | 0.107310 [0.070344, 0.190809] (n=8 trajectories) | 0.107310 [0.070344, 0.190809] (n=8 trajectories) | no |
| gemma3-1b | I2 | saturation_p1 | 38 | 0.145888 [0.096140, 0.250280] (n=8 trajectories) | 0.008242 [-0.059145, 0.131674] (n=8 trajectories) | 0.185624 [0.141068, 0.294853] (n=8 trajectories) | 0.185624 [0.141068, 0.294853] (n=8 trajectories) | yes |
| gemma3-1b | I2 | saturation_p2 | 38 | 0.080024 [0.031359, 0.186129] (n=8 trajectories) | -0.002678 [-0.058432, 0.134371] (n=8 trajectories) | 0.251489 [0.206886, 0.359804] (n=8 trajectories) | 0.251489 [0.206886, 0.359804] (n=8 trajectories) | yes |
| gemma3-270m | I1 | zero | 52 | 0.035416 [0.022230, 0.049120] (n=16 trajectories) | -0.026576 [-0.043386, -0.010746] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I1 | constant | 52 | 0.035416 [0.022230, 0.049120] (n=16 trajectories) | -0.026576 [-0.043386, -0.010746] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I1 | T_only | 52 | 0.038617 [0.032565, 0.043863] (n=16 trajectories) | 0.017945 [0.000810, 0.034009] (n=16 trajectories) | -0.003201 [-0.016864, 0.011101] (n=16 trajectories) | -0.003201 [-0.016864, 0.011101] (n=16 trajectories) | no |
| gemma3-270m | I1 | reuse_only | 52 | 0.031799 [0.026308, 0.037261] (n=16 trajectories) | 0.011994 [-0.000775, 0.023926] (n=16 trajectories) | 0.003617 [-0.006587, 0.014329] (n=16 trajectories) | 0.003617 [-0.006587, 0.014329] (n=16 trajectories) | no |
| gemma3-270m | I1 | two_dimensional | 52 | 0.035311 [0.028335, 0.041282] (n=16 trajectories) | 0.009882 [-0.007417, 0.026083] (n=16 trajectories) | 0.000105 [-0.010516, 0.011372] (n=16 trajectories) | 0.000105 [-0.010516, 0.011372] (n=16 trajectories) | no |
| gemma3-270m | I1 | saturation_p1 | 52 | 0.032101 [0.025304, 0.039078] (n=16 trajectories) | 0.008013 [-0.005730, 0.020824] (n=16 trajectories) | 0.003315 [-0.004942, 0.012093] (n=16 trajectories) | 0.003315 [-0.004942, 0.012093] (n=16 trajectories) | no |
| gemma3-270m | I1 | saturation_p2 | 52 | 0.023796 [0.019069, 0.029185] (n=16 trajectories) | 0.010217 [0.001796, 0.017374] (n=16 trajectories) | 0.011620 [0.000495, 0.023093] (n=16 trajectories) | 0.011620 [0.000495, 0.023093] (n=16 trajectories) | no |
| gemma3-270m | I2 | zero | 28 | 0.116628 [0.101038, 0.147173] (n=7 trajectories) | 0.110007 [0.094668, 0.143130] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-270m | I2 | constant | 28 | 0.116628 [0.101038, 0.147173] (n=7 trajectories) | 0.110007 [0.094668, 0.143130] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-270m | I2 | T_only | 28 | 0.116454 [0.099888, 0.147510] (n=7 trajectories) | 0.109885 [0.094646, 0.143774] (n=7 trajectories) | 0.000175 [-0.000337, 0.001150] (n=7 trajectories) | 0.000175 [-0.000337, 0.001150] (n=7 trajectories) | no |
| gemma3-270m | I2 | reuse_only | 28 | 0.056294 [0.044140, 0.080107] (n=7 trajectories) | 0.000066 [-0.015024, 0.034674] (n=7 trajectories) | 0.060334 [0.055999, 0.067066] (n=7 trajectories) | 0.060334 [0.055999, 0.067066] (n=7 trajectories) | yes |
| gemma3-270m | I2 | two_dimensional | 28 | 0.083622 [0.068760, 0.113737] (n=7 trajectories) | 0.038433 [0.021192, 0.077781] (n=7 trajectories) | 0.033007 [0.031441, 0.034872] (n=7 trajectories) | 0.033007 [0.031441, 0.034872] (n=7 trajectories) | yes |
| gemma3-270m | I2 | saturation_p1 | 28 | 0.063411 [0.048926, 0.090047] (n=7 trajectories) | 0.020425 [0.000096, 0.057868] (n=7 trajectories) | 0.053217 [0.049127, 0.057126] (n=7 trajectories) | 0.053217 [0.049127, 0.057126] (n=7 trajectories) | yes |
| gemma3-270m | I2 | saturation_p2 | 28 | 0.034725 [0.019156, 0.064501] (n=7 trajectories) | 0.007056 [-0.011811, 0.047115] (n=7 trajectories) | 0.081903 [0.078842, 0.084216] (n=7 trajectories) | 0.081903 [0.078842, 0.084216] (n=7 trajectories) | yes |
| gemma3-4b | I1 | zero | 28 | 0.108774 [0.021846, 0.214535] (n=7 trajectories) | -0.108025 [-0.213985, -0.020679] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I1 | constant | 28 | 0.108774 [0.021846, 0.214535] (n=7 trajectories) | -0.108025 [-0.213985, -0.020679] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I1 | T_only | 28 | 0.128412 [0.087549, 0.181037] (n=7 trajectories) | -0.001032 [-0.112865, 0.090646] (n=7 trajectories) | -0.019638 [-0.066968, 0.028673] (n=7 trajectories) | -0.019638 [-0.066968, 0.028673] (n=7 trajectories) | no |
| gemma3-4b | I1 | reuse_only | 28 | 0.107548 [0.068496, 0.158607] (n=7 trajectories) | -0.003778 [-0.095597, 0.069524] (n=7 trajectories) | 0.001226 [-0.047304, 0.050996] (n=7 trajectories) | 0.001226 [-0.047304, 0.050996] (n=7 trajectories) | no |
| gemma3-4b | I1 | two_dimensional | 28 | 0.127391 [0.081555, 0.185223] (n=7 trajectories) | -0.006851 [-0.120489, 0.086449] (n=7 trajectories) | -0.018616 [-0.062487, 0.026535] (n=7 trajectories) | -0.018616 [-0.062487, 0.026535] (n=7 trajectories) | no |
| gemma3-4b | I1 | saturation_p1 | 28 | 0.102065 [0.057982, 0.158444] (n=7 trajectories) | -0.005804 [-0.090579, 0.064919] (n=7 trajectories) | 0.006710 [-0.040334, 0.057129] (n=7 trajectories) | 0.006710 [-0.040334, 0.057129] (n=7 trajectories) | no |
| gemma3-4b | I1 | saturation_p2 | 28 | 0.067134 [0.051769, 0.093542] (n=7 trajectories) | 0.023398 [-0.024838, 0.056405] (n=7 trajectories) | 0.041640 [-0.034156, 0.122161] (n=7 trajectories) | 0.041640 [-0.034156, 0.122161] (n=7 trajectories) | no |
| gemma3-4b | I2 | zero | 14 | 0.501157 [0.478840, 0.523474] (n=4 trajectories; BELOW SIX) | 0.498900 [0.474326, 0.523474] (n=4 trajectories; BELOW SIX) | 0.000000 [0.000000, 0.000000] (n=4 trajectories; BELOW SIX) | 0.000000 [0.000000, 0.000000] (n=4 trajectories; BELOW SIX) | no |
| gemma3-4b | I2 | constant | 14 | 0.501157 [0.478840, 0.523474] (n=4 trajectories; BELOW SIX) | 0.498900 [0.474326, 0.523474] (n=4 trajectories; BELOW SIX) | 0.000000 [0.000000, 0.000000] (n=4 trajectories; BELOW SIX) | 0.000000 [0.000000, 0.000000] (n=4 trajectories; BELOW SIX) | no |
| gemma3-4b | I2 | T_only | 14 | 0.501536 [0.479580, 0.523493] (n=4 trajectories; BELOW SIX) | 0.499582 [0.475739, 0.523424] (n=4 trajectories; BELOW SIX) | -0.000379 [-0.000740, -0.000019] (n=4 trajectories; BELOW SIX) | -0.000379 [-0.000740, -0.000019] (n=4 trajectories; BELOW SIX) | no |
| gemma3-4b | I2 | reuse_only | 14 | 0.305889 [0.273583, 0.338196] (n=4 trajectories; BELOW SIX) | 0.195736 [0.164535, 0.226937] (n=4 trajectories; BELOW SIX) | 0.195267 [0.185278, 0.205257] (n=4 trajectories; BELOW SIX) | 0.195267 [0.185278, 0.205257] (n=4 trajectories; BELOW SIX) | yes |
| gemma3-4b | I2 | two_dimensional | 14 | 0.347756 [0.309891, 0.385622] (n=4 trajectories; BELOW SIX) | 0.158301 [0.121686, 0.194917] (n=4 trajectories; BELOW SIX) | 0.153401 [0.137852, 0.168949] (n=4 trajectories; BELOW SIX) | 0.153401 [0.137852, 0.168949] (n=4 trajectories; BELOW SIX) | yes |
| gemma3-4b | I2 | saturation_p1 | 14 | 0.236501 [0.188260, 0.284742] (n=4 trajectories; BELOW SIX) | 0.013704 [-0.031937, 0.059346] (n=4 trajectories; BELOW SIX) | 0.264655 [0.238731, 0.290580] (n=4 trajectories; BELOW SIX) | 0.264655 [0.238731, 0.290580] (n=4 trajectories; BELOW SIX) | yes |
| gemma3-4b | I2 | saturation_p2 | 14 | 0.126207 [0.068032, 0.184381] (n=4 trajectories; BELOW SIX) | 0.024303 [-0.030095, 0.078701] (n=4 trajectories; BELOW SIX) | 0.374950 [0.339092, 0.410808] (n=4 trajectories; BELOW SIX) | 0.374950 [0.339092, 0.410808] (n=4 trajectories; BELOW SIX) | yes |

#### leave_one_pool_seed_out_within_student: qa / 2WikiMultihopQA:075005b82489 — retrospective development-set

| Student | Target | Candidate | Pairs | MAE [95% CI] | Bias [95% CI] | Gain over zero [95% CI] | Gain over constant [95% CI] | Clears target rule |
|---|---|---|---:|---|---|---|---|---|
| all | I1 | zero | 134 | 0.856061 [0.606933, 1.124245] (n=40 trajectories) | -0.826850 [-1.103091, -0.569582] (n=40 trajectories) | 0.000000 [0.000000, 0.000000] (n=40 trajectories) | 0.000000 [-0.000000, 0.000000] (n=40 trajectories) | no |
| all | I1 | constant | 134 | 0.856061 [0.606933, 1.124245] (n=40 trajectories) | -0.826850 [-1.103091, -0.569582] (n=40 trajectories) | 0.000000 [-0.000000, 0.000000] (n=40 trajectories) | 0.000000 [0.000000, 0.000000] (n=40 trajectories) | no |
| all | I1 | T_only | 134 | 0.808428 [0.556122, 1.078668] (n=40 trajectories) | -0.765232 [-1.043895, -0.504121] (n=40 trajectories) | 0.047634 [0.016674, 0.076524] (n=40 trajectories) | 0.047634 [0.016674, 0.076524] (n=40 trajectories) | no |
| all | I1 | reuse_only | 134 | 0.731689 [0.492546, 0.983339] (n=40 trajectories) | -0.672820 [-0.937525, -0.422658] (n=40 trajectories) | 0.124372 [0.088882, 0.160062] (n=40 trajectories) | 0.124372 [0.088882, 0.160062] (n=40 trajectories) | yes |
| all | I1 | two_dimensional | 134 | 0.638080 [0.423212, 0.870737] (n=40 trajectories) | -0.420691 [-0.700897, -0.159227] (n=40 trajectories) | 0.217982 [0.171902, 0.268318] (n=40 trajectories) | 0.217982 [0.171902, 0.268318] (n=40 trajectories) | yes |
| all | I1 | saturation_p1 | 134 | 0.494569 [0.398158, 0.596793] (n=40 trajectories) | -0.048793 [-0.241772, 0.131546] (n=40 trajectories) | 0.361492 [0.177134, 0.563641] (n=40 trajectories) | 0.361492 [0.177134, 0.563641] (n=40 trajectories) | no |
| all | I1 | saturation_p2 | 134 | 0.334419 [0.266003, 0.407238] (n=40 trajectories) | -0.037197 [-0.131463, 0.050164] (n=40 trajectories) | 0.521642 [0.314678, 0.737209] (n=40 trajectories) | 0.521642 [0.314678, 0.737209] (n=40 trajectories) | yes |
| all | I2 | zero | 80 | 3.478397 [2.336934, 4.607533] (n=19 trajectories) | 3.477784 [2.336888, 4.607533] (n=19 trajectories) | 0.000000 [0.000000, 0.000000] (n=19 trajectories) | 0.000000 [-0.000000, 0.000000] (n=19 trajectories) | no |
| all | I2 | constant | 80 | 3.478397 [2.336934, 4.607533] (n=19 trajectories) | 3.477784 [2.336888, 4.607533] (n=19 trajectories) | 0.000000 [-0.000000, 0.000000] (n=19 trajectories) | 0.000000 [0.000000, 0.000000] (n=19 trajectories) | no |
| all | I2 | T_only | 80 | 3.477724 [2.337294, 4.605301] (n=19 trajectories) | 3.477119 [2.336326, 4.605301] (n=19 trajectories) | 0.000673 [-0.001152, 0.003140] (n=19 trajectories) | 0.000673 [-0.001152, 0.003140] (n=19 trajectories) | no |
| all | I2 | reuse_only | 80 | 3.153386 [2.277670, 4.131861] (n=19 trajectories) | 3.116187 [2.257332, 4.105607] (n=19 trajectories) | 0.325011 [0.015259, 0.582863] (n=19 trajectories) | 0.325011 [0.015259, 0.582863] (n=19 trajectories) | no |
| all | I2 | two_dimensional | 80 | 2.155977 [1.519646, 2.870970] (n=19 trajectories) | 1.162495 [0.416218, 2.038805] (n=19 trajectories) | 1.322420 [0.763870, 1.947340] (n=19 trajectories) | 1.322420 [0.763870, 1.947340] (n=19 trajectories) | yes |
| all | I2 | saturation_p1 | 80 | 1.231814 [0.890507, 1.812060] (n=19 trajectories) | 0.043883 [-0.651738, 0.763738] (n=19 trajectories) | 2.246584 [1.320865, 3.295834] (n=19 trajectories) | 2.246584 [1.320865, 3.295834] (n=19 trajectories) | yes |
| all | I2 | saturation_p2 | 80 | 0.718472 [0.421877, 1.266910] (n=19 trajectories) | -0.312843 [-1.005477, 0.391984] (n=19 trajectories) | 2.759925 [1.665183, 3.898999] (n=19 trajectories) | 2.759925 [1.665183, 3.898999] (n=19 trajectories) | yes |
| gemma3-1b | I1 | zero | 54 | 0.937554 [0.577239, 1.343379] (n=17 trajectories) | -0.925268 [-1.336060, -0.559609] (n=17 trajectories) | 0.000000 [0.000000, 0.000000] (n=17 trajectories) | 0.000000 [-0.000000, 0.000000] (n=17 trajectories) | no |
| gemma3-1b | I1 | constant | 54 | 0.937554 [0.577239, 1.343379] (n=17 trajectories) | -0.925268 [-1.336060, -0.559609] (n=17 trajectories) | 0.000000 [-0.000000, 0.000000] (n=17 trajectories) | 0.000000 [0.000000, 0.000000] (n=17 trajectories) | no |
| gemma3-1b | I1 | T_only | 54 | 0.815254 [0.456884, 1.218491] (n=17 trajectories) | -0.782531 [-1.202144, -0.411640] (n=17 trajectories) | 0.122300 [0.106484, 0.139013] (n=17 trajectories) | 0.122300 [0.106484, 0.139013] (n=17 trajectories) | yes |
| gemma3-1b | I1 | reuse_only | 54 | 0.743510 [0.408244, 1.120755] (n=17 trajectories) | -0.698474 [-1.095041, -0.347878] (n=17 trajectories) | 0.194044 [0.164363, 0.226681] (n=17 trajectories) | 0.194044 [0.164363, 0.226681] (n=17 trajectories) | yes |
| gemma3-1b | I1 | two_dimensional | 54 | 0.652446 [0.359352, 0.978879] (n=17 trajectories) | -0.399976 [-0.821154, -0.028711] (n=17 trajectories) | 0.285108 [0.204733, 0.375063] (n=17 trajectories) | 0.285108 [0.204733, 0.375063] (n=17 trajectories) | yes |
| gemma3-1b | I1 | saturation_p1 | 54 | 0.506074 [0.387942, 0.611848] (n=17 trajectories) | 0.013400 [-0.251824, 0.247035] (n=17 trajectories) | 0.431480 [0.108296, 0.796502] (n=17 trajectories) | 0.431480 [0.108296, 0.796502] (n=17 trajectories) | no |
| gemma3-1b | I1 | saturation_p2 | 54 | 0.413226 [0.326563, 0.506664] (n=17 trajectories) | 0.009475 [-0.093332, 0.129088] (n=17 trajectories) | 0.524329 [0.218751, 0.873827] (n=17 trajectories) | 0.524329 [0.218751, 0.873827] (n=17 trajectories) | no |
| gemma3-1b | I2 | zero | 38 | 4.146750 [3.629495, 5.348755] (n=8 trajectories) | 4.146508 [3.629495, 5.348755] (n=8 trajectories) | 0.000000 [0.000000, 0.000000] (n=8 trajectories) | 0.000000 [-0.000000, 0.000000] (n=8 trajectories) | no |
| gemma3-1b | I2 | constant | 38 | 4.146750 [3.629495, 5.348755] (n=8 trajectories) | 4.146508 [3.629495, 5.348755] (n=8 trajectories) | 0.000000 [-0.000000, 0.000000] (n=8 trajectories) | 0.000000 [0.000000, 0.000000] (n=8 trajectories) | no |
| gemma3-1b | I2 | T_only | 38 | 4.145051 [3.629453, 5.343158] (n=8 trajectories) | 4.144842 [3.629453, 5.343158] (n=8 trajectories) | 0.001699 [-0.002020, 0.005597] (n=8 trajectories) | 0.001699 [-0.002020, 0.005597] (n=8 trajectories) | no |
| gemma3-1b | I2 | reuse_only | 38 | 3.643049 [3.063906, 4.810226] (n=8 trajectories) | 3.590263 [2.965274, 4.810226] (n=8 trajectories) | 0.503701 [0.427850, 0.580560] (n=8 trajectories) | 0.503701 [0.427850, 0.580560] (n=8 trajectories) | yes |
| gemma3-1b | I2 | two_dimensional | 38 | 2.458401 [2.200495, 2.773483] (n=8 trajectories) | 1.479529 [0.718613, 2.773483] (n=8 trajectories) | 1.688350 [1.330569, 2.575272] (n=8 trajectories) | 1.688350 [1.330569, 2.575272] (n=8 trajectories) | yes |
| gemma3-1b | I2 | saturation_p1 | 38 | 1.172722 [0.886195, 1.489352] (n=8 trajectories) | 0.088031 [-0.395226, 0.899257] (n=8 trajectories) | 2.974028 [2.381169, 4.449498] (n=8 trajectories) | 2.974028 [2.381169, 4.449498] (n=8 trajectories) | yes |
| gemma3-1b | I2 | saturation_p2 | 38 | 0.499163 [0.340328, 0.618823] (n=8 trajectories) | -0.374843 [-0.618823, -0.100555] (n=8 trajectories) | 3.647587 [3.025639, 5.008427] (n=8 trajectories) | 3.647587 [3.025639, 5.008427] (n=8 trajectories) | yes |
| gemma3-270m | I1 | zero | 52 | 0.634605 [0.343807, 0.935531] (n=16 trajectories) | -0.599639 [-0.915598, -0.296186] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | 0.000000 [-0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I1 | constant | 52 | 0.634605 [0.343807, 0.935531] (n=16 trajectories) | -0.599639 [-0.915598, -0.296186] (n=16 trajectories) | 0.000000 [-0.000000, 0.000000] (n=16 trajectories) | 0.000000 [0.000000, 0.000000] (n=16 trajectories) | no |
| gemma3-270m | I1 | T_only | 52 | 0.689661 [0.384828, 1.001570] (n=16 trajectories) | -0.662076 [-0.987269, -0.347839] (n=16 trajectories) | -0.055057 [-0.073953, -0.035195] (n=16 trajectories) | -0.055057 [-0.073953, -0.035195] (n=16 trajectories) | no |
| gemma3-270m | I1 | reuse_only | 52 | 0.630407 [0.329922, 0.939129] (n=16 trajectories) | -0.590403 [-0.913840, -0.276076] (n=16 trajectories) | 0.004197 [-0.012710, 0.020753] (n=16 trajectories) | 0.004197 [-0.012710, 0.020753] (n=16 trajectories) | no |
| gemma3-270m | I1 | two_dimensional | 52 | 0.518584 [0.245450, 0.804510] (n=16 trajectories) | -0.394394 [-0.721078, -0.078291] (n=16 trajectories) | 0.116021 [0.088745, 0.141527] (n=16 trajectories) | 0.116021 [0.088745, 0.141527] (n=16 trajectories) | yes |
| gemma3-270m | I1 | saturation_p1 | 52 | 0.402292 [0.252730, 0.553780] (n=16 trajectories) | -0.114353 [-0.359837, 0.117113] (n=16 trajectories) | 0.232313 [0.056220, 0.414076] (n=16 trajectories) | 0.232313 [0.056220, 0.414076] (n=16 trajectories) | no |
| gemma3-270m | I1 | saturation_p2 | 52 | 0.230572 [0.133416, 0.331235] (n=16 trajectories) | -0.083806 [-0.212607, 0.028974] (n=16 trajectories) | 0.404032 [0.168428, 0.646369] (n=16 trajectories) | 0.404032 [0.168428, 0.646369] (n=16 trajectories) | no |
| gemma3-270m | I2 | zero | 28 | 2.132141 [1.725509, 2.621309] (n=7 trajectories) | 2.130718 [1.719817, 2.621309] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-270m | I2 | constant | 28 | 2.132141 [1.725509, 2.621309] (n=7 trajectories) | 2.130718 [1.719817, 2.621309] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-270m | I2 | T_only | 28 | 2.132236 [1.726955, 2.620115] (n=7 trajectories) | 2.130793 [1.721183, 2.620115] (n=7 trajectories) | -0.000095 [-0.001446, 0.001195] (n=7 trajectories) | -0.000095 [-0.001446, 0.001195] (n=7 trajectories) | no |
| gemma3-270m | I2 | reuse_only | 28 | 2.188709 [1.676157, 2.646163] (n=7 trajectories) | 2.184835 [1.660663, 2.646163] (n=7 trajectories) | -0.056568 [-0.125385, 0.049351] (n=7 trajectories) | -0.056568 [-0.125385, 0.049351] (n=7 trajectories) | no |
| gemma3-270m | I2 | two_dimensional | 28 | 1.482448 [1.092386, 1.889204] (n=7 trajectories) | 0.701371 [0.173402, 1.250343] (n=7 trajectories) | 0.649693 [0.602552, 0.732105] (n=7 trajectories) | 0.649693 [0.602552, 0.732105] (n=7 trajectories) | yes |
| gemma3-270m | I2 | saturation_p1 | 28 | 0.985064 [0.619912, 1.298438] (n=7 trajectories) | -0.022366 [-0.606716, 0.507568] (n=7 trajectories) | 1.147077 [1.065700, 1.322872] (n=7 trajectories) | 1.147077 [1.065700, 1.322872] (n=7 trajectories) | yes |
| gemma3-270m | I2 | saturation_p2 | 28 | 0.565273 [0.334405, 0.800968] (n=7 trajectories) | -0.231332 [-0.800968, 0.334149] (n=7 trajectories) | 1.566868 [0.924540, 1.927636] (n=7 trajectories) | 1.566868 [0.924540, 1.927636] (n=7 trajectories) | yes |
| gemma3-4b | I1 | zero | 28 | 1.110172 [0.489851, 1.846079] (n=7 trajectories) | -1.059011 [-1.822080, -0.414526] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I1 | constant | 28 | 1.110172 [0.489851, 1.846079] (n=7 trajectories) | -1.059011 [-1.822080, -0.414526] (n=7 trajectories) | 0.000000 [-0.000000, 0.000000] (n=7 trajectories) | 0.000000 [0.000000, 0.000000] (n=7 trajectories) | no |
| gemma3-4b | I1 | T_only | 28 | 1.015828 [0.373431, 1.788718] (n=7 trajectories) | -0.923443 [-1.749222, -0.239391] (n=7 trajectories) | 0.094344 [0.042059, 0.147220] (n=7 trajectories) | 0.094344 [0.042059, 0.147220] (n=7 trajectories) | no |
| gemma3-4b | I1 | reuse_only | 28 | 0.896987 [0.312940, 1.615791] (n=7 trajectories) | -0.776403 [-1.560295, -0.135862] (n=7 trajectories) | 0.213185 [0.155557, 0.276230] (n=7 trajectories) | 0.213185 [0.155557, 0.276230] (n=7 trajectories) | yes |
| gemma3-4b | I1 | two_dimensional | 28 | 0.832293 [0.277434, 1.509811] (n=7 trajectories) | -0.509476 [-1.348734, 0.186593] (n=7 trajectories) | 0.277879 [0.178090, 0.373952] (n=7 trajectories) | 0.277879 [0.178090, 0.373952] (n=7 trajectories) | yes |
| gemma3-4b | I1 | saturation_p1 | 28 | 0.643753 [0.423541, 0.983043] (n=7 trajectories) | -0.046984 [-0.637426, 0.407153] (n=7 trajectories) | 0.466419 [0.020137, 0.979385] (n=7 trajectories) | 0.466419 [0.020137, 0.979385] (n=7 trajectories) | no |
| gemma3-4b | I1 | saturation_p2 | 28 | 0.375293 [0.209717, 0.582732] (n=7 trajectories) | -0.040650 [-0.368801, 0.189958] (n=7 trajectories) | 0.734879 [0.226069, 1.297636] (n=7 trajectories) | 0.734879 [0.226069, 1.297636] (n=7 trajectories) | no |
| gemma3-4b | I2 | zero | 14 | 4.356809 [3.634667, 5.078952] (n=4 trajectories; BELOW SIX) | 4.356809 [3.634667, 5.078952] (n=4 trajectories; BELOW SIX) | 0.000000 [0.000000, 0.000000] (n=4 trajectories; BELOW SIX) | 0.000000 [-0.000000, 0.000000] (n=4 trajectories; BELOW SIX) | no |
| gemma3-4b | I2 | constant | 14 | 4.356809 [3.634667, 5.078952] (n=4 trajectories; BELOW SIX) | 4.356809 [3.634667, 5.078952] (n=4 trajectories; BELOW SIX) | 0.000000 [0.000000, 0.000000] (n=4 trajectories; BELOW SIX) | 0.000000 [0.000000, 0.000000] (n=4 trajectories; BELOW SIX) | no |
| gemma3-4b | I2 | T_only | 14 | 4.357384 [3.635819, 5.078949] (n=4 trajectories; BELOW SIX) | 4.357384 [3.635819, 5.078949] (n=4 trajectories; BELOW SIX) | -0.000574 [-0.001152, 0.000003] (n=4 trajectories; BELOW SIX) | -0.000574 [-0.001152, 0.000003] (n=4 trajectories; BELOW SIX) | no |
| gemma3-4b | I2 | reuse_only | 14 | 3.753656 [2.887490, 4.619822] (n=4 trajectories; BELOW SIX) | 3.692111 [2.793883, 4.590340] (n=4 trajectories; BELOW SIX) | 0.603154 [0.459130, 0.747177] (n=4 trajectories; BELOW SIX) | 0.603154 [0.459130, 0.747177] (n=4 trajectories; BELOW SIX) | yes |
| gemma3-4b | I2 | two_dimensional | 14 | 2.682173 [1.759315, 3.605032] (n=4 trajectories; BELOW SIX) | 1.224222 [0.030452, 2.417992] (n=4 trajectories; BELOW SIX) | 1.674636 [1.473920, 1.875352] (n=4 trajectories; BELOW SIX) | 1.674636 [1.473920, 1.875352] (n=4 trajectories; BELOW SIX) | yes |
| gemma3-4b | I2 | saturation_p1 | 14 | 1.885704 [1.262530, 2.508878] (n=4 trajectories; BELOW SIX) | 0.056547 [-1.262530, 1.375624] (n=4 trajectories; BELOW SIX) | 2.471106 [2.372137, 2.570074] (n=4 trajectories; BELOW SIX) | 2.471106 [2.372137, 2.570074] (n=4 trajectories; BELOW SIX) | yes |
| gemma3-4b | I2 | saturation_p2 | 14 | 1.620139 [1.542275, 1.698002] (n=4 trajectories; BELOW SIX) | -0.307583 [-1.698002, 1.082837] (n=4 trajectories; BELOW SIX) | 2.736671 [1.936664, 3.536677] (n=4 trajectories; BELOW SIX) | 2.736671 [1.936664, 3.536677] (n=4 trajectories; BELOW SIX) | yes |

### Per-trajectory breakdown

Each cell is MAE / signed bias. For I2, a pair appears in both endpoint trajectories' descriptive breakdowns, but once in the overall metric. No checkpoint bootstrap or within-trajectory interval is reported: each row conditions on one focal trajectory (below six).

#### leave_one_student_size_out: code / MBPP:e80dbf3c3adc, I1 — retrospective development-set

| Trajectory | Incident pairs | zero | constant | T_only | reuse_only | two_dimensional | saturation_p1 | saturation_p2 |
|---|---:|---|---|---|---|---|---|---|
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 2 | 0.053158 / -0.053158 | 0.053158 / -0.053158 | 0.020675 / 0.020675 | 0.014747 / 0.014747 | 0.014169 / 0.014169 | 0.012609 / 0.012609 | 0.006490 / 0.006490 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 2 | 0.029257 / -0.029257 | 0.029257 / -0.029257 | 0.044612 / 0.044612 | 0.038706 / 0.038706 | 0.038104 / 0.038104 | 0.036589 / 0.036589 | 0.030454 / 0.030454 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 2 | 0.028883 / -0.028883 | 0.028883 / -0.028883 | 0.045767 / 0.045767 | 0.040484 / 0.040484 | 0.039189 / 0.039189 | 0.038401 / 0.038401 | 0.032577 / 0.032577 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 2 | 0.055636 / -0.048411 | 0.055636 / -0.048411 | 0.043459 / 0.025587 | 0.047823 / 0.019494 | 0.044531 / 0.019067 | 0.051111 / 0.017466 | 0.050841 / 0.011103 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 2 | 0.037685 / -0.034890 | 0.037685 / -0.034890 | 0.038505 / 0.038505 | 0.032507 / 0.032507 | 0.032038 / 0.032038 | 0.032492 / 0.030452 | 0.032264 / 0.024192 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 2 | 0.055746 / -0.055746 | 0.055746 / -0.055746 | 0.018531 / 0.018531 | 0.012409 / 0.012409 | 0.011987 / 0.011987 | 0.010318 / 0.010318 | 0.004012 / 0.004012 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.035326 / -0.035326 | 0.035326 / -0.035326 | 0.046998 / 0.046998 | 0.023307 / 0.023307 | 0.039744 / 0.039744 | 0.020975 / 0.020975 | 0.008862 / 0.008862 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.024164 / -0.024164 | 0.024164 / -0.024164 | 0.056885 / 0.056885 | 0.036314 / 0.033588 | 0.049744 / 0.049744 | 0.031557 / 0.030955 | 0.020357 / 0.018650 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.032288 / -0.032288 | 0.032288 / -0.032288 | 0.050016 / 0.050016 | 0.027422 / 0.025615 | 0.042764 / 0.042764 | 0.023100 / 0.023100 | 0.011736 / 0.010657 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 4 | 0.018102 / -0.018102 | 0.018102 / -0.018102 | 0.064541 / 0.064001 | 0.045215 / 0.036048 | 0.059375 / 0.056767 | 0.040809 / 0.034386 | 0.027240 / 0.020820 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 4 | 0.021258 / -0.021258 | 0.021258 / -0.021258 | 0.061160 / 0.061160 | 0.033566 / 0.033300 | 0.053899 / 0.053899 | 0.031585 / 0.031585 | 0.018003 / 0.018003 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 4 | 0.020441 / -0.020441 | 0.020441 / -0.020441 | 0.061812 / 0.061812 | 0.033548 / 0.033548 | 0.054565 / 0.054565 | 0.031813 / 0.031813 | 0.018897 / 0.018168 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2rep_lora_dseed11_seed1 | 4 | 0.018974 / -0.018974 | 0.018974 / -0.018974 | 0.064638 / 0.063129 | 0.045312 / 0.035176 | 0.059472 / 0.055895 | 0.040906 / 0.033515 | 0.027337 / 0.019948 |
| gemma3-1b/gpt-5.6-luna_full_75_p2dev_lora_dseed11 | 2 | 0.285374 / -0.285374 | 0.285374 / -0.285374 | 0.183777 / -0.183777 | 0.187874 / -0.187874 | 0.192729 / -0.192729 | 0.199402 / -0.199402 | 0.151768 / -0.151768 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 4 | 0.184870 / -0.184870 | 0.184870 / -0.184870 | 0.103425 / -0.103425 | 0.104425 / -0.095605 | 0.110601 / -0.110601 | 0.111046 / -0.102545 | 0.084214 / -0.071758 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 4 | 0.229718 / -0.229718 | 0.229718 / -0.229718 | 0.147970 / -0.147970 | 0.139228 / -0.139228 | 0.155173 / -0.155173 | 0.146186 / -0.146186 | 0.118129 / -0.114284 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 4 | 0.148479 / -0.148479 | 0.148479 / -0.148479 | 0.065651 / -0.065651 | 0.060103 / -0.058548 | 0.072949 / -0.072949 | 0.066892 / -0.065651 | 0.040136 / -0.034964 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 2 | 0.066790 / -0.066790 | 0.066790 / -0.066790 | 0.021787 / 0.005031 | 0.018002 / 0.003417 | 0.021247 / 0.002049 | 0.020278 / 0.002987 | 0.023264 / 0.000425 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 2 | 0.049795 / -0.049795 | 0.049795 / -0.049795 | 0.038173 / 0.022062 | 0.034373 / 0.020471 | 0.037600 / 0.019079 | 0.036627 / 0.020024 | 0.039549 / 0.017431 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 2 | 0.050626 / -0.050626 | 0.050626 / -0.050626 | 0.046410 / 0.021991 | 0.042496 / 0.021093 | 0.046013 / 0.018976 | 0.044844 / 0.020643 | 0.048311 / 0.018444 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 2 | 0.064188 / -0.064188 | 0.064188 / -0.064188 | 0.017024 / 0.007794 | 0.020792 / 0.006019 | 0.017516 / 0.004805 | 0.018496 / 0.005480 | 0.015603 / 0.002553 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 2 | 0.066181 / -0.066181 | 0.066181 / -0.066181 | 0.008720 / 0.005214 | 0.004995 / 0.003500 | 0.008203 / 0.002250 | 0.007244 / 0.002998 | 0.010099 / 0.000200 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 2 | 0.058570 / -0.058570 | 0.058570 / -0.058570 | 0.013685 / 0.013685 | 0.011897 / 0.011897 | 0.010685 / 0.010685 | 0.011410 / 0.011410 | 0.008596 / 0.008596 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.026551 / -0.026551 | 0.026551 / -0.026551 | 0.053530 / 0.053530 | 0.042726 / 0.034069 | 0.051402 / 0.050204 | 0.044488 / 0.033626 | 0.039825 / 0.022420 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.034592 / -0.034592 | 0.034592 / -0.034592 | 0.052460 / 0.044249 | 0.042594 / 0.025118 | 0.050070 / 0.040975 | 0.047135 / 0.024960 | 0.043143 / 0.013588 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.030468 / -0.030468 | 0.030468 / -0.030468 | 0.049594 / 0.049594 | 0.033217 / 0.029398 | 0.046269 / 0.046269 | 0.038044 / 0.029133 | 0.033993 / 0.017496 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 4 | 0.034198 / -0.020386 | 0.034198 / -0.020386 | 0.062657 / 0.059481 | 0.050248 / 0.035600 | 0.060819 / 0.056165 | 0.054822 / 0.034256 | 0.047582 / 0.021225 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 4 | 0.027292 / -0.025520 | 0.027292 / -0.025520 | 0.065554 / 0.054653 | 0.052683 / 0.030887 | 0.063701 / 0.051324 | 0.054072 / 0.029596 | 0.047153 / 0.016561 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 4 | 0.024607 / -0.024607 | 0.024607 / -0.024607 | 0.055406 / 0.055406 | 0.032186 / 0.031212 | 0.052083 / 0.052083 | 0.033487 / 0.029954 | 0.026578 / 0.016795 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 4 | 0.104490 / -0.104490 | 0.104490 / -0.104490 | 0.028303 / -0.025263 | 0.017486 / -0.012198 | 0.029447 / -0.028552 | 0.021136 / -0.006760 | 0.045775 / 0.041568 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 4 | 0.124938 / -0.124938 | 0.124938 / -0.124938 | 0.045416 / -0.045416 | 0.031381 / -0.031381 | 0.048718 / -0.048718 | 0.025890 / -0.025890 | 0.033214 / 0.023963 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 4 | 0.098255 / -0.098255 | 0.098255 / -0.098255 | 0.017683 / -0.017683 | 0.008974 / -0.005275 | 0.021029 / -0.021029 | 0.005617 / 0.000352 | 0.048502 / 0.048502 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2rep_lora_dseed11_seed1 | 4 | 0.115679 / -0.115679 | 0.115679 / -0.115679 | 0.036452 / -0.036452 | 0.023387 / -0.023387 | 0.039742 / -0.039742 | 0.017950 / -0.017950 | 0.033973 / 0.030378 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.033651 / -0.033651 | 0.033651 / -0.033651 | 0.041752 / 0.041752 | 0.023655 / 0.020976 | 0.038898 / 0.038898 | 0.026927 / 0.022409 | 0.019482 / 0.009415 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.032523 / -0.032523 | 0.032523 / -0.032523 | 0.041713 / 0.041713 | 0.021284 / 0.021284 | 0.038902 / 0.038902 | 0.024196 / 0.022925 | 0.018783 / 0.009712 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.047256 / -0.047256 | 0.047256 / -0.047256 | 0.028130 / 0.028130 | 0.007659 / 0.006692 | 0.025276 / 0.025276 | 0.010073 / 0.008254 | 0.009764 / -0.005145 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed11 | 4 | 0.027915 / -0.027915 | 0.027915 / -0.027915 | 0.053066 / 0.047287 | 0.035984 / 0.022536 | 0.051033 / 0.044440 | 0.038887 / 0.023488 | 0.024014 / 0.008857 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed12 | 4 | 0.023444 / -0.023168 | 0.023444 / -0.023168 | 0.052322 / 0.052322 | 0.027663 / 0.027663 | 0.049465 / 0.049465 | 0.028653 / 0.028653 | 0.018089 / 0.014005 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed11 | 4 | 0.184047 / -0.184047 | 0.184047 / -0.184047 | 0.109448 / -0.109448 | 0.100879 / -0.100879 | 0.112272 / -0.112272 | 0.096174 / -0.096174 | 0.058184 / -0.058184 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed12 | 4 | 0.147953 / -0.147953 | 0.147953 / -0.147953 | 0.073077 / -0.073077 | 0.063645 / -0.063645 | 0.075911 / -0.075911 | 0.058926 / -0.058926 | 0.019609 / -0.019609 |

#### leave_one_student_size_out: code / MBPP:e80dbf3c3adc, I2 — retrospective development-set

| Trajectory | Incident pairs | zero | constant | T_only | reuse_only | two_dimensional | saturation_p1 | saturation_p2 |
|---|---:|---|---|---|---|---|---|---|
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 42 | 0.096822 / 0.088172 | 0.096822 / 0.088172 | 0.096423 / 0.087878 | 0.070798 / 0.032585 | 0.072756 / 0.058615 | 0.069997 / 0.040841 | 0.060096 / 0.030240 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 43 | 0.095592 / 0.075787 | 0.095592 / 0.075787 | 0.096116 / 0.074968 | 0.094951 / 0.021077 | 0.095216 / 0.046434 | 0.093968 / 0.029206 | 0.084620 / 0.018758 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 43 | 0.095866 / 0.088511 | 0.095866 / 0.088511 | 0.095809 / 0.088832 | 0.069676 / 0.033453 | 0.071473 / 0.059401 | 0.068835 / 0.041684 | 0.059475 / 0.031398 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 43 | 0.096899 / 0.081843 | 0.096899 / 0.081843 | 0.095993 / 0.081090 | 0.065393 / 0.027222 | 0.070611 / 0.052560 | 0.064571 / 0.035319 | 0.055010 / 0.024964 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 43 | 0.093299 / 0.078562 | 0.093299 / 0.078562 | 0.093113 / 0.078130 | 0.080084 / 0.024277 | 0.080466 / 0.049617 | 0.079392 / 0.032346 | 0.069711 / 0.021982 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 43 | 0.106576 / 0.079614 | 0.106576 / 0.079614 | 0.105534 / 0.079075 | 0.056013 / 0.025238 | 0.077619 / 0.050564 | 0.061238 / 0.033319 | 0.051042 / 0.022936 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_throughput_lora_dseed31 | 5 | 0.011891 / -0.000709 | 0.011891 / -0.000709 | 0.012917 / -0.000661 | 0.013016 / -0.001379 | 0.012884 / -0.001317 | 0.013068 / -0.001268 | 0.012954 / -0.000921 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 79 | 0.104658 / 0.097840 | 0.104658 / 0.097840 | 0.104540 / 0.097099 | 0.064577 / 0.018072 | 0.082427 / 0.050796 | 0.067634 / 0.029943 | 0.053728 / 0.023518 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 73 | 0.118334 / 0.108919 | 0.118334 / 0.108919 | 0.117606 / 0.107935 | 0.067068 / 0.027268 | 0.087008 / 0.062762 | 0.070822 / 0.039360 | 0.056007 / 0.030007 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 79 | 0.099224 / 0.094390 | 0.099224 / 0.094390 | 0.099166 / 0.094095 | 0.059118 / 0.016950 | 0.075788 / 0.048735 | 0.061661 / 0.028458 | 0.047524 / 0.022751 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 72 | 0.123164 / 0.116647 | 0.123164 / 0.116647 | 0.120999 / 0.114144 | 0.071390 / 0.020780 | 0.091260 / 0.058811 | 0.074353 / 0.034962 | 0.057607 / 0.029429 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 79 | 0.118875 / 0.111844 | 0.118875 / 0.111844 | 0.116531 / 0.109664 | 0.064033 / 0.024966 | 0.084732 / 0.059524 | 0.067755 / 0.037813 | 0.053732 / 0.032661 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 80 | 0.107923 / 0.095289 | 0.107923 / 0.095289 | 0.106920 / 0.094844 | 0.065871 / 0.009141 | 0.081944 / 0.044012 | 0.067529 / 0.021974 | 0.052139 / 0.017070 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2rep_lora_dseed11_seed1 | 72 | 0.125388 / 0.119077 | 0.125388 / 0.119077 | 0.123091 / 0.116574 | 0.071113 / 0.023211 | 0.092255 / 0.061242 | 0.074489 / 0.037393 | 0.058241 / 0.031859 |
| gemma3-1b/gpt-5.6-luna_full_75_p2dev_lora_dseed11 | 57 | 0.347359 / 0.346468 | 0.347359 / 0.346468 | 0.345589 / 0.344955 | 0.186668 / 0.183804 | 0.261117 / 0.260389 | 0.208698 / 0.207937 | 0.170025 / 0.169314 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 74 | 0.242083 / 0.240856 | 0.242083 / 0.240856 | 0.244636 / 0.243545 | 0.142805 / 0.095035 | 0.194334 / 0.157673 | 0.153641 / 0.116791 | 0.116725 / 0.098073 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 78 | 0.270190 / 0.267085 | 0.270190 / 0.267085 | 0.273327 / 0.270823 | 0.177894 / 0.129108 | 0.225188 / 0.188821 | 0.186783 / 0.149771 | 0.150606 / 0.131619 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 79 | 0.191846 / 0.190036 | 0.191846 / 0.190036 | 0.191351 / 0.189624 | 0.102008 / 0.052848 | 0.145510 / 0.110537 | 0.110780 / 0.073234 | 0.074210 / 0.055183 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 40 | 0.055925 / 0.050619 | 0.055925 / 0.050619 | 0.056152 / 0.050828 | 0.027937 / -0.004291 | 0.029776 / 0.011541 | 0.030512 / -0.014402 | 0.035518 / -0.028393 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 40 | 0.055454 / 0.048061 | 0.055454 / 0.048061 | 0.055491 / 0.047879 | 0.039585 / -0.007169 | 0.042503 / 0.008611 | 0.041015 / -0.017335 | 0.039767 / -0.031440 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 40 | 0.056628 / 0.048216 | 0.056628 / 0.048216 | 0.056429 / 0.048922 | 0.037078 / -0.007489 | 0.039143 / 0.008574 | 0.039092 / -0.017807 | 0.039233 / -0.031659 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 40 | 0.057121 / 0.049127 | 0.057121 / 0.049127 | 0.056862 / 0.049003 | 0.030581 / -0.006059 | 0.033062 / 0.009723 | 0.033193 / -0.016199 | 0.037248 / -0.030206 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 40 | 0.054217 / 0.048476 | 0.054217 / 0.048476 | 0.054401 / 0.048562 | 0.034042 / -0.006432 | 0.036857 / 0.009349 | 0.035971 / -0.016550 | 0.036700 / -0.030572 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 40 | 0.056420 / 0.048938 | 0.056420 / 0.048938 | 0.056106 / 0.049018 | 0.027391 / -0.005985 | 0.029326 / 0.009791 | 0.030003 / -0.016111 | 0.035465 / -0.030140 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 74 | 0.067862 / 0.066827 | 0.067862 / 0.066827 | 0.068201 / 0.066609 | 0.035240 / -0.022773 | 0.041185 / -0.005232 | 0.044443 / -0.039289 | 0.050838 / -0.045038 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 68 | 0.069942 / 0.069612 | 0.069942 / 0.069612 | 0.069778 / 0.069287 | 0.036976 / -0.021037 | 0.037614 / 0.000497 | 0.048591 / -0.037699 | 0.062054 / -0.048767 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 74 | 0.068732 / 0.065728 | 0.068732 / 0.065728 | 0.071350 / 0.066451 | 0.034250 / -0.021257 | 0.040533 / -0.004302 | 0.043137 / -0.037373 | 0.048227 / -0.041987 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 74 | 0.070471 / 0.067361 | 0.070471 / 0.067361 | 0.069684 / 0.065359 | 0.041001 / -0.032561 | 0.045265 / -0.014812 | 0.054717 / -0.050803 | 0.058060 / -0.053543 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 74 | 0.071026 / 0.067044 | 0.071026 / 0.067044 | 0.070167 / 0.065150 | 0.040424 / -0.032252 | 0.043001 / -0.014487 | 0.055325 / -0.050373 | 0.059291 / -0.053326 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 75 | 0.081183 / 0.079513 | 0.081183 / 0.079513 | 0.080887 / 0.079279 | 0.035008 / -0.018943 | 0.042147 / -0.001125 | 0.045579 / -0.037096 | 0.050050 / -0.039567 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 67 | 0.129744 / 0.128535 | 0.129744 / 0.128535 | 0.132691 / 0.131427 | 0.043606 / -0.016539 | 0.067210 / 0.016334 | 0.061097 / -0.043466 | 0.078891 / -0.066725 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 73 | 0.144131 / 0.142731 | 0.144131 / 0.142731 | 0.147298 / 0.146171 | 0.056148 / 0.009841 | 0.086155 / 0.040056 | 0.052706 / -0.014931 | 0.054221 / -0.036949 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 74 | 0.101230 / 0.100491 | 0.101230 / 0.100491 | 0.101641 / 0.100840 | 0.059937 / -0.031148 | 0.054847 / -0.001945 | 0.082529 / -0.055421 | 0.104352 / -0.077008 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2rep_lora_dseed11_seed1 | 67 | 0.148758 / 0.147864 | 0.148758 / 0.147864 | 0.151591 / 0.150756 | 0.048482 / 0.002790 | 0.081170 / 0.035663 | 0.053283 / -0.024137 | 0.060289 / -0.047396 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 41 | 0.097803 / 0.081288 | 0.097803 / 0.081288 | 0.098673 / 0.081998 | 0.059912 / 0.017767 | 0.072184 / 0.031728 | 0.057258 / 0.010523 | 0.043901 / 0.006332 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 37 | 0.106444 / 0.081932 | 0.106444 / 0.081932 | 0.106632 / 0.082429 | 0.052260 / 0.017232 | 0.067792 / 0.034603 | 0.049288 / 0.009896 | 0.031307 / 0.001356 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 41 | 0.086636 / 0.077263 | 0.086636 / 0.077263 | 0.087285 / 0.078046 | 0.047926 / 0.018261 | 0.062420 / 0.030660 | 0.045384 / 0.011523 | 0.030634 / 0.008931 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed11 | 41 | 0.101760 / 0.097749 | 0.101760 / 0.097749 | 0.101222 / 0.096310 | 0.053614 / 0.025940 | 0.069638 / 0.039696 | 0.051166 / 0.017860 | 0.032079 / 0.016225 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed12 | 41 | 0.107548 / 0.102817 | 0.107548 / 0.102817 | 0.106381 / 0.101444 | 0.054430 / 0.031599 | 0.071526 / 0.045366 | 0.051099 / 0.023595 | 0.034002 / 0.021753 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed11 | 41 | 0.230813 / 0.208507 | 0.230813 / 0.208507 | 0.232192 / 0.210268 | 0.121754 / 0.061136 | 0.159683 / 0.096302 | 0.114398 / 0.044357 | 0.070071 / 0.029772 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed12 | 40 | 0.180074 / 0.156514 | 0.180074 / 0.156514 | 0.181650 / 0.158813 | 0.083892 / 0.008076 | 0.118325 / 0.043287 | 0.078755 / -0.008879 | 0.038219 / -0.023454 |

#### leave_one_student_size_out: math / MATH-500:3816ece4b994, I1 — retrospective development-set

| Trajectory | Incident pairs | zero | constant | T_only | reuse_only | two_dimensional | saturation_p1 | saturation_p2 |
|---|---:|---|---|---|---|---|---|---|
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 2 | 0.042544 / -0.042544 | 0.042544 / -0.042544 | 0.018423 / 0.018423 | 0.016854 / 0.016854 | 0.012384 / 0.012384 | 0.013262 / 0.013262 | 0.015796 / 0.015796 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 2 | 0.034096 / -0.034096 | 0.034096 / -0.034096 | 0.026902 / 0.026902 | 0.025352 / 0.025352 | 0.020860 / 0.020860 | 0.021751 / 0.021751 | 0.024248 / 0.024248 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 2 | 0.041417 / -0.041417 | 0.041417 / -0.041417 | 0.020225 / 0.020225 | 0.019260 / 0.019260 | 0.014120 / 0.014120 | 0.015721 / 0.015721 | 0.018506 / 0.018506 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 2 | 0.040584 / -0.040584 | 0.040584 / -0.040584 | 0.024391 / 0.020520 | 0.027612 / 0.018815 | 0.025387 / 0.014468 | 0.027316 / 0.015131 | 0.022212 / 0.017316 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 2 | 0.041720 / -0.041720 | 0.041720 / -0.041720 | 0.023986 / 0.018885 | 0.027173 / 0.017233 | 0.025033 / 0.012882 | 0.026924 / 0.013598 | 0.021846 / 0.015881 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 2 | 0.050775 / -0.050775 | 0.050775 / -0.050775 | 0.010560 / 0.010560 | 0.008843 / 0.008843 | 0.004485 / 0.004485 | 0.005180 / 0.005180 | 0.007507 / 0.007507 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.025615 / -0.016418 | 0.025615 / -0.016418 | 0.051560 / 0.051560 | 0.034870 / 0.034870 | 0.044827 / 0.044827 | 0.029585 / 0.029585 | 0.025989 / 0.025989 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.020150 / -0.014581 | 0.020150 / -0.014581 | 0.052345 / 0.052345 | 0.035937 / 0.035937 | 0.045717 / 0.045717 | 0.030897 / 0.030897 | 0.027223 / 0.027223 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.018748 / -0.018748 | 0.018748 / -0.018748 | 0.049214 / 0.049214 | 0.031901 / 0.031901 | 0.042483 / 0.042483 | 0.026701 / 0.026701 | 0.022828 / 0.022828 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 4 | 0.009414 / -0.002425 | 0.009414 / -0.002425 | 0.065372 / 0.065372 | 0.044942 / 0.044942 | 0.058657 / 0.058657 | 0.038495 / 0.038495 | 0.033498 / 0.033498 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 4 | 0.012076 / -0.007548 | 0.012076 / -0.007548 | 0.060508 / 0.060508 | 0.040175 / 0.040175 | 0.053767 / 0.053767 | 0.033770 / 0.033770 | 0.028790 / 0.028790 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 4 | 0.016399 / -0.004475 | 0.016399 / -0.004475 | 0.063445 / 0.063445 | 0.042750 / 0.042750 | 0.056718 / 0.056718 | 0.036356 / 0.036356 | 0.032817 / 0.031280 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2rep_lora_dseed11_seed1 | 4 | 0.009893 / -0.000535 | 0.009893 / -0.000535 | 0.067262 / 0.067262 | 0.046831 / 0.046831 | 0.060547 / 0.060547 | 0.040385 / 0.040385 | 0.035387 / 0.035387 |
| gemma3-1b/gpt-5.6-luna_full_75_p2dev_lora_dseed11 | 2 | 0.293027 / -0.293027 | 0.293027 / -0.293027 | 0.209134 / -0.209134 | 0.207742 / -0.207742 | 0.217443 / -0.217443 | 0.204148 / -0.204148 | 0.130945 / -0.130945 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 4 | 0.184483 / -0.184483 | 0.184483 / -0.184483 | 0.131331 / -0.117230 | 0.131941 / -0.106401 | 0.136102 / -0.123891 | 0.130967 / -0.104017 | 0.087691 / -0.056417 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 4 | 0.280118 / -0.280118 | 0.280118 / -0.280118 | 0.220292 / -0.212615 | 0.221172 / -0.200965 | 0.224993 / -0.219301 | 0.220283 / -0.198452 | 0.176112 / -0.149578 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 4 | 0.207806 / -0.207806 | 0.207806 / -0.207806 | 0.151031 / -0.139411 | 0.152385 / -0.129141 | 0.155811 / -0.146185 | 0.151266 / -0.126655 | 0.108119 / -0.079116 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 2 | 0.024218 / -0.024218 | 0.024218 / -0.024218 | 0.051425 / 0.051425 | 0.053694 / 0.053694 | 0.050058 / 0.050058 | 0.052815 / 0.052815 | 0.060880 / 0.060880 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 2 | 0.024483 / -0.024483 | 0.024483 / -0.024483 | 0.051198 / 0.051198 | 0.053495 / 0.053495 | 0.049830 / 0.049830 | 0.052554 / 0.052554 | 0.060576 / 0.060576 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 2 | 0.019245 / -0.019245 | 0.019245 / -0.019245 | 0.057235 / 0.057235 | 0.060344 / 0.060344 | 0.055853 / 0.055853 | 0.059383 / 0.059383 | 0.067956 / 0.067956 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 2 | 0.024928 / -0.023659 | 0.024928 / -0.023659 | 0.052153 / 0.052153 | 0.054253 / 0.054253 | 0.050783 / 0.050783 | 0.053002 / 0.053002 | 0.060574 / 0.060574 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 2 | 0.017739 / -0.017739 | 0.017739 / -0.017739 | 0.057454 / 0.057454 | 0.059589 / 0.059589 | 0.056096 / 0.056096 | 0.058459 / 0.058459 | 0.066158 / 0.066158 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 2 | 0.022200 / -0.022200 | 0.022200 / -0.022200 | 0.053898 / 0.053898 | 0.055999 / 0.055999 | 0.052523 / 0.052523 | 0.054927 / 0.054927 | 0.062690 / 0.062690 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.018033 / -0.000256 | 0.018033 / -0.000256 | 0.084086 / 0.084086 | 0.067018 / 0.067018 | 0.082562 / 0.082562 | 0.066263 / 0.066263 | 0.060833 / 0.060833 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.018180 / -0.003187 | 0.018180 / -0.003187 | 0.079849 / 0.079849 | 0.063076 / 0.063076 | 0.078348 / 0.078348 | 0.063293 / 0.063293 | 0.057677 / 0.057677 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.011185 / -0.005503 | 0.011185 / -0.005503 | 0.078819 / 0.078819 | 0.060934 / 0.060934 | 0.077295 / 0.077295 | 0.060788 / 0.060788 | 0.054717 / 0.054717 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 4 | 0.020188 / 0.004930 | 0.020188 / 0.004930 | 0.089046 / 0.089046 | 0.067060 / 0.067060 | 0.087526 / 0.087526 | 0.063303 / 0.063303 | 0.055821 / 0.055821 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 4 | 0.016262 / 0.004489 | 0.016262 / 0.004489 | 0.088928 / 0.088928 | 0.067087 / 0.067087 | 0.087402 / 0.087402 | 0.063512 / 0.063512 | 0.056043 / 0.056043 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 4 | 0.016868 / 0.002131 | 0.016868 / 0.002131 | 0.086401 / 0.086401 | 0.064076 / 0.064076 | 0.084878 / 0.084878 | 0.060610 / 0.060610 | 0.052884 / 0.052884 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 4 | 0.066970 / -0.066970 | 0.066970 / -0.066970 | 0.026697 / 0.016472 | 0.035450 / 0.035450 | 0.025881 / 0.014964 | 0.054310 / 0.054310 | 0.129872 / 0.129872 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 4 | 0.089460 / -0.087935 | 0.089460 / -0.087935 | 0.030567 / -0.004183 | 0.029232 / 0.015890 | 0.031631 / -0.005696 | 0.034920 / 0.034920 | 0.112564 / 0.112564 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 4 | 0.071199 / -0.061757 | 0.071199 / -0.061757 | 0.023102 / 0.023102 | 0.041427 / 0.041427 | 0.021577 / 0.021568 | 0.060932 / 0.060932 | 0.136121 / 0.136121 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2rep_lora_dseed11_seed1 | 4 | 0.065659 / -0.065659 | 0.065659 / -0.065659 | 0.020195 / 0.017783 | 0.036762 / 0.036762 | 0.019379 / 0.016276 | 0.055622 / 0.055622 | 0.131184 / 0.131184 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.025700 / -0.025700 | 0.025700 / -0.025700 | 0.033824 / 0.033824 | 0.018645 / 0.018645 | 0.030045 / 0.030045 | 0.016432 / 0.016432 | 0.011787 / 0.011787 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.027921 / -0.025814 | 0.027921 / -0.025814 | 0.032789 / 0.032789 | 0.017865 / 0.017865 | 0.029069 / 0.029069 | 0.016019 / 0.016019 | 0.011631 / 0.011231 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.029460 / -0.029460 | 0.029460 / -0.029460 | 0.030050 / 0.030050 | 0.014333 / 0.014333 | 0.026272 / 0.026272 | 0.012337 / 0.012337 | 0.008874 / 0.007339 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed11 | 4 | 0.011673 / -0.011673 | 0.011673 / -0.011673 | 0.047692 / 0.047692 | 0.030788 / 0.029281 | 0.043923 / 0.043923 | 0.031772 / 0.025804 | 0.024565 / 0.019960 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed12 | 4 | 0.021528 / -0.018393 | 0.021528 / -0.018393 | 0.041200 / 0.041200 | 0.022870 / 0.022870 | 0.037416 / 0.037416 | 0.019460 / 0.019460 | 0.013617 / 0.013617 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed11 | 4 | 0.360675 / -0.360675 | 0.360675 / -0.360675 | 0.301786 / -0.301786 | 0.293162 / -0.293162 | 0.305524 / -0.305524 | 0.287156 / -0.287156 | 0.245719 / -0.245719 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed12 | 4 | 0.284463 / -0.284463 | 0.284463 / -0.284463 | 0.225355 / -0.225355 | 0.216024 / -0.216024 | 0.229108 / -0.229108 | 0.209931 / -0.209931 | 0.167307 / -0.167307 |

#### leave_one_student_size_out: math / MATH-500:3816ece4b994, I2 — retrospective development-set

| Trajectory | Incident pairs | zero | constant | T_only | reuse_only | two_dimensional | saturation_p1 | saturation_p2 |
|---|---:|---|---|---|---|---|---|---|
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 42 | 0.107480 / 0.103270 | 0.107480 / 0.103270 | 0.107210 / 0.103027 | 0.073599 / 0.054647 | 0.075285 / 0.066484 | 0.074680 / 0.040972 | 0.062384 / 0.031423 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 43 | 0.105263 / 0.095125 | 0.105263 / 0.095125 | 0.104740 / 0.094448 | 0.089135 / 0.047268 | 0.090635 / 0.058801 | 0.090314 / 0.033873 | 0.078806 / 0.024385 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 43 | 0.108573 / 0.107620 | 0.108573 / 0.107620 | 0.108669 / 0.107886 | 0.065053 / 0.059459 | 0.072481 / 0.071150 | 0.066093 / 0.045790 | 0.054283 / 0.036660 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 43 | 0.104551 / 0.101210 | 0.104551 / 0.101210 | 0.104140 / 0.100588 | 0.076219 / 0.053432 | 0.077897 / 0.064948 | 0.077137 / 0.040069 | 0.065402 / 0.030701 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 43 | 0.106037 / 0.099253 | 0.106037 / 0.099253 | 0.105695 / 0.098896 | 0.068101 / 0.051769 | 0.071641 / 0.063286 | 0.069072 / 0.038424 | 0.057059 / 0.029068 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 43 | 0.113333 / 0.101654 | 0.113333 / 0.101654 | 0.112376 / 0.101209 | 0.066019 / 0.054089 | 0.077398 / 0.065598 | 0.059207 / 0.040732 | 0.047448 / 0.031353 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_throughput_lora_dseed31 | 5 | 0.005842 / 0.002349 | 0.005842 / 0.002349 | 0.006334 / 0.002388 | 0.006213 / 0.001763 | 0.006067 / 0.001571 | 0.006131 / 0.001582 | 0.006201 / 0.002059 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 79 | 0.120678 / 0.116324 | 0.120678 / 0.116324 | 0.120383 / 0.115712 | 0.077904 / 0.046549 | 0.090242 / 0.057882 | 0.073412 / 0.026952 | 0.055483 / 0.024087 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 73 | 0.132211 / 0.127561 | 0.132211 / 0.127561 | 0.131827 / 0.126748 | 0.080868 / 0.056138 | 0.093016 / 0.070322 | 0.075324 / 0.036152 | 0.057480 / 0.029773 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 79 | 0.115409 / 0.110694 | 0.115409 / 0.110694 | 0.116240 / 0.110449 | 0.072914 / 0.042955 | 0.085318 / 0.053809 | 0.069585 / 0.023881 | 0.051061 / 0.021835 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 72 | 0.137532 / 0.130394 | 0.137532 / 0.130394 | 0.136100 / 0.128326 | 0.087047 / 0.046537 | 0.099644 / 0.059174 | 0.083737 / 0.023295 | 0.062035 / 0.022321 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 79 | 0.130447 / 0.124095 | 0.130447 / 0.124095 | 0.128815 / 0.122295 | 0.078635 / 0.048100 | 0.091973 / 0.059634 | 0.074085 / 0.027033 | 0.055350 / 0.025990 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 80 | 0.123337 / 0.117152 | 0.123337 / 0.117152 | 0.122973 / 0.116784 | 0.077845 / 0.041797 | 0.089418 / 0.053310 | 0.075078 / 0.020563 | 0.055233 / 0.020059 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2rep_lora_dseed11_seed1 | 72 | 0.139528 / 0.132052 | 0.139528 / 0.132052 | 0.138142 / 0.129984 | 0.087308 / 0.048195 | 0.100393 / 0.060832 | 0.083607 / 0.024953 | 0.062050 / 0.023979 |
| gemma3-1b/gpt-5.6-luna_full_75_p2dev_lora_dseed11 | 57 | 0.334077 / 0.329578 | 0.334077 / 0.329578 | 0.332920 / 0.328329 | 0.195256 / 0.187292 | 0.227747 / 0.222707 | 0.167084 / 0.147411 | 0.124383 / 0.109911 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 74 | 0.247999 / 0.237649 | 0.247999 / 0.237649 | 0.249647 / 0.239869 | 0.164124 / 0.110096 | 0.192677 / 0.132735 | 0.148633 / 0.073632 | 0.104314 / 0.060405 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 78 | 0.362267 / 0.360437 | 0.362267 / 0.360437 | 0.365154 / 0.363524 | 0.275977 / 0.239746 | 0.305996 / 0.261251 | 0.256254 / 0.205004 | 0.215454 / 0.192108 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 79 | 0.261522 / 0.248986 | 0.261522 / 0.248986 | 0.260730 / 0.248645 | 0.179230 / 0.128984 | 0.207087 / 0.149895 | 0.163824 / 0.095181 | 0.122301 / 0.081718 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 40 | 0.041731 / 0.041161 | 0.041731 / 0.041161 | 0.041720 / 0.041381 | 0.029543 / -0.019775 | 0.027915 / -0.015218 | 0.059386 / -0.053500 | 0.075370 / -0.069183 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 40 | 0.042290 / 0.041842 | 0.042290 / 0.041842 | 0.042124 / 0.041650 | 0.029414 / -0.019450 | 0.027676 / -0.014938 | 0.059551 / -0.053360 | 0.075651 / -0.069197 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 40 | 0.044629 / 0.044593 | 0.044629 / 0.044593 | 0.045514 / 0.045337 | 0.030008 / -0.017225 | 0.027580 / -0.012772 | 0.060202 / -0.051638 | 0.076513 / -0.067037 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 40 | 0.042532 / 0.041743 | 0.042532 / 0.041743 | 0.042325 / 0.041613 | 0.032659 / -0.019499 | 0.032497 / -0.014991 | 0.061524 / -0.053324 | 0.077703 / -0.069035 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 40 | 0.041682 / 0.039703 | 0.041682 / 0.039703 | 0.041556 / 0.039794 | 0.034498 / -0.021231 | 0.035650 / -0.016705 | 0.062194 / -0.054980 | 0.078165 / -0.070703 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 40 | 0.042431 / 0.040955 | 0.042431 / 0.040955 | 0.042281 / 0.041039 | 0.029369 / -0.019996 | 0.027540 / -0.015480 | 0.059460 / -0.053773 | 0.075623 / -0.069492 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 74 | 0.049749 / 0.045455 | 0.049749 / 0.045455 | 0.050114 / 0.045225 | 0.056654 / -0.053979 | 0.060893 / -0.058300 | 0.111403 / -0.109074 | 0.114059 / -0.110823 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 68 | 0.053189 / 0.051362 | 0.053189 / 0.051362 | 0.053622 / 0.051019 | 0.050406 / -0.049236 | 0.049395 / -0.048113 | 0.105958 / -0.104819 | 0.115227 / -0.113926 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 74 | 0.049767 / 0.046556 | 0.049767 / 0.046556 | 0.052529 / 0.047318 | 0.053503 / -0.049975 | 0.058124 / -0.054601 | 0.106898 / -0.103730 | 0.107520 / -0.103970 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 74 | 0.051431 / 0.045523 | 0.051431 / 0.045523 | 0.050906 / 0.043414 | 0.066761 / -0.065365 | 0.073666 / -0.072185 | 0.127660 / -0.126235 | 0.124856 / -0.123278 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 74 | 0.054883 / 0.050327 | 0.054883 / 0.050327 | 0.053765 / 0.048333 | 0.060560 / -0.059866 | 0.067133 / -0.066493 | 0.120899 / -0.120331 | 0.118692 / -0.117720 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 75 | 0.055008 / 0.050834 | 0.055008 / 0.050834 | 0.055149 / 0.050588 | 0.060352 / -0.058427 | 0.067338 / -0.065275 | 0.120948 / -0.118986 | 0.117651 / -0.115526 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 67 | 0.101386 / 0.097069 | 0.101386 / 0.097069 | 0.104414 / 0.100115 | 0.073909 / -0.063927 | 0.079419 / -0.065605 | 0.162957 / -0.153719 | 0.184930 / -0.175724 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 73 | 0.121584 / 0.119212 | 0.121584 / 0.119212 | 0.124924 / 0.122835 | 0.061802 / -0.028262 | 0.070906 / -0.029926 | 0.131159 / -0.110867 | 0.151433 / -0.131941 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 74 | 0.080525 / 0.077847 | 0.080525 / 0.077847 | 0.080912 / 0.078214 | 0.081656 / -0.068240 | 0.083658 / -0.069875 | 0.161997 / -0.149204 | 0.183625 / -0.170085 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2rep_lora_dseed11_seed1 | 67 | 0.093134 / 0.088237 | 0.093134 / 0.088237 | 0.096162 / 0.091283 | 0.082566 / -0.072759 | 0.085287 / -0.074437 | 0.171927 / -0.162551 | 0.194103 / -0.184556 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 41 | 0.176634 / 0.167586 | 0.176634 / 0.167586 | 0.177566 / 0.168146 | 0.140882 / 0.116022 | 0.151706 / 0.120768 | 0.130132 / 0.097763 | 0.116191 / 0.097476 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 37 | 0.201110 / 0.181681 | 0.201110 / 0.181681 | 0.201803 / 0.182073 | 0.158092 / 0.129159 | 0.167829 / 0.136999 | 0.146797 / 0.110653 | 0.132398 / 0.106360 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 41 | 0.159865 / 0.150886 | 0.159865 / 0.150886 | 0.160682 / 0.151503 | 0.127083 / 0.102989 | 0.137289 / 0.106840 | 0.117851 / 0.085992 | 0.104525 / 0.086969 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed11 | 41 | 0.186109 / 0.181755 | 0.186109 / 0.181755 | 0.185765 / 0.180619 | 0.146644 / 0.123463 | 0.158327 / 0.127293 | 0.135229 / 0.103239 | 0.120912 / 0.105572 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed12 | 41 | 0.180223 / 0.175302 | 0.180223 / 0.175302 | 0.179942 / 0.174218 | 0.141814 / 0.117489 | 0.153734 / 0.121396 | 0.130810 / 0.097450 | 0.116615 / 0.099553 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed11 | 41 | 0.451893 / 0.429827 | 0.451893 / 0.429827 | 0.453107 / 0.431217 | 0.364322 / 0.310195 | 0.389585 / 0.323801 | 0.338044 / 0.267859 | 0.304298 / 0.262686 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed12 | 40 | 0.409550 / 0.387223 | 0.409550 / 0.387223 | 0.410922 / 0.389038 | 0.319085 / 0.266726 | 0.344687 / 0.280145 | 0.293106 / 0.223936 | 0.258441 / 0.218879 |

#### leave_one_student_size_out: qa / 2WikiMultihopQA:075005b82489, I1 — retrospective development-set

| Trajectory | Incident pairs | zero | constant | T_only | reuse_only | two_dimensional | saturation_p1 | saturation_p2 |
|---|---:|---|---|---|---|---|---|---|
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 2 | 0.878766 / -0.878766 | 0.878766 / -0.878766 | 0.852664 / -0.852664 | 0.754666 / -0.754666 | 0.558331 / -0.558331 | 0.270573 / -0.270573 | 0.386563 / -0.386563 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 2 | 0.817480 / -0.817480 | 0.817480 / -0.817480 | 0.791365 / -0.791365 | 0.693274 / -0.693274 | 0.589820 / -0.496885 | 0.302895 / -0.210885 | 0.327205 / -0.327205 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 2 | 0.751121 / -0.751121 | 0.751121 / -0.751121 | 0.724729 / -0.724729 | 0.687528 / -0.624348 | 0.654832 / -0.427138 | 0.376970 / -0.143153 | 0.354780 / -0.254562 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 2 | 0.814274 / -0.814274 | 0.814274 / -0.814274 | 0.788114 / -0.788114 | 0.690173 / -0.690173 | 0.559048 / -0.493123 | 0.277002 / -0.215151 | 0.335528 / -0.335528 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 2 | 0.922466 / -0.922466 | 0.922466 / -0.922466 | 0.896519 / -0.896519 | 0.799295 / -0.799295 | 0.603933 / -0.603933 | 0.323355 / -0.323355 | 0.441820 / -0.441820 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 2 | 1.061255 / -1.061255 | 1.061255 / -1.061255 | 1.034996 / -1.034996 | 0.936697 / -0.936697 | 0.738890 / -0.738890 | 0.455575 / -0.455575 | 0.574891 / -0.574891 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.421393 / -0.319270 | 0.421393 / -0.319270 | 0.401034 / -0.290166 | 0.346474 / -0.212114 | 0.172716 / 0.038015 | 0.338702 / 0.218018 | 0.278136 / 0.004891 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.345975 / -0.344077 | 0.345975 / -0.344077 | 0.325056 / -0.315423 | 0.268962 / -0.238531 | 0.102862 / 0.007677 | 0.370669 / 0.219686 | 0.308138 / 0.003648 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.381101 / -0.381101 | 0.381101 / -0.381101 | 0.352004 / -0.352004 | 0.280359 / -0.275279 | 0.106987 / -0.023901 | 0.313678 / 0.169593 | 0.244463 / -0.048924 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 4 | 0.305248 / -0.305248 | 0.305248 / -0.305248 | 0.276222 / -0.276222 | 0.206285 / -0.206285 | 0.074498 / 0.051082 | 0.358200 / 0.169680 | 0.211567 / -0.068120 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 4 | 0.382704 / -0.372494 | 0.382704 / -0.372494 | 0.361767 / -0.343357 | 0.309953 / -0.272786 | 0.125683 / -0.014799 | 0.271398 / 0.108653 | 0.165510 / -0.129424 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 4 | 0.350068 / -0.298431 | 0.350068 / -0.298431 | 0.329670 / -0.269352 | 0.279580 / -0.199763 | 0.212768 / 0.058548 | 0.441155 / 0.180965 | 0.339110 / -0.058091 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2rep_lora_dseed11_seed1 | 4 | 0.336576 / -0.336576 | 0.336576 / -0.336576 | 0.307549 / -0.307549 | 0.237613 / -0.237613 | 0.078653 / 0.019754 | 0.336147 / 0.138352 | 0.207412 / -0.099448 |
| gemma3-1b/gpt-5.6-luna_full_75_p2dev_lora_dseed11 | 2 | 1.778698 / -1.778698 | 1.778698 / -1.778698 | 1.742780 / -1.742780 | 1.600511 / -1.600511 | 1.337765 / -1.337765 | 0.392001 / -0.392001 | 0.431213 / 0.431213 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 4 | 2.048540 / -2.048540 | 2.048540 / -2.048540 | 2.019747 / -2.019747 | 1.885402 / -1.885402 | 1.695066 / -1.695066 | 1.052169 / -1.052169 | 0.545765 / -0.528684 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 4 | 2.258046 / -2.258046 | 2.258046 / -2.258046 | 2.229145 / -2.229145 | 2.092671 / -2.092671 | 1.903258 / -1.903258 | 1.257731 / -1.257731 | 0.715239 / -0.715239 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 4 | 2.315301 / -2.315301 | 2.315301 / -2.315301 | 2.286019 / -2.286019 | 2.150947 / -2.150947 | 1.955828 / -1.955828 | 1.306532 / -1.306532 | 0.784693 / -0.784693 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 2 | 0.349384 / -0.349384 | 0.349384 / -0.349384 | 0.254858 / -0.191859 | 0.246100 / -0.065623 | 0.190031 / 0.165946 | 0.554883 / 0.554883 | 0.379573 / 0.379573 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 2 | 0.307738 / -0.296221 | 0.307738 / -0.296221 | 0.277499 / -0.138619 | 0.267371 / -0.012219 | 0.219364 / 0.219364 | 0.606160 / 0.606160 | 0.430413 / 0.430413 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 2 | 0.267430 / -0.235745 | 0.267430 / -0.235745 | 0.246444 / -0.076476 | 0.244577 / 0.054126 | 0.285290 / 0.285290 | 0.671273 / 0.671273 | 0.502113 / 0.502113 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 2 | 0.349664 / -0.349664 | 0.349664 / -0.349664 | 0.227444 / -0.191787 | 0.220775 / -0.065901 | 0.168433 / 0.166817 | 0.543221 / 0.543221 | 0.362015 / 0.362015 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 2 | 0.356636 / -0.356636 | 0.356636 / -0.356636 | 0.254614 / -0.200047 | 0.246642 / -0.074999 | 0.192556 / 0.155634 | 0.535178 / 0.535178 | 0.356653 / 0.356653 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 2 | 0.330428 / -0.330428 | 0.330428 / -0.330428 | 0.258138 / -0.171955 | 0.251780 / -0.045620 | 0.199704 / 0.188004 | 0.571204 / 0.571204 | 0.391336 / 0.391336 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.206487 / -0.077534 | 0.206487 / -0.077534 | 0.098106 / 0.098106 | 0.167482 / 0.167482 | 0.497058 / 0.497058 | 0.725030 / 0.718496 | 0.446231 / 0.411702 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.243697 / -0.146586 | 0.243697 / -0.146586 | 0.117453 / 0.026335 | 0.106611 / 0.094750 | 0.419110 / 0.419110 | 0.680845 / 0.680845 | 0.399959 / 0.369750 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.186364 / -0.186364 | 0.186364 / -0.186364 | 0.077216 / -0.010767 | 0.065811 / 0.055603 | 0.388090 / 0.388090 | 0.735358 / 0.624977 | 0.459628 / 0.310653 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 4 | 0.161557 / -0.137621 | 0.161557 / -0.137621 | 0.097192 / 0.037548 | 0.127010 / 0.088661 | 0.435433 / 0.435433 | 0.715268 / 0.570838 | 0.466104 / 0.231516 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 4 | 0.306617 / -0.213365 | 0.306617 / -0.213365 | 0.183153 / -0.037524 | 0.164826 / 0.014622 | 0.361887 / 0.361887 | 0.576402 / 0.503747 | 0.264671 / 0.163911 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 4 | 0.236849 / -0.147146 | 0.236849 / -0.147146 | 0.115954 / 0.028344 | 0.137484 / 0.078464 | 0.426955 / 0.426955 | 0.640011 / 0.566589 | 0.404931 / 0.225490 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 4 | 1.426980 / -1.426980 | 1.426980 / -1.426980 | 1.253213 / -1.253213 | 1.141877 / -1.053956 | 0.961917 / -0.858516 | 0.225681 / 0.008810 | 0.704334 / 0.704334 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 4 | 1.746421 / -1.746421 | 1.746421 / -1.746421 | 1.572009 / -1.572009 | 1.450564 / -1.368283 | 1.271906 / -1.175845 | 0.497499 / -0.303141 | 0.418318 / 0.418318 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 4 | 1.220446 / -1.220446 | 1.220446 / -1.220446 | 1.043731 / -1.043731 | 0.878705 / -0.844643 | 0.694865 / -0.642335 | 0.232283 / 0.232283 | 0.925297 / 0.925297 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2rep_lora_dseed11_seed1 | 4 | 1.533802 / -1.533802 | 1.533802 / -1.533802 | 1.360036 / -1.360036 | 1.235783 / -1.160779 | 1.055823 / -0.965339 | 0.371193 / -0.098013 | 0.597512 / 0.597512 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.477527 / -0.301450 | 0.477527 / -0.301450 | 0.430029 / -0.233549 | 0.389417 / -0.175428 | 0.210244 / 0.105209 | 0.351081 / 0.276546 | 0.260934 / 0.045285 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.668078 / -0.626214 | 0.668078 / -0.626214 | 0.619272 / -0.559364 | 0.577504 / -0.502084 | 0.479975 / -0.225851 | 0.154150 / -0.020660 | 0.255542 / -0.255542 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.538666 / -0.473831 | 0.538666 / -0.473831 | 0.491794 / -0.405946 | 0.452509 / -0.349377 | 0.303914 / -0.067270 | 0.171162 / 0.118050 | 0.119169 / -0.119169 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed11 | 4 | 0.272893 / -0.251074 | 0.272893 / -0.251074 | 0.224533 / -0.183354 | 0.188411 / -0.134686 | 0.164504 / 0.154498 | 0.432760 / 0.260407 | 0.313266 / 0.004253 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed12 | 4 | 0.606045 / -0.552509 | 0.606045 / -0.552509 | 0.557198 / -0.484529 | 0.520485 / -0.435245 | 0.313505 / -0.145382 | 0.111984 / -0.034403 | 0.290950 / -0.290950 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed11 | 4 | 2.906997 / -2.906997 | 2.906997 / -2.906997 | 2.839820 / -2.839820 | 2.715134 / -2.715134 | 2.504675 / -2.504675 | 1.839905 / -1.839905 | 1.305650 / -1.305650 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed12 | 4 | 2.300999 / -2.300999 | 2.300999 / -2.300999 | 2.233572 / -2.233572 | 2.106506 / -2.106506 | 1.897182 / -1.897182 | 1.229471 / -1.229471 | 0.675285 / -0.675285 |

#### leave_one_student_size_out: qa / 2WikiMultihopQA:075005b82489, I2 — retrospective development-set

| Trajectory | Incident pairs | zero | constant | T_only | reuse_only | two_dimensional | saturation_p1 | saturation_p2 |
|---|---:|---|---|---|---|---|---|---|
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 42 | 1.341496 / 1.069311 | 1.341496 / 1.069311 | 1.341303 / 1.069207 | 1.240788 / 0.967723 | 0.860616 / 0.582391 | 0.727406 / 0.249834 | 0.556018 / 0.071548 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 43 | 1.352177 / 1.130152 | 1.352177 / 1.130152 | 1.351781 / 1.129863 | 1.252971 / 1.030167 | 0.880019 / 0.652287 | 0.564122 / 0.323236 | 0.386366 / 0.147486 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 43 | 1.326252 / 1.045882 | 1.326252 / 1.045882 | 1.326167 / 1.045995 | 1.228812 / 0.945260 | 0.859093 / 0.559746 | 0.717931 / 0.229811 | 0.552799 / 0.056889 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 43 | 1.328417 / 1.283238 | 1.328417 / 1.283238 | 1.328104 / 1.282972 | 1.229710 / 1.183416 | 0.857625 / 0.805819 | 0.577079 / 0.479048 | 0.400535 / 0.304912 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 43 | 1.522009 / 1.254834 | 1.522009 / 1.254834 | 1.521847 / 1.254681 | 1.424187 / 1.155624 | 1.054378 / 0.779544 | 0.732918 / 0.454136 | 0.559200 / 0.279821 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 43 | 1.678023 / 1.288958 | 1.678023 / 1.288958 | 1.677639 / 1.288767 | 1.579413 / 1.189581 | 1.208395 / 0.813089 | 0.881963 / 0.487239 | 0.805299 / 0.312605 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_throughput_lora_dseed31 | 5 | 0.140027 / -0.078623 | 0.140027 / -0.078623 | 0.140390 / -0.078606 | 0.142084 / -0.079847 | 0.145430 / -0.089224 | 0.139899 / -0.089398 | 0.137577 / -0.083375 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 79 | 1.624024 / 1.585918 | 1.624024 / 1.585918 | 1.623669 / 1.585656 | 1.502438 / 1.440137 | 1.179851 / 0.813862 | 0.836207 / 0.407677 | 0.556102 / 0.303174 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 73 | 1.733145 / 1.711688 | 1.733145 / 1.711688 | 1.732837 / 1.711340 | 1.593184 / 1.562465 | 1.177970 / 0.956958 | 0.819161 / 0.510600 | 0.557654 / 0.355836 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 79 | 1.530284 / 1.498515 | 1.530284 / 1.498515 | 1.530115 / 1.498410 | 1.401263 / 1.356988 | 1.074488 / 0.744683 | 0.754981 / 0.356540 | 0.492915 / 0.264274 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 72 | 1.848533 / 1.809765 | 1.848533 / 1.809765 | 1.847679 / 1.808880 | 1.688420 / 1.634563 | 1.244107 / 0.877775 | 0.833799 / 0.398621 | 0.549777 / 0.310697 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 79 | 1.622774 / 1.600962 | 1.622774 / 1.600962 | 1.621995 / 1.600192 | 1.480260 / 1.442188 | 1.095339 / 0.756942 | 0.745976 / 0.322380 | 0.487798 / 0.240353 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 80 | 1.588913 / 1.527870 | 1.588913 / 1.527870 | 1.588561 / 1.527712 | 1.455942 / 1.370430 | 1.088048 / 0.682454 | 0.748477 / 0.255306 | 0.488015 / 0.177479 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2rep_lora_dseed11_seed1 | 72 | 1.787141 / 1.756541 | 1.787141 / 1.756541 | 1.786281 / 1.755656 | 1.627320 / 1.581339 | 1.213654 / 0.824551 | 0.824273 / 0.345397 | 0.536063 / 0.257473 |
| gemma3-1b/gpt-5.6-luna_full_75_p2dev_lora_dseed11 | 57 | 4.102417 / 3.998941 | 4.102417 / 3.998941 | 4.101482 / 3.998406 | 3.805546 / 3.701663 | 2.696592 / 2.587971 | 1.708899 / 1.601615 | 1.079646 / 0.948942 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 74 | 3.012524 / 2.913782 | 3.012524 / 2.913782 | 3.013201 / 2.914733 | 2.759059 / 2.647285 | 2.105872 / 1.505536 | 1.392957 / 0.759970 | 0.792015 / 0.449069 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 78 | 2.851496 / 2.728933 | 2.851496 / 2.728933 | 2.852146 / 2.730255 | 2.628204 / 2.476773 | 2.058544 / 1.390938 | 1.389725 / 0.685549 | 0.803546 / 0.383804 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 79 | 3.198819 / 3.023968 | 3.198819 / 3.023968 | 3.198178 / 3.023822 | 2.976581 / 2.773248 | 2.411442 / 1.710233 | 1.733388 / 1.000715 | 1.146013 / 0.700467 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 40 | 0.760022 / 0.686411 | 0.760022 / 0.686411 | 0.760137 / 0.686869 | 0.545105 / 0.464476 | 0.540590 / 0.014418 | 0.555896 / -0.406502 | 0.654563 / -0.643681 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 40 | 0.755948 / 0.678508 | 0.755948 / 0.678508 | 0.755513 / 0.678108 | 0.556134 / 0.455278 | 0.576137 / 0.003760 | 0.598817 / -0.421436 | 0.668823 / -0.660359 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 40 | 0.767256 / 0.626911 | 0.767256 / 0.626911 | 0.767184 / 0.628460 | 0.656642 / 0.401765 | 0.712079 / -0.059366 | 0.708804 / -0.486348 | 0.727246 / -0.721407 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 40 | 0.765603 / 0.692997 | 0.765603 / 0.692997 | 0.765032 / 0.692726 | 0.547153 / 0.469950 | 0.519112 / 0.018492 | 0.546107 / -0.404574 | 0.654586 / -0.641823 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 40 | 0.772426 / 0.691269 | 0.772426 / 0.691269 | 0.772473 / 0.691457 | 0.554136 / 0.469342 | 0.500864 / 0.019564 | 0.526744 / -0.402274 | 0.652864 / -0.639980 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 40 | 0.791176 / 0.705210 | 0.791176 / 0.705210 | 0.790489 / 0.705385 | 0.571776 / 0.483223 | 0.442428 / 0.033223 | 0.489046 / -0.388466 | 0.643079 / -0.626425 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 74 | 1.088059 / 0.940418 | 1.088059 / 0.940418 | 1.088102 / 0.939940 | 0.827259 / 0.578274 | 0.763074 / -0.293152 | 0.883141 / -0.845913 | 0.981041 / -0.945120 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 68 | 0.990205 / 0.976475 | 0.990205 / 0.976475 | 0.991285 / 0.975761 | 0.701201 / 0.610091 | 0.599850 / -0.205616 | 0.871596 / -0.825817 | 1.063825 / -1.014717 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 74 | 0.866521 / 0.811101 | 0.866521 / 0.811101 | 0.871622 / 0.812687 | 0.631046 / 0.459526 | 0.674398 / -0.396388 | 0.932736 / -0.925686 | 1.018782 / -1.006623 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 74 | 0.904192 / 0.827862 | 0.904192 / 0.827862 | 0.904311 / 0.823471 | 0.653089 / 0.424002 | 0.781958 / -0.562594 | 1.172392 / -1.158341 | 1.217915 / -1.205288 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 74 | 0.925673 / 0.869200 | 0.925673 / 0.869200 | 0.924135 / 0.865047 | 0.646068 / 0.467869 | 0.729714 / -0.511259 | 1.110012 / -1.104108 | 1.162525 / -1.154600 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 75 | 0.905205 / 0.858648 | 0.905205 / 0.858648 | 0.905771 / 0.858135 | 0.633414 / 0.460712 | 0.746731 / -0.521866 | 1.119352 / -1.104337 | 1.166007 / -1.148663 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 67 | 1.927838 / 1.926734 | 1.927838 / 1.926734 | 1.933861 / 1.933079 | 1.450351 / 1.340377 | 1.050994 / -0.023935 | 1.208282 / -0.966548 | 1.535996 / -1.364986 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 73 | 2.217729 / 2.217153 | 2.217729 / 2.217153 | 2.224972 / 2.224698 | 1.782203 / 1.680040 | 1.380407 / 0.424741 | 1.055672 / -0.440183 | 1.121373 / -0.817665 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 74 | 1.516351 / 1.513476 | 1.516351 / 1.513476 | 1.516764 / 1.514242 | 1.089966 / 0.981419 | 0.839172 / -0.246212 | 1.427611 / -1.109174 | 1.795946 / -1.476618 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2rep_lora_dseed11_seed1 | 67 | 2.003835 / 2.003469 | 2.003835 / 2.003469 | 2.010012 / 2.009814 | 1.527375 / 1.417112 | 1.144602 / 0.052800 | 1.185413 / -0.889813 | 1.458784 / -1.288251 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 41 | 1.635308 / 1.512676 | 1.635308 / 1.512676 | 1.635501 / 1.513316 | 1.522277 / 1.366138 | 1.234642 / 0.783139 | 0.950379 / 0.437180 | 0.666741 / 0.379333 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 37 | 1.748346 / 1.701611 | 1.748346 / 1.701611 | 1.749168 / 1.702058 | 1.604966 / 1.552350 | 1.201361 / 1.006459 | 0.871348 / 0.611666 | 0.620088 / 0.488360 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 41 | 1.450783 / 1.369313 | 1.450783 / 1.369313 | 1.452416 / 1.370018 | 1.341200 / 1.233198 | 1.106721 / 0.682287 | 0.831446 / 0.370195 | 0.580092 / 0.335949 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed11 | 41 | 1.731732 / 1.687749 | 1.731732 / 1.687749 | 1.731302 / 1.686453 | 1.584329 / 1.522090 | 1.260460 / 0.852828 | 0.913371 / 0.480829 | 0.640786 / 0.462774 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed12 | 41 | 1.576282 / 1.467540 | 1.576282 / 1.467540 | 1.576692 / 1.466304 | 1.438055 / 1.303245 | 1.169705 / 0.640831 | 0.867071 / 0.271156 | 0.583820 / 0.250004 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed11 | 41 | 4.394040 / 3.972011 | 4.394040 / 3.972011 | 4.395308 / 3.973597 | 4.074699 / 3.632036 | 3.342481 / 2.318953 | 2.566066 / 1.481602 | 1.945055 / 1.273961 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed12 | 40 | 3.337530 / 2.901434 | 3.337530 / 2.901434 | 3.339242 / 2.903505 | 3.029120 / 2.558998 | 2.224466 / 1.228811 | 1.457672 / 0.386763 | 0.816801 / 0.179279 |

#### leave_one_pool_seed_out_within_student: code / MBPP:e80dbf3c3adc, I1 — retrospective development-set

| Trajectory | Incident pairs | zero | constant | T_only | reuse_only | two_dimensional | saturation_p1 | saturation_p2 |
|---|---:|---|---|---|---|---|---|---|
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 2 | 0.053158 / -0.053158 | 0.053158 / -0.053158 | 0.013957 / 0.013957 | 0.012642 / 0.012642 | 0.013548 / 0.013548 | 0.014504 / 0.014504 | 0.014067 / 0.014067 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 2 | 0.029257 / -0.029257 | 0.029257 / -0.029257 | 0.038460 / 0.038460 | 0.037035 / 0.037035 | 0.038329 / 0.038329 | 0.039414 / 0.039414 | 0.038203 / 0.038203 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 2 | 0.028883 / -0.028883 | 0.028883 / -0.028883 | 0.038972 / 0.038972 | 0.038332 / 0.038332 | 0.038940 / 0.038940 | 0.040697 / 0.040697 | 0.040163 / 0.040163 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 2 | 0.055636 / -0.048411 | 0.055636 / -0.048411 | 0.044579 / 0.018779 | 0.048079 / 0.017265 | 0.044610 / 0.018587 | 0.041785 / 0.019108 | 0.039489 / 0.018151 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 2 | 0.037685 / -0.034890 | 0.037685 / -0.034890 | 0.032189 / 0.032189 | 0.030672 / 0.030672 | 0.031950 / 0.031950 | 0.032552 / 0.032552 | 0.031518 / 0.031518 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 2 | 0.055746 / -0.055746 | 0.055746 / -0.055746 | 0.011307 / 0.011307 | 0.009847 / 0.009847 | 0.011131 / 0.011131 | 0.012551 / 0.011844 | 0.014956 / 0.011164 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.035326 / -0.035326 | 0.035326 / -0.035326 | 0.040231 / 0.040231 | 0.020874 / 0.020874 | 0.040323 / 0.040323 | 0.030117 / 0.024142 | 0.018383 / 0.013090 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.024164 / -0.024164 | 0.024164 / -0.024164 | 0.050821 / 0.050821 | 0.034759 / 0.031458 | 0.050858 / 0.050858 | 0.044370 / 0.035851 | 0.032252 / 0.023810 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.032288 / -0.032288 | 0.032288 / -0.032288 | 0.043598 / 0.043598 | 0.025828 / 0.023313 | 0.043510 / 0.043510 | 0.035924 / 0.026999 | 0.025776 / 0.015295 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 4 | 0.018102 / -0.018102 | 0.018102 / -0.018102 | 0.053156 / 0.048059 | 0.040198 / 0.029135 | 0.052369 / 0.046957 | 0.044087 / 0.028253 | 0.034376 / 0.019982 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 4 | 0.021258 / -0.021258 | 0.021258 / -0.021258 | 0.048985 / 0.048985 | 0.030249 / 0.028755 | 0.047235 / 0.047235 | 0.032453 / 0.025774 | 0.023203 / 0.018087 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 4 | 0.020441 / -0.020441 | 0.020441 / -0.020441 | 0.051965 / 0.051965 | 0.031264 / 0.031264 | 0.051299 / 0.051299 | 0.029896 / 0.028571 | 0.024483 / 0.021269 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2rep_lora_dseed11_seed1 | 4 | 0.018974 / -0.018974 | 0.018974 / -0.018974 | 0.053253 / 0.047187 | 0.040294 / 0.028263 | 0.052466 / 0.046085 | 0.044184 / 0.027381 | 0.034473 / 0.019110 |
| gemma3-1b/gpt-5.6-luna_full_75_p2dev_lora_dseed11 | 2 | 0.285374 / -0.285374 | 0.285374 / -0.285374 | 0.203505 / -0.203505 | 0.200321 / -0.200321 | 0.204868 / -0.204868 | 0.183755 / -0.183755 | 0.108273 / -0.108273 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 4 | 0.184870 / -0.184870 | 0.184870 / -0.184870 | 0.119240 / -0.119240 | 0.111366 / -0.107000 | 0.120333 / -0.120333 | 0.102845 / -0.096938 | 0.058181 / -0.045982 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 4 | 0.229718 / -0.229718 | 0.229718 / -0.229718 | 0.160046 / -0.160046 | 0.146768 / -0.146768 | 0.161782 / -0.161782 | 0.137794 / -0.137794 | 0.090298 / -0.085825 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 4 | 0.148479 / -0.148479 | 0.148479 / -0.148479 | 0.075567 / -0.075567 | 0.062371 / -0.062352 | 0.076238 / -0.076238 | 0.050314 / -0.048367 | 0.015036 / 0.009327 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 2 | 0.066790 / -0.066790 | 0.066790 / -0.066790 | 0.021319 / 0.002451 | 0.016890 / -0.005042 | 0.020368 / -0.002798 | 0.014326 / -0.005167 | 0.013587 / -0.009657 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 2 | 0.049795 / -0.049795 | 0.049795 / -0.049795 | 0.037715 / 0.019676 | 0.033189 / 0.012146 | 0.036754 / 0.014667 | 0.030905 / 0.012446 | 0.029948 / 0.008096 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 2 | 0.050626 / -0.050626 | 0.050626 / -0.050626 | 0.046083 / 0.019510 | 0.041825 / 0.012586 | 0.045409 / 0.014399 | 0.039401 / 0.012823 | 0.038908 / 0.008856 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 2 | 0.064188 / -0.064188 | 0.064188 / -0.064188 | 0.017442 / 0.005257 | 0.021763 / -0.002420 | 0.018305 / 0.000009 | 0.024406 / -0.002522 | 0.025143 / -0.007121 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 2 | 0.066181 / -0.066181 | 0.066181 / -0.066181 | 0.008297 / 0.002791 | 0.004794 / -0.004794 | 0.007389 / -0.002417 | 0.004846 / -0.004846 | 0.009472 / -0.009472 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 2 | 0.058570 / -0.058570 | 0.058570 / -0.058570 | 0.011126 / 0.011126 | 0.009292 / 0.003418 | 0.005889 / 0.005889 | 0.011911 / 0.003346 | 0.012592 / -0.001027 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.026551 / -0.026551 | 0.026551 / -0.026551 | 0.052221 / 0.051729 | 0.038653 / 0.026484 | 0.049309 / 0.046306 | 0.037236 / 0.026965 | 0.031976 / 0.016166 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.034592 / -0.034592 | 0.034592 / -0.034592 | 0.050900 / 0.042112 | 0.040478 / 0.017487 | 0.046918 / 0.036658 | 0.036864 / 0.017530 | 0.031248 / 0.006367 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.030468 / -0.030468 | 0.030468 / -0.030468 | 0.048112 / 0.048112 | 0.031386 / 0.021988 | 0.042498 / 0.042498 | 0.026983 / 0.022211 | 0.021440 / 0.011054 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 4 | 0.034198 / -0.020386 | 0.034198 / -0.020386 | 0.057931 / 0.050954 | 0.048062 / 0.027975 | 0.054459 / 0.044689 | 0.042096 / 0.025570 | 0.037126 / 0.016676 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 4 | 0.027292 / -0.025520 | 0.027292 / -0.025520 | 0.062518 / 0.049200 | 0.048305 / 0.023111 | 0.059022 / 0.042920 | 0.045564 / 0.021975 | 0.039828 / 0.011686 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 4 | 0.024607 / -0.024607 | 0.024607 / -0.024607 | 0.052412 / 0.052412 | 0.028815 / 0.025264 | 0.046570 / 0.046570 | 0.026918 / 0.024446 | 0.020775 / 0.013067 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 4 | 0.104490 / -0.104490 | 0.104490 / -0.104490 | 0.033722 / -0.033722 | 0.024767 / -0.024767 | 0.039936 / -0.039936 | 0.034368 / -0.034368 | 0.022825 / -0.007398 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 4 | 0.124938 / -0.124938 | 0.124938 / -0.124938 | 0.050825 / -0.050825 | 0.044279 / -0.044279 | 0.057054 / -0.057054 | 0.051967 / -0.051967 | 0.025852 / -0.025852 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 4 | 0.098255 / -0.098255 | 0.098255 / -0.098255 | 0.020698 / -0.020698 | 0.015184 / -0.015184 | 0.026581 / -0.026581 | 0.020901 / -0.020901 | 0.007440 / 0.007440 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2rep_lora_dseed11_seed1 | 4 | 0.115679 / -0.115679 | 0.115679 / -0.115679 | 0.044911 / -0.044911 | 0.035956 / -0.035956 | 0.051126 / -0.051126 | 0.045557 / -0.045557 | 0.018587 / -0.018587 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.033651 / -0.033651 | 0.033651 / -0.033651 | 0.062152 / 0.062152 | 0.037269 / 0.037269 | 0.053324 / 0.053324 | 0.035258 / 0.035258 | 0.018674 / 0.018109 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.032523 / -0.032523 | 0.032523 / -0.032523 | 0.064409 / 0.064409 | 0.038534 / 0.038534 | 0.054312 / 0.054312 | 0.034170 / 0.034170 | 0.019959 / 0.017585 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.047256 / -0.047256 | 0.047256 / -0.047256 | 0.050100 / 0.050100 | 0.023349 / 0.023349 | 0.039896 / 0.039896 | 0.019881 / 0.019881 | 0.009345 / 0.002936 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed11 | 4 | 0.027915 / -0.027915 | 0.027915 / -0.027915 | 0.059931 / 0.056899 | 0.045628 / 0.035823 | 0.050887 / 0.044235 | 0.034440 / 0.027645 | 0.021160 / 0.013418 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed12 | 4 | 0.023444 / -0.023168 | 0.023444 / -0.023168 | 0.066730 / 0.066730 | 0.046574 / 0.046574 | 0.054797 / 0.054797 | 0.042324 / 0.042324 | 0.025109 / 0.025109 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed11 | 4 | 0.184047 / -0.184047 | 0.184047 / -0.184047 | 0.099913 / -0.099913 | 0.078977 / -0.078977 | 0.112475 / -0.112475 | 0.096338 / -0.096338 | 0.059754 / -0.059754 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed12 | 4 | 0.147953 / -0.147953 | 0.147953 / -0.147953 | 0.058786 / -0.058786 | 0.035419 / -0.032280 | 0.070623 / -0.070623 | 0.043133 / -0.041256 | 0.007961 / 0.007764 |

#### leave_one_pool_seed_out_within_student: code / MBPP:e80dbf3c3adc, I2 — retrospective development-set

| Trajectory | Incident pairs | zero | constant | T_only | reuse_only | two_dimensional | saturation_p1 | saturation_p2 |
|---|---:|---|---|---|---|---|---|---|
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 12 | 0.327212 / 0.327212 | 0.327212 / 0.327212 | 0.325294 / 0.325294 | 0.182189 / 0.143316 | 0.217734 / 0.166631 | 0.150698 / 0.078795 | 0.089027 / 0.050156 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 7 | 0.320606 / 0.320393 | 0.320606 / 0.320393 | 0.321403 / 0.321403 | 0.199060 / 0.136734 | 0.234278 / 0.148013 | 0.177801 / 0.069506 | 0.115771 / 0.066408 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 7 | 0.209762 / 0.201276 | 0.209762 / 0.201276 | 0.207958 / 0.199573 | 0.097780 / 0.011204 | 0.130031 / 0.006068 | 0.080041 / -0.080041 | 0.080678 / -0.080678 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2rep_lora_dseed11_seed1 | 12 | 0.330510 / 0.330510 | 0.330510 / 0.330510 | 0.328593 / 0.328593 | 0.184615 / 0.146614 | 0.220792 / 0.169929 | 0.151238 / 0.082093 | 0.089567 / 0.053455 |
| gemma3-1b/gpt-5.6-luna_full_75_p2dev_lora_dseed11 | 10 | 0.414856 / 0.414856 | 0.414856 / 0.414856 | 0.410302 / 0.410302 | 0.215932 / 0.215932 | 0.251683 / 0.251683 | 0.166007 / 0.146442 | 0.105956 / 0.094488 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 14 | 0.267436 / 0.267436 | 0.267436 / 0.267436 | 0.267401 / 0.267401 | 0.160166 / 0.094275 | 0.196106 / 0.108707 | 0.140226 / 0.033303 | 0.077398 / 0.021317 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 7 | 0.320606 / 0.320393 | 0.320606 / 0.320393 | 0.321403 / 0.321403 | 0.199060 / 0.136734 | 0.234278 / 0.148013 | 0.177801 / 0.069506 | 0.115771 / 0.066408 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 7 | 0.209762 / 0.201276 | 0.209762 / 0.201276 | 0.207958 / 0.199573 | 0.097780 / 0.011204 | 0.130031 / 0.006068 | 0.080041 / -0.080041 | 0.080678 / -0.080678 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 14 | 0.157782 / 0.156786 | 0.157782 / 0.156786 | 0.157811 / 0.156749 | 0.053898 / -0.020496 | 0.098817 / 0.058028 | 0.066675 / 0.019698 | 0.028293 / 0.009827 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 7 | 0.179026 / 0.176314 | 0.179026 / 0.176314 | 0.179589 / 0.177389 | 0.063871 / -0.002272 | 0.114237 / 0.074103 | 0.075583 / 0.034924 | 0.038546 / 0.026733 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 7 | 0.131157 / 0.129504 | 0.131157 / 0.129504 | 0.129237 / 0.127692 | 0.053824 / -0.053824 | 0.072924 / 0.015040 | 0.039909 / -0.026966 | 0.035804 / -0.035804 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 7 | 0.147539 / 0.145546 | 0.147539 / 0.145546 | 0.147634 / 0.145509 | 0.047819 / -0.031736 | 0.092302 / 0.046788 | 0.060595 / 0.008458 | 0.022213 / -0.001413 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 7 | 0.179026 / 0.176314 | 0.179026 / 0.176314 | 0.179589 / 0.177389 | 0.063871 / -0.002272 | 0.114237 / 0.074103 | 0.075583 / 0.034924 | 0.038546 / 0.026733 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 7 | 0.131157 / 0.129504 | 0.131157 / 0.129504 | 0.129237 / 0.127692 | 0.053824 / -0.053824 | 0.072924 / 0.015040 | 0.039909 / -0.026966 | 0.035804 / -0.035804 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2rep_lora_dseed11_seed1 | 7 | 0.168026 / 0.168026 | 0.168026 / 0.168026 | 0.167988 / 0.167988 | 0.059978 / -0.009256 | 0.105331 / 0.069268 | 0.072755 / 0.030938 | 0.034372 / 0.021067 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed11 | 7 | 0.272982 / 0.272982 | 0.272982 / 0.272982 | 0.272938 / 0.272938 | 0.122311 / 0.039335 | 0.189251 / 0.133317 | 0.139213 / 0.090840 | 0.086616 / 0.070098 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed12 | 7 | 0.225543 / 0.220521 | 0.225543 / 0.220521 | 0.226221 / 0.221815 | 0.073763 / -0.035589 | 0.129666 / 0.038876 | 0.082231 / -0.008630 | 0.039726 / -0.039726 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed11 | 7 | 0.272982 / 0.272982 | 0.272982 / 0.272982 | 0.272938 / 0.272938 | 0.122311 / 0.039335 | 0.189251 / 0.133317 | 0.139213 / 0.090840 | 0.086616 / 0.070098 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed12 | 7 | 0.225543 / 0.220521 | 0.225543 / 0.220521 | 0.226221 / 0.221815 | 0.073763 / -0.035589 | 0.129666 / 0.038876 | 0.082231 / -0.008630 | 0.039726 / -0.039726 |

#### leave_one_pool_seed_out_within_student: math / MATH-500:3816ece4b994, I1 — retrospective development-set

| Trajectory | Incident pairs | zero | constant | T_only | reuse_only | two_dimensional | saturation_p1 | saturation_p2 |
|---|---:|---|---|---|---|---|---|---|
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 2 | 0.042544 / -0.042544 | 0.042544 / -0.042544 | 0.022642 / 0.022642 | 0.023073 / 0.023073 | 0.022300 / 0.022300 | 0.024010 / 0.024010 | 0.029172 / 0.029172 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 2 | 0.034096 / -0.034096 | 0.034096 / -0.034096 | 0.031421 / 0.031421 | 0.031782 / 0.031782 | 0.031347 / 0.031347 | 0.032705 / 0.032705 | 0.037701 / 0.037701 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 2 | 0.041417 / -0.041417 | 0.041417 / -0.041417 | 0.024256 / 0.024256 | 0.025415 / 0.025415 | 0.024183 / 0.024183 | 0.026297 / 0.026297 | 0.031977 / 0.031977 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 2 | 0.040584 / -0.040584 | 0.040584 / -0.040584 | 0.024817 / 0.024817 | 0.026898 / 0.025023 | 0.024725 / 0.024725 | 0.025654 / 0.025654 | 0.030376 / 0.030376 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 2 | 0.041720 / -0.041720 | 0.041720 / -0.041720 | 0.023267 / 0.023006 | 0.026414 / 0.023289 | 0.023290 / 0.022874 | 0.024024 / 0.024024 | 0.028884 / 0.028884 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 2 | 0.050775 / -0.050775 | 0.050775 / -0.050775 | 0.014508 / 0.014508 | 0.014779 / 0.014779 | 0.014299 / 0.014299 | 0.015323 / 0.015323 | 0.020552 / 0.020552 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.025615 / -0.016418 | 0.025615 / -0.016418 | 0.057303 / 0.057303 | 0.039713 / 0.039713 | 0.057825 / 0.057825 | 0.042350 / 0.042350 | 0.034832 / 0.034832 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.020150 / -0.014581 | 0.020150 / -0.014581 | 0.058288 / 0.058288 | 0.040835 / 0.040835 | 0.058628 / 0.058628 | 0.044733 / 0.044733 | 0.036520 / 0.036520 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.018748 / -0.018748 | 0.018748 / -0.018748 | 0.055326 / 0.055326 | 0.036792 / 0.036792 | 0.055550 / 0.055550 | 0.040082 / 0.040082 | 0.031763 / 0.031763 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 4 | 0.009414 / -0.002425 | 0.009414 / -0.002425 | 0.063164 / 0.063164 | 0.046413 / 0.046413 | 0.063304 / 0.063304 | 0.043275 / 0.043275 | 0.041489 / 0.041489 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 4 | 0.012076 / -0.007548 | 0.012076 / -0.007548 | 0.058374 / 0.058374 | 0.040358 / 0.040358 | 0.056646 / 0.056646 | 0.036306 / 0.036306 | 0.032020 / 0.032020 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 4 | 0.016399 / -0.004475 | 0.016399 / -0.004475 | 0.063862 / 0.063862 | 0.045342 / 0.045342 | 0.062778 / 0.062778 | 0.041973 / 0.040762 | 0.037951 / 0.037504 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2rep_lora_dseed11_seed1 | 4 | 0.009893 / -0.000535 | 0.009893 / -0.000535 | 0.065053 / 0.065053 | 0.048303 / 0.048303 | 0.065194 / 0.065194 | 0.045165 / 0.045165 | 0.043378 / 0.043378 |
| gemma3-1b/gpt-5.6-luna_full_75_p2dev_lora_dseed11 | 2 | 0.293027 / -0.293027 | 0.293027 / -0.293027 | 0.211866 / -0.211866 | 0.205093 / -0.205093 | 0.211692 / -0.211692 | 0.169638 / -0.169638 | 0.066377 / -0.066377 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 4 | 0.184483 / -0.184483 | 0.184483 / -0.184483 | 0.132900 / -0.119420 | 0.130464 / -0.103975 | 0.132800 / -0.119281 | 0.112479 / -0.082385 | 0.049505 / -0.011059 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 4 | 0.280118 / -0.280118 | 0.280118 / -0.280118 | 0.221780 / -0.214732 | 0.220993 / -0.200660 | 0.222985 / -0.216445 | 0.210371 / -0.187398 | 0.158647 / -0.129240 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 4 | 0.207806 / -0.207806 | 0.207806 / -0.207806 | 0.150735 / -0.138992 | 0.149811 / -0.124824 | 0.151505 / -0.140083 | 0.135048 / -0.107394 | 0.078175 / -0.043467 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 2 | 0.024218 / -0.024218 | 0.024218 / -0.024218 | 0.017843 / 0.017843 | 0.014080 / 0.014080 | 0.010362 / 0.010362 | 0.011090 / 0.011090 | 0.009640 / 0.007865 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 2 | 0.024483 / -0.024483 | 0.024483 / -0.024483 | 0.017581 / 0.017581 | 0.013831 / 0.013831 | 0.010087 / 0.010087 | 0.010853 / 0.010853 | 0.007621 / 0.007621 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 2 | 0.019245 / -0.019245 | 0.019245 / -0.019245 | 0.023163 / 0.023163 | 0.019804 / 0.019804 | 0.015677 / 0.015677 | 0.017820 / 0.016903 | 0.016774 / 0.013835 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 2 | 0.024928 / -0.023659 | 0.024928 / -0.023659 | 0.018699 / 0.018699 | 0.020504 / 0.014791 | 0.019208 / 0.011099 | 0.022624 / 0.011724 | 0.021895 / 0.008339 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 2 | 0.017739 / -0.017739 | 0.017739 / -0.017739 | 0.024296 / 0.024296 | 0.020442 / 0.020442 | 0.016874 / 0.016874 | 0.017451 / 0.017451 | 0.014092 / 0.014092 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 2 | 0.022200 / -0.022200 | 0.022200 / -0.022200 | 0.020217 / 0.020217 | 0.016315 / 0.016315 | 0.012593 / 0.012593 | 0.015313 / 0.013237 | 0.014518 / 0.009907 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.018033 / -0.000256 | 0.018033 / -0.000256 | 0.047177 / 0.047177 | 0.032565 / 0.032565 | 0.039565 / 0.039565 | 0.030541 / 0.030541 | 0.023607 / 0.023607 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.018180 / -0.003187 | 0.018180 / -0.003187 | 0.043664 / 0.043664 | 0.029203 / 0.029203 | 0.035939 / 0.035939 | 0.026815 / 0.026815 | 0.019896 / 0.019896 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.011185 / -0.005503 | 0.011185 / -0.005503 | 0.042287 / 0.042287 | 0.027032 / 0.027032 | 0.034321 / 0.034321 | 0.024542 / 0.024542 | 0.017615 / 0.017615 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 4 | 0.020188 / 0.004930 | 0.020188 / 0.004930 | 0.047963 / 0.047963 | 0.036969 / 0.034853 | 0.039211 / 0.039211 | 0.034598 / 0.031500 | 0.031969 / 0.025326 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 4 | 0.016262 / 0.004489 | 0.016262 / 0.004489 | 0.049184 / 0.049184 | 0.034023 / 0.034023 | 0.040758 / 0.040758 | 0.030619 / 0.030619 | 0.027479 / 0.024298 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 4 | 0.016868 / 0.002131 | 0.016868 / 0.002131 | 0.048267 / 0.048267 | 0.032577 / 0.032577 | 0.040261 / 0.040261 | 0.030331 / 0.030331 | 0.022957 / 0.022957 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 4 | 0.066970 / -0.066970 | 0.066970 / -0.066970 | 0.034008 / -0.024282 | 0.034538 / -0.017642 | 0.040226 / -0.032964 | 0.040872 / -0.024856 | 0.022990 / -0.004261 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 4 | 0.089460 / -0.087935 | 0.089460 / -0.087935 | 0.058287 / -0.043603 | 0.060629 / -0.038950 | 0.064164 / -0.051961 | 0.067310 / -0.046411 | 0.051759 / -0.028408 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 4 | 0.071199 / -0.061757 | 0.071199 / -0.061757 | 0.038417 / -0.015299 | 0.040959 / -0.011042 | 0.044106 / -0.023360 | 0.045377 / -0.015991 | 0.027957 / 0.003917 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2rep_lora_dseed11_seed1 | 4 | 0.065659 / -0.065659 | 0.065659 / -0.065659 | 0.031872 / -0.022971 | 0.032402 / -0.016331 | 0.038091 / -0.031652 | 0.038736 / -0.023545 | 0.020854 / -0.002950 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.025700 / -0.025700 | 0.025700 / -0.025700 | 0.095316 / 0.095316 | 0.072025 / 0.072025 | 0.093607 / 0.093607 | 0.074364 / 0.074364 | 0.059113 / 0.059113 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.027921 / -0.025814 | 0.027921 / -0.025814 | 0.094765 / 0.094765 | 0.071104 / 0.071104 | 0.091986 / 0.091986 | 0.074148 / 0.074148 | 0.058462 / 0.058462 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.029460 / -0.029460 | 0.029460 / -0.029460 | 0.092167 / 0.092167 | 0.067106 / 0.067106 | 0.089644 / 0.089644 | 0.069168 / 0.069168 | 0.054014 / 0.054014 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed11 | 4 | 0.011673 / -0.011673 | 0.011673 / -0.011673 | 0.083786 / 0.083786 | 0.069220 / 0.069220 | 0.073947 / 0.073947 | 0.043906 / 0.043906 | 0.054628 / 0.054628 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed12 | 4 | 0.021528 / -0.018393 | 0.021528 / -0.018393 | 0.079795 / 0.079795 | 0.065967 / 0.065967 | 0.072707 / 0.072707 | 0.047701 / 0.047701 | 0.054196 / 0.054196 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed11 | 4 | 0.360675 / -0.360675 | 0.360675 / -0.360675 | 0.265980 / -0.265980 | 0.245080 / -0.227323 | 0.275741 / -0.275741 | 0.249706 / -0.222419 | 0.144075 / -0.112172 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed12 | 4 | 0.284463 / -0.284463 | 0.284463 / -0.284463 | 0.187073 / -0.187073 | 0.162334 / -0.144545 | 0.194104 / -0.194104 | 0.155460 / -0.127496 | 0.045450 / -0.004456 |

#### leave_one_pool_seed_out_within_student: math / MATH-500:3816ece4b994, I2 — retrospective development-set

| Trajectory | Incident pairs | zero | constant | T_only | reuse_only | two_dimensional | saturation_p1 | saturation_p2 |
|---|---:|---|---|---|---|---|---|---|
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 12 | 0.317105 / 0.312129 | 0.317105 / 0.312129 | 0.315249 / 0.310228 | 0.180258 / 0.122002 | 0.197079 / 0.099155 | 0.112471 / -0.023701 | 0.047873 / -0.043635 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 7 | 0.405852 / 0.405007 | 0.405852 / 0.405007 | 0.406348 / 0.405956 | 0.290701 / 0.229080 | 0.318302 / 0.220984 | 0.250280 / 0.131674 | 0.186129 / 0.134371 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 7 | 0.301677 / 0.294225 | 0.301677 / 0.294225 | 0.299974 / 0.292617 | 0.191437 / 0.111093 | 0.219848 / 0.087357 | 0.152821 / -0.009324 | 0.084096 / -0.002960 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2rep_lora_dseed11_seed1 | 12 | 0.319960 / 0.314262 | 0.319960 / 0.314262 | 0.318105 / 0.312361 | 0.182154 / 0.124135 | 0.198975 / 0.101287 | 0.114367 / -0.021569 | 0.047904 / -0.041502 |
| gemma3-1b/gpt-5.6-luna_full_75_p2dev_lora_dseed11 | 10 | 0.390802 / 0.390802 | 0.390802 / 0.390802 | 0.386287 / 0.386287 | 0.197462 / 0.185138 | 0.200185 / 0.175208 | 0.096140 / 0.028266 | 0.031359 / -0.020572 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 14 | 0.266911 / 0.257762 | 0.266911 / 0.257762 | 0.266955 / 0.257728 | 0.169595 / 0.078733 | 0.196485 / 0.046659 | 0.125761 / -0.058993 | 0.059696 / -0.058280 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 7 | 0.405852 / 0.405007 | 0.405852 / 0.405007 | 0.406348 / 0.405956 | 0.290701 / 0.229080 | 0.318302 / 0.220984 | 0.250280 / 0.131674 | 0.186129 / 0.134371 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 7 | 0.301677 / 0.294225 | 0.301677 / 0.294225 | 0.299974 / 0.292617 | 0.191437 / 0.111093 | 0.219848 / 0.087357 | 0.152821 / -0.009324 | 0.084096 / -0.002960 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 14 | 0.109151 / 0.099915 | 0.109151 / 0.099915 | 0.109208 / 0.099893 | 0.050465 / -0.009777 | 0.075994 / 0.027380 | 0.057337 / 0.011869 | 0.027622 / -0.003540 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 7 | 0.147173 / 0.143130 | 0.147173 / 0.143130 | 0.147510 / 0.143774 | 0.080107 / 0.034674 | 0.113737 / 0.077781 | 0.090047 / 0.057868 | 0.064501 / 0.047115 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 7 | 0.101038 / 0.097066 | 0.101038 / 0.097066 | 0.099888 / 0.095980 | 0.044140 / -0.014857 | 0.068760 / 0.021192 | 0.048926 / 0.000096 | 0.019156 / -0.011811 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 7 | 0.113935 / 0.105162 | 0.113935 / 0.105162 | 0.113992 / 0.105140 | 0.052562 / -0.004530 | 0.079063 / 0.032627 | 0.059434 / 0.017116 | 0.029719 / 0.001707 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 7 | 0.147173 / 0.143130 | 0.147173 / 0.143130 | 0.147510 / 0.143774 | 0.080107 / 0.034674 | 0.113737 / 0.077781 | 0.090047 / 0.057868 | 0.064501 / 0.047115 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 7 | 0.101038 / 0.097066 | 0.101038 / 0.097066 | 0.099888 / 0.095980 | 0.044140 / -0.014857 | 0.068760 / 0.021192 | 0.048926 / 0.000096 | 0.019156 / -0.011811 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2rep_lora_dseed11_seed1 | 7 | 0.104366 / 0.094668 | 0.104366 / 0.094668 | 0.104424 / 0.094646 | 0.048367 / -0.015024 | 0.072925 / 0.022133 | 0.055239 / 0.006622 | 0.025524 / -0.008787 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed11 | 7 | 0.523474 / 0.523474 | 0.523474 / 0.523474 | 0.523493 / 0.523424 | 0.338196 / 0.226937 | 0.385622 / 0.194917 | 0.284742 / 0.059346 | 0.184381 / 0.078701 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed12 | 7 | 0.478840 / 0.474326 | 0.478840 / 0.474326 | 0.479580 / 0.475739 | 0.273583 / 0.164535 | 0.309891 / 0.121686 | 0.188260 / -0.031937 | 0.068032 / -0.030095 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed11 | 7 | 0.523474 / 0.523474 | 0.523474 / 0.523474 | 0.523493 / 0.523424 | 0.338196 / 0.226937 | 0.385622 / 0.194917 | 0.284742 / 0.059346 | 0.184381 / 0.078701 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed12 | 7 | 0.478840 / 0.474326 | 0.478840 / 0.474326 | 0.479580 / 0.475739 | 0.273583 / 0.164535 | 0.309891 / 0.121686 | 0.188260 / -0.031937 | 0.068032 / -0.030095 |

#### leave_one_pool_seed_out_within_student: qa / 2WikiMultihopQA:075005b82489, I1 — retrospective development-set

| Trajectory | Incident pairs | zero | constant | T_only | reuse_only | two_dimensional | saturation_p1 | saturation_p2 |
|---|---:|---|---|---|---|---|---|---|
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 2 | 0.878766 / -0.878766 | 0.878766 / -0.878766 | 0.708551 / -0.708551 | 0.599658 / -0.599658 | 0.379710 / -0.360347 | 0.037788 / -0.004294 | 0.180528 / -0.180528 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 2 | 0.817480 / -0.817480 | 0.817480 / -0.817480 | 0.654192 / -0.654192 | 0.612589 / -0.544896 | 0.553266 / -0.306364 | 0.203365 / 0.054260 | 0.199653 / -0.122236 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 2 | 0.751121 / -0.751121 | 0.751121 / -0.751121 | 0.675263 / -0.582193 | 0.675301 / -0.469271 | 0.628941 / -0.230649 | 0.291831 / 0.127746 | 0.272547 / -0.043438 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 2 | 0.814274 / -0.814274 | 0.814274 / -0.814274 | 0.649121 / -0.649121 | 0.580427 / -0.540768 | 0.527400 / -0.300804 | 0.184137 / 0.048627 | 0.177559 / -0.133168 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 2 | 0.922466 / -0.922466 | 0.922466 / -0.922466 | 0.767079 / -0.767079 | 0.657912 / -0.657912 | 0.518346 / -0.424107 | 0.176960 / -0.067646 | 0.244512 / -0.244512 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 2 | 1.061255 / -1.061255 | 1.061255 / -1.061255 | 0.909152 / -0.909152 | 0.797939 / -0.797939 | 0.563258 / -0.563258 | 0.200965 / -0.200965 | 0.377874 / -0.377874 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.421393 / -0.319270 | 0.421393 / -0.319270 | 0.277744 / -0.113919 | 0.253846 / -0.079630 | 0.286541 / 0.277687 | 0.493111 / 0.453167 | 0.382350 / 0.147886 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.345975 / -0.344077 | 0.345975 / -0.344077 | 0.198597 / -0.142208 | 0.173824 / -0.108147 | 0.242461 / 0.242461 | 0.553247 / 0.460992 | 0.413996 / 0.150079 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.381101 / -0.381101 | 0.381101 / -0.381101 | 0.212476 / -0.176683 | 0.190329 / -0.145229 | 0.214455 / 0.214455 | 0.552981 / 0.408264 | 0.348557 / 0.094198 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 4 | 0.305248 / -0.305248 | 0.305248 / -0.305248 | 0.223930 / -0.223930 | 0.182604 / -0.177366 | 0.179979 / 0.179979 | 0.559177 / 0.351590 | 0.314511 / 0.048636 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 4 | 0.382704 / -0.372494 | 0.382704 / -0.372494 | 0.281853 / -0.232141 | 0.247757 / -0.187542 | 0.142480 / 0.142480 | 0.477909 / 0.313701 | 0.266996 / -0.011749 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 4 | 0.350068 / -0.298431 | 0.350068 / -0.298431 | 0.261087 / -0.171582 | 0.226952 / -0.126095 | 0.293196 / 0.202167 | 0.572536 / 0.370072 | 0.426233 / 0.048254 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2rep_lora_dseed11_seed1 | 4 | 0.336576 / -0.336576 | 0.336576 / -0.336576 | 0.255257 / -0.255257 | 0.208694 / -0.208694 | 0.148651 / 0.148651 | 0.537124 / 0.320262 | 0.310356 / 0.017308 |
| gemma3-1b/gpt-5.6-luna_full_75_p2dev_lora_dseed11 | 2 | 1.778698 / -1.778698 | 1.778698 / -1.778698 | 1.678072 / -1.678072 | 1.548441 / -1.548441 | 1.178264 / -1.178264 | 0.062376 / 0.062376 | 1.195956 / 1.195956 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 4 | 2.048540 / -2.048540 | 2.048540 / -2.048540 | 1.967874 / -1.967874 | 1.837729 / -1.837729 | 1.567201 / -1.567201 | 0.701358 / -0.701358 | 0.685421 / 0.028338 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 4 | 2.258046 / -2.258046 | 2.258046 / -2.258046 | 2.118834 / -2.118834 | 1.951286 / -1.951286 | 1.747259 / -1.747259 | 0.861953 / -0.861953 | 0.438796 / -0.154218 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 4 | 2.315301 / -2.315301 | 2.315301 / -2.315301 | 2.187566 / -2.187566 | 2.028236 / -2.028236 | 1.811206 / -1.811206 | 0.943891 / -0.943891 | 0.667014 / -0.287913 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 2 | 0.349384 / -0.349384 | 0.349384 / -0.349384 | 0.379965 / -0.379965 | 0.306431 / -0.306431 | 0.242996 / -0.126389 | 0.089914 / 0.089914 | 0.002872 / 0.001356 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 2 | 0.307738 / -0.296221 | 0.307738 / -0.296221 | 0.326271 / -0.326271 | 0.301565 / -0.252792 | 0.264709 / -0.071960 | 0.142545 / 0.142545 | 0.053464 / 0.053464 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 2 | 0.267430 / -0.235745 | 0.267430 / -0.235745 | 0.271175 / -0.264165 | 0.263803 / -0.189733 | 0.237258 / -0.006766 | 0.204222 / 0.204222 | 0.118437 / 0.118437 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 2 | 0.349664 / -0.349664 | 0.349664 / -0.349664 | 0.381622 / -0.381622 | 0.307847 / -0.307847 | 0.216713 / -0.126578 | 0.081900 / 0.081900 | 0.029608 / -0.009950 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 2 | 0.356636 / -0.356636 | 0.356636 / -0.356636 | 0.388645 / -0.388645 | 0.315368 / -0.315368 | 0.243423 / -0.135907 | 0.075095 / 0.075095 | 0.015189 / -0.015189 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 2 | 0.330428 / -0.330428 | 0.330428 / -0.330428 | 0.363146 / -0.363146 | 0.288890 / -0.288890 | 0.247752 / -0.107976 | 0.105389 / 0.105389 | 0.014637 / 0.014637 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.206487 / -0.077534 | 0.206487 / -0.077534 | 0.221100 / -0.098424 | 0.178751 / -0.037864 | 0.187476 / 0.187476 | 0.321459 / 0.313306 | 0.212466 / 0.147904 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.243697 / -0.146586 | 0.243697 / -0.146586 | 0.262477 / -0.172309 | 0.216762 / -0.109671 | 0.114254 / 0.108598 | 0.274774 / 0.270292 | 0.206414 / 0.100508 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.186364 / -0.186364 | 0.186364 / -0.186364 | 0.212006 / -0.212006 | 0.159696 / -0.149325 | 0.073084 / 0.072805 | 0.341847 / 0.218707 | 0.210526 / 0.047398 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 4 | 0.161557 / -0.137621 | 0.161557 / -0.137621 | 0.259842 / -0.259842 | 0.186385 / -0.171826 | 0.076237 / -0.000263 | 0.347880 / 0.170748 | 0.280174 / 0.012868 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 4 | 0.306617 / -0.213365 | 0.306617 / -0.213365 | 0.366271 / -0.296386 | 0.311555 / -0.220133 | 0.180640 / -0.028395 | 0.150969 / 0.093741 | 0.081112 / -0.070094 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 4 | 0.236849 / -0.147146 | 0.236849 / -0.147146 | 0.277564 / -0.205187 | 0.225354 / -0.131054 | 0.132200 / 0.071718 | 0.320985 / 0.209495 | 0.240695 / 0.025097 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 4 | 1.426980 / -1.426980 | 1.426980 / -1.426980 | 1.548221 / -1.548221 | 1.483365 / -1.483365 | 1.290722 / -1.290722 | 0.756290 / -0.746355 | 0.387338 / -0.323373 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 4 | 1.746421 / -1.746421 | 1.746421 / -1.746421 | 1.828766 / -1.828766 | 1.757646 / -1.757646 | 1.562954 / -1.562954 | 1.069697 / -1.069697 | 0.717575 / -0.670009 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 4 | 1.220446 / -1.220446 | 1.220446 / -1.220446 | 1.278893 / -1.278893 | 1.193642 / -1.193642 | 1.000054 / -1.000054 | 0.443181 / -0.443181 | 0.062793 / -0.010962 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2rep_lora_dseed11_seed1 | 4 | 1.533802 / -1.533802 | 1.533802 / -1.533802 | 1.655044 / -1.655044 | 1.590187 / -1.590187 | 1.397544 / -1.397544 | 0.853178 / -0.853178 | 0.481243 / -0.430196 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 4 | 0.477527 / -0.301450 | 0.477527 / -0.301450 | 0.292932 / -0.037564 | 0.236981 / 0.042600 | 0.407473 / 0.407473 | 0.579200 / 0.579200 | 0.393299 / 0.239839 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 4 | 0.668078 / -0.626214 | 0.668078 / -0.626214 | 0.521828 / -0.376466 | 0.498538 / -0.291889 | 0.403291 / 0.050105 | 0.318271 / 0.282782 | 0.114922 / -0.058793 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 4 | 0.538666 / -0.473831 | 0.538666 / -0.473831 | 0.357152 / -0.210941 | 0.319730 / -0.135670 | 0.249080 / 0.228634 | 0.452347 / 0.420595 | 0.155299 / 0.074083 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed11 | 4 | 0.272893 / -0.251074 | 0.272893 / -0.251074 | 0.268244 / -0.244564 | 0.176142 / -0.117784 | 0.159670 / 0.145776 | 0.522196 / 0.392610 | 0.375185 / 0.091083 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed12 | 4 | 0.606045 / -0.552509 | 0.606045 / -0.552509 | 0.548514 / -0.472443 | 0.438992 / -0.323554 | 0.274496 / -0.066273 | 0.387251 / 0.242929 | 0.177217 / -0.069733 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed11 | 4 | 2.906997 / -2.906997 | 2.906997 / -2.906997 | 2.900539 / -2.900539 | 2.687270 / -2.687270 | 2.513327 / -2.513327 | 1.603643 / -1.603643 | 0.924355 / -0.924355 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed12 | 4 | 2.300999 / -2.300999 | 2.300999 / -2.300999 | 2.221585 / -2.221585 | 1.921257 / -1.921257 | 1.818717 / -1.818717 | 0.643364 / -0.643364 | 0.486771 / 0.363328 |

#### leave_one_pool_seed_out_within_student: qa / 2WikiMultihopQA:075005b82489, I2 — retrospective development-set

| Trajectory | Incident pairs | zero | constant | T_only | reuse_only | two_dimensional | saturation_p1 | saturation_p2 |
|---|---:|---|---|---|---|---|---|---|
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 12 | 4.384498 / 4.384498 | 4.384498 / 4.384498 | 4.382142 / 4.382142 | 3.910823 / 3.886652 | 2.534483 / 1.828560 | 1.082034 / 0.270487 | 0.438322 / -0.339681 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 7 | 3.644467 / 3.644467 | 3.644467 / 3.644467 | 3.646486 / 3.646486 | 3.063906 / 2.965274 | 2.200495 / 0.718613 | 1.263297 / -0.395226 | 0.618823 / -0.618823 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 7 | 3.947265 / 3.945948 | 3.947265 / 3.945948 | 3.944103 / 3.942964 | 3.418725 / 3.312429 | 2.566010 / 1.157137 | 1.489352 / 0.059094 | 0.489015 / -0.138049 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2rep_lora_dseed11_seed1 | 12 | 4.318367 / 4.318367 | 4.318367 / 4.318367 | 4.316010 / 4.316010 | 3.843965 / 3.820521 | 2.469991 / 1.762429 | 1.025874 / 0.204356 | 0.496122 / -0.405812 |
| gemma3-1b/gpt-5.6-luna_full_75_p2dev_lora_dseed11 | 10 | 5.348755 / 5.348755 | 5.348755 / 5.348755 | 5.343158 / 5.343158 | 4.810226 / 4.810226 | 2.773483 / 2.773483 | 0.899257 / 0.899257 | 0.340328 / -0.113587 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 14 | 3.639060 / 3.639060 | 3.639060 / 3.639060 | 3.639017 / 3.639017 | 3.211085 / 3.170273 | 2.308491 / 1.096931 | 1.164452 / -0.235318 | 0.557860 / -0.557860 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 7 | 3.644467 / 3.644467 | 3.644467 / 3.644467 | 3.646486 / 3.646486 | 3.063906 / 2.965274 | 2.200495 / 0.718613 | 1.263297 / -0.395226 | 0.618823 / -0.618823 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 7 | 3.947265 / 3.945948 | 3.947265 / 3.945948 | 3.944103 / 3.942964 | 3.418725 / 3.312429 | 2.566010 / 1.157137 | 1.489352 / 0.059094 | 0.489015 / -0.138049 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 14 | 2.090873 / 2.090873 | 2.090873 / 2.090873 | 2.090937 / 2.090937 | 2.216258 / 2.216258 | 1.474101 / 0.690869 | 1.010953 / 0.004842 | 0.383225 / -0.229254 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 7 | 2.621309 / 2.621309 | 2.621309 / 2.621309 | 2.620115 / 2.620115 | 2.646163 / 2.646163 | 1.889204 / 1.250343 | 1.298438 / 0.507568 | 0.693673 / 0.334149 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 7 | 1.725509 / 1.719817 | 1.725509 / 1.719817 | 1.726955 / 1.721183 | 1.676157 / 1.660663 | 1.092386 / 0.173402 | 0.619912 / -0.606716 | 0.800968 / -0.800968 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 7 | 2.052567 / 2.052567 | 2.052567 / 2.052567 | 2.052630 / 2.052630 | 2.177951 / 2.177951 | 1.421574 / 0.652563 | 0.958427 / -0.033464 | 0.334405 / -0.267561 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 7 | 2.621309 / 2.621309 | 2.621309 / 2.621309 | 2.620115 / 2.620115 | 2.646163 / 2.646163 | 1.889204 / 1.250343 | 1.298438 / 0.507568 | 0.693673 / 0.334149 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 7 | 1.725509 / 1.719817 | 1.725509 / 1.719817 | 1.726955 / 1.721183 | 1.676157 / 1.660663 | 1.092386 / 0.173402 | 0.619912 / -0.606716 | 0.800968 / -0.800968 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2rep_lora_dseed11_seed1 | 7 | 2.129180 / 2.129180 | 2.129180 / 2.129180 | 2.129243 / 2.129243 | 2.254565 / 2.254565 | 1.526627 / 0.729176 | 1.063480 / 0.043149 | 0.432044 / -0.190948 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed11 | 7 | 5.078952 / 5.078952 | 5.078952 / 5.078952 | 5.078949 / 5.078949 | 4.619822 / 4.590340 | 3.605032 / 2.417992 | 2.508878 / 1.375624 | 1.542275 / 1.082837 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed12 | 7 | 3.634667 / 3.634667 | 3.634667 / 3.634667 | 3.635819 / 3.635819 | 2.887490 / 2.793883 | 1.759315 / 0.030452 | 1.262530 / -1.262530 | 1.698002 / -1.698002 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed11 | 7 | 5.078952 / 5.078952 | 5.078952 / 5.078952 | 5.078949 / 5.078949 | 4.619822 / 4.590340 | 3.605032 / 2.417992 | 2.508878 / 1.375624 | 1.542275 / 1.082837 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed12 | 7 | 3.634667 / 3.634667 | 3.634667 / 3.634667 | 3.635819 / 3.635819 | 2.887490 / 2.793883 | 1.759315 / 0.030452 | 1.262530 / -1.262530 | 1.698002 / -1.698002 |

### Included trajectory accounting

| Trajectory | Registered U | Registered D_U | Data seed | Training seed | Checkpoints with T > 0 | Actual T range | Baseline |
|---|---:|---:|---:|---:|---:|---|---|
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 200 | 52958 | 31 | 0 | 4 | [0, 296771] | own update 0 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 200 | 52964 | 32 | 0 | 4 | [0, 302781] | own update 0 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 200 | 51800 | 33 | 0 | 4 | [0, 293958] | own update 0 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 200 | 53528 | 34 | 0 | 4 | [0, 301610] | own update 0 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 200 | 53332 | 35 | 0 | 4 | [0, 297635] | own update 0 |
| gemma3-270m/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 200 | 53411 | 36 | 0 | 4 | [0, 301189] | own update 0 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 375 | 99504 | 21 | 0 | 5 | [0, 300036] | own update 0 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 375 | 99380 | 22 | 0 | 5 | [0, 299401] | own update 0 |
| gemma3-270m/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 375 | 102522 | 23 | 0 | 5 | [0, 306590] | own update 0 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 450 | 118791 | 11 | 0 | 5 | [0, 295139] | own update 0 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 450 | 117848 | 12 | 0 | 5 | [0, 296387] | own update 0 |
| gemma3-270m/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 450 | 120199 | 13 | 0 | 5 | [0, 299780] | own update 0 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 75 | 20010 | 11 | 0 | 5 | [0, 294023] | own update 0 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 75 | 19418 | 12 | 0 | 5 | [0, 292678] | own update 0 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 75 | 20552 | 13 | 0 | 5 | [0, 303908] | own update 0 |
| gemma3-270m/gpt-5.6-luna_full_75_p2v2rep_lora_dseed11_seed1 | 75 | 20010 | 11 | 1 | 5 | [0, 294023] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed31 | 200 | 52958 | 31 | 0 | 4 | [0, 296771] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed32 | 200 | 52964 | 32 | 0 | 4 | [0, 302781] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed33 | 200 | 51800 | 33 | 0 | 4 | [0, 293958] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed34 | 200 | 53528 | 34 | 0 | 4 | [0, 301610] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed35 | 200 | 53332 | 35 | 0 | 4 | [0, 297635] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_lora_dseed36 | 200 | 53411 | 36 | 0 | 4 | [0, 301189] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_throughput_lora_dseed31 | 200 | 52958 | 31 | 0 | 1 | [0, 51385] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 375 | 99504 | 21 | 0 | 5 | [0, 300036] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 375 | 99380 | 22 | 0 | 5 | [0, 299401] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 375 | 102522 | 23 | 0 | 5 | [0, 306590] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed11 | 450 | 118791 | 11 | 0 | 5 | [0, 295139] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed12 | 450 | 117848 | 12 | 0 | 5 | [0, 296387] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed13 | 450 | 120199 | 13 | 0 | 5 | [0, 299780] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_450_p2v2rep_lora_dseed11_seed1 | 450 | 118791 | 11 | 1 | 5 | [0, 295139] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_75_p2dev_lora_dseed11 | 75 | 20010 | 11 | 0 | 3 | [0, 300150] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed11 | 75 | 20010 | 11 | 0 | 5 | [0, 294023] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed12 | 75 | 19418 | 12 | 0 | 5 | [0, 292678] | own update 0 |
| gemma3-1b/gpt-5.6-luna_full_75_p2v2_lora_dseed13 | 75 | 20552 | 13 | 0 | 5 | [0, 303908] | own update 0 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | 375 | 99504 | 21 | 0 | 5 | [0, 300036] | own update 0 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | 375 | 99380 | 22 | 0 | 5 | [0, 299401] | own update 0 |
| gemma3-4b/gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | 375 | 102522 | 23 | 0 | 5 | [0, 306590] | own update 0 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed11 | 450 | 118791 | 11 | 0 | 5 | [0, 295139] | own update 0 |
| gemma3-4b/gpt-5.6-luna_full_450_p2v2test_lora_dseed12 | 450 | 117848 | 12 | 0 | 5 | [0, 296387] | own update 0 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed11 | 75 | 20010 | 11 | 0 | 5 | [0, 294023] | own update 0 |
| gemma3-4b/gpt-5.6-luna_full_75_p2v2test_lora_dseed12 | 75 | 19418 | 12 | 0 | 5 | [0, 292678] | own update 0 |

## Retrospective development-set identifiability

For each fixed p=1 and p=2, profile tau on a fixed 161-point logarithmic grid from min positive T to max T in the fit's training set. At each tau refit a,b by least squares to delta. Calibrate a profile-loss confidence set with 5000 whole-trajectory bootstrap resamples, seed 0: within each resample refit a,b at every grid tau, and calculate excess mean squared loss at the original fitted tau over the resample optimum. The 95th percentile of this excess sets the cutoff for the observed profile. Report the envelope of all accepted grid points as the approximate 95% profile interval; disconnected components and the full grid are recorded in JSON. This diagnostic uses delta squared loss only for parameter profiling, not for candidate performance ranking; no checkpoint independence assumption is used. Boundary contact means tau is not identified.

The observed range means the positive supervised-token range in that fit's training data; tau cannot be zero. A profile set touching either range boundary is reported as not identified. No statement about an asymptotic floor or an optimal budget is made. All-data fits below are descriptive; fold fits are the actual functions used for intervention predictions.

| Split / fold | Capability / distribution | Candidate | Fitted tau | Profile 95% interval, clusters | Observed positive T range | Lower / upper boundary | Identifiability |
|---|---|---|---:|---|---|---|---|
| descriptive_all_data / all | code / MBPP:e80dbf3c3adc | saturation_p1 | 107907.2 | [35011.0, 306590.0] (n=41 trajectories) | [35011, 306590] | True / True | not identified |
| descriptive_all_data / all | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 39556.0] (n=41 trajectories) | [35011, 306590] | True / False | not identified |
| descriptive_all_data / all | math / MATH-500:3816ece4b994 | saturation_p1 | 96812.8 | [37978.9, 306590.0] (n=41 trajectories) | [35011, 306590] | False / True | not identified |
| descriptive_all_data / all | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=41 trajectories) | [35011, 306590] | True / False | not identified |
| descriptive_all_data / all | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=41 trajectories) | [35011, 306590] | True / False | not identified |
| descriptive_all_data / all | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=41 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=11 | code / MBPP:e80dbf3c3adc | saturation_p1 | 86859.0 | [35011.0, 306590.0] (n=14 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=11 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 38497.5] (n=14 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=11 | math / MATH-500:3816ece4b994 | saturation_p1 | 90465.7 | [35011.0, 306590.0] (n=14 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=11 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 306590.0] (n=14 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=11 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=14 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=11 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=14 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=12 | code / MBPP:e80dbf3c3adc | saturation_p1 | 97581.3 | [35536.0, 306590.0] (n=16 trajectories) | [35536, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=12 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35536.0 | [35536.0, 35536.0] (n=16 trajectories) | [35536, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=12 | math / MATH-500:3816ece4b994 | saturation_p1 | 96275.9 | [35536.0, 306590.0] (n=16 trajectories) | [35536, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=12 | math / MATH-500:3816ece4b994 | saturation_p2 | 35536.0 | [35536.0, 35536.0] (n=16 trajectories) | [35536, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=12 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35536.0 | [35536.0, 37001.3] (n=16 trajectories) | [35536, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=12 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35536.0 | [35536.0, 35536.0] (n=16 trajectories) | [35536, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=13 | code / MBPP:e80dbf3c3adc | saturation_p1 | 90465.7 | [35011.0, 306590.0] (n=16 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=13 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=16 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=13 | math / MATH-500:3816ece4b994 | saturation_p1 | 94222.2 | [35011.0, 306590.0] (n=16 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=13 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 306590.0] (n=16 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=13 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 36962.7] (n=16 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=13 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=16 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=21 | code / MBPP:e80dbf3c3adc | saturation_p1 | 67129.1 | [35011.0, 243463.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=21 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=21 | math / MATH-500:3816ece4b994 | saturation_p1 | 73814.0 | [35011.0, 185626.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=21 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 306590.0] (n=17 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=21 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=21 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=22 | code / MBPP:e80dbf3c3adc | saturation_p1 | 64452.7 | [35011.0, 227501.6] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=22 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=22 | math / MATH-500:3816ece4b994 | saturation_p1 | 72819.7 | [35011.0, 180658.9] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=22 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 157747.4] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=22 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=22 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=23 | code / MBPP:e80dbf3c3adc | saturation_p1 | 66952.3 | [35011.0, 244843.4] (n=17 trajectories) | [35011, 303908] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=23 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 303908] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=23 | math / MATH-500:3816ece4b994 | saturation_p1 | 73591.4 | [35011.0, 184376.4] (n=17 trajectories) | [35011, 303908] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=23 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 165492.5] (n=17 trajectories) | [35011, 303908] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=23 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 303908] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=23 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 303908] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=31 | code / MBPP:e80dbf3c3adc | saturation_p1 | 70871.1 | [35011.0, 306590.0] (n=16 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=31 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=16 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=31 | math / MATH-500:3816ece4b994 | saturation_p1 | 77928.7 | [35011.0, 240183.5] (n=16 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=31 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 64452.7] (n=16 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=31 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=16 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=31 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=16 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=32 | code / MBPP:e80dbf3c3adc | saturation_p1 | 68045.6 | [35011.0, 306590.0] (n=17 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=32 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=32 | math / MATH-500:3816ece4b994 | saturation_p1 | 77928.7 | [35011.0, 236948.3] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=32 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 54772.8] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=32 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=32 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=33 | code / MBPP:e80dbf3c3adc | saturation_p1 | 68045.6 | [35011.0, 306590.0] (n=17 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=33 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=33 | math / MATH-500:3816ece4b994 | saturation_p1 | 77928.7 | [35011.0, 230607.9] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=33 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 70871.1] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=33 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=33 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=34 | code / MBPP:e80dbf3c3adc | saturation_p1 | 69916.5 | [35011.0, 306590.0] (n=17 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=34 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=34 | math / MATH-500:3816ece4b994 | saturation_p1 | 77928.7 | [35011.0, 233756.6] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=34 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 61049.6] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=34 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=34 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=35 | code / MBPP:e80dbf3c3adc | saturation_p1 | 69916.5 | [35011.0, 306590.0] (n=17 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=35 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=35 | math / MATH-500:3816ece4b994 | saturation_p1 | 77928.7 | [35011.0, 233756.6] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=35 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 65332.8] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=35 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=35 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=36 | code / MBPP:e80dbf3c3adc | saturation_p1 | 69916.5 | [35011.0, 306590.0] (n=17 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=36 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=36 | math / MATH-500:3816ece4b994 | saturation_p1 | 78992.7 | [35011.0, 236948.3] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=36 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 73814.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=36 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-1b/seed=36 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=17 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=11 | code / MBPP:e80dbf3c3adc | saturation_p1 | 60227.2 | [35011.0, 113922.4] (n=13 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=11 | code / MBPP:e80dbf3c3adc | saturation_p2 | 49812.3 | [39023.1, 61049.6] (n=13 trajectories) | [35011, 306590] | False / False | identified inside the observed T range (grid profile) |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=11 | math / MATH-500:3816ece4b994 | saturation_p1 | 35011.0 | [35011.0, 306590.0] (n=13 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=11 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=13 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=11 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 36464.8] (n=13 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=11 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35489.0] (n=13 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=12 | code / MBPP:e80dbf3c3adc | saturation_p1 | 66029.3 | [35536.0, 134818.7] (n=14 trajectories) | [35536, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=12 | code / MBPP:e80dbf3c3adc | saturation_p2 | 49096.7 | [40659.5, 58491.5] (n=14 trajectories) | [35536, 306590] | False / False | identified inside the observed T range (grid profile) |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=12 | math / MATH-500:3816ece4b994 | saturation_p1 | 35536.0 | [35536.0, 36017.9] (n=14 trajectories) | [35536, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=12 | math / MATH-500:3816ece4b994 | saturation_p2 | 35536.0 | [35536.0, 35536.0] (n=14 trajectories) | [35536, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=12 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35536.0 | [35536.0, 35536.0] (n=14 trajectories) | [35536, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=12 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35536.0 | [35536.0, 35536.0] (n=14 trajectories) | [35536, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=13 | code / MBPP:e80dbf3c3adc | saturation_p1 | 66224.8 | [35011.0, 168814.9] (n=14 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=13 | code / MBPP:e80dbf3c3adc | saturation_p2 | 47182.2 | [39023.1, 56278.7] (n=14 trajectories) | [35011, 306590] | False / False | identified inside the observed T range (grid profile) |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=13 | math / MATH-500:3816ece4b994 | saturation_p1 | 35011.0 | [35011.0, 306590.0] (n=14 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=13 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=14 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=13 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=14 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=13 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=14 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=21 | code / MBPP:e80dbf3c3adc | saturation_p1 | 85689.0 | [35011.0, 306590.0] (n=15 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=21 | code / MBPP:e80dbf3c3adc | saturation_p2 | 50492.5 | [41761.0, 60227.2] (n=15 trajectories) | [35011, 306590] | False / False | identified inside the observed T range (grid profile) |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=21 | math / MATH-500:3816ece4b994 | saturation_p1 | 35011.0 | [35011.0, 306590.0] (n=15 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=21 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=21 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=21 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=22 | code / MBPP:e80dbf3c3adc | saturation_p1 | 83396.1 | [35011.0, 306590.0] (n=15 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=22 | code / MBPP:e80dbf3c3adc | saturation_p2 | 49141.4 | [40643.5, 57826.1] (n=15 trajectories) | [35011, 306590] | False / False | identified inside the observed T range (grid profile) |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=22 | math / MATH-500:3816ece4b994 | saturation_p1 | 35011.0 | [35011.0, 61049.6] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=22 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=22 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=22 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=23 | code / MBPP:e80dbf3c3adc | saturation_p1 | 84233.6 | [35011.0, 303908.0] (n=15 trajectories) | [35011, 303908] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=23 | code / MBPP:e80dbf3c3adc | saturation_p2 | 50417.7 | [41731.2, 60095.1] (n=15 trajectories) | [35011, 303908] | False / False | identified inside the observed T range (grid profile) |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=23 | math / MATH-500:3816ece4b994 | saturation_p1 | 35011.0 | [35011.0, 53216.5] (n=15 trajectories) | [35011, 303908] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=23 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 303908] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=23 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 303908] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=23 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 303908] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=31 | code / MBPP:e80dbf3c3adc | saturation_p1 | 76879.0 | [35011.0, 260544.2] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=31 | code / MBPP:e80dbf3c3adc | saturation_p2 | 49812.3 | [41198.5, 59416.0] (n=15 trajectories) | [35011, 306590] | False / False | identified inside the observed T range (grid profile) |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=31 | math / MATH-500:3816ece4b994 | saturation_p1 | 35011.0 | [35011.0, 45301.1] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=31 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=31 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=31 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=32 | code / MBPP:e80dbf3c3adc | saturation_p1 | 82272.8 | [35011.0, 306590.0] (n=15 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=32 | code / MBPP:e80dbf3c3adc | saturation_p2 | 51181.9 | [42909.2, 60227.2] (n=15 trajectories) | [35011, 306590] | False / False | identified inside the observed T range (grid profile) |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=32 | math / MATH-500:3816ece4b994 | saturation_p1 | 35011.0 | [35011.0, 45301.1] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=32 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=32 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=32 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=33 | code / MBPP:e80dbf3c3adc | saturation_p1 | 81164.6 | [35011.0, 306590.0] (n=15 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=33 | code / MBPP:e80dbf3c3adc | saturation_p2 | 51181.9 | [42909.2, 60227.2] (n=15 trajectories) | [35011, 306590] | False / False | identified inside the observed T range (grid profile) |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=33 | math / MATH-500:3816ece4b994 | saturation_p1 | 35011.0 | [35011.0, 46546.6] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=33 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=33 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=33 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=34 | code / MBPP:e80dbf3c3adc | saturation_p1 | 75843.4 | [35011.0, 253572.4] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=34 | code / MBPP:e80dbf3c3adc | saturation_p2 | 49812.3 | [41761.0, 58615.6] (n=15 trajectories) | [35011, 306590] | False / False | identified inside the observed T range (grid profile) |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=34 | math / MATH-500:3816ece4b994 | saturation_p1 | 35011.0 | [35011.0, 42909.2] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=34 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=34 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=34 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=35 | code / MBPP:e80dbf3c3adc | saturation_p1 | 76879.0 | [35011.0, 267707.7] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=35 | code / MBPP:e80dbf3c3adc | saturation_p2 | 49812.3 | [41761.0, 59416.0] (n=15 trajectories) | [35011, 306590] | False / False | identified inside the observed T range (grid profile) |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=35 | math / MATH-500:3816ece4b994 | saturation_p1 | 35011.0 | [35011.0, 42909.2] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=35 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=35 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=35 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=36 | code / MBPP:e80dbf3c3adc | saturation_p1 | 76879.0 | [35011.0, 257034.7] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=36 | code / MBPP:e80dbf3c3adc | saturation_p2 | 50492.5 | [41761.0, 59416.0] (n=15 trajectories) | [35011, 306590] | False / False | identified inside the observed T range (grid profile) |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=36 | math / MATH-500:3816ece4b994 | saturation_p1 | 35011.0 | [35011.0, 44088.9] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=36 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=36 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-270m/seed=36 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=15 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=11 | code / MBPP:e80dbf3c3adc | saturation_p1 | 35011.0 | [35011.0, 64452.7] (n=5 trajectories; BELOW SIX) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=11 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 40096.1] (n=5 trajectories; BELOW SIX) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=11 | math / MATH-500:3816ece4b994 | saturation_p1 | 218431.5 | [35011.0, 306590.0] (n=5 trajectories; BELOW SIX) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=11 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=5 trajectories; BELOW SIX) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=11 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=5 trajectories; BELOW SIX) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=11 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=5 trajectories; BELOW SIX) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=12 | code / MBPP:e80dbf3c3adc | saturation_p1 | 35536.0 | [35536.0, 62566.2] (n=5 trajectories; BELOW SIX) | [35536, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=12 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35536.0 | [35536.0, 36017.9] (n=5 trajectories; BELOW SIX) | [35536, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=12 | math / MATH-500:3816ece4b994 | saturation_p1 | 148147.8 | [35536.0, 306590.0] (n=5 trajectories; BELOW SIX) | [35536, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=12 | math / MATH-500:3816ece4b994 | saturation_p2 | 35536.0 | [35536.0, 36017.9] (n=5 trajectories; BELOW SIX) | [35536, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=12 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 40115.5 | [35536.0, 64274.4] (n=5 trajectories; BELOW SIX) | [35536, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=12 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35536.0 | [35536.0, 35536.0] (n=5 trajectories; BELOW SIX) | [35536, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=21 | code / MBPP:e80dbf3c3adc | saturation_p1 | 35011.0 | [35011.0, 306590.0] (n=6 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=21 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 40643.5] (n=6 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=21 | math / MATH-500:3816ece4b994 | saturation_p1 | 74821.8 | [35011.0, 306590.0] (n=6 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=21 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 35489.0] (n=6 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=21 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35973.6] (n=6 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=21 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=6 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=22 | code / MBPP:e80dbf3c3adc | saturation_p1 | 35011.0 | [35011.0, 306590.0] (n=6 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=22 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 44088.9] (n=6 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=22 | math / MATH-500:3816ece4b994 | saturation_p1 | 75843.4 | [35011.0, 306590.0] (n=6 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=22 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 36464.8] (n=6 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=22 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 36464.8] (n=6 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=22 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=6 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=23 | code / MBPP:e80dbf3c3adc | saturation_p1 | 35011.0 | [35011.0, 300036.0] (n=6 trajectories) | [35011, 300036] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=23 | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 40583.2] (n=6 trajectories) | [35011, 300036] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=23 | math / MATH-500:3816ece4b994 | saturation_p1 | 77310.2 | [35011.0, 300036.0] (n=6 trajectories) | [35011, 300036] | True / True | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=23 | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 35484.2] (n=6 trajectories) | [35011, 300036] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=23 | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 36450.0] (n=6 trajectories) | [35011, 300036] | True / False | not identified |
| leave_one_pool_seed_out_within_student / gemma3-4b/seed=23 | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=6 trajectories) | [35011, 300036] | True / False | not identified |
| leave_one_student_size_out / gemma3-1b | code / MBPP:e80dbf3c3adc | saturation_p1 | 57047.2 | [35011.0, 180658.9] (n=23 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_student_size_out / gemma3-1b | code / MBPP:e80dbf3c3adc | saturation_p2 | 41761.0 | [35011.0, 54035.0] (n=23 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_student_size_out / gemma3-1b | math / MATH-500:3816ece4b994 | saturation_p1 | 139622.3 | [35011.0, 306590.0] (n=23 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_student_size_out / gemma3-1b | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 36962.7] (n=23 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_student_size_out / gemma3-1b | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=23 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_student_size_out / gemma3-1b | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=23 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_student_size_out / gemma3-270m | code / MBPP:e80dbf3c3adc | saturation_p1 | 89247.2 | [35011.0, 306590.0] (n=25 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_student_size_out / gemma3-270m | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=25 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_student_size_out / gemma3-270m | math / MATH-500:3816ece4b994 | saturation_p1 | 86859.0 | [35011.0, 250156.8] (n=25 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_student_size_out / gemma3-270m | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 109380.5] (n=25 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_student_size_out / gemma3-270m | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=25 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_student_size_out / gemma3-270m | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=25 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_student_size_out / gemma3-4b | code / MBPP:e80dbf3c3adc | saturation_p1 | 60227.2 | [35011.0, 306590.0] (n=34 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_student_size_out / gemma3-4b | code / MBPP:e80dbf3c3adc | saturation_p2 | 35011.0 | [35011.0, 42331.2] (n=34 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_student_size_out / gemma3-4b | math / MATH-500:3816ece4b994 | saturation_p1 | 100832.8 | [35011.0, 306590.0] (n=34 trajectories) | [35011, 306590] | True / True | not identified |
| leave_one_student_size_out / gemma3-4b | math / MATH-500:3816ece4b994 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=34 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_student_size_out / gemma3-4b | qa / 2WikiMultihopQA:075005b82489 | saturation_p1 | 35011.0 | [35011.0, 35011.0] (n=34 trajectories) | [35011, 306590] | True / False | not identified |
| leave_one_student_size_out / gemma3-4b | qa / 2WikiMultihopQA:075005b82489 | saturation_p2 | 35011.0 | [35011.0, 35011.0] (n=34 trajectories) | [35011, 306590] | True / False | not identified |

## Every dropped run and its reason

Retrospective development-set analysis on already unblinded data; exclusions use input availability and identity, never fit performance.

- `gemma3-270m/claude-sonnet-4-6_answer_only_600`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-270m/claude-sonnet-4-6_full_600`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-270m/gpt-5.6-luna_answer_only_600`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-270m/gpt-5.6-luna_full_150`: No pool entry in V47 register for U=150, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-270m/gpt-5.6-luna_full_16`: No pool entry in V47 register for U=16, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-270m/gpt-5.6-luna_full_300`: No pool entry in V47 register for U=300, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-270m/gpt-5.6-luna_full_600`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-270m/gpt-5.6-luna_full_75`: No pool entry in V47 register for U=75, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/claude-sonnet-4-6_answer_only_600`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/claude-sonnet-4-6_full_600`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_answer_only_600`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_150`: No pool entry in V47 register for U=150, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_225_lora_dseed1`: No pool entry in V47 register for U=225, data_seed=1; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_225_lora_dseed1_seed1`: No pool entry in V47 register for U=225, data_seed=1; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_225_lora_dseed2`: No pool entry in V47 register for U=225, data_seed=2; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_225_lora_dseed2_seed1`: No pool entry in V47 register for U=225, data_seed=2; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_225_lora_dseed3`: No pool entry in V47 register for U=225, data_seed=3; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_225_lora_dseed3_seed1`: No pool entry in V47 register for U=225, data_seed=3; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_300`: No pool entry in V47 register for U=300, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_300_seed1`: No pool entry in V47 register for U=300, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_300_seed2`: No pool entry in V47 register for U=300, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_600`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_600_seed1`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_600_seed2`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_600_uxseen`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_600_uxseen_seed1`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_600_uxseen_seed2`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_75`: No pool entry in V47 register for U=75, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_75_seed1`: No pool entry in V47 register for U=75, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_75_seed2`: No pool entry in V47 register for U=75, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_75_uxseen`: No pool entry in V47 register for U=75, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_75_uxseenE`: No pool entry in V47 register for U=75, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_75_uxseenE_seed1`: No pool entry in V47 register for U=75, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_75_uxseenE_seed2`: No pool entry in V47 register for U=75, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_75_uxseen_seed1`: No pool entry in V47 register for U=75, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_full_75_uxseen_seed2`: No pool entry in V47 register for U=75, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-1b/gpt-5.6-luna_no_code_fence_600`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-4b/claude-sonnet-4-6_full_600`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-4b/gpt-5.6-luna_full_150`: No pool entry in V47 register for U=150, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-4b/gpt-5.6-luna_full_300`: No pool entry in V47 register for U=300, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-4b/gpt-5.6-luna_full_300_seed1`: No pool entry in V47 register for U=300, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-4b/gpt-5.6-luna_full_300_seed2`: No pool entry in V47 register for U=300, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-4b/gpt-5.6-luna_full_600`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-4b/gpt-5.6-luna_full_600_seed1`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-4b/gpt-5.6-luna_full_600_seed2`: No pool entry in V47 register for U=600, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-4b/gpt-5.6-luna_full_75`: No pool entry in V47 register for U=75, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-4b/gpt-5.6-luna_full_75_seed1`: No pool entry in V47 register for U=75, data_seed=None; D_U cannot be assigned from the required source.

- `gemma3-4b/gpt-5.6-luna_full_75_seed2`: No pool entry in V47 register for U=75, data_seed=None; D_U cannot be assigned from the required source.

### Checkpoint exclusions and duplicate aliases

- `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_throughput_lora_dseed31/eval.json` : Duplicate update 36; retained results/v12-distill/gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_throughput_lora_dseed31/trajectory/update-00000036/eval.json.

Full endpoint records, fold membership, fitted coefficients, grid profiles, distribution definitions, input paths and SHA-256 digests are in `summary.json`. Retrospective development-set analysis on already unblinded data.
