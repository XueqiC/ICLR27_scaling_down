# A2 curvature and interaction report

Status: **complete**. CPU only; no training or new evaluation.

The predeclared primary targets are code × budget intervention and new 2Wiki QA × data intervention. Exact positive-budget data interventions and T-by-E rectangles are absent. Tolerance-matched contrasts do not establish fixed-budget causality. These are reused development measurements, not an independent confirmation sample.

## Protocol

**T_ref**: 100000.0

**p_range**: [-1,3]

**p_initial_regions**: [[-1.0,0.3333333333333333],[0.3333333333333333,1.6666666666666667],[1.6666666666666667,3.0]]

**p_profile_grid**: [-1.0,-0.875,-0.75,-0.625,-0.5,-0.375,-0.25,-0.125,0.0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0,1.125,1.25,1.375,1.5,1.625,1.75,1.875,2.0,2.125,2.25,2.375,2.5,2.625,2.75,2.875,3.0]

**coefficients**: unconstrained signs

**structures**: {"F_curv":"(a+a_prime*z)*u+(b+b_prime*z)*h_p(E)","F_int":"(a+a_prime*z)*u+(b+b_prime*z)*v+k*u*v","F_log":"(a+a_prime*z)*u+(b+b_prime*z)*v"}

**descriptors**: ["log_parameters","initial_loss"]

**lambda_grid**: [0.0001,0.01,1.0]

**fixed_lambda**: 0.01

**descriptor_calibration_cost**: Log parameter count requires exact architecture metadata; initial_loss requires the target student's own recorded pretraining evaluation on that distribution. This is one baseline evaluation, not a post-training target calibration. Both use training-fold standardisation.

**selection**: Use >=3 usable inner folds of same grouping as outer split; select structure/descriptor/lambda by mean inner response MAE. Otherwise F_log/log_parameters/lambda=.01 are fixed. Each selected F predicts every target. All baseline surfaces share the selected descriptor/lambda.

**standardisation**: Equal training students then trajectories; log non-embedding count or own distribution-specific initial loss. Feature RMS learned in each fit, no column centering. RMS-normalised ridge minimises weighted MSE+lambda*sum(theta^2).

**fit_weights**: Positive responses only: equal trajectories, then equal budgets within trajectory; exact zero anchors imposed algebraically. 132 trajectories receive no extra weight from dense checkpoints.

**intervention_weights**: Equal student, physical pool pair, fixed A1 budget band, pair. Duplicate pair appearances across held groups share weight.

**largest_budget**: All T>=150000 withheld; fit only T<150000. QA scope has only final positive checkpoints and is unscorable here.

**endpoints**: Training pair calibration requires both endpoints in training. Evaluation needs at least one held endpoint, allowing a known reference in leave-rung/seed/budget folds. Both predictions always from same fitted F. The student's initial-loss descriptor is held at the first endpoint's recorded value in both predictions; small cross-run baseline discrepancies are audited, not interpreted as a data effect.

**primary_targets**: ["code/training_probe:MBPP x I_T","qa/2wiki_new x I_U; training_probe:2Wiki also reported separately"]

**secondary**: Math, training-probe QA, MuSiQue and TriviaQA; all distributions from canonical DEVELOPMENT CSV only

**I_U**: Exact I_U has no observed positive matched-T pairs. Decision MAEs labelled 1% tolerance proxy use actual endpoints; fixed-T model predictions are stored but not scored against mismatched outcomes. 0.5%/5% separate sensitivity, never pooled.

**I_T**: Stored [1.7,2.3] budget ratios, approximate doubling; no exact 2x claim

**error_interval**: 95% frozen-prediction trajectory multiplier intervals; same-run pair gets one weight, cross-run pair product. No refitting; shared pool seed dependence and baseline oracle selection not covered. Pointwise, not simultaneous.

**new_configuration_prediction_intervals**: {"interval":null,"reason":"Only three students and four partly rung-confounded pool seeds; no independent calibration configurations. Frozen error and parameter intervals are not new-configuration prediction intervals.","status":"not estimable with calibrated coverage"}

**external_diagnostic_file_read**: false

**previous_F_int**: V96 source specifies 18 matrix2 trajectories, different selection/regularisation and no 132 rung. Those inputs/folds do not match A2; prior scores not read or recomputed. A2 is a reanalysis of overlapping measurements, not independent new evidence. Identical A2 fingerprints are reused on rerun.

## Canonical input limitations

Stored effects remain differences of the canonical deltas. Their differing zero baselines are retained as a measurement limitation. An initial-loss-conditioned intervention holds the first endpoint's initial loss fixed.

| Student / distribution | Distinct recorded initial losses | Range |
|---|---:|---:|
| gemma3-1b / QA-2Wiki | 2 | 0.0032103 |
| gemma3-270m / QA-2Wiki | 2 | 0.0041491 |
| gemma3-1b / QA-MuSiQue | 2 | 0.0022117 |
| gemma3-270m / QA-MuSiQue | 2 | 0.00062889 |
| gemma3-1b / QA-TriviaQA | 2 | 0.00028058 |
| gemma3-270m / QA-TriviaQA | 2 | 0.00060051 |

Complete grouped-fold eligibility counts are saved in summary.json even when missing descriptors prevent fitting. No forbidden metadata or external diagnostic measurement file was opened.

## Baselines and calibration

| Baseline | Parameters | Effective degrees of freedom | Fitting unit | Information | Calibration cost |
|---|---|---|---|---|---|
| zero | 0 | 0 | none | T=0 anchor | none |
| constant | 1 | 1 | trajectory/budget weighted positive response | Only training-fold canonical responses; no extra measured checkpoints | one scalar from existing training responses |
| budget_only | 1 | trace ridge hat matrix, <=1; recorded per fold | weighted response | u only | Only training-fold canonical responses; no extra measured checkpoints |
| reuse_only | 1 | trace ridge hat matrix, <=1; recorded per fold | weighted response | v only | Only training-fold canonical responses; no extra measured checkpoints |
| mean_effect | 1 per intervention type; 2 in total, no response predictor | 1 per estimable mean, 0 when unavailable | training pairs, hierarchical student/pool/budget weights | observed nonzero signed budget-doubling or data-change effects separately | requires both existing endpoints of training pairs; no new measurements |
| surface | 7 | trace ridge hat matrix, <=7; recorded per fold | weighted response | fixed u,v,u^2,v^2,uv,zu,zv, exactly same inputs/scaler/lambda as F | Only training-fold canonical responses; no extra measured checkpoints |
| empirical_interpolation_diagnostic | one value per distinct training knot | number of distinct knots | same-student response knots | training (u,v) closed convex hull, barycentric linear, no extrapolation or cross-student transfer | dense local responses already present; unsupported points abstain |

The constant positive-budget response has identically zero differenced intervention. The signed mean-effect baseline is separately estimated for I_T and I_U; it is not forced to zero. F_log has four coefficients, F_curv four coefficients plus p, and F_int five coefficients. Conditional ridge EDF and the approximate additional p degree of freedom are saved per fold.

## Decision table

MAE in native-token nats. Positive gain favours the primary F. The strongest observed baseline is an outer-score oracle, shown as requested; its interval is conditional and does not include choosing the baseline. The separate inner-selected baseline comparisons below avoid that outer selection. Candidate identity always comes from inner responses, never the target outer score.

| Distribution | Target | Split | Primary F | Strongest observed baseline | MAE F | MAE baseline | Paired gain [95% trajectory interval] |
|---|---|---|---|---|---:|---:|---|
| code | response | data_rung | F_curv, F_int, F_log | surface | 0.050845 | 0.049757 | -0.0010889 [-0.0040183, 0.0018998] |
| code **primary** | I_T | data_rung | F_curv, F_int, F_log | reuse_only | 0.039335 | 0.030617 | -0.0087184 [-0.013631, -0.0039494] |
| code | I_U (1% proxy) | data_rung | F_curv, F_int, F_log | reuse_only | 0.069743 | 0.06155 | -0.0081929 [-0.023453, 0.0040595] |
| code | response | student | F_log | reuse_only | 0.064261 | 0.045792 | -0.01847 [-0.025704, -0.01145] |
| code **primary** | I_T | student | F_log | surface | 0.0357 | 0.030806 | -0.0048944 [-0.0093197, -0.0015638] |
| code | I_U (1% proxy) | student | F_log | surface | 0.057042 | 0.052286 | -0.0047566 [-0.019268, 0.015749] |
| code | response | largest_budget | F_log | surface | 0.067912 | 0.045308 | -0.022604 [-0.031706, -0.010468] |
| code **primary** | I_T | largest_budget | F_log | surface | 0.045698 | 0.030334 | -0.015364 [-0.02217, -0.0071457] |
| code | I_U (1% proxy) | largest_budget | F_log | surface | 0.085489 | 0.056342 | -0.029147 [-0.049996, -0.0060227] |
| code | response | pool_seed | F_curv, F_int | surface | 0.040909 | 0.036291 | -0.0046175 [-0.0086273, -0.0010893] |
| code **primary** | I_T | pool_seed | F_curv, F_int | surface | 0.035623 | 0.031459 | -0.004164 [-0.0086358, 8.2026e-05] |
| code | I_U (1% proxy) | pool_seed | F_curv, F_int | surface | 0.055841 | 0.047825 | -0.0080161 [-0.019564, 0.0032712] |
| math | response | data_rung | F_curv, F_int, F_log | surface | 0.053221 | 0.045124 | -0.0080974 [-0.013894, -0.0026509] |
| math | I_T | data_rung | F_curv, F_int, F_log | reuse_only | 0.039433 | 0.034804 | -0.0046288 [-0.0085139, -0.00095377] |
| math | I_U (1% proxy) | data_rung | F_curv, F_int, F_log | surface | 0.079559 | 0.069408 | -0.010151 [-0.021717, 0.0017148] |
| math | response | student | F_log | surface | 0.066891 | 0.063813 | -0.0030773 [-0.0062524, 0.00051298] |
| math | I_T | student | F_log | surface | 0.035555 | 0.030979 | -0.0045755 [-0.0081057, -0.0013345] |
| math | I_U (1% proxy) | student | F_log | surface | 0.089154 | 0.06562 | -0.023534 [-0.042887, -0.0087228] |
| math | response | largest_budget | F_log | reuse_only | 0.071841 | 0.066746 | -0.0050952 [-0.021254, 0.013428] |
| math | I_T | largest_budget | F_log | reuse_only | 0.046944 | 0.042192 | -0.0047518 [-0.012577, 0.0028415] |
| math | I_U (1% proxy) | largest_budget | F_log | surface | 0.10375 | 0.08049 | -0.023262 [-0.032498, -0.010793] |
| math | response | pool_seed | F_curv, F_int | surface | 0.040459 | 0.038638 | -0.0018212 [-0.0054737, 0.0017876] |
| math | I_T | pool_seed | F_curv, F_int | reuse_only | 0.035944 | 0.034876 | -0.0010682 [-0.007806, 0.0083693] |
| math | I_U (1% proxy) | pool_seed | F_curv, F_int | surface | 0.053556 | 0.057674 | 0.0041177 [-0.0055385, 0.016569] |
| QA-2Wiki | response | data_rung | F_curv, F_int, F_log | surface | 3.8888 | 1.2418 | -2.6469 [-5.3326, -0.92725] |
| QA-2Wiki | I_T | data_rung | NA | NA | NA | NA | unscorable |
| QA-2Wiki **primary** | I_U (1% proxy) | data_rung | F_curv, F_int, F_log | surface | 4.5106 | 1.3369 | -3.1736 [-5.1598, -1.1032] |
| QA-2Wiki | response | student | F_log | surface | 1.1958 | 1.1048 | -0.090945 [-0.22057, 0.039323] |
| QA-2Wiki | I_T | student | NA | NA | NA | NA | unscorable |
| QA-2Wiki **primary** | I_U (1% proxy) | student | F_log | surface | 1.5662 | 1.1615 | -0.40474 [-0.67418, -0.15266] |
| QA-2Wiki | response | largest_budget | NA | NA | NA | NA | unscorable |
| QA-2Wiki | I_T | largest_budget | NA | NA | NA | NA | unscorable |
| QA-2Wiki **primary** | I_U (1% proxy) | largest_budget | NA | NA | NA | NA | unscorable |
| QA-2Wiki | response | pool_seed | F_curv, F_log | surface | 1.056 | 0.81995 | -0.236 [-0.3483, -0.08709] |
| QA-2Wiki | I_T | pool_seed | NA | NA | NA | NA | unscorable |
| QA-2Wiki **primary** | I_U (1% proxy) | pool_seed | F_curv, F_log | surface | 1.4987 | 1.3162 | -0.18242 [-0.50286, 0.26067] |
| QA-MuSiQue | response | data_rung | F_curv, F_int | surface | 0.77433 | 0.47426 | -0.30007 [-0.57231, -0.096931] |
| QA-MuSiQue | I_T | data_rung | NA | NA | NA | NA | unscorable |
| QA-MuSiQue | I_U (1% proxy) | data_rung | F_curv, F_int | surface | 0.81997 | 0.51673 | -0.30324 [-0.51116, -0.1221] |
| QA-MuSiQue | response | student | F_log | surface | 0.30267 | 0.21541 | -0.087253 [-0.12415, -0.052946] |
| QA-MuSiQue | I_T | student | NA | NA | NA | NA | unscorable |
| QA-MuSiQue | I_U (1% proxy) | student | F_log | surface | 0.47873 | 0.30642 | -0.1723 [-0.25531, -0.098509] |
| QA-MuSiQue | response | largest_budget | NA | NA | NA | NA | unscorable |
| QA-MuSiQue | I_T | largest_budget | NA | NA | NA | NA | unscorable |
| QA-MuSiQue | I_U (1% proxy) | largest_budget | NA | NA | NA | NA | unscorable |
| QA-MuSiQue | response | pool_seed | F_curv, F_int | surface | 0.46572 | 0.39976 | -0.065961 [-0.12632, -0.014877] |
| QA-MuSiQue | I_T | pool_seed | NA | NA | NA | NA | unscorable |
| QA-MuSiQue | I_U (1% proxy) | pool_seed | F_curv, F_int | surface | 0.70208 | 0.58768 | -0.1144 [-0.18971, -0.038949] |
| QA-probe | response | data_rung | F_int | surface | 0.73977 | 0.5524 | -0.18736 [-0.24269, -0.12573] |
| QA-probe | I_T | data_rung | F_int | surface | 0.50973 | 0.45361 | -0.056121 [-0.10056, -0.017921] |
| QA-probe | I_U (1% proxy) | data_rung | F_int | surface | 1.0184 | 0.7639 | -0.25449 [-0.4117, -0.063724] |
| QA-probe | response | student | F_log | surface | 1.036 | 0.60793 | -0.42809 [-0.59259, -0.27947] |
| QA-probe | I_T | student | F_log | surface | 0.83315 | 0.46514 | -0.36802 [-0.5363, -0.23449] |
| QA-probe | I_U (1% proxy) | student | F_log | surface | 1.2921 | 0.80976 | -0.48231 [-0.81833, -0.15486] |
| QA-probe | response | largest_budget | F_log | surface | 2.1809 | 1.0335 | -1.1474 [-1.8844, -0.57614] |
| QA-probe | I_T | largest_budget | F_log | mean_effect | 1.7044 | 0.78862 | -0.91581 [-0.99225, -0.77781] |
| QA-probe | I_U (1% proxy) | largest_budget | F_log | surface | 1.7871 | 1.1526 | -0.63454 [-1.5619, 0.33722] |
| QA-probe | response | pool_seed | F_int | surface | 0.51935 | 0.44546 | -0.073881 [-0.13434, -0.016828] |
| QA-probe | I_T | pool_seed | F_int | surface | 0.42022 | 0.39957 | -0.020646 [-0.048936, 0.0081802] |
| QA-probe | I_U (1% proxy) | pool_seed | F_int | surface | 0.84372 | 0.71315 | -0.13056 [-0.30779, 0.096763] |
| QA-TriviaQA | response | data_rung | F_curv, F_int, F_log | surface | 0.35095 | 0.33936 | -0.01159 [-0.063546, 0.036724] |
| QA-TriviaQA | I_T | data_rung | NA | NA | NA | NA | unscorable |
| QA-TriviaQA | I_U (1% proxy) | data_rung | F_curv, F_int, F_log | reuse_only | 0.38794 | 0.3604 | -0.027531 [-0.19319, 0.11548] |
| QA-TriviaQA | response | student | F_log | reuse_only | 0.8628 | 0.64558 | -0.21722 [-0.40528, -0.090228] |
| QA-TriviaQA | I_T | student | NA | NA | NA | NA | unscorable |
| QA-TriviaQA | I_U (1% proxy) | student | F_log | reuse_only | 0.57674 | 0.4239 | -0.15283 [-0.4513, 0.040677] |
| QA-TriviaQA | response | largest_budget | NA | NA | NA | NA | unscorable |
| QA-TriviaQA | I_T | largest_budget | NA | NA | NA | NA | unscorable |
| QA-TriviaQA | I_U (1% proxy) | largest_budget | NA | NA | NA | NA | unscorable |
| QA-TriviaQA | response | pool_seed | F_curv, F_int, F_log | surface | 0.30391 | 0.28504 | -0.018876 [-0.049035, 0.010214] |
| QA-TriviaQA | I_T | pool_seed | NA | NA | NA | NA | unscorable |
| QA-TriviaQA | I_U (1% proxy) | pool_seed | F_curv, F_int, F_log | surface | 0.32883 | 0.26801 | -0.06082 [-0.090153, -0.027817] |

## T-by-E coverage

| Student | Trajectories | Positive checkpoints | Distinct T | T with ≥2 E | Actual four-corner rectangles |
|---|---:|---:|---:|---:|---:|
| gemma3-1b | 8 | 52 | 52 | 0 | 0 |
| gemma3-270m | 6 | 24 | 24 | 0 | 0 |
| gemma3-4b | 8 | 52 | 52 | 0 | 0 |

None: no observed four-corner interaction statistic exists; model-imputed or interpolated corners cannot exclude additivity

A repeated (0,0) anchor is degenerate. Within each positive T there is only one E at fixed student; no horizontal two-E edge exists. Two pools at two approximately equal nominal budgets do not provide the required corners.

- `gemma3-1b/gpt-5.6-luna_full_132_critical_lora_dseed51`: U=132, D_U=34503, seed=51; update-00000000 T=0 E=0; update-00000019 T=25541 E=25541/34503; update-00000029 T=40419 E=13473/11501; update-00000039 T=53788 E=7684/4929; update-00000047 T=64061 E=64061/34503; update-00000058 T=80211 E=26737/11501; update-00000067 T=92212 E=92212/34503; update-00000077 T=105738 E=35246/11501; update-00000086 T=119046 E=39682/11501; update-00000095 T=131781 E=1417/371; update-00000105 T=144266 E=2722/651; update-00000114 T=157044 E=52348/11501; update-00000124 T=171664 E=171664/34503; update-00000134 T=185412 E=61804/11501; update-00000144 T=198773 E=198773/34503
- `gemma3-1b/gpt-5.6-luna_full_132_critical_lora_dseed52`: U=132, D_U=34639, seed=52; update-00000000 T=0 E=0; update-00000019 T=27231 E=27231/34639; update-00000029 T=40421 E=40421/34639; update-00000038 T=53160 E=53160/34639; update-00000048 T=67385 E=67385/34639; update-00000057 T=79379 E=79379/34639; update-00000067 T=93026 E=93026/34639; update-00000077 T=106886 E=106886/34639; update-00000086 T=119014 E=119014/34639; update-00000095 T=131885 E=131885/34639; update-00000105 T=145248 E=145248/34639; update-00000115 T=158685 E=158685/34639; update-00000124 T=171913 E=171913/34639; update-00000134 T=186132 E=186132/34639; update-00000144 T=201230 E=201230/34639
- `gemma3-1b/gpt-5.6-luna_full_198_matrix2_lora_dseed41`: U=198, D_U=51065, seed=41; update-00000000 T=0 E=0; update-00000018 T=24623 E=24623/51065; update-00000036 T=49836 E=49836/51065; update-00000073 T=101139 E=101139/51065; update-00000144 T=198736 E=198736/51065
- `gemma3-1b/gpt-5.6-luna_full_198_matrix2_lora_dseed42`: U=198, D_U=51812, seed=42; update-00000000 T=0 E=0; update-00000018 T=25396 E=6349/12953; update-00000036 T=50541 E=50541/51812; update-00000072 T=101373 E=101373/51812; update-00000143 T=199848 E=49962/12953
- `gemma3-1b/gpt-5.6-luna_full_594_matrix2_lora_dseed41`: U=594, D_U=156826, seed=41; update-00000000 T=0 E=0; update-00000019 T=27919 E=27919/156826; update-00000037 T=53525 E=53525/156826; update-00000073 T=104539 E=104539/156826; update-00000142 T=198722 E=99361/78413
- `gemma3-1b/gpt-5.6-luna_full_594_matrix2_lora_dseed42`: U=594, D_U=156653, seed=42; update-00000000 T=0 E=0; update-00000018 T=24067 E=24067/156653; update-00000036 T=50038 E=50038/156653; update-00000072 T=102110 E=102110/156653; update-00000143 T=201946 E=201946/156653
- `gemma3-1b/gpt-5.6-luna_full_66_matrix2_lora_dseed41`: U=66, D_U=16962, seed=41; update-00000000 T=0 E=0; update-00000019 T=25460 E=12730/8481; update-00000038 T=50563 E=50563/16962; update-00000076 T=100114 E=50057/8481; update-00000152 T=199166 E=9053/771
- `gemma3-1b/gpt-5.6-luna_full_66_matrix2_lora_dseed42`: U=66, D_U=17104, seed=42; update-00000000 T=0 E=0; update-00000019 T=24822 E=12411/8552; update-00000038 T=50869 E=50869/17104; update-00000075 T=99576 E=12447/2138; update-00000151 T=199673 E=199673/17104
- `gemma3-270m/gpt-5.6-luna_full_198_matrix2_lora_dseed41`: U=198, D_U=51065, seed=41; update-00000000 T=0 E=0; update-00000018 T=24623 E=24623/51065; update-00000036 T=49836 E=49836/51065; update-00000073 T=101139 E=101139/51065; update-00000144 T=198736 E=198736/51065
- `gemma3-270m/gpt-5.6-luna_full_198_matrix2_lora_dseed42`: U=198, D_U=51812, seed=42; update-00000000 T=0 E=0; update-00000018 T=25396 E=6349/12953; update-00000036 T=50541 E=50541/51812; update-00000072 T=101373 E=101373/51812; update-00000143 T=199848 E=49962/12953
- `gemma3-270m/gpt-5.6-luna_full_594_matrix2_lora_dseed41`: U=594, D_U=156826, seed=41; update-00000000 T=0 E=0; update-00000019 T=27919 E=27919/156826; update-00000037 T=53525 E=53525/156826; update-00000073 T=104539 E=104539/156826; update-00000142 T=198722 E=99361/78413
- `gemma3-270m/gpt-5.6-luna_full_594_matrix2_lora_dseed42`: U=594, D_U=156653, seed=42; update-00000000 T=0 E=0; update-00000018 T=24067 E=24067/156653; update-00000036 T=50038 E=50038/156653; update-00000072 T=102110 E=102110/156653; update-00000143 T=201946 E=201946/156653
- `gemma3-270m/gpt-5.6-luna_full_66_matrix2_lora_dseed41`: U=66, D_U=16962, seed=41; update-00000000 T=0 E=0; update-00000019 T=25460 E=12730/8481; update-00000038 T=50563 E=50563/16962; update-00000076 T=100114 E=50057/8481; update-00000152 T=199166 E=9053/771
- `gemma3-270m/gpt-5.6-luna_full_66_matrix2_lora_dseed42`: U=66, D_U=17104, seed=42; update-00000000 T=0 E=0; update-00000019 T=24822 E=12411/8552; update-00000038 T=50869 E=50869/17104; update-00000075 T=99576 E=12447/2138; update-00000151 T=199673 E=199673/17104
- `gemma3-4b/gpt-5.6-luna_full_132_critical_lora_dseed51`: U=132, D_U=34503, seed=51; update-00000000 T=0 E=0; update-00000019 T=25541 E=25541/34503; update-00000029 T=40419 E=13473/11501; update-00000039 T=53788 E=7684/4929; update-00000047 T=64061 E=64061/34503; update-00000058 T=80211 E=26737/11501; update-00000067 T=92212 E=92212/34503; update-00000077 T=105738 E=35246/11501; update-00000086 T=119046 E=39682/11501; update-00000095 T=131781 E=1417/371; update-00000105 T=144266 E=2722/651; update-00000114 T=157044 E=52348/11501; update-00000124 T=171664 E=171664/34503; update-00000134 T=185412 E=61804/11501; update-00000144 T=198773 E=198773/34503
- `gemma3-4b/gpt-5.6-luna_full_132_critical_lora_dseed52`: U=132, D_U=34639, seed=52; update-00000000 T=0 E=0; update-00000019 T=27231 E=27231/34639; update-00000029 T=40421 E=40421/34639; update-00000038 T=53160 E=53160/34639; update-00000048 T=67385 E=67385/34639; update-00000057 T=79379 E=79379/34639; update-00000067 T=93026 E=93026/34639; update-00000077 T=106886 E=106886/34639; update-00000086 T=119014 E=119014/34639; update-00000095 T=131885 E=131885/34639; update-00000105 T=145248 E=145248/34639; update-00000115 T=158685 E=158685/34639; update-00000124 T=171913 E=171913/34639; update-00000134 T=186132 E=186132/34639; update-00000144 T=201230 E=201230/34639
- `gemma3-4b/gpt-5.6-luna_full_198_matrix2_lora_dseed41`: U=198, D_U=51065, seed=41; update-00000000 T=0 E=0; update-00000018 T=24623 E=24623/51065; update-00000036 T=49836 E=49836/51065; update-00000073 T=101139 E=101139/51065; update-00000144 T=198736 E=198736/51065
- `gemma3-4b/gpt-5.6-luna_full_198_matrix2_lora_dseed42`: U=198, D_U=51812, seed=42; update-00000000 T=0 E=0; update-00000018 T=25396 E=6349/12953; update-00000036 T=50541 E=50541/51812; update-00000072 T=101373 E=101373/51812; update-00000143 T=199848 E=49962/12953
- `gemma3-4b/gpt-5.6-luna_full_594_matrix2_lora_dseed41`: U=594, D_U=156826, seed=41; update-00000000 T=0 E=0; update-00000019 T=27919 E=27919/156826; update-00000037 T=53525 E=53525/156826; update-00000073 T=104539 E=104539/156826; update-00000142 T=198722 E=99361/78413
- `gemma3-4b/gpt-5.6-luna_full_594_matrix2_lora_dseed42`: U=594, D_U=156653, seed=42; update-00000000 T=0 E=0; update-00000018 T=24067 E=24067/156653; update-00000036 T=50038 E=50038/156653; update-00000072 T=102110 E=102110/156653; update-00000143 T=201946 E=201946/156653
- `gemma3-4b/gpt-5.6-luna_full_66_matrix2_lora_dseed41`: U=66, D_U=16962, seed=41; update-00000000 T=0 E=0; update-00000019 T=25460 E=12730/8481; update-00000038 T=50563 E=50563/16962; update-00000076 T=100114 E=50057/8481; update-00000152 T=199166 E=9053/771
- `gemma3-4b/gpt-5.6-luna_full_66_matrix2_lora_dseed42`: U=66, D_U=17104, seed=42; update-00000000 T=0 E=0; update-00000019 T=24822 E=12411/8552; update-00000038 T=50869 E=50869/17104; update-00000075 T=99576 E=12447/2138; update-00000151 T=199673 E=199673/17104

## Curvature stability and parameter intervals

Fold p ranges are sensitivity summaries, not confidence intervals. Any boundary optimum is labelled curvature not identified. Parameter intervals refit coefficients and p under trajectory multipliers, conditional on descriptor/lambda and observed students. Frozen-prediction error intervals resample errors without refitting. New-configuration prediction intervals are unavailable; neither of the preceding interval types substitutes for them.

| Distribution | p min / median / max | Boundary hits / folds | Leave-one-student p |
|---|---|---|---|
| code | -1 / 0.6099 / 2.3136 | 1/12 | gemma3-1b=0.41398; gemma3-270m=0.86435; gemma3-4b=0.79995 |
| math | -1 / 0.99221 / 1.2317 | 2/12 | gemma3-1b=1.0825; gemma3-270m=1.2317; gemma3-4b=1.0935 |
| QA-2Wiki | 0.17489 / 1.3938 / 3 | 1/11 | gemma3-1b=1.1839; gemma3-270m=1.3938; gemma3-4b=2.3772 |
| QA-MuSiQue | 1.1561 / 1.7375 / 2.5867 | 0/11 | gemma3-1b=1.3638; gemma3-270m=1.549; gemma3-4b=2.1407 |
| QA-probe | 1.6617 / 1.8119 / 3 | 1/12 | gemma3-1b=1.6617; gemma3-270m=1.7744; gemma3-4b=2.2135 |
| QA-TriviaQA | 0.69355 / 0.92454 / 1.8651 | 0/11 | gemma3-1b=0.84463; gemma3-270m=0.92454; gemma3-4b=1.8651 |

code: full-development conditional parameter interval for p [0.2203, 1.1414]; refit boundary fraction 0. Full fit p=0.69052: interior optimum; identification requires stability. Coefficient intervals and complete profile objectives are in summary.json.

math: full-development conditional parameter interval for p [0.34721, 1.401]; refit boundary fraction 0. Full fit p=1.1199: interior optimum; identification requires stability. Coefficient intervals and complete profile objectives are in summary.json.

QA-2Wiki: full-development conditional parameter interval for p [1.0182, 2.2168]; refit boundary fraction 0. Full fit p=1.4399: interior optimum; identification requires stability. Coefficient intervals and complete profile objectives are in summary.json.

QA-MuSiQue: full-development conditional parameter interval for p [1.0242, 2.304]; refit boundary fraction 0. Full fit p=1.5491: interior optimum; identification requires stability. Coefficient intervals and complete profile objectives are in summary.json.

QA-probe: full-development conditional parameter interval for p [1.5628, 2.0285]; refit boundary fraction 0. Full fit p=1.7396: interior optimum; identification requires stability. Coefficient intervals and complete profile objectives are in summary.json.

QA-TriviaQA: full-development conditional parameter interval for p [0.29514, 2.659]; refit boundary fraction 0.015. Full fit p=1.0579: interior optimum; identification requires stability. Coefficient intervals and complete profile objectives are in summary.json.

## Missing-corner development design

maximum corner response disagreement >=0.01 native-token nat, fixed before evaluation.

Three trajectories total (one existing, two new). Exact supervised stops at 50563 and 101126; preserve the recorded fixed schedule horizon and recipe. Construct and verify distinct pools with exactly 33924/67848 supervised tokens. If these counts/stops cannot be realised without changing protocol, this design is infeasible and must be redesigned before collection; nominal U or ordinal checkpoints do not substitute.

Evaluate all four corners on the same distributions, including 2wiki_new; QA scope requires measurement even at the existing training checkpoint. Pool parentage must be recorded; existing seed labels do not prove nesting.

| Corner | T | E | D_U | Existing recorded training corner |
|---|---:|---|---:|---|
| 1 | 50563 | 50563/33924 | 33924 | False |
| 2 | 50563 | 50563/16962 | 16962 | True |
| 3 | 101126 | 50563/33924 | 67848 | False |
| 4 | 101126 | 50563/16962 | 33924 | False |

One 1B trajectory at D_U=16962 (existing U66, seed41), one new D_U=33924 trajectory at both budgets, and one new D_U=67848 trajectory at the higher budget cover the design. New pools' source-example counts cannot be inferred from token totals; they must be recorded after assembly. This is a concrete token-level requirement, not an assertion that existing named pools already satisfy it.

| Distribution | Max additive/interaction corner disagreement | Material at 0.01 | F_int second difference |
|---|---:|---|---:|
| code | 0.014375 | True | 0.0030315 |
| math | 0.021486 | True | 0.0028211 |
| QA-2Wiki | 0.87111 | True | 0.13258 |
| QA-MuSiQue | 0.36583 | True | 0.055675 |
| QA-probe | 1.5546 | True | 0.43258 |
| QA-TriviaQA | 0.23205 | True | 0.035316 |

Predictions in this design table are planning diagnostics, not imputed observations or structural evidence.

## Per-fold decisions, inner-selected baselines and stability

All coefficients, standardizers, training keys, predictor IDs, profile grids and per-pair endpoint predictions are retained in summary.json. The same predictor ID applies to both endpoints.

### code / data_rung

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| 66 | response | F_curv | surface | 0.079289 | 0.07767 | -0.0016194 [-0.0092929, 0.0055676] | -1/True |
| 66 | I_T | F_curv | mean_effect | 0.065018 | 0.062378 | -0.0026397 [-0.0042333, -0.0010537] | -1/True |
| 66 | I_U | F_curv | zero | 0.13582 | 0.15096 | 0.015139 [0.0049392, 0.026861] | -1/True |
| 132 | response | F_log | surface | 0.043283 | 0.037972 | -0.0053112 [-0.0064373, -0.0042544] | 0.58366/False |
| 132 | I_T | F_log | reuse_only | 0.023339 | 0.02034 | -0.0029985 [-0.0040774, -0.0010968] | 0.58366/False |
| 132 | I_U | F_log | surface | 0.081971 | 0.071849 | -0.010122 [-0.024246, 0.0028728] | 0.58366/False |
| 198 | response | F_int | surface | 0.037531 | 0.03777 | 0.00023891 [-0.0015776, 0.0025814] | 0.58711/False |
| 198 | I_T | F_int | surface | 0.028744 | 0.034769 | 0.0060252 [0.0039619, 0.0072307] | 0.58711/False |
| 198 | I_U | F_int | surface | 0.035401 | 0.031343 | -0.0040576 [-0.0088609, -4.3901e-05] | 0.58711/False |
| 594 | response | F_log | reuse_only | 0.040757 | 0.051511 | 0.010754 [0.0045139, 0.019715] | 0.59326/False |
| 594 | I_T | F_log | reuse_only | 0.036985 | 0.032618 | -0.0043675 [-0.0074799, -0.0015466] | 0.59326/False |
| 594 | I_U | F_log | reuse_only | 0.045587 | 0.05695 | 0.011363 [-0.0042755, 0.022385] | 0.59326/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=88 F MAE=0.08785; 5% n=332 F MAE=0.055504.

Held 66: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.01. Empirical interpolation support 0/24, MAE=NA, F MAE on identical support=NA.

Held 132: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.01. Empirical interpolation support 54/56, MAE=0.024868, F MAE on identical support=0.041898.

Held 198: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=1.0. Empirical interpolation support 24/24, MAE=0.025474, F MAE on identical support=0.037531.

Held 594: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.01. Empirical interpolation support 0/24, MAE=NA, F MAE on identical support=NA.

### code / student

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| gemma3-1b | response | F_log | constant | 0.054838 | 0.070955 | 0.016117 [-0.0051485, 0.037736] | 0.41398/False |
| gemma3-1b | I_T | F_log | mean_effect | 0.043361 | 0.037078 | -0.0062828 [-0.011983, 0.0025706] | 0.41398/False |
| gemma3-1b | I_U | F_log | mean_effect | 0.080463 | 0.10149 | 0.021031 [0.0026757, 0.03448] | 0.41398/False |
| gemma3-270m | response | F_log | constant | 0.075535 | 0.05554 | -0.019995 [-0.04386, 0.0052959] | 0.86435/False |
| gemma3-270m | I_T | F_log | mean_effect | 0.03169 | 0.025288 | -0.0064027 [-0.012263, 0.0001424] | 0.86435/False |
| gemma3-270m | I_U | F_log | mean_effect | 0.034655 | 0.061944 | 0.027289 [0.011312, 0.041707] | 0.86435/False |
| gemma3-4b | response | F_log | constant | 0.065229 | 0.064282 | -0.00094708 [-0.032209, 0.026892] | 0.79995/False |
| gemma3-4b | I_T | F_log | mean_effect | 0.032049 | 0.033367 | 0.001318 [-0.002899, 0.0064559] | 0.79995/False |
| gemma3-4b | I_U | F_log | mean_effect | 0.056009 | 0.079805 | 0.023796 [0.0053602, 0.042315] | 0.79995/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=44 F MAE=0.065145; 5% n=166 F MAE=0.047478.

Held gemma3-1b: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/52, MAE=NA, F MAE on identical support=NA.

Held gemma3-270m: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/24, MAE=NA, F MAE on identical support=NA.

Held gemma3-4b: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/52, MAE=NA, F MAE on identical support=NA.

### code / largest_budget

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| T>=150000 | response | F_log | constant | 0.067912 | 0.10608 | 0.038171 [0.0031124, 0.075978] | 2.3136/False |
| T>=150000 | I_T | F_log | mean_effect | 0.045698 | 0.039847 | -0.0058503 [-0.01354, 0.0033099] | 2.3136/False |
| T>=150000 | I_U | F_log | mean_effect | 0.085489 | 0.12112 | 0.03563 [0.022534, 0.048405] | 2.3136/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=31 F MAE=0.09021; 5% n=60 F MAE=0.079619.

Held T>=150000: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/34, MAE=NA, F MAE on identical support=NA.

### code / pool_seed

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| 41 | response | F_int | surface | 0.040912 | 0.038363 | -0.0025488 [-0.0082129, 0.0016657] | 0.77946/False |
| 41 | I_T | F_int | mean_effect | 0.034413 | 0.030749 | -0.0036632 [-0.010354, 0.0050238] | 0.77946/False |
| 41 | I_U | F_int | surface | 0.043648 | 0.037507 | -0.0061411 [-0.015984, 0.0041525] | 0.77946/False |
| 42 | response | F_int | budget_only | 0.04414 | 0.05901 | 0.01487 [0.0067193, 0.027123] | 0.4607/False |
| 42 | I_T | F_int | mean_effect | 0.040232 | 0.041235 | 0.0010032 [-0.003644, 0.0066678] | 0.4607/False |
| 42 | I_U | F_int | budget_only | 0.076465 | 0.10409 | 0.027624 [0.010514, 0.046769] | 0.4607/False |
| 51 | response | F_curv | surface | 0.027591 | 0.025258 | -0.002334 [-0.0062078, 0.0015698] | 0.66903/False |
| 51 | I_T | F_curv | surface | 0.029475 | 0.022646 | -0.0068284 [-0.0080321, -0.005634] | 0.66903/False |
| 51 | I_U | F_curv | surface | 0.054096 | 0.056364 | 0.0022675 [0.0006804, 0.0043159] | 0.66903/False |
| 52 | response | F_curv | surface | 0.03967 | 0.041089 | 0.0014188 [-0.0011474, 0.004005] | 0.62654/False |
| 52 | I_T | F_curv | reuse_only | 0.031282 | 0.021592 | -0.0096904 [-0.012657, -0.0067468] | 0.62654/False |
| 52 | I_U | F_curv | surface | 0.085088 | 0.087216 | 0.0021277 [-0.0014173, 0.0056284] | 0.62654/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=76 F MAE=0.071224; 5% n=275 F MAE=0.044243.

Held 41: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=1.0. Empirical interpolation support 12/36, MAE=0.015014, F MAE on identical support=0.035764.

Held 42: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.0001. Empirical interpolation support 20/36, MAE=0.02696, F MAE on identical support=0.036557.

Held 51: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.01. Empirical interpolation support 28/28, MAE=0.017385, F MAE on identical support=0.027591.

Held 52: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.01. Empirical interpolation support 26/28, MAE=0.01883, F MAE on identical support=0.036104.

### math / data_rung

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| 66 | response | F_curv | surface | 0.071492 | 0.068394 | -0.0030985 [-0.0098813, 0.0016705] | -1/True |
| 66 | I_T | F_curv | reuse_only | 0.076029 | 0.064678 | -0.011351 [-0.016443, -0.0041645] | -1/True |
| 66 | I_U | F_curv | surface | 0.1562 | 0.14405 | -0.012145 [-0.027873, -0.0018729] | -1/True |
| 132 | response | F_log | surface | 0.046013 | 0.040433 | -0.0055797 [-0.0062226, -0.0049365] | 0.99602/False |
| 132 | I_T | F_log | reuse_only | 0.028473 | 0.026181 | -0.0022923 [-0.0088123, 0.0035594] | 0.99602/False |
| 132 | I_U | F_log | surface | 0.083478 | 0.071777 | -0.011701 [-0.02773, 0.0027653] | 0.99602/False |
| 198 | response | F_int | surface | 0.033637 | 0.039423 | 0.0057852 [0.0057497, 0.0058347] | 0.88156/False |
| 198 | I_T | F_int | reuse_only | 0.029839 | 0.019974 | -0.0098649 [-0.015341, -0.0036792] | 0.88156/False |
| 198 | I_U | F_int | surface | 0.044478 | 0.040506 | -0.0039725 [-0.0096756, 0.00022383] | 0.88156/False |
| 594 | response | F_log | reuse_only | 0.059339 | 0.050027 | -0.0093121 [-0.011745, -0.0065429] | 0.7318/False |
| 594 | I_T | F_log | reuse_only | 0.023827 | 0.028401 | 0.004574 [0.0025337, 0.0063607] | 0.7318/False |
| 594 | I_U | F_log | reuse_only | 0.058745 | 0.070322 | 0.011577 [-0.022699, 0.043437] | 0.7318/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=88 F MAE=0.10347; 5% n=332 F MAE=0.063812.

Held 66: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.01. Empirical interpolation support 0/24, MAE=NA, F MAE on identical support=NA.

Held 132: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.01. Empirical interpolation support 54/56, MAE=0.018534, F MAE on identical support=0.045138.

Held 198: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=1.0. Empirical interpolation support 24/24, MAE=0.013591, F MAE on identical support=0.033637.

Held 594: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.0001. Empirical interpolation support 0/24, MAE=NA, F MAE on identical support=NA.

### math / student

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| gemma3-1b | response | F_log | constant | 0.045389 | 0.065996 | 0.020607 [0.001754, 0.033904] | 1.0825/False |
| gemma3-1b | I_T | F_log | mean_effect | 0.028683 | 0.024565 | -0.0041178 [-0.013393, 0.0073494] | 1.0825/False |
| gemma3-1b | I_U | F_log | mean_effect | 0.061604 | 0.090118 | 0.028514 [0.0030672, 0.045807] | 1.0825/False |
| gemma3-270m | response | F_log | constant | 0.088354 | 0.054318 | -0.034037 [-0.069098, -0.0074921] | 1.2317/False |
| gemma3-270m | I_T | F_log | mean_effect | 0.02115 | 0.038426 | 0.017276 [-0.00041605, 0.029587] | 1.2317/False |
| gemma3-270m | I_U | F_log | mean_effect | 0.076368 | 0.076079 | -0.00028882 [-0.076667, 0.053953] | 1.2317/False |
| gemma3-4b | response | F_log | constant | 0.072295 | 0.085334 | 0.013039 [-0.011175, 0.035339] | 1.0935/False |
| gemma3-4b | I_T | F_log | mean_effect | 0.056831 | 0.064442 | 0.0076109 [0.00082572, 0.013239] | 1.0935/False |
| gemma3-4b | I_U | F_log | mean_effect | 0.12949 | 0.14715 | 0.017655 [0.0063911, 0.033752] | 1.0935/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=44 F MAE=0.11375; 5% n=166 F MAE=0.071982.

Held gemma3-1b: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/52, MAE=NA, F MAE on identical support=NA.

Held gemma3-270m: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/24, MAE=NA, F MAE on identical support=NA.

Held gemma3-4b: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/52, MAE=NA, F MAE on identical support=NA.

### math / largest_budget

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| T>=150000 | response | F_log | constant | 0.071841 | 0.094718 | 0.022876 [-0.011349, 0.061097] | -1/True |
| T>=150000 | I_T | F_log | mean_effect | 0.046944 | 0.048106 | 0.0011619 [-0.014882, 0.018161] | -1/True |
| T>=150000 | I_U | F_log | mean_effect | 0.10375 | 0.14088 | 0.037131 [0.023129, 0.056256] | -1/True |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=31 F MAE=0.11976; 5% n=60 F MAE=0.093686.

Held T>=150000: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/34, MAE=NA, F MAE on identical support=NA.

### math / pool_seed

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| 41 | response | F_int | surface | 0.047128 | 0.045724 | -0.0014039 [-0.0066328, 0.0026365] | 0.95043/False |
| 41 | I_T | F_int | budget_only | 0.036593 | 0.037582 | 0.00098915 [-0.0077651, 0.010273] | 0.95043/False |
| 41 | I_U | F_int | surface | 0.064969 | 0.058459 | -0.0065099 [-0.014801, 0.0026029] | 0.95043/False |
| 42 | response | F_curv | surface | 0.039542 | 0.033653 | -0.0058889 [-0.010591, 0.00020158] | 0.9884/False |
| 42 | I_T | F_curv | surface | 0.037577 | 0.038225 | 0.000648 [-0.0072647, 0.014616] | 0.9884/False |
| 42 | I_U | F_curv | surface | 0.045899 | 0.05785 | 0.011952 [0.002267, 0.028787] | 0.9884/False |
| 51 | response | F_curv | surface | 0.028592 | 0.034123 | 0.0055312 [0.0028298, 0.0082118] | 1.0781/False |
| 51 | I_T | F_curv | surface | 0.040883 | 0.033605 | -0.0072775 [-0.010411, -0.0041199] | 1.0781/False |
| 51 | I_U | F_curv | surface | 0.050386 | 0.067025 | 0.016639 [0.0029481, 0.045069] | 1.0781/False |
| 52 | response | F_curv | surface | 0.026444 | 0.033698 | 0.0072536 [0.0052846, 0.0092074] | 1.071/False |
| 52 | I_T | F_curv | surface | 0.02705 | 0.021519 | -0.005531 [-0.0094111, -0.0016208] | 1.071/False |
| 52 | I_U | F_curv | surface | 0.057095 | 0.073811 | 0.016717 [0.0019902, 0.045037] | 1.071/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=76 F MAE=0.07015; 5% n=275 F MAE=0.039877.

Held 41: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=1.0. Empirical interpolation support 12/36, MAE=0.010051, F MAE on identical support=0.032978.

Held 42: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.01. Empirical interpolation support 20/36, MAE=0.017439, F MAE on identical support=0.024568.

Held 51: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.01. Empirical interpolation support 28/28, MAE=0.011036, F MAE on identical support=0.028592.

Held 52: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.01. Empirical interpolation support 26/28, MAE=0.012558, F MAE on identical support=0.023667.

### QA-2Wiki / data_rung

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| 66 | response | F_curv | zero | 11.242 | 2.9964 | -8.2454 [-12.754, -3.0775] | 3/True |
| 66 | I_T | NA | NA | NA | NA | unscorable | 3/True |
| 66 | I_U | F_curv | zero | 11.123 | 3.3747 | -7.7488 [-12.493, -1.1108] | 3/True |
| 132 | response | F_log | surface | 0.77702 | 0.68993 | -0.087092 [-0.2679, 0.15594] | 2.0551/False |
| 132 | I_T | NA | NA | NA | NA | unscorable | 2.0551/False |
| 132 | I_U | F_log | surface | 1.1166 | 0.86882 | -0.24777 [-0.46553, -0.02622] | 2.0551/False |
| 198 | response | F_log | surface | 1.2084 | 0.96052 | -0.24784 [-0.25272, -0.24219] | 0.17489/False |
| 198 | I_T | NA | NA | NA | NA | unscorable | 0.17489/False |
| 198 | I_U | F_log | mean_effect | 1.4471 | 1.761 | 0.3139 [-0.42175, 1.0841] | 0.17489/False |
| 594 | response | F_int | surface | 1.2906 | 1.1511 | -0.13945 [-0.17742, -0.10141] | 1.6791/False |
| 594 | I_T | NA | NA | NA | NA | unscorable | 1.6791/False |
| 594 | I_U | F_int | surface | 1.3653 | 1.1859 | -0.17944 [-0.32491, -0.0027909] | 1.6791/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=62 F MAE=5.075; 5% n=120 F MAE=4.1185.

Held 66: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.01. Empirical interpolation support 0/6, MAE=NA, F MAE on identical support=NA.

Held 132: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/4, MAE=NA, F MAE on identical support=NA.

Held 198: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.0001. Empirical interpolation support 3/6, MAE=0.76463, F MAE on identical support=1.2615.

Held 594: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/6, MAE=NA, F MAE on identical support=NA.

### QA-2Wiki / student

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| gemma3-1b | response | F_log | constant | 0.88212 | 1.5872 | 0.70506 [0.09844, 1.3122] | 1.1839/False |
| gemma3-1b | I_T | NA | NA | NA | NA | unscorable | 1.1839/False |
| gemma3-1b | I_U | F_log | mean_effect | 1.0836 | 1.3281 | 0.24455 [-0.35996, 0.89627] | 1.1839/False |
| gemma3-270m | response | F_log | constant | 1.4639 | 1.4631 | -0.00074108 [-1.4897, 1.0892] | 1.3938/False |
| gemma3-270m | I_T | NA | NA | NA | NA | unscorable | 1.3938/False |
| gemma3-270m | I_U | F_log | mean_effect | 1.6312 | 1.7982 | 0.16702 [-0.5768, 1.4561] | 1.3938/False |
| gemma3-4b | response | F_log | constant | 1.3083 | 2.3757 | 1.0674 [0.35612, 1.9152] | 2.3772/False |
| gemma3-4b | I_T | NA | NA | NA | NA | unscorable | 2.3772/False |
| gemma3-4b | I_U | F_log | mean_effect | 1.9839 | 2.4566 | 0.4727 [-0.22192, 1.4197] | 2.3772/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=31 F MAE=1.6508; 5% n=60 F MAE=1.4624.

Held gemma3-1b: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/8, MAE=NA, F MAE on identical support=NA.

Held gemma3-270m: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/6, MAE=NA, F MAE on identical support=NA.

Held gemma3-4b: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/8, MAE=NA, F MAE on identical support=NA.

### QA-2Wiki / largest_budget

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| T>=150000 | all | NA | NA | NA | NA | unscorable: no positive train or test responses | NA |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=0 F MAE=NA; 5% n=0 F MAE=NA.

### QA-2Wiki / pool_seed

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| 41 | response | F_log | surface | 0.99316 | 0.60467 | -0.38849 [-0.51803, -0.12319] | 0.84174/False |
| 41 | I_T | NA | NA | NA | NA | unscorable | 0.84174/False |
| 41 | I_U | F_log | surface | 1.5093 | 1.192 | -0.31737 [-0.73791, 0.27259] | 0.84174/False |
| 42 | response | F_log | surface | 1.2088 | 1.1077 | -0.10116 [-0.22821, 0.062483] | 0.49443/False |
| 42 | I_T | NA | NA | NA | NA | unscorable | 0.49443/False |
| 42 | I_U | F_log | mean_effect | 1.6815 | 1.6204 | -0.061091 [-0.61205, 0.55348] | 0.49443/False |
| 51 | response | F_curv | surface | 0.99368 | 0.79155 | -0.20213 [-0.43761, 0.035178] | 1.4037/False |
| 51 | I_T | NA | NA | NA | NA | unscorable | 1.4037/False |
| 51 | I_U | F_curv | surface | 0.82027 | 0.73371 | -0.086563 [-0.41225, 0.1218] | 1.4037/False |
| 52 | response | F_curv | surface | 0.71283 | 0.52236 | -0.19047 [-0.38899, 0.0095935] | 1.3375/False |
| 52 | I_T | NA | NA | NA | NA | unscorable | 1.3375/False |
| 52 | I_U | F_curv | surface | 0.64425 | 0.63217 | -0.012079 [-0.35004, 0.28494] | 1.3375/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=50 F MAE=1.5477; 5% n=102 F MAE=1.4441.

Held 41: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.0001. Empirical interpolation support 0/9, MAE=NA, F MAE on identical support=NA.

Held 42: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.0001. Empirical interpolation support 2/9, MAE=0.62501, F MAE on identical support=0.63832.

Held 51: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.0001. Empirical interpolation support 0/2, MAE=NA, F MAE on identical support=NA.

Held 52: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.0001. Empirical interpolation support 0/2, MAE=NA, F MAE on identical support=NA.

### QA-MuSiQue / data_rung

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| 66 | response | F_curv | surface | 1.7734 | 0.7318 | -1.0416 [-1.595, -0.57752] | 2.5867/False |
| 66 | I_T | NA | NA | NA | NA | unscorable | 2.5867/False |
| 66 | I_U | F_curv | surface | 1.7866 | 0.7118 | -1.0748 [-1.8272, -0.5108] | 2.5867/False |
| 132 | response | F_curv | reuse_only | 0.47202 | 0.494 | 0.021976 [-0.079826, 0.12344] | 2.5346/False |
| 132 | I_T | NA | NA | NA | NA | unscorable | 2.5346/False |
| 132 | I_U | F_curv | surface | 0.5402 | 0.61383 | 0.073634 [-0.10559, 0.28006] | 2.5346/False |
| 198 | response | F_curv | reuse_only | 0.13624 | 0.61293 | 0.47668 [0.24762, 0.71727] | 1.4394/False |
| 198 | I_T | NA | NA | NA | NA | unscorable | 1.4394/False |
| 198 | I_U | F_curv | mean_effect | 0.19395 | 0.76484 | 0.57089 [0.27627, 0.7889] | 1.4394/False |
| 594 | response | F_int | surface | 0.61487 | 0.5343 | -0.080575 [-0.094873, -0.066226] | 1.7375/False |
| 594 | I_T | NA | NA | NA | NA | unscorable | 1.7375/False |
| 594 | I_U | F_int | surface | 0.66438 | 0.57124 | -0.093141 [-0.14839, -0.027407] | 1.7375/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=62 F MAE=0.89759; 5% n=120 F MAE=0.78133.

Held 66: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/6, MAE=NA, F MAE on identical support=NA.

Held 132: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=1.0. Empirical interpolation support 0/4, MAE=NA, F MAE on identical support=NA.

Held 198: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 3/6, MAE=0.37261, F MAE on identical support=0.16707.

Held 594: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/6, MAE=NA, F MAE on identical support=NA.

### QA-MuSiQue / student

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| gemma3-1b | response | F_log | constant | 0.30274 | 0.58962 | 0.28688 [0.039081, 0.5221] | 1.3638/False |
| gemma3-1b | I_T | NA | NA | NA | NA | unscorable | 1.3638/False |
| gemma3-1b | I_U | F_log | mean_effect | 0.47447 | 0.60097 | 0.1265 [-0.1275, 0.34668] | 1.3638/False |
| gemma3-270m | response | F_log | constant | 0.14473 | 0.8681 | 0.72338 [0.40783, 0.95282] | 1.549/False |
| gemma3-270m | I_T | NA | NA | NA | NA | unscorable | 1.549/False |
| gemma3-270m | I_U | F_log | mean_effect | 0.28582 | 0.819 | 0.53318 [0.27083, 0.88702] | 1.549/False |
| gemma3-4b | response | F_log | constant | 0.42105 | 0.94997 | 0.52892 [0.021417, 1.1394] | 2.1407/False |
| gemma3-4b | I_T | NA | NA | NA | NA | unscorable | 2.1407/False |
| gemma3-4b | I_U | F_log | mean_effect | 0.67589 | 0.97892 | 0.30303 [-0.14146, 0.80632] | 2.1407/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=31 F MAE=0.47663; 5% n=60 F MAE=0.45184.

Held gemma3-1b: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/8, MAE=NA, F MAE on identical support=NA.

Held gemma3-270m: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/6, MAE=NA, F MAE on identical support=NA.

Held gemma3-4b: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/8, MAE=NA, F MAE on identical support=NA.

### QA-MuSiQue / largest_budget

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| T>=150000 | all | NA | NA | NA | NA | unscorable: no positive train or test responses | NA |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=0 F MAE=NA; 5% n=0 F MAE=NA.

### QA-MuSiQue / pool_seed

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| 41 | response | F_int | surface | 0.42738 | 0.35937 | -0.068013 [-0.132, -0.01368] | 1.8242/False |
| 41 | I_T | NA | NA | NA | NA | unscorable | 1.8242/False |
| 41 | I_U | F_int | mean_effect | 0.77008 | 0.68337 | -0.08671 [-0.40273, 0.23446] | 1.8242/False |
| 42 | response | F_int | surface | 0.51284 | 0.45538 | -0.057463 [-0.11846, -0.0066602] | 1.6/False |
| 42 | I_T | NA | NA | NA | NA | unscorable | 1.6/False |
| 42 | I_U | F_int | mean_effect | 0.79603 | 0.68606 | -0.10997 [-0.39839, 0.243] | 1.6/False |
| 51 | response | F_curv | surface | 0.51625 | 0.28782 | -0.22842 [-0.49154, 0.036732] | 2.138/False |
| 51 | I_T | NA | NA | NA | NA | unscorable | 2.138/False |
| 51 | I_U | F_curv | surface | 0.49538 | 0.33075 | -0.16463 [-0.61636, 0.20957] | 2.138/False |
| 52 | response | F_curv | surface | 0.37562 | 0.44312 | 0.067499 [0.00067764, 0.13381] | 1.1561/False |
| 52 | I_T | NA | NA | NA | NA | unscorable | 1.1561/False |
| 52 | I_U | F_curv | surface | 0.33957 | 0.42193 | 0.082356 [0.0013905, 0.18241] | 1.1561/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=50 F MAE=0.80315; 5% n=102 F MAE=0.65329.

Held 41: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=1.0. Empirical interpolation support 0/9, MAE=NA, F MAE on identical support=NA.

Held 42: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=1.0. Empirical interpolation support 2/9, MAE=0.2607, F MAE on identical support=0.29344.

Held 51: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/2, MAE=NA, F MAE on identical support=NA.

Held 52: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.0001. Empirical interpolation support 0/2, MAE=NA, F MAE on identical support=NA.

### QA-probe / data_rung

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| 66 | response | F_int | surface | 1.0128 | 0.67936 | -0.33344 [-0.39642, -0.28808] | 2.9285/False |
| 66 | I_T | F_int | zero | 0.72948 | 1.2775 | 0.54803 [0.16459, 0.85793] | 2.9285/False |
| 66 | I_U | F_int | surface | 1.2783 | 0.84025 | -0.43803 [-0.72514, -0.0042447] | 2.9285/False |
| 132 | response | F_int | surface | 0.71702 | 0.64154 | -0.075481 [-0.22264, 0.070667] | 2.0883/False |
| 132 | I_T | F_int | surface | 0.43918 | 0.46173 | 0.022548 [0.00075387, 0.041459] | 2.0883/False |
| 132 | I_U | F_int | surface | 1.0321 | 0.83566 | -0.19644 [-0.38198, -0.017175] | 2.0883/False |
| 198 | response | F_int | surface | 0.59028 | 0.45258 | -0.1377 [-0.21185, -0.043966] | 1.8645/False |
| 198 | I_T | F_int | surface | 0.31412 | 0.35106 | 0.036942 [0.018599, 0.049415] | 1.8645/False |
| 198 | I_U | F_int | surface | 0.76824 | 0.63542 | -0.13282 [-0.29507, 0.078469] | 1.8645/False |
| 594 | response | F_int | surface | 0.63138 | 0.46585 | -0.16554 [-0.19858, -0.13506] | 1.7978/False |
| 594 | I_T | F_int | surface | 0.53593 | 0.43609 | -0.099843 [-0.11476, -0.077191] | 1.7978/False |
| 594 | I_U | F_int | surface | 1.1604 | 0.82409 | -0.33626 [-0.47179, -0.10849] | 1.7978/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=88 F MAE=1.1406; 5% n=332 F MAE=0.85653.

Held 66: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/24, MAE=NA, F MAE on identical support=NA.

Held 132: Inner grouped response MAE; same F for all three targets; descriptor=initial_loss, lambda=0.01. Empirical interpolation support 54/56, MAE=0.19571, F MAE on identical support=0.69807.

Held 198: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 24/24, MAE=0.21652, F MAE on identical support=0.59028.

Held 594: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/24, MAE=NA, F MAE on identical support=NA.

### QA-probe / student

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| gemma3-1b | response | F_log | constant | 1.0169 | 0.76661 | -0.25029 [-0.33346, -0.14055] | 1.6617/False |
| gemma3-1b | I_T | F_log | mean_effect | 0.68099 | 0.51754 | -0.16345 [-0.2592, -0.079115] | 1.6617/False |
| gemma3-1b | I_U | F_log | mean_effect | 1.0203 | 1.3845 | 0.36423 [-0.095506, 0.71225] | 1.6617/False |
| gemma3-270m | response | F_log | constant | 0.71494 | 0.48555 | -0.22939 [-0.44248, -0.074287] | 1.7744/False |
| gemma3-270m | I_T | F_log | mean_effect | 0.67061 | 0.55548 | -0.11512 [-0.47802, 0.15041] | 1.7744/False |
| gemma3-270m | I_U | F_log | mean_effect | 0.96643 | 1.2371 | 0.27063 [-0.7266, 1.0478] | 1.7744/False |
| gemma3-4b | response | F_log | constant | 1.296 | 1.1162 | -0.17978 [-0.30274, -0.085014] | 2.2135/False |
| gemma3-4b | I_T | F_log | mean_effect | 1.1479 | 0.90063 | -0.24724 [-0.3295, -0.13191] | 2.2135/False |
| gemma3-4b | I_U | F_log | mean_effect | 1.8895 | 1.9791 | 0.08958 [-0.15035, 0.33143] | 2.2135/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=44 F MAE=1.7108; 5% n=166 F MAE=1.0002.

Held gemma3-1b: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/52, MAE=NA, F MAE on identical support=NA.

Held gemma3-270m: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/24, MAE=NA, F MAE on identical support=NA.

Held gemma3-4b: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/52, MAE=NA, F MAE on identical support=NA.

### QA-probe / largest_budget

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| T>=150000 | response | F_log | constant | 2.1809 | 1.4432 | -0.73775 [-0.90318, -0.56652] | 3/True |
| T>=150000 | I_T | F_log | mean_effect | 1.7044 | 0.78862 | -0.91581 [-0.99225, -0.77781] | 3/True |
| T>=150000 | I_U | F_log | mean_effect | 1.7871 | 2.0313 | 0.24416 [0.067769, 0.46682] | 3/True |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=31 F MAE=2.0325; 5% n=60 F MAE=1.6668.

Held T>=150000: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/34, MAE=NA, F MAE on identical support=NA.

### QA-probe / pool_seed

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| 41 | response | F_int | surface | 0.493 | 0.4383 | -0.054697 [-0.13003, 0.037828] | 1.7497/False |
| 41 | I_T | F_int | surface | 0.40258 | 0.37258 | -0.030001 [-0.070308, 0.0063195] | 1.7497/False |
| 41 | I_U | F_int | surface | 0.79775 | 0.66628 | -0.13146 [-0.30962, 0.13697] | 1.7497/False |
| 42 | response | F_int | surface | 0.5651 | 0.49622 | -0.068879 [-0.11385, -0.018145] | 1.7171/False |
| 42 | I_T | F_int | surface | 0.45953 | 0.42139 | -0.038142 [-0.082315, 0.0027307] | 1.7171/False |
| 42 | I_U | F_int | surface | 0.90409 | 0.77934 | -0.12475 [-0.28085, 0.065445] | 1.7171/False |
| 51 | response | F_int | surface | 0.44766 | 0.32876 | -0.1189 [-0.28242, 0.043375] | 1.8063/False |
| 51 | I_T | F_int | surface | 0.39034 | 0.43588 | 0.04554 [-0.012616, 0.10325] | 1.8063/False |
| 51 | I_U | F_int | surface | 0.80768 | 0.53126 | -0.27642 [-0.52446, -0.041268] | 1.8063/False |
| 52 | response | F_int | surface | 0.50372 | 0.36601 | -0.13771 [-0.37602, 0.098777] | 1.8176/False |
| 52 | I_T | F_int | surface | 0.36525 | 0.3873 | 0.022052 [-0.058769, 0.10225] | 1.8176/False |
| 52 | I_U | F_int | surface | 0.99601 | 0.72952 | -0.26649 [-0.72461, 0.33191] | 1.8176/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=76 F MAE=0.9519; 5% n=275 F MAE=0.67023.

Held 41: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 12/36, MAE=0.1117, F MAE on identical support=0.45551.

Held 42: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 20/36, MAE=0.1221, F MAE on identical support=0.4629.

Held 51: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.0001. Empirical interpolation support 28/28, MAE=0.14274, F MAE on identical support=0.44766.

Held 52: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.0001. Empirical interpolation support 26/28, MAE=0.1477, F MAE on identical support=0.48032.

### QA-TriviaQA / data_rung

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| 66 | response | F_int | surface | 0.61771 | 0.62605 | 0.0083385 [-0.00069324, 0.013293] | 0.87913/False |
| 66 | I_T | NA | NA | NA | NA | unscorable | 0.87913/False |
| 66 | I_U | F_int | surface | 0.59282 | 0.60545 | 0.012627 [0.010223, 0.015527] | 0.87913/False |
| 132 | response | F_log | surface | 0.31916 | 0.28318 | -0.035973 [-0.051994, -0.0057964] | 0.80566/False |
| 132 | I_T | NA | NA | NA | NA | unscorable | 0.80566/False |
| 132 | I_U | F_log | surface | 0.2834 | 0.2499 | -0.033499 [-0.072799, 0.00027243] | 0.80566/False |
| 198 | response | F_curv | surface | 0.29816 | 0.2313 | -0.066862 [-0.1943, 0.094493] | 1.0673/False |
| 198 | I_T | NA | NA | NA | NA | unscorable | 1.0673/False |
| 198 | I_U | F_curv | surface | 0.25713 | 0.26652 | 0.0093894 [-0.056739, 0.065895] | 1.0673/False |
| 594 | response | F_curv | surface | 0.15816 | 0.19817 | 0.040009 [-0.015067, 0.083397] | 1.0578/False |
| 594 | I_T | NA | NA | NA | NA | unscorable | 1.0578/False |
| 594 | I_U | F_curv | surface | 0.35488 | 0.33076 | -0.024126 [-0.093934, 0.044082] | 1.0578/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=62 F MAE=0.4263; 5% n=120 F MAE=0.35786.

Held 66: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/6, MAE=NA, F MAE on identical support=NA.

Held 132: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/4, MAE=NA, F MAE on identical support=NA.

Held 198: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=1.0. Empirical interpolation support 3/6, MAE=0.10894, F MAE on identical support=0.30377.

Held 594: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=1.0. Empirical interpolation support 0/6, MAE=NA, F MAE on identical support=NA.

### QA-TriviaQA / student

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| gemma3-1b | response | F_log | constant | 0.53759 | 0.55099 | 0.013401 [-0.2505, 0.29386] | 0.84463/False |
| gemma3-1b | I_T | NA | NA | NA | NA | unscorable | 0.84463/False |
| gemma3-1b | I_U | F_log | mean_effect | 0.25213 | 0.3152 | 0.063075 [-0.12624, 0.26299] | 0.84463/False |
| gemma3-270m | response | F_log | constant | 1.1725 | 0.44531 | -0.72723 [-1.3459, -0.2794] | 0.92454/False |
| gemma3-270m | I_T | NA | NA | NA | NA | unscorable | 0.92454/False |
| gemma3-270m | I_U | F_log | mean_effect | 0.82922 | 0.33443 | -0.4948 [-0.92764, 0.01543] | 0.92454/False |
| gemma3-4b | response | F_log | constant | 0.95573 | 0.92089 | -0.034835 [-0.20672, 0.13638] | 1.8651/False |
| gemma3-4b | I_T | NA | NA | NA | NA | unscorable | 1.8651/False |
| gemma3-4b | I_U | F_log | mean_effect | 0.64886 | 0.72231 | 0.073449 [-0.0041651, 0.16392] | 1.8651/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=31 F MAE=0.62881; 5% n=60 F MAE=0.56764.

Held gemma3-1b: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/8, MAE=NA, F MAE on identical support=NA.

Held gemma3-270m: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/6, MAE=NA, F MAE on identical support=NA.

Held gemma3-4b: Fixed parsimonious settings: fewer than three usable inner grouped folds; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/8, MAE=NA, F MAE on identical support=NA.

### QA-TriviaQA / largest_budget

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| T>=150000 | all | NA | NA | NA | NA | unscorable: no positive train or test responses | NA |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=0 F MAE=NA; 5% n=0 F MAE=NA.

### QA-TriviaQA / pool_seed

| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |
|---|---|---|---|---:|---:|---|---|
| 41 | response | F_int | surface | 0.30896 | 0.28512 | -0.023839 [-0.081558, 0.036629] | 1.2509/False |
| 41 | I_T | NA | NA | NA | NA | unscorable | 1.2509/False |
| 41 | I_U | F_int | surface | 0.38029 | 0.30423 | -0.076059 [-0.11106, -0.030085] | 1.2509/False |
| 42 | response | F_int | surface | 0.29603 | 0.29296 | -0.0030709 [-0.033647, 0.028592] | 0.69355/False |
| 42 | I_T | NA | NA | NA | NA | unscorable | 0.69355/False |
| 42 | I_U | F_int | surface | 0.31863 | 0.26937 | -0.049258 [-0.087195, 0.0014496] | 0.69355/False |
| 51 | response | F_log | surface | 0.33295 | 0.28201 | -0.050931 [-0.05277, -0.049106] | 0.90528/False |
| 51 | I_T | NA | NA | NA | NA | unscorable | 0.90528/False |
| 51 | I_U | F_log | surface | 0.26532 | 0.237 | -0.028318 [-0.07296, 0.0087394] | 0.90528/False |
| 52 | response | F_curv | surface | 0.28763 | 0.25203 | -0.035606 [-0.06683, -0.00414] | 0.92877/False |
| 52 | I_T | NA | NA | NA | NA | unscorable | 0.92877/False |
| 52 | I_U | F_curv | surface | 0.18145 | 0.18668 | 0.0052299 [-0.04772, 0.049577] | 0.92877/False |

Data-pair tolerance sensitivity (same frozen F, no retuning): 0.5% n=50 F MAE=0.36341; 5% n=102 F MAE=0.2965.

Held 41: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=1.0. Empirical interpolation support 0/9, MAE=NA, F MAE on identical support=NA.

Held 42: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 2/9, MAE=0.068886, F MAE on identical support=0.3214.

Held 51: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.01. Empirical interpolation support 0/2, MAE=NA, F MAE on identical support=NA.

Held 52: Inner grouped response MAE; same F for all three targets; descriptor=log_parameters, lambda=0.0001. Empirical interpolation support 0/2, MAE=NA, F MAE on identical support=NA.

