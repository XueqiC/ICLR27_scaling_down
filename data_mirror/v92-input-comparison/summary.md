V92 VERDICT — development only; lower MAE is better.
Statistics budget vs K0 + dense anchor, SAME form; gain must exceed the FULL 95% interval width.
Positive gains are MAE reductions in nats. OLS and ridge are separate fixed comparisons.

| Arm | Capability | Statistics helped? | OLS gain [95% CI]; width | Ridge gain [95% CI]; width | Source clusters |
|---|---|---|---|---|---|
| pruning | math | YES: ridge | 0.02606 [0.00167, 0.06089]; 0.05922 | 0.08356 [0.04882, 0.12227]; 0.07345 | 9 |
| pruning | code | NO under fixed rule | -0.01182 [-0.07016, 0.04275]; 0.11290 | 0.05321 [0.01983, 0.08508]; 0.06525 | 9 |
| pruning | qa | YES: ridge | 0.12777 [-0.13266, 0.38482]; 0.51748 | 0.10897 [0.06206, 0.15684]; 0.09478 | 9 |
| grouped_quantization | math | YES: ridge | 0.07527 [-0.08062, 0.20412]; 0.28474 | 0.13712 [0.08715, 0.18417]; 0.09702 | 9 |
| grouped_quantization | code | NO under fixed rule | 0.00990 [-0.09775, 0.12628]; 0.22403 | 0.06829 [-0.00295, 0.13769]; 0.14064 | 9 |
| grouped_quantization | qa | YES: ridge | 0.49851 [-0.00050, 1.03445]; 1.03495 | 0.31827 [0.22170, 0.42892]; 0.20722 | 9 |
| per_channel_quantization | math | NO under fixed rule | -0.01889 [-0.11625, 0.05688]; 0.17313 | 0.08726 [0.03153, 0.14338]; 0.11185 | 9 |
| per_channel_quantization | code | NO under fixed rule | -0.03841 [-0.14117, 0.05976]; 0.20093 | 0.08110 [-0.00865, 0.17812]; 0.18678 | 9 |
| per_channel_quantization | qa | NO under fixed rule | -0.53423 [-1.42322, 0.20249]; 1.62571 | 0.17537 [0.01184, 0.34076]; 0.32892 | 9 |

In 9/9 arm/capability pairs, both statistics-augmented linear candidates still have higher MAE than the K0 source-free median curve. Passing the incremental input rule is not a win over the delivered predictor.

Most-fragile capability accuracy (ridge, dense anchor → dense statistics): pruning 58.3% → 41.7%; grouped_quantization 40.7% → 49.4%; per_channel_quantization 36.1% → 41.7%. These are descriptive ranking results, separate from the MAE reading rule.

Statistics clear the fixed reading rule in 4 of 9 arm-and-capability pairs, all through ridge: pruning math, pruning QA, grouped-quantization math, and grouped-quantization QA.

Development protocol and coverage

Pruning, grouped RTN, and per-channel RTN each have 9 source states: 3 sizes × steps 16k/64k/143k. Primary/size evaluation contains 459 response rows (108 pruning, 243 grouped quantization, 108 per-channel quantization). All grouped step64k files are present; no source states are missing. No missing responses or descriptors are imputed.
Coverage is read from the saved observations and provenance; the pinned summary.json retains its historical coverage note.
Primary/size evaluation uses the common measured configuration grid: pruning densities .6/.7/.8/.9; grouped bits 3/4/5 × group sizes 64/128/256; per-channel bits 3/4/6/8. Budgets use exactly the same rows within each arm.
Four additional pruning state-density cells (.55/.65 for 160M-step143k and 1.4B-step64k) are scored separately. Grouped g32/g512 confirmation additions in the V54 files are excluded. Only the listed development response files and V91 dense descriptors are read; no frozen prediction/test artifacts are accessed.
The V91 dense measurements are reused, not rerun: 64 probe references per source/capability. L0 is the true dense anchor recorded with each response arm. Descriptor L differences, sample counts, revisions and file hashes are retained in the audit. Historical arm-local L0 and V91 L differ by at most 0.01319 nats; the cause is not remeasured, and descriptor L is never substituted.

Input definitions

| Budget | Inputs |
|---|---|
| K0 | N0, D0, density or bits (and group size for grouped RTN) |
| dense_anchor | K0 + arm-local dense capability loss L0 |
| dense_statistics | dense_anchor + capability-specific B, V, W from V91 dense descriptors |

N0 counts transformer matrices excluding embeddings/head, following V36; D0 = step × 2,097,152 processed tokens. OLS/ridge use raw inputs standardized on training response rows, separately per capability. There is one coefficient per input plus intercept; no nonlinear expansion or interaction search. B = z[y] − E_p[z], V = 1 − Σp², W = Var_p(z), pooled over reference tokens. Descriptor L is audited but never added as a predictor.

Candidate and uncertainty definitions

Zero predicts signed ΔL=0. Constant is the training capability mean. Median is the equal-source median at each training configuration. OLS is unregularized least squares. Ridge minimizes mean squared error + α‖β_nonintercept‖²; α ∈ [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] is selected by inner grouped CV MAE. Size holdouts use inner size folds; all other evaluations use inner source folds. Every inner fold refits its scaler. The delivered new-source branch is the median for all three arms/capabilities (analysis/final_rule.py); it does not consume L0/B/V/W when predicting ΔL. Its absolute-loss presentation would add L0, which is unavailable to K0, so all candidates are evaluated directly on ΔL.

MAE and paired improvements weight source states equally, then configurations equally within a state. 95% percentile intervals resample whole source states 20,000 times (seed 9201), pairing predictions on identical rows. Intervals condition on the fitted cross-validation predictions; they do not rerun fitting on bootstrap samples. Leave-one-size-out has three size folds but intervals still cluster on the 9 source states in every arm; these small-cluster intervals are development evidence, not independent confirmation. No candidate is selected by outer-fold MAE. The rule is applied separately to the two fixed linear estimators; YES names the one that clears, without selecting a new predictor or adjusting for multiple comparisons.

In grouped quantization, the saved statistics-budget OLS design is full rank in every primary fold (rank 9 of 9 columns), with 8 training source states. OLS uses least squares; ridge regularizes the same design. Design ranks, coefficients, fold-local scalers and penalty scores are recorded in JSON. Audit row sets are interned as ordered indices into observations to avoid repeating the same training rows thousands of times.

MAE in nats: primary and size holdout

| Split | Arm | Capability | Budget | Zero | Constant | Median | OLS | Ridge | Delivered | Clusters |
|---|---|---|---|---|---|---|---|---|---|---|
| leave_one_source_state_out | pruning | math | K0 | 0.55789 | 0.66758 | 0.37594 | 0.65987 | 0.48884 | 0.37594 | 9 |
| leave_one_source_state_out | pruning | math | dense_anchor | 0.55789 | 0.66758 | 0.37594 | 0.48323 | 0.50534 | 0.37594 | 9 |
| leave_one_source_state_out | pruning | math | dense_statistics | 0.55789 | 0.66758 | 0.37594 | 0.45718 | 0.42178 | 0.37594 | 9 |
| leave_one_source_state_out | pruning | code | K0 | 0.66769 | 0.83207 | 0.47609 | 0.81943 | 0.63247 | 0.47609 | 9 |
| leave_one_source_state_out | pruning | code | dense_anchor | 0.66769 | 0.83207 | 0.47609 | 0.60903 | 0.58721 | 0.47609 | 9 |
| leave_one_source_state_out | pruning | code | dense_statistics | 0.66769 | 0.83207 | 0.47609 | 0.62085 | 0.53399 | 0.47609 | 9 |
| leave_one_source_state_out | pruning | qa | K0 | 0.49968 | 0.69747 | 0.52545 | 0.86388 | 0.69483 | 0.52545 | 9 |
| leave_one_source_state_out | pruning | qa | dense_anchor | 0.49968 | 0.69747 | 0.52545 | 1.00313 | 0.67815 | 0.52545 | 9 |
| leave_one_source_state_out | pruning | qa | dense_statistics | 0.49968 | 0.69747 | 0.52545 | 0.87536 | 0.56917 | 0.52545 | 9 |
| leave_one_size_out | pruning | math | K0 | 0.55789 | 0.68316 | 0.37939 | 1.49747 | 0.84790 | 0.37939 | 9 |
| leave_one_size_out | pruning | math | dense_anchor | 0.55789 | 0.68316 | 0.37939 | 0.74804 | 0.62502 | 0.37939 | 9 |
| leave_one_size_out | pruning | math | dense_statistics | 0.55789 | 0.68316 | 0.37939 | 0.53750 | 0.51029 | 0.37939 | 9 |
| leave_one_size_out | pruning | code | K0 | 0.66769 | 0.85752 | 0.54320 | 1.81920 | 1.06694 | 0.54320 | 9 |
| leave_one_size_out | pruning | code | dense_anchor | 0.66769 | 0.85752 | 0.54320 | 0.97013 | 0.94058 | 0.54320 | 9 |
| leave_one_size_out | pruning | code | dense_statistics | 0.66769 | 0.85752 | 0.54320 | 0.80617 | 0.73002 | 0.54320 | 9 |
| leave_one_size_out | pruning | qa | K0 | 0.49968 | 0.78284 | 0.59788 | 1.95713 | 0.64547 | 0.59788 | 9 |
| leave_one_size_out | pruning | qa | dense_anchor | 0.49968 | 0.78284 | 0.59788 | 2.41617 | 0.77809 | 0.59788 | 9 |
| leave_one_size_out | pruning | qa | dense_statistics | 0.49968 | 0.78284 | 0.59788 | 0.74022 | 0.74223 | 0.59788 | 9 |
| leave_one_source_state_out | grouped_quantization | math | K0 | 0.99040 | 1.42215 | 0.83278 | 1.86041 | 1.27794 | 0.83278 | 9 |
| leave_one_source_state_out | grouped_quantization | math | dense_anchor | 0.99040 | 1.42215 | 0.83278 | 1.33041 | 1.18150 | 0.83278 | 9 |
| leave_one_source_state_out | grouped_quantization | math | dense_statistics | 0.99040 | 1.42215 | 0.83278 | 1.25514 | 1.04438 | 0.83278 | 9 |
| leave_one_source_state_out | grouped_quantization | code | K0 | 1.09952 | 1.53431 | 0.91010 | 1.86331 | 1.34500 | 0.91010 | 9 |
| leave_one_source_state_out | grouped_quantization | code | dense_anchor | 1.09952 | 1.53431 | 0.91010 | 1.37473 | 1.20859 | 0.91010 | 9 |
| leave_one_source_state_out | grouped_quantization | code | dense_statistics | 1.09952 | 1.53431 | 0.91010 | 1.36483 | 1.14031 | 0.91010 | 9 |
| leave_one_source_state_out | grouped_quantization | qa | K0 | 0.96321 | 1.58730 | 0.97957 | 2.10696 | 1.52348 | 0.97957 | 9 |
| leave_one_source_state_out | grouped_quantization | qa | dense_anchor | 0.96321 | 1.58730 | 0.97957 | 2.39513 | 1.50312 | 0.97957 | 9 |
| leave_one_source_state_out | grouped_quantization | qa | dense_statistics | 0.96321 | 1.58730 | 0.97957 | 1.89662 | 1.18485 | 0.97957 | 9 |
| leave_one_size_out | grouped_quantization | math | K0 | 0.99040 | 1.46904 | 0.82773 | 3.69100 | 2.07329 | 0.82773 | 9 |
| leave_one_size_out | grouped_quantization | math | dense_anchor | 0.99040 | 1.46904 | 0.82773 | 2.97626 | 1.40224 | 0.82773 | 9 |
| leave_one_size_out | grouped_quantization | math | dense_statistics | 0.99040 | 1.46904 | 0.82773 | 1.71333 | 0.97925 | 0.82773 | 9 |
| leave_one_size_out | grouped_quantization | code | K0 | 1.09952 | 1.59451 | 0.91815 | 3.79694 | 2.15841 | 0.91815 | 9 |
| leave_one_size_out | grouped_quantization | code | dense_anchor | 1.09952 | 1.59451 | 0.91815 | 3.26253 | 1.35099 | 0.91815 | 9 |
| leave_one_size_out | grouped_quantization | code | dense_statistics | 1.09952 | 1.59451 | 0.91815 | 1.47167 | 1.29525 | 0.91815 | 9 |
| leave_one_size_out | grouped_quantization | qa | K0 | 0.96321 | 1.66740 | 1.06468 | 4.28578 | 1.35119 | 1.06468 | 9 |
| leave_one_size_out | grouped_quantization | qa | dense_anchor | 0.96321 | 1.66740 | 1.06468 | 5.60415 | 1.65551 | 1.06468 | 9 |
| leave_one_size_out | grouped_quantization | qa | dense_statistics | 0.96321 | 1.66740 | 1.06468 | 2.60624 | 1.63267 | 1.06468 | 9 |
| leave_one_source_state_out | per_channel_quantization | math | K0 | 1.58035 | 2.31155 | 1.19239 | 2.14090 | 1.87530 | 1.19239 | 9 |
| leave_one_source_state_out | per_channel_quantization | math | dense_anchor | 1.58035 | 2.31155 | 1.19239 | 1.97760 | 1.83096 | 1.19239 | 9 |
| leave_one_source_state_out | per_channel_quantization | math | dense_statistics | 1.58035 | 2.31155 | 1.19239 | 1.99650 | 1.74369 | 1.19239 | 9 |
| leave_one_source_state_out | per_channel_quantization | code | K0 | 1.84075 | 2.67155 | 1.36677 | 2.41574 | 2.16014 | 1.36677 | 9 |
| leave_one_source_state_out | per_channel_quantization | code | dense_anchor | 1.84075 | 2.67155 | 1.36677 | 2.30085 | 2.11537 | 1.36677 | 9 |
| leave_one_source_state_out | per_channel_quantization | code | dense_statistics | 1.84075 | 2.67155 | 1.36677 | 2.33925 | 2.03427 | 1.36677 | 9 |
| leave_one_source_state_out | per_channel_quantization | qa | K0 | 1.52474 | 2.29831 | 1.31277 | 2.24138 | 1.92320 | 1.31277 | 9 |
| leave_one_source_state_out | per_channel_quantization | qa | dense_anchor | 1.52474 | 2.29831 | 1.31277 | 2.48823 | 1.95403 | 1.31277 | 9 |
| leave_one_source_state_out | per_channel_quantization | qa | dense_statistics | 1.52474 | 2.29831 | 1.31277 | 3.02246 | 1.77866 | 1.31277 | 9 |
| leave_one_size_out | per_channel_quantization | math | K0 | 1.58035 | 2.30907 | 1.12514 | 3.82583 | 2.41980 | 1.12514 | 9 |
| leave_one_size_out | per_channel_quantization | math | dense_anchor | 1.58035 | 2.30907 | 1.12514 | 2.91796 | 1.89951 | 1.12514 | 9 |
| leave_one_size_out | per_channel_quantization | math | dense_statistics | 1.58035 | 2.30907 | 1.12514 | 2.05493 | 1.85700 | 1.12514 | 9 |
| leave_one_size_out | per_channel_quantization | code | K0 | 1.84075 | 2.68372 | 1.26931 | 4.64477 | 2.87759 | 1.26931 | 9 |
| leave_one_size_out | per_channel_quantization | code | dense_anchor | 1.84075 | 2.68372 | 1.26931 | 3.40042 | 2.31461 | 1.26931 | 9 |
| leave_one_size_out | per_channel_quantization | code | dense_statistics | 1.84075 | 2.68372 | 1.26931 | 3.58101 | 2.31455 | 1.26931 | 9 |
| leave_one_size_out | per_channel_quantization | qa | K0 | 1.52474 | 2.29568 | 1.20301 | 4.41296 | 2.69826 | 1.20301 | 9 |
| leave_one_size_out | per_channel_quantization | qa | dense_anchor | 1.52474 | 2.29568 | 1.20301 | 5.39009 | 1.94186 | 1.20301 | 9 |
| leave_one_size_out | per_channel_quantization | qa | dense_statistics | 1.52474 | 2.29568 | 1.20301 | 2.42867 | 2.25494 | 1.20301 | 9 |

What the extra inputs buy with the same form

Zero/constant/median/delivered ignore the added inputs, so both adjacent-budget gains are exactly zero. The table gives OLS/ridge paired gains; positive means the upper budget improves. The complete comparisons and fold audits are in summary.json.

| Split | Arm | Capability | Budget transition | Form | Gain | 95% CI | Width | Clusters |
|---|---|---|---|---|---|---|---|---|
| leave_one_source_state_out | pruning | math | K0 → dense_anchor | ols | 0.17664 | [0.04999, 0.32904] | 0.27905 | 9 |
| leave_one_source_state_out | pruning | math | K0 → dense_anchor | ridge | -0.01650 | [-0.09968, 0.06338] | 0.16306 | 9 |
| leave_one_source_state_out | pruning | math | dense_anchor → dense_statistics | ols | 0.02606 | [0.00167, 0.06089] | 0.05922 | 9 |
| leave_one_source_state_out | pruning | math | dense_anchor → dense_statistics | ridge | 0.08356 | [0.04882, 0.12227] | 0.07345 | 9 |
| leave_one_source_state_out | pruning | code | K0 → dense_anchor | ols | 0.21039 | [0.05887, 0.37124] | 0.31237 | 9 |
| leave_one_source_state_out | pruning | code | K0 → dense_anchor | ridge | 0.04526 | [-0.03528, 0.12227] | 0.15755 | 9 |
| leave_one_source_state_out | pruning | code | dense_anchor → dense_statistics | ols | -0.01182 | [-0.07016, 0.04275] | 0.11290 | 9 |
| leave_one_source_state_out | pruning | code | dense_anchor → dense_statistics | ridge | 0.05321 | [0.01983, 0.08508] | 0.06525 | 9 |
| leave_one_source_state_out | pruning | qa | K0 → dense_anchor | ols | -0.13925 | [-0.37247, 0.05684] | 0.42932 | 9 |
| leave_one_source_state_out | pruning | qa | K0 → dense_anchor | ridge | 0.01668 | [-0.02541, 0.05464] | 0.08005 | 9 |
| leave_one_source_state_out | pruning | qa | dense_anchor → dense_statistics | ols | 0.12777 | [-0.13266, 0.38482] | 0.51748 | 9 |
| leave_one_source_state_out | pruning | qa | dense_anchor → dense_statistics | ridge | 0.10897 | [0.06206, 0.15684] | 0.09478 | 9 |
| leave_one_size_out | pruning | math | K0 → dense_anchor | ols | 0.74942 | [0.23088, 1.35155] | 1.12067 | 9 |
| leave_one_size_out | pruning | math | K0 → dense_anchor | ridge | 0.22288 | [0.06573, 0.40527] | 0.33955 | 9 |
| leave_one_size_out | pruning | math | dense_anchor → dense_statistics | ols | 0.21055 | [0.01201, 0.42596] | 0.41395 | 9 |
| leave_one_size_out | pruning | math | dense_anchor → dense_statistics | ridge | 0.11473 | [-0.01154, 0.25507] | 0.26661 | 9 |
| leave_one_size_out | pruning | code | K0 → dense_anchor | ols | 0.84907 | [0.20474, 1.63073] | 1.42599 | 9 |
| leave_one_size_out | pruning | code | K0 → dense_anchor | ridge | 0.12636 | [-0.08503, 0.32982] | 0.41484 | 9 |
| leave_one_size_out | pruning | code | dense_anchor → dense_statistics | ols | 0.16397 | [-0.08586, 0.44694] | 0.53280 | 9 |
| leave_one_size_out | pruning | code | dense_anchor → dense_statistics | ridge | 0.21056 | [-0.11976, 0.58010] | 0.69987 | 9 |
| leave_one_size_out | pruning | qa | K0 → dense_anchor | ols | -0.45904 | [-0.86797, -0.02116] | 0.84681 | 9 |
| leave_one_size_out | pruning | qa | K0 → dense_anchor | ridge | -0.13262 | [-0.27007, -0.00023] | 0.26984 | 9 |
| leave_one_size_out | pruning | qa | dense_anchor → dense_statistics | ols | 1.67594 | [0.50907, 3.01895] | 2.50988 | 9 |
| leave_one_size_out | pruning | qa | dense_anchor → dense_statistics | ridge | 0.03586 | [0.00093, 0.07641] | 0.07549 | 9 |
| leave_one_source_state_out | grouped_quantization | math | K0 → dense_anchor | ols | 0.53000 | [0.12337, 1.01002] | 0.88665 | 9 |
| leave_one_source_state_out | grouped_quantization | math | K0 → dense_anchor | ridge | 0.09644 | [-0.09711, 0.31135] | 0.40847 | 9 |
| leave_one_source_state_out | grouped_quantization | math | dense_anchor → dense_statistics | ols | 0.07527 | [-0.08062, 0.20412] | 0.28474 | 9 |
| leave_one_source_state_out | grouped_quantization | math | dense_anchor → dense_statistics | ridge | 0.13712 | [0.08715, 0.18417] | 0.09702 | 9 |
| leave_one_source_state_out | grouped_quantization | code | K0 → dense_anchor | ols | 0.48858 | [0.10109, 0.94617] | 0.84508 | 9 |
| leave_one_source_state_out | grouped_quantization | code | K0 → dense_anchor | ridge | 0.13641 | [-0.08317, 0.34426] | 0.42742 | 9 |
| leave_one_source_state_out | grouped_quantization | code | dense_anchor → dense_statistics | ols | 0.00990 | [-0.09775, 0.12628] | 0.22403 | 9 |
| leave_one_source_state_out | grouped_quantization | code | dense_anchor → dense_statistics | ridge | 0.06829 | [-0.00295, 0.13769] | 0.14064 | 9 |
| leave_one_source_state_out | grouped_quantization | qa | K0 → dense_anchor | ols | -0.28817 | [-0.89980, 0.26541] | 1.16521 | 9 |
| leave_one_source_state_out | grouped_quantization | qa | K0 → dense_anchor | ridge | 0.02036 | [-0.11452, 0.15609] | 0.27061 | 9 |
| leave_one_source_state_out | grouped_quantization | qa | dense_anchor → dense_statistics | ols | 0.49851 | [-0.00050, 1.03445] | 1.03495 | 9 |
| leave_one_source_state_out | grouped_quantization | qa | dense_anchor → dense_statistics | ridge | 0.31827 | [0.22170, 0.42892] | 0.20722 | 9 |
| leave_one_size_out | grouped_quantization | math | K0 → dense_anchor | ols | 0.71474 | [0.04646, 1.49356] | 1.44710 | 9 |
| leave_one_size_out | grouped_quantization | math | K0 → dense_anchor | ridge | 0.67106 | [0.24164, 1.16929] | 0.92764 | 9 |
| leave_one_size_out | grouped_quantization | math | dense_anchor → dense_statistics | ols | 1.26293 | [0.15387, 2.52301] | 2.36914 | 9 |
| leave_one_size_out | grouped_quantization | math | dense_anchor → dense_statistics | ridge | 0.42299 | [0.01490, 0.89238] | 0.87748 | 9 |
| leave_one_size_out | grouped_quantization | code | K0 → dense_anchor | ols | 0.53440 | [-0.13953, 1.30908] | 1.44861 | 9 |
| leave_one_size_out | grouped_quantization | code | K0 → dense_anchor | ridge | 0.80742 | [0.27827, 1.38370] | 1.10543 | 9 |
| leave_one_size_out | grouped_quantization | code | dense_anchor → dense_statistics | ols | 1.79086 | [0.24893, 3.43127] | 3.18235 | 9 |
| leave_one_size_out | grouped_quantization | code | dense_anchor → dense_statistics | ridge | 0.05575 | [-0.33537, 0.45459] | 0.78997 | 9 |
| leave_one_size_out | grouped_quantization | qa | K0 → dense_anchor | ols | -1.31837 | [-2.48480, -0.06096] | 2.42383 | 9 |
| leave_one_size_out | grouped_quantization | qa | K0 → dense_anchor | ridge | -0.30431 | [-0.61008, -0.03807] | 0.57202 | 9 |
| leave_one_size_out | grouped_quantization | qa | dense_anchor → dense_statistics | ols | 2.99791 | [0.92718, 5.27997] | 4.35279 | 9 |
| leave_one_size_out | grouped_quantization | qa | dense_anchor → dense_statistics | ridge | 0.02284 | [-0.02228, 0.06322] | 0.08549 | 9 |
| leave_one_source_state_out | per_channel_quantization | math | K0 → dense_anchor | ols | 0.16330 | [-0.02961, 0.39912] | 0.42873 | 9 |
| leave_one_source_state_out | per_channel_quantization | math | K0 → dense_anchor | ridge | 0.04434 | [-0.06563, 0.15645] | 0.22208 | 9 |
| leave_one_source_state_out | per_channel_quantization | math | dense_anchor → dense_statistics | ols | -0.01889 | [-0.11625, 0.05688] | 0.17313 | 9 |
| leave_one_source_state_out | per_channel_quantization | math | dense_anchor → dense_statistics | ridge | 0.08726 | [0.03153, 0.14338] | 0.11185 | 9 |
| leave_one_source_state_out | per_channel_quantization | code | K0 → dense_anchor | ols | 0.11489 | [-0.13504, 0.42491] | 0.55996 | 9 |
| leave_one_source_state_out | per_channel_quantization | code | K0 → dense_anchor | ridge | 0.04476 | [-0.10065, 0.17864] | 0.27929 | 9 |
| leave_one_source_state_out | per_channel_quantization | code | dense_anchor → dense_statistics | ols | -0.03841 | [-0.14117, 0.05976] | 0.20093 | 9 |
| leave_one_source_state_out | per_channel_quantization | code | dense_anchor → dense_statistics | ridge | 0.08110 | [-0.00865, 0.17812] | 0.18678 | 9 |
| leave_one_source_state_out | per_channel_quantization | qa | K0 → dense_anchor | ols | -0.24686 | [-0.69402, 0.13745] | 0.83147 | 9 |
| leave_one_source_state_out | per_channel_quantization | qa | K0 → dense_anchor | ridge | -0.03083 | [-0.16601, 0.07960] | 0.24561 | 9 |
| leave_one_source_state_out | per_channel_quantization | qa | dense_anchor → dense_statistics | ols | -0.53423 | [-1.42322, 0.20249] | 1.62571 | 9 |
| leave_one_source_state_out | per_channel_quantization | qa | dense_anchor → dense_statistics | ridge | 0.17537 | [0.01184, 0.34076] | 0.32892 | 9 |
| leave_one_size_out | per_channel_quantization | math | K0 → dense_anchor | ols | 0.90787 | [0.15588, 1.78023] | 1.62435 | 9 |
| leave_one_size_out | per_channel_quantization | math | K0 → dense_anchor | ridge | 0.52029 | [0.10122, 1.00948] | 0.90825 | 9 |
| leave_one_size_out | per_channel_quantization | math | dense_anchor → dense_statistics | ols | 0.86303 | [0.14786, 1.66071] | 1.51286 | 9 |
| leave_one_size_out | per_channel_quantization | math | dense_anchor → dense_statistics | ridge | 0.04251 | [-0.20111, 0.30167] | 0.50279 | 9 |
| leave_one_size_out | per_channel_quantization | code | K0 → dense_anchor | ols | 1.24435 | [-0.27453, 2.84234] | 3.11688 | 9 |
| leave_one_size_out | per_channel_quantization | code | K0 → dense_anchor | ridge | 0.56299 | [-0.08828, 1.24151] | 1.32979 | 9 |
| leave_one_size_out | per_channel_quantization | code | dense_anchor → dense_statistics | ols | -0.18059 | [-0.93384, 0.51425] | 1.44809 | 9 |
| leave_one_size_out | per_channel_quantization | code | dense_anchor → dense_statistics | ridge | 0.00006 | [-0.30187, 0.31384] | 0.61571 | 9 |
| leave_one_size_out | per_channel_quantization | qa | K0 → dense_anchor | ols | -0.97713 | [-1.86334, -0.08176] | 1.78159 | 9 |
| leave_one_size_out | per_channel_quantization | qa | K0 → dense_anchor | ridge | 0.75640 | [-0.02157, 1.70039] | 1.72196 | 9 |
| leave_one_size_out | per_channel_quantization | qa | dense_anchor → dense_statistics | ols | 2.96143 | [0.28076, 5.97092] | 5.69016 | 9 |
| leave_one_size_out | per_channel_quantization | qa | dense_anchor → dense_statistics | ridge | -0.31308 | [-0.57330, -0.07332] | 0.49998 | 9 |

What changing the form buys at a fixed budget

The source-free median already uses a flexible configuration curve; a source-dependent linear form is a different form, not necessarily a richer configuration curve. Ridge and OLS have the same linear functional class; their contrast is regularization. Every within-budget pair and two exact additive decompositions of each adjacent-budget contrast are stored in JSON. For median(lower budget) → linear(upper budget): total gain = same-form input gain + fixed-budget form gain. Following the median path makes input gain zero, whereas following the linear path isolates the actual input increment. A gain after changing both budget and form cannot be attributed entirely to inputs.

| Arm | Capability | Budget | Constant → median | Median → OLS | Median → ridge | OLS → ridge (regularization) |
|---|---|---|---|---|---|---|
| pruning | math | K0 | 0.29164 [0.22100, 0.35812] | -0.28393 [-0.44711, -0.10203] | -0.11291 [-0.21553, 0.00644] | 0.17103 [0.05628, 0.29867] |
| pruning | math | dense_anchor | 0.29164 [0.22100, 0.35812] | -0.10730 [-0.23305, 0.05140] | -0.12941 [-0.21913, 0.00491] | -0.02211 [-0.08800, 0.03365] |
| pruning | math | dense_statistics | 0.29164 [0.22100, 0.35812] | -0.08124 [-0.19580, 0.06837] | -0.04584 [-0.13785, 0.07886] | 0.03540 [-0.04173, 0.11403] |
| pruning | code | K0 | 0.35598 [0.23839, 0.45621] | -0.34334 [-0.59153, -0.07079] | -0.15638 [-0.32783, 0.03138] | 0.18696 [0.04203, 0.36029] |
| pruning | code | dense_anchor | 0.35598 [0.23839, 0.45621] | -0.13295 [-0.35332, 0.13596] | -0.11112 [-0.28997, 0.11182] | 0.02183 [-0.08343, 0.12672] |
| pruning | code | dense_statistics | 0.35598 [0.23839, 0.45621] | -0.14476 [-0.34531, 0.09456] | -0.05790 [-0.22446, 0.15828] | 0.08686 [-0.02200, 0.19086] |
| pruning | qa | K0 | 0.17203 [0.03887, 0.29581] | -0.33844 [-0.60482, -0.06583] | -0.16938 [-0.30856, -0.03092] | 0.16906 [-0.01383, 0.38083] |
| pruning | qa | dense_anchor | 0.17203 [0.03887, 0.29581] | -0.47769 [-0.82420, -0.15039] | -0.15270 [-0.28130, -0.01930] | 0.32499 [0.07709, 0.62453] |
| pruning | qa | dense_statistics | 0.17203 [0.03887, 0.29581] | -0.34991 [-0.59173, -0.11015] | -0.04372 [-0.14852, 0.07014] | 0.30619 [0.10029, 0.52974] |
| grouped_quantization | math | K0 | 0.58937 [0.38720, 0.73904] | -1.02763 [-1.49952, -0.51025] | -0.44516 [-0.68810, -0.19551] | 0.58247 [0.22033, 0.99355] |
| grouped_quantization | math | dense_anchor | 0.58937 [0.38720, 0.73904] | -0.49763 [-0.79058, -0.11303] | -0.34872 [-0.56981, -0.12683] | 0.14891 [-0.10410, 0.37974] |
| grouped_quantization | math | dense_statistics | 0.58937 [0.38720, 0.73904] | -0.42236 [-0.64711, -0.16085] | -0.21160 [-0.42461, 0.02540] | 0.21076 [0.01205, 0.39096] |
| grouped_quantization | code | K0 | 0.62421 [0.42566, 0.76122] | -0.95321 [-1.42006, -0.43882] | -0.43490 [-0.68157, -0.16695] | 0.51831 [0.18045, 0.89987] |
| grouped_quantization | code | dense_anchor | 0.62421 [0.42566, 0.76122] | -0.46463 [-0.79425, -0.05082] | -0.29849 [-0.53603, -0.06397] | 0.16614 [-0.10269, 0.40957] |
| grouped_quantization | code | dense_statistics | 0.62421 [0.42566, 0.76122] | -0.45473 [-0.79471, -0.04860] | -0.23020 [-0.43094, -0.03398] | 0.22452 [-0.04064, 0.45520] |
| grouped_quantization | qa | K0 | 0.60773 [0.36527, 0.81570] | -1.12739 [-1.68646, -0.53303] | -0.54391 [-0.84792, -0.24156] | 0.58348 [0.15201, 1.09344] |
| grouped_quantization | qa | dense_anchor | 0.60773 [0.36527, 0.81570] | -1.41556 [-2.15654, -0.68677] | -0.52355 [-0.76880, -0.25556] | 0.89201 [0.30234, 1.55942] |
| grouped_quantization | qa | dense_statistics | 0.60773 [0.36527, 0.81570] | -0.91705 [-1.54773, -0.28454] | -0.20528 [-0.40062, 0.05194] | 0.71178 [0.22377, 1.26237] |
| per_channel_quantization | math | K0 | 1.11916 [0.85255, 1.38345] | -0.94851 [-1.27375, -0.57668] | -0.68291 [-1.05193, -0.32005] | 0.26560 [0.01642, 0.49458] |
| per_channel_quantization | math | dense_anchor | 1.11916 [0.85255, 1.38345] | -0.78522 [-0.96291, -0.54403] | -0.63857 [-0.94586, -0.33258] | 0.14664 [-0.12832, 0.42205] |
| per_channel_quantization | math | dense_statistics | 1.11916 [0.85255, 1.38345] | -0.80411 [-1.05628, -0.51229] | -0.55131 [-0.84361, -0.25681] | 0.25280 [-0.04153, 0.54194] |
| per_channel_quantization | code | K0 | 1.30477 [0.88027, 1.72384] | -1.04897 [-1.52674, -0.57730] | -0.79337 [-1.31669, -0.26894] | 0.25560 [-0.03014, 0.51777] |
| per_channel_quantization | code | dense_anchor | 1.30477 [0.88027, 1.72384] | -0.93407 [-1.19069, -0.66960] | -0.74860 [-1.19607, -0.29104] | 0.18547 [-0.09437, 0.46591] |
| per_channel_quantization | code | dense_statistics | 1.30477 [0.88027, 1.72384] | -0.97248 [-1.28246, -0.63942] | -0.66750 [-1.08221, -0.24760] | 0.30498 [0.00745, 0.59616] |
| per_channel_quantization | qa | K0 | 0.98554 [0.69599, 1.28986] | -0.92860 [-1.28555, -0.52928] | -0.61042 [-1.01265, -0.21622] | 0.31818 [0.05719, 0.57036] |
| per_channel_quantization | qa | dense_anchor | 0.98554 [0.69599, 1.28986] | -1.17546 [-1.54483, -0.74215] | -0.64125 [-1.00095, -0.26882] | 0.53420 [0.03581, 1.08735] |
| per_channel_quantization | qa | dense_statistics | 0.98554 [0.69599, 1.28986] | -1.70969 [-2.26178, -1.20236] | -0.46589 [-0.77340, -0.15390] | 1.24380 [0.52610, 2.01911] |

Which capability is most fragile at a fixed configuration

Fragility is the largest signed ΔL among math/code/QA, not the largest absolute loss. Top-1 accuracy awards the expected credit under uniform prediction ties (zero gets 1/3); regret is the actual maximum ΔL minus the actual ΔL of the predicted most-fragile capability. JSON includes pairwise ranking accuracy, source-cluster intervals, and each state's actual/predicted capability.

| Split | Arm | Budget/form | Top-1 accuracy | Regret (nats) | Clusters |
|---|---|---|---|---|---|
| leave_one_source_state_out | pruning | K0/zero | 0.33333 | 0.18956 | 9 |
| leave_one_source_state_out | pruning | K0/constant | 0.50000 | 0.01690 | 9 |
| leave_one_source_state_out | pruning | K0/median_curve | 0.33333 | 0.07732 | 9 |
| leave_one_source_state_out | pruning | K0/ols | 0.61111 | 0.01494 | 9 |
| leave_one_source_state_out | pruning | K0/ridge | 0.52778 | 0.01922 | 9 |
| leave_one_source_state_out | pruning | K0/delivered | 0.33333 | 0.07732 | 9 |
| leave_one_source_state_out | pruning | dense_anchor/ols | 0.47222 | 0.13335 | 9 |
| leave_one_source_state_out | pruning | dense_anchor/ridge | 0.58333 | 0.02566 | 9 |
| leave_one_source_state_out | pruning | dense_statistics/ols | 0.61111 | 0.04779 | 9 |
| leave_one_source_state_out | pruning | dense_statistics/ridge | 0.41667 | 0.08427 | 9 |
| leave_one_size_out | pruning | K0/zero | 0.33333 | 0.18956 | 9 |
| leave_one_size_out | pruning | K0/constant | 0.50000 | 0.01690 | 9 |
| leave_one_size_out | pruning | K0/median_curve | 0.16667 | 0.09566 | 9 |
| leave_one_size_out | pruning | K0/ols | 0.72222 | 0.00630 | 9 |
| leave_one_size_out | pruning | K0/ridge | 0.44444 | 0.23155 | 9 |
| leave_one_size_out | pruning | K0/delivered | 0.16667 | 0.09566 | 9 |
| leave_one_size_out | pruning | dense_anchor/ols | 0.27778 | 0.13890 | 9 |
| leave_one_size_out | pruning | dense_anchor/ridge | 0.36111 | 0.10674 | 9 |
| leave_one_size_out | pruning | dense_statistics/ols | 0.44444 | 0.08774 | 9 |
| leave_one_size_out | pruning | dense_statistics/ridge | 0.52778 | 0.10121 | 9 |
| leave_one_source_state_out | grouped_quantization | K0/zero | 0.33333 | 0.15427 | 9 |
| leave_one_source_state_out | grouped_quantization | K0/constant | 0.70370 | 0.05135 | 9 |
| leave_one_source_state_out | grouped_quantization | K0/median_curve | 0.64198 | 0.05236 | 9 |
| leave_one_source_state_out | grouped_quantization | K0/ols | 0.60494 | 0.05899 | 9 |
| leave_one_source_state_out | grouped_quantization | K0/ridge | 0.55556 | 0.05951 | 9 |
| leave_one_source_state_out | grouped_quantization | K0/delivered | 0.64198 | 0.05236 | 9 |
| leave_one_source_state_out | grouped_quantization | dense_anchor/ols | 0.33333 | 0.20985 | 9 |
| leave_one_source_state_out | grouped_quantization | dense_anchor/ridge | 0.40741 | 0.12267 | 9 |
| leave_one_source_state_out | grouped_quantization | dense_statistics/ols | 0.30864 | 0.20733 | 9 |
| leave_one_source_state_out | grouped_quantization | dense_statistics/ridge | 0.49383 | 0.14288 | 9 |
| leave_one_size_out | grouped_quantization | K0/zero | 0.33333 | 0.15427 | 9 |
| leave_one_size_out | grouped_quantization | K0/constant | 0.70370 | 0.05135 | 9 |
| leave_one_size_out | grouped_quantization | K0/median_curve | 0.64198 | 0.05301 | 9 |
| leave_one_size_out | grouped_quantization | K0/ols | 0.58025 | 0.05977 | 9 |
| leave_one_size_out | grouped_quantization | K0/ridge | 0.46914 | 0.15862 | 9 |
| leave_one_size_out | grouped_quantization | K0/delivered | 0.64198 | 0.05301 | 9 |
| leave_one_size_out | grouped_quantization | dense_anchor/ols | 0.44444 | 0.14109 | 9 |
| leave_one_size_out | grouped_quantization | dense_anchor/ridge | 0.34568 | 0.18629 | 9 |
| leave_one_size_out | grouped_quantization | dense_statistics/ols | 0.06173 | 0.29445 | 9 |
| leave_one_size_out | grouped_quantization | dense_statistics/ridge | 0.29630 | 0.21673 | 9 |
| leave_one_source_state_out | per_channel_quantization | K0/zero | 0.33333 | 0.21680 | 9 |
| leave_one_source_state_out | per_channel_quantization | K0/constant | 0.44444 | 0.01364 | 9 |
| leave_one_source_state_out | per_channel_quantization | K0/median_curve | 0.41667 | 0.01855 | 9 |
| leave_one_source_state_out | per_channel_quantization | K0/ols | 0.58333 | 0.01278 | 9 |
| leave_one_source_state_out | per_channel_quantization | K0/ridge | 0.47222 | 0.01358 | 9 |
| leave_one_source_state_out | per_channel_quantization | K0/delivered | 0.41667 | 0.01855 | 9 |
| leave_one_source_state_out | per_channel_quantization | dense_anchor/ols | 0.47222 | 0.11546 | 9 |
| leave_one_source_state_out | per_channel_quantization | dense_anchor/ridge | 0.36111 | 0.20672 | 9 |
| leave_one_source_state_out | per_channel_quantization | dense_statistics/ols | 0.27778 | 0.20240 | 9 |
| leave_one_source_state_out | per_channel_quantization | dense_statistics/ridge | 0.41667 | 0.13762 | 9 |
| leave_one_size_out | per_channel_quantization | K0/zero | 0.33333 | 0.21680 | 9 |
| leave_one_size_out | per_channel_quantization | K0/constant | 0.44444 | 0.01364 | 9 |
| leave_one_size_out | per_channel_quantization | K0/median_curve | 0.38889 | 0.04103 | 9 |
| leave_one_size_out | per_channel_quantization | K0/ols | 0.52778 | 0.09358 | 9 |
| leave_one_size_out | per_channel_quantization | K0/ridge | 0.41667 | 0.09365 | 9 |
| leave_one_size_out | per_channel_quantization | K0/delivered | 0.38889 | 0.04103 | 9 |
| leave_one_size_out | per_channel_quantization | dense_anchor/ols | 0.44444 | 0.18410 | 9 |
| leave_one_size_out | per_channel_quantization | dense_anchor/ridge | 0.22222 | 0.36847 | 9 |
| leave_one_size_out | per_channel_quantization | dense_statistics/ols | 0.30556 | 0.31617 | 9 |
| leave_one_size_out | per_channel_quantization | dense_statistics/ridge | 0.22222 | 0.29986 | 9 |

Unseen strength: separate from the primary result

Each common density/bit-width is held out globally in turn. The same-sources diagnostic retains other strengths of the scored source; the new-source analysis removes that source entirely as well. Endpoint holdouts extrapolate; interior holdouts interpolate. For example grouped b4 trains on b3/b5 at all three group sizes. Extra pruning d=.65 (interior) and d=.55 (exterior) train only on the common .6/.7/.8/.9 grid and exclude the scored source. They have only two source clusters. Median baselines use adjacent linear interpolation/extension (density/raw bits; log2(qmax) for grouped bits). The delivered new-source rule is scored only where defined: pruning [.6,.9], grouped no bit extrapolation, channel seen bits. In the same-sources diagnostic 'delivered' remains this source-free new-source reference, not the paper's seen-state power/interpolation branch. Delivered coverage is explicit in JSON; its partial-coverage MAE must not be compared with full-coverage MAEs.

| Split | Arm | Capability | Held strength | Form | K0 MAE | Anchor MAE | Statistics MAE | Clusters |
|---|---|---|---|---|---|---|---|---|
| unseen_strength_same_sources | pruning | math | 0.6 | median_curve | 1.14654 | 1.14654 | 1.14654 | 9 |
| unseen_strength_same_sources | pruning | math | 0.7 | median_curve | 0.45475 | 0.45475 | 0.45475 | 9 |
| unseen_strength_same_sources | pruning | math | 0.8 | median_curve | 0.13586 | 0.13586 | 0.13586 | 9 |
| unseen_strength_same_sources | pruning | math | 0.9 | median_curve | 0.13788 | 0.13788 | 0.13788 | 9 |
| unseen_strength_same_sources | pruning | math | 0.6 | ols | 0.89341 | 0.90719 | 0.91340 | 9 |
| unseen_strength_same_sources | pruning | math | 0.7 | ols | 0.57886 | 0.42680 | 0.42680 | 9 |
| unseen_strength_same_sources | pruning | math | 0.8 | ols | 0.43075 | 0.40102 | 0.32978 | 9 |
| unseen_strength_same_sources | pruning | math | 0.9 | ols | 0.78272 | 1.04752 | 1.06709 | 9 |
| unseen_strength_same_sources | pruning | math | 0.6 | ridge | 1.09407 | 1.09223 | 0.93393 | 9 |
| unseen_strength_same_sources | pruning | math | 0.7 | ridge | 0.52450 | 0.39411 | 0.32009 | 9 |
| unseen_strength_same_sources | pruning | math | 0.8 | ridge | 0.39304 | 0.40100 | 0.32755 | 9 |
| unseen_strength_same_sources | pruning | math | 0.9 | ridge | 0.24998 | 1.02734 | 0.94017 | 9 |
| unseen_strength_same_sources | pruning | code | 0.6 | median_curve | 1.52992 | 1.52992 | 1.52992 | 9 |
| unseen_strength_same_sources | pruning | code | 0.7 | median_curve | 0.61240 | 0.61240 | 0.61240 | 9 |
| unseen_strength_same_sources | pruning | code | 0.8 | median_curve | 0.17060 | 0.17060 | 0.17060 | 9 |
| unseen_strength_same_sources | pruning | code | 0.9 | median_curve | 0.08439 | 0.08439 | 0.08439 | 9 |
| unseen_strength_same_sources | pruning | code | 0.6 | ols | 1.01138 | 1.03052 | 1.06040 | 9 |
| unseen_strength_same_sources | pruning | code | 0.7 | ols | 0.76267 | 0.51193 | 0.48932 | 9 |
| unseen_strength_same_sources | pruning | code | 0.8 | ols | 0.53090 | 0.50943 | 0.38917 | 9 |
| unseen_strength_same_sources | pruning | code | 0.9 | ols | 0.92814 | 1.20397 | 1.23715 | 9 |
| unseen_strength_same_sources | pruning | code | 0.6 | ridge | 1.26442 | 1.25721 | 1.05882 | 9 |
| unseen_strength_same_sources | pruning | code | 0.7 | ridge | 0.70638 | 0.57864 | 0.49443 | 9 |
| unseen_strength_same_sources | pruning | code | 0.8 | ridge | 0.45809 | 0.50559 | 0.41436 | 9 |
| unseen_strength_same_sources | pruning | code | 0.9 | ridge | 0.31058 | 1.18343 | 1.21445 | 9 |
| unseen_strength_same_sources | pruning | qa | 0.6 | median_curve | 1.21431 | 1.21431 | 1.21431 | 9 |
| unseen_strength_same_sources | pruning | qa | 0.7 | median_curve | 0.48690 | 0.48690 | 0.48690 | 9 |
| unseen_strength_same_sources | pruning | qa | 0.8 | median_curve | 0.15975 | 0.15975 | 0.15975 | 9 |
| unseen_strength_same_sources | pruning | qa | 0.9 | median_curve | 0.07850 | 0.07850 | 0.07850 | 9 |
| unseen_strength_same_sources | pruning | qa | 0.6 | ols | 1.08335 | 1.15642 | 1.06613 | 9 |
| unseen_strength_same_sources | pruning | qa | 0.7 | ols | 0.70475 | 0.66505 | 0.35397 | 9 |
| unseen_strength_same_sources | pruning | qa | 0.8 | ols | 0.51556 | 0.54439 | 0.49787 | 9 |
| unseen_strength_same_sources | pruning | qa | 0.9 | ols | 0.69229 | 0.70556 | 0.93493 | 9 |
| unseen_strength_same_sources | pruning | qa | 0.6 | ridge | 1.23133 | 1.23682 | 1.08218 | 9 |
| unseen_strength_same_sources | pruning | qa | 0.7 | ridge | 0.64977 | 0.65728 | 0.41076 | 9 |
| unseen_strength_same_sources | pruning | qa | 0.8 | ridge | 0.34182 | 0.38321 | 0.37740 | 9 |
| unseen_strength_same_sources | pruning | qa | 0.9 | ridge | 0.29271 | 0.26714 | 0.51905 | 9 |
| unseen_strength_new_source | pruning | math | 0.6 | median_curve | 1.16976 | 1.16976 | 1.16976 | 9 |
| unseen_strength_new_source | pruning | math | 0.7 | median_curve | 0.47290 | 0.47290 | 0.47290 | 9 |
| unseen_strength_new_source | pruning | math | 0.8 | median_curve | 0.13940 | 0.13940 | 0.13940 | 9 |
| unseen_strength_new_source | pruning | math | 0.9 | median_curve | 0.14294 | 0.14294 | 0.14294 | 9 |
| unseen_strength_new_source | pruning | math | 0.6 | ols | 0.93680 | 0.98063 | 0.90877 | 9 |
| unseen_strength_new_source | pruning | math | 0.7 | ols | 0.78800 | 0.52961 | 0.39631 | 9 |
| unseen_strength_new_source | pruning | math | 0.8 | ols | 0.54610 | 0.31445 | 0.30741 | 9 |
| unseen_strength_new_source | pruning | math | 0.9 | ols | 0.86428 | 0.97576 | 1.16777 | 9 |
| unseen_strength_new_source | pruning | math | 0.6 | ridge | 1.09134 | 1.04384 | 1.03327 | 9 |
| unseen_strength_new_source | pruning | math | 0.7 | ridge | 0.61013 | 0.50828 | 0.36858 | 9 |
| unseen_strength_new_source | pruning | math | 0.8 | ridge | 0.43449 | 0.25933 | 0.24261 | 9 |
| unseen_strength_new_source | pruning | math | 0.9 | ridge | 0.28519 | 0.91266 | 0.94987 | 9 |
| unseen_strength_new_source | pruning | code | 0.6 | median_curve | 1.49108 | 1.49108 | 1.49108 | 9 |
| unseen_strength_new_source | pruning | code | 0.7 | median_curve | 0.63440 | 0.63440 | 0.63440 | 9 |
| unseen_strength_new_source | pruning | code | 0.8 | median_curve | 0.18562 | 0.18562 | 0.18562 | 9 |
| unseen_strength_new_source | pruning | code | 0.9 | median_curve | 0.11004 | 0.11004 | 0.11004 | 9 |
| unseen_strength_new_source | pruning | code | 0.6 | ols | 1.12306 | 1.14204 | 1.07778 | 9 |
| unseen_strength_new_source | pruning | code | 0.7 | ols | 1.01640 | 0.69829 | 0.49526 | 9 |
| unseen_strength_new_source | pruning | code | 0.8 | ols | 0.71610 | 0.44204 | 0.63647 | 9 |
| unseen_strength_new_source | pruning | code | 0.9 | ols | 1.01274 | 1.11189 | 1.45446 | 9 |
| unseen_strength_new_source | pruning | code | 0.6 | ridge | 1.28652 | 1.31707 | 1.31577 | 9 |
| unseen_strength_new_source | pruning | code | 0.7 | ridge | 0.79928 | 0.65020 | 0.60316 | 9 |
| unseen_strength_new_source | pruning | code | 0.8 | ridge | 0.55352 | 0.43005 | 0.27601 | 9 |
| unseen_strength_new_source | pruning | code | 0.9 | ridge | 0.36291 | 0.96250 | 1.12881 | 9 |
| unseen_strength_new_source | pruning | qa | 0.6 | median_curve | 1.22221 | 1.22221 | 1.22221 | 9 |
| unseen_strength_new_source | pruning | qa | 0.7 | median_curve | 0.56940 | 0.56940 | 0.56940 | 9 |
| unseen_strength_new_source | pruning | qa | 0.8 | median_curve | 0.15814 | 0.15814 | 0.15814 | 9 |
| unseen_strength_new_source | pruning | qa | 0.9 | median_curve | 0.08714 | 0.08714 | 0.08714 | 9 |
| unseen_strength_new_source | pruning | qa | 0.6 | ols | 1.20464 | 1.43024 | 1.38407 | 9 |
| unseen_strength_new_source | pruning | qa | 0.7 | ols | 0.98317 | 1.07280 | 0.95033 | 9 |
| unseen_strength_new_source | pruning | qa | 0.8 | ols | 0.75910 | 0.90976 | 0.70019 | 9 |
| unseen_strength_new_source | pruning | qa | 0.9 | ols | 0.78637 | 0.86121 | 0.78808 | 9 |
| unseen_strength_new_source | pruning | qa | 0.6 | ridge | 1.27028 | 1.28745 | 1.26933 | 9 |
| unseen_strength_new_source | pruning | qa | 0.7 | ridge | 0.72647 | 0.75084 | 0.61853 | 9 |
| unseen_strength_new_source | pruning | qa | 0.8 | ridge | 0.46839 | 0.53354 | 0.29797 | 9 |
| unseen_strength_new_source | pruning | qa | 0.9 | ridge | 0.33004 | 0.36120 | 0.47232 | 9 |
| extra_pruning_strength_new_source | pruning | math | 0.55 | median_curve | 3.96013 | 3.96013 | 3.96013 | 2 |
| extra_pruning_strength_new_source | pruning | math | 0.65 | median_curve | 1.79903 | 1.79903 | 1.79903 | 2 |
| extra_pruning_strength_new_source | pruning | math | 0.55 | ols | 4.15098 | 3.83593 | 3.40247 | 2 |
| extra_pruning_strength_new_source | pruning | math | 0.65 | ols | 1.58301 | 1.28793 | 0.88457 | 2 |
| extra_pruning_strength_new_source | pruning | math | 0.55 | ridge | 4.30916 | 4.12900 | 3.86219 | 2 |
| extra_pruning_strength_new_source | pruning | math | 0.65 | ridge | 1.62678 | 1.48470 | 1.07434 | 2 |
| extra_pruning_strength_new_source | pruning | code | 0.55 | median_curve | 4.04077 | 4.04077 | 4.04077 | 2 |
| extra_pruning_strength_new_source | pruning | code | 0.65 | median_curve | 2.68534 | 2.68534 | 2.68534 | 2 |
| extra_pruning_strength_new_source | pruning | code | 0.55 | ols | 4.13783 | 3.73940 | 2.93027 | 2 |
| extra_pruning_strength_new_source | pruning | code | 0.65 | ols | 2.34626 | 2.02910 | 0.92497 | 2 |
| extra_pruning_strength_new_source | pruning | code | 0.55 | ridge | 4.29991 | 4.16300 | 4.02196 | 2 |
| extra_pruning_strength_new_source | pruning | code | 0.65 | ridge | 2.43256 | 2.15234 | 2.08823 | 2 |
| extra_pruning_strength_new_source | pruning | qa | 0.55 | median_curve | 3.86681 | 3.86681 | 3.86681 | 2 |
| extra_pruning_strength_new_source | pruning | qa | 0.65 | median_curve | 2.46751 | 2.46751 | 2.46751 | 2 |
| extra_pruning_strength_new_source | pruning | qa | 0.55 | ols | 3.50712 | 3.71861 | 3.95010 | 2 |
| extra_pruning_strength_new_source | pruning | qa | 0.65 | ols | 2.17761 | 2.44666 | 2.32708 | 2 |
| extra_pruning_strength_new_source | pruning | qa | 0.55 | ridge | 3.68686 | 3.70015 | 3.77519 | 2 |
| extra_pruning_strength_new_source | pruning | qa | 0.65 | ridge | 2.53390 | 2.54918 | 2.16331 | 2 |
| unseen_strength_same_sources | grouped_quantization | math | 3.0 | median_curve | 2.43692 | 2.43692 | 2.43692 | 9 |
| unseen_strength_same_sources | grouped_quantization | math | 4.0 | median_curve | 0.42462 | 0.42462 | 0.42462 | 9 |
| unseen_strength_same_sources | grouped_quantization | math | 5.0 | median_curve | 0.57102 | 0.57102 | 0.57102 | 9 |
| unseen_strength_same_sources | grouped_quantization | math | 3.0 | ols | 2.00988 | 2.02330 | 2.03020 | 9 |
| unseen_strength_same_sources | grouped_quantization | math | 4.0 | ols | 1.34966 | 1.21342 | 0.99726 | 9 |
| unseen_strength_same_sources | grouped_quantization | math | 5.0 | ols | 2.17124 | 3.04181 | 3.12884 | 9 |
| unseen_strength_same_sources | grouped_quantization | math | 3.0 | ridge | 2.18405 | 2.18405 | 2.03029 | 9 |
| unseen_strength_same_sources | grouped_quantization | math | 4.0 | ridge | 1.00936 | 0.99862 | 0.99612 | 9 |
| unseen_strength_same_sources | grouped_quantization | math | 5.0 | ridge | 0.68571 | 2.98660 | 1.30680 | 9 |
| unseen_strength_same_sources | grouped_quantization | code | 3.0 | median_curve | 2.68322 | 2.68322 | 2.68322 | 9 |
| unseen_strength_same_sources | grouped_quantization | code | 4.0 | median_curve | 0.59838 | 0.59838 | 0.59838 | 9 |
| unseen_strength_same_sources | grouped_quantization | code | 5.0 | median_curve | 0.86590 | 0.86590 | 0.86590 | 9 |
| unseen_strength_same_sources | grouped_quantization | code | 3.0 | ols | 2.14962 | 2.17586 | 2.19748 | 9 |
| unseen_strength_same_sources | grouped_quantization | code | 4.0 | ols | 1.40897 | 1.24922 | 1.07603 | 9 |
| unseen_strength_same_sources | grouped_quantization | code | 5.0 | ols | 2.29702 | 3.14177 | 3.24048 | 9 |
| unseen_strength_same_sources | grouped_quantization | code | 3.0 | ridge | 2.55992 | 2.36652 | 2.19458 | 9 |
| unseen_strength_same_sources | grouped_quantization | code | 4.0 | ridge | 1.09542 | 1.06562 | 1.06562 | 9 |
| unseen_strength_same_sources | grouped_quantization | code | 5.0 | ridge | 0.71859 | 0.99738 | 3.22983 | 9 |
| unseen_strength_same_sources | grouped_quantization | qa | 3.0 | median_curve | 2.44587 | 2.44587 | 2.44587 | 9 |
| unseen_strength_same_sources | grouped_quantization | qa | 4.0 | median_curve | 0.41475 | 0.41475 | 0.41475 | 9 |
| unseen_strength_same_sources | grouped_quantization | qa | 5.0 | median_curve | 0.30849 | 0.30849 | 0.30849 | 9 |
| unseen_strength_same_sources | grouped_quantization | qa | 3.0 | ols | 2.24230 | 2.31292 | 2.24761 | 9 |
| unseen_strength_same_sources | grouped_quantization | qa | 4.0 | ols | 1.43488 | 1.62057 | 1.12802 | 9 |
| unseen_strength_same_sources | grouped_quantization | qa | 5.0 | ols | 2.19086 | 2.34017 | 3.15653 | 9 |
| unseen_strength_same_sources | grouped_quantization | qa | 3.0 | ridge | 2.39325 | 2.39344 | 2.24736 | 9 |
| unseen_strength_same_sources | grouped_quantization | qa | 4.0 | ridge | 0.99506 | 1.12635 | 1.02831 | 9 |
| unseen_strength_same_sources | grouped_quantization | qa | 5.0 | ridge | 0.74032 | 1.00116 | 1.40731 | 9 |
| unseen_strength_new_source | grouped_quantization | math | 3.0 | median_curve | 2.43131 | 2.43131 | 2.43131 | 9 |
| unseen_strength_new_source | grouped_quantization | math | 4.0 | median_curve | 0.46593 | 0.46593 | 0.46593 | 9 |
| unseen_strength_new_source | grouped_quantization | math | 5.0 | median_curve | 0.63959 | 0.63959 | 0.63959 | 9 |
| unseen_strength_new_source | grouped_quantization | math | 3.0 | ols | 2.06357 | 2.15586 | 2.07510 | 9 |
| unseen_strength_new_source | grouped_quantization | math | 4.0 | ols | 1.79800 | 0.85920 | 0.76455 | 9 |
| unseen_strength_new_source | grouped_quantization | math | 5.0 | ols | 2.48352 | 2.69787 | 3.35600 | 9 |
| unseen_strength_new_source | grouped_quantization | math | 3.0 | ridge | 2.29195 | 2.17304 | 2.22459 | 9 |
| unseen_strength_new_source | grouped_quantization | math | 4.0 | ridge | 1.15163 | 1.01597 | 0.70123 | 9 |
| unseen_strength_new_source | grouped_quantization | math | 5.0 | ridge | 0.83661 | 1.71283 | 0.96927 | 9 |
| unseen_strength_new_source | grouped_quantization | code | 3.0 | median_curve | 2.66551 | 2.66551 | 2.66551 | 9 |
| unseen_strength_new_source | grouped_quantization | code | 4.0 | median_curve | 0.63208 | 0.63208 | 0.63208 | 9 |
| unseen_strength_new_source | grouped_quantization | code | 5.0 | median_curve | 0.89741 | 0.89741 | 0.89741 | 9 |
| unseen_strength_new_source | grouped_quantization | code | 3.0 | ols | 2.20880 | 2.32476 | 2.42160 | 9 |
| unseen_strength_new_source | grouped_quantization | code | 4.0 | ols | 1.87687 | 0.94805 | 1.00548 | 9 |
| unseen_strength_new_source | grouped_quantization | code | 5.0 | ols | 2.63036 | 2.81445 | 3.32597 | 9 |
| unseen_strength_new_source | grouped_quantization | code | 3.0 | ridge | 2.51315 | 2.42565 | 2.47389 | 9 |
| unseen_strength_new_source | grouped_quantization | code | 4.0 | ridge | 1.39050 | 1.07871 | 1.08384 | 9 |
| unseen_strength_new_source | grouped_quantization | code | 5.0 | ridge | 0.89645 | 0.82369 | 2.71650 | 9 |
| unseen_strength_new_source | grouped_quantization | qa | 3.0 | median_curve | 2.45164 | 2.45164 | 2.45164 | 9 |
| unseen_strength_new_source | grouped_quantization | qa | 4.0 | median_curve | 0.44468 | 0.44468 | 0.44468 | 9 |
| unseen_strength_new_source | grouped_quantization | qa | 5.0 | median_curve | 0.31780 | 0.31780 | 0.31780 | 9 |
| unseen_strength_new_source | grouped_quantization | qa | 3.0 | ols | 2.35996 | 2.58492 | 2.58844 | 9 |
| unseen_strength_new_source | grouped_quantization | qa | 4.0 | ols | 2.04786 | 2.51186 | 1.67844 | 9 |
| unseen_strength_new_source | grouped_quantization | qa | 5.0 | ols | 2.48108 | 2.58846 | 2.52296 | 9 |
| unseen_strength_new_source | grouped_quantization | qa | 3.0 | ridge | 2.40895 | 2.42951 | 2.46346 | 9 |
| unseen_strength_new_source | grouped_quantization | qa | 4.0 | ridge | 1.41416 | 1.35609 | 0.84160 | 9 |
| unseen_strength_new_source | grouped_quantization | qa | 5.0 | ridge | 1.03287 | 1.02192 | 1.56128 | 9 |
| unseen_strength_same_sources | per_channel_quantization | math | 3.0 | median_curve | 5.60792 | 5.60792 | 5.60792 | 9 |
| unseen_strength_same_sources | per_channel_quantization | math | 4.0 | median_curve | 2.14503 | 2.14503 | 2.14503 | 9 |
| unseen_strength_same_sources | per_channel_quantization | math | 6.0 | median_curve | 0.06545 | 0.06545 | 0.06545 | 9 |
| unseen_strength_same_sources | per_channel_quantization | math | 8.0 | median_curve | 0.12066 | 0.12066 | 0.12066 | 9 |
| unseen_strength_same_sources | per_channel_quantization | math | 3.0 | ols | 5.26214 | 5.26214 | 5.26214 | 9 |
| unseen_strength_same_sources | per_channel_quantization | math | 4.0 | ols | 3.48675 | 3.48675 | 3.48675 | 9 |
| unseen_strength_same_sources | per_channel_quantization | math | 6.0 | ols | 1.69710 | 1.55444 | 1.47466 | 9 |
| unseen_strength_same_sources | per_channel_quantization | math | 8.0 | ols | 4.06718 | 4.54911 | 4.61158 | 9 |
| unseen_strength_same_sources | per_channel_quantization | math | 3.0 | ridge | 5.44428 | 5.44428 | 5.44428 | 9 |
| unseen_strength_same_sources | per_channel_quantization | math | 4.0 | ridge | 2.47040 | 2.47040 | 2.47040 | 9 |
| unseen_strength_same_sources | per_channel_quantization | math | 6.0 | ridge | 1.63934 | 1.63934 | 1.63934 | 9 |
| unseen_strength_same_sources | per_channel_quantization | math | 8.0 | ridge | 1.15059 | 1.37197 | 1.74880 | 9 |
| unseen_strength_same_sources | per_channel_quantization | code | 3.0 | median_curve | 6.48674 | 6.48674 | 6.48674 | 9 |
| unseen_strength_same_sources | per_channel_quantization | code | 4.0 | median_curve | 3.32192 | 3.32192 | 3.32192 | 9 |
| unseen_strength_same_sources | per_channel_quantization | code | 6.0 | median_curve | 0.09038 | 0.09038 | 0.09038 | 9 |
| unseen_strength_same_sources | per_channel_quantization | code | 8.0 | median_curve | 0.16388 | 0.16388 | 0.16388 | 9 |
| unseen_strength_same_sources | per_channel_quantization | code | 3.0 | ols | 6.08508 | 6.08508 | 6.08508 | 9 |
| unseen_strength_same_sources | per_channel_quantization | code | 4.0 | ols | 4.02663 | 4.02663 | 4.02663 | 9 |
| unseen_strength_same_sources | per_channel_quantization | code | 6.0 | ols | 1.84227 | 1.78208 | 1.72227 | 9 |
| unseen_strength_same_sources | per_channel_quantization | code | 8.0 | ols | 4.71792 | 5.15101 | 5.21848 | 9 |
| unseen_strength_same_sources | per_channel_quantization | code | 3.0 | ridge | 6.30328 | 6.30328 | 6.30328 | 9 |
| unseen_strength_same_sources | per_channel_quantization | code | 4.0 | ridge | 2.84752 | 2.84752 | 2.84752 | 9 |
| unseen_strength_same_sources | per_channel_quantization | code | 6.0 | ridge | 1.90777 | 1.90777 | 1.90777 | 9 |
| unseen_strength_same_sources | per_channel_quantization | code | 8.0 | ridge | 1.27303 | 4.46913 | 1.87060 | 9 |
| unseen_strength_same_sources | per_channel_quantization | qa | 3.0 | median_curve | 5.59782 | 5.59782 | 5.59782 | 9 |
| unseen_strength_same_sources | per_channel_quantization | qa | 4.0 | median_curve | 2.42349 | 2.42349 | 2.42349 | 9 |
| unseen_strength_same_sources | per_channel_quantization | qa | 6.0 | median_curve | 0.07518 | 0.07518 | 0.07518 | 9 |
| unseen_strength_same_sources | per_channel_quantization | qa | 8.0 | median_curve | 0.06881 | 0.06881 | 0.06881 | 9 |
| unseen_strength_same_sources | per_channel_quantization | qa | 3.0 | ols | 5.04058 | 5.04058 | 5.04058 | 9 |
| unseen_strength_same_sources | per_channel_quantization | qa | 4.0 | ols | 3.36271 | 3.36271 | 3.36271 | 9 |
| unseen_strength_same_sources | per_channel_quantization | qa | 6.0 | ols | 1.70052 | 1.98565 | 1.42593 | 9 |
| unseen_strength_same_sources | per_channel_quantization | qa | 8.0 | ols | 3.83414 | 3.95718 | 4.57397 | 9 |
| unseen_strength_same_sources | per_channel_quantization | qa | 3.0 | ridge | 5.32182 | 5.32182 | 5.19526 | 9 |
| unseen_strength_same_sources | per_channel_quantization | qa | 4.0 | ridge | 2.39855 | 2.39855 | 2.39855 | 9 |
| unseen_strength_same_sources | per_channel_quantization | qa | 6.0 | ridge | 1.52357 | 1.62863 | 1.52357 | 9 |
| unseen_strength_same_sources | per_channel_quantization | qa | 8.0 | ridge | 1.14455 | 1.26845 | 1.82671 | 9 |
| unseen_strength_new_source | per_channel_quantization | math | 3.0 | median_curve | 5.57418 | 5.57418 | 5.57418 | 9 |
| unseen_strength_new_source | per_channel_quantization | math | 4.0 | median_curve | 2.13916 | 2.13916 | 2.13916 | 9 |
| unseen_strength_new_source | per_channel_quantization | math | 6.0 | median_curve | 0.07723 | 0.07723 | 0.07723 | 9 |
| unseen_strength_new_source | per_channel_quantization | math | 8.0 | median_curve | 0.14400 | 0.14400 | 0.14400 | 9 |
| unseen_strength_new_source | per_channel_quantization | math | 3.0 | ols | 5.27845 | 5.33662 | 5.25807 | 9 |
| unseen_strength_new_source | per_channel_quantization | math | 4.0 | ols | 3.39265 | 3.07852 | 2.63666 | 9 |
| unseen_strength_new_source | per_channel_quantization | math | 6.0 | ols | 1.88470 | 1.22396 | 0.87803 | 9 |
| unseen_strength_new_source | per_channel_quantization | math | 8.0 | ols | 4.17638 | 4.54188 | 4.90947 | 9 |
| unseen_strength_new_source | per_channel_quantization | math | 3.0 | ridge | 5.48576 | 5.47924 | 5.49891 | 9 |
| unseen_strength_new_source | per_channel_quantization | math | 4.0 | ridge | 2.54234 | 2.40944 | 2.24865 | 9 |
| unseen_strength_new_source | per_channel_quantization | math | 6.0 | ridge | 1.58971 | 1.15780 | 1.30124 | 9 |
| unseen_strength_new_source | per_channel_quantization | math | 8.0 | ridge | 1.25078 | 1.77451 | 1.47463 | 9 |
| unseen_strength_new_source | per_channel_quantization | code | 3.0 | median_curve | 6.44687 | 6.44687 | 6.44687 | 9 |
| unseen_strength_new_source | per_channel_quantization | code | 4.0 | median_curve | 3.02331 | 3.02331 | 3.02331 | 9 |
| unseen_strength_new_source | per_channel_quantization | code | 6.0 | median_curve | 0.10625 | 0.10625 | 0.10625 | 9 |
| unseen_strength_new_source | per_channel_quantization | code | 8.0 | median_curve | 0.18831 | 0.18831 | 0.18831 | 9 |
| unseen_strength_new_source | per_channel_quantization | code | 3.0 | ols | 6.10639 | 6.17551 | 6.15126 | 9 |
| unseen_strength_new_source | per_channel_quantization | code | 4.0 | ols | 3.93810 | 3.72003 | 3.27860 | 9 |
| unseen_strength_new_source | per_channel_quantization | code | 6.0 | ols | 1.99925 | 1.50853 | 1.58463 | 9 |
| unseen_strength_new_source | per_channel_quantization | code | 8.0 | ols | 4.82544 | 5.10463 | 5.51564 | 9 |
| unseen_strength_new_source | per_channel_quantization | code | 3.0 | ridge | 6.35515 | 6.34969 | 6.37667 | 9 |
| unseen_strength_new_source | per_channel_quantization | code | 4.0 | ridge | 2.93643 | 2.80384 | 2.71685 | 9 |
| unseen_strength_new_source | per_channel_quantization | code | 6.0 | ridge | 1.85920 | 1.43280 | 1.40241 | 9 |
| unseen_strength_new_source | per_channel_quantization | code | 8.0 | ridge | 1.38009 | 3.59774 | 2.50857 | 9 |
| unseen_strength_new_source | per_channel_quantization | qa | 3.0 | median_curve | 5.56427 | 5.56427 | 5.56427 | 9 |
| unseen_strength_new_source | per_channel_quantization | qa | 4.0 | median_curve | 2.39041 | 2.39041 | 2.39041 | 9 |
| unseen_strength_new_source | per_channel_quantization | qa | 6.0 | median_curve | 0.06514 | 0.06514 | 0.06514 | 9 |
| unseen_strength_new_source | per_channel_quantization | qa | 8.0 | median_curve | 0.04834 | 0.04834 | 0.04834 | 9 |
| unseen_strength_new_source | per_channel_quantization | qa | 3.0 | ols | 5.06588 | 5.17984 | 5.13551 | 9 |
| unseen_strength_new_source | per_channel_quantization | qa | 4.0 | ols | 3.36557 | 3.34074 | 4.12320 | 9 |
| unseen_strength_new_source | per_channel_quantization | qa | 6.0 | ols | 1.93972 | 2.44404 | 2.54036 | 9 |
| unseen_strength_new_source | per_channel_quantization | qa | 8.0 | ols | 3.97602 | 4.12205 | 4.84027 | 9 |
| unseen_strength_new_source | per_channel_quantization | qa | 3.0 | ridge | 5.31964 | 5.33058 | 5.30008 | 9 |
| unseen_strength_new_source | per_channel_quantization | qa | 4.0 | ridge | 2.69599 | 2.50996 | 2.35907 | 9 |
| unseen_strength_new_source | per_channel_quantization | qa | 6.0 | ridge | 1.47570 | 1.64506 | 0.92875 | 9 |
| unseen_strength_new_source | per_channel_quantization | qa | 8.0 | ridge | 1.24049 | 1.37386 | 1.61254 | 9 |

Reading-rule qualifications

- pruning/math: ols has a positive same-source strength-transfer input gain but does not clear the across-source reading rule; it does not count.
- pruning/code: ols, ridge has a positive same-source strength-transfer input gain but does not clear the across-source reading rule; it does not count.
- pruning/code: switching anchor OLS to statistics ridge improves MAE by 0.07504, but this mixes inputs and regularization and does not count.
- pruning/qa: ols has a positive same-source strength-transfer input gain but does not clear the across-source reading rule; it does not count.
- grouped_quantization/math: ols has a positive same-source strength-transfer input gain but does not clear the across-source reading rule; it does not count.
- grouped_quantization/code: ols has a positive same-source strength-transfer input gain but does not clear the across-source reading rule; it does not count.
- grouped_quantization/code: switching anchor OLS to statistics ridge improves MAE by 0.23442, but this mixes inputs and regularization and does not count.
- per_channel_quantization/math: ols has a positive same-source strength-transfer input gain but does not clear the across-source reading rule; it does not count.
- per_channel_quantization/math: switching anchor OLS to statistics ridge improves MAE by 0.23391, but this mixes inputs and regularization and does not count.
- per_channel_quantization/code: ridge has a positive same-source strength-transfer input gain but does not clear the across-source reading rule; it does not count.
- per_channel_quantization/code: switching anchor OLS to statistics ridge improves MAE by 0.26658, but this mixes inputs and regularization and does not count.
- per_channel_quantization/qa: switching anchor OLS to statistics ridge improves MAE by 0.70957, but this mixes inputs and regularization and does not count.

All requested comparisons use existing scalar responses. No compressed-model diagnostic enters a predictor. The small Pythia panel tests transfer between source states/sizes within one family; it does not establish architecture transfer. Signed improvements in reference CE can be negative and do not by themselves establish downstream task accuracy.
