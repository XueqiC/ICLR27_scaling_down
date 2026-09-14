> Historical V92 snapshot: this report accompanies the adjacent pre-gap-fill JSON (378 primary rows, 6 grouped states, 3/9 qualifying pairs). The [current completed-panel report](../../../results/v92-input-comparison/summary.md) has 459 primary rows, 9 states in every arm, and 4/9 qualifying pairs. The snapshot below is retained for reproduction and legacy-loader compatibility.

V92 VERDICT — development only; lower MAE is better.
Statistics budget vs K0 + dense anchor, SAME form; gain must exceed the FULL 95% interval width.
Positive gains are MAE reductions in nats. OLS and ridge are separate fixed comparisons.

| Arm | Capability | Statistics helped? | OLS gain [95% CI]; width | Ridge gain [95% CI]; width | Source clusters |
|---|---|---|---|---|---|
| pruning | math | YES: ridge | 0.02606 [0.00167, 0.06089]; 0.05922 | 0.08356 [0.04882, 0.12227]; 0.07345 | 9 |
| pruning | code | NO under fixed rule | -0.01182 [-0.07016, 0.04275]; 0.11290 | 0.05321 [0.01983, 0.08508]; 0.06525 | 9 |
| pruning | qa | YES: ridge | 0.12777 [-0.13266, 0.38482]; 0.51748 | 0.10897 [0.06206, 0.15684]; 0.09478 | 9 |
| grouped_quantization | math | YES: ridge | -0.00398 [-0.14263, 0.12737]; 0.27000 | 0.35172 [0.25743, 0.43691]; 0.17949 | 6 |
| grouped_quantization | code | NO under fixed rule | -0.07811 [-0.37594, 0.10564]; 0.48159 | 0.31262 [0.07250, 0.51472]; 0.44223 | 6 |
| grouped_quantization | qa | NO under fixed rule | 2.05567 [0.99378, 3.42335]; 2.42957 | 1.29369 [0.24962, 3.13225]; 2.88263 | 6 |
| per_channel_quantization | math | NO under fixed rule | -0.01889 [-0.11625, 0.05688]; 0.17313 | 0.08726 [0.03153, 0.14338]; 0.11185 | 9 |
| per_channel_quantization | code | NO under fixed rule | -0.03841 [-0.14117, 0.05976]; 0.20093 | 0.08110 [-0.00865, 0.17812]; 0.18678 | 9 |
| per_channel_quantization | qa | NO under fixed rule | -0.53423 [-1.42322, 0.20249]; 1.62571 | 0.17537 [0.01184, 0.34076]; 0.32892 | 9 |

In 9/9 arm/capability pairs, both statistics-augmented linear candidates still have higher MAE than the K0 source-free median curve. Passing the incremental input rule is not a win over the delivered predictor.

Most-fragile capability accuracy (ridge, dense anchor → dense statistics): pruning 58.3% → 41.7%; grouped_quantization 38.9% → 18.5%; per_channel_quantization 36.1% → 41.7%. These are descriptive ranking results, separate from the MAE reading rule.

Development protocol and coverage

Pruning and per-channel RTN: nine source states, three sizes × steps 16k/64k/143k. Grouped RTN: six available states; all three step64k files are absent. No missing responses or descriptors are imputed.
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

MAE and paired improvements weight source states equally, then configurations equally within a state. 95% percentile intervals resample whole source states 20,000 times (seed 9201), pairing predictions on identical rows. Intervals condition on the fitted cross-validation predictions; they do not rerun fitting on bootstrap samples. Leave-one-size-out has three size folds but intervals still cluster on the 9 or 6 source states as requested; these small-cluster intervals are development evidence, not independent confirmation. No candidate is selected by outer-fold MAE. The rule is applied separately to the two fixed linear estimators; YES names the one that clears, without selecting a new predictor or adjusting for multiple comparisons.

In grouped quantization, the statistics-budget OLS design is rank deficient in every primary fold: only five training source states support six source inputs plus an intercept, alongside configuration inputs. OLS uses the minimum-norm least-squares solution; ridge stabilizes the same design. Design ranks, coefficients, fold-local scalers and penalty scores are recorded in JSON. Audit row sets are interned as ordered indices into observations to avoid repeating the same training rows thousands of times.

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
| leave_one_source_state_out | grouped_quantization | math | K0 | 1.32241 | 2.02767 | 1.21618 | 2.89085 | 1.92979 | 1.21618 | 6 |
| leave_one_source_state_out | grouped_quantization | math | dense_anchor | 1.32241 | 2.02767 | 1.21618 | 1.86804 | 1.85839 | 1.21618 | 6 |
| leave_one_source_state_out | grouped_quantization | math | dense_statistics | 1.32241 | 2.02767 | 1.21618 | 1.87202 | 1.50667 | 1.21618 | 6 |
| leave_one_source_state_out | grouped_quantization | code | K0 | 1.42412 | 2.12489 | 1.29599 | 2.89539 | 2.01726 | 1.29599 | 6 |
| leave_one_source_state_out | grouped_quantization | code | dense_anchor | 1.42412 | 2.12489 | 1.29599 | 1.98179 | 1.90329 | 1.29599 | 6 |
| leave_one_source_state_out | grouped_quantization | code | dense_statistics | 1.42412 | 2.12489 | 1.29599 | 2.05989 | 1.59067 | 1.29599 | 6 |
| leave_one_source_state_out | grouped_quantization | qa | K0 | 1.31791 | 2.23769 | 1.31976 | 3.25938 | 2.14307 | 1.31976 | 6 |
| leave_one_source_state_out | grouped_quantization | qa | dense_anchor | 1.31791 | 2.23769 | 1.31976 | 4.06865 | 2.97719 | 1.31976 | 6 |
| leave_one_source_state_out | grouped_quantization | qa | dense_statistics | 1.31791 | 2.23769 | 1.31976 | 2.01298 | 1.68350 | 1.31976 | 6 |
| leave_one_size_out | grouped_quantization | math | K0 | 1.32241 | 2.01111 | 1.21842 | 5.28681 | 2.91696 | 1.21842 | 6 |
| leave_one_size_out | grouped_quantization | math | dense_anchor | 1.32241 | 2.01111 | 1.21842 | 3.44299 | 2.20326 | 1.21842 | 6 |
| leave_one_size_out | grouped_quantization | math | dense_statistics | 1.32241 | 2.01111 | 1.21842 | 2.43377 | 1.50892 | 1.21842 | 6 |
| leave_one_size_out | grouped_quantization | code | K0 | 1.42412 | 2.11654 | 1.28642 | 5.30348 | 3.00528 | 1.28642 | 6 |
| leave_one_size_out | grouped_quantization | code | dense_anchor | 1.42412 | 2.11654 | 1.28642 | 4.49046 | 2.09396 | 1.28642 | 6 |
| leave_one_size_out | grouped_quantization | code | dense_statistics | 1.42412 | 2.11654 | 1.28642 | 1.81170 | 2.03833 | 1.28642 | 6 |
| leave_one_size_out | grouped_quantization | qa | K0 | 1.31791 | 2.23798 | 1.42510 | 5.88932 | 3.25684 | 1.42510 | 6 |
| leave_one_size_out | grouped_quantization | qa | dense_anchor | 1.31791 | 2.23798 | 1.42510 | 7.21084 | 2.23336 | 1.42510 | 6 |
| leave_one_size_out | grouped_quantization | qa | dense_statistics | 1.31791 | 2.23798 | 1.42510 | 2.39065 | 2.14154 | 1.42510 | 6 |
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
| leave_one_source_state_out | grouped_quantization | math | K0 → dense_anchor | ols | 1.02281 | [0.24916, 1.84088] | 1.59172 | 6 |
| leave_one_source_state_out | grouped_quantization | math | K0 → dense_anchor | ridge | 0.07139 | [-0.13590, 0.27947] | 0.41537 | 6 |
| leave_one_source_state_out | grouped_quantization | math | dense_anchor → dense_statistics | ols | -0.00398 | [-0.14263, 0.12737] | 0.27000 | 6 |
| leave_one_source_state_out | grouped_quantization | math | dense_anchor → dense_statistics | ridge | 0.35172 | [0.25743, 0.43691] | 0.17949 | 6 |
| leave_one_source_state_out | grouped_quantization | code | K0 → dense_anchor | ols | 0.91360 | [0.17577, 1.70119] | 1.52542 | 6 |
| leave_one_source_state_out | grouped_quantization | code | K0 → dense_anchor | ridge | 0.11397 | [-0.11961, 0.35097] | 0.47057 | 6 |
| leave_one_source_state_out | grouped_quantization | code | dense_anchor → dense_statistics | ols | -0.07811 | [-0.37594, 0.10564] | 0.48159 | 6 |
| leave_one_source_state_out | grouped_quantization | code | dense_anchor → dense_statistics | ridge | 0.31262 | [0.07250, 0.51472] | 0.44223 | 6 |
| leave_one_source_state_out | grouped_quantization | qa | K0 → dense_anchor | ols | -0.80927 | [-2.65123, 0.33489] | 2.98613 | 6 |
| leave_one_source_state_out | grouped_quantization | qa | K0 → dense_anchor | ridge | -0.83412 | [-2.66602, 0.19607] | 2.86209 | 6 |
| leave_one_source_state_out | grouped_quantization | qa | dense_anchor → dense_statistics | ols | 2.05567 | [0.99378, 3.42335] | 2.42957 | 6 |
| leave_one_source_state_out | grouped_quantization | qa | dense_anchor → dense_statistics | ridge | 1.29369 | [0.24962, 3.13225] | 2.88263 | 6 |
| leave_one_size_out | grouped_quantization | math | K0 → dense_anchor | ols | 1.84382 | [0.00661, 4.05054] | 4.04393 | 6 |
| leave_one_size_out | grouped_quantization | math | K0 → dense_anchor | ridge | 0.71370 | [-0.01768, 1.56762] | 1.58530 | 6 |
| leave_one_size_out | grouped_quantization | math | dense_anchor → dense_statistics | ols | 1.00922 | [0.31375, 1.72585] | 1.41210 | 6 |
| leave_one_size_out | grouped_quantization | math | dense_anchor → dense_statistics | ridge | 0.69434 | [-0.10950, 1.56963] | 1.67914 | 6 |
| leave_one_size_out | grouped_quantization | code | K0 → dense_anchor | ols | 0.81302 | [-1.04801, 2.90593] | 3.95393 | 6 |
| leave_one_size_out | grouped_quantization | code | K0 → dense_anchor | ridge | 0.91132 | [0.08135, 1.80137] | 1.72002 | 6 |
| leave_one_size_out | grouped_quantization | code | dense_anchor → dense_statistics | ols | 2.67876 | [1.03668, 4.41561] | 3.37892 | 6 |
| leave_one_size_out | grouped_quantization | code | dense_anchor → dense_statistics | ridge | 0.05563 | [-0.24645, 0.37917] | 0.62562 | 6 |
| leave_one_size_out | grouped_quantization | qa | K0 → dense_anchor | ols | -1.32152 | [-3.25466, 0.61162] | 3.86629 | 6 |
| leave_one_size_out | grouped_quantization | qa | K0 → dense_anchor | ridge | 1.02348 | [-0.02653, 2.34463] | 2.37116 | 6 |
| leave_one_size_out | grouped_quantization | qa | dense_anchor → dense_statistics | ols | 4.82019 | [1.06812, 9.12334] | 8.05522 | 6 |
| leave_one_size_out | grouped_quantization | qa | dense_anchor → dense_statistics | ridge | 0.09182 | [-0.00860, 0.27901] | 0.28762 | 6 |
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
| grouped_quantization | math | K0 | 0.81149 [0.39014, 1.14148] | -1.67467 [-2.57609, -0.65499] | -0.71360 [-1.06212, -0.26443] | 0.96106 [0.19097, 1.78965] |
| grouped_quantization | math | dense_anchor | 0.81149 [0.39014, 1.14148] | -0.65186 [-1.17966, -0.01405] | -0.64221 [-1.05371, -0.21976] | 0.00965 [-0.28443, 0.31134] |
| grouped_quantization | math | dense_statistics | 0.81149 [0.39014, 1.14148] | -0.65584 [-1.19971, 0.04309] | -0.29048 [-0.72006, 0.13400] | 0.36535 [-0.00348, 0.69067] |
| grouped_quantization | code | K0 | 0.82890 [0.42415, 1.13782] | -1.59940 [-2.49333, -0.58764] | -0.72127 [-1.06881, -0.25394] | 0.87812 [0.14805, 1.65714] |
| grouped_quantization | code | dense_anchor | 0.82890 [0.42415, 1.13782] | -0.68580 [-1.22478, -0.03638] | -0.60730 [-1.04614, -0.19476] | 0.07850 [-0.22095, 0.37213] |
| grouped_quantization | code | dense_statistics | 0.82890 [0.42415, 1.13782] | -0.76390 [-1.15376, -0.35276] | -0.29468 [-0.57457, -0.03502] | 0.46923 [0.27595, 0.65869] |
| grouped_quantization | qa | K0 | 0.91793 [0.43478, 1.31961] | -1.93962 [-3.03874, -0.76441] | -0.82331 [-1.24752, -0.29612] | 1.11632 [0.23066, 2.09925] |
| grouped_quantization | qa | dense_anchor | 0.91793 [0.43478, 1.31961] | -2.74889 [-4.61998, -1.09668] | -1.65743 [-3.71116, -0.36037] | 1.09146 [0.24424, 2.06361] |
| grouped_quantization | qa | dense_statistics | 0.91793 [0.43478, 1.31961] | -0.69322 [-1.36585, 0.17663] | -0.36374 [-0.72529, 0.09151] | 0.32948 [-0.08613, 0.66885] |
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
| leave_one_source_state_out | grouped_quantization | K0/zero | 0.33333 | 0.15494 | 6 |
| leave_one_source_state_out | grouped_quantization | K0/constant | 0.70370 | 0.07569 | 6 |
| leave_one_source_state_out | grouped_quantization | K0/median_curve | 0.61111 | 0.09375 | 6 |
| leave_one_source_state_out | grouped_quantization | K0/ols | 0.50000 | 0.12646 | 6 |
| leave_one_source_state_out | grouped_quantization | K0/ridge | 0.64815 | 0.07628 | 6 |
| leave_one_source_state_out | grouped_quantization | K0/delivered | 0.61111 | 0.09375 | 6 |
| leave_one_source_state_out | grouped_quantization | dense_anchor/ols | 0.25926 | 0.25606 | 6 |
| leave_one_source_state_out | grouped_quantization | dense_anchor/ridge | 0.38889 | 0.21027 | 6 |
| leave_one_source_state_out | grouped_quantization | dense_statistics/ols | 0.44444 | 0.12840 | 6 |
| leave_one_source_state_out | grouped_quantization | dense_statistics/ridge | 0.18519 | 0.20167 | 6 |
| leave_one_size_out | grouped_quantization | K0/zero | 0.33333 | 0.15494 | 6 |
| leave_one_size_out | grouped_quantization | K0/constant | 0.70370 | 0.07569 | 6 |
| leave_one_size_out | grouped_quantization | K0/median_curve | 0.66667 | 0.07861 | 6 |
| leave_one_size_out | grouped_quantization | K0/ols | 0.51852 | 0.10155 | 6 |
| leave_one_size_out | grouped_quantization | K0/ridge | 0.62963 | 0.07849 | 6 |
| leave_one_size_out | grouped_quantization | K0/delivered | 0.66667 | 0.07861 | 6 |
| leave_one_size_out | grouped_quantization | dense_anchor/ols | 0.50000 | 0.14469 | 6 |
| leave_one_size_out | grouped_quantization | dense_anchor/ridge | 0.20370 | 0.22196 | 6 |
| leave_one_size_out | grouped_quantization | dense_statistics/ols | 0.27778 | 0.19503 | 6 |
| leave_one_size_out | grouped_quantization | dense_statistics/ridge | 0.57407 | 0.04887 | 6 |
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
| unseen_strength_same_sources | grouped_quantization | math | 3.0 | median_curve | 3.26224 | 3.26224 | 3.26224 | 6 |
| unseen_strength_same_sources | grouped_quantization | math | 4.0 | median_curve | 0.54620 | 0.54620 | 0.54620 | 6 |
| unseen_strength_same_sources | grouped_quantization | math | 5.0 | median_curve | 0.70688 | 0.70688 | 0.70688 | 6 |
| unseen_strength_same_sources | grouped_quantization | math | 3.0 | ols | 2.75958 | 2.75071 | 2.76561 | 6 |
| unseen_strength_same_sources | grouped_quantization | math | 4.0 | ols | 1.81639 | 1.48215 | 1.31988 | 6 |
| unseen_strength_same_sources | grouped_quantization | math | 5.0 | ols | 2.63722 | 3.91725 | 3.96747 | 6 |
| unseen_strength_same_sources | grouped_quantization | math | 3.0 | ridge | 3.11302 | 2.75086 | 2.77013 | 6 |
| unseen_strength_same_sources | grouped_quantization | math | 4.0 | ridge | 1.36341 | 1.30706 | 1.30643 | 6 |
| unseen_strength_same_sources | grouped_quantization | math | 5.0 | ridge | 1.44203 | 3.91656 | 1.80268 | 6 |
| unseen_strength_same_sources | grouped_quantization | code | 3.0 | median_curve | 3.42856 | 3.42856 | 3.42856 | 6 |
| unseen_strength_same_sources | grouped_quantization | code | 4.0 | median_curve | 0.71942 | 0.71942 | 0.71942 | 6 |
| unseen_strength_same_sources | grouped_quantization | code | 5.0 | median_curve | 0.94732 | 0.94732 | 0.94732 | 6 |
| unseen_strength_same_sources | grouped_quantization | code | 3.0 | ols | 2.79500 | 2.78949 | 2.82585 | 6 |
| unseen_strength_same_sources | grouped_quantization | code | 4.0 | ols | 1.80953 | 1.52964 | 1.34771 | 6 |
| unseen_strength_same_sources | grouped_quantization | code | 5.0 | ols | 2.71246 | 3.92491 | 4.00674 | 6 |
| unseen_strength_same_sources | grouped_quantization | code | 3.0 | ridge | 3.26001 | 3.00128 | 3.01403 | 6 |
| unseen_strength_same_sources | grouped_quantization | code | 4.0 | ridge | 1.32253 | 1.32253 | 1.32253 | 6 |
| unseen_strength_same_sources | grouped_quantization | code | 5.0 | ridge | 0.90752 | 1.25658 | 1.72730 | 6 |
| unseen_strength_same_sources | grouped_quantization | qa | 3.0 | median_curve | 3.30607 | 3.30607 | 3.30607 | 6 |
| unseen_strength_same_sources | grouped_quantization | qa | 4.0 | median_curve | 0.54301 | 0.54301 | 0.54301 | 6 |
| unseen_strength_same_sources | grouped_quantization | qa | 5.0 | median_curve | 0.36845 | 0.36845 | 0.36845 | 6 |
| unseen_strength_same_sources | grouped_quantization | qa | 3.0 | ols | 3.03955 | 3.07453 | 2.99529 | 6 |
| unseen_strength_same_sources | grouped_quantization | qa | 4.0 | ols | 1.84339 | 1.98100 | 1.24584 | 6 |
| unseen_strength_same_sources | grouped_quantization | qa | 5.0 | ols | 2.63635 | 3.05023 | 4.08583 | 6 |
| unseen_strength_same_sources | grouped_quantization | qa | 3.0 | ridge | 3.23918 | 3.23553 | 3.03654 | 6 |
| unseen_strength_same_sources | grouped_quantization | qa | 4.0 | ridge | 1.43085 | 1.35175 | 1.22527 | 6 |
| unseen_strength_same_sources | grouped_quantization | qa | 5.0 | ridge | 1.42467 | 1.42467 | 1.95725 | 6 |
| unseen_strength_new_source | grouped_quantization | math | 3.0 | median_curve | 3.26224 | 3.26224 | 3.26224 | 6 |
| unseen_strength_new_source | grouped_quantization | math | 4.0 | median_curve | 0.57782 | 0.57782 | 0.57782 | 6 |
| unseen_strength_new_source | grouped_quantization | math | 5.0 | median_curve | 0.70688 | 0.70688 | 0.70688 | 6 |
| unseen_strength_new_source | grouped_quantization | math | 3.0 | ols | 2.98998 | 3.01388 | 2.99177 | 6 |
| unseen_strength_new_source | grouped_quantization | math | 4.0 | ols | 2.92194 | 0.88562 | 1.21777 | 6 |
| unseen_strength_new_source | grouped_quantization | math | 5.0 | ols | 3.31473 | 3.77992 | 3.81647 | 6 |
| unseen_strength_new_source | grouped_quantization | math | 3.0 | ridge | 3.11528 | 3.06631 | 3.04725 | 6 |
| unseen_strength_new_source | grouped_quantization | math | 4.0 | ridge | 1.79113 | 1.75265 | 0.84420 | 6 |
| unseen_strength_new_source | grouped_quantization | math | 5.0 | ridge | 1.45289 | 2.04061 | 1.92515 | 6 |
| unseen_strength_new_source | grouped_quantization | code | 3.0 | median_curve | 3.42856 | 3.42856 | 3.42856 | 6 |
| unseen_strength_new_source | grouped_quantization | code | 4.0 | median_curve | 0.75703 | 0.75703 | 0.75703 | 6 |
| unseen_strength_new_source | grouped_quantization | code | 5.0 | median_curve | 0.94732 | 0.94732 | 0.94732 | 6 |
| unseen_strength_new_source | grouped_quantization | code | 3.0 | ols | 3.03876 | 3.11375 | 3.20347 | 6 |
| unseen_strength_new_source | grouped_quantization | code | 4.0 | ols | 2.96865 | 1.23608 | 1.28706 | 6 |
| unseen_strength_new_source | grouped_quantization | code | 5.0 | ols | 3.40460 | 3.89317 | 3.81550 | 6 |
| unseen_strength_new_source | grouped_quantization | code | 3.0 | ridge | 3.26653 | 3.18422 | 3.20965 | 6 |
| unseen_strength_new_source | grouped_quantization | code | 4.0 | ridge | 1.92096 | 1.58638 | 1.20624 | 6 |
| unseen_strength_new_source | grouped_quantization | code | 5.0 | ridge | 1.28586 | 1.30218 | 1.72387 | 6 |
| unseen_strength_new_source | grouped_quantization | qa | 3.0 | median_curve | 3.32109 | 3.32109 | 3.32109 | 6 |
| unseen_strength_new_source | grouped_quantization | qa | 4.0 | median_curve | 0.57121 | 0.57121 | 0.57121 | 6 |
| unseen_strength_new_source | grouped_quantization | qa | 5.0 | median_curve | 0.36887 | 0.36887 | 0.36887 | 6 |
| unseen_strength_new_source | grouped_quantization | qa | 3.0 | ols | 3.31729 | 3.56175 | 3.37280 | 6 |
| unseen_strength_new_source | grouped_quantization | qa | 4.0 | ols | 3.28560 | 4.40644 | 1.13750 | 6 |
| unseen_strength_new_source | grouped_quantization | qa | 5.0 | ols | 3.38300 | 4.36728 | 3.93658 | 6 |
| unseen_strength_new_source | grouped_quantization | qa | 3.0 | ridge | 3.29626 | 3.62907 | 3.27342 | 6 |
| unseen_strength_new_source | grouped_quantization | qa | 4.0 | ridge | 1.91211 | 2.92370 | 1.24539 | 6 |
| unseen_strength_new_source | grouped_quantization | qa | 5.0 | ridge | 1.48795 | 1.88060 | 1.48085 | 6 |
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
- grouped_quantization/code: switching anchor OLS to statistics ridge improves MAE by 0.39112, but this mixes inputs and regularization and does not count.
- grouped_quantization/qa: switching anchor OLS to statistics ridge improves MAE by 2.38515, but this mixes inputs and regularization and does not count.
- per_channel_quantization/math: ols has a positive same-source strength-transfer input gain but does not clear the across-source reading rule; it does not count.
- per_channel_quantization/math: switching anchor OLS to statistics ridge improves MAE by 0.23391, but this mixes inputs and regularization and does not count.
- per_channel_quantization/code: ridge has a positive same-source strength-transfer input gain but does not clear the across-source reading rule; it does not count.
- per_channel_quantization/code: switching anchor OLS to statistics ridge improves MAE by 0.26658, but this mixes inputs and regularization and does not count.
- per_channel_quantization/qa: switching anchor OLS to statistics ridge improves MAE by 0.70957, but this mixes inputs and regularization and does not count.

All requested comparisons use existing scalar responses. No compressed-model diagnostic enters a predictor. The small Pythia panel tests transfer between source states/sizes within one family; it does not establish architecture transfer. Signed improvements in reference CE can be negative and do not by themselves establish downstream task accuracy.
