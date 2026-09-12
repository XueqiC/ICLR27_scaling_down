# V75 distillation form audit and CPU checks

Selected E (math and code): a*log(1+E), one parameter, no intercept, zero at E=0, OLS. Baseline E-only: c+b*log(1+E), two parameters, unpenalized intercept c and standardized-ridge slope b; response at E=0 is c.

**Dense drift:** primary CI sign changes: 0/12 shifted comparisons; budget-specific CI sign changes: 0/36. Positive paired differences favor the selected form.

**Item uncertainty:** per-example losses are not stored in any of the 48 inspected snapshots. The requested 2,000-resample, seed-0 item bootstrap cannot be computed; no item interval is reported. Each evaluation contains 64 items per capability and scores 13,198 math, 4,516 code, and 251 QA target tokens.

## Forms, fit data, and coefficients

All capabilities/methods fit all 100 rows jointly across students; no token-count or equal-student weighting. Baseline choice varies by student, but fitted coefficients do not.

D = 25 trajectories × 4 checkpoints = 100 responses per capability (12 former dev + 9 former test + 4 former held-out-4B). There are 9 trajectories each for 270M and 1B, and 7 for 4B. Fitting uses raw weight 1 per row; normalized weights are 1/25 per trajectory × 1/4 per checkpoint = 1/100 per row. LOCO uses 25 folds with 96 training points each, refitting references and scaling within folds.

- `T`: T_c: cumulative supervised completion tokens, including repetitions
- `D_U`: unique training-pool completion tokens
- `E`: T_c / D_U
- `u`: log(1 + T_c / 35000)
- `w`: log(1 + E)
- `v`: log(D_U / D_ref)
- `n`: log(N / N_ref), a student-size descriptor
- `N`: Frozen V50 total meta-device parameter count, not nominal size or trainable adapter count
- `z_L`: (L0_student,cap - mean_L0_cap) / population_std_L0_cap
- `z_N`: (log(N_student) - mean_logN) / population_std_logN
- `s`: 1 - exp(-T_c / T_star)
- `logs`: natural logarithms

O: Unpenalized OLS: minimize sum_i (x_i beta - y_i)^2.

R: Minimize sum_i (Z_i beta_std - y_i)^2 + 0.001 sum_j penalized_j beta_std_j^2. Population-SD column scaling; center nonconstant columns only when an intercept exists. Intercept exempt; every other coefficient penalized. F1/F2 are not centered, preserving zero response at zero budget. coef is mapped back to the raw basis.

With mean squared loss over 100 rows, lambda is 0.00001, not 0.001.

One discrete fitted parameter chosen by lowest unpenalized training SSE on [17500,35000,70000,140000,280000]; four ridge coefficients plus T_star = five parameters.

| Form | Exact formula | Free parameters | Intercept | Intercept penalized | Student descriptor | Fit |
|---|---|---:|---|---|---|---|
| constant | `c` | 1 | yes | no | none | D/O |
| constant+src | `c+d n` | 2 | yes | no | n | D/O |
| T | `a u` | 1 | no | n/a | none | D/O |
| T+src | `a u+d n u` | 2 | no | n/a | n | D/O |
| E | `a w` | 1 | no | n/a | none | D/O |
| E+src | `a w+d n w` | 2 | no | n/a | n | D/O |
| joint | `a u+b u^2+c u v` | 3 | no | n/a | none | D/O |
| joint+src | `a u+b u^2+c u v+d n u` | 4 | no | n/a | n | D/O |
| F1:L0 | `(a+b z_L)w` | 2 | no | n/a | L0 | D/R |
| F1:logN | `(a+b z_N)w` | 2 | no | n/a | logN | D/R |
| F2:L0 | `(a+b z_L)s+(c+d z_L)w` | 5 | no | n/a | L0 | D/R |
| F2:logN | `(a+b z_N)s+(c+d z_N)w` | 5 | no | n/a | logN | D/R |
| T-only | `c+b u` | 2 | yes | no | none | D/R |
| E-only | `c+b w` | 2 | yes | no | none | D/R |
| surface:L0 | `c+a u+b w+d z_L` | 4 | yes | no | L0 | D/R |
| surface:logN | `c+a u+b w+d z_N` | 4 | yes | no | logN | D/R |

Frozen coefficients below follow each formula's displayed order. These are raw-basis coefficients, not standardized ridge coefficients. Full precision and normalization constants are retained in summary.json.

| Form | Coefficient order | Math | Code | QA |
|---|---|---|---|---|
| constant | c | 0.160732024 | 0.173201672 | -0.432171315 |
| constant+src | c, d | 0.165728433, 0.0496466195 | 0.1749345, 0.0172181742 | -0.424642511, 0.0748096552 |
| T | a | 0.114114607 | 0.124073741 | -0.051685636 |
| T+src | a, d | 0.117983079, 0.0380758982 | 0.125207745, 0.0111615703 | -0.0419870271, 0.0954597101 |
| E | a | 0.151278411 | 0.15508921 | 0.327317256 |
| E+src | a, d | 0.15911482, 0.059728297 | 0.1574277, 0.0178237279 | 0.355887112, 0.217756472 |
| joint | a, b, c | 0.0574369957, 0.0107932815, -0.0823476696 | 0.0827450015, 0.0073276267, -0.0622261747 | -2.96881165, 1.35615826, -1.02368379 |
| joint+src | a, b, c, d | 0.0605077541, 0.0109561656, -0.0839798192, 0.0403715196 | 0.0837243846, 0.00737957669, -0.06274673, 0.012876031 | -2.95932583, 1.35666142, -1.02872563, 0.124710917 |
| F1:L0 | a, b | 0.159780981, -0.0583658675 | 0.157691194, -0.0186939303 | 0.333959788, 0.209514969 |
| F1:logN | a, b | 0.159114264, 0.0594530031 | 0.157427205, 0.0177415302 | 0.355885758, 0.216752905 |
| F2:L0 | a, b, c, d | -0.0511954549, 0.0590538851, 0.187534664, -0.0906120448; T_star=70000 | 0.0609528815, 0.00410247568, 0.132529724, -0.0199636499; T_star=140000 | -2.91040842, -0.510556695, 2.19442302, 0.538571927; T_star=17500 |
| F2:logN | a, b, c, d | -0.0508894384, -0.0688461496, 0.186694193, 0.0966806263; T_star=70000 | 0.0605580624, -0.0113819736, 0.132483251, 0.0220729856; T_star=140000 | -2.96471959, -0.551243265, 2.25944091, 0.602666694; T_star=17500 |
| T-only | c, b | -0.00354162482, 0.11628241 | -0.015371568, 0.133483069 | -2.65593552, 1.5741092 |
| E-only | c, b | -0.0111847762, 0.158559374 | 0.0171610492, 0.143916729 | -2.67574528, 2.06925492 |
| surface:L0 | c, a, b, d | 0.0121578623, -0.0229750651, 0.172481011, -0.0535115142 | -0.00565072293, 0.0283814834, 0.130120302, -0.0217673619 | -2.52991953, -0.168942425, 2.15704634, 0.0964447045 |
| surface:logN | c, a, b, d | 0.0114903929, -0.022676478, 0.172094499, 0.0525711877 | -0.00599990961, 0.0285376893, 0.129918098, 0.0195509969 | -2.52068482, -0.173073497, 2.16239389, 0.114164589 |

Per capability, lowest leave-one-trajectory-out (LOCO) MAE, equal weight per trajectory and per checkpoint within trajectory. All 12 candidates within 0.02 nats (inclusive) of the minimum tie: choose fewer fitted parameters, then lower MAE, then declared candidate order. F2 counts four coefficients plus one fitted discrete T_star. Refit coefficients, descriptor references, column scaling and F2 T_star inside each training fold; T_star minimizes training SSE per capability on V56's fixed five-value grid.

Per student and capability, lowest development LOCO MAE among constant, intercept+T-only, intercept+E-only, and same-input surfaces with L0/logN. Exact ties: fewer parameters, then declared baseline order. Frozen before confirmation; no baseline selection or refitting on confirmation errors.

## Dense-drift sensitivity

delta'_trajectory,cap,budget = delta_trajectory,cap,budget +/- recorded max_abs_drift_trajectory. The same nonnegative trajectory maximum shifts every capability and budget; predictions and descriptors stay frozen.

Two deterministic sensitivity scenarios, not a probabilistic drift CI or a bound over all possible mixed-sign perturbations.

The recorded trajectory maxima are 0.0009337649402398895 for every 270M pool and 0.0007196634189547968 for every 1B pool. Primary results use V70's 5,000 PCG64 pool resamples, seed 0, with the same six pool indices across methods, students, capabilities, and sensitivity scenarios. Each pool retains all three budgets. Predictions remain the stored planned-budget predictions.

| Student | Cap. | Selected / frozen baseline | Shift | Selected MAE | Baseline MAE | Paired difference [95% pool CI] | CI sign |
|---|---|---|---|---:|---:|---|---|
| gemma3-270m | math | E / T-only | original | 0.074254358 | 0.065444382 | -0.008809976 [-0.009781847, -0.008105177] | negative |
| gemma3-270m | math | E / T-only | plus_drift | 0.073320593 | 0.064510617 | -0.008809976 [-0.009781847, -0.008105177] | negative |
| gemma3-270m | math | E / T-only | minus_drift | 0.075188123 | 0.066378147 | -0.008809976 [-0.009781847, -0.008105177] | negative |
| gemma3-270m | code | E / E-only | original | 0.019251711 | 0.023005919 | 0.003754209 [0.002534092, 0.004965926] | positive |
| gemma3-270m | code | E / E-only | plus_drift | 0.018525449 | 0.022279658 | 0.003754209 [0.002534092, 0.004965926] | positive |
| gemma3-270m | code | E / E-only | minus_drift | 0.019977972 | 0.023732181 | 0.003754209 [0.002534092, 0.004965926] | positive |
| gemma3-270m | qa | joint / E-only | original | 0.514982277 | 0.609664491 | 0.094682214 [0.077442232, 0.109336218] | positive |
| gemma3-270m | qa | joint / E-only | plus_drift | 0.514048512 | 0.609353236 | 0.095304724 [0.078064742, 0.109958728] | positive |
| gemma3-270m | qa | joint / E-only | minus_drift | 0.515916042 | 0.609975746 | 0.094059704 [0.076819722, 0.108713708] | positive |
| gemma3-1b | math | E / surface:L0 | original | 0.056896931 | 0.033486853 | -0.023410078 [-0.023537359, -0.023277550] | negative |
| gemma3-1b | math | E / surface:L0 | plus_drift | 0.056177267 | 0.032908885 | -0.023268382 [-0.023537359, -0.022994189] | negative |
| gemma3-1b | math | E / surface:L0 | minus_drift | 0.057616594 | 0.034152376 | -0.023464218 [-0.023563000, -0.023328004] | negative |
| gemma3-1b | code | E / E-only | original | 0.048525030 | 0.053212499 | 0.004687469 [0.004157699, 0.004968518] | positive |
| gemma3-1b | code | E / E-only | plus_drift | 0.047885329 | 0.052492836 | 0.004607506 [0.003917811, 0.004968518] | positive |
| gemma3-1b | code | E / E-only | minus_drift | 0.049164731 | 0.053932163 | 0.004767432 [0.004397587, 0.004968518] | positive |
| gemma3-1b | qa | joint / surface:logN | original | 0.463356721 | 0.450151814 | -0.013204907 [-0.019848841, -0.009775078] | negative |
| gemma3-1b | qa | joint / surface:logN | plus_drift | 0.462717020 | 0.449512113 | -0.013204907 [-0.019848841, -0.009775078] | negative |
| gemma3-1b | qa | joint / surface:logN | minus_drift | 0.463996421 | 0.450791515 | -0.013204907 [-0.019848841, -0.009775078] | negative |

All 16 methods' shifted MAEs, individual shifted responses/errors, and selected/baseline MAE intervals are in summary.json. No form or baseline is reselected using these errors.

## Evaluation-item uncertainty and separate budget-specific pool intervals

measurement_samples contains integer counts, not samples/losses; post_training and capability_losses contain one aggregate float per capability. No per-example loss sums or token denominators are stored. An item bootstrap cannot be recovered from these aggregates.

Scored-token counts are per evaluation, shared across snapshots/students here; do not sum repeated evaluations as independent items. Pool-cluster CIs condition on these fixed evaluation items and do not quantify item-sampling or training-seed uncertainty.

| Student | Capability | Items per evaluation | Scored target tokens per evaluation | Snapshots inspected |
|---|---|---:|---:|---:|
| gemma3-270m | math | 64 | 13198 | 24 |
| gemma3-270m | code | 64 | 4516 | 24 |
| gemma3-270m | qa | 64 | 251 | 24 |
| gemma3-1b | math | 64 | 13198 | 24 |
| gemma3-1b | code | 64 | 4516 | 24 |
| gemma3-1b | qa | 64 | 251 | 24 |

The following intervals resample **six training pools at one budget** (5,000 resamples, seed 0). They are supplementary pool-cluster intervals, not evaluation-item intervals; item intervals are unavailable at every budget. The primary V70 intervals above average over all three budgets within each pool.

| Student | Cap. | Planned T | Original paired difference [pool CI] | +drift [pool CI] | -drift [pool CI] | Any CI sign change | Item CI |
|---|---|---:|---|---|---|---|---|
| gemma3-270m | math | 50000 | -0.000887148 [-0.001621723, -0.000354838] | -0.000887148 [-0.001621723, -0.000354838] | -0.000887148 [-0.001621723, -0.000354838] | no | unavailable |
| gemma3-270m | math | 100000 | -0.006956256 [-0.007943968, -0.006240011] | -0.006956256 [-0.007943968, -0.006240011] | -0.006956256 [-0.007943968, -0.006240011] | no | unavailable |
| gemma3-270m | math | 200000 | -0.018586524 [-0.019779849, -0.017720684] | -0.018586524 [-0.019779849, -0.017720684] | -0.018586524 [-0.019779849, -0.017720684] | no | unavailable |
| gemma3-270m | code | 50000 | 0.009737026 [0.009682775, 0.009776339] | 0.009737026 [0.009682775, 0.009776339] | 0.009737026 [0.009682775, 0.009776339] | no | unavailable |
| gemma3-270m | code | 100000 | 0.001829065 [-0.001735105, 0.005366101] | 0.001829065 [-0.001735105, 0.005366101] | 0.001829065 [-0.001735105, 0.005366101] | no | unavailable |
| gemma3-270m | code | 200000 | -0.000303465 [-0.000391596, -0.000239519] | -0.000303465 [-0.000391596, -0.000239519] | -0.000303465 [-0.000391596, -0.000239519] | no | unavailable |
| gemma3-270m | qa | 50000 | 0.033644206 [-0.016000308, 0.075354054] | 0.035511736 [-0.014132778, 0.077221584] | 0.031776676 [-0.017867838, 0.073486524] | no | unavailable |
| gemma3-270m | qa | 100000 | 0.183875068 [0.183618746, 0.184075315] | 0.183875068 [0.183618746, 0.184075315] | 0.183875068 [0.183618746, 0.184075315] | no | unavailable |
| gemma3-270m | qa | 200000 | 0.066527368 [0.063430611, 0.068786543] | 0.066527368 [0.063430611, 0.068786543] | 0.066527368 [0.063430611, 0.068786543] | no | unavailable |
| gemma3-1b | math | 50000 | -0.021119045 [-0.021336739, -0.020799992] | -0.020693960 [-0.021336739, -0.019880740] | -0.021281466 [-0.021356073, -0.021178511] | no | unavailable |
| gemma3-1b | math | 100000 | -0.023519952 [-0.023620338, -0.023381518] | -0.023519952 [-0.023620338, -0.023381518] | -0.023519952 [-0.023620338, -0.023381518] | no | unavailable |
| gemma3-1b | math | 200000 | -0.025591236 [-0.025712589, -0.025423984] | -0.025591236 [-0.025712589, -0.025423984] | -0.025591236 [-0.025712589, -0.025423984] | no | unavailable |
| gemma3-1b | code | 50000 | 0.009050035 [0.007621801, 0.009776339] | 0.008810147 [0.006902138, 0.009776339] | 0.009289923 [0.008341465, 0.009776339] | no | unavailable |
| gemma3-1b | code | 100000 | 0.005315837 [0.005242891, 0.005368734] | 0.005315837 [0.005242891, 0.005368734] | 0.005315837 [0.005242891, 0.005368734] | no | unavailable |
| gemma3-1b | code | 200000 | -0.000303465 [-0.000391596, -0.000239519] | -0.000303465 [-0.000391596, -0.000239519] | -0.000303465 [-0.000391596, -0.000239519] | no | unavailable |
| gemma3-1b | qa | 50000 | -0.252458247 [-0.253495910, -0.251006933] | -0.252458247 [-0.253495910, -0.251006933] | -0.252458247 [-0.253495910, -0.251006933] | no | unavailable |
| gemma3-1b | qa | 100000 | 0.194654219 [0.194413488, 0.195006010] | 0.194654219 [0.194413488, 0.195006010] | 0.194654219 [0.194413488, 0.195006010] | no | unavailable |
| gemma3-1b | qa | 200000 | 0.018189307 [-0.001464648, 0.029320795] | 0.018189307 [-0.001464648, 0.029320795] | 0.018189307 [-0.001464648, 0.029320795] | no | unavailable |

## Reproduction and write scope

Run `python -B analysis/v75_distill_audit.py`; verify existing artifacts without writes using `python -B analysis/v75_distill_audit.py --check`. Only NumPy/CPU operations are used.

Validation reproduced 48 full-development fits, checked 4,800 independently specified basis vectors, reconstructed frozen planned-budget predictions from coefficients, verified recorded responses against eval.json, and matched all six original V70 MAEs and paired intervals. Shifted results also match V70's paired-summary implementation. The SHA256 of every file under results/v70-distill-confirm/ is checked before and after execution.

Paper artifacts are staged under results/v75-distill-audit/paper/ to respect the write-only restriction, preserving paper/paper/tables and paper/code/analysis relative paths. Top-level paper files are untouched. No commit was made.

- summary_json: `results/v75-distill-audit/summary.json`
- summary_markdown: `results/v75-distill-audit/summary.md`
- table: `results/v75-distill-audit/paper/paper/tables/distill_forms_audit.tex`
- code_mirror: `results/v75-distill-audit/paper/analysis/v75_distill_audit.py`
