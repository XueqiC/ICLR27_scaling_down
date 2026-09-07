# Quantization regions (V30b)

The unchanged V30 full-domain shared-η MAE is **1.04509**, versus zero-change **2.77695** nats. The low-bit region supplies **98.7% of its net full-domain gain**. Region-specific results below qualify the all-bit win.

In the 4/5-bit region, shared η improves over zero-change by only 0.05707 [-0.02543, 0.13967] nats; the gain is unresolved when its interval includes zero. The strongest simple baseline here is zero-change, which is included on the identical cells in every comparison.

capability LOSS: Delta L_c(b) = L_c(b) - L_c(b_ref), native-token CE nats. Negative changes, QA improvements, near-zero values, and collapse cells are retained. No observed-damage ratios, log-damage fits, or outcome-based censoring are used.

## Common comparison and freeze boundary

Bit-defined regions only: high b>=6, measurable b=4/5, collapse b<=3. Descriptive regime names are not outcome filters; full domain retains every V30 dev cell.

The primary tables reuse and verify every saved V30 cell prediction against its frozen fold coefficients and the original V10 JSON. Candidates use the same models, probes, reference anchors, ridge mapping (λ=1), and target information. Broad and shape512 remain separate; dense/16 is b_ref=16. An 8-bit fallback would cost one compressed reference point; no current input uses it. The terms near-zero, measurable non-collapse, and collapse describe typical behavior, not all cells. In particular 5-bit signals can be tiny and QA can improve.

Paired whole-model bootstrap; all capabilities/bits/candidates stay together, equal observed-cell weights. MAE fits held fixed; eta comparison refits each draw. 95% percentile intervals are exploratory, not multiplicity-adjusted or item/seed CIs.

- **v30_lomo_and_v30b**: Eta and amplitude mapping are frozen before scoring each held-out model. Amplitude uses only target size, family, dense reference L_c: zero compressed calibration points for current dense anchors. Basic-input prediction conditional on measured dense loss; development LOMO is not prospective new-source validation.
- **v30_paired_4_to_5**: CONDITIONAL SHAPE TRANSFER: observed target Delta L(4) calibrates amplitude (one compressed point). Only shape is frozen; not full basic-parameter prediction.
- **v30_descriptive_eta**: Per-development-model/capability amplitudes profiled on development curves; this is an in-sample shape estimate, not held-out amplitude prediction.

Here a_c means the raw multiplier of g(b). V30 fits signed amplitudes normalized to ΔL(4), then predicts them from log nominal model size, reference capability loss, and family indicators. The held-out a_c is predicted by that frozen mapping; it is not fitted to the target quantization curve. An unknown target dense loss still prevents a unique numerical basic-input prediction.

## Full actual-step function

The candidates are fixed g(b)=4^(-b)−4^(-b_ref), actual-step below, and shared g(b)=2^(-η(b−4))−2^(-η(b_ref−4)). One η is shared across development models and capabilities.

`q_max(b) = 2^(b-1) - 1`

`g(b) = q_max(b)^(-2) - q_max(b_ref)^(-2)`

V30b implements this **full-bitwidth function directly**, including reference subtraction. η≈2.20 is only its local 4→5 equivalent slope and is never substituted for the actual-step candidate. Dividing each g by g(4) changes the amplitude units, not predictions.

## Regional MAE: unchanged V30 predictions

### broad

| Region / capability (models; cells) | Candidate | MAE [95% CI] | Gain over zero-change [95% CI] |
|---|---|---:|---:|
| High bits (b≥6) / math (12; 24) | Zero-change | 0.00964 [0.00448, 0.01710] | -0.00000 [-0.00000, -0.00000] |
| High bits (b≥6) / math (12; 24) | 4^-b | 0.10098 [0.08474, 0.11871] | -0.09134 [-0.10840, -0.07419] |
| High bits (b≥6) / math (12; 24) | Actual step² | 0.05875 [0.04880, 0.06911] | -0.04911 [-0.06027, -0.03502] |
| High bits (b≥6) / math (12; 24) | Shared η | 0.00937 [0.00431, 0.01677] | 0.00028 [0.00005, 0.00049] |
| High bits (b≥6) / code (12; 24) | Zero-change | 0.00924 [0.00585, 0.01339] | -0.00000 [-0.00000, -0.00000] |
| High bits (b≥6) / code (12; 24) | 4^-b | 0.10315 [0.08339, 0.12356] | -0.09390 [-0.11454, -0.07418] |
| High bits (b≥6) / code (12; 24) | Actual step² | 0.06097 [0.04880, 0.07336] | -0.05173 [-0.06475, -0.03877] |
| High bits (b≥6) / code (12; 24) | Shared η | 0.00887 [0.00542, 0.01309] | 0.00037 [0.00019, 0.00054] |
| High bits (b≥6) / qa (12; 24) | Zero-change | 0.02929 [0.01972, 0.03847] | -0.00000 [-0.00000, -0.00000] |
| High bits (b≥6) / qa (12; 24) | 4^-b | 0.09791 [0.07857, 0.11709] | -0.06862 [-0.08860, -0.04871] |
| High bits (b≥6) / qa (12; 24) | Actual step² | 0.06532 [0.04961, 0.08078] | -0.03603 [-0.05200, -0.01861] |
| High bits (b≥6) / qa (12; 24) | Shared η | 0.02949 [0.01996, 0.03857] | -0.00020 [-0.00038, -0.00000] |
| High bits (b≥6) / pooled (12; 72) | Zero-change | 0.01606 [0.01096, 0.02160] | -0.00000 [-0.00000, -0.00000] |
| High bits (b≥6) / pooled (12; 72) | 4^-b | 0.10068 [0.08348, 0.11835] | -0.08462 [-0.10269, -0.06743] |
| High bits (b≥6) / pooled (12; 72) | Actual step² | 0.06168 [0.05060, 0.07255] | -0.04562 [-0.05756, -0.03247] |
| High bits (b≥6) / pooled (12; 72) | Shared η | 0.01591 [0.01081, 0.02142] | 0.00015 [0.00004, 0.00026] |
| 4–5 bits / math (12; 24) | Zero-change | 0.38813 [0.21747, 0.59030] | -0.00000 [-0.00000, -0.00000] |
| 4–5 bits / math (12; 24) | 4^-b | 1.67086 [1.34490, 2.02513] | -1.28274 [-1.68995, -0.86479] |
| 4–5 bits / math (12; 24) | Actual step² | 1.12559 [0.86941, 1.39281] | -0.73746 [-1.08671, -0.34089] |
| 4–5 bits / math (12; 24) | Shared η | 0.24285 [0.11489, 0.40818] | 0.14528 [0.04819, 0.24055] |
| 4–5 bits / code (12; 24) | Zero-change | 0.38799 [0.17710, 0.64930] | -0.00000 [-0.00000, -0.00000] |
| 4–5 bits / code (12; 24) | 4^-b | 1.66980 [1.25633, 2.10975] | -1.28182 [-1.82703, -0.69900] |
| 4–5 bits / code (12; 24) | Actual step² | 1.12537 [0.78740, 1.46924] | -0.73738 [-1.20744, -0.19652] |
| 4–5 bits / code (12; 24) | Shared η | 0.30401 [0.13442, 0.52369] | 0.08398 [0.00239, 0.16285] |
| 4–5 bits / qa (12; 24) | Zero-change | 0.28700 [0.18195, 0.40336] | -0.00000 [-0.00000, -0.00000] |
| 4–5 bits / qa (12; 24) | 4^-b | 1.58092 [1.19681, 1.97650] | -1.29392 [-1.68872, -0.87184] |
| 4–5 bits / qa (12; 24) | Actual step² | 1.15029 [0.83243, 1.46618] | -0.86329 [-1.18958, -0.49970] |
| 4–5 bits / qa (12; 24) | Shared η | 0.34504 [0.22472, 0.47095] | -0.05804 [-0.18165, 0.06875] |
| 4–5 bits / pooled (12; 72) | Zero-change | 0.35437 [0.21005, 0.52002] | -0.00000 [-0.00000, -0.00000] |
| 4–5 bits / pooled (12; 72) | 4^-b | 1.64053 [1.28953, 2.01956] | -1.28616 [-1.70806, -0.87010] |
| 4–5 bits / pooled (12; 72) | Actual step² | 1.13375 [0.85935, 1.43045] | -0.77938 [-1.12926, -0.40873] |
| 4–5 bits / pooled (12; 72) | Shared η | 0.29730 [0.18386, 0.42690] | 0.05707 [-0.02543, 0.13967] |
| Low bits (b≤3) / math (12; 12) | Zero-change | 13.98412 [9.76441, 17.97693] | -0.00000 [-0.00000, -0.00000] |
| Low bits (b≤3) / math (12; 12) | 4^-b | 4.57467 [2.80704, 6.56006] | 9.40945 [5.37152, 12.89639] |
| Low bits (b≤3) / math (12; 12) | Actual step² | 4.54195 [2.78559, 6.50097] | 9.44217 [5.28155, 13.09146] |
| Low bits (b≤3) / math (12; 12) | Shared η | 4.50384 [2.73527, 6.44841] | 9.48028 [5.17161, 13.29834] |
| Low bits (b≤3) / code (12; 12) | Zero-change | 14.19945 [10.04599, 18.25158] | -0.00000 [-0.00000, -0.00000] |
| Low bits (b≤3) / code (12; 12) | 4^-b | 4.23878 [2.25327, 6.46180] | 9.96067 [6.06938, 13.39012] |
| Low bits (b≤3) / code (12; 12) | Actual step² | 4.28218 [2.36330, 6.40447] | 9.91727 [5.92654, 13.43368] |
| Low bits (b≤3) / code (12; 12) | Shared η | 4.38797 [2.58804, 6.38177] | 9.81148 [5.76130, 13.39131] |
| Low bits (b≤3) / qa (12; 12) | Zero-change | 11.24804 [7.29435, 15.31754] | -0.00000 [-0.00000, -0.00000] |
| Low bits (b≤3) / qa (12; 12) | 4^-b | 4.79154 [2.64437, 7.30529] | 6.45651 [3.02726, 9.69847] |
| Low bits (b≤3) / qa (12; 12) | Actual step² | 4.84400 [2.75090, 7.28944] | 6.40405 [2.86437, 9.70702] |
| Low bits (b≤3) / qa (12; 12) | Shared η | 4.90522 [2.84149, 7.29859] | 6.34283 [2.69998, 9.72219] |
| Low bits (b≤3) / pooled (12; 36) | Zero-change | 13.14387 [9.06819, 17.12847] | -0.00000 [-0.00000, -0.00000] |
| Low bits (b≤3) / pooled (12; 36) | 4^-b | 4.53500 [2.66805, 6.70660] | 8.60888 [4.86056, 11.98916] |
| Low bits (b≤3) / pooled (12; 36) | Actual step² | 4.55604 [2.73308, 6.66311] | 8.58783 [4.74117, 12.07283] |
| Low bits (b≤3) / pooled (12; 36) | Shared η | 4.59901 [2.81015, 6.64147] | 8.54486 [4.58341, 12.13490] |
| Full domain / math (12; 60) | Zero-change | 2.95593 [2.06648, 3.80993] | -0.00000 [-0.00000, -0.00000] |
| Full domain / math (12; 60) | 4^-b | 1.62367 [1.24775, 2.00579] | 1.33226 [0.52830, 2.00699] |
| Full domain / math (12; 60) | Actual step² | 1.38213 [1.03178, 1.74448] | 1.57381 [0.72305, 2.30228] |
| Full domain / math (12; 60) | Shared η | 1.00165 [0.62588, 1.42934] | 1.95428 [1.07496, 2.72597] |
| Full domain / code (12; 60) | Zero-change | 2.99878 [2.12702, 3.85924] | -0.00000 [-0.00000, -0.00000] |
| Full domain / code (12; 60) | 4^-b | 1.55694 [1.13794, 2.00355] | 1.44184 [0.68600, 2.09403] |
| Full domain / code (12; 60) | Actual step² | 1.33097 [0.93795, 1.74688] | 1.66781 [0.87011, 2.36574] |
| Full domain / code (12; 60) | Shared η | 1.00275 [0.62872, 1.44944] | 1.99604 [1.17567, 2.70886] |
| Full domain / qa (12; 60) | Zero-change | 2.37613 [1.55250, 3.21139] | -0.00000 [-0.00000, -0.00000] |
| Full domain / qa (12; 60) | 4^-b | 1.62984 [1.19196, 2.13163] | 0.74629 [0.09644, 1.33320] |
| Full domain / qa (12; 60) | Actual step² | 1.45504 [1.04302, 1.93139] | 0.92108 [0.23683, 1.54269] |
| Full domain / qa (12; 60) | Shared η | 1.13086 [0.72888, 1.60026] | 1.24527 [0.52705, 1.89859] |
| Full domain / pooled (12; 180) | Zero-change | 2.77695 [1.92340, 3.62373] | -0.00000 [-0.00000, -0.00000] |
| Full domain / pooled (12; 180) | 4^-b | 1.60348 [1.21593, 2.02690] | 1.17346 [0.45346, 1.79786] |
| Full domain / pooled (12; 180) | Actual step² | 1.38938 [1.02896, 1.78322] | 1.38757 [0.62398, 2.05772] |
| Full domain / pooled (12; 180) | Shared η | 1.04509 [0.67770, 1.48344] | 1.73186 [0.93392, 2.43832] |

### shape512

| Region / capability (models; cells) | Candidate | MAE [95% CI] | Gain over zero-change [95% CI] |
|---|---|---:|---:|
| High bits (b≥6) | No measured cells | N/A | N/A |
| 4–5 bits / math (7; 11) | Zero-change | 0.64650 [0.38054, 1.03216] | -0.00000 [-0.00000, -0.00000] |
| 4–5 bits / math (7; 11) | 4^-b | 0.45328 [0.28730, 0.67827] | 0.19322 [-0.01871, 0.41793] |
| 4–5 bits / math (7; 11) | Actual step² | 0.44427 [0.27955, 0.67237] | 0.20223 [-0.01398, 0.42495] |
| 4–5 bits / math (7; 11) | Shared η | 0.42439 [0.25317, 0.66520] | 0.22211 [0.00656, 0.43773] |
| 4–5 bits / code (7; 11) | Zero-change | 0.65695 [0.29084, 1.18098] | -0.00000 [-0.00000, -0.00000] |
| 4–5 bits / code (7; 11) | 4^-b | 0.65573 [0.30417, 1.14027] | 0.00121 [-0.33598, 0.23276] |
| 4–5 bits / code (7; 11) | Actual step² | 0.65080 [0.30066, 1.14258] | 0.00614 [-0.33195, 0.23436] |
| 4–5 bits / code (7; 11) | Shared η | 0.63536 [0.28088, 1.13573] | 0.02159 [-0.31917, 0.23935] |
| 4–5 bits / qa (7; 11) | Zero-change | 0.41917 [0.22916, 0.62274] | -0.00000 [-0.00000, -0.00000] |
| 4–5 bits / qa (7; 11) | 4^-b | 0.45962 [0.26404, 0.71393] | -0.04044 [-0.23859, 0.10409] |
| 4–5 bits / qa (7; 11) | Actual step² | 0.45877 [0.26123, 0.71245] | -0.03960 [-0.24323, 0.10922] |
| 4–5 bits / qa (7; 11) | Shared η | 0.45345 [0.25004, 0.71065] | -0.03428 [-0.25830, 0.13382] |
| 4–5 bits / pooled (7; 33) | Zero-change | 0.57421 [0.33869, 0.89681] | -0.00000 [-0.00000, -0.00000] |
| 4–5 bits / pooled (7; 33) | 4^-b | 0.52288 [0.34729, 0.76918] | 0.05133 [-0.17386, 0.21163] |
| 4–5 bits / pooled (7; 33) | Actual step² | 0.51795 [0.34053, 0.76865] | 0.05626 [-0.17310, 0.21702] |
| 4–5 bits / pooled (7; 33) | Shared η | 0.50440 [0.31986, 0.76977] | 0.06981 [-0.17413, 0.23092] |
| Low bits (b≤3) | No measured cells | N/A | N/A |
| Full domain / math (7; 11) | Zero-change | 0.64650 [0.38054, 1.03216] | -0.00000 [-0.00000, -0.00000] |
| Full domain / math (7; 11) | 4^-b | 0.45328 [0.28730, 0.67827] | 0.19322 [-0.01871, 0.41793] |
| Full domain / math (7; 11) | Actual step² | 0.44427 [0.27955, 0.67237] | 0.20223 [-0.01398, 0.42495] |
| Full domain / math (7; 11) | Shared η | 0.42439 [0.25317, 0.66520] | 0.22211 [0.00656, 0.43773] |
| Full domain / code (7; 11) | Zero-change | 0.65695 [0.29084, 1.18098] | -0.00000 [-0.00000, -0.00000] |
| Full domain / code (7; 11) | 4^-b | 0.65573 [0.30417, 1.14027] | 0.00121 [-0.33598, 0.23276] |
| Full domain / code (7; 11) | Actual step² | 0.65080 [0.30066, 1.14258] | 0.00614 [-0.33195, 0.23436] |
| Full domain / code (7; 11) | Shared η | 0.63536 [0.28088, 1.13573] | 0.02159 [-0.31917, 0.23935] |
| Full domain / qa (7; 11) | Zero-change | 0.41917 [0.22916, 0.62274] | -0.00000 [-0.00000, -0.00000] |
| Full domain / qa (7; 11) | 4^-b | 0.45962 [0.26404, 0.71393] | -0.04044 [-0.23859, 0.10409] |
| Full domain / qa (7; 11) | Actual step² | 0.45877 [0.26123, 0.71245] | -0.03960 [-0.24323, 0.10922] |
| Full domain / qa (7; 11) | Shared η | 0.45345 [0.25004, 0.71065] | -0.03428 [-0.25830, 0.13382] |
| Full domain / pooled (7; 33) | Zero-change | 0.57421 [0.33869, 0.89681] | -0.00000 [-0.00000, -0.00000] |
| Full domain / pooled (7; 33) | 4^-b | 0.52288 [0.34729, 0.76918] | 0.05133 [-0.17386, 0.21163] |
| Full domain / pooled (7; 33) | Actual step² | 0.51795 [0.34053, 0.76865] | 0.05626 [-0.17310, 0.21702] |
| Full domain / pooled (7; 33) | Shared η | 0.50440 [0.31986, 0.76977] | 0.06981 [-0.17413, 0.23092] |

OLMo3-32B is the int3 exception and stays in the low-bit bucket and full-domain score: ΔL_math=0.68241, ΔL_code=1.22120, ΔL_qa=0.37514 nats. It does not show the catastrophic damage typical of the other int3 curves.

### Where the all-bit gain comes from

Contributions use all 180 broad cells as the denominator and sum to each candidate's full-domain gain. They do not confuse a region's per-cell MAE with its weight in the headline result.

| Region | Candidate | Contribution to full-domain gain [95% CI] | Share of net gain |
|---|---|---:|---:|
| High bits (b≥6) | 4^-b | -0.03385 [-0.04107, -0.02697] | -2.9% |
| High bits (b≥6) | Actual step² | -0.01825 [-0.02302, -0.01299] | -1.3% |
| High bits (b≥6) | Shared η | 0.00006 [0.00002, 0.00010] | 0.0% |
| 4–5 bits | 4^-b | -0.51446 [-0.68322, -0.34804] | -43.8% |
| 4–5 bits | Actual step² | -0.31175 [-0.45170, -0.16349] | -22.5% |
| 4–5 bits | Shared η | 0.02283 [-0.01017, 0.05587] | 1.3% |
| Low bits (b≤3) | 4^-b | 1.72178 [0.97211, 2.39783] | 146.7% |
| Low bits (b≤3) | Actual step² | 1.71757 [0.94823, 2.41457] | 123.8% |
| Low bits (b≤3) | Shared η | 1.70897 [0.91668, 2.42698] | 98.7% |

The baseline comparison and candidate ranking answer different questions. Almost all shared-η gain over zero-change comes from int3, where all candidates predict large damage. Shared η actually has slightly worse point MAE than both fixed candidates in that bucket. Its all-bit advantage over those candidates comes from avoiding their overprediction in the high-bit and 4/5-bit regions. Paired candidate comparisons below quantify that distinction; negative differences favor the first candidate.

| Region | First / second candidate | First MAE − second MAE [95% CI] |
|---|---|---:|
| High bits (b≥6) | 4^-b / Actual step² | 0.03900 [0.03191, 0.04658] |
| High bits (b≥6) | 4^-b / Shared η | 0.08477 [0.06748, 0.10296] |
| High bits (b≥6) | Actual step² / Shared η | 0.04577 [0.03255, 0.05779] |
| 4–5 bits | 4^-b / Actual step² | 0.50678 [0.41504, 0.60491] |
| 4–5 bits | 4^-b / Shared η | 1.34323 [0.95362, 1.75698] |
| 4–5 bits | Actual step² / Shared η | 0.83645 [0.49489, 1.16722] |
| Low bits (b≤3) | 4^-b / Actual step² | -0.02104 [-0.19979, 0.16171] |
| Low bits (b≤3) | 4^-b / Shared η | -0.06401 [-0.41620, 0.29722] |
| Low bits (b≤3) | Actual step² / Shared η | -0.04297 [-0.21979, 0.13952] |
| Full domain | 4^-b / Actual step² | 0.21410 [0.15988, 0.26933] |
| Full domain | 4^-b / Shared η | 0.55840 [0.39211, 0.73198] |
| Full domain | Actual step² / Shared η | 0.34429 [0.20789, 0.47982] |

## Validity range: development fits on 4/5 versus all bits

On the same 12 broad models, η(all bits)=4.68833 [4.06422, 5.94957]; η(4/5 only)=2.66690 [2.51733, 2.93913]. The paired-refit difference is 2.02143 [1.40070, 3.25856].

The following sensitivity refits **every candidate's amplitude labels and mapping**, plus shared η, using only training-model 4/5-bit cells. Every held-out bit is still scored, with zero compressed target calibration. Thus comparisons within each table have identical training information. Differences between training ranges are not an η-only intervention, and extrapolation to int3 is explicitly retained.

| Region / capability (models; cells) | Candidate | MAE [95% CI] | Gain over zero-change [95% CI] |
|---|---|---:|---:|
| High bits (b≥6) / pooled (12; 72) | Zero-change | 0.01606 [0.01096, 0.02160] | -0.00000 [-0.00000, -0.00000] |
| High bits (b≥6) / pooled (12; 72) | 4^-b | 0.01829 [0.01396, 0.02270] | -0.00223 [-0.00617, 0.00253] |
| High bits (b≥6) / pooled (12; 72) | Actual step² | 0.01671 [0.01249, 0.02087] | -0.00065 [-0.00380, 0.00301] |
| High bits (b≥6) / pooled (12; 72) | Shared η | 0.01393 [0.00934, 0.01896] | 0.00213 [0.00015, 0.00394] |
| 4–5 bits / pooled (12; 72) | Zero-change | 0.35437 [0.21005, 0.52002] | -0.00000 [-0.00000, -0.00000] |
| 4–5 bits / pooled (12; 72) | 4^-b | 0.32587 [0.22582, 0.43930] | 0.02850 [-0.04371, 0.10742] |
| 4–5 bits / pooled (12; 72) | Actual step² | 0.32201 [0.22115, 0.43712] | 0.03236 [-0.03987, 0.11047] |
| 4–5 bits / pooled (12; 72) | Shared η | 0.31473 [0.21135, 0.43277] | 0.03964 [-0.03215, 0.11657] |
| Low bits (b≤3) / pooled (12; 36) | Zero-change | 13.14387 [9.06819, 17.12847] | -0.00000 [-0.00000, -0.00000] |
| Low bits (b≤3) / pooled (12; 36) | 4^-b | 11.62709 [7.67073, 15.53724] | 1.51678 [0.89943, 2.15144] |
| Low bits (b≤3) / pooled (12; 36) | Actual step² | 11.07056 [7.16822, 14.98067] | 2.07332 [1.20857, 2.95612] |
| Low bits (b≤3) / pooled (12; 36) | Shared η | 10.68476 [6.83015, 14.54952] | 2.45912 [1.42370, 3.50010] |
| Full domain / pooled (12; 180) | Zero-change | 2.77695 [1.92340, 3.62373] | -0.00000 [-0.00000, -0.00000] |
| Full domain / pooled (12; 180) | 4^-b | 2.46308 [1.65442, 3.26476] | 0.31386 [0.17669, 0.45588] |
| Full domain / pooled (12; 180) | Actual step² | 2.34960 [1.54952, 3.15220] | 0.42735 [0.24102, 0.61859] |
| Full domain / pooled (12; 180) | Shared η | 2.26842 [1.47579, 3.06436] | 0.50853 [0.28785, 0.73329] |

| Evaluation region | Candidate | All-bit-fit MAE − 4/5-fit MAE [paired 95% CI] |
|---|---|---:|
| High bits (b≥6) | 4^-b | 0.08239 [0.06643, 0.09869] |
| High bits (b≥6) | Actual step² | 0.04497 [0.03375, 0.05595] |
| High bits (b≥6) | Shared η | 0.00198 [0.00009, 0.00371] |
| 4–5 bits | 4^-b | 1.31465 [0.91367, 1.74242] |
| 4–5 bits | Actual step² | 0.81174 [0.46510, 1.16437] |
| 4–5 bits | Shared η | -0.01743 [-0.09142, 0.05878] |
| Low bits (b≤3) | 4^-b | -7.09209 [-10.42969, -3.46880] |
| Low bits (b≤3) | Actual step² | -6.51452 [-9.98177, -2.80166] |
| Low bits (b≤3) | Shared η | -6.08575 [-9.68119, -2.24822] |
| Full domain | 4^-b | -0.85960 [-1.48197, -0.16241] |
| Full domain | Actual step² | -0.96022 [-1.62529, -0.22111] |
| Full domain | Shared η | -1.22333 [-1.92326, -0.47068] |

The range-dependent η and these cross-range errors are a validity-range problem for a single exponent. A good all-bit score does not establish a universal non-collapse loss law.

## Fill the missing 5-bit cells (commands only)

Missing 5-bit records: **gemma3-270m, gemma3-4b, olmo3-7b** (3 of 7). Complete 4/5 models: gemma3-12b, gemma3-1b, gemma4-31b, muse-30b. Completes the same seven-model 4/5 comparison and reduces comparison-set variation.

These exact V10 commands are listed, **not executed**. `--n-probe 512` builds 512 probes per capability and V10 evaluates the odd-indexed half (256); it does not evaluate 512 held-out items. bf16 follows job_hpg_v10shape.slurm; saved aggregate JSON lacks dtype/item IDs. Confirm the original runtime/probes before collecting matched fills.

```bash
python analysis/v10_quantization.py --model gemma3-270m --device cuda:0 --model-dtype bf16 --n-probe 512 --bits 5 --output-base results/v10-quant-shape512-fill5
python analysis/v10_quantization.py --model gemma3-4b --device cuda:0 --model-dtype bf16 --n-probe 512 --bits 5 --output-base results/v10-quant-shape512-fill5
python analysis/v10_quantization.py --model olmo3-7b --device cuda:0 --model-dtype bf16 --n-probe 512 --bits 5 --output-base results/v10-quant-shape512-fill5
```

V10 always measures dense and overwrites its dense key on merge. Stage fills separately; check dense/probe/dtype agreement, then add only the 5-bit record to shape512 without replacing its dense/4-bit anchor. If mismatched, rerun dense/4/5 together as a separate panel.

### Sensitivity with and without the three incomplete models

The seven-model full-domain result retains all 33 observed cells. Removing the incomplete models leaves four models and 24 cells. First we hold V30 predictions fixed to isolate scoring-set changes; then we rerun LOMO on those four models to expose training-set changes. Cross-set MAE changes are descriptive; paired refit comparisons use the same 24 cells. Missing 5-bit outcomes are never imputed.

| Scoring / training set | Candidate | Full-domain MAE [95% CI] | Gain over zero [95% CI] |
|---|---|---:|---:|
| with_missing_models | Zero-change | 0.57421 [0.33869, 0.89681] | -0.00000 [-0.00000, -0.00000] |
| with_missing_models | 4^-b | 0.52288 [0.34729, 0.76918] | 0.05133 [-0.17386, 0.21163] |
| with_missing_models | Actual step² | 0.51795 [0.34053, 0.76865] | 0.05626 [-0.17310, 0.21702] |
| with_missing_models | Shared η | 0.50440 [0.31986, 0.76977] | 0.06981 [-0.17413, 0.23092] |
| without_missing_models_fixed_v30_predictions | Zero-change | 0.49451 [0.28135, 0.67745] | -0.00000 [-0.00000, -0.00000] |
| without_missing_models_fixed_v30_predictions | 4^-b | 0.40696 [0.24510, 0.54007] | 0.08756 [0.03625, 0.13738] |
| without_missing_models_fixed_v30_predictions | Actual step² | 0.39875 [0.24103, 0.53316] | 0.09576 [0.04032, 0.14429] |
| without_missing_models_fixed_v30_predictions | Shared η | 0.37622 [0.22508, 0.52224] | 0.11829 [0.06223, 0.15532] |
| without_missing_models_refitted_lomo | Zero-change | 0.49451 [0.28135, 0.67745] | -0.00000 [-0.00000, -0.00000] |
| without_missing_models_refitted_lomo | 4^-b | 0.43221 [0.37797, 0.50718] | 0.06230 [-0.09858, 0.18888] |
| without_missing_models_refitted_lomo | Actual step² | 0.42552 [0.37024, 0.50211] | 0.06899 [-0.09161, 0.19585] |
| without_missing_models_refitted_lomo | Shared η | 0.40143 [0.33917, 0.49003] | 0.09308 [-0.06755, 0.21501] |

| Same 24 cells | Candidate | Original-fit MAE − complete-model-refit MAE [95% CI] |
|---|---|---:|
| Complete models | 4^-b | -0.02525 [-0.14670, 0.05798] |
| Complete models | Actual step² | -0.02677 [-0.14372, 0.05746] |
| Complete models | Shared η | -0.02521 [-0.12978, 0.07183] |

| 5-bit scoring set: same four models / 12 cells | Candidate | MAE [95% CI] | Gain over zero [95% CI] |
|---|---|---:|---:|
| all_seven_training_models | Zero-change | 0.12956 [0.05886, 0.20026] | -0.00000 [-0.00000, -0.00000] |
| all_seven_training_models | 4^-b | 0.15268 [0.07098, 0.23437] | -0.02312 [-0.06815, 0.01093] |
| all_seven_training_models | Actual step² | 0.13678 [0.06081, 0.21276] | -0.00723 [-0.03648, 0.01676] |
| all_seven_training_models | Shared η | 0.09255 [0.02828, 0.17572] | 0.03701 [0.01574, 0.06461] |
| four_complete_training_models | Zero-change | 0.12956 [0.05886, 0.20026] | -0.00000 [-0.00000, -0.00000] |
| four_complete_training_models | 4^-b | 0.18641 [0.12926, 0.24357] | -0.05685 [-0.08291, -0.01726] |
| four_complete_training_models | Actual step² | 0.17187 [0.10929, 0.23444] | -0.04231 [-0.07003, -0.00646] |
| four_complete_training_models | Shared η | 0.12290 [0.03571, 0.21009] | 0.00666 [-0.02127, 0.03111] |

## Qwen3-14B discrimination pre-check

**Verdict: INCONCLUSIVE_FOR_ACTUAL_TARGET.** Obtain matching dense-only losses and a measurement-precision estimate before committing to a shape-discrimination run. The all-bit predictor gaps are not uniformly near zero in dev-reference scenarios; local 4/5 shape discrimination can still be weak. Do not treat all-bit amplitude-map disagreement as clean shape confirmation.

MISSING: numeric table is a dev-Qwen median-reference scenario, not measured Qwen3-14B. No Qwen3-14B quantized losses or V28 results were read. Both dev-only candidate coefficient sets below are saved in the V30b JSON; neither was calibrated on Qwen3-14B compressed outcomes.

The exact frozen predictions are supplied in JSON as `intercept + reference_loss_coefficient * L_c(dense)` for each candidate, capability and bit. The numeric default table substitutes the median dense loss of development Qwen3-0.6/1.7/4B, and also records pair-gap ranges over those anchors. These scenarios are not Qwen3-14B observations, estimates of its dense loss, or guaranteed bounds on its reference loss. No prior ability/accuracy data are converted to capability CE.

abs(DeltaL_shape512 - DeltaL_broad), matched model/capability/bit; each panel uses its own dense anchor. Includes probe-composition/protocol effects; not a measurement standard error or formal power calculation.

Gap > upper 95% model-bootstrap CI of mean absolute cross-panel dev discrepancy. Heuristic screening only, not significance/power.

Broad Qwen3-0.6B also records a dense protocol gap of −0.0003/−0.0008/+0.0133 nats (math/code/QA). This isolated discrepancy is not a variance estimate. High-bit damage and LOMO prediction errors likewise are not treated as measurement noise.

### Frozen development fit: all_bits

| Capability / bit | Reference scenario | Pred ΔL: 4^-b | Pred ΔL: step² | Pred ΔL: η | Gap 4^-b/step² | Gap 4^-b/η | Gap step²/η | Dev discrepancy [95% CI] | Screen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| math / 4 | 1.00424 | 2.16301 | 1.63140 | 0.35347 | 0.53161 | 1.80953 | 1.27793 | 0.02570 [0.01075, 0.04453] | ALL_PAIRS_ABOVE_PROXY |
| math / 5 | 1.00424 | 0.54075 | 0.35528 | 0.01371 | 0.18547 | 0.52704 | 0.34157 | 0.00463 [0.00162, 0.00743] | ALL_PAIRS_ABOVE_PROXY |
| code / 4 | 1.59109 | 1.49325 | 1.13042 | 0.24692 | 0.36283 | 1.24633 | 0.88350 | 0.02334 [0.00822, 0.04074] | ALL_PAIRS_ABOVE_PROXY |
| code / 5 | 1.59109 | 0.37331 | 0.24618 | 0.00958 | 0.12713 | 0.36373 | 0.23660 | 0.01742 [0.00308, 0.04182] | ALL_PAIRS_ABOVE_PROXY |
| qa / 4 | 6.38872 | 1.47133 | 1.11582 | 0.24463 | 0.35552 | 1.22670 | 0.87119 | 0.17623 [0.09934, 0.25297] | ALL_PAIRS_ABOVE_PROXY |
| qa / 5 | 6.38872 | 0.36783 | 0.24300 | 0.00949 | 0.12483 | 0.35835 | 0.23351 | 0.09605 [0.02096, 0.17115] | SOME_PAIRS_ABOVE_PROXY |
### Frozen development fit: bits_4_5

| Capability / bit | Reference scenario | Pred ΔL: 4^-b | Pred ΔL: step² | Pred ΔL: η | Gap 4^-b/step² | Gap 4^-b/η | Gap step²/η | Dev discrepancy [95% CI] | Screen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| math / 4 | 1.00424 | 0.45056 | 0.45575 | 0.46333 | 0.00518 | 0.01276 | 0.00758 | 0.02570 [0.01075, 0.04453] | INCONCLUSIVE_SMALL_GAPS |
| math / 5 | 1.00424 | 0.11264 | 0.09925 | 0.07296 | 0.01339 | 0.03968 | 0.02629 | 0.00463 [0.00162, 0.00743] | ALL_PAIRS_ABOVE_PROXY |
| code / 4 | 1.59109 | -0.02104 | -0.02172 | -0.02293 | 0.00068 | 0.00189 | 0.00121 | 0.02334 [0.00822, 0.04074] | INCONCLUSIVE_NEAR_ZERO |
| code / 5 | 1.59109 | -0.00526 | -0.00473 | -0.00361 | 0.00053 | 0.00165 | 0.00112 | 0.01742 [0.00308, 0.04182] | INCONCLUSIVE_NEAR_ZERO |
| qa / 4 | 6.38872 | -0.16675 | -0.16695 | -0.16643 | 0.00020 | 0.00032 | 0.00052 | 0.17623 [0.09934, 0.25297] | INCONCLUSIVE_NEAR_ZERO |
| qa / 5 | 6.38872 | -0.04169 | -0.03636 | -0.02621 | 0.00533 | 0.01548 | 0.01015 | 0.09605 [0.02096, 0.17115] | INCONCLUSIVE_NEAR_ZERO |

The all-bit candidates do not all predict near-zero damage in these scenarios: the fixed laws carry substantial all-bit-fitted amplitudes into b4/b5. That makes them potentially separable as complete predictors, but is entangled with collapse-region amplitude fitting. The local 4/5 fit is a more relevant shape screen: its b4 amplitudes nearly coincide; small code/QA b5 gaps can make those tests inconclusive. A dev-proxy screen cannot settle the actual target without its basic inputs and measurement precision. The JSON keeps each pair's verdict and range, not only the maximum gap.

## Reproduction and artifact boundary

Run `python analysis/v30b_quant_regions.py --bootstrap 10000`; add `--dry-run` for no writes. If matching target dense CE is available, pass `--target-reference-losses '{"math": ..., "code": ..., "qa": ...}'` with actual numeric values. This option accepts only three dense losses, never quantized outcomes. Tests: `python -m pytest -q tests/test_v30b.py tests/test_v30.py tests/test_prediction_audits.py`.

Only `results/v30b-quant-regions/summary.json`, its `report.md`, and `paper/docs/QUANT_REGIONS.md` are written. V28 is untouched and V30's summary is read-only. Source hashes are verified before writing; changed V10 data fail closed rather than silently changing the V30 comparison. All fits use NumPy/SciPy on CPU. No fill or Qwen3-14B quantization command is run.
