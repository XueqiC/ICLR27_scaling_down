# Quantization bit-response candidates (V30)

The broad-panel pooled all-bit LOMO winner by point MAE is **Shared η**: MAE 1.04509 versus zero-change 2.77695 nats; improvement 1.73186 [0.93392, 2.43832]. This is exploratory development-panel model holdout, not a new-source prospective validation.

At 5 bits, shared η improves pooled LOMO MAE over zero-change by 0.00954 [0.00456, 0.01452] nats in the broad panel and 0.03701 [0.01574, 0.06461] in the 512-probe panel. The 512-probe shared exponent is 3.46686 [2.97976, 4.41138]. Its paired tests reject both fixed ratios for math, code, pooled; joint rejection remains unresolved for qa. The 512-probe pairwise MAE comparison does not resolve shared η versus actual-step, despite the point ranking and rejection of the fixed mean ratio restriction.

The endpoint is capability **LOSS**, ΔL_c(b)=L_c(b)−L_c(b_ref), in native-token CE nats. Negative changes are retained, including QA improvements. No ratios with observed near-zero denominators, logarithms of observed damage, cliff filters, or sign censoring enter any fit.

## Inputs, reference, and scope

The two saved probe panels are analyzed separately. Bits and dense anchors are never spliced between panels. Reference priority is explicit `16`, then `dense` as the uncompressed **b_ref=16** anchor, then `8` if neither is present. Every current input uses `dense`/16; an 8-bit fallback would count as one compressed reference measurement. Dense is an operational 16-bit anchor, not a claim that a separate 16-bit fake-quantization run was measured.

| Panel | Development models | Bits by model | Reference |
|---|---|---|---|
| broad | qwen3-0.6b | 3, 4, 5, 6, 8 | dense → 16 |
| broad | qwen3-1.7b | 3, 4, 5, 6, 8 | dense → 16 |
| broad | qwen3-4b | 3, 4, 5, 6, 8 | dense → 16 |
| broad | gemma3-12b | 3, 4, 5, 6, 8 | dense → 16 |
| broad | gemma3-1b | 3, 4, 5, 6, 8 | dense → 16 |
| broad | gemma3-270m | 3, 4, 5, 6, 8 | dense → 16 |
| broad | gemma3-27b | 3, 4, 5, 6, 8 | dense → 16 |
| broad | gemma3-4b | 3, 4, 5, 6, 8 | dense → 16 |
| broad | gemma4-31b | 3, 4, 5, 6, 8 | dense → 16 |
| broad | muse-30b | 3, 4, 5, 6, 8 | dense → 16 |
| broad | olmo3-32b | 3, 4, 5, 6, 8 | dense → 16 |
| broad | olmo3-7b | 3, 4, 5, 6, 8 | dense → 16 |
| shape512 | gemma3-12b | 4, 5 | dense → 16 |
| shape512 | gemma3-1b | 4, 5 | dense → 16 |
| shape512 | gemma3-270m | 4 | dense → 16 |
| shape512 | gemma3-4b | 4 | dense → 16 |
| shape512 | gemma4-31b | 4, 5 | dense → 16 |
| shape512 | muse-30b | 4, 5 | dense → 16 |
| shape512 | olmo3-7b | 4 | dense → 16 |

`shape512` is the supplied 512-probe high-precision panel. Only four of its seven models currently have a 5-bit record; the other three contribute their 4-bit cells to all-bit LOMO, but cannot inform an exponent or the paired 4/5 test. The available aggregate JSON cannot independently establish item identities, numeric evaluation dtype, or item/seed uncertainty. The broader panel retains Qwen3-0.6B's recorded 5-bit protocol discrepancy (dense-gap QA +0.0133 nats); no correction is invented.

## Shapes and estimation

For ΔL̂_c(b)=a_c(x0)g(b), the candidates are:

1. **4^-b:** g(b)=4^-b−4^-b_ref.
2. **Actual step²:** g(b)=(2^(b−1)−1)^−2−(2^(b_ref−1)−1)^−2.
3. **Shared η:** g(b)=2^(−η(b−4))−2^(−η(b_ref−4)); one η across models, capabilities, and families within the quantization panel/fold.

The verified quantizer uses symmetric per-output-channel round-to-nearest, fixed clip at max_abs, and scale=max_abs/(2^(b−1)−1). Thus its derived 5/4 step² ratio is (7/15)²=0.217777778, not 0.25. With the finite 16-bit subtraction, the ratios are 0.217777742 and 0.249999955. η=2 gives exactly the 4^-b shape up to an amplitude scale; η≈2.20 matches the actual-step **local 4→5 slope**, not its entire bit curve. Error-power scaling alone does not prove an identical capability-loss law.

For each shape, development curves yield signed least-squares amplitudes; g is normalized by g(4) for conditioning, so the label equals fitted ΔL(4). A capability-specific ridge mapping (fixed λ=1, unpenalized intercept) predicts that label from log nominal model size in billions parsed from the model identifier, measured reference L_c, and centered family indicators. Nominal size is not a measured non-embedding parameter count. Features are standardized on training models only; unseen families receive zero family correction. The same mapping and penalty apply to all candidates and are not selected using holdout errors.

Shared η minimizes original loss-space SSE after profiling the development model/capability amplitudes, over [0.05, 8], using a grid and bounded refinement. This is one fitted η, never a separate exponent per capability/model/family. Single-bit curves have zero profile information. Every LOMO fold excludes all capabilities and bits of the held-out model before refitting η, amplitude labels, preprocessing, and ridge coefficients. Its target inputs are only basic metadata and the reference loss: **0 compressed calibration points** for these dense-reference inputs. The full-development η estimates below are descriptive; they are never substituted into LOMO folds.

## Paired 4/5-bit differences

D_k=ΔL(5)−kΔL(4). Each 95% interval resamples whole models, preserving the 4/5 pair, reference, candidates, and all capabilities together. A CI excluding zero rejects the fixed ratio's population mean paired-difference restriction on this panel. A CI containing zero does not establish individual-model fit. Requested k=0.250 and 0.218 are shown, with the exact finite-reference step² ratio as a rounding sensitivity.

The improvement column additionally scores kΔL(4) as a prediction of ΔL(5). It spends **one own-model 4-bit calibration point**; it is distinct from the zero-calibration LOMO comparison. Observed response amplitude means mean |ΔL|, not the unstable median ratio.

| Panel / capability (models) | k | Mean D_k [95% CI] | Observed amplitude: mean abs(ΔL5) / abs(ΔL4) | Improvement over zero-change [95% CI] | Calibration points used | Candidate shape excluded |
|---|---:|---:|---:|---:|---:|---|
| broad / math (12) | 0.250 | -0.07828 [-0.13065, -0.03692] | 0.09263 / 0.68362 | 0.01306 [-0.05632, 0.09278] | 1 | 4^-b ratio |
| broad / math (12) | 0.218 | -0.05640 [-0.10200, -0.02074] | 0.09263 / 0.68362 | 0.03423 [-0.03551, 0.12058] | 1 | step² ratio |
| broad / math (12) | exact step² | -0.05625 [-0.10181, -0.02058] | 0.09263 / 0.68362 | 0.03438 [-0.03534, 0.12078] | 1 | step² ratio |
| broad / code (12) | 0.250 | -0.05706 [-0.09439, -0.02600] | 0.11025 / 0.66572 | 0.04780 [-0.00489, 0.10995] | 1 | 4^-b ratio |
| broad / code (12) | 0.218 | -0.03575 [-0.06148, -0.01246] | 0.11025 / 0.66572 | 0.06507 [0.00430, 0.13760] | 1 | step² ratio |
| broad / code (12) | exact step² | -0.03561 [-0.06124, -0.01235] | 0.11025 / 0.66572 | 0.06519 [0.00437, 0.13772] | 1 | step² ratio |
| broad / qa (12) | 0.250 | -0.08612 [-0.14323, -0.01765] | 0.12229 / 0.45171 | 0.00216 [-0.06036, 0.06198] | 1 | 4^-b ratio |
| broad / qa (12) | 0.218 | -0.08098 [-0.13533, -0.01684] | 0.12229 / 0.45171 | 0.00956 [-0.04885, 0.06701] | 1 | step² ratio |
| broad / qa (12) | exact step² | -0.08094 [-0.13531, -0.01681] | 0.12229 / 0.45171 | 0.00961 [-0.04880, 0.06708] | 1 | step² ratio |
| broad / pooled (12) | 0.250 | -0.07382 [-0.11137, -0.03733] | 0.10839 / 0.60035 | 0.02101 [-0.00026, 0.04563] | 1 | 4^-b ratio |
| broad / pooled (12) | 0.218 | -0.05771 [-0.08754, -0.02765] | 0.10839 / 0.60035 | 0.03629 [0.01024, 0.06896] | 1 | step² ratio |
| broad / pooled (12) | exact step² | -0.05760 [-0.08740, -0.02760] | 0.10839 / 0.60035 | 0.03639 [0.01030, 0.06913] | 1 | step² ratio |
| shape512 / math (4) | 0.250 | -0.14563 [-0.25103, -0.04023] | 0.09804 / 0.97471 | -0.04759 [-0.18349, 0.08832] | 1 | 4^-b ratio |
| shape512 / math (4) | 0.218 | -0.11444 [-0.21026, -0.01863] | 0.09804 / 0.97471 | -0.01640 [-0.14271, 0.10992] | 1 | step² ratio |
| shape512 / math (4) | exact step² | -0.11422 [-0.20997, -0.01848] | 0.09804 / 0.97471 | -0.01618 [-0.14243, 0.11007] | 1 | step² ratio |
| shape512 / code (4) | 0.250 | -0.07444 [-0.11864, -0.03024] | 0.14910 / 0.89417 | 0.07466 [-0.02873, 0.17804] | 1 | 4^-b ratio |
| shape512 / code (4) | 0.218 | -0.04583 [-0.07681, -0.01416] | 0.14910 / 0.89417 | 0.10087 [-0.01353, 0.22058] | 1 | step² ratio |
| shape512 / code (4) | exact step² | -0.04563 [-0.07652, -0.01395] | 0.14910 / 0.89417 | 0.10100 [-0.01342, 0.22093] | 1 | step² ratio |
| shape512 / qa (4) | 0.250 | -0.22310 [-0.36771, 0.01558] | 0.14153 / 0.70952 | -0.14760 [-0.24441, -0.05078] | 1 | none resolved |
| shape512 / qa (4) | 0.218 | -0.20356 [-0.34289, 0.02086] | 0.14153 / 0.70952 | -0.12489 [-0.20550, -0.04428] | 1 | none resolved |
| shape512 / qa (4) | exact step² | -0.20342 [-0.34266, 0.02084] | 0.14153 / 0.70952 | -0.12473 [-0.20523, -0.04424] | 1 | none resolved |
| shape512 / pooled (4) | 0.250 | -0.14772 [-0.22894, -0.03936] | 0.12956 / 0.85946 | -0.04018 [-0.12050, 0.00933] | 1 | 4^-b ratio |
| shape512 / pooled (4) | 0.218 | -0.12128 [-0.19190, -0.02648] | 0.12956 / 0.85946 | -0.01347 [-0.08662, 0.03329] | 1 | step² ratio |
| shape512 / pooled (4) | exact step² | -0.12109 [-0.19165, -0.02639] | 0.12956 / 0.85946 | -0.01330 [-0.08638, 0.03347] | 1 | step² ratio |

## Shared effective exponent

η uncertainty uses whole-model bootstrap **refits**, including all capabilities and available bits per sampled model. Duplicate draws weight the complete model profile SSE. Bounds are fixed before fitting; unidentified draws (only one-bit models sampled) are counted and omitted from the conditional percentile interval. Loss-space SSE weights large loss cliffs strongly.

| Fit | η [95% CI] | Implied 5/4 ratio [95% CI] | Observed response amplitude | Improvement over zero-change | Calibration points used | Candidate shape excluded |
|---|---:|---:|---:|---|---|---|
| broad, all available bits | 4.68833 [4.06130, 5.93865] | 0.03879 [0.01630, 0.05990] | 2.77695 | N/A: descriptive fit | dev 5 bits/curve; target N/A | 4^-b (η=2); actual-step local approximation (η=2.20) |
| shape512, all available bits | 3.46686 [2.97976, 4.41138] | 0.09044 [0.04699, 0.12677] | 0.57421 | N/A: descriptive fit | dev 1,2 bits/curve; target N/A | 4^-b (η=2); actual-step local approximation (η=2.20) |
| broad, 4/5 only (local sensitivity) | 2.66690 [2.52100, 2.93795] | 0.15746 [0.13049, 0.17422] | 0.35437 | N/A: descriptive fit | dev 2 bits/curve; target N/A | 4^-b (η=2); actual-step local approximation (η=2.20) |

- broad, all available bits: the interval lies above 2.20, supporting faster effective decay than both local fixed-law slopes; 12 informative models, 0/10000 unidentified bootstrap draws and 0 boundary draws (point fit at boundary: False).

- shape512, all available bits: the interval lies above 2.20, supporting faster effective decay than both local fixed-law slopes; 4 informative models, 21/10000 unidentified bootstrap draws and 0 boundary draws (point fit at boundary: False).

- broad, 4/5 only (local sensitivity): the interval lies above 2.20, supporting faster effective decay than both local fixed-law slopes; 12 informative models, 0/10000 unidentified bootstrap draws and 0 boundary draws (point fit at boundary: False).

An exponent above 2.2 means damage magnitude decays faster with increasing bits than either fixed law's local 4→5 prediction. A positive ratio of about 0.13 corresponds to η=−log2(0.13)≈2.94; that is an illustrative conversion, not a fabricated panel estimate. The all-bit fit includes the 3-bit cliffs, so its effective exponent need not equal the 4/5-only value. The latter is a separately labeled sensitivity and does not replace the specified all-bit candidate in LOMO. Excluding η=2.20 in a constant-exponent model is not an exact global rejection of the non-exponential actual-step curve; use the direct paired test and held-out actual-step comparison as well.

## Held-out model prediction

Every candidate and zero-change score exactly the same held-out cells. All-bit MAE includes 3/4/5/6/8 where available; the 5-bit view isolates the small-response endpoint using those same all-bit-trained folds. Each cell has equal weight. The 512 panel has unequal bit coverage, so its all-bit estimate is a cell-weighted available-data estimate.

Intervals use 10,000 paired model-bootstrap draws. MAE/gain intervals hold OOF predictions fixed, per prediction_audit.py; they describe panel variation, not retraining, seed, or shared-item uncertainty. Pooled resampling keeps all capabilities of each model together. Gains are MAE(zero-change)−MAE(candidate), positive favoring the candidate. Zero-change is the user-specified strongest simple baseline; this analysis does not reselect among additional simple baselines.

The last column identifies **candidate predictors** with significantly worse paired MAE than the row predictor (95% CI for other−row above zero). It does not exclude a mathematical shape independent of this amplitude mapping. All intervals are exploratory, marginal, and unadjusted for multiple comparisons. A point winner alone is not a resolved win.

### all_bits

| Panel / capability (models; cells) | Candidate | LOMO MAE [95% CI] | Observed response amplitude [95% CI] | Improvement over zero-change [95% CI] | Calibration points used | Candidate predictors excluded |
|---|---|---:|---:|---:|---:|---|
| broad / math (12; 60) | 4^-b | 1.62367 [1.24775, 2.00579] | 2.95593 [2.06648, 3.80993] | 1.33226 [0.52830, 2.00699] | 0 | none resolved |
| broad / math (12; 60) | Actual step² | 1.38213 [1.03178, 1.74448] | 2.95593 [2.06648, 3.80993] | 1.57381 [0.72305, 2.30228] | 0 | 4^-b |
| broad / math (12; 60) | Shared η | 1.00165 [0.62588, 1.42934] | 2.95593 [2.06648, 3.80993] | 1.95428 [1.07496, 2.72597] | 0 | 4^-b, Actual step² |
| broad / code (12; 60) | 4^-b | 1.55694 [1.13794, 2.00355] | 2.99878 [2.12702, 3.85924] | 1.44184 [0.68600, 2.09403] | 0 | none resolved |
| broad / code (12; 60) | Actual step² | 1.33097 [0.93795, 1.74688] | 2.99878 [2.12702, 3.85924] | 1.66781 [0.87011, 2.36574] | 0 | 4^-b |
| broad / code (12; 60) | Shared η | 1.00275 [0.62872, 1.44944] | 2.99878 [2.12702, 3.85924] | 1.99604 [1.17567, 2.70886] | 0 | 4^-b, Actual step² |
| broad / qa (12; 60) | 4^-b | 1.62984 [1.19196, 2.13163] | 2.37613 [1.55250, 3.21139] | 0.74629 [0.09644, 1.33320] | 0 | none resolved |
| broad / qa (12; 60) | Actual step² | 1.45504 [1.04302, 1.93139] | 2.37613 [1.55250, 3.21139] | 0.92108 [0.23683, 1.54269] | 0 | 4^-b |
| broad / qa (12; 60) | Shared η | 1.13086 [0.72888, 1.60026] | 2.37613 [1.55250, 3.21139] | 1.24527 [0.52705, 1.89859] | 0 | 4^-b, Actual step² |
| broad / pooled (12; 180) | 4^-b | 1.60348 [1.21593, 2.02690] | 2.77695 [1.92340, 3.62373] | 1.17346 [0.45346, 1.79786] | 0 | none resolved |
| broad / pooled (12; 180) | Actual step² | 1.38938 [1.02896, 1.78322] | 2.77695 [1.92340, 3.62373] | 1.38757 [0.62398, 2.05772] | 0 | 4^-b |
| broad / pooled (12; 180) | Shared η | 1.04509 [0.67770, 1.48344] | 2.77695 [1.92340, 3.62373] | 1.73186 [0.93392, 2.43832] | 0 | 4^-b, Actual step² |
| shape512 / math (7; 11) | 4^-b | 0.45328 [0.28730, 0.67827] | 0.64650 [0.38054, 1.03216] | 0.19322 [-0.01871, 0.41793] | 0 | none resolved |
| shape512 / math (7; 11) | Actual step² | 0.44427 [0.27955, 0.67237] | 0.64650 [0.38054, 1.03216] | 0.20223 [-0.01398, 0.42495] | 0 | 4^-b |
| shape512 / math (7; 11) | Shared η | 0.42439 [0.25317, 0.66520] | 0.64650 [0.38054, 1.03216] | 0.22211 [0.00656, 0.43773] | 0 | none resolved |
| shape512 / code (7; 11) | 4^-b | 0.65573 [0.30417, 1.14027] | 0.65695 [0.29084, 1.18098] | 0.00121 [-0.33598, 0.23276] | 0 | none resolved |
| shape512 / code (7; 11) | Actual step² | 0.65080 [0.30066, 1.14258] | 0.65695 [0.29084, 1.18098] | 0.00614 [-0.33195, 0.23436] | 0 | none resolved |
| shape512 / code (7; 11) | Shared η | 0.63536 [0.28088, 1.13573] | 0.65695 [0.29084, 1.18098] | 0.02159 [-0.31917, 0.23935] | 0 | none resolved |
| shape512 / qa (7; 11) | 4^-b | 0.45962 [0.26404, 0.71393] | 0.41917 [0.22916, 0.62274] | -0.04044 [-0.23859, 0.10409] | 0 | none resolved |
| shape512 / qa (7; 11) | Actual step² | 0.45877 [0.26123, 0.71245] | 0.41917 [0.22916, 0.62274] | -0.03960 [-0.24323, 0.10922] | 0 | none resolved |
| shape512 / qa (7; 11) | Shared η | 0.45345 [0.25004, 0.71065] | 0.41917 [0.22916, 0.62274] | -0.03428 [-0.25830, 0.13382] | 0 | none resolved |
| shape512 / pooled (7; 33) | 4^-b | 0.52288 [0.34729, 0.76918] | 0.57421 [0.33869, 0.89681] | 0.05133 [-0.17386, 0.21163] | 0 | none resolved |
| shape512 / pooled (7; 33) | Actual step² | 0.51795 [0.34053, 0.76865] | 0.57421 [0.33869, 0.89681] | 0.05626 [-0.17310, 0.21702] | 0 | none resolved |
| shape512 / pooled (7; 33) | Shared η | 0.50440 [0.31986, 0.76977] | 0.57421 [0.33869, 0.89681] | 0.06981 [-0.17413, 0.23092] | 0 | none resolved |

Point winners and gains over zero-change (a negative gain means the baseline wins):

- broad/math: Shared η, gain 1.95428 [1.07496, 2.72597]; gain resolved.
- broad/code: Shared η, gain 1.99604 [1.17567, 2.70886]; gain resolved.
- broad/qa: Shared η, gain 1.24527 [0.52705, 1.89859]; gain resolved.
- broad/pooled: Shared η, gain 1.73186 [0.93392, 2.43832]; gain resolved.
- shape512/math: Shared η, gain 0.22211 [0.00656, 0.43773]; gain resolved.
- shape512/code: Shared η, gain 0.02159 [-0.31917, 0.23935]; gain unresolved.
- shape512/qa: Shared η, gain -0.03428 [-0.25830, 0.13382]; gain unresolved.
- shape512/pooled: Shared η, gain 0.06981 [-0.17413, 0.23092]; gain unresolved.

### 5bit

| Panel / capability (models; cells) | Candidate | LOMO MAE [95% CI] | Observed response amplitude [95% CI] | Improvement over zero-change [95% CI] | Calibration points used | Candidate predictors excluded |
|---|---|---:|---:|---:|---:|---|
| broad / math (12; 12) | 4^-b | 0.73097 [0.60756, 0.86538] | 0.09263 [0.04229, 0.16398] | -0.63834 [-0.77419, -0.49271] | 0 | none resolved |
| broad / math (12; 12) | Actual step² | 0.44877 [0.36357, 0.53609] | 0.09263 [0.04229, 0.16398] | -0.35614 [-0.45669, -0.22585] | 0 | 4^-b |
| broad / math (12; 12) | Shared η | 0.07259 [0.02414, 0.14438] | 0.09263 [0.04229, 0.16398] | 0.02004 [0.01410, 0.02591] | 0 | 4^-b, Actual step² |
| broad / code (12; 12) | 4^-b | 0.71357 [0.55572, 0.88270] | 0.11025 [0.04281, 0.19156] | -0.60332 [-0.79876, -0.40073] | 0 | none resolved |
| broad / code (12; 12) | Actual step² | 0.43174 [0.31542, 0.55021] | 0.11025 [0.04281, 0.19156] | -0.32149 [-0.47934, -0.14485] | 0 | 4^-b |
| broad / code (12; 12) | Shared η | 0.09507 [0.02970, 0.17649] | 0.11025 [0.04281, 0.19156] | 0.01518 [0.00626, 0.02334] | 0 | 4^-b, Actual step² |
| broad / qa (12; 12) | 4^-b | 0.70126 [0.54157, 0.86580] | 0.12229 [0.07842, 0.16850] | -0.57897 [-0.74597, -0.41564] | 0 | none resolved |
| broad / qa (12; 12) | Actual step² | 0.47789 [0.35339, 0.60096] | 0.12229 [0.07842, 0.16850] | -0.35560 [-0.47964, -0.22745] | 0 | 4^-b |
| broad / qa (12; 12) | Shared η | 0.12887 [0.08395, 0.17750] | 0.12229 [0.07842, 0.16850] | -0.00659 [-0.01622, 0.00361] | 0 | 4^-b, Actual step² |
| broad / pooled (12; 36) | 4^-b | 0.71527 [0.58063, 0.86282] | 0.10839 [0.06888, 0.15349] | -0.60688 [-0.75876, -0.46732] | 0 | none resolved |
| broad / pooled (12; 36) | Actual step² | 0.45280 [0.36122, 0.55349] | 0.10839 [0.06888, 0.15349] | -0.34441 [-0.45541, -0.23566] | 0 | 4^-b |
| broad / pooled (12; 36) | Shared η | 0.09885 [0.06064, 0.14335] | 0.10839 [0.06888, 0.15349] | 0.00954 [0.00456, 0.01452] | 0 | 4^-b, Actual step² |
| shape512 / math (4; 4) | 4^-b | 0.10738 [0.04856, 0.17944] | 0.09804 [0.05890, 0.15748] | -0.00933 [-0.03882, 0.03513] | 0 | none resolved |
| shape512 / math (4; 4) | Actual step² | 0.08208 [0.03451, 0.13738] | 0.09804 [0.05890, 0.15748] | 0.01597 [-0.01387, 0.05120] | 0 | 4^-b |
| shape512 / math (4; 4) | Shared η | 0.02434 [0.00310, 0.04657] | 0.09804 [0.05890, 0.15748] | 0.07370 [0.03478, 0.12800] | 0 | 4^-b |
| shape512 / code (4; 4) | 4^-b | 0.15627 [0.05091, 0.26164] | 0.14910 [0.04501, 0.28609] | -0.00717 [-0.06266, 0.04831] | 0 | none resolved |
| shape512 / code (4; 4) | Actual step² | 0.14083 [0.03972, 0.24704] | 0.14910 [0.04501, 0.28609] | 0.00827 [-0.03195, 0.04848] | 0 | none resolved |
| shape512 / code (4; 4) | Shared η | 0.09403 [0.00391, 0.26428] | 0.14910 [0.04501, 0.28609] | 0.05507 [0.01896, 0.11369] | 0 | none resolved |
| shape512 / qa (4; 4) | 4^-b | 0.19437 [0.06941, 0.31933] | 0.14153 [0.07110, 0.21196] | -0.05284 [-0.12419, 0.00859] | 0 | none resolved |
| shape512 / qa (4; 4) | Actual step² | 0.18745 [0.06839, 0.30650] | 0.14153 [0.07110, 0.21196] | -0.04591 [-0.10851, 0.00823] | 0 | none resolved |
| shape512 / qa (4; 4) | Shared η | 0.15928 [0.06929, 0.24927] | 0.14153 [0.07110, 0.21196] | -0.01775 [-0.04862, 0.00692] | 0 | none resolved |
| shape512 / pooled (4; 12) | 4^-b | 0.15268 [0.07098, 0.23437] | 0.12956 [0.05886, 0.20026] | -0.02312 [-0.06815, 0.01093] | 0 | none resolved |
| shape512 / pooled (4; 12) | Actual step² | 0.13678 [0.06081, 0.21276] | 0.12956 [0.05886, 0.20026] | -0.00723 [-0.03648, 0.01676] | 0 | 4^-b |
| shape512 / pooled (4; 12) | Shared η | 0.09255 [0.02828, 0.17572] | 0.12956 [0.05886, 0.20026] | 0.03701 [0.01574, 0.06461] | 0 | 4^-b |

Point winners and gains over zero-change (a negative gain means the baseline wins):

- broad/math: Shared η, gain 0.02004 [0.01410, 0.02591]; gain resolved.
- broad/code: Shared η, gain 0.01518 [0.00626, 0.02334]; gain resolved.
- broad/qa: Shared η, gain -0.00659 [-0.01622, 0.00361]; gain unresolved.
- broad/pooled: Shared η, gain 0.00954 [0.00456, 0.01452]; gain resolved.
- shape512/math: Shared η, gain 0.07370 [0.03478, 0.12800]; gain resolved.
- shape512/code: Shared η, gain 0.05507 [0.01896, 0.11369]; gain resolved.
- shape512/qa: Shared η, gain -0.01775 [-0.04862, 0.00692]; gain unresolved.
- shape512/pooled: Shared η, gain 0.03701 [0.01574, 0.06461]; gain resolved.

Observed response amplitude equals zero-change MAE exactly. The JSON also records signed means, every pairwise candidate-MAE interval, and best-candidate gain intervals with the winner reselected within each bootstrap draw; named-winner intervals in the report are conditional on that candidate and are not selection-adjusted. Cross-model native-token nats and nominal size/family features limit cross-tokenizer and mechanistic interpretation. The four paired 512 models are a small available subset, not a random missingness guarantee.

## Prospective boundary and reproduction

**The frozen Qwen3-8B prospective result in `results/v28-new-source-pred` stays as-is.** This script never reads its predictions, never reads Qwen3-8B quantization loss JSON, and never writes to that directory. It does not revise the V28 method or retrospective result. Qwen3-8B is not presented as independent validation of these newly explored V30 candidates. **Qwen3-14B is reserved for the next frozen new-source test**; no Qwen3-14B numbers are loaded, fitted, or fabricated. Every model used here is development data for these candidates and cannot later validate them as an independent new source. Separate-panel fits and LOMO do not restore prospective independence to already-seen models.

Run `python analysis/v30_quant_shape_candidates.py --bootstrap 10000`; add `--dry-run` to compute without output writes. Tests: `python -m pytest -q tests/test_v30.py`. Only V30 outputs are written: `results/v30-quant-candidates/summary.json`, `predictions.csv`, `report.md`, and this report. Summary JSON contains input SHA-256 hashes, cell predictions, fold training membership/coefficients/η, model coverage, all comparisons, and η objective profiles. Source hashes are checked again immediately before output writes.
