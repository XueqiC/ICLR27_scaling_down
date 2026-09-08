# Unified information budgets: K0, K1 and Oracle (V35)

**K0 is the main basic-input prediction goal. K1 predicts the remaining configurations after one calibration. Oracle is a post-hoc diagnostic.** This CPU-only analysis imports saved coefficients and existing losses; it performs no model runs or development refits.

capability LOSS: Delta L_c = compressed/post-training L_c minus own dense L_c, native-token CE nats. Each arm keeps its own dense reference; distillation uses the student's dense loss. Native-token losses are not tokenizer-invariant, and this report does not pool arms into one MAE.

At the declared descriptive 0.1-nat adequacy tolerance, the arms show different failure signatures:

- **Pruning:** all three development capabilities retain appreciable Oracle form error. The Qwen math/code curves have much smaller post-hoc Oracle errors than K0, implicating source mapping; QA also has residual form error, especially Qwen14. Density-0.9 calibration worsens math and QA. A good Oracle fit is compatibility after seeing the curve, not independent shape prediction.
- **Quantization:** K0 has useful collapse-region predictions; 4-bit K1 worsens all three development capabilities and all Qwen14 capabilities. Oracle's near-zero int3 error is a consequence of its all-target fit: the largest shape weight makes the L1 scalar fit that point exactly. Full-curve errors expose the remaining 4-bit/QA mismatch. The prospective shape test remains inconclusive.
- **Distillation:** development math/code K0 is accurate and D75 calibration worsens it; QA retains Oracle form error. On Qwen4, calibration repairs much of the math/QA source shift but makes code worse. The two-budget all-target Oracle still misses code/QA at the stated tolerance; its code remainder error is zero because it fits D600 exactly. Neither that zero nor the trivial one-test-point lower bound establishes shape transfer.

## Information and scoring contract

- **pruning: K1 coordinate 0.9**. V28 pre-specified dev density 0.9, unchanged; no target-based tuning.
- **quantization: K1 coordinate 4**. V30 dev paired 4-to-5 design's 4-bit anchor, unchanged; shared-eta shape and mapping stay frozen.
- **distillation: K1 coordinate 75**. Smallest positive budget in fixed V25 dev grid: min(75,150,300,600)=75. Cost rule uses coordinates only, no outcomes. V35 replaces V28's D150 anchor for ALL students, without changing any frozen coefficient.

The same coordinate is used for every held-out model and capability in an arm. These rules use development coordinates/designs only. They are specified for this retrospective audit, not claimed to predate the existing target runs.

K0/K1/Oracle/zero share identical remainder cells. The calibration point is excluded from every primary score. Full-curve diagnostics include it and therefore K1 full-curve scores are not predictive.

DIAGNOSTIC only: independent signed scalar per target/capability, all-target MAE optimum with frozen shape; includes calibration in its fit. Best possible performance (lower error bound) on the all-target objective, not a predictor.

All-target optimum need not bound remainder MAE. Both all-target diagnostic MAEs and each target's remainder-only scalar lower bound are saved; the latter is trivial with one test coordinate.

Pruning V28 .9/.8/.7/.6 (score .8/.7/.6); quant all available V30 broad bits 3/4/5/6/8 (score excludes 4); distill V25 LoRA budgets (score excludes 75), Qwen4B seed0 D75/600. No imputed Qwen14 5-bit or Qwen4 D150/300.

abs(delta)<= 0.01 nat is flagged, signed and retained; exact zero is retained. No division by observed response; no clipping/sign filtering. abs(shape(cal))<=1e-12 explicitly falls back to K0 after consuming the one point.

No collapse censoring. Pruning/distill delta>1 nat flagged; quant b<=3 is a coordinate-defined region (including OLMo exceptions). Flags never affect scores, calibration or fitting.

Paired whole-model percentile bootstrap via prediction_audit.compare_predictions; all candidates and all configurations for a sampled model stay together. Equal observed-cell weights, including unequal distillation grids. Fixed fits/scalars; no refitting, item/seed/measurement uncertainty or multiplicity adjustment. Separate prospective model rows have degenerate CIs, n=1.

Oracle optimizes MAE, not squared error: its scalar is the weighted median of ΔL/x with weights |x|; a midpoint resolves median ties. It replaces the signed amplitude of the frozen shape (equivalently rescales a nonzero K0 curve). A zero K0 amplitude does not disable the shape's diagnostic amplitude. No target gamma, eta, offset or curvature is fitted.

## Primary remainder scores

All values are nats with paired-model 95% CIs. Gains are zero-change MAE minus method MAE; positive helps. Each prospective row is separate (n=1, degenerate CI). Development rows use saved model-held-out fits.

| Arm / cohort | Capability | Models / cells | Zero MAE | K0 MAE [95% CI] | K1 MAE [95% CI] | Oracle MAE [95% CI] |
|---|---|---:|---:|---|---|---|
| pruning / dev LOMO | math | 12 / 36 | 1.56887 [0.73220, 2.51960] | 1.41473 [0.84500, 2.06268] | 9.29039 [2.09719, 21.74722] | 0.38180 [0.07256, 0.77804] |
| pruning / dev LOMO | code | 12 / 36 | 1.57652 [0.72986, 2.54230] | 1.29818 [0.71641, 1.92430] | 1.80860 [0.87896, 3.09608] | 0.35012 [0.11168, 0.64942] |
| pruning / dev LOMO | qa | 12 / 36 | 1.69267 [0.64673, 2.87452] | 1.91500 [1.19195, 2.77552] | 34.49935 [15.84184, 55.88683] | 0.28007 [0.15940, 0.40947] |
| pruning / prospective Qwen3-14B | math | 1 / 3 | 0.06886 [0.06886, 0.06886] | 0.76743 [0.76743, 0.76743] | 2.73184 [2.73184, 2.73184] | 0.01432 [0.01432, 0.01432] |
| pruning / prospective Qwen3-14B | code | 1 / 3 | 0.10533 [0.10533, 0.10533] | 0.36300 [0.36300, 0.36300] | 0.03330 [0.03330, 0.03330] | 0.03290 [0.03290, 0.03290] |
| pruning / prospective Qwen3-14B | qa | 1 / 3 | 0.48525 [0.48525, 0.48525] | 0.81590 [0.81590, 0.81590] | 1.12996 [1.12996, 1.12996] | 0.14519 [0.14519, 0.14519] |
| pruning / prospective Qwen3-8B | math | 1 / 3 | 0.06378 [0.06378, 0.06378] | 0.73463 [0.73463, 0.73463] | 0.98747 [0.98747, 0.98747] | 0.00388 [0.00388, 0.00388] |
| pruning / prospective Qwen3-8B | code | 1 / 3 | 0.12701 [0.12701, 0.12701] | 0.38938 [0.38938, 0.38938] | 0.29623 [0.29623, 0.29623] | 0.00608 [0.00608, 0.00608] |
| pruning / prospective Qwen3-8B | qa | 1 / 3 | 0.53146 [0.53146, 0.53146] | 0.73577 [0.73577, 0.73577] | 22.68855 [22.68855, 22.68855] | 0.09817 [0.09817, 0.09817] |
| quantization / dev LOMO | math | 12 / 48 | 3.52401 [2.47680, 4.56274] | 1.14879 [0.70156, 1.64768] | 2.78515 [1.18050, 4.78342] | 0.02257 [0.00816, 0.04346] |
| quantization / dev LOMO | code | 12 / 48 | 3.58205 [2.54515, 4.62560] | 1.12520 [0.68215, 1.62272] | 3.43328 [1.32712, 6.13719] | 0.02807 [0.01072, 0.04984] |
| quantization / dev LOMO | qa | 12 / 48 | 2.85723 [1.86276, 3.91044] | 1.27327 [0.77420, 1.85794] | 3.28244 [1.87343, 4.77430] | 0.04560 [0.03216, 0.05929] |
| quantization / prospective Qwen3-14B | math | 1 / 3 | 3.10365 [3.10365, 3.10365] | 0.02847 [0.02847, 0.02847] | 2.32918 [2.32918, 2.32918] | 0.00243 [0.00243, 0.00243] |
| quantization / prospective Qwen3-14B | code | 1 / 3 | 3.60292 [3.60292, 3.60292] | 0.42162 [0.42162, 0.42162] | 0.90573 [0.90573, 0.90573] | 0.00627 [0.00627, 0.00627] |
| quantization / prospective Qwen3-14B | qa | 1 / 3 | 2.17259 [2.17259, 2.17259] | 0.08217 [0.08217, 0.08217] | 4.14952 [4.14952, 4.14952] | 0.02334 [0.02334, 0.02334] |
| quantization / prospective Qwen3-8B | math | 1 / 4 | 3.22654 [3.22654, 3.22654] | 0.82004 [0.82004, 0.82004] | 1.44689 [1.44689, 1.44689] | 0.00533 [0.00533, 0.00533] |
| quantization / prospective Qwen3-8B | code | 1 / 4 | 3.42340 [3.42340, 3.42340] | 0.94067 [0.94067, 0.94067] | 0.33635 [0.33635, 0.33635] | 0.03175 [0.03175, 0.03175] |
| quantization / prospective Qwen3-8B | qa | 1 / 4 | 2.45399 [2.45399, 2.45399] | 0.79464 [0.79464, 0.79464] | 7.62355 [7.62355, 7.62355] | 0.09617 [0.09617, 0.09617] |
| distillation / dev LOMO | math | 3 / 8 | 0.10534 [0.08416, 0.15166] | 0.03017 [0.01373, 0.07503] | 0.05861 [0.02908, 0.07711] | 0.01310 [0.00647, 0.01896] |
| distillation / dev LOMO | code | 3 / 8 | 0.13320 [0.11429, 0.15405] | 0.03363 [0.01581, 0.05882] | 0.07699 [0.04827, 0.10684] | 0.01473 [0.00928, 0.01686] |
| distillation / dev LOMO | qa | 3 / 8 | 1.13421 [0.99202, 1.22384] | 0.28596 [0.23867, 0.35992] | 0.27457 [0.23946, 0.32875] | 0.24676 [0.20659, 0.28747] |
| distillation / prospective Qwen3-4B | math | 1 / 1 | 0.10545 [0.10545, 0.10545] | 0.19571 [0.19571, 0.19571] | 0.06561 [0.06561, 0.06561] | 0.03281 [0.03281, 0.03281] |
| distillation / prospective Qwen3-4B | code | 1 / 1 | 0.21641 [0.21641, 0.21641] | 0.40332 [0.40332, 0.40332] | 0.89549 [0.89549, 0.89549] | 0.00000 [0.00000, 0.00000] |
| distillation / prospective Qwen3-4B | qa | 1 / 1 | 2.40571 [2.40571, 2.40571] | 1.26553 [1.26553, 1.26553] | 0.23533 [0.23533, 0.23533] | 0.11766 [0.11766, 0.11766] |

| Arm / cohort | Capability | K0 gain [95% CI] | K1 gain [95% CI] | Oracle gain [95% CI], diagnostic |
|---|---|---|---|---|
| pruning / dev LOMO | math | 0.15414 [-0.36182, 0.67097] | -7.72153 [-19.72192, -0.93283] | 1.18707 [0.60767, 1.83120] |
| pruning / dev LOMO | code | 0.27835 [-0.26963, 0.83457] | -0.23207 [-1.10361, 0.64225] | 1.22640 [0.58442, 1.95789] |
| pruning / dev LOMO | qa | -0.22234 [-0.90992, 0.48238] | -32.80668 [-53.61265, -14.56515] | 1.41259 [0.48378, 2.46648] |
| pruning / prospective Qwen3-14B | math | -0.69857 [-0.69857, -0.69857] | -2.66298 [-2.66298, -2.66298] | 0.05454 [0.05454, 0.05454] |
| pruning / prospective Qwen3-14B | code | -0.25767 [-0.25767, -0.25767] | 0.07203 [0.07203, 0.07203] | 0.07243 [0.07243, 0.07243] |
| pruning / prospective Qwen3-14B | qa | -0.33064 [-0.33064, -0.33064] | -0.64471 [-0.64471, -0.64471] | 0.34006 [0.34006, 0.34006] |
| pruning / prospective Qwen3-8B | math | -0.67085 [-0.67085, -0.67085] | -0.92369 [-0.92369, -0.92369] | 0.05991 [0.05991, 0.05991] |
| pruning / prospective Qwen3-8B | code | -0.26237 [-0.26237, -0.26237] | -0.16922 [-0.16922, -0.16922] | 0.12093 [0.12093, 0.12093] |
| pruning / prospective Qwen3-8B | qa | -0.20431 [-0.20431, -0.20431] | -22.15709 [-22.15709, -22.15709] | 0.43329 [0.43329, 0.43329] |
| quantization / dev LOMO | math | 2.37522 [1.27809, 3.32816] | 0.73886 [-0.95880, 2.16560] | 3.50145 [2.46111, 4.52751] |
| quantization / dev LOMO | code | 2.45685 [1.45203, 3.35627] | 0.14877 [-2.14764, 1.98738] | 3.55397 [2.52093, 4.59035] |
| quantization / dev LOMO | qa | 1.58396 [0.67323, 2.43779] | -0.42521 [-1.51035, 0.87691] | 2.81163 [1.82003, 3.86698] |
| quantization / prospective Qwen3-14B | math | 3.07518 [3.07518, 3.07518] | 0.77447 [0.77447, 0.77447] | 3.10122 [3.10122, 3.10122] |
| quantization / prospective Qwen3-14B | code | 3.18130 [3.18130, 3.18130] | 2.69719 [2.69719, 2.69719] | 3.59665 [3.59665, 3.59665] |
| quantization / prospective Qwen3-14B | qa | 2.09042 [2.09042, 2.09042] | -1.97693 [-1.97693, -1.97693] | 2.14925 [2.14925, 2.14925] |
| quantization / prospective Qwen3-8B | math | 2.40650 [2.40650, 2.40650] | 1.77965 [1.77965, 1.77965] | 3.22120 [3.22120, 3.22120] |
| quantization / prospective Qwen3-8B | code | 2.48273 [2.48273, 2.48273] | 3.08705 [3.08705, 3.08705] | 3.39165 [3.39165, 3.39165] |
| quantization / prospective Qwen3-8B | qa | 1.65935 [1.65935, 1.65935] | -5.16956 [-5.16956, -5.16956] | 2.35782 [2.35782, 2.35782] |
| distillation / dev LOMO | math | 0.07517 [0.07043, 0.07894] | 0.04673 [0.01853, 0.07651] | 0.09224 [0.07668, 0.13743] |
| distillation / dev LOMO | code | 0.09957 [0.07963, 0.12240] | 0.05621 [0.02848, 0.08993] | 0.11846 [0.09743, 0.14477] |
| distillation / dev LOMO | qa | 0.84824 [0.75335, 0.96707] | 0.85964 [0.75256, 0.96707] | 0.88744 [0.78543, 0.96707] |
| distillation / prospective Qwen3-4B | math | -0.09026 [-0.09026, -0.09026] | 0.03984 [0.03984, 0.03984] | 0.07264 [0.07264, 0.07264] |
| distillation / prospective Qwen3-4B | code | -0.18691 [-0.18691, -0.18691] | -0.67908 [-0.67908, -0.67908] | 0.21641 [0.21641, 0.21641] |
| distillation / prospective Qwen3-4B | qa | 1.14018 [1.14018, 1.14018] | 2.17039 [2.17039, 2.17039] | 2.28805 [2.28805, 2.28805] |

## Failure signatures and full-curve diagnostic bound

Descriptive adequacy threshold 0.1 nat fixed across arms/capabilities, not a scientifically established success cutoff. (a) all-target Oracle MAE>threshold; (b) remainder K1>K0; (c) all-target K0>threshold and Oracle<=threshold. Nonexclusive; raw errors/gaps and CI support provided.

The bound below fits AND scores all declared target configurations, including calibration; it is the true minimum error for this scalar form. CI support is descriptive for this panel. In a one-model row it cannot establish population evidence. Mapping gaps may remain even when the form also fails; the three flags are not an exhaustive causal decomposition.

| Arm / cohort | Capability | Full-curve K0 MAE | Full-curve Oracle MAE [95% CI] | Remainder K1−K0 [95% CI] | Failure signatures |
|---|---|---:|---|---|---|
| pruning / dev LOMO | math | 1.06559 | 0.29434 [0.05775, 0.60031] | 7.87567 [0.83933, 20.04873] | (a) functional form insufficient [point estimate]; (b) calibration unstable [CI supports] |
| pruning / dev LOMO | code | 0.97930 | 0.26965 [0.08731, 0.49961] | 0.51042 [-0.43968, 1.58159] | (a) functional form insufficient [point estimate]; (b) calibration unstable [point estimate] |
| pruning / dev LOMO | qa | 1.45877 | 0.23164 [0.13441, 0.33513] | 32.58434 [14.09024, 53.62373] | (a) functional form insufficient [CI supports]; (b) calibration unstable [CI supports] |
| pruning / prospective Qwen3-14B | math | 0.57730 | 0.01316 [0.01316, 0.01316] | 1.96441 [1.96441, 1.96441] | (b) calibration unstable [point estimate]; (c) source mapping insufficient [point estimate] |
| pruning / prospective Qwen3-14B | code | 0.27321 | 0.02468 [0.02468, 0.02468] | -0.32970 [-0.32970, -0.32970] | (c) source mapping insufficient [point estimate] |
| pruning / prospective Qwen3-14B | qa | 0.61214 | 0.10957 [0.10957, 0.10957] | 0.31406 [0.31406, 0.31406] | (a) functional form insufficient [point estimate]; (b) calibration unstable [point estimate] |
| pruning / prospective Qwen3-8B | math | 0.55120 | 0.00377 [0.00377, 0.00377] | 0.25284 [0.25284, 0.25284] | (b) calibration unstable [point estimate]; (c) source mapping insufficient [point estimate] |
| pruning / prospective Qwen3-8B | code | 0.29231 | 0.00544 [0.00544, 0.00544] | -0.09316 [-0.09316, -0.09316] | (c) source mapping insufficient [point estimate] |
| pruning / prospective Qwen3-8B | qa | 0.56787 | 0.08923 [0.08923, 0.08923] | 21.95278 [21.95278, 21.95278] | (b) calibration unstable [point estimate]; (c) source mapping insufficient [point estimate] |
| quantization / dev LOMO | math | 1.00165 | 0.09684 [0.04632, 0.15789] | 1.63636 [0.04003, 3.54330] | (b) calibration unstable [CI supports]; (c) source mapping insufficient [point estimate] |
| quantization / dev LOMO | code | 1.00275 | 0.11963 [0.05347, 0.20355] | 2.30808 [0.22320, 4.86271] | (a) functional form insufficient [point estimate]; (b) calibration unstable [CI supports] |
| quantization / dev LOMO | qa | 1.13086 | 0.14199 [0.08673, 0.20024] | 2.00918 [0.39666, 3.64590] | (a) functional form insufficient [point estimate]; (b) calibration unstable [CI supports] |
| quantization / prospective Qwen3-14B | math | 0.08828 | 0.06950 [0.06950, 0.06950] | 2.30071 [2.30071, 2.30071] | (b) calibration unstable [point estimate] |
| quantization / prospective Qwen3-14B | code | 0.33030 | 0.03086 [0.03086, 0.03086] | 0.48412 [0.48412, 0.48412] | (b) calibration unstable [point estimate]; (c) source mapping insufficient [point estimate] |
| quantization / prospective Qwen3-14B | qa | 0.17995 | 0.13754 [0.13754, 0.13754] | 4.06735 [4.06735, 4.06735] | (a) functional form insufficient [point estimate]; (b) calibration unstable [point estimate] |
| quantization / prospective Qwen3-8B | math | 0.67545 | 0.04893 [0.04893, 0.04893] | 0.62685 [0.62685, 0.62685] | (b) calibration unstable [point estimate]; (c) source mapping insufficient [point estimate] |
| quantization / prospective Qwen3-8B | code | 0.77125 | 0.03483 [0.03483, 0.03483] | -0.60432 [-0.60432, -0.60432] | (c) source mapping insufficient [point estimate] |
| quantization / prospective Qwen3-8B | qa | 0.84793 | 0.31087 [0.31087, 0.31087] | 6.82891 [6.82891, 6.82891] | (a) functional form insufficient [point estimate]; (b) calibration unstable [point estimate] |
| distillation / dev LOMO | math | 0.03309 | 0.02317 [0.01085, 0.03053] | 0.02844 [0.00012, 0.06041] | (b) calibration unstable [CI supports] |
| distillation / dev LOMO | code | 0.03166 | 0.01831 [0.01505, 0.02199] | 0.04336 [0.03246, 0.05114] | (b) calibration unstable [CI supports] |
| distillation / dev LOMO | qa | 0.26262 | 0.19968 [0.16397, 0.24656] | -0.01140 [-0.03117, 0.00079] | (a) functional form insufficient [CI supports] |
| distillation / prospective Qwen3-4B | math | 0.22851 | 0.03281 [0.03281, 0.03281] | -0.13009 [-0.13009, -0.13009] | (c) source mapping insufficient [point estimate] |
| distillation / prospective Qwen3-4B | code | 0.36526 | 0.11280 [0.11280, 0.11280] | 0.49217 [0.49217, 0.49217] | (a) functional form insufficient [point estimate]; (b) calibration unstable [point estimate] |
| distillation / prospective Qwen3-4B | qa | 1.38319 | 0.11766 [0.11766, 0.11766] | -1.03020 [-1.03020, -1.03020] | (a) functional form insufficient [point estimate] |

## Quantization regions and scope

INCONCLUSIVE on prospective Qwen3-14B; no shape-transfer claim. All-bit gains can be driven by int3 collapse; report bit regions separately.

4-bit is the K1 calibration point, so the remaining measurable region is 5-bit only. Qwen3-14B has no measured 5-bit point: its measurable remainder is unavailable, not zero error. Regional scores use the SAME all-bit Oracle scalar; they are not independently refitted region oracles.

| Cohort / region | Capability | Models / cells | Zero MAE | K0 MAE [95% CI] | K1 MAE [95% CI] | Oracle MAE [95% CI] |
|---|---|---:|---:|---|---|---|
| dev LOMO / high_bits | math | 12 / 24 | 0.00964 | 0.00937 [0.00432, 0.01654] | 0.00928 [0.00426, 0.01637] | 0.00935 [0.00430, 0.01649] |
| dev LOMO / measurable_bits | math | 12 / 12 | 0.09263 | 0.07259 [0.02504, 0.14318] | 0.06791 [0.02495, 0.13047] | 0.07156 [0.02348, 0.14098] |
| dev LOMO / collapse_bits | math | 12 / 12 | 13.98412 | 4.50384 [2.75997, 6.44977] | 11.05415 [4.68113, 19.02335] | 0.00000 [0.00000, 0.00000] |
| dev LOMO / high_bits | code | 12 / 24 | 0.00924 | 0.00887 [0.00544, 0.01295] | 0.00884 [0.00553, 0.01277] | 0.00887 [0.00544, 0.01294] |
| dev LOMO / measurable_bits | code | 12 / 12 | 0.11025 | 0.09507 [0.02960, 0.17884] | 0.08789 [0.03238, 0.15767] | 0.09456 [0.02958, 0.17748] |
| dev LOMO / collapse_bits | code | 12 / 12 | 14.19945 | 4.38797 [2.61933, 6.35381] | 13.62754 [5.25649, 24.36525] | 0.00000 [0.00000, 0.00000] |
| dev LOMO / high_bits | qa | 12 / 24 | 0.02929 | 0.02949 [0.02012, 0.03871] | 0.02922 [0.01983, 0.03852] | 0.02950 [0.02014, 0.03872] |
| dev LOMO / measurable_bits | qa | 12 / 12 | 0.12229 | 0.12887 [0.08317, 0.17655] | 0.11540 [0.07566, 0.15512] | 0.12341 [0.07571, 0.17252] |
| dev LOMO / collapse_bits | qa | 12 / 12 | 11.24804 | 4.90522 [2.87892, 7.26294] | 12.95592 [7.35681, 18.89020] | 0.00000 [0.00000, 0.00000] |
| prospective Qwen3-14B / high_bits | math | 1 / 2 | 0.00392 | 0.00365 [0.00365, 0.00365] | 0.00385 [0.00385, 0.00385] | 0.00364 [0.00364, 0.00364] |
| prospective Qwen3-14B / measurable_bits | math | 0 / 0 | unavailable | unavailable | unavailable | unavailable |
| prospective Qwen3-14B / collapse_bits | math | 1 / 1 | 9.30311 | 0.07812 [0.07812, 0.07812] | 6.97984 [6.97984, 6.97984] | 0.00000 [0.00000, 0.00000] |
| prospective Qwen3-14B / high_bits | code | 1 / 2 | 0.00972 | 0.00944 [0.00944, 0.00944] | 0.00948 [0.00948, 0.00948] | 0.00940 [0.00940, 0.00940] |
| prospective Qwen3-14B / measurable_bits | code | 0 / 0 | unavailable | unavailable | unavailable | unavailable |
| prospective Qwen3-14B / collapse_bits | code | 1 / 1 | 10.78933 | 1.24598 [1.24598, 1.24598] | 2.69824 [2.69824, 2.69824] | 0.00000 [0.00000, 0.00000] |
| prospective Qwen3-14B / high_bits | qa | 1 / 2 | 0.03483 | 0.03501 [0.03501, 0.03501] | 0.03466 [0.03466, 0.03466] | 0.03502 [0.03502, 0.03502] |
| prospective Qwen3-14B / measurable_bits | qa | 0 / 0 | unavailable | unavailable | unavailable | unavailable |
| prospective Qwen3-14B / collapse_bits | qa | 1 / 1 | 6.44812 | 0.17649 [0.17649, 0.17649] | 12.37926 [12.37926, 12.37926] | 0.00000 [0.00000, 0.00000] |
| prospective Qwen3-8B / high_bits | math | 1 / 2 | 0.00120 | 0.00092 [0.00092, 0.00092] | 0.00100 [0.00100, 0.00100] | 0.00083 [0.00083, 0.00083] |
| prospective Qwen3-8B / measurable_bits | math | 1 / 1 | 0.03903 | 0.02457 [0.02457, 0.02457] | 0.02833 [0.02833, 0.02833] | 0.01967 [0.01967, 0.01967] |
| prospective Qwen3-8B / collapse_bits | math | 1 / 1 | 12.86471 | 3.25374 [3.25374, 3.25374] | 5.75724 [5.75724, 5.75724] | 0.00000 [0.00000, 0.00000] |
| prospective Qwen3-8B / high_bits | code | 1 / 2 | 0.01398 | 0.01369 [0.01369, 0.01369] | 0.01362 [0.01362, 0.01362] | 0.01358 [0.01358, 0.01358] |
| prospective Qwen3-8B / measurable_bits | code | 1 / 1 | 0.12020 | 0.10528 [0.10528, 0.10528] | 0.10165 [0.10165, 0.10165] | 0.09982 [0.09982, 0.09982] |
| prospective Qwen3-8B / collapse_bits | code | 1 / 1 | 13.54543 | 3.63000 [3.63000, 3.63000] | 1.21652 [1.21652, 1.21652] | 0.00000 [0.00000, 0.00000] |
| prospective Qwen3-8B / high_bits | qa | 1 / 2 | 0.09260 | 0.09279 [0.09279, 0.09279] | 0.09199 [0.09199, 0.09199] | 0.09287 [0.09287, 0.09287] |
| prospective Qwen3-8B / measurable_bits | qa | 1 / 1 | 0.18473 | 0.19473 [0.19473, 0.19473] | 0.15358 [0.15358, 0.15358] | 0.19894 [0.19894, 0.19894] |
| prospective Qwen3-8B / collapse_bits | qa | 1 / 1 | 9.44603 | 2.79824 [2.79824, 2.79824] | 30.15664 [30.15664, 30.15664] | 0.00000 [0.00000, 0.00000] |

## Pruning range sensitivity

Every extra observed pruning density, including infills/deep collapse, is retained in a separate all-available diagnostic; same frozen gamma/map, new diagnostic scalar. These densities never change primary fits or scores.

| Capability | Models / remainder cells | K0 MAE [95% CI] | K1 MAE [95% CI] | Oracle MAE [95% CI] | All-target failure signatures |
|---|---:|---|---|---|---|
| math | 12 / 119 | 8.96794 [6.25687, 11.70503] | 126.01804 [20.78957, 317.88876] | 2.39103 [1.28257, 3.65791] | (a) functional form insufficient [CI supports]; (b) calibration unstable [CI supports] |
| code | 12 / 119 | 4.60372 [3.19963, 6.17286] | 14.62732 [5.47380, 29.75498] | 2.20651 [1.26995, 3.22233] | (a) functional form insufficient [CI supports]; (b) calibration unstable [CI supports] |
| qa | 12 / 119 | 11.62031 [8.37744, 15.22246] | 404.10211 [172.73167, 688.81726] | 2.31688 [1.37599, 3.41956] | (a) functional form insufficient [CI supports]; (b) calibration unstable [CI supports] |

## Every held-out development target

Per-target n=1 model-bootstrap CIs are degenerate: [MAE, MAE]. The compact values below therefore also specify those CIs. Prospective targets appear separately above. Gains and complete paired CIs for each target are saved in summary.json.

| Arm / target | Capability | Remainder cells | Zero MAE | K0 MAE | K1 MAE | Oracle MAE |
|---|---|---:|---:|---:|---:|---:|
| pruning / Qwen3-0.6B | math | 3 | 0.76372 | 0.22809 | 1.84681 | 0.02163 |
| pruning / Qwen3-0.6B | code | 3 | 0.92931 | 0.14183 | 0.86185 | 0.07282 |
| pruning / Qwen3-0.6B | qa | 3 | 0.14927 | 0.15777 | 48.96717 | 0.12663 |
| pruning / Qwen3-1.7B | math | 3 | 0.49133 | 0.29242 | 3.52149 | 0.02868 |
| pruning / Qwen3-1.7B | code | 3 | 0.37631 | 0.10940 | 2.35907 | 0.06680 |
| pruning / Qwen3-1.7B | qa | 3 | 0.49425 | 0.54380 | 17.88637 | 0.15586 |
| pruning / Qwen3-4B | math | 3 | 0.05100 | 1.00611 | 1.40543 | 0.01404 |
| pruning / Qwen3-4B | code | 3 | 0.03311 | 0.48297 | 1.58305 | 0.03249 |
| pruning / Qwen3-4B | qa | 3 | 0.94876 | 1.84264 | 0.22394 | 0.19871 |
| pruning / gemma3-12b | math | 3 | 1.65391 | 0.73148 | 2.50874 | 0.16015 |
| pruning / gemma3-12b | code | 3 | 0.83619 | 2.06431 | 0.48967 | 0.08476 |
| pruning / gemma3-12b | qa | 3 | 2.76960 | 1.03680 | 16.10766 | 0.50524 |
| pruning / gemma3-1b | math | 3 | 1.93091 | 0.49321 | 11.65300 | 0.14563 |
| pruning / gemma3-1b | code | 3 | 2.32177 | 0.32593 | 1.58681 | 0.31376 |
| pruning / gemma3-1b | qa | 3 | 1.38439 | 1.06641 | 72.79185 | 0.25709 |
| pruning / gemma3-270m | math | 3 | 3.65503 | 2.94042 | 74.59139 | 0.96295 |
| pruning / gemma3-270m | code | 3 | 4.36154 | 2.72367 | 7.81603 | 0.75485 |
| pruning / gemma3-270m | qa | 3 | 3.61146 | 3.94938 | 102.53943 | 0.52523 |
| pruning / gemma3-27b | math | 3 | 4.38918 | 2.97243 | 5.48999 | 1.08084 |
| pruning / gemma3-27b | code | 3 | 4.46539 | 3.31958 | 1.16905 | 1.01662 |
| pruning / gemma3-27b | qa | 3 | 4.23971 | 2.62901 | 98.45440 | 0.61900 |
| pruning / gemma3-4b | math | 3 | 1.06572 | 1.45776 | 4.73475 | 0.10903 |
| pruning / gemma3-4b | code | 3 | 1.58267 | 1.39500 | 1.43332 | 0.17556 |
| pruning / gemma3-4b | qa | 3 | 0.17613 | 2.51225 | 19.30928 | 0.14826 |
| pruning / gemma4-31b | math | 3 | 4.43305 | 3.59739 | 4.24132 | 2.04050 |
| pruning / gemma4-31b | code | 3 | 3.58572 | 2.66969 | 3.40663 | 1.59829 |
| pruning / gemma4-31b | qa | 3 | 6.26939 | 5.19628 | 25.55596 | 0.64868 |
| pruning / muse-30b | math | 3 | 0.33613 | 1.50934 | 1.29534 | 0.01241 |
| pruning / muse-30b | code | 3 | 0.35907 | 1.05533 | 0.85313 | 0.07342 |
| pruning / muse-30b | qa | 3 | 0.13206 | 1.95993 | 7.05930 | 0.09395 |
| pruning / olmo3-32b | math | 3 | 0.01988 | 0.97655 | 0.01580 | 0.00172 |
| pruning / olmo3-32b | code | 3 | 0.03534 | 0.42992 | 0.02229 | 0.00590 |
| pruning / olmo3-32b | qa | 3 | 0.02609 | 1.37070 | 2.51118 | 0.00251 |
| pruning / olmo3-7b | math | 3 | 0.03655 | 0.77151 | 0.18064 | 0.00401 |
| pruning / olmo3-7b | code | 3 | 0.03186 | 0.86047 | 0.12227 | 0.00616 |
| pruning / olmo3-7b | qa | 3 | 0.11087 | 0.71505 | 2.58560 | 0.07970 |
| quantization / Qwen3-0.6B | math | 4 | 2.23058 | 1.11933 | 1.47773 | 0.02025 |
| quantization / Qwen3-0.6B | code | 4 | 2.73980 | 0.66268 | 2.09775 | 0.02932 |
| quantization / Qwen3-0.6B | qa | 4 | 1.69384 | 1.07911 | 0.29762 | 0.04376 |
| quantization / Qwen3-1.7B | math | 4 | 2.39004 | 0.40382 | 0.15070 | 0.00374 |
| quantization / Qwen3-1.7B | code | 4 | 2.37280 | 0.30926 | 1.35346 | 0.01260 |
| quantization / Qwen3-1.7B | qa | 4 | 2.08545 | 0.20596 | 3.43543 | 0.06303 |
| quantization / Qwen3-4B | math | 4 | 2.69370 | 0.22584 | 1.85336 | 0.00697 |
| quantization / Qwen3-4B | code | 4 | 2.98837 | 1.01467 | 1.77350 | 0.02064 |
| quantization / Qwen3-4B | qa | 4 | 1.54187 | 0.56585 | 1.03685 | 0.04112 |
| quantization / gemma3-12b | math | 4 | 5.07132 | 0.62988 | 2.84782 | 0.01257 |
| quantization / gemma3-12b | code | 4 | 5.73905 | 1.36285 | 3.95563 | 0.00597 |
| quantization / gemma3-12b | qa | 4 | 5.22613 | 1.75308 | 7.56173 | 0.02916 |
| quantization / gemma3-1b | math | 4 | 3.88930 | 1.92375 | 2.56516 | 0.05208 |
| quantization / gemma3-1b | code | 4 | 3.57344 | 2.07242 | 0.47546 | 0.03727 |
| quantization / gemma3-1b | qa | 4 | 2.47267 | 2.49435 | 3.50601 | 0.02614 |
| quantization / gemma3-270m | math | 4 | 6.66303 | 2.82491 | 10.78999 | 0.12111 |
| quantization / gemma3-270m | code | 4 | 7.09173 | 3.13679 | 13.43883 | 0.10073 |
| quantization / gemma3-270m | qa | 4 | 6.51545 | 3.71449 | 1.11718 | 0.02812 |
| quantization / gemma3-27b | math | 4 | 4.62621 | 0.24819 | 2.74898 | 0.00305 |
| quantization / gemma3-27b | code | 4 | 4.29966 | 0.10497 | 3.00109 | 0.00349 |
| quantization / gemma3-27b | qa | 4 | 4.10631 | 0.58239 | 7.90338 | 0.08920 |
| quantization / gemma3-4b | math | 4 | 5.46222 | 0.89827 | 1.11599 | 0.01885 |
| quantization / gemma3-4b | code | 4 | 4.82781 | 0.37208 | 1.81803 | 0.01027 |
| quantization / gemma3-4b | qa | 4 | 3.86059 | 0.33702 | 5.44934 | 0.04160 |
| quantization / gemma4-31b | math | 4 | 5.10267 | 2.22912 | 1.19559 | 0.00776 |
| quantization / gemma4-31b | code | 4 | 4.97960 | 1.65007 | 1.10158 | 0.00947 |
| quantization / gemma4-31b | qa | 4 | 3.79529 | 1.52306 | 4.91845 | 0.06587 |
| quantization / muse-30b | math | 4 | 2.80947 | 0.47556 | 8.45917 | 0.02137 |
| quantization / muse-30b | code | 4 | 2.90378 | 0.36267 | 11.90441 | 0.10446 |
| quantization / muse-30b | qa | 4 | 2.17240 | 0.57943 | 2.92903 | 0.08452 |
| quantization / olmo3-32b | math | 4 | 0.17296 | 2.03031 | 0.06360 | 0.00210 |
| quantization / olmo3-32b | code | 4 | 0.30713 | 1.59003 | 0.09197 | 0.00136 |
| quantization / olmo3-32b | qa | 4 | 0.09986 | 1.54478 | 0.06055 | 0.00622 |
| quantization / olmo3-7b | math | 4 | 1.17663 | 0.77655 | 0.15376 | 0.00093 |
| quantization / olmo3-7b | code | 4 | 1.16141 | 0.86388 | 0.18764 | 0.00131 |
| quantization / olmo3-7b | qa | 4 | 0.71691 | 0.89968 | 1.17374 | 0.02846 |
| distillation / gemma3-1b | math | 3 | 0.09564 | 0.01670 | 0.07711 | 0.01896 |
| distillation / gemma3-1b | code | 3 | 0.11429 | 0.03466 | 0.08581 | 0.01686 |
| distillation / gemma3-1b | qa | 3 | 1.22384 | 0.35992 | 0.32875 | 0.28747 |
| distillation / gemma3-270m | math | 3 | 0.08416 | 0.01373 | 0.02908 | 0.00647 |
| distillation / gemma3-270m | code | 3 | 0.13820 | 0.01581 | 0.04827 | 0.01624 |
| distillation / gemma3-270m | qa | 3 | 0.99202 | 0.23867 | 0.23946 | 0.20659 |
| distillation / gemma3-4b | math | 2 | 0.15166 | 0.07503 | 0.07515 | 0.01423 |
| distillation / gemma3-4b | code | 2 | 0.15405 | 0.05882 | 0.10684 | 0.00928 |
| distillation / gemma3-4b | qa | 2 | 1.21302 | 0.24595 | 0.24595 | 0.24595 |

## Retention, provenance and limitations

- pruning: retained 9 near-zero, 21 negative and 30 collapse-labelled score cells; 21 near-zero capability calibration responses; 0 numerical fallbacks.
- quantization: retained 55 near-zero, 47 negative and 42 collapse-labelled score cells; 0 near-zero capability calibration responses; 0 numerical fallbacks.
- distillation: retained 0 near-zero, 11 negative and 0 collapse-labelled score cells; 0 near-zero capability calibration responses; 0 numerical fallbacks.

One declared recipe exclusion: Gemma3-4B/D600 used full training, outside V28's LoRA ladder (all three capabilities excluded). Only seed0 final distillation runs enter: no seed averaging, trajectory snapshots, or matched exposure controls. Qwen14 pruning aliases are verified identical and counted once. Raw metadata caveats, including the Qwen0.6B 5-bit dense protocol discrepancy, are retained in the inventory.

The frozen V30 map uses log nominal model size, family and dense L_c, not V28 non-embedding N0. This remains pre-compression basic information; converting its feature to N0 would require a forbidden refit.

Frozen V28/V25 math and QA constant response, code beta*log(1+D/150). No size/family effect is fitted; K0 uses a subset of allowed basic inputs. Preserve each run's own dense anchor, including small historical drift across Gemma budgets; K1 uses only D75's dense anchor. Cross-family math/code signs differ; QA improves in both Gemma and Qwen.

Rows name held-out new sources; V35 is an existing-data replay, not a newly timestamped preregistration. Qwen8 quant uses V30b shared_eta, not the original V28 fixed-4^-b forecast. Qwen8 pruning actuals cannot establish the V28 named Base checkpoint identity. Qwen14 pruning extends the frozen mapping with architecture-only N0; its old JSON lacks checkpoint revision identity. Qwen4 seed0 only; seed replicas/control runs are not extra models or extra K1 measurements.

Qwen14 pruning N0=13,212,057,600 non-embedding 2-D parameters is computed from the embedded architecture transcription; see `qwen14_basic_metadata` for the exact formula and cached config revision. This is basic architecture information, not a fit to compression outcomes.

Reproduce: `python analysis/v35_info_budget.py --n-boot 10000`. Validate without writing: add `--dry-run`. Tests: `python -m pytest tests/test_v35.py`. [Machine-readable summary](../../results/v35-info-budget/summary.json) contains per-cell predictions, calibration audit, all-target scalar fits, lower-bound caveats, per-target/region gains and CIs, exclusions and SHA256 input/code provenance. [Implementation](../../analysis/v35_info_budget.py) imports `prediction_audit.py`, V28 prediction functions and V30 prediction functions. Development fits come from V28/V30 JSON; prospective quantization uses V30b's already-saved full-development map.
