# A5: Corner second difference

Status: **PENDING**.

**The registered decision needs both students and cannot be reached yet.**

| Student | Distribution / role | I | I ± 2 × noise | 2 × noise | Individual band |
|---|---|---:|---|---:|---|
| gemma3-1b | 2WikiMultihopQA / PRIMARY | -0.2434013944 | [-0.5650013944, +0.0781986056] | 0.3216 | inside |
| gemma3-1b | MATH-500 / SECONDARY, marginal | +0.0075769056 | [-0.0100230944, +0.0251769056] | 0.0176 | inside |
| gemma3-1b | MBPP / underpowered | -0.0139780779 | [-0.0647780779, +0.0368219221] | 0.0508 | inside |
| gemma3-4b | PENDING | — | — | — | — |

## Registered analysis

`I = delta_3 - delta_4 - delta_1 + delta_2`. Delta is checkpoint loss minus the same trajectory's own update-0 loss, using A1's delta and distribution-identity functions. Units are native-token nats/token. Students are evaluated separately and never averaged.

On an exact rectangle, A(T) cancels within each reuse contrast and B*h(E) cancels between budgets, for any h. The achieved rectangle's residuals are reported below.

Source: `paper/docs/prereg/corner_second_difference_prereg.md`, including its same-day amendment. SHA256: `aefa1e0d79824803ced6c99aa1842c92fc35c883b9999992debad6e7bd7c75e5`. Numeric noise and power figures are parsed from the amendment table at runtime. The original pooled-QA figure is superseded.

| Readout | Role | Matched seed pairs | Noise on I | Prediction | Registered ratio |
|---|---|---:|---:|---:|---:|
| QA probe, 2Wiki, measured in the trajectory | PRIMARY | 30 | 0.1608 | 1.5546 | 9.67 |
| MATH-500 probe | SECONDARY, marginal | 30 | 0.0088 | 0.0215 | 2.44 |
| MBPP probe | underpowered | 30 | 0.0254 | 0.0144 | 0.57 |

The displayed intervals are I ± twice the registered noise, the registered decision band translated to I. They are not calibrated confidence intervals; the noise is a propagated pool-seed variability measure, not a fitted standard error. Pool B enters twice with a minus sign, so a constant pool-B offset enters I at double weight.

Rejection requires both students outside the band with the same sign. Both inside the band gives survival for the powered/marginal readouts; opposite signs are reported as a size-dependent interaction. A same-sign mixed-band case remains unresolved. MBPP is underpowered whatever it shows; a null there never supports additivity.

## Registered decisions

- **2WikiMultihopQA (PRIMARY): PENDING**. The registered decision needs both students and cannot be reached yet.
- **MATH-500 (SECONDARY, marginal): PENDING**. The registered decision needs both students and cannot be reached yet.
- **MBPP (underpowered): PENDING**. The registered decision needs both students and cannot be reached yet.

## Checkpoints, achieved budgets and reuse

Targets below are the plan's predicted completed-update coordinates, not nominal prompt-inclusive milestones. E = achieved supervised T / the plan's supervised pool size D. D is identified by exact pool-hash equality. PENDING entries show predictions only. The imperfect equality of budgets/reuse across paired corners is checked separately.

| Student | Corner / pool | Predicted update | Achieved update | Target T | Achieved T | D | Target E | Achieved E | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| gemma3-1b | 1 / B | 36 | 36 | 50558 | 50558 | 35376 | 1.429161013116 | 1.429161013116 | VERIFIED |
| gemma3-1b | 2 / A | 38 | 38 | 50563 | 50563 | 16962 | 2.980957434265 | 2.980957434265 | VERIFIED |
| gemma3-1b | 3 / C | 74 | 74 | 105433 | 105433 | 73766 | 1.429289916764 | 1.429289916764 | VERIFIED |
| gemma3-1b | 4 / B | 74 | 74 | 105450 | 105450 | 35376 | 2.980834464043 | 2.980834464043 | VERIFIED |
| gemma3-4b | 1 / B | 36 | — | 50558 | — | 35376 | 1.429161013116 | — | PENDING |
| gemma3-4b | 2 / A | 38 | 38 | 50563 | 50563 | 16962 | 2.980957434265 | 2.980957434265 | VERIFIED |
| gemma3-4b | 3 / C | 74 | — | 105433 | — | 73766 | 1.429289916764 | — | PENDING |
| gemma3-4b | 4 / B | 74 | — | 105450 | — | 35376 | 2.980834464043 | — | PENDING |

### Sources and missing checkpoints

#### gemma3-1b: COMPLETE

- Corner 1: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000036/eval.json`; own initial: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000000/eval.json`.
- Corner 2: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000038/eval.json`; own initial: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000000/eval.json`.
- Corner 3: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000074/eval.json`; own initial: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000000/eval.json`.
- Corner 4: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000074/eval.json`; own initial: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000000/eval.json`.

#### gemma3-4b: PENDING

- Corner 1: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000036/eval.json`; own initial: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000000/eval.json`.
- Corner 2: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000038/eval.json`; own initial: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000000/eval.json`.
- Corner 3: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000074/eval.json`; own initial: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000000/eval.json`.
- Corner 4: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000074/eval.json`; own initial: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000000/eval.json`.

Missing checkpoints (including required own initial measurements):

- `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000036/eval.json`
- `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000074/eval.json`
- `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000000/eval.json`
- `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000074/eval.json`

## Four raw responses

The following deltas are the four response values in I. Initial and checkpoint losses are included to audit their provenance. All printed raw values retain round-trip precision.

| Student | Distribution | Corner | Own initial loss | Checkpoint loss | Delta |
|---|---|---:|---:|---:|---:|
| gemma3-1b | 2WikiMultihopQA | 1 | 5.567791334661354 | 4.2270293824701195 | -1.3407619521912348 |
| gemma3-1b | 2WikiMultihopQA | 2 | 5.567791334661354 | 4.352900896414343 | -1.2148904382470116 |
| gemma3-1b | 2WikiMultihopQA | 3 | 5.567791334661354 | 4.3438745019920315 | -1.2239168326693228 |
| gemma3-1b | 2WikiMultihopQA | 4 | 5.567791334661354 | 4.713147410358566 | -0.8546439243027883 |
| gemma3-1b | MATH-500 | 1 | 1.2205068949840885 | 1.2893809668131535 | 0.06887407182906502 |
| gemma3-1b | MATH-500 | 2 | 1.2205068949840885 | 1.288092892862555 | 0.06758599787846653 |
| gemma3-1b | MATH-500 | 3 | 1.2205068949840885 | 1.3268298227004092 | 0.10632292771632068 |
| gemma3-1b | MATH-500 | 4 | 1.2205068949840885 | 1.3179648431580542 | 0.09745794817396569 |
| gemma3-1b | MBPP | 1 | 1.060091895482728 | 1.15613928255093 | 0.09604738706820193 |
| gemma3-1b | MBPP | 2 | 1.060091895482728 | 1.158658104517272 | 0.09856620903454383 |
| gemma3-1b | MBPP | 3 | 1.060091895482728 | 1.1693976970770594 | 0.10930580159433134 |
| gemma3-1b | MBPP | 4 | 1.060091895482728 | 1.1858945969884853 | 0.12580270150575723 |
| gemma3-4b | 2WikiMultihopQA | 2 | 5.477153884462151 | 4.407557270916334 | -1.0695966135458166 |
| gemma3-4b | MATH-500 | 2 | 0.7404436278223974 | 0.8470222761024397 | 0.10657864828004238 |
| gemma3-4b | MBPP | 2 | 0.8017050487156776 | 0.8995515943312666 | 0.09784654561558903 |

## Four residual mismatches

| Residual | Definition | Plan / verified achieved fraction | Percent |
|---|---|---|---:|
| budget_low | abs(T1 − T2) / T2 | 5/50563 | 0.009888653759% |
| budget_high | abs(T3 − T4) / min(T3,T4) | 17/105433 | 0.016123983952% |
| reuse_low | abs(E1 − E3) / min(E1,E3) | 7645/84760487 | 0.009019532887% |
| reuse_high | abs(E2 − E4) / min(E2,E4) | 559/13550325 | 0.004125362307% |

Verified against achieved coordinates for: gemma3-1b. Incomplete students retain planned residuals only.

## Residual contamination

**Unimplementable total:** Total contamination and the requested comparison of |I| with that total are unimplementable from plan.json without an additional surface assumption. The plan supplies four dependent, fixed-pool budget-equivalent sensitivities and explicitly says 'do not sum them'. They do not identify fixed-budget reuse effects or a joint bias bound. Each is reported below; no total is invented, no new slopes are estimated, and no corner is interpolated.

The stored plan uses L4 − L3 − L2 + L1, opposite to the registered I. Signed sensitivities are multiplied by −1; their absolute magnitudes are unchanged. Each row uses only stored adjacent-checkpoint slopes, multiplied by the plan's exact equivalent supervised displacement. The range is the envelope across those stored runs. JSON retains every source slope and checkpoint pair.

| Student | Distribution | Residual | Equivalent ΔT | Signed sensitivity range for I | Max absolute sensitivity | abs(I) | abs(I) / individual sensitivity |
|---|---|---|---|---|---:|---:|---:|
| gemma3-1b | 2WikiMultihopQA | budget_low | -5 | [-3.41935585e-05, 0.0001031046648] | 0.0001031046648 | 0.2434013944 | 2360.72 |
| gemma3-1b | 2WikiMultihopQA | budget_high | -17 | [-0.0004368521498, -6.470156649e-05] | 0.0004368521498 | 0.2434013944 | 557.171 |
| gemma3-1b | 2WikiMultihopQA | reuse_low | 7645/804 | [3.618989434e-05, 0.0002443469919] | 0.0002443469919 | 0.2434013944 | 996.13 |
| gemma3-1b | 2WikiMultihopQA | reuse_high | -1118/257 | [1.655672953e-05, 0.0001117877554] | 0.0001117877554 | 0.2434013944 | 2177.35 |
| gemma3-1b | MATH-500 | budget_low | -5 | [-1.090288887e-05, 9.44226999e-06] | 1.090288887e-05 | 0.007576905592 | 694.945 |
| gemma3-1b | MATH-500 | budget_high | -17 | [-2.485503897e-05, 2.598494958e-07] | 2.485503897e-05 | 0.007576905592 | 304.844 |
| gemma3-1b | MATH-500 | reuse_low | 7645/804 | [-1.45343093e-07, 1.390230999e-05] | 1.390230999e-05 | 0.007576905592 | 545.011 |
| gemma3-1b | MATH-500 | reuse_high | -1118/257 | [-6.649387418e-08, 6.360250301e-06] | 6.360250301e-06 | 0.007576905592 | 1191.29 |
| gemma3-1b | MBPP | budget_low | -5 | [-2.362781635e-06, 1.951061893e-05] | 1.951061893e-05 | 0.01397807795 | 716.434 |
| gemma3-1b | MBPP | budget_high | -17 | [-3.33108019e-05, 3.123414328e-06] | 3.33108019e-05 | 0.01397807795 | 419.626 |
| gemma3-1b | MBPP | reuse_low | 7645/804 | [-1.74703706e-06, 1.863191985e-05] | 1.863191985e-05 | 0.01397807795 | 750.222 |
| gemma3-1b | MBPP | reuse_high | -1118/257 | [-7.992623526e-07, 8.524027585e-06] | 8.524027585e-06 | 0.01397807795 | 1639.84 |

These individual comparisons are descriptive and cannot establish that I exceeds total contamination. No sensitivity is added to or subtracted from I or the registered noise band.

## Fresh-sample secondary check

- gemma3-1b: Fresh-sample distributions have no saved measurement at corner 2. The four-corner secondary check is unavailable and dropped. No endpoint is substituted and no new evaluation is run. Their registered noise estimates use only two matched seed pairs.
- gemma3-4b: Fresh-sample distributions have no saved measurement at corner 2. The four-corner secondary check is unavailable and dropped. No endpoint is substituted and no new evaluation is run. Their registered noise estimates use only two matched seed pairs.

## Discrepancies and execution

No discrepancies in the available selected checkpoints: update, achieved supervised total, processed total, pool identity, protocol, own baseline, saved delta and probe identity passed.

CPU and standard-library analysis of saved JSON only. No fitting, training, model loading or new evaluation. SHA256 input provenance is in summary.json. Existing corner 2 is checked against the SHA256 recorded in plan.json.

```bash
python -B -m pytest -q tests/test_a5_corner_second_difference.py
python -B analysis/a5_corner_second_difference.py
# Require all corners for an explicitly named student:
python -B analysis/a5_corner_second_difference.py --students gemma3-1b
```

Default: both students, missing checkpoints PENDING, exit zero. Explicit --students: missing checkpoints are still listed as PENDING but make the overall run FAILED with a non-zero exit. Malformed records and discrepancies fail in either mode. An unrequested student is NOT_REQUESTED and cannot contribute to a registered decision.
