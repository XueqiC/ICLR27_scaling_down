# A5: Corner second difference

Status: **COMPLETE**.

| Student | Distribution / role | I | I ± 2 × noise | 2 × noise | Individual band |
|---|---|---:|---|---:|---|
| gemma3-1b | 2WikiMultihopQA / PRIMARY | -0.2434013944 | [-0.5650013944, +0.0781986056] | 0.3216 | inside |
| gemma3-1b | MATH-500 / SECONDARY, marginal | +0.0075769056 | [-0.0100230944, +0.0251769056] | 0.0176 | inside |
| gemma3-1b | MBPP / underpowered | -0.0139780779 | [-0.0647780779, +0.0368219221] | 0.0508 | inside |
| gemma3-4b | 2WikiMultihopQA / PRIMARY | -0.2123381474 | [-0.5339381474, +0.1092618526] | 0.3216 | inside |
| gemma3-4b | MATH-500 / SECONDARY, marginal | -0.0368521746 | [-0.0544521746, -0.0192521746] | 0.0176 | outside |
| gemma3-4b | MBPP / underpowered | -0.0946080602 | [-0.1454080602, -0.0438080602] | 0.0508 | outside |

### Fresh-sample SECONDARY

Noise estimate is weak: only two matched seed pairs per distribution. The PRIMARY QA probe alone carries the registered decision; SECONDARY results never change it.

| Student | Distribution / role | I | I ± 2 × noise | 2 × noise | Individual band |
|---|---|---:|---|---:|---|
| gemma3-1b | 2wiki_new / SECONDARY | -0.3339129441 | [-1.0985129441, +0.4306870559] | 0.7646 | inside |
| gemma3-1b | triviaqa / SECONDARY | -0.1134321070 | [-0.4014321070, +0.1745678930] | 0.2880 | inside |
| gemma3-1b | musique / SECONDARY | +0.0434420384 | [-1.6285579616, +1.7154420384] | 1.6720 | inside |
| gemma3-4b | 2wiki_new / SECONDARY | -0.3070141066 | [-1.0716141066, +0.4575858934] | 0.7646 | inside |
| gemma3-4b | triviaqa / SECONDARY | -0.6584902635 | [-0.9464902635, -0.3704902635] | 0.2880 | outside |
| gemma3-4b | musique / SECONDARY | -0.0100411696 | [-1.6820411696, +1.6619588304] | 1.6720 | inside |

## Registered analysis

`I = delta_3 - delta_4 - delta_1 + delta_2`. Delta is checkpoint loss minus the same trajectory's own update-0 loss, using A1's delta and distribution-identity functions. Units are native-token nats/token. Students are evaluated separately and never averaged.

On an exact rectangle, A(T) cancels within each reuse contrast and B*h(E) cancels between budgets, for any h. The achieved rectangle's residuals are reported below.

Source: `paper/docs/prereg/corner_second_difference_prereg.md`, including its same-day amendment. SHA256: `aefa1e0d79824803ced6c99aa1842c92fc35c883b9999992debad6e7bd7c75e5`. Numeric noise and power figures are parsed from the amendment table at runtime. The original pooled-QA figure is superseded.

| Readout | Role | Matched seed pairs | Noise on I | Prediction | Registered ratio |
|---|---|---:|---:|---:|---:|
| QA probe, 2Wiki, measured in the trajectory | PRIMARY | 30 | 0.1608 | 1.5546 | 9.67 |
| MATH-500 probe | SECONDARY, marginal | 30 | 0.0088 | 0.0215 | 2.44 |
| MBPP probe | underpowered | 30 | 0.0254 | 0.0144 | 0.57 |
| 2Wiki, fresh sample | SECONDARY | 2 | 0.3823 | 0.8711 | 2.28 |
| TriviaQA | SECONDARY | 2 | 0.144 | 0.2321 | 1.61 |
| MuSiQue | SECONDARY | 2 | 0.836 | 0.3658 | 0.44 |

The displayed intervals are I ± twice the registered noise, the registered decision band translated to I. They are not calibrated confidence intervals; the noise is a propagated pool-seed variability measure, not a fitted standard error. Pool B enters twice with a minus sign, so a constant pool-B offset enters I at double weight.

Rejection requires both students outside the band with the same sign. Both inside the band gives survival for the powered/marginal readouts; opposite signs are reported as a size-dependent interaction. A same-sign mixed-band case remains unresolved. MBPP is underpowered whatever it shows; a null there never supports additivity.

## Registered decisions

- **2WikiMultihopQA (PRIMARY): SURVIVES**. Additivity survives its one fit-free test. Failure of the three fitted structures is attributed to the form of the individual terms or to irreducible variation rather than to a missing interaction.
- **MATH-500 (SECONDARY, marginal): SIZE_DEPENDENT_INTERACTION**. Students disagree in sign: size-dependent interaction; do not average them.
- **MBPP (underpowered): UNRESOLVED**. The students do not jointly satisfy rejection or survival. The registration specifies no conclusive outcome for this mixed-band case. MBPP remains underpowered whatever the outcome; a null cannot support additivity.

## Checkpoints, achieved budgets and reuse

Targets below are the plan's predicted completed-update coordinates, not nominal prompt-inclusive milestones. E = achieved supervised T / the plan's supervised pool size D. D is identified by exact pool-hash equality. PENDING entries show predictions only. The imperfect equality of budgets/reuse across paired corners is checked separately.

| Student | Corner / pool | Predicted update | Achieved update | Target T | Achieved T | D | Target E | Achieved E | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| gemma3-1b | 1 / B | 36 | 36 | 50558 | 50558 | 35376 | 1.429161013116 | 1.429161013116 | VERIFIED |
| gemma3-1b | 2 / A | 38 | 38 | 50563 | 50563 | 16962 | 2.980957434265 | 2.980957434265 | VERIFIED |
| gemma3-1b | 3 / C | 74 | 74 | 105433 | 105433 | 73766 | 1.429289916764 | 1.429289916764 | VERIFIED |
| gemma3-1b | 4 / B | 74 | 74 | 105450 | 105450 | 35376 | 2.980834464043 | 2.980834464043 | VERIFIED |
| gemma3-4b | 1 / B | 36 | 36 | 50558 | 50558 | 35376 | 1.429161013116 | 1.429161013116 | VERIFIED |
| gemma3-4b | 2 / A | 38 | 38 | 50563 | 50563 | 16962 | 2.980957434265 | 2.980957434265 | VERIFIED |
| gemma3-4b | 3 / C | 74 | 74 | 105433 | 105433 | 73766 | 1.429289916764 | 1.429289916764 | VERIFIED |
| gemma3-4b | 4 / B | 74 | 74 | 105450 | 105450 | 35376 | 2.980834464043 | 2.980834464043 | VERIFIED |

### Sources and missing checkpoints

#### gemma3-1b: COMPLETE

- Corner 1: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000036/eval.json`; own initial: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000000/eval.json`.
- Corner 2: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000038/eval.json`; own initial: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000000/eval.json`.
- Corner 3: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000074/eval.json`; own initial: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000000/eval.json`.
- Corner 4: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000074/eval.json`; own initial: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000000/eval.json`.

#### gemma3-4b: COMPLETE

- Corner 1: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000036/eval.json`; own initial: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000000/eval.json`.
- Corner 2: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000038/eval.json`; own initial: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000000/eval.json`.
- Corner 3: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000074/eval.json`; own initial: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000000/eval.json`.
- Corner 4: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000074/eval.json`; own initial: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000000/eval.json`.

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
| gemma3-4b | 2WikiMultihopQA | 1 | 5.477153884462151 | 4.057582171314741 | -1.41957171314741 |
| gemma3-4b | 2WikiMultihopQA | 2 | 5.477153884462151 | 4.407557270916334 | -1.0695966135458166 |
| gemma3-4b | 2WikiMultihopQA | 3 | 5.477153884462151 | 4.32445219123506 | -1.1527016932270913 |
| gemma3-4b | 2WikiMultihopQA | 4 | 5.477153884462151 | 4.886765438247012 | -0.5903884462151394 |
| gemma3-4b | MATH-500 | 1 | 0.7404436278223974 | 0.8670821336566147 | 0.12663850583421732 |
| gemma3-4b | MATH-500 | 2 | 0.7404436278223974 | 0.8470222761024397 | 0.10657864828004238 |
| gemma3-4b | MATH-500 | 3 | 0.7404436278223974 | 0.894775723594484 | 0.15433209577208662 |
| gemma3-4b | MATH-500 | 4 | 0.7404436278223974 | 0.911568040612214 | 0.1711244127898166 |
| gemma3-4b | MBPP | 1 | 0.8017050487156776 | 0.9446689548272807 | 0.14296390611160315 |
| gemma3-4b | MBPP | 2 | 0.8017050487156776 | 0.8995515943312666 | 0.09784654561558903 |
| gemma3-4b | MBPP | 3 | 0.8017050487156776 | 0.9406831266607617 | 0.1389780779450841 |
| gemma3-4b | MBPP | 4 | 0.8017050487156776 | 0.9901738263950398 | 0.1884687776793622 |

## Four residual mismatches

| Residual | Definition | Plan / verified achieved fraction | Percent |
|---|---|---|---:|
| budget_low | abs(T1 − T2) / T2 | 5/50563 | 0.009888653759% |
| budget_high | abs(T3 − T4) / min(T3,T4) | 17/105433 | 0.016123983952% |
| reuse_low | abs(E1 − E3) / min(E1,E3) | 7645/84760487 | 0.009019532887% |
| reuse_high | abs(E2 − E4) / min(E2,E4) | 559/13550325 | 0.004125362307% |

Verified against achieved coordinates for: gemma3-1b, gemma3-4b. Incomplete students retain planned residuals only.

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
| gemma3-4b | 2WikiMultihopQA | budget_low | -5 | [-1.865876726e-05, 0.0003055262452] | 0.0003055262452 | 0.2123381474 | 694.992 |
| gemma3-4b | 2WikiMultihopQA | budget_high | -17 | [-0.0006327166931, 6.55719304e-05] | 0.0006327166931 | 0.2123381474 | 335.598 |
| gemma3-4b | 2WikiMultihopQA | reuse_low | 7645/804 | [-3.667671992e-05, 0.0003539010183] | 0.0003539010183 | 0.2123381474 | 599.993 |
| gemma3-4b | 2WikiMultihopQA | reuse_high | -1118/257 | [-1.677945026e-05, 0.0001619082772] | 0.0001619082772 | 0.2123381474 | 1311.47 |
| gemma3-4b | MATH-500 | budget_low | -5 | [-7.595473538e-06, 1.728470598e-05] | 1.728470598e-05 | 0.03685217457 | 2132.07 |
| gemma3-4b | MATH-500 | budget_high | -17 | [-4.990149326e-05, 2.260634838e-05] | 4.990149326e-05 | 0.03685217457 | 738.498 |
| gemma3-4b | MATH-500 | reuse_low | 7645/804 | [-1.264453712e-05, 2.79116854e-05] | 2.79116854e-05 | 0.03685217457 | 1320.31 |
| gemma3-4b | MATH-500 | reuse_high | -1118/257 | [-5.784824327e-06, 1.27694826e-05] | 1.27694826e-05 | 0.03685217457 | 2885.96 |
| gemma3-4b | MBPP | budget_low | -5 | [4.236969654e-07, 1.531132207e-05] | 1.531132207e-05 | 0.09460806023 | 6178.96 |
| gemma3-4b | MBPP | budget_high | -17 | [-2.345764909e-05, 1.130540186e-05] | 2.345764909e-05 | 0.09460806023 | 4033.14 |
| gemma3-4b | MBPP | reuse_low | 7645/804 | [-6.323514576e-06, 1.312069998e-05] | 1.312069998e-05 | 0.09460806023 | 7210.6 |
| gemma3-4b | MBPP | reuse_high | -1118/257 | [-2.892982211e-06, 6.0026669e-06] | 6.0026669e-06 | 0.09460806023 | 15761 |

These individual comparisons are descriptive and cannot establish that I exceeds total contamination. No sensitivity is added to or subtracted from I or the registered noise band.

## Fresh-sample SECONDARY check

`I = delta_3 - delta_4 - delta_1 + delta_2`, using each saved scope file's `delta_from_update_0`. Every available scope corner's `actual_supervised_tokens` is checked against the plan's predicted corner budget before its deltas are used. Missing scope corners leave only the SECONDARY check PENDING; endpoints are never substituted.

- gemma3-1b: **COMPLETE**. SECONDARY only: the registered decision is carried by the PRIMARY QA probe. These fresh-sample results never change that decision. Their noise estimate is weak: only two matched seed pairs per distribution.
- gemma3-4b: **COMPLETE**. SECONDARY only: the registered decision is carried by the PRIMARY QA probe. These fresh-sample results never change that decision. Their noise estimate is weak: only two matched seed pairs per distribution.

| Student | Corner / pool | Update | Predicted T | Scope actual T | Status |
|---|---|---:|---:|---:|---|
| gemma3-1b | 1 / B | 36 | 50558 | 50558 | VERIFIED |
| gemma3-1b | 2 / A | 38 | 50563 | 50563 | VERIFIED |
| gemma3-1b | 3 / C | 74 | 105433 | 105433 | VERIFIED |
| gemma3-1b | 4 / B | 74 | 105450 | 105450 | VERIFIED |
| gemma3-4b | 1 / B | 36 | 50558 | 50558 | VERIFIED |
| gemma3-4b | 2 / A | 38 | 50563 | 50563 | VERIFIED |
| gemma3-4b | 3 / C | 74 | 105433 | 105433 | VERIFIED |
| gemma3-4b | 4 / B | 74 | 105450 | 105450 | VERIFIED |

### Scope sources and raw deltas

- gemma3-1b corner 1 (VERIFIED): `results/v99-scope/a5-corners-gemma3-1b-poolB/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101-f8defe2cc5d54af5/update-00000036.json`; `delta_from_update_0`: {"2wiki_new": -1.4230219762277958, "musique": -0.01820134206431523, "triviaqa": 0.21497875612173534}.
- gemma3-1b corner 2 (VERIFIED): `results/v99-scope/a5-corners-gemma3-1b-poolA/gpt-5.6-luna_full_66_matrix2_lora_dseed41-01b39086b0f5e684/update-00000038.json`; `delta_from_update_0`: {"2wiki_new": -1.2833521094566356, "musique": 0.12205410399377614, "triviaqa": 0.19262786409748145}.
- gemma3-1b corner 3 (VERIFIED): `results/v99-scope/a5-corners-gemma3-1b-poolC/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234-98e5d57052842e92/update-00000074.json`; `delta_from_update_0`: {"2wiki_new": -1.3519992489550683, "musique": 0.25126507391078823, "triviaqa": 0.2533613062616604}.
- gemma3-1b corner 4 (VERIFIED): `results/v99-scope/a5-corners-gemma3-1b-poolB/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101-f8defe2cc5d54af5/update-00000074.json`; `delta_from_update_0`: {"2wiki_new": -0.8784164380877746, "musique": 0.34807848158713695, "triviaqa": 0.3444425212803175}.
- gemma3-4b corner 1 (VERIFIED): `results/v99-scope/a5-corners-gemma3-4b-poolB/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101-15fadb650795b560/update-00000036.json`; `delta_from_update_0`: {"2wiki_new": -1.2092007085945662, "musique": -0.021766403008298907, "triviaqa": 0.7405688840951492}.
- gemma3-4b corner 2 (VERIFIED): `results/v99-scope/a5-corners-gemma3-4b-poolA/gpt-5.6-luna_full_66_matrix2_lora_dseed41-0e02d1102f506ae9/update-00000038.json`; `delta_from_update_0`: {"2wiki_new": -0.9320569651253914, "musique": 0.20728896524896268, "triviaqa": 0.4890923798973881}.
- gemma3-4b corner 3 (VERIFIED): `results/v99-scope/a5-corners-gemma3-4b-poolC/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234-2da4fe4688510efb/update-00000074.json`; `delta_from_update_0`: {"2wiki_new": -0.9843525502873565, "musique": 0.16498962655601646, "triviaqa": 0.6951106284981341}.
- gemma3-4b corner 4 (VERIFIED): `results/v99-scope/a5-corners-gemma3-4b-poolB/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101-15fadb650795b560/update-00000074.json`; `delta_from_update_0`: {"2wiki_new": -0.40019470023510983, "musique": 0.40408616441908696, "triviaqa": 1.1021243878264926}.

## Discrepancies and execution

No discrepancies in the available selected checkpoints: update, achieved supervised total, processed total, pool identity, protocol, own baseline, saved delta and probe identity passed.

CPU and standard-library analysis of saved JSON only. No fitting, training, model loading or new evaluation. SHA256 input provenance is in summary.json. Existing corner 2 is checked against the SHA256 recorded in plan.json.

```bash
python -B -m pytest -q tests/test_a5_corner_second_difference.py
python -B analysis/a5_corner_second_difference.py
# Require all corners for an explicitly named student:
python -B analysis/a5_corner_second_difference.py --students gemma3-1b
```

Default: both students, missing checkpoints PENDING, exit zero. Explicit --students: missing primary checkpoints are still listed as PENDING but make the overall run FAILED with a non-zero exit. Malformed records and discrepancies fail in either mode. Missing SECONDARY corners remain PENDING without failing the PRIMARY; secondary discrepancies fail the run but do not change the primary readouts or registered decision. An unrequested student is NOT_REQUESTED and cannot contribute to a registered decision.
