# V83 Figure 2: latest frozen confirmations

Build: `python -B analysis/plot_fig2_confirm.py` (CPU; NumPy and matplotlib Agg; no fitting).
Outputs: `paper/paper/figs/final_confirmations.pdf` and `.png`, 5.5 × 2.4 inches; 300 dpi PNG, 1650 × 720 pixels. Minimum visible font: 7.2 pt. PDF width equals the 5.5-inch text width in `paper/paper/iclr2027_conference.sty`. The main Figure 2 inclusion and caption in `paper/paper/general.tex` use `figs/final_confirmations.pdf` at `\textwidth`.
Font: `/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf`; bold Times-like serif, embedded TrueType in the PDF.

Each bar is **MAE(baseline) − MAE(candidate)** in native-token nats/token. Blue is positive (candidate better); orange is negative (baseline better). All 18 bars are retained, including failures. Numbers immediately above the bar endpoint are **candidate MAE / baseline MAE**, rounded to three decimals for display only. Each arm has its own linear x scale; zero is marked. Distillation whiskers use the stored 95% paired pool-cluster interval on the difference, not intervals on either MAE.

## Cohorts and aggregation

- **Pruning, three-checkpoint panel:** exactly 410M@48k, 1.4B@112k, 6.9B@80k, densities 0.85, 0.675, 0.575. Pool the nine absolute errors separately for each method and capability, then compare power with the lower pooled MAE of A2 and median_curve. Do not choose a different baseline per checkpoint or density. Equal counts make this equivalent to the mean of the three state MAEs. A2 is the per-density source regression.
- **Pruning, 2.8B repeat:** use the stored pooled `scores` for power and median_curve; six rows/capability at d=0.85,0.75,0.65 under the 16k and 143k revision labels. Retain the supplied scoring convention. These labels load identical learned weights and are one state measured twice, not two independent source states (documented in `paper/paper/appendix.tex`, weight-identity and pruning-repeat paragraphs). No pruning confidence interval is available in these compare files; none is fabricated.
- **Grouped quantization, development-state boundary:** filter `rows` to `test_set=development_state_boundary`: two states (410M@143k, 1.4B@16k), b=3,4,5 and g=32,512; 12 equally weighted configurations/capability. Compare same_input_interpolation with the lower pooled MAE of low_order_2d and median. The surface is the selected baseline for math/code; the median for QA.
- **Grouped quantization, new state:** pool both `new_state_boundary` and `new_state_interior`, all nine configurations of 1.4B@112k per capability (six boundary plus three interior cells). Candidate is the per-configuration median; baseline is same_input_interpolation. Weight individual configurations equally, so the six-cell boundary stratum has twice the weight of the three-cell interior stratum.
- **Distillation:** use all six `groups`, two students (Gemma-3 270M, 1B) × math/code/QA. Candidate and strongest baseline are taken exactly from each group and checked against `freeze.json::selected` and `strongest_baseline`. Math/code use the zero-anchored reuse form (`E`); QA uses the joint form (`joint`). Candidate `E` and baseline `E-only` are distinct forms (the latter includes an intercept). No baseline is reselected using the confirmation scores.
- **Distillation intervals:** stored `paired_difference.ci95`; 5,000 paired percentile bootstrap samples with NumPy PCG64 seed 0. Average the three planned budgets (50k,100k,200k tokens) within each U=200 pool, then give each of six pools equal weight. Keep all budgets together and use shared pool draws across students and capabilities. The plotted predictions remain the stored planned-budget predictions; actual budget overshoot does not cause prediction recomputation. Intervals describe variation over these sampled pools, not training-seed variability or 18 independent points.

**Selection timing:** all plotted predictions were frozen before measurement. The requested better-of comparisons in the pruning panel and quantization boundary group choose the strongest *realized confirmation MAE* among the specified frozen alternatives. Likewise, the displayed quantization interpolation/median roles are the requested delivered comparisons, not the original development-selected `selected_candidate` (surface for math, median for code, zero for QA). This figure does not claim that these post-confirmation choices were preselected. Distillation selections, including the strongest baseline, were fixed using development LOCO.

## Exact plotted numbers

Values below use Python's round-trip float representation (no display rounding). n counts configuration/checkpoint rows per capability, not independent clusters.

| Arm / confirmation group | Capability | Candidate | Baseline | n | Candidate MAE | Baseline MAE | Baseline − candidate | 95% pool-cluster interval |
|---|---|---|---|---:|---:|---:|---:|---|
| Pruning / Three-checkpoint panel | Math | Power form (`power`) | Per-density source regression (`A2`) | 9 | 0.2430560743287057 | 0.22984457123512214 | -0.013211503093583571 | — |
| Pruning / Three-checkpoint panel | Code | Power form (`power`) | Median density curve (`median_curve`) | 9 | 0.23612291498241916 | 0.21352144573858442 | -0.022601469243834743 | — |
| Pruning / Three-checkpoint panel | QA | Power form (`power`) | Median density curve (`median_curve`) | 9 | 0.6784256894645805 | 0.2206420640578797 | -0.4577836254067008 | — |
| Pruning / 2.8B repeat | Math | Power form (`power`) | Median density curve (`median_curve`) | 6 | 0.2918249928910655 | 0.07991576366368745 | -0.21190922922737804 | — |
| Pruning / 2.8B repeat | Code | Power form (`power`) | Median density curve (`median_curve`) | 6 | 0.39335501944909756 | 0.05379427145234136 | -0.3395607479967562 | — |
| Pruning / 2.8B repeat | QA | Power form (`power`) | Median density curve (`median_curve`) | 6 | 0.6788475813133066 | 0.06522546340304179 | -0.6136221179102648 | — |
| Grouped quantization / Development-state boundary | Math | Same-input interpolation (`same_input_interpolation`) | Low-order surface (`low_order_2d`) | 12 | 0.06531429346249056 | 0.21488512888177 | 0.14957083541927946 | — |
| Grouped quantization / Development-state boundary | Code | Same-input interpolation (`same_input_interpolation`) | Low-order surface (`low_order_2d`) | 12 | 0.1162554201531743 | 0.3086031972120603 | 0.192347777058886 | — |
| Grouped quantization / Development-state boundary | QA | Same-input interpolation (`same_input_interpolation`) | Per-configuration median (`median`) | 12 | 0.47658908992693516 | 0.45793973780101377 | -0.018649352125921392 | — |
| Grouped quantization / New state: 1.4B at 112k | Math | Per-configuration median (`median`) | Same-input interpolation (`same_input_interpolation`) | 9 | 0.08817791136070029 | 0.1411768545686477 | 0.052998943207947416 | — |
| Grouped quantization / New state: 1.4B at 112k | Code | Per-configuration median (`median`) | Same-input interpolation (`same_input_interpolation`) | 9 | 0.15318054565142752 | 0.2032231282336886 | 0.050042582582261075 | — |
| Grouped quantization / New state: 1.4B at 112k | QA | Per-configuration median (`median`) | Same-input interpolation (`same_input_interpolation`) | 9 | 0.14119600232361654 | 0.605247069561631 | 0.4640510672380145 | — |
| Distillation / Gemma-3 270M | Math | Zero-anchored reuse form (`E`) | Token-budget regression (`T-only`) | 18 | 0.07425435842205592 | 0.0654443822373542 | -0.008809976184701723 | [-0.009781846803438903, -0.008105177298178598] |
| Distillation / Gemma-3 270M | Code | Zero-anchored reuse form (`E`) | Reuse regression (`E-only`) | 18 | 0.019251710710199432 | 0.02300591946643296 | 0.003754208756233529 | [0.002534091660976484, 0.004965926244298827] |
| Distillation / Gemma-3 270M | QA | Joint token/reuse form (`joint`) | Reuse regression (`E-only`) | 18 | 0.5149822769205946 | 0.6096644909789816 | 0.09468221405838695 | [0.07744223245545379, 0.10933621766581009] |
| Distillation / Gemma-3 1B | Math | Zero-anchored reuse form (`E`) | Surface with dense loss (`surface:L0`) | 18 | 0.056896930528940855 | 0.033486852869408196 | -0.02341007765953266 | [-0.023537359400026416, -0.023277549662340778] |
| Distillation / Gemma-3 1B | Code | Zero-anchored reuse form (`E`) | Reuse regression (`E-only`) | 18 | 0.04852503018632907 | 0.053212499226748616 | 0.004687469040419549 | [0.004157698706268864, 0.0049685181087731276] |
| Distillation / Gemma-3 1B | QA | Joint token/reuse form (`joint`) | Surface with log size (`surface:logN`) | 18 | 0.4633567206141849 | 0.4501518138333222 | -0.013204906780862724 | [-0.019848840748508734, -0.00977507758016205] |

## Comparator audit and exact source fields

- **Pruning / Three-checkpoint panel / Math:** eligible reference MAEs: `A2`=0.22984457123512214; `median_curve`=0.27725981439004505. Source: `results/v53-prune-dev/compare_pythia-410m@step48000.json; results/v53-prune-dev/compare_pythia-1.4b@step112000.json; results/v53-prune-dev/compare_pythia-6.9b@step80000.json :: absolute_errors.math[*]`.
- **Pruning / Three-checkpoint panel / Code:** eligible reference MAEs: `A2`=0.2217045267411343; `median_curve`=0.21352144573858442. Source: `results/v53-prune-dev/compare_pythia-410m@step48000.json; results/v53-prune-dev/compare_pythia-1.4b@step112000.json; results/v53-prune-dev/compare_pythia-6.9b@step80000.json :: absolute_errors.code[*]`.
- **Pruning / Three-checkpoint panel / QA:** eligible reference MAEs: `A2`=0.6822702828662187; `median_curve`=0.2206420640578797. Source: `results/v53-prune-dev/compare_pythia-410m@step48000.json; results/v53-prune-dev/compare_pythia-1.4b@step112000.json; results/v53-prune-dev/compare_pythia-6.9b@step80000.json :: absolute_errors.qa[*]`.
- **Pruning / 2.8B repeat / Math:** eligible reference MAEs: `median_curve`=0.07991576366368745. Source: `results/v72-prune-repeat/compare.json :: scores[method].by_capability.math.mae`.
- **Pruning / 2.8B repeat / Code:** eligible reference MAEs: `median_curve`=0.05379427145234136. Source: `results/v72-prune-repeat/compare.json :: scores[method].by_capability.code.mae`.
- **Pruning / 2.8B repeat / QA:** eligible reference MAEs: `median_curve`=0.06522546340304179. Source: `results/v72-prune-repeat/compare.json :: scores[method].by_capability.qa.mae`.
- **Grouped quantization / Development-state boundary / Math:** eligible reference MAEs: `low_order_2d`=0.21488512888177; `median`=0.49983851406575447. Source: `results/v69-quant-confirm/compare.json :: rows[capability=math, test_set in ('development_state_boundary',)].absolute_errors`.
- **Grouped quantization / Development-state boundary / Code:** eligible reference MAEs: `low_order_2d`=0.3086031972120603; `median`=0.5567357974803898. Source: `results/v69-quant-confirm/compare.json :: rows[capability=code, test_set in ('development_state_boundary',)].absolute_errors`.
- **Grouped quantization / Development-state boundary / QA:** eligible reference MAEs: `low_order_2d`=1.2281847007517377; `median`=0.45793973780101377. Source: `results/v69-quant-confirm/compare.json :: rows[capability=qa, test_set in ('development_state_boundary',)].absolute_errors`.
- **Grouped quantization / New state: 1.4B at 112k / Math:** eligible reference MAEs: `same_input_interpolation`=0.1411768545686477. Source: `results/v69-quant-confirm/compare.json :: rows[capability=math, test_set in ('new_state_boundary', 'new_state_interior')].absolute_errors`.
- **Grouped quantization / New state: 1.4B at 112k / Code:** eligible reference MAEs: `same_input_interpolation`=0.2032231282336886. Source: `results/v69-quant-confirm/compare.json :: rows[capability=code, test_set in ('new_state_boundary', 'new_state_interior')].absolute_errors`.
- **Grouped quantization / New state: 1.4B at 112k / QA:** eligible reference MAEs: `same_input_interpolation`=0.605247069561631. Source: `results/v69-quant-confirm/compare.json :: rows[capability=qa, test_set in ('new_state_boundary', 'new_state_interior')].absolute_errors`.
- **Distillation / Gemma-3 270M / Math:** eligible reference MAEs: `T-only`=0.0654443822373542. Source: `results/v70-distill-confirm/compare.json :: groups[student=gemma3-270m, capability=math]`.
- **Distillation / Gemma-3 270M / Code:** eligible reference MAEs: `E-only`=0.02300591946643296. Source: `results/v70-distill-confirm/compare.json :: groups[student=gemma3-270m, capability=code]`.
- **Distillation / Gemma-3 270M / QA:** eligible reference MAEs: `E-only`=0.6096644909789816. Source: `results/v70-distill-confirm/compare.json :: groups[student=gemma3-270m, capability=qa]`.
- **Distillation / Gemma-3 1B / Math:** eligible reference MAEs: `surface:L0`=0.033486852869408196. Source: `results/v70-distill-confirm/compare.json :: groups[student=gemma3-1b, capability=math]`.
- **Distillation / Gemma-3 1B / Code:** eligible reference MAEs: `E-only`=0.053212499226748616. Source: `results/v70-distill-confirm/compare.json :: groups[student=gemma3-1b, capability=code]`.
- **Distillation / Gemma-3 1B / QA:** eligible reference MAEs: `surface:logN`=0.4501518138333222. Source: `results/v70-distill-confirm/compare.json :: groups[student=gemma3-1b, capability=qa]`.

## Input SHA-256

All prediction hashes and freeze links are checked; compare-row absolute errors, pooled MAEs and all six stored bootstrap intervals are independently reproduced. Source files are rehashed before outputs are written. The renderer checks signed bar lengths, complete bar/label counts, full interval extents, minimum fonts, page and panel bounds, and text collisions at the fixed physical page size. Only the two figure files and this new note are written; no experimental results are modified. No tight bounding-box crop is applied.

- `results/v53-prune-dev/compare_pythia-410m@step48000.json`: `6cbfcff19a7196a5377e83417538159b86797996d3045c976fcb3e29dd321399`
- `results/v53-prune-dev/compare_pythia-1.4b@step112000.json`: `23dd3cd34588ea33666f0158fc05004b6bc8273ab8947bb43aad954085857406`
- `results/v53-prune-dev/compare_pythia-6.9b@step80000.json`: `288af4793219be8325a33fcaa0d2dbc0dd96defff4890de040aea0723e177eb9`
- `results/v53-prune-dev/register.json`: `7498103830f139f6bea16ee440b717a05dee41e45b9835fdac6a2251b833974d`
- `results/v53-prune-dev/predictions_pythia-410m@step48000.json`: `6a3e987278dcb2d59d37eb72161bc0a062dd5cbe67116087b9fe7374bf426d6a`
- `results/v53-prune-dev/predictions_pythia-1.4b@step112000.json`: `862e299cc30a55672c785de418b0c184c204e3d5c2b222fb6174351e507489f4`
- `results/v53-prune-dev/predictions_pythia-6.9b@step80000.json`: `c14ec20711138aa57d7ae4d394c97ba8b08c05ed65926c537473e6ec499558d3`
- `results/v72-prune-repeat/compare.json`: `639eef759655038f393e0451b4d7471883dae31d4da08ffe32a6277eaec76fb9`
- `results/v72-prune-repeat/freeze.json`: `3f4e45d0bd505b7d65815f2246a5c3b343f3a49be465aedf724aaa208b0de8b9`
- `results/v69-quant-confirm/compare.json`: `12944c61ed42c61f1beaf8f84ef5c87d900b3b1a38d65d175976e27d3d4e6bac`
- `results/v69-quant-confirm/freeze.json`: `6ade0a3ff576781cf286978519b6a922500d37e3bc72cef4e3e635e826cccdc2`
- `results/v70-distill-confirm/compare.json`: `983f53033b07e035b1cb552429b8ce6ab23f8d6aee4365237970e2b46b0a829b`
- `results/v70-distill-confirm/freeze.json`: `d7b28c7251dbc6113c3de7c957560827e603d4e77a48010f99835a8468db0c24`
