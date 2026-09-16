# A9 measurement-efficiency study

CPU only; no training or GPU use. Retrospective subsampling of existing development labels, evaluated on fixed historical confirmation cells. The A9 plan was written before any A9 fit; these are not newly blinded confirmation results.

## Result

The compact power form has a demonstrated smaller tested measurement budget in 6 predictor-pair/target comparisons where both predictors reach the target. It alone reaches the target in 12 additional comparisons; those establish no finite savings estimate for the comparator. Targets are 0.15, 0.25 and 0.35 nats, with all 20 fits required at a qualifying budget.

The advantage is restricted to density reduction: five finite comparisons against A2 and one against the median curve (new-state code at 0.35 nats: 9 versus 18 measurements). The five A2 comparisons partly reflect its inability to fit all replicates at sparse budgets. Source reduction shows no finite budget savings where both predictors reach the target. New-state QA favors the median curve in every complete three-predictor comparison; the compact form reaches none of the three QA error targets there. These statements apply only to this fixed panel, estimator and sampling plan.

| Reduction | Cells | Capability | Target | Power measurements | Comparator | Comparator measurements |
|---|---|---|---:|---:|---|---:|
| densities | seen | math | 0.35 | 18 | A2 | 36 |
| densities | new | math | 0.25 | 18 | A2 | 36 |
| densities | new | math | 0.35 | 9 | A2 | 36 |
| densities | new | code | 0.25 | 18 | A2 | 36 |
| densities | new | code | 0.35 | 9 | A2 | 36 |
| densities | new | code | 0.35 | 9 | median_curve | 18 |
| sources | seen | math | 0.15 | 20 | A2 | not reached |
| sources | seen | math | 0.15 | 20 | median_curve | not reached |
| sources | seen | math | 0.25 | 20 | A2 | not reached |
| sources | seen | math | 0.25 | 20 | median_curve | not reached |
| sources | seen | math | 0.35 | 20 | median_curve | not reached |
| sources | seen | qa | 0.25 | 20 | median_curve | not reached |
| sources | seen | qa | 0.35 | 20 | median_curve | not reached |
| densities | seen | math | 0.15 | 18 | A2 | not reached |
| densities | seen | math | 0.15 | 18 | median_curve | not reached |
| densities | seen | math | 0.25 | 18 | A2 | not reached |
| densities | seen | math | 0.25 | 18 | median_curve | not reached |
| densities | seen | math | 0.35 | 18 | median_curve | not reached |

Among 18 primary capability × axis × budget × test-panel comparisons with 20/20 fits for all three predictors, the median curve has the lowest (or tied-lowest) median MAE in 4 and power is strictly lowest in 4. The median curve does not dominate all these complete comparisons.

Power is lowest: sources/seen/math at 20; sources/seen/math at 36; sources/seen/qa at 20; densities/seen/math at 36.
Median is lowest: sources/new/math at 20; sources/new/qa at 20; sources/new/qa at 36; densities/new/qa at 36.

These are descriptive medians and empirical threshold crossings on a three-budget grid, not significance tests or continuous sample-complexity estimates. An unavailable A2 fit is an identifiability limit under the fixed unregularized estimator, not evidence that its prediction error is larger.

## Provenance and fixed design

A1/C73 is Gemma distillation, not Pythia pruning. Use the explicit nine-state V36/V40/V92 pruning panel, cross-checked with data_mirror. The pruning A2 is V42/V53, not A2/C74 distillation.

The development table is exactly 9 states (160M/410M/1.4B × 16k/64k/143k) × densities 0.6/0.7/0.8/0.9 × math/code/QA: 36 measurements per capability, 108 scalar labels. Input files, mirror checks, row-level provenance and all labels are in summary.json. No confirmation cell enters fitting.

Plan: `0b12b06d3a9919c737cdc6b89480dc5461606a981d2ccff2a090a6ee06e07a95`; written 2026-09-16T13:36:28.441818+00:00; fitting began 2026-09-16T13:45:30.265999+00:00.

R = 20, seeds 91000–91019, nested subsets. Source reduction retains the two seen-test states (1.4B@64k, 160M@143k), then samples the other seven: 2/5/9 states yield 8/20/36 measurements per capability (realized 22.2%/55.6%/100%; nominal 25%/50%/100%). This conditional source design keeps the fixed seen-state panel actually seen. At 25% the source subset is identical across replicates. Density reduction independently samples 1/2/4 strengths per source: 9/18/36 measurements (exact 25%/50%/100%). Axes are never mixed. At full budget all replicates repeat the same fit.

The x axis counts pruning responses per capability (also the number of source-density evaluations). Multiply by three for all-capability scalar labels. Dense anchors and N0/D0 metadata are equally available and excluded from this response-measurement count; training dense-anchor counts are 2/5/9 per capability for source reduction and 9 throughout density reduction. Test dense anchors are fixed inputs.

All predictors receive identical selected labels and source inputs. Power and A2 use the same training-only standardizer and unregularized least squares; no validation or hyperparameter search is allowed for any predictor. Power fits its structural exponent on the fixed V53 grid 0.5–6.0 by 0.05 using training SSE only. The source-free median deliberately ignores source covariates. A2 and median interpolate adjacent observed density anchors and linearly extend the nearest pair outside their sampled hull, without clipping. The compact form has five parameters; A2 has four per observed density (16 at full budget, not the paper's 20 from a different five-anchor panel); median has one estimated anchor per observed density.

Full column rank is required at relative SVD tolerance 1e-10; power additionally needs local Jacobian rank five. No ridge or pseudoinverse rescue is used. The table reports conditional medians/IQRs with fitted counts, and every failed attempt remains in summary.json with rank, nominal parameter count and reason. Threshold crossings require 20/20 successful fits and median MAE at or below the target; missing crossings are not extrapolated.

## Held-out cells

| Panel | In-range cells/capability | Boundary cells/capability | Source states or labels |
|---|---:|---:|---:|
| seen | 2 | 2 | 2 |
| new | 42 | 9 | 11 |
| 2.8B audit | 6 | 0 | 2 |

Primary curves and thresholds use the original density range [0.6, 0.9]. Boundary points (including 0.55 and V53's 0.575, outside this nine-state panel) are retained and scored separately and in all-cell summaries. New-state cells are the union of V38/V42/V46/V49 and V53; V49 protocol B duplicates the same observations and is not counted twice. V36 leave-stage-out rows are development results and are not treated as confirmation. All predictors and replicates use the exact same fixed cells within each panel.

V72's two 2.8B stage labels have identical learned weights (C52), so they are one weight state measured twice, not two independent training stages. All 18 labelled capability cells are scored in a separate audit using their recorded dense inputs and D0 labels. Their actual training histories are unverified; they are excluded from the primary new-state aggregate and all efficiency claims. The full audit, boundary and all-cell curves/thresholds are in summary.json.

## Primary error curves

MAE in nats: median [25th, 75th percentile] over fitted replicates (fitted/20). IQRs measure subset variation, not uncertainty across independent experiments. `seen` means unseen densities of the two retained development states.

| Reduction | Cells | Capability | Measurements | Power | A2 | Median curve |
|---|---|---|---:|---|---|---|
| sources | seen | math | 8 | unfittable (0/20) | unfittable (0/20) | 1.767 [1.767, 1.767] (20/20) |
| sources | seen | math | 20 | 0.135 [0.127, 0.165] (20/20) | 0.302 [0.300, 0.337] (20/20) | 1.767 [1.767, 1.767] (20/20) |
| sources | seen | math | 36 | 0.083 [0.083, 0.083] (20/20) | 0.261 [0.261, 0.261] (20/20) | 1.767 [1.767, 1.767] (20/20) |
| sources | seen | code | 8 | unfittable (0/20) | unfittable (0/20) | 2.653 [2.653, 2.653] (20/20) |
| sources | seen | code | 20 | 0.466 [0.437, 0.491] (20/20) | 0.310 [0.279, 0.324] (20/20) | 2.653 [2.653, 2.653] (20/20) |
| sources | seen | code | 36 | 0.484 [0.484, 0.484] (20/20) | 0.319 [0.319, 0.319] (20/20) | 2.653 [2.653, 2.653] (20/20) |
| sources | seen | qa | 8 | unfittable (0/20) | unfittable (0/20) | 2.379 [2.379, 2.379] (20/20) |
| sources | seen | qa | 20 | 0.175 [0.075, 0.232] (20/20) | 0.214 [0.177, 0.222] (20/20) | 2.379 [2.379, 2.379] (20/20) |
| sources | seen | qa | 36 | 0.772 [0.772, 0.772] (20/20) | 0.646 [0.646, 0.646] (20/20) | 2.379 [2.379, 2.379] (20/20) |
| sources | new | math | 8 | unfittable (0/20) | unfittable (0/20) | 1.029 [1.029, 1.029] (20/20) |
| sources | new | math | 20 | 0.248 [0.241, 0.279] (20/20) | 0.232 [0.221, 0.258] (20/20) | 0.187 [0.182, 0.210] (20/20) |
| sources | new | math | 36 | 0.189 [0.189, 0.189] (20/20) | 0.182 [0.182, 0.182] (20/20) | 0.187 [0.187, 0.187] (20/20) |
| sources | new | code | 8 | unfittable (0/20) | unfittable (0/20) | 1.365 [1.365, 1.365] (20/20) |
| sources | new | code | 20 | 0.208 [0.200, 0.213] (20/20) | 0.155 [0.140, 0.173] (20/20) | 0.203 [0.181, 0.264] (20/20) |
| sources | new | code | 36 | 0.166 [0.166, 0.166] (20/20) | 0.125 [0.125, 0.125] (20/20) | 0.182 [0.182, 0.182] (20/20) |
| sources | new | qa | 8 | unfittable (0/20) | unfittable (0/20) | 1.361 [1.361, 1.361] (20/20) |
| sources | new | qa | 20 | 0.758 [0.698, 0.853] (20/20) | 0.818 [0.789, 0.908] (20/20) | 0.379 [0.312, 0.447] (20/20) |
| sources | new | qa | 36 | 0.537 [0.537, 0.537] (20/20) | 0.561 [0.561, 0.561] (20/20) | 0.307 [0.307, 0.307] (20/20) |
| densities | seen | math | 9 | 0.442 [0.312, 0.817] (20/20) | unfittable (0/20) | 1.767 [1.767, 1.767] (20/20) |
| densities | seen | math | 18 | 0.141 [0.079, 0.206] (20/20) | 0.539 [0.245, 2.470] (6/20) | 1.767 [1.767, 1.767] (20/20) |
| densities | seen | math | 36 | 0.083 [0.083, 0.083] (20/20) | 0.261 [0.261, 0.261] (20/20) | 1.767 [1.767, 1.767] (20/20) |
| densities | seen | code | 9 | 1.053 [0.579, 1.605] (20/20) | unfittable (0/20) | 2.653 [2.653, 2.653] (20/20) |
| densities | seen | code | 18 | 0.460 [0.318, 0.819] (20/20) | 1.405 [0.948, 1.955] (6/20) | 2.653 [2.653, 2.653] (20/20) |
| densities | seen | code | 36 | 0.484 [0.484, 0.484] (20/20) | 0.319 [0.319, 0.319] (20/20) | 2.653 [2.653, 2.653] (20/20) |
| densities | seen | qa | 9 | 1.509 [0.967, 2.276] (20/20) | unfittable (0/20) | 2.379 [2.379, 2.379] (20/20) |
| densities | seen | qa | 18 | 1.179 [0.754, 1.331] (20/20) | 1.007 [0.763, 1.460] (6/20) | 2.379 [2.379, 2.379] (20/20) |
| densities | seen | qa | 36 | 0.772 [0.772, 0.772] (20/20) | 0.646 [0.646, 0.646] (20/20) | 2.379 [2.379, 2.379] (20/20) |
| densities | new | math | 9 | 0.254 [0.199, 0.322] (20/20) | unfittable (0/20) | 0.331 [0.197, 0.655] (20/20) |
| densities | new | math | 18 | 0.195 [0.169, 0.220] (20/20) | 0.237 [0.205, 0.345] (6/20) | 0.210 [0.177, 0.263] (20/20) |
| densities | new | math | 36 | 0.189 [0.189, 0.189] (20/20) | 0.182 [0.182, 0.182] (20/20) | 0.187 [0.187, 0.187] (20/20) |
| densities | new | code | 9 | 0.304 [0.210, 0.526] (20/20) | unfittable (0/20) | 0.475 [0.214, 0.874] (20/20) |
| densities | new | code | 18 | 0.174 [0.150, 0.189] (20/20) | 0.277 [0.211, 0.367] (6/20) | 0.206 [0.185, 0.270] (20/20) |
| densities | new | code | 36 | 0.166 [0.166, 0.166] (20/20) | 0.125 [0.125, 0.125] (20/20) | 0.182 [0.182, 0.182] (20/20) |
| densities | new | qa | 9 | 0.607 [0.418, 1.058] (20/20) | unfittable (0/20) | 0.584 [0.369, 0.898] (20/20) |
| densities | new | qa | 18 | 0.519 [0.453, 0.588] (20/20) | 0.764 [0.633, 0.837] (6/20) | 0.327 [0.308, 0.384] (20/20) |
| densities | new | qa | 36 | 0.537 [0.537, 0.537] (20/20) | 0.561 [0.561, 0.561] (20/20) | 0.307 [0.307, 0.307] (20/20) |

## Measurements needed for the pre-set error levels

Entries are minimum tested measurements per capability for 0.15 / 0.25 / 0.35 nats; NR = not reached with 20/20 fits. No interpolation between budgets and no monotonic smoothing.

| Reduction | Cells | Capability | Power | A2 | Median curve |
|---|---|---|---|---|---|
| sources | seen | math | 20 / 20 / 20 | NR / NR / 20 | NR / NR / NR |
| sources | seen | code | NR / NR / NR | NR / NR / 20 | NR / NR / NR |
| sources | seen | qa | NR / 20 / 20 | NR / 20 / 20 | NR / NR / NR |
| sources | new | math | NR / 20 / 20 | NR / 20 / 20 | NR / 20 / 20 |
| sources | new | code | NR / 20 / 20 | 36 / 20 / 20 | NR / 20 / 20 |
| sources | new | qa | NR / NR / NR | NR / NR / NR | NR / NR / 36 |
| densities | seen | math | 18 / 18 / 18 | NR / NR / 36 | NR / NR / NR |
| densities | seen | code | NR / NR / NR | NR / NR / 36 | NR / NR / NR |
| densities | seen | qa | NR / NR / NR | NR / NR / NR | NR / NR / NR |
| densities | new | math | NR / 18 / 9 | NR / 36 / 36 | NR / 18 / 9 |
| densities | new | code | NR / 18 / 9 | 36 / 36 / 36 | NR / 18 / 18 |
| densities | new | qa | NR / NR / NR | NR / NR / NR | NR / NR / 18 |

## Fit availability and identifiability

Counts below aggregate the three capabilities: 60 attempts per row. Every individual rank, singular spectrum, parameter count, fitted coefficient, gamma profile and prediction is recorded in summary.json.

| Reduction | Measurements/capability | Predictor | Fitted/60 | Parameters | Design ranks |
|---|---:|---|---:|---|---|
| sources | 8 | power | 0/60 | 5 | 2 |
| sources | 8 | A2 | 0/60 | 16 | 8 |
| sources | 8 | median_curve | 60/60 | 4 | 4 |
| sources | 20 | power | 60/60 | 5 | 4 |
| sources | 20 | A2 | 60/60 | 16 | 16 |
| sources | 20 | median_curve | 60/60 | 4 | 4 |
| sources | 36 | power | 60/60 | 5 | 4 |
| sources | 36 | A2 | 60/60 | 16 | 16 |
| sources | 36 | median_curve | 60/60 | 4 | 4 |
| densities | 9 | power | 60/60 | 5 | 4 |
| densities | 9 | A2 | 0/60 | 12, 16 | 8, 9 |
| densities | 9 | median_curve | 60/60 | 3, 4 | 3, 4 |
| densities | 18 | power | 60/60 | 5 | 4 |
| densities | 18 | A2 | 18/60 | 16 | 12, 13, 14, 15, 16 |
| densities | 18 | median_curve | 60/60 | 4 | 4 |
| densities | 36 | power | 60/60 | 5 | 4 |
| densities | 36 | A2 | 60/60 | 16 | 16 |
| densities | 36 | median_curve | 60/60 | 4 | 4 |

## Boundary and duplicated-state audit

Full-budget median MAEs; all 20 repeats are identical. These rows do not enter the primary efficiency claims.

| Panel | Region | Capability | Power | A2 | Median curve |
|---|---|---|---:|---:|---:|
| seen | boundary | math | 1.053 | 0.496 | 3.931 |
| seen | boundary | code | 1.102 | 0.491 | 4.078 |
| seen | boundary | qa | 0.706 | 1.014 | 4.060 |
| new | boundary | math | 0.823 | 0.869 | 0.899 |
| new | boundary | code | 0.760 | 0.678 | 0.785 |
| new | boundary | qa | 2.264 | 1.960 | 0.799 |
| 2.8B audit | in_range | math | 0.372 | 0.429 | 0.058 |
| 2.8B audit | in_range | code | 0.442 | 0.486 | 0.102 |
| 2.8B audit | in_range | qa | 0.236 | 0.327 | 0.203 |

## Artifacts and checks

`summary.json` contains every selected subset, development and held-out row, fit/failure, prediction, MAE, median/IQR, paired difference and threshold crossing (including per-replicate crossings). `a9_efficiency.pdf` contains four pages of primary curves. Per-panel PDFs/PNGs and numeric/style sidecars are under `figs/`, following analysis/paper_figure_style.py typography, palette and full-canvas conventions.

Panel order: a–c source reduction/seen states (math, code, QA); d–f source reduction/new states; g–i density reduction/seen states; j–l density reduction/new states. Lines show conditional medians, bands/error bars show IQRs, horizontal guides show the pre-set targets. Missing fits are gaps; availability is explicit above and in each sidecar.

Validation passed: 1080 fit attempts, 222 explicitly reported failures; same inputs and fixed test-cell hashes; recomputed MAEs; identical full-budget fits; nested single-axis subsets; synthetic law recovery, rank rejection, interpolation and median checks. Full-budget A2 reproduces historical V42 predictions to 1.3e-15 nats. A runtime guard confines analysis writes to A9 outputs and temporary caches, and blocks subprocesses/network operations. This analysis wrote no paper TeX, figures or tables. No commit was made.
