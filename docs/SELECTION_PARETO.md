# Selection, measured Pareto domination, and model uncertainty (V33b)

**Offline existing-data analysis; no new runs, no refitting, no prospective calibration.** The frozen selector is miscalibrated due to prediction error on this replay. Its pruning failures do not establish pruning domination.

Across 12 development models and 2210 total budget vectors, pruning has **24 measured Pareto-frontier configurations across 6/12 models**. The selector's 31 pruning picks have 0 actual successes, while feasible pruning configurations are wrongly excluded at rate **0.45975 [0.28683, 0.60554]**. These are distinct statements about measured candidates and predictor decisions.

## Measured frontier and true domination

Minimize (nominal storage, measured math CE, measured code CE, measured QA CE) per model; no worse in all, strictly better in at least one; tolerance 1e-12. Identical vectors retain all ties.

This is a four-objective frontier: loss capabilities are never averaged, and predictions never enter dominance. Every measured configuration is checked against every other configuration of the same model, including within-arm rivals. An arm is absent only when all its measured configurations are dominated; different rivals may dominate different configurations. This establishes domination only within the available measured candidate set, not every possible density, bit width, compression algorithm, or noisy population loss.

Independent of quantile grid: test cheaper feasible rivals at tau_c=max(0, actual_delta_c-1e-12), the minimal budget under the feasibility tolerance. Equal-cost optima are retained, even if loss-dominated; signed-loss Pareto membership is reported separately.

The minimal budget is a witness: if no strictly cheaper rival is feasible there, the candidate is cost optimal including ties at that budget; otherwise that rival remains feasible wherever the candidate is feasible. Negative measured deltas are retained. A frontier point with stronger negative deltas can still be unnecessary when budgets must be nonnegative. A loss-dominated point can tie a rival's cost. Thus neither sampled-grid oracle counts nor arbitrary tie breaking define Pareto membership.

| Model | Dense frontier IDs | Pruning frontier IDs | Quantization frontier IDs | Pruning optimal at any nonnegative budget, including ties? |
|---|---|---|---|---|
| Qwen3-0.6B | dense | pruning:0.7, pruning:0.8, pruning:0.9 | quantization:3, quantization:4, quantization:5, quantization:6, quantization:8 | True |
| Qwen3-1.7B | none (all dominated) | pruning:0.55, pruning:0.575, pruning:0.6, pruning:0.625, pruning:0.65, pruning:0.675, pruning:0.7, pruning:0.8, pruning:0.9 | quantization:3, quantization:4, quantization:5, quantization:6, quantization:8 | False |
| Qwen3-4B | none (all dominated) | pruning:0.45, pruning:0.5, pruning:0.55, pruning:0.6, pruning:0.7 | quantization:3, quantization:4, quantization:5, quantization:6, quantization:8 | True |
| gemma3-12b | dense | none (all dominated) | quantization:3, quantization:4, quantization:5, quantization:6, quantization:8 | False |
| gemma3-1b | dense | none (all dominated) | quantization:3, quantization:4, quantization:5, quantization:6, quantization:8 | False |
| gemma3-270m | dense | none (all dominated) | quantization:3, quantization:4, quantization:5, quantization:6, quantization:8 | False |
| gemma3-27b | dense | none (all dominated) | quantization:3, quantization:4, quantization:5, quantization:6, quantization:8 | False |
| gemma3-4b | dense | none (all dominated) | quantization:3, quantization:4, quantization:5, quantization:6, quantization:8 | False |
| gemma4-31b | dense | none (all dominated) | quantization:3, quantization:4, quantization:5, quantization:6, quantization:8 | False |
| muse-30b | dense | pruning:0.9 | quantization:3, quantization:4, quantization:5, quantization:6, quantization:8 | False |
| olmo3-32b | dense | pruning:0.3, pruning:0.35, pruning:0.4, pruning:0.45, pruning:0.9 | quantization:3, quantization:4, quantization:5, quantization:6, quantization:8 | True |
| olmo3-7b | none (all dominated) | pruning:0.9 | quantization:3, quantization:4, quantization:5, quantization:6, quantization:8 | False |

`summary.json` supplies every configuration's measured objective vector, complete dominator IDs, cross-arm dominator IDs, frontier flag, minimal budget witness and cheaper rivals. Frontier flags are deterministic conditional on aggregate measurements; bootstrap intervals do not estimate noise in those measurements.

All measured pruning configurations are dominated in: gemma3-12b, gemma3-1b, gemma3-270m, gemma3-27b, gemma3-4b, gemma4-31b. Pruning can minimize nominal cost at some nonnegative budget (including ties) in: Qwen3-0.6B, Qwen3-4B, olmo3-32b. The other frontier appearances involve loss improvements beyond what nonnegative budgets require.

## Policies on the unchanged V33 grids

The 2210 vectors are spread across the 12 models (unequal grids), not 2210 independent replicates per model. The saved outcome-informed quantile grids, signed loss deltas from each V6 source dense, and V33 V28/V30 frozen held-out predictions are reused exactly. Raw V6/V10 measurements and nominal costs are checked against the saved candidates; saved decisions and input SHA-256 hashes are checked before reuse. V10 predictions keep V33's dense-anchor rebasing.

Selector: cheapest candidate meeting every predicted capability constraint. Quant-only: same rule restricted to quantization plus dense fallback. Oracle: cheapest actually feasible candidate. Always-cheapest ignores the budget; always-dense chooses the exact source reference. Oracle is a privileged reference. Ties use nominal storage, method name, configuration ID. Every reported satisfaction score uses all three actual constraints.

**Nominal storage** is pruning density, quantization bits/16, and dense 1, relative to the source at 16 bits. These ratios omit sparse indices, quantization metadata, untouched tensors, packing and runtime representation. **No latency claims.**

| Policy | Actual satisfaction [95% CI] | Nominal storage [95% CI] | Signed nominal gap vs oracle [95% CI] |
|---|---|---|---|
| selector | 0.70889 [0.61614, 0.79530] | 0.55945 [0.50464, 0.61158] | 0.06677 [0.00144, 0.13242] |
| oracle | 1.00000 [1.00000, 1.00000] | 0.49268 [0.43932, 0.54545] | 0.00000 [0.00000, 0.00000] |
| quant_only | 0.71676 [0.63387, 0.79687] | 0.56144 [0.50679, 0.61291] | 0.06877 [0.00486, 0.13291] |
| always_cheapest | 0.00705 [0.00530, 0.00905] | 0.18750 [0.18750, 0.18750] | -0.30518 [-0.35795, -0.25182] |
| always_dense | 1.00000 [1.00000, 1.00000] | 1.00000 [1.00000, 1.00000] | 0.50732 [0.45455, 0.56068] |

Distill-only on the full 12-model compression panel: **N/A**. No paired smaller source-to-student compression panel for all 12 sources. The covered-subset distill_only diagnostic selects a fixed V12 GPT/full/600 adaptation of the source itself, at nominal storage 1; it is not size reduction or a commercial-teacher storage ratio. Missing models are not dense-imputed.

Signed gaps above use all decisions; lower cost obtained by violating a constraint is not a successful saving. Conditional-on-success gaps and paired policy differences are in JSON. For the selector the conditional gap is 0.15289 [0.09126, 0.21184].

| Frozen-predictor exclusion | Rate [95% model CI] |
|---|---|
| all_compressed | 0.32706 [0.21361, 0.42409] |
| pruning | 0.45975 [0.28683, 0.60554] |
| quantization | 0.20947 [0.13551, 0.28228] |

V33 predicted-infeasible AND actually-feasible compressed configuration-budget pairs / actually-feasible pairs; normalize counts by each model's budget count before pooling; zero denominators undefined.

Selected pruning satisfaction is 0.00000 [0.00000, 0.00000]; 1103 of 10000 bootstrap draws contain no pruning picks and are undefined for this conditional statistic. The zero observed successes are not a population guarantee.

## Nominal cost at matched actual satisfaction

Uniform mixtures: if target r exceeds policy satisfaction s, mix with dense; if r<s, mix with cheapest; alpha=(r-s_anchor)/(s-s_anchor) is base-policy probability. Expected cost and actual satisfaction are mixed on the same model distribution. Recompute alpha inside every paired bootstrap draw; no extrapolation. These are retrospective randomized policy families, not the original deterministic policies or prospectively calibrated guarantees. Exact target refers to pooled equal-model expected satisfaction, not each model.

Each cell below is expected nominal storage [95% model CI] at the row's exact expected satisfaction, using all decisions. The mixture coin is independent of the model, budget and whether a decision actually fails. No outcome-dependent rescue of individual failures is used. Mixture weights use replay outcomes and must not be described as deployable calibration. Matched-rate intervals are degenerate by construction after re-estimating weights; they are not uncertainty intervals for the reliability of a fixed deployed mixture. Always-dense and always-cheapest trace the same two-anchor mixture; their equality is intentional. Below 100%, even the oracle mixture deliberately permits failures via the cheapest anchor.

| Matched satisfaction | selector | oracle | quant_only | always_cheapest | always_dense |
|---|---|---|---|---|---|
| 75% | 0.62166 [0.52802, 0.70842] | 0.41584 [0.37582, 0.45545] | 0.61291 [0.52665, 0.69640] | 0.79543 [0.79502, 0.79579] | 0.79543 [0.79502, 0.79579] |
| 80% | 0.69733 [0.58353, 0.76674] | 0.43121 [0.38852, 0.47346] | 0.69033 [0.58088, 0.75712] | 0.83635 [0.83602, 0.83663] | 0.83635 [0.83602, 0.83663] |
| 90% | 0.84866 [0.78891, 0.88337] | 0.46194 [0.41392, 0.50946] | 0.84517 [0.78778, 0.87856] | 0.91817 [0.91801, 0.91832] | 0.91817 [0.91801, 0.91832] |
| 95% | 0.92433 [0.89446, 0.94168] | 0.47731 [0.42662, 0.52745] | 0.92258 [0.89389, 0.93928] | 0.95909 [0.95900, 0.95916] | 0.95909 [0.95900, 0.95916] |
| 99% | 0.98487 [0.97889, 0.98834] | 0.48960 [0.43678, 0.54185] | 0.98452 [0.97878, 0.98786] | 0.99182 [0.99180, 0.99183] | 0.99182 [0.99180, 0.99183] |
| 100% | 1.00000 [1.00000, 1.00000] | 0.49268 [0.43932, 0.54545] | 1.00000 [1.00000, 1.00000] | 1.00000 [1.00000, 1.00000] | 1.00000 [1.00000, 1.00000] |

| Matched satisfaction | Selector minus oracle nominal storage [paired 95% CI] | Selector minus quant-only [paired 95% CI] |
|---|---|---|
| 75% | 0.20582 [0.12757, 0.27568] | 0.00875 [0.00000, 0.02000] |
| 80% | 0.26612 [0.17074, 0.32184] | 0.00700 [0.00000, 0.01660] |
| 90% | 0.38672 [0.33612, 0.42511] | 0.00350 [0.00000, 0.00831] |
| 95% | 0.44702 [0.40401, 0.48750] | 0.00175 [0.00000, 0.00416] |
| 99% | 0.49526 [0.44521, 0.54475] | 0.00035 [0.00000, 0.00083] |
| 100% | 0.50732 [0.45455, 0.56068] | 0.00000 [0.00000, 0.00000] |

At 100% the chosen mixing construction can collapse an imperfect policy to dense. That is a property of this uniform-fallback family, not a lower bound on what a future calibrated selector could achieve. Mixture probabilities, anchors, achieved rates and all paired cost differences are stored in JSON; unreachable targets/draws are undefined, never extrapolated.

## Distill-only: common-coverage adaptation diagnostic

No paired smaller source-to-student compression panel for all 12 sources. The covered-subset distill_only diagnostic selects a fixed V12 GPT/full/600 adaptation of the source itself, at nominal storage 1; it is not size reduction or a commercial-teacher storage ratio. Missing models are not dense-imputed.

Use the fixed teacher gpt-5.6-luna, full recipe, 600 traces per domain; no choice of run by outcome. The post-training candidate is the same source checkpoint size, evaluated against that source's V6 dense loss. It is selected on every budget for distill-only. The subset oracle also sees this additional candidate; the frozen selector/quant-only candidate sets stay as in V33. All policies in the following tables use exactly the same covered models and budgets. Nominal merged model storage is 1; adapter overhead and training cost are omitted.

Missing fixed-run models: gemma3-27b, gemma4-31b, muse-30b, olmo3-32b.

Covered sources: 8 (Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, gemma3-12b, gemma3-1b, gemma3-270m, gemma3-4b, olmo3-7b); 1499 budgets, bootstrap over these 8 models.

| Policy | Actual satisfaction [95% CI] | Nominal storage [95% CI] | Signed nominal gap vs oracle [95% CI] |
|---|---|---|---|
| selector | 0.69786 [0.57149, 0.81344] | 0.51754 [0.45619, 0.57707] | 0.03769 [-0.04887, 0.12795] |
| oracle | 1.00000 [1.00000, 1.00000] | 0.47985 [0.41192, 0.54308] | 0.00000 [0.00000, 0.00000] |
| quant_only | 0.70966 [0.59849, 0.81457] | 0.52054 [0.45852, 0.58009] | 0.04068 [-0.04380, 0.13010] |
| always_cheapest | 0.00712 [0.00462, 0.00994] | 0.18750 [0.18750, 0.18750] | -0.29235 [-0.35558, -0.22442] |
| always_dense | 1.00000 [1.00000, 1.00000] | 1.00000 [1.00000, 1.00000] | 0.52015 [0.45692, 0.58808] |
| distill_only | 0.59666 [0.37755, 0.83163] | 1.00000 [1.00000, 1.00000] | 0.52015 [0.45692, 0.58808] |

Matched expected nominal storage [95% model CI]:

| Matched satisfaction | selector | oracle | quant_only | always_cheapest | always_dense | distill_only |
|---|---|---|---|---|---|---|
| 75% | 0.60080 [0.48185, 0.71416] | 0.40624 [0.35527, 0.45377] | 0.58715 [0.48129, 0.69703] | 0.79542 [0.79484, 0.79593] | 0.79542 [0.79484, 0.79593] | 1.00000 [0.91937, 1.00000] |
| 80% | 0.68064 [0.52679, 0.77133] | 0.42096 [0.36660, 0.47164] | 0.66972 [0.52539, 0.75762] | 0.83633 [0.83587, 0.83675] | 0.83633 [0.83587, 0.83675] | 1.00000 [0.96876, 1.00000] |
| 90% | 0.84032 [0.75049, 0.88566] | 0.45041 [0.38926, 0.50736] | 0.83486 [0.74970, 0.87881] | 0.91817 [0.91793, 0.91837] | 0.91817 [0.91793, 0.91837] | 1.00000 [1.00000, 1.00000] |
| 95% | 0.92016 [0.87524, 0.94283] | 0.46513 [0.40059, 0.52522] | 0.91743 [0.87485, 0.93941] | 0.95908 [0.95897, 0.95919] | 0.95908 [0.95897, 0.95919] | 1.00000 [1.00000, 1.00000] |
| 99% | 0.98403 [0.97505, 0.98857] | 0.47691 [0.40965, 0.53951] | 0.98349 [0.97497, 0.98788] | 0.99182 [0.99179, 0.99184] | 0.99182 [0.99179, 0.99184] | 1.00000 [1.00000, 1.00000] |
| 100% | 1.00000 [1.00000, 1.00000] | 0.47985 [0.41192, 0.54308] | 1.00000 [1.00000, 1.00000] | 1.00000 [1.00000, 1.00000] | 1.00000 [1.00000, 1.00000] | 1.00000 [1.00000, 1.00000] |

The same-size adapted candidate is on the expanded measured frontier in 8/8 covered models, and can tie minimum nominal cost at some nonnegative budget in 2 models. Loss improvements can preserve its frontier membership even though always-dense has higher aggregate constraint satisfaction at the same nominal cost. Expanded frontiers and witnesses for every covered model are in JSON.

V12 identity, benchmarks, probe source/seed, odd-half convention and counts are validated. The JSON retains V12 own-dense minus V6 source-dense offsets; raw post-training losses are used without an invented compressed-loss correction. Aggregate artifacts cannot verify item identities, so this remains a measurement-qualified adaptation diagnostic, not proof of source-to-smaller-student compression. No losses from different model tokenizers are subtracted.

## Model-level uncertainty and limits

All headline satisfaction, nominal cost/gap, wrong-exclusion and policy-difference intervals use 10000 paired whole-model bootstrap draws, seed 330907, through `prediction_audit.bootstrap_means`. One sufficient-statistics row per model gives each model unit budget mass, regardless of its grid size. Resample all policies and ratio numerators/denominators together. Each draw recomputes matched-reliability mixture weights and cost differences. Undefined ratio or unattainable matching draws are counted explicitly. V33 already used whole-model resampling; V33b retains and extends it, rather than claiming that its existing CIs were based on independent budgets.

The 12 development models are a small, heterogeneous convenience panel. Shared frozen LOMO training folds are conditioned on, not independently retrained in the bootstrap. Intervals describe variation across these model clusters; they omit item, seed, dense-anchor, measurement, fit and scenario-distribution uncertainty. Within-model budget duplication does not create additional model evidence. No correction for multiple comparisons is applied.

The original measurement caveats remain: Qwen3-0.6B 5-bit metadata flags an odd-half protocol discrepancy; dense offsets, pruning infills and extrapolated frozen shapes are retained. Tight-budget decisions and measured frontier flags inherit those limitations. Missing configurations are not imputed, and archived/512-probe panels are not pooled.

Separate Qwen3-8B offline transfer is retained in JSON as a descriptive one-model result, excluded from the 12-model CIs. Legacy Qwen3-8B aggregate tables do not establish Base versus post-trained identity. This separate offline transfer is not validation of the named Qwen/Qwen3-8B-Base V28 target.

## Conclusion

**Established on this replay:** the frozen selector is miscalibrated due to predictor error, both accepting infeasible candidates and wrongly excluding feasible ones. **Measured domination test:** pruning survives on the actual loss/nominal-cost Pareto frontier in 6 of 12 models. Consequently a blanket claim that pruning is dominated is unsupported. Per-model and per-configuration domination findings are explicit above and in JSON. Matched-rate nominal costs describe retrospective mixtures, not a calibrated future selector or measured latency.

## Reproduce

```bash
python analysis/v33b_selection_pareto.py --bootstrap 10000
python analysis/v33b_selection_pareto.py --dry-run --bootstrap 10000
python -m pytest -q tests/test_v33b.py tests/test_v33.py
```

Outputs: `results/v33b-selection-pareto/summary.json`, its `report.md`, and `paper/docs/SELECTION_PARETO.md`. Input/code hashes are recorded and rechecked before writing. Dry-run writes nothing. No model inference, fitting, training or GPU imports occur.
