# Loss-constrained nominal compression selection (V33)

**OFFLINE REPLAY — not prospective.** Saved predictions select a method/configuration; actual measured capability losses score that selection. No fitting, target compressed calibration, threshold tuning, or safety-margin tuning occurs in this replay.

The main panel has 12 development models and 2210 budget vectors. Actual constraint satisfaction is **0.70889 [0.61614, 0.79530]**; the signed nominal storage gap against the actual oracle is **0.06677 [0.00144, 0.13242]**. Brackets are paired-model bootstrap 95% intervals.

## Frozen prediction and measured endpoint

Pruning imports V28 `predict_arm` and its saved Mode A development-LOMO coefficients. Quantization imports V30 `predict_candidate` and saved broad-panel development-LOMO shared-eta coefficients. The full-development fits contain these 12 targets, so the main panel must use their already-saved held-out folds. Every mapping excludes the complete target model. If a saved V30 fold/shape is unavailable, the saved V28 quantization dev-LOMO Mode A fit is the explicitly labelled fallback; no new fold is fitted. The shared-eta quantization predictor is a fixed replay choice, not chosen by selection scores; this replay does not establish that it is the best predictor. There are **0 target compressed calibration points**, although dense measurements are inputs and other models supplied historical development outcomes. Shapes and fits remain frozen even when poor or extrapolated.

V28's basic inputs are log non-embedding N0, family and measured dense capability loss; V30 uses log nominal model size parsed from the identifier, family and measured dense capability loss. No target pruning/quantization loss or target fitted amplitude is an input. V28 pruning evaluates a_c(x0)((1-density)/0.3)^gamma_c; V30 quantization evaluates a_c(x0) g(bits)/g(4), with g(b)=2^(-eta(b-4))-2^(-eta(16-4)). All a mappings, gamma and eta values come from the saved appropriate fold.

The common reference is each source's V6 `prune_losses.json["1.0"]`. For both methods, actual delta_c = measured L_c(config) - L_c(source dense). V10 has its own measurement of that source dense; its prediction is rebased as delta_hat_common = delta_hat_V10 + L_dense,V10 - L_dense,V6. This offset uses dense measurements only. The small differences are exposed below, not silently identified as equal or corrected using compressed outcomes. Native-token CE nats are compared within each model; cross-model averages are descriptive despite different tokenizers.

All measured numeric density/bit settings enter the candidate set, plus the source dense candidate with exact zero delta. Missing configurations are not imputed. Pruning below 0.6 extrapolates the V28 shape fitted on 0.6–0.9; signed predictions and losses, cliffs and infills are retained. Archived copies and the separate 512-probe panel are not mixed in. Each candidate, prediction source, extrapolation flag, measured loss, and exclusion is in `summary.json`.

## Budgets, nominal costs, and scoring

For each model/capability use quantiles [0.0, 0.1, 0.25, 0.5, 0.75, 1.0] of its observed compressed deltas, clip only these budget thresholds at zero, add zero, deduplicate, and take the Cartesian product across math/code/QA. Thus capabilities vary independently and the dense candidate is always feasible. These outcome-informed scenarios make this retrospective, not a prospective budget distribution. Actual outcomes only construct scenarios and score decisions; fixed predictions plus a supplied budget determine the selector.

Minimize **nominal storage**: pruning = density, quantization = bits/16, dense = 1. The auxiliary **nominal active-parameter ratio** is density for pruning and 1 for quantization/dense; it is not the selection objective. These nominal ratios omit sparse indices, packing, scales, untouched tensors and runtime representation. Saved fake-quantized or masked tensors need not occupy these nominal sizes. **No latency claims.** Equal nominal storage breaks ties by method name then configuration id, without measured losses.

For each budget the selector chooses the minimum nominal storage satisfying all predicted delta_hat_c <= tau_c (tolerance 1e-12 nats). The oracle applies the same rule to actual deltas. Always-cheapest ignores budgets; always-dense selects the exact reference. Constraint satisfaction requires every actual capability constraint. The signed nominal gap is C_selected - C_oracle over all decisions. A negative nominal gap can accompany a violation and is not successful compression. The conditional nominal gap reports only actually satisfying decisions.

Wrong exclusion = predicted-infeasible but actually-feasible compressed configuration-budget pairs / actually-feasible compressed configuration-budget pairs. Dense is omitted from this denominator. A budget with no feasible compressed candidates has an undefined fraction, not zero; its zero numerator/denominator still participates correctly in the pooled ratio. This measures predictor exclusion, so it is not assigned to the two trivial policies.

All intervals use 10000 paired whole-model bootstrap draws (seed 330907) through `prediction_audit.bootstrap_means`. Normalize each model to unit budget mass before pooling, retaining all policies/capabilities/configurations together; resample ratio numerators and denominators together. Baseline differences use the very same draws. Intervals condition on frozen fits and fixed observed grids and omit retraining, item/seed and dense-anchor uncertainty. Within-model rows below are descriptive points, not independent-model confidence intervals.

## Main development-LOMO replay

| Policy | Actual constraint satisfaction [95% CI] | Mean nominal storage [95% CI] | Signed nominal gap vs actual oracle [95% CI] | Nominal gap when actually satisfying [95% CI] |
|---|---|---|---|---|
| selector | 0.70889 [0.61614, 0.79530] | 0.55945 [0.50464, 0.61158] | 0.06677 [0.00144, 0.13242] | 0.15289 [0.09126, 0.21184] |
| always_cheapest | 0.00705 [0.00530, 0.00905] | 0.18750 [0.18750, 0.18750] | -0.30518 [-0.35795, -0.25182] | 0.00000 [0.00000, 0.00000] |
| always_dense | 1.00000 [1.00000, 1.00000] | 1.00000 [1.00000, 1.00000] | 0.50732 [0.45455, 0.56068] | 0.50732 [0.45455, 0.56068] |

The advisor's four evidence columns are below. Baseline improvements distinguish constraint satisfaction from nominal savings; neither is a scalar utility that rewards violating a budget.

| Response amplitude context | Improvement vs always-cheapest and always-dense | Calibration points used | Which candidate excluded |
|---|---|---|---|
| Mean absolute actual compressed delta 4.99725 [3.26070, 6.73312] nats | vs always_cheapest: satisfaction gain 0.70184 [0.60988, 0.78781], nominal storage saving -0.37195 [-0.42408, -0.31714]; vs always_dense: satisfaction gain -0.29111 [-0.38386, -0.20470], nominal storage saving 0.44055 [0.38842, 0.49536] | **0** | pruning wrongly excluded 0.45975 [0.28683, 0.60554]; quantization wrongly excluded 0.20947 [0.13551, 0.28228]; method never selected: none; exact configuration IDs and failed capabilities recorded for every budget |

| Method | Mean absolute actual delta, nats [95% CI] | Mean predicted minus actual delta, nats [95% CI] | Actually-feasible configurations wrongly excluded [95% CI] |
|---|---|---|---|
| all_compressed | 4.99725 [3.26070, 6.73312] | 3.29185 [0.75606, 5.64178] | 0.32706 [0.21361, 0.42409] |
| pruning | 6.06396 [3.85840, 8.28599] | 4.89077 [1.36163, 8.24971] | 0.45975 [0.28683, 0.60554] |
| quantization | 2.77713 [1.93340, 3.63904] | -0.03762 [-0.73280, 0.59691] | 0.20947 [0.13551, 0.28228] |

Positive mean prediction-minus-actual indicates overprediction of damage on this panel. The wrongly-excluded fraction measures its decision consequence directly; it is not inferred from prediction error alone.

## Method domination on the tested nominal range

Here 'dominated' means never selected on the tested budget/nominal-cost grid, as requested. This is a finite-grid operational finding, not a proof of global Pareto domination. Oracle counts and optimal ties distinguish predictor rejection from actual nominal-cost inferiority. No optimal region is assumed for either compression method.

| Method | Selector budget count (actually satisfying) | Actual oracle budget count | Never selected by selector? | Never actually nominal-cost optimal, including ties? | Selector share [95% CI] |
|---|---|---|---|---|---|
| dense | 674 (674) | 420 | False | False | 0.31554 [0.21529, 0.40655] |
| pruning | 31 (0) | 81 | False | False | 0.00787 [0.00000, 0.02302] |
| quantization | 1505 (809) | 1709 | False | False | 0.67660 [0.58619, 0.77785] |

Pruning is never selected for 10 of the 12 development models. Of its 31 selected budgets, 0 actually satisfy all constraints. Selection alone therefore does not establish a successful optimal region.

| Source model | Budgets | Actual satisfaction | Signed nominal gap | Feasible pruning wrongly excluded | Feasible quantization wrongly excluded | Methods never selected |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 144 | 0.52083 | -0.03056 | 0.19850 | 0.08462 | none |
| Qwen3-1.7B | 75 | 0.78667 | 0.24750 | 0.43813 | 0.38498 | pruning |
| Qwen3-4B | 75 | 0.86667 | 0.21350 | 0.78862 | 0.21311 | pruning |
| gemma3-12b | 245 | 0.55510 | -0.14107 | 0.22024 | 0.04511 | dense, pruning |
| gemma3-1b | 252 | 0.88492 | 0.09623 | 0.23607 | 0.10956 | pruning |
| gemma3-270m | 343 | 0.38776 | -0.07642 | 0.01829 | 0.00000 | none |
| gemma3-27b | 168 | 0.91071 | 0.17374 | 0.04848 | 0.30462 | pruning |
| gemma3-4b | 245 | 0.71429 | -0.02066 | 0.34444 | 0.15285 | pruning |
| gemma4-31b | 216 | 0.60185 | 0.13571 | 0.19588 | 0.19241 | pruning |
| muse-30b | 180 | 0.68333 | 0.16146 | 0.47489 | 0.33651 | pruning |
| olmo3-32b | 147 | 0.72789 | 0.02883 | 0.81152 | 0.43558 | pruning |
| olmo3-7b | 120 | 0.86667 | 0.01302 | 0.70968 | 0.15472 | dense, pruning |

## Reference and measurement caveats

| Model | V10 dense minus common source dense (math / code / QA), nats | Pruning / quantization configuration counts | Extrapolated configurations |
|---|---|---|---|
| Qwen3-0.6B | 0.0001307 / -0.0013002 / -0.0063999 | 10 / 5 | 6 |
| Qwen3-1.7B | 0.0003267 / 0.0004334 / -0.0027675 | 15 / 5 | 8 |
| Qwen3-4B | -0.0012881 / 0.0000000 / 0.0049477 | 10 / 5 | 6 |
| gemma3-12b | -0.0007672 / -0.0004429 / -0.0042019 | 10 / 5 | 6 |
| gemma3-1b | 0.0000000 / 0.0000000 / 0.0000000 | 16 / 5 | 6 |
| gemma3-270m | 0.0000000 / 0.0000000 / 0.0000000 | 10 / 5 | 6 |
| gemma3-27b | 0.0000000 / 0.0000000 / 0.0000000 | 10 / 5 | 6 |
| gemma3-4b | 0.0000284 / 0.0005536 / -0.0029258 | 10 / 5 | 6 |
| gemma4-31b | 0.0000000 / 0.0000000 / 0.0000000 | 10 / 5 | 6 |
| muse-30b | 0.0000000 / 0.0000000 / 0.0000000 | 10 / 5 | 6 |
| olmo3-32b | 0.0000000 / 0.0000000 / 0.0000000 | 10 / 5 | 6 |
| olmo3-7b | -0.0001013 / -0.0004342 / -0.0025980 | 10 / 5 | 6 |
| Qwen3-8B | 0.0001867 / 0.0000000 / 0.0035747 | 4 / 5 | 1 |

Qwen3-0.6B's saved 5-bit metadata records an odd-half protocol discrepancy (reported dense-gap QA +0.0133 nats). It remains included and flagged; no invented correction is applied. Infill metadata is preserved in the JSON. Aggregate loss files cannot verify item identities or estimate measurement uncertainty, so tight-budget outcomes inherit these limits.

## Separate offline transfer and excluded candidates

Legacy Qwen3-8B aggregate tables do not establish Base versus post-trained identity. This separate offline transfer is not validation of the named Qwen/Qwen3-8B-Base V28 target.

Qwen3-8B: 72 budgets, V28 frozen full-development Mode A coefficients for both methods, with zero target compressed calibration. V30 contains no full-development basic-input mapping for this target, so the available V28 frozen quantization mapping is used. This result is kept separate from the 12-model LOMO aggregate. Its one-model bootstrap intervals are degenerate and carry no across-model uncertainty.

| Response amplitude context | Improvement vs always-cheapest and always-dense | Calibration points used | Which candidate excluded |
|---|---|---|---|
| 1.48895 [1.48895, 1.48895] nats | vs always_cheapest: satisfaction gain 0.84722 [0.84722, 0.84722], nominal storage saving -0.58507 [-0.58507, -0.58507]; vs always_dense: satisfaction gain -0.13889 [-0.13889, -0.13889], nominal storage saving 0.22743 [0.22743, 0.22743] | **0** | pruning: wrongly excluded 0.70732 [0.70732, 0.70732]; quantization: wrongly excluded 0.62617 [0.62617, 0.62617] |

Actual satisfaction: 0.86111 [0.86111, 0.86111]; signed nominal gap: 0.32378 [0.32378, 0.32378]. Methods never selected: pruning.

**Distillation excluded:** V12 clean student-own-dense adaptation deltas do not supply a paired source-to-student loss/cost table relative to these source dense models; commercial teachers and student sizes are not interchangeable source checkpoints.

- `results/v6-capability-geometry/Qwen--Qwen3-14B/prune_losses.json`: No complete paired panel with supported frozen basic-input prediction.
- `results/v6-capability-geometry/Qwen3-14B/prune_losses.json`: No complete paired panel with supported frozen basic-input prediction.

Qwen3-14B currently has pruning results but no paired quantization loss table and no supported saved basic-input prediction panel; it cannot support the same cross-method comparison. Duplicate aliases are not additional models.

## Reproduction and artifacts

```bash
python analysis/v33_loss_constrained_selection.py --bootstrap 10000
python analysis/v33_loss_constrained_selection.py --dry-run --bootstrap 10000
python -m pytest -q tests/test_v33.py
```

`results/v33-selection/summary.json` contains candidates, budget axes, every decision, actual oracle, excluded configuration IDs with failing predicted capabilities, feasible/wrongly-excluded counts, per-model sufficient statistics, paired intervals, and input/code SHA-256 hashes. The V28 payload seal is validated; all read inputs are hashed and rechecked before writing. The dry run writes nothing. Execution is CPU-only and loads no model or GPU framework.
