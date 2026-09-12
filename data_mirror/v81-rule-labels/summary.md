# V81 selection-confirmation labels

Generated on CPU from the sealed `results/v78-rule-confirm/compare.json` and freezes. All four states × 17 budgets are retained per objective. No fitting, measurements, GPU access, V78 edits, or commits are performed.

## Candidate-set coverage: definitions and verbatim code

In `tab:rule-confirm-candidate-sizes`, **Method coverage** (now **Set contains oracle method**) is the fraction of the 68 state–budget cells whose heuristic `candidate_methods` contains the measured oracle's method. **Exact config coverage** (now **Set contains oracle configuration**) is the fraction whose `candidate_config_ids` contains the measured oracle's exact configuration ID. The configuration set contains one predicted best configuration per included method. Both columns describe the **no-clear-winner candidate set**, not the single chosen configuration.

The denominator is **all 68 cells, including singleton sets**, as in V78 and V80. It is not restricted to cells flagged no-clear-winner. The separate `method_coverage_no_clear_winner` statistic uses only flagged cells. On this panel that flag holds exactly when set size ≥2. The best predicted method is always included; each additional method is included when its predicted gap is strictly below the maximum of the two frozen development LOSO MAEs. These are retrospective heuristic sets, not calibrated uncertainty.

`analysis/v64_selection_feasible.py:450–474` (verbatim):

```text
450: def ambiguity(feasible, objective, dense, maes):
451:     """V60's retrospective no-clear-winner heuristic; not calibrated uncertainty."""
452:     if not feasible:
453:         return {"no_clear_winner_heuristic": None, "candidate_methods": [], "methods": [],
454:                 "best_two_gap": None, "threshold": None}
455:     winners = [choose([q for q in feasible if q["method"] == m], objective, dense)
456:                for m in METHODS if any(q["method"] == m for q in feasible)]
457:     winners.sort(key=lambda q: (score(q, objective, dense, "predicted"), *tie_key(q)))
458:     best = winners[0]
459:     records = []
460:     candidates = [best["method"]]
461:     for q in winners:
462:         gap = score(q, objective, dense, "predicted") - score(best, objective, dense, "predicted")
463:         threshold = max(maes[best["law"]]["mae"][objective], maes[q["law"]]["mae"][objective])
464:         if q is not best and gap < threshold:
465:             candidates.append(q["method"])
466:         records.append({"method": q["method"], "config_id": q["id"], "law": q["law"],
467:                         "predicted_score": score(q, objective, dense, "predicted"),
468:                         "gap_from_best": gap, "mae_threshold": threshold})
469:     # The flag is exactly the specified best-two-method comparison. Candidate
470:     # membership extends the same pairwise rule to every remaining method.
471:     no_winner = len(records) > 1 and records[1]["gap_from_best"] < records[1]["mae_threshold"]
472:     return {"no_clear_winner_heuristic": no_winner, "candidate_methods": candidates,
473:             "methods": records, "best_two_gap": records[1]["gap_from_best"] if len(records) > 1 else None,
474:             "threshold": records[1]["mae_threshold"] if len(records) > 1 else None}
```

`analysis/v78_rule_confirm.py:360–361` (verbatim):

```text
360:                         "candidate_config_ids": [r["config_id"] for r in ambiguity["methods"]
361:                                                  if r["method"] in ambiguity["candidate_methods"]]})
```

`analysis/v78_rule_confirm.py:543–547` (verbatim):

```text
543:                 sets[name] = {"candidate_methods": entry["candidate_methods"],
544:                     "candidate_config_ids": entry["candidate_config_ids"],
545:                     "no_clear_winner_heuristic": entry["no_clear_winner_heuristic"],
546:                     "oracle_method_covered": oracle["method"] in entry["candidate_methods"],
547:                     "oracle_config_covered": oracle["id"] in entry["candidate_config_ids"]}
```

`analysis/v78_rule_confirm.py:567–570` (verbatim):

```text
567:             coverage[cap][policy] = {"n_cells": len(sets), "n_no_clear_winner": len(ambiguous),
568:                 "method_coverage": v64.mean_or_none(r["oracle_method_covered"] for r in sets),
569:                 "config_coverage": v64.mean_or_none(r["oracle_config_covered"] for r in sets),
570:                 "method_coverage_no_clear_winner": v64.mean_or_none(r["oracle_method_covered"] for r in ambiguous)}
```

V80's aggregation, retained verbatim in the updated generator:

`analysis/v80_rule_addenda.py:254–258` (verbatim):

```text
254:                 "oracle_method_covered": sum(r["oracle_method_covered"] for r in candidates),
255:                 "oracle_config_covered": sum(r["oracle_config_covered"] for r in candidates),
256:                 "method_coverage": float(np.mean([r["oracle_method_covered"] for r in candidates])),
257:                 "config_coverage": float(np.mean([r["oracle_config_covered"] for r in candidates])),
258:                 "method_coverage_no_clear_winner": float(np.mean([r["oracle_method_covered"] for r in ambiguous]))}
```

| Objective | Policy | Set contains oracle method | Set contains oracle configuration |
|---|---|---:|---:|
| Math | Frozen selection rule | 68/68 (100.0%) | 68/68 (100.0%) |
| Math | Source-conditioned predictor | 68/68 (100.0%) | 39/68 (57.4%) |
| Code | Frozen selection rule | 68/68 (100.0%) | 37/68 (54.4%) |
| Code | Source-conditioned predictor | 68/68 (100.0%) | 23/68 (33.8%) |
| QA (2Wiki) | Frozen selection rule | 68/68 (100.0%) | 18/68 (26.5%) |
| QA (2Wiki) | Source-conditioned predictor | 66/68 (97.1%) | 23/68 (33.8%) |
| Multi | Frozen selection rule | 68/68 (100.0%) | 54/68 (79.4%) |
| Multi | Source-conditioned predictor | 68/68 (100.0%) | 62/68 (91.2%) |

## Single chosen configuration: exact agreement ≤ method agreement

For each feasible cell, exact agreement is `choice['config_id'] == row['oracle_id']`; method agreement is `choice['method'] == row['oracle_method']`. Exact agreement implies method agreement in every cell. Counts and percentages below use the same feasible-cell denominator for each pair, and method agreement reproduces `compare.json`'s table statistic. **The inequality holds for every objective and every policy in compare.json**, including the three additional source-conditioned method-only variants. The six named policies follow; all nine are recorded in `summary.json`.

| Objective | Policy | Exact-configuration agreement | Method agreement |
|---|---|---:|---:|
| Math | Frozen selection rule | 67/68 (98.5%) | 67/68 (98.5%) |
| Math | Source-conditioned predictor | 38/68 (55.9%) | 67/68 (98.5%) |
| Math | Quantization-only | 64/68 (94.1%) | 64/68 (94.1%) |
| Math | Pruning-only | 0/36 (0.0%) | 0/36 (0.0%) |
| Math | Distillation-only | 1/17 (5.9%) | 1/17 (5.9%) |
| Math | Cheapest feasible | 1/68 (1.5%) | 50/68 (73.5%) |
| Code | Frozen selection rule | 36/68 (52.9%) | 65/68 (95.6%) |
| Code | Source-conditioned predictor | 23/68 (33.8%) | 68/68 (100.0%) |
| Code | Quantization-only | 35/68 (51.5%) | 66/68 (97.1%) |
| Code | Pruning-only | 0/36 (0.0%) | 0/36 (0.0%) |
| Code | Distillation-only | 1/17 (5.9%) | 1/17 (5.9%) |
| Code | Cheapest feasible | 1/68 (1.5%) | 51/68 (75.0%) |
| QA (2Wiki) | Frozen selection rule | 11/68 (16.2%) | 52/68 (76.5%) |
| QA (2Wiki) | Source-conditioned predictor | 12/68 (17.6%) | 42/68 (61.8%) |
| QA (2Wiki) | Quantization-only | 3/68 (4.4%) | 33/68 (48.5%) |
| QA (2Wiki) | Pruning-only | 11/36 (30.6%) | 25/36 (69.4%) |
| QA (2Wiki) | Distillation-only | 4/17 (23.5%) | 10/17 (58.8%) |
| QA (2Wiki) | Cheapest feasible | 4/68 (5.9%) | 43/68 (63.2%) |
| Multi | Frozen selection rule | 54/68 (79.4%) | 68/68 (100.0%) |
| Multi | Source-conditioned predictor | 60/68 (88.2%) | 66/68 (97.1%) |
| Multi | Quantization-only | 49/68 (72.1%) | 63/68 (92.6%) |
| Multi | Pruning-only | 0/36 (0.0%) | 0/36 (0.0%) |
| Multi | Distillation-only | 1/17 (5.9%) | 1/17 (5.9%) |
| Multi | Cheapest feasible | 1/68 (1.5%) | 49/68 (72.1%) |

## Regenerated artifacts and checks

- `paper/paper/figs/rule_maps_main.pdf`: original four-panel layout, exact title `QA (2Wiki)`, saved with `bbox_inches='tight'` and `pad_inches=0.04` to retain edge ticks and row labels.
- `paper/paper/figs/rule_maps_full.pdf`: row suffixes `frozen` and `source-cond.`; all oracle markers and no-clear-winner hatching retained.
- `paper/paper/figs/rule_regret.pdf`: logarithmic axis, 16 bars, each mean printed above its bar with three significant digits; numeric labels rotated to fit within 3.2 inches. Bar tops and annotations use the unchanged means over the 68 cells.
- `paper/paper/tables/rule_confirm.tex` and `rule_confirm_by_state.tex`: reader-facing policy names and explicit candidate-set coverage definitions; internal policy names appear only in table footnotes. Numeric entries and candidate-set size counts remain unchanged.
- V78 predictions, selections, oracles, regrets, candidate sets, seals, input hashes, and directory inventory are verified. Figure text extents, legend separation, annotation counts, label overlaps, and saved regret width are checked in `validation.json`.
- `analysis/v80_rule_addenda.py` is updated in place and mirrored byte-for-byte to `paper/code/analysis/v80_rule_addenda.py`. Reproduce with `python -B analysis/v80_rule_addenda.py`. PDF/PNG previews and both table files are also saved under `results/v81-rule-labels/`; SHA256 provenance is in `manifest.json`.

V78 verdicts are unchanged: Math, Code, and QA confirmed; Multi retrospective. QA is restricted to 2Wiki. The two historical distillation students remain excluded from the prediction fits (7 development students).
