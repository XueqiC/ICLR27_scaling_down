# V80 rule-confirmation addenda

Generated on CPU from `results/v78-rule-confirm/compare.json` and `freeze.json`. The independent freeze, original source code and historical measurements are read only for verification. All 4 states × 17 budgets are retained for each objective; no models are fitted or measured.

## 1. Multi-capability objective

The coded objective is **max_c [L_c(M) − L_c(M0)]**, where M0 is the source dense model, not max_c L_c(M). Selection minimizes this maximum increase over feasible candidates. The individual objectives use absolute deployed losses. KD predictions add the predicted change to the student's dense loss, before the multi objective subtracts the source dense loss.

`analysis/final_rule.py:14–19` (verbatim):

```text
14: All returns are absolute deployed losses in native-token nats. Pruning/RTN
15: add a development response to source L0; KD adds it to STUDENT dense loss.
16: Group interpolation uses v69's frozen boundary rule: adjacent log-coordinate
17: pairs, zero floor only outside the group grid, and no bit extrapolation.
18: Selection is downstream: feasible nominal storage, argmin absolute loss per
19: capability, or max_c(L_c-source_L0c), with v64's candidate-set heuristic.
```

`analysis/v78_rule_confirm.py:74–78` (verbatim):

```text
74:     "The max map uses max_c(L_c-source_L0c), as in v64. Quant-only pools channel and "
75:     "grouped RTN and uses locked predictions; its v64-prediction variant is also reported. "
76:     "KD candidates reuse two historical v39 outcomes, not new independent KD experiments. "
77:     "Both students are purged from both KD fits, as in v64. Constants are v39's arithmetic "
78:     "mean-delta baseline on the remaining development students; math uses its +D0 linear form. "
```

`analysis/v64_selection_feasible.py:407–410` (verbatim):

```text
407: def score(q, objective, dense, field):
408:     if objective == "multi":
409:         return max(q[field][c] - dense[c] for c in CAPS)
410:     return q[field][objective]
```

`analysis/v64_selection_feasible.py:442–447` (verbatim):

```text
442: def tie_key(q):
443:     return METHODS.index(q["method"]), q["r"], q["id"]
444: 
445: 
446: def choose(configs, objective, dense, field="predicted"):
447:     return min(configs, key=lambda q: (score(q, objective, dense, field), *tie_key(q)))
```

`analysis/v78_rule_confirm.py:524–536` (verbatim):

```text
524:             measured = [{**q, "actual": actuals[tag][q["id"]]} for q in s["configs"]]
525:             feasible = [q for q in measured if q["r"] <= budget + 1e-12]
526:             oracle = v64.choose(feasible, cap, s["dense"], "actual")
527:             best = v64.score(oracle, cap, s["dense"], "actual")
528:             lookup = {q["id"]: q for q in feasible}
529:             policies = {"locked-rule": locked["policies"]["MAP"], "v64-law": old["policies"]["MAP"],
530:                         **{p: locked["policies"][p] for p in POLICIES[2:]},
531:                         **{"v64-" + p: old["policies"][p] for p in ("quant-only", "prune-only", "distill-only")}}
532:             results = {}
533:             for policy, choice in policies.items():
534:                 selected = lookup.get(choice["config_id"])
535:                 require(choice["config_id"] is None or selected is not None, "Frozen selection is infeasible")
536:                 regret = v64.score(selected, cap, s["dense"], "actual") - best if selected else None
```

## 2. Quant-only fairness and no fallback

**Quant-only uses the same quantization candidates, the same locked-rule predictor, and the same feasibility rule as the locked-rule map, with no fallback.** Each state has five channel RTN candidates (8, 6, 5, 4, 3 bits) and nine grouped RTN candidates (3, 4, 5 bits × groups 64, 128, 256). Both maps first restrict candidates by `r <= budget + 1e-12`. Quant-only then filters that pool to method `quant` and uses the identical objective and tie rule. An empty pool yields `None` and `INFEASIBLE`; no other method or dense fallback is substituted. The headline `quant-only` comes from the locked map; `v64-quant-only` is reported separately.

`analysis/v78_rule_confirm.py:127–143` (verbatim):

```text
127: def candidate_grid(tag):
128:     require(tag in TAGS, f"Unregistered panel state: {tag}")
129:     source = v64.state_info(tag, prune)
130:     def config(method, law, key, ratio, **extra):
131:         return {"id": f"{method}:{key}", "method": method, "law": law, "config": key,
132:                 "r": float(ratio), "target_state": tag, **extra}
133:     configs = [config("dense", "dense", "source", 1.)]
134:     configs += [config("prune", "prune_power", f"d{d:g}", d, d=d) for d in (.9, .8, .7, .6)]
135:     configs += [config("quant", "quant_channel", f"channel_b{b}", b / 16, bit=b) for b in v64.BITS]
136:     configs += [config("quant", "quant_group", f"b{b}_g{g}", (b + 16 / g) / 16,
137:                        bit=b, group_size=g) for b in (3, 4, 5) for g in (64, 128, 256)]
138:     if tag == TAGS[-1]:
139:         for student in STUDENTS:
140:             info = v64.state_info(student, prune)
141:             configs.append(config("distill", "distill_linear", student, info["N0"] / source["N0"],
142:                                   target_state=student))
143:     return {**source, "state_status": "new_stage", "configs": configs}
```

`analysis/v78_rule_confirm.py:309–325` (verbatim):

```text
309: def predict_config(source, q, models, dense=None):
310:     """All candidate/capability entries, including explicit unresolved L0s."""
311:     arm = {"prune_power": "pruning", "quant_channel": "per-channel",
312:            "quant_group": "grouped", "distill_linear": "distillation", "dense": "dense"}[q["law"]]
313:     predictions = {p: {} for p in ("locked-rule", "v64-law")}
314:     for cap in CAPS:
315:         inputs = {**source, **q, "L0": dense[cap] if dense is not None else 0.,
316:                   "models": models["locked"], "qa_distribution": "2Wiki"}
317:         if q["method"] == "distill":
318:             inputs["student"] = models["students"][q["target_state"]]
319:         value = rule.predict(arm, cap, "new_stage", inputs)
320:         independent = q["method"] == "distill"
321:         predictions["locked-rule"][cap] = {"absolute_loss": value if dense is not None or independent else None,
322:             "delta": value - dense[cap] if dense is not None and not independent else
323:                      (value if not independent else value - inputs["student"]["dense"][cap]),
324:             "anchor": "student_dense" if independent else "source_dense",
325:             "status": "FROZEN" if dense is not None or independent else "WAITING_FOR_DENSE"}
```

`analysis/v78_rule_confirm.py:338–361` (verbatim):

```text
338: def build_maps(states, maes):
339:     maps = {policy: {c: [] for c in OBJECTIVES} for policy in ("locked-rule", "v64-law")}
340:     for policy in maps:
341:         for source in states:
342:             configs = [{**q, "predicted": {c: q["predictions"][policy][c]["absolute_loss"] for c in CAPS}}
343:                        for q in source["configs"]]
344:             require(all(q["predicted"][c] is not None for q in configs for c in CAPS), "Maps require dense anchors")
345:             for cap in OBJECTIVES:
346:                 for budget in BUDGETS:
347:                     feasible = [q for q in configs if q["r"] <= budget + 1e-12]
348:                     selected = {"MAP": v64.choose(feasible, cap, source["dense"]) if feasible else None,
349:                                 "cheapest": min(feasible, key=lambda q: (q["r"], *v64.tie_key(q))) if feasible else None}
350:                     for m in ("quant", "prune", "distill"):
351:                         within = [q for q in feasible if q["method"] == m]
352:                         selected[m + "-only"] = v64.choose(within, cap, source["dense"]) if within else None
353:                     ambiguity = v64.ambiguity(feasible, cap, source["dense"], maes)
354:                     maps[policy][cap].append({"state": source["tag"], "budget": budget,
355:                         "n_feasible": len(feasible), "policies": {name: {
356:                             "status": "FEASIBLE" if q else "INFEASIBLE", "config_id": q["id"] if q else None,
357:                             "method": q["method"] if q else None, "r": q["r"] if q else None,
358:                             "predicted_score": v64.score(q, cap, source["dense"], "predicted") if q else None}
359:                             for name, q in selected.items()}, **ambiguity,
360:                         "candidate_config_ids": [r["config_id"] for r in ambiguity["methods"]
361:                                                  if r["method"] in ambiguity["candidate_methods"]]})
```

`analysis/v78_rule_confirm.py:529–531` (verbatim):

```text
529:             policies = {"locked-rule": locked["policies"]["MAP"], "v64-law": old["policies"]["MAP"],
530:                         **{p: locked["policies"][p] for p in POLICIES[2:]},
531:                         **{"v64-" + p: old["policies"][p] for p in ("quant-only", "prune-only", "distill-only")}}
```

## 3. Distillation provenance and prediction-fit membership

**The pythia-1b@step64000 KD candidates are the historical V39 students pythia-160m@step64000 and pythia-410m@step64000, but their post-training losses did not enter the V78 KD prediction fits supplied to final_rule.predict. They are not in-sample outcomes for those predictions.** Both students belong to the original nine-student V39 cohort. V78 reconstructs that cohort, excludes both selectable students from the V39 +D0 linear fit used for Math, and excludes them from the mean-delta baselines used for Code and QA. The V64 KD fits also exclude both students and the source. Seven development students remain. `final_rule.py` evaluates the supplied fit objects; it does not load a global V39 fit. The two historical `post_training` outcomes are reused only when scoring these candidate endpoints. Their teacher was `gpt-5.6-luna` (run `gpt-5.6-luna_full_600_lora`), not the 1B source; these are reused alternatives, not fresh KD experiments trained from that source. Original V39 cohort membership and reuse in development analyses do not make this a fully independent prospective KD experiment.

`analysis/v39_distill_controlled.py:23–40` (verbatim):

```text
23: SIZES = ("160m", "410m", "1.4b")
24: STEPS = (16000, 64000, 143000)
25: CAPS = ("math", "code", "qa")
26: RUN = "gpt-5.6-luna_full_600_lora"
27: 
28: def load_cells():
29:     rows = []
30:     for size, step in itertools.product(SIZES, STEPS):
31:         tag = f"pythia-{size}--step{step}"
32:         p = ROOT / f"results/v12-distill/{tag}/{RUN}/eval.json"
33:         if not p.exists():
34:             raise FileNotFoundError(f"missing distill cell: {p}")
35:         e = json.loads(p.read_text())
36:         assert e["training_mode"] == "lora", f"{tag} not lora"
37:         for cap in CAPS:
38:             rows.append({"size": size, "step": step, "cap": cap,
39:                          "N0": v36.matrix_n0(size), "D0": step * v36.TOKENS_PER_STEP,
40:                          "L0": e["dense"][cap], "delta": e["delta"][cap]})
```

`analysis/v78_rule_confirm.py:49–51` (verbatim):

```text
49: TAGS = ("pythia-160m@step32000", "pythia-410m@step32000",
50:         "pythia-1.4b@step32000", "pythia-1b@step64000")
51: STUDENTS = ("pythia-160m@step64000", "pythia-410m@step64000")
```

`analysis/v78_rule_confirm.py:266–293` (verbatim):

```text
266:     require(len(old["students"]) == reference["panel"]["n_cells"], "Incomplete v39 cohort")
267:     for s in old["students"].values():
268:         require(s["tag"] not in TAGS, "Confirmation student in development")
269:         for cap in CAPS:
270:             drows.append({**v64.state_info(s["tag"], prune), "cap": cap, "L0": s["dense"][cap],
271:                           "delta": s["delta"][cap]})
272:     modules = (prune, group, v36, distill, None)
273:     fold = v64.fit_fold(candidate_grid(TAGS[-1]), prows, qrows, grows, drows, modules)
274:     constant = {c: float(np.mean([r["delta"] for r in drows if r["cap"] == c and r["tag"] not in STUDENTS]))
275:                 for c in CAPS}
276:     locked = {"prune": {"standardization": p["standardization"], "models": p["models"]},
277:               "grouped": {"standardization": g["standardization"], "models": g["models"]},
278:               "channel": {"models": fold["quant_channel"]["models"],
279:                           "median": {c: {str(b): float(np.median([r["observed"] for r in qrows
280:                               if r["capability"] == c and r["config"] == b])) for b in v64.BITS} for c in CAPS}},
281:               "distill": {"linear": fold["distill_linear"]["models"], "constant": constant,
282:                           "excluded_students": list(STUDENTS)}}
283:     students = {t: {k: s[k] for k in ("tag", "N0", "D0", "dense", "path", "teacher")}
284:                 for t, s in old["students"].items() if t in STUDENTS}
285:     for s in students.values():
286:         path = ROOT / s["path"]
287:         hashes.update(hash_inputs([path]))
288:         require(hashes[label(path)] == old["input_sha256"][s["path"]], "Reused student artifact changed")
289:     check_hashes(hashes)
290:     return {"locked": locked, "v64": fold, "students": students,
291:             "maes": old["law_loso_mae"], "development_roster": sorted(set(rosters)), "input_sha256": hashes,
292:             "probe_sha256": g["measurement_protocol"]["probe_sha256"],
293:             "fit_rule": "Delivered V53 and V69 fits; V64 frozen paired rows; v39 excludes both eligible students",
```

`analysis/v64_selection_feasible.py:365–371` (verbatim):

```text
365:     if drows:
366:         purged = {tag} | {q["target_state"] for q in source["configs"] if q["method"] == "distill"}
367:         dtrain = [r for r in drows if r["tag"] not in purged]
368:         df = {c: distill._fit([r for r in dtrain if r["cap"] == c], True) for c in CAPS}
369:         result["distill_linear"] = {"models": df, "excluded_states": sorted(purged),
370:                                     "train_states": sorted({r["tag"] for r in dtrain}),
371:                                     "n_train_students": len(dtrain) // 3}
```

`analysis/final_rule.py:60–65` (verbatim):

```text
60: def predict(arm, capability, state_status, inputs):
61:     """Return the absolute loss under the docstring's locked table.
62: 
63:     This pure function only evaluates supplied development-fitted objects;
64:     it does not read outcomes, refit, change inputs, or mutate module state.
65:     """
```

`analysis/final_rule.py:75–87` (verbatim):

```text
75:     elif arm == "distillation":
76:         if state_status not in NEW:
77:             raise ValueError("Locked KD rule requires a new source")
78:         s = inputs["student"]
79:         if s["N0"] >= inputs["N0"] or s["D0"] != inputs["D0"]:
80:             raise ValueError("KD candidate must be smaller and at the same stage")
81:         if c == "qa" and inputs.get("qa_distribution") != "2Wiki":
82:             raise ValueError("Locked KD QA is restricted to 2Wiki")
83:         anchor = float(s["dense"][c])
84:         f = models["distill"]
85:         delta = (float(distill._predict(f["linear"][c], [{**s, "L0": anchor}])[0])
86:                  if c == "math" else float(f["constant"][c]))
87:         value = anchor + delta
```

`analysis/v78_rule_confirm.py:668–673` (verbatim):

```text
668:         for q in candidate_grid(tag)["configs"]:
669:             if q["method"] == "distill":
670:                 path = ROOT / initial["models"]["students"][q["target_state"]]["path"]
671:                 ev = json.loads(path.read_bytes())
672:                 require(ev["student"] == q["target_state"] and ev["training_mode"] == "lora", "Student identity changed")
673:                 actuals[tag][q["id"]] = prune.checked_losses(ev["post_training"], path)
```

Frozen evidence: `freeze-independent.json` → `models.locked.distill.excluded_students` contains both alternatives; `models.v64.distill_linear.train_states` contains the seven students below, and its models equal `models.locked.distill.linear`. The frozen constants were checked against the arithmetic mean of the same seven development deltas.

- `pythia-1.4b@step143000`
- `pythia-1.4b@step16000`
- `pythia-1.4b@step64000`
- `pythia-160m@step143000`
- `pythia-160m@step16000`
- `pythia-410m@step143000`
- `pythia-410m@step16000`

## Per-state regret

Regret is in nats, averaged over exactly 17 budgets. K counts distinct configuration IDs, not methods.

| State | Objective | Locked-rule | V64-law | Quant-only | Cheapest | K |
|---|---|---:|---:|---:|---:|---:|
| pythia-160m@step32000 | math | 0.000000 | 0.000512 | 0.000035 | 2.902545 | 7 |
| pythia-410m@step32000 | math | 0.000000 | 0.028442 | 0.000009 | 1.607308 | 7 |
| pythia-1.4b@step32000 | math | 0.000024 | 0.030380 | 0.000000 | 2.248675 | 7 |
| pythia-1b@step64000 | math | 0.000000 | 0.000434 | 0.026970 | 0.327902 | 7 |
| pythia-160m@step32000 | code | 0.000000 | 0.000000 | 0.000035 | 3.767303 | 7 |
| pythia-410m@step32000 | code | 0.001014 | 0.410475 | 0.000923 | 2.053675 | 7 |
| pythia-1.4b@step32000 | code | 0.011972 | 0.013315 | 0.012025 | 3.526863 | 7 |
| pythia-1b@step64000 | code | 0.004321 | 0.025644 | 0.027814 | 0.390606 | 7 |
| pythia-160m@step32000 | qa | 0.220466 | 0.141708 | 0.126440 | 1.882328 | 4 |
| pythia-410m@step32000 | qa | 0.046997 | 0.183243 | 0.216010 | 1.459195 | 4 |
| pythia-1.4b@step32000 | qa | 0.244136 | 0.523932 | 0.387497 | 2.013168 | 4 |
| pythia-1b@step64000 | qa | 0.052428 | 0.005406 | 0.280362 | 0.052428 | 1 |
| pythia-160m@step32000 | multi | 0.001169 | 0.003603 | 0.001204 | 3.764622 | 7 |
| pythia-410m@step32000 | multi | 0.002180 | 0.001677 | 0.002334 | 2.050533 | 7 |
| pythia-1.4b@step32000 | multi | 0.004743 | 0.002122 | 0.005323 | 3.507343 | 7 |
| pythia-1b@step64000 | multi | 0.000000 | 0.000000 | 0.023494 | 0.382624 | 7 |

## Candidate-set sizes and coverage

Each size distribution uses all 68 cells, including singleton sets. On this panel, the no-clear-winner flag is exactly equivalent to size ≥2. The JSON also includes percentages and size counts conditional on that flag. **V64 Code has one size-4 set** at pythia-1b@step64000, budget 1.00: quant, dense, prune, distill. Keeping this fourth column avoids silently dropping a cell. Exact configuration coverage compares the oracle with the predicted best configuration of each included method.

| Objective | Policy | Size 1 | Size 2 | Size 3 | Size 4 | Method coverage | Config coverage | No clear winner |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| math | locked-rule | 30 | 28 | 10 | 0 | 68/68 (100.0%) | 68/68 (100.0%) | 38 |
| math | v64-law | 31 | 27 | 10 | 0 | 68/68 (100.0%) | 39/68 (57.4%) | 37 |
| code | locked-rule | 30 | 28 | 10 | 0 | 68/68 (100.0%) | 37/68 (54.4%) | 38 |
| code | v64-law | 30 | 26 | 11 | 1 | 68/68 (100.0%) | 23/68 (33.8%) | 38 |
| qa | locked-rule | 24 | 32 | 12 | 0 | 68/68 (100.0%) | 18/68 (26.5%) | 44 |
| qa | v64-law | 26 | 30 | 12 | 0 | 66/68 (97.1%) | 23/68 (33.8%) | 42 |
| multi | locked-rule | 30 | 28 | 10 | 0 | 68/68 (100.0%) | 54/68 (79.4%) | 38 |
| multi | v64-law | 32 | 26 | 10 | 0 | 68/68 (100.0%) | 62/68 (91.2%) | 36 |

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

## Figures and verification

- `paper/paper/figs/rule_maps_main.pdf`: 5.5 × 2.1 in (the paper style defines a 5.5-in text width at `paper/paper/iclr2027_conference.sty:49`); four 4 × 17 panels, locked rule only, method colours and white circles only where the oracle method differs. No hatching in this view.
- `paper/paper/figs/rule_regret.pdf`: 2.95 × 2.6 in, all 68 cells per objective, logarithmic regret axis. Bar tops show the means; all bars start at the displayed 10⁻⁶-nat axis floor. Budget cells share states and are not independent replicates; no error bars are inferred.
- `paper/paper/figs/rule_maps_full.pdf`: 5.5 × 4.8 in, two map rows per state, all oracle symbols and the original no-clear-winner hatching. QA is titled `QA (2Wiki)`. Legends are below the panels, separate from titles and axis labels.
- All figures use V64's shared Times-metric bold fonts (at least 8 pt), method palette, white cell grids, PDF fonttype 42, deterministic metadata and 300-dpi PNG previews. The full view preserves V78's oracle symbol shapes. Preview PNGs are under `results/v80-rule-addenda/figs/`.
- `paper/paper/tables/rule_confirm_by_state.tex` contains two `[H]` tables, with six decimals for regret so that the small nonzero Math regret remains visible.
- All recorded V78 input hashes and freeze seals checked; locked predictions, feasible selections, measured oracles, regrets and candidate sets reproduced without fitting. Aggregate values match `compare.json`; figure extents, font sizes, marker counts, hatching counts and legend separation are checked in `validation.json`.
- The full V78 directory inventory and input hashes were unchanged after generation. Only the authorized outputs and script mirror are written. No commit is made.

The existing V78 verdict remains Math, Code and QA confirmed; Multi retrospective (locked mean regret exceeds V64). Candidate-set coverage is a retrospective heuristic using frozen V64 development LOSO MAE thresholds, not calibrated uncertainty.

Reproduce with `python -B analysis/v80_rule_addenda.py` or the identical paper code mirror. Exact input/output SHA256 values are recorded in `manifest.json`.
