# C17 scope audit — V34

**Verdict: scope_closed_with_provenance_limits; branch (b), training-benchmark reuse.** The observed C17 effect belongs to the old mixed QA panel. Current distillation still uses HotpotQA training data, so a literal ‘old grid only, no current data dependence’ verdict is excluded. The current default measurement uses 2Wiki, and the saved questions show no exact overlap with current traces or the old HotpotQA panel. This is a scope closeout with explicit provenance limits, not a blanket cleanliness certificate. Keep C17 and all original results. No model runs were performed.

## Evidence and mechanism

Entry points: [prior inventory](../../notes/prior_inventory.md), [IRT audit §c/§h](../../notes/review/irt_audit.md), [V3 source](../../analysis/v3_measurement_audit.py), [original V3 result](../../results/v3-measurement-audit/summary.md). The old frozen grid is `irt-grid-fill-locked-2026-07-28/`, with the cell inventory `cells-grid-fill-2026-07-28.csv`. V3's QA evidence specifically reuses the 8B-teacher campaign: 52 respondents (4 dense + 4 recipes × 4 student sizes × 3 seeds), each evaluated on 50 HotpotQA and 50 2Wiki questions. It is not a measurement of contamination across every prior-grid cell.

The audit also verifies all 100 V3 item IDs against the frozen QA panel (355 respondents) and fingerprints its 1065-row cell table. This binds the reanalysis to the old grid instead of relying on directory names.

The old sweep `scaling_pilot_qwen3_hotpotqa.sh` reads the local AFlow/MetaGPT `hotpotqa_test.jsonl`: candidate training slice [0:200], evaluation slice [400:450]. The generator retains teacher responses with F1≥0.5. ‘Trained half’ means the **evaluation half from the training benchmark**, not literal training rows. Old eval JSONs preserve questions and local indices 400–449. Those local integers are not official HotpotQA row IDs; the upstream file and original teacher trace files for these four arms are absent, so complete training lineage cannot be certified. The declared slices are disjoint. The observed split is consistent with benchmark/format fitting; it does not identify duplicate leakage or causally measure the fraction attributable to fitting.

A derived artifact does preserve **150 M0 training questions**: `gold-qwen3-2026-07-21/traces/gold-hotpotqa.jsonl`. The gold sweep explicitly names the old M0 keep-correct trace source, and `make_matched_gold_from_m0.py` copies its user/system messages verbatim while replacing the assistant target. This provides question-level M0 lineage; it does not substitute for the missing original trace bytes or establish M1/M2/M3 membership. These recovered questions are also compared with current training, current measurement, and old HotpotQA evaluation below.

| Recipe | Paired cells | HotpotQA Δacc | 2Wiki Δacc | Difference | Saved 90% CI |
|---|---:|---:|---:|---:|---|
| M0 | 12 | +0.055000 | -0.051667 | +0.106667 | [0.07333333333333335, 0.13999999999999996] |
| M1 | 12 | +0.090000 | -0.043333 | +0.133333 | [0.05999999999999998, 0.20166666666666658] |
| M2 | 12 | +0.198333 | +0.126667 | +0.071667 | [0.02833333333333334, 0.11508333333333329] |
| M3 | 12 | -0.000000 | -0.025000 | +0.025000 | [-0.01833333333333334, 0.06666666666666665] |

These point estimates were recomputed from the raw saved item scores (score≥0.5), matched by recipe/size/seed against each size's dense baseline, and agree with V3 within 1e-12. Original bootstrap intervals are read, not refit. For M2, the benchmark gap / HotpotQA gain is 36.1%, explaining the old ‘~1/3’ shorthand. For an equal-weight 100-item panel, the gain is 0.1625, and excess over the transfer-only gain is 22.1% of that panel gain. Neither ratio is a contaminated-item rate. The original result is preserved; this clarifies its denominator.

The separate [V1 anchor result](../../results/v1-recipe-strat/summary.md) uses the frozen grid's Rasch θ: QA M1 anchor mean −1.051 fails, whereas M0 +0.076/M2 +0.028/M3 +0.017 pass the original 2SE rule; pooling minus M1 passes (+0.043). This is recipe mixing, not a current dataset leak. V3's 1PL/2PL rank agreement is likewise prior-grid evidence, not a current loss calibration.

## Current provenance and overlap

[V12 source](../../analysis/v12_distill.py) declares training GSM8K/CodeAlpaca/HotpotQA and measurement MATH-500/MBPP/2WikiMultihopQA. The actual [V6 builder](../../analysis/v6_capability_geometry.py) loads `framolfese/2WikiMultihopQA`, split `validation`, after math and code draws from a shared seed-0 RNG. V12 takes `[1::2]` **after sampling 128 probes**, yielding 64 QA examples; these are odd sampled positions, not odd dataset rows. V15 saved indices 0–63 are already renumbered measurement examples and are not sliced again.

Inspected 2 current QA trace files (600 rows/teacher in this snapshot), 660 V12 eval files and 16 V16 residual files. Metadata conflicts: 0. 3 saved default V15 panels contain the same ordered 64 QA prompts/references; 32 other V15 artifacts use the easy benchmark suite (QA=TriviaQA). The default panels are dense/pruned evaluations, not saved default-panel distilled behavioral results. Easy-suite behavior must not be described as 2Wiki behavior.

Trace JSONLs contain prompt, response, teacher, model_version, timestamp and usage; no dataset/config/split/row IDs. V12 selects the first n rows per domain before training shuffles, so checking both full 600-row files covers the available smaller current prefixes. Historical trace bytes are not bound by an immutable run manifest. The source declaration alone does not prove HotpotQA's official training split; the optional existing-cache check supplies independent question identification.

Local HotpotQA distractor cache status: **read_existing_cache**. Only existing Arrow streams were read; no download or model/data-loader execution.

- current_training: 600/600 unique questions identified; split matches {'train': 600}.
- old_hotpot_eval: 50/50 unique questions identified; split matches {'validation': 50}.
- old_M0_training: 150/150 unique questions identified; split matches {'validation': 150}.
- current_measurement: 0/64 unique questions identified; split matches {}.

The supplied cache identifies current trace questions as official train rows 0–599, and the old 50 HotpotQA evaluation questions in official validation, with no train matches. The local AFlow filename ‘test’ therefore must not be treated as an official split label. Cache identification does not recover the historical generator revision or original teacher trace membership.

The cache check also queries all 64 current measurement questions against every cached HotpotQA train/validation row, rather than just the selected 600 training questions. Its zero-match result, when available in the coverage table, is exact-question evidence for these 64 items, not a semantic whole-corpus guarantee.

Question comparison uses Unicode NFKC, casefold and whitespace normalization, preserving punctuation. Blank/missing questions give unknown, never a clean zero. Teacher wording and answers are excluded from identity. Different dataset ID namespaces and local integer indices are not treated as proof of disjointness.

| Comparison | Unique questions left / right | Exact matches | Status |
|---|---:|---:|---|
| old_hotpot_eval_vs_current_training | 50 / 600 | 0 | no_exact_question_overlap |
| current_training_vs_current_measurement | 600 / 64 | 0 | no_exact_question_overlap |
| old_hotpot_eval_vs_current_measurement | 50 / 64 | 0 | no_exact_question_overlap |
| old_2wiki_vs_current_measurement | 0 / 64 | None | unknown |
| old_M0_training_vs_current_training | 150 / 600 | 0 | no_exact_question_overlap |
| old_M0_training_vs_current_measurement | 150 / 64 | 0 | no_exact_question_overlap |
| old_M0_training_vs_old_hotpot_eval | 150 / 50 | 0 | no_exact_question_overlap |

Old 2Wiki evals retain task IDs but not questions; current V15 examples retain questions but not source IDs. Their item intersection is unresolved (not zero). That would be reuse of old **transfer/evaluation** items, not proof of exposure to old teacher training traces.

HotpotQA and 2Wiki are separate QA collections with different construction procedures, not two splits of one benchmark. Both use Wikipedia; 2Wiki also uses Wikidata and explicit reasoning construction. [HotpotQA authors](https://hotpotqa.github.io/); [2Wiki authors' paper](https://aclanthology.org/2020.coling-main.580/). Thus ‘different corpora’ is defensible at question-benchmark level, not as a guarantee of disjoint source articles. Rendered current prompts share 3 title prefixes: e. v. v. satyanarayana, george gershwin, manuel romero. There are 0 identical normalized rendered context lines. Title reuse is not itself answer leakage. Truncation, serialization and title-prefix parsing prevent a full passage-overlap certificate.

## Claim disposition and minimal validation

The saved V21 summary explicitly names 11 raw files (0 missing, 0 under the prior tree): V12 evaluations and V6 dense anchors, all fingerprinted by this audit. V31/V31b summaries record the training design and aggregate contrasts but no raw-path manifest; their historical item binding cannot be independently reconstructed from those summaries. See the machine-readable claim dependency inventory.

| Claim | Data dependence | Disposition |
|---|---|---|
| C17 | old locked grid and V3 8B-teacher mixed panel | Preserve original result; add this scope finding. Benchmark fitting, recipe anchor, and IRT findings stay prior-grid scoped. |
| C9b QA / current distillation QA behavior | V12 HotpotQA traces -> default 2Wiki loss; V15 behavioral support mostly easy TriviaQA | Retain numeric loss observations. Independently validate any broad QA capability/behavioral-gain claim on the frozen clean set below. |
| C18 QA interpretation | V16 uses V12 adapters, default 2Wiki probes and C4 generic control | Math/code residual row is not an old-grid QA estimate. Retain residual observations; QA residual improvement is not behavioral proof. Clean-set validation required for a QA capability interpretation. |
| C21 QA (including V31/V31b) | V12 loss/trajectory outcomes; HotpotQA training benchmark persists; V31 summaries lack per-item records | Retain signed loss/data/reuse observations conditional on the measured panel. Require clean-set revalidation before claiming independent QA transfer generality or contamination-free historical provenance. |
| C1 and secondary HotpotQA controls | V9/V23/V26/V27 include HotpotQA for dense/pruned/quantized models | Shared benchmark alone does not transfer a distillation-fitting effect to untrained compression controls. No C17-driven revalidation for those controls; audit training provenance if applying them to HotpotQA-trained adapters. |
| C19/C23 behavioral link | V15/V19 few-shot accuracy; easy QA is TriviaQA, including trained adapters | No old mixed-panel numeric input. Preserve measured benchmark-specific links; do not relabel TriviaQA as 2Wiki or use this link as proof that every QA transfer panel is clean. |
| C10/C11 trace recovery | trace-trained recovery shares current training pool; C4-only recovery does not | Apply the same QA generalization gate to trace-recovery claims; keep numerical recovery observations and C4 controls. |

Branch (b) is triggered at **training-benchmark/distribution level**, not by an observed current train/eval question collision. Preserve numerical fixed-panel loss observations: C17 gives no reason to subtract one third from them, delete them, or rerun training. C18's actual ledger statement is about math/code residuals; the QA interpretation is an extension requiring its own validation. Generic-loss subtraction does not establish behavior.

Branch b only at benchmark/distribution level. Known old M0/eval questions have no current training or measurement matches; item-level branch b is not demonstrated. The old mixed-panel fitting mechanism is not the current default measurement design. Neither an unconditional all-current-sets-clean branch a nor confirmed current item contamination follows.

No new model run is needed solely to preserve the present, qualified fixed-panel observations. The revalidation gate below applies to stronger independent/general QA claims and any assertion of fully verified historical cleanliness; it is not a mandatory retraining request caused by benchmark reuse.

Minimal existing candidate: the 64 exact-question-disjoint default 2Wiki measurement examples, whose ordered prompt/reference hashes are saved in `summary.json`. A stricter rendered-title-disjoint candidate retains 61 examples after excluding measurement indices [8, 37, 41]. This smaller candidate still needs full-context/near-duplicate screening before a corpus-independence claim. Selection here uses provenance only, never correctness or loss. No arbitrary new sample size/power claim is made.

Freeze ordered prompt/reference SHA256 and recover upstream IDs/revision first. Use all 64 question-disjoint items for a fixed-panel claim; for a corpus-independence claim exclude shared-title items and screen full untruncated passages/near-duplicates. Pair each relevant existing dense and distilled/recovery checkpoint on the same retained items; keep reference loss and behavioral EM/F1 separate. No new training is indicated by C17 alone. Reuse saved per-item outputs if available; aggregate losses cannot be subset-rescored. If full provenance cannot be recovered, freeze an independently sourced replacement set; do not silently declare this set clean.

Revalidation is a gate for independent QA behavioral/generalization claims and unresolved historical clean-set assertions (C9b QA, C18 QA extension, C21 QA/V31/V31b, trace-recovery QA). It is not evidence that those saved loss numbers are wrong. V12/V16 aggregate outcomes cannot establish item-level historical provenance or be recomputed on a filtered panel; saved dense-only V15 records cannot supply a missing distilled comparison. This CPU audit closes the scope question and specifies remaining validation; it does not claim to have performed that validation.

## Excluded interpretations and limits

- A literal no-current-data-dependence verdict: current distillation still trains on HotpotQA.
- C17 proves duplicate-item leakage or says one third of current 2Wiki results are contaminated.
- Different benchmark names establish disjoint Wikipedia passages or historical item provenance.
- The current default measurement mixes 50 HotpotQA and 50 2Wiki questions like the old panel.

- Not established: complete original M0/M1/M2/M3 trace lineage (original trace files absent; 150 M0 prompts recoverable through matched gold control).
- Not established: every historical V12/V16/C21 measurement panel bound to immutable question hashes.
- Not established: old 2Wiki IDs versus current 2Wiki IDs (current source IDs absent).
- Not established: semantic/paraphrase overlap, full-context overlap, or model pretraining cleanliness.

## Reproduce

```bash
python analysis/v34_c17_scope_audit.py
# Optional read-only resolution against the existing local HotpotQA cache:
python analysis/v34_c17_scope_audit.py --hotpot-cache /home/xueqi/.cache/huggingface/datasets/hotpotqa___hotpot_qa
python -m pytest -q tests/test_v34.py
```

The first command reports cache provenance as unavailable; the second produces this snapshot's enriched split identification. Standard library only unless reading cached Arrow (pyarrow). [Machine-readable result](../../results/v34-c17-scope/summary.json) includes every input SHA256, raw respondent paths, run inventory, overlap counts, source IDs recovered from cache and clean-set indices/hashes. Inputs are hashed again before output; only the V34 summary/report are written. No network, model run, training, IRT refit, old-result mutation or claim-number correction occurs.
