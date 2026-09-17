# Measurement follow-ups (v27c)

**The saved empty-reasoning panel has one value across all three losses.** The audit checks 768 empty-r model/probe records, with 0 disagreements. The byte prediction sensitivity below replays the original V28 LOMO folds and refits their coefficients. No model was run. MAIN remains V27 L_full, mean_i(NLL_i / target tokens_i), in nats/token.

## 1. Exact empty-reasoning audit

The three saved panels are Gemma3-1B, Gemma3-4B, and OLMo3-7B at pruning density 0.7. References are the existing V6 seed-0 odd-index measurement probes (64 per benchmark; MBPP/HumanEval code and 2WikiMultihopQA/HotpotQA answers have r=empty). All three definitions use the same context p and target y when r is empty. V27 caches by (context IDs, target IDs), so it copies the same NLL record into all three losses. There is no prompt, scored-span, or per-sample normalization difference among them.

The two different numbers actually present in each scoring group are `L_c` (equal-example mean of NLL/token) and `token_weighted_L_c` (sum NLL / sum target tokens). This is **example weighting**, not full versus direct/given, and neither is a per-sequence loss. `sum_ce` is the unnormalized per-sequence NLL before division. The following saved columns pinpoint this genuine two-value distinction; each column is identical across all three losses.

| Model | Benchmark | Empty r n | L_full = L_direct = L_given (L_c) | token_weighted_L_c |
| --- | --- | --- | --- | --- |
| gemma3-1b | code | 64 | 2.540593 | 2.289969 |
| gemma3-1b | code_humaneval | 64 | 2.297865 | 2.070092 |
| gemma3-1b | qa | 64 | 9.090130 | 6.395412 |
| gemma3-1b | qa_hotpotqa | 64 | 9.396929 | 7.126968 |
| gemma3-4b | code | 64 | 2.561770 | 2.280197 |
| gemma3-4b | code_humaneval | 64 | 1.241949 | 1.084313 |
| gemma3-4b | qa | 64 | 8.373747 | 5.715827 |
| gemma3-4b | qa_hotpotqa | 64 | 8.828467 | 6.625154 |
| olmo3-7b | code | 64 | 1.428404 | 1.157142 |
| olmo3-7b | code_humaneval | 64 | 0.617727 | 0.527412 |
| olmo3-7b | qa | 64 | 7.662739 | 5.175384 |
| olmo3-7b | qa_hotpotqa | 64 | 7.889360 | 6.134479 |

Format-control values belong to different targets: e.g. `Answer: ` is prepended to y, and its tokens/bytes enter both numerator and denominator. MATH/GSM8K have nonempty r and generally three different losses. Neither observation supplies a second empty-r conditioning loss. Without a specific contrary output, attributing an alleged two-loss split to a hidden prompt change would be unsupported.

### Per-sample prompts and scored spans

The following three examples use fixed model/benchmark/index identities, without loss-based selection. Prompts are reconstructed offline from the original cached references; all primary probe hashes and saved canonical context-token hashes match. Decoded strings below show exactly the retained token sequence, including special tokens. JSON escapes preserve whitespace. Authoritative integer context/target IDs for each of the three losses, raw untruncated prompt, token pieces, and hashes are in summary.json. Positions are zero-based, half-open in the concatenated model input. Prompt tokens are masked; target has no BOS/EOS. Target token at input position j is scored by logit j−1.

#### gemma3-1b / code / measurement 0 (V6 index 1)

Source: `results/v27-scoring-units/gemma3-1b/dense_prune-d0.7/all-L_full-scoring-units.json`; probe SHA-256 `af8c893750086cb3fa2cd9fd301b9d4fb1ba8cdc82e907355a2eeb4ad566dc3e`. r=`""`.

Prompt fed (same for L_full/L_direct/L_given; decoded retained IDs):

```json
"<bos># Task: Write a function to find the circumference of a circle.\n# Write a Python function.\n"
```

Scored text y (same for all three):

```json
"def circle_circumference(r):\r\n  perimeter=2*3.1415*r\r\n  return perimeter"
```

| Loss | Input target span | Target token IDs | NLL (nats/sequence) | Target tokens | NLL/token |
| --- | --- | --- | --- | --- | --- |
| L_full | [23, 51] | [2063, 10059, 236779, 154792, 2172, 236769, 236750, 1473, 251, 107, 138, 212757, 236784, 236778, 236829, 236800, 236761, 236770, 236812, 236770, 236810, 236829, 236750, 251, 107, 138, 2060, 44803] | 70.932236 | 28 | 2.533294 |
| L_direct | [23, 51] | [2063, 10059, 236779, 154792, 2172, 236769, 236750, 1473, 251, 107, 138, 212757, 236784, 236778, 236829, 236800, 236761, 236770, 236812, 236770, 236810, 236829, 236750, 251, 107, 138, 2060, 44803] | 70.932236 | 28 | 2.533294 |
| L_given | [23, 51] | [2063, 10059, 236779, 154792, 2172, 236769, 236750, 1473, 251, 107, 138, 212757, 236784, 236778, 236829, 236800, 236761, 236770, 236812, 236770, 236810, 236829, 236750, 251, 107, 138, 2060, 44803] | 70.932236 | 28 | 2.533294 |

#### gemma3-4b / qa / measurement 0 (V6 index 1)

Source: `results/v27-scoring-units/gemma3-4b/dense_prune-d0.7/all-L_full-scoring-units.json`; probe SHA-256 `1d3d9e7bf8e10edd7b4e5ce17ed7e1e7892d41978d709a1397849328d85d3f6b`. r=`""`.

Prompt fed (same for L_full/L_direct/L_given; decoded retained IDs):

```json
"<bos>Context:\nDana Blankstein: Dana Blankstein- Cohen( born March 3, 1981) is the director of the Israeli Academy of Film and Television. She is a film director, and an Israeli culture entrepreneur.\nIan Barry (director): Ian Barry is an Australian director of film and TV.\nMark N. Hopkins: Mark N. Hopkins is an English- American filmmaker, best known for his award- winning film\" Living in Emergency\".\nOlav Aaraas: Olav Aaraas( born 10 July 1950) is a Norwegian historian and museum director. He was born in Fredrikstad. From 1982 to 1993 he was the director of Sogn Folk Museum, from 1993 to 2010 he was the director of Maihaugen and from 2001 he has been the director of the Norwegian Museum of Cultural History. In 2010 he was decorated with the Royal Norwegian Order of St. Olav.\nJesse E. Hobson: Jesse Edward Hobson( May 2, 1911 – November 5, 1970) was the director of SRI International from 1947 to 1955. Prior to SRI, he was the director of the Armour Research Foundation.\nS. N. Mathur: S.N. Mathur was the Director of the Indian Intelligence Bureau between September 1975 and February 1980. He was also the Director General of Police in Punjab.\nPeter Levin: Peter Levin is an American director of film, television and theatre.\nSteven Okazaki: Steven Toll Okazaki (born March 12, 1952, in Venice, California) is an American filmmaker. He is Sansei Japanese American (3rd generation) and is based in the San Francisco Bay Area. He has received a Peabody Award and been nominated for four Academy Awards, winning an Oscar for the documentary short subject,  (1990).\nLiving on Tokyo Time: Living on Tokyo Time is a 1987 film starring Minako Ohashi and Ken Nakagawa and directed by Steven Okazaki. It is a romantic comedy revolving around Japanese American rock musician Ken and his marriage of convenience to Kyoko, a young immigré from Japan who speaks limited English. The film received a nomination for a Grand Jury Prize at the 1987 Sundance"
```

Scored text y (same for all three):

```json
" Venice"
```

| Loss | Input target span | Target token IDs | NLL (nats/sequence) | Target tokens | NLL/token |
| --- | --- | --- | --- | --- | --- |
| L_full | [512, 513] | [43586] | 15.804455 | 1 | 15.804455 |
| L_direct | [512, 513] | [43586] | 15.804455 | 1 | 15.804455 |
| L_given | [512, 513] | [43586] | 15.804455 | 1 | 15.804455 |

#### olmo3-7b / qa / measurement 1 (V6 index 3)

Source: `results/v27-scoring-units/olmo3-7b/dense_prune-d0.7/all-L_full-scoring-units.json`; probe SHA-256 `12201dfa96fa3d6f00400eeb7e98ad7aa6cb91974eaf4457ec285a9c47544890`. r=`""`.

Prompt fed (same for L_full/L_direct/L_given; decoded retained IDs):

```json
"Context:\nFranciszek Ksawery Branicki: Franciszek Ksawery Branicki (1730, Barwałd Górny, Poland – 1819 Biała Cerkiew, Russian Empire) was a Polish nobleman, magnate, French count, diplomat, politician, military commander, one of the leaders of the Targowica Confederation and a grand traitor who participated with the Russians in the dismemberment of his nation. He was appointed Great Crown Podstoli in 1764, Ambassador to Berlin in 1765, Master of the Hunt of the Crown in 1766–1773, Artillery General of Lithuania in 1768–1773, Ambassador to Moscow in 1771, Crown Hetman in 1773 and was Great Crown Hetman of the Polish–Lithuanian Commonwealth between 1774 and 1794. In 1774 Stanisław August Poniatowski ceded to him, as mark of his confidence and esteem, the immense estate of Bila Tserkva in the Kiev Oblast. He opposed the reforms of the Great Sejm (1788–1792), and supported the Hetman Party instead. During the Kościuszko Uprising (1794) he was sentenced by the Supreme Criminal Court, \"in absentia\", to hang for treason, witness his decades long pro-Russian stance and anti-patriotic politics and plotting against the state, the Polish-Lithuanian Commonwealth. However, he escaped the death penalty. Branicki was awarded the Order of the White Eagle in December 1764. He married Aleksandra von Engelhardt, a supposed niece of Prince Potemkin, in 1781 making him the putative son-in-law of Empress Catherine of Russia.\nPlace of birth: The place of birth( POB) or birthplace is the place where a person was born. This place is often used in legal documents, together with name and date of birth, to uniquely identify a person. As a general rule with respect to passports, the place of birth is determined to be the country that currently has\" sovereignty\" over the actual place of birth, regardless of when the birth actually occurred. The place of birth is not necessarily the place where the parents of the new baby live. If the baby is born in a hospital in another place, that place is the place of birth. In many countries, this also means that the government requires that the birth of the new baby"
```

Scored text y (same for all three):

```json
" Bila Tserkva"
```

| Loss | Input target span | Target token IDs | NLL (nats/sequence) | Target tokens | NLL/token |
| --- | --- | --- | --- | --- | --- |
| L_full | [512, 518] | [426, 10746, 350, 805, 74, 6723] | 19.693224 | 6 | 3.282204 |
| L_direct | [512, 518] | [426, 10746, 350, 805, 74, 6723] | 19.693224 | 6 | 3.282204 |
| L_given | [512, 518] | [426, 10746, 350, 805, 74, 6723] | 19.693224 | 6 | 3.282204 |

Both QA examples hit the 512-token prompt cap before the final question and `Answer:` cue. That retained prompt is verified against the saved token hash and is shared by all three conditions; it does not explain a difference among their losses.

## 2. Per-byte prediction-error sensitivity

This is the **same V28 compression-response LOMO**, including its original training models, score cells, coefficients/forms, own-dense anchors, and exclusions. Pruning: 12 models, train densities .9/.8/.7/.6, test .8/.7/.6. Quantization: the same 12 models, train bits 8/6/4, test 5/4. Distillation: 3 Gemma3 sizes, original LoRA rows, teacher gpt-5.6-luna/full, D=75/150/300/600 with full-training 4B/D600 excluded. D150 is calibration-only for test scoring. Mode A uses no compressed target calibration; B uses exactly the original one point (.9, 6 bits, D150). Every coefficient and preprocessor is fitted anew without the held-out model. No prospective outcomes enter.

V28's historical prediction endpoint is a **corpus** ratio, sum NLL / sum target tokens, so this sensitivity preserves corpus weighting: L_byte = L_token × T/B, where T and B are the totals for the same 64 reference completions, capped at 512 target tokens. MATH uses all 64 original references, including the 29 excluded from V27's reasoning decomposition. Dense inputs, compressed losses, deltas and calibration points all change units together. This conversion does not infer byte loss from an equal-example aggregate and does not substitute V27's pruned absolute losses for historical prediction deltas.

V6/V10 aggregate-only losses lack per-item hashes and historical tokenizer revisions. Their conversion is conditional on the declared V6 odd64/seed0/512-target-token protocol and cached tokenizer. V27 hashes/spans and available V12 total token counts validate reconstruction, but do not retroactively certify every historical run.

All 143 available final V12 evals for development models match the reconstructed per-capability token/sample totals. V27 legacy per-item token and byte counts match for its three models. The remaining historical runs retain the conditional provenance above. Full denominator manifests and tokenizer/cache SHA-256s are in summary.json.

Numerical MAEs in different units naturally have different scales. To distinguish rescaling from altered prediction quality, `byte refit → token MAE` divides each held-out error by its own model's T/B before averaging over the same cells. `MAE/zero` is a dimensionless comparison against zero-change, separately in each unit. A ≥10% relative change in the converted-back MAE is flagged descriptively as material; this is not an uncertainty/significance claim. A byte score of merely rescaled frozen predictions is also saved, separately from the refit.

| Arm | Capability | Mode | Cells | Token MAE | Byte MAE | Byte refit → token MAE | Relative change | MAE/zero token / byte | ≥10% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pruning | math | A | 36 | 1.414725 | 0.584016 | 1.427533 | +0.91% | 0.9017 / 0.9030 | no |
| pruning | math | B | 36 | 9.290393 | 3.829327 | 9.285685 | -0.05% | 5.9217 / 5.9209 | no |
| pruning | math | zero | 36 | 1.568867 | 0.646746 | 1.568867 | -0.00% | 1.0000 / 1.0000 | no |
| pruning | code | A | 36 | 1.298176 | 0.585440 | 1.486240 | +14.49% | 0.8234 / 0.9002 | yes |
| pruning | code | B | 36 | 1.808598 | 0.710766 | 1.796159 | -0.69% | 1.1472 / 1.0929 | no |
| pruning | code | zero | 36 | 1.576524 | 0.650341 | 1.576524 | +0.00% | 1.0000 / 1.0000 | no |
| pruning | qa | A | 36 | 1.915002 | 0.468475 | 1.859837 | -2.88% | 1.1314 / 1.1011 | no |
| pruning | qa | B | 36 | 34.499345 | 8.642501 | 34.153008 | -1.00% | 20.3817 / 20.3130 | no |
| pruning | qa | zero | 36 | 1.692666 | 0.425467 | 1.692666 | +0.00% | 1.0000 / 1.0000 | no |
| quantization | math | A | 24 | 0.263659 | 0.106280 | 0.262053 | -0.61% | 0.6793 / 0.6723 | no |
| quantization | math | B | 24 | 0.223552 | 0.090418 | 0.223552 | -0.00% | 0.5760 / 0.5720 | no |
| quantization | math | zero | 24 | 0.388126 | 0.158086 | 0.388126 | +0.00% | 1.0000 / 1.0000 | no |
| quantization | code | A | 24 | 0.336182 | 0.136076 | 0.362149 | +7.72% | 0.8665 / 0.9154 | no |
| quantization | code | B | 24 | 0.274628 | 0.102965 | 0.274628 | +0.00% | 0.7078 / 0.6927 | no |
| quantization | code | zero | 24 | 0.387987 | 0.148646 | 0.387987 | +0.00% | 1.0000 / 1.0000 | no |
| quantization | qa | A | 24 | 0.356733 | 0.091330 | 0.356595 | -0.04% | 1.2430 / 1.2561 | no |
| quantization | qa | B | 24 | 0.408733 | 0.105017 | 0.408733 | -0.00% | 1.4242 / 1.4444 | no |
| quantization | qa | zero | 24 | 0.287000 | 0.072706 | 0.287000 | +0.00% | 1.0000 / 1.0000 | no |
| distillation | math | A | 8 | 0.032152 | 0.013258 | 0.032152 | +0.00% | 0.3647 / 0.3647 | no |
| distillation | math | B | 8 | 0.031862 | 0.013139 | 0.031862 | +0.00% | 0.3614 / 0.3614 | no |
| distillation | math | zero | 8 | 0.088158 | 0.036354 | 0.088158 | +0.00% | 1.0000 / 1.0000 | no |
| distillation | code | A | 8 | 0.037223 | 0.015685 | 0.037223 | +0.00% | 0.3459 / 0.3459 | no |
| distillation | code | B | 8 | 0.026129 | 0.011010 | 0.026129 | +0.00% | 0.2428 / 0.2428 | no |
| distillation | code | zero | 8 | 0.107621 | 0.045350 | 0.107621 | +0.00% | 1.0000 / 1.0000 | no |
| distillation | qa | A | 8 | 0.257162 | 0.064227 | 0.257162 | +0.00% | 0.2435 / 0.2435 | no |
| distillation | qa | B | 8 | 0.303007 | 0.075676 | 0.303007 | -0.00% | 0.2869 / 0.2869 | no |
| distillation | qa | zero | 8 | 1.056104 | 0.263763 | 1.056104 | +0.00% | 1.0000 / 1.0000 | no |

**Prediction conclusion:** pruning CODE Mode A changes from 1.298176 nats/token MAE to 0.585440 nats/byte; its converted-back MAE is 1.486240, a +14.49% change that meets the descriptive 10% threshold. Quantization CODE Mode A changes by +7.72%, below that threshold but numerically nonzero. Other A/B converted-back MAE changes are below 3%; distillation is pure unit scaling. No A/B comparison against zero-change reverses. Stable benchmark rankings therefore coexist with a material prediction-error change in pruning CODE.

### Law coefficients

These full-development coefficients are descriptive; the MAEs above use the separate saved LOMO memberships and newly fitted fold coefficients, all retained in JSON. Amplitudes carry loss units, so their raw numerical changes alone are not evidence of a changed law. Pruning gamma is dimensionless and can change because model-dependent T/B changes the relative weight of curves in the profiled SSE. Quantization's exponent remains fixed at 2 in the original V28 form. Distillation uses identical Gemma target denominators within capability; a common rescaling leaves its converted-back predictions unchanged.

| Arm | Capability | Coefficient | Token units | Byte units | Gamma token | Gamma byte |
| --- | --- | --- | --- | --- | --- | --- |
| pruning | math | mean a at density .7 | 0.914439 | 0.376972 | 4.679569 | 4.679411 |
| pruning | code | mean a at density .7 | 1.145011 | 0.472983 | 3.757519 | 3.749218 |
| pruning | qa | mean a at density .7 | 0.809727 | 0.201418 | 4.870229 | 4.864335 |
| quantization | math | mean q/1024 | 0.170495 | 0.069371 | N/A | N/A |
| quantization | code | mean q/1024 | 0.165976 | 0.063672 | N/A | N/A |
| quantization | qa | mean q/1024 | 0.039764 | 0.010090 | N/A | N/A |
| distillation | math | mu | 0.090257 | 0.037220 | N/A | N/A |
| distillation | code | beta (log1p D/150) | 0.116131 | 0.048936 | N/A | N/A |
| distillation | qa | mu | -1.140184 | -0.284762 | N/A | N/A |

To compare the transferred law amplitude itself, the next table uses each held-out Mode A amplitude (a, q/1024, mu, or beta), converts its byte fit back with that target's T/B, and reports sum absolute amplitude changes / sum absolute original amplitudes. This avoids unstable percentage changes of individual near-zero signed coefficients. The same 10% descriptive threshold is used. Gamma reports the maximum absolute relative change across LOMO folds. The full ridge mappings and every fold amplitude are retained in JSON.

| Arm | Capability | Amplitude change in common units | ≥10% | Max fold gamma change |
| --- | --- | --- | --- | --- |
| pruning | math | 2.07% | no | 0.010% |
| pruning | code | 20.06% | yes | 0.542% |
| pruning | qa | 10.84% | yes | 0.190% |
| quantization | math | 3.47% | no | fixed / N/A |
| quantization | code | 14.43% | yes | fixed / N/A |
| quantization | qa | 9.39% | no | fixed / N/A |
| distillation | math | 0.00% | no | fixed / N/A |
| distillation | code | 0.00% | no | fixed / N/A |
| distillation | qa | 0.00% | no | fixed / N/A |

The 0/18 rank flips in V27b therefore do not establish prediction-error invariance. Use the matched-fold error and coefficient comparisons here for that question. V27 native-token MAIN and all frozen historical/prospective artifacts are unchanged.

## Reproduce and provenance

```bash
python analysis/v27c_measurement_followups.py
python -m pytest -q tests/test_v27c.py
```

Requires numpy, scipy, pyarrow, tokenizers, the existing results, and the read-only local Hugging Face cache (override with --hf-cache). Missing/mismatched references or denominators raise an error; no download or model fallback exists. --dry-run validates and computes without writes. Outputs: results/v27c/summary.json and this report. Summary includes source file hashes, frozen prediction digest, exact per-sample IDs/spans, all fold fits/predictions, conversion denominators, and native replay errors. Historical uncaptured tokenizer revisions remain explicit.
