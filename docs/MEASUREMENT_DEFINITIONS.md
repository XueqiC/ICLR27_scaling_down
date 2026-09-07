# Measurement definitions — v27 closed

This is a numerical closeout of existing outputs, before new training. The MAIN endpoint remains the native-token capability loss: v27 **L_full**, the equal-example mean of target NLL/token, separately by benchmark. No model was run, no prediction endpoint was changed, and no auxiliary result was selected as MAIN.

Numerical readout across the saved models and primary/secondary benchmarks (nats/token; the detailed paired contrasts below retain every benchmark):

| Capability | Mean within-item format range | Mean given−direct change | Candidate interpretation excluded |
| --- | --- | --- | --- |
| math | 0.203554 to 0.303080 | -3.767369 to -0.677889 | excludes equivalence of direct and given-reference-reasoning answer loss |
| code | 0.450511 to 1.293250 | +0.000000 to +0.000000 | excludes surface invariance; empty-r conditioning identity |
| qa | 3.204216 to 4.205446 | +0.000000 to +0.000000 | excludes surface invariance; empty-r conditioning identity |

The legacy V6/V12 and prospective results retain their original versions, cohorts, token units and aggregation. In particular V12 uses a corpus-token ratio; it is not replaced by v27's equal-example full-solution metric. V27 also retains `legacy_v6` separately. This document closes definitions and the available readout; missing measurements are explicitly unavailable, rather than a request to reopen v27 or launch training.

Observed input schema: the local files use `benchmarks` (not the anticipated `capabilities` wrapper), with `metric_definitions`, per-benchmark specs, `scoring`, `format_control`, `legacy_v6`, per-item NLL/token/byte counts, and reference decompositions. The reader also accepts a capabilities wrapper containing those benchmark records.

| Model | Prune density | Quant bits | Adapter | Tokenizer | Source |
| --- | --- | --- | --- | --- | --- |
| gemma3-1b | 0.7 | None | None | google/gemma-3-1b-pt | `results/v27-scoring-units/gemma3-1b/dense_prune-d0.7/all-L_full-scoring-units.json` |
| gemma3-4b | 0.7 | None | None | google/gemma-3-4b-pt | `results/v27-scoring-units/gemma3-4b/dense_prune-d0.7/all-L_full-scoring-units.json` |
| olmo3-7b | 0.7 | None | None | allenai/Olmo-3-1025-7B | `results/v27-scoring-units/olmo3-7b/dense_prune-d0.7/all-L_full-scoring-units.json` |

There are 3 saved panels. Their absolute losses alone cannot establish a compression delta or a QA pruning sign without a matched dense anchor. Protocol and eligibility are retained explicitly below; counts are benchmark-specific. Conditioning comparisons always pair the same items, and format comparisons use the common format/scoring subset. This excludes attributing a within-benchmark contrast to different selected items.

| Model | Seed | Requested | Probe half | Max length | Prompt cap |
| --- | --- | --- | --- | --- | --- |
| gemma3-1b | 0 | 128 | measurement (odd indices, v[1::2]) | 1024 | 512 |
| gemma3-4b | 0 | 128 | measurement (odd indices, v[1::2]) | 1024 | 512 |
| olmo3-7b | 0 | 128 | measurement (odd indices, v[1::2]) | 1024 | 512 |

| Model | Benchmark | Scoring / available | Format n | Conditioning exclusions | Extra format exclusions |
| --- | --- | --- | --- | --- | --- |
| gemma3-1b | MBPP | 64 / 64 | 64 | none | 0 |
| gemma3-1b | HumanEval | 64 / 64 | 64 | none | 0 |
| gemma3-1b | MATH-500 | 35 / 64 | 35 | 29: information after final box; cannot isolate final answer safely | 0 |
| gemma3-1b | GSM8K | 64 / 64 | 64 | none | 0 |
| gemma3-1b | 2WikiMultihopQA | 64 / 64 | 64 | none | 0 |
| gemma3-1b | HotpotQA | 64 / 64 | 64 | none | 0 |
| gemma3-4b | MBPP | 64 / 64 | 64 | none | 0 |
| gemma3-4b | HumanEval | 64 / 64 | 64 | none | 0 |
| gemma3-4b | MATH-500 | 35 / 64 | 35 | 29: information after final box; cannot isolate final answer safely | 0 |
| gemma3-4b | GSM8K | 64 / 64 | 64 | none | 0 |
| gemma3-4b | 2WikiMultihopQA | 64 / 64 | 64 | none | 0 |
| gemma3-4b | HotpotQA | 64 / 64 | 64 | none | 0 |
| olmo3-7b | MBPP | 64 / 64 | 64 | none | 0 |
| olmo3-7b | HumanEval | 64 / 64 | 64 | none | 0 |
| olmo3-7b | MATH-500 | 35 / 64 | 35 | 29: information after final box; cannot isolate final answer safely | 0 |
| olmo3-7b | GSM8K | 64 / 64 | 64 | none | 0 |
| olmo3-7b | 2WikiMultihopQA | 64 / 64 | 64 | none | 0 |
| olmo3-7b | HotpotQA | 64 / 64 | 64 | none | 0 |

The three losses are distinct definitions (natural logarithms, total NLL before normalization):

| Version | Name | Total NLL | Scored region | Conditioning |
| --- | --- | --- | --- | --- |
| v27-native-token / MAIN | L_full — full solution loss | −log p(r,y\|x) | r+y | x |
| v27-direct / auxiliary | L_direct — direct answer loss | −log p(y\|x) | y | x |
| v27-given / auxiliary | L_given — given-reference-reasoning answer loss | −log p(y\|x,r) | y | x,r |

L_given is still an **answer loss**; its conditioning differs from L_direct. For MATH-500, y contains the final boxed answer plus punctuation; for GSM8K it contains the final `####` delimiter and answer. Reference r/y are encoded separately, with identical y token IDs in all three conditions. MBPP/HumanEval use the whole reference code as y. 2WikiMultihopQA/HotpotQA use the reference answer as y with benchmark context in x. CODE/QA have r empty: equality of their three losses is an identity, not three independent confirmations and not evidence that reasoning generally has no value.

| Model | Capability / benchmark | n | L_full MAIN | L_direct | L_given | direct−full | given−full | given−direct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma3-1b | code / MBPP | 64 | 2.540593 | 2.540593 | 2.540593 | +0.000000 | +0.000000 | +0.000000 |
| gemma3-1b | code / HumanEval | 64 | 2.297865 | 2.297865 | 2.297865 | +0.000000 | +0.000000 | +0.000000 |
| gemma3-1b | math / MATH-500 | 35 | 2.788209 | 7.253993 | 6.576104 | +4.465784 | +3.787895 | -0.677889 |
| gemma3-1b | math / GSM8K | 64 | 2.381296 | 7.660634 | 5.391164 | +5.279338 | +3.009867 | -2.269471 |
| gemma3-1b | qa / 2WikiMultihopQA | 64 | 9.090130 | 9.090130 | 9.090130 | +0.000000 | +0.000000 | +0.000000 |
| gemma3-1b | qa / HotpotQA | 64 | 9.396929 | 9.396929 | 9.396929 | +0.000000 | +0.000000 | +0.000000 |
| gemma3-4b | code / MBPP | 64 | 2.561770 | 2.561770 | 2.561770 | +0.000000 | +0.000000 | +0.000000 |
| gemma3-4b | code / HumanEval | 64 | 1.241949 | 1.241949 | 1.241949 | +0.000000 | +0.000000 | +0.000000 |
| gemma3-4b | math / MATH-500 | 35 | 1.783422 | 6.967276 | 4.665833 | +5.183854 | +2.882411 | -2.301443 |
| gemma3-4b | math / GSM8K | 64 | 1.547366 | 7.244200 | 5.226193 | +5.696834 | +3.678827 | -2.018007 |
| gemma3-4b | qa / 2WikiMultihopQA | 64 | 8.373747 | 8.373747 | 8.373747 | +0.000000 | +0.000000 | +0.000000 |
| gemma3-4b | qa / HotpotQA | 64 | 8.828467 | 8.828467 | 8.828467 | +0.000000 | +0.000000 | +0.000000 |
| olmo3-7b | code / MBPP | 64 | 1.428404 | 1.428404 | 1.428404 | +0.000000 | +0.000000 | +0.000000 |
| olmo3-7b | code / HumanEval | 64 | 0.617727 | 0.617727 | 0.617727 | +0.000000 | +0.000000 | +0.000000 |
| olmo3-7b | math / MATH-500 | 35 | 0.788228 | 4.867547 | 1.965808 | +4.079319 | +1.177580 | -2.901739 |
| olmo3-7b | math / GSM8K | 64 | 0.808409 | 6.458135 | 2.690766 | +5.649727 | +1.882358 | -3.767369 |
| olmo3-7b | qa / 2WikiMultihopQA | 64 | 7.662739 | 7.662739 | 7.662739 | +0.000000 | +0.000000 | +0.000000 |
| olmo3-7b | qa / HotpotQA | 64 | 7.889360 | 7.889360 | 7.889360 | +0.000000 | +0.000000 | +0.000000 |

All numbers above are nats/token, equal-example means. The negative given−direct math contrasts exclude equivalence of the two answer-conditioning questions on this panel. A lower normalized full loss does not imply that producing reasoning is free: full and answer-only losses have different target lengths. Never interpret their normalized difference as reasoning NLL.

True format control is separate. The context and reference body are fixed; only the outer prefix changes to a newline, blank line, `Answer: `, or `The answer is `. Internal code whitespace stays fixed. All saved controls condition as L_full. Entries below are paired mean variant−canonical changes, with the mean within-item max−min range over all five pre-specified variants. The range is a sensitivity statistic, not a rule for choosing the best format. Given−direct is recomputed on the format cohort; the final column is its mean absolute paired change.

| Model | Benchmark | newline Δ | blank line Δ | Answer: Δ | The answer is Δ | format range | given−direct Δ | mean \|given−direct\| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma3-1b | MBPP | +0.070324 | +0.080667 | +0.422349 | +0.631745 | 0.631745 | +0.000000 | 0.000000 |
| gemma3-1b | HumanEval | +0.150319 | +0.281329 | +1.110826 | +1.282866 | 1.293250 | +0.000000 | 0.000000 |
| gemma3-1b | MATH-500 | +0.057639 | +0.108392 | +0.232324 | +0.244063 | 0.258096 | -0.677889 | 1.086693 |
| gemma3-1b | GSM8K | +0.053515 | +0.133812 | +0.276615 | +0.268844 | 0.285349 | -2.269471 | 2.269471 |
| gemma3-1b | 2WikiMultihopQA | +1.644508 | +1.666744 | +1.469925 | +0.498645 | 4.205446 | +0.000000 | 0.000000 |
| gemma3-1b | HotpotQA | +0.868199 | +1.029018 | +1.518383 | +0.358582 | 4.144568 | +0.000000 | 0.000000 |
| gemma3-4b | MBPP | +0.237173 | +0.255072 | +0.263500 | +0.247181 | 0.450511 | +0.000000 | 0.000000 |
| gemma3-4b | HumanEval | +0.437565 | +0.594175 | +1.094802 | +1.133207 | 1.181130 | +0.000000 | 0.000000 |
| gemma3-4b | MATH-500 | +0.096411 | +0.130449 | +0.234808 | +0.218383 | 0.238258 | -2.301443 | 2.301443 |
| gemma3-4b | GSM8K | +0.072054 | +0.129498 | +0.277155 | +0.261222 | 0.281691 | -2.018007 | 2.035227 |
| gemma3-4b | 2WikiMultihopQA | +0.277629 | +0.640350 | +0.686595 | -0.489364 | 3.538899 | +0.000000 | 0.000000 |
| gemma3-4b | HotpotQA | -0.424726 | +0.021046 | +0.645870 | -0.466729 | 3.467969 | +0.000000 | 0.000000 |
| olmo3-7b | MBPP | +0.249298 | +0.287409 | +0.397693 | +0.607412 | 0.607412 | +0.000000 | 0.000000 |
| olmo3-7b | HumanEval | +0.428637 | +0.459037 | +0.937855 | +1.040158 | 1.041745 | +0.000000 | 0.000000 |
| olmo3-7b | MATH-500 | +0.170649 | +0.194962 | +0.129460 | +0.164555 | 0.203554 | -2.901739 | 2.901739 |
| olmo3-7b | GSM8K | +0.184567 | +0.202075 | +0.256217 | +0.297741 | 0.303080 | -3.767369 | 3.767369 |
| olmo3-7b | 2WikiMultihopQA | -0.297467 | +0.626956 | -1.190093 | -0.755463 | 3.204216 | +0.000000 | 0.000000 |
| olmo3-7b | HotpotQA | -0.368033 | +0.236718 | -1.433421 | -0.725760 | 3.208265 | +0.000000 | 0.000000 |

These numbers exclude perfect surface invariance. They do not identify a causal share of distillation damage: there are no paired dense/distilled format panels. In math the format control scores r+y, while the conditioning contrast scores y, so their normalized magnitudes are descriptive sensitivities on different target regions. In CODE/QA, the format movement coexists with exactly zero conditioning movement because r is empty. Prefix prediction and denominator dilution are included; body-only or syntax/semantic NLL is unavailable. To expose denominator effects, the following are mean **total** NLL changes (nats/example), before length normalization:

| Model | Benchmark | newline Δ total | blank line Δ total | Answer: Δ total | The answer is Δ total | given−direct Δ total |
| --- | --- | --- | --- | --- | --- | --- |
| gemma3-1b | MBPP | +6.178234 | +6.772805 | +30.175060 | +45.387954 | +0.000000 |
| gemma3-1b | HumanEval | +7.081226 | +11.885537 | +50.692840 | +64.057583 | +0.000000 |
| gemma3-1b | MATH-500 | +10.776811 | +17.282269 | +40.422929 | +46.310754 | -6.777877 |
| gemma3-1b | GSM8K | +8.034109 | +16.794794 | +38.192498 | +40.340158 | -9.660090 |
| gemma3-1b | 2WikiMultihopQA | +14.951330 | +15.503410 | +39.424046 | +44.669162 | +0.000000 |
| gemma3-1b | HotpotQA | +13.284053 | +13.943721 | +40.868658 | +44.581908 | +0.000000 |
| gemma3-4b | MBPP | +13.849868 | +14.871494 | +21.278286 | +18.982622 | +0.000000 |
| gemma3-4b | HumanEval | +16.012033 | +21.547243 | +45.396161 | +51.888858 | +0.000000 |
| gemma3-4b | MATH-500 | +15.581348 | +20.255598 | +39.504706 | +39.140815 | -18.247954 |
| gemma3-4b | GSM8K | +9.514038 | +15.630641 | +35.751621 | +36.161453 | -8.596967 |
| gemma3-4b | 2WikiMultihopQA | +10.352375 | +11.988633 | +33.701110 | +35.598207 | +0.000000 |
| gemma3-4b | HotpotQA | +8.366126 | +10.256826 | +34.222260 | +36.449017 | +0.000000 |
| olmo3-7b | MBPP | +11.167007 | +12.686688 | +21.047152 | +32.429748 | +0.000000 |
| olmo3-7b | HumanEval | +13.681430 | +14.667878 | +34.297917 | +40.216197 | +0.000000 |
| olmo3-7b | MATH-500 | +24.279921 | +27.429264 | +20.139162 | +25.970731 | -22.334647 |
| olmo3-7b | GSM8K | +17.284621 | +18.933455 | +25.496739 | +30.381738 | -11.690389 |
| olmo3-7b | 2WikiMultihopQA | +7.951692 | +11.515703 | +19.974148 | +29.880561 | +0.000000 |
| olmo3-7b | HotpotQA | +7.282970 | +9.774729 | +17.523784 | +29.789412 | +0.000000 |

For a concrete denominator check, OLMo7B/2WikiMultihopQA with `Answer: ` changes mean token loss by -1.190093 nats/token but mean total NLL by +19.974148 nats/example. This excludes treating the normalized change as the same change in total sequence probability. The original answer's body-only likelihood cannot be recovered from a wrapped-target total.

Byte versions use each sample's saved total negative log probability and **exact UTF-8 bytes of its scored target**: b_i=len(target.encode('utf-8')). The readout verifies byte counts and target hashes against the saved r/y strings and prefixes. Context/BOS/EOS bytes are excluded. It does not convert an aggregate per-token mean into bytes. Both weighting conventions are explicit: equal-example token=mean_i(NLL_i/n_i), equal-example byte=mean_i(NLL_i/b_i), corpus token=ΣNLL_i/Σn_i, corpus byte=ΣNLL_i/Σb_i. The corpus-byte version is the existing v27 byte convention; the equal-example byte version isolates a unit change while keeping MAIN's example weighting.

| Model | Benchmark | Loss | Native token mean | Byte mean | Corpus token | Corpus byte | ΣNLL | Σtokens | Σbytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma3-1b | MBPP | L_full | 2.540593 | 1.050082 | 2.289969 | 0.964962 | 10341.501221 | 4516 | 10717 |
| gemma3-1b | MBPP | L_direct | 2.540593 | 1.050082 | 2.289969 | 0.964962 | 10341.501221 | 4516 | 10717 |
| gemma3-1b | MBPP | L_given | 2.540593 | 1.050082 | 2.289969 | 0.964962 | 10341.501221 | 4516 | 10717 |
| gemma3-1b | HumanEval | L_full | 2.297865 | 0.849403 | 2.070092 | 0.736049 | 8288.647375 | 4004 | 11261 |
| gemma3-1b | HumanEval | L_direct | 2.297865 | 0.849403 | 2.070092 | 0.736049 | 8288.647375 | 4004 | 11261 |
| gemma3-1b | HumanEval | L_given | 2.297865 | 0.849403 | 2.070092 | 0.736049 | 8288.647375 | 4004 | 11261 |
| gemma3-1b | MATH-500 | L_full | 2.788209 | 1.136224 | 2.619373 | 1.088235 | 18657.790985 | 7123 | 17145 |
| gemma3-1b | MATH-500 | L_direct | 7.253993 | 3.549650 | 6.248889 | 3.154201 | 1987.146725 | 318 | 630 |
| gemma3-1b | MATH-500 | L_given | 6.576104 | 3.206995 | 5.502896 | 2.777652 | 1749.921034 | 318 | 630 |
| gemma3-1b | GSM8K | L_full | 2.381296 | 1.031438 | 2.298701 | 0.969983 | 18031.007446 | 7844 | 18589 |
| gemma3-1b | GSM8K | L_direct | 7.660634 | 4.338770 | 7.289363 | 4.286020 | 1997.285393 | 274 | 466 |
| gemma3-1b | GSM8K | L_given | 5.391164 | 3.032261 | 5.032991 | 2.959312 | 1379.039621 | 274 | 466 |
| gemma3-1b | 2WikiMultihopQA | L_full | 9.090130 | 1.972178 | 6.395412 | 1.597262 | 1605.248495 | 251 | 1005 |
| gemma3-1b | 2WikiMultihopQA | L_direct | 9.090130 | 1.972178 | 6.395412 | 1.597262 | 1605.248495 | 251 | 1005 |
| gemma3-1b | 2WikiMultihopQA | L_given | 9.090130 | 1.972178 | 6.395412 | 1.597262 | 1605.248495 | 251 | 1005 |
| gemma3-1b | HotpotQA | L_full | 9.396929 | 2.157812 | 7.126968 | 1.688543 | 1646.329627 | 231 | 975 |
| gemma3-1b | HotpotQA | L_direct | 9.396929 | 2.157812 | 7.126968 | 1.688543 | 1646.329627 | 231 | 975 |
| gemma3-1b | HotpotQA | L_given | 9.396929 | 2.157812 | 7.126968 | 1.688543 | 1646.329627 | 231 | 975 |
| gemma3-4b | MBPP | L_full | 2.561770 | 1.052690 | 2.280197 | 0.960844 | 10297.367699 | 4516 | 10717 |
| gemma3-4b | MBPP | L_direct | 2.561770 | 1.052690 | 2.280197 | 0.960844 | 10297.367699 | 4516 | 10717 |
| gemma3-4b | MBPP | L_given | 2.561770 | 1.052690 | 2.280197 | 0.960844 | 10297.367699 | 4516 | 10717 |
| gemma3-4b | HumanEval | L_full | 1.241949 | 0.459234 | 1.084313 | 0.385542 | 4341.588120 | 4004 | 11261 |
| gemma3-4b | HumanEval | L_direct | 1.241949 | 0.459234 | 1.084313 | 0.385542 | 4341.588120 | 4004 | 11261 |
| gemma3-4b | HumanEval | L_given | 1.241949 | 0.459234 | 1.084313 | 0.385542 | 4341.588120 | 4004 | 11261 |
| gemma3-4b | MATH-500 | L_full | 1.783422 | 0.719699 | 1.638742 | 0.680826 | 11672.756210 | 7123 | 17145 |
| gemma3-4b | MATH-500 | L_direct | 6.967276 | 3.410356 | 5.845052 | 2.950360 | 1858.726517 | 318 | 630 |
| gemma3-4b | MATH-500 | L_given | 4.665833 | 2.284031 | 3.836629 | 1.936584 | 1220.048140 | 318 | 630 |
| gemma3-4b | GSM8K | L_full | 1.547366 | 0.672767 | 1.464025 | 0.617775 | 11483.813087 | 7844 | 18589 |
| gemma3-4b | GSM8K | L_direct | 7.244200 | 4.110683 | 6.917897 | 4.067604 | 1895.503691 | 274 | 466 |
| gemma3-4b | GSM8K | L_given | 5.226193 | 2.950039 | 4.909846 | 2.886905 | 1345.297826 | 274 | 466 |
| gemma3-4b | 2WikiMultihopQA | L_full | 8.373747 | 1.809860 | 5.715827 | 1.427535 | 1434.672523 | 251 | 1005 |
| gemma3-4b | 2WikiMultihopQA | L_direct | 8.373747 | 1.809860 | 5.715827 | 1.427535 | 1434.672523 | 251 | 1005 |
| gemma3-4b | 2WikiMultihopQA | L_given | 8.373747 | 1.809860 | 5.715827 | 1.427535 | 1434.672523 | 251 | 1005 |
| gemma3-4b | HotpotQA | L_full | 8.828467 | 2.068230 | 6.625154 | 1.569652 | 1530.410689 | 231 | 975 |
| gemma3-4b | HotpotQA | L_direct | 8.828467 | 2.068230 | 6.625154 | 1.569652 | 1530.410689 | 231 | 975 |
| gemma3-4b | HotpotQA | L_given | 8.828467 | 2.068230 | 6.625154 | 1.569652 | 1530.410689 | 231 | 975 |
| olmo3-7b | MBPP | L_full | 1.428404 | 0.446883 | 1.157142 | 0.373045 | 3997.925823 | 3455 | 10717 |
| olmo3-7b | MBPP | L_direct | 1.428404 | 0.446883 | 1.157142 | 0.373045 | 3997.925823 | 3455 | 10717 |
| olmo3-7b | MBPP | L_given | 1.428404 | 0.446883 | 1.157142 | 0.373045 | 3997.925823 | 3455 | 10717 |
| olmo3-7b | HumanEval | L_full | 0.617727 | 0.194701 | 0.527412 | 0.157507 | 1773.685622 | 3363 | 11261 |
| olmo3-7b | HumanEval | L_direct | 0.617727 | 0.194701 | 0.527412 | 0.157507 | 1773.685622 | 3363 | 11261 |
| olmo3-7b | HumanEval | L_given | 0.617727 | 0.194701 | 0.527412 | 0.157507 | 1773.685622 | 3363 | 11261 |
| olmo3-7b | MATH-500 | L_full | 0.788228 | 0.306673 | 0.718581 | 0.291666 | 5000.606915 | 6959 | 17145 |
| olmo3-7b | MATH-500 | L_direct | 4.867547 | 2.515132 | 4.046356 | 2.100251 | 1323.158373 | 327 | 630 |
| olmo3-7b | MATH-500 | L_given | 1.965808 | 1.024625 | 1.655797 | 0.859438 | 541.445719 | 327 | 630 |
| olmo3-7b | GSM8K | L_full | 0.808409 | 0.285289 | 0.762548 | 0.262004 | 4870.391386 | 6387 | 18589 |
| olmo3-7b | GSM8K | L_direct | 6.458135 | 2.817754 | 6.409217 | 2.764491 | 1288.252587 | 201 | 466 |
| olmo3-7b | GSM8K | L_given | 2.690766 | 1.174975 | 2.686904 | 1.158944 | 540.067717 | 201 | 466 |
| olmo3-7b | 2WikiMultihopQA | L_full | 7.662739 | 1.736723 | 5.175384 | 1.349205 | 1355.950713 | 262 | 1005 |
| olmo3-7b | 2WikiMultihopQA | L_direct | 7.662739 | 1.736723 | 5.175384 | 1.349205 | 1355.950713 | 262 | 1005 |
| olmo3-7b | 2WikiMultihopQA | L_given | 7.662739 | 1.736723 | 5.175384 | 1.349205 | 1355.950713 | 262 | 1005 |
| olmo3-7b | HotpotQA | L_full | 7.889360 | 2.007811 | 6.134479 | 1.447108 | 1410.930144 | 230 | 975 |
| olmo3-7b | HotpotQA | L_direct | 7.889360 | 2.007811 | 6.134479 | 1.447108 | 1410.930144 | 230 | 975 |
| olmo3-7b | HotpotQA | L_given | 7.889360 | 2.007811 | 6.134479 | 1.447108 | 1410.930144 | 230 | 975 |

Cross-model ordering uses the intersection of probe identities with equal scored-target hashes and bytes, separately by benchmark/loss and compression configuration. All eligible items are shared in this panel. Lower loss sorts first; ties are retained explicitly. The table groups identical CODE/QA loss identities; summary.json retains every loss.

| Benchmark | Loss | Equal-example token order | Equal-example byte order | Corpus byte order | Example unit flip? | Corpus unit flip? |
| --- | --- | --- | --- | --- | --- | --- |
| MBPP | all three | olmo3-7b < gemma3-1b < gemma3-4b | olmo3-7b < gemma3-1b < gemma3-4b | olmo3-7b < gemma3-4b < gemma3-1b | False | False |
| HumanEval | all three | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | False | False |
| MATH-500 | L_full | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | False | False |
| MATH-500 | L_direct | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | False | False |
| MATH-500 | L_given | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | False | False |
| GSM8K | L_full | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | False | False |
| GSM8K | L_direct | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | False | False |
| GSM8K | L_given | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | False | False |
| 2WikiMultihopQA | all three | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | False | False |
| HotpotQA | all three | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | olmo3-7b < gemma3-4b < gemma3-1b | False | False |

**Ordering conclusion:** over 18 benchmark/loss comparisons, token→byte ordering changes in 0 at equal-example weighting and 0 at corpus weighting. Comparing MAIN's weighting to corpus bytes changes 3 orders (CODE/QA identities are counted separately by loss).

MBPP: gemma3-1b−gemma3-4b is -0.021177 native nats/token, -0.002609 equal-example nats/byte, +0.009773 corpus nats/token, and +0.004118 corpus nats/byte.

This reversal already occurs when changing example weighting to corpus weighting with token units fixed; it does not demonstrate a tokenizer-induced reversal.

Fixed-weight rankings, including OLMo versus Gemma, are unit-robust in this panel. This is not universal measurement invariance.

The reversals exclude a claim that every numerical cross-model ranking survives a simultaneous unit/aggregation change.

Token-limited prompts can retain different text across tokenizers, and exact retained prompt text and immutable tokenizer revisions were not saved.

Region accounting uses NLL_full−NLL_given **before normalization** to recover −log p(r|x). Answer NLL is the saved given-reference-reasoning answer NLL. Fractions are shares of full total NLL at these pruned checkpoints, not shares of a distillation or pruning delta.

| Model | Benchmark | Reasoning ΣNLL | Answer ΣNLL | Answer fraction | Reasoning nats/token |
| --- | --- | --- | --- | --- | --- |
| gemma3-1b | MBPP | 0.000000 | 10341.501221 | 1.000000 | N/A |
| gemma3-1b | HumanEval | 0.000000 | 8288.647375 | 1.000000 | N/A |
| gemma3-1b | MATH-500 | 16907.869951 | 1749.921034 | 0.093790 | 2.574775 |
| gemma3-1b | GSM8K | 16651.967825 | 1379.039621 | 0.076482 | 2.271410 |
| gemma3-1b | 2WikiMultihopQA | 0.000000 | 1605.248495 | 1.000000 | N/A |
| gemma3-1b | HotpotQA | 0.000000 | 1646.329627 | 1.000000 | N/A |
| gemma3-4b | MBPP | 0.000000 | 10297.367699 | 1.000000 | N/A |
| gemma3-4b | HumanEval | 0.000000 | 4341.588120 | 1.000000 | N/A |
| gemma3-4b | MATH-500 | 10452.708071 | 1220.048140 | 0.104521 | 1.618015 |
| gemma3-4b | GSM8K | 10138.515262 | 1345.297826 | 0.117147 | 1.408475 |
| gemma3-4b | 2WikiMultihopQA | 0.000000 | 1434.672523 | 1.000000 | N/A |
| gemma3-4b | HotpotQA | 0.000000 | 1530.410689 | 1.000000 | N/A |
| olmo3-7b | MBPP | 0.000000 | 3997.925823 | 1.000000 | N/A |
| olmo3-7b | HumanEval | 0.000000 | 1773.685622 | 1.000000 | N/A |
| olmo3-7b | MATH-500 | 4459.161195 | 541.445719 | 0.108276 | 0.726143 |
| olmo3-7b | GSM8K | 4330.323669 | 540.067717 | 0.110888 | 0.737717 |
| olmo3-7b | 2WikiMultihopQA | 0.000000 | 1355.950713 | 1.000000 | N/A |
| olmo3-7b | HotpotQA | 0.000000 | 1410.930144 | 1.000000 | N/A |

For all CODE/QA rows the answer fraction is 1.000000 and reasoning total is 0.000000: the relevant region is the whole code answer or short QA answer, respectively. This excludes a separately scored reasoning-span explanation for the CODE response or QA sign. It does not locate effects within code syntax, whitespace, semantics, or particular QA answer tokens; those per-token likelihoods are absent.

Distillation is read from the existing V12 eval JSON in its **original corpus-token version**, subtracting each run's own dense anchor. These are not v27 measurements. `answer_only` and `no_code_fence` below name training recipes, not different evaluation losses and not fixed-model format controls. All final configurations are reported, including sign exceptions; dependent trajectory snapshots are preserved separately in summary.json and do not enter the final-configuration counts.

| Capability | Final configs | Positive Δ | Negative Δ | Zero Δ | Min Δ | Max Δ |
| --- | --- | --- | --- | --- | --- | --- |
| math | 45 | 40 | 5 | 0 | -0.130148 | +1.122102 |
| code | 45 | 43 | 2 | 0 | -0.160322 | +0.868772 |
| qa | 45 | 3 | 42 | 0 | -2.345076 | +5.383622 |

| Model | Run | CODE dense | CODE post | CODE Δ | QA dense | QA post | QA Δ | CODE / QA target tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | gpt-5.6-luna_full_600 | 1.590581 | 1.667076 | +0.076495 | 5.626326 | 4.344442 | -1.281884 | 3461 / 271 |
| Qwen3-1.7B | gpt-5.6-luna_full_600 | 1.602138 | 1.441816 | -0.160322 | 6.391720 | 4.046644 | -2.345076 | 3461 / 271 |
| gemma3-12b | claude-sonnet-4-6_full_600 | 0.750526 | 0.845494 | +0.094968 | 5.431462 | 5.160156 | -0.271305 | 4516 / 251 |
| gemma3-12b | gpt-5.6-luna_full_600 | 0.750526 | 0.888950 | +0.138424 | 5.431462 | 4.245222 | -1.186239 | 4516 / 251 |
| gemma3-1b | claude-sonnet-4-6_answer_only_600 | 1.060092 | 1.132058 | +0.071966 | 5.567791 | 4.991005 | -0.576787 | 4516 / 251 |
| gemma3-1b | claude-sonnet-4-6_full_600 | 1.060812 | 1.102469 | +0.041657 | 5.567138 | 5.521663 | -0.045474 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_answer_only_600 | 1.060812 | 1.170090 | +0.109278 | 5.567138 | 4.424614 | -1.142524 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_150 | 1.060812 | 1.138674 | +0.077862 | 5.567138 | 4.081238 | -1.485900 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_300 | 1.060092 | 1.186780 | +0.126688 | 5.567791 | 4.129482 | -1.438309 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_300_seed1 | 1.060341 | 1.100670 | +0.040329 | 5.574670 | 4.146539 | -1.428131 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_300_seed2 | 1.060341 | 1.099701 | +0.039360 | 5.574670 | 4.168700 | -1.405970 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_600 | 1.060092 | 1.198406 | +0.138314 | 5.567791 | 4.820468 | -0.747323 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_600_seed1 | 1.060341 | 1.123090 | +0.062749 | 5.574670 | 4.087898 | -1.486772 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_600_seed2 | 1.060341 | 1.122177 | +0.061836 | 5.574670 | 4.075573 | -1.499097 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_600_uxseen | 1.060812 | 1.202668 | +0.141857 | 5.567138 | 4.789280 | -0.777857 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_600_uxseen_seed1 | 1.060812 | 1.203333 | +0.142521 | 5.567138 | 5.134711 | -0.432427 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_600_uxseen_seed2 | 1.060812 | 1.186587 | +0.125775 | 5.567138 | 4.917144 | -0.649994 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_75 | 1.060812 | 1.070998 | +0.010186 | 5.567138 | 4.376494 | -1.190644 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_75_seed1 | 1.060341 | 1.060590 | +0.000249 | 5.574670 | 4.811877 | -0.762793 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_75_seed2 | 1.060341 | 1.064050 | +0.003709 | 5.574670 | 4.819877 | -0.754793 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_75_uxseen | 1.060812 | 1.929584 | +0.868772 | 5.567138 | 10.950759 | +5.383622 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_75_uxseenE | 1.060812 | 1.069669 | +0.008857 | 5.567138 | 4.379980 | -1.187158 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_75_uxseen_seed1 | 1.060812 | 1.813496 | +0.752685 | 5.567138 | 10.453063 | +4.885925 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_full_75_uxseen_seed2 | 1.060812 | 1.863209 | +0.802397 | 5.567138 | 10.862363 | +5.295225 | 4516 / 251 |
| gemma3-1b | gpt-5.6-luna_no_code_fence_600 | 1.060812 | 1.266580 | +0.205768 | 5.567138 | 4.779133 | -0.788004 | 4516 / 251 |
| gemma3-270m | claude-sonnet-4-6_answer_only_600 | 1.190296 | 1.242554 | +0.052259 | 5.275741 | 4.409176 | -0.866565 | 4516 / 251 |
| gemma3-270m | claude-sonnet-4-6_full_600 | 1.190296 | 1.313718 | +0.123422 | 5.275741 | 4.903386 | -0.372354 | 4516 / 251 |
| gemma3-270m | gpt-5.6-luna_answer_only_600 | 1.190960 | 1.322354 | +0.131394 | 5.274807 | 4.218563 | -1.056244 | 4516 / 251 |
| gemma3-270m | gpt-5.6-luna_full_150 | 1.190960 | 1.271036 | +0.080076 | 5.274807 | 4.126494 | -1.148313 | 4516 / 251 |
| gemma3-270m | gpt-5.6-luna_full_16 | 1.190296 | 1.192704 | +0.002408 | 5.275741 | 5.275336 | -0.000405 | 4516 / 251 |
| gemma3-270m | gpt-5.6-luna_full_300 | 1.190960 | 1.352358 | +0.161398 | 5.274807 | 4.100349 | -1.174458 | 4516 / 251 |
| gemma3-270m | gpt-5.6-luna_full_600 | 1.190296 | 1.363430 | +0.173134 | 5.275741 | 4.622448 | -0.653293 | 4516 / 251 |
| gemma3-270m | gpt-5.6-luna_full_75 | 1.190960 | 1.223123 | +0.032163 | 5.274807 | 4.323705 | -0.951102 | 4516 / 251 |
| gemma3-4b | claude-sonnet-4-6_full_600 | 0.801788 | 0.847127 | +0.045339 | 5.473855 | 5.088894 | -0.384960 | 4516 / 251 |
| gemma3-4b | gpt-5.6-luna_full_150 | 0.801705 | 0.912090 | +0.110385 | 5.477154 | 4.018177 | -1.458977 | 4516 / 251 |
| gemma3-4b | gpt-5.6-luna_full_300 | 0.801705 | 0.999419 | +0.197714 | 5.477154 | 4.510085 | -0.967069 | 4516 / 251 |
| gemma3-4b | gpt-5.6-luna_full_300_seed1 | 0.801788 | 0.868357 | +0.066569 | 5.473855 | 3.906188 | -1.567667 | 4516 / 251 |
| gemma3-4b | gpt-5.6-luna_full_300_seed2 | 0.801788 | 0.860745 | +0.058957 | 5.473855 | 4.268364 | -1.205491 | 4516 / 251 |
| gemma3-4b | gpt-5.6-luna_full_600 | 0.801788 | 0.887483 | +0.085695 | 5.473855 | 4.255852 | -1.218003 | 4516 / 251 |
| gemma3-4b | gpt-5.6-luna_full_600_seed1 | 0.801788 | 0.881615 | +0.079827 | 5.473855 | 4.283429 | -1.190426 | 4516 / 251 |
| gemma3-4b | gpt-5.6-luna_full_600_seed2 | 0.801788 | 0.888840 | +0.087052 | 5.473855 | 4.221489 | -1.252366 | 4516 / 251 |
| gemma3-4b | gpt-5.6-luna_full_75 | 0.801705 | 0.823074 | +0.021368 | 5.477154 | 4.150523 | -1.326631 | 4516 / 251 |
| gemma3-4b | gpt-5.6-luna_full_75_seed1 | 0.801788 | 0.814410 | +0.012622 | 5.473855 | 4.340949 | -1.132906 | 4516 / 251 |
| gemma3-4b | gpt-5.6-luna_full_75_seed2 | 0.801788 | 0.817842 | +0.016054 | 5.473855 | 4.260769 | -1.213085 | 4516 / 251 |
| olmo3-7b | gpt-5.6-luna_full_600 | 1.129378 | 1.125760 | -0.003618 | 5.026718 | 4.152698 | -0.874020 | 3455 / 262 |

For a compact size/family view, the original gpt-5.6-luna/full/600 run in each available model is shown below (native V12 corpus nats/token). This is a named configuration slice; it does not replace the complete table or select by outcome.

| Model | CODE Δ | QA Δ |
| --- | --- | --- |
| Qwen3-0.6B | +0.076495 | -1.281884 |
| Qwen3-1.7B | -0.160322 | -2.345076 |
| gemma3-12b | +0.138424 | -1.186239 |
| gemma3-1b | +0.138314 | -0.747323 |
| gemma3-270m | +0.173134 | -0.653293 |
| gemma3-4b | +0.085695 | -1.218003 |
| olmo3-7b | -0.003618 | -0.874020 |

**CODE response:** the sign exceptions above exclude a universal positive CODE response across the observed models/configurations. The response concerns the code answer target, not a scored rationale. The `no_code_fence` training recipe's delta is retained in the complete table; that comparison changes training and cannot isolate a pure evaluation-format contribution.

**QA sign:** the complete-table positive exceptions exclude an unconditional negative-QA rule. The empty-r identity excludes an evaluation-reasoning-conditioning explanation. Short QA targets (see the recorded token counts above) and the measured format sensitivity leave surface/answer-token effects plausible. Neither sign establishes a behavioral accuracy improvement or decline.

**Identification limit at closeout:** no v27 dense/distilled pair exists locally; V12 eval JSON contains aggregate losses, not item NLL/byte lengths or token-region losses. Therefore the separate **numerical distillation deltas** of the three v27 losses, any body-versus-wrapper causal share, and V12 per-byte distillation deltas are unavailable (JSON null), not zero. The empty-r identity locates CODE/QA at the answer target but does not reconstruct new v27 measurements. Borrowing bytes or dense anchors from the pruning panel would conflate versions and is refused. The measurement definitions are closed with this boundary; the original prospective results stay in their original version.

Reproduce using only the standard library:

```bash
CUDA_VISIBLE_DEVICES='' python analysis/v27b_scoring_readout.py
python -m pytest -q tests/test_v27b.py
```

Outputs: `results/v27b-readout/summary.json` and this document. The JSON preserves unrounded values, source metric definitions, exclusions, all format variants in both units/weightings, same-item ordering comparisons, all V12 losses, and SHA-256 hashes of every input. Input files and prediction artifacts are never written. The source panel is small and fixed; no independent-seed uncertainty or general measurement-invariance claim is inferred from these descriptive values.

Source inventory (raw-file SHA-256; paths relative to repository root):

| Kind | Source | SHA-256 |
| --- | --- | --- |
| v27_panel | `results/v27-scoring-units/gemma3-1b/dense_prune-d0.7/all-L_full-scoring-units.json` | 7f0e6a3ae501d71c19502a0547ab4258886f0c482c66adfc876ec07f75179f14 |
| v27_panel | `results/v27-scoring-units/gemma3-4b/dense_prune-d0.7/all-L_full-scoring-units.json` | bedd06e3d38eb2ca9c7b08e55a54c22d1ad130eb6bbce993365e169374398146 |
| v27_panel | `results/v27-scoring-units/olmo3-7b/dense_prune-d0.7/all-L_full-scoring-units.json` | 1bbce39d150bdd1ca46812e5c38549939c6696ca851953b893ff83e655431d37 |
| eval | `results/v12-distill/Qwen3-0.6B/gpt-5.6-luna_full_600/eval.json` | c4d2b2f3a7edebefa8274569c1b277abc6f9f41aa3a432e9bf1570709fa9d324 |
| eval | `results/v12-distill/Qwen3-1.7B/gpt-5.6-luna_full_600/eval.json` | 41c93b0dd4f3c50c32e851824ff6af47fe7d4e8794876f4f425063dbaf4ae671 |
| eval | `results/v12-distill/gemma3-12b/claude-sonnet-4-6_full_600/eval.json` | 174038dc28beaa897bdc0299a6386ac427b1db98e65b6c6247b0d804aa11eecc |
| eval | `results/v12-distill/gemma3-12b/gpt-5.6-luna_full_600/eval.json` | e5ac2232c3e18963c8de7d065ad0f61db872b326cc7ab1ec7fa6b42f66859fc4 |
| eval | `results/v12-distill/gemma3-1b/claude-sonnet-4-6_answer_only_600/eval.json` | 27cab6c503644ebc56f3d2c1fe25da03532e4300de32136f64e5b54e7e27a99d |
| eval | `results/v12-distill/gemma3-1b/claude-sonnet-4-6_full_600/eval.json` | 16c0578a07055e0349e34cdfa2a47dd9afe7c256f69fc131d0aeb5e5539e52db |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_answer_only_600/eval.json` | 3d957a63862d6c14324ce3c9d323e078e0df353b45fc4ec6c483a9dbcb822850 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_150/eval.json` | fe8c35fed5571b3bd1108b69d6f56a084abbab60a94aeb71f22747b161b1e82d |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_300/eval.json` | 94fa40f2f499b61f3dc706a9bfb802020ae43984a314a22978332ba37eee1f46 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_300_seed1/eval.json` | 777e74c68ff1f15dfb3e7f6c336194e5a3bc268fdf5046f7d5bf47edce8e6097 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_300_seed2/eval.json` | 0ad8f25acf28f022cf7c56e4a2da75575c4ade95e00ff465ec6de033a8b449f7 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_600/eval.json` | d46e98c48078365c0481258d8010f21abba641986361d6240bcf80f3dee0285d |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_600_seed1/eval.json` | fce7bbdb233f91a2e8ea80b85636d0c4514d9a9dbeb2bb11c8ce4990b8c19bed |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_600_seed2/eval.json` | 332e903ebb0c1fc9d9ae8f41f3d2fd6cd697df4b7cf4e946080c2df6d5c359e7 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_600_uxseen/eval.json` | 90032a3e351dd97186fad038d44e40f289bb8093c9f2ede20826055ff8145858 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_600_uxseen/trajectory/update-00000000/eval.json` | 66bb1ef5ddf771a70ce7ec548455a3e2ab668ba33df30c099465a7eade8ff39d |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_600_uxseen/trajectory/update-00000105/eval.json` | 254d5891cc14e232501bfd3a494d5e41a5d11741f13ce762e258202a6cd0cdaf |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_600_uxseen/trajectory/update-00000209/eval.json` | f2f7f78b8d9cc7038eeecec386fda1b365af57c777c97f7d337ca0b59c8adb7a |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_600_uxseen_seed1/eval.json` | 812931c2e0411b3097ae4d6484efeaf984303a67ba3e1f70a414eed952889c26 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_600_uxseen_seed1/trajectory/update-00000000/eval.json` | b9d68554d8d9275aadee46e8f7df62fe9e73978f7dbe5907b87a9aa6db1a4b34 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_600_uxseen_seed1/trajectory/update-00000104/eval.json` | 8205005238515b69661c660569facea8cdaaa416d0410f4eb2388279fdecdd71 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_600_uxseen_seed1/trajectory/update-00000210/eval.json` | d5483019db28aec778d30bae2c1655a9c8c4e7891565803822a12399db1abd8b |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_600_uxseen_seed2/eval.json` | 319948cef58a25a9593c6864acf3a95a4af9b06505fb2af140cd5a5d5dd471c3 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_600_uxseen_seed2/trajectory/update-00000000/eval.json` | 2ba23159affa5b3bd3bfa9893a322d2351bd2c985e0b1225c91597661b3a19ae |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_600_uxseen_seed2/trajectory/update-00000105/eval.json` | b767ced1c70934a6a70e1c1c94bc7378af3f714cf099cd7c2c5396d068a49f5d |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_600_uxseen_seed2/trajectory/update-00000210/eval.json` | 50e52fc1d34e3d25f6a51c1fe9e1268be5c74ab91651faa92ae9ef4de195858c |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75/eval.json` | 8bf9c5acc3585785ac01642a6e2a6882142918a1fb6b8f314ea2a4773b43989e |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_seed1/eval.json` | 4c5a7b365273bd2cd1184911a05c431bedd8e97cfc94af3b94c048a4cf97b87b |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_seed2/eval.json` | 3c0becfa9bc2329b7de4dcd55bb92022a078d3599fde02256c9619f4adf2d55c |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseen/eval.json` | 1b6cd46a13596fd6e4e541fa7b8468a4d98fb52cf07b45c7855de188e38e0295 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseen/trajectory/update-00000000/eval.json` | 70556f8eb8b88931c1be66e1f4ac351f2a33d6a765c5d110d4c84b7584fe49f7 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseen/trajectory/update-00000105/eval.json` | 7e5e52b3fb2c6928607edfba1835524f2c17030775ca9e2267b07d9624ddb1a1 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseen/trajectory/update-00000209/eval.json` | 5b1456eeba97f82d0b1cf3af06e7b9465e4080369f1e7e0798e47178f7b9a3b5 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseenE/eval.json` | e2fec9f6846f76f07507cb635973ea228070eec6b39cd1b8d85d1bbd15e31a34 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseenE/trajectory/update-00000000/eval.json` | 251047d698d6fc186028d4a17743e2a8c460171626ef99d2f763778cf48bc360 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseenE/trajectory/update-00000014/eval.json` | 2b79a9d07ea0801af743e6b2125f6e83100b21a249006c3d1db6e38b6bfd377f |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseenE/trajectory/update-00000027/eval.json` | 37a48d21b653f644d0f3c535d6db6c0b880d862467efc5b6eb3dc37c4a9cb5a8 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseenE_seed1/trajectory/update-00000000/eval.json` | 24a110b74d0d15cf6378d47e92b008f24bcbe78453cf72da37056ea4ac6b070b |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseenE_seed1/trajectory/update-00000013/eval.json` | d38de68aa39f790455119c9319f045a69105512fce0daee170e454a625f5f3b7 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseen_seed1/eval.json` | ebe7f8ca106bfbd9c5cab4920909244788cbb7e889198ad06afebe577ef56efb |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseen_seed1/trajectory/update-00000000/eval.json` | 5d6600c2bfc92f941980ce8df25e3bf0cc99628768ac98104be6d38980a1a40d |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseen_seed1/trajectory/update-00000105/eval.json` | e5fcf481b82357e2d0ef95aca634d58b51e545f8e648ced7b967c22671c350a5 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseen_seed1/trajectory/update-00000209/eval.json` | d2d2982da91ecd70ac43459dfdff3c18ae8c434ae518f4d387c9a32855f08438 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseen_seed2/eval.json` | e33f68682aadc3c2c5c9b5f8ba1431bacb45037ed78862691cb68df5085d4b93 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseen_seed2/trajectory/update-00000000/eval.json` | cb2f7ba5cb89fc35fce0915ef0008cf94660b0af411058a0c217fbe3e652c748 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseen_seed2/trajectory/update-00000105/eval.json` | 11a57192dcbe523d3c268208f7e155a584eca98e2d88bdf993a7ec52b65e3fa0 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseen_seed2/trajectory/update-00000209/eval.json` | 7fb3d6194bafc0720bbd39a949c38142b1c2a721c557b2b54c324e5f3779d923 |
| eval | `results/v12-distill/gemma3-1b/gpt-5.6-luna_no_code_fence_600/eval.json` | df08fdecc410e6680a2f1b425a73428155917d0be01450158967435ad550a480 |
| eval | `results/v12-distill/gemma3-270m/claude-sonnet-4-6_answer_only_600/eval.json` | bf4c131b9dadaabb78d711905bd870a50db6897f017044505c4902b3f411524b |
| eval | `results/v12-distill/gemma3-270m/claude-sonnet-4-6_full_600/eval.json` | d0ee205fed6037ab9e5fd542c1308169f4087b83e84b05d1a0cbc452576e08a1 |
| eval | `results/v12-distill/gemma3-270m/gpt-5.6-luna_answer_only_600/eval.json` | 54a444f1477e48586288c65efd93667b9eb86ffda10078251ee17bbf94a50e0a |
| eval | `results/v12-distill/gemma3-270m/gpt-5.6-luna_full_150/eval.json` | 15a7bc2eca930e9da935329b20e527c3bfb6fc7c122497f13b0c6bee16867245 |
| eval | `results/v12-distill/gemma3-270m/gpt-5.6-luna_full_16/eval.json` | cf855b842bcea878673181fd724f703b40c80372707cccf5c016663c65bcfb0f |
| eval | `results/v12-distill/gemma3-270m/gpt-5.6-luna_full_300/eval.json` | 75770bc645317504df196c87a4281d531bcff6f754d2dae699f8994737b2f43f |
| eval | `results/v12-distill/gemma3-270m/gpt-5.6-luna_full_600/eval.json` | 46a1d073a01f2527522f5e010976aae77ee2c5279ab49beb20047812c6493383 |
| eval | `results/v12-distill/gemma3-270m/gpt-5.6-luna_full_75/eval.json` | f747d43bdce820ef624f32e876972cad917902927fb8cf5695acf14c3101a5bf |
| eval | `results/v12-distill/gemma3-4b/claude-sonnet-4-6_full_600/eval.json` | 1bae1161f09faddd59da022582ce2bde014a68163846ffd7b0ca3cd663d12843 |
| eval | `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_150/eval.json` | 1106bef6275997ab17f11320c607e49102d18459bf050a51862d631b51af45be |
| eval | `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_300/eval.json` | fd8fdd25014c66eebc97ebd223a98ad6696a328be1d43ce23c4d5ffcd66ab8e6 |
| eval | `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_300_seed1/eval.json` | 4f67874d0acb7b8e46d8abfa5a169fd5283d9d7a1a646452c9ae45af3cea4741 |
| eval | `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_300_seed2/eval.json` | 6b1f7007699465590e0aba7b4843eb8fa29e4612d5a574182c4fe5f62b93c12b |
| eval | `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_600/eval.json` | debcaa65071cc4f341798653ae7a2e37ebb7f7bce2a667197e77ad78b0f4d620 |
| eval | `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_600_seed1/eval.json` | 78791c1ad64e64963f66042cc5b150c0977e341d24c4e3936faf172d28a64fcc |
| eval | `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_600_seed2/eval.json` | 308a8fbb4ad209f8fc9d13d43b986fd2324e800a7db763d70d88a20a13a2c315 |
| eval | `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_75/eval.json` | d9784279cd0a372156b853ca69f5711cbe6797e85bb0f335ea27f55a99ec7e85 |
| eval | `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_75_seed1/eval.json` | 898b1b3134bff5b90a319919597171ce8eb7becad7f7a7abe434d32dc78ec50f |
| eval | `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_75_seed2/eval.json` | 60f38fb25f8e9dd7298b117119bf7549b600351f70fc9e0aed994640ad8844c7 |
| eval | `results/v12-distill/olmo3-7b/gpt-5.6-luna_full_600/eval.json` | f2623f1257bfb2bdd78f83a1aba0ed054d3ad53b3443c261acfeee0e00f30231 |
