# Behavioral loss-to-accuracy links

Generated 2026-09-04 by `analysis/v19_links.py` from existing V15 artifacts only. This was a CPU-only refit; it performed no inference or training and did not mutate its inputs.

## Result

The fitted link is `A=A_max/(1+exp(k(L-L_half)))`. Rows are equally weighted checkpoint observations. Parameter intervals are deterministic model-cluster bootstrap 95% intervals, so all four checkpoints from a sampled model move together. They do not measure benchmark-item or seed uncertainty; every checkpoint has one recorded seed and 64 evaluation items.

| Outcome | n | A_max | L_half (95% CI) | k (95% CI) | shared LOFO MAE | family-specific transport LOFO MAE | paired shared−family error (95% CI) | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GSM8K accuracy | 28 | 1.000 | 0.881 [0.835, 0.998] | 7.911 [4.746, 11.729] | 0.077 | 0.068 | 0.008 [-0.009, 0.027] | compatible with a shared form; no resolved family-specific gain |
| MBPP pass@1 | 28 | 0.596 | 1.322 [0.888, 1.681] | 7.590 [2.301, 100.000] | 0.128 | 0.116 | 0.012 [-0.013, 0.037] | compatible with a shared form; no resolved family-specific gain |
| TriviaQA exact match | 28 | 0.674 | 2.793 [2.171, 3.669] | 6.909 [0.613, 15.195] | 0.154 | 0.154 | 0.000 [-0.038, 0.036] | compatible with a shared form; no resolved family-specific gain |
| TriviaQA token F1 | 28 | 1.000 | 2.968 [2.445, 3.836] | 0.684 [0.515, 4.473] | 0.105 | 0.095 | 0.010 [-0.010, 0.029] | compatible with a shared form; no resolved family-specific gain |

Here, “family-specific transport” is deliberately leakage-free: each training family gets its own sigmoid, and their predictions are averaged equally for the never-seen family. A literal `g_{c,f}` has no learned parameters for a held-out family; fitting it on that family's test rows would make the comparison in-sample. The singleton Gemma-4 and Muse families each supply only four curve points, so family-specific conclusions are thin.

**Interpretation.** Math, code, and both QA metrics are compatible with a shared cross-family sigmoid within each outcome because no family-specific transport improvement is resolved by the paired held-out intervals. This is strongest for the existence of a monotone form, not equality across capabilities: code has a weakly identified steepness whose bootstrap reaches the upper fit bound, TriviaQA EM has broad parameter intervals, and token-F1 has a much shallower fitted slope. Thus code/QA support shared-within-capability behavior provisionally, while the link parameters and the literal zero-accuracy cliff remain capability-specific.

## Cliff coincidence

A loss cliff is the first measured pruning cell with `L-L_dense > 1` nat. “At floor” means at most one correct item (`<=1/64`) for accuracy/pass@1/EM and `F1<=0.05` for token-F1. These thresholds are declared diagnostics, not fitted change points.

| Outcome | observed loss cliffs / 7 | already at accuracy floor | same first measured density | loss-cliff censored |
|---|---:|---:|---:|---:|
| GSM8K accuracy | 4/7 | 4/4 | 3/4 | 3 |
| MBPP pass@1 | 4/7 | 3/4 | 2/4 | 3 |
| TriviaQA exact match | 5/7 | 1/5 | 1/5 | 2 |
| TriviaQA token F1 | 5/7 | 0/5 | 0/5 | 2 |

The exact-zero coincidence test is stricter than the sigmoid half-accuracy transition. In particular, TriviaQA often retains partial credit at the first >1-nat loss crossing, so the data support a monotone behavioral transition more strongly than literal loss-cliff = zero-accuracy coincidence.

## QA loss-improvement audit

**Flag: lower QA loss does not reliably imply higher QA accuracy in this panel.** There are 4 QA metric-cells below their model's dense loss and only 0 have strictly higher corresponding accuracy.

| Model | density | QA metric | Δ loss vs dense | Δ accuracy vs dense |
|---|---:|---|---:|---:|
| olmo3-32b | 0.8 | qa_em | -0.0133 | +0.0000 |
| olmo3-32b | 0.7 | qa_em | -0.0653 | -0.0156 |
| olmo3-32b | 0.8 | qa_f1 | -0.0133 | -0.0016 |
| olmo3-32b | 0.7 | qa_f1 | -0.0653 | -0.0108 |

This directly limits the stronger “QA free lunch” interpretation: a negative teacher-forced QA loss delta is not, by itself, evidence of improved TriviaQA behavior. The present audit is pruning-only and therefore does not validate distillation's negative QA loss delta behaviorally.

## Fixed-evaluation provenance

All 28 requested files passed all gates: V15, `accuracy_benchmark_mode=easy`, GSM8K/MBPP/TriviaQA registry, 64 measurement examples per capability, `analysis.v15_accuracy_link.build_easy_probes`, and a decoding record containing both domain-delimiter stopping and post-hoc first-block truncation. Paths containing `_zeroshot_floor` are rejected before scores are read.

Inputs: `results/v15-accuracy/<model>/{dense,prune-d0.8,prune-d0.7,prune-d0.6}__easy/accuracy.json`. Exact paths and SHA-256 hashes are recorded in `results/v19-links/summary.json`; the fitted observations are in `results/v19-links/observations.csv`.

## Limitations

- Bootstrap count: 500; model clusters: 7; raw families: Gemma-3, Gemma-4, Muse, and OLMo-3.
- Each accuracy is based on only 64 items, one decoding run, and seed 0. Apparent non-monotonic cells can be item-sampling noise.
- The four loss ranges differ substantially. OLMo-3 never enters the steep accuracy-collapse regime at the tested densities, while singleton-family links are nearly saturated by four points.
- These are absolute loss links from each benchmark's own V15 measurement prompts. They should not be substituted with V6 losses, especially for QA, whose probe construction differs.
