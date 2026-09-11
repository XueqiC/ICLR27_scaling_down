The locked rule is confirmed for a capability if its regret is within the v64 map's and below quant-only's on the new panel; otherwise it is reported as retrospective.

Operational rule, fixed before measurement: mean regret over the same 4 states x 17 budgets (equal state and budget weights), locked <= v64 and locked < quant-only, with no fitted margin. All three policies are feasible on the full planned panel. Missing measurements prevent confirmation; never silently shrink the panel. The max map uses max_c(L_c-source_L0c), as in v64. Quant-only pools channel and grouped RTN and uses locked predictions; its v64-prediction variant is also reported. KD candidates reuse two historical v39 outcomes, not new independent KD experiments. Both students are purged from both KD fits, as in v64. Constants are v39's arithmetic mean-delta baseline on the remaining development students; math uses its +D0 linear form. QA claims apply only to 2Wiki. No-clear-winner sets reuse v64's frozen development LOSO MAE thresholds for both maps, unchanged; they are retrospective heuristics, not calibrated uncertainty for the locked predictors. Storage is nominal matrix storage: d, b/16, (b+16/g)/16, and N_student/N_source; sparse index overhead is excluded.

math: **confirmed**

| Policy | Feasible / 68 | Mean regret (nats) | Method agreement |
|---|---:|---:|---:|
| locked-rule | 68 | 0.000006 | 0.985 |
| v64-law | 68 | 0.014942 | 0.985 |
| quant-only | 68 | 0.006754 | 0.941 |
| prune-only | 36 | 0.306644 | 0.000 |
| distill-only | 17 | 0.168404 | 0.059 |
| cheapest | 68 | 1.771608 | 0.735 |
| v64-quant-only | 68 | 0.021693 | 0.941 |
| v64-prune-only | 36 | 0.306644 | 0.000 |
| v64-distill-only | 17 | 0.168404 | 0.059 |

| Candidate set | Method coverage | Exact config coverage | Ambiguous cells |
|---|---:|---:|---:|
| locked-rule | 1.000 | 1.000 | 38 |
| v64-law | 1.000 | 0.574 | 37 |

code: **confirmed**

| Policy | Feasible / 68 | Mean regret (nats) | Method agreement |
|---|---:|---:|---:|
| locked-rule | 68 | 0.004327 | 0.956 |
| v64-law | 68 | 0.112358 | 1.000 |
| quant-only | 68 | 0.010199 | 0.971 |
| prune-only | 36 | 0.347673 | 0.000 |
| distill-only | 17 | 0.247644 | 0.059 |
| cheapest | 68 | 2.434612 | 0.750 |
| v64-quant-only | 68 | 0.118212 | 0.971 |
| v64-prune-only | 36 | 0.347673 | 0.000 |
| v64-distill-only | 17 | 0.247644 | 0.059 |

| Candidate set | Method coverage | Exact config coverage | Ambiguous cells |
|---|---:|---:|---:|
| locked-rule | 1.000 | 0.544 | 38 |
| v64-law | 1.000 | 0.338 | 38 |

qa: **confirmed**

| Policy | Feasible / 68 | Mean regret (nats) | Method agreement |
|---|---:|---:|---:|
| locked-rule | 68 | 0.141007 | 0.765 |
| v64-law | 68 | 0.213573 | 0.618 |
| quant-only | 68 | 0.252577 | 0.485 |
| prune-only | 36 | 0.201737 | 0.694 |
| distill-only | 17 | 0.052428 | 0.588 |
| cheapest | 68 | 1.351780 | 0.632 |
| v64-quant-only | 68 | 0.377332 | 0.485 |
| v64-prune-only | 36 | 0.231540 | 0.694 |
| v64-distill-only | 17 | 0.005406 | 0.588 |

| Candidate set | Method coverage | Exact config coverage | Ambiguous cells |
|---|---:|---:|---:|
| locked-rule | 1.000 | 0.265 | 44 |
| v64-law | 0.971 | 0.338 | 42 |

multi: **retrospective**

| Policy | Feasible / 68 | Mean regret (nats) | Method agreement |
|---|---:|---:|---:|
| locked-rule | 68 | 0.002023 | 1.000 |
| v64-law | 68 | 0.001850 | 0.971 |
| quant-only | 68 | 0.008089 | 0.926 |
| prune-only | 36 | 0.367926 | 0.000 |
| distill-only | 17 | 0.239662 | 0.059 |
| cheapest | 68 | 2.426280 | 0.721 |
| v64-quant-only | 68 | 0.007733 | 0.926 |
| v64-prune-only | 36 | 0.367926 | 0.000 |
| v64-distill-only | 17 | 0.239662 | 0.059 |

| Candidate set | Method coverage | Exact config coverage | Ambiguous cells |
|---|---:|---:|---:|
| locked-rule | 1.000 | 0.794 | 38 |
| v64-law | 1.000 | 0.912 | 36 |
