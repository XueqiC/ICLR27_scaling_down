# V56 distillation forms

Mode `dev`. Dev: 12/12 runs, 48 checkpoints (144 capability observations); 12 runs have all four checkpoints. Test: 0 runs, 0 checkpoints.

Tc = completion_tokens_seen; E = Tc / registered D_U_completion; delta = post_training - dense, in nats. A cluster is (student, U, data_seed), so all checkpoints/capabilities from a run stay together. Metrics pool held-out checkpoints; bias = prediction - actual. Missing runs are listed in JSON.

L0 is standardized separately per capability, equally over available dev students; log N uses the two dev students. These fixed, pretreatment-only references are used in both CV variants and test prediction. Regression column scaling uses training data only. Ridge = 0.001; intercepts are unpenalized. F1/F2 columns are scaled without centering to preserve zero response at zero exposure.

T_star is chosen from {17.5k, 35k, 70k, 140k, 280k} by pooled training SSE across capabilities, separately in every fold. Final fits use all dev points. Test outcomes never select a model, scale, timescale or subset.

Part A: each cell is **MAE/signed bias**. P is the total coefficient count across all three capabilities; F2 additionally selects one five-choice timescale. LOCO leaves one run out; LOSO leaves one student out.

| Form | P | LOCO math | LOCO code | LOCO qa | LOSO math | LOSO code | LOSO qa |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 0 | 0.152/-0.152 | 0.178/-0.178 | 1.429/+0.143 | 0.152/-0.152 | 0.178/-0.178 | 1.429/+0.143 |
| constant | 3 | 0.095/+0.000 | 0.103/+0.000 | 1.400/+0.000 | 0.093/+0.000 | 0.100/+0.000 | 1.358/-0.000 |
| T-only | 6 | 0.082/+0.000 | 0.072/+0.000 | 1.092/-0.000 | 0.089/+0.000 | 0.070/+0.000 | 1.075/-0.000 |
| E-only | 6 | 0.073/-0.003 | 0.055/-0.002 | 0.798/-0.029 | 0.091/+0.000 | 0.056/+0.000 | 0.949/+0.000 |
| surface:L0 | 12 | 0.072/+0.000 | 0.054/-0.000 | 0.812/+0.001 | 0.091/+0.000 | 0.054/+0.000 | 0.944/-0.000 |
| surface:logN | 12 | 0.072/+0.000 | 0.054/-0.000 | 0.812/+0.001 | 0.091/+0.000 | 0.054/+0.000 | 0.944/+0.000 |
| F1:L0 | 6 | 0.057/+0.003 | 0.052/+0.003 | 1.518/+0.662 | 0.152/-0.152 | 0.178/-0.178 | 1.429/+0.143 |
| F1:logN | 6 | 0.057/+0.003 | 0.052/+0.003 | 1.518/+0.662 | 0.152/-0.152 | 0.178/-0.178 | 1.429/+0.143 |
| F2:L0 | 12 | 0.062/-0.007 | 0.057/-0.005 | 0.706/-0.069 | 0.152/-0.152 | 0.178/-0.178 | 1.429/+0.143 |
| F2:logN | 12 | 0.062/-0.007 | 0.057/-0.005 | 0.706/-0.069 | 0.152/-0.152 | 0.178/-0.178 | 1.429/+0.143 |

Bases: t = log(1+Tc/35000), e = log(1+E), s = 1-exp(-Tc/T_star). T-only = a+b*t; E-only = a+b*e; surface = a+b*t+c*e+d*z; F1 = (a+lambda*z)*e; F2 = (a+lambda*z)*s+(b+mu*z)*e.

With two dev students, standardized L0 and log N are identical up to sign within each capability; Part A cannot distinguish them on dev. LOSO has only one student in training, so descriptor effects are not separately identifiable from base coefficients. Ranks and ridge solutions are recorded in JSON.

Part B uses primary L0 and **equal fitted coefficient counts**, with a trade between offsets and capability slopes. Shared shape: three capability offsets plus q shared slopes (q=1 for E-only, 2 for T+E, 4 for F2). Per-capability: one shared offset plus q+2 capability-specific slopes. E-only has all three E slopes (P=4). T+E gives one capability both slopes and each other capability one slope (P=5; 12 subsets). F2 gives each capability one of {s,z*s} and one of {e,z*e} (P=7; 64 subsets). These are reduced, nonnested per-capability models, not unrestricted fits. Subsets are chosen by training SSE within each fold; the extra discrete search is reported and is not an equal search budget. Both F2 models also select one five-choice T_star. Unrestricted per-capability offsets/slopes would require 6, 9 and 15 coefficients.

Part B: LOCO MAE; gain = shared minus per-capability. `>0.02` uses unrounded nats and is a descriptive threshold, not a significance test.

| Structure | P shared/per | Capability | Shared MAE | Per-cap MAE | Gain | >0.02 |
|---|---:|---|---:|---:|---:|:---:|
| E-only | 4/4 | math | 0.431 | 0.409 | 0.022 | yes |
| E-only | 4/4 | code | 0.417 | 0.412 | 0.006 | no |
| E-only | 4/4 | qa | 0.967 | 1.150 | -0.183 | no |
| E-only | 4/4 | macro | 0.605 | 0.657 | -0.052 | no |
| T+E | 5/5 | math | 0.439 | 0.327 | 0.111 | yes |
| T+E | 5/5 | code | 0.419 | 0.314 | 0.105 | yes |
| T+E | 5/5 | qa | 0.979 | 0.853 | 0.126 | yes |
| T+E | 5/5 | macro | 0.612 | 0.498 | 0.114 | yes |
| F2 | 7/7 | math | 0.536 | 0.145 | 0.390 | yes |
| F2 | 7/7 | code | 0.527 | 0.107 | 0.420 | yes |
| F2 | 7/7 | qa | 1.044 | 0.826 | 0.219 | yes |
| F2 | 7/7 | macro | 0.702 | 0.359 | 0.343 | yes |

Per-capability gain >0.02 nats: E-only/math, T+E/math, T+E/code, T+E/qa, F2/math, F2/code, F2/qa.
No gain >0.02 nats: E-only/code, E-only/qa.

JSON contains every candidate fit, fold, prediction, parameter count, input point, dense-loss source, config hash and SHA256 of all analysis inputs. Results are deterministic for a fixed set of input files.
