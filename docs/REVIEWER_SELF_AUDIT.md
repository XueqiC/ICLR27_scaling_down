# Reviewer-perspective self-audit (2026-09-11 21:30 EDT; paper main at the commit that adds this file)

Rule of the audit: every claimed contribution must map to (i) an actual formula or procedure in the paper, (ii) the strongest
same-information baseline it was compared with, (iii) an independent test with predictions frozen before measurement, and
(iv) an explicit scope. Anything that fails a column is listed under gaps with the fix applied or the honest wording kept.

## Contribution 1: the prediction problem and the development-then-validation design

| Column | Evidence |
|---|---|
| Formula/procedure | Eq. taskloss (L_{c,j}), Eq. interface (L_hat = L_ref + Delta_hat(x, theta)); reference model M0; §2.3 fitting/evaluation; four-part origin code |
| Strongest baseline | zero change, per-capability constant, strength-only curve, median development curve; same-input alternatives; no-D0 ablation (§2.3, App. I) |
| Independent test | every registered test in Table 1 / App. D lists freeze commit and origin code; C33-C54 in the ledger |
| Scope | dense architectures only (App. A); loss endpoint, not task success (§2.1); native-token units within model |

Reviewer risks: (a) "law" wording: the paper delivers empirical relations with measured ranges, and says so (§1, §7); the title keeps
"laws" as the object of study, not as a proven result. (b) Loss vs behaviour: App. E loss-to-accuracy audit covers pruning only.

## Contribution 2: per-method relations and their tests

### Pruning
| Column | Evidence |
|---|---|
| Formula | Eq. power (5 params/cap); A2 per-density regression; median curve L0 + f~_c(d) |
| Strongest baseline | A2 (same inputs), strength-only, median curve, zero (Table 1 rows; App. D) |
| Independent tests | 1B@32k/112k, 6.9B@32k/112k (C33/C34); confirmation panel 410M@48k, 1.4B@112k, 6.9B@80k (C35); 2.8B (C50/C52) — all frozen |
| Scope | seen sizes, unseen d in [0.6,0.9], math/code; QA and new sizes -> median curve (Table final) |

Gap kept honest: the power form is not repeatable on 2.8B (one state, measured twice) and loses on QA everywhere; exponent
values are fitted constants, not established as necessary. Fixed today: 2.8B reported as one state.

### Quantization
| Column | Evidence |
|---|---|
| Formula | per-bit source regression (seen bits); Eq. quant2d surface; same-input piecewise interpolation with boundary rule; per-config median |
| Strongest baseline | per-bit/per-config median, zero, bilinear, bit-only (App. C, App. D three-way table) |
| Independent tests | b=4 and g=128 on dev states (C36, frozen); confirmation at g in {32,512} and new state 1.4B@112k (C44/C46, frozen) |
| Scope | interpolation for math/code on seen states; median for QA and new states; surface not delivered |

Gap kept honest: the surface's early success rests on a rank-deficient parametrization (C41); the delivered rule (interpolation +
median) was chosen after the confirmation (R) and is stated as such in the three-way table.

### Distillation
| Column | Evidence |
|---|---|
| Formula | delta_c = a_c log(1+E) (math, code); joint u(a+bu+qv) (QA); reference = initial student; Eq. kdforms for F1/F2 (retrospective) |
| Strongest baseline | frozen per-capability constant; intercept forms T-only/E-only; same-input surfaces (App. C audit table) |
| Independent tests | nine unseen-pool trajectories (C38); confirmation at U=200 x three budgets (C47/C48); paired CIs (C43); drift sensitivity (C52) |
| Scope | development students; pool direction (code confirmed; QA on 270M); math selection worse than intercept baselines; student size not covered; QA on the 2Wiki distribution only (C45, C49) |

Gap kept honest: the headline rule of the first test was stated after the test (R); the confirmation's math selection lost to
simpler frozen forms; no item-level intervals (per-example losses not stored).

## Contribution 3: from relations to selection

| Column | Evidence |
|---|---|
| Procedure | Eq. selection; locked rule (final_rule.py; App. H text); feasibility without fallback (C42) |
| Strongest baseline | quantization-only under the same candidates/predictor/feasibility; earlier law maps; cheapest; fixed policies (C42, C54) |
| Independent test | four fresh states, predictions and maps frozen before measurement (C53/C54): math/code/QA criterion met; largest-loss-increase objective prospective, criterion not met |
| Scope | nominal storage only; 17 budgets x 4 states (68 cells from four models); distillation candidates reused from v39 (their outcomes purged from the fits that predict them); QA restricted to 2Wiki; QA confirmation is an aggregate over heterogeneous states (160M@32k trails quant-only) |

Reviewer risks: quant-only is close on math/code (the storage axis favours quantization); the value of cross-method selection is
capability-specific (QA) and is stated so.

## Cross-cutting gaps and their status

1. Capability conditioning value (V76): the shared-curve baseline used signed scales and none came out negative (V79); the
   QA gain is attributed to a sign-reversing, non-proportional aggregate curve (R^2 0.49) and §4.3 now says "shape", not "sign";
   the 2.8B duplicate merged (four clusters) leaves the reading unchanged (macro 0.059 [0.021,0.103]).
2. Small correlated panels: three sizes x three stages; intervals are descriptive where clusters < 6 (stated in App. C/D).
3. One quantizer family, one teacher, one probe set per capability (§7); MoE not covered (App. A).
4. Multi-capability objective naming unified to max_c[L_c(M) - L_c(M0)] (fixed today).
5. 2.8B duplicate: affects V72's state count and V76's pruning cluster count (recomputed in V79); V78 unaffected (development data only).
6. Figures: main-text map simplified to locked rows with disagreement marks and a separate legend (V80, fig:rule_main); full maps, regret chart, per-state and set-size tables in App. H.
