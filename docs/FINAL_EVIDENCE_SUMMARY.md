# FINAL_EVIDENCE_SUMMARY — per-claim evidence, scope, counterexamples, revised wording

Every main claim with its evidence path, validity range, counterexample, and the old→new wording after the
closeout Package A/B/C diagnostics. Numbers trace to results/v*/summary.json (see paths). Frozen prospective
predictions (v38/v40/v41) are unchanged; A/B/C are retrospective diagnostics.

## Central thesis (retained, unchanged)
Capability loss is a common OUTCOME coordinate (its map to behavioral accuracy shares a form across tested
families), whereas compression ratio is NOT a common INTERVENTION coordinate (method-agnostic resource-ratio
laws fail to extrapolate across methods). Evidence: v6/v10/v12 heterogeneous panel; loss-to-accuracy link
(v15/v23); matched-storage non-equivalence. Scope: 12 checkpoints, 4 families, 270M–32B. Unification of a
single functional form remains OPEN (only tested resource-ratio models are negated).

## Pruning
- CLAIM: source-conditioned inputs {N0, L0, D0} predict per-capability pruning damage at UNSEEN densities,
  far better than any compression-strength-only baseline (math/code decisively; QA weaker).
  Evidence: v40 (frozen prospective, step96000, d=0.65 interp / 0.55 extrap) + v42 (same-input baselines).
  MAE: candidate math 0.371 / code 0.524 / qa 0.759 vs strength-only 1.37/1.64/1.61, median 1.48/1.87/1.92.
- REVISED (Package A): the win is a SOURCE-INFORMATION win, NOT proof of the shared power form. A simpler
  per-density source regression + linear interp/extrap (A2, no shared exponent) TIES/BEATS the candidate for
  math (0.283) and code (0.279); the candidate only beats the γ=1 variant (A1). → DROP "the exponent γ_c is
  independently confirmed"; KEEP "source-conditioned inputs predict unseen density".
- COUNTEREXAMPLE / boundary: at d=0.55 signed errors are MIXED (6 under, 3 over); the most-fragile source
  160m@143k is OVER-predicted (math +1.25, code +1.57). QA amplitude does not follow the math/code law.
- SCOPE: Pythia controlled panel; d∈[0.55,0.9]; source-transfer at a training-STAGE interpolation (step96000
  between 64k/143k), not a longer-training extrapolation. Small panel, point estimates.

- **P1 new-source frozen prospective (C32, P-new):** on pythia-1b@96k the source-conditioned family SUCCEEDS at
  the interpolated density (A1/A2/power 0.15–0.19 vs strength-only 0.70) and FAILS at the extrapolated density
  (all 1.14–1.69, strength-only 0.27; QA sign missed) → validity range = density interpolation. One source.

## Quantization
- CLAIM: on the discrete integer bit-widths {8,6,4,3}, the frozen {N0,L0,D0} predictor is compared to simple
  per-bit baselines per source. Evidence: v38 (160M/1.4B@96k), v40 (410M@96k), v44 (partition table).
- FINDINGS (Package C wording): ≥4-bit responses are near-zero; candidate and per-bit-median errors are both
  tiny, so NO incremental predictive value over a simple baseline is demonstrated there (not a proof of none).
  int3 is REAL measured collapse (not an artifact); the candidate's int3 advantage is unstable across sources
  (math +0.45, code −1.12, qa −0.32). The fixed 4^{-b} shape misfits under this RTN definition (learned
  exponential better, C8), within the tested range only.
- NOT CLAIMED: "no unseen-bit axis / quantization has no law / int3 artifact / a validated collapse-RISK law".
  v38 and v40 are separate source measurements, not one merged prospective.
- SCOPE: weights-only per-output-channel RTN; discrete bits; Pythia @96k sources.

- **P1 (C32):** on the new source the frozen full {N0,L0,D0} predictor beats no-D0/per-bit-median/zero at both
  int4 (0.074) and int3 (0.397); D0 carries information on this source; still per-bit, one source.

## Distillation
- CLAIM: dense-student baseline + signed transfer response δ_c = L_c(S_KD) − L_c(S0); reuse-count E = T/D_U
  is the dominant coordinate on the endpoint pools; predicting an unseen middle pool U=225 from endpoint pools.
  Evidence: v31/v37 (E dominance), v41 (frozen new-pool prediction), v43 (residual analysis).
- FINDINGS: on U=225, E-only beats a constant only for QA (MAE 0.706 vs 1.449); math/code are best fit by a
  per-capability constant. REVISED (Package B): the U225 error is dominated by SYSTEMATIC BIAS of the frozen
  predictors, not pool variation — math/code (constant) show small over-prediction bias with tiny variation;
  QA (E-only) shows a LARGE budget-dependent bias that SIGN-FLIPS (−0.87 → +0.48) = a form misfit, so E-only
  merely beats a constant rather than transferring cleanly.
- COUNTEREXAMPLE / boundary: endpoints are first-U prefixes, U225 is random-sampled → "unseen pool size AND
  changed sampling protocol", not isolated pool-size; U225 E (2.34–4.70) interpolates inside endpoints
  (0.94–14.94). 3 pools × 2 seeds; 2 checkpoints share a trajectory (thin variance estimate).
- NOT CLAIMED: "pool-std/train-std ~10× explains the residual", "residual is a causal data-selection effect",
  "E is universally sufficient". matched-T controls optimizer steps + supervised tokens (C21/DISTILL_STEP_CONTROL),
  so the pool-size gap is not "just more tokens/steps", but E-sufficiency still needs new-config validation.

## Cross-arm conclusion (retained, corrected emphasis)
Output space, evaluation, and information budgets are shared; whether the response FORM unifies is decided by
evidence and currently does not, so we unify at the decision layer (method-selection map). This is NOT a
"unified three-arm negative", and it is NOT a claim that all three arms established a universal law.

## Supporting results (appendix-level, not core predictors)
Capability geometry/ablation (v9/regions) — motivation/mechanism, appendix. capability-specificity math/QA vs
code limits retained. accuracy link (v15) — metric-boundary appendix. recovery (v13) — fixed-protocol
phenomenon, appendix; 16M tokens NOT called an asymptotic floor. Resource-ratio non-equivalence, Pareto,
mis-ranking — implications (nominal storage, no real latency; no held-out selection ⇒ no reliable-selector claim).
Dense scaling/reference and prior compression/distillation laws — attributed to prior work, not our contribution.

## Open limitations affecting submission (≤3 substantive)
1. Small controlled panel (3 sizes × 3 steps; 3 pools) — intervals are panel-conditional, not population.
2. The shared pruning power form is not independently necessary (A2 ties it); the paper reports the
   source-conditioned family with the power shape as a non-necessary variant.
3. No held-out agent/method-selection validation; the selection map is a framework deliverable, not a
   validated selector. (External: PDF compile on Overleaf; 7.1G model backup awaits a Drive upload path.)
