# FINAL_EVIDENCE_SUMMARY — per-claim evidence, scope, counterexamples, revised wording

Every main claim with its evidence path, validity range, counterexample, and the old→new wording after the
closeout Package A/B/C diagnostics. Numbers trace to results/v*/summary.json (see paths). Frozen prospective
predictions (v38/v40/v41) are unchanged; A/B/C are retrospective diagnostics.

## Central thesis (round-7 wording; accuracy link removed from the core)
The per-capability conditional loss vector is the common prediction endpoint for all three arms ("common" =
evaluation interface; not cross-tokenizer NLL comparability, not proven independence of the three losses);
the arms share one information budget (K0/K1/Oracle) and one validation standard; the response form and its
range of validity are determined separately per arm by evidence. Laws using only resource ratios fail to
extrapolate across the methods tested (matched-storage non-equivalence), which negates the tested
resource-ratio models but leaves open whether some other single form unifies the arms. Behavioral
(loss-to-accuracy) results are appendix-level support (v15/v23), not a central claim. Scope: 12 checkpoints,
4 families, 270M–32B; controlled Pythia panel + one new source (P1).

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

## Cross-arm conclusion (round-7 wording)
The three arms share the prediction endpoint, the information budget, and the validation standard; response
form and validity range are determined per arm by evidence: pruning = source-conditioned prediction valid in
the density-interpolation regime (independently confirmed on a new source; extrapolation fails), quantization
= per-configuration validation (near-zero region without demonstrated gain; int3 real, magnitude transfer
source-dependent), distillation = capability-specific transfer (reuse coordinate helps QA; constant suffices
for math/code; bias-dominated residual). Whether one response form unifies remains open. The method-selection
map is DISCUSSION (future use), not a delivered/validated selector; Pareto and mis-ranking analyses are
implications of prediction error. This is neither a "unified three-arm negative" nor a claim of universal laws.

## Supporting results (appendix-level, not core predictors)
Capability geometry/ablation (v9/regions) — motivation/mechanism, appendix. capability-specificity math/QA vs
code limits retained. accuracy link (v15) — metric-boundary appendix. recovery (v13) — fixed-protocol
phenomenon, appendix; 16M tokens NOT called an asymptotic floor. Resource-ratio non-equivalence, Pareto,
mis-ranking — implications (nominal storage, no real latency; no held-out selection ⇒ no reliable-selector claim).
Dense scaling/reference and prior compression/distillation laws — attributed to prior work, not our contribution.

## Open limitations (≤3 substantive)
1. Small controlled panel (3 sizes × 3 steps; 3 pools) — intervals are panel-conditional, not population.
2. The shared pruning power form is not independently necessary (A2 ties it); the paper reports the
   source-conditioned family with the power shape as a non-necessary variant.
3. No held-out agent/method-selection validation; the selection map is a framework deliverable, not a
   validated selector. (External: PDF compile on Overleaf; 7.1G model backup awaits a Drive upload path.)

## Round v2 update (2026-09-09 17:20 EDT): P1-v2 frozen prospectives on two new Pythia sizes (C33, C34)
- 1B @32k/@112k (in-range size): the source-conditioned per-density regression (A2) is the best pruning predictor in the
  interpolation regime at the late stage (protocol A: 0.112 vs median 0.173, strength-only 0.365), on par or slightly
  behind the source-free median curve at the early stage (0.261 vs 0.204 under A; 0.218 vs 0.224 under B). Density
  extrapolation to 0.55 fails under both protocols. Quantization class-indicator: helps only at 3-bit; >=4-bit the
  per-bit median is as good or better.
- 6.9B @32k/@112k (~5x size extrapolation): R in both arms and both protocols. The panel size trend for QA does not
  continue (predicted 1-3 nats improvement vs measured <0.65, reversing at d=0.55/112k); math/code over-predicted ~2x.
- Transfer range of the current source-conditioned predictors: roughly the development size range x the interpolation
  density range. Neither axis extrapolates. Tables: paper/tables/p1v2.tex (per stage, per protocol), tab:main (+8 rows).
