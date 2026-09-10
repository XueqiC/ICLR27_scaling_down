# Restructure summary — "developing predictive laws" (2026-09-10)

Start: paper main 79cf7cb. Data cut-off: results present on 2026-09-10 12:40 EDT (C35, C36, C37 included; the frozen
P2-v2 multi-student test matrix and the 4B dev-pool runs were in progress and are written as "frozen, pending").

## The new story
Specialized workloads need particular capabilities; compression changes capabilities unevenly across capabilities and
model states; so the loss of each capability after compression must be predicted from what is observable beforehand.
We build method-specific empirical relations from the initial model state (N0, D0, dense capability losses) and the
method's configuration on a controlled Pythia panel, compare low-dimensional forms on the same inputs, freeze them, and
test them on unseen configurations and unseen model states, delivering each relation with coefficients and a range.

## What each arm delivers
- Pruning: a five-parameter power form in (1-d) with a source-conditioned amplitude (Eq. 3), tied on development with a
  per-density regression; predicts math and code at unseen densities inside the fitted range (three new checkpoints:
  0.24 nats vs 0.45 strength-only), fails beyond d=0.55 and for QA; coefficients in App. B.
- Quantization: a per-bit source regression at fixed bit-width (no continuous bit law); with group size as a second
  axis, a 20-parameter two-dimensional form (Eq. 4) that predicts unseen b=4 and unseen g=128 on the development states
  and is mixed on one new state; the separable amplitude-times-shape candidate fails; coefficients in App. B.
- Distillation: delta relative to the initial student; a linear source form that carries math (not QA); reuse-count
  forms on the pool axis that carry QA on an unseen pool with a budget-dependent bias; descriptor forms F1/F2 shown as
  development-only candidates (retrospective on tests); the multi-student frozen test pending.

## Update 2026-09-10 14:00 EDT: three-layer main line
Section 4.3 now carries the shared-structure / parameter-range / one-point-calibration results (v59); a new Section 5
'From Laws to Capability-Specific Selection' (v60: leave-one-state-out selection maps with regret vs oracle, method
agreement, fixed-method and cheapest-feasible baselines, no-clear-winner cells) precedes Related Work (now 6) and
Discussion (7); contribution (3), the abstract, and the discussion were adjusted. Main text is ~13 pages with four figures.

## Main text vs appendix
Main (about 11.5 pages including Figs 1-3 and Tables 1-3; references begin on page 12; 36 pages total): problem and inputs (Eq. 2), loss definition with three boundaries, panel roles, fitting/evaluation
rules, the three arms' forms and development comparisons (Sec. 3), generalization by unseen settings / unseen states /
capabilities (Sec. 4) with Table 1 (panel roles), Table 2 (models and domains), Table 3 (held-out performance, one row per test) and Figs 1 (responses vs frozen predictions), 2 (generalization matrix incl. the confirmation panel and grouped tests), 3 (inputs, forms, capability-specific shape),
related work (3 paragraphs), discussion and conclusion.
Appendix A-I: experimental details and cohort list; fitted coefficients; alternative forms and model selection (v53/v55
LOSO, v56 forms and capability conditioning); complete prediction tables (three-line provenance tables, per-stage
pairs, confirmation panel, grouped tests) and the paired-improvement figure; measurement checks; heterogeneous-model
tables and phenomena; mechanistic analyses; recovery, resource-ratio and selection; registration, information budgets,
reproduction.

## Strong statements corrected in this pass
- "Beats source-free curves for in-range sizes" -> matches them (pooled stages); best at the late stage only.
- D0 increment separated from "beats the strongest baseline"; adding the dense anchor is the interval-supported input.
- No universal law; the pruning exponent is a fitted constant "not established as necessary".
- Quantization: per-bit regression is "not a law over b"; the fixed 4^-b misfit is not "no law"; grouped results are
  configuration transfer on seen states, new-state test mixed.
- Distillation: F1/F2 labelled development-only and retrospective; no joint student-pool-budget law claimed; QA
  constant-better on the source axis and reuse-better on the pool axis stated side by side.
- Sengupta et al. described with their v2 hold-out and per-task fits; P2 law credited for initial-state inputs.

## Unfinished results deliberately not written as results
P2-v2 multi-student test (nine U375 runs, 4B dev-pool runs), P3 measurement check, any GPTQ comparison.

## Verification
tectonic compile without errors, no undefined references or duplicate labels; every number in the text traced to a
generated table (v51, v52, v57, v58) or a ledger entry with file hashes (docs/CLAIM_EVIDENCE_MATRIX.md); all pages
rendered and inspected (main text and appendix); overfull boxes limited to <7pt inside Table 2 cells; figure order and
numbering follow the directive (1 responses, 2 generalization, 3 inputs/forms).
