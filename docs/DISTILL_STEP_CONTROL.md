# Distillation matched-T already controls optimizer steps (advisor P3 pre-check)

The advisor asked, before running any matched-optimizer-step design: check whether the existing
matched-T experiment already matched update steps. It does — to within ~1-3%. **No rerun needed.**

## The matched-T trajectory points (Qwen3-4B, gpt-5.6-luna teacher, uxseen)

At each matched processed-token count T, the small pool (U75, 65,476 pool tokens) and the large pool
(U600, 521,533 pool tokens) are compared. Their per-step batch is identical (16 sequences), so matched-T
forces matched steps:

| matched-T | U75 step | U600 step | U75 compl-tok | U600 compl-tok | U75 E | U600 E |
|-----------|----------|-----------|---------------|----------------|-------|--------|
| ~500k     | 107      | 108       | 138,998       | 143,729        | 7.65  | 0.97   |
| ~1000k    | 214      | 214       | 278,388       | 283,935        | 15.28 | 1.92   |

- **Optimizer steps matched**: 107≈108, 214=214.
- **Supervised (completion) tokens matched**: within ~3%.
- **Processed tokens matched**: within ~1% (by construction).
- **The ONLY variable that differs is reuse count E** (~8× at both points).

## Consequence for the distillation story

- The large matched-T pool-size gap (Qwen3-4B: math +0.16/+0.48, qa +2.4/+2.8) is observed at matched
  steps, matched processed tokens, and matched supervised tokens. It is therefore attributable to
  **reuse count E = T/D_U**, NOT to optimizer-step count or training exposure. This is exactly the
  matched-optimizer-step evidence the advisor wanted — it already exists in the data.
- Symmetrically, matched-E holds E fixed while steps/T vary ~8× (28 vs 224), and the gap collapses to a
  small residual (math −0.01/−0.04). So an 8× exposure/step swing at fixed E barely moves the response:
  **training exposure is a WEAK driver**; it does not "explain" the residual.

## Corrected E-residual statement (replaces "confounded with exposure, unresolved")

> On matched-T (steps, processed tokens and supervised tokens all matched to ~1-3%), the pool-size
> response gap is driven by reuse count E, not by optimizer steps or exposure. On matched-E, an ~8×
> step/exposure swing produces only a small residual. Together: E is the dominant distillation
> coordinate and step/exposure is controlled and weak; a small matched-E residual remains whose
> attribution (unique-data volume vs sampling) is a minor open item, NOT an exposure confound.

## What P3 should actually do (rerun no longer needed for the step question)
The step/exposure question is answered. The remaining open item is whether the small matched-E residual
is unique-data-volume driven — addressable by the low-DOF new-pool-size prediction (P3 in the plan),
distinguishing data-pool sampling seed from training seed, NOT by a matched-optimizer-step rerun.
