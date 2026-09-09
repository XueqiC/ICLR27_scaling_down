# CLOSEOUT_PLAN — frozen scope, information budgets, comparison rules, stop conditions

Closeout round (2026-09-09), zero new GPU / training / API. Goal: an evidence-consistent, reviewable full
manuscript from existing data. This file is the authority on what is frozen and how comparisons are judged.

## Repo
Submission artifact = `paper/.git` → github.com/XueqiC/ICLR27_scaling_down (tracks paper/*.tex, docs/,
code/ mirror, data_mirror/). Project-root git is local-only. All closeout commits go to the paper repo.

## Frozen positioning (unchanged)
A systematic investigation toward predictive, capability-conditioned scaling-down laws for LLM compression.
Three parallel arms (pruning / quantization / distillation), same evidence standard; not pruning-only.
Whether one functional form unifies stays OPEN — existing results only negate the resource-ratio models
actually tested, not all possible unifications. Parallel arms need not all yield an equally successful law.

## Unified interface (not a unified response form)
L̂_{m,c} = L_{c,ref(m)} + F_{m,c}(x0, u_m; ρ_m).
- pruning/quantization reference = the source's own uncompressed model.
- distillation reference = initial student S0: δ_c = L_c(S_KD) − L_c(S0); L̂_c(S_KD) = L_c(S0) + δ̂_c.
  A gap to a large model B is reported split as [L_c(S0)−L_c(B)] + δ_c, never as a distillation size law.

## Information budgets (reported separately; never mixed)
- K0: pre-compression info only — N0, D0 where available, dense L_c0, family/recipe, intervention params.
  Cost of measuring dense L0 is disclosed; it is NOT a target-compression calibration.
- K1: K0 + one pre-specified target-compression measurement point.
- Oracle: post-hoc rescale using target outcomes; diagnostic only.
Per-model int3 calibration is K1-class, never packaged as K0. Models without D0 are not given a fabricated
value; predictors at different budgets are not compared as if same-input.

## Loss endpoint & units
Conditional CE under fixed benchmark/prompt/reference/mask/length rules; signed response kept (no clipping
of negative δ). Same-tokenizer token-loss units reused from frozen results; cross-tokenizer only via existing
per-byte results or reported separately. If units change, fit+prediction+baseline+error all change together.
QA loss decreases are NOT called accuracy/agent-performance gains.

## Comparison rules (this round)
- Primary metric = frozen MAE with an explicit aggregation unit; per-source/config/capability kept alongside.
- "improvement vs strongest simple baseline" and "gain from adding D0" are reported SEPARATELY, never merged.
- Near-zero true responses: no unstable percentage error, no inflated sign-accuracy, no ad-hoc error bands.
- Intervals are labeled by uncertainty type (probe / training-seed / pool-sampling / source-generalization).
- A CI covering zero is NOT an equivalence proof; when indistinguishable, pick the simpler model and write
  "evidence does not distinguish", never "proven equivalent".
- New source-state ≠ independent new family; same-model multi-stage and same-trajectory checkpoints stay
  correlated. With only 3 size/step/pool clusters, prefer per-fold/per-pool display over a population ratio.
- No post-hoc smooth-region filtering to improve headline error; any response filter is a conditional
  diagnostic, not a prediction-time domain.

## Original frozen predictions are IMMUTABLE
v38/v40/v41 registers unchanged. New baselines/analyses this round are RETROSPECTIVE DIAGNOSTICS in new
versioned dirs (v42/v43/v44) and labeled as such. Where a diagnostic overturns a headline, the claim is
disclosed and downgraded, never overwritten while still called a "frozen prospective success".

## Task status
- [x] Package A — pruning same-input baselines (v42 / PRUNE_SAMEINPUT.md).
- [x] Package B — distillation residuals + pool/protocol audit (v43 / DISTILL_RESIDUALS.md).
- [x] Package C — quantization partition table (v44 / QUANT_PARTITION.md).
- [ ] Manuscript sections updated to corrected conclusions + main table/figures.
- [ ] FINAL_EVIDENCE_SUMMARY consistency with main table.
- [ ] PDF compile (requires Overleaf; no LaTeX toolchain on rai — flagged as external step).

## Stop conditions
Each package stops when its specified outputs exist; no new function classes, densities, bits, pools,
training, or GPU. Success = a checkable honest judgment, not three SUPPORTED cells. Remaining external
dependencies (PDF compile, Drive backup upload) are limitations, not blockers to the analysis closeout.
