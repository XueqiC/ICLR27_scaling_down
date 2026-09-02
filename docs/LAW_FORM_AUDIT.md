# Law-Form Audit

A running check on **whether each proposed law form is actually the right thing to fit** —
not whether we *can* fit it. Ubiquitous good in-sample fit is a *warning sign*, not a success.

**Workflow (user, 2026-09-02).** The goal is: **(1) establish one rigorous, correct law form
for EACH compression method separately, then (2) test whether they unify (M1/M2/M3) or feed a
method-matching/selection map.** Step 1 must be rigorous.

**The audit's job is to FIX forms, not just grade them.** When the evidence says the current
form is wrong, the required output is a **corrected form derived from the experimental results** —
a concrete revision, stated below as each arm's *Revision path*. Retreating to a weaker
non-parametric "honest report" is the **floor / last resort**, never the target. "Our form did
not fit" is a non-result; the value is iterating to the right form.

## Probation rule (applies to every form below)
A functional form earns the right to be fitted and reported as a *law* only if:
1. **Held-out extrapolation beats the simplest baseline** — on at least one of
   {leave-one-family-out, leave-largest-model-out, fit-shallow-predict-deep,
   leave-one-bit/one-budget-out}, and by a margin above the noise floor.
2. **No sign errors** on held-out cells (predicting improvement where damage occurs, or vice versa,
   is disqualifying regardless of magnitude MAE).
3. **A reduced form in observable-only inputs** does not lose materially to the mechanism form —
   otherwise the "law" is not a usable prediction interface.

If a form fails, we do **not** force-fit it, and we do **not** stop at a fallback: we apply the
arm's **Revision path** to derive the next candidate form, re-run the gate, and iterate. The
non-parametric fallback is adopted only after the revision path is exhausted.

Gate discipline: **no form is scaled to large models before it passes the gate on mid-size models.**

Status vocabulary: ON-PROBATION / REVISING / PROVISIONALLY-KEPT / REJECTED / FALLBACK-ADOPTED.

---

## 1. Pruning: `ΔL_c(d) = ḡ_c·δw + B·m_c(d)^γ`  — ON-PROBATION
- **What it is.** Mechanism form: measured signed first-order alignment `ḡ_c·δw` plus a
  deleted-capability-Fisher-mass term `m_c^γ`, γ shared across capabilities within a model.
- **Derivation status.** First-order term is derived (Taylor); the `m_c^γ` term is a reduced-form
  ansatz motivated by the concentration↔resilience observation, not derived from first principles.
- **Strongest evidence FOR.** After removing the first-order term, capabilities collapse onto one
  curve (R² 0.82–0.96 across 24 fits); γ clusters 1.2–2.2 and is ≈constant across capabilities
  within a model → capability differences are carried by the *measured* quantities, not free params.
  Leave-family-out MAE 2.17 vs 8.2/10.3 for baselines.
- **Why the form may be WRONG.** It silently glues two regimes together. Pre-cliff Taylor works;
  post-cliff it fails catastrophically (0.6B: math d=0.5 predicted −0.19 vs measured **+4.68**).
  Fitting one continuous curve across the cliff is not honest.
- **Revision path (corrected form).** Do not fit one curve across the cliff. Split into two
  explicit objects and, critically, **change the cliff's coordinate**:
  (a) perturbative branch `ΔL_c = ḡ_c·δw + B·m_c^γ`, valid pre-cliff;
  (b) cliff as a phase transition **re-parameterized in capability-mass survival fraction
  `S_c(d)` rather than density `d`** — v11 showed `S_c` is the leading indicator, so the corrected
  collapse variable is `S_c/S*_f` with a family-specific critical `S*_f` (§2). Concretely the next
  candidate is `ΔL_c(d) = [perturbative] + A_c·Φ((S*_f − S_c(d))/w_f)` with Φ a sharp sigmoid,
  width `w_f` per family. This is the evidence-driven modification, not a retreat.
- **Gate.** Fit the pre-cliff branch and test fit-shallow→predict-deep up to the cliff; separately
  test whether the `S_c/S*_f` collapse predicts held-out cliff onset (§2).
- **Fallback (last resort only, if revision also fails gate).** Per-family damage curve family +
  measured cliff location, no closed form.

## 2. Cliff / critical survival fraction  — STRONG-VERSION REJECTED, family-conditioned kept
- **What it is.** Behavioral cliff (ΔL≈1) corresponds to capability-mass retention crossing a
  critical value.
- **Evidence.** v11: retention decays smoothly and *leads* the behavioral cliff (gemma3-1b
  retention at cliff ≈0.10). But across families the critical value is **NOT universal**
  (Qwen 0.13–0.23 vs Gemma 0.035–0.10); it *is* consistent within a family across sizes
  (two Gemma sizes coincide).
- **Revision path (corrected form).** Universal-threshold is falsified → the corrected form is a
  **family-rescaled collapse**: `ΔL_c = Ψ(S_c(d)/S*_f)` with one shared shape `Ψ` and a
  per-family critical `S*_f` fitted as a constant. Test next whether `S*_f` itself is predictable
  from cheap family descriptors (params, pretraining tokens, depth); with only ~4 families we can
  only *screen*, not fit, so report `S*_f` per family plus the descriptor screen. "Form travels,
  coefficients split by family" is the *headline result*, not a nuisance.
- **Gate.** Predict cliff density on held-out sizes *within* a family from that family's `S*_f`;
  report cross-family `S*_f` spread as a finding, do not average it away.

## 3. Recovery: saturating `ΔL_c(D_R)=ΔL_c(0)[r+(1−r)(1+D_R/D0)^−β]`  — ON-PROBATION (under-identified)
- **What it is.** Anchored saturating recovery in adaptation tokens; `r` = asymptotic residual
  fraction under a fixed recipe (explicitly *not* claimed fundamentally unrecoverable).
- **Status.** v14 flagged it **under-identified** from single-point data — this is exactly why the
  B7 multi-budget grid (1M/4M/16M/64M × {c4, traces} × {prune, quant}) is running now.
- **Why the form may be WRONG.** Two independent risks: (a) if the multi-point fit shows `r`
  unstable across configs/seeds, the 3-param saturating form is over-specified; (b) we already
  **observed re-damage under over-training** (aligned recovery data heals fast then hurts) — a
  *monotone* saturating form structurally cannot express that.
- **Revision path (corrected form), decided by what the B7 grid shows:**
  - if recovery keeps declining with no asymptote → drop `r`: `ΔL_c(D_R)=ΔL_c(0)(1+D_R/D0)^−β`;
  - if fast-then-plateau → keep saturating but fit `r,β` from the multi-budget grid;
  - if fast-then-**re-damage** (as observed) → the corrected form is **non-monotone**:
    `ΔL_c(D_R)=ΔL_c(0)(1+D_R/D0)^−β + κ·(D_R/D_over)^p`, i.e. an over-training penalty term with
    an interior optimum `D_R*` — and `D_R*` (the optimal recovery budget) becomes the quantity the
    method-selection map actually consumes.
- **Gate.** The chosen form must predict a held-out budget and, if non-monotone, recover `D_R*`
  within CI. Else adopt the fallback.
- **Fallback (last resort only).** Non-parametric recovery curve + reported optimal budget `D_R*`.

## 4. Distillation loss-space damage law  — ON-PROBATION (measurement-validity risk)
- **What it is.** Per-capability template with capacity floor, teacher capability vector, data
  power law in trace tokens, coverage factor χ_c.
- **Evidence.** V12 pilot: math/code teacher-forced loss rises in every cell and grows with trace
  count (1b: math ∝ n^0.84, code ∝ n^1.2); QA loss falls, non-monotone in n.
- **CRITICAL RISK (new, 2026-09-02).** These are teacher-forced losses vs reference solutions; a
  rise may be **style drift toward the teacher's format, not capability loss.** B3 accuracy on the
  same probes is at the floor for base students (gemma3-1b dense: math 0/64, code 1/64, qa EM 0)
  → we currently **cannot** confirm the loss changes track real capability.
- **Revision path (corrected form).** Do not fit the distillation law on raw teacher-forced loss.
  Two evidence-driven corrections, in order:
  1. **Move the target to a capability-valid signal**: fit the law in accuracy space
     `A_c = g_c(...)` (B3 few-shot), or on a *style-residualized* loss
     `ΔL_c^cap = ΔL_c − ΔL_c^style`, where `ΔL_c^style` is the loss change on format/boilerplate
     tokens (measurable by masking content tokens). If the rise is style, `ΔL_c^cap ≈ 0` and the
     "damage" disappears — that itself is the finding.
  2. Only on the capability signal, fit the template (capacity floor + trace-token power law +
     coverage χ_c). The V12 `math ∝ n^0.84 / code ∝ n^1.2` exponents must be **re-measured on the
     corrected target** before they mean anything.
- **Gate.** B3 few-shot accuracy must (a) clear the floor on dense at the tested scale and (b) move
  monotonically with the corrected capability signal across configs, before any distillation law is
  fitted. If accuracy stays at floor even few-shot at 1B, escalate scale/benchmark (GSM8K) rather
  than declaring a loss-space law.
- **Fallback (last resort only).** Distillation ΔL_cap + accuracy as measurements + recipe/teacher
  contrasts, no predictive law.

## 5. Quantization: degradation vs bit-width `b`  — ON-PROBATION
- **What it is.** Candidate forms at bit-width `b`: `q_c·4^−b`, learned exponential `q_c·e^−kb`,
  polynomial, or an effective-parameter multiplier; validated leave-one-bit-out + held-out
  model/family, with a matched-storage comparison against pruning.
- **Evidence.** v10 (7 models, bits 8/6/4/3): int8/int6 near-lossless, int4 hurts small models,
  **b=3 collapses for all models (a quantization cliff)**; Qwen QA int4 improves (−0.21). So the
  degradation is *not* smooth — there is a cliff, mirroring pruning.
- **Why the form may be WRONG.** A single smooth `q_c·4^−b` cannot express the b=3 collapse; and the
  capability-selective sign (QA improves) means a scalar per-capability prefactor is insufficient.
- **Revision path (corrected form).** Same two-regime treatment as pruning, in the quantization
  basis: a smooth pre-cliff term (`q_c·4^−b` or exponential) **plus** a bit-cliff modeled with the
  *same* order-parameter machinery — capability-mass surviving above the quantization noise floor,
  crossing a family-specific critical value between b=4 and b=3. This is the direct test of whether
  pruning and quantization share the survival-fraction collapse (the unification hypothesis M1/M2).
- **Gate.** Leave-one-bit-out (fit 8/6/4, predict 3) must catch the collapse direction; matched-
  storage cross-check vs pruning at equal `r_storage`.
- **Fallback (last resort only).** Per-model bit-degradation table + measured quantization cliff.

---

## Cross-cutting lesson
The recurring finding across arms is **"the functional form travels across families/sizes but its
coefficients do not."** That is a stronger and more defensible thesis than any single universal law,
and it is exactly what the M1(fully-shared)/M2(hierarchical)/M3(method-specific) held-out comparison
is designed to adjudicate. When in doubt, demote a universal-law claim to a form+family-coefficient
claim rather than force the universal fit.
