# Law-Form Audit

A running check on **whether each proposed law form is actually the right thing to fit** —
not whether we *can* fit it. Standing directive (user, 2026-09-02): "we must continually
ask whether the law form is really correct and worth fitting, or whether there is a better
representation." Ubiquitous good in-sample fit is a *warning sign*, not a success.

## Probation rule (applies to every form below)
A functional form earns the right to be fitted and reported as a *law* only if:
1. **Held-out extrapolation beats the simplest baseline** — on at least one of
   {leave-one-family-out, leave-largest-model-out, fit-shallow-predict-deep,
   leave-one-bit/one-budget-out}, and by a margin above the noise floor.
2. **No sign errors** on held-out cells (predicting improvement where damage occurs, or vice versa,
   is disqualifying regardless of magnitude MAE).
3. **A reduced form in observable-only inputs** does not lose materially to the mechanism form —
   otherwise the "law" is not a usable prediction interface.

If a form fails, we do **not** force-fit it. We fall back to the weaker-but-honest statement named
in each section (curve family + working-region map). For ICLR that is a legitimate, useful result.

Gate discipline: **no form is scaled to large models before it passes the gate on mid-size models.**

Status vocabulary: ON-PROBATION / PROVISIONALLY-KEPT / REJECTED / FALLBACK-ADOPTED.

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
- **Better representation to consider.** Two explicitly separate objects:
  (a) a perturbative branch (mechanism form) valid pre-cliff, and
  (b) the cliff as a **phase transition** parameterized by an order parameter (capability-mass
  survival fraction, §2), reporting a critical value rather than a smooth extrapolation.
- **Gate.** Fit only the positive-damage pre-cliff branch; test fit-shallow→predict-deep *up to
  but not across* the cliff. Cliff handled by §2.
- **Fallback if rejected.** Per-family damage curve family + measured cliff location; no closed form.

## 2. Cliff / critical survival fraction  — STRONG-VERSION REJECTED, family-conditioned kept
- **What it is.** Behavioral cliff (ΔL≈1) corresponds to capability-mass retention crossing a
  critical value.
- **Evidence.** v11: retention decays smoothly and *leads* the behavioral cliff (gemma3-1b
  retention at cliff ≈0.10). But across families the critical value is **NOT universal**
  (Qwen 0.13–0.23 vs Gemma 0.035–0.10); it *is* consistent within a family across sizes
  (two Gemma sizes coincide).
- **Verdict.** The universal-threshold form is **falsified**. Correct form: **shape is
  cross-family, coefficient (critical retention) is per-family.** Treat this recurring
  "form travels, coefficients split by family" pattern as a *headline result*, not a nuisance.
- **Gate.** Predict cliff density on held-out sizes *within* a family from that family's critical
  retention; report cross-family critical-value spread as a finding, do not average it away.

## 3. Recovery: saturating `ΔL_c(D_R)=ΔL_c(0)[r+(1−r)(1+D_R/D0)^−β]`  — ON-PROBATION (under-identified)
- **What it is.** Anchored saturating recovery in adaptation tokens; `r` = asymptotic residual
  fraction under a fixed recipe (explicitly *not* claimed fundamentally unrecoverable).
- **Status.** v14 flagged it **under-identified** from single-point data — this is exactly why the
  B7 multi-budget grid (1M/4M/16M/64M × {c4, traces} × {prune, quant}) is running now.
- **Why the form may be WRONG.** If the multi-point fit shows `r` is unstable across configs/seeds,
  the three-parameter saturating form is over-specified and should be dropped.
- **Better representation.** Non-parametric recovery curve family + a single reported quantity:
  "recovery budget to reach X% of dense capability," which is what the method-selection map needs
  anyway.
- **Gate.** Multi-budget fit must give a stable `r` (and `β`) with a bootstrap CI that excludes the
  degenerate cases, and must predict a held-out budget. Else adopt the fallback.

## 4. Distillation loss-space damage law  — ON-PROBATION (measurement-validity risk)
- **What it is.** Per-capability template with capacity floor, teacher capability vector, data
  power law in trace tokens, coverage factor χ_c.
- **Evidence.** V12 pilot: math/code teacher-forced loss rises in every cell and grows with trace
  count (1b: math ∝ n^0.84, code ∝ n^1.2); QA loss falls, non-monotone in n.
- **CRITICAL RISK (new, 2026-09-02).** These are teacher-forced losses vs reference solutions; a
  rise may be **style drift toward the teacher's format, not capability loss.** B3 accuracy on the
  same probes is at the floor for base students (gemma3-1b dense: math 0/64, code 1/64, qa EM 0)
  → we currently **cannot** confirm the loss changes track real capability.
- **Verdict.** **Do not fit a distillation law in loss space** until B3 (few-shot accuracy) shows a
  real capability signal with dynamic range. If accuracy stays at floor even few-shot at this scale,
  the distillation arm is **a measurement report, not a law.**
- **Gate.** B3 few-shot accuracy must (a) clear the floor on dense and (b) move monotonically with
  loss across configs, before any distillation law is fitted.
- **Fallback.** Report distillation ΔL and accuracy as measurements + the recipe/teacher contrasts,
  without a predictive law.

---

## Cross-cutting lesson
The recurring finding across arms is **"the functional form travels across families/sizes but its
coefficients do not."** That is a stronger and more defensible thesis than any single universal law,
and it is exactly what the M1(fully-shared)/M2(hierarchical)/M3(method-specific) held-out comparison
is designed to adjudicate. When in doubt, demote a universal-law claim to a form+family-coefficient
claim rather than force the universal fit.
