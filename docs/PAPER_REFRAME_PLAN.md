# Paper Reframe Plan (2026-09-01)

## Old framing
Mechanism-first: capability geometry as the central object; laws written in
mechanism coordinates (deleted Fisher mass, signed alignment); geometry
presented as prediction-adjacent; strong wording (causal carriers,
unrecoverable price, universal).

## New framing
Predictive scaling-down laws for capability-specialized agents:
(dense source state, common compression coordinates, method-specific
controls) -> post-compression capability vector -> workload performance ->
cost-aware method selection. Prediction-time inputs must be observable
BEFORE compression (anchored setting: L_obs + predicted delta). Geometry =
mechanism evidence and upper-bound comparison only. Do not presume
cross-method unification: compare M1 (shared) / M2 (hierarchical) /
M3 (method-specific) by held-out prediction error.

## Claims to remove or weaken
- "causal carriers" -> "causally implicated, partially separable regions"
  ONLY after multi-model + control ablations (until then: preliminary).
- "unrecoverable damage / true price" -> "asymptotic residual under a
  fixed recovery recipe".
- Dense per-capability law as a contribution -> capability-conditioned
  specialization/validation of established laws (placeholder until D1-D4
  comparison done).
- Universal 3-bit cliff -> observed across 9 models; functional form and
  held-out prediction pending.
- Law claims: in-sample fits are PRELIMINARY until the held-out battery
  (deepest-density, leave-largest-out, leave-family-out) passes.

## Verified contributions (current)
Capability ordering invariance (12 models); cliff band + non-perturbative
Taylor failure (descriptive); prior-grid forensic re-analysis; retired
concentration predictor (honest negative).

## Target contributions (need experiments in EXPERIMENT_PLAN.md)
Anchored predictive laws with held-out validation; sign prediction with
uncertainty; geometric order parameter multi-model; cross-method
M1/M2/M3 decision; agent workload validation + method-selection map.

## Section mapping (new structure)
1 Introduction (specialized agents, grid-search problem, predictive laws)
2 Related work (7 strands incl. compression method selection)
3 Problem formulation (capability vector, source state, coordinate taxonomy,
  workload requirement, cost-aware selection)
4 Predictive framework (dense refs D1-D4, anchored setting, shared+method-
  specific hierarchy, cliff models, recovery transform, uncertainty)
5 Capability measurement & mechanistic analysis (benchmarks, tokenizer-
  normalized loss, link functions, geometry as evidence, ablations+controls)
6 Experimental setup  7 Results (organized by research questions)
8 Limitations and scope

## Required figures/tables
Block-structure figure regenerated with all 7 models; ablation matrix with
controls; per-arm damage tables with bootstrap CIs; held-out error tables
(law vs baselines); recovery curves (saturating form); method-selection map.

## Three-parallel-arms structure (directive 2026-09-05, §11)

**Positioning (fixed):** a systematic investigation aimed at developing
capability-specific predictive scaling-down laws for LLM compression.
Pruning, distillation, quantization are THREE CO-EQUAL main arms, same
evidence standard, experiment count allocated by each arm's variables and gaps.
Dense reference = inherited/validated base. Recovery = cross-method training-budget
axis. Capability geometry/Fisher = explanatory/diagnostic. No presupposition that
the three share a form, a cliff, or a capability-improvement; openness kept.

### Section order
1. Motivation & research questions (workload capability needs are selective;
   average performance + single compression ratio insufficient).
2. Common formulation / measurement / validation (interface
   L̂^(m)=F_m(x_base,z_shared,z_m;η̂), Â_c=g_c(L̂_c); three information conditions:
   meta-only / +dense-cap / +k-model calibration).
3. **Pruning** (co-equal) ┐
4. **Distillation** (co-equal) ├ each: Inputs & candidate forms → Fit & simple
5. **Quantization** (co-equal) ┘ baseline → Independent prediction → Domain of validity.
6. Shared structure & transfer (M1/M2/M3; what travels, what is family-specific).
7. Training-budget analysis (recovery as common budget across methods).
8. Behavior & method selection (loss→accuracy links; preregistered held-out selection).
9. Mechanistic evidence & limitations (Fisher block structure, ablation, nulls).

### Per-arm section template (identical across the three)
- **Observable inputs & information budget**: base descriptors (real params,
  family, checkpoint/stage, disclosed pretraining tokens, cost-labeled dense
  capability), shared vars (target size, precision, adapt-data, budget, recipe),
  method vars (density/pattern/criterion | bit/group/quantizer | Ns/Ds/teacher).
  Separate prediction-time-known inputs from dev-estimated frozen coefficients.
- **Candidate forms & simple baselines**: list actual formulas+parameter counts
  (not code names like density_only); include the trivial baseline.
- **Independent prediction**: held-out tables (shallow→deep, leave-size-out,
  leave-family-out, leave-bit-out), MAE + behavioral error + PI coverage/width;
  label each split as interpolation / extrapolation / calibration-transfer.
- **Domain of validity & failure**: where it holds, boundary/cliff prediction,
  input ablation, algorithm-transfer mini-panel (Wanda / GPTQ).

### Main figures
- Three-column figure, one column per method, unified capability color scheme.
- Each method keeps its NATIVE x-axis (density / bit-width / student-size). Do NOT
  transform coordinates to manufacture spurious curve overlap across methods.
- Show measured, predicted, and held-out points per panel.

### Delivery rules per arm
Actual formula + coefficient table, input availability, validity domain,
parameter-identifiability note, held-out prediction table, behavioral link,
failure cases. A simple law CAN be the final answer if its error and domain are
shown useful. Unified form / shared-partial-params / independent forms are all
acceptable final answers. Negative results edit the affected claim in place, not
only appended to the ledger tail.
