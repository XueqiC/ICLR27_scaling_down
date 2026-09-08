# Controlled-panel predictor: exact form, dual baselines, by-config decomposition

Consolidates the Pythia controlled-panel prediction test (v36 / v36b) at the standard the advisor
asked for: the actual predictor (not just an input set), BOTH improvement CIs, per-held-out-fold
error, and a per-config decomposition. Panel = pythia-{160m,410m,1.4b} × step{16000,64000,143000}
= **9 source-states**; arms = pruning (v6) and quantization (v10), forward-only. Endpoint = signed
config-level ΔL_c (native-token CE nats). Distillation is NOT in this table (see gap below).

## 1. The actual predictor (this is a source-transfer model at fixed compression, not yet a law over d/b)

For each arm, one unweighted OLS solve on all signed config-level ΔL observations:

    ΔL̂_c = Σ_q 1[config=q] · ( β0_q + β1_q·z(log N0) + β2_q·z(L_{c,0}) [ + β3_q·z(log D0) ] )

- **config q = the compression setting itself** (pruning density d∈{.9,.8,.7,.6}; quant bit b∈{8,6,4,3}),
  entered as an INDICATOR interacted with every continuous term. So each d/b gets its **own** intercept
  and slopes — this is effectively a **separate linear model per compression config**.
- Continuous inputs z-scored on train. No regularization (plain lstsq). No a_c labels — ΔL fit directly.
- Coefficient count per arm/capability: **3 configs-worth** — with D0, 4 coeffs × 4 configs = 16 fit to
  16 train cells in leave-one-out → **saturated** (full-rank but zero residual DOF); without D0, 12 coeffs.
  This saturation is disclosed in v36b metadata and is why held-out (not train) error is the only honest read.

**Consequence (answers the advisor's key question directly):** because d/b enters as indicators, this
model predicts **source-state transfer AT EACH FIXED compression setting** — "given a new (size, step),
predict ΔL at this same density/bit." It does **NOT** parameterize ΔL as a smooth function of d or b, so
it is **not yet a scaling-down law over the compression axis**. A law over d/b (e.g. the pruning power
((1−d)/0.3)^γ or a quant g(b)) is a separate, still-open prediction target.

## 2. Two different improvements — report BOTH (advisor #2.2)

"4 CIs exclude 0" refers to the **D0-incremental** test (does D0 help beyond {N0,L0}). It does NOT by
itself show the model beats the strongest simple baseline. The two are reported side by side.

### 2a. D0-incremental value: B={N0,L0,D0} vs A={N0,L0}  (v36; split matters!)

| arm/cap    | leave-one-SIZE-out         | leave-one-STEP-out         |
|------------|----------------------------|----------------------------|
| prune-math | **+0.469 [0.302, 0.775]** ✓ | +0.361 [−0.037, 0.605]     |
| prune-code | −0.060 [−0.770, 0.311]     | +0.402 [−0.009, 0.616]     |
| prune-qa   | −0.171 [−0.576, 0.039]     | −0.031 [−0.105, 0.050]     |
| quant-math | **+1.257 [0.814, 1.891]** ✓ | +0.845 [−0.134, 1.671]     |
| quant-code | **+1.583 [0.924, 2.773]** ✓ | **+1.451 [0.153, 2.104]** ✓ |
| quant-qa   | −0.339 [−1.248, 0.121]     | −0.025 [−0.357, 0.485]     |

→ **4/12 comparisons have a CI excluding 0** (size-out prune-math, quant-math, quant-code; step-out
quant-code). D0's incremental value is **split-dependent** (prune-code is +0.40 step-out but −0.06
size-out) and **null for QA**. The clean statement: *expanding to 3 sizes surfaced incremental D0
predictive evidence not detectable on the 2-size panel, concentrated in quant and in math/code, not QA.*

### 2b. Full model vs STRONGEST simple baseline (v36b; baseline = per-config median unless noted)

| arm/cap    | strongest baseline | full MAE | improvement vs baseline   |
|------------|--------------------|----------|---------------------------|
| prune-math | config_median      | 0.255    | +0.159 [−0.048, +0.470]   |
| prune-code | config_median      | 0.298    | **+0.220 [+0.029, +0.586]** ✓ |
| prune-qa   | zero               | 0.796    | −0.299 [−0.884, +0.115]   |
| quant-math | config_median      | 1.144    | +0.266 [−0.291, +0.808]   |
| quant-code | config_median      | 0.626    | **+0.927 [+0.111, +1.877]** ✓ |
| quant-qa   | config_median      | 1.989    | −0.517 [−1.534, +0.831]   |

→ Against the **strongest** simple baseline, the full model significantly wins (CI excludes 0) **only for
CODE, both arms**. For MATH the point improvement is positive but the CI includes 0. QA is worse than
baseline. This is weaker than the earlier "beats baseline for math/code" — which rested on the weaker
zero/constant baseline.

## 3. By-config decomposition — the gains live in the AGGRESSIVE configs (advisor #2.4)

Full-vs-baseline improvement, split by compression setting (leave-one-step-out):

| arm  | cfg (mild→aggressive)         | improvement by config                    |
|------|-------------------------------|------------------------------------------|
| prune| d=0.9 / 0.8 / 0.7 / 0.6       | −0.01 / +0.00 / +0.13 / **+0.52** (math) |
| prune| d=0.9 / 0.8 / 0.7 / 0.6       | +0.00 / +0.05 / +0.10 / **+0.73** (code) |
| quant| b=8 / 6 / 4 / 3               | −0.00 / −0.01 / −0.08 / **+1.15** (math) |
| quant| b=8 / 6 / 4 / 3               | −0.01 / −0.01 / −0.08 / **+3.80** (code) |

→ **Essentially ALL the quant improvement is the int3 collapse region** (b≥4 ≈ 0 or slightly negative);
this matches C26/v30b (98.7% collapse artifact). Pruning gains concentrate at d=0.6 (approaching the
cliff); the mild region d≥0.8 gets ~0. **In the smooth, practically-relevant region the controlled
predictor adds little** — its measured value is largely predicting WHERE/HOW HARD the aggressive-setting
damage lands.

## 4. Per-held-out-fold error (advisor #2.3: 9 source-states, not many independent models)

Full-vs-baseline improvement by which step was held out (leave-one-step-out):

| arm/cap    | held 16000 | held 64000 | held 143000 (latest, max D0) |
|------------|-----------|-----------|------------------------------|
| prune-math | +0.211    | +0.262    | **+0.003**                   |
| prune-code | +0.315    | +0.307    | **+0.037**                   |
| quant-math | +0.259    | +0.544    | **−0.006**                   |
| quant-code | +0.994    | +1.336    | +0.453                       |

→ The gain **collapses when the LATEST (highest-D0) step is held out** — the model interpolates between
observed steps but does not extrapolate to a more-trained source. Uncertainty here is *panel-conditional*
(3 overlapping-trajectory step clusters, shared probes); probe bootstrap describes measurement noise on
these 9 states, NOT a population of independent models. Two-of-three folds carry the signal.

## 5. Honest headline (replaces "combined beats baseline for math/code")

> On the controlled 9-state Pythia panel, adding training tokens D0 to {N0, dense L0} lowers held-out
> error beyond dense loss alone in 4/12 arm×cap×split comparisons (CI excludes 0), for math/code not QA.
> But against the strongest simple per-config baseline, the full predictor significantly wins only for
> CODE; the gains are concentrated in the most aggressive configs (pruning d=0.6, quant int3 collapse)
> and vanish when the most-trained checkpoint is the held-out target. The current model predicts
> source-transfer at a FIXED compression setting, not the compression axis. This is a real but narrow
> positive that needs a frozen prospective confirmation, not a headline "law".

## 6. Gaps this table exposes (feed the next round)
- **Distillation is absent from this controlled table** — only 2 LoRA trend points exist (advisor #3).
  Need the full 3-size × 3-step LoRA panel to give distillation a parallel source-transfer test.
- **No compression-axis law** — indicators, not g(d)/g(b). A real scaling-down law must predict across
  unseen d/b, not just unseen (size,step) at seen d/b.
- **Extrapolation to higher D0 unverified** — every gain leans on interpolation between the 3 steps.
