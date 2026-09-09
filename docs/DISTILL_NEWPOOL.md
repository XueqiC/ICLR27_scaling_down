# Distillation new-pool prediction (P3): endpoint pools -> unseen U=225

Endpoint fit checkpoints: 18 (U75+U600 uxseen +seeds). U=225 runs: 6 (data-seed x train-seed). Predictors frozen on endpoints, predict U=225.

E range — endpoints 0.94-14.94, U=225 2.34-4.70 (interpolation).

## Predictor MAE on U=225 (per capability)

| cap | constant | T-only | E-only | 2D |
|---|---|---|---|---|
| math | 0.122 | 0.254 | 0.191 | 0.215 | (best: constant)
| code | 0.069 | 0.174 | 0.117 | 0.145 | (best: constant)
| qa | 1.449 | 1.041 | 0.706 | 0.710 | (best: E)

## Error source: pool-sampling vs training (std of final-checkpoint delta)
| cap | pool-sampling std | training std |
|---|---|---|
| math | 0.008 | 0.006 |
| code | 0.027 | 0.007 |
| qa | 0.427 | 0.043 |

## Read (P3 result)

Frozen predictors (fit on U75+U600 endpoint trajectories, 18 checkpoints, E 0.94–14.94) predicting the
never-fit middle pool U=225 (E 2.34–4.70, interpolation), 6 runs = 3 pool-seeds × 2 train-seeds:

| cap | constant | T-only | E-only | 2D | best |
|-----|----------|--------|--------|-----|------|
| math | **0.122** | 0.254 | 0.191 | 0.215 | constant |
| code | **0.069** | 0.174 | 0.117 | 0.145 | constant |
| qa   | 1.449 | 1.041 | **0.706** | 0.710 | E-only |

- **QA: the reuse-count (E) law transfers to the new pool** — E-only nearly halves the constant's error
  (0.71 vs 1.45). This is the one place the E coordinate has real predictive value on an unseen pool.
- **math/code: a per-capability CONSTANT wins** — the endpoint E/T relationship over-extrapolates to U=225
  (E-only/T-only/2D all worse than constant). δ_math/δ_code are small (~0.07–0.12) and pool-invariant here,
  so there is little pool-dependent signal to predict beyond the mean.

### Error source: pool-sampling vs training randomness (std of final-checkpoint δ)

| cap | pool-sampling std | training std | ratio |
|-----|-------------------|--------------|-------|
| math | 0.008 | 0.006 | ~1.3× |
| code | 0.027 | 0.007 | ~3.9× |
| qa   | 0.427 | 0.043 | ~9.9× |

- **Variability is dominated by pool SAMPLING, not training noise** — strongly for QA (~10×) and code (~4×).
  WHICH examples land in the pool matters far more than the training seed. This directly answers the
  distillation "residual" question: the unexplained response variation is largely a data-pool-selection
  effect, not optimization stochasticity.

## Honest headline (P3)

> Predicting an unseen middle distillation pool (U=225) from endpoint pools: the reuse-count E law transfers
> for QA (E-only 0.71 vs constant 1.45) but NOT for math/code (a per-capability constant is best; the E/T fit
> over-extrapolates). Response variability across seeds is dominated by pool SAMPLING, not training
> randomness (QA ~10×, code ~4×) — the distillation residual is a data-selection effect. Small panel (6 runs,
> 18 endpoint checkpoints), single teacher/recipe; endpoint pools first-U vs U=225 sampled.

## Package B correction (see DISTILL_RESIDUALS.md)
The U225 error is dominated by SYSTEMATIC BIAS of the frozen predictors, not pool variation. QA E-only bias sign-flips across budget (-0.87 -> +0.48) = form misfit; it beats a constant but does not cleanly transfer. The pool>training split describes only the small variation around the bias. Endpoints are first-U prefixes, U225 is random-sampled (protocol also changes). Drop the 'variability is a data-selection effect (~10x)' causal wording.
