# Distillation prediction residuals & pool variation (Closeout Package B, retrospective)

Frozen v41 predictors; residual r = delta - delta_hat. Per budget (b0 ~= first milestone, b1 ~= second).
Systematic bias = mean(r); variation split into pool-sampling vs training. 3 pools x 2 seeds; the two
checkpoints per run share a trajectory (not independent). No causal claim; no universal-E claim.

## math  (decisive predictor: constant)
- b0: bias(mean r)=-0.160, MAE=0.160, resid-std=0.008; pool-sampling std=0.008 vs training std=0.002
- b1: bias(mean r)=-0.085, MAE=0.085, resid-std=0.012; pool-sampling std=0.008 vs training std=0.006

## code  (decisive predictor: constant)
- b0: bias(mean r)=-0.109, MAE=0.109, resid-std=0.023; pool-sampling std=0.022 vs training std=0.004
- b1: bias(mean r)=-0.019, MAE=0.030, resid-std=0.029; pool-sampling std=0.027 vs training std=0.007

## qa  (decisive predictor: E)
- b0: bias(mean r)=-0.868, MAE=0.868, resid-std=0.175; pool-sampling std=0.173 vs training std=0.021
- b1: bias(mean r)=+0.478, MAE=0.543, resid-std=0.423; pool-sampling std=0.420 vs training std=0.043

## B3 Sampling-protocol audit (from manifests)
- U75: selection='first n_per_domain rows; seeded initial and epoch shuffles', pool_tokens=67027, n=223
- U600: selection='first n_per_domain rows; seeded initial and epoch shuffles', pool_tokens=533869, n=1781
- U225: selection='random sample of n_per_domain rows per domain via data_seed=1; seeded shuffles', data_seed(example)=1, pool_tokens=202298, n=671
- E range: endpoints 0.94-14.94; U225 2.34-4.70 (U225 E is INSIDE the endpoint range -> configuration interpolation, not range extrapolation).

## Verdict (Package B)
- The frozen E-only predictor's advantage on QA and its math/code shortfall are re-expressed as
  RESIDUALS with an explicit bias/variation split (above), not as raw-delta std ratios.
- If the systematic bias (mean residual) is small relative to the pool-to-pool spread, the
  frozen predictor is not systematically off but pool-sensitive; if bias is large, the form itself
  misfits. Both are reported per capability/budget; with 3 pools the spread estimate is THIN.
- Protocol boundary: endpoints are pool PREFIXES (first-U), U225 is RANDOM-sampled -> the test is
  'unseen pool size AND changed sampling protocol', not isolated pool-size. U225 E interpolates.
- Claims corrected: drop 'pool-std/train-std ~10x explains the prediction residual' and any causal
  'residual is a data-selection effect' phrasing; state only the described, design-limited variation.
