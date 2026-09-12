# Pre-registration: V88, logit displacement and the coefficient-free second-order account

Written and committed **before any V88 measurement was taken**. Nothing in it is fitted:
the predictions tested here have no free parameters, so what is registered is the cell
list, the quantities, the regime split and the reading rules.

## Question

For a compressed model, is the change in conditional loss explained by the displacement it
causes in the final logits, and is that displacement structured as shrinkage plus noise?

For a scored reference token with dense final logits `z`, `p = softmax(z)`, reference index
`y`, and displacement `r = z_compressed - z_dense`:

```
exact second order:      dL_hat = (E_p[r] - r_y) + 1/2 * Var_p(r)
shrinkage-plus-noise:    dL_tilde = eps_hat * B + 1/2 * sigma2_hat * V
  eps_hat    = -<r, z> / |z|^2
  s          = r + eps_hat * z            (residual after removing shrinkage)
  sigma2_hat = Var_p(s)
  B          = z_y - sum_v p_v z_v        V = 1 - sum_v p_v^2
```

`dL_hat` uses the measured displacement and no assumption. `dL_tilde` additionally assumes
the displacement is a shrinkage of the logits plus noise that is isotropic under `p`.

## Status of this experiment

A **diagnostic on already-measured cells**, not a prediction test and not a claim about
transfer. It uses development-panel states deliberately, because the point is to decompose a
known response, and it is to be reported as such. It cannot upgrade any predictive claim in
the paper by itself; its role is to say where a predictable regime ends and why.

## Cells

Five weight-identity-distinct Pythia source states, spanning three sizes and two training
stages, all previously measured:

| state | size | stage |
|---|---|---|
| pythia-160m@step16000 | 160M | early |
| pythia-160m@step143000 | 160M | final |
| pythia-410m@step143000 | 410M | final |
| pythia-1.4b@step16000 | 1.4B | early |
| pythia-1.4b@step143000 | 1.4B | final |

Seven configurations per state, chosen before measurement to span mild to severe damage:
pruning at density 0.9, 0.8, 0.7, 0.6, and grouped round-to-nearest at
(b=5, g=128), (b=4, g=128), (b=3, g=128). That is 35 state-configuration pairs, each scored
on all three capabilities, so 105 cells. No cell may be dropped after the fact, including
cells that collapse.

## Protocol

`bfloat16` weights, matching the frozen measurements' `model_dtype`; probe seed 0, 128 probe
items, odd-indexed half, the completion-token-weighted loss of the existing scorer, pooled
tokens for aggregation. Dense and compressed models are held in memory together and scored
in the same run, so the comparison never crosses hardware or dtype. All vocabulary
reductions in float32. The compression transforms are deterministic and applied through the
existing code paths, so each cell is exactly reproducible.

Every cell reports the measured `dL` recomputed in this run. Where a frozen measurement
exists for the same state and configuration, the difference between the two is reported as a
reproduction check. The frozen value stands; a difference is stated, never corrected.

## Reading rules, fixed in advance

Regimes, by the magnitude of the measured `dL` in the cell: **mild** below 0.1 nats per
token, **moderate** 0.1 to 0.5, **severe** above 0.5. Reported separately, always.

- **T1 passes** if, in the mild regime, the median absolute relative error of `dL_hat`
  against the measured `dL` is at most 20%, and its sign agreement is at least 90%.
- **T2 passes** if, in the mild regime, the median absolute relative error of `dL_tilde` is
  at most 1.5 times that of `dL_hat`. This asks whether the structural reduction is cheap,
  not whether it is exact.
- If T1 fails in the mild regime, the branch stops. The deliverable is then the boundary
  itself: the displacement magnitude beyond which the loss change is not a local function of
  the displacement.
- If T1 passes and T2 fails, `B` and `V` survive only as sensitivity summaries, and the
  shrinkage-plus-noise story is dropped with the evidence that killed it.
- If both pass, the next step is the one that carries a predictive claim: whether
  `(eps, sigma^2)` can be predicted from configuration and source metadata on states that
  entered no fit. That step gets its own registration and its own frozen predictions.

## Budget

At most 2 GPU-hours, on one GPU the user has released for this work, selected by UUID. A
pilot of one state and two configurations runs first as a functional check.
