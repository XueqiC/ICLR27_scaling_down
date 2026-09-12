# V88 findings: what the logit displacement explains, and where it stops

Registered before measurement in `paper/docs/prereg/v88_displacement_prereg.md`, including
the cell list, the damage regimes and the pass rules, with one amendment to the T2 formula
written before any cell was measured. Five weight-identity-distinct Pythia states, seven
configurations each, three capabilities: 105 non-zero cells. Ten GPU-minutes on one released
GPU, bfloat16 weights, float32 vocabulary reductions, dense and compressed models scored in
the same run so no comparison crosses hardware. Probe identity matches the frozen protocol
(`probe_sha256` 33d118c7…).

Nothing here is fitted. Both predictions are functions of the measured displacement.

## T1: the loss change is a local functional of the displacement, and it passes

For each scored token with dense logits `z`, `p = softmax(z)`, reference `y`, and
displacement `r = z_compressed - z_dense`, the prediction is
`(E_p[r] - r_y) + 0.5*Var_p(r)`.

| Regime | Cells | Median relative error | Sign agreement |
|---|---:|---:|---:|
| mild, \|dL\| < 0.1 | 41 | **2.3%** | 97.6% |
| moderate, 0.1 to 0.5 | 31 | 15.3% | 93.5% |
| severe, \|dL\| > 0.5 | 33 | 38.8% | 100% |

The registered rule for the mild regime was 20% and 90%. T1 passes with room to spare.

## Where it stops, in units a practitioner controls

Accuracy degrades monotonically with the size of the displacement relative to the logits:

| Measured \|dL\| | Cells | Median relative error | Median \|r\|/\|z\| |
|---|---:|---:|---:|
| below 0.01 | 11 | 0.9% | 0.07 |
| 0.01 to 0.03 | 15 | 2.3% | 0.15 |
| 0.03 to 0.10 | 15 | 4.5% | 0.21 |
| 0.10 to 0.30 | 22 | 12.8% | 0.33 |
| 0.30 to 1.00 | 19 | 23.7% | 0.46 |

By configuration, which is what the paper's ranges are stated in:

| Configuration | Median \|dL\| | Median relative error |
|---|---:|---:|
| pruning d=0.9 | 0.016 | 0.9% |
| pruning d=0.8 | 0.076 | 6.9% |
| pruning d=0.7 | 0.280 | 18.4% |
| pruning d=0.6 | 1.136 | 45.1% |
| grouped RTN 5 bit, g=128 | 0.023 | 1.5% |
| grouped RTN 4 bit, g=128 | 0.097 | 8.3% |
| grouped RTN 3 bit, g=128 | 0.936 | 32.4% |

This is a principled account of the ranges the paper reports empirically: a local expansion
in the displacement is accurate to about ten percent for densities at or above 0.8 and for
four bits or more, and it stops being a useful description below that. The boundary is a
property of the perturbation size, not of the capability.

## T2: the displacement is not a shrinkage of the logits, and it fails

The amended T2 predictor summarises the whole displacement by one scalar per cell,
`eps*B + 0.5*eps^2*W` with `eps = -<r,z>/|z|^2`, `B = z_y - E_p[z]`, `W = Var_p(z)`.

| Regime | Median relative error | Ratio to T1 | Sign agreement |
|---|---:|---:|---:|
| mild | 98.0% | 42× | 65.9% |
| moderate | 97.5% | 6.4× | 64.5% |
| severe | 99.6% | 2.6× | 69.7% |

The registered threshold was 1.5 times T1's error. T2 fails in every regime, and it gets the
sign wrong on a third of the cells. The projected shrinkage is small throughout (median
`eps` 0.030) and explains almost none of the response.

## What carries the damage

Median first-order term across cells: **-0.042**. Median second-order term: **+0.258**. The
loss change is dominated by the p-weighted *variance* of the displacement, while the
first-order component is slightly negative on average. Compression damage in these panels is
second-order noise in logit space, not a systematic shift toward or away from the reference
token. The displacement also spreads across the vocabulary rather than concentrating: the
sixteen largest coordinates carry 2.4% of the p-weighted squared displacement in the mild
regime, rising to 20% in the severe one.

The full decomposition
`eps*B + (E_p[s] - s_y) + 0.5*eps^2*W - eps*Cov_p(z,s) + 0.5*Var_p(s)`
reproduces the exact second-order value to 8.0e-5 across all 105 cells, which is the
bookkeeping check that no term was dropped.

## Consequence for the research plan

The advisor's package A rests on the hypothesis that pruning and quantization act as logit
shrinkage plus isotropic noise. Measured directly, that hypothesis is false. `B` and `V`
survive only as sensitivity summaries, exactly as the registration said they would if T2
failed, and the shrinkage story is dropped on evidence rather than on taste.

What replaces it is sharper. The quantity that carries the damage, `0.5*Var_p(r)`, is
measurable, capability-independent in form, and needs no labels. The next question, which
needs its own registration and its own frozen predictions, is whether it can be predicted
from the configuration and from pre-compression information, on states that entered no fit.

## Reproduction check against the frozen measurements

Recomputed losses are within 0.004 to 0.019 nats of the frozen values for math and code.
QA differs by up to 0.065 nats, which is large relative to its own response, because the QA
probe scores only 263 tokens and the run used a different GPU from the frozen one. QA
readings here are correspondingly weaker, and the frozen values stand unchanged.
