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

## T2: the displacement is not a pure shrinkage of the logits, and it fails

What T2 tests, after the amendment, is the shrinkage term alone. Its failure rules out the
pure-shrinkage account. It does **not** rule out a shrinkage-plus-noise account, whose noise
part was left to a later check, and this document must not be read as retiring that.

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

## Which term carries the damage, stated per capability rather than as one median

Two medians across cells would not settle this, because the two terms cancel: they have
opposite signs in 81 of 105 cells, and the variance term is non-negative by construction.
Counted per cell:

| Capability | Variance term larger in magnitude | Median \|first order\| | Median \|second order\| |
|---|---|---:|---:|
| math | 35 of 35 | 0.021 | 0.214 |
| code | 35 of 35 | 0.031 | 0.255 |
| QA (2Wiki) | 16 of 35 | 0.610 | 0.448 |

So the variance term dominates for math and code in every cell measured here, and it does not
dominate for QA, where the first-order term is the larger one in more than half the cells.

Seventeen cells have a **negative** measured change, that is, compression that lowers the
loss. A non-negative variance term cannot produce those. In all seventeen the first-order
term is negative and outweighs the variance term, and the second-order account still gets the
sign right in sixteen of them. Any statement that damage is carried by the variance must
therefore be qualified by capability and by the sign of the response.

The displacement spreads across the vocabulary rather than concentrating: the sixteen largest
coordinates carry 2.4% of the p-weighted squared displacement in the mild regime, rising to
20% in the severe one.

The full decomposition
`eps*B + (E_p[s] - s_y) + 0.5*eps^2*W - eps*Cov_p(z,s) + 0.5*Var_p(s)`
reproduces the exact second-order value to 8.0e-5 across all 105 cells, which is the
bookkeeping check that no term was dropped.

## Consequence for the research plan

The pure-shrinkage account of the displacement is measurably wrong on these panels. The
broader shrinkage-plus-noise account is **not** settled here: the registered T2 carries only
the shrinkage term, and the isotropic-noise reading was deferred by the same amendment. `B`
and `V` therefore keep the status the registration gave them if T2 failed, sensitivity
summaries, and the noise part of the hypothesis remains open pending the checks below.

Two limits of this experiment constrain how far any of it can be pushed. The displacement is
computed from the compressed model, so T1 is a post-hoc decomposition of a measured loss, not
a prediction from pre-compression information, and it cannot become one by relabelling. And
the shrinkage coefficient is an uncentred projection: adding a constant to every logit leaves
the probabilities and the loss unchanged but does change that coefficient, so the shrinkage
component needs a shift-invariant definition before its explanatory share is interpreted.
Both are addressed in the follow-up rather than argued away.

What the experiment does hand forward is a candidate quantity. For math and code the
variance term `0.5*Var_p(r)` is the larger of the two in every cell measured, it needs no
reference labels, and its form does not depend on the capability, although computing it still
needs a forward pass of the compressed model. The next question, which needs its own
registration and its own frozen predictions, is whether it can be predicted from the
configuration and from pre-compression information on states that entered no fit. For QA the
first-order term is comparable or larger, so QA would need its own treatment.

## Reproduction check against the frozen measurements

Recomputed losses are within 0.004 to 0.019 nats of the frozen values for math and code.
QA differs by up to 0.065 nats, which is large relative to its own response, because the QA
probe scores only 263 tokens and the run used a different GPU from the frozen one. QA
readings here are correspondingly weaker, and the frozen values stand unchanged.
