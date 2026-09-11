# V74: the V69 median and interpolation algorithms

This describes the implementation frozen in V69, verified against its recorded code SHA-256 values. No model is refitted and no test response is used to construct anchors.

## Development data and response

The development snapshot is `dev_rows` in [develop.json](../v69-quant-confirm/develop.json), loaded by `load_dev` in [analysis/v69_quant_confirm.py](../../analysis/v69_quant_confirm.py). It uses all 54 previously unblinded V54 state/configuration cells: six development states, each at b in {3,4,5} and g in {64,128,256}, with math, code and QA responses (162 capability rows). The six states and their original measurement files are:

- `pythia-160m@step16000`: `results/v54-quant-group/pythia-160m--step16000/quant_group_losses.json`
- `pythia-160m@step143000`: `results/v54-quant-group/pythia-160m--step143000/quant_group_losses.json`
- `pythia-410m@step16000`: `results/v54-quant-group/pythia-410m--step16000/quant_group_losses.json`
- `pythia-410m@step143000`: `results/v54-quant-group/pythia-410m--step143000/quant_group_losses.json`
- `pythia-1.4b@step16000`: `results/v54-quant-group/pythia-1.4b--step16000/quant_group_losses.json`
- `pythia-1.4b@step143000`: `results/v54-quant-group/pythia-1.4b--step143000/quant_group_losses.json`

For each capability c and state s, the signed response is `dL[s,c,b,g] = loss[s,c,b,g] - dense_loss[s,c]` in nats, with the dense reference from the same V54 JSON file. Development uses only the nine configurations above, even if the original files now also contain confirmation measurements. g=32/512 were never measured in the development data; 1.4B@112000 is absent from all development fits. This is V69's full 54-cell development set, not V55's original 24-cell development split.

## Per-configuration median

In V69 `fit_all`, independently for each capability and each of the nine (b,g) configurations, the scalar anchor is:

```text
M[c,b,g] = np.median([dL[s,c,b,g] for s in the six development states])
```

For six states, `np.median` averages the third and fourth sorted signed values. It does not take absolute values or floor the anchors. Each capability has nine anchors, with no ridge fit and no dependence on the target state's features. The same frozen median predictions apply to every target state. During development LOSO only, each fold instead computes anchors over its five training states; the confirmation freeze uses all six.

V69 `predict_all` passes these anchors to its own `interpolate`, also used by the piecewise interpolation candidate. Thus an unmeasured group size has a defined prediction; no median is computed at g=32/512. At each tested b:

```text
prediction[c,b,32]  = max(0, 2*M[c,b,64]  - M[c,b,128])
prediction[c,b,512] = max(0, 2*M[c,b,256] - M[c,b,128])
prediction[c,b,128] = M[c,b,128]  # signed; no floor inside the grid
```

The floor is applied after extrapolation, not to individual anchors.

## Piecewise interpolation and its boundary rule

The interpolation candidate's anchors differ from the median's. V69 `design` and `fit_all` fit a four-coefficient ridge regression for each capability/configuration on the six development states (36 coefficients per capability): `A[c,b,g;s] = beta[c,b,g] dot phi(s,c)`. The configuration-indicator design makes these nine regression blocks independent. Ridge lambda is 0.001 and penalizes every coefficient, including intercepts. [analysis/v55_quant_group_fit.py](../../analysis/v55_quant_group_fit.py) supplies `ridge_fit`, `coordinates`, `standardization` and `phi`: `phi = [1, z(log(N0)), z(L0c), z(log(D0))]`, using natural logs and population mean/std pooled over the development rows and capabilities; constant scales become 1. N0 counts transformer matrices excluding embeddings/head; D0 = step * 2097152. At prediction time the frozen scaler and the target's frozen `phi_raw` produce nine scalar anchors. These are fitted predictions for the target state, not its measured quantized losses.

V69 `grid_weights` and `interpolate` then use the same rule for either anchor type:

1. Coordinates are `x = log2(2**(b-1)-1)` and `v = log2(g/128)`. The measured grid has x at b={3,4,5} and v={-1,0,1} for g={64,128,256}.
2. Choose adjacent bit anchors using `searchsorted(xs, x, side='right') - 1`, clipped to indices 0 or 1. Choose g=64,128 if v<=0 and g=128,256 otherwise. For local coordinates tx and tv, sum the four anchors with product weights `(1-tx)*(1-tv)`, `(1-tx)*tv`, `tx*(1-tv)`, and `tx*tv`. All confirmation bits are measured bit anchors, so the other bit's weights are zero.
3. For g<64, extend the line through g=64,128 in v; for g>256, extend the line through g=128,256 in v. Do not clamp g to the grid or use the two outermost g=64,256 anchors. At g=32, v=-2 and the g weights are (2,-1); at g=512, v=2 and they are (-1,2). The formulas above apply with A replacing M.
4. Return `max(0, weighted_sum)` only when `abs(v)>1` (g<64 or g>256). For 64<=g<=256, return the signed weighted sum with no floor. There is no bit extrapolation; V69's confirmation protocol permits b=3,4,5 only.

The boundary helper is V69's implementation, not V55's older four-corner unclipped interpolation function. The exact boundary rule recorded in both development and freeze is:

> Piecewise bilinear interpolation in x=log2(qmax) and v=log2(g/128) on the 3x3 measured grid. For g<64 use g=64,128; for g>256 use g=128,256, linearly extrapolating in v at the same b. Floor the extrapolated predicted signed dL at zero: max(0, dL_extrapolated). Interior predictions remain signed. Apply this identical rule to same-input interpolation and per-config medians. No bit extrapolation; only b=3,4,5 is in the confirmation protocol.

## Frozen median anchors and resulting boundary predictions

Values below are in nats, rounded to six decimals. g=128 predictions equal the middle anchor, including a negative QA anchor at b=5.

| Capability | b | M(g=64) | M(g=128) | M(g=256) | Prediction g=32 | Prediction g=512 |
|---|---:|---:|---:|---:|---:|---:|
| Math | 3 | 0.529680 | 0.794214 | 1.273353 | 0.265147 | 1.752491 |
| Math | 4 | 0.067102 | 0.082457 | 0.107243 | 0.051748 | 0.132030 |
| Math | 5 | 0.014188 | 0.015325 | 0.018963 | 0.013051 | 0.022601 |
| Code | 3 | 0.723170 | 1.047718 | 1.709472 | 0.398621 | 2.371227 |
| Code | 4 | 0.087681 | 0.102953 | 0.123722 | 0.072409 | 0.144491 |
| Code | 5 | 0.016906 | 0.019075 | 0.025790 | 0.014737 | 0.032505 |
| QA | 3 | 0.134030 | 0.355929 | 0.740984 | 0.000000 | 1.126040 |
| QA | 4 | 0.064282 | 0.053425 | 0.017214 | 0.075140 | 0.000000 |
| QA | 5 | 0.006045 | -0.001753 | -0.013902 | 0.013843 | 0.000000 |

For example, QA at b=3,g=32 extrapolates to -0.087868346 nats and is floored to zero; at b=5,g=128 the -0.001752614-nat median remains signed.

## Three-way table and verification

D is the frozen development selection: surface for math, median for code, zero for QA. F rows give all six frozen candidates. R is the supplied post-test recommendation: interpolation for math/code on seen development states; median for QA and all new-state capabilities. R uses existing frozen predictions but the rule choice is retrospective. It is not chosen by minimizing each panel's test error (for example, new-state boundary code favors interpolation, and new-state g=128 QA favors zero).

The V74 script verifies the develop-to-freeze-to-compare SHA-256 chain, unchanged frozen predictions, all 27 median anchors from development responses, all 126 median and interpolation predictions, and all 54 panel/capability/candidate MAEs. MAE averages absolute errors against signed dL equally over measured cells. Each panel is balanced across states, so its macro state MAE equals its cell MAE. The JSON retains full precision; the paper table rounds to four decimals.

Regenerate with `python -B analysis/v74_quant_threeway.py`; verify generated artifacts without writes with `python -B analysis/v74_quant_threeway.py --check`.

### Input SHA-256

- `results/v69-quant-confirm/develop.json`: `9f035cb37a94d425064f37d822490b52dcad4c669e62864f27cba4ffa5e5643d`
- `results/v69-quant-confirm/freeze.json`: `6ade0a3ff576781cf286978519b6a922500d37e3bc72cef4e3e635e826cccdc2`
- `results/v69-quant-confirm/compare.json`: `12944c61ed42c61f1beaf8f84ef5c87d900b3b1a38d65d175976e27d3d4e6bac`
- `analysis/v69_quant_confirm.py`: `1f19cf4ad81eedae36d401a2373c4f9f6de4c152ce688ee70e78efca9c063138`
- `analysis/v55_quant_group_fit.py`: `596015ebcc8f635e8454f914bda159654cc85b2fc463f614243ff13df3c3f926`
