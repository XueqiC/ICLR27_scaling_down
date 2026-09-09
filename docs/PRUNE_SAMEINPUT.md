# Pruning same-input baselines (Closeout Package A, retrospective diagnostic)

v40 candidate is frozen; A1 = source-conditioned with gamma fixed to 1 (beta refit on dev);
A2 = per-density source regression + linear-in-d interp(0.65)/extrap(0.55). Same phi/dev/units.

## Per-capability MAE (full test set) and paired improvement vs same-input baselines
| cap | candidate | A1(g=1) | A2(per-d) | strength-only | median | zero | cand vs A1 | cand vs A2 |
|---|---|---|---|---|---|---|---|---|
| math | **0.371** | 0.859 | 0.283 | 1.369 | 1.478 | 2.374 | +0.488 | -0.088 |
| code | **0.524** | 0.941 | 0.279 | 1.635 | 1.874 | 2.726 | +0.418 | -0.244 |
| qa | **0.759** | 0.985 | 0.818 | 1.608 | 1.918 | 1.773 | +0.227 | +0.059 |

(improvement = MAE(baseline) - MAE(candidate); positive = candidate better)

## Signed per-point errors (candidate) — for correct over/under-prediction wording
| source | cap | d | kind | obs | cand | signed(cand-obs) | A1 |A2 |
|---|---|---|---|---|---|---|---|---|
| 1.4b@64000 | math | 0.65 | interp(0.65) | +0.429 | +0.494 | +0.064 | +0.462 | +0.507 |
| 1.4b@64000 | math | 0.55 | extrap(0.55) | +2.091 | +1.175 | -0.916 | +0.595 | +1.126 |
| 1.4b@64000 | code | 0.65 | interp(0.65) | +0.336 | +0.599 | +0.263 | +0.537 | +0.585 |
| 1.4b@64000 | code | 0.55 | extrap(0.55) | +2.086 | +1.241 | -0.845 | +0.690 | +1.337 |
| 1.4b@64000 | qa | 0.65 | interp(0.65) | -0.598 | -0.177 | +0.421 | -0.184 | -0.220 |
| 1.4b@64000 | qa | 0.55 | extrap(0.55) | +0.258 | -0.390 | -0.648 | -0.237 | -0.259 |
| 160m@143000 | math | 0.65 | interp(0.65) | +3.964 | +4.094 | +0.130 | +4.021 | +4.408 |
| 160m@143000 | math | 0.55 | extrap(0.55) | +8.493 | +9.742 | +1.249 | +5.170 | +8.466 |
| 160m@143000 | code | 0.65 | interp(0.65) | +5.641 | +5.041 | -0.600 | +4.791 | +5.253 |
| 160m@143000 | code | 0.55 | extrap(0.55) | +8.874 | +10.448 | +1.574 | +6.160 | +9.107 |
| 160m@143000 | qa | 0.65 | interp(0.65) | +4.160 | +3.047 | -1.113 | +2.960 | +3.246 |
| 160m@143000 | qa | 0.55 | extrap(0.55) | +7.293 | +6.725 | -0.568 | +3.806 | +5.782 |
| 410m@96000 | math | 0.8 | seen(0.8) | +0.098 | +0.098 | +0.001 | +0.372 | +0.087 |
| 410m@96000 | math | 0.6 | seen(0.6) | +1.211 | +1.076 | -0.135 | +0.745 | +1.105 |
| 410m@96000 | math | 0.65 | interp(0.65) | +0.648 | +0.679 | +0.031 | +0.652 | +0.716 |
| 410m@96000 | math | 0.55 | extrap(0.55) | +2.059 | +1.616 | -0.443 | +0.838 | +1.493 |
| 410m@96000 | code | 0.8 | seen(0.8) | +0.080 | +0.184 | +0.104 | +0.486 | +0.105 |
| 410m@96000 | code | 0.6 | seen(0.6) | +1.541 | +1.374 | -0.167 | +0.972 | +1.483 |
| 410m@96000 | code | 0.65 | interp(0.65) | +0.808 | +0.933 | +0.125 | +0.850 | +0.931 |
| 410m@96000 | code | 0.55 | extrap(0.55) | +2.447 | +1.933 | -0.513 | +1.093 | +2.036 |
| 410m@96000 | qa | 0.8 | seen(0.8) | -0.083 | +0.170 | +0.253 | +0.528 | +0.126 |
| 410m@96000 | qa | 0.6 | seen(0.6) | +0.454 | +1.509 | +1.055 | +1.056 | +1.544 |
| 410m@96000 | qa | 0.65 | interp(0.65) | -0.087 | +0.991 | +1.078 | +0.924 | +1.017 |
| 410m@96000 | qa | 0.55 | extrap(0.55) | +1.253 | +2.187 | +0.934 | +1.188 | +2.071 |

## Sensitivity: MAE excluding the max-damage source (160m@143k)
| cap | candidate | A1 | A2 |
|---|---|---|---|
| math | 0.265 | 0.583 | 0.299 |
| code | 0.336 | 0.661 | 0.269 |
| qa | 0.732 | 0.533 | 0.686 |

## Verdict (Package A)

- **Source-conditioned input is what carries the prediction, not the specific shared power form.** Both the
  v40 candidate and the far simpler A2 (per-density source regression + linear-in-d interp/extrap, NO shared
  exponent) beat every compression-strength-only baseline by a wide margin (MAE ~0.28-0.82 vs strength-only
  1.4-1.6, median 1.5-1.9, zero 1.8-2.4). The win in v40 is a SOURCE-INFORMATION win.
- **The shared power exponent is NOT independently confirmed.** A2 ties or beats the candidate for math
  (candidate 0.371 vs A2 0.283) and code (0.524 vs 0.279), and is within noise for qa. The candidate beats
  only the gamma=1 variant A1 (by +0.23 to +0.49), i.e. SOME curvature helps, but no evidence singles out the
  learned exponent over a per-density fit. Revised claim: keep "source-conditioned inputs improve unseen-density
  prediction"; DROP "the specific exponent gamma_c is independently confirmed."
- **Signed-error correction.** At d=0.55 the candidate's signed errors are MIXED (6 of 9 under-predict, 3 over-
  predict), not systematic under-prediction. The most-fragile source (160m@143k) is OVER-predicted (math +1.25,
  code +1.57) while less-fragile sources (1.4b@64k, 410m@96k) are under-predicted. The earlier "systematically
  under-predicts near the cliff" wording is wrong and is replaced by this signed account.
- Status: RETROSPECTIVE DIAGNOSTIC (v40 frozen prediction unchanged). A2 being simplest-and-competitive, the
  paper reports the source-conditioned family with A2 as the honest reduced form and the power shape as a
  not-independently-necessary variant.
