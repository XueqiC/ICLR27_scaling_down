# P1 new-source frozen prospective: pythia-1b@step96000 (P-new)

Predictions frozen+committed before measurement (b1bf631). Signed = pred - actual.

| arm | cap | config | actual dL | power/full | A2/noD0 | A1/median | strength-only | median | zero |
|---|---|---|---|---|---|---|---|---|---|
| prune | math | d=0.65 | +0.720 | +0.740 (0.020) | +0.780 (0.060) | +0.711 (0.009) | +0.943 (0.223) | +0.573 (0.147) | +0.000 (0.720) |
| prune | math | d=0.55 | +2.692 | +1.761 (0.930) | +1.640 (1.051) | +0.914 (1.777) | +2.419 (0.273) | +1.567 (1.125) | +0.000 (2.692) |
| prune | code | d=0.65 | +0.482 | +0.575 (0.092) | +0.560 (0.078) | +0.513 (0.031) | +1.130 (0.648) | +0.558 (0.076) | +0.000 (0.482) |
| prune | code | d=0.55 | +2.741 | +1.191 (1.550) | +1.383 (1.358) | +0.660 (2.081) | +2.759 (0.018) | +1.526 (1.215) | +0.000 (2.741) |
| prune | qa | d=0.65 | -0.770 | -0.304 (0.466) | -0.390 (0.380) | -0.363 (0.407) | +0.470 (1.240) | -0.201 (0.569) | +0.000 (0.770) |
| prune | qa | d=0.55 | +0.755 | -0.671 (1.426) | -0.253 (1.009) | -0.466 (1.222) | +1.285 (0.530) | -0.316 (1.072) | +0.000 (0.755) |
| quant | math | int4 | +0.198 | +0.314 (0.116) | +0.015 (0.184) | +0.130 (0.068) | +0.000 (0.198) | | |
| quant | math | int3 | +5.276 | +5.966 (0.690) | +2.952 (2.324) | +3.862 (1.414) | +0.000 (5.276) | | |
| quant | code | int4 | +0.176 | +0.130 (0.046) | -0.218 (0.394) | +0.164 (0.012) | +0.000 (0.176) | | |
| quant | code | int3 | +5.864 | +5.545 (0.318) | +2.441 (3.423) | +5.864 (0.001) | +0.000 (5.864) | | |
| quant | qa | int4 | -0.314 | -0.256 (0.059) | -0.260 (0.054) | -0.062 (0.253) | +0.000 (0.314) | | |
| quant | qa | int3 | +3.465 | +3.283 (0.182) | +1.831 (1.635) | +3.911 (0.446) | +0.000 (3.465) | | |

## MAE by config (over 3 capabilities)
- prune d=0.65: power=0.193, A2=0.172, A1_gamma1=0.149, strength_only=0.704, median_curve=0.264, zero=0.657
- prune d=0.55: power=1.302, A2=1.139, A1_gamma1=1.693, strength_only=0.273, median_curve=1.137, zero=2.063
- quant int4: full_N0_L0_D0=0.074, noD0_N0_L0=0.211, per_bit_median=0.111, zero=0.230
- quant int3: full_N0_L0_D0=0.397, noD0_N0_L0=2.460, per_bit_median=0.620, zero=4.868

## Verdict (P-new: genuinely new source, predictions frozen at b1bf631 before measurement)

- **Pruning, d=0.65 (density interpolation, stage interpolation, new size inside range): SUCCESS for
  source-conditioned prediction.** Power 0.193, A2 0.172, A1(γ=1) 0.149 vs strength-only 0.704, median-curve
  0.264, zero 0.657. Math/code absolute errors 0.02–0.09; QA is the weak capability (0.38–0.47). The three
  source-conditioned forms are within 0.04 of each other: no form is singled out; by parsimony the simplest
  (per-density or γ=1) is adequate. This is the independent confirmation A2 lacked after v42 — on ONE source.
- **Pruning, d=0.55 (density extrapolation): FAILURE of every source-conditioned candidate.** Power 1.302, A2
  1.139, A1 1.693; all UNDER-predict this source's collapse (math +2.69, code +2.74 actual) by 1–1.5 nats, and
  QA's sign is missed (+0.755 actual vs negative predictions). The source-blind strength-only curve is closest
  (0.273) because this source's deep damage happens to sit near the global curve. The transfer range of the
  source-conditioned family is therefore narrowed to the interpolation regime; no A3 is sought.
- **Quantization (seen bits on a new source): the frozen full {N0,L0,D0} predictor wins at both bits.** int4
  MAE 0.074 vs no-D0 0.211 / per-bit median 0.111 / zero 0.230; int3 0.397 vs median 0.620 / no-D0 2.46 / zero
  4.87. D0 carries real information here (no-D0 is worse than the median). Reported per bit; one source does
  not establish a quantization law.
- Statistical unit: one source-state (three capabilities × configs); point estimates; no population claim.
