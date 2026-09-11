# V69 quantization confirmation

Measured 21/21 cells; 63/63 capability responses. All errors use frozen predictions.

Near-zero: measured |dL| < 0.1 nats. Clear-damage: |dL| >= 0.1, including negative responses; this is the complementary magnitude regime. Counts differ by capability. MAE below weights measured cells equally; JSON also reports macro MAE by state and separate development-state boundary, new-state boundary, and new-state interior panels.

Selections (development LOSO only): math: low_order_2d; code: median; qa: zero.

## all

| Candidate | Math MAE (n) | Code MAE (n) | QA MAE (n) |
|---|---:|---:|---:|
| 2-D surface | 0.252781 (21) | 0.317250 (21) | 0.954517 (21) |
| Bilinear | 0.340442 (21) | 0.394596 (21) | 1.143284 (21) |
| Piecewise interpolation | 0.097827 (21) | 0.153527 (21) | 0.531728 (21) |
| Bit only | 0.422389 (21) | 0.573241 (21) | 1.175435 (21) |
| Per-config median | 0.323413 (21) | 0.383784 (21) | 0.322192 (21) |
| Zero | 0.534292 (21) | 0.628711 (21) | 0.356154 (21) |

## near_zero

| Candidate | Math MAE (n) | Code MAE (n) | QA MAE (n) |
|---|---:|---:|---:|
| 2-D surface | 0.109495 (11) | 0.125884 (9) | 0.882040 (16) |
| Bilinear | 0.159500 (11) | 0.116000 (9) | 1.150968 (16) |
| Piecewise interpolation | 0.011902 (11) | 0.025842 (9) | 0.204008 (16) |
| Bit only | 0.143912 (11) | 0.201970 (9) | 1.173193 (16) |
| Per-config median | 0.023017 (11) | 0.032284 (9) | 0.134821 (16) |
| Zero | 0.037525 (11) | 0.027084 (9) | 0.038641 (16) |

## clear_damage

| Candidate | Math MAE (n) | Code MAE (n) | QA MAE (n) |
|---|---:|---:|---:|
| 2-D surface | 0.410396 (10) | 0.460775 (12) | 1.186443 (5) |
| Bilinear | 0.539478 (10) | 0.603543 (12) | 1.118694 (5) |
| Piecewise interpolation | 0.192344 (10) | 0.249291 (12) | 1.580432 (5) |
| Bit only | 0.728714 (10) | 0.851695 (12) | 1.182609 (5) |
| Per-config median | 0.653848 (10) | 0.647408 (12) | 0.921780 (5) |
| Zero | 1.080736 (10) | 1.079931 (12) | 1.372196 (5) |

The new state's frozen L0 comes from its pre-existing V53/V6 dense evaluation. Targets subtract V54's same-file dense reference; each difference from frozen L0 is reported in compare.json. No refitting or test-based selection occurs.
