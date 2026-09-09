# P1-v2 new-source frozen prospective: pythia-6.9b@step112000 (P-new)

Predictions committed before measurement (see git log). Signed = pred − actual. MAE over 3 capabilities.

## Protocol A (full dev)
| regime | power | A2 | A1 | cont | strength_only | median_curve | zero |
|---|---|---|---|---|---|---|---|
| prune interp_0.9-0.6 | 0.436 | 0.463 | 0.610 | 0.479 | 0.375 | 0.092 | 0.294 |
| prune extrap_0.55 | 2.121 | 1.939 | 2.223 | 2.061 | 0.190 | 1.272 | 2.198 |

| bits | full | noD0 | per_bit_mean | per_bit_median | zero |
|---|---|---|---|---|---|
| int8 | 0.026 | 0.018 | 0.018 | 0.013 | 0.014 |
| int6 | 0.062 | 0.039 | 0.040 | 0.011 | 0.010 |
| int5 (interp-rule) | 0.402 | 0.247 | 0.253 | 0.033 | 0.027 |
| int4 | 0.595 | 0.428 | 0.309 | 0.112 | 0.191 |
| int3 | 3.873 | 3.550 | 0.430 | 1.896 | 6.442 |
| ≥4-bit | 0.271 | 0.183 | 0.155 | 0.042 | 0.061 |

per-point prune (cap, d, actual, power, A2, cont):
- math d=0.9: act +0.005 | power +0.012 | A2 +0.022 | cont -0.113
- math d=0.8: act +0.043 | power +0.134 | A2 +0.117 | cont +0.087
- math d=0.75: act +0.109 | power +0.289 | A2 +0.271 | cont +0.304
- math d=0.7: act +0.217 | power +0.542 | A2 +0.424 | cont +0.599
- math d=0.65: act +0.481 | power +0.922 | A2 +0.965 | cont +0.972
- math d=0.6: act +1.051 | power +1.461 | A2 +1.506 | cont +1.424
- math d=0.55: act +2.596 | power +2.194 | A2 +2.047 | cont +1.953
- code d=0.9: act +0.014 | power +0.023 | A2 +0.043 | cont -0.073
- code d=0.8: act +0.027 | power +0.172 | A2 +0.136 | cont +0.116
- code d=0.75: act +0.093 | power +0.329 | A2 +0.280 | cont +0.308
- code d=0.7: act +0.215 | power +0.558 | A2 +0.424 | cont +0.567
- code d=0.65: act +0.473 | power +0.872 | A2 +0.886 | cont +0.890
- code d=0.6: act +1.132 | power +1.285 | A2 +1.348 | cont +1.280
- code d=0.55: act +2.546 | power +1.808 | A2 +1.810 | cont +1.735
- qa d=0.9: act -0.075 | power -0.038 | A2 -0.090 | cont -0.047
- qa d=0.8: act -0.209 | power -0.318 | A2 -0.410 | cont -0.484
- qa d=0.75: act -0.341 | power -0.628 | A2 -0.888 | cont -0.848
- qa d=0.7: act -0.384 | power -1.095 | A2 -1.366 | cont -1.310
- qa d=0.65: act -0.306 | power -1.753 | A2 -1.938 | cont -1.869
- qa d=0.6: act -0.113 | power -2.634 | A2 -2.510 | cont -2.525
- qa d=0.55: act +1.452 | power -3.773 | A2 -3.081 | cont -3.279

## Protocol B (dev step<=64k only)
| regime | power | A2 | A1 | cont | strength_only | median_curve | zero |
|---|---|---|---|---|---|---|---|
| prune interp_0.9-0.6 | 0.131 | 0.106 | 0.235 | 0.140 | 0.111 | 0.095 | 0.294 |
| prune extrap_0.55 | 1.441 | 1.387 | 1.892 | 1.525 | 1.294 | 1.408 | 2.198 |

| bits | full | noD0 | per_bit_mean | per_bit_median | zero |
|---|---|---|---|---|---|
| int8 | 0.009 | 0.007 | 0.013 | 0.012 | 0.014 |
| int6 | 0.018 | 0.008 | 0.010 | 0.008 | 0.010 |
| int5 (interp-rule) | 0.043 | 0.025 | 0.025 | 0.023 | 0.027 |
| int4 | 0.121 | 0.198 | 0.120 | 0.138 | 0.191 |
| int3 | 1.294 | 4.033 | 3.770 | 4.137 | 6.442 |
| ≥4-bit | 0.048 | 0.060 | 0.042 | 0.045 | 0.061 |

per-point prune (cap, d, actual, power, A2, cont):
- math d=0.9: act +0.005 | power +0.004 | A2 +0.002 | cont -0.115
- math d=0.8: act +0.043 | power +0.063 | A2 +0.033 | cont +0.007
- math d=0.75: act +0.109 | power +0.153 | A2 +0.125 | cont +0.158
- math d=0.7: act +0.217 | power +0.318 | A2 +0.217 | cont +0.368
- math d=0.65: act +0.481 | power +0.589 | A2 +0.627 | cont +0.637
- math d=0.6: act +1.051 | power +1.004 | A2 +1.038 | cont +0.966
- math d=0.55: act +2.596 | power +1.608 | A2 +1.448 | cont +1.355
- code d=0.9: act +0.014 | power +0.005 | A2 +0.004 | cont -0.165
- code d=0.8: act +0.027 | power +0.075 | A2 +0.053 | cont -0.027
- code d=0.75: act +0.093 | power +0.183 | A2 +0.095 | cont +0.155
- code d=0.7: act +0.215 | power +0.380 | A2 +0.138 | cont +0.413
- code d=0.65: act +0.473 | power +0.704 | A2 +0.709 | cont +0.746
- code d=0.6: act +1.132 | power +1.201 | A2 +1.279 | cont +1.155
- code d=0.55: act +2.546 | power +1.924 | A2 +1.850 | cont +1.640
- qa d=0.9: act -0.075 | power -0.003 | A2 -0.013 | cont -0.005
- qa d=0.8: act -0.209 | power -0.049 | A2 -0.079 | cont -0.132
- qa d=0.75: act -0.341 | power -0.120 | A2 -0.260 | cont -0.240
- qa d=0.7: act -0.384 | power -0.249 | A2 -0.441 | cont -0.379
- qa d=0.65: act -0.306 | power -0.462 | A2 -0.583 | cont -0.548
- qa d=0.6: act -0.113 | power -0.788 | A2 -0.725 | cont -0.747
- qa d=0.55: act +1.452 | power -1.262 | A2 -0.867 | cont -0.977

