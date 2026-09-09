# P1-v2 new-source frozen prospective: pythia-6.9b@step32000 (P-new)

Predictions committed before measurement (see git log). Signed = pred − actual. MAE over 3 capabilities.

## Protocol A (full dev)
| regime | power | A2 | A1 | cont | strength_only | median_curve | zero |
|---|---|---|---|---|---|---|---|
| prune interp_0.9-0.6 | 0.476 | 0.482 | 0.608 | 0.501 | 0.483 | 0.165 | 0.160 |
| prune extrap_0.55 | 1.635 | 1.423 | 0.939 | 1.457 | 1.493 | 0.274 | 0.889 |

| bits | full | noD0 | per_bit_mean | per_bit_median | zero |
|---|---|---|---|---|---|
| int8 | 0.005 | 0.010 | 0.005 | 0.001 | 0.001 |
| int6 | 0.057 | 0.070 | 0.033 | 0.004 | 0.003 |
| int5 (interp-rule) | 0.373 | 0.457 | 0.237 | 0.067 | 0.029 |
| int4 | 0.576 | 0.730 | 0.484 | 0.064 | 0.098 |
| int3 | 3.972 | 5.103 | 3.300 | 1.834 | 2.711 |
| ≥4-bit | 0.253 | 0.317 | 0.190 | 0.034 | 0.033 |

per-point prune (cap, d, actual, power, A2, cont):
- math d=0.9: act +0.002 | power +0.011 | A2 +0.021 | cont -0.101
- math d=0.8: act +0.016 | power +0.123 | A2 +0.110 | cont +0.083
- math d=0.75: act +0.058 | power +0.266 | A2 +0.252 | cont +0.282
- math d=0.7: act +0.130 | power +0.498 | A2 +0.395 | cont +0.553
- math d=0.65: act +0.244 | power +0.848 | A2 +0.889 | cont +0.895
- math d=0.6: act +0.499 | power +1.344 | A2 +1.383 | cont +1.309
- math d=0.55: act +1.190 | power +2.018 | A2 +1.878 | cont +1.794
- code d=0.9: act -0.002 | power +0.010 | A2 +0.034 | cont -0.033
- code d=0.8: act -0.004 | power +0.073 | A2 +0.054 | cont +0.046
- code d=0.75: act +0.006 | power +0.139 | A2 +0.107 | cont +0.128
- code d=0.7: act +0.076 | power +0.236 | A2 +0.161 | cont +0.238
- code d=0.65: act +0.173 | power +0.369 | A2 +0.369 | cont +0.375
- code d=0.6: act +0.486 | power +0.543 | A2 +0.578 | cont +0.541
- code d=0.55: act +1.121 | power +0.764 | A2 +0.787 | cont +0.734
- qa d=0.9: act -0.023 | power -0.041 | A2 -0.040 | cont +0.096
- qa d=0.8: act -0.048 | power -0.344 | A2 -0.265 | cont -0.338
- qa d=0.75: act -0.101 | power -0.679 | A2 -0.763 | cont -0.753
- qa d=0.7: act -0.100 | power -1.184 | A2 -1.262 | cont -1.300
- qa d=0.65: act -0.288 | power -1.894 | A2 -2.043 | cont -1.980
- qa d=0.6: act -0.631 | power -2.847 | A2 -2.824 | cont -2.792
- qa d=0.55: act -0.357 | power -4.077 | A2 -3.605 | cont -3.737

## Protocol B (dev step<=64k only)
| regime | power | A2 | A1 | cont | strength_only | median_curve | zero |
|---|---|---|---|---|---|---|---|
| prune interp_0.9-0.6 | 0.160 | 0.166 | 0.249 | 0.187 | 0.163 | 0.136 | 0.160 |
| prune extrap_0.55 | 0.705 | 0.549 | 0.677 | 0.630 | 0.252 | 0.148 | 0.889 |

| bits | full | noD0 | per_bit_mean | per_bit_median | zero |
|---|---|---|---|---|---|
| int8 | 0.007 | 0.007 | 0.001 | 0.002 | 0.001 |
| int6 | 0.019 | 0.018 | 0.004 | 0.002 | 0.003 |
| int5 (interp-rule) | 0.047 | 0.052 | 0.058 | 0.057 | 0.029 |
| int4 | 0.057 | 0.122 | 0.055 | 0.038 | 0.098 |
| int3 | 0.489 | 1.792 | 0.174 | 0.407 | 2.711 |
| ≥4-bit | 0.032 | 0.050 | 0.029 | 0.025 | 0.033 |

per-point prune (cap, d, actual, power, A2, cont):
- math d=0.9: act +0.002 | power +0.003 | A2 -0.002 | cont -0.086
- math d=0.8: act +0.016 | power +0.050 | A2 +0.029 | cont +0.012
- math d=0.75: act +0.058 | power +0.121 | A2 +0.110 | cont +0.131
- math d=0.7: act +0.130 | power +0.251 | A2 +0.190 | cont +0.296
- math d=0.65: act +0.244 | power +0.466 | A2 +0.502 | cont +0.507
- math d=0.6: act +0.499 | power +0.794 | A2 +0.815 | cont +0.765
- math d=0.55: act +1.190 | power +1.272 | A2 +1.128 | cont +1.068
- code d=0.9: act -0.002 | power +0.002 | A2 -0.000 | cont -0.103
- code d=0.8: act -0.004 | power +0.035 | A2 -0.001 | cont -0.048
- code d=0.75: act +0.006 | power +0.085 | A2 +0.000 | cont +0.040
- code d=0.7: act +0.076 | power +0.176 | A2 +0.002 | cont +0.167
- code d=0.65: act +0.173 | power +0.326 | A2 +0.308 | cont +0.333
- code d=0.6: act +0.486 | power +0.557 | A2 +0.614 | cont +0.539
- code d=0.55: act +1.121 | power +0.892 | A2 +0.920 | cont +0.785
- qa d=0.9: act -0.023 | power -0.005 | A2 +0.002 | cont +0.111
- qa d=0.8: act -0.048 | power -0.084 | A2 -0.016 | cont -0.069
- qa d=0.75: act -0.101 | power -0.206 | A2 -0.260 | cont -0.268
- qa d=0.7: act -0.100 | power -0.427 | A2 -0.503 | cont -0.540
- qa d=0.65: act -0.288 | power -0.790 | A2 -0.916 | cont -0.884
- qa d=0.6: act -0.631 | power -1.348 | A2 -1.329 | cont -1.301
- qa d=0.55: act -0.357 | power -2.160 | A2 -1.741 | cont -1.791

