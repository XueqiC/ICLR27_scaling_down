# P1-v2 new-source frozen prospective: pythia-1b@step32000 (P-new)

Predictions committed before measurement (see git log). Signed = pred − actual. MAE over 3 capabilities.

## Protocol A (full dev)
| regime | power | A2 | A1 | cont | strength_only | median_curve | zero |
|---|---|---|---|---|---|---|---|
| prune interp_0.9-0.6 | 0.254 | 0.261 | 0.393 | 0.275 | 0.363 | 0.204 | 0.412 |
| prune extrap_0.55 | 2.272 | 2.130 | 2.394 | 2.221 | 0.637 | 1.509 | 2.435 |

| bits | full | noD0 | per_bit_mean | per_bit_median | zero |
|---|---|---|---|---|---|
| int8 | 0.003 | 0.005 | 0.005 | 0.001 | 0.001 |
| int6 | 0.028 | 0.039 | 0.041 | 0.012 | 0.008 |
| int5 (interp-rule) | 0.179 | 0.272 | 0.254 | 0.044 | 0.017 |
| int4 | 0.204 | 0.380 | 0.487 | 0.067 | 0.157 |
| int3 | 1.050 | 2.648 | 2.903 | 1.438 | 3.108 |
| ≥4-bit | 0.103 | 0.174 | 0.197 | 0.031 | 0.046 |

per-point prune (cap, d, actual, power, A2, cont):
- math d=0.9: act +0.005 | power +0.007 | A2 +0.010 | cont -0.071
- math d=0.8: act +0.048 | power +0.078 | A2 +0.062 | cont +0.044
- math d=0.75: act +0.116 | power +0.169 | A2 +0.152 | cont +0.172
- math d=0.7: act +0.306 | power +0.317 | A2 +0.241 | cont +0.346
- math d=0.65: act +0.811 | power +0.540 | A2 +0.563 | cont +0.568
- math d=0.6: act +2.307 | power +0.855 | A2 +0.885 | cont +0.836
- math d=0.55: act +3.805 | power +1.284 | A2 +1.207 | cont +1.150
- code d=0.9: act -0.003 | power +0.013 | A2 +0.013 | cont -0.072
- code d=0.8: act +0.003 | power +0.098 | A2 +0.051 | cont +0.029
- code d=0.75: act +0.036 | power +0.188 | A2 +0.119 | cont +0.144
- code d=0.7: act +0.162 | power +0.319 | A2 +0.187 | cont +0.301
- code d=0.65: act +0.514 | power +0.499 | A2 +0.493 | cont +0.502
- code d=0.6: act +1.488 | power +0.735 | A2 +0.799 | cont +0.746
- code d=0.55: act +2.690 | power +1.035 | A2 +1.105 | cont +1.032
- qa d=0.9: act -0.022 | power -0.019 | A2 -0.050 | cont -0.061
- qa d=0.8: act -0.117 | power -0.154 | A2 -0.233 | cont -0.283
- qa d=0.75: act -0.238 | power -0.305 | A2 -0.488 | cont -0.454
- qa d=0.7: act -0.554 | power -0.531 | A2 -0.743 | cont -0.666
- qa d=0.65: act -0.614 | power -0.850 | A2 -0.962 | cont -0.918
- qa d=0.6: act -0.078 | power -1.278 | A2 -1.180 | cont -1.210
- qa d=0.55: act +0.808 | power -1.830 | A2 -1.399 | cont -1.543

## Protocol B (dev step<=64k only)
| regime | power | A2 | A1 | cont | strength_only | median_curve | zero |
|---|---|---|---|---|---|---|---|
| prune interp_0.9-0.6 | 0.249 | 0.218 | 0.321 | 0.239 | 0.236 | 0.224 | 0.412 |
| prune extrap_0.55 | 1.772 | 1.777 | 2.150 | 1.863 | 1.531 | 1.645 | 2.435 |

| bits | full | noD0 | per_bit_mean | per_bit_median | zero |
|---|---|---|---|---|---|
| int8 | 0.002 | 0.003 | 0.001 | 0.002 | 0.001 |
| int6 | 0.004 | 0.005 | 0.010 | 0.009 | 0.008 |
| int5 (interp-rule) | 0.025 | 0.013 | 0.035 | 0.034 | 0.017 |
| int4 | 0.080 | 0.106 | 0.061 | 0.060 | 0.157 |
| int3 | 0.606 | 1.003 | 0.627 | 0.803 | 3.108 |
| ≥4-bit | 0.028 | 0.032 | 0.027 | 0.026 | 0.046 |

per-point prune (cap, d, actual, power, A2, cont):
- math d=0.9: act +0.005 | power +0.003 | A2 +0.004 | cont -0.070
- math d=0.8: act +0.048 | power +0.045 | A2 +0.038 | cont +0.022
- math d=0.75: act +0.116 | power +0.111 | A2 +0.110 | cont +0.130
- math d=0.7: act +0.306 | power +0.229 | A2 +0.182 | cont +0.278
- math d=0.65: act +0.811 | power +0.425 | A2 +0.461 | cont +0.466
- math d=0.6: act +2.307 | power +0.725 | A2 +0.741 | cont +0.696
- math d=0.55: act +3.805 | power +1.162 | A2 +1.020 | cont +0.966
- code d=0.9: act -0.003 | power +0.003 | A2 +0.004 | cont -0.093
- code d=0.8: act +0.003 | power +0.048 | A2 +0.036 | cont +0.001
- code d=0.75: act +0.036 | power +0.118 | A2 +0.086 | cont +0.117
- code d=0.7: act +0.162 | power +0.245 | A2 +0.136 | cont +0.280
- code d=0.65: act +0.514 | power +0.455 | A2 +0.474 | cont +0.489
- code d=0.6: act +1.488 | power +0.776 | A2 +0.811 | cont +0.745
- code d=0.55: act +2.690 | power +1.243 | A2 +1.149 | cont +1.047
- qa d=0.9: act -0.022 | power -0.001 | A2 -0.001 | cont -0.031
- qa d=0.8: act -0.117 | power -0.016 | A2 -0.042 | cont -0.082
- qa d=0.75: act -0.238 | power -0.040 | A2 -0.140 | cont -0.115
- qa d=0.7: act -0.554 | power -0.083 | A2 -0.237 | cont -0.154
- qa d=0.65: act -0.614 | power -0.153 | A2 -0.224 | cont -0.197
- qa d=0.6: act -0.078 | power -0.261 | A2 -0.210 | cont -0.245
- qa d=0.55: act +0.808 | power -0.418 | A2 -0.197 | cont -0.299

