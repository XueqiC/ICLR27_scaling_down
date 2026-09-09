# P1-v2 new-source frozen prospective: pythia-1b@step112000 (P-new)

Predictions committed before measurement (see git log). Signed = pred − actual. MAE over 3 capabilities.

## Protocol A (full dev)
| regime | power | A2 | A1 | cont | strength_only | median_curve | zero |
|---|---|---|---|---|---|---|---|
| prune interp_0.9-0.6 | 0.191 | 0.112 | 0.301 | 0.175 | 0.365 | 0.173 | 0.400 |
| prune extrap_0.55 | 1.398 | 1.235 | 1.838 | 1.356 | 0.311 | 1.311 | 2.237 |

| bits | full | noD0 | per_bit_mean | per_bit_median | zero |
|---|---|---|---|---|---|
| int8 | 0.006 | 0.004 | 0.004 | 0.002 | 0.001 |
| int6 | 0.022 | 0.029 | 0.034 | 0.005 | 0.005 |
| int5 (interp-rule) | 0.151 | 0.138 | 0.251 | 0.035 | 0.025 |
| int4 | 0.106 | 0.272 | 0.443 | 0.103 | 0.222 |
| int3 | 0.676 | 3.805 | 0.802 | 1.231 | 5.777 |
| ≥4-bit | 0.071 | 0.111 | 0.183 | 0.036 | 0.063 |

per-point prune (cap, d, actual, power, A2, cont):
- math d=0.9: act +0.004 | power +0.010 | A2 +0.016 | cont -0.092
- math d=0.8: act +0.059 | power +0.112 | A2 +0.099 | cont +0.077
- math d=0.75: act +0.131 | power +0.242 | A2 +0.233 | cont +0.259
- math d=0.7: act +0.314 | power +0.455 | A2 +0.367 | cont +0.506
- math d=0.65: act +0.759 | power +0.774 | A2 +0.814 | cont +0.818
- math d=0.6: act +1.666 | power +1.227 | A2 +1.260 | cont +1.195
- math d=0.55: act +2.902 | power +1.842 | A2 +1.707 | cont +1.637
- code d=0.9: act +0.005 | power +0.020 | A2 +0.006 | cont -0.121
- code d=0.8: act +0.031 | power +0.147 | A2 +0.069 | cont +0.026
- code d=0.75: act +0.082 | power +0.281 | A2 +0.163 | cont +0.200
- code d=0.7: act +0.187 | power +0.477 | A2 +0.256 | cont +0.440
- code d=0.65: act +0.475 | power +0.747 | A2 +0.731 | cont +0.747
- code d=0.6: act +1.306 | power +1.100 | A2 +1.207 | cont +1.122
- code d=0.55: act +2.869 | power +1.547 | A2 +1.682 | cont +1.563
- qa d=0.9: act +0.009 | power -0.009 | A2 -0.071 | cont -0.165
- qa d=0.8: act -0.165 | power -0.074 | A2 -0.261 | cont -0.306
- qa d=0.75: act -0.400 | power -0.145 | A2 -0.420 | cont -0.368
- qa d=0.7: act -0.651 | power -0.253 | A2 -0.579 | cont -0.425
- qa d=0.65: act -0.784 | power -0.405 | A2 -0.514 | cont -0.475
- qa d=0.6: act -0.179 | power -0.609 | A2 -0.450 | cont -0.519
- qa d=0.55: act +0.940 | power -0.872 | A2 -0.385 | cont -0.558

## Protocol B (dev step<=64k only)
| regime | power | A2 | A1 | cont | strength_only | median_curve | zero |
|---|---|---|---|---|---|---|---|
| prune interp_0.9-0.6 | 0.223 | 0.177 | 0.355 | 0.235 | 0.219 | 0.195 | 0.400 |
| prune extrap_0.55 | 0.957 | 0.995 | 1.685 | 1.127 | 1.333 | 1.447 | 2.237 |

| bits | full | noD0 | per_bit_mean | per_bit_median | zero |
|---|---|---|---|---|---|
| int8 | 0.002 | 0.004 | 0.002 | 0.003 | 0.001 |
| int6 | 0.012 | 0.003 | 0.004 | 0.002 | 0.005 |
| int5 (interp-rule) | 0.046 | 0.019 | 0.027 | 0.025 | 0.025 |
| int4 | 0.094 | 0.142 | 0.120 | 0.125 | 0.222 |
| int3 | 1.120 | 1.981 | 3.106 | 3.472 | 5.777 |
| ≥4-bit | 0.038 | 0.042 | 0.038 | 0.039 | 0.063 |

per-point prune (cap, d, actual, power, A2, cont):
- math d=0.9: act +0.004 | power +0.004 | A2 +0.008 | cont -0.106
- math d=0.8: act +0.059 | power +0.067 | A2 +0.054 | cont +0.030
- math d=0.75: act +0.131 | power +0.163 | A2 +0.158 | cont +0.188
- math d=0.7: act +0.314 | power +0.338 | A2 +0.261 | cont +0.406
- math d=0.65: act +0.759 | power +0.626 | A2 +0.677 | cont +0.685
- math d=0.6: act +1.666 | power +1.067 | A2 +1.093 | cont +1.024
- math d=0.55: act +2.902 | power +1.710 | A2 +1.508 | cont +1.423
- code d=0.9: act +0.005 | power +0.005 | A2 +0.006 | cont -0.139
- code d=0.8: act +0.031 | power +0.072 | A2 +0.063 | cont +0.000
- code d=0.75: act +0.082 | power +0.177 | A2 +0.126 | cont +0.174
- code d=0.7: act +0.187 | power +0.367 | A2 +0.189 | cont +0.417
- code d=0.65: act +0.475 | power +0.679 | A2 +0.702 | cont +0.730
- code d=0.6: act +1.306 | power +1.158 | A2 +1.215 | cont +1.112
- code d=0.55: act +2.869 | power +1.856 | A2 +1.728 | cont +1.563
- qa d=0.9: act +0.009 | power +0.001 | A2 -0.022 | cont -0.132
- qa d=0.8: act -0.165 | power +0.011 | A2 -0.109 | cont -0.146
- qa d=0.75: act -0.400 | power +0.026 | A2 -0.154 | cont -0.107
- qa d=0.7: act -0.651 | power +0.054 | A2 -0.199 | cont -0.039
- qa d=0.65: act -0.784 | power +0.101 | A2 +0.030 | cont +0.058
- qa d=0.6: act -0.179 | power +0.172 | A2 +0.259 | cont +0.186
- qa d=0.55: act +0.940 | power +0.275 | A2 +0.488 | cont +0.343

