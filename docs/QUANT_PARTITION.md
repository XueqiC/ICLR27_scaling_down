# Quantization partition error table (Closeout Package C, retrospective)

Frozen config-indicator {N0,L0,D0} predictor vs simple baselines on new @step96000 sources, per bit.
int3 is real measured data. v38-measured (160M/1.4B@96k) and v40-first-measured (410M@96k) are NOT
merged into one prospective. Signed bias = pred - actual.

| source (ver) | bit | actual dL | pred | signed bias | |err| cand | |err| median-base | improve |
|---|---|---|---|---|---|---|---|
| 160m@96000 (v38) | 8 | -0.002 | +0.007 | +0.009 | 0.009 | 0.002 | -0.007 |
| 160m@96000 (v38) | 6 | +0.030 | +0.049 | +0.019 | 0.019 | 0.023 | +0.004 |
| 160m@96000 (v38) | 4 | +0.691 | +0.843 | +0.152 | 0.152 | 0.561 | +0.409 |
| 160m@96000 (v38) | 3 | +6.003 | +8.725 | +2.721 | 2.721 | 2.142 | -0.580 |
| 160m@96000 (v38) | 8 | -0.013 | +0.017 | +0.030 | 0.030 | 0.013 | -0.016 |
| 160m@96000 (v38) | 6 | +0.029 | +0.083 | +0.054 | 0.054 | 0.025 | -0.028 |
| 160m@96000 (v38) | 4 | +0.714 | +1.287 | +0.573 | 0.573 | 0.550 | -0.023 |
| 160m@96000 (v38) | 3 | +6.512 | +11.284 | +4.773 | 4.773 | 0.647 | -4.125 |
| 160m@96000 (v38) | 8 | +0.026 | +0.010 | -0.016 | 0.016 | 0.030 | +0.015 |
| 160m@96000 (v38) | 6 | -0.049 | +0.079 | +0.128 | 0.128 | 0.053 | -0.075 |
| 160m@96000 (v38) | 4 | +0.865 | +0.833 | -0.032 | 0.032 | 0.927 | +0.895 |
| 160m@96000 (v38) | 3 | +6.672 | +8.424 | +1.752 | 1.752 | 2.761 | +1.009 |
| 1.4b@96000 (v38) | 8 | +0.000 | -0.001 | -0.001 | 0.001 | 0.001 | -0.000 |
| 1.4b@96000 (v38) | 6 | +0.005 | +0.002 | -0.003 | 0.003 | 0.002 | -0.001 |
| 1.4b@96000 (v38) | 4 | +0.162 | +0.150 | -0.013 | 0.013 | 0.032 | +0.019 |
| 1.4b@96000 (v38) | 3 | +6.631 | +4.997 | -1.634 | 1.634 | 2.769 | +1.136 |
| 1.4b@96000 (v38) | 8 | -0.000 | +0.002 | +0.003 | 0.003 | 0.001 | -0.002 |
| 1.4b@96000 (v38) | 6 | +0.009 | +0.005 | -0.004 | 0.004 | 0.006 | +0.002 |
| 1.4b@96000 (v38) | 4 | +0.117 | +0.301 | +0.184 | 0.184 | 0.047 | -0.137 |
| 1.4b@96000 (v38) | 3 | +7.361 | +6.737 | -0.625 | 0.625 | 1.497 | +0.873 |
| 1.4b@96000 (v38) | 8 | -0.013 | +0.009 | +0.022 | 0.022 | 0.008 | -0.013 |
| 1.4b@96000 (v38) | 6 | +0.019 | +0.006 | -0.013 | 0.013 | 0.015 | +0.001 |
| 1.4b@96000 (v38) | 4 | -0.162 | +0.188 | +0.350 | 0.350 | 0.100 | -0.250 |
| 1.4b@96000 (v38) | 3 | +5.749 | +5.861 | +0.112 | 0.112 | 1.838 | +1.725 |
| 410m@96000 (v40) | 8 | -0.000 | -0.000 | +0.000 | 0.000 | 0.001 | +0.001 |
| 410m@96000 (v40) | 6 | +0.014 | +0.005 | -0.009 | 0.009 | 0.007 | -0.002 |
| 410m@96000 (v40) | 4 | +0.261 | +0.205 | -0.056 | 0.056 | 0.131 | +0.074 |
| 410m@96000 (v40) | 3 | +4.753 | +4.861 | +0.108 | 0.108 | 0.891 | +0.783 |
| 410m@96000 (v40) | 8 | -0.000 | +0.001 | +0.001 | 0.001 | 0.001 | -0.001 |
| 410m@96000 (v40) | 6 | +0.015 | +0.000 | -0.015 | 0.015 | 0.012 | -0.003 |
| 410m@96000 (v40) | 4 | +0.353 | +0.280 | -0.073 | 0.073 | 0.189 | +0.116 |
| 410m@96000 (v40) | 3 | +5.480 | +5.972 | +0.492 | 0.492 | 0.384 | -0.108 |
| 410m@96000 (v40) | 8 | +0.030 | +0.006 | -0.024 | 0.024 | 0.035 | +0.011 |
| 410m@96000 (v40) | 6 | +0.010 | +0.076 | +0.065 | 0.065 | 0.006 | -0.060 |
| 410m@96000 (v40) | 4 | -0.091 | +0.811 | +0.902 | 0.902 | 0.029 | -0.873 |
| 410m@96000 (v40) | 3 | +4.494 | +8.769 | +4.275 | 4.275 | 0.583 | -3.692 |

## Partition summary (MAE, candidate vs per-bit median baseline)
| cap | region | cand MAE | median MAE | improvement |
|---|---|---|---|---|
| math | ge4 | 0.029 | 0.084 | +0.055 |
| math | int3 | 1.488 | 1.934 | +0.446 |
| code | ge4 | 0.104 | 0.094 | -0.010 |
| code | int3 | 1.963 | 0.843 | -1.120 |
| qa | ge4 | 0.173 | 0.134 | -0.039 |
| qa | int3 | 2.046 | 1.727 | -0.319 |

## Verdict (Package C)
- Validation range is the DISCRETE integer bit-widths {8,6,4,3}; there is no integer point between 3 and 4,
  so this is a per-configuration validation, not a claim that quantization admits no law.
- In the >=4-bit region the responses are near-zero and candidate/median errors are both tiny: at this error
  scale no incremental predictive value over a simple per-bit baseline is demonstrated (not a proof of none).
- int3 responses are REAL measured collapse, not an artifact; the candidate's advantage there is unstable
  across sources (helps on some, misses on others) -> magnitude transfer at int3 is not established.
- The fixed 4^{-b} shape is inconsistent with the measured decay under this RTN quantizer definition; a
  learned exponential fits better (prior C8), reported within this tested range only.
- No collapse-RISK prediction law is claimed (no independent risk model validated here).
- v38 (160M/1.4B@96k) and v40 (410M@96k) are separate source measurements, not one merged prospective.
