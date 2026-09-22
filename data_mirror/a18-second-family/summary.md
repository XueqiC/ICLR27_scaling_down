# A18 second-family confirmation: mean absolute error over the test states (nats per token)

| Capability | Predictor | Probes | New items |
|---|---|---:|---:|
| math | power_full | 0.012 | 0.010 |
| math | quadratic_full | 0.045 | 0.038 |
| math | strength_full | 0.019 | 0.018 |
| math | median_full | 0.022 | 0.021 |
| math | per_density_full | 0.019 | 0.013 |
| math | power_reduced | 0.017 | 0.012 |
| math | quadratic_reduced | 0.026 | 0.030 |
| math | strength_reduced | 0.018 | 0.019 |
| math | median_reduced | 0.026 | 0.030 |
| math | per_density_reduced | 0.025 | 0.030 |
| math | pythia_zero_calibration | 0.724 | 0.717 |
| code | power_full | 0.028 | 0.037 |
| code | quadratic_full | 0.062 | 0.064 |
| code | strength_full | 0.031 | 0.037 |
| code | median_full | 0.036 | 0.037 |
| code | per_density_full | 0.027 | 0.036 |
| code | power_reduced | 0.020 | 0.029 |
| code | quadratic_reduced | 0.045 | 0.056 |
| code | strength_reduced | 0.029 | 0.037 |
| code | median_reduced | 0.039 | 0.051 |
| code | per_density_reduced | 0.035 | 0.046 |
| code | pythia_zero_calibration | 1.883 | 1.864 |
| qa | power_full | 0.055 | 0.051 |
| qa | quadratic_full | 0.057 | 0.053 |
| qa | strength_full | 0.069 | 0.060 |
| qa | median_full | 0.069 | 0.053 |
| qa | per_density_full | 0.060 | 0.036 |
| qa | power_reduced | 0.062 | 0.045 |
| qa | quadratic_reduced | 0.061 | 0.041 |
| qa | strength_reduced | 0.076 | 0.050 |
| qa | median_reduced | 0.074 | 0.048 |
| qa | per_density_reduced | 0.062 | 0.043 |
| qa | pythia_zero_calibration | 0.382 | 0.412 |

Item bootstrap (1000 resamples, seed 2027; states fixed): 95 percent interval of the mean absolute error and of the difference power_reduced minus per_density_full

| Capability | Items | power_reduced | per_density_full | difference [95%] |
|---|---|---|---|---|
| math | probes | [0.013, 0.022] | [0.013, 0.026] | -0.002 [-0.005, +0.001] |
| math | new | [0.010, 0.016] | [0.010, 0.019] | -0.001 [-0.004, +0.003] |
| code | probes | [0.017, 0.026] | [0.023, 0.034] | -0.007 [-0.010, -0.004] |
| code | new | [0.023, 0.038] | [0.030, 0.044] | -0.006 [-0.008, -0.001] |
| qa | probes | [0.045, 0.101] | [0.039, 0.095] | +0.006 [+0.000, +0.014] |
| qa | new | [0.029, 0.096] | [0.027, 0.090] | +0.005 [-0.008, +0.015] |
