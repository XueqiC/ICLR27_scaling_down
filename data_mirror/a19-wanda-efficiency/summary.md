# A18 second-family confirmation: mean absolute error over the test states (nats per token)

| Capability | Predictor | Probes | New items |
|---|---|---:|---:|
| math | power_full | 0.023 | 0.025 |
| math | quadratic_full | 0.037 | 0.038 |
| math | strength_full | 0.031 | 0.033 |
| math | median_full | 0.029 | 0.028 |
| math | per_density_full | 0.025 | 0.027 |
| math | power_reduced | 0.023 | 0.025 |
| math | quadratic_reduced | 0.026 | 0.028 |
| math | strength_reduced | 0.031 | 0.033 |
| math | median_reduced | 0.029 | 0.029 |
| math | per_density_reduced | 0.024 | 0.026 |
| math | pythia_zero_calibration | 0.186 | 0.186 |
| math | zero_change | 0.043 | 0.043 |
| code | power_full | 0.029 | 0.034 |
| code | quadratic_full | 0.049 | 0.049 |
| code | strength_full | 0.044 | 0.049 |
| code | median_full | 0.045 | 0.049 |
| code | per_density_full | 0.027 | 0.031 |
| code | power_reduced | 0.026 | 0.030 |
| code | quadratic_reduced | 0.048 | 0.052 |
| code | strength_reduced | 0.046 | 0.050 |
| code | median_reduced | 0.046 | 0.050 |
| code | per_density_reduced | 0.032 | 0.034 |
| code | pythia_zero_calibration | 0.199 | 0.192 |
| code | zero_change | 0.058 | 0.065 |
| qa | power_full | 0.051 | 0.039 |
| qa | quadratic_full | 0.036 | 0.045 |
| qa | strength_full | 0.048 | 0.038 |
| qa | median_full | 0.034 | 0.048 |
| qa | per_density_full | 0.061 | 0.048 |
| qa | power_reduced | 0.058 | 0.048 |
| qa | quadratic_reduced | 0.038 | 0.042 |
| qa | strength_reduced | 0.039 | 0.039 |
| qa | median_reduced | 0.033 | 0.049 |
| qa | per_density_reduced | 0.060 | 0.049 |
| qa | pythia_zero_calibration | 0.265 | 0.244 |
| qa | zero_change | 0.036 | 0.039 |

Item bootstrap (1000 resamples, seed 2027; states fixed): 95 percent interval of the mean absolute error and of the difference power_reduced minus per_density_full

| Capability | Items | power_reduced | per_density_full | difference [95%] |
|---|---|---|---|---|
| math | probes | [0.021, 0.025] | [0.023, 0.026] | -0.001 [-0.002, -0.001] |
| math | new | [0.022, 0.027] | [0.024, 0.029] | -0.002 [-0.002, -0.001] |
| code | probes | [0.023, 0.031] | [0.024, 0.031] | -0.001 [-0.002, -0.000] |
| code | new | [0.026, 0.035] | [0.026, 0.035] | -0.001 [-0.001, -0.000] |
| qa | probes | [0.042, 0.090] | [0.046, 0.092] | -0.003 [-0.009, +0.002] |
| qa | new | [0.037, 0.073] | [0.040, 0.074] | -0.002 [-0.009, +0.003] |
