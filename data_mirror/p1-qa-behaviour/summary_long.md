# P1b long-generation readout (96 new tokens, gold-blind first-sentence extraction)

| Reference | Model | Roles | EM 32-block | EM 96-span | EM 96-line | F1 32 | F1 96-span | Truncated@96 | Span words | dEM vs quant-only [95%] | dEM vs initial |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma3-1b | quant:b3_g32 | quant_oracle | 0.010 | 0.026 | 0.026 | 0.056 | 0.069 | 0.872 | 22.3 | -0.016 [-0.039, +0.008] | -- |
| gemma3-1b | quant:channel_b4 | quant-only | 0.036 | 0.042 | 0.039 | 0.054 | 0.057 | 0.883 | 44.9 | -- | -- |
| gemma3-1b | dense:source | dense | 0.211 | 0.195 | 0.195 | 0.282 | 0.270 | 0.073 | 6.2 | +0.154 [+0.117, +0.193] | -- |
| gemma3-1b | quant:b4_g128 | quant_oracle | 0.188 | 0.180 | 0.177 | 0.237 | 0.238 | 0.169 | 6.4 | +0.138 [+0.102, +0.177] | -- |
| gemma3-1b | pristine:student | pristine_student | 0.128 | 0.133 | 0.135 | 0.195 | 0.200 | 0.346 | 8.6 | +0.091 [+0.060, +0.125] | -- |
| gemma3-1b | distill:new_s3 | oracle,priority,rule | 0.000 | 0.115 | 0.000 | 0.081 | 0.171 | 0.990 | 4.3 | +0.073 [+0.036, +0.107] | -0.018 [-0.055, +0.018] |
| gemma3-4b | quant:channel_b4 | quant-only | 0.086 | 0.081 | 0.083 | 0.180 | 0.167 | 0.266 | 18.4 | -- | -- |
| gemma3-4b | quant:b3_g32 | quant_oracle | 0.135 | 0.141 | 0.130 | 0.176 | 0.178 | 0.240 | 8.2 | +0.060 [+0.021, +0.099] | -- |
| gemma3-4b | dense:source | dense | 0.276 | 0.234 | 0.232 | 0.360 | 0.313 | 0.036 | 5.3 | +0.154 [+0.112, +0.198] | -- |
| gemma3-4b | pristine:student | pristine_student | 0.211 | 0.190 | 0.190 | 0.282 | 0.269 | 0.081 | 6.3 | +0.109 [+0.073, +0.148] | -- |
| gemma3-4b | distill:new_s3 | oracle,priority,rule | 0.003 | 0.172 | 0.089 | 0.109 | 0.249 | 0.974 | 3.7 | +0.091 [+0.052, +0.130] | -0.018 [-0.057, +0.018] |
| pythia-1.4b--step120000 | quant:channel_b4 | quant-only,quant_oracle | 0.167 | 0.146 | 0.143 | 0.208 | 0.189 | 0.128 | 8.0 | -- | -- |
| pythia-1.4b--step120000 | pristine:student | pristine_student | 0.193 | 0.182 | 0.182 | 0.227 | 0.219 | 0.219 | 7.9 | +0.036 [-0.003, +0.078] | -- |
| pythia-1.4b--step120000 | dense:source | dense | 0.198 | 0.182 | 0.182 | 0.233 | 0.226 | 0.089 | 6.8 | +0.036 [+0.005, +0.068] | -- |
| pythia-1.4b--step120000 | distill:new_s3 | oracle,priority,rule | 0.065 | 0.091 | 0.070 | 0.105 | 0.138 | 0.628 | 4.7 | -0.055 [-0.094, -0.016] | -0.091 [-0.133, -0.049] |
| pythia-410m--step120000 | quant:channel_b4 | oracle,priority,quant-only,quant_oracle,rule | 0.044 | 0.049 | 0.049 | 0.060 | 0.069 | 0.424 | 12.2 | -- | -- |
| pythia-410m--step120000 | quant:b4_g32 | quant_oracle | 0.094 | 0.102 | 0.102 | 0.123 | 0.122 | 0.375 | 15.5 | +0.052 [+0.023, +0.081] | -- |
| pythia-410m--step120000 | quant:channel_b6 | quant_oracle | 0.174 | 0.143 | 0.143 | 0.213 | 0.183 | 0.229 | 9.3 | +0.094 [+0.060, +0.130] | -- |
| pythia-410m--step120000 | dense:source | dense | 0.193 | 0.182 | 0.182 | 0.227 | 0.219 | 0.219 | 7.9 | +0.133 [+0.094, +0.172] | -- |
| pythia-410m--step120000 | prune:d0.8 | oracle | 0.109 | 0.117 | 0.117 | 0.138 | 0.140 | 0.315 | 12.8 | +0.068 [+0.039, +0.099] | -- |
| pythia-410m--step120000 | pristine:student | pristine_student | 0.070 | 0.078 | 0.078 | 0.107 | 0.110 | 0.508 | 25.4 | +0.029 [-0.003, +0.060] | -- |
| pythia-410m--step120000 | distill:new_s3 | oracle,priority,rule | 0.029 | 0.089 | 0.089 | 0.074 | 0.126 | 0.729 | 5.1 | +0.039 [+0.005, +0.073] | +0.010 [-0.023, +0.042] |
