# P2 Wanda panel (retrospective)

| Model | Density | Capability | Magnitude dL | Wanda dL |
|---|---|---|---:|---:|
| qwen3-4b | 0.9 | math | +0.005 | +0.006 |
| qwen3-4b | 0.9 | code | +0.018 | +0.007 |
| qwen3-4b | 0.9 | qa | -0.002 | -0.002 |
| qwen3-4b | 0.8 | math | +0.005 | +0.020 |
| qwen3-4b | 0.8 | code | -0.010 | +0.000 |
| qwen3-4b | 0.8 | qa | -0.293 | +0.039 |
| qwen3-4b | 0.7 | math | -0.003 | +0.047 |
| qwen3-4b | 0.7 | code | -0.088 | +0.042 |
| qwen3-4b | 0.7 | qa | -0.793 | +0.083 |
| qwen3-4b | 0.6 | math | +0.145 | +0.105 |
| qwen3-4b | 0.6 | code | -0.001 | +0.132 |
| qwen3-4b | 0.6 | qa | -1.760 | -0.009 |
| qwen3-4b | 0.5 | math | +1.308 | +0.189 |
| qwen3-4b | 0.5 | code | +0.904 | +0.244 |
| qwen3-4b | 0.5 | qa | -1.476 | -0.412 |
| qwen3-4b | 0.4 | math | +4.998 | +0.612 |
| qwen3-4b | 0.4 | code | +5.827 | +0.673 |
| qwen3-4b | 0.4 | qa | +1.369 | -1.217 |
| qwen3-4b | 0.3 | math | +12.343 | +1.697 |
| qwen3-4b | 0.3 | code | +11.948 | +2.510 |
| qwen3-4b | 0.3 | qa | +6.841 | -1.877 |
| gemma3-4b | 0.9 | math | +0.020 | +0.002 |
| gemma3-4b | 0.9 | code | +0.002 | -0.000 |
| gemma3-4b | 0.9 | qa | +0.053 | -0.004 |
| gemma3-4b | 0.8 | math | +0.153 | +0.012 |
| gemma3-4b | 0.8 | code | +0.114 | +0.014 |
| gemma3-4b | 0.8 | qa | +0.222 | -0.043 |
| gemma3-4b | 0.7 | math | +0.831 | +0.045 |
| gemma3-4b | 0.7 | code | +1.474 | +0.058 |
| gemma3-4b | 0.7 | qa | +0.241 | -0.091 |
| gemma3-4b | 0.6 | math | +2.214 | +0.165 |
| gemma3-4b | 0.6 | code | +3.159 | +0.161 |
| gemma3-4b | 0.6 | qa | +0.065 | -0.203 |
| gemma3-4b | 0.5 | math | +7.002 | +0.479 |
| gemma3-4b | 0.5 | code | +11.063 | +0.497 |
| gemma3-4b | 0.5 | qa | +2.837 | -0.393 |
| gemma3-4b | 0.4 | math | +19.774 | +1.501 |
| gemma3-4b | 0.4 | code | +10.497 | +1.915 |
| gemma3-4b | 0.4 | qa | +18.923 | -0.450 |
| gemma3-4b | 0.3 | math | +20.611 | +2.103 |
| gemma3-4b | 0.3 | code | +16.172 | +3.027 |
| gemma3-4b | 0.3 | qa | +16.631 | -0.700 |

| Capability | alpha (magnitude, pooled) | Model | scale-refit MAE | free-fit MAE (alpha) | K1 MAE |
|---|---:|---|---:|---:|---:|
| math | 4.00 | qwen3-4b | 0.148 | 0.183 (2.72) | 4.544 |
| math | 4.00 | gemma3-4b | 0.076 | 0.142 (3.74) | 0.764 |
| code | 3.98 | qwen3-4b | 0.232 | 0.260 (3.77) | 4.648 |
| code | 3.98 | gemma3-4b | 0.293 | 0.173 (4.41) | 0.380 |
| qa | 2.79 | qwen3-4b | 0.012 | 0.000 (1.84) | 0.039 |
| qa | 2.79 | gemma3-4b | too few positive cells (0) | | |
