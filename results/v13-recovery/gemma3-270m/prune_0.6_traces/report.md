# V13 recovery report — google/gemma-3-270m

Compression: `prune_0.6_traces`. Recovery checkpoints continue from one LoRA run; reported token counts are completion tokens for traces and causal document-target tokens for C4.

## Anchors

| capability | dense loss | damaged loss | initial damage |
|---|---:|---:|---:|
| math | 1.315995 | 8.299364 | +6.983369 |
| code | 1.190296 | 9.790523 | +8.600227 |
| qa | 5.275741 | 12.939243 | +7.663502 |

## Recovery ladder

| requested tokens | tokens seen | capability | loss | delta vs damaged | delta vs dense |
|---:|---:|---|---:|---:|---:|
| 500,000 | 500,005 | math | 2.757823 | -5.541540 | +1.441828 |
| 500,000 | 500,005 | code | 2.645261 | -7.145261 | +1.454966 |
| 500,000 | 500,005 | qa | 4.951133 | -7.988110 | -0.324608 |
| 2,000,000 | 2,000,289 | math | 2.587532 | -5.711831 | +1.271537 |
| 2,000,000 | 2,000,289 | code | 2.536094 | -7.254429 | +1.345798 |
| 2,000,000 | 2,000,289 | qa | 6.437251 | -6.501992 | +1.161510 |
| 8,000,000 | 8,000,749 | math | 2.680861 | -5.618503 | +1.364866 |
| 8,000,000 | 8,000,749 | code | 2.862489 | -6.928034 | +1.672193 |
| 8,000,000 | 8,000,749 | qa | 9.751992 | -3.187251 | +4.476251 |

## Recovery-law fits

Fit: `L_c(D_R) = L_c^inf + B'_c D_R^(-beta'_c)`.

| capability | points | status | L_inf | B | beta | R^2 | L_inf - L_dense |
|---|---:|---|---:|---:|---:|---:|---:|
| math | 3 | ok | 2.63416 | 3.86328e+27 | 5 | 0.699785 | 1.31816 |
| code | 3 | ok | 2.68128 | 3.17898e+08 | 5 | -2.22045e-16 | 1.49099 |
| qa | 3 | ok | 7.04679 | 1.95899e-11 | 0.000100103 | -1.9984e-15 | 1.77105 |
