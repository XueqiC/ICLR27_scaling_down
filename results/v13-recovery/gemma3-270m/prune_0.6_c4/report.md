# V13 recovery report — google/gemma-3-270m

Compression: `prune_0.6_c4`. Recovery checkpoints continue from one LoRA run; reported token counts are completion tokens for traces and causal document-target tokens for C4.

## Anchors

| capability | dense loss | damaged loss | initial damage |
|---|---:|---:|---:|
| math | 1.315995 | 8.299364 | +6.983369 |
| code | 1.190296 | 9.790523 | +8.600227 |
| qa | 5.275741 | 12.939243 | +7.663502 |

## Recovery ladder

| requested tokens | tokens seen | capability | loss | delta vs damaged | delta vs dense |
|---:|---:|---|---:|---:|---:|
| 500,000 | 501,220 | math | 3.472193 | -4.827171 | +2.156198 |
| 500,000 | 501,220 | code | 4.248339 | -5.542183 | +3.058044 |
| 500,000 | 501,220 | qa | 5.409861 | -7.529382 | +0.134120 |
| 2,000,000 | 2,005,156 | math | 2.701849 | -5.597515 | +1.385854 |
| 2,000,000 | 2,005,156 | code | 3.393601 | -6.396922 | +2.203305 |
| 2,000,000 | 2,005,156 | qa | 5.100286 | -7.838957 | -0.175454 |
| 8,000,000 | 8,004,309 | math | 2.560994 | -5.738369 | +1.244999 |
| 8,000,000 | 8,004,309 | code | 3.185673 | -6.604849 | +1.995378 |
| 8,000,000 | 8,004,309 | qa | 5.109998 | -7.829246 | -0.165743 |

## Recovery-law fits

Fit: `L_c(D_R) = L_c^inf + B'_c D_R^(-beta'_c)`.

| capability | points | status | L_inf | B | beta | R^2 | L_inf - L_dense |
|---|---:|---|---:|---:|---:|---:|---:|
| math | 3 | ok | 2.52935 | 9.0681e+06 | 1.2251 | 1 | 1.21336 |
| code | 3 | ok | 3.11857 | 727456 | 1.01909 | 1 | 1.92828 |
| qa | 3 | ok | 5.105 | 9.64368e+27 | 5 | 0.999192 | -0.170745 |
