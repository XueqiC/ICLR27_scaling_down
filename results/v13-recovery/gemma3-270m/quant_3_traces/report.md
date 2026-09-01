# V13 recovery report — google/gemma-3-270m

Compression: `quant_3_traces`. Recovery checkpoints continue from one LoRA run; reported token counts are completion tokens for traces and causal document-target tokens for C4.

## Anchors

| capability | dense loss | damaged loss | initial damage |
|---|---:|---:|---:|
| math | 1.315995 | 27.461130 | +26.145136 |
| code | 1.190296 | 29.130204 | +27.939908 |
| qa | 5.275741 | 31.202689 | +25.926948 |

## Recovery ladder

| requested tokens | tokens seen | capability | loss | delta vs damaged | delta vs dense |
|---:|---:|---|---:|---:|---:|
| 500,000 | 500,005 | math | 6.311790 | -21.149341 | +4.995795 |
| 500,000 | 500,005 | code | 6.528344 | -22.601860 | +5.338048 |
| 500,000 | 500,005 | qa | 8.771414 | -22.431275 | +3.495674 |
| 2,000,000 | 2,000,289 | math | 4.136687 | -23.324443 | +2.820693 |
| 2,000,000 | 2,000,289 | code | 4.500221 | -24.629982 | +3.309926 |
| 2,000,000 | 2,000,289 | qa | 5.938621 | -25.264069 | +0.662880 |
| 8,000,000 | 8,000,749 | math | 3.968025 | -23.493105 | +2.652031 |
| 8,000,000 | 8,000,749 | code | 4.104739 | -25.025465 | +2.914443 |
| 8,000,000 | 8,000,749 | qa | 8.565737 | -22.636952 | +3.289996 |

## Recovery-law fits

Fit: `L_c(D_R) = L_c^inf + B'_c D_R^(-beta'_c)`.

| capability | points | status | L_inf | B | beta | R^2 | L_inf - L_dense |
|---|---:|---|---:|---:|---:|---:|---:|
| math | 3 | ok | 3.95384 | 7.63492e+10 | 1.84424 | 1 | 2.63785 |
| code | 3 | ok | 4.00891 | 1.32078e+07 | 1.17908 | 1 | 2.81861 |
| qa | 3 | ok | 7.25208 | 4.74416e+28 | 5 | 0.307606 | 1.97634 |
