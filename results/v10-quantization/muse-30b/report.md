# V10 quantization report — meta-models/Muse-Glimmer-30B

All language weight matrices use symmetric round-to-nearest fake quantization with one scale per output channel. Losses use the odd-indexed measurement half of each V6 probe set.

## Dense losses

| capability | dense loss |
|---|---:|
| math | 0.692746 |
| code | 0.969909 |
| qa | 5.506674 |

## Quantization loss damage

Damage is `Delta L_c(b) = L_c(b) - L_c(dense)`.

| bits | Delta L_math | Delta L_code | Delta L_qa |
|---:|---:|---:|---:|
| 8 | -0.000763 | -0.002370 | +0.003890 |
| 6 | +0.009405 | +0.011679 | -0.101784 |
| 4 | +1.583282 | +2.059357 | -0.106781 |
| 3 | +11.138065 | +11.182880 | +8.362253 |

## Exponential fit check

Least-squares fit of `ln(Delta L_c)` against bit-width, using only positive damage. The proposed `4^(-b)` law predicts slope `-ln(4) = -1.3863` and fitted base `4`.

| capability | positive points | q_c | slope | fitted base | R^2 |
|---|---:|---:|---:|---:|---:|
| math | 3 | 17147.3137 | -2.3881 | 10.8929 | 0.9960 |
| code | 3 | 15708.4027 | -2.3307 | 10.2851 | 0.9911 |
| qa | 2 | 835.0614 | -1.5346 | 4.6394 | 1.0000 |
