# V10 quantization report — Qwen/Qwen3-4B

All language weight matrices use symmetric round-to-nearest fake quantization with one scale per output channel. Losses use the odd-indexed measurement half of each V6 probe set.

## Dense losses

| capability | dense loss |
|---|---:|
| math | 0.910534 |
| code | 1.545796 |
| qa | 6.734505 |

## Quantization loss damage

Damage is `Delta L_c(b) = L_c(b) - L_c(dense)`.

| bits | Delta L_math | Delta L_code | Delta L_qa |
|---:|---:|---:|---:|
| 8 | +0.001251 | +0.001011 | +0.014713 |
| 6 | -0.001839 | +0.020514 | -0.082986 |
| 4 | +0.132374 | +0.191346 | +0.393724 |
| 3 | +10.730893 | +11.851777 | +5.994047 |

## Exponential fit check

Least-squares fit of `ln(Delta L_c)` against bit-width, using only positive damage. The proposed `4^(-b)` law predicts slope `-ln(4) = -1.3863` and fitted base `4`.

| capability | positive points | q_c | slope | fitted base | R^2 |
|---|---:|---:|---:|---:|---:|
| math | 3 | 412.9567 | -1.6269 | 5.0879 | 0.9031 |
| code | 4 | 693.1254 | -1.7204 | 5.5870 | 0.9350 |
| qa | 3 | 77.2056 | -1.0933 | 2.9842 | 0.9240 |
