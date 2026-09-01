# V2: item-level damage spectrum, prune ladders (2026-08-25)

Models per sparsity (1 free param each): uniform Rasch shift vs deletion mixture. ΔLL = LL_mix − LL_shift; positive favors quanta-style discrete deletion.

## src32b-hotpotqa (items with locked difficulty: 50/50)
| sparsity | acc | ΔLL (mix−shift) | verdict | dead frac π̂ | Gini(drop) |
|---|---|---|---|---|---|
| 0.100 | 0.84 | +1.4 | tie | 0.00 | 0.98 |
| 0.200 | 0.60 | -0.9 | tie | 0.26 | 0.70 |
| 0.300 | 0.44 | +1.2 | tie | 0.42 | 0.60 |
| 0.400 | 0.26 | +6.8 | mix | 0.66 | 0.42 |
| 0.750 | 0.00 | -0.0 | tie | 1.00 | 0.16 |
| 0.875 | 0.00 | -0.0 | tie | 1.00 | 0.16 |
| 0.947 | 0.00 | -0.0 | tie | 1.00 | 0.16 |
| 0.981 | 0.00 | -0.0 | tie | 1.00 | 0.16 |

## src32b-humaneval (items with locked difficulty: 31/31)
| sparsity | acc | ΔLL (mix−shift) | verdict | dead frac π̂ | Gini(drop) |
|---|---|---|---|---|---|
| 0.100 | 0.84 | -1.6 | tie | 0.00 | 0.97 |
| 0.200 | 0.77 | -1.9 | tie | 0.00 | 0.94 |
| 0.300 | 0.81 | -1.3 | tie | 0.00 | 0.94 |
| 0.400 | 0.71 | +1.7 | tie | 0.06 | 0.87 |
| 0.750 | 0.00 | -0.0 | tie | 1.00 | 0.16 |
| 0.875 | 0.00 | -0.0 | tie | 1.00 | 0.16 |
| 0.947 | 0.00 | -0.0 | tie | 1.00 | 0.16 |
| 0.981 | 0.00 | -0.0 | tie | 1.00 | 0.16 |

## src8b-gsm8k (items with locked difficulty: 50/50)
| sparsity | acc | ΔLL (mix−shift) | verdict | dead frac π̂ | Gini(drop) |
|---|---|---|---|---|---|
| 0.100 | 0.82 | -4.3 | shift | 0.00 | 0.92 |
| 0.200 | 0.82 | -3.4 | shift | 0.00 | 0.94 |
| 0.300 | 0.84 | -8.0 | shift | 0.01 | 0.90 |
| 0.400 | 0.70 | -9.8 | shift | 0.14 | 0.80 |
| 0.500 | 0.20 | -6.6 | shift | 0.75 | 0.32 |

## src8b-hotpotqa (items with locked difficulty: 50/50)
| sparsity | acc | ΔLL (mix−shift) | verdict | dead frac π̂ | Gini(drop) |
|---|---|---|---|---|---|
| 0.100 | 0.80 | -5.4 | shift | 0.00 | 0.96 |
| 0.200 | 0.76 | -8.9 | shift | 0.00 | 0.92 |
| 0.300 | 0.60 | -2.7 | shift | 0.21 | 0.76 |
| 0.400 | 0.24 | -1.3 | tie | 0.66 | 0.48 |
| 0.500 | 0.10 | +3.6 | mix | 0.86 | 0.36 |

## src8b-humaneval (items with locked difficulty: 31/31)
| sparsity | acc | ΔLL (mix−shift) | verdict | dead frac π̂ | Gini(drop) |
|---|---|---|---|---|---|
| 0.100 | 0.77 | +6.5 | mix | 0.00 | - |
| 0.200 | 0.74 | -1.7 | tie | 0.00 | 0.94 |
| 0.300 | 0.74 | -1.7 | tie | 0.00 | 0.94 |
| 0.400 | 0.55 | -6.0 | shift | 0.25 | 0.74 |
| 0.500 | 0.16 | -3.9 | shift | 0.78 | 0.39 |
