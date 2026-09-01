# V9c capability-coordinate ablation — google/gemma-3-1b-pt

Top fraction per capability before exclusivity: 0.0500%. Coordinates are selected from the capability-average residual log-Fisher and removed if either other capability also selects them.

## Selected coordinates

| capability | top set | exclusive set | exclusive share of scope |
|---|---:|---:|---:|
| math | 499,875 | 497,408 | 0.049753% |
| code | 499,875 | 497,408 | 0.049753% |
| qa | 499,875 | 499,875 | 0.050000% |

## Dense completion loss

| math | code | qa |
|---:|---:|---:|
| 1.266257 | 0.848574 | 5.817808 |

## Ablated completion loss

| ablated capability | math | code | qa |
|---|---:|---:|---:|
| math | 5.898384 | 1.280201 | 5.864789 |
| code | 1.688645 | 7.989212 | 5.818999 |
| qa | 1.272591 | 0.848665 | 8.106513 |

## Damage relative to dense

| ablated capability | math | code | qa |
|---|---:|---:|---:|
| math | +4.632127 | +0.431627 | +0.046981 |
| code | +0.422388 | +7.140638 | +0.001191 |
| qa | +0.006333 | +0.000091 | +2.288705 |

## Prediction check

The gold-standard prediction is row-wise diagonal dominance: deleting a capability-exclusive region should damage that capability more than either of the others.

- math: yes
- code: yes
- qa: yes
