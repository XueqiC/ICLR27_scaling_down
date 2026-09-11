# V65 distillation paired intervals

Candidate: frozen `joint+src`; baselines: frozen `constant` and zero change.

Each estimate averages points within (student, role, capability). Paired difference is mean(|baseline error| - |candidate error|), in nats. Relative improvement is (baseline MAE - candidate MAE) / baseline MAE; the tables express this as a percentage. Positive values favor the candidate.

95% percentile intervals use 5,000 trajectory-cluster bootstrap resamples with replacement, seed 0 (NumPy PCG64). The cluster is student × pool; every budget stays together. Each resample draws the original number of trajectories and averages all resulting points. The same draws are used across both baselines and all capabilities within each student/role. Relative improvement is recomputed from each resample's MAEs. Predictions are fixed; no fitting or form selection is performed.

There are only three test trajectories per student and four held-out-student development trajectories; the intervals describe resampling of these observed trajectories and have limited resolution.

## Test pools

| Student | Cap. | Points / trajectories | Candidate MAE | Constant MAE | Diff vs constant [95% CI] | Relative % [95% CI] | Zero MAE | Diff vs zero [95% CI] | Relative % [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 270M | math | 12 / 3 | 0.0453 | 0.0605 | 0.0152 [0.0020, 0.0271] | 25.2 [3.9, 39.9] | 0.0915 | 0.0462 [0.0433, 0.0489] | 50.5 [49.2, 51.5] |
| 270M | code | 12 / 3 | 0.0364 | 0.0471 | 0.0107 [0.0062, 0.0172] | 22.7 [14.0, 35.0] | 0.1383 | 0.1019 [0.0968, 0.1091] | 73.7 [72.4, 75.2] |
| 270M | qa | 12 / 3 | 0.6055 | 0.8621 | 0.2565 [0.1727, 0.3760] | 29.8 [21.5, 39.3] | 1.0052 | 0.3997 [0.3159, 0.5192] | 39.8 [33.4, 47.2] |
| 1B | math | 12 / 3 | 0.0503 | 0.0571 | 0.0068 [-0.0008, 0.0145] | 11.8 [-1.6, 23.4] | 0.0949 | 0.0446 [0.0396, 0.0520] | 47.0 [42.8, 50.8] |
| 1B | code | 12 / 3 | 0.0289 | 0.0688 | 0.0399 [0.0317, 0.0443] | 58.0 [52.5, 64.0] | 0.1090 | 0.0801 [0.0680, 0.0888] | 73.5 [67.3, 77.0] |
| 1B | qa | 12 / 3 | 0.4439 | 1.0044 | 0.5604 [0.5550, 0.5657] | 55.8 [53.9, 58.0] | 1.1475 | 0.7036 [0.6982, 0.7089] | 61.3 [59.5, 63.4] |
| 4B | math | 12 / 3 | 0.0888 | 0.0205 | -0.0683 [-0.0699, -0.0655] | -333.8 [-486.0, -265.6] | 0.1532 | 0.0644 [0.0434, 0.0799] | 42.0 [31.1, 48.8] |
| 4B | code | 12 / 3 | 0.0504 | 0.0383 | -0.0121 [-0.0292, -0.0029] | -31.5 [-71.8, -7.9] | 0.1777 | 0.1273 [0.1168, 0.1330] | 71.6 [65.4, 76.9] |
| 4B | qa | 12 / 3 | 1.1408 | 1.0239 | -0.1169 [-0.1193, -0.1154] | -11.4 [-11.8, -11.1] | 1.1670 | 0.0262 [0.0238, 0.0277] | 2.2 [2.1, 2.4] |

## Development pools, held-out 4B student

| Student | Cap. | Points / trajectories | Candidate MAE | Constant MAE | Diff vs constant [95% CI] | Relative % [95% CI] | Zero MAE | Diff vs zero [95% CI] | Relative % [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4B | math | 16 / 4 | 0.1065 | 0.1556 | 0.0491 [-0.0681, 0.1663] | 31.6 [-573.4, 55.6] | 0.2938 | 0.1874 [0.0692, 0.3056] | 63.8 [46.4, 69.7] |
| 4B | code | 16 / 4 | 0.0482 | 0.0977 | 0.0495 [-0.0186, 0.1176] | 50.7 [-58.9, 72.0] | 0.2303 | 0.1821 [0.1028, 0.2614] | 79.1 [67.2, 87.0] |
| 4B | qa | 16 / 4 | 0.9580 | 1.7904 | 0.8324 [0.1249, 1.5398] | 46.5 [11.7, 62.4] | 1.8620 | 0.9039 [0.2681, 1.5398] | 48.5 [22.1, 63.1] |

Input: `results/v50-p2v2/compare_test.json` (156 rows).
SHA-256: `4c07abf0f940e95992a3f46d79b6d9a3ef9524ba55adaefa824b0cdec8a601aa`.
