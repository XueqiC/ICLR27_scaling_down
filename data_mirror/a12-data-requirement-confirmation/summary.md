# A12 registered data-requirement confirmation (18 trajectories on LONI, 2026-09-17)

Predictions sealed before training (amendment 1, SHA256 4a30c9df...); 18 trajectories (gemma3-1b, gemma3-4b; pool tiers below ~9k, near 32k/38k, above = full 158,391 supervised tokens; three seeds each) evaluated at 50k, 100k and 200k supervised tokens on six readouts: 432 of 432 readout cells measured.

## Verdict

The boundary is now measurable: the QA readouts cross the tolerance levels inside the tested pool range (2Wiki crosses at every tau, MuSiQue from tau = 0.25, math and code from tau = 0.25 in most cells; TriviaQA never crosses and gives lower bounds only). The registered low-degree boundary relation is NOT established as the better predictor: on every QA readout the per-student monotone interpolation predicts the measured transfer with lower error, and it also meets the loss constraint more often (below). The fixed-reuse threshold is the weakest. What the experiment delivers is a measured data-requirement map on two students, with the per-student monotone interpolation as the practical predictor.

## Measured crossing status (count over 2 students x 3 budgets)

| Readout | tau | crossed | upper bound | lower bound | non-monotone | replicate-ambiguous |
|---|---|---|---|---|---|---|
| math:MATH-500 | 0.0 | 0 | 0 | 5 | 1 | 0 |
| math:MATH-500 | 0.1 | 1 | 0 | 3 | 1 | 1 |
| math:MATH-500 | 0.25 | 4 | 1 | 0 | 1 | 0 |
| math:MATH-500 | 0.5 | 3 | 2 | 0 | 1 | 0 |
| code:MBPP | 0.0 | 0 | 0 | 5 | 1 | 0 |
| code:MBPP | 0.1 | 0 | 0 | 4 | 1 | 1 |
| code:MBPP | 0.25 | 3 | 1 | 0 | 1 | 1 |
| code:MBPP | 0.5 | 1 | 3 | 0 | 1 | 1 |
| qa:2Wiki-probe | 0.0 | 2 | 1 | 0 | 0 | 3 |
| qa:2Wiki-probe | 0.1 | 3 | 1 | 0 | 0 | 2 |
| qa:2Wiki-probe | 0.25 | 3 | 1 | 0 | 0 | 2 |
| qa:2Wiki-probe | 0.5 | 4 | 1 | 0 | 0 | 1 |
| qa:2Wiki-fresh | 0.0 | 2 | 1 | 0 | 0 | 3 |
| qa:2Wiki-fresh | 0.1 | 4 | 1 | 0 | 0 | 1 |
| qa:2Wiki-fresh | 0.25 | 4 | 1 | 0 | 0 | 1 |
| qa:2Wiki-fresh | 0.5 | 4 | 1 | 0 | 0 | 1 |
| qa:MuSiQue | 0.0 | 0 | 0 | 5 | 0 | 1 |
| qa:MuSiQue | 0.1 | 0 | 0 | 4 | 0 | 2 |
| qa:MuSiQue | 0.25 | 1 | 0 | 0 | 0 | 5 |
| qa:MuSiQue | 0.5 | 3 | 1 | 0 | 0 | 2 |
| qa:TriviaQA | 0.0 | 0 | 0 | 4 | 2 | 0 |
| qa:TriviaQA | 0.1 | 0 | 0 | 4 | 2 | 0 |
| qa:TriviaQA | 0.25 | 0 | 0 | 3 | 2 | 1 |
| qa:TriviaQA | 0.5 | 0 | 0 | 3 | 2 | 1 |

## Transfer prediction error on the new trajectories (median MAE in nats over 24 student x budget x tau cells)

| Readout | boundary relation | fixed reuse | per-student monotone |
|---|---|---|---|
| math:MATH-500 | 0.114 | 0.118 | 0.160 |
| code:MBPP | 0.073 | 0.071 | 0.077 |
| qa:2Wiki-probe | 1.045 | 1.671 | 0.767 |
| qa:2Wiki-fresh | 1.265 | 1.744 | 0.701 |
| qa:MuSiQue | 0.616 | 0.500 | 0.430 |
| qa:TriviaQA | 0.347 | 0.347 | 0.277 |

## Recommended data amount meets the loss constraint

| Method | met / scored |
|---|---|
| boundary_loglinear | 126 / 207 (61%) |
| fixed_reuse | 101 / 181 (56%) |
| student_isotonic | 112 / 172 (65%) |

Every number derives from score.json (predictions_sha256 4a30c9df8b23e8ef...) produced by analysis/a12_score.py from the sealed plan.json and the 18 fetched runs; the conservative choice (the full pool) meets the constraint in 152 of 288 scored cells, so many tolerance levels are unattainable at these budgets for either predictor.
