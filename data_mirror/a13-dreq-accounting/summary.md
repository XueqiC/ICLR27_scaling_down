# A13: independent-data accounting

student x nominal supervised training budget x readout x tau; 144 requests

Boundary recommendations are primary; delta (loss-based) recommendations are a separate companion analysis. Never pool the two rules into a 288-request denominator.

Fresh 2Wiki, MuSiQue, TriviaQA (72 requests); original 2Wiki probe separately (24). QA including original probe (96) is also reported.

A12 delta_mae and common_support_delta_mae are loss-prediction errors in nats per supervised evaluation token, not errors in required data.

D_U is the independent supervised-token pool size, not processed tokens, examples, or D_U_seen. Training budget T counts supervised tokens, including reuse.

Natural-log distance to the recorded measured interval: max(0, ln(L/R), ln(R/U)), omitting unbounded sides; R is recommended_D_U. Zero means compatibility with the interval closure, not an exact minimum or a successful recommendation.

For lower/upper bounds, distance is only a lower bound on distance to an unknown crossing; even zero does not identify the requirement. One-sided log widths are unbounded (JSON null plus width_status).

Uses the closed recorded envelope, including endpoints. A failing lower endpoint is still geometrically in its closure. Empirical constraint success is reported independently.

Replicate-ambiguous and non-monotone cells retain null intervals/distances; they stay in all coverage and solved denominators.

For finite crossed intervals: U-L supervised tokens and ln(U/L). No midpoint or interpolated minimum is estimated.

Recommendations / all requests in group, for each method and recommendation rule.

Recommendations meeting the constraint in all three measured replicates / recommendations. A12 recommendation_meets_loss_constraint is used directly.

Successful recommendations / the identical intersection of requests where all three methods recommend; the intersection is independent of success and crossing status.

Successful recommendations / all requests in group; abstentions and unresolved measurements do not leave the denominator.

conservative_D_U is a comparator recommendation, not the measured crossing. Its recorded unverified-ceiling and success flags are preserved; savings are credited only when both recommendations pass.

tau = 0.25 nats is the level A10 reported and the plan's primary level; all three registered budgets are plotted, with 200k the plan's primary budget.

The upper tier uses training seeds on the same full pool; the lower tiers use pool seeds. These are not exchangeable pool replicates.

For scored crossings only, bars run from the largest failing measured pool to the smallest passing measured pool. A12's slightly wider seed-envelope interval (smallest failing-tier pool to largest passing-tier pool) remains the accounting target and is retained alongside the plotted endpoints.

Crosses show seed-mixed tested tiers, or all tested tiers when means are non-monotone. Their y positions are measured pool sizes, not boundary estimates. A full-pool failure in a non-monotone cell is not promoted to a lower bound.

Measured request statuses: {"non_monotone": 16, "upper_bound": 16, "replicate_ambiguous": 30, "lower_bound": 40, "crossed": 42}.

Recommendation rule: boundary

| Readout group | Metric | boundary_loglinear | fixed_reuse | student_isotonic |
|---|---|---:|---:|---:|
| QA distributions | coverage | 54/72 (75.0%) | 57/72 (79.2%) | 42/72 (58.3%) |
| QA distributions | conditional | 30/54 (55.6%) | 25/57 (43.9%) | 33/42 (78.6%) |
| QA distributions | common_reliability | 30/42 (71.4%) | 25/42 (59.5%) | 33/42 (78.6%) |
| QA distributions | solved | 30/72 (41.7%) | 25/72 (34.7%) | 33/72 (45.8%) |
| math/code | coverage | 17/48 (35.4%) | 16/48 (33.3%) | 25/48 (52.1%) |
| math/code | conditional | 11/17 (64.7%) | 13/16 (81.2%) | 22/25 (88.0%) |
| math/code | common_reliability | 10/14 (71.4%) | 11/14 (78.6%) | 12/14 (85.7%) |
| math/code | solved | 11/48 (22.9%) | 13/48 (27.1%) | 22/48 (45.8%) |
| 2Wiki original probe | coverage | 24/24 (100.0%) | 24/24 (100.0%) | 19/24 (79.2%) |
| 2Wiki original probe | conditional | 18/24 (75.0%) | 19/24 (79.2%) | 16/19 (84.2%) |
| 2Wiki original probe | common_reliability | 14/19 (73.7%) | 15/19 (78.9%) | 16/19 (84.2%) |
| 2Wiki original probe | solved | 18/24 (75.0%) | 19/24 (79.2%) | 16/24 (66.7%) |
| QA including original probe | coverage | 78/96 (81.2%) | 81/96 (84.4%) | 61/96 (63.5%) |
| QA including original probe | conditional | 48/78 (61.5%) | 44/81 (54.3%) | 49/61 (80.3%) |
| QA including original probe | common_reliability | 44/61 (72.1%) | 40/61 (65.6%) | 49/61 (80.3%) |
| QA including original probe | solved | 48/96 (50.0%) | 44/96 (45.8%) | 49/96 (51.0%) |
| all requests | coverage | 95/144 (66.0%) | 97/144 (67.4%) | 86/144 (59.7%) |
| all requests | conditional | 59/95 (62.1%) | 57/97 (58.8%) | 71/86 (82.6%) |
| all requests | common_reliability | 54/75 (72.0%) | 51/75 (68.0%) | 61/75 (81.3%) |
| all requests | solved | 59/144 (41.0%) | 57/144 (39.6%) | 71/144 (49.3%) |

Recommendation rule: delta

| Readout group | Metric | boundary_loglinear | fixed_reuse | student_isotonic |
|---|---|---:|---:|---:|
| QA distributions | coverage | 59/72 (81.9%) | 26/72 (36.1%) | 42/72 (58.3%) |
| QA distributions | conditional | 34/59 (57.6%) | 20/26 (76.9%) | 17/42 (40.5%) |
| QA distributions | common_reliability | 20/22 (90.9%) | 20/22 (90.9%) | 9/22 (40.9%) |
| QA distributions | solved | 34/72 (47.2%) | 20/72 (27.8%) | 17/72 (23.6%) |
| math/code | coverage | 29/48 (60.4%) | 34/48 (70.8%) | 25/48 (52.1%) |
| math/code | conditional | 14/29 (48.3%) | 20/34 (58.8%) | 15/25 (60.0%) |
| math/code | common_reliability | 12/24 (50.0%) | 16/24 (66.7%) | 14/24 (58.3%) |
| math/code | solved | 14/48 (29.2%) | 20/48 (41.7%) | 15/48 (31.2%) |
| 2Wiki original probe | coverage | 24/24 (100.0%) | 24/24 (100.0%) | 19/24 (79.2%) |
| 2Wiki original probe | conditional | 19/24 (79.2%) | 4/24 (16.7%) | 9/19 (47.4%) |
| 2Wiki original probe | common_reliability | 15/19 (78.9%) | 4/19 (21.1%) | 9/19 (47.4%) |
| 2Wiki original probe | solved | 19/24 (79.2%) | 4/24 (16.7%) | 9/24 (37.5%) |
| QA including original probe | coverage | 83/96 (86.5%) | 50/96 (52.1%) | 61/96 (63.5%) |
| QA including original probe | conditional | 53/83 (63.9%) | 24/50 (48.0%) | 26/61 (42.6%) |
| QA including original probe | common_reliability | 35/41 (85.4%) | 24/41 (58.5%) | 18/41 (43.9%) |
| QA including original probe | solved | 53/96 (55.2%) | 24/96 (25.0%) | 26/96 (27.1%) |
| all requests | coverage | 112/144 (77.8%) | 84/144 (58.3%) | 86/144 (59.7%) |
| all requests | conditional | 67/112 (59.8%) | 44/84 (52.4%) | 41/86 (47.7%) |
| all requests | common_reliability | 47/65 (72.3%) | 40/65 (61.5%) | 32/65 (49.2%) |
| all requests | solved | 67/144 (46.5%) | 44/144 (30.6%) | 41/144 (28.5%) |

Median natural-log recommendation distance (sample counts in parentheses).
Crossed = distance to a finite bracket; censored = lower bound on error.

| Rule | Readout group | Method | Crossed | Common crossed | Censored lower bound | All identified | Finite log width | Finite width (tokens) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| boundary | QA distributions | boundary_loglinear | 0.0000 (18) | 0.0000 (18) | 0.0000 (15) | 0.0000 (33) | 1.4417 (18) | 29851.0000 (18) |
| boundary | QA distributions | fixed_reuse | 0.0000 (18) | 0.0000 (18) | 0.0000 (17) | 0.0000 (35) | 1.4417 (18) | 29851.0000 (18) |
| boundary | QA distributions | student_isotonic | 0.0000 (18) | 0.0000 (18) | 1.2617 (7) | 0.0000 (25) | 1.4417 (18) | 29851.0000 (18) |
| boundary | math/code | boundary_loglinear | 0.0000 (9) | 0.0000 (8) | 1.4323 (3) | 0.0000 (12) | 1.3645 (12) | 26691.5000 (12) |
| boundary | math/code | fixed_reuse | 0.0000 (9) | 0.0000 (8) | 0.0000 (2) | 0.0000 (11) | 1.3645 (12) | 26691.5000 (12) |
| boundary | math/code | student_isotonic | 0.0000 (10) | 0.0000 (8) | 2.8562 (7) | 0.0000 (17) | 1.3645 (12) | 26691.5000 (12) |
| boundary | 2Wiki original probe | boundary_loglinear | 0.0000 (12) | 0.0000 (8) | 1.2617 (4) | 0.0000 (16) | 1.4417 (12) | 29851.0000 (12) |
| boundary | 2Wiki original probe | fixed_reuse | 0.0000 (12) | 0.0000 (8) | 0.6308 (4) | 0.0000 (16) | 1.4417 (12) | 29851.0000 (12) |
| boundary | 2Wiki original probe | student_isotonic | 0.0000 (8) | 0.0000 (8) | 2.8561 (4) | 0.0000 (12) | 1.4417 (12) | 29851.0000 (12) |
| boundary | QA including original probe | boundary_loglinear | 0.0000 (30) | 0.0000 (26) | 1.2617 (19) | 0.0000 (49) | 1.4417 (30) | 29851.0000 (30) |
| boundary | QA including original probe | fixed_reuse | 0.0000 (30) | 0.0000 (26) | 0.0000 (21) | 0.0000 (51) | 1.4417 (30) | 29851.0000 (30) |
| boundary | QA including original probe | student_isotonic | 0.0000 (26) | 0.0000 (26) | 1.2617 (11) | 0.0000 (37) | 1.4417 (30) | 29851.0000 (30) |
| boundary | all requests | boundary_loglinear | 0.0000 (39) | 0.0000 (34) | 1.2617 (22) | 0.0000 (61) | 1.4417 (42) | 29851.0000 (42) |
| boundary | all requests | fixed_reuse | 0.0000 (39) | 0.0000 (34) | 0.0000 (23) | 0.0000 (62) | 1.4417 (42) | 29851.0000 (42) |
| boundary | all requests | student_isotonic | 0.0000 (36) | 0.0000 (34) | 1.4281 (18) | 0.0000 (54) | 1.4417 (42) | 29851.0000 (42) |
| delta | QA distributions | boundary_loglinear | 0.0000 (18) | 0.0000 (10) | 0.0000 (16) | 0.0000 (34) | 1.4417 (18) | 29851.0000 (18) |
| delta | QA distributions | fixed_reuse | 1.4091 (10) | 1.4091 (10) | 1.2617 (7) | 1.4091 (17) | 1.4417 (18) | 29851.0000 (18) |
| delta | QA distributions | student_isotonic | 0.0000 (18) | 0.0000 (10) | 0.0000 (7) | 0.0000 (25) | 1.4417 (18) | 29851.0000 (18) |
| delta | math/code | boundary_loglinear | 0.0000 (11) | 0.0000 (9) | 0.0000 (8) | 0.0000 (19) | 1.3645 (12) | 26691.5000 (12) |
| delta | math/code | fixed_reuse | 0.0000 (12) | 0.0000 (9) | 0.0000 (12) | 0.0000 (24) | 1.3645 (12) | 26691.5000 (12) |
| delta | math/code | student_isotonic | 0.0000 (10) | 0.0000 (9) | 0.0000 (7) | 0.0000 (17) | 1.3645 (12) | 26691.5000 (12) |
| delta | 2Wiki original probe | boundary_loglinear | 0.0000 (12) | 0.0000 (8) | 0.0000 (4) | 0.0000 (16) | 1.4417 (12) | 29851.0000 (12) |
| delta | 2Wiki original probe | fixed_reuse | 0.0000 (12) | 0.7129 (8) | 0.0000 (4) | 0.0000 (16) | 1.4417 (12) | 29851.0000 (12) |
| delta | 2Wiki original probe | student_isotonic | 0.0000 (8) | 0.0000 (8) | 0.0000 (4) | 0.0000 (12) | 1.4417 (12) | 29851.0000 (12) |
| delta | QA including original probe | boundary_loglinear | 0.0000 (30) | 0.0000 (18) | 0.0000 (20) | 0.0000 (50) | 1.4417 (30) | 29851.0000 (30) |
| delta | QA including original probe | fixed_reuse | 0.7046 (22) | 1.4091 (18) | 0.0000 (11) | 0.0000 (33) | 1.4417 (30) | 29851.0000 (30) |
| delta | QA including original probe | student_isotonic | 0.0000 (26) | 0.0000 (18) | 0.0000 (11) | 0.0000 (37) | 1.4417 (30) | 29851.0000 (30) |
| delta | all requests | boundary_loglinear | 0.0000 (41) | 0.0000 (27) | 0.0000 (28) | 0.0000 (69) | 1.4417 (42) | 29851.0000 (42) |
| delta | all requests | fixed_reuse | 0.0000 (34) | 0.0000 (27) | 0.0000 (23) | 0.0000 (57) | 1.4417 (42) | 29851.0000 (42) |
| delta | all requests | student_isotonic | 0.0000 (36) | 0.0000 (27) | 0.0000 (18) | 0.0000 (54) | 1.4417 (42) | 29851.0000 (42) |

Interpretation: coverage, conditional reliability, and solved fractions answer different questions. No method is uniformly best across the readout groups and recommendation rules. Zero interval distances inside broad brackets do not establish accurate point predictions or a universal data-requirement law. C84's pooled recommendation counts combine two rules per request; use the separated same-denominator comparisons above.
