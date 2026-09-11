# V64: selection with explicit feasibility

Retrospective leave-one-state-out validation of fixed per-method law families; predictions receive K0 plus dense anchors, never target compressed losses. No target fitting, measurement execution, randomness, timestamps or GPU dependencies.

## Endpoint definition (unchanged from v60)

Every candidate endpoint is the absolute measured/predicted loss of the deployed model. For pruning/quantization, predicted loss is source dense (the measurement arm's own anchor) + predicted response; measured loss is that configuration's recorded absolute loss. For KD, predicted loss is student dense + predicted student post-training change, and measured loss is student dense + measured student post-training change (= post_training). The student change is never added to the source dense. This is v60's endpoint definition, kept unchanged here. The multi objective then takes max_c(deployed loss_c - source dense_c).

## States and measured configurations

Counts are configurations, each with math/code/QA losses; dense is counted once per state.

| State | Matrix N0 | Prune | Channel RTN | Grouped RTN | Smaller student | Dense |
|---|---:|---:|---:|---:|---:|---:|
| pythia-160m@step16000 | 84,934,656 | 4 | 4 | 9 | 0 | 1 |
| pythia-160m@step64000 | 84,934,656 | 4 | 4 | 0 | 0 | 1 |
| pythia-160m@step96000 | 84,934,656 | 4 | 4 | 0 | 0 | 1 |
| pythia-160m@step143000 | 84,934,656 | 6 | 4 | 9 | 0 | 1 |
| pythia-410m@step16000 | 301,989,888 | 4 | 4 | 9 | 1 | 1 |
| pythia-410m@step64000 | 301,989,888 | 4 | 4 | 0 | 1 | 1 |
| pythia-410m@step96000 | 301,989,888 | 6 | 4 | 0 | 0 | 1 |
| pythia-410m@step143000 | 301,989,888 | 4 | 4 | 9 | 1 | 1 |
| pythia-1b@step32000 | 805,306,368 | 7 | 5 | 0 | 0 | 1 |
| pythia-1b@step96000 | 805,306,368 | 2 | 2 | 3 | 0 | 1 |
| pythia-1b@step112000 | 805,306,368 | 7 | 5 | 0 | 0 | 1 |
| pythia-1.4b@step16000 | 1,207,959,552 | 4 | 4 | 9 | 2 | 1 |
| pythia-1.4b@step64000 | 1,207,959,552 | 6 | 4 | 0 | 2 | 1 |
| pythia-1.4b@step96000 | 1,207,959,552 | 4 | 4 | 0 | 0 | 1 |
| pythia-1.4b@step143000 | 1,207,959,552 | 4 | 4 | 9 | 2 | 1 |
| pythia-6.9b@step32000 | 6,442,450,944 | 7 | 5 | 0 | 0 | 1 |
| pythia-6.9b@step112000 | 6,442,450,944 | 7 | 5 | 0 | 0 | 1 |
| **Total** | — | 84 | 70 | 57 | 9 | 17 |

Exact measured config IDs, resource ratios, dense anchors, predictions and actual losses are in summary.json.

## Candidate coverage and minimum available storage

Counts above exclude the dense source from pruning/quantization/KD. KD counts are 0, 1 or 2 measured smaller same-stage students. N/A means no measured candidate. Quant includes both RTN variants.

| State | Min r: prune | Min r: channel | Min r: group | Min r: quant (combined) | Min r: KD | Min r: dense |
|---|---:|---:|---:|---:|---:|---:|
| pythia-160m@step16000 | 0.6000000 | 0.1875000 | 0.1914062 | 0.1875000 | N/A | 1.0000000 |
| pythia-160m@step64000 | 0.6000000 | 0.1875000 | N/A | 0.1875000 | N/A | 1.0000000 |
| pythia-160m@step96000 | 0.6000000 | 0.1875000 | N/A | 0.1875000 | N/A | 1.0000000 |
| pythia-160m@step143000 | 0.5500000 | 0.1875000 | 0.1914062 | 0.1875000 | N/A | 1.0000000 |
| pythia-410m@step16000 | 0.6000000 | 0.1875000 | 0.1914062 | 0.1875000 | 0.2812500 | 1.0000000 |
| pythia-410m@step64000 | 0.6000000 | 0.1875000 | N/A | 0.1875000 | 0.2812500 | 1.0000000 |
| pythia-410m@step96000 | 0.5500000 | 0.1875000 | N/A | 0.1875000 | N/A | 1.0000000 |
| pythia-410m@step143000 | 0.6000000 | 0.1875000 | 0.1914062 | 0.1875000 | 0.2812500 | 1.0000000 |
| pythia-1b@step32000 | 0.5500000 | 0.1875000 | N/A | 0.1875000 | N/A | 1.0000000 |
| pythia-1b@step96000 | 0.5500000 | 0.1875000 | 0.1953125 | 0.1875000 | N/A | 1.0000000 |
| pythia-1b@step112000 | 0.5500000 | 0.1875000 | N/A | 0.1875000 | N/A | 1.0000000 |
| pythia-1.4b@step16000 | 0.6000000 | 0.1875000 | 0.1914062 | 0.1875000 | 0.0703125 | 1.0000000 |
| pythia-1.4b@step64000 | 0.5500000 | 0.1875000 | N/A | 0.1875000 | 0.0703125 | 1.0000000 |
| pythia-1.4b@step96000 | 0.6000000 | 0.1875000 | N/A | 0.1875000 | N/A | 1.0000000 |
| pythia-1.4b@step143000 | 0.6000000 | 0.1875000 | 0.1914062 | 0.1875000 | 0.0703125 | 1.0000000 |
| pythia-6.9b@step32000 | 0.5500000 | 0.1875000 | N/A | 0.1875000 | N/A | 1.0000000 |
| pythia-6.9b@step112000 | 0.5500000 | 0.1875000 | N/A | 0.1875000 | N/A | 1.0000000 |

## Rules and interpretation

- State universe: all non-2.8b Pythia states with both pruning and per-channel quantization files, ordered by transformer-matrix N0 then stage. All measured configs on these states are candidates.
- Storage: pruning r=d (unstructured pruning realizes this only with a sparse format; no index overhead); channel RTN r=b/16; grouped RTN r=(b+16/g)/16; smaller same-stage student r=matrix_n0(S)/matrix_n0(source); dense r=1, zero change. These ratios exclude embeddings/head/vectors, channel-scale overhead, unmerged adapter overhead, runtime memory and execution cost.
- Pruning imports v53 fit_all/predict_all unchanged, selects the prescribed power form only: (beta.phi)*((1-d)/0.3)^gamma. Ridge=1e-3 including intercept; gamma=0.5..6.0 by 0.05; training SSE selects gamma. Training-only population standardization pools response rows and capabilities.
- Channel RTN reuses v36 covariates/design/predict, the v38/v49 path: per-capability training-only population standardization, configuration indicators crossed with four phi terms, OLS. Extend its bit vocabulary to [8,6,5,4,3]; underidentified 5-bit uses the same least-squares minimum-norm solve. This is the requested per-bit regression, not v49's 4/6 interpolation rule.
- Grouped RTN imports v55 fit_all/predict_all unchanged and uses low_order_2d only: 20 ridge coefficients for phi crossed with [1,u,v,u*v,u^2]. Only its 24 original dev cells (six states, b3/b5 x g64/g256) can train; remove every cell of the held-out state. Remaining measured bit/granularity/joint cells are scored only. Predict grouped RTN only on measured source configs.
- Distillation imports v39 _fit/_predict unchanged: per-capability OLS on the STUDENT's standardized log N0, dense L0c, log D0. Delta=raw post_training-dense. A source fold excludes the source and ALL its smaller same-stage candidate students from distillation training. This extra purge is stricter than source-only LOSO and prevents fitting any candidate's target outcome. Prediction anchors on the student's measured dense, never the source's dense.
- Feasible means measured r<=r_max on 0.20,0.25,...,1.00 (1e-12 floating tolerance). MAP minimizes predicted absolute loss; ORACLE minimizes recorded actual loss. Both may choose dense only at r_max=1. Fixed policies minimize predicted loss within prune/quant/distill. Any policy with no feasible candidate returns INFEASIBLE with null config, loss, regret and agreement, including at r_max=1 if its named method has no candidates. There is no source fallback. Dense belongs only to the unchanged MAP/ORACLE/CHEAPEST candidate sets. Feasible regret is nonnegative (1e-12 tolerance).
- CHEAPEST minimizes r. All exact ties use method order prune,quant,distill,dense; remaining ties use smaller r then lexical config ID. Quant-only includes channel and grouped RTN.
- No-clear-winner heuristic: compare the predicted winners of the best two distinct feasible methods. Flag strictly when their gap is smaller than max(their respective winning-config laws' LOSO MAEs). Dense law MAE is zero. Candidate set includes the best method plus each other method satisfying that pairwise gap/MAE rule against the best. With one feasible method, the flag is false. The heuristic flag belongs to the shared state/budget candidate space. With no feasible method it is null; its summary denominator includes only evaluable cells.
- Feasibility coverage is the share of all 289 (state,budget) cells with a feasible choice. Own-feasible regret and oracle-method agreement average only each policy's feasible cells. Common-feasible regret and agreement average the intersection where all six compared policies (including ORACLE and CHEAPEST) are feasible, against the same unrestricted budget-feasible ORACLE. Empty subsets return null (N/A in tables), never zero. Every included (state,budget) has equal weight. Law MAEs weight each evaluated (held-out source, measured config) once, including shared student alternatives; the selected configuration is never tuned using these errors.
- Multi-capability objective is max_c(predicted Lc - source dense L0c), using all three capabilities on the same configuration. Its oracle minimizes max_c(actual Lc - source dense L0c); regret is the difference of these actual worst-degradation objectives. Its no-clear-winner heuristic threshold uses each law's MAE on this same scalar max objective, recomputed from held-out predictions. This is a minimax requirement, not an absolute pass/fail threshold or an average of capabilities.

## Law errors from this run

Errors average configurations once per held-out source, never repeated budget cells. Distillation counts source/student alternatives; student reuse is disclosed above.

| Law | Configs | Math MAE | Code MAE | QA MAE | Multi objective MAE |
|---|---:|---:|---:|---:|---:|
| prune_power | 84 | 0.328282 | 0.354821 | 0.798090 | 0.428189 |
| quant_channel | 70 | 0.599269 | 0.611925 | 1.082234 | 0.769966 |
| quant_group | 57 | 0.645116 | 1.382582 | 3.169445 | 1.954296 |
| distill_linear | 9 | 0.021588 | 0.092944 | 0.294701 | 0.082220 |
| dense | 17 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Math

| Policy | Feasible / all | Coverage | Regret: own | Regret: common | Agreement: own | Agreement: common |
|---|---:|---:|---:|---:|---:|---:|
| MAP | 289/289 | 100.00% | 0.193625 | 0.201754 | 95.85% | 94.55% |
| ORACLE | 289/289 | 100.00% | 0.000000 | 0.000000 | 100.00% | 100.00% |
| prune-only | 161/289 | 55.71% | 0.587531 | 0.289635 | 0.00% | 0.00% |
| quant-only | 289/289 | 100.00% | 0.208928 | 0.201804 | 96.54% | 92.73% |
| distill-only | 96/289 | 33.22% | 0.296752 | 0.322484 | 1.04% | 0.00% |
| CHEAPEST | 289/289 | 100.00% | 4.583879 | 2.093130 | 79.93% | 43.64% |

Common-feasible subset: 55/289 cells, intersecting all six policies (MAP, ORACLE, prune-only, quant-only, distill-only, CHEAPEST). Regret is in nats; agreement uses exactly the same own/common subsets. Empty subsets report N/A.

No-clear-winner heuristic: 134/289 evaluable cells; oracle method in heuristic candidate set: 99.65% of evaluable cells; 100.00% within heuristic no-clear-winner cells.

## Code

| Policy | Feasible / all | Coverage | Regret: own | Regret: common | Agreement: own | Agreement: common |
|---|---:|---:|---:|---:|---:|---:|
| MAP | 289/289 | 100.00% | 0.174335 | 0.111188 | 94.81% | 96.36% |
| ORACLE | 289/289 | 100.00% | 0.000000 | 0.000000 | 100.00% | 100.00% |
| prune-only | 161/289 | 55.71% | 0.621848 | 0.309753 | 4.97% | 0.00% |
| quant-only | 289/289 | 100.00% | 0.194287 | 0.111188 | 94.12% | 96.36% |
| distill-only | 96/289 | 33.22% | 0.336551 | 0.362588 | 3.12% | 0.00% |
| CHEAPEST | 289/289 | 100.00% | 5.174546 | 2.250188 | 78.89% | 47.27% |

Common-feasible subset: 55/289 cells, intersecting all six policies (MAP, ORACLE, prune-only, quant-only, distill-only, CHEAPEST). Regret is in nats; agreement uses exactly the same own/common subsets. Empty subsets report N/A.

No-clear-winner heuristic: 164/289 evaluable cells; oracle method in heuristic candidate set: 100.00% of evaluable cells; 100.00% within heuristic no-clear-winner cells.

## QA

| Policy | Feasible / all | Coverage | Regret: own | Regret: common | Agreement: own | Agreement: common |
|---|---:|---:|---:|---:|---:|---:|
| MAP | 289/289 | 100.00% | 0.418401 | 0.157800 | 73.36% | 67.27% |
| ORACLE | 289/289 | 100.00% | 0.000000 | 0.000000 | 100.00% | 100.00% |
| prune-only | 161/289 | 55.71% | 0.724505 | 0.446886 | 44.10% | 9.09% |
| quant-only | 289/289 | 100.00% | 0.490451 | 0.573525 | 47.75% | 7.27% |
| distill-only | 96/289 | 33.22% | 0.056414 | 0.066484 | 83.33% | 83.64% |
| CHEAPEST | 289/289 | 100.00% | 4.333390 | 1.839234 | 64.71% | 58.18% |

Common-feasible subset: 55/289 cells, intersecting all six policies (MAP, ORACLE, prune-only, quant-only, distill-only, CHEAPEST). Regret is in nats; agreement uses exactly the same own/common subsets. Empty subsets report N/A.

No-clear-winner heuristic: 142/289 evaluable cells; oracle method in heuristic candidate set: 91.00% of evaluable cells; 100.00% within heuristic no-clear-winner cells.

## Multi-capability: minimum worst degradation

| Policy | Feasible / all | Coverage | Regret: own | Regret: common | Agreement: own | Agreement: common |
|---|---:|---:|---:|---:|---:|---:|
| MAP | 289/289 | 100.00% | 0.028792 | 0.017413 | 93.77% | 94.55% |
| ORACLE | 289/289 | 100.00% | 0.000000 | 0.000000 | 100.00% | 100.00% |
| prune-only | 161/289 | 55.71% | 0.673114 | 0.329482 | 1.24% | 0.00% |
| quant-only | 289/289 | 100.00% | 0.042774 | 0.017830 | 93.08% | 90.91% |
| distill-only | 96/289 | 33.22% | 0.333210 | 0.360621 | 3.12% | 0.00% |
| CHEAPEST | 289/289 | 100.00% | 5.178227 | 2.270036 | 78.20% | 43.64% |

Common-feasible subset: 55/289 cells, intersecting all six policies (MAP, ORACLE, prune-only, quant-only, distill-only, CHEAPEST). Regret is in nats; agreement uses exactly the same own/common subsets. Empty subsets report N/A.

No-clear-winner heuristic: 158/289 evaluable cells; oracle method in heuristic candidate set: 99.65% of evaluable cells; 100.00% within heuristic no-clear-winner cells.

## Figure

The figure retains v60's three MAP panels (math/code/QA), state order and budget axes. Grey cross-hatching denotes INFEASIBLE; diagonal hatching denotes the no-clear-winner heuristic. MAP has 100% feasibility on this measured panel, so there are no infeasible MAP cells to hatch. Fixed-policy infeasibility is reported in the tables and per-cell JSON.

## Data gaps and limits

- Excluded pythia-1.4b@step112000 from training and selection: no paired quantization file.
- Excluded pythia-410m@step48000 from training and selection: no paired quantization file.
- Excluded pythia-6.9b@step80000 from training and selection: no paired quantization file.
- pythia-160m@step16000: missing v53 pruning anchors [0.55]; selection uses only measured densities [0.9, 0.8, 0.7, 0.6].
- pythia-160m@step16000: no measured per-channel bits [5]; these configs are not candidates.
- pythia-160m@step64000: missing v53 pruning anchors [0.55]; selection uses only measured densities [0.9, 0.8, 0.7, 0.6].
- pythia-160m@step64000: no measured per-channel bits [5]; these configs are not candidates.
- pythia-160m@step64000: no grouped RTN measurements; no grouped configs are synthesized.
- pythia-160m@step96000: missing v53 pruning anchors [0.55]; selection uses only measured densities [0.9, 0.8, 0.7, 0.6].
- pythia-160m@step96000: no measured per-channel bits [5]; these configs are not candidates.
- pythia-160m@step96000: no grouped RTN measurements; no grouped configs are synthesized.
- pythia-160m@step143000: no measured per-channel bits [5]; these configs are not candidates.
- pythia-410m@step16000: missing v53 pruning anchors [0.55]; selection uses only measured densities [0.9, 0.8, 0.7, 0.6].
- pythia-410m@step16000: no measured per-channel bits [5]; these configs are not candidates.
- pythia-410m@step64000: missing v53 pruning anchors [0.55]; selection uses only measured densities [0.9, 0.8, 0.7, 0.6].
- pythia-410m@step64000: no measured per-channel bits [5]; these configs are not candidates.
- pythia-410m@step64000: no grouped RTN measurements; no grouped configs are synthesized.
- pythia-410m@step96000: no measured per-channel bits [5]; these configs are not candidates.
- pythia-410m@step96000: no grouped RTN measurements; no grouped configs are synthesized.
- pythia-410m@step143000: missing v53 pruning anchors [0.55]; selection uses only measured densities [0.9, 0.8, 0.7, 0.6].
- pythia-410m@step143000: no measured per-channel bits [5]; these configs are not candidates.
- pythia-1b@step32000: no grouped RTN measurements; no grouped configs are synthesized.
- pythia-1b@step96000: missing v53 pruning anchors [0.9, 0.8, 0.7, 0.6]; selection uses only measured densities [0.65, 0.55].
- pythia-1b@step96000: no measured per-channel bits [8, 6, 5]; these configs are not candidates.
- pythia-1b@step96000: grouped grid is partial; missing ['b3_g256', 'b3_g64', 'b4_g256', 'b4_g64', 'b5_g256', 'b5_g64'].
- pythia-1b@step112000: no grouped RTN measurements; no grouped configs are synthesized.
- pythia-1.4b@step16000: missing v53 pruning anchors [0.55]; selection uses only measured densities [0.9, 0.8, 0.7, 0.6].
- pythia-1.4b@step16000: no measured per-channel bits [5]; these configs are not candidates.
- pythia-1.4b@step64000: no measured per-channel bits [5]; these configs are not candidates.
- pythia-1.4b@step64000: no grouped RTN measurements; no grouped configs are synthesized.
- pythia-1.4b@step96000: missing v53 pruning anchors [0.55]; selection uses only measured densities [0.9, 0.8, 0.7, 0.6].
- pythia-1.4b@step96000: no measured per-channel bits [5]; these configs are not candidates.
- pythia-1.4b@step96000: no grouped RTN measurements; no grouped configs are synthesized.
- pythia-1.4b@step143000: missing v53 pruning anchors [0.55]; selection uses only measured densities [0.9, 0.8, 0.7, 0.6].
- pythia-1.4b@step143000: no measured per-channel bits [5]; these configs are not candidates.
- pythia-6.9b@step32000: no grouped RTN measurements; no grouped configs are synthesized.
- pythia-6.9b@step112000: no grouped RTN measurements; no grouped configs are synthesized.
- pythia-160m@step16000: no recovered smaller same-stage v39 student; distill-only is INFEASIBLE at every budget.
- pythia-160m@step64000: no recovered smaller same-stage v39 student; distill-only is INFEASIBLE at every budget.
- pythia-160m@step96000: no recovered smaller same-stage v39 student; distill-only is INFEASIBLE at every budget.
- pythia-160m@step143000: no recovered smaller same-stage v39 student; distill-only is INFEASIBLE at every budget.
- pythia-410m@step96000: no recovered smaller same-stage v39 student; distill-only is INFEASIBLE at every budget.
- pythia-1b@step32000: no recovered smaller same-stage v39 student; distill-only is INFEASIBLE at every budget.
- pythia-1b@step96000: no recovered smaller same-stage v39 student; distill-only is INFEASIBLE at every budget.
- pythia-1b@step112000: no recovered smaller same-stage v39 student; distill-only is INFEASIBLE at every budget.
- pythia-1.4b@step96000: no recovered smaller same-stage v39 student; distill-only is INFEASIBLE at every budget.
- pythia-6.9b@step32000: no recovered smaller same-stage v39 student; distill-only is INFEASIBLE at every budget.
- pythia-6.9b@step112000: no recovered smaller same-stage v39 student; distill-only is INFEASIBLE at every budget.
- Distillation measurements use the fixed external teacher gpt-5.6-luna, full/600, two epochs, LoRA, seed 0. They measure smaller-student alternatives, not transfer from each plotted source; no other student sizes/stages or recipes are inferred. 9 raw students supply fits; 6 distinct students supply selectable alternatives (shared across source choices, not independent runs).
- Dense references are not identical across measurement arms; maximum absolute offsets from pruning dense (nats) are {'math': 0.002096021513881219, 'code': 0.013192298550035897, 'qa': 0.012298003802280633}. Fit signed changes against each file's own dense anchor and retain recorded absolute losses. Dense source and multi-capability reference use pruning key '1.0'; no post-hoc recentering. These offsets limit tiny-gap comparisons.
- Per-channel 5-bit has only four measured states. Its four-coefficient OLS blocks have rank 3 from three training observations when holding out pythia-1b@step112000, pythia-1b@step32000, pythia-6.9b@step112000, pythia-6.9b@step32000. Retain NumPy's deterministic minimum-norm least-squares solution; it is underidentified. All other measured-bit blocks have rank 4; no 5-bit interpolation is used.
- The measured configuration grid is heterogeneous: no missing loss is interpolated or imputed. Method agreement measures method identity, not exact configuration recovery. Budget cells share states and raw runs; no independence-based confidence intervals are claimed.
- No-clear-winner heuristic thresholds use same-run pooled out-of-fold MAEs, including the scored fold. They are retrospective heuristic annotations, not independently calibrated uncertainty or inputs to MAP selection. Grouped RTN uncertainty uses its own law, although both RTN variants count as the single quant method.

## Dense-anchor differences

Signed offset from the source pruning dense anchor; all nonzero differences are listed.

| State | Anchor | Math | Code | QA |
|---|---|---:|---:|---:|
| pythia-160m@step16000 | quant_group | -0.000988689 | -0.000118850 | -0.000861454 |
| pythia-160m@step16000 | student | -0.000988689 | -0.000118850 | -0.000861454 |
| pythia-160m@step64000 | student | +0.000533892 | -0.000178274 | +0.004277567 |
| pythia-160m@step143000 | quant_channel | -0.002096022 | -0.013192299 | +0.002732890 |
| pythia-410m@step16000 | quant_group | -0.000158190 | +0.000772522 | -0.002435837 |
| pythia-410m@step16000 | student | -0.000158190 | +0.000772522 | -0.002435837 |
| pythia-410m@step64000 | student | +0.000612987 | -0.000297124 | +0.012298004 |
| pythia-410m@step143000 | quant_group | -0.000276833 | -0.005823627 | +0.011347433 |
| pythia-410m@step143000 | student | -0.000276833 | -0.005823627 | +0.011347433 |
| pythia-1.4b@step16000 | quant_group | -0.000573440 | +0.000297124 | -0.001841730 |
| pythia-1.4b@step16000 | student | -0.000573440 | +0.000297124 | -0.001841730 |
| pythia-1.4b@step64000 | quant_channel | -0.000771178 | +0.000059425 | +0.000148527 |
| pythia-1.4b@step143000 | quant_group | -0.000158190 | -0.002852389 | -0.011466255 |
| pythia-1.4b@step143000 | student | -0.000158190 | -0.002852389 | -0.011466255 |

## Reuse and validation

- Imported unchanged: v53 fit_all/predict_all; v55 fit_all/predict_all; v39 _fit/_predict/_cv. v36 basic_input/covariates/design_matrix/predict supply the quantization path, with the documented 5-bit vocabulary/rank exception. All source files are hashed.
- Reconstructed v39 raw deltas agree with saved delta fields; original leave-size/leave-step MAEs reproduce summary.json to 1e-10 relative / 1e-12 absolute tolerance when the cohort is available.
- Each fold asserts source exclusion; each student prediction asserts target-student exclusion. Prediction interfaces receive no actual loss. Every feasible policy obeys its budget and has nonnegative regret.
- Plot text is checked for >=8 pt and containment within the saved canvas. PDF creation/modification timestamps are suppressed; duplicate font-family matches are ordered by absolute filename. Inputs are rehashed before exclusive output creation.
- Read-only v60 regression: all states/configs, predictions, fits, MAEs, students and budgets match its saved summary exactly. 5652 retained policy choices are identical; 1284 source fallbacks become INFEASIBLE across four objectives. The no-clear-winner heuristic is unchanged. Own/common metrics are independently recomputed. Both v60 result files are hashed and rechecked without modification.

## Files read (SHA-256)

Every explicit input/source/font read is printed, hashed and rechecked before outputs. Standard Python/NumPy/matplotlib installation files are represented by package versions.

```text
cfbe3d2edd46ae844b7eb95012f9274776801873e27b2230bbf751645f011888  /usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf
8d6951ea5fc4a9656df4802227292c7943186364929f00b97629e70daf228439  /usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf
f86031b17211cc060873da1ecc6880c6f3f11246005b2d5eab9605a3b6ec7d67  analysis/plot_fig1_responses.py
8bb4785999878ab48eb0940c6a13391f0d7a6de858ee3eeb115acacdc6102eeb  analysis/prediction_audit.py
ac665407235616c5a66801b51efae0c6f6258859e53fd9ee62243ad9e64ef91d  analysis/v36_pythia_controlled_fit.py
f051d85b8e93bd8245217717de86c58aea9ddcac99a97f4c738ef991be437914  analysis/v38_prospective_step.py
437dc7576814ef1697a706a7ab0d3ba70b57abd1c7a65a53a8fbc8b4a6f9f888  analysis/v39_distill_controlled.py
b285ebc20757bb6b8c9e408aeafc1a92fcd21e5d84e489d41b0ed3409c7571e1  analysis/v49_p1v2.py
5a3fa776ab1ddf491fdc70bad4bf7e7ee000754a735ec6ea10431c32a661bff2  analysis/v53_prune_dev.py
596015ebcc8f635e8454f914bda159654cc85b2fc463f614243ff13df3c3f926  analysis/v55_quant_group_fit.py
d8043bf79941963d7847c505c65b9a5f1c572ead88c350fc26a5f08e9b419ace  analysis/v60_selection_maps.py
e50d7fd98e9d0cced86d5c165daf0878e68ba744accf40585c2d05238a90b847  analysis/v64_selection_feasible.py
5330aaa9da7183204c37d1ff1e2ea64c651ed588bb2f72f0ed7aaed72a595243  results/v10-quantization/pythia-1.4b--step143000/quant_losses.json
a903eaa306be8c4ba70fda979bcfafa4cc1334fab7b62997e4b6e7f96cf2c945  results/v10-quantization/pythia-1.4b--step16000/quant_losses.json
1087bbf6102d85fded469961d958c713908e4ba936e24cd2d4985ee8b6e2a274  results/v10-quantization/pythia-1.4b--step64000/quant_losses.json
c733d0349c8b0fd1585cc7dae4374bf7bc81a213dd5097f3fc166e830701b74a  results/v10-quantization/pythia-1.4b--step96000/quant_losses.json
96aa01041b947077090736afe19a1becd8b9135c9b7b4037a5ea457d6fd0b334  results/v10-quantization/pythia-160m--step143000/quant_losses.json
11d1845671057066a1e15286b7f84853c6db11511b44bb7bb72328d3fd0d3178  results/v10-quantization/pythia-160m--step16000/quant_losses.json
dde3f0ad83db6997054a79318959dee2c33195ded3d11247fcb12a9992fba4ca  results/v10-quantization/pythia-160m--step64000/quant_losses.json
e26eb6b26ef5e4a27763742340714a3d1f05d056bb6881514f9336e8198becdb  results/v10-quantization/pythia-160m--step96000/quant_losses.json
16e3e67391239f53fad9194b8d378eadce39c106bd037e4fed436e96e42ce956  results/v10-quantization/pythia-1b--step112000/quant_losses.json
251ab24328fe4cb9fbbfa4ed5c691b423482ff6e6c15e7a3431e3da00188eb3b  results/v10-quantization/pythia-1b--step32000/quant_losses.json
dcf1c3d70c920830e4da7e841637006de3a154ebc275dacf92820cc23772d7e2  results/v10-quantization/pythia-1b--step96000/quant_losses.json
6f0e6c037d8b993301839d55f1b8e1261680cc1ed93864b08a2237a1b5698790  results/v10-quantization/pythia-410m--step143000/quant_losses.json
ae2ae21276a632816c48bfad123d6963e3f0e4b96c56c47a046220f39b00aed7  results/v10-quantization/pythia-410m--step16000/quant_losses.json
6d0bafbaabcf80fc426ee0587d1deceed793b9ad3ef4ddad189e0426212102e0  results/v10-quantization/pythia-410m--step64000/quant_losses.json
ca19f3629667b745ce6377e6d88916c514a8ed12e231031ea98f2ae542cad3eb  results/v10-quantization/pythia-410m--step96000/quant_losses.json
9a743a4fdc43f3d3189dac056197a9f7969db8743783fcd3a4ad1ba4ce99cdd4  results/v10-quantization/pythia-6.9b--step112000/quant_losses.json
36286d54d78217724d7ae69fb176d82c4028f63a6a46fb20551acdb1bb44a4d1  results/v10-quantization/pythia-6.9b--step32000/quant_losses.json
03bfc7ba006d4c939d121ab05bfca341d6b10663ef79d98f8a905f6fbcd2d806  results/v12-distill/pythia-1.4b--step143000/gpt-5.6-luna_full_600_lora/eval.json
8272ea3e5be3d21da3bbb2107cc3c0c4d0b2b8e06b85518276745e747103f37d  results/v12-distill/pythia-1.4b--step16000/gpt-5.6-luna_full_600_lora/eval.json
5223d7076aee9ca12e4b5d1e492f803492e67840fd85f1b8e65b11a68e007953  results/v12-distill/pythia-1.4b--step64000/gpt-5.6-luna_full_600_lora/eval.json
667f240a9596a8953a1f1bddf2437e10b496de149084f6b5ba52eff2c275184c  results/v12-distill/pythia-160m--step143000/gpt-5.6-luna_full_600_lora/eval.json
649fdb3a06ef15e6e8df8da9e111de582948b027698410e02a10f5029901797d  results/v12-distill/pythia-160m--step16000/gpt-5.6-luna_full_600_lora/eval.json
2f23cd700f575b1ad10fed3e74ca1b222bd586b5d1701729fb198152679f23de  results/v12-distill/pythia-160m--step64000/gpt-5.6-luna_full_600_lora/eval.json
905daebf301bbaf6277f72c4fb240d1493b04b5063866e81d2e1286231b4d555  results/v12-distill/pythia-410m--step143000/gpt-5.6-luna_full_600_lora/eval.json
c695fccc72aadde505d91b4f10a4bdd48f0dbf4447d9d4a6ece7fa98cb65396b  results/v12-distill/pythia-410m--step16000/gpt-5.6-luna_full_600_lora/eval.json
03b1a47d9a4252e6eb57ed516f77bee46277a53b8863df81a10aecf7666f0d4b  results/v12-distill/pythia-410m--step64000/gpt-5.6-luna_full_600_lora/eval.json
eba6667131108d3218d76f4f19e0415d0487fffd26a2d83bad5232a6dfbea473  results/v39-distill-controlled/summary.json
3b9d7d9b5584d542e8a9c88c8d02311b6a88ded22815ac215437b40b7183e406  results/v54-quant-group/pythia-1.4b--step143000/quant_group_losses.json
41bc3efd814274b8a5da559dd313724b0de85460985b7d79aa371b16da140566  results/v54-quant-group/pythia-1.4b--step16000/quant_group_losses.json
803b5327fa5f747a23f23793f57803765fe3200f0b08a9edbd3216fb3d7c322f  results/v54-quant-group/pythia-160m--step143000/quant_group_losses.json
67cb8d973d7116ee26ccbc295f36c8aeabc1f733e10e3e3e28e3a31c7a8c5939  results/v54-quant-group/pythia-160m--step16000/quant_group_losses.json
3d5b139ddcd811afe4257dcc6f5dacf6e700b9bb153d96c841cc086184039426  results/v54-quant-group/pythia-1b--step96000/quant_group_losses.json
bae7431f3c984976e1a22f02bf830254196f060c7d6f052a613a37e85f5e1389  results/v54-quant-group/pythia-410m--step143000/quant_group_losses.json
1f92ad048199581e23c3626c5d2065c860c37c1a53351552d2bd2068d50eae81  results/v54-quant-group/pythia-410m--step16000/quant_group_losses.json
bfc02e7d54eb623c2e8e4f1237b546aa62ecd14f23330b50e2a833e226916c42  results/v6-capability-geometry/pythia-1.4b--step112000/prune_losses.json
5ec4800f980d6317520f3d7eb85f4516631b629a146d5a964e33092eb1507315  results/v6-capability-geometry/pythia-1.4b--step143000/prune_losses.json
95cbe7e98739be4ea472781bf4f8a27c28827250e22d1bdbd8c5c576bd73938a  results/v6-capability-geometry/pythia-1.4b--step16000/prune_losses.json
a458cd11272a9face425eda62134b25ef0bf4af622f5902c63f13233028c0b15  results/v6-capability-geometry/pythia-1.4b--step64000/prune_losses.json
291926ecf463d6052e16336f0473aa364c4098db5a99185febe832e83d531f81  results/v6-capability-geometry/pythia-1.4b--step96000/prune_losses.json
87d9ebcad62e27b675e038b584671175f66aed91fc18da786ffc0f553af68e37  results/v6-capability-geometry/pythia-160m--step143000/prune_losses.json
46c5f9c6cfb0b5581b13f517951b861cb1c52998c333945a37639b1031862525  results/v6-capability-geometry/pythia-160m--step16000/prune_losses.json
f5791d151a0507ac90aeaddaf97d4686ceead336d4fa7b128ec711f4793ad354  results/v6-capability-geometry/pythia-160m--step64000/prune_losses.json
477e2715ec4b9033fb27a7fb169a3b7e08c0a6682572712190aed7178d36a6bc  results/v6-capability-geometry/pythia-160m--step96000/prune_losses.json
28542d8fe21c9be339b5b946416cf53b80fa3b1d8d8aedb04676b4eed80c16f3  results/v6-capability-geometry/pythia-1b--step112000/prune_losses.json
ce73eda3b0059614505fe20e03f2bf135f1e82d80e49b881f267174580ac5e87  results/v6-capability-geometry/pythia-1b--step32000/prune_losses.json
0f8d7866ab0d6a013b464aceebcbf274d0cc7ed9413be0281a38b5ac1b962d2f  results/v6-capability-geometry/pythia-1b--step96000/prune_losses.json
f46f388cee3fbb1bb7a0175c1dfde457b13bee0fa262092d9ac1e41845df0e7b  results/v6-capability-geometry/pythia-410m--step143000/prune_losses.json
228e436b0b58d734eb924f2ddc45b4c8135aa3b2765bb3b764db7a456123dcf9  results/v6-capability-geometry/pythia-410m--step16000/prune_losses.json
e71893a8e75f8abf1dc023400ab910a8765d81171d3abc991ffb1d77e012738f  results/v6-capability-geometry/pythia-410m--step48000/prune_losses.json
81bafd7adac31f27f2973545407005177fad63cf6d6e2ef942109e10fd8f3756  results/v6-capability-geometry/pythia-410m--step64000/prune_losses.json
a2a5307ace6cc58f1e987c3a7c8cbac8da1e6d17f4c9487cf55ca95d1b47a4c8  results/v6-capability-geometry/pythia-410m--step96000/prune_losses.json
a7952f3fb14575f0090dac6eeec4f103b556f69781685fb16d914dbe1905d9cb  results/v6-capability-geometry/pythia-6.9b--step112000/prune_losses.json
90b4e5a53afcfd24b9c355578d0bbb2753d006169661f521b2a82b2d17f885d2  results/v6-capability-geometry/pythia-6.9b--step32000/prune_losses.json
9f37ba12f066a94d93c07a93a180f89b9de08b77b0a7b3697a6de6a02dc1f3dd  results/v6-capability-geometry/pythia-6.9b--step80000/prune_losses.json
a0c5893c8b5f1784e648a1643c4a4bd0dde43fb17abf6ec605604e881dd11c4d  results/v60-selection-maps/summary.json
e8565b890c82d7c0e830c91c0d5f41042e4c3462ee8b4e09c969cdbe03492a3d  results/v60-selection-maps/summary.md
```

## Outputs

- `summary.json`
- `summary.md`
- `paper/tables/selection_feasible.tex`
- `paper/tables/candidate_coverage.tex`
- `paper/figs/selection_feasible.pdf`
- `paper/figs/selection_feasible.png`
- `paper/code/analysis/v64_selection_feasible.py`
