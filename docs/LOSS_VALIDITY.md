# Capability-loss validity: held-out prediction gain

V26 reads the six-model × five-density V23 pruning panel (0.90/0.85/0.80/0.75/0.70). Endpoint: mean per-example target-token CE; deltas use each benchmark's own dense anchor. Pairs: MATH-500→GSM8K, MBPP→HumanEval, 2WikiMultihopQA→HotpotQA. Dense anchors are not scored. All compression cells, including negative changes and cliffs, remain in the main analysis. Different tokenizers still imply different token units; generalization is empirical within this panel.

**Primary evidence is prediction gain, not correlation.** For each secondary target, fit y=intercept+slope×ΔL_primary on development cells. Every source uses the same two-parameter affine OLS mapping, folds and target cells. A constant source falls back to the training mean. Neither coefficients nor source selection use held-out secondary losses. The held-out primary ΔL is an input: this tests transfer between measured benchmarks, not prediction from density alone.

Main folds hold out a whole model, a whole density, or both: the joint test excludes all rows of the target model AND all rows at the target density from development and tests their intersection. Cross-selected chooses between the two other primary capabilities using inner development CV (model folds for model/joint tests; density folds for density tests). All individual cross mappings are also reported so selection cannot hide a strong comparator.

Paired intervals use 10,000 model-bootstrap draws (six clusters), holding OOF fits fixed. Positive gain = MAE(cross-selected)−MAE(same), favoring capability specificity. These small-panel descriptive CIs omit retraining and shared-probe uncertainty; multiple targets and protocols are exploratory. No post-hoc practical-effect threshold is imposed.

## all_densities: prediction

| Protocol | Secondary target | Cells | Zero MAE | Mean MAE | Math→target | Code→target | QA→target | Cross-selected | Same-over-cross gain (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| leave_model_out | math | 30 | 0.25348 | 0.36192 | 0.07393 | 0.11544 | 0.36115 | 0.11544 | 0.04152 [0.03158, 0.04990] |
| leave_model_out | code | 30 | 0.19534 | 0.27341 | 0.15611 | 0.12065 | 0.28816 | 0.15611 | 0.03545 [-0.02552, 0.13268] |
| leave_model_out | qa | 30 | 0.26563 | 0.28530 | 0.30515 | 0.29274 | 0.16751 | 0.30707 | 0.13956 [0.01828, 0.27410] |
| leave_density_out | math | 30 | 0.25348 | 0.36137 | 0.08777 | 0.22382 | 0.30446 | 0.22382 | 0.13605 [0.02657, 0.29506] |
| leave_density_out | code | 30 | 0.19534 | 0.27379 | 0.08849 | 0.15000 | 0.26760 | 0.08849 | -0.06151 [-0.16462, -0.00222] |
| leave_density_out | qa | 30 | 0.26563 | 0.27101 | 0.34264 | 0.34095 | 0.12353 | 0.33794 | 0.21441 [0.02135, 0.52196] |
| leave_model_and_density_out | math | 30 | 0.25348 | 0.38438 | 0.10315 | 0.26281 | 0.32771 | 0.26281 | 0.15967 [0.03061, 0.35859] |
| leave_model_and_density_out | code | 30 | 0.19534 | 0.28928 | 0.14083 | 0.19725 | 0.25147 | 0.14083 | -0.05643 [-0.14501, -0.00361] |
| leave_model_and_density_out | qa | 30 | 0.26563 | 0.27743 | 0.24471 | 0.21981 | 0.17678 | 0.22442 | 0.04764 [-0.02669, 0.12198] |

## without_strongest: prediction

| Protocol | Secondary target | Cells | Zero MAE | Mean MAE | Math→target | Code→target | QA→target | Cross-selected | Same-over-cross gain (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| leave_model_out | math | 24 | 0.15069 | 0.21180 | 0.03388 | 0.03892 | 0.22573 | 0.03892 | 0.00504 [-0.00311, 0.01211] |
| leave_model_out | code | 24 | 0.11139 | 0.15367 | 0.04390 | 0.05113 | 0.15477 | 0.04390 | -0.00722 [-0.01741, 0.00303] |
| leave_model_out | qa | 24 | 0.19493 | 0.19710 | 0.23370 | 0.26230 | 0.13741 | 0.26482 | 0.12741 [0.05745, 0.22038] |
| leave_density_out | math | 24 | 0.15069 | 0.22065 | 0.04438 | 0.03610 | 0.31812 | 0.03610 | -0.00828 [-0.03088, 0.00644] |
| leave_density_out | code | 24 | 0.11139 | 0.15827 | 0.05486 | 0.07014 | 0.22048 | 0.05486 | -0.01528 [-0.03370, -0.00389] |
| leave_density_out | qa | 24 | 0.19493 | 0.18666 | 0.26422 | 0.27836 | 0.15834 | 0.26422 | 0.10589 [0.02909, 0.21111] |
| leave_model_and_density_out | math | 24 | 0.15069 | 0.23046 | 0.04105 | 0.03900 | 0.25450 | 0.03900 | -0.00205 [-0.01462, 0.00794] |
| leave_model_and_density_out | code | 24 | 0.11139 | 0.16527 | 0.04125 | 0.04878 | 0.17673 | 0.04125 | -0.00753 [-0.01078, -0.00382] |
| leave_model_and_density_out | qa | 24 | 0.19493 | 0.19634 | 0.46400 | 0.46311 | 0.15811 | 0.47925 | 0.32114 [0.03107, 0.84021] |

## Capability-specific interpretation

- math, joint holdout: all_densities: same is better in this paired interval; without_strongest: same-versus-cross advantage is unresolved.
- code, joint holdout: all_densities: cross is better in this paired interval; without_strongest: cross is better in this paired interval.
- qa, joint holdout: all_densities: same-versus-cross advantage is unresolved; without_strongest: same is better in this paired interval.

Where same-capability prediction does not improve over cross-capability, high Pearson correlation is consistent with general compression damage a_j·h(d); it does not establish capability-specific structure. That explanation is compatible with the data, not a uniquely identified causal mechanism. A gain that disappears without density 0.70 is sensitive to the deepest point. Inspect absolute gains, their intervals, and individual cross-source errors.

## Pearson is descriptive: full ladder versus excluding density 0.70

| Model | Target | Same, all 5 | Same, mildest 4 | Cross math, mildest 4 | Cross code, mildest 4 | Cross QA, mildest 4 |
|---|---|---:|---:|---:|---:|---:|
| gemma3-12b | math | 0.99453 | 0.99781 | N/A | 0.99615 | -0.59901 |
| gemma3-12b | code | 0.99695 | 0.99687 | 0.99717 | N/A | -0.60114 |
| gemma3-12b | qa | 0.96262 | 0.94469 | -0.85156 | -0.78099 | N/A |
| gemma3-27b | math | 0.97709 | 0.99070 | N/A | 0.99652 | 0.47142 |
| gemma3-27b | code | 0.93180 | 0.93366 | 0.95136 | N/A | 0.04189 |
| gemma3-27b | qa | 0.96281 | 0.73908 | -0.36819 | -0.31879 | N/A |
| gemma3-4b | math | 0.99567 | 0.99931 | N/A | 0.99495 | 0.49069 |
| gemma3-4b | code | 0.99610 | 0.99922 | 0.99864 | N/A | 0.44819 |
| gemma3-4b | qa | 0.98440 | 0.97920 | 0.43403 | 0.35890 | N/A |
| muse-30b | math | 0.99461 | 0.97695 | N/A | 0.93724 | 0.88183 |
| muse-30b | code | 0.99707 | 0.99164 | 0.99089 | N/A | 0.97108 |
| muse-30b | qa | 0.87725 | 0.98529 | 0.91209 | 0.95989 | N/A |
| olmo3-32b | math | -0.58379 | -0.96327 | N/A | -0.88559 | -0.19960 |
| olmo3-32b | code | 0.98863 | 0.97540 | 0.99871 | N/A | 0.41934 |
| olmo3-32b | qa | -0.57182 | -0.60389 | -0.96347 | -0.93714 | N/A |
| olmo3-7b | math | 0.15860 | -0.67890 | N/A | 0.15192 | -0.63381 |
| olmo3-7b | code | 0.84498 | 0.76608 | 0.97292 | N/A | 0.93864 |
| olmo3-7b | qa | 0.27452 | 0.65900 | 0.60228 | 0.96399 | N/A |

## OLMo: measured changes versus sampling resolution

Subtract dense/compressed losses ITEM BY ITEM, checking probe hashes and target lengths. Noise half-width is median_d[1.96×SD(item ΔL_d)/sqrt(n)]. Signal is RMS of the centered mean ΔL trajectory. Its sampling resolution is RMS of bootstrap SDs after centering each resampled trajectory; the same sampled items are used across densities, preserving covariance from the shared dense anchor. Signal/resolution <2 is labeled insufficient resolution as a descriptive screen, not a formal correlation hypothesis test. This is benchmark-item sampling uncertainty, not measured numerical or seed noise: no repeated-run noise estimate exists.

| Model | Capability | Benchmark role | Mean abs ΔL | Max abs ΔL | Median 95% noise half-width | Signal/resolution | Resolution assessment |
|---|---|---|---:|---:|---:|---:|---|
| olmo3-32b | math | primary | 0.00697 | 0.01732 | 0.00355 | 4.82881 | variation exceeds 2x estimated sampling resolution |
| olmo3-32b | math | secondary | 0.00335 | 0.00550 | 0.00342 | 1.42254 | insufficient resolution |
| olmo3-32b | code | primary | 0.01772 | 0.03861 | 0.00499 | 5.10578 | variation exceeds 2x estimated sampling resolution |
| olmo3-32b | code | secondary | 0.00762 | 0.01637 | 0.00331 | 3.42813 | variation exceeds 2x estimated sampling resolution |
| olmo3-32b | qa | primary | 0.00792 | 0.02014 | 0.05242 | 0.42244 | insufficient resolution |
| olmo3-32b | qa | secondary | 0.06699 | 0.11151 | 0.04613 | 1.53632 | insufficient resolution |
| olmo3-7b | math | primary | 0.01039 | 0.02727 | 0.00304 | 7.01087 | variation exceeds 2x estimated sampling resolution |
| olmo3-7b | math | secondary | 0.00444 | 0.00706 | 0.00495 | 2.06075 | variation exceeds 2x estimated sampling resolution |
| olmo3-7b | code | primary | 0.01595 | 0.03255 | 0.00733 | 4.45241 | variation exceeds 2x estimated sampling resolution |
| olmo3-7b | code | secondary | 0.01252 | 0.03724 | 0.00517 | 4.95544 | variation exceeds 2x estimated sampling resolution |
| olmo3-7b | qa | primary | 0.11252 | 0.22804 | 0.07737 | 1.85822 | insufficient resolution |
| olmo3-7b | qa | secondary | 0.03530 | 0.09158 | 0.07322 | 1.25032 | insufficient resolution |

Low OLMo correlations are not 'noise-confirmed'. If either benchmark lacks trajectory resolution, capability agreement is unresolved at the available signal scale. Where both trajectories are resolved yet disagree, insufficient resolution alone is not an adequate explanation. Cell-level paired-item intervals are in summary.json. Neither correlation nor prediction gain establishes accuracy validity, a universal latent capability, or absence of benchmark contamination.

Reproduce CPU-only: `python analysis/v26_loss_validity_prediction.py --dry-run`, then without `--dry-run`. Outputs: `results/v26-loss-validity-pred/summary.json` and `report.md`; the latter rebuilds this document. JSON retains all held-out predictions, mapping coefficients, inner source selection scores, same-versus-each-cross paired intervals, item resolution, descriptive correlations and SHA-256 input hashes. Original V23 inputs are read only.
