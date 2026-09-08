# Pythia controlled training-history analysis (V36)

On the supplied grid, adding D₀ lowers point-estimate MAE in 7 of 12 arm/capability/holdout comparisons; 4 improvement intervals lie wholly above zero. The tables keep the arms and capabilities separate; this count is descriptive and is not a pooled score.

The headline question is whether training history D₀ reduces held-out prediction error for capability **loss** damage beyond model size N₀ and measured dense loss L₀. The target is the observed signed response ΔL_c = L_c(config) − L₀,c. This is a loss endpoint; it does not by itself establish changes in task accuracy.

This is a **fixed-training-recipe SERIES**, using standard (non-deduped) Pythia checkpoints. The suite shares training data/order and token accounting; architectures and hyperparameters still vary across sizes (including learning rate). See the [Pythia paper](https://proceedings.mlr.press/v202/biderman23a/biderman23a.pdf) and [official training documentation](https://github.com/EleutherAI/pythia). Within a size, D₀ also tracks checkpoint age, cumulative optimization and the learning-rate schedule. A conditional prediction gain is evidence about this series, not an isolated causal token effect or a universal compression law.

**Distillation is PENDING and not included:** its training runs are blocked on flaky GPUs. This analysis launches no training or inference.

## Inputs and fixed analysis specification

The 18 existing V6/V10 loss JSON files yield 9 size×step cells per arm and 36 compressed observations per capability/arm (216 total). Pruning uses densities 0.9/0.8/0.7/0.6 with dense key `1.0`; quantization uses bits 8/6/4/3 with dense key `dense`. Each response and predictor L₀ use that file's own measured capability-specific dense reference. The maximum dense-reference discrepancy between arms is 0.00000. Arms and capabilities are fitted and scored separately in nats per native token, with no pooled headline score.

N₀ is computed from official GPT-NeoX architecture configs: `layers × (4h² + 2hm)` for attention QKV/output and the two MLP matrices. It excludes embeddings, the LM head, biases and normalization vectors. These are exact matrix counts, not nominal model names or the paper's slightly larger non-embedding counts including vectors. The dimensions below were transcribed from local cached official configs, verified equal across the three revisions per size; pinned config URLs and file SHA256s are in summary.json. Reproduction needs no weights or config download.

| Size label | Layers | h | m | N₀ matrix parameters |
|---|---:|---:|---:|---:|
| [160m](https://huggingface.co/EleutherAI/pythia-160m/blob/b56d9bee36300031aeea723b73c4d62ac7fa71a2/config.json) | 12 | 768 | 3072 | 84,934,656 |
| [410m](https://huggingface.co/EleutherAI/pythia-410m/blob/bba6a464f54bbf08fc174cfb351d9794d58af21d/config.json) | 24 | 1024 | 4096 | 301,989,888 |
| [1.4b](https://huggingface.co/EleutherAI/pythia-1.4b/blob/9cc5c8c8148a4e0115d9e29c6b4f21124cfe748a/config.json) | 24 | 2048 | 8192 | 1,207,959,552 |
| [2.8b](https://huggingface.co/EleutherAI/pythia-2.8b/blob/dbe7ae300a54abcdc475a33907b3dff81d25709f/config.json) | 32 | 2560 | 10240 | 2,516,582,400 |

The existing measurement code compresses **all language matrices, including embeddings and the LM head** (`language_weight_parameters` in V6, reused by V10). Thus N₀ is a size covariate rather than the total compressed parameter count. V6 uses global magnitude pruning; V10 uses symmetric per-output-channel fake quantization. The grid runner specifies bf16 and 128 probes, with the measurement routines using the odd-indexed held-out half. Only aggregate capability losses are available here, so no probe-level or seed uncertainty is estimated.

D₀ is **processed pretraining tokens**: `step × 1024 × 2048`, from the [Pythia training batch specification](https://proceedings.mlr.press/v202/biderman23a/biderman23a.pdf). It is not unique-token count or distillation data volume.

| Step | D₀ tokens | Billions (rounded) |
|---|---:|---:|
| 16000 | 33,554,432,000 | 33.6 |
| 64000 | 134,217,728,000 | 134.2 |
| 143000 | 299,892,736,000 | 299.9 |

For each arm and capability, let q index its four measured configurations:

```text
A: F(N₀,L₀,q)    = α_q + β_q log(N₀/10⁹) + γ_q L₀
B: F(N₀,L₀,D₀,q) = α_q + β_q log(N₀/10⁹) + γ_q L₀ + δ_q log(D₀/10⁹)
```

All coefficients are fitted jointly by ordinary least squares against the **raw configuration-level ΔL responses**. There is no per-source a_c estimation followed by label regression. Configuration indicators and their covariate interactions let int3 have a different response from high bits without forcing a common amplitude or shape. A has 12 coefficients and B has 16; setting B's four D₀ coefficients to zero gives A. Each configuration effectively has six training cells and three (A) or four (B) coefficients. No model-form selection, regularization, hyperparameter tuning, log-response transform or prediction clipping is used. Input standardization uses only the training rows of each fold. Full-rank designs are required. The domain is the four measured configurations; there is no unseen-configuration prediction claim. Dense anchors supply references and are not fitted/scored as artificial zero-damage rows.

The same A/B specification is used for **leave-one-SIZE-out** (all three steps/configurations of that size withheld) and, separately, **leave-one-STEP-out** (all sizes/configurations at that step withheld). Each of the three folds has 24 training and 12 test observations per arm/capability, covering six training and three test source cells. The middle size/step is interpolation and the endpoints require extrapolation. Every observation is predicted once per holdout scheme. Target compressed outcomes never enter training or preprocessing; target dense L₀ is an explicitly permitted input.

**Pre-specified handling for this run:** retain and flag all int3 rows, all `|ΔL| ≤ 0.01` near-zero rows, all negative responses, and all `ΔL > 1` large-damage rows. The latter is a descriptive large-damage threshold, not a validated definition of collapse. No censoring, clipping, winsorization or response-dependent weights enter either fit or primary MAE. Missing/nonfinite values fail the run instead of disappearing. These rules and forms were fixed before running V36 fits; the existing smoke results were known, so this is **retrospective**, not a new prospective preregistration.

## Held-out prediction: does D₀ help beyond L₀?

MAE weights all 36 held-out observations equally within a capability/arm. Positive **A−B** means adding D₀ reduces error. Brackets are paired 95% cluster-bootstrap intervals. The bootstrap enumerates all 27 ordered resamples of the three whole held-out sizes or steps, carrying every configuration and the other axis together, and uses discrete inverse-CDF percentiles. Fits remain fixed. With only three groups, overlapping training folds and shared trajectories/probes, these are coarse **panel-conditional descriptive intervals**, not independent-seed, retraining, population, probe or causal uncertainty. The leave-one-group-out score range in summary.json is a sensitivity range, not another CI. There is no multiplicity correction across the 12 comparisons.

| Arm | Capability | Holdout | A MAE [95%] | B MAE [95%] | A−B [95%] | Assessment |
|---|---|---|---:|---:|---:|---|
| pruning | math | size | 0.61546 [0.36748, 1.04180] | 0.14694 [0.06532, 0.26660] | 0.46851 [0.30217, 0.77519] | lower error throughout interval |
| pruning | math | step | 0.61667 [0.10618, 0.93482] | 0.25539 [0.14354, 0.41856] | 0.36128 [-0.03735, 0.60493] | inconclusive |
| pruning | code | size | 0.93939 [0.58566, 1.46607] | 0.99983 [0.27479, 2.23603] | -0.06044 [-0.76996, 0.31087] | inconclusive |
| pruning | code | step | 0.69997 [0.10246, 1.11884] | 0.29796 [0.11123, 0.50305] | 0.40201 [-0.00877, 0.61580] | inconclusive |
| pruning | qa | size | 0.92703 [0.56219, 1.14662] | 1.09827 [1.04947, 1.13770] | -0.17124 [-0.57551, 0.03899] | inconclusive |
| pruning | qa | step | 0.76506 [0.39030, 0.98763] | 0.79629 [0.42889, 1.02247] | -0.03123 [-0.10524, 0.05012] | inconclusive |
| quantization | math | size | 2.02839 [1.06915, 3.43045] | 0.77131 [0.25473, 1.53985] | 1.25708 [0.81442, 1.89060] | lower error throughout interval |
| quantization | math | step | 1.98931 [0.33542, 3.06177] | 1.14448 [0.46978, 2.06359] | 0.84483 [-0.13436, 1.67069] | inconclusive |
| quantization | code | size | 2.85022 [1.25874, 5.46418] | 1.26701 [0.33478, 2.69135] | 1.58321 [0.92396, 2.77283] | lower error throughout interval |
| quantization | code | step | 2.07693 [0.37423, 3.24288] | 0.62600 [0.22094, 1.14772] | 1.45094 [0.15329, 2.10435] | lower error throughout interval |
| quantization | qa | size | 2.27202 [1.40639, 2.97373] | 2.61091 [2.32588, 2.85237] | -0.33889 [-1.24809, 0.12136] | inconclusive |
| quantization | qa | step | 1.96411 [0.92058, 2.62249] | 1.98938 [1.12520, 2.70581] | -0.02527 [-0.35654, 0.48535] | inconclusive |

For **pruning, held-out size**, D₀ lowers point-estimate MAE for math; the descriptive improvement interval lies wholly above zero for math. This answers the conditional question for the stated model class and panel.

For **pruning, held-out step**, D₀ lowers point-estimate MAE for math, code; the descriptive improvement interval lies wholly above zero for no capability. This answers the conditional question for the stated model class and panel.

For **quantization, held-out size**, D₀ lowers point-estimate MAE for math, code; the descriptive improvement interval lies wholly above zero for math, code. This answers the conditional question for the stated model class and panel.

For **quantization, held-out step**, D₀ lowers point-estimate MAE for math, code; the descriptive improvement interval lies wholly above zero for code. This answers the conditional question for the stated model class and panel.

A strong raw checkpoint trend alone does not establish a D₀ gain beyond dense L₀. Conversely, failure of this simple B model does not rule out a nonlinear history effect. D₀ and dense L₀ are strongly related during training; coefficient signs should not be interpreted causally. Training-design condition numbers and D₀/L₀ correlations are retained for every fold.

### Every fold (positive improvement favors B)

| Arm | Capability | Axis | Held out | A MAE | B MAE | A−B | B design condition number |
|---|---|---|---|---:|---:|---:|---:|
| pruning | math | size | 160m | 1.04180 | 0.26660 | 0.77519 | 14.20 |
| pruning | math | size | 410m | 0.36748 | 0.06532 | 0.30217 | 3.97 |
| pruning | math | size | 1.4b | 0.43709 | 0.10891 | 0.32818 | 3.07 |
| pruning | math | step | 16000 | 0.80900 | 0.20406 | 0.60493 | 4.33 |
| pruning | math | step | 64000 | 0.10618 | 0.14354 | -0.03735 | 3.69 |
| pruning | math | step | 143000 | 0.93482 | 0.41856 | 0.51626 | 14.19 |
| pruning | code | size | 160m | 1.46607 | 2.23603 | -0.76996 | 12.96 |
| pruning | code | size | 410m | 0.58566 | 0.27479 | 0.31087 | 3.65 |
| pruning | code | size | 1.4b | 0.76644 | 0.48867 | 0.27777 | 3.14 |
| pruning | code | step | 16000 | 0.87860 | 0.27960 | 0.59901 | 3.56 |
| pruning | code | step | 64000 | 0.10246 | 0.11123 | -0.00877 | 3.29 |
| pruning | code | step | 143000 | 1.11884 | 0.50305 | 0.61580 | 7.40 |
| pruning | qa | size | 160m | 1.07227 | 1.04947 | 0.02280 | 1.54 |
| pruning | qa | size | 410m | 0.56219 | 1.13770 | -0.57551 | 4.36 |
| pruning | qa | size | 1.4b | 1.14662 | 1.10763 | 0.03899 | 1.87 |
| pruning | qa | step | 16000 | 0.91724 | 1.02247 | -0.10524 | 1.78 |
| pruning | qa | step | 64000 | 0.39030 | 0.42889 | -0.03858 | 2.21 |
| pruning | qa | step | 143000 | 0.98763 | 0.93751 | 0.05012 | 2.48 |
| quantization | math | size | 160m | 3.43045 | 1.53985 | 1.89060 | 14.20 |
| quantization | math | size | 410m | 1.06915 | 0.25473 | 0.81442 | 3.97 |
| quantization | math | size | 1.4b | 1.58556 | 0.51934 | 1.06622 | 3.07 |
| quantization | math | step | 16000 | 2.57075 | 0.90007 | 1.67069 | 4.33 |
| quantization | math | step | 64000 | 0.33542 | 0.46978 | -0.13436 | 3.69 |
| quantization | math | step | 143000 | 3.06177 | 2.06359 | 0.99818 | 14.19 |
| quantization | code | size | 160m | 5.46418 | 2.69135 | 2.77283 | 12.96 |
| quantization | code | size | 410m | 1.25874 | 0.33478 | 0.92396 | 3.65 |
| quantization | code | size | 1.4b | 1.82775 | 0.77491 | 1.05284 | 3.14 |
| quantization | code | step | 16000 | 2.61369 | 0.50934 | 2.10435 | 3.56 |
| quantization | code | step | 64000 | 0.37423 | 0.22094 | 0.15329 | 3.29 |
| quantization | code | step | 143000 | 3.24288 | 1.14772 | 2.09517 | 7.40 |
| quantization | qa | size | 160m | 2.43593 | 2.32588 | 0.11005 | 1.54 |
| quantization | qa | size | 410m | 1.40639 | 2.65447 | -1.24809 | 4.36 |
| quantization | qa | size | 1.4b | 2.97373 | 2.85237 | 0.12136 | 1.87 |
| quantization | qa | step | 16000 | 2.34927 | 2.70581 | -0.35654 | 1.78 |
| quantization | qa | step | 64000 | 0.92058 | 1.12520 | -0.20461 | 2.21 |
| quantization | qa | step | 143000 | 2.62249 | 2.13714 | 0.48535 | 2.48 |

### Configuration contributions

These are slices of the same primary out-of-fold predictions, with no refitting or removals. They expose int3's contribution and the near-zero high-bit errors; primary comparisons above retain all configurations.

| Arm | Capability | Config | Size A / B / A−B | Step A / B / A−B |
|---|---|---:|---:|---:|
| pruning | math | 0.9 | 0.03484 / 0.02259 / 0.01225 | 0.03409 / 0.02852 / 0.00558 |
| pruning | math | 0.8 | 0.18474 / 0.09434 / 0.09039 | 0.19012 / 0.12380 / 0.06632 |
| pruning | math | 0.7 | 0.58467 / 0.16147 / 0.42320 | 0.59999 / 0.28505 / 0.31494 |
| pruning | math | 0.6 | 1.65758 / 0.30937 / 1.34821 | 1.64246 / 0.58418 / 1.05828 |
| pruning | code | 0.9 | 0.06762 / 0.05629 / 0.01133 | 0.05030 / 0.03061 / 0.01969 |
| pruning | code | 0.8 | 0.33524 / 0.34127 / -0.00602 | 0.24885 / 0.13393 / 0.11492 |
| pruning | code | 0.7 | 1.02473 / 1.40477 / -0.38004 | 0.75742 / 0.48786 / 0.26956 |
| pruning | code | 0.6 | 2.32997 / 2.19700 / 0.13297 | 1.74330 / 0.53944 / 1.20387 |
| pruning | qa | 0.9 | 0.12644 / 0.13205 / -0.00560 | 0.09503 / 0.08936 / 0.00568 |
| pruning | qa | 0.8 | 0.35997 / 0.41375 / -0.05379 | 0.31582 / 0.30907 / 0.00675 |
| pruning | qa | 0.7 | 1.05103 / 1.29245 / -0.24142 | 0.91070 / 0.94993 / -0.03923 |
| pruning | qa | 0.6 | 2.17067 / 2.55482 / -0.38415 | 1.73867 / 1.83680 / -0.09813 |
| quantization | math | 8 | 0.00810 / 0.00506 / 0.00304 | 0.00871 / 0.00647 / 0.00224 |
| quantization | math | 6 | 0.04694 / 0.02368 / 0.02326 | 0.04842 / 0.03137 / 0.01705 |
| quantization | math | 4 | 0.72107 / 0.23281 / 0.48826 | 0.71489 / 0.51893 / 0.19596 |
| quantization | math | 3 | 7.33743 / 2.82368 / 4.51376 | 7.18522 / 4.02114 / 3.16408 |
| quantization | code | 8 | 0.01918 / 0.02061 / -0.00142 | 0.01336 / 0.01479 / -0.00143 |
| quantization | code | 6 | 0.08027 / 0.06478 / 0.01549 | 0.06034 / 0.04478 / 0.01556 |
| quantization | code | 4 | 1.15040 / 1.17455 / -0.02414 | 0.84376 / 0.65575 / 0.18801 |
| quantization | code | 3 | 10.15104 / 3.80812 / 6.34292 | 7.39028 / 1.78867 / 5.60160 |
| quantization | qa | 8 | 0.01585 / 0.01997 / -0.00412 | 0.02224 / 0.02199 / 0.00025 |
| quantization | qa | 6 | 0.12103 / 0.14200 / -0.02097 | 0.11839 / 0.12610 / -0.00770 |
| quantization | qa | 4 | 1.19087 / 1.60496 / -0.41409 | 1.28921 / 1.42944 / -0.14023 |
| quantization | qa | 3 | 7.76032 / 8.67671 / -0.91639 | 6.42660 / 6.37999 / 0.04661 |

## Raw D₀ trend at fixed size and configuration

Fragility here means **signed loss damage**: higher ΔL is worse. Negative ΔL means the measured loss improved under compression. Monotonicity checks both adjacent differences across all three steps (numerical tolerance 10⁻¹², not a measurement-noise test). The smoke's roughly 3–5× claim is checked against all matched trajectories, rather than assumed to hold across capabilities/configurations. Ratios are descriptive and reported only when both early and late damage exceed 0.01; unstable or nonpositive endpoint ratios are marked N/A with raw observations retained. Absolute-damage trajectories and endpoint changes are also recorded separately in summary.json.

| Arm | Capability | Late > early / 12 | Increasing / decreasing / flat / nonmonotonic | Positive-damage ratio median [range]; eligible n |
|---|---|---:|---|---|
| pruning | math | 12 / 12 | 12 / 0 / 0 / 0 | 6.09 [3.15, 10.12]; n=9 |
| pruning | code | 12 / 12 | 12 / 0 / 0 / 0 | 6.67 [2.78, 16.89]; n=9 |
| pruning | qa | 7 / 12 | 3 / 2 / 0 / 7 | 15.01 [12.19, 17.83]; n=2 |
| quantization | math | 11 / 12 | 10 / 0 / 0 / 2 | 8.08 [3.53, 21.76]; n=6 |
| quantization | code | 12 / 12 | 8 / 0 / 0 / 4 | 11.69 [5.43, 21.48]; n=7 |
| quantization | qa | 9 / 12 | 4 / 1 / 0 / 7 | 10.66 [10.18, 64.24]; n=5 |

At pruning config 0.6, the late/early damage ratios are 410m math 4.45× (increasing); 1.4b math 3.15× (increasing); 410m code 6.67× (increasing); 1.4b code 2.78× (increasing).

At quantization config 3, the late/early damage ratios are 410m math 6.51× (increasing); 1.4b math 9.07× (increasing); 410m code 5.43× (increasing); 1.4b code 8.20× (increasing).

The 3–5× smoke description is therefore not a universal magnitude across configurations and capabilities. QA's signed responses and the full three-step trajectories are shown below; a loss improvement is not relabeled as positive damage. Any exact-repeat checkpoint flags qualify the corresponding flat trajectories and aggregate ratios.


### All matched step trajectories

Steps are 16000 → 64000 → 143000. Flags apply if any step is near zero (`Z`), negative (`−`), above 1 nat damage (`H`), has a duplicated checkpoint payload (`C`), or if the configuration is int3 (`3`). All flagged observations enter the primary analysis.

| Arm | Capability | Size | Config | ΔL at three steps | Late−early | Monotonicity | Late/early | Flags |
|---|---|---|---:|---|---:|---|---:|---|
| pruning | math | 160m | 0.9 | 0.00888 → 0.00890 → 0.12457 | 0.11570 | increasing | N/A | Z |
| pruning | math | 160m | 0.8 | 0.07326 → 0.10789 → 0.74128 | 0.66802 | increasing | 10.11822 | none |
| pruning | math | 160m | 0.7 | 0.25706 → 0.44436 → 2.39037 | 2.13331 | increasing | 9.29892 | H |
| pruning | math | 160m | 0.6 | 0.87210 → 1.42177 → 6.49027 | 5.61817 | increasing | 7.44209 | H |
| pruning | math | 410m | 0.9 | 0.00225 → 0.00825 → 0.03411 | 0.03186 | increasing | N/A | Z |
| pruning | math | 410m | 0.8 | 0.02064 → 0.04742 → 0.19665 | 0.17601 | increasing | 9.52586 | none |
| pruning | math | 410m | 0.7 | 0.10061 → 0.19637 → 0.61281 | 0.51220 | increasing | 6.09100 | none |
| pruning | math | 410m | 0.6 | 0.37365 → 0.71237 → 1.66246 | 1.28882 | increasing | 4.44930 | H |
| pruning | math | 1.4b | 0.9 | 0.00105 → 0.00708 → 0.01010 | 0.00906 | increasing | N/A | Z |
| pruning | math | 1.4b | 0.8 | 0.01505 → 0.03838 → 0.07063 | 0.05558 | increasing | 4.69382 | none |
| pruning | math | 1.4b | 0.7 | 0.07811 → 0.19738 → 0.29481 | 0.21670 | increasing | 3.77443 | none |
| pruning | math | 1.4b | 0.6 | 0.36538 → 0.99409 → 1.14965 | 0.78427 | increasing | 3.14644 | H |
| pruning | code | 160m | 0.9 | 0.01361 → 0.01652 → 0.22986 | 0.21625 | increasing | 16.89083 | none |
| pruning | code | 160m | 0.8 | 0.07565 → 0.18481 → 1.09395 | 1.01830 | increasing | 14.46112 | H |
| pruning | code | 160m | 0.7 | 0.27799 → 0.61380 → 3.39036 | 3.11237 | increasing | 12.19602 | H |
| pruning | code | 160m | 0.6 | 1.13365 → 2.00446 → 7.29148 | 6.15783 | increasing | 6.43188 | H |
| pruning | code | 410m | 0.9 | 0.00184 → 0.00618 → 0.00957 | 0.00773 | increasing | N/A | Z |
| pruning | code | 410m | 0.8 | 0.00499 → 0.06216 → 0.21613 | 0.21114 | increasing | N/A | Z |
| pruning | code | 410m | 0.7 | 0.11320 → 0.16817 → 0.80604 | 0.69283 | increasing | 7.12021 | none |
| pruning | code | 410m | 0.6 | 0.34550 → 0.99287 → 2.30550 | 1.96001 | increasing | 6.67303 | H |
| pruning | code | 1.4b | 0.9 | 0.00006 → 0.00398 → 0.02110 | 0.02104 | increasing | N/A | Z |
| pruning | code | 1.4b | 0.8 | 0.01254 → 0.01795 → 0.05152 | 0.03898 | increasing | 4.10900 | none |
| pruning | code | 1.4b | 0.7 | 0.04956 → 0.08533 → 0.17406 | 0.12449 | increasing | 3.51199 | none |
| pruning | code | 1.4b | 0.6 | 0.35399 → 0.97938 → 0.98277 | 0.62877 | increasing | 2.77623 | none |
| pruning | qa | 160m | 0.9 | 0.01209 → -0.04396 → 0.14734 | 0.13525 | nonmonotonic | 12.18673 | − |
| pruning | qa | 160m | 0.8 | -0.02718 → -0.11122 → 0.75665 | 0.78383 | nonmonotonic | N/A | − |
| pruning | qa | 160m | 0.7 | -0.17363 → 0.26675 → 2.89057 | 3.06419 | increasing | N/A | −H |
| pruning | qa | 160m | 0.6 | 0.34847 → 1.52519 → 6.21186 | 5.86339 | increasing | 17.82593 | H |
| pruning | qa | 410m | 0.9 | -0.01646 → -0.01444 → -0.11538 | -0.09892 | nonmonotonic | N/A | − |
| pruning | qa | 410m | 0.8 | -0.03274 → -0.11811 → -0.16683 | -0.13409 | decreasing | N/A | − |
| pruning | qa | 410m | 0.7 | -0.18102 → -0.29004 → 0.12773 | 0.30876 | nonmonotonic | N/A | − |
| pruning | qa | 410m | 0.6 | -0.39134 → -0.24774 → 1.25677 | 1.64811 | increasing | N/A | −H |
| pruning | qa | 1.4b | 0.9 | 0.00986 → -0.02665 → 0.02822 | 0.01836 | nonmonotonic | N/A | Z− |
| pruning | qa | 1.4b | 0.8 | -0.00250 → -0.05980 → -0.06381 | -0.06131 | decreasing | N/A | Z− |
| pruning | qa | 1.4b | 0.7 | -0.14169 → -0.42666 → -0.41394 | -0.27225 | nonmonotonic | N/A | − |
| pruning | qa | 1.4b | 0.6 | -0.31993 → -0.50163 → -0.44519 | -0.12527 | nonmonotonic | N/A | − |
| quantization | math | 160m | 8 | -0.00028 → 0.00000 → 0.03164 | 0.03191 | increasing | N/A | Z− |
| quantization | math | 160m | 6 | 0.00210 → 0.00716 → 0.18777 | 0.18568 | increasing | N/A | Z |
| quantization | math | 160m | 4 | 0.13039 → 0.22157 → 2.83742 | 2.70703 | increasing | 21.76130 | H |
| quantization | math | 160m | 3 | 1.55355 → 3.86182 → 23.03002 | 21.47647 | increasing | 14.82415 | H3 |
| quantization | math | 410m | 8 | 0.00063 → 0.00119 → 0.00275 | 0.00212 | increasing | N/A | Z |
| quantization | math | 410m | 6 | 0.00431 → 0.00799 → 0.02343 | 0.01912 | increasing | N/A | Z |
| quantization | math | 410m | 4 | 0.07884 → 0.11957 → 0.55946 | 0.48062 | increasing | 7.09631 | none |
| quantization | math | 410m | 3 | 1.07684 → 2.88361 → 7.01246 | 5.93562 | increasing | 6.51206 | H3 |
| quantization | math | 1.4b | 8 | -0.00018 → 0.00146 → -0.00127 | -0.00109 | nonmonotonic | N/A | Z− |
| quantization | math | 1.4b | 6 | 0.00075 → 0.00686 → 0.00473 | 0.00397 | nonmonotonic | N/A | Z |
| quantization | math | 1.4b | 4 | 0.07544 → 0.11843 → 0.26627 | 0.19084 | increasing | 3.52975 | none |
| quantization | math | 1.4b | 3 | 0.83101 → 4.41752 → 7.53381 | 6.70280 | increasing | 9.06582 | H3 |
| quantization | code | 160m | 8 | 0.00095 → -0.00155 → 0.05408 | 0.05313 | nonmonotonic | N/A | Z− |
| quantization | code | 160m | 6 | 0.01349 → 0.00351 → 0.26866 | 0.25517 | nonmonotonic | 19.91630 | Z |
| quantization | code | 160m | 4 | 0.16407 → 0.31953 → 3.52359 | 3.35952 | increasing | 21.47591 | H |
| quantization | code | 160m | 3 | 2.12461 → 5.99828 → 24.84104 | 22.71643 | increasing | 11.69203 | H3 |
| quantization | code | 410m | 8 | 0.00053 → -0.00101 → 0.00808 | 0.00755 | nonmonotonic | N/A | Z− |
| quantization | code | 410m | 6 | 0.00256 → 0.00053 → 0.00939 | 0.00683 | nonmonotonic | N/A | Z |
| quantization | code | 410m | 4 | 0.04914 → 0.12081 → 0.67471 | 0.62556 | increasing | 13.72914 | none |
| quantization | code | 410m | 3 | 1.28019 → 3.72296 → 6.94788 | 5.66770 | increasing | 5.42724 | H3 |
| quantization | code | 1.4b | 8 | -0.00166 → -0.00059 → 0.00256 | 0.00422 | increasing | N/A | Z− |
| quantization | code | 1.4b | 6 | -0.00897 → 0.00178 → 0.00909 | 0.01807 | increasing | N/A | Z− |
| quantization | code | 1.4b | 4 | 0.05473 → 0.08700 → 0.30455 | 0.24982 | increasing | 5.56460 | none |
| quantization | code | 1.4b | 3 | 1.06566 → 5.86415 → 8.73502 | 7.66936 | increasing | 8.19679 | H3 |
| quantization | qa | 160m | 8 | -0.00478 → 0.00737 → -0.00927 | -0.00449 | nonmonotonic | N/A | Z− |
| quantization | qa | 160m | 6 | 0.03357 → 0.00440 → 0.35789 | 0.32432 | nonmonotonic | 10.66195 | Z |
| quantization | qa | 160m | 4 | 0.05665 → -0.11478 → 3.63926 | 3.58261 | nonmonotonic | 64.24331 | −H |
| quantization | qa | 160m | 3 | 1.33433 → 3.91124 → 23.95342 | 22.61909 | increasing | 17.95160 | H3 |
| quantization | qa | 410m | 8 | -0.00796 → 0.01046 → 0.01248 | 0.02044 | increasing | N/A | Z− |
| quantization | qa | 410m | 6 | 0.01972 → -0.00974 → -0.00291 | -0.02264 | nonmonotonic | N/A | Z− |
| quantization | qa | 410m | 4 | -0.06197 → -0.13563 → 0.53024 | 0.59221 | nonmonotonic | N/A | − |
| quantization | qa | 410m | 3 | 0.60355 → 2.20901 → 6.14175 | 5.53820 | increasing | 10.17600 | H3 |
| quantization | qa | 1.4b | 8 | -0.00588 → -0.02142 → 0.03654 | 0.04242 | nonmonotonic | N/A | Z− |
| quantization | qa | 1.4b | 6 | -0.01497 → -0.01830 → 0.00662 | 0.02160 | nonmonotonic | N/A | Z− |
| quantization | qa | 1.4b | 4 | 0.03000 → -0.07512 → -0.13881 | -0.16882 | decreasing | N/A | − |
| quantization | qa | 1.4b | 3 | 0.58965 → 4.75502 → 6.02605 | 5.43640 | increasing | 10.21970 | H3 |

## Reproduction and audit trail

```bash
python3 analysis/v36_pythia_controlled_fit.py --dry-run
python3 analysis/v36_pythia_controlled_fit.py
python3 -m pytest -q tests/test_v36.py
```

Only NumPy and the Python standard library are used for analysis. The dry run prints this report without writing. `results/v36-pythia-controlled/summary.json` contains the SHA256 of each of the 18 loss inputs, architecture provenance, all fold memberships, training-only standardizers, direct-fit coefficients/ranks/condition numbers, every observed and held-out predicted response, paired uncertainty, flags and raw trajectories. Inputs are rehashed before outputs are written. This report is generated from that same summary; no results are manually substituted.
