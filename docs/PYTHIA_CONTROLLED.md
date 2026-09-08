# Pythia controlled training-history analysis (V36)

On the supplied grid, adding D₀ lowers point-estimate MAE in 4 of 12 arm/capability/holdout comparisons; 0 improvement intervals lie wholly above zero. The tables keep the arms and capabilities separate; this count is descriptive and is not a pooled score.

The headline question is whether training history D₀ reduces held-out prediction error for capability **loss** damage beyond model size N₀ and measured dense loss L₀. The target is the observed signed response ΔL_c = L_c(config) − L₀,c. This is a loss endpoint; it does not by itself establish changes in task accuracy.

This is a **fixed-training-recipe SERIES**, using standard (non-deduped) Pythia checkpoints. The suite shares training data/order and token accounting; architectures and hyperparameters still vary across sizes (including learning rate). See the [Pythia paper](https://proceedings.mlr.press/v202/biderman23a/biderman23a.pdf) and [official training documentation](https://github.com/EleutherAI/pythia). Within a size, D₀ also tracks checkpoint age, cumulative optimization and the learning-rate schedule. A conditional prediction gain is evidence about this series, not an isolated causal token effect or a universal compression law.

**Distillation is PENDING and not included:** its training runs are blocked on flaky GPUs. This analysis launches no training or inference.

## Inputs and fixed analysis specification

The 18 existing V6/V10 loss JSON files yield 9 size×step cells per arm and 36 compressed observations per capability/arm (216 total). Pruning uses densities 0.9/0.8/0.7/0.6 with dense key `1.0`; quantization uses bits 8/6/4/3 with dense key `dense`. Each response and predictor L₀ use that file's own measured capability-specific dense reference. The maximum dense-reference discrepancy between arms is 0.00000. Arms and capabilities are fitted and scored separately in nats per native token, with no pooled headline score.

N₀ is computed from official GPT-NeoX architecture configs: `layers × (4h² + 2hm)` for attention QKV/output and the two MLP matrices. It excludes embeddings, the LM head, biases and normalization vectors. These are exact matrix counts, not nominal model names or the paper's slightly larger non-embedding counts including vectors. The dimensions below were transcribed from local cached official configs, verified equal across the three revisions per size; pinned config URLs and file SHA256s are in summary.json. Reproduction needs no weights or config download.

| Size label | Layers | h | m | N₀ matrix parameters |
|---|---:|---:|---:|---:|
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
| pruning | math | step | 0.16534 [0.11483, 0.19752] | 0.17962 [0.10409, 0.23344] | -0.01429 [-0.03592, 0.01073] | inconclusive |
| pruning | code | step | 0.23842 [0.17444, 0.30378] | 0.21689 [0.09708, 0.30071] | 0.02153 [-0.06367, 0.07736] | inconclusive |
| pruning | qa | step | 0.90475 [0.17295, 2.26890] | 0.78448 [0.21911, 1.85351] | 0.12027 [-0.04615, 0.41539] | inconclusive |
| quantization | math | step | 0.78871 [0.46844, 1.22732] | 0.62262 [0.38925, 0.78874] | 0.16609 [-0.11837, 0.53746] | inconclusive |
| quantization | code | step | 0.61390 [0.40428, 0.89691] | 1.35934 [0.35735, 3.35359] | -0.74544 [-2.81310, 0.53956] | inconclusive |
| quantization | qa | step | 2.12044 [0.21989, 5.14867] | 1.48716 [0.42306, 3.55695] | 0.63327 [-0.20317, 1.59172] | inconclusive |

For **pruning, held-out step**, D₀ lowers point-estimate MAE for code, qa; the descriptive improvement interval lies wholly above zero for no capability. This answers the conditional question for the stated model class and panel.

For **quantization, held-out step**, D₀ lowers point-estimate MAE for math, qa; the descriptive improvement interval lies wholly above zero for no capability. This answers the conditional question for the stated model class and panel.

A strong raw checkpoint trend alone does not establish a D₀ gain beyond dense L₀. Conversely, failure of this simple B model does not rule out a nonlinear history effect. D₀ and dense L₀ are strongly related during training; coefficient signs should not be interpreted causally. Training-design condition numbers and D₀/L₀ correlations are retained for every fold.

### Every fold (positive improvement favors B)

| Arm | Capability | Axis | Held out | A MAE | B MAE | A−B | B design condition number |
|---|---|---|---|---:|---:|---:|---:|
| pruning | math | step | 16000 | 0.19752 | 0.23344 | -0.03592 | 24.00 |
| pruning | math | step | 64000 | 0.11483 | 0.10409 | 0.01073 | 13.45 |
| pruning | math | step | 143000 | 0.18367 | 0.20134 | -0.01767 | 19.38 |
| pruning | code | step | 16000 | 0.23704 | 0.30071 | -0.06367 | 11.82 |
| pruning | code | step | 64000 | 0.17444 | 0.09708 | 0.07736 | 16.97 |
| pruning | code | step | 143000 | 0.30378 | 0.25289 | 0.05089 | 109.45 |
| pruning | qa | step | 16000 | 2.26890 | 1.85351 | 0.41539 | 12.98 |
| pruning | qa | step | 64000 | 0.17295 | 0.21911 | -0.04615 | 1.54 |
| pruning | qa | step | 143000 | 0.27239 | 0.28081 | -0.00842 | 1.80 |
| quantization | math | step | 16000 | 1.22732 | 0.68986 | 0.53746 | 24.00 |
| quantization | math | step | 64000 | 0.46844 | 0.38925 | 0.07919 | 13.45 |
| quantization | math | step | 143000 | 0.67037 | 0.78874 | -0.11837 | 19.38 |
| quantization | code | step | 16000 | 0.89691 | 0.35735 | 0.53956 | 11.82 |
| quantization | code | step | 64000 | 0.40428 | 0.36707 | 0.03721 | 16.97 |
| quantization | code | step | 143000 | 0.54050 | 3.35359 | -2.81310 | 109.45 |
| quantization | qa | step | 16000 | 5.14867 | 3.55695 | 1.59172 | 12.98 |
| quantization | qa | step | 64000 | 0.21989 | 0.42306 | -0.20317 | 1.54 |
| quantization | qa | step | 143000 | 0.99275 | 0.48148 | 0.51127 | 1.80 |

### Configuration contributions

These are slices of the same primary out-of-fold predictions, with no refitting or removals. They expose int3's contribution and the near-zero high-bit errors; primary comparisons above retain all configurations.

| Arm | Capability | Config | Size A / B / A−B | Step A / B / A−B |
|---|---|---:|---:|---:|
| pruning | math | 0.9 | 0.01181 / 0.01066 / 0.00114 |
| pruning | math | 0.8 | 0.08375 / 0.05908 / 0.02467 |
| pruning | math | 0.7 | 0.21729 / 0.16963 / 0.04766 |
| pruning | math | 0.6 | 0.34849 / 0.47912 / -0.13063 |
| pruning | code | 0.9 | 0.01456 / 0.01620 / -0.00164 |
| pruning | code | 0.8 | 0.07988 / 0.07661 / 0.00327 |
| pruning | code | 0.7 | 0.33142 / 0.11354 / 0.21788 |
| pruning | code | 0.6 | 0.52782 / 0.66122 / -0.13340 |
| pruning | qa | 0.9 | 0.22892 / 0.25061 / -0.02169 |
| pruning | qa | 0.8 | 0.09325 / 0.06377 / 0.02948 |
| pruning | qa | 0.7 | 0.69312 / 0.62859 / 0.06452 |
| pruning | qa | 0.6 | 2.60372 / 2.19494 / 0.40877 |
| quantization | math | 8 | 0.00249 / 0.00344 / -0.00095 |
| quantization | math | 6 | 0.00641 / 0.00911 / -0.00269 |
| quantization | math | 4 | 0.30539 / 0.19246 / 0.11292 |
| quantization | math | 3 | 2.84056 / 2.28546 / 0.55510 |
| quantization | code | 8 | 0.00689 / 0.00871 / -0.00182 |
| quantization | code | 6 | 0.00807 / 0.02914 / -0.02108 |
| quantization | code | 4 | 0.40666 / 0.14627 / 0.26039 |
| quantization | code | 3 | 2.03397 / 5.25322 / -3.21925 |
| quantization | qa | 8 | 0.04785 / 0.09417 / -0.04632 |
| quantization | qa | 6 | 0.01595 / 0.04703 / -0.03108 |
| quantization | qa | 4 | 1.22495 / 1.12926 / 0.09569 |
| quantization | qa | 3 | 7.19300 / 4.67819 / 2.51481 |

## Raw D₀ trend at fixed size and configuration

Fragility here means **signed loss damage**: higher ΔL is worse. Negative ΔL means the measured loss improved under compression. Monotonicity checks both adjacent differences across all three steps (numerical tolerance 10⁻¹², not a measurement-noise test). The smoke's roughly 3–5× claim is checked against all matched trajectories, rather than assumed to hold across capabilities/configurations. Ratios are descriptive and reported only when both early and late damage exceed 0.01; unstable or nonpositive endpoint ratios are marked N/A with raw observations retained. Absolute-damage trajectories and endpoint changes are also recorded separately in summary.json.

| Arm | Capability | Late > early / 12 | Increasing / decreasing / flat / nonmonotonic | Positive-damage ratio median [range]; eligible n |
|---|---|---:|---|---|
| pruning | math | 8 / 12 | 8 / 0 / 0 / 0 | 4.57 [3.15, 9.53]; n=6 |
| pruning | code | 8 / 12 | 8 / 0 / 0 / 0 | 4.11 [2.78, 7.12]; n=5 |
| pruning | qa | 3 / 12 | 1 / 2 / 0 / 5 | N/A; n=0 |
| quantization | math | 7 / 12 | 6 / 0 / 0 / 2 | 6.80 [3.53, 9.07]; n=4 |
| quantization | code | 8 / 12 | 6 / 0 / 0 / 2 | 6.88 [5.43, 13.73]; n=4 |
| quantization | qa | 6 / 12 | 3 / 1 / 0 / 4 | 10.20 [10.18, 10.22]; n=2 |

At pruning config 0.6, the late/early damage ratios are 410m math 4.45× (increasing); 1.4b math 3.15× (increasing); 410m code 6.67× (increasing); 1.4b code 2.78× (increasing).

At quantization config 3, the late/early damage ratios are 410m math 6.51× (increasing); 1.4b math 9.07× (increasing); 410m code 5.43× (increasing); 1.4b code 8.20× (increasing).

The 3–5× smoke description is therefore not a universal magnitude across configurations and capabilities. QA's signed responses and the full three-step trajectories are shown below; a loss improvement is not relabeled as positive damage. Any exact-repeat checkpoint flags qualify the corresponding flat trajectories and aggregate ratios.


### All matched step trajectories

Steps are 16000 → 64000 → 143000. Flags apply if any step is near zero (`Z`), negative (`−`), above 1 nat damage (`H`), has a duplicated checkpoint payload (`C`), or if the configuration is int3 (`3`). All flagged observations enter the primary analysis.

| Arm | Capability | Size | Config | ΔL at three steps | Late−early | Monotonicity | Late/early | Flags |
|---|---|---|---:|---|---:|---|---:|---|
| pruning | math | 410m | 0.9 | 0.00225 → 0.00825 → 0.03411 | 0.03186 | increasing | N/A | Z |
| pruning | math | 410m | 0.8 | 0.02064 → 0.04742 → 0.19665 | 0.17601 | increasing | 9.52586 | none |
| pruning | math | 410m | 0.7 | 0.10061 → 0.19637 → 0.61281 | 0.51220 | increasing | 6.09100 | none |
| pruning | math | 410m | 0.6 | 0.37365 → 0.71237 → 1.66246 | 1.28882 | increasing | 4.44930 | H |
| pruning | math | 1.4b | 0.9 | 0.00105 → 0.00708 → 0.01010 | 0.00906 | increasing | N/A | Z |
| pruning | math | 1.4b | 0.8 | 0.01505 → 0.03838 → 0.07063 | 0.05558 | increasing | 4.69382 | none |
| pruning | math | 1.4b | 0.7 | 0.07811 → 0.19738 → 0.29481 | 0.21670 | increasing | 3.77443 | none |
| pruning | math | 1.4b | 0.6 | 0.36538 → 0.99409 → 1.14965 | 0.78427 | increasing | 3.14644 | H |
| pruning | code | 410m | 0.9 | 0.00184 → 0.00618 → 0.00957 | 0.00773 | increasing | N/A | Z |
| pruning | code | 410m | 0.8 | 0.00499 → 0.06216 → 0.21613 | 0.21114 | increasing | N/A | Z |
| pruning | code | 410m | 0.7 | 0.11320 → 0.16817 → 0.80604 | 0.69283 | increasing | 7.12021 | none |
| pruning | code | 410m | 0.6 | 0.34550 → 0.99287 → 2.30550 | 1.96001 | increasing | 6.67303 | H |
| pruning | code | 1.4b | 0.9 | 0.00006 → 0.00398 → 0.02110 | 0.02104 | increasing | N/A | Z |
| pruning | code | 1.4b | 0.8 | 0.01254 → 0.01795 → 0.05152 | 0.03898 | increasing | 4.10900 | none |
| pruning | code | 1.4b | 0.7 | 0.04956 → 0.08533 → 0.17406 | 0.12449 | increasing | 3.51199 | none |
| pruning | code | 1.4b | 0.6 | 0.35399 → 0.97938 → 0.98277 | 0.62877 | increasing | 2.77623 | none |
| pruning | qa | 410m | 0.9 | -0.01646 → -0.01444 → -0.11538 | -0.09892 | nonmonotonic | N/A | − |
| pruning | qa | 410m | 0.8 | -0.03274 → -0.11811 → -0.16683 | -0.13409 | decreasing | N/A | − |
| pruning | qa | 410m | 0.7 | -0.18102 → -0.29004 → 0.12773 | 0.30876 | nonmonotonic | N/A | − |
| pruning | qa | 410m | 0.6 | -0.39134 → -0.24774 → 1.25677 | 1.64811 | increasing | N/A | −H |
| pruning | qa | 1.4b | 0.9 | 0.00986 → -0.02665 → 0.02822 | 0.01836 | nonmonotonic | N/A | Z− |
| pruning | qa | 1.4b | 0.8 | -0.00250 → -0.05980 → -0.06381 | -0.06131 | decreasing | N/A | Z− |
| pruning | qa | 1.4b | 0.7 | -0.14169 → -0.42666 → -0.41394 | -0.27225 | nonmonotonic | N/A | − |
| pruning | qa | 1.4b | 0.6 | -0.31993 → -0.50163 → -0.44519 | -0.12527 | nonmonotonic | N/A | − |
| quantization | math | 410m | 8 | 0.00063 → 0.00119 → 0.00275 | 0.00212 | increasing | N/A | Z |
| quantization | math | 410m | 6 | 0.00431 → 0.00799 → 0.02343 | 0.01912 | increasing | N/A | Z |
| quantization | math | 410m | 4 | 0.07884 → 0.11957 → 0.55946 | 0.48062 | increasing | 7.09631 | none |
| quantization | math | 410m | 3 | 1.07684 → 2.88361 → 7.01246 | 5.93562 | increasing | 6.51206 | H3 |
| quantization | math | 1.4b | 8 | -0.00018 → 0.00146 → -0.00127 | -0.00109 | nonmonotonic | N/A | Z− |
| quantization | math | 1.4b | 6 | 0.00075 → 0.00686 → 0.00473 | 0.00397 | nonmonotonic | N/A | Z |
| quantization | math | 1.4b | 4 | 0.07544 → 0.11843 → 0.26627 | 0.19084 | increasing | 3.52975 | none |
| quantization | math | 1.4b | 3 | 0.83101 → 4.41752 → 7.53381 | 6.70280 | increasing | 9.06582 | H3 |
| quantization | code | 410m | 8 | 0.00053 → -0.00101 → 0.00808 | 0.00755 | nonmonotonic | N/A | Z− |
| quantization | code | 410m | 6 | 0.00256 → 0.00053 → 0.00939 | 0.00683 | nonmonotonic | N/A | Z |
| quantization | code | 410m | 4 | 0.04914 → 0.12081 → 0.67471 | 0.62556 | increasing | 13.72914 | none |
| quantization | code | 410m | 3 | 1.28019 → 3.72296 → 6.94788 | 5.66770 | increasing | 5.42724 | H3 |
| quantization | code | 1.4b | 8 | -0.00166 → -0.00059 → 0.00256 | 0.00422 | increasing | N/A | Z− |
| quantization | code | 1.4b | 6 | -0.00897 → 0.00178 → 0.00909 | 0.01807 | increasing | N/A | Z− |
| quantization | code | 1.4b | 4 | 0.05473 → 0.08700 → 0.30455 | 0.24982 | increasing | 5.56460 | none |
| quantization | code | 1.4b | 3 | 1.06566 → 5.86415 → 8.73502 | 7.66936 | increasing | 8.19679 | H3 |
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
