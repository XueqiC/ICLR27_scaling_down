# V79 audit of V76 capability conditioning

CPU audit using frozen V76 predictions and historical development inputs. No V76 files were modified. All gains below use **MAE(B) minus MAE(A)** (positive favors A); literal MAE(A)-MAE(B) has the opposite sign.

## 1. Variant B scales and fitting constraints

From `results/v76-cap-conditioning/summary.json`, `arms.<arm>.fit.scales`, reproduced from development inputs:

| Arm | Math scale | Code scale | QA scale | Negative allowed? | Negative occurred? |
|---|---:|---:|---:|:---:|:---:|
| Pruning | 1.287773394985 | 1.270558982252 | 0.327893977614 | Yes | No |
| Quantization | 1.069770291520 | 1.431305776417 | 0.556504983556 | Yes | No |
| Distillation | 0.716184413646 | 0.734225554903 | 1.549590031451 | Yes | No |

Negative scales are allowed by the stored specification and actual code, but none occurred in any primary arm or stored sensitivity fit. Pruning has no output clipping. Quantization alone keeps V69's boundary zero floor after extrapolation; that is not a scale sign constraint.

The actual `corrections` implementation at `analysis/v76_cap_conditioning.py:116` uses `scale = x @ y / (x @ x)` in its OLS branch. `fit_anchors` passes the shared anchors and capability median anchors with equal anchor weight. There is no intercept, clipping of the scale, ridge penalty, or sign bound. The source hash matches the one recorded in V76's summary. Executing that same function on known negative targets returned scales `(2, -3, -4)`. All stored sensitivity scales are also positive (full values in JSON).

## 2. Pruning gain attribution

Fit the signed multiple `s_c h(d)` through the origin on the seven development median anchors. The reported variance fraction is centered `R² = 1 - Σ(m_c-s_c h)² / Σ(m_c-mean(m_c))²`; no extra intercept is fitted. The uncentered quantity uses `Σm_c²` in the denominator and is shown only to remove ambiguity about through-origin regression conventions.

| Capability | Centered R² | Uncentered R² | Median sign pattern, increasing density | Five-label gain | Four-state gain |
|---|---:|---:|---|---:|---:|
| math | 0.938239 | 0.962321 | + + + + + + + | 0.018977 | 0.004829 |
| code | 0.973857 | 0.982947 | + + + + + + + | 0.009219 | 0.003967 |
| qa | 0.489380 | 0.494705 | + − − − − − − | 0.189877 | 0.168254 |

Density order: 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9.

| Density | Shared h | Math median | Code median | QA median |
|---:|---:|---:|---:|---:|
| 0.55 | 2.130193820 | 2.643666456 | 2.617869028 | 0.873975166 |
| 0.6 | 0.737332315 | 1.126384165 | 1.132606370 | -0.146179895 |
| 0.65 | 0.183206172 | 0.684212608 | 0.478874495 | -0.452337809 |
| 0.7 | 0.021570279 | 0.275933323 | 0.180324459 | -0.180400428 |
| 0.75 | -0.037508106 | 0.112522740 | 0.059097932 | -0.289522933 |
| 0.8 | 0.004003691 | 0.055643439 | 0.045816496 | -0.075020794 |
| 0.9 | -0.002034750 | 0.007514039 | 0.007873782 | -0.015446768 |

**QA changes sign across density:** its median is positive at 0.55, then negative at every observed density from 0.60 through 0.90. Linear interpolation crosses zero at d=0.592835. The shared curve has a different sign pattern; a single positive or negative scale cannot reproduce the QA curve. Low-density positive anchors dominate its OLS projection, giving a positive QA scale despite predominantly negative QA medians.

Per-source QA direction variability uses the **arithmetic mean**, not the median, at each density. Each available development source state receives one vote; opposite means `response * mean < 0`.

| Density | Sources | QA mean | QA median | Positive / negative / zero | Opposite mean | Share |
|---:|---:|---:|---:|---:|---:|---:|
| 0.55 | 8 | 1.550317104 | 0.873975166 | 7 / 1 / 0 | 1/8 | 12.5% |
| 0.6 | 16 | 0.584834690 | -0.146179895 | 6 / 10 / 0 | 10/16 | 62.5% |
| 0.65 | 8 | 0.089201372 | -0.452337809 | 1 / 7 / 0 | 7/8 | 87.5% |
| 0.7 | 16 | 0.051449620 | -0.180400428 | 4 / 12 / 0 | 12/16 | 75.0% |
| 0.75 | 4 | -0.269917419 | -0.289522933 | 0 / 4 / 0 | 0/4 | 0.0% |
| 0.8 | 16 | 0.013324694 | -0.075020794 | 2 / 14 / 0 | 14/16 | 87.5% |
| 0.9 | 16 | 0.000317476 | -0.015446768 | 6 / 10 / 0 | 10/16 | 62.5% |

The historical grid is unbalanced (4-16 sources per density); do not impute missing source-density pairs or add a dense anchor. Median-curve shape is an aggregate over the available sources.

A majority can oppose the arithmetic mean when a few large positive responses dominate it. This occurs here at 0.60, 0.65, 0.70, 0.80 and 0.90. The small positive mean at 0.90 is reported with its actual sign; no significance threshold or zero tolerance is imposed. Individual response values and source IDs are in JSON, for all three capabilities.

The measured A-over-B gain is primarily a non-proportional aggregate QA curve shape (including a density sign reversal), not a prohibited negative scale. B already minimizes signed OLS over its entire one-scale function family. Math/code curves are much closer to proportional; their gain intervals include zero.

QA directions vary across development sources, and its arithmetic mean is positive at several densities with a negative median/majority. A and B are both source-free: neither adapts its sign to a source at fixed density. These diagnostics establish heterogeneity, but do not causally decompose its contribution to the aggregate curve or confirmation gain. The observed A-B contrast measures extra curve flexibility; it does not demonstrate learned source-specific direction.

No nonnegativity, regularization, clipping, or optimizer-bound artifact in pruning. B is structurally restricted to proportional curves; OLS versus confirmation MAE is a criterion choice, not a sign constraint. V76's other fitting conventions are sensitivity analyses, not refits on confirmation labels.

QA accounts for 87.07% of the original summed capability gains and 95.03% after deduplication. These are descriptive shares of the observed gain, not a causal variance decomposition.

## 3. Four-cluster pruning confirmation

Distinct serialized blob hashes do not imply distinct learned weights. The one differing learned entry is +0.0/-0.0; extra checkpoint buffers also differ. These two labels represent one measured state cluster.

The stored V77 full-tensor audit (`results/v77-model-arch/pythia_2_8b_tensor_identity.json`) checks all 388 learned tensors: 387 are byte-identical, and the remaining layer-normalization bias has one +0.0/-0.0 difference. Its blob identities match the V72 freeze. V79 checks this stored evidence and independently verifies that the three paired measurement files and all V76 A/B/C/D predictions are identical across labels. It does not reload model weights.

Count each physical state-density-capability once: merge identical cells within the 2.8B cluster, retaining step143000 as representative. 36 rows, 12 state-density cells, four equally weighted states. This is also the mean error within each cluster followed by equal-state averaging.

All frozen predictions are retained unchanged in JSON; the representative rows are a subset of them. Use V76's 5,000 paired state-bootstrap resamples, NumPy PCG64 seed 0, and 95% percentile intervals. A sampled state brings every density and capability, using the same draws across methods/capabilities. The intervals condition on the fixed development fit and probes and are descriptive with only four states.

| Capability | A MAE, 5 labels | B MAE, 5 labels | Gain, 5 labels [95% CI] | A MAE, 4 states | B MAE, 4 states | Gain, 4 states [95% CI] |
|---|---:|---:|---|---:|---:|---|
| math | 0.198322 | 0.217299 | 0.018977 [-0.029150, 0.068998] | 0.227924 | 0.232753 | 0.004829 [-0.049485, 0.059144] |
| code | 0.149631 | 0.158850 | 0.009219 [-0.011634, 0.027267] | 0.173590 | 0.177556 | 0.003967 [-0.018129, 0.022823] |
| qa | 0.158475 | 0.348352 | 0.189877 [0.125212, 0.254542] | 0.181788 | 0.350042 | 0.168254 [0.105439, 0.243418] |
| macro | 0.168809 | 0.241500 | 0.072691 [0.029958, 0.115424] | 0.194434 | 0.253450 | 0.059017 [0.020555, 0.102507] |

Macro gain decreases by 0.013674 nats (18.81%); relative MAE reduction changes from 30.10% to 23.29%. The qualitative A-versus-B reading is unchanged: QA and macro intervals remain positive; math and code intervals include zero. The macro interval shifts downward and becomes slightly narrower; deduplication need not widen every percentile interval because it also changes the empirical state distribution. These numbers exactly reproduce V76's already-stored `unique_outcome_sensitivity`. V79 establishes the weight-identity basis for treating it as the four-state comparison. V72 alone has one unique cluster, so no between-state bootstrap interval can be estimated.

For completeness, all V76 contrasts after the same merge (positive baseline-minus-A):

| Capability | B-A [95% CI] | C-A [95% CI] | D-A [95% CI] |
|---|---|---|---|
| math | 0.004829 [-0.049485, 0.059144] | -0.003062 [-0.019498, 0.024800] | -0.060175 [-0.177102, 0.056753] |
| code | 0.003967 [-0.018129, 0.022823] | 0.001419 [-0.019273, 0.042803] | -0.063835 [-0.134873, 0.007204] |
| qa | 0.168254 [0.105439, 0.243418] | 0.280853 [0.185778, 0.345059] | 0.434749 [0.354158, 0.500279] |
| macro | 0.059017 [0.020555, 0.102507] | 0.093070 [0.084212, 0.102096] | 0.103580 [0.052812, 0.154348] |

A separate **cluster-only sensitivity** avoids conflating dependence correction with reweighting: Keep all 45 rows and only pair the two aliases in one cluster. Four bootstrap units, but the duplicate state retains double point-estimate weight. This preserves original gains and changes only their intervals.

| Capability | Gain with original 45-row weighting, 4 clusters [95% CI] |
|---|---|
| math | 0.018977 [-0.049485, 0.064618] |
| code | 0.009219 [-0.018129, 0.025844] |
| qa | 0.189877 [0.105439, 0.257539] |
| macro | 0.072691 [0.020555, 0.113170] |

Scanned all 30 freeze/compare-named files under `results/` using `rg --files --hidden --no-ignore`, then matched both full tags (including `--` path variants); a broader check for separated `2.8b`, `step16000`, and `step143000` fields found the same files:

- `results/v72-prune-repeat/compare.json` (first tag matches at lines 192, 30).
- `results/v72-prune-repeat/freeze.json` (first tag matches at lines 18, 6).

Freeze has two target labels; compare reports six state-density configurations, 18 capability rows, and two by_state entries. V72 treats them as two states; its compare has no state-bootstrap interval. Covariate-dependent power/A2 predictions can differ across labels even though measurements agree; the V76 A/B/C/D predictions being audited agree exactly.

V76 pooled confirmation counts five state labels and bootstraps them separately; its unique_outcome_sensitivity already counts these aliases once.

No other freeze/compare file under results/ matched both labels. V77's identity files discuss the aliases as audit evidence, not independent performance states. This is a literal-tag/separated-field inventory, not proof against unnamed reuse.

## 4. Distillation interpretation

**Distillation A==B equivalence is algebraic for the single-coefficient forms and supports no conclusion about conditioning. With w=log(1+E), a_c=(w^T y_c)/(w^T w), a=(w^T mean_c(y_c))/(w^T w), and a!=0, s_c=((a*w)^T y_c)/((a*w)^T(a*w))=a_c/a. Thus A=a_c*w=a*s_c*w=B. The equality is guaranteed by parameterization; it is not empirical evidence that conditioning helps, is unnecessary, or generalizes.**

The fitted shared coefficient is 0.211228292536, so the nonzero-shared-coefficient condition holds. Maximum frozen prediction difference A-B is 1.39e-16 nats (floating-point rounding).

## Reproduction and outputs

Run `python -B analysis/v79_cond_audit.py --selftest`, then `python -B analysis/v79_cond_audit.py`. Use `--check` to recompute and verify all output bytes without writing. The audit reproduces all 216 V76 frozen capability rows, checks historical input hashes, and verifies that every V76 file is unchanged before and after the run. Input SHA-256 values, raw signed responses, source identities, scale code evidence, and complete scores are recorded in `summary.json`.

Paper artifacts are staged under this results directory to obey the explicit write-only boundary:

- [Table, with `[H]`](paper/paper/tables/cond_audit.tex), for `paper/paper/tables/cond_audit.tex`.
- [Byte-identical code mirror](paper/analysis/v79_cond_audit.py), for `paper/analysis/v79_cond_audit.py`.
- No files in the top-level paper tree are changed; no commit is made.
