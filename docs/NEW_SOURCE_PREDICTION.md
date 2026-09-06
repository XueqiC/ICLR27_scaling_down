# New-source prediction with frozen coefficients (V28, plan item 4)

The basic-parameter prediction target is

\[
\widehat L_c^{(m)}=L_{c,\mathrm{ref}}+
F_{m,c}(x_{\mathrm{dense}},u_{\mathrm{shared}},u_m).
\]

For pruning and quantization, the reference is the **source's own dense
checkpoint**, not a smaller dense model. For distillation it is the **student's
own dense checkpoint** S0. The target endpoint is reference-completion loss,
not accuracy. This analysis reads saved development JSON on CPU; it neither
loads model weights nor measures or trains a target model.

The artifact at `results/v28-new-source-pred/frozen_predictions.json` freezes
the mappings, preprocessing, formulas, evaluation coordinates, calibration
coordinates, and residual intervals. Target dense losses have not been
supplied. Consequently pruning/quantization entries are **conditional affine
functions of dense L_c**, with null numeric predictions. Distillation delta
predictions are numeric; their absolute-loss anchors remain pending. None of
these entries is a measured held-out result. The final independent test still
requires the declared dense panel and the later held-out compression runs.

## Information budgets

| Mode | Target inputs | Target compressed configurations | Interpretation |
|---|---|---:|---|
| A: NO-compression-calibration | Metadata, pre-declared dense L_c, requested coordinate and fixed recipe | 0 | Basic-parameter prediction |
| B: FEW-compression-calibration | Mode A information plus one pre-specified target compressed configuration | 1 **per arm** | One-point calibration transfer |

One configuration provides math/code/QA losses, not three separate
configurations. Testing Mode B across both pruning and quantization therefore
costs two configurations in total. Its calibration point is excluded from the
evaluation coordinates. Mode A never receives a calibration object. Mode B's
single amplitude is fitted independently per capability; it does not identify
a target-specific pruning exponent, quantization cliff, or distillation
data/size curvature.

Dense/basic inputs to the pruning and quantization mappings are log(N0/1e9),
family and dense L_c. N0 consistently means **non-embedding language matrix
parameters**: the V6 two-dimensional language-weight count minus token
embeddings and a separately stored output head. This excludes one-dimensional
norm/bias parameters and non-language towers; it is not an exact count of all
non-embedding checkpoint tensors. It never falls back to the nominal model
label or an embedding-inclusive count. This operational convention must also
be used for a custom source's `--source-n0`.

`configs/v28_source_metadata.json` records dimension sources, embedding/head
exclusions and derived N0 for all 12 development models. Qwen3 8B has
N0=6,945,767,424 under this matrix convention, calculated from its architecture.
Gemma3 multimodal configurations omit vocabulary size; their documented
configuration-class default is 262,208, distinct from the 262,144 used by the
270M/1B text configurations. These embedding dimensions are accounted for.

Training stage and disclosed pretraining tokens are metadata only. **D0 is
candidate-only and is not a regression input**: these sparse commercial size
ladders cannot distinguish N from D0 effects. Unknown disclosures are not
imputed. Distillation D below is adaptation traces per domain, not D0 and not
tokens. The reduced distillation response currently has no identified student
size/family term; recorded student metadata does not imply such an effect was
estimated.

## Development fits and frozen forms

All models in the existing 12-model V6 panel are retained. The panel is a fixed
allowlist; future directories and archive copies do not expand it. Pruning uses
only the pre-declared densities 0.9/0.8/0.7/0.6. Quantization coefficient labels
use bits 8/6/4; development bit 5 is used for scoring, not label fitting.
There is no observed-loss, cliff, or sign filter. This broader prospective
eligibility differs from V18's outcome-filtered pre-cliff comparisons.

For each capability and each leave-one-model-out fold, the script fits all
shape parameters, model coefficient labels, feature standardizers, family
frequencies and ridge coefficients using the other **11 entire models**.
The held-out model contributes only its own arm-specific dense anchor to
Mode A. Its compressed outcomes score predictions and, for Mode B only, the
one declared point calibrates its amplitude. The final mapping is then fitted
on all 12 development models and frozen for the new source. Regularization
strength is fixed at lambda=1; the intercept is unpenalized. Family indicators
are centered on training frequencies; an unseen family's correction is zero.
No tuning or coordinate change is made after inspecting these LOMO results.

Pruning uses

\[
\widehat{\Delta L}_c(d)=\widehat a_c(x)
\left(\frac{1-d}{0.3}\right)^{\gamma_c}
=\widehat A_c(x)(1-d)^{\gamma_c}.
\]

Gamma is shared across development models within a capability and fitted in
[0.05,8] by profiling signed per-model amplitudes. The ridge mapping predicts
a=A(0.3)^gamma, a numerical reparameterization of A. It allows signed changes
and anchors delta(1)=0. It cannot represent every nonmonotone curve or a cliff.
Mode B retains the development gamma and sets a from the target's **density
0.9** delta alone. Predictions are frozen for **0.8/0.7/0.6**.

Quantization uses

\[
\widehat{\Delta L}_c(b)=\widehat q_c(x)(4^{-b}-4^{-16}).
\]

Per-model signed q labels are least-squares fits on development 8/6/4-bit
points. The ridge mapping predicts q/1024 for numerical conditioning. Mode B
sets q from **bit 6** alone. Predictions are frozen for **bits 5/4**, under the
existing V10 symmetric per-output-channel fake weight quantizer. This does not
claim transfer to GPTQ or a different quantizer. Pruning retains the V6 global
sampled-threshold magnitude recipe, which acts on language matrices including
embeddings/head even though N0 excludes those matrices.

Distillation uses the V25 LoRA-only panel: Gemma3 270M/1B/4B, D=75/150/300/600,
excluding 4B/D600 because that cell uses full training. The exclusion is based
on recipe, not outcome. Per-capability forms are fixed from prior V25 evidence:

\[
\widehat\delta_{\mathrm{math}}(D)=\mu_{\mathrm{math}},\quad
\widehat\delta_{\mathrm{QA}}(D)=\mu_{\mathrm{QA}},\quad
\widehat\delta_{\mathrm{code}}(D)=\beta_{\mathrm{code}}\log(1+D/150),
\qquad D>0.
\]

At D=0 the response is defined as zero; the math/QA constants are local
corrections for the positive training budgets, not continuous small-D laws.
Their coefficients are fitted on development outcomes relative to each run's
own dense student anchor. No API teacher parameter count is invented. LOMO
holds out all budgets of one student; Mode B uses that student's D=150 point
to set its response scale. LOMO evaluation scores D=75/300/600 where available,
always excluding D=150 in both modes. The form choice used prior evidence on
this panel, so these development scores are retrospective rather than fresh
model-selection validation.

The frozen untested configuration is **Gemma3 12B, D=300 traces/domain,
gpt-5.6-luna teacher, full trace recipe, LoRA**. The loader verifies the
development training logs and adapters: rank 16, alpha 32, dropout 0, two
epochs, seed 0, learning rate 1e-4, AdamW/cosine, warmup ratio .03, effective
batch 16, maximum length 1024, bf16. These are declarations for a later run;
V28 does not execute training. At freeze time the corresponding saved eval
file is absent. This is student-size extrapolation beyond 4B with an in-range
data budget. Cross-family or alternative-teacher configurations remain outside
the demonstrated development scope.

## Intervals and calibration cost

For each arm/capability/mode/coordinate, the frozen interval is the predicted
delta plus the 2.5th/97.5th percentiles of signed LOMO residuals. These are
**descriptive prediction intervals**, based on 12 source models or only three
student sizes. They have no finite-sample coverage guarantee and omit
item/seed uncertainty, dense-measurement uncertainty and unrepresented
distribution shift. Residual intervals can be asymmetric and need not contain
the point estimate. They are not confidence intervals for a coefficient.
No coverage claim is made from reusing these residuals on the same panel.
For a custom distillation budget absent from development, pooled residual
offsets are explicitly flagged; that does not validate extrapolated coverage.

Mode A/B MAEs use identical evaluation cells, separately per capability.
The gain is MAE_A minus MAE_B, with 2,000 paired model-bootstrap draws of fixed
out-of-fold errors. A positive gain means the single calibration point helped.
Zero change and a development mean coefficient are reported on the same cells.
One-point calibration is not assumed to help. In particular, density 0.9
calibration amplifies tiny changes under the fitted power and produces severe
pruning errors; the point and failure are retained. Quantization code has a
positive paired gain on this panel; its math/QA gains remain unresolved. QA
Mode A does not beat zero change here. These outcomes constrain the claim;
they are not evidence of successful new-source validation.

## Target identity, measurement and replay

The command defaults retain the requested name `Qwen3-8B`, HF ID
`Qwen/Qwen3-8B`, stage `Base`, family `qwen3`, and report the identity conflict.
The [official Qwen3-8B card](https://huggingface.co/Qwen/Qwen3-8B) labels that
HF checkpoint as pretrained and post-trained and identifies
`Qwen/Qwen3-8B-Base` as its base. Therefore the **paper artifact explicitly uses
`--source-hf Qwen/Qwen3-8B-Base`** to match the requested Base experiment.
The [Base configuration](https://huggingface.co/Qwen/Qwen3-8B-Base/blob/main/config.json)
has the architecture used for the non-embedding matrix count. This checkpoint
choice does not introduce any target outcomes. Dense and calibration records
must match the frozen HF ID and declare a checkpoint revision; a pinned
`--source-revision` or `--student-revision` is enforced when supplied at freeze.

The dense panel is declared before compression: math=MATH-500, code=MBPP,
QA=2WikiMultihopQA, seed 0, 128 requested probes, odd-indexed 64 examples per
capability. Use the V6 reference-completion span and token-weighted aggregation
(sum target NLL / sum target tokens), not item-average CE or the later V27
per-byte endpoint. This is a forward-only dense measurement with no Fisher,
gradient or geometry calculation. Native-token losses differ across
tokenizers; this freeze does not solve the cross-tokenizer unit problem. The
historical V6/V10 dense discrepancies are retained arm by arm. Saved aggregates
also do not certify exact historical revisions or scored-text hashes; those
limitations persist.

Inspect without writing:

```bash
python analysis/v28_new_source_prediction.py --dry-run
python analysis/v28_new_source_prediction.py --source-hf Qwen/Qwen3-8B-Base --dry-run
```

The second command without `--dry-run` created the paper artifact. Writes use
exclusive creation: rerunning it cannot overwrite the freeze. A different
explicit `--output` creates a separately timestamped artifact. The commit hash
identifies the base checkout; SHA-256 hashes identify the actual code,
metadata and data inputs, including untracked work. The artifact has its own
content hash. These records are local provenance, not externally certified
preregistration.

Supply real measurements later using a JSON object with the schema below.
**Nulls are placeholders and are rejected until measured values are supplied.**
`student` can be omitted when only source dense measurements are available.

```json
{
  "protocol_id": "v6-v10-v12-reference-completion-native-token-odd64-seed0",
  "source": {
    "model": "Qwen3-8B",
    "hf_id": "Qwen/Qwen3-8B-Base",
    "revision": "SUPPLY_CHECKPOINT_REVISION",
    "losses": {"math": null, "code": null, "qa": null}
  },
  "student": {
    "model": "gemma3-12b",
    "hf_id": "google/gemma-3-12b-pt",
    "revision": "SUPPLY_CHECKPOINT_REVISION",
    "losses": {"math": null, "code": null, "qa": null}
  }
}
```

```bash
python analysis/v28_new_source_prediction.py \
  --apply-frozen results/v28-new-source-pred/frozen_predictions.json \
  --dense-json path/to/measured_dense.json --dry-run
```

Without `--dry-run`, this writes a separate `bound_predictions.json`. Replay
evaluates only the serialized affine functions and interval offsets; it reads
no development data and performs no refit. Dense measurements already recorded
inside a freeze cannot be replaced. Calibration is optional and only accepted
when applying an existing freeze. Extra compressed fields in a dense JSON are
rejected.

A Mode B calibration JSON has the same `protocol_id` and optional per-arm
objects named `pruning`, `quantization`, `distillation`. Each object contains
exactly `model`, `hf_id`, `revision`, `coordinate`, `losses` as above, with
coordinate **0.9**, **6**, or **150**, respectively. Its identity/revision must
match the relevant source or student dense record. There is one object per
arm, not a list of points. Pass it with `--calibration-json path/to/calibration.json`
and use a distinct `--output` for the Mode B report. Mode A predictions remain
identical when calibration is supplied.

Custom sources require `--source`, `--source-hf`, `--source-family`, and
`--source-n0`; the new source cannot alias any development checkpoint.
Custom student tags use `--student`, `--student-hf`, and `--student-budget`;
the reduced response remains restricted by its recorded training scope.
`--pretraining-tokens` records a disclosure without adding a fitted D0 effect.

Validation: `python -m pytest tests/test_v28_new_source_prediction.py tests/test_prediction_audits.py -q`.
Tests include coefficient recovery on explicitly synthetic fixtures, outer-fold
leakage perturbations, two-mode input separation, immutable formula replay,
checkpoint identity/revision checks, signed losses, missing inputs, zero
boundaries, metadata counting, no-write dry runs and no torch/transformers
imports. Empirical artifacts use only saved development measurements.

## Frozen tables

The following tables are rendered directly from the saved JSON. Prediction
coefficients in JSON retain full precision; displayed coefficients are rounded.

Freeze UTC: `2026-09-06T15:51:23.960151+00:00`; base commit: `26e2c08a74353d16fedc0ed363d49cb271050fd1`; artifact content hash: `cc8788cf0addea1a94b24d17ba6e6e24e6d5b4cc91fd5feed2f846edcfa59eb2`.

| Arm | Mode | Capability | Coordinate | Predicted delta (nats) / frozen function | 95% interval |
|---|---|---|---:|---|---|
| pruning | A | math | 0.8 | 0.0558172 +0.0185953 L_dense | function + [-0.203526, 0.85786] |
| pruning | A | math | 0.7 | 0.37222 +0.124004 L_dense | function + [-0.89043, 3.31511] |
| pruning | A | math | 0.6 | 1.4304 +0.476534 L_dense | function + [-3.57948, 8.77551] |
| pruning | B | math | 0.8 | -25.6266 L_dense +25.6266 L_cal | function + [-2.74058, 0.420334] |
| pruning | B | math | 0.7 | -170.892 L_dense +170.892 L_cal | function + [-28.1764, 1.80984] |
| pruning | B | math | 0.6 | -656.721 L_dense +656.721 L_cal | function + [-140.933, 11.0939] |
| pruning | A | code | 0.8 | 0.5903 -0.37355 L_dense | function + [-0.357433, 0.750871] |
| pruning | A | code | 0.7 | 2.70856 -1.71402 L_dense | function + [-1.42743, 3.71409] |
| pruning | A | code | 0.6 | 7.9836 -5.05214 L_dense | function + [-3.91814, 6.87717] |
| pruning | B | code | 0.8 | -13.5246 L_dense +13.5246 L_cal | function + [-0.366515, 0.402376] |
| pruning | B | code | 0.7 | -62.0572 L_dense +62.0572 L_cal | function + [-3.11393, 2.19224] |
| pruning | B | code | 0.6 | -182.916 L_dense +182.916 L_cal | function + [-14.8254, 8.34742] |
| pruning | A | qa | 0.8 | 0.190807 -0.0270458 L_dense | function + [-0.31409, 0.828525] |
| pruning | A | qa | 0.7 | 1.37467 -0.194852 L_dense | function + [-1.30695, 2.72658] |
| pruning | A | qa | 0.6 | 5.58057 -0.791014 L_dense | function + [-5.83453, 11.6098] |
| pruning | B | qa | 0.8 | -29.2473 L_dense +29.2473 L_cal | function + [-6.22013, 3.10411] |
| pruning | B | qa | 0.7 | -210.712 L_dense +210.712 L_cal | function + [-53.7354, 23.13] |
| pruning | B | qa | 0.6 | -855.402 L_dense +855.402 L_cal | function + [-244.293, 95.0258] |
| quantization | A | math | 5 | -0.10411 +0.207189 L_dense | function + [-0.159691, 0.152525] |
| quantization | A | math | 4 | -0.41644 +0.828758 L_dense | function + [-0.471408, 1.27927] |
| quantization | B | math | 5 | -4 L_dense +4 L_cal | function + [0.00116899, 0.0692802] |
| quantization | B | math | 4 | -16 L_dense +16 L_cal | function + [0.0504434, 1.25494] |
| quantization | A | code | 5 | 0.183594 -0.102995 L_dense | function + [-0.159338, 0.363671] |
| quantization | A | code | 4 | 0.734378 -0.411982 L_dense | function + [-0.572416, 1.9724] |
| quantization | B | code | 5 | -4 L_dense +4 L_cal | function + [-0.0088697, 0.321398] |
| quantization | B | code | 4 | -16 L_dense +16 L_cal | function + [-0.096683, 1.80331] |
| quantization | A | qa | 5 | -0.140731 +0.0204601 L_dense | function + [-0.268715, 0.279488] |
| quantization | A | qa | 4 | -0.562925 +0.0818403 L_dense | function + [-0.66207, 1.40826] |
| quantization | B | qa | 5 | -4 L_dense +4 L_cal | function + [-0.28077, 0.396897] |
| quantization | B | qa | 4 | -16 L_dense +16 L_cal | function + [-0.659577, 1.78745] |
| distillation | A | math | 300 | 0.0902573 | [0.0741569, 0.174864] |
| distillation | B | math | 300 | -1 L_dense +1 L_cal | function + [0.00500265, 0.0280994] |
| distillation | A | code | 300 | 0.127583 | [0.112584, 0.206233] |
| distillation | B | code | 300 | -1.58496 L_dense +1.58496 L_cal | function + [0.00425387, 0.0338941] |
| distillation | A | qa | 300 | -1.14018 | [-1.4615, -1.01235] |
| distillation | B | qa | 300 | -1 L_dense +1 L_cal | function + [-0.0224586, 0.469692] |

LOMO development evaluation; errors in nats. Each A/B pair scores identical cells. Positive gain means one calibration point helped.

| Arm | Capability | Models | A MAE | B MAE | Zero MAE | Dev mean MAE | A−B MAE (95% paired model bootstrap) |
|---|---|---:|---:|---:|---:|---:|---|
| pruning | math | 12 | 1.41473 | 9.29039 | 1.56887 | 1.53428 | -7.87567 [-20.49633, -0.87153] |
| pruning | code | 12 | 1.29818 | 1.80860 | 1.57652 | 1.60451 | -0.51042 [-1.62912, 0.47888] |
| pruning | qa | 12 | 1.91500 | 34.49935 | 1.69267 | 2.07631 | -32.58434 [-53.65222, -14.24359] |
| quantization | math | 12 | 0.26366 | 0.22355 | 0.38813 | 0.31600 | 0.04011 [-0.02162, 0.11184] |
| quantization | code | 12 | 0.33618 | 0.27463 | 0.38799 | 0.35781 | 0.06155 [0.01174, 0.11935] |
| quantization | qa | 12 | 0.35673 | 0.40873 | 0.28700 | 0.32930 | -0.05200 [-0.25069, 0.11055] |
| distillation | math | 3 | 0.03215 | 0.03186 | 0.08816 | 0.03215 | 0.00029 [-0.00580, 0.00657] |
| distillation | code | 3 | 0.03722 | 0.02613 | 0.10762 | 0.06564 | 0.01109 [-0.00023, 0.01835] |
| distillation | qa | 3 | 0.25716 | 0.30301 | 1.05610 | 0.25716 | -0.04584 [-0.13235, 0.06495] |

central 95% empirical signed LOMO residual offsets, separately per arm/capability/mode/coordinate. Descriptive prediction intervals; 12 source models or 3 student sizes, no finite-sample coverage guarantee, no item/seed/dense-measurement uncertainty. Distillation unseen budgets use all LOMO budget residuals with explicit extrapolation flag.

NOT added: sparse commercial size ladders cannot separate N from D0 effects. Disclosed tokens/stage are recorded as metadata, not fitted covariates; no identifiable independent token/stage effect is established.

