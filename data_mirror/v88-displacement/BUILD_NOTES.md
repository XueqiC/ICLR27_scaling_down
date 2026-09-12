# V88 displacement build notes

Implemented `analysis/v88_displacement.py` and CPU tests in
`tests/test_v88_displacement.py`. No GPU work was run, no CUDA environment
variable was set, and no existing script or frozen JSON was edited. The only
artifact created in the repository's `results/` tree is this file; synthetic
measurement and summary fixtures were written under pytest's `/tmp` directories.

## Measurement and CLI

`run_state(state, configs, probes=None, device="cpu", dtype="bfloat16",
output_dir=..., revision=None, ...)` resolves registry aliases and revisions,
loads a dense model, and deep-copies it for each compression configuration.
Both models remain resident together and use the same dtype and device. Dense
and compressed forwards are paired for every scored token within each sample.
Dense measurements are recomputed for every configuration; no frozen or
previously saved CE enters the displacement calculation.

Supported configuration strings:

- `prune_d0.9` (also `p0.9`): V6 global magnitude pruning. The returned sampled
  threshold, seed, and sampling rule are stored with the configuration. Replaying
  a saved configuration reuses its threshold without sampling again. Density 1
  is the V6 identity case; its internally negative-infinite threshold is stored
  as JSON `null` with the identity rule stated explicitly.
- `b3_g64`: V54 grouped symmetric round-to-nearest quantization.
- `b4_g0` (also `b4`): V10 symmetric per-output-channel quantization.

Transforms operate on the existing language-weight scope, including embeddings
and the LM head. Grouped quantization calls the existing function without a
mode override, preserving its default symmetric arithmetic and padding rule.

Modes, implemented but not used for real model measurements during this build:

```text
python3 -m analysis.v88_displacement --pilot --states pythia-410m@step16000
python3 -m analysis.v88_displacement --full --states STATE1 STATE2 --configs prune_d0.9,b3_g64,b4_g0 --device DEVICE
python3 -m analysis.v88_displacement --summarize
python3 -m analysis.v88_displacement --selftest --output-dir /tmp/UNUSED_DIRECTORY
```

`--pilot` requires exactly one state and two configurations. The default config
pair is pruning at density 0.9 and grouped RTN `(3,64)`. Full mode accepts a list
of states. `--configs-json` accepts a JSON list of configuration dictionaries,
including saved pruning thresholds, for replay into a fresh output directory.
`--probe-set` accepts already-selected measurement samples keyed by capability;
no further split is applied. Without this argument, the code uses V6
`build_probes(n_probe, seed=0)` and selects `[1::2]` for each capability, as V6/V10/V54
do. V6 has a default seed argument rather than a module `PROBE_SEED`; the imported
V54 `PROBE_SEED=0` explicitly names the same seed.

CPU is the default device. Both measurement models default to bf16. Float32
inputs are available for CPU diagnostic models such as the tiny transition
model; the runtime guard still prohibits their fp32 matrix products. Non-CPU
float32 model requests fail before loading. `--selftest` is CPU-only and uses the
existing `TinyModel`, `TinyTokenizer`, and `PROBES`, with bf16 weights and two
configurations. Its outputs have `selftest=true` and are excluded from ordinary
summaries.

## Exact formulas as coded

All the following vocabulary arithmetic is float32, after model forwards.
There are no fp32 matrix products in the reducer. Let `z` and `q` be the native
final dense/compressed logits cast to float32, `r=q-z`, `p=softmax(z)`, and `y`
the reference index. Each formula is evaluated separately for each token:

```text
dense_ce                 = logsumexp(z) - z[y]
compressed_ce            = logsumexp(q) - q[y]
measured_delta           = compressed_ce - dense_ce
mu                      = sum(p*r)
p_r2                    = sum(p*r*r)
first_order             = mu - r[y]
second_order            = 0.5 * max(p_r2 - mu*mu, 0)
second_order_prediction  = first_order + second_order
B                       = z[y] - sum(p*z)       [existing descriptor reducer]
V                       = 1 - sum(p*p)          [existing descriptor reducer]
r_norm2                 = sum(r*r)
z_norm2                 = sum(z*z)
eps_hat                 = -sum(r*z) / z_norm2
s                       = r + eps_hat*z
sigma2_hat              = max(sum(p*s*s) - sum(p*s)**2, 0)
shrinkage_prediction     = eps_hat*B + 0.5*sigma2_hat*V
I                       = indices of min(16, vocab_size) largest abs(r)
top16_p_r2              = sum(p[I]*r[I]*r[I])
top16_r2_share           = top16_p_r2 / p_r2
```

When `z_norm2=0`, the projection is unidentifiable and `eps_hat=0` by convention.
When `p_r2=0`, the diagnostic share is 0. Variances are clamped at 0 and the
diagnostic share to `[0,1]` only to handle roundoff. Any nonfinite statistic
raises an error instead of writing invalid JSON. Projection uses the uncentered,
unweighted Euclidean inner product requested in the contract.

Token values are converted immediately to Python floats. Per-sample records
contain sums and means for every metric, identity/index/hash, and scored-token
counts. Pooled values execute the existing V10 aggregation through
`eval_records.aggregate_records`: summed token quantities divided by total
scored tokens, never a mean of sample means. A zero-token sample has zero sums
and null means; zero-token capabilities retain V10's zero pooled-value convention
and are excluded from summary statistics. Two extra diagnostic aggregates report
the mean share among nonzero displacements and the ratio of pooled top-16 mass to
pooled total mass. These distinguish zero damage and unequal displacement sizes.

### Mathematical inconsistency in requested test (b)

For pure shrinkage `r=-eps*z`, the requested projection recovers `eps` and its
residual variance is zero. However, the exact second-order Taylor polynomial is

```text
eps*B + 0.5*eps**2*Var_p(z)
```

while the stipulated shrinkage prediction is only `eps*B`. Thus the requested
equality to `1e-9` cannot hold for a general nonconstant `z` and nonzero `eps`.
The implementation preserves both requested prediction formulas. The test
checks projection recovery and zero residual, then checks the missing quadratic
term explicitly to `1e-9` in float32 arithmetic. A constant-logit fixture also
checks the requested equality in the case where the quadratic term is zero.
This discrepancy is stated in measurement provenance and both summary formats;
no coefficient was fitted and no prediction term was silently changed.

Likewise, the requested `sigma2_hat` is the **p-weighted residual variance**, not
the coordinate variance of isotropic noise. Before projection,
`E[Var_p(eta)] = sigma_coordinate**2 * V`; projection further removes a direction.
The isotropic-noise test uses a large vocabulary, near-uniform probabilities,
and independent seeded draws, so `V` is near 1 and the distinction is smaller
than its sampling tolerance. The recorded formula remains exactly the one
specified in the task.

The `1e-9` synthetic closed-form check uses dyadic values and uniform
probabilities that are exactly representable in float32. It is not a claim that
arbitrary float32 reductions approximate real-number formulas to `1e-9`.

## Scoring, memory, and matrix-product protection

The frozen `completion_loss` function executes with a private globals dictionary
whose CE function returns shape-only zeros. Its model callback returns an integer
ID view with a singleton dummy vocabulary dimension. This directly reuses the
frozen BOS handling, prompt/reference truncation, shift, and scored-suffix count.
No frozen module global is patched. Real paired scoring then uses those same IDs
and that exact suffix count.

Every scored token is evaluated on its causal prefix with `use_cache=False`.
`TokenForward` either requests one token via the model's explicit last-logit API
or temporarily slices a Linear output head's input to its last hidden state.
The output is always `model(...).logits`, after final softcaps/multipliers. Hooks
are removed even on failure. Unsupported heads fail before their first forward.
The existing `TinyModel` is a context-free transition model, so only its last
input ID is needed; its original forward and final softcap are reused.

This avoids even native bf16 token-by-vocabulary logit allocations. Only one
vocabulary vector from each model and token-local intermediate vectors exist;
none are retained between tokens or serialized. Hidden-state and attention
storage still depend on prefix length, as usual. Prefix recomputation is slower
than a full-sequence forward or cached decoding; it is the explicit cost of
keeping the no-cache semantics and strict vocabulary-memory bound.

The frozen model loader is reused with its sanity callback privately rebound
to a streaming check, preserving its checkpoint-integrity checks without its
otherwise full-sequence sanity logits. The same matrix-product guard applies
to sanity and measurement forwards. Transformers can perform fp32 matrix
products inside RoPE despite bf16 model weights. V88 replaces singleton-axis
outer products with mathematically equivalent elementwise broadcast
multiplication. Other fp32 matrix products fail before PyTorch dispatches the
underlying kernel, on CPU as well as accelerator devices.

## Output and summaries

Each measurement is exclusively created at `<output-dir>/<state-tag>/<config>.json`.
Existing measurement files are refused before model loading; replay requires a
fresh directory. Writes within the repository's `results/` tree are restricted
to `results/v88-displacement/`, including checks for symlink escapes.

Artifacts record the requested and resolved state/revision, dtype/device/device
name, probe SHA-256 (using V54's serialization), config and pruning threshold,
per-capability token counts and pooled quantities, flat per-sample record array,
and provenance with Git commit, actual source-file hashes, software versions,
formulas, forward method, and elapsed time. A small model's named state dictionary
is hashed when it is at most 1 MiB; otherwise a cached single-file HF LFS SHA-256
is used if available without reading weights. The hash basis is explicit; large
or sharded checkpoints without a cheap single hash record null and a reason.

Only `--summarize` writes `summary.json` and `summary.md`. It weights each pooled
state/config/capability measurement equally and divides damage regimes into
`abs(dL)<0.1`, `0.1<=abs(dL)<=0.5`, and `abs(dL)>0.5`. For both predictions it
reports MAE, median `abs(pred-dL)/abs(dL)`, and exact sign agreement with
`sign(0)=0`. Zero-damage rows are excluded from relative errors and counted
explicitly; empty groups report null. Isotropy summaries give mean, median,
minimum, and maximum shares by regime, including nonzero-only and pooled-mass
versions. Large top-16 shares indicate concentration, not proof against isotropy.

Frozen V6 pruning and V54 quantization files are read only. Matching state,
revision, density or bit/group settings produce a reproduction row with frozen
dL, recomputed dL, and `recomputed-frozen` difference. Rows include source paths,
SHA-256s, available probe-hash agreement, and frozen dtype. Missing V6 probe hashes
and realized thresholds remain explicitly unverified. These rows are reproduction
checks, never corrections. Tests check that frozen fixture bytes are unchanged.

## Existing functions reused (source file:line)

- `analysis/model_registry.py:111`: `resolve_model_and_revision`.
- `analysis/model_registry.py:133`: `require_compliant`.
- `analysis/v6_capability_geometry.py:58`: `model_output_tag`.
- `analysis/v6_capability_geometry.py:176`: `load_text_causal_lm`, with private
  streaming sanity callback; underlying loading-integrity and text-adapter logic
  remains unchanged.
- `analysis/v6_capability_geometry.py:263`: `_sample_abs_weights`, called through
  the existing pruning function when no stored threshold was supplied.
- `analysis/v6_capability_geometry.py:283`: `language_weight_parameters`.
- `analysis/v6_capability_geometry.py:289`: `apply_global_magnitude_pruning`.
- `analysis/v6_capability_geometry.py:384`: `build_probes`.
- `analysis/v6_capability_geometry.py:446`: `completion_loss`, executed with a
  private shape-only CE callback for the exact frozen token mask.
- `analysis/descriptor_bv.py:51`: `reduce_final_logits(..., chunk_tokens=1)` for B/V.
- `analysis/v10_quantization.py:73`: `fake_quantize_per_output_channel`.
- `analysis/v54_quant_group.py:152`: `fake_quantize_grouped`, default symmetric mode.
- `analysis/eval_records.py:70`: `aggregate_records`, which executes
  `analysis/v10_quantization.py:157` `_measure_capability_losses`; division is at
  `analysis/v10_quantization.py:179`.
- `analysis/tiny_test_model.py:12` and `:31`: `TinyTokenizer` and `TinyModel`;
  the same module supplies the offline `PROBES`.

## Validation and GPU limits

Required command:

```text
python3 -m pytest tests/test_v88_displacement.py -q
39 passed
```

Tests cover synthetic first/second-order closed forms, the pure-shrinkage
discrepancy, isotropic noise, hand-computed unequal-length pooling, allocation
interception during an entire tiny run, exact frozen token masks, float32
descriptor reductions, output schema/sample counts, reproducible compression,
final softcaps, a real random-config bf16 GPT-NeoX forward without downloads,
RoPE fp32-product interception, config validation, CLI modes/selftest, frozen
reproduction comparisons, empty/zero-damage summaries, and output protection.
The allocation guard also has a negative check proving that it catches a full
native token-by-vocabulary buffer. Tests block CUDA initialization/device queries.

Additional tests for reused components:

```text
MKL_THREADING_LAYER=GNU python3 -m pytest tests/test_capability_geometry.py tests/test_v10.py tests/test_registry.py tests/test_eval_records.py tests/test_descriptor_bv.py tests/test_v54_quant_group.py -q
74 passed in 7.12s
```

The initial additional-suite run had 72 passes and two descriptor subprocess
failures caused by the local Intel-MKL/libgomp threading-layer conflict. Using
the GNU threading layer resolved both, without changing any existing file or any
CUDA setting.

No real checkpoint measurement, dataset download, GPU forward, GPU memory peak,
target-hardware kernel compatibility, or real frozen-dL reproduction was run.
The CPU tests verify the code paths using synthetic inputs, including bf16
Transformers forwards; they cannot certify all architectures or custom/fused
GPU kernels. Full-prefix versus last-token output-head shapes can cause numerical
kernel differences, and frozen V6/V54 native-dtype CE rounding can differ from
V88's mandated float32 CE. The reproduction report exposes resulting differences
instead of adjusting frozen values. Real pilot/full output and scientific MAE
results therefore remain to be generated by a separately authorized measurement
run.
