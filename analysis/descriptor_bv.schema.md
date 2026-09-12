# V87 descriptor and evaluation formats (schema version 1)

## Scoring and aggregation

The endpoint is the legacy reference-completion CE used by V6/V10/V12, not
V27's separate reasoning/answer endpoints. `v6_capability_geometry.completion_loss`
handles direct tokenization, Gemma BOS, half-length prompt/reference truncation,
causal shifting and prompt masking. Its returned scored-token count selects the
scored suffix from final `model(...).logits`. No head hooks or duplicate masking
implementation are used. A model's final softcap is therefore included.

V10 `analysis/v10_quantization.py:177-179` accumulates:

```python
total_loss += float(loss)
total_tokens += n_tokens
losses[capability] = total_loss / max(total_tokens, 1)
```

V12 `analysis/v12_distill.py:517-523` uses the same pooled-token rule and raises
on an entirely unscored capability. This is **not** a mean of sample means.
`eval_records.aggregate_records` executes the existing V10 function's code with
an isolated scorer binding, preserving sample accumulation order and its zero
denominator convention. Neither frozen module is edited. Descriptor aggregation
uses this function for NLL, B and V sums, separately within each capability and
distribution. Byte units use the same ratio-of-sums rule with exact scored bytes.

For each scored token, float32 reductions compute
`B_token = z[y] - sum(softmax(z) * z)` and
`V_token = 1 - sum(softmax(z)**2)`. Native logits from one forward batch are
transient; at most `chunk_tokens` rows are promoted to float32 at once. No logits
or probabilities are accumulated across samples, copied to a dataset-sized
buffer, or serialized. The current model forward still allocates its native
sequence logits; this is batch streaming, not a custom transformer decoder.
Legacy NLL remains in the original scorer's arithmetic/dtype for compatibility.

**Sign correction:** writing `CE(z) = logsumexp(z) - z[y]` gives
`d CE((1-eps)z)/d eps = z[y] - sum(p*z) = B`. Thus the requested formula and the
requested *negative* shrinkage derivative cannot both hold. The implementation
keeps the formula and verifies the positive derivative. Equivalently, B is the
negative derivative under **expansion**, `z -> (1+eps)z`. V is the Hessian trace.

## Descriptor JSON

`python -m analysis.descriptor_bv --model MODEL --revision REV --probe-set FILE
--distribution J --dtype bf16 --device cpu --output-dir results/v87-prep/RUN`
uses locally cached models/tokenizers only. `--device` defaults to CPU and rejects
other devices. Input JSON is `{capability: [{prompt, completion, sample_id?,
distribution?}, ...]}`; `id` is also accepted as a sample ID. A sample distribution
overrides the CLI distribution. The list is already the requested scored set:
the descriptor does not silently apply an additional train/measurement split.

Top-level fields in `descriptor_bv.json` and `descriptor_selftest.json`:

| Field | Type / interpretation |
|---|---|
| `schema_version` | Integer 1 |
| `model_id`, `model_revision` | Requested model and explicit revision |
| `resolved_model_revision` | Model config commit hash, or null if unavailable |
| `commit` | Repository HEAD at execution; source hashes also identify uncommitted work |
| `input_hashes` | Canonical probe SHA256; CLI also records raw probe-file and tokenizer-vocabulary SHA256 |
| `code_sha256` | Exact source hashes of descriptor, record adapter, frozen scorer/aggregator and byte helper |
| `device`, `max_len`, `chunk_tokens` | Execution/scoring configuration |
| `probability_reduction_dtype` | Always `torch.float32` |
| `aggregation_rule`, `byte_aggregation_rule` | Explicit ratio-of-sums rules |
| `logit_source`, `shrinkage_identity` | Final-output location and correct sign identity |
| `per_sample` | Ordered array of the records described below; includes unscored inputs with count zero |
| `aggregates` | `{capability: {distribution: aggregate}}` |
| `selftest` | Selftest output only: numeric oracles, errors, tolerances and status |

Each per-sample record contains:

- `sample_id` (supplied ID, otherwise content hash), `sample_index` (within
  capability), `capability`, `distribution`.
- `prompt_sha256`, `reference_sha256`, `prompt_reference_sha256`: SHA256 of
  canonical compact UTF-8 JSON, using sorted keys and `ensure_ascii=False`.
  Duplicate content retains a shared hash for later clustering; indices retain
  the original repeated sample contributions.
- `summed_nll`, `scored_token_count`, `reference_byte_count` (full original UTF-8
  reference, before truncation).
- `B_sum`, `V_sum`, and token-normalized `B`, `V`, `L`.
- `scored_reference_byte_count`: UTF-8 bytes corresponding to the actual scored
  token prefix, obtained through V27's `target_byte_count`; `byte_count_error`
  records why exact accounting is unavailable. Full-reference bytes are never
  substituted for truncated bytes. If a tokenizer cannot prove the exact prefix
  (including a partial Unicode token), bytes and byte-normalized values are null.
- `B_per_byte`, `V_per_byte`, `L_per_byte` and forward/reduction dtype strings.

Each aggregate has `B`, `V`, `L`, their `_per_byte` variants, `n_samples`,
`n_scored_samples`, `scored_token_count` and `scored_reference_byte_count`.
If any contributing sample lacks exact byte accounting, the entire group's
byte aggregate is null (no silent cohort change). An empty scored cohort has
zero token aggregates following V10, and null byte aggregates. Unscored sample
means are null. Per-sample sums and identities support future clustered intervals;
no intervals or capability claims are estimated here.

## Additive evaluation records

Frozen entry points remain unchanged. For new runs use:

```bash
python -m analysis.eval_records distill --distribution J --output-dir results/v87-prep/distill <V12 arguments>
python -m analysis.eval_records quantize --distribution J --output-dir results/v87-prep/quantize <V10 arguments>
```

These adapters default to CPU and reject other devices. They preserve the frozen
scorers, loops and existing scalar fields. V12 final, dense-baseline trajectory,
and intermediate trajectory files automatically acquire records when written.
V54 calls the recording evaluator directly in both modes. There is no backfill
of historical results and no change to frozen V6/V10/V12 source. For CPU training,
the adapter preserves Python/NumPy/CPU Torch RNG state and skips optimizer CUDA
health queries after checking all parameters are on CPU. Original GPU protocols
remain available through their unchanged frozen entry points.

The new key is `per_example_v87`, with fields `schema_version`,
`aggregation_rule`, `reference_bytes_rule` and
`evaluations: {existing_loss_key: [sample_record, ...]}`.
V12 and V54 put this key at top level. V10 puts it **inside `dense`**, since V17
casts every top-level non-dense key to a numeric bit width. Every original
capability/scalar field is preserved. Resume retains previously recorded bit
measurements. Scored records include identity/hash/distribution fields, summed
NLL, token count and full-reference byte count as listed above; zero-token
examples are omitted. Default distributions are MATH-500, MBPP and 2WikiMultihopQA.
The full-reference byte count in these records is provenance, not authorization
to compute truncated per-byte CE; use descriptor scored-byte accounting for that.

`read_eval(path)` returns legacy or new JSON without inventing absent records.
For new files it verifies positive token counts, finite NLL and agreement with
stored aggregates at absolute tolerance `1e-9`. `aggregate_records(records)`
recomputes capability losses in the project's exact accumulation order.

## Grouped RTN JSON

V54 adds `--mode symmetric|asymmetric` (default `symmetric`) and `--output-dir`.
Existing config keys `bBITS_gSIZE` and symmetric protocol strings stay intact.
Modes cannot be merged within one directory; the protocol guard rejects them.
Completed historical configs are not rewritten/backfilled.

Symmetric tensor quantization delegates to the original V54 implementation,
including input-dtype scales/arithmetic, ties-to-even rounding, zero-group safe
divisors, zero padding and g0/None's V10 path. Asymmetric uses float32 arithmetic:
include zero in each group's min/max, `scale = (max-min)/(2**bits-1)`, integer
`zero_point = clamp(round(-min/safe_scale), 0, 2**bits-1)`, then
`q = clamp(round(w/safe_scale)+zero_point, 0, 2**bits-1)` and
`w_hat = ((q-zero_point)*scale).to(input_dtype)`. Zero-only groups use safe divisor
one and realised scale zero. Partial groups' padding does not affect the range.

Each `_meta.configs.CONFIG.quantization_statistics.PARAMETER` contains:

- `step`, `zero_point`: min/mean/max/population-standard-deviation over groups of
  the realised scale and integer zero point (zero in symmetric mode).
- `clipping_rule`, `fraction_weights_clipped` and
  `fraction_clipped_per_group` summary moments. Clipping counts rounded codes
  actually changed by the clamp, not ordinary rounding. Padding is excluded.
- `rms_rounding_error_per_group` summary moments and `rms_rounding_error` pooled
  over real weights, measured after dequantization/cast to the input dtype.
- `n_groups`, `n_weights`, `arithmetic_dtype`, `mode`, `statistics_weighting`.

Statistics are per-tensor aggregate group summaries, never per-weight arrays.
The stored realised scale is the quantizer's scale; bf16 casting can make adjacent
represented weight differences nonuniform. Quantization metadata describes fake
quantization, not packed model size or runtime acceleration.
