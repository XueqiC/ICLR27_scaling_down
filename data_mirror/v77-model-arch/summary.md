# V77 model architecture and weight identity audit

CPU/config/meta inspection only; weight files are streamed for hashes and never loaded as tensors. The 12 original heterogeneous models are identified by v51's `cohort=panel` and checked against v6/v9/v10 result files. All six registered Pythia sizes and two later Qwen3 prospective additions are also reported; the three Gemma-3 students are already panel members.

**Panel MoE models:** none. **Models missing cached config:** none.

**V9 coverage gap:** Gemma-3-27B and OLMo-3-32B have empty v9 result directories. Both have completed v6/v10 results and remain members of the original 12-model panel.

## Parameter conventions

- **total_parameters:** Text LM sum(p.numel() for p in model.parameters()) inside accelerate.init_empty_weights, before explicitly restoring tied weights. This reproduces v50's literal meta counting procedure, restricted to text_config.
- **total_unique_parameters:** Same text LM on meta after tie_weights() outside init_empty_weights; shared embedding/head Parameter counted once. Includes embeddings, head, norms and biases; excludes vision/audio/projectors and buffers.
- **active_parameters_per_token:** Dense: total_parameters. MoE: total_parameters minus all routed expert weights plus k/E of routed expert weights per layer. Router and always-on shared dense MLP stay in the non-expert count. This is a parameter convention, not FLOPs or the number of embedding rows accessed.
- **active_unique_parameters_per_token:** Same active formula with total_unique_parameters.
- **transformer_matrix_parameters:** Decoder-block matrix/tensor parameters (rank >= 2), excluding embeddings, LM head, biases and normalization vectors. Pythia: exactly v53/v64 N0 = L*(4*h*h + 2*h*m). For other architectures the corresponding actual block matrices are summed; this is an extension, not an assertion that v53 fitted them.
- **v50_frozen_total_parameters:** Unchanged results/v50-p2v2/freeze.json refs.N: literal full-config meta count. Gemma-3-4B includes vision/projector parameters; tied embedding/head weights are counted twice in all three students.
- **v64:** V64 uses v53.matrix_n0 for Pythia sources and students; it is a transformer-matrix convention, not v50's full-model meta total.

These counts expose, but do not change, the existing laws or their frozen covariates. The literal meta counts must not be presented as independent physical weights when the embedding and LM head are tied. No model size is inferred from its marketing name.

## Architecture table (exact integer counts)

| Model | Cohort | Text architecture | Type | Total meta | Active meta | Unique text | Transformer matrices |
|---|---|---|---|---:|---:|---:|---:|
| gemma3-270m | heterogeneous_12 | Gemma3ForCausalLM | dense | 435,870,336 | 435,870,336 | 268,098,176 | 100,270,080 |
| gemma3-1b | heterogeneous_12 | Gemma3ForCausalLM | dense | 1,301,875,840 | 1,301,875,840 | 999,885,952 | 697,761,792 |
| gemma3-4b | heterogeneous_12 | Gemma3ForCausalLM | dense | 4,551,515,648 | 4,551,515,648 | 3,880,263,168 | 3,208,642,560 |
| gemma3-12b | heterogeneous_12 | Gemma3ForCausalLM | dense | 12,772,912,896 | 12,772,912,896 | 11,766,034,176 | 10,758,389,760 |
| gemma3-27b | heterogeneous_12 | Gemma3ForCausalLM | dense | 28,418,976,512 | 28,418,976,512 | 27,009,346,304 | 25,598,361,600 |
| gemma4-31b | heterogeneous_12 | Gemma4ForCausalLM | dense | 32,106,631,424 | 32,106,631,424 | 30,697,345,280 | 29,286,727,680 |
| muse-30b | heterogeneous_12 | MuseGlimmerTextModel + Linear LM head | dense | 27,854,780,928 | 27,854,780,928 | 27,854,780,928 | 25,163,726,848 |
| olmo3-7b | heterogeneous_12 | Olmo3ForCausalLM | dense | 7,298,011,136 | 7,298,011,136 | 7,298,011,136 | 6,476,005,376 |
| olmo3-32b | heterogeneous_12 | Olmo3ForCausalLM | dense | 32,233,522,176 | 32,233,522,176 | 32,233,522,176 | 31,205,621,760 |
| pythia-6.9b | pythia | GPTNeoXForCausalLM | dense | 6,857,302,016 | 6,857,302,016 | 6,857,302,016 | 6,442,450,944 |
| pythia-1b | pythia | GPTNeoXForCausalLM | dense | 1,011,781,632 | 1,011,781,632 | 1,011,781,632 | 805,306,368 |
| pythia-160m | pythia | GPTNeoXForCausalLM | dense | 162,322,944 | 162,322,944 | 162,322,944 | 84,934,656 |
| pythia-410m | pythia | GPTNeoXForCausalLM | dense | 405,334,016 | 405,334,016 | 405,334,016 | 301,989,888 |
| pythia-1.4b | pythia | GPTNeoXForCausalLM | dense | 1,414,647,808 | 1,414,647,808 | 1,414,647,808 | 1,207,959,552 |
| pythia-2.8b | pythia | GPTNeoXForCausalLM | dense | 2,775,208,960 | 2,775,208,960 | 2,775,208,960 | 2,516,582,400 |
| Qwen3-0.6B | heterogeneous_12 | Qwen3ForCausalLM | dense | 751,632,384 | 751,632,384 | 596,049,920 | 440,401,920 |
| Qwen3-1.7B | heterogeneous_12 | Qwen3ForCausalLM | dense | 2,031,739,904 | 2,031,739,904 | 1,720,574,976 | 1,409,286,144 |
| Qwen3-4B | heterogeneous_12 | Qwen3ForCausalLM | dense | 4,411,424,256 | 4,411,424,256 | 4,022,468,096 | 3,633,315,840 |
| Qwen3-8B | prospective_addition | Qwen3ForCausalLM | dense | 8,190,735,360 | 8,190,735,360 | 8,190,735,360 | 6,945,767,424 |
| Qwen3-14B | prospective_addition | Qwen3ForCausalLM | dense | 14,768,307,200 | 14,768,307,200 | 14,768,307,200 | 13,212,057,600 |

## Frozen Gemma-3 student convention

| Student | V50 frozen full-config meta | Text-only meta | Unique text | Non-text in frozen count |
|---|---:|---:|---:|---:|
| gemma3-270m | 435,870,336 | 435,870,336 | 268,098,176 | 0 |
| gemma3-1b | 1,301,875,840 | 1,301,875,840 | 999,885,952 | 0 |
| gemma3-4b | 4,971,331,952 | 4,551,515,648 | 3,880,263,168 | 419,816,304 |

## Cached MoE outside the panel

`google/gemma-4-26B-A4B-it` is absent from the registry and original v6/v9/v10 panel. Its cached `text_config` explicitly enables MoE: 128 routed experts, top_k_experts=8. The implementation also executes one always-on dense MLP per layer; no explicit shared-expert count appears in the config.

Text meta total 25,971,339,264; active meta 4,560,728,064; unique total 25,233,141,760; active unique 3,822,530,560. Routed expert parameters 22,837,985,280; selected expert parameters 1,427,374,080; non-expert meta parameters 3,133,353,984. This supplementary cache check does not add it to the panel.

## Pythia identity

Audited 23 cached revisions covering 22 measured pruning/quantization states. Pythia-2.8B step16000 and step143000 select distinct safetensors blobs; step64000 shares the step143000 blob. No other selected weight-set coincidences were found. See [weight_identity.md](weight_identity.md) and [weight_identity.json](weight_identity.json) for every full SHA-256 and provenance limit.

**Critical:** the full learned-tensor audit finds that all three Pythia-2.8B revision labels have the same parameter values modulo signed zero. Between step16000 and step143000, 387/388 parameter tensors are byte-identical; the remaining tensor differs only at one +0.0/-0.0 entry. Extra serialized buffers also change the file hash. V72's two distinct blob hashes do not establish two independent training states.

## Validation and artifacts

- Every original panel model's language matrix count matches its v6 Fisher metadata.
- All six Pythia matrix counts match the exact v53 formula; all three frozen v50 student totals reproduce exactly.
- Every selected blob is hashed from bytes and checked against its LFS address; v72's three cached Pythia-2.8B selections match frozen provenance.
- All 388 learned Pythia-2.8B parameter tensors are compared across the two distinct serialized blobs; the only payload difference is the sign bit of one zero.
- `model_arch.json` retains full cached/resolved text configs, top-level architecture classes, parameter inventories, implementation source hashes, snapshot commits and result evidence.
- `paper/paper/tables/model_arch.tex` and `paper/analysis/v77_model_arch.py` are staged under this results directory to respect the requested write boundary. No commit is made.
