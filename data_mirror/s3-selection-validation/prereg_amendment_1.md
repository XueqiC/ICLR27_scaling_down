# S3 preregistration amendment 1 — registered before outcomes

This amendment supersedes only the decisions below in `prereg.md` and the blocked
preparation described in `PRINTED_PREPARATION.md`. Both documents, `plan.json`, its
sidecar, the frozen probe file, and `inputs/locked_models.json` remain byte-identical.
`plan_v2.json` is the operative registration. Its initial status is
`REGISTERED_WAITING_FOR_ANCHORS`; numeric prediction/map seals are null until the
CPU freeze succeeds. No student, compressed outcome, or A12 artifact is used to
choose these decisions. No fit, GPU work, submission, or commit occurs in S3b.

1. **Reference roster.** Exactly four deployment references, with twelve candidates
   each, in the following fixed order. Independence remains a condition to verify,
   not a claim established by different revision labels.

   | Task | Reference | New student initialization | Revision(s) |
   |---:|---|---|---|
   | 0 | EleutherAI/pythia-410m | EleutherAI/pythia-160m | both `step120000` |
   | 1 | EleutherAI/pythia-1.4b | EleutherAI/pythia-410m | both `step120000` |
   | 2 | google/gemma-3-1b-pt | google/gemma-3-270m | reference `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`; student `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1` |
   | 3 | google/gemma-3-4b-pt | google/gemma-3-1b-pt | reference `cc012e0a6d0787b4adcc0fa2c4da74402494554d`; student `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29` |

   Drop the Pythia-2.8B candidate slot entirely because of the recorded learned-tensor
   identity problem (`results/v77-model-arch/weight_identity.json`,
   `analysis/v77_model_arch.py:pythia_tensor_identity`). Its excluded historical
   revisions remain comparison inputs to the identity audit, never candidates.
   The packet pins Pythia by step labels, without immutable commit IDs; the identity
   report must record the exact resolved commits and file/tensor hashes before anchors.
   The Pythia-410M and Gemma-1B bases are intentionally shared between a reference
   role and another reference's student initialization. These are not additional
   independent snapshots. Shared teacher traces and probe texts are not replications.

2. **New-student recipe.** Replace the corner-style 200-row/100k-supervision recipe
   with the historical V12 recipe on which the V78 locked KD branch was fitted.
   The fixed recipe is documented by `analysis/v39_distill_controlled.py` (module
   contract and `load_cells`), `scripts/run_pythia_lora_grid.sh`, and the nine
   `results/v12-distill/pythia-{160m,410m,1.4b}--step{16000,64000,143000}/gpt-5.6-luna_full_600_lora/eval.json`
   records. V78's locked fit excludes its two registered reused students; their
   recipe records can verify protocol, but their outcomes are not refitted or reused.

   | Setting | Registered value | Source |
   |---|---|---|
   | Teacher and trace files | `gpt-5.6-luna`; the same three `results/traces-pilot/gpt-5.6-luna_{math,qa,code}.jsonl` files, sealed by hash | V39 module contract; V12 `TRACE_BASE`, `load_sft_records`; grid launcher |
   | Domains and training benchmarks | `math,qa,code`; GSM8K, HotpotQA, CodeAlpaca | `analysis/v12_distill.py:DOMAINS,TRAINING_BENCHMARKS` |
   | Pool and recipe | first 600 nonblank source rows/domain; `full` teacher completion, omit empty completions; identical trace preparation | V12 `load_sft_records`, `apply_recipe`; historical eval `data_selection` |
   | Pool selection seed | `data_seed=None` (omit `--data-seed`); **not** randomized sampling with seed 0 | V12 `load_sft_records`, CLI default; grid launcher |
   | Training and shuffles | training seed 0; initial and per-epoch shuffle seed 0 (`random.Random(0+epoch)`) | V12 `SEED`, `load_sft_records`, `_train`; historical eval |
   | Budget / selected checkpoint | two complete epochs; final eval only; no schedule-token override, trajectory selection, early stopping, or 100k token target | V39 contract; grid launcher; V12 CLI and `_train` |
   | Learning rate and optimizer | 1e-4, AdamW; inherited defaults betas=(.9,.999), eps=1e-8, weight_decay=.01; no gradient clipping | V12 `LORA_LR`, `_train` (`torch.optim.AdamW(parameters, lr=learning_rate)`); installed torch signature |
   | LR schedule | cosine, warmup `int(.03 * total_updates)`; `total_updates=2*ceil(retained_examples/16)` | V12 `WARMUP_RATIO`, `_train` |
   | Batch / objective | effective 16 sequences; one sequence per forward/backward, mean sequence losses accumulated as `loss/len(group)`; partial final groups retained | V12 `EFFECTIVE_BATCH_SIZE`, `_train`, `masked_causal_loss` |
   | Tokenization | max length 1024, at most 512 prompt + 512 completion tokens; prompt masked; completion has no added special tokens; Gemma single-BOS guard | V12 `MAX_LEN`, `tokenize_sft_example` |
   | Precision | bf16 initial weights; PEFT's existing adapter handling; no precision override | V12 `run_distillation` loader and `configure_training` |
   | Training mode / LoRA | strict LoRA, r=16, alpha=32, dropout=0, bias=none; no full-finetuning fallback | V12 `configure_training`, `resolve_training_mode`; historical adapter configs |
   | Pythia targets | `query_key_value,dense,dense_h_to_4h,dense_4h_to_h` | V12 `lora_target_modules` |
   | Gemma targets | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | V12 `LORA_TARGET_MODULES`, `lora_target_modules` |
   | Evaluation | frozen 128 probes/capability, seed 0, odd half 64; completion-token-weighted CE; MATH-500, MBPP, 2Wiki; native-token nats | V12 `measure_capability_losses`; V6 `completion_loss`; original S3 probe hash |
   | Output | new S3 directory per reference, pristine pinned base; final adapter and all three fresh evaluations | `analysis/s3_run.py:students`; original S3 exclusion rule |

   CPU tokenization registers the pool hash, retained count, processed and supervised
   totals, 2-epoch update count and warmup for each student in `plan_v2.json`. Pythia's
   pool hash must match all nine historical recipe records. LONI identity must replay
   the pool using the actual downloaded student tokenizer before anchors. Training
   and scoring enforce the registered final totals; they do not select a checkpoint
   by observed loss. No corner accounting code or A12 output enters the new recipe.

   **Exact-domain limitations.** Pythia matches the historical recipe and same-stage
   construction. Step120000 is a new stage for testing, not a fitted student outcome.
   Gemma cannot match the Pythia architecture, tokenizer, exact projection names,
   token totals per epoch, or a registered common pretraining stage/Ds. Its LoRA
   targets follow V12's existing family-specific convention and its pool/budget use
   the same source rows/two epochs, but this is cross-family transfer. The fallback
   below resolves missing inputs; it does not prove in-domain Gemma coverage.
   Historical library/hardware versions and all implicit PEFT defaults are not fully
   recorded, so bitwise historical optimization equivalence cannot be claimed.
   The same V12 code/defaults are sealed, and no undocumented retuning is permitted.

3. **KD prediction inputs.** For both Pythia reference/student pairs,
   `Ds = Dref = 120000 * (1024 * 2048) = 251658240000` pretraining tokens.
   This is `analysis/v36_pythia_controlled_fit.py:TOKENS_PER_STEP`, used directly by
   `analysis/v39_distill_controlled.py:load_cells`. Math evaluates the frozen +Ds
   linear branch at `(log Ns, student dense math loss, log Ds)` with its stored
   centering/scaling/coefficient arrays. Code/QA use the frozen arithmetic means.
   For **both Gemma pairs**, Ds remains null; math uses the already frozen
   `models.locked.distill.constant.math = 0.0666998113015672`, added to the pristine
   student's math loss. Code adds `0.1804815104078237`; QA adds
   `-0.4803817558392179`. These are the existing V78 arithmetic-mean coefficients,
   not new estimates. `analysis/s3_freeze.py` implements this explicit Gemma override;
   `analysis/final_rule.py` and `inputs/locked_models.json` are not changed. No zeros
   or invented pretraining Ds substitute for missing data. QA remains 2Wiki only.

4. **Anchors and numeric freeze.** Run identity first. The new one-GPU, two-hour
   `scripts/loni_s3_anchors.slurm` measures only pristine reference/student dense
   losses on frozen probes, with no compression or training. There are eight role
   records under `anchors/`, representing six unique snapshots; identical shared
   snapshots are evaluated once and explicitly reused as dense inputs. Every anchor
   binds registration, probe, identity report, commit, learned tensors, tokenizer,
   family/units, token counts and SLURM job ID. A changed or partial anchor set fails.

   CPU `analysis/s3_freeze.py` requires the sealed identity report and all eight
   sealed role records. It verifies current target snapshot files, evaluates all
   12 candidates × 4 references × 3 capabilities (144 losses), and separately stores
   48 largest-increase values `max_c(Lhat_candidate,c - Ldense_reference,c)`.
   It freezes both selectors over all 17 budgets: 272 reference/target/budget cells
   per policy, with explicit null infeasibility. It writes `predictions_sha256` and
   `maps_sha256` into `plan_v2.json`, refreshes its file sidecar and `SHA256SUMS_v2`,
   and preserves the prior registered bytes as `plan_v2.registered.json` plus sidecar.
   It never reads compressed/student outcomes, fits a coefficient, changes a budget,
   or removes a missing cell. Re-freeze and freezing after any S3 student/panel output
   directory exists are refused. Student and panel execution require and recompute
   the complete numeric seal before any model loading.

5. **Identity.** CPU `analysis/s3_identity.py` resolves exact snapshots, verifies all
   indexed shards, and fingerprints all learned parameters against a meta-device
   architecture schema. Wrapper and tied-weight aliases are normalized; disagreeing
   tied aliases are rejected; saved non-learned buffers are excluded. Floating values
   use a common exact float64 representation with negative zero converted to positive
   zero. Learned tensors cannot silently disappear. Serialized file SHA256 alone is
   insufficient. Each of six unique target snapshots is compared against **every one
   of the 30 historical revisions in the original `plan.json` exclusions** (180
   comparisons), including different shapes and the dropped 2.8B history. Distinct
   target snapshots must also differ from one another; intentional reuse of the same
   pinned snapshot in two roles is recorded rather than counted as independent.
   Missing historical snapshots or equal learned values fail closed, without a
   reduced-history fallback. Original exclusions do not expand to unrelated studies;
   this audit establishes independence relative to that registered roster only.

6. **Execution and concurrency.** Sync the v2 scripts, transitive predictor imports,
   packet, traces and tests via `scripts/s3_sync_list.txt`. Students use array
   `0-3%3`, one GPU/task; panels use one GPU and must depend on successful completion
   of the whole student array. The panel dispatcher independently checks `sacct`
   records for all four completed tasks using `S3_STUDENTS_JOB_ID`. Thus at most
   three S3 GPUs run concurrently. Anchors finish before CPU freeze; no student or
   transformed-model forward is allowed before the seal. All jobs are offline after
   download. Scoring is CPU-only after all twelve candidates per reference exist.

7. **Unchanged endpoints and validation.** Preserve twelve candidates, nominal
   matrix storage, budgets .20:.05:1.00, V64 tie ordering, largest-increase definition,
   per-reference/family reporting and the complete 4×16 paired decision denominator.
   Quant-only remains undefined at 20%; null is not zero. Preserve original support
   criterion and report opportunity separately from quant-only selector regret.
   CPU tests exercise numeric branches, complete maps, identity normalization,
   missing/altered inputs, immutable registration and execution gates. Real trainer
   and panel parsers plus shell/self checks run without weights, GPU, or submission.

Remaining operational unknowns are LONI cache completeness, resolved Pythia commits,
actual learned-tensor independence against every excluded snapshot, tokenizer and
pool compatibility, installed library support, GPU memory/time sufficiency, and
unmeasured dense losses. These are gates for the stated operations, not permission
to change the roster, recipe, predictors or endpoints after outcomes. If a gate
fails, stop and record the failure; no replacement reference or fallback is chosen.
The separate `LONI_OPERATIONS_v2.md` gives executable commands in the registered order.
