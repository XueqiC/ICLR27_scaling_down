# Experiment Plan
IDs are stable. Status: PLANNED / RUNNING / DONE / BLOCKED. Compute tiers:
low (rai co-located, <1 GPU-day), medium (rai multi-GPU or 1-2 B200 jobs),
full (B200 batches). Artifacts land under `results/<id>/` on rai with
curated copies here.

## Priority A (audit + existing-data refitting) — no new heavy compute
- A1 artifact audit [DONE 2026-09-01]: inventory + inconsistency list (see PROGRESS).
- A2 pruning-law held-out battery [RUNNING; codex-built v14]: fit shallow ->
  predict deep pre-cliff; leave-largest-model-out; leave-family-out; sign
  prediction; cliff-density prediction; baselines (sparsity, remaining-N,
  removed weight norm, anchor-only, size+density, family-conditioned);
  bootstrap CIs. Success: mechanism/reduced forms beat baselines on MAE.
- A3 quantization functional forms [RUNNING; part of v14]: 4^-b vs learned
  exponential vs polynomial vs EPM; held-out bit width/model.
- A4 recovery refit [RUNNING; part of v14]: saturating form
  dL*(r+(1-r)(1+D/D0)^-beta); D_R=0 anchor check.
- A5 dense reference comparison D1-D4 [PLANNED]: needs Stage-1 grid (E2/E3)
  for (N,D) identification; observed-anchor variant computable now.
- A6 negative-damage significance [RUNNING; part of v14]: benchmark
  bootstrap over measurement halves for all improvement claims.

## Priority B
- B1 geometry ablation controls [PLANNED->next codex]: random 0.05%,
  magnitude-matched random, global-Fisher top, no-residualization,
  fractions {0.01,0.05,0.2}%; models gemma3-1b/4b, Qwen3-1.7B, olmo3-7b.
  Success: capability-selective >> all controls, multi-model.
- B2 block-structure statistics [PLANNED]: example bootstrap + permutation
  test + prompt/template stability for V9 similarities.
- B3 link functions A_c=g_c(L_c) [PLANNED]: fit on compression configs,
  validate on held-out configs; per-benchmark variance.
- B4 third code benchmark + MuSiQue [PLANNED]: extend probe registry;
  freeze prompts/decoding/normalization.
- B5 cliff localization densities (0.025 steps near d*) [PLANNED, medium].
- B6 small-model distillation grid per directive Stage 6 (token-based D_S
  ladder, matched-token answer_only control, 3 seeds small) [PARTIAL:
  12-run pilot running as v12; extension PLANNED].
- B7 recovery grid: models {gemma3-1b, Qwen3-1.7B}, matched-damage
  prune/quant configs, D_R {1M,4M,16M,64M}, sources {c4, traces, mixed},
  LoRA vs full-FT control (small) [PLANNED, medium].
- B8 V11 order parameter: olmo3-7b, gemma3-270m, muse/hpg-class [PARTIAL].

## Priority C
- C1 Pythia dedup (N,D) identification grid [PLANNED, medium; login-node
  downloads + rai small sizes first].
- C2 OLMo-2 ladder (1B/7B/13B/32B) dense refs [PLANNED].
- C3 large-model held-out validation (27B-32B class) [PARTIAL via hpg].
- C4 agent workloads (coding agent on SWE-bench-Verified subset; math
  tool-use; retrieval QA) + G_W calibration + method-selection evaluation
  [PLANNED, full; needs harness design].
- C5 hardware latency measurements [PLANNED].
- C6 M1/M2/M3 cross-method comparison [blocked on A2/A3/B7 outputs].

## BLOCKED
- Qwen3 8B/14B/32B geometry: cluster policy prohibits PRC models on the
  shared cluster; rai-only when GPUs free. Needs: idle 80GB+ rai GPU.
