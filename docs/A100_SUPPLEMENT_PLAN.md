# A100_SUPPLEMENT_PLAN — bounded single-GPU supplement (2026-09-09)

Authorization: one NVIDIA A100 80GB PCIe on rai (idx3, `GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0`, verified
free at 15:15 EDT; the other cards hold labmates' vLLM engines / tc-alignment and are not used). Total budget
**24 GPU-hours** (model compute/train/eval; downloads and CPU analysis excluded). Historical evidence frozen;
v38/v40/v41 registers immutable. No hpg. Packages: P0 (CPU), P1 (≤4h), P2 (≤16h), P3 (≤4h, only if data
missing and budget remains). Stop at 24h or when the pre-registered runs complete.

## Repo / code
Submission repo = `paper/.git` → github.com/XueqiC/ICLR27_scaling_down (HEAD at start `b9ba574`). Analysis
scripts mirrored to `paper/code/analysis/`, result JSONs to `paper/data_mirror/`.

## Load / throughput probe (recorded BEFORE scheduling)
- pythia-1b@step96000 dense eval (main protocol, 128-probe measurement half): load 35 s, eval 13 s,
  peak VRAM 2.3 GB, 1.33k tok/s. P1 (dense + 2 densities + 2 bits) ≈ minutes.
- Gemma3-1B LoRA 1-epoch U75 probe: 14 updates / 66k processed tokens, 5 min 22 s wall including dense +
  post evals, peak GPU ≈14 GB, RSS 2.8 GB. Estimate per 1.0M-processed run ≈ 35 min (training ≈ 30 min +
  four evals) → nine P2 runs ≈ 5.3 GPU-h, well inside the 16 h allocation; P1 ≈ 0.2 GPU-h.

## P1 — new source frozen prediction (pruning + quantization)
- Target: **pythia-1b@step96000** (EleutherAI/pythia-1b, revision `134b25682ad4`, config sha `302d6702…`).
  Verified NEVER measured/fit/used (no results dirs, not in registry or panel). Same Pythia series as the
  panel (not v0/dedup). $N_0$ (matrix) = 805,306,368 (between 410M=302M and 1.4B=1,208M → **new size inside
  the range**); step96000 = **stage interpolation** (between 64k and 143k).
- Order enforced: model mapping frozen → dense $L_0$ measured (v46_dense_eval, `results/v46-p1-newsource/dense.json`)
  → predictions written and committed (`predictions_frozen.json`, commit `b1bf631`) → compressed measurement.
- States: dense; magnitude pruning d=0.65 (interp) and d=0.55 (extrap, per frozen training range 0.6–0.9);
  RTN int4, int3 (seen bits on a new source, NOT unseen-bit validation). Probes/mask/units = main protocol.
- Pruning candidates (all K0, dev-only fits): frozen shared power (v40 register); A2 per-density source
  regression + linear interp/extrap anchored on predicted 0.7/0.6; A1 γ=1 refit; strength-only; median-curve;
  zero. Quantization: frozen config-indicator {N0,L0,D0}; no-D0 {N0,L0}; per-bit median; zero. No literature
  baseline added (P0: incompatible endpoints/inputs).
- Report: per cell prediction/actual/signed/abs; MAE per candidate per config (0.65 vs 0.55; int4 vs int3
  separately). One source → no population claim.

## P2 — distillation random-pool protocol control + new middle pool (≤9 LoRA trajectories)
- Student Gemma-3-1b-pt, snapshot `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29` (the cached original). Teacher
  data: existing gpt-5.6-luna traces (no API). Recipe full trace. **Strict LoRA** (r=16, α=32, dropout 0,
  targets q,k,v,o,gate,up,down), AdamW lr 1e-4, cosine, warmup 0.03 — unchanged from the original protocol.
- Pool inventory: mother pool = **600 per domain** × 3 domains (U = per-domain count). Since U600 equals the
  full mother pool, pre-registered rule applies: **U_L = 75, U_H = 450, U_* = 375** (75 < 375 < 450 < 600).
  Pools are random samples via `--data-seed` (never first-U prefixes). Dev data-seeds {11,12,13}; test
  data-seeds {21,22,23} (all unused before). One fixed training seed 0 for every run; the data pool is the
  replicate unit. Pool-set distinctness checked by sorted sample-ID hashes; overlap reported.
- Budget coordinates: $D_U$ = supervised completion tokens of one pass; $T$ = cumulative supervised completion
  tokens; $E = T/D_U$; processed tokens and optimizer steps recorded. Two frozen milestones from the old
  matched-token logs: **T1 ≈ 148k, T2 ≈ 294k completion tokens** (the old runs reached these at 500k / 1.0M
  processed; the trajectory trigger is processed-based, so we keep the 500k/1.0M triggers and analyze at the
  ACTUAL completion tokens, reporting deviation). Epochs = smallest integer with epochs × pool_processed ≥ 1.0M
  (same rule for all nine runs; the cosine horizon therefore scales with the pool exactly as in the historical
  protocol — an inherited limitation, stated, not silently changed).
- P2-A (dev, 6 runs): U75 × {11,12,13}, U450 × {11,12,13}. Before training: save the OLD frozen v41 predictors'
  predictions for these pools at the planned milestones. After: fit and freeze the four low-DOF candidates
  (constant, T-only, E-only, 2D) on the six dev trajectories only (selection rules dev-only; whole
  trajectories move together by data seed).
- P2-B (test, 3 runs): U375 × {21,22,23}. Order: 6 dev done → freeze formulas/coefficients/selection rule →
  write predictions for U375 → train/eval the 3 test runs. Old U225 = historical diagnostic only.
- Main results: per-capability MAE, signed bias, improvement vs constant and vs δ=0, per milestone (no
  averaging across budgets), per pool. 3 pools = 3 replicates (not 6 checkpoints); training randomness not
  covered (single seed).

## P3 — measurement check (≤4h, only if existing per-benchmark data is insufficient)
Reuse first. If run: Gemma3-1B dense, prune d=0.7, RTN int4, and four P2 adapters (U_L/U_H × two budgets,
data-seed fixed in advance = 11); one independent secondary benchmark per capability (128 samples, existing
references; no accuracy, no teacher calls); same-capability primary/secondary loss consistency across arms.

## Status log
- [x] Verification: GPU identity/health, repo HEAD, P1 target provenance, pool inventory, protocol constants.
- [x] P0 contrast table (P0_RELATED_CONTRAST.md).
- [x] P1 dense L0 measured; predictions frozen + committed (b1bf631); compressed measurement queued.
- [ ] P2-A six dev runs; [ ] P2 freeze + U375 predictions; [ ] P2-B three test runs; [ ] P3 decision.
- GPU-hours used: (updated per package below).
