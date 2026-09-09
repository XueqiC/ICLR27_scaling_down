# A100_EXPLORATION_PLAN — authoritative plan for the expanded single-GPU round (2026-09-09 v2)

Supersedes A100_SUPPLEMENT_PLAN.md (24 h / 9 LoRA / two bases). Authorization: one A100 80GB PCIe on rai
(idx3, `GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0`), **72 GPU-hours** counted by physical-card occupancy
(concurrent jobs share one hour), internal checkpoints at 36 h and 60 h, no auto-extension. **No Qwen** in any
new load/train/quant/prune/calibration. No hpg, no new teacher API, no pretraining. Historical compliant results
retained; v38/v40/v41 frozen predictions and v42–v45 diagnostics retained; new analyses in new versioned dirs.

Labels: **P-new** = frozen prediction this round; **R** = post-hoc diagnostic/selection; **L** = existing
leave-out; **D** = development exploration. No row may hide post-test selection behind "P+R".

## Carry-over from the superseded plan (per "do not kill valid tasks")
- P1 (superseded scope) on **pythia-1b@step96000** is COMPLETE (C32, P-new): interp d=0.65 success; extrap
  d=0.55 failure; quant full wins int4/int3. Kept as-is; not re-claimed as new.
- Old-protocol P2-A: one valid run (Gemma3-1B, U75, data-seed 11, epoch-based cosine, 500k/1M triggers)
  finishes and is kept as a **protocol-limited historical comparison**; the remaining 5 dev + 3 test old-protocol
  runs were cancelled before starting (no GPU spent). Its pools (U75/U450 seeds 11–13; U375 seeds 21–23) were
  registered but not trained under the new protocol; U375 seeds 21–23 remain UNUSED test pools.
- Probe/schedule-test runs are quarantined to `_trash/` and never enter fits.

## Resources verified (15:45 EDT)
A100: 68 GB free (one 12.5 GB job running); RAM 1 TB; disk 2.1 T. Cached: gemma-3-270m, gemma-3-1b-pt
(`fcf18a2a…`), gemma-3-4b-pt, pythia-1b. Downloading (not GPU budget): pythia-6.9b step32000 (`554e5adf`),
step112000 (`9c7bcdb5`). Concurrency: start 2-way; admission target ≈64 GiB; measure single vs paired
throughput before committing to pairing; 3-way only if throughput improves and data loading keeps up.

## Priority & budget plan (allocation, not a promise)
P0 CPU/literature (done: P0_RELATED_CONTRAST.md) → P1 (forward-only, ~52 configs; est. ≤6 h incl. 6.9B) and
P2 dev (12 LoRA) in parallel → freeze → P2 tests (9 LoRA) → P3 (≤4 h) → P4 only with remaining budget.
If short: cancel P4 first, then trim P3 scoring; never drop test sources/pools/configs by candidate outcome.

## P1 — Pythia size × stage × strength joint prospective (forward-only)
Sources (all verified unused; official revisions): **1B@step32000** (`e2368797`), **1B@step112000**
(`8d1a7cf2`), **6.9B@step32000** (`554e5adf`), **6.9B@step112000** (`9c7bcdb5`). Matrix N0: 1B 0.81B
(inside 0.30–1.21B), 6.9B 6.44B (≈5.3× beyond the 1.4B max → size extrapolation). Stages 32k/112k = stage
interpolation under P1-A; 112k = stage extrapolation under P1-B.
Configs per source: dense; pruning d={0.9,0.8,0.75,0.7,0.65,0.6,0.55}; RTN b={8,6,5,4,3} → 52 evaluations
(not 52 independent models). Main-protocol probes, per-example losses saved.
Protocols: **P1-A** fits/standardization on the full 9-state dev (step16k/64k/143k); **P1-B** fits ONLY on
dev step≤64k (143k never used for fitting, scaling, or selection). Results reported separately.
Order enforced per source: mapping + coefficients frozen → dense L0 measured → predictions written and
committed → compressed measurement.
Pruning candidates: (1) frozen shared power (v40 register; P1-B refit on ≤64k dev); (2) A2 per-density
source regression with the FIXED rule (in-range: linear between adjacent fitted densities; out of range:
linear extension from the two boundary densities; anchors are predictions, never target measurements);
(3) γ=1 same-input; (4) strength-only, zero; (5) ONE new continuous candidate
ΔL̂_c = (β_c·φ)s + (ζ_c·φ)s², s=1−d, φ=(1, log N0, L_c0, log D0) (zero at d=1, signed; 8 params/cap).
Quantization: frozen config-indicator full {N0,L0,D0}, no-D0, per-bit mean/median, zero; **5-bit rule**
(pre-registered): class models have no 5-bit class → predict by linear interpolation of the 4- and 6-bit
predictions, labeled "interpolation-rule baseline"; models unable to produce a 5-bit prediction are marked
"not covering" (never zero error, never dropped). Target-calibrated int3 models are K1 and listed separately.
Analysis: per source/cap/config actual, prediction, signed, |err|; size interp vs extrap; stage interp vs
extrap; density interp vs deep extrap; quant per bit / full / ≥4 / int3; paired differences with stated
units; max-damage-point sensitivity (not a substitute); plots of dense loss, increment, and absolute
post-compression loss. Upgrade rules: new continuous candidate must beat A2 at equal information to count as
a form contribution; A2 winning → piecewise relation; 6.9B/112k success → state the extrapolation span; local
success → local range only. All four sources run regardless of the first two outcomes.

## P2 — multi-student × independent pool × exposure distillation prediction (≤21 short LoRA runs)
Students: **Gemma3-270M, Gemma3-1B** (dev + same-student test), **Gemma3-4B** (size × config test; 4B was
measured historically, so this is "new configuration on a student held out from the new dev matrix").
Teacher data: existing gpt-5.6-luna mother pool (600/domain), full-trace recipe, **strict LoRA** with one
uniform definition (r16/α32, targets q,k,v,o,gate,up,down), AdamW 1e-4, warmup 3%; trainable-parameter
counts recorded per student. **New uniform schedule**: cosine horizon and stop point defined on the SAME
absolute exposure for every U (v12 `--schedule-tokens 1000000` processed ≈ 294k supervised), so warmup/decay
do not scale with per-pool epochs (this is a versioned protocol change; old epoch-based trajectories are
historical comparisons only). Four fixed checkpoints per run at processed triggers {119k, 237k, 475k, 949k}
≈ supervised T {35k, 70k, 140k, 280k} (ratio ≈0.295 from logs; actual completion tokens recorded and used).
Pools: U = per-domain count, fixed mixture. Mother pool = 600/domain ⇒ U_L=75, U_H=450, U*=375. Random
pools via `--data-seed`; **paired across students** (same pool-seed ⇒ identical samples for 270M and 1B; test
pools shared by 270M/1B/4B). Dev pool seeds {11,12,13} (U75, U450); test pool seeds {21,22,23} (U375, never
trained under any protocol). Overlap by sorted sample-ID hash reported (U450 ≈0.60, U375 ≈0.45 Jaccard —
shared-mother-pool partial overlap, not a new distribution). One training seed (0).
Contamination: mother-pool vs evaluation-probe overlap audited (sample-ID + normalized text) BEFORE freezing;
secondary benchmarks exclude the training benchmarks (GSM8K, HotpotQA, CodeAlpaca).
Runs: dev 12 (270M, 1B × U75, U450 × 3 seeds); test 6 (270M, 1B × U375 × 3 seeds); test 3 (4B × U375 × 3).
Coordinates: D_U (supervised completion tokens per pass), T (cumulative supervised), E=T/D_U (derived);
processed tokens, steps, LR logged.
Candidates (4 classes, each with a no-source and ONE pre-registered source version, source term =
k_c·log(N_S/N_ref) on the leading basis coefficient; N_ref, T_ref, D_ref dev-only): constant (T>0); T-only;
E-only; low-DOF joint u=log(1+T/T_ref), v=log(D_U/D_ref), δ̂=u(a+bu+qv). With two dev students, log N and dense
L0 are not both identifiable; log N is the registered source term (dense-L0 alternative only as a pre-declared
dev-selected alternative). δ=0 vs constant answers "response exists?"; constant vs forms answers "structure
needed?". Freeze: 12 dev → limited dev-only selection by pool seed (whole trajectories together) → freeze all
candidates/coefficients/rules → save predictions for all 9 test trajectories × 4 checkpoints → run tests.
Report: same-student new pool vs 4B joint transfer separately; per T; per pool; bias vs variation; source term
and D_U term gains separately. 3 test pools = 3 replicates (not 36 checkpoints).

## P3 — capability measurement + conditioning value (≤4 h; reuse first)
Gemma3-1B states: dense, prune d=0.7, RTN int4, and P2 adapters (pool seed 11, U_L/U_H, two fixed T) — all
fixed before any P3 result. Primary = MATH-500/MBPP/2Wiki measurement half; secondary = **SVAMP / HumanEval /
TriviaQA** (128 samples, fixed probe seed; NOT the training benchmarks). A: same-capability primary vs
secondary ΔL direction/magnitude/uncertainty; B: only if dev suffices, same-capability vs generic/cross-
capability held-out comparison (no target-loss-derived "generic damage" passed as K0). Optional 4B replication
decided BEFORE seeing 1B results: default = do it only if ≥6 h budget remains after P2 tests.

## P4 — quantizer-condition mini-package (only if P1/P2 confirmatory budget is reserved and P3 done)
Pythia-1B (one stage) and Gemma3-4B; b∈{3,4}; RTN vs one fixed GPTQ implementation with matched group size /
axis / symmetry / excluded modules; fixed existing calibration samples disjoint from probes; ≤8 configs.
Questions: does the RTN-fit predictor transfer zero-shot to GPTQ; do the two algorithms show different
capability responses at matched settings. No GPTQ law; no latency/storage claims.

## Registration log
- 2026-09-09 15:50 EDT: sources/revisions/architectures registered (v36 ARCHITECTURES 1b, 6.9b;
  registry pythia-1b, pythia-6.9b); v12 `--schedule-tokens` implemented (absolute-exposure protocol); P3
  secondary benchmarks fixed to non-training sets.
- 16:00 EDT: P1-v2 mapping frozen (6f2e63a); 1B@32k/112k dense measured, predictions committed (cd9ed8f),
  compressed measurement running. Contamination audit clean. P2 v2 pools registered (v47 register['v2']).
  Concurrency note: card briefly at 3-4 my processes (old run + schedule test + measurement); throughput
  pairing to be measured on the P2 dev launch. A premature duplicate v6 run was killed and its partial output
  quarantined (results not used).
- 16:00 EDT: 6.9B dense measured (peak 14 GB, 81 s each); predictions committed (950312f); compressed measurement
  queued behind the 1B pair with --reference-device cpu. Schedule protocol test PASSED (270M, 30k tokens: 7/7
  updates, LR 9.5e-5 -> 0.0 at budget, 3 snapshots; overshoot ≈1 batch recorded); test run quarantined.
  P2-v2 dev lanes launched 15:58: lane A gemma3-1b (U75/U450 × 11–13), lane B gemma3-270m gated on PAIR_OK.
  Concurrency note: lane A's first run overlaps the old-protocol run and the 1B measurement, so its throughput
  reading is not a clean solo; pairing decisions use per-run completion time (runs/hour), logged per run.
- Pending: P1-v2 compares (1B, 6.9B); P2 dev completion → v50 freeze → FREEZE2_COMMITTED → test lanes.
- GPU-hours used at start of v2: ≈0.3 (probes 0.1, P1 0.02, old-protocol run ≈0.2 in flight).
- 16:12 EDT: 1B pair measured (15:47-16:12, 22 GPU-min) and compared -> C33. 6.9B predictions had been committed at
  15:55:58 (950312f); 6.9B measurement started 16:12:04 (reference on CPU).
- 16:49 EDT: 6.9B pair measured (36 GPU-min) and compared -> C34: ~5x size extrapolation refuted in both arms under both
  protocols. Paper updated (abstract/conclusion qualified; new paragraphs; tab:p1v2; tab:main +8 rows). GPU-h ≈1.0.
- P3 decision unchanged (only if >=6 h remain after P2 tests); it is also held back until the solo-throughput baseline of
  the first dev run is recorded, so it cannot contaminate the solo-vs-paired measurement.
