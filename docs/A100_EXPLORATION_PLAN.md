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

## Round v3 (2026-09-09 21:10 EDT; advisor packages) — turn phenomena into deliverable predictors
Rules carried over: no Qwen, no new teacher API, freeze-before-measure, old frozen predictions immutable, no config
dropped by outcome, budget 72 GPU-h total (card wall-clock), concurrency by measured throughput.

### V3-P (pruning): choose one deliverable predictor, then confirm once
- CPU development (v53): dev = 9 Pythia states + unblinded 1B@32k/96k/112k + 6.9B@32k/112k (all measured densities);
  candidates = shared power, A2 (per-density regression + interpolation), continuous low-order form; same inputs
  (N0, D0, L0c), same standardization, ridge on the same dev split; source-free baselines strength-only / median / zero.
  Selection rule (pre-committed): lowest leave-one-source-out MAE averaged over capabilities; ties (<0.02 nats) go to
  the form with fewer parameters that is continuous and zero at d=1. Old frozen v40/v46/v49 predictions untouched.
- Test panel: Pythia-410M@48k, 1.4B@112k, 6.9B@80k (verify unused + weight identity vs neighbouring steps), densities
  0.85 / 0.675 / 0.575 (interpolation between dev grid points; not labelled extrapolation). Order: dense L0 measured ->
  v53 freeze + commit -> 9 prune configs measured (forward only, ~1.5 GPU-h).
### V3-Q (quantization): add group size g -> F_{Q,c}(N0,D0,L0c,b,g)
- Same symmetric RTN, contiguous groups of g weights along the input dimension, per-group absmax scale (v54).
- Dev: 160M/410M/1.4B x {16k,143k} = 6 states x b{3,5} x g{64,256} (24). Tests: bit test b=4, g{64,256} (12);
  granularity test g=128, b{3,4,5} (18); joint test Pythia-1B@96k g=128, b{3,4,5} (3, a known model, new configs).
- Predictors: separable (state amplitude x shared (b,g) shape), low-order 2D with b x g interaction, same-input
  interpolation baseline; compact candidate A_c(x)(2^{b-1}-1)^{-p_c}(g/g_ref)^{q_c} is a candidate, not a derived law.
  Stop rule: if dev shows only near-zero and collapse cells with no identifiable middle response, stop the grid.
### V3-D (distillation): one controlled matrix (reuse P2-v2)
- Reuse: dev 270M/1B x U{75,450} x seeds{11,12,13} (running), tests U375 seeds{21,22,23} for 270M/1B/4B (queued).
- Add: 4B x U{75,450} x seeds{11,12} (student held out at dev pools; 4 runs) and 2 training-seed repeats (270M U75 s11
  seed 1; 1B U450 s11 seed 1). v50 freeze extended to emit 4B@U75/U450 predictions before those runs.
- Budgets = processed-token triggers 119k/237k/475k/949k (already fixed); no 750k seal in existing runs.
- Forms to add on CPU: (a_c + λ_c z_c) log(1+E) and (a_c + λ_c z_c)(1-e^{-T/T*}) + (b_c + μ_c z_c) log(1+E), z_c = L0c
  primary (log N_S alternative); compared with zero / constant / T-only / E-only / same-input surface.
### CPU common test: shared response shape across capabilities vs per-capability models at matched complexity.
### Execution order: CPU (v53 dev, v54 implementation, v50 extension) now; forward packages interleaved with the running
pipeline when a lane is idle; 4B dev runs + seed repeats after the test lanes; each package updates one paper table
(formula + coefficients, inputs, dev/test ranges, per-capability MAE, gain vs same-input baseline, bias, intervals).

## Round v4 (2026-09-10 13:10 EDT; advisor framing) — shared structure, (m,c)-specific parameters, prediction intervals (CPU only)
Goal: L_{m,c}(x) = f(x; theta_{m,c}) with a shared function family and method/capability-specific parameters, delivered
with three kinds of range kept apart (parameter confidence interval; distribution of parameters across models; prediction
interval for unseen models), applicability range, held-out error, and interval coverage/width. No new GPU work.
1. Shared-shape test: one family in a per-method normalized strength s (pruning s=(1-d)/0.3; grouped quantization
   s=(u,v) with u=log2 qmax, v=log2 g/128; distillation s=log(1+E)) with parameters fit per (m,c); compared on the same
   held-out splits with the method-specific forms already delivered (v53 power/A2, v55 2D, v56 forms).
2. Which parameters share: pruning gamma shared across capabilities vs per capability (amplitude per capability and
   per state); quantization 2D term coefficients shared vs per capability; distillation shape (v56 Part B, done).
3. Predict parameters, not just fit them: K0 amplitude from (N0, D0, L0c) vs K1 one-point calibration (amplitude set by
   the mildest measured compression point of the target, shape shared), cost counted as one compressed measurement;
   prediction intervals from leave-one-source-out residuals (per capability, per regime) with coverage and width on the
   confirmation panel, the 1B/6.9B pairs, and the grouped-quantization tests.
Deliverable: per law, form + parameter estimates and ranges + applicability + held-out error + PI coverage/width; a
paragraph in Sec. 3.1/4.3 and one table (analysis/v59_shared_structure.py -> tables/shared_structure.tex).
4. Layer 3 (v60, CPU): capability-specific selection maps from the delivered laws on the Pythia states over a nominal
   storage budget (pruning r=d with the sparse-format caveat; per-channel RTN r=b/16; grouped RTN r=(b+16/g)/16;
   distillation r=N_S/N_0 to a same-stage smaller student), validated leave-one-state-out: regret vs oracle, method
   agreement, fixed-method and cheapest-feasible baselines, 'no clear winner' cells where the predicted method gap is
   below the laws' held-out MAE; multi-capability (max-over-capabilities) version. Supporting evidence: v33 offline
   replay on 12 heterogeneous models. Paper: new section 'From laws to capability-specific selection'; contribution (3).

## Round v4 (registered 2026-09-10 22:45 EDT): confirm final predictors, fix interpretations, compress to 9 pages

Advisor package (user-forwarded). No new models. CPU packages under codex, reviewed; text and compression by the session.
- V63 quantization identifiability (R): dev bit-widths {3,5} make u^2 collinear with [1,u]; v55 register records design_rank 16 of 20 for low_order_2d. Export rank/singular values/ridge/effective dof; same-input controls without u^2 (16), bit-only (8), bit+group without interaction (12), delivered 20 (reproduce v55); score on the frozen v54 test cells; original v55 predictions untouched. Wording: separable claim narrowed to the tested power candidate.
- V64 selection feasibility: no source fallback (v60 line 478 falls back to the r=1 source when a fixed method has no feasible config); infeasible cells reported as such; per-policy feasibility coverage; regret on common-feasible cells; candidate coverage table per state; endpoint documented as the absolute loss of the deployed model (student dense + student post-training for KD; source dense + response for pruning/quantization). Contribution framed as "laws connected to an explicit decision problem, retrospective validation"; no-winner rule labeled heuristic.
- V65 distillation paired intervals: cluster bootstrap over trajectories of the paired absolute-error differences (joint+src vs frozen constant, vs zero) per student and capability; relative improvement; replaces the "2-4x" wording. Frozen-candidates vs post-hoc headline rule stated separately. P3 QA finding moves into the main text with the "conditional loss on a QA distribution != general QA ability" statement.
- V66 label rule: origin code from process only (v52/v45 assigned R when a frozen candidate lost); outcome shown separately. PI under-coverage (80% nominal, 78%/67% observed) reported with states/configs/widths.
- GPU (pending user confirmation, ~1 GPU-h): MuSiQue (answerable dev, supplied paragraphs) QA loss on existing adapters: final checkpoints of the 12 dev + 9 test + 4 4B-dev + 2 repeat runs, the 4 P3 KD checkpoints, and the three students dense; no training.
- Optional (ask): one training package, final KD form + rule pre-locked, one new pool size, three pool seeds on a seen student (~6 GPU-h).
- Decisions adopted: main text 9 pages (ICLR 2027 initial submission), budget Intro 1 / Setup 1.25 / Laws 3 / Generalization 2.25 / Selection 0.5 / Related+Discussion 1; distillation claim = new-pool prediction relation only; no Qwen/heterogeneous selection validation now.

## Round v5 (registered 2026-09-10 23:10 EDT): independent confirmation of the final predictors; 30 physical GPU-h cap

Advisor package (user-forwarded, "keep the GPU busy after the MuSiQue run"). Non-Qwen only; existing teacher traces; no new API calls. GPU: RTX 6000 Ada (GPU-20b20454, addressed by UUID); A100 idx3 only if released. Throughput measured on one existing configuration before fixing concurrency. Every package: freeze forms/coefficients/preprocessing/predictions and all baselines before the new measurements; unblinded data may enter a new development set but never rewrites old prospective records.
- V5-Q quantization: new development set = the unblinded 54-cell grid (6 states x b{3,4,5} x g{64,128,256}); report design rank and condition number; compare the 2-D surface, the bilinear form without u^2, same-input piecewise interpolation with a pre-specified boundary extrapolation rule, bit-only, median, zero; freeze per capability. Test (forward only, 21 configs): two development states + one state never used in quantization fits, each at b{3,4,5} x g{32,512}; the new state also at g=128 x b{3,4,5}. Test names: unseen group-size extrapolation; joint new-state x new-configuration. b=4 is no longer "unseen". Near-zero and clear-damage regimes reported separately. Group/tail handling of the quantizer checked before measuring.
- V5-D distillation: Gemma-3-270M and 1B x U=200 x six new pool seeds (manifest check that U=200 never entered selection), 12 short LoRA trajectories under the unchanged protocol (LoRA, optimizer, 1M processed-token schedule, warmup; early budgets are prefixes of the same schedule), checkpoints at supervised T in {50k,100k,200k}. Same pool sample reused across the two students; D_U and E from actual completion tokens. Before testing: freeze the per-capability final candidate and constant / T-only / E-only / same-input surface predictions. Report per student x capability paired error vs the strongest baseline with trajectory-cluster intervals; independent pools = 6. Goal: configuration prediction (new pool size x new budget), no student-size extrapolation; 4B keeps its current boundary. If resources are short, cut to 1B x 6 pools before seeing any result.
- V5-M capability measurement (no training): Gemma-3-1B, eight pre-specified states (dense; two pruning configs; two quantization configs; three distillation budget checkpoints) x three QA sets x 256 samples, conditional loss only: (1) a new non-overlapping sample of the primary 2Wiki distribution; (2) MuSiQue answerable dev (overlap with teacher traces checked; context and scoring protocol fixed); (3) the existing TriviaQA control. All sets fixed in advance and all reported; three readings pre-stated (both reproduce / only the new primary sample reproduces / neither). The running V67 pass (34 checkpoints x 128 MuSiQue samples) is a precursor and is reported as such.
- V5-P pruning: two source states never used in pruning fits (candidates: Pythia-2.8B at two steps), three pre-specified densities each; frozen continuous form, A2, median curve, zero at equal input budget. Answers repeatability of the compact form; if QA stays best-predicted by the median curve or zero, that is the delivered choice for QA.
- CPU: (a) per-capability response vs "shared damage curve + per-capability fixed scale" at equal development information on the new tests (how much capability conditioning adds); (b) selection feasibility (done: V64).
- Order: throughput test -> V5-Q forward grid (short) -> V5-M -> V5-D trajectories (longest) -> V5-P; CPU freezes precede each GPU test. Deliverable after the round: final formulas + reproducible coefficients + independent configuration predictions + strong-baseline comparisons + capability and applicability boundaries.

- V5-P state change (2026-09-11 00:10 EDT): the cached pythia-2.8b step64000 snapshot shares its weight blob with step143000, so the pair is step16000 + step143000 (distinct blobs); densities 0.85/0.75/0.65 unchanged.
