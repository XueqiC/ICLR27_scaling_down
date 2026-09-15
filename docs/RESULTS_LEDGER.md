# Results Ledger
Status vocabulary: VERIFIED / PRELIMINARY / RUNNING / PLANNED / BLOCKED / REJECTED.
Every manuscript claim must trace: claim -> figure/table -> processed result -> raw artifact -> config+code version.
Snapshot: 2026-09-01. Raw artifacts live in the the workstation project tree (`results/...`); curated copies in this repo under `results/`.

| ID | Claim | Status | Evidence | Models | Validation | Manuscript |
|---|---|---|---|---|---|---|
| C1 | Capability ordering under pruning in LOSS space (QA-loss most robust pre-cliff, math/code most fragile) | VERIFIED in loss space (12 models); BEHAVIORAL caveat: lower QA loss does NOT reliably imply higher QA accuracy (v19: 4 QA cells below dense loss, 0 with higher accuracy) — 'QA robustness' is a loss-space statement, not validated behaviorally | results/v6/*/report.md + v19-links | 12 | cross-family (loss); behavioral link v19 | Tab. grid |
| C2 | Cliff in narrow proportional band; non-perturbative in loss | VERIFIED (descriptive) | v6 reports + v6b report_b | 12 | two-scale Taylor-failure check | exp §curves/§mech |
| C3 | Block structure of benchmark gradient signatures | VERIFIED with stats on 1 model (block 0.0995, CI [0.072,0.123], perm p=0.001) + point estimates on 9 incl. two 30B (muse-30b, gemma4-31b); shared-component-removed cosine is the robust metric (within~0.22-0.31 vs cross~-0.18 to -0.21, block score ~0.64 at 30B), raw cosine/Jaccard family-dependent at scale | v9 reports + v9b stats.json | 9 | bootstrap+permutation (Qwen1.7B) | Fig. blocks |
| C4 | Targeted ablation selectively removes one capability | VERIFIED (2 models; controls: random & magnitude-matched ~ no damage, global-Fisher & no-residual destroy all capabilities; selectivity 10-1700x across fracs 0.01-0.2%) | results/v9c-ablation/*/controls.json | 2 | 4 control modes x 3 fractions | exp §regions |
| C5 | Pre-cliff pruning reduced form | ON-PROBATION → USE SIMPLE FORM: paired bootstrap shows complex family/size/cliff forms do NOT beat density_only (ΔMAE CI includes 0 or favors baseline); density_only is the default reduced form (held-out MAE ~0.33, calibration ~0.91-0.96). Mechanism law is upper-bound comparator only | results/v18-law-fit + LAW_FIT_REPORT.md | 12 | fit-shallow/leave-largest/leave-family + paired bootstrap | exp §law |
| C6 | Signed first-order term predicts damage sign | VERIFIED at cell level (67.6% acc, Wilson CI [58.3,75.7], n=108) | results/v14-fitting/ | 12 | Wilson CI | exp §mech |
| C7 | Geometric retention smooth; cliff at family-specific critical retention | PRELIMINARY (3 models, 2 families) | results/v11-geometry-damage/* | 3 | none | exp §orderparam |
| C8 | Quantization 4^-b shape | 4^-b OVER-predicts 5-bit damage: high-precision (512-probe) ΔL(5)/ΔL(4): small/mid 0.13, 30B EVEN LOWER (gemma4-31b 0.04-0.08, muse-30b 0.06-0.18) vs 4^-b prediction 0.25. NOTE: the quantizer uses quant_max=2^(b-1)−1 with fixed max_abs clip, so the DERIVED step²-ratio is (7/15)²≈0.218, NOT 0.25 — the correct-step baseline already predicts less than 4^-b, yet even 0.218 over-predicts most observations. 5-bit near-lossless, damage concentrates at ≤4 bit. THREE-CANDIDATE test (v30, LOMO): fixed 4^-b (η=2) AND actual-step (7/15)² (η≈2.2) BOTH REJECTED by paired-diff CI for math/code/pooled; a shared learnable η fits better and wins pooled all-bit LOMO MAE (1.045 vs zero-change 2.777), with 512-probe η=3.47 [2.98,4.41] — bit-response decays FASTER than either fixed law. Caveats: at 5-bit precision the paired MAE can NOT separate shared-η from actual-step (both beat zero only marginally, 512-probe +0.037 [0.016,0.065]); η drifts with bit range (2.67 for 4/5 only vs 4.69 all-bit) so it is NOT a clean single exponent; only 4/7 shape512 models have 5-bit. So: shape now CHARACTERIZED (steeper-than-4^-b, η>2.2, fixed laws rejected) but not reduced to one exponent. PER-REGION DEFLATION (v30b): the shared-η all-bit LOMO win is 98.7% from the int3 COLLAPSE region (easy to predict big damage); in the MEASURABLE 4/5-bit region NO candidate beats zero-change (η gain 0.057 [−0.025,+0.140], CI incl 0); at high bits (b≥6) the fixed laws OVER-predict (negative gain) while η≈zero. So the quant law has NO demonstrated predictive value beyond zero-change in the smooth measurable region — its apparent success is a collapse-region artifact. Freeze status: Mode-A amplitude uses only size/family/dense-L_c (0 calibration pts); the 4→5 paired test is CONDITIONAL SHAPE TRANSFER (1 calibration pt), not full basic-param. Qwen3-14B quant shape pre-check = INCONCLUSIVE (candidate 4/5-bit predictions coincide within ~0.001-0.015 nats < measurement precision) → do not run for shape confirmation. Missing 5-bit: gemma3-270m/4b, olmo3-7b (fill cmds in QUANT_REGIONS.md); QA erratic, OLMo near-noise. So 4^-b is not the precise shape (steeper decay); still int8/6 lossless, int3 collapse (OLMo exempt). Prospective Qwen3-8B quant prediction PARTIAL (MAE 0.23 > beats zero 0.32; math b4 pred 0.30 vs actual 0.28) | results/v10-quant-shape512 + prereg | 5+ models | 512-probe ratio + prospective | exp §quant |
| C9 | answer_only hurts math more than full traces (+0.10–0.13 nats, both students, both teachers) | PRELIMINARY (12-run pilot; single seed; teacher-forced loss confounds capability with style shift — needs B3 accuracy link) | results/v12-distill/pilot_summary.md (12 runs) | 2 students × 2 teachers | none | exp §distill |
| C9b | Distillation SFT raises math/code reference loss in every cell and the rise grows with trace count (1b: math ∝ n^0.84, code ∝ n^1.2); QA loss falls, non-monotone in n | PRELIMINARY (same confound as C9; QA measured on 251 tokens only; do NOT interpret as capability loss before B3) | results/v12-distill/pilot_summary.md | 2 students | none | exp §distill |
| C10 | General-data (c4) recovery is monotone saturating for both compression types (gemma3-1b: prune0.6 5.36→2.35→2.03→1.95; quant3 16.60→4.81→3.34→3.08); deeper damage recovers to a higher residual floor | PRELIMINARY (1 model; held-out budget fit underdetermined — recovery ΔL stays >1 nat so pre-cliff cap leaves too few points; needs denser E6 checkpoints) | results/v13-recovery/gemma3-1b/*_c4 | 1 | leave-one-budget pending | exp §recovery |
| C11 | Data source flips recovery dynamics: general (c4) monotone-saturating vs capability-aligned (traces) strongly NON-MONOTONE (prune0.6+traces 2.14→2.68→5.22, back near damaged 5.36); over-training on aligned data re-damages | PRELIMINARY (1 model/1 seed; best aligned budget ~1M among those tested; needs denser checkpoints + 2nd seed/model + continued-training control — E6) | results/v13-recovery/gemma3-1b/{prune_0.6_traces,prune_0.6_c4} | 1 | none yet | exp §recovery |
| C12 | Mid-scale robustness peak (Gemma 12B; prior Qwen 14B) | PRELIMINARY (Gemma direct; Qwen from prior ability-space grid) | v6 grid + prior re-analysis | - | none | exp §family |
| C13 | Concentration predictor retired | VERIFIED (as negative result within our grid) | v6 reports concentration lines | 12 | pre-registered then falsified | exp §family |
| C14 | Dense reference law variants (D1-D4) | PLANNED (no fits yet; placeholder in paper) | - | - | held-out per plan | method §laws |
| C15 | Cross-method unification via TESTED resource-ratio coordinates | The method-agnostic law using only tested storage/active ratios FAILS to extrapolate across pruning/quant/distillation (v17 held-out sign 0.48–0.66). This does NOT exclude shared structure that also uses dense capability, training conditions, or method descriptors — unification remains OPEN, only the resource-ratio-only model is rejected | results/v17-unification | 5 families | leave-method/family/largest-out | exp §unification |
| C16 | Method selection is capability- AND cost-axis-dependent: at matched NOMINAL storage quant near-lossless for math/code vs pruning catastrophic; distillation the only arm reliably cutting ACTIVE params (latency); pruning's apparent QA advantage is a LOSS-space effect and is NOT behaviorally validated (v19) | PRELIMINARY (gemma3 only; nominal/analytical costs, latency NOT measured; held-out selection regret pending §9; QA-niche downgraded to loss-space) | results/v17-unification/* + v19-links | gemma3 3 arms | held-out selection pending | exp §map |
| C16b | Loss-constrained selection OFFLINE REPLAY with FROZEN predictions (v33, 12 models, 2210 budget vectors, 0 target calibration, nominal storage): the frozen selector meets the actual loss budget only 70.9% [61.6,79.5] of the time at 0.559 mean nominal storage — a real gain over trivial baselines (always-cheapest 0.7% satisf / 0.19 storage; always-dense 100% / 1.0), signed cost gap vs oracle +0.067 [0.001,0.132]. But PRUNING IS UNUSABLE in the selector: 46% of actually-feasible pruning configs are WRONGLY EXCLUDED (predictor over-predicts damage) AND all 31 pruning selections that were made VIOLATED the constraint; pruning never selected for 10/12 models under frozen prediction. CORRECTION (v33b, measured-Pareto): the selector is MISCALIBRATED (predictor error), NOT proof of pruning domination — on the MEASURED 4-objective Pareto frontier (nominal storage + 3 capability losses) PRUNING SURVIVES for 6/12 models (24 frontier configs), dominated only in the other 6. So 'pruning unusable' is a selection-under-prediction-error statement, not a true-domination one. Wrong-exclusion 0.460 [0.287,0.606] (model-level bootstrap). Ties C25b/C8: pruning's amplitude-prediction failure breaks loss-constrained SELECTION, but pruning is not dominated on the actual frontier | results/v33-selection + LOSS_CONSTRAINED_SELECTION.md | 12 models offline replay | frozen-prediction replay, paired bootstrap | exp §map |
| C17 | Prior-grid re-analysis (anchor recipe artifact; QA train-half contamination ~1/3; rank-only IRT) | VERIFIED for the PRIOR grid (recomputed from committed cell tables). SCOPE CLOSED (v34, with provenance limits — branch b, training-benchmark reuse): the effect is confined to the OLD mixed-QA panel; the "~1/3" is M2 benchmark-gap/HotpotQA-gain = 36.1% (a ratio, NOT a contaminated-item rate). CURRENT MEASUREMENT (2Wiki, 64 items) is CLEAN — 0/64 exact overlap with current traces or the old HotpotQA panel, disjoint corpus → measurement not contaminated, no re-run needed. CAVEAT: current distillation still TRAINS on HotpotQA (official train rows 0-599), so QA-distillation dynamics inherit training-side benchmark-fitting risk (measurement side clean). Provenance limit: historical trace bytes lack an immutable manifest, full lineage uncertifiable. Keep C17 + originals | results/v1-v3 + v34-c17-scope + C17_SCOPE.md | prior grid + current-set overlap | recomputation + exact-question overlap audit | exp §reanalysis |
| C18 | Distillation raw math/code probe-loss rise vs a shared generic-loss shift | Under a FIXED λ_c=1 subtraction, capability-specific residuals dL_cap=dL_probe−dL_generic are non-positive — one residual OBSERVATION. This does NOT establish 'distillation does not harm math/code capability': even a fitted λ_c cannot alone prove the removed part is entirely style. Behavioral accuracy shows no corresponding degradation in tested settings | results/v16-style-residual | 5 students | generic-corpus control (λ=1) | exp §distill |
| C19 | Loss→accuracy link A_c=g_c(L_c) is a monotone sigmoid; the FORM is compatible with a single cross-family shape within each capability (math/code/QA all: paired shared−family held-out MAE CI includes 0) — closes the behavioral-interface chain | VERIFIED that a monotone shared-within-capability form exists (math L½0.88, code 1.32, QA-EM 2.79); link PARAMETERS are capability-specific and code steepness/QA CIs are broad; singleton families thin | results/v15-accuracy + v19-links/BEHAVIORAL_LINKS.md | 7 models | leave-one-family-out + bootstrap | RQ1 §link |
| C20 | OLMo3 robust under both tested pruning and quantization and across metrics; pruning↔quant robustness correlation is positive (E2 pooled Pearson 0.508 [0.036,0.838], survives OLMo removal) but NOT established as a shared family factor (Spearman CI includes 0; broad family intervals; Qwen near-zero within-family) | PRELIMINARY (suggestive; needs more sizes per family) | results/v6,v10,v15 + v20-robustness-corr | 12 models | Pearson/Spearman + bootstrap + LOFO | exp §family |
| C21 | Distillation change δ_c = L_c(S_KD)−L_c(S0); predict L̂_c(S_KD)=L_c(S0)+δ̂_c | δ_c REAL (beats dense δ=0). SIGN: δ_code>0 — more transfer data RAISES the defined code loss; this is a TRANSFER RESPONSE, not a "distillation gain". On the OLD narrow-budget D-ladder, math/QA looked ~per-capability-constant and only code showed data-dependence; BUT the later reuse-count experiments show math/QA ALSO respond strongly (to E), so 'math/QA constant' is NOT a main distillation conclusion — it was a narrow-budget artifact. CODE low→high data-dependence tested by the per-seed PAIRED diff δ(600)−δ(75) (S0 cancels), n=3, t-interval: gemma3-4b +0.068, 95% CI [+0.059,+0.076] EXCLUDES 0 (real); gemma3-1b +0.083, 95% CI [−0.014,+0.180] INCLUDES 0 (seed0 outlier +0.128 inflates variance → suggestive, not significant at n=3). So code data-dependence is CONFIRMED on 4b, not yet on 1b alone. Fine structure (4b 'D300 peak', D300 vs D600) WITHIN seed noise. math D-increase small; QA noisy (SD up to 0.35). basic-param LOMO prediction MET. RESOLVED (v31 unique-data×seen-tokens control, gemma3-1b, U∈{75,600}×3 seeds, matched processed-token milestones 500k/1000k): the transfer response has TWO axes — (i) SEEN-TOKENS/training volume drives loss UP (for fixed pool, code δ 0.38→0.80 from 500k→1000k), (ii) at MATCHED seen-tokens, UNIQUE-DATA DIVERSITY strongly MITIGATES damage — U75(heavy repetition) − U600(diverse) paired diff at 1000k: math +0.955 [+0.887,+1.022], code +0.660 [+0.522,+0.799], qa +5.799 [+4.690,+6.909], ALL CIs EXCLUDE 0, effect grows 500k→1000k. So the response is NOT explained by token exposure alone NOR unique-count alone, and the diversity effect is NOT code-specific (math & QA show it too). BUT the matched-SEEN-TOKENS contrast CONFOUNDS pool size with reuse count E=T/D_U (D_U: U75 67027, U600 533869 → at matched T, E differs ~8×). MATCHED-REUSE-COUNT control (v31b, U75 at T=62775/125550 vs U600 at 500k/1000k, same E≈0.94/1.87): the large pool-size gap COLLAPSES — math/code U75−U600 shrinks from +0.3/+0.9 (matched-T) to −0.08/−0.13 (matched-E, small, CI excl 0 but reversed sign), QA differences become non-significant (CIs incl 0). So REUSE COUNT E=T/D_U is the DOMINANT coordinate (matching E removes ~85-90% of the matched-T effect); a small residual remains (at fixed E more tokens → slightly more change) so E is not the complete story, but the simpler reuse-count law is favored over an irreducible 2-D (unique×volume) law. Also: don't constrain the law to monotonic worsening — U600 QA improves (δ<0) then partially recovers. Seed caveat: 'first n rows' ⇒ all 3 seeds share the subset (training randomness only) | results/v12-distill + v25 + v28 + v31 + v31b | gemma3 + uxseen 6 + same-E 3 trajectories | paired t-interval, matched-token AND matched-reuse controls | exp §distill |
| C22 | A scalar teacher-quality descriptor is insufficient to predict capability-specific transfer under trace distillation in our setting: teacher effect is math-invariant (|luna-sonnet|≤0.04) but QA-strong and size-growing (-0.28→-0.92) | VERIFIED as stated (4 gemma sizes × 2 teachers); strengthening to an L_T-sufficiency falsification needs both teachers' L_T measured under identical eval (E5) | results/v12-distill/*/claude-sonnet-4-6_full_600/eval.json | 4×2 | direct contrast; teacher-descriptor test pending E5 | exp §distill |
## Audit 2026-09-04 (directive §2 integrity checks)
- **stop-sequence bug residue**: the zero-shot floored accuracy is archived only under `results/v15-accuracy/_zeroshot_floor/`; it is NOT in any ledger claim or figure. All accuracy claims (C19) use the fixed few-shot path. CLEAN.
- **Qwen3-1.7B dir merge**: canonical `Qwen3-1.7B/` has 16 densities, no duplicates; stray dir quarantined to `_trash`. Anchor gap <3e-3 nats recorded. CLEAN.
- **Fake-quant r_storage is NOMINAL** (b/16), not serialized size or runtime memory. C8/C16 now say "nominal". Serialized/runtime NOT measured.
- **Unstructured pruning r_storage=d is NOMINAL**; mask/index overhead and the fact that dense execution keeps r_active=1 (no real speedup) are NOT counted. C16 flags this.
- **Cross-tokenizer NLL**: all cross-family comparisons use WITHIN-model ΔL (tokenizer-consistent); no absolute per-token NLL is compared across tokenizers. Absolute-level cross-family statements would need bits/nats-per-byte (not currently claimed).
- **Count consistency**: model/benchmark/fit counts in claims are generated from result files; manuscript body counts to be regenerated at write time (not yet re-synced).
- **Over-strong wording fixed this pass**: C8, C11, C16, C18, C20, C21, C22 downgraded to defensible wording; C15 already correctly scoped ("method-agnostic laws on tested storage/active coordinates fail to extrapolate", not "no unified law exists").
- **Still to add** (directive wants full columns): per-claim seed, exact checkpoint, eval version, CI, held-out flag as explicit table columns — LAW_FIT_REPORT.md will carry these for the four method laws.
| C23 | Behavioral-interface summary: a shared-within-capability monotone loss→accuracy link holds for math, code, and QA across tested families, but 'QA free lunch' (QA loss improvement under pruning/distillation) does NOT correspond to QA accuracy improvement in the tested panel | VERIFIED (shared monotone form) + REJECTED (behavioral QA free lunch) | results/v19-links/BEHAVIORAL_LINKS.md | 7 models | leave-one-family-out + bootstrap | RQ1 §link |

## C17 scope closeout — 2026-09-07

The original C17 row and V1–V3 results are preserved. Its pending scope check is resolved by [C17_SCOPE.md](C17_SCOPE.md) and `results/v34-c17-scope/summary.json` (CPU-only; no model runs). The observed effect belongs to the old mixed HotpotQA/2Wiki evaluation panel: benchmark/format fitting, not a demonstrated duplicate-item leak. The recovered 150 old M0 training questions and 50 old HotpotQA evaluation questions identify in official HotpotQA validation; the current 600-question training pool identifies as official HotpotQA train rows 0–599. These known old questions have no exact matches in current training or the saved 64-item 2Wiki measurement panel; current training and measurement also have no exact question matches.

Current training still reuses the HotpotQA benchmark, so “no current data dependence” is excluded. Separately constructed HotpotQA and 2Wiki questions do not guarantee disjoint Wikipedia articles (three rendered title prefixes are shared here). Complete old recipe lineage and immutable per-run V12/V16/C21 measurement identities remain unresolved. Preserve qualified fixed-panel loss results; C17 alone requires no new model run. Independent QA capability/generalization interpretations of C9b, the QA extension of C18, C21/V31/V31b, and trace recovery require the report's clean-set validation gate: freeze the existing 64 question-disjoint 2Wiki items, or the 61-item title-disjoint candidate with full-context screening, and pair relevant existing dense/trained checkpoints. This audit does not claim that validation was run.

## Stage A measurement and split audit — 2026-09-04

This addendum supersedes the earlier audit's cross-tokenizer assertion. Existing claim rows, including C21, are unchanged. Details: [MEASUREMENT_AUDIT.md](MEASUREMENT_AUDIT.md); completed-cell reuse map: [EXPERIMENT_MANIFEST.md](EXPERIMENT_MANIFEST.md).

- Capability loss is completion-token-weighted CE **nats per model-token**; within-model deltas preserve the tokenizer but do not standardize units across tokenizers. V19 fits **absolute** per-token losses across families; V17/V18/V20 compare token-unit deltas, and V21 compares source-referenced gaps. Common-text per-byte/per-character NLL is **TO-DO / TO-MEASURE**, not performed.
- V6/V10/V12/V13/V16 use the clean odd-half probe protocol in current code (64 items/capability, prompt/target separately capped at 512 tokens, no chat template). V6/V10 loss JSON lacks immutable eval-version provenance. V15 default and easy are separate protocols: easy changes both loss and accuracy benchmarks. Preserve legacy decoding variants and the old V6 same-probe archive separately.
- Density `<.55` holdout in V18 reuses four source checkpoints across the 146 training / 28 test pre-cliff rows. Bit/budget holdouts and the V21 4B D ladder likewise reuse source checkpoints; label them conditional config tests. Student-size folds avoid same-student sibling rows, but share dense references/teachers/probes. Freeze groups by originating checkpoint/revision; additionally group reference-sharing students for source/family transfer claims. Pre-cliff eligibility currently depends on all observed outcomes before splitting.
- V22 separates baseline gap from signed distillation gain for **23 V12 runs + 16 V16 pairs (117 capability rows)**. About 99%/90% of Gemma math/code size slopes are dense baseline effects. The V21 0.106 remains held-out across Qwen sizes, with **one offset per capability, three per fold**, and only one independent size contrast/capability. The hierarchical interpretation is weakened; proposed C21 wording is in `results/v22-distill-decomp/suggested_C21.md` and the new LAW_FIT_REPORT section. C21 itself was not edited.
- No measurement, GPU work, scheduler access, or running-job changes occurred in Stage A. New analyses use existing artifacts; missing measurement and provenance work remains explicitly tagged.
| C24 | Capability specificity of loss, tested per intervention (must not be judged globally from pruning alone) | Under PRUNING + these probes: same-capability prediction beats cross-capability for MATH and QA (gain +0.04–0.21, CIs mostly exclude 0) → capability-specific structure. For CODE, same-cap gives NO extra prediction gain over cross-cap → a common-damage model a_j·h(d) SUFFICES to explain the observation, but this does NOT prove code loss measures only common damage (distillation shows code data-dependence — compatible). Capability specificity must be checked separately in quant & distillation | results/v26-loss-validity-pred | 6×5 | same-vs-cross prediction gain, pruning only | RQ1 §measurement |
| C25 | Basic-parameter prediction of compression response (predict from N0/family/dense L_c only) | LOMO dev + PROSPECTIVE on held-out new source Qwen3-8B: PRUNING FAILS (Qwen3-8B prospective MAE 0.62 WORSE than zero-change 0.24); QUANTIZATION PARTIAL (prospective MAE 0.23 beats zero 0.32, math b4 pred 0.30≈actual 0.28); DISTILLATION MET in LOMO (a≫beats zero). Distillation-on-new-source not yet run. SECOND PROSPECTIVE SOURCE Qwen3-14B (the workstation, cpu-reference prune) REPLICATES the Qwen3-8B pruning failure: prune-robust (actual ΔL d0.6 math +0.126, code +0.277) with QA loss DECREASING under pruning (d0.6 −0.797, a SIGN FLIP), while frozen v28 Mode-A over-predicts math ~14× (≈+1.8 pred like Qwen3-8B) and gets QA sign wrong. So the basic-param pruning-amplitude failure is now confirmed on TWO independent held-out Qwen models. DISTILLATION PROSPECTIVE on new student Qwen3-4B (n=3 seeds, D=75/600, same teacher/recipe): δ_c is NEGATIVE at both budgets (KD IMPROVES Qwen3-4B: D75 math −0.170/code −0.277/qa −2.67; D600 −0.105/−0.206/−2.31) — OPPOSITE SIGN to the gemma students (δ>0). The data-response SHAPE transfers (δ(600)−δ(75) = +0.066 [+0.045,+0.086] math, +0.071 [+0.046,+0.096] code, both EXCLUDE 0, same direction as gemma) but the SIGN/OFFSET does NOT — any gemma-fit predictor (δ=0, positive mean, positive data-response, metadata) gets the sign wrong (δ=0 MAE math 0.14/code 0.24/qa 2.49). MATCHED CONTROLS (Qwen3-4B, U75 seed1 n=1 vs U600 n=3; seeds0/2 GPU-fault-blocked, poller upgrading): matched-SEEN-TOKENS gap (n=3, all CIs excl 0: math +0.15/+0.47, code +0.08/+0.18, qa +2.6/+3.1) SHRINKS under matched-REUSE-COUNT E to small residuals (math −0.009/−0.043, code +0.015/−0.13, qa +0.53/−0.77; some CIs excl 0). So reuse count E=T/D_U is the DOMINANT coordinate (removes ~85-95% of the gap) but NOT complete (small residual) — REPLICATES gemma3-1b (v31b) on a NEW student family. E-SUFFICIENCY (v37, gemma3-1b + Qwen3-4B, 108 signed obs, 0.05-nat tol): E-only is NOT sufficient — matched-E residual intervals fail the equivalence criterion for math/code/QA; AND adding log D_U does NOT lower held-out MAE (worsens primary; secondary QA gain 0.00065 nat). So reuse count E is the DOMINANT coordinate but leaves a real residual that unique-data volume (log D_U) does NOT explain — the distillation law needs E plus something else (not D_U). RESIDUAL DIAGNOSIS (v37b): the matched-E residual COINCIDES with an ~8× training-exposure gap (optimizer steps 28 vs 224; supervised tokens 39k vs 317k — because T=E·D_U and D_U differs 8×). A low-DOF k_c(E)·log(D_U/D_ref) INTERACTION also gives NO held-out MAE gain; only gemma-math meets the exposure-explanation rule but is indistinguishable from a constant pool gap. So the residual is CONFOUNDED with training exposure and its causal attribution (exposure vs a genuine unique-data effect) is UNRESOLVED on this data — not a new latent capability variable. δ predicted with sign; QA noisy.  Qwen3-14B QUANT (absolute-loss check, the workstation idx1 cpu-ref): int8/6 near-lossless, int4 mild (math +0.09/code +0.31) with QA IMPROVING (−0.23, same Qwen QA sign-flip as pruning), int3 CATASTROPHIC collapse (+9 to +11 nats). Confirms high-bit-lossless + int3-cliff and the QA sign-flip basic-param predictors miss (quant shape test was pre-registered INCONCLUSIVE, v30b). CROSS-CUTTING (CORRECTED 2026-09-08, per advisor): the three arms have DIFFERENT gaps, NOT a unified 'shape transfers / amplitude fails' negative. Pruning: predictor amplitude+sign fail on new Qwen, shape 'compatibility' is only post-hoc-oracle-rescaled (not independent shape prediction). Quant: Qwen3-14B shape test INCONCLUSIVE (cannot claim shape transfer); int3-collapse prediction has real value. Distillation: matched-E shrinks (not 'collapses') the gap. SIGN facts (results/sign-matrix): distillation QA δ<0 in BOTH gemma AND Qwen (KD improves QA universally, NOT Qwen-specific); the real cross-family sign difference is math/code (gemma + / Qwen3-4B −); Qwen-specific QA-improvement is under PRUNING (gemma QA damaged at d0.6) and int4 QUANT. δ=0 has no sign | results/v28-new-source-pred + Qwen3-8B + Qwen3-14B + Qwen3-4B distill actual + v29 | 12 LOMO + 3 prospective (2 prune, 1 distill) | LOMO + frozen prospective ×3 | RQ2 §prediction |
| C27b | Pythia 4-input comparison (v36b) — separates 'D_0 redundant' from 'both predictors fail' (advisor's key table) | Per-arm/cap, same leave-one-step-out holdout, MAE + improvement over strongest simple baseline (config-median/mean) + paired CI, for zero / {N0,D0} / {N0,L0} / {N0,D0,L0}. 3-SIZE (160m/410m/1.4b): the COMBINED {N0,D0,L0} is the BEST model for math/code and beats the per-config baseline substantially (quant-code base 1.55->combined 0.63; pruning-math 0.41->0.26; quant-math 1.41->1.14), while single inputs {N0,D0} or {N0,L0} ALONE do not beat baseline. On the 3-size panel the combined model is best (on 2 sizes it was worse); this is descriptive — expanding the panel changed coverage/held-out/folds together, not attributable to 'overfitting cured' alone. So both N0,D0,dense-L0 are needed and D_0 contributes real power beyond dense loss for math/code; QA remains hard (combined does not beat baseline). So the earlier 'adding D_0 gives no gain' was the combined model overfitting, not D_0 being uninformative. PRUNING: no input beats the per-config baseline (QA models HURT: {N0,L0} 0.90 vs 0.16). D_0/L0 correlated along trajectory. Also 3 curves: dense L0(D_0), increment ΔL(D_0), absolute compressed L(D_0) — more training → bigger increment but NOT necessarily worse absolute compressed loss | results/v36b-input-comparison + PYTHIA_INPUT_COMPARISON.md | 2 sizes × 3 steps | 4-input LOSO, paired bootstrap | exp §prediction |
| C27 | Controlled training-history panel (Pythia, v36) — does D_0 (training tokens) predict compression response beyond dense loss? | 3-SIZE (160m+410m+1.4b × step16k/64k/143k, 160m verified distinct; pythia-2.8b DROPPED — current public HF files show DUPLICATION: step64000 & step143000 single-file blob hashes are identical upstream; step16000 differs but has sharded files whose actually-loaded file needs separate confirmation. So dedup of the duplicated stages is justified; not a claim that all 3 original checkpoints are corrupt). RAW SIGNAL STRONG: fragility rises sharply with training (1.4b prune ΔL@d0.6 +0.37→+1.15 early→late; int4 +0.08→+0.27). BUT held-out (leave-one-step-out) fit: adding D_0 does NOT reduce ΔL prediction error beyond {N0, dense L0} (4/12 point improvements, 0 CIs above 0). UPDATED (3-size clean panel, leave-one-size-out + leave-one-step-out): adding D_0 to {N0,L0} now LOWERS held-out MAE in 7/12 arm/cap/holdout comparisons, 4 with CIs wholly above 0 (large: quant-code LOSO A2.08->B0.63, quant-math A2.03->B0.77, pruning-math A0.62->B0.15). So on the clean 3-size panel D_0 DOES provide real predictive power beyond dense loss for MATH/CODE (QA noisy, exception). More accurately: the EXPANDED 3-size panel yielded incremental predictive evidence NOT DETECTABLE on the smaller 2-size panel (which already excluded 2.8b) — expanding the panel changed training coverage, the held-out object, AND fold composition simultaneously, so the gain is NOT attributable to any single cause (not specifically 'overfitting cured'). This does NOT prove dense loss captures everything See docs/CONTROLLED_PANEL_PREDICTOR.md for the FULL dual-baseline read: D0 beats {N0,L0} in 4/12 comparisons (CI excl 0), but vs the STRONGEST simple baseline (per-config median) the full model significantly wins ONLY for CODE (both arms); math CI includes 0; and BY-CONFIG nearly all quant gain is the int3 collapse region (b>=4 ~0) and pruning gain is at d=0.6 — the smooth region adds ~0; the gain also vanishes when the latest/max-D0 step is the held-out target. Predictor uses config INDICATORS, so it is source-transfer at FIXED d/b, not a law over the compression axis. — D_0 and dense loss are highly correlated along the trajectory (hard to separate), and a CI including 0 can still hold a meaningful gain. D_0 is KEPT as a recorded candidate. (Distinct question from predicting dense capability itself from N,D_0.) Direct ΔL fit (not a_c-label-regression). Fixed-recipe SERIES (arch/hparams still vary across Pythia sizes). Distill arm DONE (the cluster smoke, pythia-1.4b step16k/143k, D=600): D_0 modulates distillation too — more training -> more code damage (+0.018->+0.066) + more QA improvement (-0.589->-1.149); QA improves under KD at both. So ALL 3 arms show D_0-dependent responses on Pythia. CAVEAT: Pythia distill fell back to FULL-FT (peft absent on the cluster tf5), not LoRA -> within-Pythia D_0 comparison valid. LoRA CONTROL (v12 assertion mode=lora, the workstation idx3) confirms the training-stage effect HOLDS under LoRA (early->late: code +0.19->+0.32, qa -0.43->-0.77, same direction as full-FT); LoRA and full-FT show a CONSISTENT training-stage TREND across the two tested stages, but with clear response-MAGNITUDE differences (NOT 'recipe only changes magnitude' — e.g. early-math full-FT -0.005 vs LoRA +0.064, both near zero); the training-stage effect is not a full-FT artifact. Needs a 3rd clean size for leave-one-size-out | results/v36-pythia-controlled + PYTHIA_CONTROLLED.md + pythia-smoke | 2 sizes × 3 steps | leave-one-step-out, direct fit | exp §prediction |
| C26 | Unified information-budget diagnosis K0/K1/Oracle (v35, 3 arms, 0.1-nat tol) — separates the 3 failure causes per arm | K0=pre-compression basic-input (0 cal, MAIN goal); K1=+1 dev-fixed calibration point (prune d0.9, quant 4-bit, distill D75); Oracle=post-hoc all-target rescale (DIAGNOSTIC upper bound, NOT a predictor). PRUNING: appreciable Oracle FORM error on all dev caps; Qwen math/code Oracle≪K0 → source-MAPPING failure; K1(0.9) WORSENS math/QA → calibration UNSTABLE; good Oracle = post-hoc compatibility, NOT independent shape prediction. QUANT: K0 useful only in collapse region; K1(4-bit) WORSENS all; Oracle int3-zero is a fit artifact; 4-bit/QA mismatch remains; prospective shape INCONCLUSIVE. DISTILL: dev math/code K0 accurate, D75 cal worsens it; QA Oracle form error; Qwen4 cal repairs math/QA source-shift but worsens code; Oracle code-zero is a 1-point-fit artifact → NO shape transfer established. So the 3 arms have DIFFERENT gaps (form / mapping / calibration), not a single unified failure | results/v35-info-budget + INFO_BUDGET_K0K1.md | 12 dev + prospective rows | K0/K1/Oracle paired-model bootstrap | exp §prediction |
| C25b | Pruning-failure diagnosis (v29): WHY basic-param pruning prediction fails, decomposed | (1) The "0.33 vs 1.3-1.9" headline gap is DOMINATED BY DAMAGE-RANGE SELECTION, not form/calibration: controlled Mode-A bridge attributes combined training+scoring range effect 1.43 [0.96,1.93] nats vs form change −0.19 [−0.56,0.20] and historical target-calibration access ~0 (0.00003 [−0.012,0.015]) — EXCLUDES "new form is worse" and "old form cheated with target calibration". (2) On the FULL range NO basic-param predictor beats zero-change (v28_A gain over zero 0.07 [−0.43,0.57]); on smooth pre-cliff rows density_only≈zero (~0.30 both), and v28's metadata mapping HURTS smooth math/code (noisy on 12-model panel). (3) Qwen3-8B failure is AMPLITUDE/SIGN, NOT shape: amplitude/sign MSE fraction math 0.99998/code 0.99975/qa 0.983; best positive scale math 0.08 code 0.24 (predictor overshoots 4-12×, Qwen3-8B anomalously prune-robust), QA scale −2.21 = SIGN REVERSAL (QA loss DROPS under pruning d0.6 −1.02). Density SHAPE transfers; AMPLITUDE coefficient a_c does NOT. (4) One-point d=0.9 calibration CATASTROPHIC (near-zero 0.9 response → 934× amplification, v28_B pooled 15.2); nested shrinkage stabilizes dev (MAE reduction excl 0 for math/qa/pooled) but still loses to zero on Qwen3-8B. DESCRIPTOR TEST (v32, signed a_c, LOMO): adding weight & layerwise-retention descriptors (V2) does NOT help — V2 is SIGNIFICANTLY WORSE than the basic-param V1 on all caps (math MAE 2.07 vs 0.87, code 2.15 vs 0.90, qa 1.84 vs 0.88; gain-over-V1 CIs all exclude 0 = worse), overfitting a small panel; V1 itself doesn't beat zero-change (math gain +0.28 [−0.13,+0.70]). So the layerwise-retention distribution does NOT rescue amplitude prediction here. CONFIRMED on 11-model cohort (muse-30b added via a tf5 env; only gemma4-31b excluded — library-unsupported by ANY transformers incl git-main, no Gemma4Config/remote code): V2 still significantly worse than V1 (math −0.85 [−1.65,−0.16], code −0.81 [−1.63,−0.11], qa −0.80 [−1.49,−0.15] gain-over-V1, all exclude 0). So the retention-descriptor negative is robust. V3 (dense-activation / Wanda-style descriptors, 11 models, big-model activations via the cluster B200 tf5) ALSO does NOT help — V3 worse than V1 (gain-over-V1 math −0.94, code −1.26 [excl 0], qa −0.58). So NEITHER layerwise-retention (V2) NOR activation (V3) descriptors rescue signed-a_c prediction; the pruning amplitude a_c is unpredictable from ALL tested descriptors (basic params, retention, activations) on this panel — a complete negative for the descriptor line. Amplitude a_c remains unpredictable from these dense descriptors | results/v29-prune-diagnosis + v32-descriptors + PRUNE_FAILURE_DIAGNOSIS.md/PRUNE_DESCRIPTORS.md | 12 LOMO + 2 prospective; V2 on 10 | matched-condition bridge + oracle-scale + shrinkage + nested descriptor LOMO | exp §prune |


### C28 — P1 frozen prospective (step96000, never-fit source-state)
Frozen 9-state predictor tested on pythia-{160m,1.4b}@step96000 (forward-only, pre-registered coeffs):
**PRUNING generalizes** — full-input beats baseline AND no-D0 for math/code/QA (math 0.090 vs 0.367;
code 0.128 vs 0.438; qa 0.426 vs 0.832). **QUANTIZATION is MIXED** (not whole-arm failure) — quant-math beats both controls; quant-qa beats constant
but not no-D0; quant-code FAILS (0.78 > baseline 0.35). Honest problem: advantage concentrated in aggressive
(int3) quant + damage-MAGNITUDE prediction unstable there — int3 response is a REAL target, not an 'artifact'. n=8/arm/cap,
point estimates. See CONTROLLED_PANEL_PREDICTOR.md §7. This is the round's strongest positive (a real
frozen prospective, not in-sample) and a MIXED quant result (aggressive-region-concentrated, magnitude-unstable).


### C29 — P2 distillation controlled source-transfer (LoRA panel, 9 cells)
Full 3-size x 3-step LoRA distillation panel (fixed teacher gpt-5.6-luna, recipe full, pool 600, E=2,
seed 0). Predict delta_c = L_c(S_KD)-L_c(S0) from {N0,L0,[D0]} (no config indicators), leave-one-size/step-out.
**MATH: source-transfer works** — {N0,L0,D0} beats no-D0 AND the constant-delta baseline in both splits
(MAE ~halved, e.g. 0.016 vs baseline 0.038); delta_math grows smoothly/monotonically with D0.
**CODE/QA: fail** — a per-capability CONSTANT delta is as good or better; D0 adds nothing (delta_code
direction inconsistent across sizes; delta_qa noisy with a sign-flip outlier at 160m@143k). Distillation
gives a WEAKER, capability-specific source-transfer relationship than pruning (C28). n=9, point MAE only.
See DISTILL_CONTROLLED.md.


### C30 — Pruning source × compression-strength (round-6 SECOND AXIS, v40)
Frozen shared-shape candidate ΔL̂_c(x,d)=A_c(x)·((1−d)/0.3)^γ_c (A_c amplitude linear in {1,z logN0,z L0,z logD0},
γ_c shared), fit on DEV (9-state grid, seen d={0.9..0.6}), predicts UNSEEN densities on a small source×strength
panel incl. NEW source 410m@96k. **Beats every compression-strength-only baseline decisively**: aggregate MAE
math 0.371 / code 0.524 / qa 0.759 vs strength-only 1.37/1.64/1.61, median-curve 1.48/1.87/1.92, zero 2.37/2.73/1.77.
Win driven by source-conditioned amplitude (source-blind curves miss that 160m@143k is ~4× more fragile).
Interpolation (d=0.65) strong incl. new source (math err 0.03, code 0.13); deeper extrap (d=0.55) systematically
UNDER-predicts near the cliff (pre-declared domain edge, not dropped); QA amplitude over-predicted (doesn't transfer).
FIRST result predicting UNSEEN compression strength, not just source-transfer at fixed d. See PRUNE_STRENGTH_AXIS.md.
**Package-A same-input correction (PRUNE_SAMEINPUT.md, retrospective):** the win is a SOURCE-INFORMATION win, NOT proof of the shared power form — a simpler per-density source regression + interp (A2) ties/beats the candidate for math (0.283 vs 0.371) and code (0.279 vs 0.524); candidate only beats the gamma=1 variant (A1). So: keep 'source-conditioned inputs predict unseen density', DROP 'exponent independently confirmed'. Signed errors at d=0.55 are MIXED (6 under / 3 over; 160m@143k OVER-predicted), not systematic under-prediction.

### C30b — Quant partition on new sources (QUANT_PARTITION.md)
Quant has no unseen-bit axis; on 3 new @96k sources, >=4bit errors are tiny (nothing to predict), int3 collapse magnitude UNSTABLE (candidate wins on 410m/1.4b math, fails 160m-code/410m-qa). Predicts int3-collapse RISK, not a smooth magnitude law. Confirms C28. **Package-C reword (QUANT_PARTITION.md/v44, retrospective):** describe as a DISCRETE-config validation range ({8,6,4,3}, no integer between 3 and 4), NOT 'no unseen-bit axis / no law'. >=4-bit: near-zero responses, no incremental value over a per-bit baseline DEMONSTRATED (not a proof of none). int3: REAL collapse (not artifact), candidate transfer unstable across sources (math +0.45, code -1.12, qa -0.32). No collapse-RISK law claimed. v38(160M/1.4B) vs v40(410M) kept as separate source measurements.


### C31 — Distillation new-pool prediction (P3, v41)
Predictors frozen on U75/U600 endpoint trajectories (18 ckpts), predict unseen middle pool U=225 (6 runs =
3 pool-seeds × 2 train-seeds; v12 --data-seed). **QA: reuse-count E law transfers** (E-only MAE 0.706 vs
constant 1.449). **math/code: per-capability CONSTANT wins** (E/T/2D over-extrapolate; δ small & pool-invariant).
**Variability dominated by POOL-SAMPLING not training noise**: QA pool-std 0.427 vs train-std 0.043 (~10×),
code ~4× — the distillation residual is a data-SELECTION effect, not optimization stochasticity. Small (6 runs); endpoint pools first-U vs U=225 sampled. See DISTILL_NEWPOOL.md.
**Package-B residual correction (DISTILL_RESIDUALS.md, retrospective):** analyzing prediction residuals r=delta-delta_hat (not raw-delta std) shows the U225 error is dominated by SYSTEMATIC BIAS of the frozen predictors, NOT pool variation: math/code (constant) have small over-prediction bias (~-0.16/-0.11) with tiny variation; QA (E-only) has a LARGE budget-dependent bias that SIGN-FLIPS (-0.87 at b0 -> +0.48 at b1) = a form misfit, so E-only merely beats a constant, it does not cleanly transfer. Around the bias, residual variation is pool>training but small. Protocol: endpoints first-U, U225 random-sampled -> 'unseen pool size AND changed sampling', U225 E interpolates (2.34-4.70 in 0.94-14.94). DROP the earlier 'pool-std/train-std~10x explains the residual / data-selection effect' causal phrasing.


### C32 — P1 new-source frozen prospective (P-new, pythia-1b@step96000; v46; frozen b1bf631)
New size (0.81B matrix params, inside 160M–1.4B) at an interpolated stage; never used before. Predictions
committed before measurement. PRUNING d=0.65 (interp): source-conditioned forms succeed and are indistinguishable
(A1 0.149 / A2 0.172 / power 0.193 vs strength-only 0.704, zero 0.657; math/code err 0.02–0.09; QA weakest).
PRUNING d=0.55 (extrap): ALL source-conditioned forms fail (1.14–1.69), under-predicting collapse by 1–1.5 nat and
missing QA's sign; strength-only closest (0.273) → transfer range = interpolation regime; no A3. QUANT: frozen
full {N0,L0,D0} best at int4 (0.074 vs no-D0 0.211, median 0.111) and int3 (0.397 vs 0.620, 2.46). One source;
no population claim. P1_NEWSOURCE.md; ≈1 GPU-min.

### C33. P1-v2 first source pair: Pythia-1B at step 32k and 112k (frozen, prospective)

Protocol: predictions for all candidates frozen and committed before measurement (`results/v49-p1v2/register.json`,
`predictions_pythia-1b@step*.json`); measured 2026-09-09 15:47-16:12 EDT on the A100 (22 GPU-min for the pair);
7 densities (0.9, 0.8, 0.75, 0.7, 0.65, 0.6 interpolation; 0.55 extrapolation) and 5 bit-widths (8, 6, 5, 4, 3; the 5-bit
prediction is the fixed interpolation rule of the 4- and 6-bit predictions). Protocol A fits on the full dev panel,
Protocol B on dev steps <= 64k only. Table generated by `analysis/v49_p1v2_ledger.py`.

Reading (bold = best per row):
- Pruning, interpolation regime. At 112k the source-conditioned A2 candidate is best under both protocols (A: 0.112 vs
  0.173 median curve and 0.365 strength-only; B: 0.177 vs 0.195 / 0.219) and best on each capability separately.
  At 32k the picture is weaker: under Protocol A the source-free median curve wins (0.204 vs A2 0.261), driven by QA
  (median 0.193 vs A2 0.339); under Protocol B A2 is first by a small margin (0.218 vs 0.224 / 0.236).
  So the C32 result (1B@96k) replicates at the late checkpoint and only partially at the early one; the early-checkpoint
  QA response appears to be the difficult component.
- Pruning, extrapolated density 0.55. Under Protocol A strength-only is far better (0.637 / 0.311) and every source-
  conditioned candidate fails (1.2-2.3 nats). Under Protocol B at 112k the power / A2 candidates are ahead (0.957 / 0.995
  vs 1.333), but all errors are near or above 1 nat, so nothing is usable there. Transfer range remains interpolation.
- Quantization. The config-indicator candidate ("full") pays off only at 3-bit, where damage is 3-6 nats and it cuts the
  error to 0.6-1.1 nats (per-bit median 0.8-3.5, zero 3.1-5.8). At >= 4 bits the per-bit median is as good or better
  (aggregate A: 0.031 / 0.036 vs full 0.103 / 0.071); the 5-bit interpolation rule applied to "full" is clearly worse than
  the median (0.15-0.18 vs 0.035-0.044 under A). Precision-regime dependence, as already stated in the paper, holds here.

Label: pruning A2 at 112k P-new (interp); at 32k P-new only under Protocol B; extrapolation R (both); quantization
P-new at 3-bit, R at >= 4-bit vs the median baseline.

#### pythia-1b@step32000

| prune MAE (nats) | power | A2 | A1 | cont | strength_only | median_curve | zero |
|---|---|---|---|---|---|---|---|
| A interp d0.9-0.6 | 0.254 | 0.261 | 0.393 | 0.275 | 0.363 | **0.204** | 0.412 |
| A extrap d0.55 | 2.272 | 2.130 | 2.394 | 2.221 | **0.637** | 1.509 | 2.435 |
| A interp math | 0.303 | 0.298 | 0.465 | 0.315 | **0.221** | 0.270 | 0.599 |
| A interp code | 0.198 | **0.147** | 0.322 | 0.183 | 0.317 | 0.149 | 0.368 |
| A interp qa | 0.261 | 0.339 | 0.393 | 0.328 | 0.550 | **0.193** | 0.271 |
| B interp d0.9-0.6 | 0.249 | **0.218** | 0.321 | 0.239 | 0.236 | 0.224 | 0.412 |
| B extrap d0.55 | 1.772 | 1.777 | 2.150 | 1.863 | **1.531** | 1.645 | 2.435 |
| B interp math | 0.342 | 0.343 | 0.461 | 0.350 | **0.325** | 0.326 | 0.599 |
| B interp code | 0.165 | **0.139** | 0.321 | 0.176 | 0.148 | 0.148 | 0.368 |
| B interp qa | 0.239 | **0.172** | 0.182 | 0.192 | 0.234 | 0.198 | 0.271 |

| quant MAE (nats) | full | noD0 | per_bit_mean | per_bit_median | zero |
|---|---|---|---|---|---|
| A int8 | 0.003 | 0.005 | 0.005 | **0.001** | 0.001 |
| A int6 | 0.028 | 0.039 | 0.041 | 0.012 | **0.008** |
| A int5 (rule interp 4,6) | 0.179 | 0.272 | 0.254 | 0.044 | **0.017** |
| A int4 | 0.204 | 0.380 | 0.487 | **0.067** | 0.157 |
| A int3 | **1.050** | 2.648 | 2.903 | 1.438 | 3.108 |
| A >=4-bit aggregate | 0.103 | 0.174 | 0.197 | **0.031** | 0.046 |
| B int8 | 0.002 | 0.003 | **0.001** | 0.002 | 0.001 |
| B int6 | **0.004** | 0.005 | 0.010 | 0.009 | 0.008 |
| B int5 (rule interp 4,6) | 0.025 | **0.013** | 0.035 | 0.034 | 0.017 |
| B int4 | 0.080 | 0.106 | 0.061 | **0.060** | 0.157 |
| B int3 | **0.606** | 1.003 | 0.627 | 0.803 | 3.108 |
| B >=4-bit aggregate | 0.028 | 0.032 | 0.027 | **0.026** | 0.046 |

#### pythia-1b@step112000

| prune MAE (nats) | power | A2 | A1 | cont | strength_only | median_curve | zero |
|---|---|---|---|---|---|---|---|
| A interp d0.9-0.6 | 0.191 | **0.112** | 0.301 | 0.175 | 0.365 | 0.173 | 0.400 |
| A extrap d0.55 | 1.398 | 1.235 | 1.838 | 1.356 | **0.311** | 1.311 | 2.237 |
| A interp math | 0.127 | **0.111** | 0.355 | 0.160 | 0.117 | 0.150 | 0.489 |
| A interp code | 0.183 | **0.091** | 0.343 | 0.160 | 0.336 | 0.107 | 0.348 |
| A interp qa | 0.262 | **0.135** | 0.204 | 0.204 | 0.643 | 0.261 | 0.365 |
| B interp d0.9-0.6 | 0.223 | **0.177** | 0.355 | 0.235 | 0.219 | 0.195 | 0.400 |
| B extrap d0.55 | **0.957** | 0.995 | 1.685 | 1.127 | 1.333 | 1.447 | 2.237 |
| B interp math | 0.133 | **0.124** | 0.349 | 0.167 | 0.214 | 0.214 | 0.489 |
| B interp code | 0.111 | **0.066** | 0.335 | 0.158 | 0.107 | 0.107 | 0.348 |
| B interp qa | 0.425 | 0.340 | 0.380 | 0.379 | 0.336 | **0.264** | 0.365 |

| quant MAE (nats) | full | noD0 | per_bit_mean | per_bit_median | zero |
|---|---|---|---|---|---|
| A int8 | 0.006 | 0.004 | 0.004 | 0.002 | **0.001** |
| A int6 | 0.022 | 0.029 | 0.034 | **0.005** | 0.005 |
| A int5 (rule interp 4,6) | 0.151 | 0.138 | 0.251 | 0.035 | **0.025** |
| A int4 | 0.106 | 0.272 | 0.443 | **0.103** | 0.222 |
| A int3 | **0.676** | 3.805 | 0.802 | 1.231 | 5.777 |
| A >=4-bit aggregate | 0.071 | 0.111 | 0.183 | **0.036** | 0.063 |
| B int8 | 0.002 | 0.004 | 0.002 | 0.003 | **0.001** |
| B int6 | 0.012 | 0.003 | 0.004 | **0.002** | 0.005 |
| B int5 (rule interp 4,6) | 0.046 | **0.019** | 0.027 | 0.025 | 0.025 |
| B int4 | **0.094** | 0.142 | 0.120 | 0.125 | 0.222 |
| B int3 | **1.120** | 1.981 | 3.106 | 3.472 | 5.777 |
| B >=4-bit aggregate | 0.038 | 0.042 | **0.038** | 0.039 | 0.063 |

### C34. P1-v2 second source pair: Pythia-6.9B at step 32k and 112k (frozen, prospective, ~5x size extrapolation)

Protocol: identical to C33; predictions committed 950312f (15:55:58 EDT) before the measurement started (16:12:04);
measured 16:12-16:48 EDT on the A100, reference model on CPU (36 GPU-min for the pair). Dense anchors: 32k
math 1.268 / code 1.299 / QA 5.108; 112k math 1.114 / code 1.210 / QA 5.086. Table generated by `analysis/v49_p1v2_ledger.py`.

Reading (bold = best per row):
- Pruning, interpolation regime, Protocol A (full dev panel). The source-conditioned family fails at this size:
  A2 0.482 / 0.463 (32k / 112k) against the source-free median curve 0.165 / 0.092 and zero change 0.160 / 0.294.
  The miss is mostly QA: the source regression extrapolates the panel's size trend into large QA improvements
  (A2 predicts -0.26 to -3.6 nats over d 0.8-0.55 at 32k) while the measured QA response stays small (-0.05 to -0.63)
  and reverses to +1.45 at d=0.55 for 112k. Math and code are over-predicted by roughly 2x at mid densities
  (112k, d=0.65: actual 0.48 / 0.47, A2 0.97 / 0.89), i.e. the 6.9B source is more robust per density than the
  <=1.4B trend implies.
- Pruning, interpolation regime, Protocol B (dev step<=64k). Predictions are far better (A2 0.166 / 0.106) but still
  never beat the median curve (0.136 / 0.095); strength-only is on par (0.163 / 0.111). On code the source forms lead
  (A2 0.058 / 0.083 vs median 0.208 / 0.074), on QA they trail (0.324 / 0.203 vs 0.084 / 0.154).
- Pruning, extrapolated density 0.55: source-free curves win under both protocols (A: strength-only 0.19 at 112k,
  median 0.274 at 32k); source-conditioned errors 0.55-2.2 nats.
- Quantization. The config-indicator candidate does not transfer: at >= 4 bits under Protocol A it is worst or near
  worst (0.253 / 0.271 vs median 0.034 / 0.042); at 3 bits it overshoots by ~4 nats (3.97 / 3.87 vs per-bit median
  1.83 / 1.90 and per-bit mean 3.30 / 0.43). Protocol B narrows everything to the 0.02-0.05 range at >= 4 bits, and
  at 3 bits "full" is best only at 112k (1.29 vs 3.8-4.1).

Label: R for the size-extrapolation claim in both arms under both protocols (the source-conditioned family is at best
on par with the source-free median curve, never better). Together with C33 (in-range size: P-new at the late stage),
the transfer range of the current source-conditioned predictors is bounded on the size axis at roughly the development
range, in addition to the density bound already recorded (C32, C33).

#### pythia-6.9b@step32000

| prune MAE (nats) | power | A2 | A1 | cont | strength_only | median_curve | zero |
|---|---|---|---|---|---|---|---|
| A interp d0.9-0.6 | 0.476 | 0.482 | 0.608 | 0.501 | 0.483 | 0.165 | **0.160** |
| A extrap d0.55 | 1.635 | 1.423 | 0.939 | 1.457 | 1.493 | **0.274** | 0.889 |
| A interp math | 0.357 | 0.350 | 0.460 | 0.380 | 0.410 | 0.187 | **0.158** |
| A interp code | 0.106 | **0.095** | 0.170 | 0.104 | 0.561 | 0.214 | 0.124 |
| A interp qa | 0.966 | 1.001 | 1.193 | 1.019 | 0.478 | **0.092** | 0.198 |
| B interp d0.9-0.6 | 0.160 | 0.166 | 0.249 | 0.187 | 0.163 | **0.136** | 0.160 |
| B extrap d0.55 | 0.705 | 0.549 | 0.677 | 0.630 | 0.252 | **0.148** | 0.889 |
| B interp math | 0.123 | 0.117 | 0.187 | 0.143 | 0.117 | **0.116** | 0.158 |
| B interp code | 0.074 | **0.058** | 0.149 | 0.080 | 0.211 | 0.208 | 0.124 |
| B interp qa | 0.284 | 0.324 | 0.411 | 0.338 | 0.162 | **0.084** | 0.198 |

| quant MAE (nats) | full | noD0 | per_bit_mean | per_bit_median | zero |
|---|---|---|---|---|---|
| A int8 | 0.005 | 0.010 | 0.005 | 0.001 | **0.001** |
| A int6 | 0.057 | 0.070 | 0.033 | 0.004 | **0.003** |
| A int5 (rule interp 4,6) | 0.373 | 0.457 | 0.237 | 0.067 | **0.029** |
| A int4 | 0.576 | 0.730 | 0.484 | **0.064** | 0.098 |
| A int3 | 3.972 | 5.103 | 3.300 | **1.834** | 2.711 |
| A >=4-bit aggregate | 0.253 | 0.317 | 0.190 | 0.034 | **0.033** |
| B int8 | 0.007 | 0.007 | 0.001 | 0.002 | **0.001** |
| B int6 | 0.019 | 0.018 | 0.004 | **0.002** | 0.003 |
| B int5 (rule interp 4,6) | 0.047 | 0.052 | 0.058 | 0.057 | **0.029** |
| B int4 | 0.057 | 0.122 | 0.055 | **0.038** | 0.098 |
| B int3 | 0.489 | 1.792 | **0.174** | 0.407 | 2.711 |
| B >=4-bit aggregate | 0.032 | 0.050 | 0.029 | **0.025** | 0.033 |

#### pythia-6.9b@step112000

| prune MAE (nats) | power | A2 | A1 | cont | strength_only | median_curve | zero |
|---|---|---|---|---|---|---|---|
| A interp d0.9-0.6 | 0.436 | 0.463 | 0.610 | 0.479 | 0.375 | **0.092** | 0.294 |
| A extrap d0.55 | 2.121 | 1.939 | 2.223 | 2.061 | **0.190** | 1.272 | 2.198 |
| A interp math | 0.242 | 0.233 | 0.368 | 0.267 | 0.251 | **0.053** | 0.318 |
| A interp code | 0.214 | 0.194 | 0.362 | 0.218 | 0.358 | **0.074** | 0.326 |
| A interp qa | 0.852 | 0.962 | 1.100 | 0.952 | 0.518 | **0.148** | 0.238 |
| B interp d0.9-0.6 | 0.131 | 0.106 | 0.235 | 0.140 | 0.111 | **0.095** | 0.294 |
| B extrap d0.55 | 1.441 | 1.387 | 1.892 | 1.525 | **1.294** | 1.408 | 2.198 |
| B interp math | 0.053 | **0.031** | 0.249 | 0.099 | 0.057 | 0.058 | 0.318 |
| B interp code | 0.102 | 0.083 | 0.302 | 0.132 | 0.074 | **0.074** | 0.326 |
| B interp qa | 0.236 | 0.203 | **0.153** | 0.188 | 0.200 | 0.154 | 0.238 |

| quant MAE (nats) | full | noD0 | per_bit_mean | per_bit_median | zero |
|---|---|---|---|---|---|
| A int8 | 0.026 | 0.018 | 0.018 | **0.013** | 0.014 |
| A int6 | 0.062 | 0.039 | 0.040 | 0.011 | **0.010** |
| A int5 (rule interp 4,6) | 0.402 | 0.247 | 0.253 | 0.033 | **0.027** |
| A int4 | 0.595 | 0.428 | 0.309 | **0.112** | 0.191 |
| A int3 | 3.873 | 3.550 | **0.430** | 1.896 | 6.442 |
| A >=4-bit aggregate | 0.271 | 0.183 | 0.155 | **0.042** | 0.061 |
| B int8 | 0.009 | **0.007** | 0.013 | 0.012 | 0.014 |
| B int6 | 0.018 | 0.008 | 0.010 | **0.008** | 0.010 |
| B int5 (rule interp 4,6) | 0.043 | 0.025 | 0.025 | **0.023** | 0.027 |
| B int4 | 0.121 | 0.198 | **0.120** | 0.138 | 0.191 |
| B int3 | **1.294** | 4.033 | 3.770 | 4.137 | 6.442 |
| B >=4-bit aggregate | 0.048 | 0.060 | **0.042** | 0.045 | 0.061 |

### C35. V3-P pruning test panel: frozen selected predictor (shared power form) on three new checkpoints at d = 0.85 / 0.675 / 0.575

Protocol: v53 development on 17 Pythia states (9 dev + 160M/410M/1.4B@96k + 1B@32k/96k/112k + 6.9B@32k/112k; all measured densities >= 0.55), LOSO by source; pre-committed rule chose the shared power form (5 params/cap) over A2 by the tie rule (LOSO mean 0.494 vs 0.479, gap 0.015 < 0.02). All candidate + baseline predictions for the three test checkpoints were committed in paper repo 40a51b7 (21:18:52 EDT) before measurement (21:52-22:11 EDT, ~18 GPU-min). Checkpoint identity verified (HF shard hashes distinct; dense vectors differ from neighbours). Files: results/v53-prune-dev/{register.json, predictions_*.json, compare_*.json}.


**pythia-410m@step48000** (observed dL | power | A2 | median curve | zero), nats

| cap | d | observed | power | A2 | median | strength-only |
|---|---|---|---|---|---|---|
| math | 0.85 | +0.015 | +0.032 | +0.041 | +0.032 | +0.010 |
| math | 0.675 | +0.260 | +0.589 | +0.471 | +0.480 | +0.567 |
| math | 0.575 | +0.953 | +1.612 | +2.023 | +1.885 | +2.317 |
| code | 0.85 | +0.018 | +0.048 | -0.011 | +0.027 | +0.018 |
| code | 0.675 | +0.252 | +0.510 | +0.238 | +0.330 | +0.670 |
| code | 0.575 | +1.140 | +1.156 | +1.574 | +1.875 | +2.362 |
| qa | 0.85 | -0.026 | +0.050 | +0.045 | -0.045 | +0.002 |
| qa | 0.675 | -0.212 | +0.524 | +0.393 | -0.316 | +0.197 |
| qa | 0.575 | -0.166 | +1.187 | +1.223 | +0.364 | +0.984 |

**pythia-1.4b@step112000** (observed dL | power | A2 | median curve | zero), nats

| cap | d | observed | power | A2 | median | strength-only |
|---|---|---|---|---|---|---|
| math | 0.85 | +0.026 | +0.030 | +0.022 | +0.032 | +0.010 |
| math | 0.675 | +0.379 | +0.540 | +0.381 | +0.480 | +0.567 |
| math | 0.575 | +1.597 | +1.477 | +1.568 | +1.885 | +2.317 |
| code | 0.85 | +0.027 | +0.073 | +0.029 | +0.027 | +0.018 |
| code | 0.675 | +0.267 | +0.772 | +0.497 | +0.330 | +0.670 |
| code | 0.575 | +1.547 | +1.750 | +1.955 | +1.875 | +2.362 |
| qa | 0.85 | -0.011 | +0.050 | +0.001 | -0.045 | +0.002 |
| qa | 0.675 | -0.645 | +0.529 | +0.151 | -0.316 | +0.197 |
| qa | 0.575 | -0.142 | +1.199 | +1.371 | +0.364 | +0.984 |

**pythia-6.9b@step80000** (observed dL | power | A2 | median curve | zero), nats

| cap | d | observed | power | A2 | median | strength-only |
|---|---|---|---|---|---|---|
| math | 0.85 | +0.019 | +0.034 | +0.031 | +0.032 | +0.010 |
| math | 0.675 | +0.241 | +0.623 | +0.513 | +0.480 | +0.567 |
| math | 0.575 | +1.204 | +1.704 | +1.647 | +1.885 | +2.317 |
| code | 0.85 | +0.005 | +0.073 | +0.056 | +0.027 | +0.018 |
| code | 0.675 | +0.222 | +0.770 | +0.566 | +0.330 | +0.670 |
| code | 0.575 | +1.295 | +1.745 | +1.779 | +1.875 | +2.362 |
| qa | 0.85 | -0.078 | -0.050 | -0.203 | -0.045 | +0.002 |
| qa | 0.675 | -0.379 | -0.526 | -1.142 | -0.316 | +0.197 |
| qa | 0.575 | -0.004 | -1.193 | -0.869 | +0.364 | +0.984 |

**Panel MAE (mean over the three checkpoints and three densities), nats**

| candidate | math | code | qa | mean |
|---|---|---|---|---|
| power | 0.243 | 0.236 | 0.678 | 0.386 |
| A1 | 0.363 | 0.403 | 0.738 | 0.501 |
| A2 | 0.230 | 0.222 | 0.682 | 0.378 |
| cont | 0.313 | 0.305 | 0.746 | 0.455 |
| strength_only | 0.450 | 0.488 | 0.579 | 0.506 |
| median_curve | 0.277 | 0.214 | 0.221 | 0.237 |
| zero | 0.522 | 0.530 | 0.185 | 0.412 |

Reading: the selected source-conditioned form beats the source-free strength curve and zero change on every checkpoint
and is the best predictor of the math response (A2 near-exact at 1.4B@112k); on code it is on par with the median curve;
on QA it is worse than every source-free curve by 0.3-0.6 nats on each checkpoint, because it over-predicts the QA
improvement. The source-free median development curve has the lowest aggregate error on all three checkpoints. Answer to
the round's question: with source state admitted, a compact continuous relation (5 params) predicts unseen densities
inside the fitted range for math and code at the level of a per-density interpolation; for QA the current inputs carry
no usable signal and a source-free curve should be delivered instead. No exponent universality is claimed
(gamma is a fitted constant per capability on this panel). Labels: P-new for the selected form on math (better than every
baseline at 1.4B@112k and 6.9B@80k; ties median at 410M@48k), R on QA at all three checkpoints; aggregate R vs the
median curve.



### C36. V3-Q grouped-RTN quantization: frozen predictors on unseen bit-width, unseen granularity, and a new state

Protocol: v54 measures symmetric RTN with contiguous input groups of g weights (per-group absmax scale; g=None reproduces the per-channel quantizer bit-exactly). Dev = 160M/410M/1.4B x {16k,143k} x b{3,5} x g{64,256} (24 cells). Candidates fit per capability on standardized phi=[1,z(logN0),z(L0c),z(logD0)]: separable (beta.phi) qmax^-p (g/128)^q (6 params); low-order 2D phi x [1,u,v,uv,u^2] with u=log2 qmax centered, v=log2(g/128) (20 params); same-input bilinear interpolation between the four dev configs (16 params); baselines per-config mean/median interpolated the same way, zero. Dev LOSO (6 states, dominated by the collapsing 160M@143k): separable 1.95/2.19/4.75, 2D 0.91/1.91/4.37, interp 1.23/1.89/4.33, median 1.69/1.76/1.78, zero 1.82/1.93/1.80 (math/code/qa). All 33 test predictions committed in paper repo 5ad319c (21:53 EDT) before the tests were measured (21:53-22:41 EDT; whole 57-cell grid ~85 min of forward passes as a third process on the card). Rule: all candidates and baselines reported on every test; no selection by outcome.

**Test MAE (nats; from results/v55-quant-group/compare.json as printed by `v55 compare` at 22:43 EDT)**

| test | candidate | math | code | qa |
|---|---|---|---|---|
| bit test: b=4 unseen, g in {64,256}, six dev states (12 cells) | separable | 0.767 | 0.798 | 1.427 |
| bit test: b=4 unseen, g in {64,256}, six dev states (12 cells) | low_order_2d | 0.193 | 0.214 | 0.436 |
| bit test: b=4 unseen, g in {64,256}, six dev states (12 cells) | same_input_interpolation | 1.296 | 1.381 | 1.942 |
| bit test: b=4 unseen, g in {64,256}, six dev states (12 cells) | mean | 1.416 | 1.556 | 1.495 |
| bit test: b=4 unseen, g in {64,256}, six dev states (12 cells) | median | 0.553 | 0.726 | 0.537 |
| bit test: b=4 unseen, g in {64,256}, six dev states (12 cells) | zero | 0.450 | 0.532 | 0.480 |
| granularity test: g=128 unseen, b in {3,4,5}, six dev states (18 cells) | separable | 0.581 | 0.643 | 1.612 |
| granularity test: g=128 unseen, b in {3,4,5}, six dev states (18 cells) | low_order_2d | 0.219 | 0.319 | 0.986 |
| granularity test: g=128 unseen, b in {3,4,5}, six dev states (18 cells) | same_input_interpolation | 0.606 | 0.711 | 1.497 |
| granularity test: g=128 unseen, b in {3,4,5}, six dev states (18 cells) | mean | 1.857 | 1.906 | 2.019 |
| granularity test: g=128 unseen, b in {3,4,5}, six dev states (18 cells) | median | 1.123 | 1.222 | 1.220 |
| granularity test: g=128 unseen, b in {3,4,5}, six dev states (18 cells) | zero | 1.239 | 1.346 | 1.239 |
| joint test: new state 1B@96k, g=128, b in {3,4,5} (3 cells) | separable | 0.586 | 0.380 | 0.679 |
| joint test: new state 1B@96k, g=128, b in {3,4,5} (3 cells) | low_order_2d | 0.345 | 0.097 | 0.269 |
| joint test: new state 1B@96k, g=128, b in {3,4,5} (3 cells) | same_input_interpolation | 0.538 | 0.241 | 0.392 |
| joint test: new state 1B@96k, g=128, b in {3,4,5} (3 cells) | mean | 1.491 | 1.558 | 1.813 |
| joint test: new state 1B@96k, g=128, b in {3,4,5} (3 cells) | median | 0.152 | 0.270 | 0.302 |
| joint test: new state 1B@96k, g=128, b in {3,4,5} (3 cells) | zero | 0.300 | 0.339 | 0.104 |

**Measured test responses (dL vs dense, math/code/qa, nats)**: results/v54-quant-group/*/quant_group_losses.json
(mirrored). Highlights: 4-bit is a 0.03-0.35 nat regime for every state except 160M@143k (+1.9 to +3.2) and shrinks with
g (g64 < g128 < g256); 3-bit at g=128 spans +0.3 (early 1.4B) to +14.5 (late 160M); 5-bit at g=128 is <= 0.1 except
late 160M (+0.4 to +0.55). 1B@96k: b3_g128 +0.83/+0.95/-0.27; b4_g128 +0.06/+0.05/+0.02; b5_g128 ~0.

Reading:
- The low-order 2D form predicts the unseen bit-width on the dev states with MAE 0.19/0.21/0.44 against 0.45/0.53/0.48
  for zero change and 0.55/0.73/0.54 for the per-config median (12 cells, all three capabilities incl. QA), and the
  unseen granularity with 0.22/0.32/0.99 against 1.24/1.35/1.24 (zero) and 1.12/1.22/1.22 (median) (18 cells; the
  collapse cell 160M@143k b3_g128 = +14.5 inflates every error). These are configuration-axis successes on SEEN source
  states (new configs, not new sources).
- The separable compact candidate A(x) qmax^-p (g/128)^q fails on every test (worse than zero on the bit test); the
  same-input bilinear interpolation also fails (extension beyond the dev corners). The response is not separable into a
  state amplitude times a shared (b,g) shape on this panel.
- Joint test on a new in-range state (1B@96k, 3 cells): 2D 0.34/0.10/0.27 vs median 0.15/0.27/0.30 and zero
  0.30/0.34/0.10: best on code, behind the median on math, behind zero on QA; too few cells to rank.
- Stop rule: not triggered (intermediate responses 0.05-3 nats exist across the grid).
Labels: low-order 2D = P-new on the bit test and the granularity test (seen states, unseen configs); mixed on the joint
new-state test (P-new on code, R on math and QA); separable and interpolation = R everywhere.

### C37. V3-D (CPU): student-descriptor forms and capability-conditioning test on the complete P2-v2 development matrix

Data: 12 dev runs (Gemma-3-270M/1B x U75/U450 x data-seeds 11-13; absolute-exposure protocol, 4 checkpoints each, 48 points); response delta_c = L_c(S_KD) - L_c(S_0); E = completion tokens / registered D_U. Analysis analysis/v56_distill_forms.py (results/v56-distill-forms/summary.{json,md}), run 2026-09-10 04:36 EDT after the v50 freeze (33c706c, 04:35). Forms F1 = (a_c + lambda_c z_c) log(1+E) and F2 = (a_c + lambda_c z_c)(1-e^{-T/T*}) + (b_c + mu_c z_c) log(1+E) with z_c = L0c (log N_S alternative identical up to sign with two dev students) were specified by the round-3 advice AFTER the v50 freeze design; their predictions for the test runs were NOT committed before measurement, so any evaluation on the test runs is retrospective (label R) and is reported separately from the frozen v50 forms.

Leave-one-run-out (12 clusters) MAE / signed bias (nats), math | code | qa:

| Form | P | LOCO math | LOCO code | LOCO qa | LOSO math | LOSO code | LOSO qa |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 0 | 0.152/-0.152 | 0.178/-0.178 | 1.429/+0.143 | 0.152/-0.152 | 0.178/-0.178 | 1.429/+0.143 |
| constant | 3 | 0.095/+0.000 | 0.103/+0.000 | 1.400/+0.000 | 0.093/+0.000 | 0.100/+0.000 | 1.358/-0.000 |
| T-only | 6 | 0.082/+0.000 | 0.072/+0.000 | 1.092/-0.000 | 0.089/+0.000 | 0.070/+0.000 | 1.075/-0.000 |
| E-only | 6 | 0.073/-0.003 | 0.055/-0.002 | 0.798/-0.029 | 0.091/+0.000 | 0.056/+0.000 | 0.949/+0.000 |
| surface:L0 | 12 | 0.072/+0.000 | 0.054/-0.000 | 0.812/+0.001 | 0.091/+0.000 | 0.054/+0.000 | 0.944/-0.000 |
| surface:logN | 12 | 0.072/+0.000 | 0.054/-0.000 | 0.812/+0.001 | 0.091/+0.000 | 0.054/+0.000 | 0.944/+0.000 |
| F1:L0 | 6 | 0.057/+0.003 | 0.052/+0.003 | 1.518/+0.662 | 0.152/-0.152 | 0.178/-0.178 | 1.429/+0.143 |
| F1:logN | 6 | 0.057/+0.003 | 0.052/+0.003 | 1.518/+0.662 | 0.152/-0.152 | 0.178/-0.178 | 1.429/+0.143 |
| F2:L0 | 12 | 0.062/-0.007 | 0.057/-0.005 | 0.706/-0.069 | 0.152/-0.152 | 0.178/-0.178 | 1.429/+0.143 |
| F2:logN | 12 | 0.062/-0.007 | 0.057/-0.005 | 0.706/-0.069 | 0.152/-0.152 | 0.178/-0.178 | 1.429/+0.143 |

Leave-one-student-out (2 folds): descriptor forms degenerate (z constant within a fold) and fall back to zero; only student-free forms are informative there.

Part B, capability conditioning at matched parameter count (shared response shape + per-capability offsets vs per-capability models), LOCO MAE:

| Structure | P shared/per | Capability | Shared MAE | Per-cap MAE | Gain | >0.02 |
|---|---:|---|---:|---:|---:|:---:|
| E-only | 4/4 | math | 0.431 | 0.409 | 0.022 | yes |
| E-only | 4/4 | code | 0.417 | 0.412 | 0.006 | no |
| E-only | 4/4 | qa | 0.967 | 1.150 | -0.183 | no |
| E-only | 4/4 | macro | 0.605 | 0.657 | -0.052 | no |
| T+E | 5/5 | math | 0.439 | 0.327 | 0.111 | yes |
| T+E | 5/5 | code | 0.419 | 0.314 | 0.105 | yes |
| T+E | 5/5 | qa | 0.979 | 0.853 | 0.126 | yes |
| T+E | 5/5 | macro | 0.612 | 0.498 | 0.114 | yes |
| F2 | 7/7 | math | 0.536 | 0.145 | 0.390 | yes |
| F2 | 7/7 | code | 0.527 | 0.107 | 0.420 | yes |
| F2 | 7/7 | qa | 1.044 | 0.826 | 0.219 | yes |
| F2 | 7/7 | macro | 0.702 | 0.359 | 0.343 | yes |

Reading: on the development matrix, F1 has the lowest math and code error (0.057, 0.052 vs 0.073/0.055 for E-only and 0.095/0.103 for a constant) and F2 the lowest QA error (0.706 vs 0.798 E-only, 1.40 constant), so a training-amount term plus a reuse term with a student descriptor describes QA better than reuse alone; per-capability shapes beat a shared shape for the T+E and F2 structures on every capability (gains 0.1-0.4 nats) and only marginally for E-only, so capability conditioning is required for the response shape, not only for the amplitude. These are cross-validated development results with two students; the frozen test (v50 forms) and the retrospective evaluation of F1/F2 on the nine test runs and the 4B dev-pool runs follow.

### C38. P2-v2 multi-student distillation test: frozen forms on 9 unseen-pool trajectories (data cut-off 2026-09-10 16:39 EDT)

Freeze: results/v50-p2v2/freeze.json committed 33c706c (04:35 EDT, before any test run) from the 12 dev runs
(Gemma-3-270M/1B x U75/U450 x data-seeds 11-13, 48 points). Eight forms per capability (constant/T/E/joint, each
with and without a source term k*log(N_S/N_ref)); no candidate was selected at freeze. Tests: U375 pools, seeds 21-23,
students 270M and 1B (same students, unseen pool) and 4B (held-out student, N_S 6.6x N_ref, outside the dev range of
log N_S); planned T {35k,70k,140k,280k}; compare committed c06c856. Selection rule for the headline row (stated after
the tests, so labeled R): the frozen form with the lowest in-sample dev MAE at freeze time = joint+src for all three
capabilities. MAE in nats over 12 points per cell (3 seeds x 4 budgets):

| student | cap | joint+src (dev-selected) | zero | best frozen form (retrospective) |
|---|---|---|---|---|
| 270M | math | 0.045 | 0.092 | constant+src 0.031 |
| 1B | math | 0.050 | 0.095 | E 0.026 |
| 4B | math | 0.089 | 0.153 | constant 0.020 |
| 270M | code | 0.036 | 0.138 | E 0.031 |
| 1B | code | 0.029 | 0.109 | E 0.018 |
| 4B | code | 0.050 | 0.178 | T 0.033 |
| 270M | QA | 0.606 | 1.005 | joint 0.544 |
| 1B | QA | 0.444 | 1.148 | joint 0.301 |
| 4B | QA | 1.141 | 1.167 | joint 0.272 |

Reading: on the unseen pool with seen students, the dev-selected frozen form beats zero on all six cells by 2-4x. On
the held-out 4B student it beats zero on code and math (math degrades with T: 0.038/0.024/0.11/0.18 at 35k-280k vs zero
0.14-0.20) and fails on QA (1.14 vs 1.17; at T=280k the source-term extrapolation predicts 2.26 nats of damage against
0.40 measured, while the same joint form without the source term gives 0.27). The source term k*log(N_S/N_ref) is fitted
on two students spanning log ratio +-0.55 and evaluated at +1.9; it should be reported as not transferable in size.
Addendum (Table 2 strongest-baseline column, generated by v52): the frozen per-capability constant scores 0.060/0.047/0.862 (270M), 0.057/0.069/1.004 (1B), 0.020/0.038/1.024 (4B) for math/code/QA. The dev-selected form beats it on every cell of the two development students and loses to it on every cell of the held-out 4B student, so at the new size only the average development response transfers; the earlier reading 'transfers for code and math on 4B' held against zero change only and is superseded.
Origin code for the paper: P/F/F/A for the frozen forms, R for the "best form" column and the selection rule.
Pending (not in this entry): 4B at the dev pools U75/U450 (extras stage B), two training-seed repeats (stage A),
retrospective F1/F2 (v56) on the test trajectories.

### C39. P3 measurement check (v48): primary vs secondary benchmark, Gemma-3-1B, six pre-registered states (58 GPU-min)

States fixed before any P3 result: magnitude pruning d=0.7 (v6 protocol), RTN int4 (v10 protocol), and the P2-v2 data-seed-11 adapters at U75/U450 at the two milestone snapshots. Primary = main-protocol probes (measurement half); secondary = one independent benchmark per capability with 128 samples and a fixed probe seed: SVAMP (math), HumanEval (code), TriviaQA (QA; the plan named HotpotQA, the run recorded TriviaQA). Values are loss changes from dense in nats per native token, primary / secondary:

| state | math P/S | code P/S | QA P/S |
|---|---|---|---|
| prune_d0.7 | +1.31 / +0.51 | +1.23 / +1.41 | +0.83 / +3.30 |
| rtn_int4 | +0.95 / +2.51 | +0.60 / +0.75 | +0.88 / +1.99 |
| kd_U75_s11_update-00000025 | +0.07 / -1.09 | +0.06 / +0.06 | -1.33 / +0.11 |
| kd_U75_s11_update-00000049 | +0.08 / +0.31 | +0.12 / +0.07 | -0.69 / +0.31 |
| kd_U450_s11_update-00000026 | +0.10 / -1.36 | +0.06 / +0.06 | -1.46 / +0.21 |
| kd_U450_s11_update-00000050 | +0.09 / -0.30 | +0.11 / +0.07 | -1.40 / +0.37 |

dense primary: {'math': 1.22, 'code': 1.06, 'qa': 5.57} secondary: {'math': 6.42, 'code': 0.63, 'qa': 2.89}
tokens primary: {'math': 13198, 'code': 4516, 'qa': 251} secondary: {'math': 260, 'code': 8393, 'qa': 328} benchmarks: {'math': 'svamp', 'code': 'humaneval', 'qa': 'triviaqa'}

Reading: code agrees in sign and magnitude on all six states. Pruning and int4 damage agrees in sign on all three capabilities, with magnitudes that differ (SVAMP math has 260 scored tokens, TriviaQA 328, so their deltas are noisy). The distillation QA improvement on the primary probe (-0.7 to -1.5 nats on 2Wiki-style probes) does NOT appear on TriviaQA (+0.1 to +0.4): the QA gain under distillation is specific to the probe distribution, and the paper must say so (Appendix E and the distillation limitation in the discussion). Distillation math on SVAMP is within its noise (-1.4 to +0.3 over 260 tokens).

### C40. P2-v3 extras: 4B at development pools, training-seed repeats, retrospective F1/F2 (cut-off 2026-09-10 19:47 EDT, 3b5feae)

4B at U75/U450 x seeds 11/12 (16 points per capability, role dev_pool_heldout_student): dev-selected joint+src 0.106/0.048/0.958 vs frozen constant 0.156/0.098/1.790 vs zero 0.294/0.230/1.862 (math/code/QA). So the structured form beats the constant on the held-out student at the development pools and loses to it at the unseen pool (C38): the size direction is not covered reliably.
Seed repeats (train-seed 1 vs 0, same pool sample): 270M@U75 |diff| math <=0.017, code <=0.030, QA 0.016/0.010/0.093/0.186 by checkpoint; 1B@U450 math <=0.004, code <=0.008, QA 0.004/0.014/0.050/0.089.
Retrospective v56 `all` (fit on 48 dev points, scored on 52 test points; R): F2 best on 270M U375 (0.022/0.020/0.316); 1B U375 E-only best QA 0.342, F2 0.478; F1 fails QA everywhere (1.15-2.26); 4B U375 constant best math/code (0.020/0.038), F2:L0 best QA 0.360 (E-only 0.367); 4B dev pools F2:L0 math 0.404 vs F2:logN 0.122 (descriptor choice unsettled). Paper: App. C paragraph, App. D seed-repeat text, §3.4/§4.1/§4.2/§7 sentences, App. I data cut-off.

### C41. V63 quantization identifiability audit (CPU, retrospective R; delivered v55 predictions unchanged)

Development bit-widths {3,5} give centred u = +-1.16 and u^2 = 1.35 on every development cell, so the u^2 column is a multiple of the intercept column: the delivered phi (x) [1,u,v,uv,u^2] design has rank 16 of 20 (nullity 4), ridge lambda 1e-3, effective df 15.9 per capability. Same-input controls fit on the same 24 cells and scored on the frozen v54/v55 test cells: without u^2 (16 coefficients, same dev column space) bit test 1.300/1.386/1.944 (math/code/QA), granularity 0.609/0.718/1.476, joint 0.527/0.222/0.320; bit-only (8) and bit+group additive (12) coincide with it (group terms vanish at g=128; on the bit test their symmetric +-v contributions cancel inside the absolute errors); delivered form 0.193/0.214/0.436, 0.219/0.319/0.986, 0.345/0.097/0.269. Reading: the unseen-bit-width success is a property of how the ridge splits a constant between the intercept and u^2 columns; no curvature in bit-width is identified by the development data. Paper: §3.3 caveat; App. C paragraph + Table tab:quant_ident; separable claim narrowed to the tested power candidate.

### C42. V64 selection with explicit feasibility (CPU, retrospective; supersedes the fallback rule of v60 in the paper)

Change vs v60: a policy with no configuration within r_max returns INFEASIBLE (v60 fell back to the r=1 source, violating the budget). Coverage: MAP/quant-only/CHEAPEST 289/289; prune-only 161/289 (55.7%); distill-only 96/289 (33.2%). Own-feasible regret (math/code/QA/multi): MAP 0.194/0.174/0.418/0.029; quant-only 0.209/0.194/0.490/0.043; prune-only 0.588/0.622/0.725/0.673; distill-only 0.297/0.337/0.056/0.333; CHEAPEST 4.58/5.17/4.33/5.18. Oracle-method agreement MAP 95.9/94.8/73.4/93.8%. Common-feasible subset 55 cells: MAP 0.202/0.111/0.158/0.017. No-clear-winner heuristic: 134/164/142/158 flagged cells; oracle method in candidate set 99.7/100/91.0/99.7% of evaluable cells, 100% within flagged cells. Endpoint documented: absolute loss of the deployed model (student dense + student change for KD). Candidate coverage per state in tables/candidate_coverage.tex. Paper: §5 rewritten (quant-only close behind MAP; contribution = laws connected to an explicit decision, no general advantage over a strong single method), App. H.2 tables/figure replaced.

### C43. V65 distillation paired intervals (CPU; candidate joint+src vs frozen constant and zero; 5000 trajectory-cluster bootstrap resamples)

Unseen pool U375 (12 points / 3 trajectories per cell), paired MAE difference constant - candidate [95% CI]: 270M math 0.015 [0.002,0.027], code 0.011 [0.006,0.017], QA 0.257 [0.173,0.376]; 1B math 0.007 [-0.001,0.015], code 0.040 [0.032,0.044], QA 0.560 [0.555,0.566]; 4B math -0.068 [-0.070,-0.066], code -0.012 [-0.029,-0.003], QA -0.117 [-0.119,-0.115]. Vs zero: 42-79% relative on every 270M/1B cell (CIs above zero); 4B 42/72/2%. Development pools, held-out 4B (16 points / 4 trajectories): vs constant math 0.049 [-0.068,0.166], code 0.050 [-0.019,0.118], QA 0.832 [0.125,1.540]. Wording rule (user, 2026-09-10): CI above zero = gain detected; crossing = point improvement not resolved; below = constant better. Provenance unchanged: candidate predictions P (frozen 33c706c), headline rule R. Paper: §3.4 sentence, §4.1 distillation paragraph, §4.2 new-students sentence, App. D table tab from tables/distill_paired.tex.

### C44. V69 quantization confirmation, development and freeze (CPU; frozen 2026-09-10 23:20 EDT, commit 6199353, freeze sha256 6ade0a3ff576781c)

New development set = the 54 unblinded grouped-RTN cells (160M/410M/1.4B x steps 16k/143k x b{3,4,5} x g{64,128,256}). With three bit levels the 2-D surface is full rank (20/20; condition 52 math, 51 code, 142 QA). Six-fold leave-one-state-out macro MAE (math/code/QA): 2-D surface 0.61/1.48/3.23; bilinear (no u^2) 0.84/1.56/3.60; piecewise interpolation with pre-specified boundary rule 0.89/1.44/3.20; bit-only 0.87/1.56/3.60; per-config median 1.22/1.30/1.32; zero 1.32/1.42/1.32. Pre-committed rule (lowest LOSO macro MAE, ties within 0.02 toward fewer coefficients) selects: math = 2-D surface, code = per-config median, QA = zero. Frozen for the 21 confirmation cells (410M@143k and 1.4B@16k development states, 1.4B@112k never in a quantization fit; b{3,4,5} x g{32,512}; new state also g=128 x 3 bits): every candidate's predictions. Test names: unseen group-size extrapolation; joint new-state x new-configuration. Quantizer tail handling checked: all confirmation widths divide by 32 and 512. Measurement pending on the Ada (scripts/run_v69_grid.sh).

### C46. V69 quantization confirmation, measured (Ada, 21 forward cells, 23:24-23:29 EDT; compare.json)

Frozen predictions (6199353, before measurement) scored on 21 cells. Overall MAE (math/code/QA): same-input piecewise interpolation 0.098/0.154/0.532; 2-D surface (dev-selected for math) 0.253/0.317/0.955; bilinear 0.340/0.395/1.143; bit-only 0.422/0.573/1.175; per-config median 0.323/0.384/0.322; zero 0.534/0.629/0.356. By test set: development states at g{32,512} (12 cells/cap): interpolation 0.065/0.116/0.477, surface 0.215/0.309/1.228, median 0.500/0.557/0.458, zero 0.609/0.678/0.457; new state 1.4B@112k at g{32,512} (6): median 0.129/0.216/0.133, interpolation 0.169/0.166/0.577, surface 0.418/0.327/0.523; new state at g=128 (3): median 0.006/0.027/0.158, surface 0.075/0.332/0.723, interpolation 0.085/0.277/0.662, zero QA 0.045. Regimes: near-zero cells favor median/zero/interpolation; clear-damage cells favor interpolation for math/code (0.19/0.25 vs surface 0.41/0.46) and median for QA. Reading: the development-selected surface loses on every test (prospective loss, origin P); delivered quantization predictor = same-input piecewise interpolation with the pre-specified boundary rule for math and code on seen states, source-free median for QA and for new states. Paper: §3.3, §4.1, abstract, intro, §7, App. D table tab:quant_confirm.

### C45. V67 MuSiQue scope check (Ada, 34 evaluations, 2026-09-10 23:03-23:22 EDT; results/v67-musique-qa)

128 MuSiQue answerable-dev questions with supporting paragraphs, V6 QA template, fixed probe seed; delta from each student's dense loss, paired with the 2Wiki primary QA of the same checkpoint. Final checkpoints: 270M U375 2Wiki -0.67..-0.93 vs MuSiQue +0.08..+0.16; 270M U450 -0.65..-0.68 vs +0.09..+0.19; 1B U375 -0.51..-0.60 vs +0.52..+0.59; 1B U450 -0.58..-0.72 vs +0.22..+0.45; 4B U375 -0.18..-0.60 vs +0.65..+0.95; 4B U450 -0.29..-0.82 vs +0.40..+0.71; U75 (E~14) both rise (2Wiki +2.0..+6.7, MuSiQue +1.3..+3.7); 1B early checkpoints 2Wiki -1.33/-0.69/-1.46/-1.40 vs MuSiQue +0.17/+0.45/+0.26/+0.40. Reading: the distillation QA response reverses sign on an independent multi-hop distribution under the same context condition; the laws predict conditional loss on the teacher-matched QA distribution only. Paper: App. E paragraph + tables/musique_scope.tex (generator v73); §3.4; §7. Precursor of V5-M (256 samples, three sets, eight states).

### C47. V70 distillation confirmation, register/develop/freeze (CPU; frozen 2026-09-10 23:37 EDT, commit 118d1f8, FREEZE_V70 sha256 d7b28c72...)

Register: U=200 pools with data seeds 31-36 sampled by the V12 rule (assertion: U=200 absent from every earlier registered pool), D_U from actual completion tokens, triggers for supervised T {50k,100k,200k}. Development: 25 trajectories (12 dev + 9 test + 4 held-out 4B), 100 points; LOCO by trajectory, rule = lowest LOCO MAE with ties within 0.02 nats toward fewer parameters: math E-only (0.076; eight forms tie), code E-only (0.048), QA joint (0.598). Frozen: 108 predictions (2 students x 6 pools x 3 budgets x 3 caps) for the selected form and for constant, T-only, E-only, and same-input surfaces; strongest frozen baseline per student x capability fixed from development LOCO (270M: T-only/E-only/E-only; 1B: surface:L0/E-only/surface:logN). Runs pending: throughput pilot on the Ada, then 12 trajectories (suffix p2v3conf).

### C48. V70 distillation confirmation, measured (Ada, 12 trajectories 00:15-04:31 EDT 09-11; compare via v70b wrapper)

Frozen (118d1f8) selected forms vs strongest development-fixed baseline, 18 points per student x capability (6 pools x 3 budgets), paired difference [95% pool-cluster CI]: 270M math E 0.074 vs T-only 0.065, -0.009 [-0.010,-0.008]; 270M code E 0.019 vs E-only 0.023, +0.004 [+0.003,+0.005]; 270M QA joint 0.515 vs E-only 0.610, +0.095 [+0.077,+0.109]; 1B math E 0.057 vs surface:L0 0.033, -0.023 [-0.024,-0.023]; 1B code E 0.049 vs E-only 0.053, +0.005 [+0.004,+0.005]; 1B QA joint 0.463 vs surface:logN 0.450, -0.013 [-0.020,-0.010]. Zero change 0.09-0.15 (math/code) and 0.90-1.00 (QA); constant 0.069/0.046/0.478 (270M) and 0.053/0.057/0.854 (1B). Selected-form error grows with budget (270M QA 0.14/0.43/0.98 at 50k/100k/200k). Reading: configuration prediction for a new pool size holds for code (gain detected on both students) and QA on 270M; for math the pre-selected single-parameter reuse form is worse than budget-only/surface baselines and on par with a constant. Technical note: dense losses re-measured on the Ada drift <=0.0009 nats from the frozen descriptor; the frozen script's exact-equality check was bypassed by analysis/v70b_compare_tolerant.py (tolerance 0.01, drift recorded in dense_drift.json; predictions and responses untouched). Paper: §3.4, §4.1, App. D table.

### C49. V71 QA scope check, measured (Ada, 24 cells, 04:31-04:52 EDT 09-11; results/v71-qa-scope)

Gemma-3-1B, eight pre-specified states x {new 2Wiki sample (256, overlap 0 with both probe halves), MuSiQue answerable-dev with supporting paragraphs (256, overlap 0 with teacher traces), TriviaQA rc.nocontext (256)}. KD U450 s11 at T 35k/140k/280k: 2Wiki_new -1.548/-1.258/-0.878; MuSiQue +0.076/+0.195/+0.204; TriviaQA +0.246/+0.344/+0.253 -> pre-stated reading at all three budgets: only the new primary sample reproduces. Prune d=0.85: +0.41/+0.22/+0.16; d=0.7: +0.71/+1.27/+2.91; int4 channel: +0.80/+1.41/+1.65; b4 g128: -0.04/+0.22/+0.37. Paper: App. E paragraph + tables/qa_scope.tex; §7.

### C50. V72 pruning repeatability on Pythia-2.8B (Ada; dense 04:52-04:53, freeze 04:53:57 EDT 09-11 committed 6089efd, pruned cells 04:57-04:59; compare.json)

States pythia-2.8b@step16000 and @step143000 (never in a pruning fit; the cached step64000 snapshot shares the step143000 weight blob and was excluded), densities 0.85/0.75/0.65, six cells per capability. MAE (math/code/QA): delivered power form 0.292/0.393/0.679; A2 0.425/0.458/0.437; median development curve 0.080/0.054/0.065; zero 0.218/0.145/0.326. Reading: the source-free median curve is 3-7x better than both source-conditioned forms on an in-range new size; the compact continuous form is not repeatable here (prospective failure, origin P). Paper: §4.2 sentence, §7, App. D table tab:prune_repeat (caption corrected to steps 16k/143k).

### C51. Formulation and scope corrections after the round-5 review (2026-09-11 10:00-10:30 EDT; paper 9351d00)

Facts checked in code before writing: (a) v69's per-configuration median predicts never-measured g=32/512 by the same piecewise/boundary interpolation rule applied to the nine per-configuration median anchors (v69_quant_confirm.py line 254); (b) v70's selected 'E' form is a*log(1+E) with no intercept (1 parameter, zero at zero budget) and the 'E-only' baseline is c + b*log(1+E) (2 parameters), both fit on the 25 development trajectories with equal weights; (c) the teacher pool's QA traces come from HotpotQA and the primary probe from 2WikiMultihopQA, so "teacher-matched distribution" was replaced by "the 2Wiki distribution"; context conditions differ (2Wiki all paragraphs, MuSiQue supporting paragraphs, TriviaQA none) and are now part of the scope statement; (d) heterogeneous-panel models Qwen3, Gemma-3, Gemma-4-31B, OLMo-3 are dense per cached configs; Gemma-4-26B-A4B (cached, 128 experts) is MoE; Muse-Glimmer-30B's architecture is being read by V77; trajectory eval.json files store only sample counts (64 per capability), not per-example losses, so an item-level bootstrap is impossible from stored data. Paper changes: reference model M0 defined (accommodates dense and sparsely activated architectures; controlled experiments focus on dense), L_{c,j} with the evaluation distribution as a fixed index, V72 written as positive evidence for L0 + f~_c(d), the final five-question table per arm (App. D, tab:final), shared-structure details moved to App. C.

### C52. V74-V77 closure analyses (CPU, 2026-09-11 10:13-11:08 EDT) and CORRECTION of C50

V74: three-way quantization table (frozen selector / frozen candidates / post-test rule) generated from v69; median algorithm documented (nine per-configuration median anchors, same piecewise/boundary rule as the interpolation candidate). V75: form audit (selected E = a*log(1+E), 1 parameter, no intercept; baseline E-only = c + b*log(1+E), 2 parameters; both OLS on 100 rows); dense-drift sensitivity: 0/12 and 0/36 interval sign changes; per-example losses not stored (64 items per capability), item interval impossible. V76: per-capability vs shared curve + per-capability scale on the frozen confirmation cells: pruning +0.073 macro [0.030,0.115] driven by QA (+0.19 [0.13,0.25]; math/code within noise; shared curve alone better for math/code), quantization +0.003 (indistinguishable), distillation identical (single-coefficient forms); shared + offset best for distillation QA in this ablation (0.63 vs 1.24 for the reuse form). V77: all 12 heterogeneous-panel models and all Pythia states are dense (Muse-Glimmer-30B dense; Gemma-4-26B-A4B cached but never in the panel); v50's frozen 4B student count includes 0.42B non-text parameters (recorded, not recomputed). CORRECTION of C50: the cached Pythia-2.8B revisions step16000/step64000/step143000 carry identical learned parameters (387/388 tensors byte-identical, one differs in the sign of zero; dense and pruned losses identical to all digits), so V72 is ONE state measured twice, not two stages; the per-capability MAEs are unchanged. Paper: App. A architecture table + weight-identity paragraph; App. D V72 wording and caption; §4.2 wording.

### C53. Round v7 confirmation panel downloaded and identity-checked (2026-09-11 15:25 EDT; logs/v78_downloads.log)

Fresh snapshots with blob SHA-256 prefixes, none coinciding with any hash in results/v77-model-arch/weight_identity.json: pythia-160m@step32000 837a7b56674b9563 (649,308,728 B); pythia-410m@step32000 c30a2dd44723938b (1,621,370,224 B); pythia-1.4b@step32000 e2cc40f3d4e82130 + a485538a70ea4f15 (two shards); pythia-1b@step64000 2d4b40480e0989af (4,047,149,576 B). No measurement of these states exists in the repository. Locked rule and panel registered in A100_EXPLORATION_PLAN.md 'Round v7'; V78 authoring in progress.

### C54. V78 locked selection rule, per-capability maps, independent confirmation (Ada 15:26-15:55 EDT 09-11; freeze 869ec2e before configuration measurement; compare.json)

Panel: pythia-160m/410m/1.4b@step32000, pythia-1b@step64000 (fresh blobs, C53; verification.json), dense + 18 configurations each (72 evaluations, 27 min). Mean regret over 68 state x budget cells per capability (nats) / method agreement: math locked 0.000006 (0.985), v64-law 0.0149 (0.985), quant-only 0.0068 (0.941), cheapest 1.77; code locked 0.0043 (0.956), v64-law 0.112 (1.000), quant-only 0.0102 (0.971), cheapest 2.43; QA locked 0.141 (0.765), v64-law 0.214 (0.618), quant-only 0.253 (0.485), distill-only on its 17 feasible cells 0.052; multi (max over capabilities) locked 0.0020 (1.000), v64-law 0.00185 (0.971), quant-only 0.0081. Pre-stated verdicts: math, code, QA CONFIRMED (locked <= v64 and < quant-only); multi (largest loss increase max_c[L_c-L_0c]) PROSPECTIVE, prespecified criterion NOT MET (trails v64 by 0.0002; beats quant-only). Distillation candidates are the historical v39 students 160m@64k/410m@64k, whose post-training outcomes were PURGED from the KD fits that predict them (v78_rule_confirm.py:366-367); pruning/quantization candidates are new measurements. Candidate-set method coverage 1.000 for the locked rule on every objective. Maps: QA map chooses pruning at r >= 0.6 on the 32k states and distillation on 1b@64k at every budget, oracle agrees; math/code maps quantization almost everywhere. Paper: §5 paragraph, App. H rule statement + tab:rule-confirm + fig:rule_maps.

### C55. V79 audit of the capability-conditioning comparison (CPU, 2026-09-11 21:15 EDT; results/v79-cond-audit)

Variant B scales were unrestricted signed OLS and none came out negative (pruning 1.29/1.27/0.33; quantization 1.07/1.43/0.56; distillation 0.72/0.73/1.55). Attribution: the QA gain comes from a non-proportional aggregate curve that reverses sign across density (median +0.87 at d=0.55, negative from 0.60 to 0.90; R^2 of the best signed multiple 0.49 vs 0.94 math, 0.97 code); QA direction also varies across development sources (majority opposite to the arithmetic mean at 0.60-0.90 except 0.75). Four-cluster recomputation with the 2.8B duplicate merged: A-B gain macro 0.059 [0.021,0.103], QA 0.168 [0.105,0.243], math 0.005 [-0.049,0.059], code 0.004 [-0.018,0.023]; reading unchanged. Only V72 and V76 used both 2.8B labels; V78 unaffected. Distillation A==B is algebraic (proof in summary.md; max prediction difference 1.4e-16). Paper: §4.3 rewritten ("shape", not "sign"); App. C paragraph + tab from cond_audit.tex.

### C56. V80 addenda to the selection confirmation (CPU, 2026-09-11 21:26 EDT; results/v80-rule-addenda)

Verified from code: multi objective = max_c[L_c(M) - L_c(M0)] (final_rule.py:18-19; v64 score:407-410); quant-only pools channel and grouped RTN with the locked predictor and the same feasibility rule, no fallback (v78:74-75, 127-143); KD candidates are historical v39 outcomes purged from the fits (v78:366-367). Per-state locked-rule regret (math/code/QA/multi): 160M@32k 0.000/0.000/0.220/0.001 (QA quant-only 0.126, v64 0.142); 410M@32k 0.000/0.001/0.047/0.002; 1.4B@32k 0.000/0.012/0.244/0.005 (QA quant-only 0.387); 1B@64k 0.000/0.004/0.052/0.000 (QA v64 0.005, quant-only 0.280). Candidate-set sizes (locked rule, 68 cells): one method 30/30/24/30, two 28/28/32/28, three 10/10/12/10 for math/code/QA/multi; oracle method always in the set; exact-configuration coverage 100/54/27/79%. Figures: rule_maps_main (locked rows, disagreement marks), rule_regret, rule_maps_full. Paper: §5 restructured; App. H tables tab:rule-confirm-by-state and tab:rule-confirm-candidate-sizes.

### C57. V81 metric definitions and reader-facing labels (CPU, 2026-09-11 22:32 EDT; results/v81-rule-labels)

Coverage columns of tab:rule-confirm-candidate-sizes describe the no-clear-winner CANDIDATE SET over all 68 cells (headers now 'Set contains oracle method' / 'Set contains oracle configuration'; v78:567-570, v64 ambiguity:450-474). Single chosen configuration (frozen rule): exact-configuration agreement 67/68 (98.5%) math, 36/68 (52.9%) code, 11/68 (16.2%) QA; method agreement 98.5/95.6/76.5%; exact <= method holds in every objective and policy. Figures relabeled: 'Frozen selection rule', 'Source-conditioned predictor', 'Quantization-only', 'Cheapest feasible' (internal names in appendix footnotes); regret chart carries numeric labels; main strip saved tight with QA titled 'QA (2Wiki)'. Paper wording: 'not proportional to the tested shared curve'; §7 limitation paragraph (advisor text, deduplicated); caption states 4 states x 17 budgets sharing model and candidates.

### C58. Round v9 restructure (2026-09-12 02:30 EDT onward; no new measurements)

Advisor full-PDF read: final predictors moved to the centre of §3 (per arm: relation -> inputs and range -> selection -> tests and boundary); the quantization surface (former Eq. 4) and the F1/F2 candidates (former Eq. 5) demoted to tested candidates; the anchor L0 distinguished from the response relation. Contradictions fixed: pruning "lowest math error" -> "A2's math error with a quarter of the parameters" (0.243 vs 0.230; median better on code 0.214 vs 0.236); new quantization state -> median best for math/QA at new group sizes, interpolation better for code (0.166 vs 0.216), zero better for QA at g=128 (0.045 vs 0.158); §4.3 "carried on every axis" -> per-method, per-range statement; v60 caption leftover ("regret can be negative") removed; d=0.55 extrapolation attributed to the nine-state freeze only. §5 now cites the exact locked rule (App. H table) instead of Table final, and states that the math/code margin over quantization-only comes almost entirely from 1B@64k (three 32k states coincide within 1e-5). Title -> "Developing Capability-Conditioned Scaling-Down Laws for LLM Compression". App. I registration list synchronized (v36b..v78 with commits; retrospective rows acknowledged). Codex V82-V85 delivered: Fig. 1 final_relations (min glyph 9.8 pt at text width), Fig. 2 final_confirmations (min 7.2 pt), tables/main_final.tex (10 rows, every number with a JSON-pointer source), tables/rule_decomp.tex (three 32k states: frozen minus quant-only math -0.000007, code +0.000001, QA -0.073; 1B@64k: -0.027/-0.023/-0.228; all four: -0.0067/-0.0059/-0.112) and tables/locked_rule.tex (24 rows, differences from Table final noted). Old Fig. 1/2 (glyphs 1.7-3.3 pt) retired to the appendix. Paper 6bb7701, nine pages.

### C59. V86 frozen-candidate table and hindsight ranking (CPU, 2026-09-12 11:31 EDT; results/v86-main-table)

Round v10 (advisor): the main table and Fig. 2 now separate (i) the frozen candidate, (ii) the strongest alternative = lowest pooled test MAE among ALL predictors frozen in that round excluding the candidate (a hindsight ranking, stated as such), and (iii) the delivered rule with when it was fixed. Quantization candidates are the frozen development selection D (surface/median/zero), not the retrospective interpolation/median; delivered interpolation (math, code, seen states) and median (QA; new states) are labelled 'fixed after test'. Pruning: three-checkpoint panel candidate power 0.243/0.236/0.678 vs A2 0.230 / median 0.214 / zero 0.185; delivered median for new states (0.277/0.214/0.221), fixed after test and reused frozen in v78. New-state quantization QA: zero 0.2217 vs median 0.1412 over nine cells (gain 0.081, not 0.464 vs interpolation). U375 rows moved to App. D (tab:main_context). NEW FINDING: for the U200 distillation confirmation the development-designated baselines (T-only, E-only, surface) are not the full-set minima; ranked over all sixteen frozen forms the selected E / E / joint is the hindsight best for no student x capability (270M: T+src 0.016, F1:L0 0.010, joint+src 0.333 vs 0.074/0.019/0.515; 1B: F2:L0 0.031, F1:L0 0.046, T-only 0.386 vs 0.057/0.049/0.463). The pre-designated comparison (code gain 0.004/0.005; QA 0.09 on 270M) stands as registered; abstract, §1, §3.4, §4.1 and App. D now say both. Locked-rule table branch labels: 'seen source state; new d' (new stage takes precedence over a seen size, final_rule.py unchanged). Registered selection criterion written out in §5/App. H: mean regret over 68 cells within the source-conditioned map's and below quantization-only's.

### C60. V88 logit-displacement diagnostic [READ WITH C63, which corrects two statements here] (GPU, 2026-09-12 14:30-14:45 EDT, 10 GPU-min; results/v88-displacement)

Registered before measurement (docs/prereg/v88_displacement_prereg.md), with one amendment written before any cell was measured: the registered shrinkage predictor omitted the shrinkage's own quadratic term, so the complete decomposition dL2 = eps*B + (E_p[s]-s_y) + 0.5*eps^2*W - eps*Cov_p(z,s) + 0.5*Var_p(s) replaced it (W = Var_p(z); verified to 1.5e-14 on fixtures and to 8.0e-5 across all measured cells). 5 weight-identity-distinct Pythia states (160M@16k/143k, 410M@143k, 1.4B@16k/143k) x 7 configurations (pruning d 0.9/0.8/0.7/0.6; grouped RTN b5/b4/b3 at g=128) x 3 capabilities = 105 non-zero cells; bf16 weights, float32 vocabulary reductions, dense and compressed models scored in the same run, probe_sha256 33d118c7 matching the frozen protocol. NOTHING FITTED.
T1 PASSES: the coefficient-free second-order account (E_p[r]-r_y) + 0.5*Var_p(r) has median relative error 2.3% and 97.6% sign agreement in the mild regime (|dL|<0.1, n=41), against a registered threshold of 20% and 90%; 15.3% moderate (n=31), 38.8% severe (n=33, sign 100%). Accuracy degrades monotonically with |r|/|z|: 0.9% at 0.07, 4.5% at 0.21, 23.7% at 0.46. By configuration: pruning d=0.9 0.9%, d=0.8 6.9%, d=0.7 18.4%, d=0.6 45.1%; RTN 5-bit 1.5%, 4-bit 8.3%, 3-bit 32.4%. This is a mechanism-level account of why the paper's compact forms hold only in limited ranges.
T2 FAILS: summarising the displacement by one scalar (eps*B + 0.5*eps^2*W) gives 98.0/97.5/99.6% median relative error (42x/6.4x/2.6x the T1 error against a 1.5x threshold) and only ~66% sign agreement; median eps is 0.030. The hypothesis that pruning and quantization act as logit shrinkage plus isotropic noise is therefore FALSE on these panels; B and V survive only as sensitivity summaries, as the registration specified.
Mechanism: median first-order term -0.042 against median second-order term +0.258, so damage is carried by the p-weighted variance of the displacement, not by a systematic shift; the displacement spreads across the vocabulary (top-16 share 2.4% mild, 20% severe).
Reproduction check against frozen measurements: math and code within 0.004-0.019 nats; QA up to 0.065 nats apart (263 scored tokens, different GPU), so QA readings here are weaker and no frozen value was changed.
Next question, needing its own registration and frozen predictions: whether 0.5*Var_p(r) is predictable from configuration and pre-compression information on states that entered no fit.

### C61. Test-suite state after the 2026-09-12 changes (CPU)

Local offline suite: 702 passed, 6 failed, 5 skipped. Fixed today: three CLI tests failed only inside the full suite because torch sets MKL_THREADING_LAYER=INTEL in the parent and the spawned child then fails against libgomp; the child now gets a compatible layer. One duplicate-coordinate test compared an error message case-sensitively.
Remaining six, none of them silent: test_v17_unification, test_v32 and test_v34 are the pre-existing failures recorded on 2026-09-08; test_v22_experiment_manifest wants results/v6-capability-geometry/Qwen--Qwen3-14B/fisher_meta.json, which was never measured; and the two test_v27c provenance tests correctly report that analysis/v12_distill.py has changed since that experiment recorded its digest, because V87 added per-sample evaluation records to the shared eval path (additive, backward compatible, covered by V87's tests). Refreshing v27c would rewrite docs/MEASUREMENT_FOLLOWUPS.md and its recorded digests; that is a deliberate re-run, not a silent fix, and has not been done.

### C62. V89 intervention gate on existing distillation trajectories [READ WITH C63, which corrects the code verdict] (CPU, 2026-09-12; results/v89-intervention-retro)

Retrospective development-set analysis on already unblinded data; not an independent test. Purpose: decide, before funding any new distillation training, whether the effect of a training INTERVENTION is predictable from what already exists. 89 run directories inspected, 41 usable, 48 dropped with reasons listed. Two targets: I1, the change in delta when the budget roughly doubles inside a trajectory (134 pairs); I2, the change in delta when the pool changes at a matched budget (1163 pairs). Seven candidates fixed before fitting (zero, per-capability constant, T-only, reuse-only, low-order two-dimensional, saturation-plus-reuse at p=1 and p=2), each fitted to delta and then differenced; the intervention effect is never fitted directly. Splits: leave-one-student-size-out and leave-one-pool-seed-out. Intervals from a 5000-draw cluster bootstrap whose unit is the trajectory.
Registered reading rule, fixed before the numbers: signal counts as present only if the SAME candidate beats both zero and the per-capability constant on BOTH targets, by more than the interval width.
VERDICT: math ABSENT, code ABSENT, QA (2Wiki) PRESENT via reuse-only and the two-dimensional form. For code, reuse-only clears on I1 (gain 0.026 against width 0.019) and on I2 (gain 0.053), but no single candidate clears both targets against both baselines; for math no candidate clears at all (T-only actually loses on I1, gain -0.008).
Identifiability: tau is NOT identified in any fold or capability; every profile interval touches a boundary of the observed positive-token range. No asymptotic floor and no optimal-budget claim is available from these trajectories.
CONSEQUENCE: the 18-trajectory crossed matrix proposed for the next round is not justified for math or code on this evidence. QA is the only arm with a predictable intervention effect, and its scope is already known to be narrow (2Wiki only; the gain reverses on MuSiQue and TriviaQA, C45/C49). Gate 0 of the next-round plan therefore does not open for math and code.

### C63. Corrections to C60 and C62 after review (CPU, 2026-09-12)

Four statements in C60 and C62 claimed more than the experiments support. The measurements stand; the readings are corrected here and in the day report and FINDINGS.

1. **C60 said the shrinkage-plus-noise hypothesis is false. It is not.** The registered T2, after its pre-measurement amendment, contains only the shrinkage term eps*B + 0.5*eps^2*W; the isotropic-noise reading was explicitly deferred by that same amendment. What V88 rules out is the PURE-SHRINKAGE account. The noise part is untested and stays open.
2. **C60's "damage is carried by the variance" was a median over cells and hides cancellation.** The two terms have opposite signs in 81 of 105 cells. Per cell, the variance term is larger for math in 35 of 35 and for code in 35 of 35, but for QA in only 16 of 35 (median |first order| 0.610 against |second order| 0.448). Seventeen cells have a negative measured change, which a non-negative variance term cannot produce; in all seventeen the first-order term is negative and outweighs it, and the second-order account still gets the sign right in sixteen.
3. **C60 understated two boundaries.** T1 uses the compressed model's logits, so it is a post-hoc decomposition, not a prediction from pre-compression inputs; a statistic that needs no labels but needs the compressed model is not a basic-input predictive law. And eps_hat is an uncentred projection: adding a constant to every logit changes it while leaving probabilities and loss unchanged, so the shrinkage share needs a shift-invariant definition before interpretation.
4. **C62 said code has no signal. The correct statement is that code did not clear the pre-registered investment gate.** The same reuse-only form improves on the no-intervention baseline for both targets: budget change MAE 0.0687 to 0.0423, improvement 0.0264 [0.0120, 0.0417]; pool change 0.1195 to 0.0667, improvement 0.0528 [0.0300, 0.0720]. Those are 38% and 44% error reductions with intervals above zero. The gate failed only because on the first target the improvement is smaller than the interval WIDTH (0.0297). The gate's verdict stands and the eighteen-trajectory matrix stays unfunded, but code is not shown to be unpredictable.
Also corrected: the two gate baselines are not independent. A per-capability constant fitted to delta and then differenced is identically zero, so "beats zero and the constant" was one comparison, as the V89 code and detailed report disclose. The seven candidates carry no student-state input, so what was tested is whether a relation in T, D_U and E alone transfers across sizes, not the student-conditioned relation the plan proposes. The historical pool-change target mixes schedules, run generations and budgets matched only to about ten percent, so it cannot adjudicate a strictly controlled matrix. And tau is not identified under the size holdout, but some 270M code pool-seed folds do give interior intervals, so "not identified in any fold" was wrong; no floor or optimal-budget claim is made either way.

### C64. V88 follow-up: shift-invariant projection and the term shares (GPU, 2026-09-12, 10 GPU-min re-run)

Answers the two checks the review asked for; the first run is kept unchanged at results/v88-displacement-uncentred-run/.
(1) SHIFT INVARIANCE. The original shrinkage coefficient used an uncentred projection; adding 7 to every logit changed it by 85% while leaving probabilities and loss unchanged. Replacing it with the p-weighted covariance projection eps = -Cov_p(r,z)/Var_p(z) makes it invariant (1e-6 on the same fixture) and zeroes the cross term by construction; a unit test asserts both, and asserts that the old coefficient is not invariant. Re-running all 105 cells: the one-scalar summary improves to 83.8/82.0/85.9% median relative error (from 98.0/97.5/99.6) with sign agreement 85% (from 66%), still 36x/5.4x/2.2x the exact account's error against a 1.5x threshold. T2's verdict is unchanged and now rests on a well-defined quantity.
(2) THE NOISE TERM. Median share of the measured response: mild, shrinkage 2.7%, residual first order -9.4%, variance 88.6%, sum 99.1%; moderate 2.8/2.1/72.0, sum 94.5; severe 3.7/-2.6/59.2, sum 61.2. Per capability over mild and moderate cells: math 3.2/-3.9/84.5, code -0.3/-19.8/88.6, QA 91.0/61.4/-94.0. For math and code the predictable-regime response is 84 to 89 percent noise with about three percent shrinkage, which SUPPORTS the noise half of the shrinkage-plus-noise hypothesis; V88 falsifies the shrinkage half, and it does so because shrinkage is a small share rather than because it was mis-estimated. QA is structurally different and needs separate treatment. Worst centred-decomposition residual 1.87e-2, 0.92% of that cell's response, from float32 cancellation at large logits.

### C65. V92 input comparison: can pre-compression information predict response across sources? (CPU, 2026-09-13; results/v92-input-comparison)

Development study, leave-one-source-state-out, on the nine-state Pythia panel with the newly measured dense statistics from results/v91-dense-stats (B, V and W, one dense forward per state and capability, 27 of 27 extracted). Three nested input budgets kept strictly separate: K0 metadata; K0 plus the dense anchor L_0; and that plus B, V, W. Candidates fixed before fitting: zero, per-capability constant, the source-free median development curve, OLS, ridge with the penalty chosen inside the training fold, and the arm's delivered form. Reading rule fixed before the numbers: the statistics count only if they beat the dense-anchor budget UNDER THE SAME FORM by more than the full interval width, clustered on source states.
RESULT, two parts, and the second decides it. (1) Under the registered rule the statistics help in 3 of 9 arm-by-capability pairs, all through ridge: pruning math 0.084 [0.049, 0.122], pruning QA 0.109 [0.062, 0.157], grouped quantization math 0.352 [0.257, 0.437]. OLS gains are within their intervals everywhere. (2) In 9 of 9 pairs BOTH statistics-augmented candidates still have higher MAE than the K0 source-free median curve. Clearing the incremental input rule is therefore not a win over what the paper already delivers.
Most-fragile-capability accuracy gets WORSE when the statistics are added: pruning 58.3% to 41.7%, grouped quantization 38.9% to 18.5%, per-channel 36.1% to 41.7% (descriptive, nine and six clusters).
CONSEQUENCE: the bar the round set, that a little pre-compression information explains and predicts cross-model differences in capability response, is NOT met. The reserved confirmation states should not be spent on this branch as it stands. Dense inputs for those states are being extracted anyway (results/v93-confirm-inputs), which costs minutes and keeps the option open without measuring any response.

### C66. V91 learning-rate protocol pilot (GPU, 2026-09-12 evening, A100, ~80 GPU-min; results/v12-distill/*/gpt-5.6-luna_full_75_lrpilot_*)

Six short trajectories, Gemma-3-1B and 4B, one fixed development pool (U=75, data seed 11), identical LoRA settings, schedule and evaluation, learning rate the only variable. Change in loss from the student's own initial checkpoint, positive means worse:
1B: 5e-5 math +0.0025, code +0.0067, QA -0.791; 1e-4 +0.0508, +0.0520, -1.312; 2e-4 +0.0614, +0.1016, -1.136.
4B: 5e-5 math +0.0575, code +0.0221, QA -1.095; 1e-4 +0.1015, +0.0720, -1.340; 2e-4 +0.1424, +0.1541, -1.180.
TWO READINGS. (1) Protocol sensitivity is large and systematic: math and code degrade monotonically with learning rate, and at 5e-5 the 1B student is essentially unharmed (+0.003, +0.007) while still gaining 0.79 on QA. Every earlier distillation result used the LoRA default of 1e-4, so the size of the reported math and code cost is a property of that choice as much as of distillation. (2) The size effect SURVIVES protocol matching: at every learning rate the 4B student degrades more than the 1B on both math and code, by roughly a factor of two on math (0.058 against 0.003 at 5e-5; 0.102 against 0.051 at 1e-4; 0.142 against 0.061 at 2e-4). So the earlier size-transfer failure is not explained away by the learning rate, though its magnitude is. QA improves for both sizes at every rate and peaks at 1e-4 for both, so the QA gain is close to size-independent and rate-peaked.
CONSEQUENCE: a single learning rate can be fixed for the controlled matrix. Recommending 1e-4, because it keeps the 41 existing trajectories comparable and it is where the QA effect is largest; the pilot stands as the sensitivity record, and any claim about math or code damage must name the rate.

### C67. V94 controlled distillation matrix and V95 intervention analysis (GPU + CPU, 2026-09-13; results/v12-distill/*/...matrix2..., results/v95-matrix-intervention)

FIRST CONTROLLED MEASUREMENT of the two axes separately. 18 trajectories: students 270M, 1B, 4B; independent-data pools of 66, 198 and 594 traces per domain (a 1:3:9 ladder colliding with no earlier pool); two pool seeds; one protocol everywhere (LoRA, learning rate 1e-4 fixed from the C66 pilot, identical schedule and evaluation); every cell stopped at the same budget with checkpoints at 85k/170k/339k/678k processed tokens, about 25k/50k/100k/200k supervised. An earlier launch was discarded as a matrix: the trajectory flag counts processed tokens, not completion tokens, and without a stop point larger pools silently trained longer.
EFFECT, at the common budget of about 200k supervised tokens, mean over the two seeds, positive means worse: at the smallest pool (reuse about 12x) 270M math/code/QA +0.18/+0.30/+0.29, 1B +0.30/+0.38/+2.55, 4B +0.56/+0.40/+4.85; at the largest pool (reuse about 1.3x) 270M +0.08/+0.14/-0.97, 1B +0.09/+0.09/-0.91, 4B +0.12/+0.14/-1.17. So at a FIXED budget, more independent data monotonically reduces math and code damage (4B math falls by a factor of five) and turns QA from a large loss into a gain, and the reuse penalty GROWS WITH STUDENT SIZE. The earlier size-transfer failure is partly a reuse effect: those runs sat in the high-reuse regime.
PREDICTABILITY (V95, registered before the numbers: same candidate must beat the single no-intervention baseline on BOTH targets under leave-one-student-out by more than the full interval width). VERDICT: not predictable for math, code or QA. But the two targets differ sharply and the report must say so. Budget doubling, code: reuse-only cuts MAE 43% (0.0337 against 0.0590), gain 0.0253 [0.0123, 0.0388], width 0.0265, so it misses the strict rule by 0.0012 with an interval well above zero; QA: 10% cut, gain 0.0614 [0.0126, 0.1105]; math: 5%, interval spans zero. Independent-data change: NO candidate helps for any capability (code 13%, math and QA about zero), which is the real negative, because the effect of that intervention is the largest one in the data. Leave-one-pool-seed-out gives the same picture, so the failure is not about crossing students.
READING: the response to reuse is large, monotone and size-dependent, and a one-parameter reuse form tracks the budget axis for code; nothing registered predicts what changing the independent data does, even under control. The 12 dropped-cell figures and per-trajectory curves are in the summary.

### C68. Two wordings corrected after review (CPU, 2026-09-13)

(1) C67 and the round report said the intervention is "not predictable". The correct statement is that the PRE-REGISTERED THRESHOLD WAS NOT MET. For the budget-doubling target on code the one-parameter reuse form cuts MAE by 43% with an interval excluding zero and misses the bar by 0.0012 nats; that is development evidence worth an independent confirmation, not a null.
(2) C64 said the noise half of the shrinkage-plus-noise hypothesis is supported because the variance term carries 84 to 89 percent of the response for math and code. That over-reads it. The quantity measured is the variance of the displacement component orthogonal to the logits. A large orthogonal residual variance says the loss change lives there; it does not establish that the residual is random noise, and it is not a pre-compression predictor, since computing it needs the compressed model. What V88 settles is that the shrinkage share is small, so the pure-shrinkage account fails on magnitude. Whether the residual has usable structure is open.

### C69. V96 joint response audit (CPU, 2026-09-13; results/v96-joint-response)

Answers the three checks the review asked for, on the 18 controlled trajectories only; the five discarded cells are excluded by assertion in code.
(A) THE T-ONLY ANOMALY IS FULLY EXPLAINED. With student and budget fixed, a budget-only predictor must predict exactly zero for a pool change. 100% of the small non-zero gains reported in C67 reconstruct from residual budget mismatch under the registered tolerance of max/min T <= 1.10; at identical T both the prediction and its gain are exactly zero. A gap in the logs is recorded rather than worked around: per-domain SUPERVISED-TOKEN totals are not stored, only total completion tokens and per-domain example counts, so per-domain token shares cannot be reported for these runs.
(B) STRONGER BASELINES ADDED as asked: the non-zero mean intervention effect estimated on the training fold, and a same-input low-order response surface. A per-capability constant differenced is identically zero and is not counted as a second baseline.
(C) THE PROPOSED JOINT FORM (a + a'z)log(1+T/T_ref) + (b + b'z)log(1+E), four coefficients per capability, zero at zero budget, one fitted parameter set generating the response and both interventions, is NOT algebraically equivalent to any earlier candidate (14 comparisons, all with unequal witness rank). It DID NOT MEET the pre-registered threshold for either descriptor or any capability, and it also fails against both new baselines.
(D) ITS FALSIFIABLE IMPLICATION FAILS. The measured pool-change effect divided by [log(1+E2) - log(1+E1)] should be flat in budget; it drifts strongly and systematically, for example code on 1B small-to-middle 0.012, 0.025, 0.120, 0.234 across the four checkpoints. Residual-slope intervals exclude zero in one common direction for all three size contrasts and both descriptors, so the gate for the single extra interaction opened; refitting with k*log(1+T/T_ref)*log(1+E) leaves the drift detectable in almost every contrast and buys no held-out gain (gains from -0.004 to +0.003, most intervals spanning zero). So the reuse coefficient is not budget-independent, and one product-of-logs interaction does not repair it.
LIMIT, stated by the analysis: the matrix contains NO pair of pools whose checkpoints land at the same supervised budget, so the exact fixed-budget falsification statistic is not computable and Part D is an observed-endpoint diagnostic confounded with the budget mismatch. That is an experimental defect, not a data-analysis choice, and it is what the next runs fix.

### C70. Displacement diagnostic extended to three families and a refined boundary (GPU, 2026-09-13/14; results/v88-displacement and results/v88-displacement-cluster)

Ran in parallel on the two free workstation cards and one cluster node, forward passes only. The grid grew from 105 to 339 non-zero cells: the four development states the first sweep missed, a third compression family (per-channel round-to-nearest at 8, 6, 5, 4 and 3 bits), intermediate pruning densities 0.85/0.75/0.65, and grouped round-to-nearest at group sizes 32 and 512. Cluster-measured cells are kept in a SEPARATE directory, because the diagnostic compares dense against compressed within one run and is internally valid, but the two machines must not be silently pooled.
MEDIAN RELATIVE ERROR of the coefficient-free second-order account, by configuration:
pruning d=0.9 0.9%, 0.85 4.3%, 0.8 5.3%, 0.75 11.1%, 0.7 16.0%, 0.65 28.2%, 0.6 44.4%;
grouped RTN b5 1.6-2.7% at every group size, b4 3.3% (g=32), 7.3% (g=128), 9.2% (g=512), b3 35.9%;
per-channel RTN b8 2.5%, b6 3.0%, b5 4.2%, b4 23.8%, b3 53.9%.
READING: the boundary is the same in all three families and it tracks the DAMAGE MAGNITUDE rather than the family or the knob. Wherever the measured loss change is near 0.1 nats the account holds to about ten percent, and it degrades from there; grouped four-bit quantization with fine groups sits at 0.06 nats and 3.3% error while per-channel four-bit sits at 0.24 nats and 23.8%, which is the same curve seen through two different knobs. This is the first version of the claim that covers a third family.

### C71. V100 critical-region test: the drift is real, and the joint family is falsified (GPU + CPU, 2026-09-14; results/v12-distill/*_132_critical_*, results/v100-critical-region)

Four new development trajectories were trained for one purpose: the matrix had no pair of pools whose checkpoints land at the same supervised budget, so the fixed-budget falsification test could only be run approximately. The new runs are gemma3-1b and gemma3-4b at U=132 per domain, seeds 51 and 52, same protocol, with fourteen checkpoints each instead of four. Combined with the matrix that gives 22 trajectories, 150 checkpoints and 450 capability responses; the five discarded launches are excluded by assertion.
RESULT. Matched-budget pairs now exist at a 1% tolerance and, for some contrasts, at 0.3%, with residual differences of 22 to 64 supervised tokens. THE DRIFT SURVIVES at both tolerances for all three capabilities: pooled ratio slopes at 1% are math 0.261 [0.105, 0.426], code 0.168 [0.083, 0.283], QA 3.99 [1.80, 6.19], each over 19 trajectory clusters; at 0.3% math 0.348 [0.096, 0.612], code 0.168 [0.072, 0.229], QA 2.34 [0.39, 4.04]. So the budget dependence of the reuse coefficient is NOT an artifact of the earlier broad tolerance.
STRUCTURE WORTH KEEPING. The drift is concentrated in the high-reuse contrasts and nearly vanishes between the two low-reuse pools: for the 66-to-198 contrast the slopes are math 0.277, code 0.202, QA 4.01, while for 198-to-594 they are math 0.038 [-0.002, 0.074], code 0.102 [-0.021, 0.248] and QA 0.72 [0.09, 1.73]. The response looks close to separable at low reuse and is clearly not separable at high reuse, which is exactly where the largest effects live.
CONSEQUENCE. Both the four-coefficient joint form and its five-coefficient interaction extension DID NOT MEET the pre-registered threshold for every capability and descriptor at both tolerances, on the combined data and with U=132 held out as a pool. The family proposed for the data-requirement law is falsified rather than merely under-powered, and a future candidate must let the reuse effect grow with budget instead of carrying a budget-independent coefficient.
LIMITS the analysis states itself: no exactly equal budgets exist, so this removes the broad tolerance rather than all mismatch; at 0.3% the new pool only populates the 200k band; and the 1% slope involving the new pool is identified by one of its two seeds, so it is not two-seed replicated.

### C72. V99 scope of the reuse effect beyond 2Wiki (GPU, 2026-09-14; results/v99-scope)

Scores the controlled trajectories' endpoints on three QA distributions with the sampling, registers and scoring of the earlier scope check, so the numbers are comparable with it: the fresh 2Wiki sample, MuSiQue answerable-dev and TriviaQA rc.nocontext. Change from each trajectory's own initial student; positive means worse. Averaged over the two pool seeds:
gemma3-1b: reuse 11.7 -> 2Wiki +2.62, MuSiQue +1.99, TriviaQA +0.99; reuse 5.8 -> -0.22, +0.66, +0.48; reuse 3.9 -> -0.77, +0.48, +0.36; reuse 1.3 -> -1.10, +0.23, +0.26.
gemma3-270m: reuse 11.7 -> +0.60, +0.63, +0.90; reuse 3.9 -> -0.92, +0.06, +0.40; reuse 1.3 -> -1.16, -0.01, +0.36.
TWO STATEMENTS, and they differ. The COST of over-reuse is distribution-general: all three distributions get worse at high reuse, and all three improve monotonically as reuse falls. The BENEFIT is specific to 2Wiki: only 2Wiki crosses zero into a gain of about one nat, while MuSiQue flattens near zero and TriviaQA stays positive, that is slightly worse than the initial student, at every reuse level measured.
CONSEQUENCE for how a data requirement may be stated: a requirement of the form 'enough independent data to keep the QA loss increase below a threshold' is supportable across distributions, while any claim of a QA improvement remains a 2Wiki statement. This extends C45/C49, which found the gain did not transfer, by showing that the reuse dependence itself does transfer.
COMPLETED with the 4B student, which makes the size interaction explicit. Mean over the two seeds, change from each trajectory's own initial student: at reuse 11.7 the 4B loses 2Wiki +5.77, MuSiQue +3.01, TriviaQA +2.49, roughly double the 1B and far above the 270M; at reuse 5.8 it is still +2.40 / +1.43 / +1.43; at 3.9 it reaches +0.06 / +0.69 / +1.29; and only at reuse 1.3 does it gain on 2Wiki, -1.00, while still losing +0.28 on MuSiQue and +0.56 on TriviaQA. So the independent data needed to reach a given outcome rises with student size, and rises further for the external distributions than for 2Wiki: at every reuse level measured, no student reaches a gain on TriviaQA.
Two tool problems were fixed rather than worked around. The 4B adapters carry no base-model name, because peft leaves that field null when the base is loaded from a local path, and the guard treated an absent name as a mismatch and refused to score; absent and mismatched are now distinguished, a genuine mismatch stays fatal, and the output records which case applied. Changing the tool changed its protocol record, so it correctly refused to write into the existing output directory and the rerun went to a new subdirectory. Noted but not changed: those refusals exit zero where they should exit non-zero.
A hardware note: the Blackwell card cannot run this evaluation path either, failing on the same fp32 matrix product, so the whole scope run was done on the Ada card. Test fixtures written into results/v99-scope by the suite were moved to _trash rather than left to contaminate the tree.

### C73. A1 canonical development table (CPU, 2026-09-14; results/a1-development-table)

Fixes the only table any fit in this round may read. 22 core trajectories (18 controlled matrix + 4 dense critical-region), 150 checkpoints, 582 rows = 450 training-probe rows + 132 fresh-QA rows. 41 historical trajectories / 684 rows go to a SEPARATE external diagnostic file the fitting code cannot read; the 5 discarded-launch trajectories are excluded by assertion.
WHAT IT BUYS: at the 0.5% budget-matching tolerance there are now 44 checkpoint pairs (225 measurement rows, all 22 trajectories) that change the pool at a genuinely matched budget, median signed residual -51 supervised tokens, max 951. The controlled matrix alone yields none; these come from the four dense trajectories. I_T is inventoried at budget ratios 1.7-2.3 (124 checkpoint pairs). Changing the pool seed at fixed pool size (16 pairs at 0.5%) and changing the training seed (0 pairs; all core training seeds are 0) are kept as separate inventories and neither is an I_U.
VERIFIED IDENTITY: a budget-only predictor gives exactly zero data-intervention effect at exactly matched budgets; this is asserted in the artifact rather than assumed.
LIMITS RECORDED: per-domain supervised-token totals were never logged and are left missing rather than approximated by example counts; D_U_pool and D_U_seen are exact algebraic reconstructions, flagged per row in row_metadata.csv.

### C74. A2 three-structure comparison: negative, and the decisive test turns out to be impossible on existing data (CPU, 2026-09-14; results/a2-curvature-interaction)

STRUCTURES: F_log = (a+a'z)u + (b+b'z)v; F_curv = (a+a'z)u + (b+b'z)h_p(E) with h_p(E)=((1+E)^p-1)/p profiled on p in [-1,3] inside each training fold; F_int = F_log + k*u*v. u=log(1+T/T_ref), v=log(1+E), T_ref=100000 fixed, never fitted. One parameter set per structure produces the response and BOTH interventions; no intervention is fitted directly. Six baselines, four grouped hold-outs (leave a data rung, leave a student, leave the largest budget, leave a pool seed), all selection inside training folds.
RESULT, PRIMARY: against the strongest baseline, EVERY cell is negative -- no capability, target or split where any structure wins. Both pre-declared primary targets fail: code x I_T loses on all four hold-outs; QA-2Wiki x I_U loses on three of four folds with intervals clear of zero and is indistinguishable from zero on the fourth. The winning baselines are the same-input low-order response surface and the one-parameter reuse-only form.
RESULT, SECONDARY AND LABELLED AS SUCH: against the INNER-selected baseline (not the strongest), F_log beats the fitted mean-effect baseline for code x I_U on all three held-out students: 1B +0.0210 [0.0027,0.0345], 270M +0.0273 [0.0113,0.0417], 4B +0.0238 [0.0054,0.0423]. Three of three, intervals above zero. NOT a primary target, NOT against the strongest baseline, and deliberately not promoted.
CURVATURE: full-development fits are interior for every capability -- code p=0.69 [0.22,1.14], math 1.12 [0.35,1.40], QA-2Wiki 1.44 [1.02,2.22], TriviaQA 1.06 [0.30,2.66]. p=0 is the old log reuse term, p=1 is linear; the data prefers linear or stronger, consistent with the reuse penalty concentrating at high reuse. Fold ranges are wide, 1-2 of ~12 folds hit a boundary for three capabilities, and the flexibility buys no held-out accuracy: a direction, not an established exponent.
A BLOCKER THAT WAS CORRECT: the first run refused to execute because the permitted inputs carry no non-embedding parameter counts and it would not read a count off a model name. Counts summed from each snapshot's safetensors tensor shapes, minus the tied embedding and (4B/12B) the vision tower: 270M -> 100,326,016; 1B -> 697,896,064; 4B -> 3,209,010,688; 12B -> 10,759,155,456. The 270M student's embedding (167.8M) is LARGER than its body, so the three students span 32x, not the 15x the names suggest. Nominal names would have distorted every size-dependent coefficient.
THE STRUCTURAL FINDING: the development set contains ZERO complete four-corner rectangles in (T,E) space, for all three students. Reuse E = T/D_U and each trajectory has one pool, so E is a function of T within a trajectory; a rectangle needs two trajectories sharing a budget at different reuse, and no two trajectories share a budget. Rectangles that survive a naive search are built on the E-level where less than one pass through the pool has occurred, which carries no reuse information. The second difference across such a rectangle is exactly zero for ANY additive budget-plus-reuse structure whatever its curvature, so it is the only statistic separating additive from interacting WITHOUT fitting. It has never been run in this project and cannot be run on data collected so far. Verified independently, under two definitions of E, before the fitting output was read. Additive and interacting predictions disagree at the missing corners by more than the pre-registered 0.01 nat for every capability, and by 0.87 nat for QA-2Wiki.
CONSEQUENCE: the Stage A launch conditions are NOT met; the 16-trajectory confirmation package does not launch and its 44 GPU-hours stay unspent. What the round does authorise on this finding is at most four development trajectories to complete one missing corner.

### C75. A4 shrunk source correction: the P/Q input branch is closed, and a number I published was stale (CPU, 2026-09-14; results/a4-shrunk-source)

THE QUESTION. V92 found that pre-compression statistics help inside a parametric family yet lose to the source-free median configuration curve in 9/9 arm-and-capability pairs. One explanation had never been tested: that the source-dependent candidates lose because they RE-LEARN THE WHOLE RESPONSE, not because source information is worthless. The test keeps the median curve frozen and adds only a heavily shrunk correction fitted to its residual: prediction = unchanged V92 median(configuration) + X beta, with beta minimising squared residual plus alpha*||beta||^2, alpha selected by grouped inner CV inside each outer training fold from a grid fixed in advance (1, 10, 100, 1e3, 1e4, 1e6, infinity), ties favouring stronger shrinkage. Same panel, same splits, same arms, same capabilities, same information budget, same scoring as V92; comparator MAEs read directly from the V92 artifact, pinned by SHA256 29383f1b0a12dc2bd7c28479e8dd0257b72333f56b029812d9c3a0bec1d79b39.
RESULT. Positive paired gains with 95% intervals excluding zero: 0/9 on the primary leave-one-source-state-out split (the pre-fixed rule required at least 5/9), 2/9 on leave-one-size-out, 0/9 on both. Maximum shrinkage was selected in only 14/108 outer folds, so the data does prefer a finite correction -- it just does not convert into held-out gain. THE BRANCH IS CLOSED, as the pre-fixed decision rule says, and no further statistics, budgets or richer corrections are proposed.
CORRECTION TO A PUBLISHED NUMBER. C66 and the two round reports state that the statistics help in 3 of 9 pairs, naming pruning math, pruning QA and grouped-quantization math. That count came from the SIX-state, 378-row panel. After the Loni gap-fill completed the nine-state panel and V92 was rerun, the saved artifact records 459 primary rows (108 pruning, 243 grouped quantization, 108 per-channel quantization), nine states in every arm, and 4 of 9 qualifying pairs: pruning math, pruning QA, grouped-quantization math AND grouped-quantization QA. I reported the pre-rerun figure after the rerun had happened. The corrected count is 4 of 9. The conclusion that decided the round is unchanged: both whole-response candidates still lose to the source-free median in 9/9.
NOTED AND NOT YET FIXED: the V92 artifact's own prose and one legacy loader test still describe six grouped states and 378 rows, while its numbers are the nine-state panel. The numbers are right and the description is stale; the description needs fixing without touching summary.json.

### C76. Four-corner second difference, part 1: design, exact landing, the 1B verdict and the noise it is judged against (GPU + CPU, 2026-09-14/15; results/a3-corner-pools, results/a5-corner-second-difference, results/a5-training-seed-noise)

DESIGN (results/a3-corner-pools/plan.json, chosen by a boundary-grid search after my first objective was wrong). Corner 2 is the EXISTING checkpoint update-38 of the U=66 seed-41 matrix run (D_A=16962, T=50563). Pool B n/domain=130 seed=101 D_B=35376; pool C n/domain=279 seed=234 D_C=73766. Pool size is an OUTPUT of update-boundary alignment, not 2*D_A as I first demanded. Corners: 1=(50558, B), 2=(50563, A), 3=(105433, C), 4=(105450, B). Worst of the four mismatches 0.016% against a 0.5% limit; budget contrast 2.085, reuse contrast 2.086; domain-share drift <=1.76pp; both new pool hashes absent from all 32 recorded pools. --trajectory-tokens carries PROCESSED tokens (I had told codex the opposite; it checked the trainer and refused to emit commands until the objective was corrected), so thresholds were set by deterministic ledger replay, verified against the corner-2 run. LANDING: every new corner on BOTH students hit its predicted supervised total to the token (50558 / 105433 / 105450).
REGISTERED BEFORE TRAINING (paper/docs/prereg/corner_second_difference_prereg.md + amendment): I = d3 - d4 - d1 + d2, exactly zero for any A(T)+B*h(E) at fixed student; pool B enters twice with the same sign, so a pool-draw offset counts double. Primary readout the 2Wiki QA training probe (sigma_I 0.1608 from 30 matched seed pairs, predicted additive-vs-interaction disagreement 1.5546, ratio 9.7); MATH-500 marginal (0.0088 vs 0.0215); MBPP UNDERPOWERED (0.0254 vs 0.0144, ratio 0.57: a null must be reported as unresolved). Decision: reject additivity only if |I| > 2 sigma_I AND both students agree in sign.
1B RESULT: QA I=-0.2434 (band 0.3216), math +0.0076 (band 0.0176), code -0.0140 (band 0.0508): all INSIDE. Secondary check on the fresh-sample distributions, all four corners evaluated on one card: 2wiki_new -0.3339 (band 0.7646), musique +0.0434 (1.672), triviaqa -0.1134 (0.288): all inside, consistent.
TRAINING-SEED NOISE, measured for the first time in this project (user-approved fifth trajectory: same pool B, --seed 1, 14 checkpoints at identical supervised budgets): sigma_I from training seed = QA 0.0768, math 0.0048, code 0.0019 -- about half the registered pool-seed figures, so the registered band is conservative. The QA seed difference grows with budget (0.003 at 46k -> 0.117 at 105k supervised). The 1B QA value is ~3x the training-seed sigma and ~1.5x the pool-seed sigma: inside the band, not zero.
HOW THIS RECONCILES WITH C71: the drift test divided the pool-change effect by the LOG reuse bracket and found it not flat, which falsifies the four-coefficient LOG form's implication, not additivity. A2's profiled curvature (p 0.7-1.7, near linear) is exactly what makes the log-bracket ratio drift while the second difference stays near zero. The two results are consistent. The 4B verdict and the registered decision are in C77.
