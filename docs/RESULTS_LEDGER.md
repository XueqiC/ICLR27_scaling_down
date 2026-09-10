# Results Ledger
Status vocabulary: VERIFIED / PRELIMINARY / RUNNING / PLANNED / BLOCKED / REJECTED.
Every manuscript claim must trace: claim -> figure/table -> processed result -> raw artifact -> config+code version.
Snapshot: 2026-09-01. Raw artifacts live in the rai project tree (`results/...`); curated copies in this repo under `results/`.

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
| C25 | Basic-parameter prediction of compression response (predict from N0/family/dense L_c only) | LOMO dev + PROSPECTIVE on held-out new source Qwen3-8B: PRUNING FAILS (Qwen3-8B prospective MAE 0.62 WORSE than zero-change 0.24); QUANTIZATION PARTIAL (prospective MAE 0.23 beats zero 0.32, math b4 pred 0.30≈actual 0.28); DISTILLATION MET in LOMO (a≫beats zero). Distillation-on-new-source not yet run. SECOND PROSPECTIVE SOURCE Qwen3-14B (rai, cpu-reference prune) REPLICATES the Qwen3-8B pruning failure: prune-robust (actual ΔL d0.6 math +0.126, code +0.277) with QA loss DECREASING under pruning (d0.6 −0.797, a SIGN FLIP), while frozen v28 Mode-A over-predicts math ~14× (≈+1.8 pred like Qwen3-8B) and gets QA sign wrong. So the basic-param pruning-amplitude failure is now confirmed on TWO independent held-out Qwen models. DISTILLATION PROSPECTIVE on new student Qwen3-4B (n=3 seeds, D=75/600, same teacher/recipe): δ_c is NEGATIVE at both budgets (KD IMPROVES Qwen3-4B: D75 math −0.170/code −0.277/qa −2.67; D600 −0.105/−0.206/−2.31) — OPPOSITE SIGN to the gemma students (δ>0). The data-response SHAPE transfers (δ(600)−δ(75) = +0.066 [+0.045,+0.086] math, +0.071 [+0.046,+0.096] code, both EXCLUDE 0, same direction as gemma) but the SIGN/OFFSET does NOT — any gemma-fit predictor (δ=0, positive mean, positive data-response, metadata) gets the sign wrong (δ=0 MAE math 0.14/code 0.24/qa 2.49). MATCHED CONTROLS (Qwen3-4B, U75 seed1 n=1 vs U600 n=3; seeds0/2 GPU-fault-blocked, poller upgrading): matched-SEEN-TOKENS gap (n=3, all CIs excl 0: math +0.15/+0.47, code +0.08/+0.18, qa +2.6/+3.1) SHRINKS under matched-REUSE-COUNT E to small residuals (math −0.009/−0.043, code +0.015/−0.13, qa +0.53/−0.77; some CIs excl 0). So reuse count E=T/D_U is the DOMINANT coordinate (removes ~85-95% of the gap) but NOT complete (small residual) — REPLICATES gemma3-1b (v31b) on a NEW student family. E-SUFFICIENCY (v37, gemma3-1b + Qwen3-4B, 108 signed obs, 0.05-nat tol): E-only is NOT sufficient — matched-E residual intervals fail the equivalence criterion for math/code/QA; AND adding log D_U does NOT lower held-out MAE (worsens primary; secondary QA gain 0.00065 nat). So reuse count E is the DOMINANT coordinate but leaves a real residual that unique-data volume (log D_U) does NOT explain — the distillation law needs E plus something else (not D_U). RESIDUAL DIAGNOSIS (v37b): the matched-E residual COINCIDES with an ~8× training-exposure gap (optimizer steps 28 vs 224; supervised tokens 39k vs 317k — because T=E·D_U and D_U differs 8×). A low-DOF k_c(E)·log(D_U/D_ref) INTERACTION also gives NO held-out MAE gain; only gemma-math meets the exposure-explanation rule but is indistinguishable from a constant pool gap. So the residual is CONFOUNDED with training exposure and its causal attribution (exposure vs a genuine unique-data effect) is UNRESOLVED on this data — not a new latent capability variable. δ predicted with sign; QA noisy.  Qwen3-14B QUANT (absolute-loss check, rai idx1 cpu-ref): int8/6 near-lossless, int4 mild (math +0.09/code +0.31) with QA IMPROVING (−0.23, same Qwen QA sign-flip as pruning), int3 CATASTROPHIC collapse (+9 to +11 nats). Confirms high-bit-lossless + int3-cliff and the QA sign-flip basic-param predictors miss (quant shape test was pre-registered INCONCLUSIVE, v30b). CROSS-CUTTING (CORRECTED 2026-09-08, per advisor): the three arms have DIFFERENT gaps, NOT a unified 'shape transfers / amplitude fails' negative. Pruning: predictor amplitude+sign fail on new Qwen, shape 'compatibility' is only post-hoc-oracle-rescaled (not independent shape prediction). Quant: Qwen3-14B shape test INCONCLUSIVE (cannot claim shape transfer); int3-collapse prediction has real value. Distillation: matched-E shrinks (not 'collapses') the gap. SIGN facts (results/sign-matrix): distillation QA δ<0 in BOTH gemma AND Qwen (KD improves QA universally, NOT Qwen-specific); the real cross-family sign difference is math/code (gemma + / Qwen3-4B −); Qwen-specific QA-improvement is under PRUNING (gemma QA damaged at d0.6) and int4 QUANT. δ=0 has no sign | results/v28-new-source-pred + Qwen3-8B + Qwen3-14B + Qwen3-4B distill actual + v29 | 12 LOMO + 3 prospective (2 prune, 1 distill) | LOMO + frozen prospective ×3 | RQ2 §prediction |
| C27b | Pythia 4-input comparison (v36b) — separates 'D_0 redundant' from 'both predictors fail' (advisor's key table) | Per-arm/cap, same leave-one-step-out holdout, MAE + improvement over strongest simple baseline (config-median/mean) + paired CI, for zero / {N0,D0} / {N0,L0} / {N0,D0,L0}. 3-SIZE (160m/410m/1.4b): the COMBINED {N0,D0,L0} is the BEST model for math/code and beats the per-config baseline substantially (quant-code base 1.55->combined 0.63; pruning-math 0.41->0.26; quant-math 1.41->1.14), while single inputs {N0,D0} or {N0,L0} ALONE do not beat baseline. On the 3-size panel the combined model is best (on 2 sizes it was worse); this is descriptive — expanding the panel changed coverage/held-out/folds together, not attributable to 'overfitting cured' alone. So both N0,D0,dense-L0 are needed and D_0 contributes real power beyond dense loss for math/code; QA remains hard (combined does not beat baseline). So the earlier 'adding D_0 gives no gain' was the combined model overfitting, not D_0 being uninformative. PRUNING: no input beats the per-config baseline (QA models HURT: {N0,L0} 0.90 vs 0.16). D_0/L0 correlated along trajectory. Also 3 curves: dense L0(D_0), increment ΔL(D_0), absolute compressed L(D_0) — more training → bigger increment but NOT necessarily worse absolute compressed loss | results/v36b-input-comparison + PYTHIA_INPUT_COMPARISON.md | 2 sizes × 3 steps | 4-input LOSO, paired bootstrap | exp §prediction |
| C27 | Controlled training-history panel (Pythia, v36) — does D_0 (training tokens) predict compression response beyond dense loss? | 3-SIZE (160m+410m+1.4b × step16k/64k/143k, 160m verified distinct; pythia-2.8b DROPPED — current public HF files show DUPLICATION: step64000 & step143000 single-file blob hashes are identical upstream; step16000 differs but has sharded files whose actually-loaded file needs separate confirmation. So dedup of the duplicated stages is justified; not a claim that all 3 original checkpoints are corrupt). RAW SIGNAL STRONG: fragility rises sharply with training (1.4b prune ΔL@d0.6 +0.37→+1.15 early→late; int4 +0.08→+0.27). BUT held-out (leave-one-step-out) fit: adding D_0 does NOT reduce ΔL prediction error beyond {N0, dense L0} (4/12 point improvements, 0 CIs above 0). UPDATED (3-size clean panel, leave-one-size-out + leave-one-step-out): adding D_0 to {N0,L0} now LOWERS held-out MAE in 7/12 arm/cap/holdout comparisons, 4 with CIs wholly above 0 (large: quant-code LOSO A2.08->B0.63, quant-math A2.03->B0.77, pruning-math A0.62->B0.15). So on the clean 3-size panel D_0 DOES provide real predictive power beyond dense loss for MATH/CODE (QA noisy, exception). More accurately: the EXPANDED 3-size panel yielded incremental predictive evidence NOT DETECTABLE on the smaller 2-size panel (which already excluded 2.8b) — expanding the panel changed training coverage, the held-out object, AND fold composition simultaneously, so the gain is NOT attributable to any single cause (not specifically 'overfitting cured'). This does NOT prove dense loss captures everything See docs/CONTROLLED_PANEL_PREDICTOR.md for the FULL dual-baseline read: D0 beats {N0,L0} in 4/12 comparisons (CI excl 0), but vs the STRONGEST simple baseline (per-config median) the full model significantly wins ONLY for CODE (both arms); math CI includes 0; and BY-CONFIG nearly all quant gain is the int3 collapse region (b>=4 ~0) and pruning gain is at d=0.6 — the smooth region adds ~0; the gain also vanishes when the latest/max-D0 step is the held-out target. Predictor uses config INDICATORS, so it is source-transfer at FIXED d/b, not a law over the compression axis. — D_0 and dense loss are highly correlated along the trajectory (hard to separate), and a CI including 0 can still hold a meaningful gain. D_0 is KEPT as a recorded candidate. (Distinct question from predicting dense capability itself from N,D_0.) Direct ΔL fit (not a_c-label-regression). Fixed-recipe SERIES (arch/hparams still vary across Pythia sizes). Distill arm DONE (hpg smoke, pythia-1.4b step16k/143k, D=600): D_0 modulates distillation too — more training -> more code damage (+0.018->+0.066) + more QA improvement (-0.589->-1.149); QA improves under KD at both. So ALL 3 arms show D_0-dependent responses on Pythia. CAVEAT: Pythia distill fell back to FULL-FT (peft absent on hpg tf5), not LoRA -> within-Pythia D_0 comparison valid. LoRA CONTROL (v12 assertion mode=lora, rai idx3) confirms the training-stage effect HOLDS under LoRA (early->late: code +0.19->+0.32, qa -0.43->-0.77, same direction as full-FT); LoRA and full-FT show a CONSISTENT training-stage TREND across the two tested stages, but with clear response-MAGNITUDE differences (NOT 'recipe only changes magnitude' — e.g. early-math full-FT -0.005 vs LoRA +0.064, both near zero); the training-stage effect is not a full-FT artifact. Needs a 3rd clean size for leave-one-size-out | results/v36-pythia-controlled + PYTHIA_CONTROLLED.md + pythia-smoke | 2 sizes × 3 steps | leave-one-step-out, direct fit | exp §prediction |
| C26 | Unified information-budget diagnosis K0/K1/Oracle (v35, 3 arms, 0.1-nat tol) — separates the 3 failure causes per arm | K0=pre-compression basic-input (0 cal, MAIN goal); K1=+1 dev-fixed calibration point (prune d0.9, quant 4-bit, distill D75); Oracle=post-hoc all-target rescale (DIAGNOSTIC upper bound, NOT a predictor). PRUNING: appreciable Oracle FORM error on all dev caps; Qwen math/code Oracle≪K0 → source-MAPPING failure; K1(0.9) WORSENS math/QA → calibration UNSTABLE; good Oracle = post-hoc compatibility, NOT independent shape prediction. QUANT: K0 useful only in collapse region; K1(4-bit) WORSENS all; Oracle int3-zero is a fit artifact; 4-bit/QA mismatch remains; prospective shape INCONCLUSIVE. DISTILL: dev math/code K0 accurate, D75 cal worsens it; QA Oracle form error; Qwen4 cal repairs math/QA source-shift but worsens code; Oracle code-zero is a 1-point-fit artifact → NO shape transfer established. So the 3 arms have DIFFERENT gaps (form / mapping / calibration), not a single unified failure | results/v35-info-budget + INFO_BUDGET_K0K1.md | 12 dev + prospective rows | K0/K1/Oracle paired-model bootstrap | exp §prediction |
| C25b | Pruning-failure diagnosis (v29): WHY basic-param pruning prediction fails, decomposed | (1) The "0.33 vs 1.3-1.9" headline gap is DOMINATED BY DAMAGE-RANGE SELECTION, not form/calibration: controlled Mode-A bridge attributes combined training+scoring range effect 1.43 [0.96,1.93] nats vs form change −0.19 [−0.56,0.20] and historical target-calibration access ~0 (0.00003 [−0.012,0.015]) — EXCLUDES "new form is worse" and "old form cheated with target calibration". (2) On the FULL range NO basic-param predictor beats zero-change (v28_A gain over zero 0.07 [−0.43,0.57]); on smooth pre-cliff rows density_only≈zero (~0.30 both), and v28's metadata mapping HURTS smooth math/code (noisy on 12-model panel). (3) Qwen3-8B failure is AMPLITUDE/SIGN, NOT shape: amplitude/sign MSE fraction math 0.99998/code 0.99975/qa 0.983; best positive scale math 0.08 code 0.24 (predictor overshoots 4-12×, Qwen3-8B anomalously prune-robust), QA scale −2.21 = SIGN REVERSAL (QA loss DROPS under pruning d0.6 −1.02). Density SHAPE transfers; AMPLITUDE coefficient a_c does NOT. (4) One-point d=0.9 calibration CATASTROPHIC (near-zero 0.9 response → 934× amplification, v28_B pooled 15.2); nested shrinkage stabilizes dev (MAE reduction excl 0 for math/qa/pooled) but still loses to zero on Qwen3-8B. DESCRIPTOR TEST (v32, signed a_c, LOMO): adding weight & layerwise-retention descriptors (V2) does NOT help — V2 is SIGNIFICANTLY WORSE than the basic-param V1 on all caps (math MAE 2.07 vs 0.87, code 2.15 vs 0.90, qa 1.84 vs 0.88; gain-over-V1 CIs all exclude 0 = worse), overfitting a small panel; V1 itself doesn't beat zero-change (math gain +0.28 [−0.13,+0.70]). So the layerwise-retention distribution does NOT rescue amplitude prediction here. CONFIRMED on 11-model cohort (muse-30b added via a tf5 env; only gemma4-31b excluded — library-unsupported by ANY transformers incl git-main, no Gemma4Config/remote code): V2 still significantly worse than V1 (math −0.85 [−1.65,−0.16], code −0.81 [−1.63,−0.11], qa −0.80 [−1.49,−0.15] gain-over-V1, all exclude 0). So the retention-descriptor negative is robust. V3 (dense-activation / Wanda-style descriptors, 11 models, big-model activations via hpg B200 tf5) ALSO does NOT help — V3 worse than V1 (gain-over-V1 math −0.94, code −1.26 [excl 0], qa −0.58). So NEITHER layerwise-retention (V2) NOR activation (V3) descriptors rescue signed-a_c prediction; the pruning amplitude a_c is unpredictable from ALL tested descriptors (basic params, retention, activations) on this panel — a complete negative for the descriptor line. Amplitude a_c remains unpredictable from these dense descriptors | results/v29-prune-diagnosis + v32-descriptors + PRUNE_FAILURE_DIAGNOSIS.md/PRUNE_DESCRIPTORS.md | 12 LOMO + 2 prospective; V2 on 10 | matched-condition bridge + oracle-scale + shrinkage + nested descriptor LOMO | exp §prune |


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

**Test MAE (nats)**

| test | candidate | math | code | qa |
|---|---|---|---|---|
| bit test: b=4 unseen, g in {64,256}, six dev states (12 cells) | (not found in compare.json: top keys ['schema_version', 'precommitted_rule', 'response', 'test_sets', 'rows', 'missing']) | | | |
| granularity test: g=128 unseen, b in {3,4,5}, six dev states (18 cells) | (not found in compare.json: top keys ['schema_version', 'precommitted_rule', 'response', 'test_sets', 'rows', 'missing']) | | | |
| joint test: new state 1B@96k, g=128, b in {3,4,5} (3 cells) | (not found in compare.json: top keys ['schema_version', 'precommitted_rule', 'response', 'test_sets', 'rows', 'missing']) | | | |

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
