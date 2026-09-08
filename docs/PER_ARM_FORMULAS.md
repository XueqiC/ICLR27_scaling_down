# Per-arm candidate formulas, domains, K0 error, and calibration (deliverable #4)

Synthesis of the frozen results (ledger C1–C27b). Endpoint = capability LOSS ΔL_c (native-token CE nats).
K0 = basic-input prediction (pre-compression info only, 0 target calibration). "Transfers" is used only
where an INDEPENDENT prospective test supports it — post-hoc oracle rescale is labeled diagnostic.

---

## Pruning

- **Candidate form**: ΔL̂_c(d) = a_c · ((1−d)/0.3)^γ_c (v28/v29). Density power; γ_c a shared shape,
  a_c a per-model amplitude. Complex family/size/cliff forms do NOT beat density-only (C5).
- **Coefficients**: γ_c shared across models; **a_c does NOT transfer from {N0, family, dense L_c}**.
  Descriptors that might predict a_c (layerwise retention V2, dense activations V3) do NOT help — worse
  than the basic-input baseline on the 11-model panel (C25b/v32).
- **Domain of validity**: pre-cliff (roughly d ≥ 0.6); the cliff is non-perturbative in loss (C2) and
  falls in a narrow proportional band. Deep/collapse region is not described by the smooth power.
- **K0 error**: on the FULL range no basic-input predictor beats zero-change; on smooth rows density-only
  ≈ a per-config constant. Prospective NEW sources FAIL: Qwen3-8B MAE 0.62 > zero 0.24; Qwen3-14B amplitude
  over-predicted ~14× and **QA sign-flips** (loss drops under pruning). Failure is amplitude/sign, not shape (C25/C25b).
- **Calibration**: one point at d=0.9 is CATASTROPHIC (near-zero response → 934× amplification); shrinkage
  stabilizes the dev set but doesn't beat zero on new sources. K1 (d=0.9) worsens math/QA (C26).
- **Controlled panel (Pythia)**: adding D_0 to {N0,L0} lowers held-out error (leave-one-size-out prune-math +0.47 CI excl 0), but vs the
  STRONGEST per-config baseline the full model significantly wins only for CODE; nearly all gain is in the
  aggressive region (d=0.6) — mild d>=0.8 ~0. Config-indicator model = source-transfer at fixed d, not a law
  over d. See CONTROLLED_PANEL_PREDICTOR.md. QA remains hard.

## Quantization

- **Candidate form**: ΔL̂_c(b) = a_c · g(b). Three g tested: fixed 4^-b (η=2), actual-step (7/15)² (η≈2.2),
  shared learnable η. **Both fixed laws REJECTED** (paired-diff CI, math/code/pooled); shared η fits better,
  512-probe η=3.47 [2.98,4.41] — decays faster than either fixed law (C8).
- **Domain**: int8/6 near-lossless, int4 measurable/selective, int3 collapse (OLMo3-32B is an int3 exception).
- **K0 error**: in the MEASURABLE 4/5-bit region NO candidate beats zero-change (η gain 0.057 [−0.025,0.140]);
  the shared-η all-bit "win" is 98.7% from the int3 collapse region (v30b). So the shape law has no demonstrated
  predictive value in the smooth region. Prospective: Qwen3-8B PARTIAL; Qwen3-14B int8/6 lossless, int4 QA sign-flip
  (−0.23), int3 collapse.
- **Calibration**: Qwen3-14B shape pre-check INCONCLUSIVE (4/5-bit candidate predictions coincide < measurement
  precision); K1 (4-bit) worsens all (C26). q_c is a per-model 3-bit calibration, not basic-parameter.
- **Controlled panel**: D_0 predicts quant-code fragility (full-vs-baseline +0.93 CI excl 0); but BY-BIT this is ~entirely int3
  collapse (b>=4 improvement ~0) — no smooth-region predictive value. Source-transfer at fixed b, not a g(b) law.

## Distillation

- **Candidate form**: δ_c = L_c(S_KD) − L_c(S0) = dense-baseline + transfer-response; the transfer-response is
  governed primarily by **reuse count E = T / D_U** (C21/v31b/v37). `floor − α log r` retired.
- **Coefficients**: math/QA ≈ per-capability constant; CODE has genuine data-dependence (confirmed on gemma3-4b,
  paired-diff CI excludes 0). Reuse count E is the DOMINANT coordinate (matched-E collapses ~85–95% of the
  matched-T pool-size gap on gemma3-1b and Qwen3-4B).
- **Domain**: positive-budget region only; δ_c(0)=0. Response is signed — δ_code>0 is a TRANSFER RESPONSE
  (more data raises the defined code loss), not a "gain". Do NOT constrain the law to monotone worsening
  (U600 QA improves then partially recovers).
- **K0 error**: dev-LOMO basic-parameter prediction MET (beats δ=0). New STUDENT Qwen3-4B: math/code sign
  DIFFERS from gemma (gemma δ>0 / Qwen δ<0); QA improves under KD in BOTH families (not Qwen-specific) (C25).
- **Sufficiency / attribution of E**: matched-T ALREADY controls optimizer steps (107≈108, 214=214), processed
  tokens and supervised tokens to ~1-3% (DISTILL_STEP_CONTROL.md) — the large matched-T pool-size gap is driven by
  reuse count E, NOT steps/exposure. matched-E supports E as the effective candidate coordinate, but does NOT establish 'exposure weak' or
  'E sufficient': U75/U600 co-vary independent-data volume, composition and repetition (T=E·D_U couples them);
  v37 0.05-nat insufficiency stands, QA ~0.5-0.8 nat residual. Sufficiency needs NEW-CONFIG validation (P3). Adding log D_U (additive or
  low-DOF interaction) does not help held-out (v37/v37b). No matched-optimizer-step rerun needed.
- **Controlled panel (P2, full 3x3 LoRA)**: delta_c source-transfer works ONLY for MATH ({N0,L0,D0} beats
  no-D0 AND constant baseline, MAE ~halved; delta_math grows monotonically with D0). CODE/QA: a per-capability
  CONSTANT delta is as good or better, D0 adds nothing (code direction size-inconsistent; qa noisy w/ outlier).
  Weaker & capability-specific vs pruning's all-cap prospective generalization (C28/C29). See DISTILL_CONTROLLED.md.

---

## Cross-arm summary

- The three arms have **different** gaps (not a unified negative): pruning = amplitude/sign transfer + calibration
  instability; quantization = no predictive value in the measurable region (collapse-region artifact) + inconclusive
  shape; distillation = reuse-count-dominated with an exposure-confounded residual.
- On **heterogeneous finished models**, per-model amplitude/sign does not transfer from {N0, family, dense L_c}.
- On the **controlled panel**, arms differ: PRUNING generalizes prospectively across all caps (C28); QUANT mixed (advantage concentrated in aggressive int3, magnitude-unstable; NOT an artifact); DISTILLATION source-transfer is math-only, code/qa best fit by a constant (C29).
- On the **controlled Pythia series**, D_0 adds incremental value beyond dense loss (4/12 CIs excl 0), but vs the
  strongest per-config baseline the full model significantly wins ONLY for CODE, concentrated in the aggressive
  configs (pruning d=0.6 / quant int3); QA excepted; and it is source-transfer at fixed d/b, not a compression-axis law.
- Qwen-family QA IMPROVES under pruning and int4 quant (Qwen-specific); under distillation QA improves in all families.

## Compression-strength axis (round-6, SECOND AXIS)
- **Pruning (C30)**: frozen shared-shape ΔL̂_c(x,d)=A_c(x)·((1−d)/0.3)^γ_c predicts UNSEEN densities and
  beats every strength-only baseline for math/code (MAE ~0.4–0.5 vs ~1.4–1.6), incl. a new source-state;
  interpolation strong, deep extrapolation under-predicts at cliff, QA amplitude doesn't transfer. First
  real unseen-strength law. See PRUNE_STRENGTH_AXIS.md.
- **Quantization (QUANT_PARTITION.md)**: no unseen-bit axis; >=4bit region has ~nothing to predict (tiny
  errors), int3 collapse magnitude is UNSTABLE (helps on some new sources, fails on others). Predicts the
  RISK of large-damage entry, not a smooth magnitude law. No clean second-axis result. Confirms C28.
- **Distillation**: P3 (hpg, new pool U=225) pending.

## Open (next round)
- A per-arm frozen prospective on a fresh source with the full {N0,D0,L0} model (Qwen3-14B pruning/quant already run;
  a distillation new-student with matched LoRA recipe pending).
- Disentangle the distillation reuse residual from training exposure with a matched-optimizer-step design.
- Prediction intervals per arm reflect the small model count (≤12); widen probes improve measurement, not model count.
