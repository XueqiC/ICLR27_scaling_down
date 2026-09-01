# Results Ledger
Status vocabulary: VERIFIED / PRELIMINARY / RUNNING / PLANNED / BLOCKED / REJECTED.
Every manuscript claim must trace: claim -> figure/table -> processed result -> raw artifact -> config+code version.
Snapshot: 2026-09-01. Raw artifacts live in the rai project tree (`results/...`); curated copies in this repo under `results/`.

| ID | Claim | Status | Evidence | Models | Validation | Manuscript |
|---|---|---|---|---|---|---|
| C1 | Capability ordering under pruning (QA most robust pre-cliff) | VERIFIED (12 models, 4 families) | results/v6-capability-geometry/*/report.md | 12 | descriptive, cross-family replication | Tab. grid |
| C2 | Cliff in narrow proportional band; non-perturbative in loss | VERIFIED (descriptive) | v6 reports + v6b report_b | 12 | two-scale Taylor-failure check | exp §curves/§mech |
| C3 | Block structure of benchmark gradient signatures | PRELIMINARY (7 models; bootstrap/permutation/prompt-stability controls pending) | results/v9-capability-regions/*/report.md | 7 | none yet (point estimates) | Fig. blocks |
| C4 | Targeted 0.05% ablation selectively destroys one capability | PRELIMINARY (1 model; random/magnitude-matched/global-Fisher controls pending) | results/v9c-ablation/gemma3-1b | 1 | none | exp §regions |
| C5 | Pre-cliff law dL = g.dw + B m^gamma, gamma in [1.2,2.2], within-model shared | PRELIMINARY (in-sample R2 only; 24 fits verified; held-out extrapolation pending) | scratch refit over v6 artifacts | 12 | in-sample only | exp §law |
| C6 | Signed first-order term predicts QA sign 4 families + OLMo size flip | PRELIMINARY (matched pred/meas points; no uncertainty; eval-noise bootstrap pending) | v6b report_b per model | 12 | none | exp §mech |
| C7 | Geometric retention smooth; cliff at family-specific critical retention | PRELIMINARY (3 models, 2 families) | results/v11-geometry-damage/* | 3 | none | exp §orderparam |
| C8 | Quantization: int8/6 lossless, int4 selective, 3-bit collapse | PRELIMINARY (9 models; no functional-form fit; no held-out) | results/v10-quantization/* | 9 | none | exp §quant |
| C9 | answer_only hurts math more than full traces | PRELIMINARY (unmatched controls: tokens/updates/format not yet controlled) | results/v12-distill/*/eval.json (5 runs) | 2 students | none | exp §distill |
| C10 | Recovery power law; asymptotic residual under fixed recipe | PRELIMINARY (1 model, 1 density, LoRA-only; form diverges at D_R=0 -> refit saturating form) | results/v13-recovery/gemma3-270m | 1 | none | exp §recovery |
| C11 | Aligned recovery fast-but-re-damaging vs general-data monotone | PRELIMINARY (single seed/model) | v13 traces vs c4 runs | 1 | none | exp §recovery |
| C12 | Mid-scale robustness peak (Gemma 12B; prior Qwen 14B) | PRELIMINARY (Gemma direct; Qwen from prior ability-space grid) | v6 grid + prior re-analysis | - | none | exp §family |
| C13 | Concentration predictor retired | VERIFIED (as negative result within our grid) | v6 reports concentration lines | 12 | pre-registered then falsified | exp §family |
| C14 | Dense reference law variants (D1-D4) | PLANNED (no fits yet; placeholder in paper) | - | - | held-out per plan | method §laws |
| C15 | Cross-method shared law (M1/M2/M3) | PLANNED | - | - | held-out incl. leave-method-out | - |
| C16 | Agent-level workload validation + method-selection map | PLANNED | - | - | held-out workloads | - |
| C17 | Prior-grid re-analysis (anchor recipe artifact; QA contamination ~1/3; rank-only IRT) | VERIFIED (recomputed from committed cell tables) | results/v1-v3 dirs (rai) | prior grid | recomputation | exp §reanalysis |
