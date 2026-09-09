# P0 — Related work vs. this study: contribution boundary and baseline list

Bounded check (abstracts/pages verified 2026-09-09; no open-ended search). Purpose: remove the "existing law with
a new benchmark" doubt without inventing firstness. Fields: (1) prediction endpoint, (2) observable inputs,
(3) target-side calibration cost, (4) compression algorithm + training setting, (5) validation split,
(6) overlap / difference with this study.

| Work | (1) Endpoint | (2) Inputs | (3) Target calibration | (4) Algorithm / training | (5) Validation | (6) Overlap / difference |
|---|---|---|---|---|---|---|
| Pruning Laws for LLMs (2504.04342 v2) | per-task post-pruning performance (8 tasks) | unpruned performance, sparsity, dense vs MoE, task; "task- and method-specific coefficients" | not stated in abstract; coefficients are task/method-specific (implies fitting per task/method) | unstructured, width, depth pruning; recovery not stated | 10 LLMs 1.3B–30B + 20B MoE; zero/one-shot transfer to unseen models; "<7% extrapolation error" | OVERLAP: uses pre-pruning performance and per-task endpoint. DIFFERENCE: we use capability-conditioned *loss* (not task accuracy), test a controlled training-state axis ($D_0$), an unseen-density axis with same-input baselines, and separate K0/K1 budgets; we do not claim a universal pruning law. |
| P2 Law (ACL 2025) | post-training loss of the pruned model (aggregate) | size before pruning, post-training tokens, pruning rate, loss before pruning | not stated; law fit across runs | common pruning methods on Llama-3/Qwen-2.5 + post-training recovery | generalizes to larger data, sizes, higher pruning rates | OVERLAP: pre-pruning loss + size as inputs (so "source information helps" is NOT new). DIFFERENCE: P2 targets aggregate loss after recovery; we target per-capability loss without recovery in the main arms, and test source-state and density transfer with frozen predictions. Recovery appears only as an appendix phenomenon here. |
| Scaling Laws for Precision (2411.04330) | loss degradation from low-precision training/inference incl. PTQ | N, D, training/inference precision, bit-width | PTQ term fit across 465+ runs | PTQ (quantizer unspecified in abstract); pretraining-scale | up to 1.7B / 26B tokens | OVERLAP: "PTQ degradation increases with more pretraining data" is THEIR claim; our $D_0$-vs-int4/int3 fragility must cite it, not present it as new. DIFFERENCE: we report per-capability responses on discrete RTN bits with same-budget baselines and find no demonstrated gain at $\ge$4 bits and unstable int3 magnitude transfer. |
| Distillation Scaling Laws (2502.08606) | distilled student performance vs compute allocation | compute budget, student/teacher sizes | fit across runs | pretraining-scale distillation (logit) | held-out compute allocations | OVERLAP: distillation predictability from size/compute. DIFFERENCE: black-box trace distillation with LoRA, per-capability signed response $\delta_c$ with the student's own dense as reference, reuse count $E=T/D_U$ as coordinate, and an unseen-pool prospective. |
| Task-Specific LLM Distillation (2606.24747) | in-domain task quality + general benchmarks under compression | dataset size, compression ratio, supervision format, pruning schedule | not stated | logit- and LoRA-based distillation under iterative structural pruning; CoT-blended loss | scaling-law derivation (split unspecified) | OVERLAP: task-specific quality under compression. DIFFERENCE: we do not combine pruning with distillation; we isolate the data-pool/reuse axis and a sampling-protocol control. |

## Contribution boundary (what we do NOT claim as first)
- "Source information helps prediction" (P2 Law uses pre-pruning loss + size).
- "More pretraining increases PTQ damage" (Scaling Laws for Precision).
- "Task-level / per-task loss measurement" (Pruning Laws; task-specific distillation).

## What this study adds (the claim we make)
A capability-conditioned, two-axis, information-budgeted comparison across three parallel arms with the same
endpoint and validation standard: (a) a controlled training-state axis (same family, known $N_0,D_0$) that
separates the incremental value of $D_0$ from dense loss; (b) unseen-configuration prediction (density, pool)
with frozen predictions and same-input baselines, which shows that source conditioning carries the pruning
result while the shared power form has no demonstrated extra advantage; (c) explicit K0/K1/Oracle budgets so
per-model calibration is never passed off as pre-compression prediction; (d) a distillation sampling-protocol
control (random endpoints) with a new-pool prospective.

## Compatible literature baseline decision
- P2 Law's functional form is defined on post-recovery aggregate loss with post-training tokens as an input;
  our main arms have no recovery and a per-capability endpoint, so a faithful implementation is INCOMPATIBLE
  at the same endpoint/inputs. Not implemented; recorded as such (no forced fit, no "beats P2" claim).
- Pruning Laws' form uses unpruned performance + sparsity with task/method coefficients: its K0-compatible
  reduced form is a source-loss-conditioned function of sparsity, which is structurally what our strength-only
  vs source-conditioned comparison already spans (A1/A2/power). No additional baseline is added; the
  comparison is cited as related, not reproduced.
- Precision law's PTQ term requires the training-precision axis we do not vary; cited for the $D_0$ claim.
Final baseline list for P1 therefore stays: power (frozen), A2, A1, strength-only, median-curve, zero
(pruning); full/no-D0/per-bit-median/zero (quantization).
