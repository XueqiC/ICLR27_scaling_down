# RESTRUCTURE_MAP — "developing predictive laws" rewrite (2026-09-10, Stage A)

Repo: paper/ (submission repo, branch main, HEAD 79cf7cb, working tree clean). Analysis repo: project root (results/ tracked).
Baseline PDF kept: deliverables/scaling_down_law_round3_2026-09-09.pdf (paper 1c8837a build).

## Data cut-off (single cut-off for text, tables and figures)
- INCLUDED: heterogeneous panel (v6/v10), controlled Pythia panel (v36b, v38, v40/v42, v44, v46, v49), round-3 pruning
  deliverable (v53, C35) and grouped-RTN quantization (v54/v55, C36), distillation source-axis (v39), old-protocol unseen
  pool (v41/v43), and the CPU forms/capability-conditioning analysis on the complete P2-v2 dev matrix (v56, C37).
- FROZEN BUT NOT YET MEASURED at cut-off: P2-v2 test matrix (v50 freeze 33c706c; nine U375 runs + 4B dev-pool runs
  running). Written only as "frozen, pending"; if the compare lands before Stage C, ALL tables/figures are regenerated
  once from that later state and this line is updated. No P3 results.
- Not written as results: anything else in the round-3 plan that is not in results/.

## New outline (default from the directive) and the one-sentence conclusion each section carries
1 Introduction — specialised workloads need per-capability prediction; we build method-specific empirical relations from
  the initial model state and the compression setting, verify them on unseen settings and states, and deliver them
  with their ranges.
2 Experimental Setup — 2.1 the capability-conditioned loss and its three stated boundaries; 2.2 panels (heterogeneous
  = phenomena + transfer limits; controlled Pythia = law development; new states/configs = validation) and the three
  operations (global magnitude pruning; per-channel and grouped RTN; teacher-trace LoRA SFT of an existing student);
  2.3 development/held-out design, the two kinds of unseen target, MAE per capability, baseline classes, no target
  calibration, retrospective strongest-baseline marking.
3 Developing Capability-Conditioned Scaling Laws — 3.1 why the initial state: the dense capability anchor L0c carries
  the signal, D0 adds little beyond it; 3.2 pruning: shared power form (5 params) and per-density regression tie on
  development, the continuous form is delivered; 3.3 quantization: per-bit source regression at fixed b (no continuous
  bit law), and the two-axis (b, g) low-order form from the grouped-RTN development set; 3.4 distillation: delta vs
  initial student, source-axis linear form (math) and reuse-count forms on the pool axis, with the new descriptor forms
  as development-only candidates.
4 Generalization to Unseen Models and Configurations — 4.1 unseen settings: densities (pooled unseen d; three new
  checkpoints between grid points), unseen bit/group configs, unseen pool; 4.2 transfer across sizes/stages: 96k stage,
  1B in-range, 6.9B five-times, protocol A/B; 4.3 across capabilities: math/code carried by source inputs, QA not;
  capability-specific shapes needed (v56 Part B).
5 Related Work — three paragraphs (law variables/targets; task-conditioned evaluation and task-stratified laws; empirical
  law fitting and generalization).
6 Discussion and Conclusion — what can be predicted now, what cannot, what the results say about further development;
  limitations kept.
Appendix: A Experimental details; B Fitted models and coefficients; C Alternative forms and model selection; D Complete
  prediction results; E Capability-loss measurement checks; F Heterogeneous-model results; G Mechanistic analyses;
  H Recovery and cross-method analyses; I Reproducibility and prediction registration.

## Old -> new mapping
| new location | old TeX / asset | scientific question | evidence + script | action |
|---|---|---|---|---|
| 1 Intro | intro.tex | motivation, problem, design, findings | — | rewrite (5 paragraphs, 3 contributions) |
| 2.1 | prelim.tex §3.3 | what is measured | v26/v19 checks (App. E) | keep core, shorten |
| 2.2 | method.tex §4.1 + Table cohorts | which panels for which role | v51 counts | compact table in main; lists to App. A |
| 2.3 | method.tex §4.3 + prelim §3.2 budgets | how fits are judged | v52 origin codes | short; budget table + codes to App. I |
| 3.1 | experiment.tex §5.2 source axis (v36b/v38/v39 D0 and L0 increments) | do initial-state inputs help | v36b input_pairs, v39, Fig 3 panel A | main paragraph + Fig 3 |
| 3.2 | method §4.2 + experiment §5.2 + round-3 paragraph | pruning form and fit | v40/v42, v53 register (App. B coefficients) | main: form, params, dev comparison |
| 3.3 | experiment §5.3 + group-size paragraph + appendix quantizer | quantization forms | v38/v44 per-bit; v55 register | main: per-bit source regression + (b,g) 2D form |
| 3.4 | experiment §5.4 + C37 | distillation forms | v39, v41, v56 | main: delta definition, T/D_U/E, forms, dev result |
| 4.1 | experiment §5.2/5.3/5.4 config-axis + C35/C36 | unseen settings | v42, v46, v49, v53, v55, v41 | main Table 2 blocks + Fig 2 |
| 4.2 | experiment §5.2 new sources; C33/C34/C35 | unseen states/sizes/stages | v38, v46, v49, v53 | main |
| 4.3 | §5.5 inputs/forms + C37 Part B | capability conditioning | v56 | main (short) |
| 5 | related.tex | — | RELATED_WORK_VERIFICATION.md | keep verified claims, compress to 3 paragraphs |
| 6 | con.tex + §5.5 limitations | deliverables and boundaries | — | rewrite |
| App. A | appendix A.1–A.3, cohorts table, LoRA hyperparameters | — | — | move |
| App. B | round3_coef + v55/v50/v40 coefficients | — | new generator v58 | new coefficient tables |
| App. C | v53 LOSO table, v55 LOSO, v56 Part A/B | model selection | v53/v55/v56 | new |
| App. D | pred_source/pred_config_prune/pred_config_qd/p1v2/round3 tables, gains figure | complete results | v52/v49/v57 | move |
| App. E | appendix A.1 checks + measurement figure | loss validity | v26/v19 | move |
| App. F | panel_prune/panel_quant + heterogeneous prose + family structure | phenomena | v51 | move |
| App. G | geometry blocks, pruning mechanism law, signed term, cliff | mechanism | v9/v14 | move, shorten |
| App. H | recovery, resource-ratio (v17), selection replay | — | — | move |
| App. I | registration, freeze commits, origin codes, reproduction | — | MANUSCRIPT_REWRITE_MAP, CLAIM matrix | move |

## Figures and tables (main)
- Fig 1 (responses): keep plot_fig1_responses (measured vs frozen predictions per arm); caption states dev-fit vs frozen.
- Fig 2 (generalization): transfer_limits matrix + distillation pool panel; add the three new checkpoints (C35) and
  grouped-RTN tests (C36) as rows/panels -> regenerate (Stage C).
- Fig 3 (inputs and forms): compact two-panel version of input_form_gain (A: full vs no-D0 / adding L0; C: power vs
  A2/A1 same-input), plus a small capability-conditioning panel from v56 Part B -> regenerate (Stage C).
- Table 1: predictive models and their domains (form, inputs, varied config, target-calibration count, test range,
  coefficient appendix) -> new generator v58.
- Table 2: held-out performance by test type (candidate MAE per capability, strongest relevant baseline name + MAE,
  test origin) -> from v52 groups + v53/v55 (Stage C).

## Structural principles adopted from the reference papers (writing note)
- Distillation Scaling Laws: "why this model -> functional form -> parametric fit -> independent prediction"; Fig 1 shows
  predictions early; profiles/coefficients/implementation in appendix.
- Scaling Laws for Precision: unified notation first, one subsection per intervention, formula next to the response
  figure; alternatives and numerical coefficients in appendix.
- P2 Law: development of the relation and its generalization tests are separate sections; failures affecting transfer
  stay in the main text; we use "developing/functional form", not "derivation".
- Pruning Laws: give the form early, then validate its range; acknowledge its task strata and hold-out tests.
