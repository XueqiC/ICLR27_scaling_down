# Restructure map: the paper as one systematic study of predictive scaling-down laws

Directive of 2026-09-15 (user): present the whole paper as a complete systematic study around
developing predictive scaling-down laws, showing what was learned, why the regularities appear, and
how that knowledge serves later research and practice. This map fixes the section order, what each
section must show, which evidence carries it, and where every existing piece moves. Prose is written
by me; generators by codex; every number enters from a frozen artifact.

## Position, kept

Pruning, quantization and distillation are parallel arms. The shared endpoint is the loss conditioned
on capability and evaluation distribution, relative to a reference measured beforehand. Forms may
differ per arm; the research question and the validation standard are shared.

## Narrative order, applied to every results section

observed → which conditions shape it → what evidence explains the variation → what relation can be
built → where it applies. Results are described by effect size, error, uncertainty and range, not by
success or failure. Each results subsection ends with one paragraph of scientific meaning or practical
lesson tied to named evidence.

## Section outline (main text, nine pages)

1. **Introduction.** The problem: capability-specific workloads served by scaled-down reference
   models; compression damages capabilities unevenly (twelve-model heterogeneity, App. H). The
   question: under which inputs, configurations and evaluation scopes can a parsimonious relation
   predict the conditional loss, and which plausible structures fail under independent test. The
   three contribution classes, each with the evidence that backs it.

2. **Problem and measurement.** Endpoint, admitted inputs (K0 state + configuration; K1 as costed
   auxiliary), the controlled Pythia panel, the Gemma distillation panel, the information-budget
   ladder, and the validation standard: development inside folds, frozen predictions before new
   measurements, pre-registered verdicts kept as written.

3. **What compression does to capabilities.** The observations first. Damage magnitude and its
   ordering across capabilities per arm; the size and stage dependence on the Pythia panel; the
   protocol dependence of distillation damage (learning-rate pilot); the reuse cost, monotone and
   size-scaling, and its distribution scope (three QA distributions). Closing paragraph: the
   observations that any law must reproduce.

4. **Why the responses vary: conditions and mechanism.** The explanatory section, promoted from the
   appendix. (a) Function shape: the pruning power form and its range; the quantization threshold
   as a geometric crossing; the reuse term near-linear rather than logarithmic (A2 profile) and the
   corner test that failed to reject additivity on QA while the larger student diverged on math and
   code. (b) Source state: what size and stage carry, and where source descriptors stop helping
   (V92, A4 closed). (c) Training protocol and data reuse: rate, budget, pool. (d) Evaluation
   distribution: cost transfers, gain does not. (e) Mechanism: the exact logit-displacement identity,
   the shift-invariant projection, the second-order account with its boundary in damage magnitude
   across three families; and the explicit statement that this account is post-hoc, not a
   pre-compression predictor. Closing: verified causes / excluded explanations / unidentified
   factors, as three short lists.

5. **Predictive relations per arm.** For each arm: relation, inputs, fitted range, error against the
   development-stage strongest baseline, and the structures that were tested and did not survive.
   The main table lives here (spec below). Closing: what a practitioner can compute today with K0
   inputs, per arm, and its stated error.

6. **Generalization tests.** New configurations versus new sources or students, kept as separate
   panels; the fresh-distribution scope shown separately; development results, frozen predictions and
   post-hoc recommendations labelled distinctly. The corner test appears as one small panel captioned
   "failed to reject". Closing: the range each relation has earned.

7. **Decision use.** The selection problem, its retrospective and prospective results, per-reference
   regret, and the stated limit that the cross-family validation with fresh candidates was not
   completed. Closing: which decisions the evidence supports and at what calibration cost.

8. **What was learned and what it enables.** Replaces the nine-line discussion. Three contribution
   classes with their evidence: (i) responses and relations; (ii) scientific understanding gained in
   building reliable laws (why the log form failed, why source descriptors did not transfer, why
   QA and code behave differently, what identifiability demands); (iii) practical guidance for
   experiment design and configuration choice (choose inputs; design rectangles by construction;
   fix protocol before measuring damage; budget calibration; state prediction range). Concrete,
   verifiable, and bounded where prediction or decision is not yet validated. One paragraph of
   limitations and the conditions under which experiments would reopen.

Related work stays one section, moved after §2 or kept before §8 depending on page fit.

## The main table (§5), spec

Rows: method × prediction task (pruning: unseen density / new source state; quantization: unseen
bit-width / unseen group size / new state; distillation: budget response / data-reuse response /
new pool). Columns: relation (form, parameter count), inputs required, tested range (states,
configurations, capabilities, distributions), error with interval, comparison against the
development-stage strongest baseline, and a status column with exactly three values: development,
frozen prediction, post-hoc recommendation. Generator `analysis/final_prediction_table.py` reads only
frozen artifacts (v69, v70, v78, v93, results/a2-curvature-interaction, a5, a7) and writes
`paper/paper/tables/main_prediction_v2.tex`; it replaces `final_deliverables.tex` and
`main_prediction.tex`.

## Figures (main text), spec

- **Fig 1, responses observed** (§3): student × reuse for math/code/QA at the matched budget, both
  seeds; the three QA distributions against reuse. Generator `analysis/plot_fig_responses_v2.py`
  from the A1 table and v99 scope.
- **Fig 2, explanation** (§4): the reuse-term profile (p across folds) beside the corner-test panel
  with per-student prediction / measurement / noise; the displacement-account boundary across
  families. Generator `analysis/plot_fig_explanation.py` from A2, A5, A7, V88.
- **Fig 3, generalization** (§6): panel A new configurations, panel B new sources or students,
  panel C new evaluation distributions; filled = frozen prediction within its stated error, hollow =
  outside; every point carries its error bar; development points drawn differently from frozen ones.
  Generator `analysis/plot_fig_generalization.py` from v69, v70, v78, v93, v99, a5.
- Selection map stays as is (§7).

## Appendix moves

Full candidate lists and forms; derivations (Taylor identity, projection); the gate audit, parameter
accounting, seed pair and corner design (A7); all pre-registered verdicts verbatim; engineering
details; heterogeneous-model tables; recovery arm; resource-ratio models.

## Order of work

1. Generators (codex) → tables and figures regenerated; I check every number against the ledger.
2. Prose: §3, §4 (new), §5, §6, §7 closing paragraphs, §8 (new), then §1 and abstract.
3. Compile with tectonic, nine-page fit, de-AI pass, contribution review, push at the milestone.
