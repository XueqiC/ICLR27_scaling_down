# Closeout writing plan: what the main text will show, and how each piece is built

Drafted 2026-09-15 for approval before any `.tex` edit. Follows the advisor's closeout note: the
main text answers research questions with final evidence; the audits, failed candidates and Taylor
details go to the appendix; every headline is bounded by a stated scope.

## The organising question

> Within which inputs, compression configurations and evaluation scopes can a parsimonious relation
> predict the capability-conditioned loss, and which plausible structures fail under independent test?

Three findings carry the paper, each answering it directly:

1. **Within-configuration prediction and cross-source transfer are different problems.** Pruning's
   compact form works inside a stated range; quantization's configuration median and interpolation
   are strong; extra source descriptors do not stably improve transfer within the inputs and models
   tested. Scope, not "source information is useless".
2. **The distillation data response depends on student and evaluation distribution.** The reuse cost
   reproduces on three QA distributions; the net gain is 2Wiki-only; budget and data responses are
   not jointly predicted by any of the parsimonious forms tried.
3. **Curve fit, structural interpretation and decision validity need separate verification.** A
   rejected log form is not a rejected additive class; an accurate second-order displacement account
   is not a pre-compression predictor; a low-regret selector is not thereby better than quant-only
   across references.

## Deliverable 1: the final prediction table (main text)

Replaces `tables/final_deliverables.tex`. One row per arm; columns fixed as: delivered predictor,
inputs required, applicable domain (states, configurations, capabilities), independent test (what
was frozen, on what), gain over the **development-stage** strongest baseline with interval, and a
separate column for development-only evidence so development and independent confirmation never
share a cell. Generator: extend `analysis/v85_selection_decomp.py`'s table writer or add
`analysis/final_prediction_table.py` reading only frozen artifacts (v69, v70, v78, v93, a2, a5) and
the audit's gate table. Numbers enter from JSON, never typed.

## Deliverable 2: the generalization figure (main text)

New `figs/generalization.pdf`, three panels with one shared visual grammar (success = filled,
failure = hollow, each point a frozen prediction with its measured error bar):
- **new configurations** (pruning densities inside/outside range; quantization group sizes; the
  distillation corner budgets),
- **new students or sources** (Pythia held-out states; the 4B as development student, labelled so;
  external models),
- **new evaluation distributions** (2Wiki fresh sample, MuSiQue, TriviaQA).
Generator: `analysis/plot_fig_generalization.py` built from `plot_fig2_confirm.py` and
`plot_fig3_transfer.py` inputs plus v99 scope and a5. Replaces `frozen_candidates.pdf` in
`general.tex`; `transfer_limits.pdf` moves to the appendix.

## Deliverable 3: the distillation controlled-response figure (main text)

New `figs/distill_response.pdf`: left, student × reuse at the matched ~200k budget for math, code, QA
(the C67 table as a figure, both seeds shown); right, the three QA distributions against reuse for
each student (the C72 table). The corner test becomes one small inset or an appendix panel with the
per-student prediction / measurement / error table from the audit, captioned as "failed to reject".
Generator: `analysis/plot_fig_distill_response.py` from the A1 table and v99 scope.

## Deliverable 4: the selection paragraph (main text, `selection.tex`)

Keep the prospective result on four fresh states and the per-reference analysis. Add one sentence
that the planned independent cross-family selection validation with fresh candidates was not
completed, and remove any phrasing that lets one historical KD candidate carry a general
practicality claim. No new figure.

## Abstract and introduction

Rewrite the promise in both to the organising question and the three findings; state ranges in the
same sentence as each claim; drop the phrase-level inventory of forms from the abstract. The
contribution list becomes: (1) the prediction problem, admitted inputs and the
development-then-validation design; (2) per-arm relations with their tested ranges and the
structures that failed under independent test; (3) the selection connection with its prospective
result and its stated limit.

## Appendix moves

Parameter accounting, the gate audit, the seed-pair comparison, all failed candidates, the corner
design and the Taylor expansion details move to the appendix, each as a short subsection with its
artifact named.

## Order of work

1. Audit results land (A7, A8) → fix any number the tables depend on.
2. Generators for deliverables 1–3 (codex), reviewed against the ledger.
3. `.tex` prose edits listed by the mirror refresh, then selection paragraph, then abstract and intro.
4. Compile, nine-page check, de-AI pass, full contribution review, push at the milestone.
