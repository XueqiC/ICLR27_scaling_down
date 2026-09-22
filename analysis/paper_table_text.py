"""Publication wording for generated tables, independent of numerical analysis.

Only table environments are edited. Comments, citations, labels, source paths,
and mathematical expressions are protected from general language substitutions.
The versioned generators keep their original data keys and numerical formatting.
"""
from functools import wraps
import re
import textwrap


TABLE = re.compile(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", re.S)
PROTECTED = re.compile(
    r"(?m)^%[^\n]*|\\(?:label|ref|eqref|cite|texttt)\{[^}]*\}|\$(?:\\.|[^$])*\$"
)

# Definitions belong to each table, even when an earlier table defines a term.
CAPTION_NOTES = {
    "tab:main-prediction-v2": "Each block names a compression method and task; the left column identifies the field and the right column gives its value. Within each error row, capabilities appear in the order Math, Code, and QA. Paired student errors follow the displayed student order.",
    "tab:models": "Rows identify prediction families; columns give predictors, parameter counts per capability, development settings, target calibration counts, tested ranges, and coefficient sources. Loss responses are in nats per native token; density and reuse are dimensionless, bit widths are in bits, and budgets are in tokens. This is a post-hoc summary of development fits and frozen prediction tests.",
    "tab:model_arch": "Rows identify models within each cohort; columns give the architecture class and parameter-count conventions. This is a post-hoc architecture audit. MoE means mixture of experts.",
    "tab:round3_coef": "Rows identify capabilities; columns give the fitted exponent and coefficients of the intercept, source size, initial loss, and pretraining tokens, respectively. These development coefficients were frozen for prediction. Loss is in nats per native token; standardized covariates and the exponent are dimensionless.",
    "tab:quant2d_coef": "Rows identify capabilities and basis terms; columns give coefficients of the intercept, source size, initial loss, and pretraining tokens. These development coefficients were frozen for prediction. Loss is in nats per native token, source size is a parameter count, and pretraining exposure is in tokens; the fitted covariates are standardized.",
    "tab:v53_loso": "Rows identify density subsets and candidate predictors; columns give mean absolute error by capability and its capability mean. A1 denotes the linear power form, and A2 denotes per-density regression with interpolation.",
    "tab:v55_loso": "Rows identify candidate predictors and columns identify capabilities. All entries are development mean absolute errors.",
    "tab:quant_ident": "Rows identify control forms and evaluation panels; capability columns report effective degrees of freedom above and mean absolute errors below.",
    "tab:v56_forms": "Rows identify forms and columns give parameter counts and capability errors under each development holdout. Each paired entry gives mean absolute error followed by signed bias. The descriptor suffix L0 means initial loss and logN means logarithmic student size.",
    "tab:v56_condition": "Rows identify response structures and capabilities; columns compare parameter counts, errors, gains, and the stated gain threshold in this development analysis. F2 denotes a saturating budget response plus a logarithmic reuse response, with a student descriptor. Paired parameter counts give the shared and capability-specific totals, respectively.",
    "tab:distill_forms_audit": "This is a post-hoc audit of frozen prediction forms. Loss responses and coefficient-scaled responses are in nats per native token. F1 denotes descriptor-modulated reuse; F2 adds a saturating budget term. L0 denotes initial loss and logN denotes logarithmic student size.",
    "tab:shared_structure": "Rows identify response families, sharing comparisons, and calibration evaluations; columns describe shared and varying terms and compare errors and interval summaries. This is a post-hoc analysis. K0 denotes no compressed-target calibration and K1 denotes one compressed-target calibration measurement.",
    "tab:cap_conditioning": "Rows identify compression families and capabilities; columns compare the defined response variants and their paired error reduction. This is a post-hoc ablation using development fits on frozen confirmation panels.",
    "tab:cond_audit": "Rows identify compression families above and capabilities below; columns give dimensionless scales and squared correlations above, and error comparisons below. This is a post-hoc audit. A denotes a separate response per capability; B denotes a shared response with a fitted capability scale.",
    "tab:main_final": "Rows identify frozen candidates and test panels; columns give candidate errors, the strongest frozen alternative, error differences, delivered rules, and their timing and scope. F1 denotes descriptor-modulated reuse and F2 denotes the saturating budget and reuse form; L0 denotes initial loss. Pool sizes count traces per domain and budgets count supervised tokens.",
    "tab:main_context": "Rows identify student and pool tests; columns give candidate errors, the strongest frozen alternative, error differences, delivered rules, and their timing and scope. Pool sizes count traces per domain and budgets count supervised tokens. These are frozen predictions with post-hoc ranking and rule selection.",
    "tab:pred_full": "Rows identify prediction tests; columns identify the held-out axis, the defined provenance code, and capability errors. Each error pair lists the candidate followed by the strongest baseline.",
    "tab:pred_source": "Rows identify tests and predictors; columns report capability errors. LoRA denotes low-rank adaptation.",
    "tab:pred_config_prune": "Rows identify frozen prediction tests and predictors; columns give mean absolute errors in nats per native token by capability. Parentheses give baseline-minus-candidate improvement; bold marks the lowest error. The provenance letters mean: P, prediction frozen before measurement; F, baseline specified before measurement; R, post-hoc baseline; A, all specified forms reported. A1 denotes the linear power form and A2 denotes per-density regression with interpolation.",
    "tab:pred_config_qd": "Rows identify frozen prediction tests and predictors; columns give mean absolute errors in nats per native token by capability. Parentheses give baseline-minus-candidate improvement; bold marks the lowest error. The provenance letters mean: P, prediction frozen before measurement; F, baseline specified before measurement; R, post-hoc selection; A, all specified forms reported. Pool sizes count traces per domain.",
    "tab:p1v2": "Rows identify source states, protocols, and compression regimes; columns give candidate errors and the predictor with the lowest observed error. A1 denotes the linear power form and A2 denotes per-density regression with interpolation. The best-predictor column is a post-hoc ranking.",
    "tab:round3_prune": "Rows identify frozen predictors; columns give parameter counts and capability errors in nats per native token.",
    "tab:round3_quant": "Rows identify frozen predictors; columns give parameter counts and capability mean absolute errors for each test panel.",
    "tab:quant_threeway": "Within each panel, rows identify predictor status and form; columns give capability mean absolute errors.",
    "tab:prune_repeat": "Rows identify frozen predictors; columns give mean absolute errors for each capability and their mean.",
    "tab:p2v2_test": "Rows identify students, pool roles, and capabilities; columns give errors for the development-selected form and baselines, the post-hoc best frozen form, and the number of test points. Pool sizes count traces per domain. R denotes post-hoc ranking.",
    "tab:distill_paired": "Rows identify students and capabilities within each pool group; columns give candidate and baseline mean absolute errors, paired differences, and relative improvements in percent. This is a post-hoc paired analysis of frozen predictions.",
    "tab:distill_confirm": "Rows identify students and capabilities; columns give selected and baseline forms, their mean absolute errors, and paired improvements. These are frozen prediction results. Pool size counts traces per domain and budgets count supervised tokens.",
    "tab:p3_check": "Rows identify measured compression states; columns give capability loss changes, with the primary benchmark followed by the secondary benchmark. This is a post-hoc measurement-scope check, with states specified before secondary measurement. Pool sizes count traces per domain.",
    "tab:musique_scope": "Rows identify students, pools, and checkpoints; columns give dimensionless reuse counts, trajectory counts, and benchmark loss changes. This is a post-hoc measurement-scope audit. Pool sizes count traces per domain.",
    "tab:qa_scope": "Rows identify compression states; paired columns give loss and change from dense for each benchmark. This is a post-hoc measurement-scope audit of pre-specified states.",
    "tab:panel_prune": "Rows identify models; columns give capability loss changes, the Math loss-increase threshold, QA rankings, and the minimum QA response. This is a descriptive development panel with prospective additions marked by daggers. A dash denotes an unmeasured setting.",
    "tab:panel_quant": "Rows identify models and series; columns give capability loss changes at the displayed bit widths and list all measured bit widths in bits. This is a descriptive development panel with prospective additions marked by daggers.",
    "tab:selection-feasible": "Rows identify policies and objectives; columns give feasibility coverage, mean regret, and oracle-method agreement. This is a post-hoc development evaluation. Own means the cells feasible for that policy; common means cells feasible for every policy.",
    "tab:rule-confirm": "Rows identify objectives and policies; columns give feasible-cell counts, mean regret, oracle-method agreement, and candidate-set coverage. Policy predictions were frozen before the new measurements; set coverage is a post-hoc diagnostic.",
    "tab:rule_decomp": "Rows identify objectives and policies; columns compare the stated source subsets and all states. This is a post-hoc decomposition of frozen selection predictions.",
    "tab:rule-confirm-by-state": "Rows identify source states and objectives; columns compare policy regret and selected-configuration counts. This is a post-hoc decomposition of frozen selection predictions.",
    "tab:rule-confirm-candidate-sizes": "Rows identify objectives and policies; columns give method-count distributions and oracle coverage. This is a post-hoc diagnostic of candidate sets based on frozen development errors.",
    "tab:candidate-coverage": "Rows identify source states; columns give availability by compression method and the minimum quantization storage ratio. Counts are numbers of configurations and storage ratios are dimensionless. This is a post-hoc inventory of the development selection panel.",
    "tab:locked_rule": "Rows identify compression families, capabilities and source-state status; the last column specifies the response predictor. Absolute losses are in nats per native token. This rule was frozen before selection confirmation. Fit keys P, Q, G, and K denote pruning, channel quantization, grouped quantization, and distillation development data, respectively.",
    "tab:final": "Rows identify compression families; columns give delivered predictors, inputs, prediction and rule-selection status, observed gains, and excluded settings. This is a post-hoc summary of frozen prediction evidence; A2 denotes per-density regression with interpolation.",
}

# Endings adjusted so that no caption note leaves a short last line on the pdfLaTeX build.
_NOTE_ENDING_FIXES = {
    "tab:pred_config_qd": ("Pool sizes count traces per domain.",
                           "Pool sizes count traces per domain and budgets count supervised tokens."),
}
for _label, (_old, _new) in _NOTE_ENDING_FIXES.items():
    assert CAPTION_NOTES[_label].count(_old) == 1, _label
    CAPTION_NOTES[_label] = CAPTION_NOTES[_label].replace(_old, _new)



# Final academic wording of captions. Applied after every other step, on the
# rendered caption text, so the search strings are the literal published words.
ACADEMIC_CAPTION_PATTERNS = [
    r" The table was generated by \\texttt\{[^}]*\}(?: from \\texttt\{[^}]*\})?\.",
    r" The register is \\texttt\{[^}]*\}\.",
]
ACADEMIC_CAPTION_REWRITES = [
    # duplicated abbreviation sentences (the note layer adds one of each)
    ("CI means confidence interval. ", ""),
    ("MAE means mean absolute error. ", ""),
    ("MAE means mean absolute error. LoRA denotes low-rank adaptation. ", "LoRA denotes low-rank adaptation. "),
    # internal experiment codes
    ("The V79 audit examines capability conditioning.", "The audit examines capability conditioning."),
    ("The table reports the independent V78 selection panel.", "The table reports the independent selection panel."),
    ("The table reports V78 regret by source state and objective.", "The table reports regret by source state and objective on the independent panel."),
    ("The table reports the V74 three-way quantization confirmation.", "The table reports the three-way quantization confirmation."),
    ("The delivered V55 predictions are unchanged.", "The delivered predictions are unchanged."),
    ("pooled V55 standardization", "pooled development standardization"),
    ("Median uses V55's per-configuration anchors and bilinear interpolation.", "The median baseline uses the per-configuration development anchors with bilinear interpolation."),
    ("using the delivered V53 register and the same available inputs", "using the delivered development register and the same available inputs"),
    ("A2 uses V53's fixed ridge anchor fits and linear interpolation.", "A2 uses the fixed ridge anchor fits of the development register with linear interpolation."),
    ("responses use each V6 pruning run's checked dense reference.", "responses use the checked dense reference of each pruning run."),
    ("distillation reuses V39 students.", "distillation reuses the earlier students."),
    ("TriviaQA uses V48's no-context control.", "TriviaQA uses the no-context control."),
    ("V50's frozen Gemma-3 student counts are", "The frozen Gemma-3 student counts are"),
    ("Pythia T is exactly the v53 and v64 law covariate", "Pythia T equals the law covariate"),
    ("The specific forms are v53 power, v55 low-order two-dimensional form, v56 F2 with $L_0$.", "The specific forms are the power form, the low-order two-dimensional form, and F2 with $L_0$."),
    ("The P1-v2 tests evaluate frozen predictions for new sources", "The new-source tests evaluate frozen predictions"),
    ("for the two P1-v2 pairs and the single-stage Pythia-1B at step 96k prospective;", "applies to the two new-source pairs and to the prospective single-stage Pythia-1B at step 96k;"),
    ("the early checkpoints of the P3 states", "the early checkpoints of the pre-specified states"),
    ("on Gemma-3-1B (P3 check).", "on Gemma-3-1B."),
    ("The table reports the Round-3 pruning confirmation.", "The table reports the third-round pruning confirmation."),
    ("The table reports the Round-3 grouped-quantization confirmation", "The table reports the third-round grouped-quantization confirmation"),
    ("The table specifies the exact locked selection rule in \\texttt{analysis/final\\_rule.py}.", "The table specifies the locked selection rule."),
    ("Pythia-410M@143k", "Pythia-410M at step 143k"),
    # equations-as-prose and colon lists
    ("bold = better)", "bold marks the lower error)"),
    ("bold = smallest in the group;", "bold marks the smallest error in the group;"),
    ("(positive = candidate better)", "(positive values favor the candidate)"),
    ("Gain = alternative minus candidate; negative values favor the alternative.", "The gain is the alternative error minus the candidate error; negative values favor the alternative."),
    ("Frozen development selection = response surface, development median, and zero change, respectively.", "The frozen development selection is the response surface for Math, the development median for Code, and zero change for QA."),
    ("Test = three new checkpoints", "The test uses three new checkpoints"),
    ("development set = 17 Pythia states", "the development set comprises 17 Pythia states"),
    ("Forms: power $(\\beta_c\\cdot\\phi)((1-d)/0.3)^{\\gamma_c}$; A2 = per-density ordinary least squares on $\\phi$ at five anchors with linear interpolation; A1 = power form with $\\gamma_c{=}1$; cont $=(\\beta_c\\cdot\\phi)s+(\\zeta_c\\cdot\\phi)s^2$, $s{=}1{-}d$; strength only $A((1-d)/0.3)^{\\gamma}$, the development median curve, and zero change are source-free.",
     "The forms are the power form $(\\beta_c\\cdot\\phi)((1-d)/0.3)^{\\gamma_c}$, A2 (per-density ordinary least squares on $\\phi$ at five anchors with linear interpolation), A1 (the power form with $\\gamma_c{=}1$), and the continuous two-term form $(\\beta_c\\cdot\\phi)s+(\\zeta_c\\cdot\\phi)s^2$ with $s{=}1{-}d$; the strength-only curve $A((1-d)/0.3)^{\\gamma}$, the development median curve, and zero change are source-free."),
    ("Gain of the selected form versus A2 (same input): Math -0.013, Code -0.014, QA +0.004; versus the strongest source-free curve: Math +0.034, Code -0.023, QA -0.494.",
     "The gain of the selected form over A2 with the same inputs is -0.013 for Math, -0.014 for Code, and +0.004 for QA; over the strongest source-free curve it is +0.034, -0.023, and -0.494."),
    ("Forms: low-order two-dimensional form $=\\phi\\cdot[1,u,v,uv,u^2]$ with $u=\\log_2 q_{\\max}$ (centred) and $v=\\log_2(g/128)$, one 4-vector of state coefficients per term; separable $=(\\beta_c\\cdot\\phi)\\,q_{\\max}^{-p_c}(g/128)^{q_c}$; same-input interpolation = bilinear between the four development configurations; median and mean = per-configuration statistics interpolated the same way (source-free).",
     "The low-order two-dimensional form is $\\phi\\cdot[1,u,v,uv,u^2]$ with $u=\\log_2 q_{\\max}$ (centred) and $v=\\log_2(g/128)$ and one 4-vector of state coefficients per term; the separable form is $(\\beta_c\\cdot\\phi)\\,q_{\\max}^{-p_c}(g/128)^{q_c}$; same-input interpolation is bilinear between the four development configurations; the median and mean baselines are per-configuration statistics interpolated in the same way and are source-free."),
    ("development = six Pythia states", "the development set comprises six Pythia states"),
    ("Bit test: unseen $b{=}4$ at $g\\in\\{64,256\\}$ on the development states; granularity test: unseen $g{=}128$ at $b\\in\\{3,4,5\\}$ on the development states; joint test: new state Pythia-1B at step 96k at $g{=}128$.",
     "The bit-width test holds out $b{=}4$ at $g\\in\\{64,256\\}$ on the development states; the group-size test holds out $g{=}128$ at $b\\in\\{3,4,5\\}$ on the development states; the joint test uses the new state Pythia-1B at step 96k at $g{=}128$."),
    ("Signed bias of the 2D form (Math, Code, and QA): bit +0.108/+0.060/+0.077; granularity +0.162/+0.137/+0.129; joint +0.345/+0.097/+0.250.",
     "The signed bias of the two-dimensional form for Math, Code, and QA is +0.108, +0.060, and +0.077 on the bit-width test, +0.162, +0.137, and +0.129 on the group-size test, and +0.345, +0.097, and +0.250 on the joint test."),
    ("($N_0$ = transformer weight-matrix parameters, $D_0$ = tokens)", "($N_0$ counts transformer weight-matrix parameters and $D_0$ counts tokens)"),
    ("$z$ = standardized $L_{0,c}$", "$z$ is the standardized $L_{0,c}$"),
    ("Forms: $t=", "The forms use $t="),
    ("fall back to zero.", "reduce to the zero-change prediction."),
    ("Scored completion tokens, primary then secondary: Math 13198; 260, Code 4516; 8393, QA 251; 328.", "The scored completion tokens, primary followed by secondary, are 13198 and 260 for Math, 4516 and 8393 for Code, and 251 and 328 for QA."),
    ("24/24 cells measured.", "All 24 cells were measured."),
    ("distillation labels denote nominal supervised-token budgets.", "Distillation labels denote nominal supervised-token budgets."),
    ("are within 0.01 nats of dense for every listed model and are omitted.", "are within 0.01 nats of dense for each listed model and are omitted."),
    ("$^{\\dagger}$prospective additions measured at four densities only.", "Daggers mark prospective additions measured at four densities only."),
    ("$^{\\dagger}$prospective additions.", "Daggers mark prospective additions."),
    ("``QA least and most damaged'' counts,", "the least- and most-damaged QA columns count,"),
    ("``strongest same-information baseline'' is fixed", "the strongest same-information baseline is fixed"),
    ("``student'' marks the \\mbox{held-out} Gemma-3-4B.", "the student marker denotes the \\mbox{held-out} Gemma-3-4B."),
    ("``unseen'' lists what the test holds out", "the unseen column lists what the test holds out"),
    ("(`all')", "(all rows)"),
    ("M: literal meta-device parameter count before restoring tied weights; U: unique parameters after restoring ties; T: transformer matrices, excluding embeddings, language-model head, norms and biases.",
     "M is the literal meta-device parameter count before restoring tied weights, U the unique parameter count after restoring ties, and T the transformer matrices, excluding embeddings, language-model head, norms and biases."),
    ("G3: Gemma3ForCausalLM; G4: Gemma4ForCausalLM; MG: MuseGlimmerTextModel plus language-model head; O3: Olmo3ForCausalLM; PN: GPTNeoXForCausalLM; Q3: Qwen3ForCausalLM.",
     "The class abbreviations are G3 for Gemma3ForCausalLM, G4 for Gemma4ForCausalLM, MG for MuseGlimmerTextModel with a language-model head, O3 for Olmo3ForCausalLM, PN for GPTNeoXForCausalLM, and Q3 for Qwen3ForCausalLM."),
    ("A: per-capability response; B: shared response with a fixed capability scale; C: shared response with a fixed capability offset; D: shared response only.",
     "Variant A is a per-capability response, B a shared response with a fixed capability scale, C a shared response with a fixed capability offset, and D a shared response only."),
    ("The frozen distillation confirmation evaluates $U=200$ and supervised $T=50,100,200$k.", "The frozen distillation confirmation evaluates the pool $U=200$ at supervised budgets $T=50,100,200$k."),
    ("(18 points, not 18 replicates)", "(18 points that are not independent replicates)"),
    ("seed 0; all budgets stay together.", "seed 0; all budgets of a pool are resampled together."),
    ("Intervals have limited resolution with so few trajectories.", "With so few trajectories the intervals have limited resolution."),
    ("among the round's other frozen predictors, a diagnostic minimum that selection favours; the registered comparison is primary.", "among the other frozen predictors of the round; this minimum is a diagnostic that favours the alternative, and the registered comparison is primary."),
    ("(hindsight ranking)", "(a retrospective ranking)"),
    ("Each compression family is summarized by its predictor, inputs, evidence status, gains, and scope. Errors in nats per token;", "Each compression family is summarized by its predictor, inputs, evidence status, gains, and scope. Errors are in nats per token;"),
    ("The table lists all prediction tests across the three arms (Errors are mean absolute errors (MAE) in nats per native token per capability, candidate followed by strongest baseline, bold marks the lower error).",
     "The table lists all prediction tests across the three arms. Errors are mean absolute errors in nats per native token per capability, candidate followed by strongest baseline; bold marks the lower error."),
    ("$E$-only form for the pool of 225 traces per domain distillation pool test;", "reuse-only form for the distillation test with the pool of 225 traces per domain;"),
    ("For pool of 375 traces per domain, the headline is", "For the pool of 375 traces per domain, the headline is"),
    ("Each pool of 375 traces per domain group has 12 points per capability;", "Each group with the pool of 375 traces per domain has 12 points per capability;"),
    ("The pool of 375 traces per domain headline is the frozen form", "The headline for the pool of 375 traces per domain is the frozen form"),
    ("named by Code: nD denotes the variant without pretraining tokens, med = per-configuration median or development median curve, 0 = zero change, so = strength-only curve, A1 = $\\gamma{=}1$ power form, A2 = per-density regression with fixed interpolation, ct = continuous two-term form, pbm and pba denote the per-bit development median and mean, respectively, c denotes a constant, TE denotes the joint budget and reuse form.",
     "named by code: nD denotes the variant without pretraining tokens, med the per-configuration median or development median curve, 0 zero change, so the strength-only curve, A1 the power form with $\\gamma{=}1$, A2 per-density regression with fixed interpolation, ct the continuous two-term form, pbm and pba the per-bit development median and mean, c a constant, and TE the joint budget and reuse form."),
    ("Origin Code, four letters:", "The origin code has four letters:"),
    ("Origin Code: prediction origin", "The origin code gives the prediction origin"),
    ("Format, bold, improvement in parentheses, and origin Code as in Table~\\ref{tab:pred_source}.", "The format, the bold marking, the improvement in parentheses, and the origin code follow Table~\\ref{tab:pred_source}."),
    ("Configuration axis, pruning: does a relation fit at seen densities and sources predict unseen densities, sizes and stages?", "The configuration-axis tests for pruning ask whether a relation fitted at seen densities and sources predicts unseen densities, sizes and stages."),
    ("Source axis: does the pre-compression source state predict the response at a fixed intervention setting?", "The source-axis tests ask whether the pre-compression source state predicts the response at a fixed intervention setting."),
    ("(joint with a source term $k\\log(N_S/N_{\\mathrm{ref}})$ for every capability; this selection rule was stated after the tests)", "(joint with a source term $k\\log(N_S/N_{\\mathrm{ref}})$ for each capability; this selection rule was stated after the tests)"),
    ("(joint with a source term for every capability); this selection rule was stated after the tests (R).", "(joint with a source term for each capability); this selection rule was stated after the tests (R)."),
    ("This is a retrospective quantization identifiability audit (R).", "The table reports a retrospective audit of quantization identifiability (R)."),
    ("Bottom: Errors are mean absolute errors (MAE) in nats on the frozen bit (12 cells/capability), granularity (18), and joint 1B at 96k (3) tests, plus development leave-one-state-out (leave-one-state-out, 24), with training-only standardization refit in each fold.",
     "The lower panel reports mean absolute errors in nats on the frozen bit-width test (12 cells per capability), the group-size test (18), and the joint test on Pythia-1B at step 96k (3), and under development leave-one-state-out evaluation (24), with training-only standardization refit in each fold."),
    ("\\textbf{D}: FROZEN development-LOSO selection (Math: response surface; Code: development median; QA: zero), including the 0.02-nat tie rule. F: every frozen candidate. \\textbf{R}: \\mbox{post-hoc} recommended rule (piecewise interpolation for Math and Code on development states; median for QA and for all capabilities on new states). R reuses frozen candidate predictions but its choice is retrospective, not a prospective selection or the test minimum.",
     "\\textbf{D} is the frozen development leave-one-state-out selection (the response surface for Math, the development median for Code, and zero change for QA), including the 0.02-nat tie rule; F lists all frozen candidates; \\textbf{R} is the \\mbox{post-hoc} recommended rule (piecewise interpolation for Math and Code on development states; the median for QA and for all capabilities on new states). R reuses frozen candidate predictions, and its choice is retrospective."),
    ("Development states tested: Pythia-410M at step 143k and Pythia-1.4B at step 16k; new state: Pythia-1.4B at step 112k.", "The development states tested are Pythia-410M at step 143k and Pythia-1.4B at step 16k; the new state is Pythia-1.4B at step 112k."),
    ("The selected reuse response is $a\\log(1+E)$ (Math and Code): one parameter, no intercept, and zero at zero budget.", "The selected reuse response for Math and Code is $a\\log(1+E)$; it has one parameter, no intercept, and is zero at zero budget."),
    ("Step 32k comprises 160M, 410M, and 1.4B: pruning and quantization candidates with newly measured outcomes.", "The step-32k subset comprises the 160M, 410M, and 1.4B states, whose pruning and quantization candidates have newly measured outcomes."),
    ("Multi minimizes", "The maximum objective minimizes"),
    ("Own and common refer to the subsets defined below.", "Own and common refer to the subsets defined at the end of the caption."),
    ("for every trajectory of the uniform protocol", "for each trajectory of the uniform protocol"),
    ("The original dense candidate is retained at budget 1.0 in every state.", "The original dense candidate is retained at budget 1.0 in each state."),
    ("which selected the power form over A2 (gap 0.015). Baselines do not select.", "which selected the power form over A2 (gap 0.015); baselines are not subject to selection."),
    ("so the development set does not rank the forms;", "so the development set does not rank the forms, and"),
    ("dominates every \\mbox{held-out} error", "dominates each \\mbox{held-out} error"),
    ("Both coverage columns describe the set, not the single choice,", "Both coverage columns describe the candidate set,"),
    # second pass: fragments that remained after the first rewrite
    ("$\\Delta L_{\\mathrm{math}}\\ge 1$; Models are grouped by series", "$\\Delta L_{\\mathrm{math}}\\ge 1$; models are grouped by series"),
    ("protocol B in Table~\\ref{tab:p1v2}.", "protocol B is reported in Table~\\ref{tab:p1v2}."),
    ("8 bits and 6 bits are within 0.01 nats", "Bit widths 8 and 6 are within 0.01 nats"),
    ("The provenance letters mean: P, prediction frozen before measurement; F, baseline specified before measurement; R, \\mbox{post-hoc} baseline; A, all specified forms reported.", "The provenance letters are P for a prediction frozen before measurement, F for a baseline specified before measurement, R for a \\mbox{post-hoc} baseline, and A for all specified forms reported."),
    ("The provenance letters mean: P, prediction frozen before measurement; F, baseline specified before measurement; R, \\mbox{post-hoc} selection; A, all specified forms reported.", "The provenance letters are P for a prediction frozen before measurement, F for a baseline specified before measurement, R for a \\mbox{post-hoc} selection, and A for all specified forms reported."),
    ("1B sources are in-range sizes and 6.9B is", "The 1B sources are in-range sizes and 6.9B is"),
    ("Each method column gives configuration count followed by", "Each method column gives the configuration count followed by"),
    (". the unseen column lists what the test holds out (size$\\uparrow$ = five times beyond the development range).", ". The unseen column lists what the test holds out (size$\\uparrow$ denotes five times beyond the development range)."),
    ("$^{*}$ = bootstrap 95\\% interval of that improvement excludes zero", "an asterisk marks an improvement whose bootstrap 95\\% interval excludes zero"),
    ("Coefficients in Appendix Table~\\ref{tab:round3_coef}.", "Coefficients are given in Appendix Table~\\ref{tab:round3_coef}."),
    ("Same inputs $\\phi$ as Table~\\ref{tab:round3_prune};", "The inputs $\\phi$ are those of Table~\\ref{tab:round3_prune};"),
    ("predictions committed before measurement; MAE and mean signed error (prediction minus observation) over nine cells per capability.", "with predictions committed before measurement; the table reports MAE and the mean signed error (prediction minus observation) over nine cells per capability."),
    ("No refitting, test-based selection or outcome censoring.", "There was no refitting, test-based selection, or outcome censoring."),
    ("Source-free median and zero ignore source covariates.", "The source-free median and zero-change predictors ignore source covariates."),
    ("selected by the Frozen selection rule across these budgets.", "selected by the frozen selection rule across these budgets."),
    ("Both coverage columns describe the candidate set, and report count out of 68 (percent).", "Both coverage columns describe the candidate set and report the count out of 68 with the percentage."),
    ("Quantization only pools per-channel and grouped candidates using frozen-rule predictions.", "The quantization-only policy pools per-channel and grouped candidates using frozen-rule predictions."),
    ("Differences are computed before rounding; negative favors the frozen rule.", "Differences are computed before rounding; negative values favor the frozen rule."),
    ("common means cells feasible for every policy.", "common means cells feasible for each policy."),
    ("Family comparisons use registered development splits: 17 pruning sources, six quantization states, 12 distillation runs.", "Family comparisons use the registered development splits of 17 pruning sources, six quantization states, and 12 distillation runs."),
    ("over all \\mbox{held-out} rows (all rows) and over the densities off the coarse grid", "over all \\mbox{held-out} rows (all densities) and over the densities off the coarse grid"),
    ("All candidates and baselines are reported on every test (", "All candidates and baselines are reported on each test ("),
    ("Errors are mean absolute errors (MAE) in nats (lower is better), equally weighting measured cells at $b\\in\\{3,4,5\\}$; $n$ is per capability.", "Errors are mean absolute errors in nats, with the measured cells at $b\\in\\{3,4,5\\}$ weighted equally; $n$ counts cells per capability."),
]


def _academic_caption(text):
    def body(match):
        line = match.group()
        for pattern in ACADEMIC_CAPTION_PATTERNS:
            line = re.sub(pattern, "", line)
        for before, after in ACADEMIC_CAPTION_REWRITES:
            line = line.replace(before, after)
        return line
    return re.sub(r"\\caption\*?\{[^\n]*", body, text)


def _replace_words(text):
    replacements = {
        "math/code/QA": "Math, Code, and QA", "math/code": "Math and Code",
        "QA/macro": "QA and the capability mean", "scales/offsets": "scales and offsets",
        "means/SDs": "means and standard deviations", "student/capability": "student and capability",
        "student/pool-role": "student and pool-role", "pool/budget": "pool and budget",
        "embedding/head": "embedding and output head", "Gemma/Muse": "Gemma and Muse",
        "registry/panel": "registry and panel", "new size/stage": "new size or stage",
        "nats/token": "nats per native token",
        "per-config": "per-configuration", "config-indicator": "configuration-indicator",
        "cand.": "candidate", "Cap.": "Capability", "cap.": "capability",
        "Prot.": "Protocol", "Coeff.": "Coefficients", "Target cal.": "Target calibration",
        "pred.": "prediction", "obs.": "observation", "Diff.": "Difference",
        "Rel.": "Relative improvement", "Base MAE": "Baseline MAE",
        "Int. penalized?": "Intercept penalized?", "Free ": "Free parameters ",
        "Class/type": "Class and type", "Conv.": "Convention",
        "str.-only": "strength only", "strength-only": "strength only",
        "median-curve": "development median curve", "per-bit-median": "per-bit development median",
        "per-bit-mean": "per-bit development mean", "T-only": "budget only",
        "E-only": "reuse only", "Bit-only": "Bit width only", "bit-only": "bit width only",
        "Bit+group additive": "Additive bit width and group size",
        "T+E": "Budget and reuse", "joint$+$src": "joint with student size",
        "constant$+$src": "constant with student size", "joint+src": "joint with student size",
        "Joint+src": "Joint with student size", "constant+src": "constant with student size",
        "Const.+src": "Constant with student size", "T+src": "budget with student size",
        "E+src": "reuse with student size", "surface:L0": "response surface with initial loss",
        "surface:logN": "response surface with student size", "F1:L0": "F1 with initial loss",
        "F1:logN": "F1 with student size", "F2:L0": "F2 with initial loss",
        "F2:logN": "F2 with student size", "F1/F2": "F1 and F2",
        "2-D surface": "two-dimensional response surface", "low-order 2D": "low-order two-dimensional form",
        "low order 2d": "low-order two-dimensional form", "2D term": "two-dimensional term",
        "Grouped quant.": "Grouped quantization", "Distill. context": "Distillation context",
        "Distill exposure": "Distillation exposure", "Dev LOSO": "Development leave-one-state-out",
        "Prune dev LOSO": "Pruning development leave-one-state-out",
        "Quant 2D dev LOSO": "Quantization surface development leave-one-state-out",
        "Quant 2D": "Quantization surface", "Quant sep": "Separable quantization",
        "bit\\_test": "bit-width test", "granularity\\_test": "group-size test",
        "joint\\_test": "joint test", "min/med/max": "minimum, median, maximum:",
        "coverage/full width": "coverage and full width", "OOF": "out-of-fold predictions",
        "Quantization-only": "Quantization only", "quant-only": "Quantization only",
        "prune-only": "Pruning only", "distill-only": "Distillation only",
        "All (max)": "Maximum across capabilities", "Multi (max)": "Maximum across capabilities",
        "Frozen minus quant.": "Frozen rule minus quantization only", "Step-32k": "Step 32k",
        "Dev-selected": "Selected on development data", "FROZEN development-selected": "Frozen development selection",
        "Post-test recommended rule": "Post-hoc recommended rule", "dev-selected": "selected on development data",
        "CHEAPEST": "Cheapest feasible configuration", "ORACLE": "Measured oracle",
        "MAP": "Predicted selection", "INFEASIBLE": "infeasible", "N/A": "not available", "n/a": "not applicable",
        "LoRA": "low-rank adaptation", "OLS": "ordinary least squares", "SDs": "standard deviations",
        "CIs": "confidence intervals", "RTN": "round-to-nearest quantization", "KD": "distillation",
        "Channel round-to-nearest quantization": "Per-channel round-to-nearest quantization",
        "dev.": "development", "Dev.": "Development", "dev": "development", "cfg": "configuration",
        "interp.": "interpolation", "extrap.": "extrapolation", "interp": "interpolation",
        "extrap": "extrapolation", "incl.": "including", "regr.": "regression",
        "min QA": "Minimum QA response", "math@0.5": "Math at density 0.5",
        "noD0": "without pretraining tokens", "no-D0": "without pretraining tokens",
        "D/O": "development ordinary least squares", "D/R": "development ridge regression",
        "LM head": "language-model head", "LOCO": "leave-one-run-out",
        "LOSO": "leave-one-state-out", "post hoc": "post-hoc",
    }
    # Longest match first prevents partial replacement of names such as per-config.
    pattern = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)) + r")(?![\w-])")
    text = pattern.sub(lambda m: replacements[m.group()], text)
    text = re.sub(r"\b(?:math|code|qa)\b", lambda m: {"math": "Math", "code": "Code", "qa": "QA"}[m.group()], text)
    text = re.sub(r"\b(?:macro|Macro)\b", "Capability mean", text)
    text = re.sub(r"\bU(\d+)\b", r"pool of \1 traces per domain", text)
    text = re.sub(r"\bint(\d+)\b", r"\1 bits", text)
    text = re.sub(r"\b(pythia|gemma3|gemma4|olmo3|muse)-(\d+(?:\.\d+)?)([mb])\b",
                  lambda m: {"pythia": "Pythia", "gemma3": "Gemma-3", "gemma4": "Gemma-4", "olmo3": "OLMo-3", "muse": "Muse"}[m[1]] + "-" + m[2] + m[3].upper(), text)
    text = re.sub(r"(?<![\w.-])(\d+(?:\.\d+)?)([mMbB])\s*@\s*(?:step)?(\d+k?)",
                  lambda m: "Pythia-" + m[1] + m[2].upper() + " at step " + m[3], text)
    text = text.replace("Pythia-Pythia-", "Pythia-")
    return text


def _language(text):
    parts = PROTECTED.split(text)
    matches = list(PROTECTED.finditer(text))
    return "".join(_replace_words(part) + (matches[i].group() if i < len(matches) else "") for i, part in enumerate(parts))


def _caption_addition(text, note):
    match = re.search(r"\\caption\*?\{", text)
    if not match:
        return re.sub(r"\\end\{table(\*?)\}", lambda m: r"\caption*{" + note + "}\n" + m.group(), text)
    if note in text:
        return text
    depth = 1
    end = match.end()
    while depth:
        if text[end] in "{}" and text[end - 1] != "\\":
            depth += 1 if text[end] == "{" else -1
        end += 1
    return text[:end - 1] + " " + note + text[end - 1:]


def _table(text, fallback_label=None):
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    label = next((label for label in labels if label in CAPTION_NOTES), fallback_label)
    if label is None:
        return text
    saved = []
    def protect(match):
        saved.append(match.group())
        return f"EDITORIALPROTECTEDTOKEN{len(saved) - 1}ENDTOKEN"
    text = re.sub(r"(?m)^%[^\n]*|\\(?:label|ref|eqref|cite|texttt)\{[^}]*\}", protect, text)
    text = _language(text)
    text = _specific(label, text)
    text = _caption_sentences(text)
    text = _caption_addition(text, CAPTION_NOTES[label])
    if re.search(r"\bMAE\b", text):
        text = _caption_addition(text, "MAE denotes mean absolute error.")
    if re.search(r"\bCI\b", text):
        text = _caption_addition(text, "CI denotes confidence interval.")
    text = _wrap_table_words(label, text)
    # Keep status phrases together without changing formulas or model names.
    text = re.sub(r"(?<!\\mbox\{)\b(?:post-hoc|held-out|pre-committed|pre-specified)\b",
                  lambda m: r"\mbox{" + m.group() + "}", text, flags=re.I)
    text = re.sub(r"EDITORIALPROTECTEDTOKEN(\d+)ENDTOKEN", lambda m: saved[int(m[1])], text)
    return _academic_caption(text)


# Final concise captions, keyed by table label: what the table shows, the unit, and the reading
# rule a reader needs. Applied last, after the wording layer; nothing outside the caption changes.
CONCISE_CAPTIONS = {
    "tab:models": (
        r"Predictive models and their domains. Rows are prediction families; columns give the predictor, "
        r"parameters per capability, development settings, target calibration count (zero throughout), tested "
        r"range, and coefficient source. All inputs are $\mathbf x=(N_0,D_0,L_{0,c})$; losses are in nats per "
        r"native token, bit widths in bits, and budgets in tokens. This is a \mbox{post-hoc} summary of "
        r"development fits and frozen prediction tests."),
    "tab:panel_prune": (
        r"Per-capability pruning damage on the heterogeneous panel, in nats per native token within each model. "
        r"$d^{*}_{\mathrm{math}}$ is the largest measured density with $\Delta L_{\mathrm{math}}\ge 1$; the QA "
        r"ranking columns count, over the pre-cliff densities, how often QA is the least and the most affected "
        r"capability; the last column gives the most negative QA response and its density. Daggers mark "
        r"prospective additions measured at four densities only; a dash marks an unmeasured setting."),
    "tab:panel_quant": (
        r"Per-capability loss change under per-output-channel symmetric round-to-nearest weight quantization, in nats "
        r"per native token, at the displayed bit widths; 8 and 6 bits lie within 0.01 nats of dense for each "
        r"model and are omitted. The last column lists all measured bit widths. Daggers mark prospective additions."),
    "tab:round3_coef": (
        r"Frozen coefficients of the selected pruning form $\widehat{\Delta L}_c=(\beta_c\cdot\phi)((1-d)/0.3)^{\gamma_c}$ "
        r"with $\phi=[1,z(\log N_0),z(L_{0,c}),z(\log D_0)]$, standardized with centers 20.248, 2.661, 25.614 and "
        r"scales 1.407, 1.691, 0.739 for $(\log N_0, L_{0,c}, \log D_0)$. Rows are capabilities; columns give the "
        r"exponent and the coefficients of the intercept, source size, initial loss, and pretraining tokens. Loss "
        r"is in nats per native token."),
    "tab:quant2d_coef": (
        r"Frozen coefficients of the grouped-quantization form $\widehat{\Delta L}_c=\sum_k \beta_{c,k}\cdot\phi(\mathbf x)\,t_k(b,g)$ "
        r"with $t_k\in\{1,u,v,uv,u^2\}$, $u=\log_2 q_{\max}(b)$ centred at the development mean, and "
        r"$v=\log_2(g/128)$; standardization uses centers 19.565, 2.746, 25.332 and scales 1.084, 1.623, 1.095 for "
        r"$(\log N_0, L_{0,c}, \log D_0)$, with ridge penalty $10^{-3}$ in the fit on standardized inputs. Rows are capabilities and basis terms; "
        r"columns give the coefficients of the intercept, source size, initial loss, and pretraining tokens."),
    "tab:shared_structure": (
        r"Shared structure and one-point calibration, in nats pooled equally over capabilities. Rows are response "
        r"families, parameter-sharing comparisons, and calibration evaluations; columns describe the shared and "
        r"varying terms and compare errors and prediction-interval summaries (coverage and full width for K0, then "
        r"K1). K0 uses no compressed-target measurement and K1 uses one per source, excluded from scoring. This is "
        r"a \mbox{post-hoc} analysis."),
    "tab:cap_conditioning": (
        r"Capability conditioning on the frozen confirmation panels (mean absolute error, nats). A is a "
        r"per-capability response, B a shared response with a fixed capability scale, C a shared response with a "
        r"fixed capability offset, and D a shared response only, all from the same development information. "
        r"Positive $\Delta_B=B-A$ favors A; brackets give 95\% paired cluster-bootstrap intervals. This is a "
        r"\mbox{post-hoc} ablation."),
    "tab:final": (
        r"Summary of each compression family: delivered predictor, inputs, prediction and rule-selection status, "
        r"observed gains, and excluded settings. Errors are in nats per token against the strongest "
        r"same-information baseline fixed from development before the test. A2 denotes per-density regression "
        r"with interpolation. This is a \mbox{post-hoc} summary of frozen prediction evidence."),
    "tab:pred_full": (
        r"\scriptsize All prediction tests across the three arms. Each error pair gives the mean absolute error in "
        r"nats per native token of the candidate followed by the strongest baseline, per capability; bold marks "
        r"the lower error. The unseen column names what the test holds out. The origin code gives the prediction origin, the same-input baseline origin, the "
        r"simple baseline origin (P frozen before measurement, R retrospective, F \mbox{pre-specified}, -- none), "
        r"and the selection (A all \mbox{pre-specified} forms reported, S chosen after the test). Baseline codes: "
        r"nD without pretraining tokens, med the per-configuration or development median, 0 zero change, so the "
        r"strength-only curve, A1 the power form with $\gamma{=}1$, A2 per-density regression with interpolation, "
        r"ct the continuous two-term form, pbm and pba the per-bit median and mean, c a constant, and TE the joint "
        r"budget and reuse form. Improvements, both baselines, and bootstrap intervals are in "
        r"Appendix Tables~\ref{tab:pred_source}--\ref{tab:pred_config_qd}."),
    "tab:distill_confirm": (
        r"Frozen distillation confirmation at $U=200$ and $T=50,100,200$k. Each student and "
        r"capability is scored on six sampled pools with three dependent checkpoints per pool. Columns give the "
        r"selected and baseline forms, chosen by development leave-one-run-out before confirmation, their mean "
        r"absolute errors in nats, and the paired improvement (baseline minus selected) with 95\% percentile "
        r"intervals from 5,000 paired pool-cluster bootstrap resamples."),
    "tab:locked_rule": (
        r"The locked selection rule, frozen before the selection confirmation. Rows are compression families, "
        r"capabilities and source-state status; the last column gives the response predictor. Each absolute "
        r"prediction adds that response to a dense anchor: the source dense loss for pruning and quantization, "
        r"and the initial student's dense loss for distillation. Every method is fitted on its own development "
        r"panel, listed below the table."),
    "tab:rule-confirm": (
        r"The independent selection panel. Rows are objectives and policies; columns give "
        r"mean regret in nats, agreement (the percentage of single choices matching the oracle method), and "
        r"set coverage (the percentage of heuristic candidate sets containing that method, a \mbox{post-hoc} "
        r"diagnostic). QA is restricted to 2Wiki; policy predictions were frozen before the new measurements."),
    "tab:model_arch": (
        r"Cached text-model architectures and parameter counts ($10^9$). M is the meta-device count before "
        r"restoring tied weights, U the unique count after restoring ties, and T the transformer matrices "
        r"excluding embeddings, language-model head, norms and biases; Pythia T equals the law covariate "
        r"$L(4h^2+2hm)$. Multimodal checkpoints use their text decoder only. The frozen Gemma-3 student counts are "
        r"0.435870336B, 1.301875840B, and 4.971331952B, the last including 0.419816304B non-text parameters. "
        r"Class abbreviations: G3 Gemma3ForCausalLM, G4 Gemma4ForCausalLM, MG MuseGlimmerTextModel, O3 "
        r"Olmo3ForCausalLM, PN GPTNeoXForCausalLM, and Q3 Qwen3ForCausalLM."),
    "tab:v53_loso": (
        r"Pruning development on 17 Pythia states: leave-one-source-out mean absolute error in nats over all "
        r"\mbox{held-out} densities and over the densities off the coarse grid (0.75, 0.65, 0.55). The "
        r"\mbox{pre-committed} rule selects the lowest capability mean among the four source-conditioned "
        r"candidates, resolving gaps below 0.02 nats toward the continuous zero-at-$d{=}1$ form with fewer "
        r"parameters, which selected the power form over A2 (gap 0.015 in mean absolute error). A1 is the linear power form and A2 "
        r"per-density regression with interpolation."),
    "tab:v55_loso": (
        r"Grouped quantization: leave-one-state-out mean absolute error in nats on the 24 development cells of "
        r"six states, by candidate predictor and capability. The late-stage 160M state, which collapses at 3 bits, "
        r"dominates each \mbox{held-out} error, so the development set does not rank the forms; all three "
        r"candidates and the baselines were frozen and tested (\S\ref{sec:unseen_settings})."),
    "tab:quant_ident": (
        r"Retrospective audit of quantization identifiability; the delivered predictions are unchanged. The four "
        r"forms use the same 24 development cells per capability, pooled standardization, and ridge "
        r"$\lambda=10^{-3}$. The upper panel reports rank and effective degrees of freedom; the delivered "
        r"20-coefficient design has rank 16 because two development bit levels make $u^2=(u_3+u_5)u-u_3u_5$. The "
        r"lower panel reports mean absolute errors in nats on the frozen bit-width test (12 cells per capability), "
        r"the group-size test (18), the joint test on Pythia-1B at step 96k (3), and development "
        r"leave-one-state-out evaluation (24)."),
    "tab:v56_forms": (
        r"Distillation development matrix (12 runs, 48 points): \mbox{held-out} mean absolute error followed by "
        r"signed bias, in nats, per form and capability under leave-one-run-out and leave-one-student-out "
        r"evaluation. Forms use $t=\log(1+T_c/35000)$, $e=\log(1+E)$, and $s=1-e^{-T_c/T_*}$: budget only $a+bt$, "
        r"reuse only $a+be$, surface $a+bt+ce+dz$, F1 $(a+\lambda z)e$, and F2 $(a+\lambda z)s+(b+\mu z)e$, with "
        r"$z$ the standardized $L_{0,c}$ or $\log N_S$. Under leave-one-student-out the descriptor forms reduce to "
        r"zero change."),
    "tab:v56_condition": (
        r"Capability conditioning at matched parameter count (leave-one-run-out mean absolute error, nats): a "
        r"shared response shape with per-capability offsets against per-capability models with the same total "
        r"count. Paired parameter counts give the shared and capability-specific totals; a gain above 0.02 nats is "
        r"a descriptive threshold for comparing the fitted responses. F2 is the saturating budget response plus a logarithmic reuse response with a "
        r"student descriptor."),
    "tab:distill_forms_audit": (
        r"Audit of the sixteen frozen distillation forms for the response $\Delta L=L_{\rm post}-L_0$ in nats per "
        r"native token. Panel A specifies each form and its fit; panel B gives the frozen raw-basis coefficients by "
        r"capability. The selected reuse response for Math and Code is $a\log(1+E)$, with one parameter and no "
        r"intercept; the reuse-only baseline $c+b\log(1+E)$ has an unpenalized intercept. F1 denotes "
        r"descriptor-modulated reuse and F2 adds a saturating budget term."),
    "tab:cond_audit": (
        r"Audit of capability conditioning, where A fits a separate response per capability and B a shared "
        r"response with a fitted, unrestricted signed capability scale. The upper panel gives the dimensionless "
        r"scales and squared correlations; the lower panel gives the error comparison, with positive $\Delta=B-A$ "
        r"the reduction in mean absolute error from A, in nats. The pruning confirmation counts the two identical "
        r"Pythia-2.8B revisions as one state. This is a \mbox{post-hoc} audit."),
    "tab:p3_check": (
        r"Primary and secondary benchmarks on Gemma-3-1B: loss change from dense in nats per native token on the "
        r"main-protocol probe followed by an independent benchmark of the same capability (Math: SVAMP; Code: "
        r"HumanEval; QA: TriviaQA; 128 samples, fixed probe seed). The six states were fixed before any secondary "
        r"measurement. Scored completion tokens, primary then secondary, are 13198 and 260 for Math, 4516 and 8393 "
        r"for Code, and 251 and 328 for QA."),
    "tab:musique_scope": (
        r"Distillation QA responses on the primary 2Wiki probe and on 128 answerable MuSiQue development questions "
        r"with supporting paragraphs supplied, as changes in conditional loss from the dense loss of the student in "
        r"nats per native token, for each trajectory of the uniform protocol at its final checkpoint (three data "
        r"seeds per pool size) and for the early checkpoints of the \mbox{pre-specified} states. Entries give the "
        r"mean over trajectories with the range in brackets."),
    "tab:qa_scope": (
        r"QA measurement scope on Gemma-3-1B: conditional loss and $\Delta L$ from dense in native-token nats "
        r"(negative is improvement) with 256 fixed references per set. 2Wiki supplies all context, MuSiQue supplies "
        r"supporting paragraphs, and TriviaQA is the no-context control; distillation labels give nominal "
        r"supervised-token budgets. All 24 cells of the \mbox{pre-specified} states were measured."),
    "tab:main_final": (
        r"\scriptsize Frozen candidates against delivered rules. Errors are mean absolute errors in nats per token, "
        r"stacked as Math, Code, and QA. The strongest alternative is the frozen predictor of the round with the "
        r"lowest pooled test error, a diagnostic that favours the alternative; the gain is the alternative error "
        r"minus the candidate error, so negative values favor the alternative. Rules fixed after a test are "
        r"retrospective for it. F1 denotes descriptor-modulated reuse, F2 the saturating budget and reuse form, and "
        r"L0 the initial loss."),
    "tab:pred_source": (
        r"Source-axis tests: whether the pre-compression source state predicts the response at a fixed "
        r"intervention setting. Entries are mean absolute errors in nats per native token per capability; bold "
        r"marks the smallest error in the group; parentheses give the improvement of the candidate over the "
        r"strongest listed baseline, with an asterisk where the bootstrap 95\% interval excludes zero. The origin "
        r"code gives the prediction origin, the same-input baseline origin, the simple baseline origin, and the "
        r"selection (P frozen before measurement, R retrospective, F \mbox{pre-specified}, -- none; A all forms "
        r"reported, S chosen after the test)."),
    "tab:pred_config_prune": (
        r"Configuration-axis tests for pruning: whether a relation fitted at seen densities and sources predicts "
        r"unseen densities, sizes and stages. Protocol A (full development panel) applies to the two new-source "
        r"pairs and the prospective Pythia-1B at step 96k; protocol B is in Table~\ref{tab:p1v2}. Entries are mean "
        r"absolute errors in nats per native token by capability; bold marks the lowest error, parentheses give "
        r"the baseline-minus-candidate improvement, and the origin code follows Table~\ref{tab:pred_source}. A1 is "
        r"the linear power form and A2 per-density regression with interpolation."),
    "tab:pred_config_qd": (
        r"Configuration-axis tests for quantization and distillation: unseen sizes and stages at seen bit-widths "
        r"(protocol A; the 5-bit prediction is a fixed interpolation rule) and unseen distillation pools. For the "
        r"pool of 375 traces per domain the headline is the frozen form with the lowest per-capability development "
        r"error, a rule stated after the tests (R), with 12 points per capability and Gemma-3-4B \mbox{held-out}. "
        r"Entries, bold marking, parentheses, and the origin code follow Table~\ref{tab:pred_source}."),
    "tab:p1v2": (
        r"New-source tests of frozen predictions: mean absolute error over three capabilities in nats per native "
        r"token. Protocol A fits on the full nine-state development panel; protocol B fits on steps $\le$64k only, "
        r"so its 112k rows are training-stage extrapolations. The 1B sources are in-range sizes and 6.9B is a "
        r"$\sim$5$\times$ size extrapolation. Columns give candidate errors and the predictor with the lowest "
        r"observed error, a \mbox{post-hoc} ranking. A1 is the linear power form and A2 per-density regression "
        r"with interpolation."),
    "tab:round3_prune": (
        r"Third-round pruning confirmation on three new checkpoints (Pythia-410M at step 48k, Pythia-1.4B at step "
        r"112k, Pythia-6.9B at step 80k) at $d\in\{0.85,0.675,0.575\}$, with predictions committed before "
        r"measurement: mean absolute error and mean signed error (prediction minus observation) over nine cells "
        r"per capability, in nats per native token. Candidates are the power form, A2 (per-density least squares "
        r"at five anchors with linear interpolation), A1 (the power form with $\gamma_c{=}1$), the continuous "
        r"two-term form, and the source-free strength-only curve, development median, and zero change. "
        r"Coefficients are given in Appendix Table~\ref{tab:round3_coef}."),
    "tab:round3_quant": (
        r"Third-round grouped-quantization confirmation (mean absolute error, nats), with all predictions "
        r"committed before the tests. The low-order two-dimensional form is $\phi\cdot[1,u,v,uv,u^2]$ with "
        r"$u=\log_2 q_{\max}$ (centred) and $v=\log_2(g/128)$; the separable form is "
        r"$(\beta_c\cdot\phi)\,q_{\max}^{-p_c}(g/128)^{q_c}$; same-input interpolation is bilinear between the four "
        r"development configurations. The bit-width test holds "
        r"out $b{=}4$ and the group-size test $g{=}128$ on the development states; the joint test uses Pythia-1B at "
        r"step 96k at $g{=}128$. The signed bias of the two-dimensional form for Math, Code, and QA is +0.108, "
        r"+0.060, and +0.077 on the bit-width test, +0.162, +0.137, and +0.129 on the group-size test, and +0.345, "
        r"+0.097, and +0.250 on the joint test of source state and configuration."),
    "tab:quant_threeway": (
        r"Three-way quantization confirmation: mean absolute errors in nats with the measured cells at "
        r"$b\in\{3,4,5\}$ weighted equally; $n$ counts cells per capability. \textbf{D} is the frozen development "
        r"leave-one-state-out selection (response surface for Math, development median for Code, zero change for "
        r"QA, with the 0.02-nat tie rule); F lists all frozen candidates; \textbf{R} is the \mbox{post-hoc} "
        r"recommended rule (piecewise interpolation for Math and Code on development states, the median "
        r"otherwise), which reuses frozen candidate predictions. Development states are Pythia-410M at step 143k "
        r"and Pythia-1.4B at step 16k; the new state is Pythia-1.4B at step 112k."),
    "tab:prune_repeat": (
        r"Pruning repeatability on Pythia-2.8B at the final checkpoint, whose two cached revisions carry identical "
        r"weights; the six cells per capability cover $d\in\{0.85,0.75,0.65\}$ measured on each revision. "
        r"Entries are mean absolute prediction errors for signed $\Delta L$ in nats with equal weights. All four "
        r"predictors were frozen before the target measurements from the development register with inputs "
        r"$(N_0,D_0,L_{0,c},d)$ and no target calibration; A2 uses the fixed ridge anchor fits with linear "
        r"interpolation."),
    "tab:p2v2_test": (
        r"Multi-student distillation on the uniform absolute-exposure protocol. Eight forms per capability were "
        r"frozen from the twelve development runs (Gemma-3-270M and 1B, $U\in\{75,450\}$, three data seeds) ; the "
        r"headline column is the frozen form with the lowest development error, a joint form with a source term "
        r"$k\log(N_S/N_{\mathrm{ref}})$, under a rule stated after the tests. Errors are mean absolute errors in "
        r"nats over all test points of the row against the frozen constant, zero change, and the best frozen form "
        r"chosen after the fact (R). Gemma-3-4B is a \mbox{held-out} student ($\log N_S$ 1.9 above the reference; "
        r"development span $\pm0.55$)."),
    "tab:main_context": (
        r"\scriptsize Earlier distillation tests. Errors are mean absolute errors in nats per token, stacked as "
        r"Math, Code, and QA. The strongest alternative is the predictor frozen in that round, other than the "
        r"candidate, with the lowest pooled test error, a retrospective ranking; the gain is the alternative error "
        r"minus the candidate error, so negative values favor the alternative. Rules fixed after a test are "
        r"retrospective for it; the joint-with-student-size rule was stated \mbox{post-hoc} and not delivered."),
    "tab:distill_paired": (
        r"Paired distillation errors for the frozen joint-with-student-size candidate: mean absolute errors and "
        r"paired differences in nats, with positive differences and relative improvements favoring the candidate. "
        r"Differences are point means of $|e_{\mathrm{baseline}}|-|e_{\mathrm{candidate}}|$ and relative "
        r"improvement is $100(\mathrm{MAE}_{\mathrm{baseline}}-\mathrm{MAE}_{\mathrm{candidate}})/\mathrm{MAE}_{\mathrm{baseline}}$. "
        r"Brackets give 95\% percentile intervals from 5,000 paired trajectory bootstrap resamples with student "
        r"$\times$ pool clusters; test rows use 12 points from 3 trajectories and \mbox{held-out}-student "
        r"development rows 16 points from 4, so the intervals have limited resolution."),
    "tab:rule-confirm-by-state": (
        r"Regret by source state and objective on the independent panel: the mean over the 17 nominal storage "
        r"budgets (0.20--1.00 in steps of 0.05), in nats, and $K$, the number of distinct configurations selected "
        r"by the frozen rule across these budgets. This is a \mbox{post-hoc} decomposition of frozen selection "
        r"predictions."),
    "tab:rule-confirm-candidate-sizes": (
        r"Heuristic candidate-set sizes and oracle coverage on all 68 cells per objective. Size columns count "
        r"methods in the no-clear-winner candidate set, including singletons; coverage columns give the count out "
        r"of 68 and the percentage of sets containing the oracle method. This is a \mbox{post-hoc} diagnostic "
        r"based on frozen development errors."),
    "tab:candidate-coverage": (
        r"Measured candidate coverage by source state on the development selection panel. Each method column gives "
        r"the configuration count followed by the minimum available storage ratio $r$ (dimensionless). This is a "
        r"\mbox{post-hoc} inventory."),
    "tab:selection-feasible": (
        r"Feasible leave-one-state-out selection by policy and objective: feasibility coverage and oracle-method "
        r"agreement in percent and mean regret in nats. Own denotes the cells feasible for that policy and common "
        r"the cells feasible for each policy. This is a \mbox{post-hoc} development evaluation."),
    "tab:rule_decomp": (
        r"Decomposition of selection regret in native-token nats (lower is better) over 17 storage budgets "
        r"(0.20--1.00 in steps of 0.05), weighting each state--budget cell equally; all-state means weight the "
        r"step-32k subset (160M, 410M, 1.4B, with newly measured pruning and quantization candidates) and Pythia-1B "
        r"at step 64k (which adds two historical distillation candidates) 3:1. The maximum objective minimizes "
        r"$\max_c(L_c-L_{0,c}^{\mathrm{source}})$; differences are computed before rounding, and negative values "
        r"favor the frozen rule."),
}

# Endings adjusted so that no numbered caption leaves a short last line on the pdfLaTeX build.
_CAPTION_ENDING_FIXES = {
    "tab:panel_prune": ("a dash marks an unmeasured setting.",
                        "a dash marks a setting that was not measured for that model."),
    "tab:quant2d_coef": ("initial loss, and pretraining tokens.",
                         "initial loss, and pretraining tokens, in standardized units."),
    "tab:shared_structure": (r"This is a \mbox{post-hoc} analysis.",
                             r"This is a \mbox{post-hoc} analysis of the development fits and their calibration."),
    "tab:distill_forms_audit": ("F2 adds a saturating budget term.",
                                "F2 adds a saturating budget term to the reuse response."),
    "tab:cond_audit": (r"This is a \mbox{post-hoc} audit.",
                       r"This is a \mbox{post-hoc} audit of the conditioning comparison."),
    "tab:p3_check": ("and 251 and 328 for QA.", "and 251 and 328 for QA, in the same benchmark order."),
    "tab:musique_scope": ("with the range in brackets.", "with the range across trajectories in brackets."),
    "tab:pred_config_prune": ("A2 per-density regression with interpolation.",
                              "A2 the per-density regression with interpolation between densities."),
    "tab:pred_config_qd": (r"follow Table~\ref{tab:pred_source}.", r"follow Table~\ref{tab:pred_source} throughout."),
    "tab:rule-confirm-candidate-sizes": ("based on frozen development errors.",
                                         "based on frozen development errors, without new measurements."),
    "tab:candidate-coverage": (r"This is a \mbox{post-hoc} inventory.",
                               r"This is a \mbox{post-hoc} inventory of the measured candidates."),
}
for _label, (_old, _new) in _CAPTION_ENDING_FIXES.items():
    assert CONCISE_CAPTIONS[_label].count(_old) == 1, _label
    CONCISE_CAPTIONS[_label] = CONCISE_CAPTIONS[_label].replace(_old, _new)


SIZE_PREFIX = re.compile(r"^\\(?:scriptsize|footnotesize|small)\s+")


def _concise(text):
    """Replace the numbered caption of a table environment by its concise version, if one is registered."""
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    label = next((label for label in labels if label in CONCISE_CAPTIONS), None)
    if label is None:
        return text
    match = re.search(r"\\caption\{", text)
    if not match:
        return text
    depth, end = 1, match.end()
    while depth:
        if text[end] in "{}" and text[end - 1] != "\\":
            depth += 1 if text[end] == "{" else -1
        end += 1
    body = text[match.end():end - 1]
    prefix = SIZE_PREFIX.match(body)
    new = CONCISE_CAPTIONS[label]
    if prefix and not SIZE_PREFIX.match(new):
        new = prefix.group() + new
    text = text[:match.end()] + new + text[end - 1:]
    # Drop appended glossary notes; keep continuation notes.
    def drop(m):
        return "" if "Continued" not in m.group() else m.group()
    return re.sub(r"\\caption\*\{(?:[^{}]|\{[^{}]*\})*\}\n?", drop, text)


# No table gets a page to itself: [!htbp] both allows a float page and tells
# LaTeX to ignore the fractions that would otherwise prevent one, so the
# published tables float to the top or bottom of a page that also carries text.
FLOAT_PLACEMENT = re.compile(r"(\\begin\{table\*?\})\[[^\]]*\]")


def table_text(text):
    """Apply editorial changes to tables without touching surrounding prose."""
    text = FLOAT_PLACEMENT.sub(r"\1[tb]", text)
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    fallback = next((label for label in labels if label in CAPTION_NOTES), None)
    text = TABLE.sub(lambda m: _concise(_table(m.group(), fallback)), text)
    text = text.replace("reuse count", "reuse ratio")
    for phrase in NOTE_TRIMS:
        text = text.replace(phrase, "")
    for old, new in NOTE_ENDINGS:
        text = text.replace(old, new)
    return plain_language(fit_table_height(text))


# Method restatement that a table note does not need: the fitting rules are in
# the appendix section the table sits in, and keeping them here made the float
# tall enough to leave the rest of its page empty. What stays is what a reader
# needs to read the numbers: the panel sizes, the interval basis, the reason a
# distillation gain is exactly zero, and the retrospective status.
NOTE_TRIMS = [
    "All capabilities receive equal Capability mean weight. ",
    "Shared anchors are medians of statewise capability means; scales and offsets "
    "minimize development median-anchor squared error. ",
    "pruning development interpolation and grouped-quantization interpolation boundary "
    "flooring are retained. Distillation uses development ordinary least squares and "
    "frozen planned reuse ratios. ",
    "State intervals are descriptive because clusters are few. ",
    "The two development state labels record identical response vectors; counting that "
    "vector once is reported as a sensitivity in the accompanying summary. ",
]

NOTE_ENDINGS = [
    ("and supports no conclusion about conditioning.",
     "and provides no evidence about the benefit of capability conditioning."),
    ("per-capability arithmetic mean development response (not a median).",
     "arithmetic mean development response for each capability, rather than its median."),
    ("invokes the new-stage branch for all four states.",
     "invokes the new-stage branch for all four states included in the independent confirmation panel."),
    ("This retrospective ablation does not constitute a new preregistration.",
     "This retrospective ablation does not constitute a new preregistration of the comparison it reports."),
    ("selection uses the fixed-recipe student-state math form and code/QA constants.",
     "selection uses the fixed-recipe student-state math form together with the code and QA constants fixed at the same freeze."),
    ("The frozen selection rule uses the locked policy; the source-conditioned predictor uses the earlier selection laws.",
     "The frozen selection rule uses the locked policy, and the source-conditioned predictor uses the earlier selection laws."),
    ("historical V39 distillation students, excluded from the V78 prediction fits.",
     "historical distillation students, which were excluded from the prediction fits."),
]


def fit_table_height(text):
    """Compatibility entry point: fit columns without scaling table fonts."""
    if __package__:
        from .paper_table_layout import table_layout
    else:
        from paper_table_layout import table_layout
    return table_layout(text)


def proofread_table(render):
    """Keep wording fixes in the generator's pure rendering path."""
    @wraps(render)
    def wrapped(*args, **kwargs):
        result = render(*args, **kwargs)
        return table_text(result)
    return wrapped


def _caption_sentences(text):
    openings = {
        "Predictive models and their domains.": "The table summarizes predictive models and their domains.",
        "Cached text-model architectures and parameter counts": "The table lists cached text-model architectures and parameter counts",
        "Selected pruning form ": "The selected pruning form is ",
        "Grouped-quantization form ": "The grouped-quantization form is ",
        "standardization centers ": "standardization uses centers ",
        "; ridge ": "; the ridge penalty is ", "Register: ": "The register is ",
        "Pruning development on 17 Pythia states: ": "Pruning development on 17 Pythia states is evaluated by ",
        "Pre-committed rule: lowest mean": "The pre-committed rule selects the lowest mean",
        "Grouped quantization: leave-one-state-out": "Grouped quantization is evaluated by leave-one-state-out",
        "Quantization identifiability audit, retrospective (R).": "This is a retrospective quantization identifiability audit (R).",
        "Top: rank and effective degrees of freedom": "The upper panel reports rank and effective degrees of freedom",
        "Bottom: MAE in nats": "The lower panel reports MAE in nats",
        "Distillation development matrix (12 runs, 48 points): ": "The distillation development matrix (12 runs, 48 points) is evaluated using ",
        "Capability conditioning at matched parameter count (leave-one-run-out MAE, nats): ": "The table compares capability conditioning at matched parameter count (leave-one-run-out MAE, nats), using a ",
        "V75 audit of all 16 V70 distillation forms.": "The audit covers all sixteen frozen distillation forms.",
        "Response: ": "The response is ",
        "Shared structure and one-point calibration.": "The table compares shared structure and one-point calibration.",
        "Specific forms: ": "The specific forms are ",
        "Value of capability conditioning on frozen confirmation panels": "The table evaluates capability conditioning on frozen confirmation panels",
        "V79 audit of capability conditioning.": "The V79 audit examines capability conditioning.",
        "Frozen candidates and delivered rules.": "The table compares frozen candidates and delivered rules.",
        "Earlier distillation context.": "The table summarizes the earlier distillation tests.",
        "Mean absolute error (MAE) in nats": "Errors are mean absolute errors (MAE) in nats",
        "Prediction tests, all three arms, complete list": "The table lists all prediction tests across the three arms",
        "Configuration axis, quantization and distillation (continued from the preceding table block): ": "The configuration-axis tests for quantization and distillation (continued from the preceding table block) cover ",
        "P1-v2 frozen predictions for new sources": "The P1-v2 tests evaluate frozen predictions for new sources",
        "protocol B on step": "protocol B fits on step",
        "Round-3 pruning deliverable.": "The table reports the Round-3 pruning confirmation.",
        "Round-3 grouped-quantization deliverable": "The table reports the Round-3 grouped-quantization confirmation",
        "V74 three-way quantization confirmation.": "The table reports the V74 three-way quantization confirmation.",
        "Pruning repeatability on Pythia-2.8B": "The table evaluates pruning repeatability on Pythia-2.8B",
        "Mean absolute prediction errors for signed ": "Entries give mean absolute prediction errors for signed ",
        "Multi-student distillation test on": "The table evaluates multi-student distillation on",
        "MAE in nats over all test points": "Columns report MAE in nats over all test points",
        "Paired distillation errors for": "The table reports paired distillation errors for",
        "V70 distillation confirmation at ": "The frozen distillation confirmation evaluates ",
        "Each student and capability has six": "Each student and capability is evaluated on six",
        "Primary versus secondary benchmark on": "The table compares primary and secondary benchmarks on",
        "Loss change from dense in nats": "Entries give loss changes from dense in nats",
        "Distillation QA response on": "The table compares distillation QA responses on",
        "Change in conditional loss from": "Entries give changes in conditional loss from",
        "QA measurement scope on Gemma-3-1B: ": "The table audits QA measurement scope on Gemma-3-1B by reporting ",
        "Per-capability pruning damage on": "The table reports per-capability pruning damage on",
        "Per-capability quantization damage (": "The table reports per-capability quantization damage (",
        "Feasible leave-one-state-out selection.": "The table evaluates feasible leave-one-state-out selection.",
        "Independent V78 selection panel.": "The table reports the independent V78 selection panel.",
        "Selection regret decomposition (": "The table decomposes selection regret (",
        "V78 regret by source state and objective.": "The table reports V78 regret by source state and objective.",
        "Heuristic candidate-set sizes and oracle coverage on": "The table summarizes heuristic candidate-set sizes and oracle coverage on",
        "Measured candidate coverage by source state.": "The table inventories measured candidate coverage by source state.",
        "Exact locked selection rule in ": "The table specifies the exact locked selection rule in ",
        "Generated by ": "The table was generated by ",
    }
    # All current generated captions occupy one source line. Keep notes and data
    # cells outside captions out of this grammatical pass.
    def sentence(match):
        line = match.group()
        for before, after in openings.items():
            line = line.replace(before, after)
        return line
    return re.sub(r"\\caption\*?\{[^\n]*", sentence, text)


def _wrap_table_words(label, text):
    """Give expanded labels readable breaks before applying column widths."""
    def wrap_tabular(match):
        block = match.group()
        paragraph_columns = "p{" in block.splitlines()[0] or r"\dimexpr" in block.splitlines()[0]
        lines = []
        for line in block.splitlines():
            if paragraph_columns or " & " not in line or "EDITORIALPROTECTEDTOKEN" in line:
                lines.append(line)
                continue
            cells = line.split(" & ")
            for i, cell in enumerate(cells):
                ending = r" \\" if cell.endswith(r" \\") else ""
                content = cell[:-len(ending)] if ending else cell
                # Formulas, existing paragraph cells, and numeric data retain
                # their own TeX formatting. Only ordinary word labels wrap.
                if len(content) > 20 and not re.search(r"[\\${}]", content):
                    words = textwrap.wrap(content, width=20, break_long_words=False, break_on_hyphens=False)
                    if len(words) > 1:
                        cells[i] = r"\shortstack[l]{" + r"\\".join(words) + "}" + ending
            lines.append(" & ".join(cells))
        block = "\n".join(lines)
        return block
    return re.sub(r"\\begin\{(tabularx?)\}.*?\\end\{\1\}", wrap_tabular, text, flags=re.S)


def _specific(label, text):
    """Context-sensitive labels and captions whose meanings vary by table."""
    # These terms are complete predictor labels, not the stored candidate keys.
    names = {
        "A2": "Per-density regression", "cont": "Continuous two-term form",
        "power": "Power form", "zero": "Zero change", "Zero": "Zero change",
        "mean": "Development mean", "median": "Development median",
        "separable": "Separable form", "full": "Full source regression",
        "constant": "Constant", "joint": "Joint budget and pool response",
        "E": "Reuse response", "T": "Budget response",
        "all": "All densities", "off coarse": "Densities off the coarse grid",
        "best": "Lowest-error predictor", "subset": "Subset", "candidate": "Candidate",
        "capability": "Capability", "term": "Term", "bits": "Measured bit widths",
        "unseen": "Held-out axis", "origin": "Provenance", "Line": "Predictor",
        "same": "Same as frozen candidate", "Cheapest": "Cheapest feasible configuration",
        "Multi": "Maximum across capabilities", "Prune": "Pruning",
        "budget only": "Budget only", "reuse only": "Reuse only", "strength only": "Strength only",
        "median curve": "Development median curve", "per-bit development median": "Per-bit development median",
        "per-bit development mean": "Per-bit development mean", "development median curve": "Development median curve",
        "response surface with initial loss": "Response surface with initial loss",
        "response surface with student size": "Response surface with student size",
        "constant with student size": "Constant with student size", "budget with student size": "Budget with student size",
        "reuse with student size": "Reuse with student size", "joint with student size": "Joint with student size",
    }
    def rename_cell(match):
        prefix, value = match[1], match[2]
        return prefix + names.get(value, value)
    text = re.sub(r"(^| & )([^&\n]*?)(?= & | \\\\\n)", rename_cell, text, flags=re.M)
    common = {
        "same input interpolation": "Interpolation with the same inputs",
        "MAE/signed bias": "mean absolute error and signed bias",
        "MAE in nats": "Mean absolute error (MAE) in nats",
        "(MAE, nats)": "(mean absolute error, nats)",
        "(MAE over three capabilities, nats per native token)": "(mean absolute error over three capabilities, in nats per native token)",
        "per-capability MAE": "mean absolute error per capability",
        "MAEs and paired differences": "Mean absolute errors and paired differences",
        "P shared/per": "Parameters, shared and specific",
        "Per-cap MAE": "Capability-specific MAE",
        "LOCO Math": "Leave-one-run-out Math", "LOCO Code": "Leave-one-run-out Code",
        "Prune Math": "Pruning Math", "Prune Code": "Pruning Code", "Prune QA": "Pruning QA",
        "Prune confirmation": "Pruning confirmation", "Prune pairs": "Pruning pairs",
        "\nForm & P &": "\nForm & Parameters &",
        "Candidate & $P$ &": "Candidate & Parameters &",
        "$P$/capability": "Parameters per capability",
        "same-input: no-$D_0$": "Same-input baseline without pretraining tokens",
        "same-input: ": "Same-input baseline: ", "simple: ": "Simple baseline: ",
        "candidate: ": "Candidate: ",
        "A2 (per-$d$ regression)": "per-density regression (A2)",
        "strength only curve": "strength-only curve",
        "Frozen / retrospective": "Prediction and rule status",
        "selected on development data, R": "selected on development data; rule chosen post-hoc",
        "Best frozen (R)": "Best frozen form (R)",
        "median development curve": "development median curve",
        "leave-one-run-out (leave-one-run-out)": "leave-one-run-out evaluation",
        "leave-one-state-out (leave-one-state-out)": "leave-one-state-out evaluation",
        "leave-one-student-out (leave-one-student-out)": "leave-one-student-out evaluation",
        "development\\ ": "development ", "interpolation\\ ": "interpolation ",
        "extrapolation\\ ": "extrapolation ", "including\\ ": "including ",
        "constant$+$src": "constant with student size", "joint$+$src": "joint with student size",
        "Channel round-to-nearest quantization": "Per-channel round-to-nearest quantization",
        "Config-indicator ordinary least squares": "Configuration-indicator ordinary least squares",
        "\n\\textbf{power (selected)}": "\n\\textbf{Power form (selected)}",
        r"Test; \emph{unseen axis}; [origin]": r"Test; \emph{held-out axis}; [provenance]",
        "32k+112k": "32k and 112k", "at step 32k and": "at steps 32k and",
        "configuration-indicator ordinary least squares uses": "Configuration-indicator ordinary least squares uses",
        "5-bit rule including": "including the 5-bit rule",
    }
    for before, after in common.items():
        text = text.replace(before, after)
    changes = {
        "tab:main-prediction-v2": {
            "Relation (parameters)": "Relation (parameter count)",
            "MAE &": "Mean absolute error &", "Baseline Mean absolute error": "Baseline mean absolute error",
            " / ": "; ", "median ": "development median ",
            "surface and reuse only": "response surface and reuse only",
            "zero change, surface and reuse only": "zero change, response surface and reuse only",
            " / surface ": "; response surface ", "; surface ": "; response surface ",
            "additive log (4); curved (5); interaction (5)": "additive logarithmic form (4); curved form (5); interaction form (5)",
            "Pythia 160M": "Pythia-160M", "Pythia 410M": "Pythia-410M", "Pythia 1.4B": "Pythia-1.4B",
            "The baseline column names": "The baseline row names",
            "Frozen prediction evidence per method and prediction task.": "Development and frozen prediction evidence by method and prediction task.",
            "Cells marked not tested": "Cells marked not evaluated",
            "not tested": "not evaluated",
        },
        "tab:models": {
            "Quantization, fixed $b$": "Quantization, fixed bit width",
            "Quantization, $(b,g)$": "Quantization, bit width and group size",
            "new state mixed": "mixed results on the new state",
            "near-zero regime uninformative": "the near-zero regime is uninformative",
            "multi-student test frozen": "multi-student predictions frozen before testing",
            "fixed pool and budget": "a fixed pool and budget",
            "pools pool of 75 traces per domain/pool of 600 traces per domain": "pools of 75 and 600 traces per domain",
            "unseen pool pool of 225 traces per domain": "an unseen pool of 225 traces per domain",
            "v38/v49": "v38 and v49", "v41/v50": "v41 and v50",
        },
        "tab:model_arch": {"/dense": "; dense", "M/U/T": "M, U, T", "v53/v64": "v53 and v64"},
        "tab:v53_loso": {"QA & Development mean": "QA & Capability mean"},
        "tab:v56_forms": {"leave-one-state-out": "leave-one-student-out", "form and capability": "form and capability", "L0 (": "initial loss ("},
        "tab:v56_condition": {" >0.02 ": " Gain exceeds 0.02 ", " & yes ": " & Yes ", " & no ": " & No "},
        "tab:distill_forms_audit": {
            "Selected form E": "The selected reuse response", "Baseline reuse only": "The reuse-only baseline",
            "The suffix +src introduces student size $n$.": "The student-size variants introduce student size $n$.",
            "QA selects joint; Math and Code select E.": "QA selects the joint form; Math and Code select the reuse response.",
            "F1 and F2 have": "F1 and F2 have", "training SSE": "training sum of squared errors",
            " & none &": " & No descriptor &", " & yes &": " & Yes &", " & no &": " & No &",
        },
        "tab:shared_structure": {
            "Held-out MAE shared vs specific": "Held-out MAE, shared versus specific",
            "K0 vs K1 MAE": "MAE, K0 versus K1", "80\\% PI coverage \\& width": "80\\% prediction interval coverage and width",
            "MAE and PI widths are nats": "Mean absolute errors and prediction interval widths are in nats",
            "PI entries": "Prediction interval entries", " vs ": " versus ",
            "sep K1": "separable K1", "2D K1": "response-surface K1",
        },
        "tab:cond_audit": {"B scale &": "Response family or diagnostic &", "QA and the capability mean intervals": "QA and capability-mean intervals"},
        "tab:main_final": {
            "Frozen $E$ / $E$ / joint": "Frozen reuse, reuse, and joint responses",
            "Frozen D": "Frozen development selection",
            "surface/median/zero": "response surface, development median, and zero change, respectively",
            "Interp.": "Interpolation", "A2 $": "Per-density regression $",
            "$T$+src": "Budget with student size", "$T$-only": "Budget only",
            "$d$ ": "densities ", "$b$ ": "bits ", "$g$ ": "group sizes ",
            " / ": "; ",
            "MAE in nats per token; stacks: Math, Code, and QA.": "Mean absolute errors are in nats per token; stacked entries give Math, Code, and QA, respectively.",
            "Strongest: lowest": "The strongest alternative has the lowest",
            "negative = alternative better": "negative values favor the alternative",
            "after test; reused in Sec. 5": "post-hoc; reused in Section 5",
            "after test": "post-hoc", "before test": "frozen before testing",
        },
        "tab:main_context": {
            "Frozen joint with student size": "Frozen joint response with student size",
            "$E$ $": "Reuse response $", "$T$ $": "Budget response $",
            "Strongest: lowest": "The strongest alternative has the lowest",
            "negative = alternative better": "negative values favor the alternative",
            "after test": "post-hoc", "after a test": "after a test",
            "stated post-hoc; not delivered": "stated post-hoc and was not delivered",
            "stacks: Math, Code, and QA": "stacked entries give Math, Code, and QA, respectively",
        },
        "tab:pred_full": {
            "prune:": "Pruning:", "quant:": "Quantization:", "distill:": "Distillation:",
            "unseen pool pool of": "unseen pool of", "prune ": "pruning ",
            "nD = no-$D_0$ variant": "nD denotes the variant without pretraining tokens",
            "pbm/pba = per-bit median/mean": "pbm and pba denote the per-bit development median and mean, respectively",
            "c = constant": "c denotes a constant", "TE = joint $T{+}E$ form": "TE denotes the joint budget and reuse form",
            "candidate\\,/\\,strongest baseline": "candidate followed by strongest baseline",
        },
        "tab:pred_config_prune": {"development sources +": "development sources and", "A2 (per-$d$ regression)": "per-density regression (A2)"},
        "tab:pred_config_qd": {
            "unseen pool pool of": "unseen pool of", "endpoint pools pool of 75 traces per domain/pool of 600 traces per domain": "endpoint pools of 75 and 600 traces per domain",
            "$E$-only (reuse count)": "reuse only", "$T{+}E$ joint": "joint budget and reuse response",
        },
        "tab:p1v2": {
            "prune interpolation d": "Pruning interpolation at densities ",
            "prune extrapolation d": "Pruning extrapolation at density ",
            "quant ": "Quantization at ", "(rule)": "(fixed interpolation rule)",
            "new-source frozen prospectives": "frozen predictions for new sources",
        },
        "tab:round3_prune": {"test MAE": "Test MAE", "signed bias (prediction\\ $-$ observation)": "Signed bias (prediction $-$ observation)", "vs A2": "versus A2", "vs the": "versus the"},
        "tab:round3_quant": {"bit test (12)": "Bit-width test (12)", "granularity test (18)": "Group-size test (18)", "joint, ": "Joint test, "},
        "tab:quant_threeway": {
            "\\textbf{D} &": "\\textbf{Frozen selection} &", "\nF &": "\nFrozen candidate &",
            "\\textbf{R} &": "\\textbf{Post-hoc rule} &", "development-leave-one-state-out": "development leave-one-state-out",
            "post-test": "post-hoc", "surface;": "response surface;", "median;": "development median;",
        },
        "tab:prune_repeat": {"Signed $\\Delta L$ errors": "Mean absolute prediction errors for signed $\\Delta L$"},
        "tab:distill_confirm": {"270m &": "Gemma-3-270M &", "1b &": "Gemma-3-1B &", "Brackets:": "Brackets give"},
        "tab:p3_check": {"probe / on": "probe and on", "primary / secondary:": "primary then secondary:", "round-to-nearest quantization 4 bits": "Per-channel round-to-nearest quantization at 4 bits"},
        "tab:musique_scope": {"answerable-dev": "answerable development", " & $E$ & $n$ &": " & Reuse count & Trajectories &", ", final &": ", final checkpoint &", ", early &": ", early checkpoint &", "Mean over trajectories and the range in brackets.": "Entries give the mean over trajectories and the range in brackets."},
        "tab:qa_scope": {"Prune d=": "Pruning at density ", "round-to-nearest quantization 4 bits/channel": "Per-channel round-to-nearest quantization at 4 bits", "round-to-nearest quantization b=4 g=128": "Grouped round-to-nearest quantization at 4 bits and group size 128", "distillation T=": "Distillation at supervised budget "},
        "tab:panel_prune": {"QA least$\\,|\\,$worst": "QA least and most damaged", "none by ": "Not reached by density ", "capability the last": "capability. The last", "models are grouped": "Models are grouped", "counts, over": "counts, over"},
        "tab:selection-feasible": {"Own/common": "Own and common", "Empty subsets are not available.": "Empty subsets are marked not available.", "MAP/ORACLE/CHEAPEST": "predicted selection, measured oracle, and cheapest feasible configuration", "Predicted selection/Measured oracle/Cheapest feasible configuration": "predicted selection, measured oracle, and cheapest feasible configuration", "pruning/round-to-nearest quantization": "pruning and round-to-nearest quantization", "The multi objective": "The maximum-capability objective"},
        "tab:rule-confirm": {"Policy internal names: locked-rule, v64-law (respectively).": "The frozen selection rule uses the locked policy; the source-conditioned predictor uses the earlier selection laws.", "The multi objective": "The maximum-capability objective"},
        "tab:rule-confirm-by-state": {"Policy internal names: locked-rule, v64-law (respectively).": "The frozen selection rule uses the locked policy; the source-conditioned predictor uses the earlier selection laws.", "1B/64k": "Pythia-1B at step 64k", "Multi minimizes": "The maximum-capability objective minimizes"},
        "tab:rule-confirm-candidate-sizes": {"Policy internal names: locked-rule, v64-law (respectively).": "The frozen selection rule uses the locked policy; the source-conditioned predictor uses the earlier selection laws.", "1B/64k": "Pythia-1B at step 64k", "configuration ID": "configuration identifier", "count/68 (percent)": "count out of 68 (percent)"},
        "tab:candidate-coverage": {"Min $r$: quant": "Minimum quantization storage ratio", "configuration count / minimum": "configuration count followed by the minimum", "Quant pools": "Quantization pools", "distillation has": "Distillation has", "not available means": "Not available means"},
        "tab:locked_rule": {" & v53 power &": " & Pruning power form &", " & v53 median curve &": " & Pruning development median curve &", " & v36 regression &": " & Per-bit source regression &", " & v69 interpolation &": " & Piecewise source interpolation &", "linear (Math)": "Linear source regression", "0 (no response)": "0 (no loss change)"},
        "tab:final": {"What each arm delivers, answering the same five questions.": "Each compression family is summarized by its predictor, inputs, evidence status, gains, and scope.", "headline rule post-hoc (first test) or pre-committed (confirmation)": "the headline rule was chosen post-hoc (first test) or pre-committed (confirmation)", "Math: baseline better": "Math: the baseline performs better", "none for QA": "no gain for QA", "none over the median": "no gain over the median"},
    }
    for before, after in changes.get(label, {}).items():
        text = text.replace(before, after)
    if label == "tab:v56_forms":
        text = text.replace("leave-one-student-out (leave-one-student-out)", "leave-one-student-out evaluation")
    if label in {"tab:main_final", "tab:main_context"}:
        text = re.sub(r"(?<!\w)same(?=\\newline| &)", "Same as frozen candidate", text)
        text = text.replace("stacks: Math, Code, and QA", "stacked entries give Math, Code, and QA, respectively")
    if label == "tab:p2v2_test":
        text = re.sub(r"(?<= & )([ET])(?= \()", lambda m: {"E": "Reuse response", "T": "Budget response"}[m[1]], text)
    if label == "tab:distill_forms_audit":
        text = text.replace(" & T &", " & Budget response &").replace(" & E &", " & Reuse response &")
    if label == "tab:p2v2_test":
        text = text.replace("joint (", "Joint budget and pool response (").replace("constant (", "Constant (")
        text = text.replace("constant with student size (", "Constant with student size (").replace("joint with student size (", "Joint with student size (")
    if label == "tab:main_final":
        text = text.replace("2.8B, one state", "Pythia-2.8B, one state")
    if label == "tab:p1v2":
        text = text.replace(r"Quantization at $\ge$4-bit", r"Quantization at $\ge$4 bits")
        text = text.replace("densities 0.9-0.6", "densities 0.9 to 0.6")
    if label in {"tab:distill_paired", "tab:musique_scope"}:
        text = re.sub(r"(?m)^(270M|1B|4B)(?= &)", r"Gemma-3-\1", text)
    if label in {"tab:main_final", "tab:main_context"}:
        text = re.sub(r"(?<= & )(270M|1B|4B)(?=;)", r"Gemma-3-\1", text)
    # Slash-separated display lists are lists, not mathematical ratios.
    if label in {"tab:main-prediction-v2", "tab:main_final", "tab:main_context", "tab:pred_full", "tab:pred_config_qd", "tab:round3_quant", "tab:models"}:
        text = re.sub(r"(?<=[\dkBM])/(?=[\d])", ", ", text)
    if label in {"tab:p3_check", "tab:v56_forms", "tab:v56_condition", "tab:candidate-coverage"}:
        # Outside math, paired values are separated by a semicolon.
        text = re.sub(r"(?<=[\d$])\s*/\s*(?=[+$\d-]|not available)", "; ", text)
    return text


# Final reader-facing vocabulary. Data keys and numerical formatting remain in
# the individual generators. This pass also covers their already-rendered notes.
PLAIN_WORDS = {
    'QA': 'Question answering', 'Math': 'Mathematics',
    'MAEs': 'mean absolute errors', 'MAE': 'mean absolute error',
    'CI': 'confidence interval', 'LOSO': 'leave one state out',
    'LOCO': 'leave one run out', 'OLS': 'ordinary least squares',
    'RTN': 'round-to-nearest quantization', 'LoRA': 'low-rank adaptation',
    'A1': 'linear power form', 'A2': 'per-density regression',
    'F1': 'descriptor-modulated reuse', 'F2': 'saturating budget and reuse',
    'K0': 'no target measurement', 'K1': 'one target measurement',
    'nD': 'no pretraining-token input', 'pbm': 'per-bit median',
    'pba': 'per-bit mean', 'med': 'median', 'so': 'strength only',
    'TE': 'joint budget and reuse', 'ct': 'continuous two-term form',
    'L0': 'initial loss', 'logN': 'logarithmic student size',
    'Arm': 'Compression method', 'df': 'effective degrees of freedom',
}
EXPERIMENT_WORDS = {
    '36': 'per-bit source', '38': 'fixed-bit development',
    '39': 'fixed-recipe student', '41': 'single-student pool',
    '49': 'new-source prediction', '50': 'multi-student development',
    '53': 'pruning development', '55': 'grouped-quantization development',
    '64': 'development selection', '69': 'grouped-quantization interpolation',
    '70': 'distillation confirmation', '78': 'independent selection',
    '85': 'selection decomposition',
}
FOLD_WORDS = {
    'RFRA': 'Retrospective / fixed / retrospective',
    'PFRA': 'Frozen / fixed / retrospective',
    'P–RA': 'Frozen / absent / retrospective',
    'PRFA': 'Frozen / retrospective / fixed',
    'PFFA': 'Frozen / fixed / fixed',
}

# No new numerical claims are introduced here. The previous caption's numerical
# definitions remain, in order, in the adjacent note. Caption text is intentionally
# independent of experiment identifiers, formulas and the stored artifact names.
PLAIN_CAPTIONS = {
 'tab:models': 'The table shows which prediction forms apply to each compression method and test range. Loss is measured in nats per native token, bit width in bits, and budgets in tokens. Parameter and calibration columns give counts for each prediction family.',
 'tab:model_arch': 'The table compares model architectures and parameter counts. Counts are in billions of parameters. The columns distinguish counts before restoring tied weights, active parameters, unique parameters after restoring ties, and parameters in transformer matrices.',
 'tab:round3_coef': 'The table gives the frozen pruning exponent and coefficients for each capability. Coefficients act on standardized inputs; the resulting loss change is in nats per native token. The note gives the prediction formula and the constants used to standardize each input.',
 'tab:quant2d_coef': 'The table gives frozen grouped-quantization coefficients by capability and response term. Coefficients act on standardized inputs; the resulting loss change is in nats per native token. The note gives the prediction formula and the constants used to standardize each input.',
 'tab:v53_loso': 'The table compares pruning predictors when one source is held out. Entries are mean absolute errors in nats per native token. The rows separate all measured densities from densities outside the coarse development grid.',
 'tab:v55_loso': 'The table compares grouped-quantization predictors when one state is held out. Entries are mean absolute errors in nats per native token. A collapsing development state dominates the errors, so all candidate forms were retained for testing.',
 'tab:quant_ident': 'The table gives the number of independent coefficients the quantization designs support and how accurately they predict held-out measurements. The upper panel gives dimensionless rank and effective degrees of freedom. The lower panel gives mean absolute errors in nats per native token.',
 'tab:v56_forms': 'The table compares distillation forms under run and student holdouts. Each pair gives mean absolute error followed by signed bias, in nats per native token. Parameter counts cover all capabilities.',
 'tab:v56_condition': 'The table compares shared and capability-specific distillation responses at matched parameter counts. Errors and gains are in nats per native token. Paired parameter counts list the shared response followed by the capability-specific response.',
 'tab:distill_forms_audit': 'The table records the frozen distillation formulas and their fitted coefficients. Responses are loss changes in nats per native token. The first panel specifies the fit; the second lists coefficients in the stated order. The notes define the model inputs and the procedure used to fit each form.',
 'tab:shared_structure': 'The table compares a shared response family with capability-specific responses, and reports the spread of exponents fitted separately per source. Errors are mean absolute errors in nats per native token, pooled equally over capabilities. Distillation carries no pretraining-token input, so that term is inactive.',
 'tab:calibration_k1': 'The table reports the effect of one target measurement. Errors are mean absolute errors in nats per native token, interval widths are in the same units, and coverage is a percentage. The calibration cell itself is excluded from scoring.',
 'tab:cap_conditioning': 'The table compares separate capability responses with shared responses on frozen confirmation panels. Mean absolute errors and gains are in nats per native token. Positive gains favor separate responses; brackets give paired confidence intervals for the error differences.',
 'tab:cond_audit': 'The table compares separate capability responses with a shared response and fitted capability scales. Scales and squared correlations are dimensionless. Errors and improvements are in nats per native token; brackets give confidence intervals for the error differences.',
 'tab:main_final': 'The table compares frozen relations, alternatives, and delivered rules on confirmation tests. Mean absolute errors and gains are in nats per native token. Lists follow mathematics, code, and question answering. Gain is alternative error minus relation error; retrospective rules were chosen after testing.',
 'tab:main_context': 'The table compares the earlier distillation relation with frozen alternatives on unseen pools and students. Mean absolute errors and gains are in nats per native token. Lists follow mathematics, code, and question answering. Gain is alternative error minus relation error; these rules were not delivered.',
 'tab:pred_full': 'The table compares relations with the named strongest baselines across all prediction tests. Pairs give relation error followed by baseline error, in nats per native token; bold marks the lower error. All specified forms are reported. Fold rules distinguish frozen predictions, fixed baselines, and retrospective fits.',
 'tab:pred_source': 'The table tests predictions for unseen source states at fixed compression settings. Entries are mean absolute errors in nats per native token. Bold marks the lowest error; parentheses give improvement over the strongest baseline. Asterisks mark confidence intervals excluding zero. All specified forms are reported.',
 'tab:pred_config_prune': 'The table tests pruning predictions at unseen densities and source states. Entries are mean absolute errors in nats per native token. Bold marks the lowest error; parentheses give baseline error minus relation error. All specified forms are reported, including the frozen baselines.',
 'tab:pred_config_qd': 'The table tests quantization and distillation predictions on unseen settings, source states, and pools. Entries are mean absolute errors in nats per native token. Bold marks the lowest error; parentheses give baseline error minus relation error. All specified forms are reported, including each baseline frozen before testing.',
 'tab:p1v2': 'The table compares frozen predictors on new source states under two development-data rules. Entries are mean absolute errors over capabilities, in nats per native token. The last column reports the retrospective winner, selected by the lowest observed error.',
 'tab:round3_prune': 'The table compares frozen pruning predictors on new checkpoints and densities. Mean absolute errors and signed biases are in nats per native token. Bias is prediction minus observation; parameter counts are per capability.',
 'tab:round3_quant': 'The table compares frozen grouped-quantization predictors on unseen bit widths, group sizes, and source states. Entries are mean absolute errors in nats per native token. Parentheses in test headings give the number of measured cells per capability.',
 'tab:quant_threeway': 'The table compares development-selected predictors, all frozen candidates, and retrospective rules on the quantization confirmation panel. Entries are mean absolute errors in nats per native token, with equal weight assigned to each measured confirmation cell.',
 'tab:prune_repeat': 'The table compares pruning prediction errors when identical model weights are measured twice. Entries are mean absolute errors in nats per native token, with equal weight per measurement. All predictors were frozen before target measurement.',
 'tab:p2v2_test': 'The table compares frozen distillation predictions for unseen pools and students. Entries are mean absolute errors in nats per native token. The selected relation minimizes development error; the best frozen form is a retrospective ranking. Pool sizes count traces per domain.',
 'tab:distill_paired': 'The table compares distillation errors with the frozen constant and zero-change baselines. Errors and paired differences are in nats per native token; relative improvements are percentages. Positive differences favor the relation. Brackets give paired confidence intervals.',
 'tab:distill_confirm': 'The table compares distillation relations with baselines selected before confirmation. Mean absolute errors and paired improvements are in nats per native token. Improvement is baseline error minus relation error; brackets give paired confidence intervals. Pool sizes count traces per domain; budgets count supervised tokens.',
 'tab:p3_check': 'The table compares loss changes on primary and independent secondary benchmarks. Each pair lists the primary benchmark followed by the named secondary benchmark, in nats per native token. Compression states were specified before secondary measurement.',
 'tab:musique_scope': 'The table compares question-answering loss changes across two benchmarks and distillation checkpoints. Loss changes are in nats per native token. Entries give trajectory means with ranges in brackets; reuse is dimensionless and pool sizes count traces per domain.',
 'tab:qa_scope': 'The table compares question-answering loss across benchmarks and compression states. Loss and change from the dense model are in nats per native token. Negative changes indicate improvement. Distillation budgets count the supervised tokens used for training.',
 'tab:panel_prune': 'The table compares the pruning loss change across models and capabilities. Loss changes are in nats per native token; density is dimensionless. Ranking cells count least-affected and most-affected outcomes out of measured densities. Daggers mark prospective additions; dashes mark unmeasured settings.',
 'tab:panel_quant': 'The table compares the quantization loss change across models and capabilities. Loss changes are in nats per native token, and bit widths are in bits. The last column lists all measured widths. Daggers mark prospective additions.',
 'tab:selection-feasible': 'The table compares selection policies on their feasible compression choices. Coverage and agreement with the measured oracle are percentages; mean regret is in nats per native token. Results distinguish each policy\'s feasible cells from the cells shared by all policies.',
 'tab:rule-confirm': 'The table compares frozen selection policies on the independent confirmation panel. Mean regret is in nats per native token. Feasibility entries give feasible cells out of total cells; agreement and candidate-set coverage are reported as percentages of evaluated cells.',
 'tab:rule_decomp': 'The table separates selection regret by source-state subset and policy. Mean regret and policy differences are in nats per native token, with equal weight per state and storage budget. Negative differences favor the frozen rule.',
 'tab:rule-confirm-by-state': 'The table compares policy regret for each source state and objective. Mean regret is in nats per native token, averaged over storage budgets. The final column counts distinct configurations selected by the frozen rule.',
 'tab:rule-confirm-candidate-sizes': 'The table shows candidate-set sizes and coverage of the measured oracle. Set sizes count compression methods. Coverage entries give covered cells out of total cells, followed by the percentage in parentheses. This is a retrospective diagnostic of frozen predictions.',
 'tab:candidate-coverage': 'The table shows measured compression choices available for each source state. Method columns list configuration count followed by minimum storage ratio. Counts are numbers of configurations; storage ratios are dimensionless. Unavailable entries indicate that no candidate was measured.',
 'tab:locked_rule': 'The table gives the selection predictor for each compression method, capability and source-state status. A prediction adds the listed response, in nats per native token, to a dense anchor: the source loss for pruning and quantization, the initial student\'s loss for distillation. Each method is fitted on its own development panel, listed below. The rule was frozen before confirmation.',
 'tab:final': 'The table summarizes delivered predictors, their inputs, evidence, and applicable settings. Prediction errors and gains are in nats per native token, relative to the strongest baseline using the same information. Rules chosen after testing are marked retrospective.',
}


def _reader_words(text):
    """Expand prose tokens while protecting TeX references and mathematical names."""
    protected = re.compile(r'(?m)^%[^\n]*|\\(?:label|ref|eqref|cite|texttt)\{[^}]*\}|\$(?:\\.|[^$])*\$')
    code_phrases = {
        r'\texttt{results/v75-distill-audit/summary.json}': 'the saved distillation audit',
        r'\texttt{new\_state}': 'a new state', r'\texttt{new\_size}': 'a new size',
        r'\texttt{new\_stage}': 'a new stage', r'\texttt{new\_source}': 'a new source',
        r"\texttt{qa\_distribution='2Wiki'}": 'the primary question-answering distribution',
        r'\texttt{tables/final\_deliverables.tex}': 'The delivered-predictor table',
    }
    def prose(s):
        for old, new in PLAIN_WORDS.items():
            s = re.sub(r'(?<![\w\\])' + re.escape(old) + r'(?!\w)', lambda m: new, s)
        s = re.sub(r'\b[Vv](\d{2})\b', lambda m: EXPERIMENT_WORDS.get(m[1], 'development'), s)
        # Unit prefixes change words, never the printed digits.
        s = re.sub(r'(?<=\d)k\b', ' thousand', s)
        s = re.sub(r'(?<=\d)M\b', ' million', s)
        s = re.sub(r'(?<=\d)B\b', ' billion', s)
        return re.sub(r'(?<=\w)@(?:step)?(?=\d)', ' at step ', s)
    parts = []; start = 0
    for m in protected.finditer(text):
        parts.extend([prose(text[start:m.start()]), code_phrases.get(m.group(), m.group())]); start = m.end()
    parts.append(prose(text[start:]))
    return ''.join(parts)


def _reader_caption(block, label):
    try:
        from .paper_table_layout import group
    except ImportError:
        from paper_table_layout import group
    pattern = re.compile(r'\\caption(?:\*|\[\])?\{')
    edits = []
    for match in pattern.finditer(block):
        old, end = group(block, match.end()-1)
        if 'continued' in old.lower():
            caption = ('The table continues the preceding comparison. '
                       + PLAIN_CAPTIONS[label].split('. ', 1)[1])
            note = ''
        else:
            caption = PLAIN_CAPTIONS[label]
            # Preserve numerical statements in their original position. Readers
            # see concise captions and fully traceable technical notes.
            old = SIZE_PREFIX.sub('', old)
            if label == 'tab:pred_full':
                old = (r'The zero-change baseline predicts 0; the linear power form fixes $\gamma{=}1$. '
                       r'Improvements and uncertainty intervals appear in Appendix Tables~\ref{tab:pred_source}--\ref{tab:pred_config_qd}.')
            else:
                # Keep numerical definitions and the literal reference patches;
                # discard obsolete code glossaries and repeated reading rules.
                sentences = re.split(r'(?<=[.;]) (?=[A-Z$])', old)
                kept = []
                for sentence in sentences:
                    probe = re.sub(r'\\(?:ref|eqref)\{[^}]*\}|\b(?:[AFKGVv]\d+|L0)\b|_(?:\{[^{}]*\}|\d)', '', sentence)
                    if re.search(r'\d', probe) or r'\S\ref{sec:unseen_settings}' in sentence or r'Appendix Table~\ref{tab:round3_coef}' in sentence:
                        kept.append(sentence)
                old = ' '.join(kept)
            # A minipage keeps the note justified: inside the table's \centering
            # a bare paragraph centres its last line, which reads as an error.
            note = ('\n'+r'\par\smallskip\begin{minipage}{\linewidth}\footnotesize '
                    + old + r'\end{minipage}') if old else ''
        edits.append((match.end(), end-1, caption, note))
    for start, end, caption, note in reversed(edits):
        block = block[:start]+caption+'}'+note+block[end+1:]
    return block


def _reader_body(block, label):
    try:
        from .paper_table_layout import group
    except ImportError:
        from paper_table_layout import group
    # Remove artificial stacks before expanding words. Paragraph columns wrap.
    pattern = re.compile(r'\\shortstack(?:\[[^]]*\])?')
    while m := pattern.search(block):
        content, end = group(block, m.end())
        block = block[:m.start()]+'{'+content.replace(r'\\', ' ')+'}'+block[end:]
    block = block.replace(r'\allowbreak ', '')
    block = block.replace(r'\mbox{Held-out} axis', 'What is held out')
    block = block.replace(r'\emph{\mbox{held-out} axis}; [provenance]', 'what is held out; fold rule')
    block = block.replace('Provenance', 'Fold rule')
    block = block.replace('P/F/F/A', 'PFFA')
    for code, words in FOLD_WORDS.items():
        block = block.replace(code, words).replace('·'.join(code), words)
    # Math symbols that serve as code labels are translated; genuine formulas
    # remain next to explicit word descriptions in formula/coefficient tables.
    replacements = {
        r'$F_1(L_0)$': 'Descriptor-modulated reuse with initial loss',
        r'$F_2(L_0)$': 'Saturating budget and reuse with initial loss',
        r'$\beta_{c,0}$': 'Intercept', r'$\beta_{0}$': 'Intercept',
        r'$\beta_{c,\log N_0}$': 'Source size', r'$\beta_{\log N_0}$': 'Source size',
        r'$\beta_{c,L_0}$': 'Initial loss', r'$\beta_{L_0}$': 'Initial loss',
        r'$\beta_{c,\log D_0}$': 'Pretraining tokens', r'$\beta_{\log D_0}$': 'Pretraining tokens',
        r'$\gamma_c$': 'Exponent', r'$K$': 'Selected configurations',
        r'$N_{\rm meta}$': 'Parameters before tying', r'$N_{\rm active}$': 'Active parameters',
        r'$N_{\rm unique}$': 'Unique parameters', r'$N_{0,\rm matrix}$': 'Matrix parameters',
        r'$\Delta$ 2Wiki': 'Loss change on 2Wiki', r'$\Delta$ MuSiQue': 'Loss change on MuSiQue',
        r'$\Delta L$': 'Loss change', r'$\widehat{\Delta L}$': 'Predicted loss change',
        r'\mbox{Held-out} MAE': 'Held-out mean absolute error',
        'Best frozen form (R)': 'Best frozen form (retrospective)',
        'Intercept penalized?': 'Intercept penalty',
        'M, U, T': 'Before tying; unique; matrices',
        'G3; dense': 'Gemma three; dense', 'G4; dense': 'Gemma four; dense',
        'MG; dense': 'Muse Glimmer; dense', 'O3; dense': 'Open Language Model three; dense',
        'PN; dense': 'Pythia; dense', 'Q3; dense': 'Qwen three; dense',
        'Origin code': 'Fold rule',
    }
    for old, new in replacements.items():
        block = block.replace(old, new)
    if label == 'tab:pred_full':
        codes = dict(PLAIN_WORDS, c='constant', **{'0': 'zero change'})
        # Name each row-specific baseline in a small heading above its pair.
        block = re.sub(r'((?:\\textbf\{[\d.]+\}|[\d.]+)/(?:\\textbf\{[\d.]+\}|[\d.]+))\{\\scriptsize\\,([^}]+)\}',
                       lambda m: r'\textit{Relation / '+codes.get(m[2],m[2])+r'}\newline '+m[1].replace('/', ' / ')+(' (0)' if m[2]=='0' else ''), block)
        block = block.replace('Fold rule &', 'Fold rule: relation / input baseline / simple baseline &')
    if label in {'tab:main_final', 'tab:main_context'}:
        block = block.replace('Candidate MAE', 'Relation error (mathematics / code / question answering)')
        block = block.replace('Strongest frozen alternative (name, MAE)', 'Named baseline and error (mathematics / code / question answering)')
        block = block.replace(' & Gain &', ' & Gain (mathematics / code / question answering) &')
        block = block.replace(' & Delivered rule &', ' & Delivered rule (mathematics / code / question answering) &')
        # Make order visible horizontally, rather than a stack of anonymous values.
        block = re.sub(r'(\$[+-]?[\d.]+\$)\\newline ', r'\1 / ', block)
        block = block.replace('Median ', 'Development median ').replace('Zero ', 'Zero change ')
    if label == 'tab:p1v2':
        block = block.replace(' & A &', ' & Full development panel &').replace(' & B &', ' & Early stages only &')
        block = block.replace('Source & Protocol &', 'Source & Development data &')
    if label == 'tab:locked_rule':
        for code, words in {'P':'Pruning development', 'Q':'Channel quantization development', 'G':'Grouped quantization development', 'K':'Fixed-recipe student development'}.items():
            block = block.replace(' & '+code+' &', ' & '+words+' &')
        block = block.replace(r'$L_{0,c}$', 'Source dense loss').replace(r'$L_{S0,c}$', 'Initial student dense loss')
        block = block.replace(' & Fit & Anchor', ' & Development data & Dense reference loss')
    if label == 'tab:cap_conditioning':
        block = block.replace(' & A & B & C & D &', ' & Separate capability response & Shared response with scale & Shared response with offset & Shared response only &')
        block = block.replace(r'$\Delta_B$', 'Gain over scaled shared response')
    if label == 'tab:cond_audit':
        block = block.replace('A, 4 states', 'Separate response, 4 states').replace('B, 4 states', 'Scaled shared response, 4 states')
        block = block.replace(r'$\Delta$,', 'Gain,').replace(r'Pruning anchor $R^2$', r'Pruning anchor squared correlation ($R^2$)')
    if label == 'tab:quant2d_coef':
        for old,new in {'1':'Constant (1)', '$u$':'Bit response', '$v$':'Group-size response', '$uv$':'Bit and group interaction', '$u^2$':r'Squared bit response ($u^2$)'}.items():
            block = block.replace(' & '+old+' &', ' & '+new+' &')
    if label == 'tab:quant_ident':
        block = block.replace('Bit &', 'Bit-width test &').replace('Granularity &', 'Group-size test &').replace('Joint &', 'Joint test &')
        block = block.replace(r'Without $u^2$', r'Without squared bit response ($u^2$)')
    if label == 'tab:shared_structure':
        block = block.replace(r'$\beta_c,\gamma_c$', 'Source coefficients and exponent').replace(r'$\beta_c,p_c,q_c$', 'Source coefficients and exponents')
        block = block.replace(r'$\beta_c$', 'Source coefficients').replace(r'$\gamma$', 'Exponent')
    if label == 'tab:p3_check':
        block = block.replace('State & Math & Code & QA', 'State & Mathematics: primary / arithmetic word problems & Code: primary / HumanEval & Question answering: primary / TriviaQA')
        block = block.replace('; $', ' / $')
    if label == 'tab:panel_prune':
        block = block.replace(r'$d^{*}_{\mathrm{math}}$', 'Density at loss-increase threshold')
        block = block.replace(r'$\Delta L_c$ at $d=0.7$', 'Loss change at density 0.7')
        block = block.replace('QA least and most damaged', 'Question answering: least / measured; most / measured')
        block = block.replace(r'$\,|\,$', '; ')
        block = block.replace('Minimum QA response ($d$)', 'Minimum question-answering response (density)')
    if label == 'tab:panel_quant':
        block = block.replace(r'$\Delta L_c$ at', 'Loss change at')
    if label == 'tab:rule-confirm':
        block = block.replace(' & Feasible &', ' & Feasible / total cells &')
        block = block.replace('Method agreement', 'Method agreement (percent)').replace('Set contains oracle method', 'Set contains oracle method (percent)')
    if label == 'tab:rule-confirm-candidate-sizes':
        block = block.replace('Set contains oracle method', 'Oracle method: covered / total (percent)').replace('Set contains oracle configuration', 'Oracle configuration: covered / total (percent)')
    if label == 'tab:selection-feasible':
        block = block.replace(' & Own & Common', ' & Policy-feasible cells & Shared feasible cells')
        block = block.replace(r'(\%)', '(percent)')
    if label == 'tab:v56_forms':
        block = re.sub(r'leave-one-run-out (?=Math|Code|QA)', 'Run held out: ', block)
        block = re.sub(r'leave-one-student-out (?=Math|Code|QA)', 'Student held out: ', block)
        # All six headers explicitly specify each paired value's position.
        block = re.sub(r'((?:Run|Student) held out: (?:Math|Code|QA))', r'\1 (error / signed bias)', block)
        block = re.sub(r'(?<=\d); (?=[+-]\d)', ' / ', block)
    if label == 'tab:v56_condition':
        block = block.replace('Parameters, shared and specific', 'Parameters: shared / capability-specific')
    if label == 'tab:candidate-coverage':
        for heading in ['Pruning', 'Per-channel round-to-nearest quantization', 'Grouped round-to-nearest quantization', 'distillation students']:
            block = block.replace(' & '+heading+' &', ' & '+heading+' (count; minimum ratio) &')
            block = block.replace('{'+heading+'}', '{'+heading+' (count; minimum ratio)}')
    # Words for scalar setting labels; the printed scalar itself is untouched.
    for symbol, words in {'d':'density', 'b':'bit width', 'g':'group size', 'U':'pool size', 'T':'supervised tokens', 'n':'measured cells'}.items():
        block = re.sub(r'\$'+symbol+r'(?:\{=\}|=)([^$]+)\$', lambda m: words+' '+m[1], block)
        if symbol != 'n':
            block = block.replace('$'+symbol+'$', words)
        block = re.sub(r'\$'+symbol+r'(\\(?:in|ge|le)(?![A-Za-z])[^$]+)\$', lambda m: words+' $'+m[1]+'$', block)
    block = block.replace('size$\\uparrow$', 'larger size')
    block = _reader_words(block)
    return block


def plain_language(text):
    """Make all table surfaces readable without touching the excluded main table."""
    if r'\label{tab:main-prediction-v2}' in text:
        return text
    try:
        from .paper_table_layout import TABULAR, _tabular
    except ImportError:
        from paper_table_layout import TABULAR, _tabular
    current = next((s for s in re.findall(r'\\label\{([^}]+)\}', text) if s in PLAIN_CAPTIONS), None)
    def rewrite(match):
        nonlocal current
        block = match.group()
        labels = re.findall(r'\\label\{([^}]+)\}', block)
        current = next((s for s in labels if s in PLAIN_CAPTIONS), current)
        if current is None:
            return block
        # Rewrite captions first, then prose/cells/notes through one vocabulary.
        block = _reader_caption(block, current)
        block = _reader_body(block, current)
        block = _reader_details(block, current)
        try:
            from .paper_table_layout import reader_layout
        except ImportError:
            from paper_table_layout import reader_layout
        block = reader_layout(block, current)
        return block
    text = _reader_words(TABLE.sub(rewrite, text))
    for old, new in {
        r'\par D:': r'\par Development fit:',
        r'\par O:': r'\par Ordinary least squares:',
        r' R: $': r' Ridge regression: $',
        r'\mathrm{SSE}': r'\text{sum of squared errors}',
        r'\textbf{Development data.} P:': r'\textbf{Development data.} Pruning:',
        ' Q: ': ' Channel quantization: ', ' G: ': ' Grouped quantization: ',
        ' K: ': ' Fixed-recipe students: ',
    }.items():
        text = text.replace(old, new)
    return text


def _reader_details(block, label):
    """Context-specific descriptors and explicit orders for multivalue cells."""
    block = block.replace('per-density regression (per-density regression)', 'per-density regression')
    block = block.replace('per-density regression per-density regression', 'per-density regression')
    block = block.replace('the per-density regression per-density regression', 'per-density regression')
    block = block.replace(r'Positive $\Delta_B=B-A$ favors A',
                          'A positive gain favors the separate capability response')
    if label == 'tab:quant_threeway':
        block = block.replace(r'\textbf{D} is the frozen development leave-one-state-out selection',
                              'The frozen development selection uses leave one state out')
        block = block.replace('F lists all frozen candidates;', 'All frozen candidates are listed;')
        block = block.replace(r'\textbf{R} is the \mbox{post-hoc} recommended rule',
                              'the retrospective recommendation is the rule')
    if label in {'tab:pred_source', 'tab:pred_config_prune', 'tab:pred_config_qd'}:
        block = block.replace('what is held out; fold rule', 'what is held out; fold rule (relation / input baseline / simple baseline)')
        block = block.replace('Protocol A', 'The full-development protocol').replace('protocol B', 'the early-stage protocol')
    if label == 'tab:model_arch':
        block = re.sub(r'Class abbreviations:.*?Qwen3ForCausalLM\.', '', block)
        block = block.replace('Pythia T equals', 'The Pythia matrix count equals')
        block = block.replace(' & Convention', '').replace(' & Before tying; unique; matrices', '')
        block = block.replace(r'\multicolumn{7}', r'\multicolumn{6}')
    if label == 'tab:p1v2':
        block = block.replace('Protocol A', 'The full-development protocol').replace('protocol B', 'the early-stage protocol')
    if label == 'tab:p2v2_test':
        block = block.replace('$n$', 'Measured points').replace(' (R)', ' (retrospective)')
    if label == 'tab:distill_forms_audit':
        block = block.replace('Free parameters $k$', 'Parameter count')
        block = block.replace(' & $n$ &', ' & Logarithmic student size &')
        block = block.replace(' & $z_L$ &', ' & Standardized initial loss &')
        block = block.replace(' & $z_N$ &', ' & Standardized student size &')
        block = block.replace('$4+1=5$', '4 linear plus 1 saturation, total 5')
        block = re.sub(r'\$T_\\star=(\d+)\$', lambda m: 'saturation budget '+m[1]+' tokens', block)
        # Coefficient roles follow the exact formula in the first panel.
        roles = {
            'Constant': ['intercept'],
            'Constant with student size': ['intercept', 'student size'],
            'Budget response': ['budget'], 'Budget with student size': ['budget', 'student interaction'],
            'Reuse response': ['reuse'], 'Reuse with student size': ['reuse', 'student interaction'],
            'Joint budget and pool response': ['budget', 'budget curvature', 'pool interaction'],
            'Joint with student size': ['budget', 'budget curvature', 'pool interaction', 'student interaction'],
            'descriptor-modulated reuse with initial loss': ['reuse', 'initial-loss interaction'],
            'descriptor-modulated reuse with student size': ['reuse', 'student-size interaction'],
            'saturating budget and reuse with initial loss': ['saturation', 'initial-loss saturation', 'reuse', 'initial-loss reuse'],
            'saturating budget and reuse with student size': ['saturation', 'student-size saturation', 'reuse', 'student-size reuse'],
            'Budget only': ['intercept', 'budget'], 'Reuse only': ['intercept', 'reuse'],
            'Response surface with initial loss': ['intercept', 'budget', 'reuse', 'initial loss'],
            'Response surface with student size': ['intercept', 'budget', 'reuse', 'student size'],
        }
        lines = []
        for line in block.splitlines():
            cells = line.split(' & ')
            if len(cells) == 5 and cells[0].strip('{}') in roles:
                cells[1] = '; '.join(roles[cells[0].strip('{}')])
                # Legacy stacked vectors used whitespace for some separators.
                cells[2:] = [re.sub(r'(?<=\d) {2,}(?=[+\-\d])', ', ', c) for c in cells[2:]]
                line = ' & '.join(cells)
            lines.append(line)
        block = '\n'.join(lines)
        if 'Coefficient order &' in block:
            block = _scalar_coefficient_rows(block, roles)
    if label in {'tab:models', 'tab:final', 'tab:pred_source'}:
        for symbol, name in {
            r'$N_0$': 'source parameter count', r'$D_0$': 'pretraining tokens',
            r'$L_{0,c}$': 'initial loss', r'$D_U$': 'unique-pool tokens',
            r'$E=T/D_U$': 'reuse is supervised tokens divided by unique-pool tokens',
            r'$(b,g)$': 'bit-width and group-size',
            r'$\{N_0,L_0,D_0\}$': 'source size, initial loss, and pretraining tokens',
            r'$\delta_c$': 'student loss change',
        }.items():
            block = block.replace(symbol, name)
        block = block.replace('per-$d$', 'per-density')
    if label == 'tab:models':
        block = block.replace('Predictor &', 'Prediction form &').replace('Target calibration &', 'Target measurements &')
        block = block.replace('5 (per-density regression: 20)', '5 (per-density regression: 20)')
        block = block.replace('Eq.~', 'Equation~', 1)  # Keep the protected quantization reference verbatim.
    if label == 'tab:pred_full':
        block = block.replace(r'$\delta_c$', 'student loss change')
        block = block.replace('all density', 'all densities').replace('sources +', 'sources and')
    block = block.replace(r'$3{\times}3$', '3 sizes by 3 stages').replace(r'$3\times3$', '3 sizes by 3 stages')
    block = block.replace('Mean mean absolute error', 'Capability mean absolute error')
    if label == 'tab:v56_condition':
        block = re.sub(r'(?<=\d); (?=\d)', ' / ', block)
    if label == 'tab:panel_prune':
        block = block.replace('least / measured; most / measured', 'least / measured; most / measured')
    return block


def _scalar_coefficient_rows(block, roles):
    """One named coefficient per row, retaining the original capability order."""
    block = block.replace('Form & Coefficient order & Mathematics & Code & Question answering',
                          'Form & Capability & Coefficient or budget & Value')
    lines = []
    for line in block.splitlines():
        cells = line.split(' & ')
        name = cells[0].strip('{}')
        if len(cells) != 5 or name not in roles:
            lines.append(line)
            continue
        for capability, cell in zip(['Mathematics', 'Code', 'Question answering'], cells[2:]):
            values = re.findall(r'[+-]?\d+(?:\.\d+)?', cell)
            names = roles[name] + (['Saturation budget in tokens'] if 'saturation budget' in cell else [])
            assert len(values) == len(names), (name, cell, names, values)
            for role, value in zip(names, values):
                lines.append(' & '.join([name, capability, role.capitalize(), value])+r' \\')
    return '\n'.join(lines)
