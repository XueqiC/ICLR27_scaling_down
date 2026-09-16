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
    "tab:models": "Rows identify prediction families; columns give predictors, parameter counts per capability, development settings, target calibration counts, tested ranges, and coefficient sources. Loss responses are in nats per native token; density and reuse are dimensionless, bit widths are in bits, and budgets are in tokens. This is a post-hoc summary of development fits and frozen prediction tests. A2 denotes per-density regression with interpolation; the numbered registers identify the original experiment records.",
    "tab:model_arch": "Rows identify models within each cohort; columns give the architecture class and parameter-count conventions. This is a post-hoc architecture audit. The numbered development records identify the original fits. MoE means mixture of experts.",
    "tab:round3_coef": "Rows identify capabilities; columns give the fitted exponent and coefficients of the intercept, source size, initial loss, and pretraining tokens, respectively. These development coefficients were frozen for prediction. Loss is in nats per native token; standardized covariates and the exponent are dimensionless.",
    "tab:quant2d_coef": "Rows identify capabilities and basis terms; columns give coefficients of the intercept, source size, initial loss, and pretraining tokens. These development coefficients were frozen for prediction. Loss is in nats per native token, source size is a parameter count, and pretraining exposure is in tokens; the fitted covariates are standardized.",
    "tab:v53_loso": "Rows identify density subsets and candidate predictors; columns give mean absolute error by capability and its capability mean. A1 denotes the linear power form, and A2 denotes per-density regression with interpolation.",
    "tab:v55_loso": "Rows identify candidate predictors and columns identify capabilities. All entries are development mean absolute errors.",
    "tab:quant_ident": "Rows identify control forms and evaluation panels; capability columns report effective degrees of freedom above and mean absolute errors below. V55 identifies the original grouped-quantization development register.",
    "tab:v56_forms": "Rows identify forms and columns give parameter counts and capability errors under each development holdout. Each paired entry gives mean absolute error followed by signed bias. The descriptor suffix L0 means initial loss and logN means logarithmic student size.",
    "tab:v56_condition": "Rows identify response structures and capabilities; columns compare parameter counts, errors, gains, and the stated gain threshold in this development analysis. F2 denotes a saturating budget response plus a logarithmic reuse response, with a student descriptor. Paired parameter counts give the shared and capability-specific totals, respectively.",
    "tab:distill_forms_audit": "This is a post-hoc audit of frozen prediction forms. Loss responses and coefficient-scaled responses are in nats per native token. F1 denotes descriptor-modulated reuse; F2 adds a saturating budget term. L0 denotes initial loss and logN denotes logarithmic student size. The numbered audit and freeze identifiers refer to the original experiment records.",
    "tab:shared_structure": "Rows identify response families, sharing comparisons, and calibration evaluations; columns describe shared and varying terms and compare errors and interval summaries. This is a post-hoc analysis. K0 denotes no compressed-target calibration and K1 denotes one compressed-target calibration measurement. F2 denotes the saturating budget and logarithmic reuse form; the numbered records identify the original development fits.",
    "tab:cap_conditioning": "Rows identify compression families and capabilities; columns compare the defined response variants and their paired error reduction. This is a post-hoc ablation using development fits on frozen confirmation panels. CI means confidence interval.",
    "tab:cond_audit": "Rows identify compression families above and capabilities below; columns give dimensionless scales and squared correlations above, and error comparisons below. This is a post-hoc audit. A denotes a separate response per capability; B denotes a shared response with a fitted capability scale. CI means confidence interval; the numbered records identify the original audits and confirmations.",
    "tab:main_final": "Rows identify frozen candidates and test panels; columns give candidate errors, the strongest frozen alternative, error differences, delivered rules, and their timing and scope. F1 denotes descriptor-modulated reuse and F2 denotes the saturating budget and reuse form; L0 denotes initial loss. Pool sizes count traces per domain and budgets count supervised tokens.",
    "tab:main_context": "Rows identify student and pool tests; columns give candidate errors, the strongest frozen alternative, error differences, delivered rules, and their timing and scope. Pool sizes count traces per domain and budgets count supervised tokens. These are frozen predictions with post-hoc ranking and rule selection.",
    "tab:pred_full": "Rows identify prediction tests; columns identify the held-out axis, the defined provenance code, and capability errors. Each error pair lists the candidate followed by the strongest baseline.",
    "tab:pred_source": "Rows identify tests and predictors; columns report capability errors. MAE means mean absolute error. LoRA denotes low-rank adaptation.",
    "tab:pred_config_prune": "Rows identify frozen prediction tests and predictors; columns give mean absolute errors in nats per native token by capability. Parentheses give baseline-minus-candidate improvement; bold marks the lowest error. The provenance letters mean: P, prediction frozen before measurement; F, baseline specified before measurement; R, post-hoc baseline; A, all specified forms reported. A1 denotes the linear power form and A2 denotes per-density regression with interpolation. The numbered pair identifier names the original registration.",
    "tab:pred_config_qd": "Rows identify frozen prediction tests and predictors; columns give mean absolute errors in nats per native token by capability. Parentheses give baseline-minus-candidate improvement; bold marks the lowest error. The provenance letters mean: P, prediction frozen before measurement; F, baseline specified before measurement; R, post-hoc selection; A, all specified forms reported. Pool sizes count traces per domain.",
    "tab:p1v2": "Rows identify source states, protocols, and compression regimes; columns give candidate errors and the predictor with the lowest observed error. A1 denotes the linear power form and A2 denotes per-density regression with interpolation. The numbered pair identifier names the original registration; the best-predictor column is a post-hoc ranking.",
    "tab:round3_prune": "Rows identify frozen predictors; columns give parameter counts and capability errors in nats per native token. MAE means mean absolute error.",
    "tab:round3_quant": "Rows identify frozen predictors; columns give parameter counts and capability mean absolute errors for each test panel. The numbered round identifies the original confirmation.",
    "tab:quant_threeway": "Within each panel, rows identify predictor status and form; columns give capability mean absolute errors. The numbered confirmation identifies the original experiment record.",
    "tab:prune_repeat": "Rows identify frozen predictors; columns give mean absolute errors for each capability and their mean. V53 identifies the original pruning development register.",
    "tab:p2v2_test": "Rows identify students, pool roles, and capabilities; columns give errors for the development-selected form and baselines, the post-hoc best frozen form, and the number of test points. Pool sizes count traces per domain. R denotes post-hoc ranking.",
    "tab:distill_paired": "Rows identify students and capabilities within each pool group; columns give candidate and baseline mean absolute errors, paired differences, and relative improvements in percent. CI means confidence interval. This is a post-hoc paired analysis of frozen predictions.",
    "tab:distill_confirm": "Rows identify students and capabilities; columns give selected and baseline forms, their mean absolute errors, and paired improvements. These are frozen prediction results. CI means confidence interval. Pool size counts traces per domain and budgets count supervised tokens; the numbered confirmation identifies the original record.",
    "tab:p3_check": "Rows identify measured compression states; columns give capability loss changes, with the primary benchmark followed by the secondary benchmark. This is a post-hoc measurement-scope check, with states specified before secondary measurement. Pool sizes count traces per domain; the numbered check identifies the original registration.",
    "tab:musique_scope": "Rows identify students, pools, and checkpoints; columns give dimensionless reuse counts, trajectory counts, and benchmark loss changes. This is a post-hoc measurement-scope audit. Pool sizes count traces per domain; the numbered state check identifies the original registration.",
    "tab:qa_scope": "Rows identify compression states; paired columns give loss and change from dense for each benchmark. This is a post-hoc measurement-scope audit of pre-specified states. The numbered control identifies the original evaluation record.",
    "tab:panel_prune": "Rows identify models; columns give capability loss changes, the Math damage threshold, QA damage rankings, and the minimum QA response. This is a descriptive development panel with prospective additions marked by daggers. A dash denotes an unmeasured setting.",
    "tab:panel_quant": "Rows identify models and series; columns give capability loss changes at the displayed bit widths and list all measured bit widths in bits. This is a descriptive development panel with prospective additions marked by daggers.",
    "tab:selection-feasible": "Rows identify policies and objectives; columns give feasibility coverage, mean regret, and oracle-method agreement. This is a post-hoc development evaluation. Own means the cells feasible for that policy; common means cells feasible for every policy.",
    "tab:rule-confirm": "Rows identify objectives and policies; columns give feasible-cell counts, mean regret, oracle-method agreement, and candidate-set coverage. Policy predictions were frozen before the new measurements; set coverage is a post-hoc diagnostic. The numbered panel and student records identify the original experiments.",
    "tab:rule_decomp": "Rows identify objectives and policies; columns compare the stated source subsets and all states. This is a post-hoc decomposition of frozen selection predictions.",
    "tab:rule-confirm-by-state": "Rows identify source states and objectives; columns compare policy regret and selected-configuration counts. This is a post-hoc decomposition of frozen selection predictions. The numbered panel identifies the original experiment.",
    "tab:rule-confirm-candidate-sizes": "Rows identify objectives and policies; columns give method-count distributions and oracle coverage. This is a post-hoc diagnostic of candidate sets based on frozen development errors.",
    "tab:candidate-coverage": "Rows identify source states; columns give availability by compression method and the minimum quantization storage ratio. Counts are numbers of configurations and storage ratios are dimensionless. This is a post-hoc inventory of the development selection panel.",
    "tab:locked_rule": "Rows identify compression families, capabilities, and source-state status; columns specify the response predictor, development fit, and dense-loss anchor. Absolute losses are in nats per native token. This rule was frozen before selection confirmation. Fit keys P, Q, G, and K denote pruning, channel quantization, grouped quantization, and distillation development data, respectively; numbered predictor records identify the original development fits.",
    "tab:final": "Rows identify compression families; columns give delivered predictors, inputs, prediction and rule-selection status, observed gains, and excluded settings. This is a post-hoc summary of frozen prediction evidence; A2 denotes per-density regression with interpolation.",
}


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
    return text


def table_text(text):
    """Apply editorial changes to tables without touching surrounding prose."""
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    fallback = next((label for label in labels if label in CAPTION_NOTES), None)
    return fit_table_height(TABLE.sub(lambda m: _table(m.group(), fallback), text))


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
