#!/usr/bin/env python3
"""CPU-only decomposition of existing V12/V16 losses and the V21 size claim.

No model/dataset loading. V12 and V16 are separate measurements, never mixed.
Run with --dry-run to list every input without writing anything; the default
run writes only results/v22-distill-decomp. --update-law-report additionally
updates a marked audit section of LAW_FIT_REPORT.md, never the ledger.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

try:
    from . import v21_distill_law as v21
except ImportError:
    import v21_distill_law as v21

ROOT = Path(__file__).resolve().parents[1]
CAPS = v21.CAPABILITIES
REFERENCES = {"gemma3": "gemma3-27b", "qwen3": "Qwen3-4B", "olmo3": "olmo3-32b"}
BEGIN = "<!-- V22_DISTILL_DECOMP_START -->"
END = "<!-- V22_DISTILL_DECOMP_END -->"


def read_json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected an object: {path}")
    return result


def family_of(model: str) -> str:
    for family in REFERENCES:
        if model.lower().startswith(family):
            return family
    raise ValueError(f"no registered same-family reference for {model}")


def finite(value: object) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite loss")
    return value


def decompose(own_dense: float, distilled: float, reference: float) -> dict:
    own_dense, distilled, reference = map(finite, (own_dense, distilled, reference))
    baseline = own_dense - reference
    gain = distilled - own_dense
    total = distilled - reference
    error = total - baseline - gain
    if not math.isclose(error, 0.0, abs_tol=1e-12):
        raise ValueError("decomposition identity failed")
    return {"own_dense_loss": own_dense, "distilled_loss": distilled,
            "reference_loss": reference, "baseline_gap": baseline,
            "distillation_gain": gain, "source_referenced_gap": total,
            "identity_error": error}


def input_paths(results: Path) -> list[Path]:
    losses = sorted((results / "v12-distill").glob("*/*/eval.json"))
    losses += sorted((results / "v16-style-residual").glob("*/*/residual.json"))
    # Pythia controlled-panel distill cells (v36/v39 source-transfer study) are not heterogeneous
    # finished models and have no same-family reference here; exclude them from the v22 decomposition.
    losses = [p for p in losses if not p.parent.parent.name.lower().startswith("pythia-")]
    if not losses:
        raise FileNotFoundError("no V12/V16 loss artifacts found")
    families = {family_of(path.parent.parent.name) for path in losses}
    paths = losses + [results / "v6-capability-geometry" / REFERENCES[f] / "prune_losses.json"
                      for f in sorted(families)]
    paths.append(results / "v21-distill-law/summary.json")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(paths)


def path_label(path: Path, results: Path) -> str:
    return "results/" + path.relative_to(results).as_posix()


def load_decompositions(results: Path) -> list[dict]:
    rows = []
    for version, dirname, filename in [(12, "v12-distill", "eval.json"),
                                       (16, "v16-style-residual", "residual.json")]:
        for path in sorted((results / dirname).glob(f"*/*/{filename}")):
            model, run = path.parent.parent.name, path.parent.name
            if model.lower().startswith("pythia-"):
                continue  # controlled-panel cells (v36/v39), not heterogeneous finished models
            payload = read_json(path)
            if payload.get("version") != version:
                raise ValueError(f"unexpected measurement version: {path}")
            if payload.get("student_tag") != model or payload.get("run_name") != run:
                raise ValueError(f"path/metadata identity mismatch: {path}")
            family = family_of(model)
            ref_model = REFERENCES[family]
            ref_path = results / "v6-capability-geometry" / ref_model / "prune_losses.json"
            reference = read_json(ref_path)["1.0"]
            dense = payload["dense"] if version == 12 else payload["dense"]["L_c"]
            distilled = payload["post_training"] if version == 12 else payload["distilled"]["L_c"]
            for capability in CAPS:
                terms = decompose(dense[capability], distilled[capability], reference[capability])
                recorded = payload["delta"][capability] if version == 12 else payload["deltas"][capability]["dL_c"]
                if not math.isclose(terms["distillation_gain"], finite(recorded), abs_tol=1e-10, rel_tol=0):
                    raise ValueError(f"recorded gain disagrees with paired losses: {path}")
                rows.append({"row_id": f"v{version}|{model}|{run}|{capability}",
                             "family": family, "student": model, "run": run,
                             "capability": capability, "eval_version": version,
                             "teacher": payload["teacher"], "recipe": payload["recipe"],
                             "n_per_domain": payload["n_per_domain"],
                             "training_mode": payload.get("training_mode", "not recorded in V16"),
                             "reference_model": ref_model,
                             "reference_scope": "supplementary OLMo reference" if family == "olmo3" else "directive reference",
                             "loss_path": path_label(path, results),
                             "reference_path": path_label(ref_path, results),
                             "unit": "CE nats per model-token (completion-token weighted)",
                             **terms})
    return rows


def reference_correction(alpha: float, gemma_ref_loss: float, qwen_ref_loss: float) -> dict:
    """Correction to a Gemma prediction expressed in Qwen's original y/x units.

    x=-log(Ns/Nref). x_q_common=x_q+log(27/4),
    y_q_common=y_q+Lref_q-Lref_g. Thus p_q_corrected=p_q+k.
    This is accounting in recorded token units, not a causal family estimate.
    """
    loss = gemma_ref_loss - qwen_ref_loss
    coordinate = alpha * math.log(27.0 / 4.0)
    return {"reference_loss_shift": loss, "coordinate_shift": coordinate,
            "known_correction": loss + coordinate}


def component_fits(rows: list[dict]) -> dict:
    fits = {}
    for component in ("source_referenced_gap", "baseline_gap", "distillation_gain"):
        converted = [{"row_id": r["row_id"], "model": r["student"], "family": r["family"],
                      "capability": r["capability"], "r_storage": r["r_storage"],
                      "observed": r[component]} for r in rows]
        gemma = [r for r in converted if r["family"] == "gemma3"]
        qwen = [r for r in converted if r["family"] == "qwen3"]
        shared = v21.evaluate_shared_no_refit(gemma, qwen)
        hierarchical = v21.evaluate_hierarchical_intercept(shared["fit"], qwen)
        fits[component] = {"shared": shared, "rotating_intercept": hierarchical}
    return fits


def analyze_c21(rows: list[dict], recorded: dict) -> dict:
    selected = []
    for r in rows:
        spec = v21.SIZE_LADDERS.get(r["family"], {})
        ratio = spec.get("students", {}).get(r["student"])
        if r["eval_version"] == 12 and r["run"] == "gpt-5.6-luna_full_600" and ratio is not None:
            selected.append({**r, "r_storage": ratio})
    if len(selected) != 18:
        raise ValueError("C21 reproduction needs six complete V12 student cells (18 capability rows)")
    fits = component_fits(selected)
    total = fits["source_referenced_gap"]
    original = recorded["source_referenced_size_law"]
    for key, computed in [("shared_no_refit", total["shared"]),
                          ("hierarchical_intercept", total["rotating_intercept"])]:
        if not math.isclose(computed["metrics"]["mae"], original[key]["metrics"]["mae"], abs_tol=1e-10, rel_tol=0):
            raise ValueError(f"V21 {key} MAE no longer matches current inputs")
        expected = {(r["model"], r["capability"]): r for r in original[key]["records"]}
        for r in computed["records"]:
            old = expected[(r["model"], r["capability"])]
            for field in ("observed", "predicted"):
                if not math.isclose(r[field], old[field], abs_tol=1e-10, rel_tol=0):
                    raise ValueError(f"V21 {key} {field} record mismatch")

    attribution, corrected = [], []
    for cap in CAPS:
        params = {term: fit["shared"]["fit"]["parameters"][cap] for term, fit in fits.items()}
        alpha = params["source_referenced_gap"]["alpha"]
        refs = {r["family"]: r["reference_loss"] for r in selected if r["capability"] == cap}
        correction = reference_correction(alpha, refs["gemma3"], refs["qwen3"])
        records = [r for r in total["shared"]["records"] if r["capability"] == cap]
        offsets = [r["observed"] - r["predicted"] for r in records]
        remaining = [offset - correction["known_correction"] for offset in offsets]
        original_mae = sum(map(abs, offsets)) / 2
        corrected_mae = sum(map(abs, remaining)) / 2
        attribution.append({"capability": cap, **correction,
                            "gemma_alpha_total": alpha,
                            "gemma_alpha_baseline": params["baseline_gap"]["alpha"],
                            "gemma_alpha_gain": params["distillation_gain"]["alpha"],
                            "baseline_fraction_of_alpha": params["baseline_gap"]["alpha"] / alpha,
                            "mean_original_offset_descriptive": sum(offsets) / 2,
                            "mean_remaining_offset_descriptive": sum(remaining) / 2,
                            "original_no_refit_mae": original_mae,
                            "reference_corrected_no_refit_mae": corrected_mae,
                            "mae_reduction_fraction": 1 - corrected_mae / original_mae})
        for r in records:
            predicted = r["predicted"] + correction["known_correction"]
            corrected.append({**r, "predicted": predicted, "absolute_error": abs(predicted-r["observed"])})

    folds = []
    for fold in total["rotating_intercept"]["folds"]:
        for cap in CAPS:
            component_offsets = {}
            for component, result in fits.items():
                matched = next(f for f in result["rotating_intercept"]["folds"]
                               if f["held_out_model"] == fold["held_out_model"])
                component_offsets[component] = matched["qwen_intercepts"][cap]
            correction = next(r for r in attribution if r["capability"] == cap)
            folds.append({"held_out_model": fold["held_out_model"],
                          "calibration_model": fold["calibration_models"][0], "capability": cap,
                          "total_offset": component_offsets["source_referenced_gap"],
                          "baseline_offset": component_offsets["baseline_gap"],
                          "gain_offset": component_offsets["distillation_gain"],
                          "known_correction": correction["known_correction"],
                          "remaining_offset": component_offsets["source_referenced_gap"] - correction["known_correction"]})
    # V16 sensitivity replaces complete V12 pairs only, retaining V12 Qwen
    # where there is no V16 measurement. It is explicitly not the C21 score.
    lookup = {(r["student"], r["run"], r["capability"]): r for r in rows if r["eval_version"] == 16}
    sensitivity = [{**lookup.get((r["student"], r["run"], r["capability"]), r),
                    "r_storage": r["r_storage"]} for r in selected]
    return {"rows": selected, "component_fits": fits, "reference_attribution": attribution,
            "reference_corrected_no_qwen_student_fit": {"records": corrected, "metrics": v21.prediction_metrics(corrected)},
            "fold_intercept_decomposition": folds,
            "v16_pairs_with_v12_qwen_sensitivity": component_fits(sensitivity),
            "parameter_scope": "one Qwen offset per capability: three additional parameters per rotation",
            "independent_size_contrasts_per_capability": 1,
            "verdict": "PARTIAL: calibrated source-referenced size-gap predictor; distillation-specific family law unestablished"}


def suggested_wording(audit: dict) -> str:
    fits = audit["component_fits"]["source_referenced_gap"]
    return (
        "C21 — PARTIAL / EXPLORATORY. On the six GPT full-600 students, a Gemma fit of the "
        "source-referenced post-training loss gap predicts Qwen with MAE "
        f"{fits['shared']['metrics']['mae']:.3f} nats/token (4/6 signs). "
        "Calibrating one Qwen offset per capability on one size and predicting the other, in both rotations, "
        f"gives MAE {fits['rotating_intercept']['metrics']['mae']:.3f} (6/6 signs). "
        "This is a calibrated two-size transfer diagnostic, with three offsets per fold and only one "
        "independent size contrast per capability. About 99%/90% of the Gemma math/code log-size slopes "
        "are dense baseline-gap effects; they are not distillation gains. Unequal reference losses and "
        "27B-versus-4B coordinates partly account for the offsets in recorded token units. "
        "A distillation-specific shared exponent or intrinsic family intercept is not established. "
        "A family-specific exponent has no held-out test with two Qwen sizes. D ladders are non-monotone "
        "and the 4B ladder changes training mode; no clean D exponent is supported. QA findings are "
        "loss-space only. Evidence: results/v22-distill-decomp/report.md and V21 recorded folds."
    )


def render_report(summary: dict) -> str:
    audit = summary["c21_audit"]
    fits = audit["component_fits"]
    lines = ["## Stage A / V22: distillation decomposition and C21 reassessment", "",
             "**Verdict: PARTIAL / EXPLORATORY.** The held-out 0.106 calculation is reproducible, "
             "but it supports a calibrated predictor of the combined baseline-plus-training gap. "
             "It does not establish a distillation-specific hierarchical family law.", "",
             "### Paired loss accounting", "",
             "`L(S_KD)-L(B_ref) = [L(S0)-L(B_ref)] + [L(S_KD)-L(S0)]`.", "",
             "The first term is the student's own dense baseline gap: pre-existing size/quality, not "
             "distillation. The signed DISTILLATION GAIN is negative when training lowers loss and positive "
             "when it damages loss. B_ref is a same-family comparison checkpoint, not the API teacher. "
             "References are Gemma-3 27B and Qwen-3 4B; the single OLMo student is included separately "
             "against OLMo-3 32B and excluded from C21 fits.", "",
             f"Coverage: {summary['counts']['v12_runs']} V12 runs and {summary['counts']['v16_runs']} V16 "
             "remeasurements; all three capabilities for each. `decomposition.csv` and `summary.json` "
             "contain every term and its artifact paths. V12 reproduces C21; V16 pairs are retained as "
             "separate measurements. No V16 generic residual is substituted for capability loss.", "",
             "The following table is the C21 GPT/full/600 slice; all other teachers, recipes and budgets "
             "are in the complete table below.", "",
             "| Student | Capability | BASELINE GAP | DISTILLATION GAIN | Total source gap |",
             "|---|---|---:|---:|---:|"]
    for r in audit["rows"]:
        lines.append(f"| {r['student']} | {r['capability']} | {r['baseline_gap']:+.6f} | {r['distillation_gain']:+.6f} | {r['source_referenced_gap']:+.6f} |")
    lines += ["", "### Where the fitted size slope comes from", "",
              "Identical OLS designs imply `alpha_total = alpha_baseline + alpha_gain` exactly. "
              "These are descriptive fits on the four Gemma sizes, not new held-out exponent estimates.", "",
              "| Capability | Total alpha | Baseline alpha | Gain alpha | Baseline / total |",
              "|---|---:|---:|---:|---:|"]
    for r in audit["reference_attribution"]:
        lines.append(f"| {r['capability']} | {r['gemma_alpha_total']:.6f} | {r['gemma_alpha_baseline']:.6f} | {r['gemma_alpha_gain']:.6f} | {100*r['baseline_fraction_of_alpha']:.1f}% |")
    lines += ["", "The negative QA baseline fraction means opposing components, not an explained-variance share. "
              "Math/code slopes largely reflect dense size/quality. The QA total slope hides cancellation.", "",
              "| Response fitted and tested | Gemma→Qwen no-refit MAE | Rotating per-capability intercept MAE |",
              "|---|---:|---:|"]
    for name, result in fits.items():
        lines.append(f"| {name} | {result['shared']['metrics']['mae']:.6f} | {result['rotating_intercept']['metrics']['mae']:.6f} |")
    lines += ["", "Baseline and gain prediction errors can cancel in their sum; success on the total "
              "therefore does not validate prediction of the training effect.", "",
              "### Reference and coordinate accounting before a family interpretation", "",
              "Let `x=-log(r_storage)`. Moving Qwen to the Gemma 27B coordinate gives "
              "`x_common=x_Q+log(27/4)`; moving its loss gap to the Gemma source gives "
              "`y_common=y_Q+L_ref,Q-L_ref,G`. Thus a reference-corrected prediction in the original "
              "Qwen units is `p_Q + k_c`, where `k_c=(L_ref,G-L_ref,Q)+alpha_c*log(27/4)`. "
              "This uses only dense reference losses and Gemma-fitted alpha, no Qwen student outcomes. "
              "It retains V21's registered ratios (.010/.036/.157/.445 and .150/.425); these are not "
              "exactly nominal student B/27 (implied Gemma sizes .270/.972/4.239/12.015B). "
              "This is a coordinate-origin sensitivity, not a harmonized parameter-count refit.", "",
              "| Capability | Ref-loss shift | Log-coordinate shift | Known correction k | Mean original offset* | Mean remaining offset* | No-refit MAE after correction |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in audit["reference_attribution"]:
        lines.append(f"| {r['capability']} | {r['reference_loss_shift']:+.6f} | {r['coordinate_shift']:+.6f} | {r['known_correction']:+.6f} | {r['mean_original_offset_descriptive']:+.6f} | {r['mean_remaining_offset_descriptive']:+.6f} | {r['reference_corrected_no_refit_mae']:.6f} |")
    before = fits["source_referenced_gap"]["shared"]["metrics"]["mae"]
    after = audit["reference_corrected_no_qwen_student_fit"]["metrics"]["mae"]
    lines += ["", "*Means use both Qwen sizes and are descriptive calibration summaries, not held-out "
              "estimates. Actual fold offsets are listed below. Reference-loss differences include "
              "size, quality, and tokenizer/evaluation effects; the residual cannot be identified as "
              "an intrinsic family effect from these data.", "",
              f"The uncalibrated pooled MAE changes {before:.6f}→{after:.6f}, a "
              f"{100*(1-after/before):.1f}% reduction in error, not a causal percentage of a family effect. "
              "The correction overshoots code and moves math in the wrong direction; it accounts for "
              "part of the large QA offset. The three capabilities do not support one scalar explained "
              "fraction. With a freely calibrated intercept this constant correction is absorbed, "
              "so the rotated total-gap MAE remains 0.106.", "",
              "| Calibration → held-out size | Capability | Total offset | Baseline contribution | Gain contribution | Known correction | Remaining offset |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in audit["fold_intercept_decomposition"]:
        lines.append(f"| {r['calibration_model']} → {r['held_out_model']} | {r['capability']} | {r['total_offset']:+.6f} | {r['baseline_offset']:+.6f} | {r['gain_offset']:+.6f} | {r['known_correction']:+.6f} | {r['remaining_offset']:+.6f} |")
    sensitivity = audit["v16_pairs_with_v12_qwen_sensitivity"]["source_referenced_gap"]["rotating_intercept"]["metrics"]["mae"]
    lines += ["", "### What 0.106 does and does not test", "",
              "V21 fits the Gemma floor/exponent on four sizes, calibrates Qwen offsets on 0.6B and "
              "predicts 1.7B, then reverses those roles. 0.106 is pooled **held-out prediction MAE**, "
              "not the zero calibration-fit residual. Each fold adds **one offset per capability**, "
              "three parameters total, not one shared offset. The exponent is also capability-specific. "
              "There are only two Qwen sizes: each capability supplies one independent size contrast. "
              "Its two rotated absolute errors are identical; six scores are not six independent "
              "size tests. Shared Qwen-4B reference and probe items remain in both folds. This is "
              "within-Qwen size transfer after calibration, not a second strict family holdout. "
              "Two Qwen points saturate a separate slope/intercept fit; they cannot validate that alternative.", "",
              f"Using complete V16 Gemma pairs where available and the V12 Qwen pairs gives total-gap "
              f"rotated MAE {sensitivity:.6f} as a labeled measurement-version sensitivity. "
              "V16 and V12 discrepancies are recorded in `measurement_comparison.csv`; missing "
              "checkpoint/revision hashes prevent attributing them to numerical error alone. V12 "
              "records mixed LoRA/full training modes, including a mode switch at D=600 for Gemma 4B. "
              "D is examples per domain, not supervised tokens. Neither size nor D is an isolated "
              "training intervention here.", "",
              "All values are completion-token-weighted CE nats per model-token. Cross-family token "
              "units are not invariant. Per-byte/per-character NLL on frozen common text is TO-DO; "
              "none was measured. QA loss changes are not claims of QA accuracy improvement.", "",
              "### Suggested C21 replacement (proposal only; ledger claim unchanged)", "",
              suggested_wording(audit), "",
              "### Complete artifact decomposition", "",
              "Each row uses S0 and S_KD from the same file. Units: nats per model-token. "
              "Full precision, all three losses, and input SHA-256 hashes are in the JSON/CSV artifacts.", "",
              "| Version | Student | Run | Capability | BASELINE GAP | DISTILLATION GAIN | Total |",
              "|---|---|---|---|---:|---:|---:|"]
    for r in summary["rows"]:
        lines.append(f"| v{r['eval_version']} | {r['student']} | {r['run']} | {r['capability']} | {r['baseline_gap']:+.6f} | {r['distillation_gain']:+.6f} | {r['source_referenced_gap']:+.6f} |")
    return "\n".join(lines) + "\n"


def build_summary(results: Path) -> dict:
    paths = input_paths(results)
    rows = load_decompositions(results)
    lookup = {(r["student"], r["run"], r["capability"]): r for r in rows if r["eval_version"] == 12}
    comparisons = []
    for r in rows:
        if r["eval_version"] != 16:
            continue
        previous = lookup.get((r["student"], r["run"], r["capability"]))
        if previous is None:
            continue
        comparisons.append({"student": r["student"], "run": r["run"], "capability": r["capability"],
                            "v12_path": previous["loss_path"], "v16_path": r["loss_path"],
                            **{f"v16_minus_v12_{key}": r[key] - previous[key] for key in
                               ("own_dense_loss", "distilled_loss", "distillation_gain")}})
    return {"version": 22, "cpu_only": True, "inputs_mutated": False,
            "counts": {f"v{v}_runs": len({r["loss_path"] for r in rows if r["eval_version"] == v}) for v in (12,16)},
            "inputs": [{"path": path_label(p, results), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        "bytes": p.stat().st_size} for p in paths],
            "rows": rows, "measurement_comparison": comparisons,
            "c21_audit": analyze_c21(rows, read_json(results / "v21-distill-law/summary.json"))}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def update_law_report(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8")
    # Supersede the report's strong interpretation, preserving the V21 scores.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("- **Distillation is PARTIAL;"):
            lines[i] = (
                "- **Distillation is PARTIAL / EXPLORATORY (V22 reassessment).** The V21 total-gap "
                "MAEs remain 0.475 without Qwen calibration and 0.106 with one held-out offset per "
                "capability. About 99%/90% of Gemma math/code size slopes are dense baseline effects; "
                "reference/coordinate differences partly account for offsets. A distillation-specific "
                "hierarchical family law is not established. See the Stage A / V22 section below."
            )
        elif line.startswith("**Cross-family verdict: HIERARCHICAL.**"):
            lines[i] = (
                "**V21 recorded verdict: HIERARCHICAL; superseded in interpretation by V22 below.** "
                "The 0.475→0.106 held-out MAE and 4/6→6/6 signs remain reproducible for the "
                "source-referenced total gap. These do not isolate the distillation effect or establish "
                "an intrinsic family offset. The family-specific exponent remains untested."
            )
    text = "\n".join(lines) + "\n"
    if (BEGIN in text) != (END in text):
        raise ValueError("incomplete V22 report markers")
    block = BEGIN + "\n" + section.rstrip() + "\n" + END
    if BEGIN in text:
        text = text[:text.index(BEGIN)] + block + text[text.index(END) + len(END):]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/v22-distill-decomp")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update-law-report", action="store_true")
    args = parser.parse_args()
    results = args.results_root.resolve()
    paths = input_paths(results)
    if args.dry_run:
        print("V22 dry run — CPU-only; no files written; no model/dataset/job access")
        for path in paths:
            print(path)
        print(f"{len(paths)} inputs; planned output: {args.output_dir}")
        return
    output = args.output_dir.resolve()
    # Refuse an existing input tree even when its immediate filenames differ.
    if output == results or any(output == p or output in p.parents for p in paths):
        raise ValueError("output must not overlap an input tree")
    summary = build_summary(results)
    report = render_report(summary)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(output / "decomposition.csv", summary["rows"])
    write_csv(output / "measurement_comparison.csv", summary["measurement_comparison"])
    write_csv(output / "intercept_attribution.csv", summary["c21_audit"]["reference_attribution"])
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "suggested_C21.md").write_text(suggested_wording(summary["c21_audit"]) + "\n", encoding="utf-8")
    if args.update_law_report:
        # Keep the paper report concise; the full per-artifact table lives in results.
        section = report.split("### Complete artifact decomposition")[0]
        section += "Complete decomposition: `results/v22-distill-decomp/decomposition.csv`; provenance and folds: `summary.json` in that directory.\n"
        update_law_report(ROOT / "paper/docs/LAW_FIT_REPORT.md", section)
    print(f"Wrote {len(summary['rows'])} paired decompositions to {output}")


if __name__ == "__main__":
    main()
