#!/usr/bin/env python3
"""V25: predict signed own-dense distillation changes, CPU-only, existing data.

--dry-run validates and evaluates the full analysis without writing files.
"""
from __future__ import annotations

import argparse
try:
    from . import prediction_audit as audit
except ImportError:
    import prediction_audit as audit
import numpy as np

ROOT = audit.ROOT
OUT = ROOT / "results/v25-distill-delta"
REPORT = ROOT / "paper/docs/LAW_FIT_REPORT.md"
# Fixed coordinate units, not optimized on either development or test outcomes.
D_STAR = 150.0  # traces per training domain, not tokens
N_STAR = 1e9
SIZES = {"gemma3-270m": .27e9, "gemma3-1b": 1e9, "gemma3-4b": 4e9}
BUDGETS = (75, 150, 300, 600)
CANDIDATES = ("zero", "mean", "u", "u_u2", "u_uv", "u_u2_uv")
PROTOCOLS = {"leave_size_out": "model", "leave_budget_out": "D", "leave_cell_out": "row_id"}


def load_rows():
    rows, paths = [], []
    panel = None
    for model, size in SIZES.items():
        for budget in BUDGETS:
            path = ROOT / f"results/v12-distill/{model}/gpt-5.6-luna_full_{budget}/eval.json"
            p = audit.read_json(path); paths.append(path)
            if (p["student_tag"], p["teacher"], p["recipe"], p["n_per_domain"]) != (
                    model, "gpt-5.6-luna", "full", budget):
                raise ValueError(f"Unexpected experiment identity: {path}")
            protocol = [p[k] for k in ("measurement_benchmarks", "probe_source", "probe_seed",
                                       "n_probe_requested", "probe_half", "measurement_samples")]
            if panel is not None and panel != protocol:
                raise ValueError(f"Incompatible measurement panel: {path}")
            panel = protocol
            for cap in audit.CAPABILITIES:
                dense, post = (audit.finite(p[k][cap]) for k in ("dense", "post_training"))
                delta = post - dense
                if not np.isclose(delta, audit.finite(p["delta"][cap]), rtol=0, atol=1e-10):
                    raise ValueError(f"Inconsistent delta: {path}/{cap}")
                rows.append({"row_id": f"{model}|{budget}|{cap}", "model": model,
                             "capability": cap, "N_S": size, "D": budget,
                             "dense_loss": dense, "post_loss": post, "observed": delta,
                             "training_mode": p["training_mode"],
                             "source_path": str(path.relative_to(ROOT))})
    return rows, paths


def design(rows, candidate):
    u = np.log1p(np.array([r["D"] for r in rows]) / D_STAR)
    v = np.log(np.array([r["N_S"] for r in rows]) / N_STAR)
    columns = {"u": [u], "u_u2": [u, u*u], "u_uv": [u, u*v], "u_u2_uv": [u, u*u, u*v]}
    return np.column_stack(columns[candidate])


def fit_predict(train, test, candidate):
    if {r["row_id"] for r in train} & {r["row_id"] for r in test}:
        raise ValueError("Training/test overlap")
    if len({r["capability"] for r in train + test}) != 1:
        raise ValueError("Fit each capability separately")
    y = np.array([r["observed"] for r in train])
    if candidate == "zero":
        return np.zeros(len(test)), {"coefficients": [], "n_parameters": 0}
    if candidate == "mean":
        return np.full(len(test), y.mean()), {"coefficients": [float(y.mean())], "n_parameters": 1}
    x = design(train, candidate)
    coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != x.shape[1]:
        raise ValueError(f"Underidentified {candidate} in held-out fold")
    return design(test, candidate) @ coef, {"coefficients": coef.tolist(), "n_parameters": len(coef)}


def evaluate(rows, key, n_boot=10000):
    rs = list(rows)
    predictions = {c: np.full(len(rs), np.nan) for c in CANDIDATES}
    folds = []
    for split in audit.splits(rs, key):
        train, test = ([rs[i] for i in split[k]] for k in ("train_indices", "test_indices"))
        fit = {}
        for candidate in CANDIDATES:
            p, details = fit_predict(train, test, candidate)
            predictions[candidate][split["test_indices"]] = p
            fit[candidate] = details
        folds.append({"held_out": split["held_out"], "train_row_ids": [r["row_id"] for r in train],
                      "test_row_ids": [r["row_id"] for r in test], "fits": fit})
    metrics, errors, draws = audit.compare_predictions(rs, predictions, "mean", n_boot=n_boot)
    contrasts = {}
    for candidate, baseline in [(c, b) for c in CANDIDATES[2:] for b in ("zero", "mean")] + [
            ("u_u2", "u"), ("u_uv", "u"), ("u_u2_uv", "u_u2")]:
        i, j = CANDIDATES.index(candidate), CANDIDATES.index(baseline)
        contrasts[f"{candidate}_minus_{baseline}"] = {
            "difference": float((errors[:, i] - errors[:, j]).mean()),
            "ci95": audit.interval(draws[:, i] - draws[:, j])}
    records = [{**r, "predictions_delta": {c: float(predictions[c][i]) for c in CANDIDATES},
                "predictions_loss": {c: float(r["dense_loss"] + predictions[c][i]) for c in CANDIDATES}}
               for i, r in enumerate(rs)]
    return {"metrics": metrics, "contrasts": contrasts, "folds": folds, "records": records}


def build_summary(n_boot=10000):
    rows, paths = load_rows()
    hashes = audit.provenance(paths)
    analyses = {}
    for cohort, subset in (("full_grid", rows), ("lora_only", [r for r in rows if r["training_mode"] == "lora"])):
        analyses[cohort] = {}
        for protocol, key in PROTOCOLS.items():
            analyses[cohort][protocol] = {cap: evaluate([r for r in subset if r["capability"] == cap], key, n_boot)
                                         for cap in audit.CAPABILITIES}
    fits = {}
    for cap in audit.CAPABILITIES:
        rs = [r for r in rows if r["capability"] == cap]
        fits[cap] = {}
        for candidate in CANDIDATES[2:]:
            x = design(rs, candidate)
            coef = np.linalg.lstsq(x, [r["observed"] for r in rs], rcond=None)[0]
            fits[cap][candidate] = {"coefficients": coef.tolist(), "status": "NON-DECISIONAL full-data fit"}
    dense_spread = {model: {cap: float(np.ptp([r["dense_loss"] for r in rows
                                              if r["model"] == model and r["capability"] == cap]))
                            for cap in audit.CAPABILITIES} for model in SIZES}
    return {"version": 25, "input_sha256": hashes, "n_boot": n_boot,
            "fixed_units": {"D_star_traces_per_domain": D_STAR, "N_star_parameters": N_STAR,
                            "N_S_convention": "nominal model-label sizes, 0.27/1/4 billion"},
            "bootstrap_unit": "model, only 3 clusters; fixed OOF fits, descriptive intervals",
            "rows": rows, "analyses": analyses, "full_data_fits": fits,
            "dense_anchor_range_across_runs": dense_spread}


def render(summary):
    lines = ["### V25: dense baseline plus signed transfer response", "",
             "The main formulation is **δ_c=L_c(S_KD)−L_c(S0)** and **L̂_c=L_c(S0)+δ̂_c**. "
             "S0 is the student's own dense model; negative δ means lower loss. Each cell uses its paired "
             "eval.json dense anchor, so MAE in predicted L equals MAE in predicted δ. No reference-model "
             "gap enters the fit. Math, code and QA are fitted and scored separately, without pooled MAE.", "",
             "Grid: Gemma-3 270M/1B/4B × D=75/150/300/600 traces PER DOMAIN, teacher gpt-5.6-luna, "
             "recipe full. Fixed units: D_*=150 traces/domain and N_*=10^9 nominal parameters. "
             "u=log(1+D/D_*), v=log(N_S/N_*); units and forms are not selected by test error. "
             "The low-parameter response is δ̂_c=a_c u+b_c u²+k_c uv (OLS, no intercept). "
             "All response variants satisfy δ̂(D=0)=0. The training-mean comparator is intentionally "
             "a constant correction and need not satisfy this boundary.", "",
             "Four main candidates: zero (dense student), training mean δ, data-only u+u², and "
             "size×data u+u²+uv. The u and u+uv ablations isolate curvature and interaction. "
             "Coefficients and means are fitted only on each fold's development rows. Entire-size and "
             "entire-budget holdouts are the main tests; leave-cell-out is an interpolation diagnostic. "
             "Every candidate scores the same cells within each protocol.", "",
             "**Recipe confound:** Gemma-3 4B at D=600 uses full training; the other 11 cells use LoRA. "
             "The requested full grid is retained and a LoRA-only sensitivity excludes that cell from "
             "both training and testing. D effects in the full grid cannot be assigned solely to data volume. "
             "Small dense-anchor differences across runs are recorded, not silently replaced.", ""]
    for cohort, protocols in summary["analyses"].items():
        lines += [f"#### {cohort}: held-out MAE in nats", "",
                  "| Protocol | Capability | Cells | Dense δ=0 | Mean δ | u | Data-only u+u² | u+uv | Size×data u+u²+uv |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for protocol, caps in protocols.items():
            for cap, r in caps.items():
                values = " | ".join(audit.fmt(r["metrics"][c]["mae"]) for c in CANDIDATES)
                lines.append(f"| {protocol} | {cap} | {r['metrics']['zero']['n']} | {values} |")
    lines += ["", "#### Paired gains and complexity tests, full grid", "",
              "Differences below are candidate MAE minus comparator MAE; negative favors the added response. "
              f"95% intervals use {summary['n_boot']:,} paired model-bootstrap draws, only THREE model "
              "clusters. They are descriptive, conditional on overlapping fitted CV folds, and do not "
              "measure independent seed/item or retraining uncertainty. No multiplicity correction is applied.", "",
              "| Protocol | Capability | Contrast | ΔMAE (95% CI) |", "|---|---|---|---:|"]
    for protocol in ("leave_size_out", "leave_budget_out"):
        for cap, r in summary["analyses"]["full_grid"][protocol].items():
            for name in ("u_u2_minus_zero", "u_u2_minus_mean", "u_u2_uv_minus_zero",
                         "u_u2_uv_minus_mean", "u_u2_minus_u", "u_uv_minus_u", "u_u2_uv_minus_u_u2"):
                c = r["contrasts"][name]
                lines.append(f"| {protocol} | {cap} | {name} | {audit.with_ci(c['difference'], c['ci95'])} |")
    lines += ["", "#### Interpretation", ""]
    for cap in audit.CAPABILITIES:
        rs = [r for r in summary["rows"] if r["capability"] == cap]
        lines.append(f"- {cap}: observed δ range {audit.fmt(min(r['observed'] for r in rs))} to "
                     f"{audit.fmt(max(r['observed'] for r in rs))} nats.")
        for protocol in ("leave_size_out", "leave_budget_out"):
            result = summary["analyses"]["full_grid"][protocol][cap]
            best_simple = min(result["metrics"][b]["mae"] for b in ("zero", "mean"))
            wins = [c for c in ("u_u2", "u_u2_uv") if result["metrics"][c]["mae"] < best_simple]
            lines.append(f"  {protocol}: " + (", ".join(wins) + " beat both simple baselines in point MAE."
                         if wins else "neither main response beats both simple baselines in point MAE."))
    lines += ["", "Low total-loss MAE alone does not demonstrate learned transfer. Curvature/interaction "
              "claims require improvement over the constant correction and simpler response on these held-out "
              "axes, with uncertainty and the recipe sensitivity considered. QA is loss-space only; "
              "lower CE does not establish better QA accuracy. One family, one teacher and effectively one "
              "training realization per cell limit generalization.", "",
              "#### Full-grid coefficients (NON-DECISIONAL)", "",
              "All fits below use all 12 cells per capability, solely to specify the response; held-out "
              "comparisons above use freshly fitted development-only coefficients.", "",
              "| Capability | Form | a (u) | b (u²) | k (uv) |", "|---|---|---:|---:|---:|"]
    for cap, fits in summary["full_data_fits"].items():
        for candidate, fit in fits.items():
            coef = fit["coefficients"]
            a = coef[0]
            b = coef[1] if candidate in ("u_u2", "u_u2_uv") else 0.
            k = coef[-1] if candidate in ("u_uv", "u_u2_uv") else 0.
            lines.append(f"| {cap} | {candidate} | {audit.fmt(a)} | {audit.fmt(b)} | {audit.fmt(k)} |")
    lines += ["", "#### Simpler response and recipe sensitivity", ""]
    for cap in audit.CAPABILITIES:
        statements = []
        for protocol in ("leave_size_out", "leave_budget_out"):
            m = summary["analyses"]["full_grid"][protocol][cap]["metrics"]
            wins = m["u"]["mae"] < min(m[c]["mae"] for c in ("zero", "mean"))
            curvature = m["u_u2"]["mae"] - m["u"]["mae"]
            statements.append(f"{protocol}: u {'beats' if wins else 'does not beat'} both simple baselines "
                              f"in point MAE; adding u² changes MAE by {audit.fmt(curvature)}")
        m = summary["analyses"]["lora_only"]["leave_size_out"][cap]["metrics"]
        statements.append(f"LoRA-only size holdout: size×data MAE {audit.fmt(m['u_u2_uv']['mae'])} "
                          f"versus constant {audit.fmt(m['mean']['mae'])}")
        lines.append(f"- {cap}: " + "; ".join(statements) + ".")
    lines += ["",
              "Full-data coefficients are NON-DECISIONAL diagnostics in summary.json. The JSON also includes "
              "every fold, δ̂ and L̂, simple-baseline contrasts for all ablations, and LoRA-only contrasts. "
              "Reproduce: `python analysis/v25_distill_delta.py --dry-run`, then without `--dry-run`. "
              "Outputs: `results/v25-distill-delta/`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    summary = build_summary(args.bootstrap)
    report = render(summary)
    if args.dry_run:
        print(f"DRY RUN: validated {len(summary['rows'])} cells; CPU-only; no writes.\n{report}")
        return
    audit.write_outputs(summary, report, OUT)
    text = REPORT.read_text()
    section = text.split("## 3. Distillation\n", 1)[1].split("## 4. Recovery", 1)[0]
    marker = "### HISTORICAL candidates: source-referenced size gaps (V21)"
    historical = section.split(marker, 1)[1] if marker in section else section
    # Keep historical evidence, but demote headings beneath the historical note.
    historical = historical.replace("\n### ", "\n#### ")
    audit.replace_section(REPORT, "## 3. Distillation", "## 4. Recovery", report + "\n" + marker +
                          "\n\n" + historical.strip())
    lines = REPORT.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("Generated by `analysis/v18_law_fit.py`"):
            lines[i] = ("Pruning/recovery historical analyses: `analysis/v18_law_fit.py`. Quantization and "
                        "distillation refreshed by `analysis/v24_quant_baselines.py` and "
                        "`analysis/v25_distill_delta.py`; all are CPU-only existing-data analyses. "
                        "Historical V21/V22 candidates are retained with their original scope.")
        elif line.startswith("- **Distillation is") or line.startswith("- **Distillation: V25"):
            lines[i] = ("- **Distillation: V25 evaluates signed own-dense changes against zero and mean corrections.** "
                        "See per-capability size/budget holdouts, paired complexity contrasts and the "
                        "LoRA-only sensitivity below; low total-loss MAE alone is not evidence of transfer. The main formulation "
                        "is dense baseline plus transfer response; source-referenced size gaps are historical.")
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT} and distillation section of {REPORT}")


if __name__ == "__main__":
    main()
