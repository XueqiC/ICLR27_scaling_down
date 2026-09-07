#!/usr/bin/env python3
"""V26: held-out same/cross-capability prediction gain; CPU-only V23 JSON audit.

--dry-run performs input validation and analysis with no writes or model loads.
"""
from __future__ import annotations

import argparse
try:
    from . import prediction_audit as audit
except ImportError:
    import prediction_audit as audit
import numpy as np

ROOT = audit.ROOT
BASE = ROOT / "results/v23-loss-validity"
OUT = ROOT / "results/v26-loss-validity-pred"
REPORT = ROOT / "paper/docs/LOSS_VALIDITY.md"
CAPS = audit.CAPABILITIES
PROTOCOLS = ("leave_model_out", "leave_density_out", "leave_model_and_density_out")


def paired_item_deltas(current, dense):
    a, b = current["items"], dense["items"]
    identity = lambda item: (item["measurement_index"], item["probe_sha256"], item["n_tokens"])
    if [identity(i) for i in a] != [identity(i) for i in b] or len(a) < 2:
        raise ValueError("Unpaired benchmark items or changed target spans")
    values = np.array([audit.finite(i["loss"]) - audit.finite(j["loss"]) for i, j in zip(a, b)])
    for endpoint in (current, dense):
        if not np.isclose(np.mean([i["loss"] for i in endpoint["items"]]), endpoint["L_c"], atol=1e-10, rtol=0):
            raise ValueError("Item means disagree with L_c")
    if not np.isclose(values.mean(), current["delta_L_c"], atol=1e-10, rtol=0):
        raise ValueError("Item deltas disagree with delta_L_c")
    if not np.isclose(current["baseline_L_c"], dense["L_c"], atol=1e-10, rtol=0):
        raise ValueError("Dense anchor mismatch")
    return values


def load_rows():
    paths = sorted(BASE.glob("*/dense*/loss_validity.json"))
    payloads = {p: audit.read_json(p) for p in paths}
    rows, item_rows = [], []
    protocol_hash = None
    for path, p in payloads.items():
        if p["version"] != 23 or p["baseline_checkpoint"] != "dense" or p["adapter"] is not None:
            raise ValueError(f"Unsupported checkpoint: {path}")
        if p["protocol"]["loss_definition"] != "mean_over_examples(sum_target_token_CE / n_target_tokens)":
            raise ValueError(f"Unsupported loss: {path}")
        if protocol_hash is not None and p["protocol_sha256"] != protocol_hash:
            raise ValueError("Incompatible cross-model measurement protocols")
        protocol_hash = p["protocol_sha256"]
        if p["compression"]["method"] == "baseline":
            continue
        if p["compression"]["method"] != "prune":
            raise ValueError("This declared panel supports pruning only")
        base_path = path.parent.parent / "dense/loss_validity.json"
        dense = payloads[base_path]
        for key in ("baseline_sha256", "protocol_sha256", "tokenizer", "source_checkpoint", "model_tag"):
            if p[key] != dense[key]:
                raise ValueError(f"Mismatched {key}: {path}")
        row = {"row_id": f"{p['model_tag']}|{p['compression']['prune_density']}",
               "model": p["model_tag"], "density": audit.finite(p["compression"]["prune_density"]),
               "primary": {}, "secondary": {}, "source_path": str(path.relative_to(ROOT))}
        for cap in CAPS:
            for role in ("primary", "secondary"):
                endpoint = p["capabilities"][cap][role]
                reference = dense["capabilities"][cap][role]
                deltas = paired_item_deltas(endpoint, reference)
                row[role][cap] = audit.finite(endpoint["delta_L_c"])
                item_rows.append({"model": row["model"], "density": row["density"],
                                  "capability": cap, "role": role, "deltas": deltas,
                                  "item_ids": [i["probe_sha256"] for i in endpoint["items"]]})
        rows.append(row)
    rows.sort(key=lambda r: (r["model"], -r["density"]))
    if len({r["row_id"] for r in rows}) != len(rows) or not rows:
        raise ValueError("Empty or duplicate compression cells")
    ladders = {tuple(sorted(r["density"] for r in rows if r["model"] == m))
               for m in {r["model"] for r in rows}}
    if len(ladders) != 1:
        raise ValueError("Require matched density ladders across models")
    return rows, item_rows, paths


def folds(rows, protocol):
    if protocol != "leave_model_and_density_out":
        yield from audit.splits(rows, "model" if protocol == "leave_model_out" else "density")
        return
    for i, row in enumerate(rows):
        train = [j for j, r in enumerate(rows) if r["model"] != row["model"] and r["density"] != row["density"]]
        if not train:
            raise ValueError("Empty joint holdout development set")
        yield {"held_out": row["row_id"], "train_indices": train, "test_indices": [i]}


def mapping_predict(train, test, source, target):
    """Identical two-parameter affine OLS for every source→target mapping."""
    if {r["row_id"] for r in train} & {r["row_id"] for r in test}:
        raise ValueError("Mapping train/test overlap")
    x = np.column_stack([np.ones(len(train)), [r["primary"][source] for r in train]])
    y = [r["secondary"][target] for r in train]
    if np.ptp(x[:, 1]) <= 1e-12:
        coef = np.array([np.mean(y), 0.])
    else:
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
    prediction = coef[0] + coef[1] * np.array([r["primary"][source] for r in test])
    return prediction, coef.tolist()


def select_cross(train, target, protocol):
    """Select cross-capability source by inner CV, never outer target values."""
    inner_protocol = "leave_density_out" if protocol == "leave_density_out" else "leave_model_out"
    scores = {}
    for source in CAPS:
        if source == target:
            continue
        errors = []
        for fold in folds(train, inner_protocol):
            a, b = ([train[i] for i in fold[k]] for k in ("train_indices", "test_indices"))
            p, _ = mapping_predict(a, b, source, target)
            errors.extend(np.abs(p - np.array([r["secondary"][target] for r in b])))
        scores[source] = float(np.mean(errors))
    return min(scores, key=lambda s: (scores[s], s)), scores


def evaluate(rows, target, protocol, n_boot=10000):
    names = ("zero", "train_mean", *CAPS, "cross_selected")
    predictions = {name: np.full(len(rows), np.nan) for name in names}
    details = []
    for fold in folds(rows, protocol):
        train, test = ([rows[i] for i in fold[k]] for k in ("train_indices", "test_indices"))
        ix = fold["test_indices"]
        predictions["zero"][ix] = 0
        predictions["train_mean"][ix] = np.mean([r["secondary"][target] for r in train])
        coefficients = {}
        for source in CAPS:
            p, coef = mapping_predict(train, test, source, target)
            predictions[source][ix] = p
            coefficients[source] = coef
        selected, inner_scores = select_cross(train, target, protocol)
        predictions["cross_selected"][ix] = predictions[selected][ix]
        details.append({"held_out": fold["held_out"], "train_row_ids": [r["row_id"] for r in train],
                        "test_row_ids": [r["row_id"] for r in test], "coefficients": coefficients,
                        "cross_source": selected, "inner_cross_mae": inner_scores})
    scored = [{**r, "observed": r["secondary"][target]} for r in rows]
    metrics, errors, draws = audit.compare_predictions(scored, predictions, target, n_boot=n_boot)
    i, j = names.index("cross_selected"), names.index(target)
    records = [{"row_id": r["row_id"], "model": r["model"], "density": r["density"],
                "observed": r["secondary"][target],
                "predictions": {n: float(predictions[n][k]) for n in names}} for k, r in enumerate(rows)]
    gain = float((errors[:, i] - errors[:, j]).mean())
    ci = audit.interval(draws[:, i] - draws[:, j])
    return {"metrics": metrics, "same_over_cross_gain": gain, "gain_ci95": ci,
            "relative_gain": gain / metrics["cross_selected"]["mae"] if metrics["cross_selected"]["mae"] else None,
            "folds": details, "records": records}


def pearson(x, y):
    if len(x) < 3 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def noise_audit(item_rows, n_boot=10000):
    results = []
    for model in sorted({r["model"] for r in item_rows if r["model"].startswith("olmo")}):
        for cap in CAPS:
            for role in ("primary", "secondary"):
                rs = sorted([r for r in item_rows if (r["model"], r["capability"], r["role"]) == (model, cap, role)],
                            key=lambda r: -r["density"])
                if any(r["item_ids"] != rs[0]["item_ids"] for r in rs):
                    raise ValueError("Item identity changed along density trajectory")
                matrix = np.array([r["deltas"] for r in rs])
                n = matrix.shape[1]
                # The same sampled item indices are used at every density.
                indices = np.random.default_rng(260526).integers(n, size=(n_boot, n))
                boot = matrix[:, indices].mean(axis=2).T
                observed = matrix.mean(axis=1)
                se = matrix.std(axis=1, ddof=1) / np.sqrt(n)
                ci = np.quantile(boot, [.025, .975], axis=0).T
                centered = observed - observed.mean()
                boot_centered = boot - boot.mean(axis=1, keepdims=True)
                variation = float(np.sqrt(np.mean(centered**2)))
                resolution = float(np.sqrt(np.mean(np.var(boot_centered, axis=0, ddof=1))))
                ratio = variation / resolution if resolution else None
                results.append({"model": model, "capability": cap, "role": role, "n_items": n,
                                "mean_absolute_delta": float(np.mean(np.abs(observed))),
                                "max_absolute_delta": float(np.max(np.abs(observed))),
                                "median_95_noise_halfwidth": float(np.median(1.96 * se)),
                                "centered_signal_rms": variation, "centered_sampling_noise_rms": resolution,
                                "signal_to_resolution": ratio,
                                "resolution_status": "insufficient resolution" if ratio is None or ratio < 2 else
                                "variation exceeds 2x estimated sampling resolution",
                                "cells": [{"density": r["density"], "delta": float(observed[i]),
                                           "paired_item_se": float(se[i]), "ci95": ci[i].tolist()}
                                          for i, r in enumerate(rs)]})
    return results


def build_summary(n_boot=10000):
    rows, item_rows, paths = load_rows()
    hashes = audit.provenance(paths)
    strongest = min(r["density"] for r in rows)
    cohorts = {"all_densities": rows, "without_strongest": [r for r in rows if r["density"] != strongest]}
    analyses, correlations = {}, []
    for cohort, subset in cohorts.items():
        analyses[cohort] = {protocol: {cap: evaluate(subset, cap, protocol, n_boot) for cap in CAPS}
                            for protocol in PROTOCOLS}
        for model in sorted({r["model"] for r in subset}):
            rs = [r for r in subset if r["model"] == model]
            for target in CAPS:
                for source in CAPS:
                    correlations.append({"cohort": cohort, "model": model, "source": source,
                                         "target": target, "n": len(rs), "pearson": pearson(
                                             [r["primary"][source] for r in rs], [r["secondary"][target] for r in rs])})
    return {"version": 26, "input_sha256": hashes, "n_boot": n_boot, "rows": rows,
            "strongest_density": strongest, "analyses": analyses, "descriptive_pearson": correlations,
            "olmo_sampling_resolution": noise_audit(item_rows, n_boot),
            "bootstrap_unit": "model (6 clusters); paired, fixed OOF fits; not retraining uncertainty"}


def render(summary):
    lines = ["# Capability-loss validity: held-out prediction gain", "",
             "V26 reads the six-model × five-density V23 pruning panel (0.90/0.85/0.80/0.75/0.70). "
             "Endpoint: mean per-example target-token CE; deltas use each benchmark's own dense anchor. "
             "Pairs: MATH-500→GSM8K, MBPP→HumanEval, 2WikiMultihopQA→HotpotQA. Dense anchors are not scored. "
             "All compression cells, including negative changes and cliffs, remain in the main analysis. "
             "Different tokenizers still imply different token units; generalization is empirical within this panel.", "",
             "**Primary evidence is prediction gain, not correlation.** For each secondary target, fit "
             "y=intercept+slope×ΔL_primary on development cells. Every source uses the same two-parameter "
             "affine OLS mapping, folds and target cells. A constant source falls back to the training mean. "
             "Neither coefficients nor source selection use held-out secondary losses. The held-out primary "
             "ΔL is an input: this tests transfer between measured benchmarks, not prediction from density alone.", "",
             "Main folds hold out a whole model, a whole density, or both: the joint test excludes all rows "
             "of the target model AND all rows at the target density from development and tests their "
             "intersection. Cross-selected chooses between the two other primary capabilities using inner "
             "development CV (model folds for model/joint tests; density folds for density tests). "
             "All individual cross mappings are also reported so selection cannot hide a strong comparator.", "",
             f"Paired intervals use {summary['n_boot']:,} model-bootstrap draws (six clusters), holding OOF fits "
             "fixed. Positive gain = MAE(cross-selected)−MAE(same), favoring capability specificity. "
             "These small-panel descriptive CIs omit retraining and shared-probe uncertainty; multiple targets "
             "and protocols are exploratory. No post-hoc practical-effect threshold is imposed.", ""]
    for cohort, protocols in summary["analyses"].items():
        lines += [f"## {cohort}: prediction", "",
                  "| Protocol | Secondary target | Cells | Zero MAE | Mean MAE | Math→target | Code→target | QA→target | Cross-selected | Same-over-cross gain (95% CI) |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for protocol, caps in protocols.items():
            for cap, result in caps.items():
                m = result["metrics"]
                values = " | ".join(audit.fmt(m[c]["mae"]) for c in ("zero", "train_mean", *CAPS, "cross_selected"))
                lines.append(f"| {protocol} | {cap} | {m['zero']['n']} | {values} | "
                             f"{audit.with_ci(result['same_over_cross_gain'], result['gain_ci95'])} |")
        lines += [""]
    lines += ["## Capability-specific interpretation", ""]
    for cap in CAPS:
        statements = []
        for cohort in summary["analyses"]:
            r = summary["analyses"][cohort]["leave_model_and_density_out"][cap]
            lo, hi = r["gain_ci95"]
            verdict = "same is better in this paired interval" if lo > 0 else (
                "cross is better in this paired interval" if hi < 0 else "same-versus-cross advantage is unresolved")
            statements.append(f"{cohort}: {verdict}")
        lines.append(f"- {cap}, joint holdout: " + "; ".join(statements) + ".")
    lines += ["", "Where same-capability prediction does not improve over cross-capability, high Pearson "
              "correlation is consistent with general compression damage a_j·h(d); it does not establish "
              "capability-specific structure. That explanation is compatible with the data, not a uniquely "
              "identified causal mechanism. A gain that disappears without density 0.70 is sensitive to the "
              "deepest point. Inspect absolute gains, their intervals, and individual cross-source errors.", "",
              "## Pearson is descriptive: full ladder versus excluding density 0.70", "",
              "| Model | Target | Same, all 5 | Same, mildest 4 | Cross math, mildest 4 | Cross code, mildest 4 | Cross QA, mildest 4 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    lookup = {(r["cohort"], r["model"], r["source"], r["target"]): r["pearson"] for r in summary["descriptive_pearson"]}
    for model in sorted({r["model"] for r in summary["rows"]}):
        for cap in CAPS:
            values = [lookup[("all_densities", model, cap, cap)], lookup[("without_strongest", model, cap, cap)]]
            values += [lookup[("without_strongest", model, source, cap)] if source != cap else None for source in CAPS]
            lines.append(f"| {model} | {cap} | " + " | ".join(audit.fmt(v) for v in values) + " |")
    lines += ["", "## OLMo: measured changes versus sampling resolution", "",
              "Subtract dense/compressed losses ITEM BY ITEM, checking probe hashes and target lengths. "
              "Noise half-width is median_d[1.96×SD(item ΔL_d)/sqrt(n)]. Signal is RMS of the centered "
              "mean ΔL trajectory. Its sampling resolution is RMS of bootstrap SDs after centering each "
              "resampled trajectory; the same sampled items are used across densities, preserving covariance "
              "from the shared dense anchor. Signal/resolution <2 is labeled insufficient resolution as "
              "a descriptive screen, not a formal correlation hypothesis test. This is benchmark-item "
              "sampling uncertainty, not measured numerical or seed noise: no repeated-run noise estimate exists.", "",
              "| Model | Capability | Benchmark role | Mean |ΔL| | Max |ΔL| | Median 95% noise half-width | Signal/resolution | Resolution assessment |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for r in summary["olmo_sampling_resolution"]:
        lines.append(f"| {r['model']} | {r['capability']} | {r['role']} | {audit.fmt(r['mean_absolute_delta'])} | "
                     f"{audit.fmt(r['max_absolute_delta'])} | {audit.fmt(r['median_95_noise_halfwidth'])} | "
                     f"{audit.fmt(r['signal_to_resolution'])} | {r['resolution_status']} |")
    lines += ["", "Low OLMo correlations are not 'noise-confirmed'. If either benchmark lacks trajectory "
              "resolution, capability agreement is unresolved at the available signal scale. Where both "
              "trajectories are resolved yet disagree, insufficient resolution alone is not an adequate "
              "explanation. Cell-level paired-item intervals are in summary.json. Neither correlation nor "
              "prediction gain establishes accuracy validity, a universal latent capability, or absence of "
              "benchmark contamination.", "",
              "Reproduce CPU-only: `python analysis/v26_loss_validity_prediction.py --dry-run`, then without "
              "`--dry-run`. Outputs: `results/v26-loss-validity-pred/summary.json` and `report.md`; "
              "the latter rebuilds this document. JSON retains all held-out predictions, mapping coefficients, "
              "inner source selection scores, same-versus-each-cross paired intervals, item resolution, "
              "descriptive correlations and SHA-256 input hashes. Original V23 inputs are read only.", ""]
    return "\n".join(lines).replace("Mean |ΔL|", "Mean abs ΔL").replace("Max |ΔL|", "Max abs ΔL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    summary = build_summary(args.bootstrap)
    report = render(summary)
    if args.dry_run:
        print(f"DRY RUN: validated {len(summary['rows'])} compression cells; CPU-only; no writes.\n{report}")
        return
    audit.write_outputs(summary, report, OUT)
    REPORT.write_text(report)
    print(f"Wrote {OUT} and {REPORT}")


if __name__ == "__main__":
    main()
