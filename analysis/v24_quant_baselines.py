#!/usr/bin/env python3
"""Audit frozen 5-bit predictions against real baselines; CPU, existing JSON only.

Run with --dry-run to validate inputs and evaluate without writing any files.
"""
from __future__ import annotations

import argparse
try:
    from . import prediction_audit as audit
except ImportError:
    import prediction_audit as audit
import numpy as np

ROOT = audit.ROOT
OUT = ROOT / "results/v24-quant-baselines"
REPORT = ROOT / "paper/docs/LAW_FIT_REPORT.md"
PREREG = ROOT / "results/prereg/quant_5bit_prereg.json"
SIMPLE = ("zero_change", "nearest_4bit", "linear_4_6bit")
NAMES = (*SIMPLE, "law_frozen", "law_reconstructed", "dense_only_lomo")


def shape(bit):
    return 4.0 ** -bit - 4.0 ** -16


def calibrated_q(losses, capability):
    x = np.array([shape(b) for b in (8, 6, 4)])
    y = np.array([audit.finite(losses[str(b)][capability]) - losses["dense"][capability]
                  for b in (8, 6, 4)])
    return float(x @ y / (x @ x))


def load_rows():
    frozen = audit.read_json(PREREG)
    rows, paths = [], [PREREG]
    for model, predictions in sorted(frozen["predictions"].items()):
        path = ROOT / "results/v10-quantization" / model / "quant_losses.json"
        tag = model.removeprefix("Qwen--")
        meta_path = ROOT / "results/v6-capability-geometry" / tag / "fisher_meta.json"
        paths.extend([path, meta_path])
        loss, meta = audit.read_json(path), audit.read_json(meta_path)
        family = "qwen3" if tag.startswith("Qwen3") else tag.split("-")[0]
        for cap in audit.CAPABILITIES:
            dense = audit.finite(loss["dense"][cap])
            q = calibrated_q(loss, cap)
            prediction = audit.finite(predictions[cap])
            if abs(prediction - q * shape(5)) > .000051:
                raise ValueError(f"Frozen prediction no longer reconstructs: {model}/{cap}")
            rows.append({"row_id": f"{model}|{cap}|5", "model": model, "family": family,
                         "capability": cap, "N0": audit.finite(meta["n_params"]),
                         "dense_loss": dense, "q_calibrated": q,
                         "observed": audit.finite(loss["5"][cap]) - dense,
                         "delta4": audit.finite(loss["4"][cap]) - dense,
                         "delta6": audit.finite(loss["6"][cap]) - dense,
                         "law_frozen": prediction, "law_reconstructed": q * shape(5),
                         "measurement_caveat": loss.get("_5bit_meta"),
                         "source_path": str(path.relative_to(ROOT))})
    return rows, paths, frozen


def dense_predict(train, test):
    """Fixed ridge strength 1; all preprocessing uses other models only.

    Regress signed q/1024 (a numerical rescaling of q, not a new outcome) on
    standardized log N0, dense L_c, and centered family indicators. All slopes
    are penalized, the intercept is not. Unseen families get zero correction.
    """
    if {r["model"] for r in train} & {r["model"] for r in test}:
        raise ValueError("Held-out model leaked into dense-only calibration")
    if len({r["capability"] for r in train + test}) != 1:
        raise ValueError("Fit capabilities separately")
    numeric = lambda rs: np.array([[np.log(r["N0"] / 1e9), r["dense_loss"]] for r in rs])
    x = numeric(train)
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-12)
    families = sorted({r["family"] for r in train})
    frequencies = np.array([np.mean([r["family"] == f for r in train]) for f in families])
    def design(rs):
        effects = np.array([[float(r["family"] == f) for f in families] for r in rs]) - frequencies
        for i, r in enumerate(rs):
            if r["family"] not in families:
                effects[i] = 0
        return np.column_stack([np.ones(len(rs)), (numeric(rs) - center) / scale, effects])
    design_train = design(train)
    penalty = np.eye(design_train.shape[1]); penalty[0, 0] = 0
    coef = np.linalg.solve(design_train.T @ design_train + penalty,
                           design_train.T @ np.array([r["q_calibrated"] / 1024 for r in train]))
    prediction = design(test) @ coef * 1024 * shape(5)
    return prediction, {"train_row_ids": [r["row_id"] for r in train],
                        "test_row_ids": [r["row_id"] for r in test],
                        "center": center.tolist(), "scale": scale.tolist(),
                        "families": families, "family_frequencies": frequencies.tolist(),
                        "coefficients": coef.tolist(), "ridge_lambda": 1.0}


def build_summary(n_boot=10000):
    rows, paths, prereg = load_rows()
    hashes = audit.provenance(paths)
    results, folds = {}, []
    for cap in audit.CAPABILITIES:
        rs = [r for r in rows if r["capability"] == cap]
        predictions = {"zero_change": [0.] * len(rs), "nearest_4bit": [r["delta4"] for r in rs],
                       "linear_4_6bit": [(r["delta4"] + r["delta6"]) / 2 for r in rs],
                       "law_frozen": [r["law_frozen"] for r in rs],
                       "law_reconstructed": [r["law_reconstructed"] for r in rs],
                       "dense_only_lomo": [None] * len(rs)}
        for split in audit.splits(rs, "model"):
            train, test = ([rs[i] for i in split[k]] for k in ("train_indices", "test_indices"))
            pred, fit = dense_predict(train, test)
            folds.append({"capability": cap, "held_out": split["held_out"], **fit})
            for i, p in zip(split["test_indices"], pred):
                predictions["dense_only_lomo"][i] = float(p)
        metrics, errors, draws = audit.compare_predictions(rs, predictions, "law_frozen", n_boot=n_boot)
        best = min(SIMPLE, key=lambda name: metrics[name]["mae"])
        # Re-select the best simple baseline inside every paired draw as well.
        gain = draws[:, :3].min(axis=1) - draws[:, 3]
        results[cap] = {"metrics": metrics, "best_simple": best,
                        "gain_over_best_simple": metrics[best]["mae"] - metrics["law_frozen"]["mae"],
                        "gain_over_best_simple_ci95": audit.interval(gain),
                        "calibration_cost_exact": float((errors[:, 5] - errors[:, 4]).mean()),
                        "calibration_cost_exact_ci95": audit.interval(draws[:, 5] - draws[:, 4])}
        for i, row in enumerate(rs):
            row["predictions"] = {name: float(predictions[name][i]) for name in NAMES}
    return {"version": 24, "input_sha256": hashes, "n_boot": n_boot,
            "bootstrap_unit": "model (12 per capability), paired; fitted predictions fixed",
            "preregistration": {k: v for k, v in prereg.items() if k != "predictions"},
            "rows": rows, "dense_only_folds": folds, "by_capability": results}


def render(summary):
    lines = ["### V24: frozen 5-bit predictions versus simple baselines", "",
             "CPU-only analysis of all 12 preregistered models × 3 capabilities. All candidates score "
             "the SAME 5-bit cells; no observed-loss/cliff filter, no sign censoring, and no 5-bit refit. "
             "The frozen JSON predictions are used as recorded (four decimal places). MAE units are CE nats.", "",
             "**Calibration audit:** q_c was fitted separately for each model and capability using that "
             "model's OWN 8-, 6-, and 4-bit ΔL values. The law is ΔL=q_c(4^-b−4^-16), with a dense anchor, "
             "one fitted coefficient, and **three compressed calibration points**. This is a CALIBRATED "
             "prediction, not a basic-parameter-only law. Frozen values reconstruct to rounding precision.", "",
             "Nearest-bit uses ΔL(4); interpolation uses [ΔL(4)+ΔL(6)]/2. The dense-only LOMO variant "
             "regresses signed q_c from the other 11 models on log(N0/1e9), family, and dense L_c, separately "
             "per capability, with fixed ridge λ=1 and training-only standardization. N0 is consistently "
             "the V6 fisher_meta 2-D language-weight count, including embeddings; it is not total checkpoint "
             "storage. Unseen-family correction is zero. At inference it uses no compressed measurements "
             "of the held-out model, but it still requires that model's measured dense L_c.", "",
             f"Intervals use {summary['n_boot']:,} paired model-bootstrap draws (12 models per capability). "
             "They describe panel variation conditional on these fits, not item/seed uncertainty or "
             "retraining uncertainty. Positive candidate−law favors 4^-b.", "",
             "| Capability | Candidate | MAE | Candidate MAE − frozen 4^-b (95% CI) |",
             "|---|---|---:|---:|"]
    for cap, result in summary["by_capability"].items():
        for name in NAMES:
            m = result["metrics"][name]
            lines.append(f"| {cap} | {name} | {audit.fmt(m['mae'])} | "
                         f"{audit.with_ci(m['mae_minus_reference'], m['difference_ci95'])} |")
    lines += ["", "| Capability | Best simple baseline | Improvement of 4^-b over best simple (95% CI) | "
              "Dense-only minus exact calibrated MAE (95% CI) |", "|---|---|---:|---:|"]
    for cap, r in summary["by_capability"].items():
        lines.append(f"| {cap} | {r['best_simple']} | "
                     f"{audit.with_ci(r['gain_over_best_simple'], r['gain_over_best_simple_ci95'])} | "
                     f"{audit.with_ci(r['calibration_cost_exact'], r['calibration_cost_exact_ci95'])} |")
    lines += ["", "The best-simple column takes the minimum simple-baseline MAE within each bootstrap "
              "draw; per-candidate intervals keep the named comparator fixed. These exploratory comparisons "
              "do not supply a new independent validation set. A positive calibration-cost column quantifies "
              "the error reduction bought by measuring three compressed calibration points, for this specified "
              "dense-only estimator; it is not an optimal-estimator bound.", ""]
    for cap, r in summary["by_capability"].items():
        lo, hi = r["gain_over_best_simple_ci95"]
        verdict = "lower error than the best simple baseline" if lo > 0 else (
            "higher error than the best simple baseline" if hi < 0 else "no resolved advantage over the best simple baseline")
        lines.append(f"- {cap}: 4^-b has {verdict}.")
    lines += ["", "Qwen-0.6B's input records a 5-bit dense-protocol discrepancy (QA 0.0133 nats); "
              "the canonical dense anchor is retained for every candidate. This is a measurement caveat, "
              "not silently corrected. The numerical preregistration audit does not independently verify "
              "its historical freeze timestamp.", "", "Reproduce: `python analysis/v24_quant_baselines.py --dry-run`, then the same command "
              "without `--dry-run`. Cell predictions, fold coefficients and SHA-256 input hashes: "
              "`results/v24-quant-baselines/summary.json`.", ""]
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
    # Preserve V18's broader exploratory table, but place it after the decisive audit.
    text = REPORT.read_text()
    section = text.split("## 2. Quantization\n", 1)[1].split("## 3. Distillation", 1)[0]
    marker = "### Historical V18 quantization candidates"
    historical = section.split(marker, 1)[1] if marker in section else section
    audit.replace_section(REPORT, "## 2. Quantization", "## 3. Distillation",
                          report + "\n" + marker + "\n\n" + historical.strip())
    text = REPORT.read_text()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("- **Quantization remains") or line.startswith("- **Quantization: V24"):
            comparisons = []
            for cap, r in summary["by_capability"].items():
                lo, hi = r["gain_over_best_simple_ci95"]
                verdict = "advantage resolved" if lo > 0 else "baseline favored" if hi < 0 else "advantage unresolved"
                comparisons.append(f"{cap} {audit.fmt(r['metrics']['law_frozen']['mae'])} vs "
                                   f"{r['best_simple']} {audit.fmt(r['metrics'][r['best_simple']]['mae'])} ({verdict})")
            lines[i] = ("- **Quantization: V24 frozen 5-bit law versus best simple baseline.** " +
                        "; ".join(comparisons) + ". The law uses three own-model compressed calibration "
                        "points; the dense-only LOMO audit quantifies their cost.")
        elif line.startswith("**Simplicity verdict.**"):
            lines[i] = ("**Simplicity verdict.** The historical pruning comparisons support keeping the reduced "
                        "density-only form on probation. Quantization's decisive comparison is the V24 frozen "
                        "5-bit test against zero-change, nearest-bit and interpolation below. The absence of an "
                        "unseen category in a categorical estimator is not evidence that a continuous law beats "
                        "simple predictive baselines; use the paired best-simple gains reported in V24.")
        elif line.startswith("| Quantization | leave-one-bit-out (pre-cliff only)"):
            lines[i] = line.replace("categorical baseline cannot predict unseen bit", "historical comparison; use V24 simple baselines below")
    text = "\n".join(lines) + "\n"
    scope = ("V24/V25 supersede the quantization/distillation headline comparisons below. They retain the "
             "entire declared held-out panels without V18's outcome-based cliff filter and report each "
             "capability separately. Their paired bootstrap units and calibration requirements are defined "
             "in their sections; the older V18 conventions in the following paragraphs apply only to historical tables.")
    if scope not in text:
        text = text.replace("## Decision rule and scope\n", "## Decision rule and scope\n\n" + scope + "\n")
    REPORT.write_text(text)
    print(f"Wrote {OUT} and quantization section of {REPORT}")


if __name__ == "__main__":
    main()
