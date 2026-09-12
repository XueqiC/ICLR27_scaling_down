#!/usr/bin/env python3
"""CPU-only benchmark-transfer evidence, verified against the original V23 measurements."""
import hashlib
import sys

sys.dont_write_bytecode = True
if __package__:
    from .plot_fig1_responses import Audit, CAPS, CAP_LABEL, COLORS, ROOT, checked_close, setup_style
else:
    from plot_fig1_responses import Audit, CAPS, CAP_LABEL, COLORS, ROOT, checked_close, setup_style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

PROTOCOLS = ("leave_model_out", "leave_density_out", "leave_model_and_density_out")
PROTOCOL_LABEL = ("Model out", "Density out", "Model + density out")
COHORTS = ("all_densities", "without_strongest")
PRIMARY_COLORS = {"math": "#ad416b", "code": "#258d9e", "qa": "#8662a6"}


def load_and_verify(audit, summary):
    """Check recorded inputs, benchmark identities, and saved OOF MAEs without refitting."""
    inputs = {}
    for rel, expected in sorted(summary["input_sha256"].items()):
        payload = audit.read(rel)
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f"V23 provenance hash mismatch: {rel}")
        inputs[rel] = payload
    pairs = {}
    for rr in summary["rows"]:
        original = inputs[rr["source_path"]]
        for cap in CAPS:
            measures = original["capabilities"][cap]
            pair = tuple(measures[role]["benchmark"] for role in ("primary", "secondary"))
            if cap in pairs and pair != pairs[cap]:
                raise ValueError(f"Inconsistent benchmark pair: {cap}")
            pairs[cap] = pair
            for role in ("primary", "secondary"):
                measure = measures[role]
                checked_close(rr[role][cap], measure["delta_L_c"], f"v26/v23 {rr['row_id']}/{cap}/{role}")
                checked_close(measure["delta_L_c"], measure["L_c"] - measure["baseline_L_c"],
                              f"dense subtraction {rr['row_id']}/{cap}/{role}")
    for cohort in COHORTS:
        for protocol in PROTOCOLS:
            for cap in CAPS:
                result = summary["analyses"][cohort][protocol][cap]
                for predictor, metric in result["metrics"].items():
                    errors = [abs(r["predictions"][predictor] - r["observed"]) for r in result["records"]]
                    checked_close(np.mean(errors), metric["mae"], f"OOF MAE {cohort}/{protocol}/{cap}/{predictor}")
                    checked_close(len(errors), metric["n"], "OOF metric count")
                metrics = result["metrics"]
                checked_close(metrics["cross_selected"]["mae"] - metrics[cap]["mae"],
                              result["same_over_cross_gain"], f"paired gain {cohort}/{protocol}/{cap}")
    return pairs


def main():
    setup_style()
    audit = Audit(4, "Measurement support: primary-to-secondary benchmark transfer")
    summary = audit.read("results/v26-loss-validity-pred/summary.json")
    pairs = load_and_verify(audit, summary)
    models = sorted({r["model"] for r in summary["rows"]})
    densities = sorted({r["density"] for r in summary["rows"]}, reverse=True)
    audit.rule(f"All v26 input_sha256 files read and hash-verified against V23; {len(models)} models "
               f"({', '.join(models)}), densities {densities}. Verify primary/secondary row deltas "
               "against capabilities[cap][role].delta_L_c and L_c - baseline_L_c in original V23 JSON.")
    audit.rule("Three capability panels, math/code/QA. Each shows all_densities and without_strongest "
               "(exclude d=.70), each with leave_model_out, leave_density_out, leave_model_and_density_out. "
               "Top: recorded MAE for every primary-capability predictor, cross_selected, zero, train_mean. "
               "The same-capability primary uses a diamond. Bottom: same_over_cross_gain and its stored gain_ci95. "
               "All saved MAEs are independently checked from the OOF records without fitting models.")
    audit.rule("Gain = MAE(cross-selected) - MAE(same-capability), positive = same-capability better. "
               f"Intervals: {summary['n_boot']} paired bootstrap draws; {summary['bootstrap_unit']}. "
               "Cross-selected source is chosen inside development CV, not by the displayed target MAEs. "
               "Zero/mean are source-free controls, not accuracy measures.")
    audit.omit("The numeric JSON supports all capability panels, so no fallback TeX table is needed. "
               "This is measured-primary to measured-secondary loss transfer, not downstream accuracy "
               "validity or density-only prediction. Native-token loss units differ across models. "
               "Stored descriptive CIs condition on fixed OOF fits and omit refitting/probe uncertainty.")
    audit.omit("OLMo sampling-resolution and Pearson diagnostics are available but are outside the requested "
               "prediction-gain plot; they are not substituted for prediction evidence. No repeated-run "
               "numerical-noise estimate is available in these measurements.")
    fig = plt.figure(figsize=(8.4, 8.4))
    grid = fig.add_gridspec(2, 3, left=.18, right=.98, bottom=.125, top=.86,
                          height_ratios=[1.4, 1], hspace=.30, wspace=.20)
    row_specs = [(cohort, protocol) for cohort in COHORTS for protocol in PROTOCOLS]
    labels = [f"{'All d' if i < 3 else 'd ≥.75'} · {PROTOCOL_LABEL[i % 3]}" for i in range(6)]
    for j, cap in enumerate(CAPS):
        top = fig.add_subplot(grid[0, j])
        bottom = fig.add_subplot(grid[1, j])
        primary, secondary = pairs[cap]
        top.set_title(f"{CAP_LABEL[cap]}\n{primary} →\n{secondary}", fontsize=10, pad=8)
        predictors = (*CAPS, "cross_selected", "zero", "train_mean")
        for y, (cohort, protocol) in enumerate(row_specs):
            result = summary["analyses"][cohort][protocol][cap]
            metrics = result["metrics"]
            for k, predictor in enumerate(predictors):
                color = PRIMARY_COLORS[predictor] if predictor in CAPS else COLORS[predictor]
                marker = "D" if predictor == cap else ("s" if predictor == "cross_selected" else "o")
                top.plot(metrics[predictor]["mae"], y + (k - 2.5) * .12, marker=marker,
                         color=color, ms=3.6, linestyle="none", mfc="white" if predictor == "zero" else color)
            gain = result["same_over_cross_gain"]
            lo, hi = result["gain_ci95"]
            bottom.hlines(y, lo, hi, color=COLORS["same_cap"], lw=1.3)
            bottom.vlines([lo, hi], y - .1, y + .1, color=COLORS["same_cap"], lw=.8)
            bottom.plot(gain, y, "D", color=COLORS["same_cap"], ms=4)
            audit.rule(f"{cohort}/{protocol}/{cap}: n={metrics[cap]['n']}, "
                       + ", ".join(f"{p} MAE={metrics[p]['mae']:.9f}" for p in predictors)
                       + f"; same-over-cross gain={gain:+.9f}, CI95=[{lo:+.9f}, {hi:+.9f}].")
        for ax in (top, bottom):
            ax.axhline(2.5, color="#999999", lw=.6)
            ax.set_ylim(5.6, -.6)
            ax.set_yticks(range(6), labels if j == 0 else [])
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", color="#ededed", lw=.5)
            ax.locator_params(axis="x", nbins=4)
        top.set_xlim(left=0)
        top.set_xlabel("Prediction MAE (nats/token)")
        bottom.axvline(0, color="#444444", lw=.8)
        bottom.set_xlabel("Same-over-cross gain (nats/token)")
        bottom.set_title("Paired gain and 95% interval", fontsize=9)
    legend = [Line2D([], [], marker="o", color=PRIMARY_COLORS[c], ls="", label=f"{CAP_LABEL[c]} primary") for c in CAPS]
    legend += [Line2D([], [], marker="s", color=COLORS["cross_selected"], ls="", label="Cross-selected"),
               Line2D([], [], marker="o", color=COLORS["zero"], mfc="white", ls="", label="Zero"),
               Line2D([], [], marker="o", color=COLORS["train_mean"], ls="", label="Train mean"),
               Line2D([], [], marker="D", color="black", ls="", label="Same-capability marker")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .985), frameon=False, ncol=4)
    fig.text(.5, .035, "Positive gain = same-capability predictor better. All measured densities retained in the first three rows.\n"
             "Six model clusters; fixed-fit descriptive intervals. Primary losses are held-out inputs; no accuracy claim.",
             ha="center", fontsize=8, linespacing=1.5)
    audit.save(fig, "measurement_support")
    audit.finish()


if __name__ == "__main__":
    main()
