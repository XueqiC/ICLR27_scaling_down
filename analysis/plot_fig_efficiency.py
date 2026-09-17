#!/usr/bin/env python3
"""A14 CPU-only efficiency panels from A9 replicates and registered A11 scores.

Run: python -B analysis/plot_fig_efficiency.py
Writes eff_{a,b,legend} and appendix eff_qa_{a,b,legend} using the paper style,
plus A11 per_state.json. No fitting, model loading, or manuscript edits.
"""
from __future__ import annotations

import math
from unittest.mock import patch

if __package__:
    from . import a11_score as score
    from .paper_artifacts import ROOT, Artifacts, no_symlinks, pyplot
    from .paper_figure_style import (CAPABILITY_COLORS, PALETTE, apply_style,
        legend_strip, panel_axes, row_axis_style, save_panel, write_caption)
else:
    import a11_score as score
    from paper_artifacts import ROOT, Artifacts, no_symlinks, pyplot
    from paper_figure_style import (CAPABILITY_COLORS, PALETTE, apply_style,
        legend_strip, panel_axes, row_axis_style, save_panel, write_caption)

A9 = "results/a9-measurement-efficiency/summary.json"
A11 = "results/a11-efficiency-confirmation"
PANEL_SIZE = (2.7, 1.3)
LEGEND_SIZE = (5.5, .3)
METHODS = ("power", "A2", "median_curve")
CONFIRM_METHODS = ("power_18", "A2_36", "median_curve_36")
MARKERS = {"power": "o", "A2": "s", "median_curve": "^"}
LINES = {"power": "-", "A2": "--", "median_curve": ":"}
LABELS = {"power": "Power form", "A2": "Per-density regression", "median_curve": "Median curve"}
CAPTION = """Measurement efficiency for Math (blue) and Code (orange); QA is
shown separately in the appendix panels eff_qa_a and eff_qa_b. (a) Retrospective
A9 learning curves on the 42 new-state, in-range cells per capability, along the
densities-reduction axis. The nine development states are retained; the budgets
are 9, 18 and 36 state-density measurements per capability, excluding dense
anchors. Lines show median mean absolute error (MAE, nats) across 20 subset
replicates; light bands show the interquartile range among successful fits.
The power form, per-density regression A2 and source-free median density curve
use circle/solid, square/dashed and triangle/dotted encodings. The per-density
regression (A2) cannot be fitted at 9 measurements (0/20 fits) and has no marker
there. At 18, hollow squares and the regression band summarize only the 6/20
fitted replicates (14/20 failed); at 36 all 20/20 fit. These subset bands are not confidence intervals. (b) Registered A11
confirmation: each marker is the unweighted MAE across the six registered
densities (0.90, 0.85, 0.80, 0.75, 0.70, 0.65) of the indicated Pythia state.
Power uses 18 development measurements; A2 and the median curve each use 36.
States, left to right: 160M step 80k, 410M step 112k, 1.4B step 48k and 1B step
48k. Within each state, Math precedes Code and predictors follow the legend order;
horizontal positions are categorical. Errors are computed by a11_score from the
frozen functions and same-job dense/pruned losses, without refitting. No
independent-seed uncertainty is inferred from this shared-probe panel. The
registered 0.05-nat comparison is pooled across all four states, not a per-state
decision. Panels are 2.7 x 1.45 inches; the shared legend is 5.5 x 0.3 inches.
"""


def learning_curves(audit):
    """Read stored medians/IQRs; verify them against the recorded replicate MAEs."""
    import numpy as np
    summary = audit.read(A9)
    score.require(summary["validation"]["passed"], "A9 validation failed")
    rows = []
    for i, row in enumerate(summary["curves"]):
        if (row["axis"], row["split"], row["region"]) != ("densities", "new_state", "in_range"):
            continue
        if row["method"] not in METHODS:
            continue
        replicates = [(j, r) for j, r in enumerate(summary["records"])
                      if (r["axis"], r["cap"], r["method"], r["n_measurements_per_capability"]) ==
                      ("densities", row["cap"], row["method"], row["n_measurements_per_capability"])]
        values = [r["scores"]["new_state/in_range"]["mae"] for _, r in replicates
                  if r["scores"]["new_state/in_range"]["mae"] is not None]
        score.require(len(replicates) == row["R"] == 20 and len(values) == row["n_fitted"],
                      "A9 replicate count mismatch")
        expected = np.quantile(values, [.5, .25, .75]).tolist() if values else [None] * 3
        score.require(all(a == b or (a is not None and b is not None and math.isclose(a, b, abs_tol=1e-14))
                          for a, b in zip(expected, [row[k] for k in ("median", "q25", "q75")])),
                      "A9 stored curve disagrees with replicate scores")
        rows.append({**row, "source": f"{A9}#/curves/{i}",
                     "replicate_sources": [f"{A9}#/records/{j}/scores/new_state~1in_range" for j, _ in replicates],
                     "status_marker_y": 0 if not values else None})
    score.require(len(rows) == 27, "Expected three budgets x three predictors x three capabilities")
    for cap in score.CAPS:
        for method in METHODS:
            score.require(sorted(r["n_measurements_per_capability"] for r in rows
                                 if (r["cap"], r["method"]) == (cap, method)) == [9, 18, 36],
                          "Unexpected A9 budgets")
    return rows


def confirmation(audit):
    """Reuse all A11 validation and scoring; aggregate its six errors per state."""
    out = audit.root / A11
    # The frozen scorer remains byte-identical. Translate only authenticated
    # publication digests back to their registered identities during validation;
    # the artifact audit below records the actual published bytes.
    pairs = audit.published_pairs()
    registered = {published: original for original, (published, _) in pairs.items()}
    raw_digest = score.digest

    def registered_digest(path):
        actual = raw_digest(path)
        return registered.get(actual, actual)

    with patch.object(score, "digest", registered_digest):
        plan, frozen = score.load_registration(out, audit.root)
        prediction_hash = score.digest(out / "predictions.json")
        tables, provenance, missing = score.read_measurements(plan, frozen, out / "measurements", prediction_hash)
    score.require(not missing, "Incomplete A11 measurements: " + ", ".join(missing))
    scored = score.score_tables(plan, frozen, tables)
    stored = audit.read(f"{A11}/summary.json")
    for key in ("status", "by_capability", "cells", "efficiency_claim_supported", "decision"):
        score.require(stored[key] == scored[key], f"A11 summary disagrees with a11_score: {key}")
    score.require(stored["predictions_sha256"] == prediction_hash and
                  stored["measurement_sha256"] == provenance, "A11 summary provenance mismatch")
    for name in ("plan.json", "predictions.json", "predictions.json.sha256", "prereg.md"):
        audit.read(f"{A11}/{name}")
    # Record the validation-only runtime inputs, too; never import model code.
    audit.inputs.update({relative: raw_digest(audit.root / relative)
                         for relative in frozen["runtime_sha256"]})
    for target in plan["targets"]:
        for name in ("prune_losses.json", "metadata.json"):
            audit.read(f"{A11}/measurements/{target['state'].replace('@', '--')}/{name}")
    rows = []
    for target in plan["targets"]:
        for cap in score.CAPS:
            cells = [(i, r) for i, r in enumerate(scored["cells"])
                     if (r["state"], r["capability"]) == (target["state"], cap)]
            score.require(len(cells) == 6 and {r["density"] for _, r in cells} == set(plan["densities"]),
                          "Each state/capability must have all six densities")
            errors = {m: [r["absolute_errors"][m] for _, r in cells] for m in score.METHODS}
            # Match the scorer's centered summation, with six instead of 24 cells.
            mae = {m: v[0] + math.fsum(x - v[0] for x in v) / len(v) for m, v in errors.items()}
            rows.append(dict(state=target["state"], size=target["size"], step=target["step"],
                             capability=cap, n_cells=len(cells), densities=[r["density"] for _, r in cells],
                             mae=mae, absolute_errors=errors,
                             sources=[f"{A11}/summary.json#/cells/{i}/absolute_errors" for i, _ in cells]))
    return dict(schema_version=1, status="complete", cpu_only=True, unit="nats",
                metric="Unweighted mean absolute delta-loss error over six densities per state and capability",
                scorer="analysis/a11_score.py: load_registration, read_measurements, score_tables",
                predictions_sha256=prediction_hash, measurement_sha256=provenance,
                development_measurements_per_capability={m: frozen["coefficients"][m]["budget_per_capability"]
                                                        for m in score.METHODS},
                state_order=[t["state"] for t in plan["targets"]],
                by_capability=scored["by_capability"], records=rows)


def draw_learning(fig, rows, capabilities):
    import numpy as np
    ax = panel_axes(fig, PANEL_SIZE, left=.43, bottom=.53, top=.06)
    for cap in capabilities:
        for method in METHODS:
            part = sorted((r for r in rows if (r["cap"], r["method"]) == (cap, method)),
                          key=lambda r: r["n_measurements_per_capability"])
            xs = [r["n_measurements_per_capability"] for r in part]
            numeric = lambda key: [np.nan if r[key] is None else r[key] for r in part]
            color = CAPABILITY_COLORS[cap]
            ax.plot(xs, numeric("median"), color=color, ls=LINES[method])
            ax.fill_between(xs, numeric("q25"), numeric("q75"), color=color, alpha=.10, linewidth=0)
            for x, r in zip(xs, part):
                if not r["n_fitted"]:
                    continue  # unfittable budget: no marker (the caption states it)
                ax.plot(x, r["median"], ls="", marker=MARKERS[method],
                        color=color, mfc=color if r["n_fitted"] == r["R"] else PALETTE["white"])
    ymax = max(r["q75"] for r in rows if r["cap"] in capabilities and r["q75"] is not None)
    row_axis_style(ax)
    ax.set(xlim=(7, 38), ylim=(-.04 * ymax, 1.08 * ymax), xticks=[9, 18, 36],
           xlabel="Development measurements\nper capability", ylabel="MAE (nats)")
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))
    return ax


def draw_confirmation(fig, result, capabilities):
    from matplotlib.ticker import MaxNLocator
    ax = panel_axes(fig, PANEL_SIZE, left=.43, bottom=.49, top=.06)
    labels = []
    for index, state in enumerate(result["state_order"]):
        for ci, cap in enumerate(capabilities):
            row = next(r for r in result["records"] if (r["state"], r["capability"]) == (state, cap))
            for mi, (method, frozen_method) in enumerate(zip(METHODS, CONFIRM_METHODS)):
                x = index + (ci - (len(capabilities)-1)/2) * .30 + (mi-1) * .085
                ax.plot(x, row["mae"][frozen_method], ls="", marker=MARKERS[method], color=CAPABILITY_COLORS[cap])
        labels.append(row["size"].upper() + "\n" + str(row["step"] // 1000) + "k")
    ymax = max(r["mae"][m] for r in result["records"] if r["capability"] in capabilities for m in CONFIRM_METHODS)
    row_axis_style(ax)
    ax.set(xlim=(-.5, 3.5), ylim=(0, ymax * 1.12), xticks=range(4), xticklabels=labels,
           xlabel="Pythia size / training step", ylabel="MAE (nats)")
    ax.yaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))
    return ax


def draw_legend(fig, capabilities):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=CAPABILITY_COLORS[c], lw=2, label=c.title() if c != "qa" else "QA")
               for c in capabilities]
    handles += [Line2D([], [], color=PALETTE["reference"], ls=LINES[m], marker=MARKERS[m], label=LABELS[m])
                for m in METHODS]
    return legend_strip(fig, handles)


def generate(root=ROOT):
    audit = Artifacts(root)
    curves = learning_curves(audit)
    result = confirmation(audit)
    result["input_sha256"] = dict(audit.inputs)
    score.write_json(no_symlinks(audit.root / A11 / "per_state.json"), result)
    audit.rule(CAPTION)
    plt = pyplot(root)
    for prefix, caps in (("eff", ("math", "code")), ("eff_qa", ("qa",))):
        caption = CAPTION if prefix == "eff" else CAPTION.replace(
            "for Math (blue) and Code (orange); QA is\nshown separately in the appendix panels eff_qa_a and eff_qa_b.",
            "for QA (green), the descriptive appendix counterpart of the main Math/Code figure.").replace(
            "Math precedes Code", "only QA is shown")
        panel_rows = [r for r in curves if r["cap"] in caps]
        confirm_rows = [r for r in result["records"] if r["capability"] in caps]
        for suffix, size, kind, draw, records in (
            ("a", PANEL_SIZE, "double", lambda f: draw_learning(f, curves, caps), panel_rows),
            ("b", PANEL_SIZE, "double", lambda f: draw_confirmation(f, result, caps), confirm_rows),
            ("legend", LEGEND_SIZE, "legend", lambda f: draw_legend(f, caps), [])):
            apply_style(kind)
            fig = plt.figure(figsize=size)
            draw(fig)
            save_panel(fig, f"{prefix}_{suffix}", kind, audit, records)
            write_caption(f"{prefix}_{suffix}", audit, caption)
            plt.close(fig)
        write_caption(prefix, audit, caption)
    return result


def print_errors(result):
    print("\nA11 per-state MAE (nats; six densities per capability)")
    print("State | Capability | Power-18 | A2-36 | Median-36")
    for row in result["records"]:
        print(f"{row['state']} | {row['capability']} | " +
              " | ".join(f"{row['mae'][m]:.9f}" for m in CONFIRM_METHODS))


if __name__ == "__main__":
    print_errors(generate())
