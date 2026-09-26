#!/usr/bin/env python3
"""Pythia measured state traces and medians from frozen V36/V53/V69 artifacts."""
from __future__ import annotations

if __package__:
    from .paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
else:
    from paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp

if __package__:
    from .paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot
    from .paper_figure_style import finish_panel, panel_axes
    from .paper_panel_exports import ref, capability_handles, axes_defaults, export
else:
    from paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot
    from paper_figure_style import finish_panel, panel_axes
    from paper_panel_exports import ref, capability_handles, axes_defaults, export

P36 = "results/v36-pythia-controlled/summary.json"
P53 = "results/v53-prune-dev/register.json"
Q69 = "results/v69-quant-confirm/"
PANEL_SIZE = (1.8, 1.5)
LEGEND_SIZE = (5.5, .3)
BIT_MARKERS = {3: "o", 4: "s", 5: "^"}
CAPTIONS = {
    "a": """Pruning responses on the original nine development states: Pythia
160M, 410M and 1.4B at steps 16k, 64k and 143k. Thin faint lines show each
development state's measured loss change from its own dense loss. A bold line
with markers shows the pointwise median over the nine states. Both use only
densities measured in all nine states. Colour identifies Math, Code and QA;
positive loss change means worse performance. The loss axis is symmetric log
with a 0.1-nat linear threshold. Every saved measurement remains in the data
sidecar, including densities without all nine states. The original V36 input
manifest selects the states; the V53 register verifies the V6 loss artifacts'
digests. No held-out source is presented as a development state.
""",
    "b": """Per-output-channel symmetric round-to-nearest quantization on the
same nine V36 development states. Thin faint lines show each development
state's measured loss change from its own dense loss. A bold line with markers
shows the pointwise median over the nine states. Both use only bit-widths
measured in all nine states. The full loss range is retained, including 3-bit
collapse, on a symmetric log axis with a 0.1-nat linear threshold. The inputs are the frozen
V10 quant_losses.json files listed in V36's summary; plot_fig1_final.py does
not itself read a per-channel panel, so V36 supplies this additional artifact.
Colour follows Figure 1; positive loss change means worse performance.
""",
    "c": """Grouped symmetric RTN mathematics responses over all six development
states in V69 develop.json: Pythia 160M, 410M and 1.4B at steps 16k and 143k.
Colour and marker identify 3, 4 and 5 bits. Thin faint lines show each
development state's measured loss change from its own dense loss. A bold line
with markers shows the pointwise median over the six states. Both use group
sizes 64, 128 and 256, measured in all six states. All 54 measured rows come
from develop.json; compare.json is excluded. These are the frozen development artifacts
read by plot_fig1_final.py. Group size uses a base-2 logarithmic axis and loss
uses a symmetric log axis with a 0.1-nat linear threshold.
""",
}


def build(audit):
    manifest, register = audit.read(P36), audit.read(P53)
    paths = list(manifest["input_sha256"])
    rows = []
    for arm, directory, dense, panel in (("pruning", "v6-capability-geometry", "1.0", "a"),
                                         ("quantization", "v10-quantization", "dense", "b")):
        selected = [p for p in paths if p.startswith(f"results/{directory}/")]
        if len(selected) != 9:
            raise ValueError(f"Expected nine V36 {arm} states")
        for path in selected:
            data = audit.read(path)
            expected_hash = register["dev_hashes"][path] if panel == "a" else manifest["input_sha256"][path]
            if not audit.matches_digest(expected_hash, audit.inputs[path]):
                raise ValueError(f"Frozen {arm} digest mismatch: {path}")
            state = path.split("/")[-2].replace("--step", "@step")
            # The original manifest's file digests may predate appended densities;
            # current exact bytes are recorded by Artifacts; never hide additions.
            for key in sorted((k for k in data if not k.startswith("_") and (k != "dense")), key=float):
                for cap in CAPS:
                    rows.append({"panel": panel, "kind": "measured", "state": state,
                                 "capability": cap, "x": float(key),
                                 "delta": data[key][cap] - data[dense][cap],
                                 "loss_source": ref(path, key, cap), "dense_source": ref(path, dense, cap),
                                 "cohort_source": ref(P36, "input_sha256", path)})
    develop, freeze = [audit.read(Q69 + name + ".json") for name in ("develop", "freeze")]
    if not audit.matches_digest(freeze["provenance"]["develop_sha256"], audit.inputs[Q69 + "develop.json"]):
        raise ValueError("V69 development digest mismatch")
    group_states = {s["tag"] for s in develop["dev_states"]}
    if len(group_states) != 6:
        raise ValueError("Expected six V69 development states")
    for i, r in enumerate(develop["dev_rows"]):
        if r["capability"] != "math":
            continue
        bit, group = map(int, r["config"].replace("b", "").split("_g"))
        rows.append({"panel": "c", "kind": "measured", "state": r["state"], "capability": "math",
                     "bit": bit, "x": group, "delta": r["dL"],
                     "delta_source": ref(Q69 + "develop.json", "dev_rows", i, "dL"),
                     "config_source": ref(Q69 + "develop.json", "dev_rows", i, "config")})
    grouped = [r for r in rows if r["panel"] == "c"]
    expected = {(s, b, g) for s in group_states for b in (3, 4, 5) for g in (64, 128, 256)}
    if len(grouped) != 54 or {(r["state"], r["bit"], r["x"]) for r in grouped} != expected:
        raise ValueError("Incomplete six-state by three-bit-width by three-group panel")
    audit.rule("a/b: delta = loss_source - dense_source; x parsed from the JSON configuration key. "
               "V36 selects nine states per arm; V53 dev_hashes validate measured pruning inputs, "
               "and V36 input_sha256 validates measured per-channel quantization inputs. "
               "c: delta = V69 develop.json dL for all six development states; "
               "bit and group parsed from config; compare.json excluded. "
               "Plots summarize measured rows only at x shared by all states: "
               "thin faint lines for each state's measured loss change from its own dense loss, "
               "and a bold pointwise median line with markers; all measured rows are retained. "
               "All three loss axes are symmetric log with a 0.1-nat linear threshold.")
    return rows


def state_summary(rows, state_count):
    """Summarize complete state cohorts without changing the provenance rows: the shared
    coordinates, each state's values there, and their median."""
    import numpy as np
    states = sorted({r["state"] for r in rows})
    if len(states) != state_count:
        raise ValueError(f"Expected {state_count} states for response summary")
    cells = {}
    for r in rows:
        values = cells.setdefault(r["x"], {})
        if r["state"] in values:
            raise ValueError("Duplicate state at the same response coordinate")
        values[r["state"]] = r["delta"]
    xs = sorted(x for x, values in cells.items() if set(values) == set(states))
    if not xs:
        raise ValueError("No response coordinates shared by all states")
    values = np.array([[cells[x][s] for s in states] for x in xs])
    return xs, values, np.median(values, axis=1)


def draw_summary(ax, rows, state_count, color, marker="o"):
    """Each state as a thin faint trace, the median over states as a bold line with markers."""
    from matplotlib.collections import LineCollection
    xs, values, median = state_summary(rows, state_count)
    ax.add_collection(LineCollection([list(zip(xs, values[:, j])) for j in range(values.shape[1])],
                                     colors=[color], linewidths=.5, alpha=.3, zorder=2))
    # Keep the requested widths and exact median coordinates through export's
    # legacy Line2D normalization and marker-dodging pass.
    ax.add_collection(LineCollection([list(zip(xs, median))], colors=[color], linewidths=1.5, zorder=3))
    ax.scatter(xs, median, color=color, marker=marker, s=3.2**2, linewidths=.4, zorder=4)


def draw_panel(fig, rows, panel):
    from matplotlib.ticker import NullLocator
    ax = panel_axes(fig, PANEL_SIZE, left=.35, bottom=.30, right=.07)
    part = [r for r in rows if r["panel"] == panel]
    axes_defaults(ax)
    if panel in ("a", "b"):
        for cap in CAPS:
            measured = [r for r in part if r["kind"] == "measured" and r["capability"] == cap]
            draw_summary(ax, measured, 9, COLORS[cap])
        ax.set_yscale("symlog", linthresh=.1)
        if panel == "a":
            ax.set(xlabel="Retained density", xlim=(.535, 1.02), ylim=(-1.2, 12), xticks=[.6, .8, 1.])
        else:
            ax.set(xlabel="Bit-width", xlim=(2.7, 8.3), ylim=(-1.2, 30), xticks=[3, 4, 6, 8])
    else:
        for bit in (3, 4, 5):
            draw_summary(ax, [r for r in part if r["bit"] == bit], 6, BIT_COLORS[bit], BIT_MARKERS[bit])
        ax.set_xscale("log", base=2)
        ax.set_yscale("symlog", linthresh=.1)
        ax.set(xlabel="Group size", xlim=(56, 292), ylim=(-.025, 22))
        ax.set_xticks([64, 128, 256], ["64", "128", "256"])
        ax.xaxis.set_minor_locator(NullLocator())
    if panel in ("a", "b"):
        ax.set_yticks([-1, 0, .1, 1, 10], ["$-$1", "0", "0.1", "1", "10"])
    else:
        ax.set_yticks([0, .1, 1, 10], ["0", "0.1", "1", "10"])
        ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("Loss change (nats)")
    return finish_panel(ax)


def draw_legend(fig):
    from matplotlib.lines import Line2D
    if __package__:
        from .paper_figure_style import legend_strip
    else:
        from paper_figure_style import legend_strip
    handles = capability_handles()
    handles += [Line2D([], [], color=BIT_COLORS[b], marker=BIT_MARKERS[b], label=f"{b} bit")
                for b in (3, 4, 5)]
    handles += [Line2D([], [], color=PALETTE["dense"], lw=.5, label="One state")]
    # Measure the compact two-row strip with the exported preview's glyph
    # metrics; rounding at the default 100 dpi overestimates its height.
    fig.set_dpi(220)
    legend = legend_strip(fig, handles)
    for handle, label in zip(legend.legend_handles, legend.get_texts()):
        if label.get_text() == "One state":
            handle.set_linewidth(.5)
    return legend


def generate(root=ROOT):
    with frozen_run(root) as access:
        audit, plt = Artifacts(root), pyplot(root)
        rows = build(audit)
        panels = [(f"fig4_{p}", PANEL_SIZE, lambda f, p=p: draw_panel(f, rows, p),
                   [r for r in rows if r["panel"] == p], CAPTIONS[p]) for p in "abc"]
        layout = (
            "Three 1.8 x 1.5-inch panels in one 5.5-inch row; fig4_legend.pdf "
            "is the shared capability, bit-width and one-state key above the panels. "
            "State traces are 0.5 pt at alpha 0.3; median lines are 1.5 pt with 3.2-pt markers.\n")
        caption = "\n".join(CAPTIONS.values()) + layout
        panels = [(name, size, draw, records, text + layout)
                  for name, size, draw, records, text in panels]
        export(plt, audit, "pythia_responses", panels, caption, width=5.5,
               legend=("fig4_legend", LEGEND_SIZE, draw_legend, [], caption))
        return rows, audit, access


if __name__ == "__main__":
    generate()
