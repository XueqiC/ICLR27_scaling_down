#!/usr/bin/env python3
"""Nine-state Pythia response panels from frozen V36/V53/V69 artifacts."""
from __future__ import annotations

if __package__:
    from .paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot
    from .paper_figure_style import panel_axes
    from .paper_panel_exports import ref, capability_handles, axes_defaults, export
else:
    from paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot
    from paper_figure_style import panel_axes
    from paper_panel_exports import ref, capability_handles, axes_defaults, export

P36 = "results/v36-pythia-controlled/summary.json"
P53 = "results/v53-prune-dev/register.json"
Q69 = "results/v69-quant-confirm/"
PANEL_SIZE = (2.7, 2.0)
GROUP_STATE = "pythia-410m@step143000"
CAPTIONS = {
    "a": """Pruning responses on the original nine development states: Pythia
160M, 410M and 1.4B at steps 16k, 64k and 143k. Thin coloured lines are each
state's measured loss minus its own dense loss, including every saved density.
Heavy Math/Code lines are the pointwise median of the nine state-conditioned
V53 power predictions; the heavy QA line is the delivered V53 median curve.
These summaries evaluate frozen coefficients without fitting. Delivered curves
are shown only on their [0.6, 0.9] domain. V53's fit used 17 development states;
the nine states displayed are selected by the original V36 input manifest.
The register and V6 loss artifacts are the pruning sources used by
plot_fig1_final.py; no held-out source is presented as a development state.
""",
    "b": """Per-output-channel symmetric round-to-nearest quantization on the
same nine V36 development states. Thin lines show each state's per-capability
loss change from its own dense anchor across every saved bit-width. The full
loss range is retained, including 3-bit collapse. The inputs are the frozen
V10 quant_losses.json files listed in V36's summary; plot_fig1_final.py does
not itself read a per-channel panel, so V36 supplies this additional artifact.
Colour follows Figure 1; positive loss change means worse performance.
""",
    "c": """Grouped symmetric RTN Math responses for Pythia 410M at step143k.
Colour and marker identify 3, 4 and 5 bits. Connected markers are measured
loss changes against group size, with no fitted curve. V69 develop.json supplies
groups 64, 128 and 256; V69 compare.json supplies groups 32 and 512. The latter
were unseen group sizes at the V69 freeze. These are the same frozen grouped
quantization artifacts read by plot_fig1_final.py. All three bit-widths share
one linear loss axis and a logarithmic group-size axis.
""",
}


def build(audit):
    import numpy as np
    if __package__:
        from .plot_fig1_final import prune_curve
    else:
        from plot_fig1_final import prune_curve
    manifest, register = audit.read(P36), audit.read(P53)
    paths = list(manifest["input_sha256"])
    states = {s["path"]: (i, s) for i, s in enumerate(register["dev_states"])}
    rows, prune_states = [], []
    for arm, directory, dense, panel in (("pruning", "v6-capability-geometry", "1.0", "a"),
                                         ("quantization", "v10-quantization", "dense", "b")):
        selected = [p for p in paths if p.startswith(f"results/{directory}/")]
        if len(selected) != 9:
            raise ValueError(f"Expected nine V36 {arm} states")
        for path in selected:
            data = audit.read(path)
            expected_hash = register["dev_hashes"][path] if panel == "a" else manifest["input_sha256"][path]
            if audit.inputs[path] != expected_hash:
                raise ValueError(f"Frozen {arm} digest mismatch: {path}")
            state = path.split("/")[-2].replace("--step", "@step")
            if panel == "a":
                prune_states.append(states[path])
            # The original manifest's file digests may predate appended densities;
            # current exact bytes are recorded by Artifacts; never hide additions.
            for key in sorted((k for k in data if not k.startswith("_") and (k != "dense")), key=float):
                for cap in CAPS:
                    rows.append({"panel": panel, "kind": "measured", "state": state,
                                 "capability": cap, "x": float(key),
                                 "delta": data[key][cap] - data[dense][cap],
                                 "loss_source": ref(path, key, cap), "dense_source": ref(path, dense, cap),
                                 "cohort_source": ref(P36, "input_sha256", path)})
    for cap in CAPS:
        method = "median_curve" if cap == "qa" else "power"
        for d in np.linspace(.6, .9, 121):
            values = [float(prune_curve(register, s, cap, float(d), method)) for _, s in prune_states]
            rows.append({"panel": "a", "kind": "delivered", "capability": cap,
                         "x": float(d), "delta": float(np.median(values)), "method": method,
                         "state_predictions": values,
                         "model_source": ref(P53, "models", cap, method),
                         "standardization_source": ref(P53, "standardization"),
                         "state_sources": [ref(P53, "dev_states", i) for i, _ in prune_states],
                         "formula": "median_state(prune_curve(register,state,cap,d,method))"})
    develop, freeze, compare = [audit.read(Q69 + name + ".json") for name in ("develop", "freeze", "compare")]
    if freeze["provenance"]["develop_sha256"] != audit.inputs[Q69 + "develop.json"]:
        raise ValueError("V69 development digest mismatch")
    if compare["provenance"]["freeze_sha256"] != audit.inputs[Q69 + "freeze.json"]:
        raise ValueError("V69 freeze digest mismatch")
    for data, section, name in ((develop, "dev_rows", "develop"), (compare, "rows", "compare")):
        for i, r in enumerate(data[section]):
            if r["state"] != GROUP_STATE or r["capability"] != "math":
                continue
            bit, group = map(int, r["config"].replace("b", "").split("_g"))
            rows.append({"panel": "c", "kind": "measured", "state": r["state"], "capability": "math",
                         "bit": bit, "x": group, "delta": r["dL"],
                         "delta_source": ref(Q69 + name + ".json", section, i, "dL"),
                         "config_source": ref(Q69 + name + ".json", section, i, "config")})
    if len([r for r in rows if r["panel"] == "c"]) != 15:
        raise ValueError("Incomplete 3-bit-width by 5-group panel")
    audit.rule("a/b: delta = loss_source - dense_source; x parsed from the JSON configuration key. "
               "a delivered: reuse plot_fig1_final.prune_curve; pointwise median of nine frozen "
               "state predictions for power, source-free median_curve for QA; no fitting. "
               "c: delta = V69 dL; bit and group parsed from config.")
    return rows


def draw_panel(fig, rows, panel):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator
    ax = panel_axes(fig, PANEL_SIZE, left=.55 if panel == "c" else .51, bottom=.48, right=.10)
    part = [r for r in rows if r["panel"] == panel]
    if panel in ("a", "b"):
        for cap in CAPS:
            measured = [r for r in part if r["kind"] == "measured" and r["capability"] == cap]
            for state in sorted({r["state"] for r in measured}):
                line = sorted((r for r in measured if r["state"] == state), key=lambda r: r["x"])
                ax.plot([r["x"] for r in line], [r["delta"] for r in line],
                        color=COLORS[cap], lw=.9, alpha=.45)
            heavy = [r for r in part if r["kind"] == "delivered" and r["capability"] == cap]
            if heavy:
                ax.plot([r["x"] for r in heavy], [r["delta"] for r in heavy], color=COLORS[cap], lw=2.5)
        ax.legend(handles=capability_handles(), loc="upper right")
        if panel == "a":
            ax.set(xlabel="Density $d$", xlim=(.535, 1.02), ylim=(-1, 10), xticks=[.6, .8, 1.])
        else:
            ax.set(xlabel="Bit-width", xlim=(2.7, 8.3), ylim=(-1, 27), xticks=[3, 4, 6, 8])
    else:
        handles = []
        for bit, marker, color in zip((3, 4, 5), ("o", "s", "^"), ("#71519a", "#c26a24", "#25836b")):
            line = sorted((r for r in part if r["bit"] == bit), key=lambda r: r["x"])
            ax.plot([r["x"] for r in line], [r["delta"] for r in line], color=color, marker=marker)
            handles.append(Line2D([], [], color=color, marker=marker, label=f"{bit} bit"))
        ax.set(xscale="log", xlabel="Group size", xlim=(26, 640), ylim=(-.25, 5.4))
        ax.set_xticks([32, 128, 512], ["32", "128", "512"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend(handles=handles, loc="upper left")
    axes_defaults(ax)
    ax.set_ylabel("Δ loss (nats)")
    return ax


def generate(root=ROOT):
    with frozen_run(root) as access:
        audit, plt = Artifacts(root), pyplot(root)
        rows = build(audit)
        panels = [(f"fig4_{p}", PANEL_SIZE, lambda f, p=p: draw_panel(f, rows, p),
                   [r for r in rows if r["panel"] == p], CAPTIONS[p]) for p in "abc"]
        export(plt, audit, "pythia_responses", panels, "\n".join(CAPTIONS.values()))
        return rows, audit, access


if __name__ == "__main__":
    generate()
