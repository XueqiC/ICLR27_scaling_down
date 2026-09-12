#!/usr/bin/env python3
"""V82 Figure 1, CPU only; evaluate frozen objects, never fit or rebuild results.

Run from any directory with Python, NumPy and matplotlib. Writes only
paper/paper/figs/final_relations.{pdf,png} and results/v82-fig1/notes.md.
The caption and text-width inclusion live in paper/paper/laws.tex.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "v82-fig1-mpl"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FONT = 5.5, 2.68, 7.5
CAPS = ("math", "code", "qa")
CAP_COLORS = dict(zip(CAPS, ("#2166ac", "#c76b16", "#22834a")))
CAP_MARKERS = dict(zip(CAPS, ("o", "s", "^")))
QUANT_STATES = ("pythia-410m@step143000", "pythia-1.4b@step16000")
STUDENTS = ("gemma3-270m", "gemma3-1b")
STATE_COLORS = ("#71519a", "#b75c16")
STATE_MARKERS = ("o", "s")
INPUTS = {}


def read(relative):
    raw = (ROOT / relative).read_bytes()
    INPUTS[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def close(a, b, context):
    if not np.allclose(a, b, rtol=1e-9, atol=1e-10):
        raise ValueError(f"Frozen-object check failed: {context}: {a} != {b}")


def style():
    # Same Times-like bold setup and TrueType fallback as plot_fig1_responses.py.
    families = ["Nimbus Roman", "Liberation Serif", "Times New Roman"]
    resolved = Path(font_manager.findfont(font_manager.FontProperties(family=families, weight="bold")))
    if resolved.suffix.lower() == ".otf":
        resolved = Path(font_manager.findfont(
            font_manager.FontProperties(family="Liberation Serif", weight="bold"),
            fallback_to_default=False))
        families = ["Liberation Serif", "Nimbus Roman", "Times New Roman"]
    plt.rcParams.update({
        "font.family": "serif", "font.serif": families, "font.weight": "bold",
        "axes.labelweight": "bold", "axes.titleweight": "bold",
        "font.size": FONT, "axes.labelsize": FONT, "axes.titlesize": 8.5,
        "xtick.labelsize": FONT, "ytick.labelsize": FONT, "legend.fontsize": FONT,
        "legend.title_fontsize": FONT, "figure.titlesize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .55,
        "xtick.major.width": .5, "ytick.major.width": .5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "axes.labelpad": 2, "axes.unicode_minus": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })
    return resolved


def linear_extend(x, xp, yp):
    """Adjacent interpolation and nearest-boundary-pair linear extension."""
    x, xp, yp = np.asarray(x), np.asarray(xp), np.asarray(yp)
    i = np.clip(np.searchsorted(xp, x, side="right") - 1, 0, len(xp) - 2)
    return yp[i] + (yp[i + 1] - yp[i]) * (x - xp[i]) / (xp[i + 1] - xp[i])


def phi(raw, stats):
    return np.r_[1., (np.asarray(raw) - stats["center"]) / stats["scale"]]


def prune_curve(register, target, cap, d, method):
    model = register["models"][cap][method]
    if method == "median_curve":
        anchors = sorted((float(k), v) for k, v in model["anchors"].items())
        return linear_extend(d, *zip(*anchors))
    z = phi([np.log(target["N0"]), target["L0"][cap], np.log(target["D0"])],
            register["standardization"])
    return np.dot(model["beta"], z) * ((1 - np.asarray(d)) / .3) ** model["gamma"]


def quant_curve(freeze, state, bit, g, method):
    anchors = freeze["models"]["math"][method]["anchors"]
    if method == "same_input_interpolation":
        z = phi([np.log(state["N0"]), state["L0"]["math"], np.log(state["D0"])],
                freeze["standardization"])
        values = [np.dot(anchors[f"b{bit}_g{k}"], z) for k in (64, 128, 256)]
    else:
        values = [anchors[f"b{bit}_g{k}"] for k in (64, 128, 256)]
    g = np.asarray(g)
    y = linear_extend(np.log2(g), [6, 7, 8], values)
    # The frozen zero floor applies ONLY outside the measured group-size grid.
    return np.where((g < 64) | (g > 256), np.maximum(0., y), y)


def distill_curve(freeze, cap, method, student, du, e):
    """V70 raw-coordinate basis; coef already reverses training standardization."""
    refs = freeze["refs"]
    e = np.asarray(e)
    u, w = np.log1p(e * du / refs["T_ref"]), np.log1p(e)
    v = np.log(du / refs["D_ref"])
    beta = freeze["models"][cap][method]["coef"]
    if method == "E":
        return beta[0] * w
    if method == "joint":
        return u * (beta[0] + beta[1] * u + beta[2] * v)
    if method in ("T-only", "E-only"):
        return beta[0] + beta[1] * (u if method == "T-only" else w)
    if method.startswith("surface:"):
        descriptor = method.split(":")[1]
        stats = refs["L0"][cap] if descriptor == "L0" else refs["logN"]
        value = refs["L0_by_student"][student][cap] if descriptor == "L0" else np.log(refs["N"][student])
        z = (value - stats["mean"]) / (stats["std"] or 1.)
        return beta[0] + beta[1] * u + beta[2] * w + beta[3] * z
    raise ValueError(f"Unexpected frozen method: {method}")


def legend(fig, handles, x, y, columns=1):
    return fig.legend(handles=handles, loc="center", bbox_to_anchor=(x / WIDTH, y / HEIGHT),
                      ncol=columns, frameon=False, borderaxespad=0, borderpad=0,
                      handlelength=1.25, handletextpad=.3, columnspacing=.65, labelspacing=.1)


def key(label, color="#333333", marker=None, ls="None", **kwargs):
    return Line2D([], [], label=label, color=color, marker=marker, linestyle=ls,
                  markersize=3.1, linewidth=1., **kwargs)


def axes_at(fig, x, y, width, height):
    ax = fig.add_axes([x / WIDTH, y / HEIGHT, width / WIDTH, height / HEIGHT])
    ax.axhline(0, color="#bdbdbd", lw=.5, zorder=0)
    ax.grid(axis="y", color="#e8e8e8", lw=.45, zorder=0)
    return ax


def log_ticks(ax, values):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(values))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())


def panel_a(fig, reg, pred, losses):
    ax = axes_at(fig, .40, .55, 1.29, 1.54)
    target = pred["target"]
    assert target["tag"] == "pythia-1.4b@step112000"
    assert reg["n_dev_states"] == len(reg["dev_states"]) == 17
    assert reg["selected_candidate"] == "power"
    densities = sorted(float(k) for k in losses if not k.startswith("_"))
    support = [d for state in reg["dev_states"] for d in state["densities"]]
    ax.axvspan(min(support), max(support), color="#e7e1ce", alpha=.65, zorder=-1)
    d = np.linspace(min(support), 1., 401)
    for cap in CAPS:
        close(losses["1.0"][cap], target["L0"][cap], f"A dense/{cap}")
        for ds, methods in pred["predictions"][cap].items():
            for method in ("power", "median_curve"):
                close(prune_curve(reg, target, cap, float(ds), method), methods[method], f"A {cap}/{ds}/{method}")
        color = CAP_COLORS[cap]
        ax.plot(d, prune_curve(reg, target, cap, d, "power"), color=color, lw=1.15)
        # Show frozen median anchors only over their support; no invented dense anchor.
        anchors = sorted(float(k) for k in reg["models"][cap]["median_curve"]["anchors"])
        ax.plot(anchors, prune_curve(reg, target, cap, anchors, "median_curve"),
                color=color, lw=1., ls="--")
        ax.plot(densities, [losses[str(x)][cap] - losses["1.0"][cap] for x in densities],
                linestyle="None", marker=CAP_MARKERS[cap], color=color, ms=3.5,
                markeredgecolor="white", markeredgewidth=.3, zorder=5)
    ax.set(xlim=(.535, 1.02), ylim=(-.85, 3.05), xticks=[.6, .8, 1.], yticks=[0, 1, 2, 3],
           xlabel="Retained density d", ylabel="ΔL (nats)")
    ax.text(.98, .96, "17-state range", transform=ax.transAxes, ha="right", va="top", fontsize=FONT)
    legend(fig, [key(c.title() if c != "qa" else "QA", CAP_COLORS[c], CAP_MARKERS[c]) for c in CAPS],
           1.045, 2.235, 3)
    return len(densities), (min(support), max(support))


def panel_b(fig, freeze, develop, compare):
    assert compare["complete"]
    states = {s["tag"]: s for s in freeze["states"]}
    dev = {(r["state"], r["config"]): r for r in develop["dev_rows"] if r["capability"] == "math"}
    conf = {(r["state"], r["config"]): r for r in compare["rows"] if r["capability"] == "math"}
    frozen = {(r["state"], r["config"]): r for r in freeze["predictions"] if r["capability"] == "math"}
    g = np.unique(np.r_[np.geomspace(32, 512, 513), 32, 64, 128, 256, 512])
    assert not any(r["config"].endswith(("_g32", "_g512")) for r in develop["dev_rows"])
    for i, bit in enumerate((3, 4, 5)):
        ax = axes_at(fig, 2.18, .55 + (2 - i) * .535, 1.26, .47)
        for j, tag in enumerate(QUANT_STATES):
            state = states[tag]
            ax.plot(g, quant_curve(freeze, state, bit, g, "same_input_interpolation"),
                    color=STATE_COLORS[j], lw=1.05)
            for group in (32, 512):
                row = frozen[tag, f"b{bit}_g{group}"]
                assert row["predictions"] == conf[tag, f"b{bit}_g{group}"]["predictions"]
                for method in ("same_input_interpolation", "median"):
                    close(quant_curve(freeze, state, bit, group, method), row["predictions"][method],
                          f"B {tag}/{bit}/{group}/{method}")
                # Directly use the pre-measurement predictions for the hollow marks.
                ax.plot(group, row["predictions"]["same_input_interpolation"], "D", ms=4.,
                        mfc="white", mec=STATE_COLORS[j], mew=.85, zorder=4)
            groups = [32, 64, 128, 256, 512]
            values = [(conf if k in (32, 512) else dev)[tag, f"b{bit}_g{k}"]["dL"] for k in groups]
            ax.plot(groups, values, linestyle="None", marker=STATE_MARKERS[j], ms=2.9,
                    color=STATE_COLORS[j], markeredgecolor="white", markeredgewidth=.25, zorder=5)
        ax.plot(g, quant_curve(freeze, states[QUANT_STATES[0]], bit, g, "median"),
                color="#555555", ls="--", lw=.95)
        log_ticks(ax, [32, 64, 128, 256, 512])
        ax.set_xlim(26, 640)
        limits = [(0, 5.4), (0, .52), (-.009, .135)][i]
        ax.set_ylim(*limits)
        ax.set_yticks([[0, 2, 4], [0, .2, .4], [0, .05, .10]][i])
        ax.text(.03, .96, f"{bit} bit", transform=ax.transAxes, va="top", fontsize=FONT,
                bbox=dict(facecolor="white", edgecolor="none", pad=.2, alpha=.8))
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Group size g")
    legend(fig, [key("410M / 143k", STATE_COLORS[0], "o"), key("1.4B / 16k", STATE_COLORS[1], "s")],
           2.81, 2.27)
    return 30, 12


def panel_c(fig, freeze, compare):
    assert compare["complete"]
    rows = compare["rows"]
    assert len(rows) == 108
    frozen = {(r["student"], r["pool"], r["T_planned"], r["capability"]): r for r in freeze["predictions"]}
    for r in rows:
        fr = frozen[r["student"], r["pool"], r["T_planned"], r["capability"]]
        assert r["predictions"] == fr["predictions"]
        close(r["E_planned"], r["T_planned"] / r["DU"], "C planned reuse")
        for method in (r["selected"], r["strongest_baseline"]):
            close(distill_curve(freeze, r["capability"], method, r["student"], r["DU"], r["E_planned"]),
                  fr["predictions"][method], f"C {r['student']}/{r['pool']}/{r['capability']}/{method}")
    for i, cap in enumerate(CAPS):
        ax = axes_at(fig, 3.98, .55 + (2 - i) * .535, 1.39, .47)
        selected = freeze["selected"][cap]["method"]
        assert selected == ("joint" if cap == "qa" else "E")
        drawn = set()
        for j, student in enumerate(STUDENTS):
            for seed in range(31, 37):
                rr = sorted((r for r in rows if r["capability"] == cap and r["student"] == student
                             and r["data_seed"] == seed), key=lambda r: r["T_planned"])
                assert [r["T_planned"] for r in rr] == [50000, 100000, 200000]
                assert len({r["DU"] for r in rr}) == 1
                du = rr[0]["DU"]
                assert du == freeze["confirmation_register"]["pools"][f"U200_s{seed}"]["D_U_completion"]
                actual_e = np.array([r["T_actual"] / du for r in rr])
                # Thin, translucent connectors identify each actual three-point run.
                ax.plot(actual_e, [r["actual"] for r in rr], marker=STATE_MARKERS[j], ms=2.35,
                        color=CAP_COLORS[cap], mec="white", mew=.2, lw=.45, alpha=.60, zorder=3)
                baseline = freeze["strongest_baseline"][student][cap]["method"]
                # E-only curves coincide across students and pools. Draw these once
                # over the union domain; retain every DU-dependent frozen curve.
                for method, ls in ((selected, "-"), (baseline, ":")):
                    curve_id = (method, du if method not in ("E", "E-only") else None,
                                student if method.startswith("surface:") else None)
                    if curve_id in drawn:
                        continue
                    drawn.add(curve_id)
                    if method in ("E", "E-only"):
                        cr = [r for r in rows if r["capability"] == cap]
                        e_plot = np.geomspace(min(r["E_planned"] for r in cr),
                                              max(r["T_actual"] / r["DU"] for r in cr), 201)
                    else:
                        # Coincident curves may belong to two students whose actual
                        # last-update overshoots differ: preserve their union domain.
                        cr = [r for r in rows if r["capability"] == cap and r["DU"] == du
                              and (not method.startswith("surface:") or r["student"] == student)]
                        e_plot = np.geomspace(min(r["E_planned"] for r in cr),
                                              max(r["T_actual"] / r["DU"] for r in cr), 151)
                    ax.plot(e_plot, distill_curve(freeze, cap, method, student, du, e_plot),
                            color=CAP_COLORS[cap] if ls == "-" else ("#555555" if j == 0 else "#999999"),
                            ls=ls, lw=1.15 if ls == "-" else 1., alpha=.85, zorder=4 if ls == "-" else 2)
        log_ticks(ax, [1, 2, 4])
        ax.set_xlim(.79, 4.65)
        ax.set_ylim(*[(-.005, .29), (-.005, .29), (-1.8, 1.)][i])
        ax.set_yticks([[0, .1, .2], [0, .1, .2], [-1, 0]][i])
        ax.text(.03, .96, cap.title() if cap != "qa" else "QA", transform=ax.transAxes,
                va="top", color=CAP_COLORS[cap], fontsize=FONT,
                bbox=dict(facecolor="white", edgecolor="none", pad=.2, alpha=.8))
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Reuse count E")
    legend(fig, [key("270M", marker="o"), key("1B", marker="s")], 4.675, 2.235, 2)
    return len({(r["student"], r["pool"]) for r in rows})


def validate_figure(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    for t in texts:
        if t.get_fontsize() < 7:
            raise ValueError(f"Undersized font: {t.get_text()!r}")
        box = t.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1:
            raise ValueError(f"Text outside fixed page: {t.get_text()!r}")
    close(fig.get_size_inches(), [WIDTH, HEIGHT], "physical figure size")
    assert HEIGHT <= 2.7
    return min(t.get_fontsize() for t in texts)


def notes(font, count_a, support, count_b, marks_b, count_c, min_font):
    lines = ["# V82 Figure 1 build note", "",
        "Build: `python -B analysis/plot_fig1_final.py` (CPU, Agg, NumPy/matplotlib; no fitting).",
        f"Outputs: `paper/paper/figs/final_relations.pdf` and `.png`; {WIDTH} × {HEIGHT} in, "
        f"300 dpi PNG (1650 × 804), all figure text ≥ {min_font:g} pt. "
        "Paper inclusion/caption: `paper/paper/laws.tex`, width=`\\textwidth` (5.5 in in `iclr2027_conference.sty`).",
        f"Font: `{font}`; Times-like bold setup from `analysis/plot_fig1_responses.py`, "
        "embedded TrueType. Labels use full-size plain text, without reduced math subscripts.", "",
        "- **A, measured markers:** `results/v6-capability-geometry/pythia-1.4b--step112000/prune_losses.json`, "
        f"every numeric key ({count_a} densities including the dense reference at d=1); each capability minus its `1.0` loss. "
        "The three dense-reference markers coincide at zero. This confirmation checkpoint is absent from the 17-state fit; its model size was seen.",
        "- **A, solid power lines:** `results/v53-prune-dev/register.json::models[cap].power.{beta,gamma}` and "
        "`standardization.{center,scale}`; source inputs from "
        "`results/v53-prune-dev/predictions_pythia-1.4b@step112000.json::target.{N0,D0,L0}`. "
        "Evaluate `(beta·[1,z(log N0),z(L0),z(log D0)])*((1-d)/0.3)**gamma`. "
        "Checked against that prediction file's `predictions[cap][density].power` at all three frozen densities.",
        "- **A, dashed median lines:** the same register's `models[cap].median_curve.anchors`, "
        "joined linearly in d and shown only over their frozen support; no artificial dense anchor. "
        "Checked against `predictions[cap][density].median_curve`. "
        f"Shading [{support[0]}, {support[1]}] is the union range of `dev_states[*].densities` (17 states).",
        "- **B, measured markers:** `results/v69-quant-confirm/develop.json::dev_rows[*].dL` at g=64,128,256 "
        "and `results/v69-quant-confirm/compare.json::rows[*].dL` at g=32,512, filtered to math, "
        "`pythia-410m@step143000` and `pythia-1.4b@step16000`, b=3,4,5 "
        f"({count_b} points). Three rows use separate linear response scales; g is log2.",
        "- **B, solid interpolation lines:** `results/v69-quant-confirm/freeze.json::models.math.same_input_interpolation.anchors`, "
        "`states` (N0,D0,L0) and `standardization`; per-grid-cell dot product with standardized source features, "
        "then piecewise linear in log2(g) at each observed b. Its `boundary_rule` extends the nearest pair "
        "64/128 below 64 and 128/256 above 256, with `max(0, extrapolated dL)` outside only. "
        f"The {marks_b} hollow diamonds directly plot `predictions[*].predictions.same_input_interpolation` at g=32,512; "
        "all endpoints are checked against the formula. These group sizes were never measured before freezing; later confirmation measurements are filled markers.",
        "- **B, dashed median lines:** the same freeze's `models.math.median.anchors` with its identical "
        "piecewise/boundary rule, checked against `predictions[*].predictions.median`. "
        "One shared median line per bit row. The requested interpolation is displayed explicitly; "
        "`selected_candidate` in v69's pre-confirmation freeze is not used as the plotted method.",
        "- **C, measured trajectories:** `results/v70-distill-confirm/compare.json::rows`, "
        f"all 108 capability rows, {count_c} student/pool trajectories (270M and 1B × seeds 31–36, U=200), "
        "three planned checkpoints T=50k/100k/200k. Markers are `actual` versus `T_actual/DU`; "
        "thin translucent connectors join only the three measured points of each run. No pool averaging or invented checkpoints.",
        "- **C, solid selected lines:** `results/v70-distill-confirm/freeze.json::selected[cap].method`, "
        "`models[cap][method].coef`, `refs`, and `confirmation_register.pools[pool].D_U_completion`. "
        "Math/code use `E`: `coef[0]*log1p(E)`; QA uses `joint`: "
        "`u*(coef[0]+coef[1]*u+coef[2]*v)`, u=log1p(E*DU/refs.T_ref), v=log(DU/refs.D_ref). "
        "No student term is present in these selected forms. Draw each DU-dependent curve; exactly coincident E-only curves are drawn once over their union domain.",
        "- **C, dotted strongest-baseline lines:** the same freeze's `strongest_baseline[student][cap].method` "
        "and `models[cap][method].coef`: 270M math=`T-only`, code/QA=`E-only`; "
        "1B math=`surface:L0`, code=`E-only`, QA=`surface:logN`. "
        "`T-only`=b0+b1*u; `E-only`=b0+b1*log1p(E); `surface:*`=b0+b1*u+b2*log1p(E)+b3*z, "
        "with z from frozen `refs.L0[cap]`/`refs.L0_by_student` or `refs.logN`/`refs.N`. "
        "270M dotted curves are dark grey; 1B curves light grey; shared code baseline is drawn once. "
        "Six separate DU-dependent curves per applicable student/method; coefficients are never standardized a second time.",
        "- **C exposure convention:** curves evaluate unchanged frozen coefficients over the displayed reuse range. "
        "Each selected and baseline formula is checked against `freeze.json::predictions[*].predictions[method]` "
        "at `E_planned=T_planned/DU` (216 checks). The plotted measured x positions include actual budget overshoot; "
        "the stored planned-T predictions and confirmation scores remain unchanged. Neither curves nor baselines are selected using confirmation responses.",
        "- **Common guides:** horizontal grey lines mark zero response; pale horizontal grids are axis guides. "
        "Filled circles/squares identify the B states and C students, A uses capability markers/colours. "
        "No caption is rasterized into the figure.", "",
        "Validation: recorded prediction identities, all 12 complete trajectories, frozen-file hash links, "
        "minimum font size and page bounds checked before saving. Existing result/figure inputs are read only. "
        "Fixed PDF metadata; only the two new figure outputs and this note are written by the script.", "",
        "Exact JSON files read (SHA-256):", ""]
    lines.extend(f"- `{p}`: `{sha}`" for p, sha in INPUTS.items())
    out = ROOT / "results/v82-fig1/notes.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main():
    font = style()
    reg = read("results/v53-prune-dev/register.json")
    pred = read("results/v53-prune-dev/predictions_pythia-1.4b@step112000.json")
    losses = read("results/v6-capability-geometry/pythia-1.4b--step112000/prune_losses.json")
    qf = read("results/v69-quant-confirm/freeze.json")
    qd = read("results/v69-quant-confirm/develop.json")
    qc = read("results/v69-quant-confirm/compare.json")
    df = read("results/v70-distill-confirm/freeze.json")
    dc = read("results/v70-distill-confirm/compare.json")
    assert pred["provenance"]["register_sha256"] == INPUTS["results/v53-prune-dev/register.json"]
    assert qc["provenance"]["freeze_sha256"] == INPUTS["results/v69-quant-confirm/freeze.json"]
    assert qf["provenance"]["develop_sha256"] == INPUTS["results/v69-quant-confirm/develop.json"]
    assert dc["freeze_sha256"] == INPUTS["results/v70-distill-confirm/freeze.json"]
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    for x, title, subtitle in ((1.045, "A  Pruning", "1.4B / 112k"),
                               (2.81, "B  Group quantization", "Math · ΔL (nats)"),
                               (4.675, "C  Distillation", "U = 200 · δ (nats)")):
        fig.text(x / WIDTH, 2.60 / HEIGHT, title, ha="center", va="center", fontsize=8.5)
        fig.text(x / WIDTH, 2.435 / HEIGHT, subtitle, ha="center", va="center", fontsize=FONT)
    count_a, support = panel_a(fig, reg, pred, losses)
    count_b, marks_b = panel_b(fig, qf, qd, qc)
    count_c = panel_c(fig, df, dc)
    legend(fig, [key("Frozen relation", ls="-"), key("Median (A, B)", ls="--"),
                 key("Strongest baseline (C)", ls=":"),
                 key("Unseen g at freeze", marker="D", markerfacecolor="white")],
           WIDTH / 2, .105, 4)
    min_font = validate_figure(fig)
    for p, digest in INPUTS.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest, f"Input changed: {p}"
    for ext in ("pdf", "png"):
        out = ROOT / f"paper/paper/figs/final_relations.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        kw = {"metadata": {"CreationDate": None, "ModDate": None}} if ext == "pdf" else {}
        fig.savefig(out, dpi=300, **kw)  # Never tight-crop: preserve exact physical size.
        print(f"WROTE {out}")
    plt.close(fig)
    notes(font, count_a, support, count_b, marks_b, count_c, min_font)
    print(f"VERIFIED A: {count_a} densities/capability; B: {count_b} measurements, {marks_b} frozen diamonds; "
          f"C: {count_c} trajectories, 108 responses. {WIDTH} × {HEIGHT} in; minimum {min_font:g} pt.")


if __name__ == "__main__":
    main()
