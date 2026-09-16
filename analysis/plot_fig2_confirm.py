#!/usr/bin/env python3
"""V83 Figure 2: latest frozen confirmations, CPU only, no fitting.

Run: python -B analysis/plot_fig2_confirm.py
Writes only paper/paper/figs/final_confirmations.{pdf,png} and
results/v83-fig2/notes.md. All experimental results are read only.
"""
from __future__ import annotations

if __package__:
    from .paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
else:
    from paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "v83-fig2-mpl"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FONT = 5.5, 2.4, 7.2
CAPS = ("math", "code", "qa")
CAP_LABELS = {"math": "Math", "code": "Code", "qa": "QA"}
POSITIVE, NEGATIVE = PALETTE["math"], PALETTE["grid"]
INPUTS = {}
METHOD_LABELS = {
    "power": "Power form", "A2": "Per-density source regression",
    "median_curve": "Median density curve",
    "same_input_interpolation": "Same-input interpolation",
    "low_order_2d": "Low-order surface", "median": "Per-configuration median",
    "E": "Zero-anchored reuse form", "joint": "Joint token/reuse form",
    "T-only": "Token-budget regression", "E-only": "Reuse regression",
    "surface:L0": "Surface with dense loss", "surface:logN": "Surface with log size",
}


def require(condition, context):
    if not condition:
        raise ValueError(context)


def close(actual, expected, context):
    require(np.allclose(actual, expected, rtol=1e-11, atol=1e-13),
            f"Score/provenance mismatch: {context}: {actual} != {expected}")


def read(relative):
    raw = (ROOT / relative).read_bytes()
    INPUTS[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


@dataclass(frozen=True)
class Bar:
    arm: str
    group: str
    cap: str
    candidate: str
    baseline: str
    candidate_mae: float
    baseline_mae: float
    n: int
    references: dict
    source: str
    ci95: tuple | None = None

    @property
    def gain(self):
        return self.baseline_mae - self.candidate_mae


def paired_rows(compare, freeze, identity, observed):
    """Check frozen prediction identity and error arithmetic without refitting."""
    key = lambda r: tuple(r[k] for k in identity)
    frozen = {key(r): r for r in freeze["predictions"]}
    measured = {key(r): r for r in compare["rows"]}
    require(len(frozen) == len(freeze["predictions"]), "Duplicate frozen rows")
    require(len(measured) == len(compare["rows"]), "Duplicate measured rows")
    require(frozen.keys() == measured.keys(), "Incomplete frozen confirmation")
    for ident, row in measured.items():
        require(row["predictions"] == frozen[ident]["predictions"],
                f"Changed predictions: {ident}")
        for method, prediction in row["predictions"].items():
            close(abs(prediction - row[observed]), row["absolute_errors"][method],
                  f"{ident}/{method}")


def pruning_bars():
    paths = [f"results/v53-prune-dev/compare_pythia-{tag}.json"
             for tag in ("410m@step48000", "1.4b@step112000", "6.9b@step80000")]
    comparisons = [read(p) for p in paths]
    register_path = "results/v53-prune-dev/register.json"
    read(register_path)
    for path, comp in zip(paths, comparisons):
        prediction_path = path.replace("compare_", "predictions_")
        pred = read(prediction_path)
        require(comp["provenance"]["predictions_sha256"] == INPUTS[prediction_path],
                f"Prediction hash: {path}")
        require(pred["provenance"]["register_sha256"] == INPUTS[register_path],
                f"Register hash: {prediction_path}")
        require(comp["selected_candidate"] == "power", "Changed pruning candidate")
        close(sorted(comp["densities"]), [.575, .675, .85], "Confirmation densities")
        for cap in CAPS:
            for method in ("power", "A2", "median_curve"):
                errors = []
                for density in comp["densities"]:
                    ds = str(density)
                    error = abs(pred["predictions"][cap][ds][method]
                                - comp["observed_delta_loss"][cap][ds])
                    close(error, comp["absolute_errors"][cap][ds][method], path)
                    errors.append(error)
                close(np.mean(errors), comp["mae"][method][cap], f"{path}/{cap}/{method}")
    bars = []
    for cap in CAPS:
        # Pool all nine cells BEFORE selecting one baseline per capability.
        maes = {m: float(np.mean([c["absolute_errors"][cap][str(d)][m]
                                  for c in comparisons for d in c["densities"]]))
                for m in ("power", "A2", "median_curve")}
        refs = {m: maes[m] for m in ("A2", "median_curve")}
        baseline = min(refs, key=refs.get)
        bars.append(Bar("Pruning", "Three-checkpoint panel", cap, "power", baseline,
                        maes["power"], refs[baseline], 9, refs,
                        "; ".join(paths) + f" :: absolute_errors.{cap}[*]"))

    path = "results/v72-prune-repeat/compare.json"
    repeat = read(path)
    freeze_path = "results/v72-prune-repeat/freeze.json"
    freeze = read(freeze_path)
    require(repeat["provenance"]["freeze_sha256"] == INPUTS[freeze_path], "Repeat freeze hash")
    paired_rows(repeat, freeze, ("source", "density", "capability"), "observed_delta_loss")
    require(len(repeat["rows"]) == 18, "Expected complete 2.8B repeat")
    for cap in CAPS:
        rr = [r for r in repeat["rows"] if r["capability"] == cap]
        require(len(rr) == 6, f"Repeat count: {cap}")
        maes = {m: repeat["scores"][m]["by_capability"][cap]["mae"]
                for m in ("power", "median_curve")}
        for method, mae in maes.items():
            close(np.mean([r["absolute_errors"][method] for r in rr]), mae, f"Repeat/{cap}/{method}")
        bars.append(Bar("Pruning", "2.8B repeat", cap, "power", "median_curve",
                        maes["power"], maes["median_curve"], len(rr),
                        {"median_curve": maes["median_curve"]},
                        path + f" :: scores[method].by_capability.{cap}.mae"))
    return bars


def quantization_bars():
    path = "results/v69-quant-confirm/compare.json"
    comp = read(path)
    freeze_path = "results/v69-quant-confirm/freeze.json"
    freeze = read(freeze_path)
    require(comp["complete"] and len(comp["rows"]) == 63, "Incomplete quantization")
    require(comp["provenance"]["freeze_sha256"] == INPUTS[freeze_path], "Quantization freeze hash")
    paired_rows(comp, freeze, ("state", "config", "capability"), "dL")
    # Check each stored stratum before pooling the two new-state strata.
    for name, result in comp["test_sets"].items():
        for cap in CAPS:
            rr = [r for r in comp["rows"] if r["test_set"] == name and r["capability"] == cap]
            for method in ("same_input_interpolation", "low_order_2d", "median"):
                metric = result["scores"][method][cap]
                require(metric["n"] == len(rr), f"Quantization count: {name}/{cap}")
                close(np.mean([r["absolute_errors"][method] for r in rr]), metric["mae"],
                      f"Quantization/{name}/{cap}/{method}")
    bars = []
    for group, test_sets, n in (
        ("Development-state boundary", ("development_state_boundary",), 12),
        ("New state: 1.4B at 112k", ("new_state_boundary", "new_state_interior"), 9),
    ):
        for cap in CAPS:
            rr = [r for r in comp["rows"] if r["test_set"] in test_sets and r["capability"] == cap]
            require(len(rr) == n, f"Quantization count: {group}/{cap}")
            maes = {m: float(np.mean([r["absolute_errors"][m] for r in rr]))
                    for m in ("same_input_interpolation", "low_order_2d", "median")}
            if n == 12:
                candidate = "same_input_interpolation"
                refs = {m: maes[m] for m in ("low_order_2d", "median")}
                baseline = min(refs, key=refs.get)
            else:
                candidate, baseline = "median", "same_input_interpolation"
                refs = {baseline: maes[baseline]}
            bars.append(Bar("Grouped quantization", group, cap, candidate, baseline,
                            maes[candidate], maes[baseline], n, refs,
                            path + f" :: rows[capability={cap}, test_set in {test_sets}].absolute_errors"))
    return bars


def distillation_bars():
    path = "results/v70-distill-confirm/compare.json"
    comp = read(path)
    freeze_path = "results/v70-distill-confirm/freeze.json"
    freeze = read(freeze_path)
    require(comp["complete"] and len(comp["rows"]) == 108, "Incomplete distillation")
    require(comp["freeze_sha256"] == INPUTS[freeze_path], "Distillation freeze hash")
    paired_rows(comp, freeze, ("student", "pool", "T_planned", "capability"), "actual")
    require(comp["bootstrap"] == freeze["bootstrap"], "Changed bootstrap protocol")
    require(comp["bootstrap"]["n_resamples"] == 5000 and comp["bootstrap"]["seed"] == 0,
            "Unexpected bootstrap draws")
    # Reproduce only to check the stored CI. The plotted endpoints come from compare.json.
    draws = np.random.default_rng(0).integers(0, 6, size=(5000, 6))
    groups = {(g["student"], g["capability"]): g for g in comp["groups"]}
    require(len(groups) == len(comp["groups"]) == 6, "Expected six distillation groups")
    bars = []
    for student in ("gemma3-270m", "gemma3-1b"):
        for cap in CAPS:
            group = groups[student, cap]
            candidate, baseline = group["selected"], group["strongest_baseline"]
            require(candidate == freeze["selected"][cap]["method"], "Changed selected distillation form")
            require(baseline == freeze["strongest_baseline"][student][cap]["method"],
                    "Changed frozen strongest baseline")
            require(group["n_pools"] == 6 and group["n_checkpoints"] == 18, "Distillation group count")
            clusters = []
            for seed, cluster in zip(range(31, 37), group["clusters"]):
                pool = f"U200_s{seed}"
                require(cluster["pool"] == pool, "Changed pool ordering")
                rr = [r for r in comp["rows"] if r["student"] == student
                      and r["capability"] == cap and r["pool"] == pool]
                require(sorted(r["T_planned"] for r in rr) == [50000, 100000, 200000],
                        "Missing distillation budgets")
                values = [float(np.mean([r["absolute_errors"][m] for r in rr]))
                          for m in (candidate, baseline)]
                close(values, [cluster["mae"][m] for m in (candidate, baseline)], f"{student}/{cap}/{pool}")
                clusters.append(values)
            errors = np.asarray(clusters)
            close(errors.mean(0), [group["candidate_mae"], group["baseline_mae"]], f"{student}/{cap}")
            boot = errors[draws].mean(1)
            interval = group["paired_difference"]["ci95"]
            close(np.quantile(boot[:, 1] - boot[:, 0], [.025, .975]), interval,
                  f"Pool-cluster CI/{student}/{cap}")
            bar = Bar("Distillation", student.replace("gemma3-", "Gemma-3 ").replace("270m", "270M").replace("1b", "1B"),
                      cap, candidate, baseline, group["candidate_mae"], group["baseline_mae"], 18,
                      {baseline: group["baseline_mae"]},
                      path + f" :: groups[student={student}, capability={cap}]", tuple(interval))
            close(bar.gain, group["paired_difference"]["estimate"], f"Distillation gain/{student}/{cap}")
            require(interval[0] <= bar.gain <= interval[1], "CI does not bracket point estimate")
            bars.append(bar)
    return bars


def style():
    families = ["Liberation Serif", "Nimbus Roman", "Times New Roman"]
    font = font_manager.findfont(font_manager.FontProperties(family=families, weight="bold"))
    plt.rcParams.update({
        "font.family": "serif", "font.serif": families, "font.weight": "bold",
        "font.size": FONT, "axes.titlesize": 8.5, "axes.titleweight": "bold",
        "axes.labelsize": FONT, "xtick.labelsize": FONT, "ytick.labelsize": FONT,
        "legend.fontsize": FONT, "legend.title_fontsize": FONT,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.linewidth": .5, "xtick.major.size": 2, "xtick.major.width": .5,
        "xtick.major.pad": 2, "ytick.major.pad": 3, "axes.unicode_minus": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })
    return font


def draw_panel(fig, index, bars, title, headers, limits, ticks):
    left = .39 + index * 1.835
    ax = fig.add_axes([left / WIDTH, .43 / HEIGHT, 1.355 / WIDTH, 1.70 / HEIGHT])
    ax.set(xlim=limits, ylim=(7.05, -.95), xticks=ticks)
    positions = [.55, 1.55, 2.55, 4.55, 5.55, 6.55]
    ax.set_yticks(positions, [CAP_LABELS[b.cap] for b in bars])
    ax.tick_params(axis="y", length=0)
    fig.text((left + .50) / WIDTH, 2.29 / HEIGHT, title, fontsize=8.5, ha="center", va="center")
    for y, header in zip((-.72, 3.28), headers):
        ax.text(-.23, y, header, transform=ax.get_yaxis_transform(), va="center", fontsize=FONT)
    for low, high in ((.25, 2.85), (4.25, 6.85)):
        for tick in ticks:
            ax.vlines(tick, low, high, color=PALETTE["reference"] if tick == 0 else PALETTE["background"],
                      lw=.7 if tick == 0 else .45, zorder=0)
    annotations = []
    for y, bar in zip(positions, bars):
        color = POSITIVE if bar.gain >= 0 else NEGATIVE
        ax.barh(y, bar.gain, height=.27, color=color, edgecolor=color, linewidth=.4, zorder=2)
        if bar.ci95 is not None:
            lo, hi = bar.ci95
            ax.errorbar(bar.gain, y, xerr=[[bar.gain - lo], [hi - bar.gain]],
                        fmt="none", ecolor=PALETTE["reference"], elinewidth=.8,
                        capsize=2, capthick=.7, zorder=3)
        # The pair sits immediately above the endpoint, leaving the bar and CI unobscured.
        annotation = ax.annotate(f"{bar.candidate_mae:.3f} / {bar.baseline_mae:.3f}",
                                 (bar.gain, y), xytext=(0, 3.3), textcoords="offset points",
                                 ha="left" if bar.gain < 0 else "right", va="bottom", fontsize=FONT)
        annotations.append(annotation)
    # Prefer the inward side of the endpoint; switch sides when a short bar needs room.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for annotation in annotations:
        box = annotation.get_window_extent(renderer)
        if box.x0 < ax.bbox.x0 or box.x1 > ax.bbox.x1:
            annotation.set_ha("right" if annotation.get_ha() == "left" else "left")
    return ax, annotations


def validate_figure(fig, axes, annotations, panels):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    for text in texts:
        require(text.get_fontsize() >= 7, f"Font below 7 pt: {text.get_text()}")
        box = text.get_window_extent(renderer)
        require(box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1 + 1
                and box.y1 <= fig.bbox.y1 + 1, f"Text outside page: {text.get_text()}")
    for ax, labels, bars in zip(axes, annotations, panels):
        require(len(ax.patches) == len(labels) == len(bars) == 6, "Missing bars or MAE labels")
        for patch, label, bar in zip(ax.patches, labels, bars):
            close(patch.get_width(), bar.gain, "Signed bar width")
            box = label.get_window_extent(renderer)
            require(box.x0 >= ax.bbox.x0 - 1 and box.x1 <= ax.bbox.x1 + 1,
                    f"MAE label outside panel: {label.get_text()}")
            for value in (0, bar.gain, *(bar.ci95 or ())):
                require(ax.get_xlim()[0] <= value <= ax.get_xlim()[1], "Clipped bar/interval")
    # The visible text must not collide at the final physical figure size.
    boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in texts]
    for i, (name, box) in enumerate(boxes):
        for other, other_box in boxes[i + 1:]:
            require(not box.overlaps(other_box), f"Overlapping labels: {name!r}, {other!r}")
    close(fig.get_size_inches(), [WIDTH, HEIGHT], "Physical figure size")
    require(HEIGHT <= 2.4, "Figure exceeds requested height")
    return min(t.get_fontsize() for t in texts)


def write_notes(panels, font, minimum):
    lines = ["# V83 Figure 2: latest frozen confirmations", "",
        "Build: `python -B analysis/plot_fig2_confirm.py` (CPU; NumPy and matplotlib Agg; no fitting).",
        f"Outputs: `paper/paper/figs/final_confirmations.pdf` and `.png`, {WIDTH} × {HEIGHT} inches; "
        f"300 dpi PNG, 1650 × 720 pixels. Minimum visible font: {minimum:g} pt. "
        "PDF width equals the 5.5-inch text width in `paper/paper/iclr2027_conference.sty`. "
        "The main Figure 2 inclusion and caption in `paper/paper/general.tex` use "
        "`figs/final_confirmations.pdf` at `\\textwidth`.",
        f"Font: `{font}`; bold Times-like serif, embedded TrueType in the PDF.", "",
        "Each bar is **MAE(baseline) − MAE(candidate)** in native-token nats/token. "
        "Blue is positive (candidate better); orange is negative (baseline better). "
        "All 18 bars are retained, including failures. Numbers immediately above the bar endpoint "
        "are **candidate MAE / baseline MAE**, rounded to three decimals for display only. "
        "Each arm has its own linear x scale; zero is marked. Distillation whiskers use the stored "
        "95% paired pool-cluster interval on the difference, not intervals on either MAE.", "",
        "## Cohorts and aggregation", "",
        "- **Pruning, three-checkpoint panel:** exactly 410M@48k, 1.4B@112k, 6.9B@80k, "
        "densities 0.85, 0.675, 0.575. Pool the nine absolute errors separately for each method and "
        "capability, then compare power with the lower pooled MAE of A2 and median_curve. "
        "Do not choose a different baseline per checkpoint or density. Equal counts make this "
        "equivalent to the mean of the three state MAEs. A2 is the per-density source regression.",
        "- **Pruning, 2.8B repeat:** use the stored pooled `scores` for power and median_curve; "
        "six rows/capability at d=0.85,0.75,0.65 under the 16k and 143k revision labels. "
        "Retain the supplied scoring convention. These labels load identical learned weights "
        "and are one state measured twice, not two independent source states "
        "(documented in `paper/paper/appendix.tex`, weight-identity and pruning-repeat paragraphs). "
        "No pruning confidence interval is available in these compare files; none is fabricated.",
        "- **Grouped quantization, development-state boundary:** filter `rows` to "
        "`test_set=development_state_boundary`: two states (410M@143k, 1.4B@16k), "
        "b=3,4,5 and g=32,512; 12 equally weighted configurations/capability. Compare "
        "same_input_interpolation with the lower pooled MAE of low_order_2d and median. "
        "The surface is the selected baseline for math/code; the median for QA.",
        "- **Grouped quantization, new state:** pool both `new_state_boundary` and "
        "`new_state_interior`, all nine configurations of 1.4B@112k per capability "
        "(six boundary plus three interior cells). Candidate is the per-configuration median; "
        "baseline is same_input_interpolation. Weight individual configurations equally, "
        "so the six-cell boundary stratum has twice the weight of the three-cell interior stratum.",
        "- **Distillation:** use all six `groups`, two students (Gemma-3 270M, 1B) × "
        "math/code/QA. Candidate and strongest baseline are taken exactly from each group "
        "and checked against `freeze.json::selected` and `strongest_baseline`. Math/code use "
        "the zero-anchored reuse form (`E`); QA uses the joint form (`joint`). Candidate `E` "
        "and baseline `E-only` are distinct forms (the latter includes an intercept). "
        "No baseline is reselected using the confirmation scores.",
        "- **Distillation intervals:** stored `paired_difference.ci95`; 5,000 paired "
        "percentile bootstrap samples with NumPy PCG64 seed 0. Average the three planned "
        "budgets (50k,100k,200k tokens) within each U=200 pool, then give each of six pools "
        "equal weight. Keep all budgets together and use shared pool draws across students "
        "and capabilities. The plotted predictions remain the stored planned-budget predictions; "
        "actual budget overshoot does not cause prediction recomputation. Intervals describe "
        "variation over these sampled pools, not training-seed variability or 18 independent points.", "",
        "**Selection timing:** all plotted predictions were frozen before measurement. The "
        "requested better-of comparisons in the pruning panel and quantization boundary group "
        "choose the strongest *realized confirmation MAE* among the specified frozen alternatives. "
        "Likewise, the displayed quantization interpolation/median roles are the requested "
        "delivered comparisons, not the original development-selected `selected_candidate` "
        "(surface for math, median for code, zero for QA). This figure does not claim that these "
        "post-confirmation choices were preselected. Distillation selections, including the "
        "strongest baseline, were fixed using development LOCO.", "",
        "## Exact plotted numbers", "",
        "Values below use Python's round-trip float representation (no display rounding). "
        "n counts configuration/checkpoint rows per capability, not independent clusters.", "",
        "| Arm / confirmation group | Capability | Candidate | Baseline | n | Candidate MAE | Baseline MAE | Baseline − candidate | 95% pool-cluster interval |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for bars in panels:
        for bar in bars:
            ci = "—" if bar.ci95 is None else f"[{bar.ci95[0]!r}, {bar.ci95[1]!r}]"
            lines.append(f"| {bar.arm} / {bar.group} | {CAP_LABELS[bar.cap]} | "
                         f"{METHOD_LABELS[bar.candidate]} (`{bar.candidate}`) | "
                         f"{METHOD_LABELS[bar.baseline]} (`{bar.baseline}`) | {bar.n} | "
                         f"{bar.candidate_mae!r} | {bar.baseline_mae!r} | {bar.gain!r} | {ci} |")
    lines += ["", "## Comparator audit and exact source fields", ""]
    for bars in panels:
        for bar in bars:
            refs = "; ".join(f"`{name}`={value!r}" for name, value in bar.references.items())
            lines.append(f"- **{bar.arm} / {bar.group} / {CAP_LABELS[bar.cap]}:** "
                         f"eligible reference MAEs: {refs}. Source: `{bar.source}`.")
    lines += ["", "## Input SHA-256", "",
              "All prediction hashes and freeze links are checked; compare-row absolute errors, "
              "pooled MAEs and all six stored bootstrap intervals are independently reproduced. "
              "Source files are rehashed before outputs are written. The renderer checks signed "
              "bar lengths, complete bar/label counts, full interval extents, minimum fonts, "
              "page and panel bounds, and text collisions at the fixed physical page size. "
              "Only the two figure files and this new note are written; no experimental results "
              "are modified. No tight bounding-box crop is applied.", ""]
    lines.extend(f"- `{path}`: `{digest}`" for path, digest in INPUTS.items())
    out = ROOT / "results/v83-fig2/notes.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main():
    font = style()
    panels = [pruning_bars(), quantization_bars(), distillation_bars()]
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    specifications = [
        ("A  Pruning", ("Three checkpoints · power", "2.8B repeat · power"), (-.70, .06), [-.6, -.3, 0]),
        ("B  Grouped quantization", ("Boundary · interpolation", "New state · median"), (-.05, .51), [0, .2, .4]),
        ("C  Distillation", ("Gemma-3 270M", "Gemma-3 1B"), (-.04, .12), [0, .05, .10]),
    ]
    axes, annotations = [], []
    for i, (bars, spec) in enumerate(zip(panels, specifications)):
        ax, labels = draw_panel(fig, i, bars, *spec)
        axes.append(ax)
        annotations.append(labels)
    fig.text(.5, .19 / HEIGHT,
             "MAE(baseline) − MAE(candidate), nats/token; positive = candidate better",
             ha="center", va="center", fontsize=FONT)
    fig.text(.5, .065 / HEIGHT,
             "Bar-end MAEs: candidate / baseline; whiskers: 95% pool-cluster interval",
             ha="center", va="center", fontsize=FONT)
    minimum = validate_figure(fig, axes, annotations, panels)
    for path, digest in INPUTS.items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest,
                f"Input changed during build: {path}")
    for suffix in ("pdf", "png"):
        out = ROOT / f"paper/paper/figs/final_confirmations.{suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {}
        fig.savefig(out, dpi=300, metadata=metadata)
        print(f"WROTE {out}")
    plt.close(fig)
    write_notes(panels, font, minimum)
    print(f"VERIFIED: 18 signed bars, 36 MAEs, six stored 95% pool-cluster intervals; "
          f"{WIDTH} × {HEIGHT} in, minimum font {minimum:g} pt; inputs unchanged.")


if __name__ == "__main__":
    main()
