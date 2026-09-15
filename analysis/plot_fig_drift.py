#!/usr/bin/env python3
"""Observed logarithmic-reuse ratio drift; never rebuild missing quantities."""
from __future__ import annotations

import math
from statistics import median

if __package__:
    from .paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot, output_path, write_notes
    from .paper_figure_style import panel_axes, write_caption
    from .paper_panel_exports import ref, capability_handles, axes_defaults, export
else:
    from paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot, output_path, write_notes
    from paper_figure_style import panel_axes, write_caption
    from paper_panel_exports import ref, capability_handles, axes_defaults, export

V96 = "results/v96-joint-response/summary.json"
V100 = "results/v100-critical-region/summary.json"
STUDENTS = ("gemma3-1b", "gemma3-4b")
TOLERANCES = (.01, .003)
PANEL_SIZE = (2.7, 1.8)
CAPTION = """Observed-endpoint falsification diagnostic for logarithmic reuse,
for Gemma-3 1B (a) and 4B (b). The ordinate is the saved measured pool-change
effect divided by log(1+E2)-log(1+E1), where E=T/D_U; the abscissa is the saved
common_T=(T_first+T_second)/2, in thousands of supervised tokens. Colour denotes
Math, Code or QA. Thin lines (0.8 pt, alpha 0.35, without markers) connect the
saved ratios for each individual pool/seed trajectory dyad across budgets,
separately within each capability and tolerance. Heavy lines (2.5 pt, alpha 1)
connect the per-budget-band median of those individual plotted ratios across
pool pairs, separately for each capability and tolerance. The bands are the
saved V100 nominal 25k, 50k, 100k and 200k bands; each median is positioned at
the median common_T of its contributing records. Heavy solid lines with filled
circles denote 1% budget matching; heavy dashed lines with filled triangles
denote 0.3%. All markers are 9 pt, filled in the capability colour, with a white
edge. The tighter set is nested in the 1% set, so some points coincide. Only
occupied bands are shown; connecting segments are visual guides, with no
interpolated observations, fitted slopes or new intervals. The set of pool pairs
can change across bands, so the median lines are descriptive summaries.
At exactly matched T, a separable logarithmic reuse form predicts a horizontal
ratio for each student/capability. The zero line is a reference, not a fitted
coefficient. V100 reports zero exactly matched nonzero-budget pairs: the observed
drift also admits residual budget, composition and schedule confounding and does
not establish an exact fixed-budget causal falsification. At 0.3%, critical
U=132 matches occur only near 200k and cannot identify critical-pool drift.
V100 fixed_budget.endpoint_ratios supplies all individual plotted numbers and
budget-band labels; the heavy lines summarize these records by medians. Its saved
raw_response_curves verifies endpoint identity and E without reading trajectories.
V96 part_d supplies the original definition and matching endpoint cross-checks;
its broader ordinal matches are not added to the two requested tolerance sets.
"""


def student_rows(old, summary, student):
    """Fail the whole student panel if a required frozen field is absent."""
    curves = {(c["capability"], p["source"]): (i, j, p)
              for i, c in enumerate(summary["raw_response_curves"]) if c["student"] == student
              for j, p in enumerate(c["points"])}
    previous = {r["pair_id"]: (i, r) for i, r in enumerate(old["part_d"]["endpoint_ratios"])
                if r["kind"] == "size" and r["student"] == student}
    result = []
    for i, raw in enumerate(summary["fixed_budget"]["endpoint_ratios"]):
        if raw["student"] != student or raw["tolerance"] not in TOLERANCES:
            continue
        r = {k: raw[k] for k in ("student", "capability", "tolerance", "pair_id", "contrast", "band",
                                 "trajectory_clusters", "raw_ratio", "measured_effect", "reuse_bracket",
                                 "common_T", "T_first", "T_second", "first_source", "second_source")}
        if not all(math.isfinite(r[k]) for k in ("raw_ratio", "measured_effect", "reuse_bracket", "common_T")):
            raise ValueError("Nonfinite frozen ratio fields")
        if not r["reuse_bracket"] or not math.isclose(r["raw_ratio"], r["measured_effect"]/r["reuse_bracket"], abs_tol=1e-12):
            raise ValueError("Saved raw_ratio disagrees with saved effect/bracket")
        if r["common_T"] != (r["T_first"] + r["T_second"])/2:
            raise ValueError("Saved common_T disagrees with endpoint budgets")
        mismatch = abs(r["T_second"]-r["T_first"])/min(r["T_first"], r["T_second"])
        if mismatch > r["tolerance"]:
            raise ValueError("Endpoint exceeds its recorded matching tolerance")
        first, second = [curves[r["capability"], r[f"{end}_source"]] for end in ("first", "second")]
        if first[2]["T"] != r["T_first"] or second[2]["T"] != r["T_second"]:
            raise ValueError("Frozen response-curve budget mismatch")
        bracket = math.log1p(second[2]["E"]) - math.log1p(first[2]["E"])
        if not math.isclose(bracket, r["reuse_bracket"], abs_tol=1e-12):
            raise ValueError("Saved reuse bracket disagrees with frozen response curves")
        if not math.isclose(second[2]["delta"]-first[2]["delta"], r["measured_effect"], abs_tol=1e-12):
            raise ValueError("Saved effect disagrees with frozen response curves")
        base = ("fixed_budget", "endpoint_ratios", i)
        r.update(x=r["common_T"]/1000, delta=r["raw_ratio"],
                 x_source=ref(V100, *base, "common_T"), delta_source=ref(V100, *base, "raw_ratio"),
                 effect_source=ref(V100, *base, "measured_effect"),
                 bracket_source=ref(V100, *base, "reuse_bracket"),
                 tolerance_source=ref(V100, *base, "tolerance"),
                 band_source=ref(V100, *base, "band"),
                 endpoint_sources=[ref(V100, "raw_response_curves", ci, "points", pi) for ci, pi, _ in (first, second)],
                 v96_source=None)
        if r["pair_id"] in previous:
            oi, prior = previous[r["pair_id"]]
            if not math.isclose(prior["raw_ratio"], r["raw_ratio"], abs_tol=1e-12):
                raise ValueError("V96/V100 shared endpoint ratio mismatch")
            r["v96_source"] = ref(V96, "part_d", "endpoint_ratios", oi, "raw_ratio")
        result.append(r)
    if {(r["capability"], r["tolerance"]) for r in result} != {(c, t) for c in CAPS for t in TOLERANCES}:
        raise ValueError("Missing capability/tolerance coverage")
    return result


def build(audit):
    old, summary = audit.read(V96), audit.read(V100)
    # Required contextual fields, also recorded in each generated caption.
    audit.rule("V96 definition: " + old["part_d"]["ratio_definition"])
    audit.rule("V100 exact_nonzero_budget_pairs=" + str(summary["fixed_budget"]["exact_nonzero_budget_pairs"]))
    rows, unavailable = [], {}
    for letter, student in zip("ab", STUDENTS):
        try:
            part = student_rows(old, summary, student)
        except (KeyError, ValueError, TypeError) as exc:
            unavailable[letter] = f"{student}: missing or inconsistent frozen quantity: {exc}. Panel stopped; no trajectory fallback."
        else:
            rows.extend(dict(r, panel=letter) for r in part)
    return rows, unavailable


def draw_panel(fig, rows, panel):
    ax = panel_axes(fig, PANEL_SIZE, left=.53, bottom=.50, right=.08)
    for cap in CAPS:
        for tolerance, marker in zip(TOLERANCES, ("o", "^")):
            part = [r for r in rows if r["panel"] == panel and r["capability"] == cap and r["tolerance"] == tolerance]
            for dyad in sorted({tuple(r["trajectory_clusters"]) for r in part}):
                line = sorted((r for r in part if tuple(r["trajectory_clusters"]) == dyad), key=lambda r: r["x"])
                ax.plot([r["x"] for r in line], [r["delta"] for r in line], color=COLORS[cap],
                        lw=.8, alpha=.35, marker="", ls="-", zorder=1)
            # Use saved budget bands: actual endpoint midpoints differ by dyad.
            # Take the median of ratios, never a ratio of aggregated effects.
            points = sorted((median(r["x"] for r in part if r["band"] == band),
                             median(r["delta"] for r in part if r["band"] == band))
                            for band in {r["band"] for r in part})
            if points:
                ax.plot([x for x, _ in points], [y for _, y in points], color=COLORS[cap],
                        lw=2.5, alpha=1, marker=marker, ms=9, mfc=COLORS[cap],
                        mec="white", mew=1.2, ls="-" if tolerance == .01 else "--", zorder=3)
    axes_defaults(ax)
    ax.set(xlabel="Budget $T$ (k)", ylabel="Ratio (nats)", xlim=(0, 220), ylim=(-1, 8),
           xticks=[0, 100, 200], yticks=[0, 3, 6])
    # Six capability/tolerance entries crowd a 2.7-inch panel at shared font size.
    # Keep capability colours inside; the caption gives the tolerance marker key.
    handles = capability_handles()
    for handle in handles:
        handle.set_linewidth(2.5)
    ax.legend(handles=handles, loc="upper left", handlelength=.8, handletextpad=.3)
    return ax


def generate(root=ROOT):
    with frozen_run(root) as access:
        audit, plt = Artifacts(root), pyplot(root)
        rows, unavailable = build(audit)
        audit.rule("Figure 5 artist overrides to shared defaults: individual dyads 0.8 pt, "
                   "alpha 0.35, no markers; per-band medians 2.5 pt, alpha 1, with 9 pt "
                   "capability-filled markers and 1.2 pt white edges. Saved budget bands "
                   "group individual raw ratios; both x and y are medians within each "
                   "student/capability/tolerance/band. Circle/solid = 1%; triangle/dashed = 0.3%.")
        for panel, reason in unavailable.items():
            audit.omit(reason)
            print(f"STOPPED fig5_{panel}: {reason}")
            for suffix in (".pdf", "_data.json"):
                path = output_path(root, "figs", f"fig5_{panel}{suffix}")
                if path.exists():
                    path.unlink()  # Never leave stale numbers looking freshly validated.
            path.parent.mkdir(parents=True, exist_ok=True)
            write_caption(f"fig5_{panel}", audit, reason)
            write_notes(f"fig5_{panel}", audit, [])
        panels = [(f"fig5_{p}", PANEL_SIZE, lambda f, p=p: draw_panel(f, rows, p),
                   [r for r in rows if r["panel"] == p], f"Gemma-3 {s.rsplit('-', 1)[1].upper()}.\n" + CAPTION)
                  for p, s in zip("ab", STUDENTS) if p not in unavailable]
        if panels:
            export(plt, audit, "drift", panels, CAPTION + "\n" + "\n".join(unavailable.values()))
            print("Figure 5 artist overrides: thin lines 0.8 pt / alpha 0.35; "
                  "median lines 2.5 pt / alpha 1; filled markers 9 pt / white edges 1.2 pt.")
        else:
            for name in ("drift.png", "drift_files.json"):
                path = output_path(root, "figs", name)
                if path.exists():
                    path.unlink()
            write_caption("drift", audit, "\n".join(unavailable.values()))
            write_notes("drift", audit, [])
        return rows, audit, access


if __name__ == "__main__":
    generate()
