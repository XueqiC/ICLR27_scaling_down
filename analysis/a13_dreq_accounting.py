#!/usr/bin/env python3
"""A13: CPU-only accounting and figures from the sealed A12 plan and score.

Run: python -B analysis/a13_dreq_accounting.py
No fitting, interpolation, model imports, training, or manuscript edits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import statistics

try:
    from .paper_artifacts import Artifacts, output_path, pyplot
    from . import paper_figure_style as style
except ImportError:
    from paper_artifacts import Artifacts, output_path, pyplot
    import paper_figure_style as style

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path("results/a12-data-requirement-confirmation")
OUTPUT = Path("results/a13-dreq-accounting")
METHODS = ("boundary_loglinear", "fixed_reuse", "student_isotonic")
KINDS = ("boundary", "delta")
KEY = ("student", "nominal_supervised_tokens", "readout", "tau")
STUDENTS = {"gemma3-1b": "1B", "gemma3-4b": "4B"}
BUDGETS = (50000, 100000, 200000)
QA = ("qa:2Wiki-fresh", "qa:MuSiQue", "qa:TriviaQA")
READOUT_NAMES = {"math:MATH-500": "Mathematics (MATH-500)",
                 "code:MBPP": "Code (MBPP)", "qa:2Wiki-probe": "2Wiki (original probe)",
                 "qa:2Wiki-fresh": "2Wiki (fresh sample)",
                 "qa:MuSiQue": "MuSiQue", "qa:TriviaQA": "TriviaQA"}
GROUPS = {"QA distributions": QA,
          "math/code": ("math:MATH-500", "code:MBPP"),
          "2Wiki original probe": ("qa:2Wiki-probe",),
          "QA including original probe": (*QA, "qa:2Wiki-probe"),
          "all requests": tuple(READOUT_NAMES)}
DEFINITIONS = {
    "request": "student x nominal supervised training budget x readout x tau; 144 requests",
    "recommendation_rules": "Boundary recommendations are primary; delta (loss-based) recommendations are a separate companion analysis. Never pool the two rules into a 288-request denominator.",
    "QA distributions": "Fresh 2Wiki, MuSiQue, TriviaQA (72 requests); original 2Wiki probe separately (24). QA including original probe (96) is also reported.",
    "loss_units": "A12 delta_mae and common_support_delta_mae are loss-prediction errors in nats per supervised evaluation token, not errors in required data.",
    "data_units": "D_U is the independent supervised-token pool size, not processed tokens, examples, or D_U_seen. Training budget T counts supervised tokens, including reuse.",
    "log_distance": "Natural-log distance to the recorded measured interval: max(0, ln(L/R), ln(R/U)), omitting unbounded sides; R is recommended_D_U. Zero means compatibility with the interval closure, not an exact minimum or a successful recommendation.",
    "censoring": "For lower/upper bounds, distance is only a lower bound on distance to an unknown crossing; even zero does not identify the requirement. One-sided log widths are unbounded (JSON null plus width_status).",
    "interval_membership": "Uses the closed recorded envelope, including endpoints. A failing lower endpoint is still geometrically in its closure. Empirical constraint success is reported independently.",
    "unresolved": "Replicate-ambiguous and non-monotone cells retain null intervals/distances; they stay in all coverage and solved denominators.",
    "interval_width": "For finite crossed intervals: U-L supervised tokens and ln(U/L). No midpoint or interpolated minimum is estimated.",
    "coverage": "Recommendations / all requests in group, for each method and recommendation rule.",
    "conditional": "Recommendations meeting the constraint in all three measured replicates / recommendations. A12 recommendation_meets_loss_constraint is used directly.",
    "common_reliability": "Successful recommendations / the identical intersection of requests where all three methods recommend; the intersection is independent of success and crossing status.",
    "solved": "Successful recommendations / all requests in group; abstentions and unresolved measurements do not leave the denominator.",
    "conservative": "conservative_D_U is a comparator recommendation, not the measured crossing. Its recorded unverified-ceiling and success flags are preserved; savings are credited only when both recommendations pass.",
    "primary_tau": "tau = 0.25 nats is the level A10 reported and the plan's primary level; all three registered budgets are plotted, with 200k the plan's primary budget.",
    "seed_design": "The upper tier uses training seeds on the same full pool; the lower tiers use pool seeds. These are not exchangeable pool replicates.",
    "figure_intervals": "For scored crossings only, bars run from the largest failing measured pool to the smallest passing measured pool. A12's slightly wider seed-envelope interval (smallest failing-tier pool to largest passing-tier pool) remains the accounting target and is retained alongside the plotted endpoints.",
    "figure_unresolved": "Crosses show seed-mixed tested tiers, or all tested tiers when means are non-monotone. Their y positions are measured pool sizes, not boundary estimates. A full-pool failure in a non-monotone cell is not promoted to a lower bound.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def request_key(row):
    return tuple(row[k] for k in KEY)


def request_id(row):
    return "|".join(map(str, request_key(row)))


def interval_metrics(value, crossing):
    """Distance to the observed set, never distance to an interpolated optimum."""
    status, interval = crossing["status"], crossing["interval"]
    result = dict(log_distance=None, inside_interval=None, signed_log_distance=None,
                  distance_status="unresolved_interval", interval_width_tokens=None,
                  interval_log_width=None, width_status="unresolved",
                  measured_lower_bound_D_U=None, measured_upper_bound_D_U=None)
    if interval is None:
        return result
    lo, hi = interval
    require(lo is not None and lo >= 0 and (hi is None or hi > lo), "Invalid interval")
    require(status in ("crossed", "lower_bound", "upper_bound"), "Unknown interval status")
    if status == "crossed":
        require(lo > 0 and hi is not None, "Crossing needs finite positive endpoints")
        result.update(interval_width_tokens=hi-lo, interval_log_width=math.log(hi/lo),
                      width_status="finite")
    else:
        result["width_status"] = "unbounded_log_interval"
        if status == "lower_bound":
            require(lo > 0 and hi is None, "Lower-bound endpoint mismatch")
            result["measured_lower_bound_D_U"] = lo
        else:
            require(lo == 0 and hi is not None, "Upper-bound endpoint mismatch")
            result["measured_upper_bound_D_U"] = hi
            result["interval_width_tokens"] = hi
    if value is None:
        result["distance_status"] = "no_recommendation"
        return result
    require(math.isfinite(value) and value > 0, "Recommendation must be positive")
    signed = math.log(value/lo) if value < lo else math.log(value/hi) if hi is not None and value > hi else 0.
    result.update(log_distance=abs(signed), signed_log_distance=signed,
                  inside_interval=(value >= lo and (hi is None or value <= hi)),
                  distance_status="interval_distance" if status == "crossed" else "censored_distance_lower_bound")
    return result


def load_inputs(root=ROOT):
    audit = Artifacts(root)
    score = audit.read(INPUT / "score.json")
    plan = audit.read(INPUT / "plan.json")
    require(score["status"] == "complete" and not score["missing"], "A12 must be complete")
    require(plan["launch_ready"] and plan["n_realized"] == 18, "Incomplete A12 plan")
    require(score["measured_readout_cells"] == score["expected_readout_cells"] == 432,
            "Expected all 432 A12 readout cells")
    for field in ("predictions", "decisions"):
        require(digest(plan[field]) == plan[field+"_sha256"], f"Frozen {field} hash mismatch")
    require(score["predictions_sha256"] == plan["predictions_sha256"], "Score/plan mismatch")
    require(plan["primary_tau"] == .25, "Primary tolerance changed")
    require(set(plan["readouts"]) == set(READOUT_NAMES), "Readouts changed")
    require(set(plan["taus"]) == {0., .1, .25, .5}, "Tolerances changed")
    require(set(plan["methods"]) == set(METHODS), "Methods changed")
    expected = set(product(STUDENTS, BUDGETS, plan["readouts"], plan["taus"], KINDS, METHODS))
    frozen = {(*request_key(r), r["kind"], r["method"]): r for r in plan["decisions"]}
    keys = [(*request_key(r), r["kind"], r["method"]) for r in score["decisions"]]
    require(len(keys) == len(set(keys)) and set(keys) == expected == set(frozen),
            "Requests missing, duplicated, or unequal across methods")
    for row, key in zip(score["decisions"], keys):
        require(all(row[k] == v for k, v in frozen[key].items()), "Frozen decision changed")
        require((row["recommended_D_U"] is None) == (row["recommendation_meets_loss_constraint"] is None),
                "A recommendation has no measured success verdict")
    return score, plan, audit


def build_accounting(score, plan):
    rows, grouped = [], defaultdict(list)
    for decision in score["decisions"]:
        row = {k: decision[k] for k in (*KEY, "kind", "method", "recommended_D_U",
               "recommendation_position", "recommended_trajectory_ids", "conservative_trajectory_ids",
               "recommendation_meets_loss_constraint",
               "recommended_D_U_seen", "conservative_D_U", "conservative_D_U_seen",
               "conservative_uses_unverified_ceiling", "conservative_meets_loss_constraint",
               "credited_savings_fraction", "actual_budget_relative_span", "measured_crossing")}
        row.update(request_id=request_id(row),
                   **interval_metrics(row["recommended_D_U"], row["measured_crossing"]))
        row["conservative_interval_metrics"] = interval_metrics(row["conservative_D_U"], row["measured_crossing"])
        row["log_recommendation_over_conservative"] = (math.log(row["recommended_D_U"]/row["conservative_D_U"])
                                                        if row["recommended_D_U"] is not None else None)
        rows.append(row)
        grouped[request_key(row)].append(row)
    requests = []
    for key, group in sorted(grouped.items()):
        require(all(r["measured_crossing"] == group[0]["measured_crossing"] for r in group),
                "Measured crossing differs across methods or rules")
        requests.append({**dict(zip(KEY, key)), "request_id": request_id(group[0]),
                         "measured_crossing": group[0]["measured_crossing"],
                         **{k: group[0][k] for k in ("interval_width_tokens", "interval_log_width", "width_status",
                              "measured_lower_bound_D_U", "measured_upper_bound_D_U")},
                         "recommendations": {kind: {r["method"]: r for r in group if r["kind"] == kind}
                                             for kind in KINDS}})
    return rows, requests


def fraction(n, d):
    return {"numerator": n, "denominator": d, "fraction": n/d if d else None}


def median_record(values):
    values = [v for v in values if v is not None]
    return {"n": len(values), "median": statistics.median(values) if values else None}


def summarize(rows):
    comparisons, medians = [], []
    for kind, (group, readouts) in product(KINDS, GROUPS.items()):
        chosen = [r for r in rows if r["kind"] == kind and r["readout"] in readouts]
        methods = {m: [r for r in chosen if r["method"] == m] for m in METHODS}
        common = set.intersection(*[{r["request_id"] for r in rr if r["recommended_D_U"] is not None}
                                    for rr in methods.values()])
        for method, rr in methods.items():
            rec = [r for r in rr if r["recommended_D_U"] is not None]
            passed = sum(r["recommendation_meets_loss_constraint"] is True for r in rec)
            paired = [r for r in rr if r["request_id"] in common]
            comparisons.append(dict(kind=kind, group=group, method=method,
                coverage=fraction(len(rec), len(rr)), conditional=fraction(passed, len(rec)),
                common_reliability=fraction(sum(r["recommendation_meets_loss_constraint"] is True for r in paired), len(paired)),
                solved=fraction(passed, len(rr)), common_request_ids=sorted(common)))
            medians.append(dict(kind=kind, group=group, method=method,
                crossed=median_record(r["log_distance"] for r in rec if r["measured_crossing"]["status"] == "crossed"),
                common_crossed=median_record(r["log_distance"] for r in paired if r["measured_crossing"]["status"] == "crossed"),
                censored_lower_bound=median_record(r["log_distance"] for r in rec if r["distance_status"] == "censored_distance_lower_bound"),
                all_identified=median_record(r["log_distance"] for r in rec),
                interval_log_width=median_record(r["interval_log_width"] for r in rr),
                interval_width_tokens=median_record(r["interval_width_tokens"] for r in rr if r["width_status"] == "finite"),
                inside_interval=sum(r["inside_interval"] is True for r in rec),
                unresolved_recommendations=sum(r["measured_crossing"]["interval"] is None for r in rec)))
    return comparisons, medians


def measured_tiers(score, plan, request):
    trajectories = {t["trajectory_id"]: t for t in plan["trajectories"]}
    by_id = {}
    for error in score["checkpoint_errors"]:
        if request_key(error) != request_key(request):
            continue
        tid, delta = error["trajectory_id"], error["measured_delta"]
        require(tid not in by_id or by_id[tid] == delta, "Method-dependent measurement")
        by_id[tid] = delta
    require(len(by_id) == 9, "Expected nine measured trajectories per request")
    tiers = []
    for position in ("below", "near", "above"):
        cells = [{"trajectory_id": tid, "D_U": trajectories[tid]["pool"]["D_U_pool"],
                  "delta_nats": delta, "passes": delta <= request["tau"],
                  "data_seed": trajectories[tid]["data_seed"],
                  "training_seed": trajectories[tid].get("training_seed", plan["protocol"]["seed"])}
                 for tid, delta in sorted(by_id.items()) if trajectories[tid]["position"] == position]
        require(len(cells) == 3, "Expected three replicates per tier")
        passes = sum(c["passes"] for c in cells)
        tiers.append(dict(position=position, replicates=cells,
                          verdict="pass" if passes == 3 else "fail" if passes == 0 else "mixed",
                          D_U_min=min(c["D_U"] for c in cells), D_U_max=max(c["D_U"] for c in cells)))
    return tiers


def figure_records(score, plan, requests):
    records = []
    for request in requests:
        if request["readout"] not in QA or request["tau"] != plan["primary_tau"]:
            continue
        tiers = measured_tiers(score, plan, request)
        status = request["measured_crossing"]["status"]
        raw = [c for tier in tiers for c in tier["replicates"]]
        interval = None
        if status == "crossed":
            interval = [max(c["D_U"] for c in raw if not c["passes"]),
                        min(c["D_U"] for c in raw if c["passes"])]
            require(interval[0] < interval[1], "Empirical endpoints overlap")
        bound = (request["measured_lower_bound_D_U"] if status == "lower_bound" else
                 request["measured_upper_bound_D_U"] if status == "upper_bound" else None)
        unresolved = [t for t in tiers if status == "non_monotone" or
                      (status == "replicate_ambiguous" and t["verdict"] == "mixed")]
        records.append({**{k: request[k] for k in (*KEY, "request_id", "measured_crossing")},
                        "tiers": tiers, "plotted_interval": interval, "plotted_bound": bound,
                        "unresolved_tier_ranges": [[t["D_U_min"], t["D_U_max"]] for t in unresolved],
                        "student_label": STUDENTS[request["student"]],
                        "x_category": [request["nominal_supervised_tokens"], STUDENTS[request["student"]]],
                        "x_coordinate": BUDGETS.index(request["nominal_supervised_tokens"])*3 + list(STUDENTS).index(request["student"]),
                        "linestyle": style.STUDENT_STYLES[STUDENTS[request["student"]]]})
    require(len(records) == 18, "Expected 18 main-figure requests")
    return records


FIGURE_CAPTION = (
    "Measured independent-data requirements at a loss-increase tolerance of 0.25 nats: "
    "(a) fresh 2Wiki, (b) MuSiQue, (c) TriviaQA. Within each training-budget group, "
    "1B is on the left (dashed) and 4B on the right (solid). Vertical bars span the "
    "largest failing pool to the smallest passing pool for an identified crossing; "
    "they are observed brackets, not confidence intervals or exact minima. Upward "
    "triangles mark lower bounds when all tested pools fail, and downward triangles "
    "mark upper bounds when all tested pools pass. Crosses mark tested tiers with "
    "seed disagreement, or all tiers when mean loss is non-monotone; these requests "
    "have no identified crossing. The vertical axis is logarithmic and labels "
    "independent supervised tokens in thousands; training budgets also count "
    "supervised tokens. The upper tier uses training seeds on one full pool, while "
    "the lower tiers use pool seeds. tau = 0.25 is the level A10 reported and the "
    "plan's primary level. No between-pool minimum is interpolated."
)


def draw_panel(fig, records, color):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax = style.panel_axes(fig, (1.8, 1.35), left=.44, bottom=.31, right=.04, top=.05)
    ax.set(yscale="log", xlim=(-.6, 7.6), ylim=(6500, 220000),
           xlabel="Training tokens, T (k)", ylabel=r"$D_U$ (k tokens)")
    ax.set_xticks([.5, 3.5, 6.5], ["50", "100", "200"])
    ax.yaxis.set_major_locator(FixedLocator([10000, 40000, 160000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y/1000:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", alpha=.25)
    for r in records:
        x, ls = r["x_coordinate"], r["linestyle"]
        interval, bound = r["plotted_interval"], r["plotted_bound"]
        if interval:
            lo, hi = interval
            bars = ax.errorbar([x], [lo], yerr=[[0], [hi-lo]], fmt="none", ecolor=color,
                               elinewidth=1.1, capsize=2, capthick=.6)
            for segment in bars.lines[2]:
                segment.set_linestyle(ls)
        elif bound is not None:
            marker = "^" if r["measured_crossing"]["status"] == "lower_bound" else "v"
            # The stem conveys student identity without connecting boundary estimates.
            ends = [bound/.65, bound] if marker == "v" else [bound*.65, bound]
            ax.plot([x, x], ends, color=color, linestyle=ls)
            ax.plot([x], [bound], color=color, linestyle="none", marker=marker)
        else:
            for lo, hi in r["unresolved_tier_ranges"]:
                ax.plot([x, x], [lo*.78, hi*1.2], color=color, linestyle=ls)
                ax.plot([x], [hi], color=color, linestyle="none", marker="x")
    style.finish_panel(ax)
    return ax


def generate_figures(score, plan, requests, audit):
    plt = pyplot(audit.root)
    from matplotlib.lines import Line2D
    records = figure_records(score, plan, requests)
    for message in DEFINITIONS.values():
        audit.rule(message)
    files = []
    draws = []
    for letter, readout, color_key in zip("abc", QA, ("2wiki_new", "musique", "triviaqa")):
        subset = [r for r in records if r["readout"] == readout]
        color = style.QA_COLORS[color_key]
        draw = lambda fig, rr=subset, cc=color: draw_panel(fig, rr, cc)
        draws.append(draw)
        style.apply_style("panel")
        fig = plt.figure(figsize=(1.8, 1.35))
        draw(fig)
        stem = f"dreq_{letter}"
        style.save_panel(fig, stem, "panel", audit, dict(readout=readout, primary_tau=.25,
                         primary_tau_reason=DEFINITIONS["primary_tau"], records=subset,
                         color=color, x_categories="Budget x student; no numeric jitter"))
        style.write_caption(stem, audit, READOUT_NAMES[readout] + ". " + FIGURE_CAPTION)
        plt.close(fig)
        files.extend(output_path(audit.root, "figs", stem+s) for s in
                     (".pdf", ".png", "_data.json", "_sources.md", "_caption.txt"))
    style.apply_style("legend")
    neutral = style.PALETTE["reference"]
    handles = [Line2D([], [], color=neutral, linestyle=style.STUDENT_STYLES[s], label=s) for s in ("1B", "4B")]
    handles.extend(Line2D([], [], color=neutral, linestyle="none", marker=m, label=l)
                   for m, l in (("^", "Lower bound"), ("v", "Upper bound"), ("x", "Unresolved")))
    fig = plt.figure(figsize=(5.5, .3))
    style.legend_strip(fig, handles)
    style.save_panel(fig, "dreq_legend", "legend", audit, {"primary_tau": .25, "definitions": DEFINITIONS})
    style.write_caption("dreq_legend", audit, FIGURE_CAPTION)
    plt.close(fig)
    files.extend(output_path(audit.root, "figs", "dreq_legend"+s) for s in
                 (".pdf", ".png", "_data.json", "_sources.md", "_caption.txt"))
    caption = output_path(audit.root, "figs", "dreq_caption.txt")
    caption.write_text(FIGURE_CAPTION + "\n")
    files.append(caption)
    # A complete, final-size sheet is a review artifact, outside the panel directory.
    panels = [("panel", (.05+i*1.85, 0., 1.8, 1.35), draw) for i, draw in enumerate(draws)]
    panels.append(("legend", (.05, 1.39, 5.5, .3), lambda fig: style.legend_strip(fig, handles)))
    combined = style.combine_panels(plt, panels, (5.6, 1.75))
    from matplotlib.backends.backend_pdf import PdfPages
    review = audit.root / OUTPUT / "dreq_review.pdf"
    with PdfPages(review) as pdf:
        pdf.savefig(combined)
    combined.savefig(review.with_suffix(".png"), dpi=220, bbox_inches=None)
    plt.close(combined)
    files.extend((review, review.with_suffix(".png")))
    return records, files


def format_fraction(value):
    n, d, f = value["numerator"], value["denominator"], value["fraction"]
    return f"{n}/{d} ({100*f:.1f}%)" if d else f"{n}/{d} (undefined)"


def comparison_text(comparisons, groups=None):
    lines = []
    for kind in KINDS:
        lines += [f"Recommendation rule: {kind}", "", "| Readout group | Metric | boundary_loglinear | fixed_reuse | student_isotonic |",
                  "|---|---|---:|---:|---:|"]
        for group in (groups or GROUPS):
            rr = {r["method"]: r for r in comparisons if r["kind"] == kind and r["group"] == group}
            for metric in ("coverage", "conditional", "common_reliability", "solved"):
                lines.append("| " + " | ".join([group, metric] + [format_fraction(rr[m][metric]) for m in METHODS]) + " |")
        lines.append("")
    return "\n".join(lines)


def median_text(medians, groups=None):
    lines = ["Median natural-log recommendation distance (sample counts in parentheses).",
             "Crossed = distance to a finite bracket; censored = lower bound on error.", "",
             "| Rule | Readout group | Method | Crossed | Common crossed | Censored lower bound | All identified | Finite log width | Finite width (tokens) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in medians:
        if groups and r["group"] not in groups:
            continue
        vals = [f"{r[k]['median']:.4f} ({r[k]['n']})" if r[k]["n"] else "undefined (0)"
                for k in ("crossed", "common_crossed", "censored_lower_bound", "all_identified", "interval_log_width", "interval_width_tokens")]
        lines.append("| " + " | ".join([r["kind"], r["group"], r["method"], *vals]) + " |")
    return "\n".join(lines)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def generate(root=ROOT, *, figures=True, table=True):
    root = Path(root).resolve()
    score, plan, audit = load_inputs(root)
    rows, requests = build_accounting(score, plan)
    comparisons, medians = summarize(rows)
    out = root / OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    result = dict(stage="A13", input_sha256=audit.inputs, predictions_sha256=score["predictions_sha256"],
                  definitions=DEFINITIONS, primary_tau=plan["primary_tau"], primary_budget=plan["primary_budget"],
                  n_requests=len(requests), n_method_rule_records=len(rows),
                  crossing_status_counts=dict(Counter(r["measured_crossing"]["status"] for r in requests)),
                  comparisons=comparisons, median_distances=medians, requests=requests)
    paths = []
    if figures:
        result["figure_records"], files = generate_figures(score, plan, requests, audit)
        result["primary_figure_status_counts"] = dict(Counter(r["measured_crossing"]["status"] for r in result["figure_records"]))
        paths.extend(files)
    for name, data in (("accounting.json", result), ("requests.json", requests), ("recommendations.json", rows),
                       ("same_denominator.json", comparisons), ("median_distances.json", medians)):
        path = out / name
        write_json(path, data)
        paths.append(path)
    for name, data in (("recommendations.csv", rows), ("same_denominator.csv", comparisons)):
        path = out / name
        write_csv(path, data)
        paths.append(path)
    summary = "# A13: independent-data accounting\n\n" + "\n\n".join(DEFINITIONS.values())
    summary += "\n\nMeasured request statuses: " + json.dumps(result["crossing_status_counts"]) + ".\n\n"
    summary += comparison_text(comparisons) + "\n" + median_text(medians) + "\n"
    summary += ("\nInterpretation: coverage, conditional reliability, and solved fractions answer different "
                "questions. No method is uniformly best across the readout groups and recommendation rules. "
                "Zero interval distances inside broad brackets do not establish accurate point predictions "
                "or a universal data-requirement law. C84's pooled recommendation counts combine two rules "
                "per request; use the separated same-denominator comparisons above.\n")
    (out / "summary.md").write_text(summary)
    paths.append(out / "summary.md")
    if table:
        try:
            from .a13_dreq_table import generate as generate_table
        except ImportError:
            from a13_dreq_table import generate as generate_table
        paths.extend(generate_table(root, result))
    for relative, before in audit.inputs.items():
        require(hashlib.sha256((root/relative).read_bytes()).hexdigest() == before, "Input modified")
    manifest = out / "files_written.json"
    paths.append(manifest)
    write_json(manifest, {"files": [str(p.relative_to(root)) for p in paths],
                          "input_sha256": audit.inputs, "cpu_only": True,
                          "generators": ["analysis/a13_dreq_accounting.py", "analysis/a13_dreq_table.py"]})
    return result, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-table", action="store_true")
    args = parser.parse_args()
    result, paths = generate(args.root, figures=not args.no_figures, table=not args.no_table)
    print(comparison_text(result["comparisons"]))
    print(median_text(result["median_distances"]))
    print("\nFiles written:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
