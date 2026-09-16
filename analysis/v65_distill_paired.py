#!/usr/bin/env python3
"""Paired trajectory-bootstrap intervals from frozen v50 predictions (CPU only).

Run: python analysis/v65_distill_paired.py
Paper artifacts default to results/v65-distill-paired/paper/ to respect this
analysis's write boundary. Use --paper-dir paper only when separately authorized.
Requires NumPy; does not fit models or load model weights.
"""

try:
    from .paper_table_text import proofread_table
except ImportError:  # Direct scripts and file-based imports.
    try:
        from analysis.paper_table_text import proofread_table
    except ImportError:
        from paper_table_text import proofread_table


import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SOURCE = Path("results/v50-p2v2/compare_test.json")
ORDER = (
    ("gemma3-270m", "test_pool"),
    ("gemma3-1b", "test_pool"),
    ("gemma3-4b", "test_pool"),
    ("gemma3-4b", "dev_pool_heldout_student"),
)
CAPS = ("math", "code", "qa")
STUDENT_LABELS = {"gemma3-270m": "270M", "gemma3-1b": "1B", "gemma3-4b": "4B"}
ROLE_LABELS = {
    "test_pool": "Test pools",
    "dev_pool_heldout_student": "Development pools, held-out 4B student",
}


def validate_rows(rows):
    """Check pairing, stored errors, frozen constants, and capability coverage."""
    groups = defaultdict(list)
    seen = set()
    constants = {}
    coverage = defaultdict(set)
    for row in rows:
        student, role, cap = (row[k] for k in ("student", "role", "cap"))
        if (student, role) not in ORDER:
            continue
        if cap not in CAPS:
            raise ValueError(f"Unexpected capability: {cap}")
        key = (student, role, row["pool"], row["Tc"], cap)
        if key in seen:
            raise ValueError(f"Duplicate point: {key}")
        seen.add(key)
        for field in ("actual", "Tc", "E"):
            if not math.isfinite(row[field]):
                raise ValueError(f"Nonfinite {field}: {key}")
        for form in ("joint+src", "constant"):
            predicted, error = row["pred_at_actual"][form], row["abs"][form]
            if not (math.isfinite(predicted) and math.isfinite(error)):
                raise ValueError(f"Nonfinite {form} prediction/error: {key}")
            if not math.isclose(error, abs(predicted - row["actual"]),
                                rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Stored {form} absolute error disagrees: {key}")
        constant = row["pred_at_actual"]["constant"]
        if constant != constants.setdefault(cap, constant):
            raise ValueError(f"Frozen constant varies within capability: {cap}")
        if "abs_zero" in row and not math.isclose(
                row["abs_zero"], abs(row["actual"]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Stored zero absolute error disagrees: {key}")
        groups[student, role, cap].append(row)
        coverage[student, role, row["pool"], row["Tc"], row["E"]].add(cap)
    expected = {(s, r, c) for s, r in ORDER for c in CAPS}
    if set(groups) != expected:
        raise ValueError(f"Missing groups: {sorted(expected - set(groups))}")
    if any(caps != set(CAPS) for caps in coverage.values()):
        raise ValueError("Capabilities do not share the same trajectory budgets")
    return groups, constants


def bootstrap_maes(cluster_sums, cluster_sizes, draws):
    """Resample whole clusters, then average points (including unequal sizes)."""
    return cluster_sums[draws].sum(axis=1) / cluster_sizes[draws].sum(axis=1)[:, None]


def interval(estimate, samples):
    return {"estimate": float(estimate),
            "ci95": np.quantile(samples, [0.025, 0.975], method="linear").tolist()}


def analyze(rows, resamples=5000, seed=0):
    groups, constants = validate_rows(rows)
    rng = np.random.Generator(np.random.PCG64(seed))
    results = []
    for student, role in ORDER:
        pools = sorted({row["pool"] for row in groups[student, role, CAPS[0]]})
        # One draw matrix per student/role, shared across capabilities/baselines.
        draws = rng.integers(0, len(pools), size=(resamples, len(pools)))
        for cap in CAPS:
            points = sorted(groups[student, role, cap], key=lambda r: (r["pool"], r["Tc"]))
            errors = np.array([[p["abs"]["joint+src"], p["abs"]["constant"],
                                abs(p["actual"])] for p in points], dtype=float)
            counts = Counter(p["pool"] for p in points)
            sizes = np.array([counts[pool] for pool in pools])
            sums = np.array([errors[[p["pool"] == pool for p in points]].sum(axis=0)
                             for pool in pools])
            boot_maes = bootstrap_maes(sums, sizes, draws)
            maes = errors.mean(axis=0)
            comparisons = {}
            for col, baseline in ((1, "constant"), (2, "zero")):
                if maes[col] <= 0 or np.any(boot_maes[:, col] <= 0):
                    raise ValueError(f"Undefined relative improvement: {student}/{role}/{cap}/{baseline}")
                differences = errors[:, col] - errors[:, 0]
                boot_diff = boot_maes[:, col] - boot_maes[:, 0]
                relative = (maes[col] - maes[0]) / maes[col]
                boot_relative = boot_diff / boot_maes[:, col]
                comparisons[baseline] = {
                    "baseline_mae": float(maes[col]),
                    "paired_difference": interval(differences.mean(), boot_diff),
                    "relative_improvement": interval(relative, boot_relative),
                    "relative_improvement_percent": interval(100 * relative, 100 * boot_relative),
                }
            results.append({
                "student": student, "role": role, "cap": cap,
                "n_points": len(points), "n_trajectories": len(pools),
                "clusters": [{"student": student, "pool": pool, "n_points": counts[pool],
                              "Tc": [p["Tc"] for p in points if p["pool"] == pool]}
                             for pool in pools],
                "candidate_mae": float(maes[0]),
                "frozen_constant_prediction": constants[cap],
                "comparisons": comparisons,
            })
    return results


def format_interval(stat, digits=4):
    lo, hi = stat["ci95"]
    return f"{stat['estimate']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def markdown(summary):
    method = summary["method"]
    lines = [
        "# V65 distillation paired intervals", "",
        "Candidate: frozen `joint+src`; baselines: frozen `constant` and zero change.", "",
        "Each estimate averages points within (student, role, capability). "
        "Paired difference is mean(|baseline error| - |candidate error|), in nats. "
        "Relative improvement is (baseline MAE - candidate MAE) / baseline MAE; "
        "the tables express this as a percentage. Positive values favor the candidate.", "",
        f"95% percentile intervals use {method['n_resamples']:,} trajectory-cluster "
        f"bootstrap resamples with replacement, seed {method['seed']} (NumPy PCG64). "
        "The cluster is student × pool; every budget stays together. Each resample "
        "draws the original number of trajectories and averages all resulting points. "
        "The same draws are used across both baselines and all capabilities within "
        "each student/role. Relative improvement is recomputed from each resample's "
        "MAEs. Predictions are fixed; no fitting or form selection is performed.", "",
        "There are only three test trajectories per student and four held-out-student "
        "development trajectories; the intervals describe resampling of these observed "
        "trajectories and have limited resolution.", "",
    ]
    for role in ROLE_LABELS:
        lines += [f"## {ROLE_LABELS[role]}", "",
                  "| Student | Cap. | Points / trajectories | Candidate MAE | Constant MAE | "
                  "Diff vs constant [95% CI] | Relative % [95% CI] | Zero MAE | "
                  "Diff vs zero [95% CI] | Relative % [95% CI] |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in summary["groups"]:
            if row["role"] != role:
                continue
            const, zero = (row["comparisons"][b] for b in ("constant", "zero"))
            cells = [STUDENT_LABELS[row["student"]], row["cap"],
                     f"{row['n_points']} / {row['n_trajectories']}",
                     f"{row['candidate_mae']:.4f}", f"{const['baseline_mae']:.4f}",
                     format_interval(const["paired_difference"]),
                     format_interval(const["relative_improvement_percent"], 1),
                     f"{zero['baseline_mae']:.4f}", format_interval(zero["paired_difference"]),
                     format_interval(zero["relative_improvement_percent"], 1)]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [f"Input: `{summary['input']['path']}` ({summary['input']['n_rows']} rows).",
              f"SHA-256: `{summary['input']['sha256']}`.", ""]
    return "\n".join(lines)


def tex_interval(stat, digits):
    lo, hi = stat["ci95"]
    return (r"\shortstack{" + f"${stat['estimate']:.{digits}f}$" + r"\\"
            + f"$[{lo:.{digits}f},{hi:.{digits}f}]$" + "}")


@proofread_table
def latex(summary):
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{2pt}", r"\renewcommand{\arraystretch}{1.15}",
             r"\caption{Paired distillation errors for the frozen joint$+$src candidate. "
             r"MAEs and paired differences are in nats; positive differences and relative "
             r"improvements favor the candidate. Differences are point means of "
             r"$|e_{\mathrm{baseline}}|-|e_{\mathrm{candidate}}|$; relative improvement is "
             r"$100(\mathrm{MAE}_{\mathrm{baseline}}-\mathrm{MAE}_{\mathrm{candidate}})"
             r"/\mathrm{MAE}_{\mathrm{baseline}}$. Brackets show 95\% percentile "
             + f"intervals from {summary['method']['n_resamples']:,} paired trajectory "
             + f"bootstrap resamples (seed {summary['method']['seed']}); "
             r"a cluster is student $\times$ pool, with all budgets resampled together. "
             r"The same draws are used for both baselines and all capabilities within "
             r"each student/pool-role group, recomputing each ratio per resample. "
             r"Test rows use 12 points from 3 trajectories; held-out-student development "
             r"rows use 16 points from 4 trajectories. Intervals have limited resolution "
             r"with so few trajectories.}",
             r"\label{tab:distill_paired}", r"\begin{tabular}{@{}llrrrrrrr@{}}",
             r"\toprule",
             r"Student & Cap. & \shortstack{Candidate\\MAE} & \multicolumn{3}{c}{Frozen constant baseline} & \multicolumn{3}{c}{Zero baseline} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r"& & & MAE & Diff. [CI] & Rel. \% [CI] & MAE & Diff. [CI] & Rel. \% [CI] \\"]
    for role, label in ROLE_LABELS.items():
        lines += [r"\midrule", r"\multicolumn{9}{l}{\textit{" + label + r"}} \\"]
        previous_student = None
        for row in summary["groups"]:
            if row["role"] != role:
                continue
            student = row["student"]
            if previous_student is not None and student != previous_student:
                lines.append(r"\addlinespace")
            previous_student = student
            cells = [STUDENT_LABELS[student], row["cap"].upper() if row["cap"] == "qa"
                     else row["cap"].capitalize(), f"{row['candidate_mae']:.3f}"]
            for baseline in ("constant", "zero"):
                comp = row["comparisons"][baseline]
                cells += [f"{comp['baseline_mae']:.3f}",
                          tex_interval(comp["paired_difference"], 3),
                          tex_interval(comp["relative_improvement_percent"], 1)]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    root = next((p for p in Path(__file__).resolve().parents if (p / SOURCE).is_file()),
                Path.cwd())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / SOURCE)
    parser.add_argument("--output-dir", type=Path, default=root / "results/v65-distill-paired")
    parser.add_argument("--paper-dir", type=Path,
                        help="Default: OUTPUT_DIR/paper (staged within the write boundary)")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    source = Path(__file__).read_bytes()
    rows = json.loads(raw)["rows"]
    groups = analyze(rows)
    input_path = args.input.resolve()
    summary = {
        "analysis": "v65-distill-paired", "candidate": "joint+src",
        "baselines": ["constant", "zero"], "units": "nats",
        "input": {"path": str(input_path.relative_to(root)) if input_path.is_relative_to(root)
                  else str(input_path), "sha256": hashlib.sha256(raw).hexdigest(),
                  "n_rows": len(rows), "n_rows_used": sum(g["n_points"] for g in groups)},
        "script_sha256": hashlib.sha256(source).hexdigest(),
        "numpy_version": np.__version__,
        "method": {
            "point_estimate": "mean over points within (student, role, cap)",
            "paired_difference": "mean(abs(baseline_error) - abs(candidate_error))",
            "relative_improvement": "(baseline_mae - candidate_mae) / baseline_mae",
            "positive_favors": "candidate", "cluster": ["student", "pool"],
            "n_resamples": 5000, "seed": 0, "rng": "numpy.random.Generator(PCG64)",
            "sampling": "sample n_trajectories with replacement; include every point of each draw",
            "weighting": "point-weighted mean in original data and each bootstrap resample",
            "draw_sharing": "same draws for both baselines and all caps within each student/role",
            "group_order": [list(pair) for pair in ORDER], "pool_order": "lexicographic",
            "ci": "95% percentile; numpy.quantile([0.025, 0.975], method='linear')",
            "relative_ci": "ratio recomputed from each resample's MAEs",
            "predictions": "stored frozen predictions; no refitting or selection",
        },
        "limitations": ["Only 3 trajectories per test group and 4 per held-out development group; "
                        "bootstrap intervals have limited resolution."],
        "groups": groups,
    }
    paper_dir = args.paper_dir or args.output_dir / "paper"
    artifacts = {
        args.output_dir / "summary.json": json.dumps(summary, indent=2, allow_nan=False) + "\n",
        args.output_dir / "summary.md": markdown(summary),
        paper_dir / "tables/distill_paired.tex": latex(summary),
        paper_dir / "code/analysis/v65_distill_paired.py": source.decode("utf-8"),
    }
    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"WROTE {path}")


if __name__ == "__main__":
    main()
