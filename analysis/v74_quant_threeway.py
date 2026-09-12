#!/usr/bin/env python3
"""CPU-only V74 report from the immutable V69 develop/freeze/compare artifacts.

    python -B analysis/v74_quant_threeway.py
    python -B analysis/v74_quant_threeway.py --check

Writes results/v74-quant-threeway/{quant_threeway.json,quant_threeway.tex,
median_algorithm.md}, paper/paper/tables/quant_threeway.tex, and an identical
script at paper/analysis/v74_quant_threeway.py. No fitting or measurement.
The R mapping is a supplied post-test recommendation, not another frozen selector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


SOURCE = Path("results/v69-quant-confirm")
OUT = Path("results/v74-quant-threeway")
CAPS = ("math", "code", "qa")
METHODS = ("low_order_2d", "bilinear", "same_input_interpolation", "bit_only", "median", "zero")
LABELS = dict(zip(METHODS, ("2-D surface", "Bilinear", "Piecewise interpolation",
                          "Bit-only", "Per-configuration median", "Zero")))
SELECTED = {"math": "low_order_2d", "code": "median", "qa": "zero"}
DEV_STATES = tuple(f"pythia-{size}@step{step}" for size in ("160m", "410m", "1.4b")
                   for step in (16000, 143000))
SEEN_STATES = ("pythia-410m@step143000", "pythia-1.4b@step16000")
NEW_STATE = "pythia-1.4b@step112000"
PANELS = {
    "development_state_boundary": (SEEN_STATES, (32, 512),
                                   r"Development states, $g\in\{32,512\}$"),
    "new_state_boundary": ((NEW_STATE,), (32, 512), r"New state, $g\in\{32,512\}$"),
    "new_state_interior": ((NEW_STATE,), (128,), r"New state, $g=128$"),
}
DEV_CONFIGS = tuple(f"b{b}_g{g}" for b in (3, 4, 5) for g in (64, 128, 256))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def close(actual, expected, context):
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"{context}: {actual!r} != {expected!r}")


def key(row):
    return row["state"], row["config"], row["capability"]


def recommendation(panel, capability):
    return ("same_input_interpolation"
            if panel == "development_state_boundary" and capability in ("math", "code")
            else "median")


def anchor_prediction(anchors, config):
    """V69's rule restricted to its tested bits and g=32,128,512; for auditing."""
    bits, group = (int(part[1:]) for part in config.split("_"))
    require(bits in (3, 4, 5) and group in (32, 128, 512), f"Unexpected test config: {config}")
    middle = anchors[f"b{bits}_g128"]
    if group == 128:
        return middle
    edge = anchors[f"b{bits}_g{64 if group == 32 else 256}"]
    return max(0.0, 2 * edge - middle)


def verify_sources(root, develop, frozen, compared, hashes):
    require(frozen["provenance"]["develop_sha256"] == hashes[str(SOURCE / "develop.json")],
            "develop.json differs from the frozen source")
    require(compared["provenance"]["freeze_sha256"] == hashes[str(SOURCE / "freeze.json")],
            "freeze.json differs from the comparison source")
    require(compared["complete"] and not compared["missing"]
            and compared["n_measured_cells"] == compared["n_expected_cells"] == 21,
            "V69 comparison must contain all 21 cells")
    for artifact in (develop, frozen):
        require(artifact["methods"] == list(METHODS), "Unexpected candidate set/order")
    for artifact in (develop, frozen, compared):
        require({cap: artifact["selected"][cap]["candidate"] for cap in CAPS} == SELECTED,
                "Development selections changed")
    require(compared["selected"] == frozen["selected"], "Comparison selections differ from freeze")
    require(develop["models"] == frozen["models"]
            and develop["standardization"] == frozen["standardization"], "Frozen models changed")
    for field in ("boundary_rule", "selection_rule", "test_cells"):
        require(develop[field] == frozen[field], f"Frozen {field} changed")
    # Read the implementation used for the algorithm note; never invoke V69 modes.
    for rel in ("analysis/v69_quant_confirm.py", "analysis/v55_quant_group_fit.py"):
        require(sha256((root / rel).read_bytes()) == develop["code_sha256"][rel],
                f"{rel} differs from the V69 implementation recorded at development")

    require([s["tag"] for s in develop["dev_states"]] == list(DEV_STATES),
            "Unexpected six development states")
    require(develop["dev_configs"] == list(DEV_CONFIGS), "Unexpected development grid")
    dev_rows = {key(row): row for row in develop["dev_rows"]}
    expected_dev = {(state, config, cap) for state in DEV_STATES
                    for config in DEV_CONFIGS for cap in CAPS}
    require(len(dev_rows) == len(develop["dev_rows"]) == 162 and set(dev_rows) == expected_dev,
            "Incomplete or duplicate development data")
    for cap in CAPS:
        for config in DEV_CONFIGS:
            median = statistics.median(dev_rows[state, config, cap]["dL"] for state in DEV_STATES)
            close(median, frozen["models"][cap]["median"]["anchors"][config],
                  f"Development median {cap}/{config}")

    expected = {(state, f"b{b}_g{g}", cap): panel
                for panel, (states, groups, _) in PANELS.items()
                for state in states for b in (3, 4, 5) for g in groups for cap in CAPS}
    freeze_rows = {key(row): row for row in frozen["predictions"]}
    compare_rows = {key(row): row for row in compared["rows"]}
    require(len(freeze_rows) == len(frozen["predictions"]) == 63 and set(freeze_rows) == set(expected),
            "Incomplete or duplicate frozen panel")
    require(len(compare_rows) == len(compared["rows"]) == 63 and set(compare_rows) == set(expected),
            "Incomplete or duplicate comparison panel")
    for row_key, panel in expected.items():
        fr, cr = freeze_rows[row_key], compare_rows[row_key]
        require(fr["test_set"] == cr["test_set"] == panel, f"Wrong panel: {row_key}")
        require(all(cr[field] == value for field, value in fr.items()),
                f"Comparison altered a frozen prediction/input: {row_key}")
        require(set(fr["predictions"]) == set(METHODS), f"Missing candidates: {row_key}")
        cap, config = fr["capability"], fr["config"]
        require(fr["selected_candidate"] == SELECTED[cap], f"Wrong frozen selection: {row_key}")
        close(fr["selected_prediction"], fr["predictions"][SELECTED[cap]], f"Selected {row_key}")
        close(cr["dL"], cr["loss"] - cr["dense"], f"Signed target {row_key}")
        for method in METHODS:
            close(abs(fr["predictions"][method] - cr["dL"]), cr["absolute_errors"][method],
                  f"Absolute error {row_key}/{method}")
        stats = frozen["standardization"]
        z = [1.0] + [(x - center) / scale for x, center, scale in
                     zip(fr["phi_raw"], stats["center"], stats["scale"])]
        model = frozen["models"][cap]
        interpolation_anchors = {config: sum(a * b for a, b in zip(beta, z))
                                 for config, beta in model["same_input_interpolation"]["anchors"].items()}
        for method, anchors in (("median", model["median"]["anchors"]),
                                ("same_input_interpolation", interpolation_anchors)):
            close(anchor_prediction(anchors, config), fr["predictions"][method],
                  f"Frozen anchor/boundary prediction {row_key}/{method}")


def build_report(develop, frozen, compared, hashes):
    rows = []
    require(set(compared["test_sets"]) == set(PANELS), "Unexpected comparison panels")
    for panel, (states, groups, _) in PANELS.items():
        n = len(states) * 3 * len(groups)
        require(compared["test_sets"][panel]["n_expected_cells"] == n, f"Wrong panel size: {panel}")
        for cap in CAPS:
            cells = [r for r in compared["rows"] if r["test_set"] == panel and r["capability"] == cap]
            candidates = {}
            for method in METHODS:
                metric = compared["test_sets"][panel]["scores"][method][cap]
                require(metric["n"] == len(cells) == n and metric["n_states"] == len(states),
                        f"Wrong score counts: {panel}/{cap}/{method}")
                errors = [abs(r["predictions"][method] - r["dL"]) for r in cells]
                state_mae = {state: statistics.mean(abs(r["predictions"][method] - r["dL"])
                                                   for r in cells if r["state"] == state)
                             for state in states}
                require(set(metric["state_mae"]) == set(states), "Wrong score states")
                close(statistics.mean(errors), metric["mae"], f"MAE {panel}/{cap}/{method}")
                close(statistics.mean(state_mae.values()), metric["macro_mae"],
                      f"Macro MAE {panel}/{cap}/{method}")
                for state, value in state_mae.items():
                    close(value, metric["state_mae"][state], f"State MAE {state}/{cap}/{method}")
                candidates[method] = metric
            selected, recommended = SELECTED[cap], recommendation(panel, cap)
            rows.append({"test_set": panel, "capability": cap, "n": n, "n_states": len(states),
                         "frozen_development_selected": {"candidate": selected, **candidates[selected]},
                         "frozen_candidates": candidates,
                         "post_test_recommended": {"label": "R", "candidate": recommended,
                                                   "status": "post-test recommendation; not a frozen selection",
                                                   **candidates[recommended]}})
    return {"schema_version": 1, "source": "V69 frozen predictions and completed comparison",
            "metric": "MAE of signed dL in nats; equal weight per measured state/configuration cell",
            "selection_rule": develop["selection_rule"], "selected": SELECTED,
            "recommendation_rule": "R: piecewise interpolation for math/code on development states; "
                                   "per-configuration median for QA and for all capabilities on new states. "
                                   "Specified post-test rule, not the minimum error in each test panel.",
            "frozen_at_utc": frozen["frozen_at_utc"],
            "input_sha256": hashes,
            "validation": {"frozen_capability_rows": 63, "candidate_scores": 54,
                           "median_anchors": 27, "median_and_interpolation_predictions": 126},
            "rows": rows}


def render_table(report):
    lines = [r"% Generated by analysis/v74_quant_threeway.py; V69 inputs are read-only.",
             r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{V74 three-way quantization confirmation. MAE in nats (lower is better), "
             r"equally weighting measured cells at $b\in\{3,4,5\}$; $n$ is per capability. "
             r"\textbf{D}: FROZEN development-LOSO selection (math: surface; code: median; QA: zero), "
             r"including the 0.02-nat tie rule. F: every frozen candidate. "
             r"\textbf{R}: post-test recommended rule (piecewise interpolation for math/code on "
             r"development states; median for QA and for all capabilities on new states). "
             r"R reuses frozen candidate predictions but its choice is retrospective, not a "
             r"prospective selection or the test minimum. Development states tested: "
             r"Pythia-410M@143k and 1.4B@16k; new state: 1.4B@112k.}",
             r"\label{tab:quant_threeway}", r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r"Status & Predictor & Math & Code & QA \\", r"\midrule"]
    for index, (panel, (_, _, title)) in enumerate(PANELS.items()):
        if index:
            lines.append(r"\addlinespace")
        rows = {r["capability"]: r for r in report["rows"] if r["test_set"] == panel}
        n = rows["math"]["n"]
        lines.append(r"\multicolumn{5}{@{}l}{\textit{" + title + f"; $n={n}$" + r"}} \\")
        lines.append(" & ".join([r"\textbf{D}", "FROZEN development-selected", *[
            f"{rows[cap]['frozen_development_selected']['mae']:.4f}" for cap in CAPS]]) + r" \\")
        for method in METHODS:
            lines.append(" & ".join(["F", LABELS[method], *[
                f"{rows[cap]['frozen_candidates'][method]['mae']:.4f}" for cap in CAPS]]) + r" \\")
        lines.append(" & ".join([r"\textbf{R}", "Post-test recommended rule", *[
            f"{rows[cap]['post_test_recommended']['mae']:.4f}" for cap in CAPS]]) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def render_algorithm(develop, frozen, hashes):
    lines = ["# V74: the V69 median and interpolation algorithms", "",
             "This describes the implementation frozen in V69, verified against its recorded code "
             "SHA-256 values. No model is refitted and no test response is used to construct anchors.", "",
             "## Development data and response", "",
             "The development snapshot is `dev_rows` in "
             "[develop.json](../v69-quant-confirm/develop.json), loaded by `load_dev` in "
             "[analysis/v69_quant_confirm.py](../../analysis/v69_quant_confirm.py). "
             "It uses all 54 previously unblinded V54 state/configuration cells: six development "
             "states, each at b in {3,4,5} and g in {64,128,256}, with math, code and QA responses "
             "(162 capability rows). The six states and their original measurement files are:", ""]
    for state in develop["dev_states"]:
        lines.append(f"- `{state['tag']}`: `{state['path']}`")
    lines += ["", "For each capability c and state s, the signed response is "
              "`dL[s,c,b,g] = loss[s,c,b,g] - dense_loss[s,c]` in nats, with the dense reference "
              "from the same V54 JSON file. Development uses only the nine configurations above, "
              "even if the original files now also contain confirmation measurements. "
              "g=32/512 were never measured in the development data; 1.4B@112000 is absent from "
              "all development fits. This is V69's full 54-cell development set, not V55's original "
              "24-cell development split.", "", "## Per-configuration median", "",
              "In V69 `fit_all`, independently for each capability and each of the nine (b,g) "
              "configurations, the scalar anchor is:", "", "```text",
              "M[c,b,g] = np.median([dL[s,c,b,g] for s in the six development states])",
              "```", "", "For six states, `np.median` averages the third and fourth sorted signed "
              "values. It does not take absolute values or floor the anchors. Each capability has "
              "nine anchors, with no ridge fit and no dependence on the target state's features. "
              "The same frozen median predictions apply to every target state. During development "
              "LOSO only, each fold instead computes anchors over its five training states; the "
              "confirmation freeze uses all six.", "",
              "V69 `predict_all` passes these anchors to its own `interpolate`, also used by the "
              "piecewise interpolation candidate. Thus an unmeasured group size has a defined "
              "prediction; no median is computed at g=32/512. At each tested b:", "", "```text",
              "prediction[c,b,32]  = max(0, 2*M[c,b,64]  - M[c,b,128])",
              "prediction[c,b,512] = max(0, 2*M[c,b,256] - M[c,b,128])",
              "prediction[c,b,128] = M[c,b,128]  # signed; no floor inside the grid",
              "```", "", "The floor is applied after extrapolation, not to individual anchors.", "",
              "## Piecewise interpolation and its boundary rule", "",
              "The interpolation candidate's anchors differ from the median's. V69 `design` and "
              "`fit_all` fit a four-coefficient ridge regression for each capability/configuration "
              "on the six development states (36 coefficients per capability): "
              "`A[c,b,g;s] = beta[c,b,g] dot phi(s,c)`. The configuration-indicator design makes "
              "these nine regression blocks independent. Ridge lambda is 0.001 and penalizes "
              "every coefficient, including intercepts. "
              "[analysis/v55_quant_group_fit.py](../../analysis/v55_quant_group_fit.py) supplies "
              "`ridge_fit`, `coordinates`, `standardization` and `phi`: "
              "`phi = [1, z(log(N0)), z(L0c), z(log(D0))]`, using natural logs and population "
              "mean/std pooled over the development rows and capabilities; constant scales become "
              "1. N0 counts transformer matrices excluding embeddings/head; "
              "D0 = step * 2097152. At prediction time the frozen scaler and the target's frozen "
              "`phi_raw` produce nine scalar anchors. These are fitted predictions for the "
              "target state, not its measured quantized losses.", "",
              "V69 `grid_weights` and `interpolate` then use the same rule for either anchor type:", "",
              "1. Coordinates are `x = log2(2**(b-1)-1)` and `v = log2(g/128)`. "
              "The measured grid has x at b={3,4,5} and v={-1,0,1} for g={64,128,256}.",
              "2. Choose adjacent bit anchors using `searchsorted(xs, x, side='right') - 1`, "
              "clipped to indices 0 or 1. Choose g=64,128 if v<=0 and g=128,256 otherwise. "
              "For local coordinates tx and tv, sum the four anchors with product weights "
              "`(1-tx)*(1-tv)`, `(1-tx)*tv`, `tx*(1-tv)`, and `tx*tv`. "
              "All confirmation bits are measured bit anchors, so the other bit's weights are zero.",
              "3. For g<64, extend the line through g=64,128 in v; for g>256, extend the line "
              "through g=128,256 in v. Do not clamp g to the grid or use the two outermost "
              "g=64,256 anchors. At g=32, v=-2 and the g weights are (2,-1); at g=512, "
              "v=2 and they are (-1,2). The formulas above apply with A replacing M.",
              "4. Return `max(0, weighted_sum)` only when `abs(v)>1` (g<64 or g>256). "
              "For 64<=g<=256, return the signed weighted sum with no floor. "
              "There is no bit extrapolation; V69's confirmation protocol permits b=3,4,5 only.", "",
              "The boundary helper is V69's implementation, not V55's older four-corner "
              "unclipped interpolation function. The exact boundary rule recorded in both "
              "development and freeze is:", "", "> " + frozen["boundary_rule"], "",
              "## Frozen median anchors and resulting boundary predictions", "",
              "Values below are in nats, rounded to six decimals. g=128 predictions equal the "
              "middle anchor, including a negative QA anchor at b=5.", "",
              "| Capability | b | M(g=64) | M(g=128) | M(g=256) | Prediction g=32 | Prediction g=512 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for cap in CAPS:
        anchors = frozen["models"][cap]["median"]["anchors"]
        for b in (3, 4, 5):
            values = [anchors[f"b{b}_g{g}"] for g in (64, 128, 256)]
            values += [anchor_prediction(anchors, f"b{b}_g{g}") for g in (32, 512)]
            lines.append(f"| {cap.upper() if cap == 'qa' else cap.title()} | {b} | "
                         + " | ".join(f"{value:.6f}" for value in values) + " |")
    lines += ["", "For example, QA at b=3,g=32 extrapolates to -0.087868346 nats and is floored "
              "to zero; at b=5,g=128 the -0.001752614-nat median remains signed.", "",
              "## Three-way table and verification", "",
              "D is the frozen development selection: surface for math, median for code, zero for QA. "
              "F rows give all six frozen candidates. R is the supplied post-test recommendation: "
              "interpolation for math/code on seen development states; median for QA and all new-state "
              "capabilities. R uses existing frozen predictions but the rule choice is retrospective. "
              "It is not chosen by minimizing each panel's test error (for example, new-state boundary "
              "code favors interpolation, and new-state g=128 QA favors zero).", "",
              "The V74 script verifies the develop-to-freeze-to-compare SHA-256 chain, unchanged "
              "frozen predictions, all 27 median anchors from development responses, all 126 median "
              "and interpolation predictions, and all 54 panel/capability/candidate MAEs. "
              "MAE averages absolute errors against signed dL equally over measured cells. "
              "Each panel is balanced across states, so its macro state MAE equals its cell MAE. "
              "The JSON retains full precision; the paper table rounds to four decimals.", "",
              "Regenerate with `python -B analysis/v74_quant_threeway.py`; verify generated artifacts "
              "without writes with `python -B analysis/v74_quant_threeway.py --check`.", "",
              "### Input SHA-256", ""]
    for path, digest in hashes.items():
        lines.append(f"- `{path}`: `{digest}`")
    for path in ("analysis/v69_quant_confirm.py", "analysis/v55_quant_group_fit.py"):
        lines.append(f"- `{path}`: `{develop['code_sha256'][path]}`")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate inputs and existing outputs without writes")
    args = parser.parse_args()
    root = next((p for p in Path(__file__).resolve().parents if (p / SOURCE / "develop.json").is_file()), None)
    require(root is not None, "Cannot locate results/v69-quant-confirm/develop.json")
    # Snapshot every V69 output, including Markdown, to verify preservation.
    before = {p: (sha256(p.read_bytes()), p.stat().st_mtime_ns)
              for p in (root / SOURCE).rglob("*") if p.is_file()}
    raw = {name: (root / SOURCE / f"{name}.json").read_bytes() for name in ("develop", "freeze", "compare")}
    hashes = {str(SOURCE / f"{name}.json"): sha256(data) for name, data in raw.items()}
    develop, frozen, compared = (json.loads(raw[name]) for name in ("develop", "freeze", "compare"))
    verify_sources(root, develop, frozen, compared, hashes)
    report = build_report(develop, frozen, compared, hashes)
    table = render_table(report)
    outputs = {
        OUT / "quant_threeway.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
        OUT / "quant_threeway.tex": table,
        OUT / "median_algorithm.md": render_algorithm(develop, frozen, hashes),
        Path("paper/paper/tables/quant_threeway.tex"): table,
        Path("paper/analysis/v74_quant_threeway.py"): (root / "analysis/v74_quant_threeway.py").read_text(),
    }
    for rel, value in outputs.items():
        path = root / rel
        require(not path.is_symlink() and all(not p.is_symlink() for p in path.parents if p != root),
                f"Refusing symlink output: {rel}")
        if args.check:
            require(path.is_file() and path.read_text() == value, f"Missing/stale output: {rel}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
        print(f"{'CHECKED' if args.check else 'WROTE'} {rel}")
    after = {p: (sha256(p.read_bytes()), p.stat().st_mtime_ns)
             for p in (root / SOURCE).rglob("*") if p.is_file()}
    require(before == after, "V69 output contents or modification times changed")
    print("Validated 63 frozen rows, 54 candidate scores, 27 median anchors and 126 anchor predictions; V69 unchanged.")


if __name__ == "__main__":
    main()
