#!/usr/bin/env python3
"""Kumar PTQ baseline, retrospective (rebuttal round, 2026-11-07). CPU only.

Run: python -B analysis/r1_kumar_baseline.py
Uses recorded signed responses, never fits log(delta), never drops negative cells.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import least_squares
from analysis import v36_pythia_controlled_fit as v36
from analysis import v69_quant_confirm as v69
from analysis.paper_table_layout import house_style, table_layout

ID = "r1-kumar-baseline"
STATUS = "retrospective (rebuttal round, 2026-11-07)"
CAPS = v36.CAPS
SEED = 20261107
SOURCE = "https://proceedings.iclr.cc/paper_files/paper/2025/hash/b2cac94f82928a85055987d9fd44753f-Abstract-Conference.html"
TARGETS = tuple(f"pythia-{size}@step{step}" for size in ("1b", "6.9b") for step in (32000, 112000))
CHANNEL_METHODS = ("full", "noD0", "per_bit_mean", "per_bit_median", "zero")


def read(root, relative, hashes):
    raw = (root / relative).read_bytes()
    hashes[str(relative)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def channel_models(rows):
    models = {}
    for c in CAPS:
        cr = [r for r in rows if r["capability"] == c]
        models[c] = {m: v36.fit_direct(cr, input_fields=fields) for m, fields in
                     (("full", ("N0", "L0", "D0")), ("noD0", ("N0", "L0")))}
        for m, op in (("per_bit_mean", np.mean), ("per_bit_median", np.median)):
            models[c][m] = {b: float(op([r["observed"] for r in cr if r["config"] == b])) for b in (8, 6, 4, 3)}
    return models


def channel_predict(models, row):
    m, b = models[row["capability"]], row["config"]
    if b == 5:
        low = channel_predict(models, {**row, "config": 4})
        high = channel_predict(models, {**row, "config": 6})
        return {k: .5*(low[k]+high[k]) for k in CHANNEL_METHODS}
    out = {name: float(v36.predict(m[name], [v36.basic_input(row, input_fields=m[name]["input_fields"])])[0])
           for name in ("full", "noD0")}
    out.update({name: m[name][b] for name in ("per_bit_mean", "per_bit_median")})
    return {**out, "zero": 0.0}


def normalized(row, grouped=False):
    state = row.get("state", row.get("cell", "")).replace("--", "@")
    size, step = state.removeprefix("pythia-").split("@step")
    b, g = (map(int, row["config"][1:].split("_g")) if grouped else (int(row["config"]), 1))
    return dict(state=state, cap=row["capability"], config=row["config"], P=b, g=g,
                N0=v36.matrix_n0(size), D0=int(step)*v36.TOKENS_PER_STEP,
                y=float(row["dL"] if grouped else row["observed"]))


def load_panels(root=ROOT):
    hashes = {}
    _, dev, source_hashes, _ = v69.load_dev(root)
    hashes.update(source_hashes)
    stats, models = v69.fit_all(dev)
    cv = v69.loso(dev)
    selected = v69.select(cv["scores"])
    baseline = {c: min((m for m in v69.METHODS if m != selected[c]["candidate"]),
                      key=lambda m: cv["scores"][m][c]["macro_mae"]) for c in CAPS}
    compared = read(root, "results/v69-quant-confirm/compare.json", hashes)
    recorded_dev = read(root, "results/v69-quant-confirm/develop.json", hashes)
    assert dev == recorded_dev["dev_rows"]
    assert {c: s["candidate"] for c, s in selected.items()} == {c: s["candidate"] for c, s in compared["selected"].items()}
    for c in CAPS:
        np.testing.assert_allclose(selected[c]["loso_macro_mae"], compared["selected"][c]["loso_macro_mae"], atol=1e-10)
    test = []
    wanted = {(r["state"], r["config"]) for r in v69.test_cells()}
    assert {(r["state"], r["config"]) for r in compared["rows"]} == wanted
    for row in compared["rows"]:
        pred = v69.predict_all(models[row["capability"]], row["phi_raw"], row["config"], stats)
        for m in v69.METHODS:
            np.testing.assert_allclose(pred[m], row["predictions"][m], atol=1e-10, rtol=1e-9)
        path = v69.v55.measurement_path(root / v69.DATA_REL, row["state"]).relative_to(root)
        measured = read(root, path, hashes)
        assert row["dL"] == measured[row["config"]][row["capability"]]-measured["dense"][row["capability"]]
        delivered = "same_input_interpolation" if row["test_set"] == "development_state_boundary" and row["capability"] != "qa" else "median"
        test.append({**normalized(row, True), "subset": row["test_set"], "comparison_predictions": pred,
                     "delivered": delivered, "baseline": baseline[row["capability"]]})
    panels = {"grouped": dict(form="kumar+g", adapted=True, dev=[normalized(r, True) for r in dev], test=test,
        baseline_selection="final_prediction_table: minimum whole-state LOSO macro MAE excluding the frozen selected candidate",
        development_baseline=baseline, development_scores=cv["scores"], frozen_selected=selected)}
    all_rows, source_hashes = v36.load_grid(root=root, sizes=("160m", "410m", "1.4b"))
    hashes.update(source_hashes)
    all_rows = [r for r in all_rows if r["arm"] == "quantization"]
    for protocol, steps in (("A", (16000, 64000, 143000)), ("B", (16000, 64000))):
        dev = [r for r in all_rows if r["step"] in steps]
        models = channel_models(dev)
        errors = {c: {m: [] for m in CHANNEL_METHODS} for c in CAPS}
        for state in sorted({r["cell"] for r in dev}):
            fold = channel_models([r for r in dev if r["cell"] != state])
            for r in (r for r in dev if r["cell"] == state):
                pred = channel_predict(fold, r)
                for m in CHANNEL_METHODS:
                    errors[r["capability"]][m].append(abs(pred[m]-r["observed"]))
        cv = {c: {m: float(np.mean(v)) for m, v in e.items()} for c, e in errors.items()}
        baseline = {c: min(CHANNEL_METHODS[1:], key=lambda m: cv[c][m]) for c in CAPS}
        test = []
        for state in TARGETS:
            size, step = state.removeprefix("pythia-").split("@step")
            frozen = read(root, f"results/v49-p1v2/predictions_{state}.json", hashes)
            compared = read(root, f"results/v49-p1v2/compare_{state}.json", hashes)
            measured = read(root, f"results/v10-quantization/{state.replace('@', '--')}/quant_losses.json", hashes)
            for r in compared["protocols"][protocol]["rows_quant"]:
                raw = dict(state=state, capability=r["cap"], config=r["bit"], observed=r["actual"],
                           N0=v36.matrix_n0(size), D0=int(step)*v36.TOKENS_PER_STEP,
                           L0=frozen["target"]["dense"][r["cap"]])
                pred = channel_predict(models, raw)
                assert raw["observed"] == measured[str(r["bit"])][r["cap"]]-measured["dense"][r["cap"]]
                for m in CHANNEL_METHODS:
                    np.testing.assert_allclose(pred[m], r[m]["pred"], rtol=1e-9, atol=1e-9)
                test.append({**normalized(raw), "subset": state, "comparison_predictions": pred,
                             "delivered": "per_bit_median", "baseline": baseline[r["cap"]]})
        panels["channel_"+protocol] = dict(form="kumar", adapted=False, dev=[normalized(r) for r in dev], test=test,
            baseline_selection="Retrospective whole-state LOSO minimum among V49 noD0, per-bit mean, per-bit median, zero; full is the source-conditioned candidate. V49 did not record a development-selected winner.",
            development_baseline=baseline, development_scores=cv)
    return panels, hashes


def design(rows, grouped, per_capability):
    caps = np.array([CAPS.index(r["cap"]) for r in rows])
    raw = np.array([[-(r["P"]-3), np.log(r["D0"]/1e11), -np.log(r["N0"]/1e9),
                     *([np.log(r["g"]/128)] if grouped else [])] for r in rows])
    indicators = np.eye(3)[caps]
    x = np.column_stack((indicators, (indicators[:, :, None]*raw[:, None, :]).reshape(len(rows), -1)
                         if per_capability else raw))
    names = ["log_A_"+c for c in CAPS]
    slope_names = ["inverse_gamma_post", "gamma_D", "gamma_N"] + (["gamma_g"] if grouped else [])
    names += [f"{c}:{n}" for c in CAPS for n in slope_names] if per_capability else slope_names
    lower = [-40.]*3 + ([1e-8, -10., -10.] + ([-10.] if grouped else []))*(3 if per_capability else 1)
    upper = [40.]*3 + ([20., 10., 10.] + ([10.] if grouped else []))*(3 if per_capability else 1)
    return x, names, np.array(lower), np.array(upper)


def solve(x, y, lower, upper, starts, weights=None):
    weights = np.ones(len(y)) if weights is None else np.sqrt(weights)
    def predict(t):
        return np.exp(x @ t)
    def fun(t):
        return weights*(predict(t)-y)
    def jac(t):
        return (weights*predict(t))[:, None]*x
    best = None
    for start in starts:
        fit = least_squares(fun, np.clip(start, lower+1e-10, upper-1e-10), jac=jac,
                            bounds=(lower, upper), x_scale="jac", max_nfev=2000,
                            ftol=1e-10, xtol=1e-10, gtol=1e-9)
        if best is None or fit.cost < best.cost:
            best = fit
    return best


def fit_kumar(rows, grouped=False, per_capability=False, n_bootstrap=1000):
    x, names, lower, upper = design(rows, grouped, per_capability)
    y = np.array([r["y"] for r in rows])
    initial = np.zeros(x.shape[1])
    initial[:3] = [np.log(max(.001, np.mean([r["y"] for r in rows if r["cap"] == c and r["P"] == 3]))) for c in CAPS]
    rate_indices = [i for i, name in enumerate(names) if "inverse_gamma_post" in name]
    starts = []
    for rate in (.5, 1.5, 3.):
        for slope in (0., 1.):
            t = initial.copy()
            t[3:] = slope
            t[rate_indices] = rate
            starts.append(t)
    fit = solve(x, y, lower, upper, starts)
    if not fit.success:
        raise RuntimeError(f"Kumar NLS did not converge: {fit.message}")
    states = sorted({r["state"] for r in rows})
    state_idx = np.array([states.index(r["state"]) for r in rows])
    rng = np.random.default_rng(SEED)
    draws, failures, deficient, boundary = [], [], [], []
    # Cluster multiplicities weight ALL signed cells and capabilities of a sampled state together.
    for b in range(n_bootstrap):
        counts = np.bincount(rng.integers(0, len(states), len(states)), minlength=len(states))
        weights = counts[state_idx]
        f = solve(x, y, lower, upper, [fit.x, starts[0]], weights)
        draws.append(f.x.tolist())
        failures.append(not bool(f.success))
        deficient.append(np.linalg.matrix_rank(x[weights > 0]) < x.shape[1])
        boundary.append(bool(np.any((f.x-lower < 1e-5) | (upper-f.x < 1e-5))))
        if (b+1) % 250 == 0:
            print(f"  {'per-capability' if per_capability else 'shared'} bootstrap {b+1}/{n_bootstrap}", flush=True)
    samples = np.array(draws).reshape(n_bootstrap, len(names))
    exponents = {}
    for c in CAPS:
        get = lambda name: names.index(f"{c}:{name}" if per_capability else name)
        gamma_d, gamma_n, rate = (float(fit.x[get(n)]) for n in ("gamma_D", "gamma_N", "inverse_gamma_post"))
        g = float(fit.x[get("gamma_g")]) if grouped else 0.
        # A is the damage at P=3, N=1e9, D=1e11, g=128. log_C is in raw units.
        log_c = float(fit.x[CAPS.index(c)]+3*rate-gamma_d*np.log(1e11)+gamma_n*np.log(1e9)-g*np.log(128))
        entry = dict(gamma_D=gamma_d, gamma_N=gamma_n, gamma_post=1/rate, gamma_g=g if grouped else None,
                     log_C=log_c, C=float(np.exp(log_c)) if log_c < 700 else None,
                     A_at_reference=float(np.exp(fit.x[CAPS.index(c)])), gamma_D_positive=gamma_d > 0)
        for name in ("gamma_D", "gamma_N"):
            vals = samples[:, get(name)]
            valid = np.logical_not(np.array(failures) | np.array(deficient))
            entry[name+"_ci95"] = np.quantile(vals, [.025, .975]).tolist() if len(vals) else None
            entry[name+"_identified_ci95"] = np.quantile(vals[valid], [.025, .975]).tolist() if np.any(valid) else None
        entry["bootstrap_fraction_gamma_D_positive"] = float(np.mean(samples[:, get("gamma_D")] > 0)) if n_bootstrap else None
        exponents[c] = entry
    return dict(parameters=fit.x.tolist(), parameter_names=names, n_parameters=len(names),
        n_extra_group_parameters=(3 if per_capability else 1) if grouped else 0,
        bounds={"lower": lower.tolist(), "upper": upper.tolist()},
        sse=float(2*fit.cost), converged=bool(fit.success), optimality=float(fit.optimality),
        active_bounds=[names[i] for i in np.flatnonzero((fit.x-lower < 1e-5) | (upper-fit.x < 1e-5))],
        design_rank=int(np.linalg.matrix_rank(x)), exponents=exponents,
        bootstrap=dict(n_resamples=n_bootstrap, seed=SEED, unit="development state", states=states,
            samples=draws, failed_indices=np.flatnonzero(failures).tolist(),
            rank_deficient_indices=np.flatnonzero(deficient).tolist(), boundary_indices=np.flatnonzero(boundary).tolist(),
            interval="percentile 95%; all draws retained; identified-only sensitivity excludes failed/rank-deficient fits"))


def evaluate(panel, fits):
    test = panel["test"]
    predictions = {}
    for variant, fit in fits.items():
        x, _, _, _ = design(test, panel["adapted"], variant == "per_capability")
        predictions[variant] = np.exp(x @ fit["parameters"])
    cells = []
    for i, r in enumerate(test):
        p = {**r["comparison_predictions"], **{v: float(pred[i]) for v, pred in predictions.items()}}
        p.update(delivered=p[r["delivered"]], baseline=p[r["baseline"]])
        cells.append({**r, "predictions": p, "absolute_errors": {m: abs(v-r["y"]) for m, v in p.items()}})
    subsets = {"all": list(range(len(cells)))}
    for subset in dict.fromkeys(r["subset"] for r in cells):
        subsets[subset] = [i for i, r in enumerate(cells) if r["subset"] == subset]
    if panel["adapted"]:
        subsets["new_state"] = [i for i, r in enumerate(cells) if r["state"] == v69.NEW_TAG]
    else:
        subsets["bits_ge4"] = [i for i, r in enumerate(cells) if r["P"] >= 4]
        subsets["bit3"] = [i for i, r in enumerate(cells) if r["P"] == 3]
        subsets["bit5"] = [i for i, r in enumerate(cells) if r["P"] == 5]
        for state in TARGETS:
            subsets[state+"|bits_ge4"] = [i for i, r in enumerate(cells) if r["state"] == state and r["P"] >= 4]
    scores = []
    for subset, indices in subsets.items():
        for c in CAPS:
            cr = [cells[i] for i in indices if cells[i]["cap"] == c]
            scores.append(dict(subset=subset, cap=c, n=len(cr),
                delivered=sorted({r["delivered"] for r in cr}), baseline=panel["development_baseline"][c],
                mae={m: float(np.mean([r["absolute_errors"][m] for r in cr])) for m in cr[0]["predictions"]}))
    return cells, scores


def build(root=ROOT, n_bootstrap=1000):
    panels, hashes = load_panels(root)
    results = {}
    for name, panel in panels.items():
        print(f"Fitting {name}: {len(panel['dev'])//3} development / {len(panel['test'])//3} test cells", flush=True)
        fits = {variant: fit_kumar(panel["dev"], panel["adapted"], variant == "per_capability", n_bootstrap)
                for variant in ("shared", "per_capability")}
        cells, scores = evaluate(panel, fits)
        results[name] = {**panel, "fits": fits, "cells": cells, "scores": scores,
            "negative_cells": {split: {c: sum(r["y"] < 0 for r in panel[split] if r["cap"] == c) for c in CAPS}
                               for split in ("dev", "test")}}
    return dict(id=ID, status=STATUS, source=SOURCE, input_sha256=hashes,
        protocol={"form": "delta = C_c exp(-P/gamma_post) D^gamma_D / N^gamma_N; grouped adapted form multiplies g^gamma_g",
            "units": "P is literal bit width; N0 transformer matrices excluding embeddings/head; D0=step*2097152",
            "estimator": "Unweighted nonlinear least squares on signed delta in linear space, six deterministic starts; shared slopes with free positive C_c, then capability-specific slopes. No negative cell dropped or clipped.",
            "signs": "C_c and gamma_post positive. gamma_D and gamma_N are estimated freely in [-10,10] to test the sign, rather than impose Kumar's positive fitted finding. Functional form unchanged; this is a refit, not the paper's original coefficients or original fitting loss.",
            "bootstrap": "1000 development-state resamples by default, preserving all configurations and capabilities jointly; two starts per draw. Rank-deficient draws retained and flagged; intervals are descriptive with only 6 or 9 development states.",
            "positive_form": "A positive multiplicative form cannot predict observed negative deltas; their errors remain in every MAE and SSE.",
            "provenance_correction": "The Sec 5.2 numbers 0.05/0.07/0.15 versus 0.23/0.19/0.96 are V49 pruning at 6.9B/112k, not PTQ. V48 is secondary-probe transfer, V47 is distillation; per-channel new-state PTQ is V49 protocol A/B. Those pruning numbers are not used as quantization comparators.",
            "channel_delivery": "Per-bit development median with V49's prescribed 5-bit average of 4/6-bit predictions; source regression and all original alternatives are also replayed. Kumar predicts directly at P=5.",
            "group_delivery": "Table 2/final_prediction_table: seen states interpolate math/code and use median for QA; new state uses median. This delivered mapping was retrospective, not development-selected."},
        panels=results)


def render(summary):
    md = [f"# {ID}", "", STATUS, "", f"Kumar et al., ICLR 2025: {SOURCE}", "", *summary["protocol"].values(), ""]
    for name, panel in summary["panels"].items():
        md += [f"## {name}: {panel['form']}", "", panel["baseline_selection"], "",
            f"Signed negative cells: {panel['negative_cells']}", "",
            "| Evaluation | Capability | Kumar shared | Kumar per capability | Delivered | Development baseline | Baseline name |",
            "|---|---|---:|---:|---:|---:|---|"]
        for r in panel["scores"]:
            values = [f"{r['mae'][m]:.3f}" for m in ("shared", "per_capability", "delivered", "baseline")]
            md.append("| "+" | ".join([r["subset"], r["cap"], *values, r["baseline"]])+" |")
        md += ["", "| Variant | Capability | Parameters (all capabilities) | gamma D [95%] | gamma N [95%] | gamma D > 0 | Bootstrap P(gamma D > 0) |",
               "|---|---|---:|---|---|---|---:|"]
        for variant, fit in panel["fits"].items():
            for cap, e in fit["exponents"].items():
                interval = lambda k: f"{e[k]:.4f} ["+", ".join(f"{v:.4f}" for v in e[k+"_ci95"])+"]" if e[k+"_ci95"] else f"{e[k]:.4f}"
                md.append(f"| {variant} | {cap} | {fit['n_parameters']} | {interval('gamma_D')} | {interval('gamma_N')} | " +
                          f"{e['gamma_D_positive']} | {e['bootstrap_fraction_gamma_D_positive']} |")
            md.append(f"\n{variant}: {len(fit['bootstrap']['failed_indices'])} nonconverged, " +
                f"{len(fit['bootstrap']['rank_deficient_indices'])} rank-deficient, {len(fit['bootstrap']['boundary_indices'])} bound-touching bootstrap draws; full-fit active bounds {fit['active_bounds']}.\n")

    capabilities = {"math": "Mathematics", "code": "Code", "qa": "Question answering"}
    body = []
    panels = summary["panels"]
    for title, name, subset in (
        ("Grouped quantization, unseen group sizes of development states", "grouped", "development_state_boundary"),
        ("Grouped quantization, a new state", "grouped", "new_state"),
        ("Per-channel quantization, new states", "channel_A", "all"),
    ):
        body += [r"\midrule", r"\multicolumn{5}{l}{\emph{" + title + r"}} \\"]
        names = ("channel_A", "channel_B") if name == "channel_A" else (name,)
        for name in names:
            if name.startswith("channel_"):
                cohort = "Nine development states" if name == "channel_A" else "Six development states"
                body.append(r"\multicolumn{5}{l}{" + cohort + r"} \\")
            for row in panels[name]["scores"]:
                if row["subset"] == subset:
                    values = [f"{row['mae'][method]:.3f}" for method in
                              ("shared", "per_capability", "delivered", "baseline")]
                    body.append(" & ".join([capabilities[row["cap"]], *values]) + r" \\")
    body += [r"\midrule", r"\multicolumn{5}{l}{\emph{Fitted exponents (shared fit)}} \\",
             r"Quantization & Exponent & Estimate & 95\% interval & Bootstrap share with $\gamma_D>0$ \\"]
    for name, label in (("grouped", "Grouped"), ("channel_A", "Per-channel, nine states"),
                        ("channel_B", "Per-channel, six states")):
        exponents = panels[name]["fits"]["shared"]["exponents"][CAPS[0]]
        for exponent, symbol in (("gamma_D", r"$\gamma_D$"), ("gamma_N", r"$\gamma_N$")):
            interval = exponents[exponent + "_ci95"]
            interval = "[" + ", ".join(f"{value:.3f}" for value in interval) + "]" if interval else "--"
            share = exponents["bootstrap_fraction_gamma_D_positive"] if exponent == "gamma_D" else None
            body.append(" & ".join([label, symbol, f"{exponents[exponent]:.3f}", interval,
                                    "--" if share is None else f"{share:.3f}"]) + r" \\")
    grouped, channel = panels["grouped"]["fits"], panels["channel_A"]["fits"]
    n_resamples = grouped["shared"]["bootstrap"]["n_resamples"]
    caption = (
        "Prediction errors for the Kumar et al. baseline, the delivered relation and the strongest development baseline. "
        "Entries are mean absolute errors in nats per token on identical test cells. "
        r"The grouped variant multiplies the published form by $g^{\gamma_g}$, an adapted baseline with one extra "
        "parameter in the shared fit and one extra parameter per capability in the separate fits. "
        f"Shared and per-capability fits have {grouped['shared']['n_parameters']} and "
        f"{grouped['per_capability']['n_parameters']} parameters for grouped quantization, and "
        f"{channel['shared']['n_parameters']} and {channel['per_capability']['n_parameters']} for per-channel quantization, respectively. "
        "Signed negative responses are kept although a positive form cannot fit them. "
        "Per-channel results aggregate all test states and bit widths, separately for nine and six development states. "
        f"Shared exponents have descriptive 95\\% percentile intervals from {n_resamples:,} development-state bootstrap resamples, "
        r"including rank-deficient resamples; the share with $\gamma_D>0$ is shown on each token-exponent row. "
        "Retrospective analysis of recorded results (rebuttal round)."
    )
    # Paragraph columns keep the full-word headings within the paper's text width.
    widths = (22, 19, 20, 19, 20)
    columns = "".join(r">{\raggedright\arraybackslash}p{\dimexpr " + f"{w / 100:.3f}" +
                      r"\textwidth-" + f"{8 * w / 100:.3f}" + r"\tabcolsep\relax}" for w in widths)
    tex = ("% Generated by analysis/r1_kumar_baseline.py; do not edit.\n"
           "\\begin{table}[tb]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n"
           "\\caption{\\rev{" + caption + "}}\n\\label{tab:r1-kumar}\n"
           "\\begin{tabular}{@{}" + columns + "@{}}\n\\toprule\n" +
           r"Capability & Kumar et al., shared exponents & Kumar et al., per-capability exponents & Delivered relation & Strongest development baseline \\" +
           "\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    tex = house_style(table_layout(tex))
    # These rebuttal tables require full capability names after the shared styling pass.
    tex = tex.replace("Math &", "Mathematics &").replace("QA &", "Question answering &")
    return "\n".join(md)+"\n", tex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--table", action="store_true", help="Render the recorded summary without refitting")
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    out = ROOT / "results" / ID
    summary = json.loads((out / "summary.json").read_text()) if args.table else build(n_bootstrap=args.bootstrap)
    md, tex = render(summary)
    out.mkdir(parents=True, exist_ok=True)
    if not args.table:
        (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (out / "summary.md").write_text(md)
    (ROOT / "paper/paper/tables" / (ID+".tex")).write_text(tex)
    print("Wrote", out)


if __name__ == "__main__":
    main()
