#!/usr/bin/env python3
"""Input ablation, retrospective (rebuttal round, 2026-11-07). Recorded data, CPU only.

Run: python -B analysis/r1_input_ablation.py
No measurement, registration, or existing analysis output is changed.
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
from analysis import a9_measurement_efficiency as a9
from analysis import a11_score as a11
from analysis import a18_second_family as a18
from analysis import v53_prune_dev as p53
from analysis.paper_table_layout import house_style, table_layout

ID = "r1-input-ablation"
STATUS = "retrospective (rebuttal round, 2026-11-07)"
CAPS = ("math", "code", "qa")
FEATURES = {"shape": (0,), "loss": (0, 2), "size_tokens": (0, 1, 3), "delivered": (0, 1, 2, 3)}
FEATURE_LABELS = {"shape": "1", "loss": "1, L0", "size_tokens": "1, logN, logD",
                  "delivered": "1, logN, L0, logD"}


def read(root, relative, hashes):
    path = root / relative
    raw = path.read_bytes()
    hashes[str(relative)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def load_panels(root=ROOT):
    """Read the exact registered grid IDs; keep new-item responses out of all fits."""
    hashes = {}
    base = Path("results/a11-efficiency-confirmation")
    frozen = read(root, base / "predictions.json", hashes)
    plan = read(root, base / "plan.json", hashes)
    dev = frozen["development_rows"]
    assert sorted(r["id"] for r in dev) == plan["development"]["full_row_ids"]
    reduced = set(plan["development"]["reduced_row_ids"])
    tables = {t["state"]: read(root, base / "measurements" / t["state"].replace("@", "--") /
                              "prune_losses.json", hashes) for t in plan["targets"]}
    replay = a11.score_tables(plan, frozen, tables)
    targets = {t["state"]: t for t in plan["targets"]}
    test = []
    for r in replay["cells"]:
        t = targets[r["state"]]
        test.append(dict(state=r["state"], cap=r["capability"], d=r["density"],
                         phi_raw=p53.raw_features(t["N0"], t["D0"], r["dense_loss"]),
                         outcomes={"probes": r["observed_delta_L"]},
                         recorded={"half": r["predicted_delta_L"]["power_18"],
                                   "full": r["predicted_delta_L"]["power_36"]}))
    panels = {"pythia_magnitude": dict(estimator="training_ols", grids={
        "half": [r for r in dev if r["id"] in reduced], "full": dev}, test=test)}
    for name, directory in (("olmo2_magnitude", "a18-second-family"), ("pythia_wanda", "a19-wanda-efficiency")):
        base = Path("results") / directory
        fz = read(root, base / "freeze.json", hashes)
        recs = {tag: read(root, base / "measurements" / (tag.replace("@", "__") + ".json"), hashes)
                for tag in [*fz["development_states"], *fz["test_states"]]}
        # D0 is recorded on both panels, including the token-labelled OLMo stages.
        assert all(r["D0"] is not None and r["D0"] > 0 for r in recs.values())
        dev = [{**r, "state": r["tag"]} for r in a18.dev_rows(
            {tag: recs[tag] for tag in fz["development_states"]})]
        test = []
        for tag in fz["test_states"]:
            rec = recs[tag]
            dense = rec["measurements"]["1.0"]
            for cap in CAPS:
                # Like A18.score, use the probe L0 in predictions for BOTH evaluations.
                raw = p53.raw_features(rec["N0_block_matrices"], rec["D0"], dense["probes"][cap]["mean_loss"])
                for d in fz["test_densities"]:
                    test.append(dict(state=tag, cap=cap, d=d, phi_raw=raw,
                        outcomes={b: rec["measurements"][str(d)][b][cap]["mean_loss"] - dense[b][cap]["mean_loss"]
                                  for b in ("probes", "new")},
                        recorded={label: a18.predict("power", g["fits"][cap]["power"],
                            p53.standardize(raw, g["standardization"]), d)
                            for label, g in (("half", fz["grids"]["reduced"]), ("full", fz["grids"]["full"]))}))
        panels[name] = dict(estimator="loso_ridge", grids={
            "half": [r for r in dev if r["d"] in a18.REDUCED], "full": dev}, test=test)
        for label, old in (("half", "reduced"), ("full", "full")):
            assert len(panels[name]["grids"][label]) // 3 == fz["grids"][old]["n_configurations"]
    for p in panels.values():
        assert not ({r["state"] for r in p["grids"]["full"]} & {r["state"] for r in p["test"]})
    return panels, hashes


def fit_power(rows, cap, stats, feature, estimator):
    cr = [r for r in rows if r["cap"] == cap]
    z = p53.standardize([r["phi_raw"] for r in cr], stats)[:, FEATURES[feature]]
    y, d = np.array([r["y"] for r in cr]), np.array([r["d"] for r in cr])
    best = None
    grid = a9.GAMMAS if estimator == "training_ols" else p53.GAMMA_GRID
    for gamma in grid:
        design = lambda zz, dd, g=gamma: zz * p53.shape(dd, g)[:, None]
        x = design(z, d)
        if estimator == "training_ols":
            beta = np.linalg.lstsq(x, y, rcond=1e-10)[0]
            criterion, lam = float(np.sum((x @ beta-y)**2)), 0.0
        else:
            lam, criterion = a18.loso_lambda(cr, z, y, d, design)
            beta = a18.ridge(x, y, lam)
        if best is None or criterion < best["criterion"]:
            best = dict(beta=beta.tolist(), gamma=float(gamma), ridge=lam, criterion=criterion,
                        n_parameters=len(FEATURES[feature])+1, rank=int(np.linalg.matrix_rank(x)))
    return best


def build(root=ROOT):
    panels, hashes = load_panels(root)
    results, cells, fits = [], [], {}
    for panel, p in panels.items():
        print(f"Fitting {panel}: {p['estimator']}", flush=True)
        fits[panel] = {}
        for grid, dev in p["grids"].items():
            stats = a9.standardizer(dev) if p["estimator"] == "training_ols" else p53.zstats(dev)
            fits[panel][grid] = dict(standardization=stats, models={}, development_rows=dev)
            for feature in FEATURES:
                models = {c: fit_power(dev, c, stats, feature, p["estimator"]) for c in CAPS}
                fits[panel][grid]["models"][feature] = models
                for r in p["test"]:
                    fit = models[r["cap"]]
                    z = p53.standardize(r["phi_raw"], stats)[list(FEATURES[feature])]
                    pred = float(z @ fit["beta"] * p53.shape(r["d"], fit["gamma"]))
                    if feature == "delivered":
                        np.testing.assert_allclose(pred, r["recorded"][grid], rtol=1e-9, atol=1e-9)
                    for evaluation, y in r["outcomes"].items():
                        cells.append(dict(panel=panel, grid=grid, feature=feature, evaluation=evaluation,
                                          state=r["state"], cap=r["cap"], d=r["d"], observed=y,
                                          predicted=pred, abs_error=abs(pred-y)))
                for evaluation in p["test"][0]["outcomes"]:
                    for cap in CAPS:
                        cr = [r for r in cells if (r["panel"], r["grid"], r["feature"], r["evaluation"], r["cap"]) ==
                              (panel, grid, feature, evaluation, cap)]
                        results.append(dict(panel=panel, grid=grid, feature=feature, evaluation=evaluation, cap=cap,
                            n_test=len(cr), n_development=len(dev)//3,
                            mae=float(np.mean([r["abs_error"] for r in cr]))))
    return dict(status=STATUS, id=ID, features=FEATURE_LABELS,
        protocol={"shape": "((1-d)/0.3)^gamma; signed responses and unrestricted amplitude coefficients",
            "pythia_magnitude": "A11 registered 18/36 row IDs, A9 exponent grid and unregularized training SSE selection",
            "olmo2_magnitude_and_wanda": "A18 exponent/ridge grids and LOSO MAE selection; standardized once per development grid, exactly as A18",
            "D0": "Known on all three panels. No missing-token fallback was needed.",
            "new_items": "Same predictions using probe L0; new-item outcomes use their own dense anchors, as A18.score",
            "gamma_grid": p53.GAMMA_GRID.tolist(), "ridge_grid": list(a18.LAMBDA_GRID),
            "delivered_replay": "Every refitted full-input test prediction agrees with its recorded coefficient replay within 1e-9"},
        input_sha256=hashes, fits=fits, scores=results, cells=cells)


def render(summary):
    lines = [
        f"# {ID}", "", STATUS, "", *[str(v) for k, v in summary["protocol"].items() if not k.endswith("grid")], "",
        "| Panel | Evaluation | Grid | Capability | Shape | + L0 | + logN, logD | Delivered |",
        "|---|---|---|---|---:|---:|---:|---:|"]
    for panel in summary["fits"]:
        keys = list(dict.fromkeys((r["evaluation"], r["grid"], r["cap"]) for r in summary["scores"] if r["panel"] == panel))
        for evaluation, grid, cap in keys:
            scores = {r["feature"]: r for r in summary["scores"] if
                      (r["panel"], r["evaluation"], r["grid"], r["cap"]) == (panel, evaluation, grid, cap)}
            values = [f"{scores[f]['mae']:.3f}" for f in FEATURES]
            budget = str(scores["shape"]["n_development"])
            lines.append("| " + " | ".join([panel, evaluation, grid+f" ({budget})", cap, *values]) + " |")

    capabilities = {"math": "Mathematics", "code": "Code", "qa": "Question answering"}
    body = []
    for panel, title in (("pythia_magnitude", "Pythia, magnitude pruning"),
                         ("olmo2_magnitude", "OLMo-2, magnitude pruning"),
                         ("pythia_wanda", "Pythia, Wanda pruning")):
        body += [r"\midrule", r"\multicolumn{7}{l}{\emph{" + title + r"}} \\"]
        for evaluation, label in (("probes", "Probes"), ("new", "New items")):
            for grid in ("half", "full"):
                for cap in CAPS:
                    scores = {r["feature"]: r for r in summary["scores"] if
                              (r["panel"], r["evaluation"], r["grid"], r["cap"]) == (panel, evaluation, grid, cap)}
                    if not scores:
                        continue
                    values = [f"{scores[feature]['mae']:.3f}" for feature in FEATURES]
                    body.append(" & ".join([label, grid.capitalize(), capabilities[cap], *values]) + r" \\")
    caption = (
        "Prediction errors for ablations of the amplitude inputs on half and full development grids. "
        "Entries are mean absolute errors in nats per token on identical test cells. "
        "The columns use shape alone, shape and dense loss, shape with parameter and pretraining-token counts, "
        "and all inputs in the delivered relation. Each fit uses the original panel's estimator and exponent grid. "
        "New items use the same predictions as probes and their own dense-loss anchors. "
        "Retrospective analysis of recorded results (rebuttal round)."
    )
    widths = (12, 7, 21, 10, 15, 19, 16)
    columns = "".join(r">{\raggedright\arraybackslash}p{\dimexpr " + f"{w / 100:.3f}" +
                      r"\textwidth-" + f"{12 * w / 100:.3f}" + r"\tabcolsep\relax}" for w in widths)
    tex = ("% Generated by analysis/r1_input_ablation.py; do not edit.\n"
           "\\begin{table}[tb]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n"
           "\\caption{\\rev{" + caption + "}}\n\\label{tab:r1-ablation}\n"
           "\\begin{tabular}{@{}" + columns + "@{}}\n\\toprule\n" +
           r"Evaluation & Grid & Capability & Shape only & Shape and dense loss & Shape, parameters and tokens & Delivered (all inputs) \\" +
           "\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    tex = house_style(table_layout(tex))
    tex = tex.replace("Math &", "Mathematics &").replace("QA &", "Question answering &")
    return "\n".join(lines)+"\n", tex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", action="store_true", help="Render the recorded summary without refitting")
    args = parser.parse_args()
    out = ROOT / "results" / ID
    summary = json.loads((out / "summary.json").read_text()) if args.table else build()
    md, tex = render(summary)
    out.mkdir(parents=True, exist_ok=True)
    if not args.table:
        (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (out / "summary.md").write_text(md)
    (ROOT / "paper/paper/tables" / (ID+".tex")).write_text(tex)
    print(md)


if __name__ == "__main__":
    main()
