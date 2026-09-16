#!/usr/bin/env python3
"""CPU-only v64 feasibility correction of the unchanged v60 selection protocol.

    python -B analysis/v64_selection_feasible.py

All artifacts, including paper tables/figures and the code mirror, are created
exclusively beneath results/v64-selection-feasible/. An optional output root
must also be within that directory. V60 inputs and results are read-only.
Data loading, laws, LOSO refits, budgets and candidate policies are copied from
v60. Empty policy candidate sets return INFEASIBLE, with null scores/metrics.
The no-clear-winner annotation is a retrospective heuristic, kept unchanged.
Imports the delivered v53/v55/v39 fitting functions unchanged. Quantization
reuses v36's exact covariate/design/prediction paths used by v38/v49, extending
the categorical vocabulary to measured 5-bit cells; see fit_quant for the
explicit minimum-norm OLS treatment of underidentified 5-bit folds.

Student outcomes reused as alternatives must also be held out: distillation
folds purge the source AND every eligible same-stage smaller student. Merely
holding out the source would train on the very student outcomes being scored.
"""
from __future__ import annotations

if __package__:
    from .paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
else:
    from paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp

try:
    from .paper_table_text import proofread_table
except ImportError:  # Direct scripts and file-based imports.
    try:
        from analysis.paper_table_text import proofread_table
    except ImportError:
        from paper_table_text import proofread_table


import argparse
from contextlib import contextmanager
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAPS = ("math", "code", "qa")
OBJECTIVES = (*CAPS, "multi")
BITS = (8, 6, 5, 4, 3)
BUDGETS = [i / 100 for i in range(20, 101, 5)]
METHODS = ("prune", "quant", "distill", "dense")
POLICIES = ("MAP", "ORACLE", "prune-only", "quant-only", "distill-only", "CHEAPEST")
LAWS = ("prune_power", "quant_channel", "quant_group", "distill_linear", "dense")
OUT_ROOT = ROOT / "results/v64-selection-feasible"
OUTPUTS = ("summary.json", "summary.md", "paper/paper/tables/selection_feasible.tex",
           "paper/paper/tables/candidate_coverage.tex", "paper/paper/figs/selection_feasible.pdf",
           "paper/paper/figs/selection_feasible.png", "paper/code/analysis/v64_selection_feasible.py")
ENDPOINT = (
    "Every candidate endpoint is the absolute measured/predicted loss of the deployed model. "
    "For pruning/quantization, predicted loss is source dense (the measurement arm's own anchor) "
    "+ predicted response; measured loss is that configuration's recorded absolute loss. "
    "For KD, predicted loss is student dense + predicted student post-training change, and "
    "measured loss is student dense + measured student post-training change (= post_training). "
    "The student change is never added to the source dense. This is v60's endpoint definition, "
    "kept unchanged here. The multi objective then takes max_c(deployed loss_c - source dense_c)."
)


class Audit:
    def __init__(self):
        self.hashes = {}
        self.gaps = []

    def read(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        print(f"READ {path} sha256={sha}", flush=True)
        if label in self.hashes and self.hashes[label] != sha:
            raise RuntimeError(f"Input changed during run: {path}")
        self.hashes[label] = sha
        return raw

    def json(self, path):
        return json.loads(self.read(path))

    def gap(self, message):
        if message not in self.gaps:
            self.gaps.append(message)
            print(f"DATA GAP / LIMITATION: {message}", flush=True)

    def verify(self):
        for path in sorted(self.hashes):
            self.read(path)


def imports(audit):
    # Hash every local source dependency before import; no upstream main runs.
    names = ("prediction_audit", "v36_pythia_controlled_fit", "v38_prospective_step",
             "v49_p1v2", "v53_prune_dev", "v55_quant_group_fit",
             "v39_distill_controlled", "plot_fig1_responses", "v60_selection_maps",
             "v64_selection_feasible")
    init = ROOT / "analysis/__init__.py"
    if init.is_file():
        audit.read(init)
    for name in names:
        audit.read(f"analysis/{name}.py")
    sys.path.insert(0, str(ROOT))
    # v38/v49 are provenance only: their quantization implementation is v36.
    return tuple(importlib.import_module(f"analysis.{name}") for name in
                 ("v53_prune_dev", "v55_quant_group_fit", "v36_pythia_controlled_fit",
                  "v39_distill_controlled", "plot_fig1_responses"))


def state_info(tag, prune):
    size, step = prune.parse_tag(tag)
    return {"tag": tag, "size": size, "step": step, "N0": prune.matrix_n0(size),
            "D0": step * prune.TOKENS_PER_STEP}


def config(method, law, key, ratio, anchor, actual, target, **extra):
    return {"id": f"{method}:{key}", "method": method, "law": law, "config": key,
            "r": float(ratio), "anchor": anchor, "actual": actual,
            "target_state": target, **extra}


def load_data(audit, prune, group, distill):
    locations = {"prune": ("v6-capability-geometry", "prune_losses.json"),
                 "channel": ("v10-quantization", "quant_losses.json"),
                 "group": ("v54-quant-group", "quant_group_losses.json")}
    tables = {}
    for kind, (directory, filename) in locations.items():
        tables[kind] = {}
        for path in sorted((ROOT / "results" / directory).glob(f"pythia-*/{filename}")):
            tag = path.parent.name.replace("--", "@")
            if tag.startswith("pythia-2.8b"):
                audit.gap(f"Explicitly excluded {tag}: pythia-2.8b* is outside this panel.")
                continue
            state_info(tag, prune)  # Reject unknown architectures/stages.
            tables[kind][tag] = audit.json(path)
    paired = set(tables["prune"]) & set(tables["channel"])
    for tag in sorted(set(tables["prune"]) ^ set(tables["channel"])):
        missing = "quantization" if tag not in tables["channel"] else "pruning"
        audit.gap(f"Excluded {tag} from training and selection: no paired {missing} file.")
    expected = {f"pythia-{size}@step{step}" for size in ("160m", "410m", "1.4b")
                for step in (16000, 64000, 143000)}
    expected |= {f"pythia-{size}@step{step}" for size, steps in
                 (("1b", (32000, 96000, 112000)), ("6.9b", (32000, 112000)),
                  ("410m", (96000,)), ("160m", (96000,)), ("1.4b", (96000,)))
                 for step in steps}
    for tag in sorted(expected - paired):
        audit.gap(f"Requested state {tag} lacks a paired pruning/quantization measurement.")
    states = sorted((state_info(t, prune) for t in paired), key=lambda s: (s["N0"], s["step"]))
    if len(states) < 2:
        raise ValueError("Need at least two paired states")
    prows, qrows, grows = [], [], []
    for state in states:
        tag = state["tag"]
        pt, qt = tables["prune"][tag], tables["channel"][tag]
        dense = prune.checked_losses(pt["1.0"], f"{tag}: pruning dense")
        qdense = prune.checked_losses(qt["dense"], f"{tag}: channel dense")
        state.update(dense=dense, anchors={"prune": dense, "quant_channel": qdense}, configs=[])
        for key, value in sorted(pt.items()):
            if key == "1.0" or key.startswith("_"):
                continue
            d = float(key)
            if not 0 < d < 1:
                raise ValueError(f"Invalid density {tag}/{key}")
            actual = prune.checked_losses(value, f"{tag}/{key}")
            state["configs"].append(config("prune", "prune_power", f"d{d:g}", d,
                                             dense, actual, tag, d=d))
            # Exact v53 training-domain rule; any lower measured densities
            # remain selectable, with extrapolation explicitly disclosed.
            if d < .55:
                audit.gap(f"{tag}: d={d:g} is selectable but below v53's d>=0.55 training domain.")
                continue
            for cap in CAPS:
                prows.append({"source": tag, "cap": cap, "d": d,
                              "phi_raw": prune.raw_features(state["N0"], state["D0"], dense[cap]),
                              "y": actual[cap] - dense[cap]})
        measured_d = {q["d"] for q in state["configs"] if q["method"] == "prune"}
        state["measured_densities"] = sorted(measured_d, reverse=True)
        missing_d = [d for d in prune.ANCHORS if d not in measured_d]
        if missing_d:
            audit.gap(f"{tag}: missing v53 pruning anchors {missing_d}; selection uses only "
                      f"measured densities {state['measured_densities']}.")
        missing_bits = [b for b in BITS if str(b) not in qt]
        if missing_bits:
            audit.gap(f"{tag}: no measured per-channel bits {missing_bits}; these configs are not candidates.")
        for b in BITS:
            if str(b) not in qt:
                continue
            actual = prune.checked_losses(qt[str(b)], f"{tag}/channel/b{b}")
            state["configs"].append(config("quant", "quant_channel", f"channel_b{b}", b / 16,
                                             qdense, actual, tag, bit=b))
            for cap in CAPS:
                qrows.append({"row_id": f"{tag}|{cap}|{b}", "cell": tag, "arm": "quantization",
                              "capability": cap, "config": b, "N0": state["N0"], "D0": state["D0"],
                              "L0": qdense[cap], "observed": actual[cap] - qdense[cap]})
        gt = tables["group"].get(tag)
        if gt is None:
            audit.gap(f"{tag}: no grouped RTN measurements; no grouped configs are synthesized.")
        else:
            gdense = prune.checked_losses(gt["dense"], f"{tag}: group dense")
            state["anchors"]["quant_group"] = gdense
            gstate = {**state, "L0": gdense}
            for key in sorted(gt):
                if key == "dense" or key.startswith("_"):
                    continue
                group.coordinates(key)  # Validate with the delivered parser.
                bits, g = map(int, key[1:].split("_g"))
                actual = prune.checked_losses(gt[key], f"{tag}/{key}")
                state["configs"].append(config("quant", "quant_group", key, (bits + 16 / g) / 16,
                                                 gdense, actual, tag, bit=bits, group_size=g))
                if tag in group.DEV_TAGS and key in group.DEV_CONFIGS:
                    for cap in CAPS:
                        grows.append({"state": tag, "config": key, "capability": cap,
                                      "phi_raw": group.raw_features(gstate, cap),
                                      "dL": actual[cap] - gdense[cap]})
            planned = {f"b{b}_g{g}" for b in (3, 4, 5) for g in (64, 128, 256)}
            absent = sorted(planned - set(gt))
            if absent:
                audit.gap(f"{tag}: grouped grid is partial; missing {absent}.")
        state["configs"].append(config("dense", "dense", "source", 1, dense, dense, tag))
    for tag in sorted(set(tables["group"]) - paired):
        audit.gap(f"Grouped file {tag} has no paired panel state; excluded from this selection analysis.")
    students, drows, reproduction = {}, [], {}
    summary_path = ROOT / "results/v39-distill-controlled/summary.json"
    if not summary_path.is_file():
        audit.gap("v39 summary.json missing: cannot establish its raw student cohort; use pruning + quantization only.")
        reference = None
    else:
        reference = audit.json(summary_path)
        panel = reference["panel"]
        if panel["run"] != distill.RUN:
            raise ValueError("v39 summary recipe differs from the delivered v39 code")
        for size in panel["sizes"]:
            for step in panel["steps"]:
                tag = f"pythia-{size}@step{step}"
                path = ROOT / "results/v12-distill" / tag.replace("@", "--") / panel["run"] / "eval.json"
                if not path.is_file():
                    audit.gap(f"Cannot recover v39 student {tag}: missing {path.relative_to(ROOT)}.")
                    continue
                ev = audit.json(path)
                if not all(k in ev for k in ("dense", "post_training")):
                    audit.gap(f"Cannot recover raw delta for {tag}: dense or post_training is absent.")
                    continue
                if ev["training_mode"] != "lora" or ev["student"] != tag:
                    raise ValueError(f"Student identity/recipe mismatch: {path}")
                dense = prune.checked_losses(ev["dense"], f"{tag}: student dense")
                actual = prune.checked_losses(ev["post_training"], f"{tag}: student post_training")
                delta = {c: actual[c] - dense[c] for c in CAPS}
                if "delta" in ev:
                    np.testing.assert_allclose([delta[c] for c in CAPS], [ev["delta"][c] for c in CAPS],
                                               rtol=0, atol=1e-12)
                student = {**state_info(tag, prune), "dense": dense, "actual": actual, "delta": delta,
                           "path": str(path.relative_to(ROOT)), "teacher": ev["teacher"]}
                students[tag] = student
                for cap in CAPS:
                    drows.append({**state_info(tag, prune), "cap": cap, "L0": dense[cap], "delta": delta[cap]})
        if len(students) != panel["n_cells"]:
            audit.gap(f"Recovered only {len(students)}/{panel['n_cells']} v39 students; disable distillation "
                      "rather than change the delivered cohort. Use pruning + quantization only.")
            students, drows = {}, []
        else:
            for cap in CAPS:
                cr = [r for r in drows if r["cap"] == cap]
                reproduction[cap] = {}
                for axis in ("size", "step"):
                    got = distill._cv(cr, axis)
                    ref = reference["by_cap"][cap][axis]
                    for key in ("mae_noD0", "mae_D0", "mae_baseline_mean", "mae_baseline_median"):
                        np.testing.assert_allclose(got[key], ref[key], rtol=1e-10, atol=1e-12)
                    reproduction[cap][axis] = got
    used_students = set()
    for state in states:
        eligible = [s for s in students.values() if s["step"] == state["step"] and s["N0"] < state["N0"]]
        for student in sorted(eligible, key=lambda s: s["N0"]):
            used_students.add(student["tag"])
            state["configs"].append(config("distill", "distill_linear", student["tag"],
                                             student["N0"] / state["N0"], student["dense"],
                                             student["actual"], student["tag"]))
        if not eligible:
            audit.gap(f"{state['tag']}: no recovered smaller same-stage v39 student; distill-only is INFEASIBLE at every budget.")
        if state["tag"] in students:
            state["anchors"]["student"] = students[state["tag"]]["dense"]
        offsets = {name: {c: anchor[c] - state["dense"][c] for c in CAPS}
                   for name, anchor in state["anchors"].items() if name != "prune"}
        state["dense_anchor_offsets"] = offsets
        state["counts"] = {law: sum(q["law"] == law for q in state["configs"]) for law in LAWS}
    if students:
        audit.gap("Distillation measurements use the fixed external teacher gpt-5.6-luna, full/600, "
                  "two epochs, LoRA, seed 0. They measure smaller-student alternatives, not transfer "
                  "from each plotted source; no other student sizes/stages or recipes are inferred. "
                  f"{len(students)} raw students supply fits; {len(used_students)} distinct students "
                  "supply selectable alternatives (shared across source choices, not independent runs).")
    max_offsets = {c: max(abs(v[c]) for s in states for v in s["dense_anchor_offsets"].values()) for c in CAPS}
    audit.gap("Dense references are not identical across measurement arms; maximum absolute offsets "
              f"from pruning dense (nats) are {max_offsets}. Fit signed changes against each file's own "
              "dense anchor and retain recorded absolute losses. Dense source and multi-capability "
              "reference use pruning key '1.0'; no post-hoc recentering. These offsets limit tiny-gap comparisons.")
    return states, prows, qrows, grows, drows, students, reproduction


@contextmanager
def quant_vocabulary(v36):
    """Extend only the imported module's in-memory vocabulary; restore on exit."""
    old = v36.CONFIGS
    v36.CONFIGS = {**old, "quantization": BITS}
    try:
        yield
    finally:
        v36.CONFIGS = old


def fit_quant(rows, v36):
    """v36 OLS with the requested fifth indicator; no ridge or interpolation.

    Full-rank cases execute fit_direct unchanged. Underidentified cases reuse
    its exact input, covariate, design and np.linalg.lstsq paths, retaining the
    returned minimum-norm solution instead of its rank-deficiency exception.
    This exception is necessary for 5-bit: four cells total, three in its LOSO
    folds. It is recorded as a limitation, never described as an identified fit.
    """
    fields = ("N0", "L0", "D0")
    with quant_vocabulary(v36):
        inputs = [v36.basic_input(r, input_fields=fields) for r in rows]
        raw = v36.covariates(inputs, input_fields=fields)
        center, scale = raw.mean(axis=0), raw.std(axis=0)
        if np.any(scale <= 0):
            raise ValueError("Training covariate has no variation")
        x = v36.design_matrix(inputs, "quantization", True, center, scale, input_fields=fields)
        y = np.array([v36.audit.finite(r["observed"]) for r in rows])
        coefficients, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
        if rank == x.shape[1]:
            fit = v36.fit_direct(rows, input_fields=fields)
            np.testing.assert_allclose(coefficients, fit["coefficients"], rtol=0, atol=0)
        else:
            fit = {"arm": "quantization", "with_d0": True, "input_fields": list(fields),
                   "center": center.tolist(), "scale": scale.tolist(),
                   "coefficients": coefficients.tolist()}
        fit.update(rank=int(rank), n_parameters=x.shape[1], n_observations=len(rows),
                   singular_values=singular.tolist(), solver="numpy.linalg.lstsq(rcond=None)",
                   config_order=list(BITS), underidentified=bool(rank < x.shape[1]),
                   bit_diagnostics={str(b): {"n": sum(r["config"] == b for r in rows),
                       "rank": int(np.linalg.matrix_rank(x[np.array([r["config"] == b for r in rows])]))}
                       for b in BITS})
        return fit


def fit_fold(source, prows, qrows, grows, drows, modules):
    prune, group, v36, distill, _ = modules
    tag = source["tag"]
    ptrain = [r for r in prows if r["source"] != tag]
    qtrain = [r for r in qrows if r["cell"] != tag]
    ps, pf = prune.fit_all(ptrain)
    qf = {c: fit_quant([r for r in qtrain if r["capability"] == c], v36) for c in CAPS}
    result = {"held_out": tag, "prune": {"standardization": ps, "models": pf,
              "train_states": sorted({r["source"] for r in ptrain}), "n_train_configs": len(ptrain) // 3},
              "quant_channel": {"models": qf, "train_states": sorted({r["cell"] for r in qtrain}),
                                "n_train_configs": len(qtrain) // 3}}
    if any(q["law"] == "quant_group" for q in source["configs"]):
        gtrain = [r for r in grows if r["state"] != tag]
        gs, gf = group.fit_all(gtrain)
        result["quant_group"] = {"standardization": gs, "models": gf,
                                 "train_states": sorted({r["state"] for r in gtrain}),
                                 "n_train_configs": len(gtrain) // 3}
    if drows:
        purged = {tag} | {q["target_state"] for q in source["configs"] if q["method"] == "distill"}
        dtrain = [r for r in drows if r["tag"] not in purged]
        df = {c: distill._fit([r for r in dtrain if r["cap"] == c], True) for c in CAPS}
        result["distill_linear"] = {"models": df, "excluded_states": sorted(purged),
                                    "train_states": sorted({r["tag"] for r in dtrain}),
                                    "n_train_students": len(dtrain) // 3}
    for law, fit in result.items():
        if law != "held_out":
            assert tag not in fit["train_states"]
    return result


def predict_config(q, source, fold, modules):
    """Only dense/K0/config fields are consulted; actual losses are never read."""
    prune, group, v36, distill, _ = modules
    predicted = {}
    for cap in CAPS:
        anchor = q["anchor"][cap]
        raw = prune.raw_features(source["N0"], source["D0"], anchor)
        if q["law"] == "prune_power":
            f = fold["prune"]
            delta = prune.predict_all(f["models"][cap], prune.standardize(raw, f["standardization"]), q["d"])["power"]
        elif q["law"] == "quant_channel":
            with quant_vocabulary(v36):
                delta = float(v36.predict(fold["quant_channel"]["models"][cap],
                    [{"N0": source["N0"], "D0": source["D0"], "L0": anchor, "config": q["bit"]}])[0])
        elif q["law"] == "quant_group":
            f = fold["quant_group"]
            delta = group.predict_all(f["models"][cap], group.phi(raw, f["standardization"]),
                                      q["config"], f["standardization"])["low_order_2d"]
        elif q["law"] == "distill_linear":
            f = fold["distill_linear"]
            assert q["target_state"] not in f["train_states"]
            student = state_info(q["target_state"], prune)
            delta = float(distill._predict(f["models"][cap], [{**student, "L0": anchor}])[0])
        else:
            delta = 0.0
        predicted[cap] = float(anchor + delta)
    return predicted


def score(q, objective, dense, field):
    if objective == "multi":
        return max(q[field][c] - dense[c] for c in CAPS)
    return q[field][objective]


def evaluate_predictions(states, modules, prows, qrows, grows, drows, audit):
    folds = []
    errors = {law: {c: [] for c in OBJECTIVES} for law in LAWS}
    for state in states:
        fold = fit_fold(state, prows, qrows, grows, drows, modules)
        folds.append(fold)
        for q in state["configs"]:
            # Pass an outcome-free dictionary to the prediction interface.
            public = {k: v for k, v in q.items() if k != "actual"}
            q["predicted"] = predict_config(public, state, fold, modules)
            for cap in OBJECTIVES:
                error = score(q, cap, state["dense"], "predicted") - score(q, cap, state["dense"], "actual")
                errors[q["law"]][cap].append(abs(error))
        print(f"LOSO {state['tag']}: {len(state['configs'])} configurations; target outcomes excluded", flush=True)
    rank_gaps = [(f["held_out"], cap, b, v["n"], v["rank"])
                 for f in folds for cap in CAPS
                 for b, v in f["quant_channel"]["models"][cap]["bit_diagnostics"].items() if v["rank"] < 4]
    if rank_gaps:
        held = sorted({r[0] for r in rank_gaps})
        audit.gap("Per-channel 5-bit has only four measured states. Its four-coefficient OLS blocks "
                  "have rank 3 from three training observations when holding out " + ", ".join(held) +
                  ". Retain NumPy's deterministic minimum-norm least-squares solution; it is "
                  "underidentified. All other measured-bit blocks have rank 4; no 5-bit interpolation is used.")
    maes = {law: {"n_configs": len(errors[law]["math"]),
                   "mae": {c: float(np.mean(e)) if e else None for c, e in bycap.items()}}
            for law, bycap in errors.items()}
    return folds, maes


def tie_key(q):
    return METHODS.index(q["method"]), q["r"], q["id"]


def choose(configs, objective, dense, field="predicted"):
    return min(configs, key=lambda q: (score(q, objective, dense, field), *tie_key(q)))


def ambiguity(feasible, objective, dense, maes):
    """V60's retrospective no-clear-winner heuristic; not calibrated uncertainty."""
    if not feasible:
        return {"no_clear_winner_heuristic": None, "candidate_methods": [], "methods": [],
                "best_two_gap": None, "threshold": None}
    winners = [choose([q for q in feasible if q["method"] == m], objective, dense)
               for m in METHODS if any(q["method"] == m for q in feasible)]
    winners.sort(key=lambda q: (score(q, objective, dense, "predicted"), *tie_key(q)))
    best = winners[0]
    records = []
    candidates = [best["method"]]
    for q in winners:
        gap = score(q, objective, dense, "predicted") - score(best, objective, dense, "predicted")
        threshold = max(maes[best["law"]]["mae"][objective], maes[q["law"]]["mae"][objective])
        if q is not best and gap < threshold:
            candidates.append(q["method"])
        records.append({"method": q["method"], "config_id": q["id"], "law": q["law"],
                        "predicted_score": score(q, objective, dense, "predicted"),
                        "gap_from_best": gap, "mae_threshold": threshold})
    # The flag is exactly the specified best-two-method comparison. Candidate
    # membership extends the same pairwise rule to every remaining method.
    no_winner = len(records) > 1 and records[1]["gap_from_best"] < records[1]["mae_threshold"]
    return {"no_clear_winner_heuristic": no_winner, "candidate_methods": candidates,
            "methods": records, "best_two_gap": records[1]["gap_from_best"] if len(records) > 1 else None,
            "threshold": records[1]["mae_threshold"] if len(records) > 1 else None}


def selections(states, maes):
    cells, tables, ambiguity_summary = {}, {}, {}
    for objective in OBJECTIVES:
        entries = []
        for state in states:
            dense = state["dense"]
            for budget in BUDGETS:
                feasible = [q for q in state["configs"] if q["r"] <= budget + 1e-12]
                oracle = choose(feasible, objective, dense, "actual") if feasible else None
                selected = {"MAP": choose(feasible, objective, dense) if feasible else None,
                            "ORACLE": oracle,
                            "CHEAPEST": min(feasible, key=lambda q: (q["r"], *tie_key(q))) if feasible else None}
                for method in METHODS[:3]:
                    within = [q for q in feasible if q["method"] == method]
                    policy = f"{method}-only"
                    selected[policy] = choose(within, objective, dense) if within else None
                ambiguous = ambiguity(feasible, objective, dense, maes)
                actual_oracle = score(oracle, objective, dense, "actual") if oracle else None
                policies = {}
                for policy in POLICIES:
                    q = selected[policy]
                    if q is None:
                        policies[policy] = {"status": "INFEASIBLE", "feasible": False,
                            "config_id": None, "method": None, "r": None,
                            "predicted_score": None, "actual_score": None, "regret": None,
                            "oracle_method_agreement": None}
                        continue
                    regret = score(q, objective, dense, "actual") - actual_oracle
                    assert q["r"] <= budget + 1e-12
                    assert regret >= -1e-12
                    policies[policy] = {"status": "FEASIBLE", "feasible": True,
                                        "config_id": q["id"], "method": q["method"], "r": q["r"],
                                        "predicted_score": score(q, objective, dense, "predicted"),
                                        "actual_score": score(q, objective, dense, "actual"),
                                        "regret": regret, "oracle_method_agreement": q["method"] == oracle["method"]}
                entries.append({"state": state["tag"], "budget": budget, "n_feasible": len(feasible),
                                "oracle_method": oracle["method"] if oracle else None,
                                "common_feasible": all(p["feasible"] for p in policies.values()),
                                "policies": policies, **ambiguous,
                                "oracle_in_heuristic_candidate_set":
                                    oracle["method"] in ambiguous["candidate_methods"] if oracle else None})
        cells[objective] = entries
        no_winner = [e for e in entries if e["no_clear_winner_heuristic"]]
        evaluable = [e for e in entries if e["n_feasible"]]
        common = [e for e in entries if e["common_feasible"]]
        ambiguity_summary[objective] = {"interpretation": "retrospective heuristic",
            "n_cells": len(entries), "n_evaluable_cells": len(evaluable),
            "n_no_clear_winner_heuristic": len(no_winner),
            "no_clear_winner_heuristic_share": len(no_winner) / len(evaluable) if evaluable else None,
            "oracle_in_heuristic_candidates_share": mean_or_none(e["oracle_in_heuristic_candidate_set"] for e in evaluable),
            "oracle_in_heuristic_candidates_no_winner_share": mean_or_none(e["oracle_in_heuristic_candidate_set"] for e in no_winner),
            "n_single_method_cells": sum(len(e["methods"]) == 1 for e in entries)}
        tables[objective] = []
        for policy in POLICIES:
            pp = [e["policies"][policy] for e in entries]
            own = [p for p in pp if p["feasible"]]
            shared = [e["policies"][policy] for e in common]
            tables[objective].append({"policy": policy, "objective": objective, "n_cells": len(pp),
                "n_feasible": len(own), "n_infeasible": len(pp) - len(own),
                "feasibility_coverage": len(own) / len(pp), "n_common_feasible": len(shared),
                "common_feasible_coverage": len(shared) / len(pp),
                "mean_regret_own_feasible": mean_or_none(p["regret"] for p in own),
                "mean_regret_common_feasible": mean_or_none(p["regret"] for p in shared),
                "oracle_method_agreement_own_feasible": mean_or_none(p["oracle_method_agreement"] for p in own),
                "oracle_method_agreement_common_feasible": mean_or_none(p["oracle_method_agreement"] for p in shared)})
    return cells, tables, ambiguity_summary


def mean_or_none(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def candidate_coverage(states):
    rows = []
    for state in states:
        configs = state["configs"]
        rows.append({"state": state["tag"], "counts": dict(state["counts"]),
            "n_pruning": state["counts"]["prune_power"],
            "n_channel_quantization": state["counts"]["quant_channel"],
            "n_grouped_quantization": state["counts"]["quant_group"],
            "n_kd_students": state["counts"]["distill_linear"],
            "minimum_r_by_method": {m: min((q["r"] for q in configs if q["method"] == m), default=None)
                                    for m in METHODS},
            "minimum_r_by_law": {law: min((q["r"] for q in configs if q["law"] == law), default=None)
                                 for law in LAWS}})
    return rows


def number(value, digits=6, percent=False):
    if value is None:
        return "N/A"
    return f"{100 * value if percent else value:.{digits}f}" + ("%" if percent else "")


def to_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def markdown_table(rows):
    lines = ["| Policy | Feasible / all | Coverage | Regret: own | Regret: common | Agreement: own | Agreement: common |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['policy']} | {r['n_feasible']}/{r['n_cells']} | "
                     f"{number(r['feasibility_coverage'], 2, True)} | "
                     f"{number(r['mean_regret_own_feasible'])} | {number(r['mean_regret_common_feasible'])} | "
                     f"{number(r['oracle_method_agreement_own_feasible'], 2, True)} | "
                     f"{number(r['oracle_method_agreement_common_feasible'], 2, True)} |")
    return "\n".join(lines)


def report(summary):
    lines = ["# V64: selection with explicit feasibility", "", summary["protocol"], "",
             "## Endpoint definition (unchanged from v60)", "", summary["endpoint_definition"], "",
             "## States and measured configurations", "",
             "Counts are configurations, each with math/code/QA losses; dense is counted once per state.", "",
             "| State | Matrix N0 | Prune | Channel RTN | Grouped RTN | Smaller student | Dense |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for s in summary["states"]:
        lines.append(f"| {s['tag']} | {s['N0']:,} | " + " | ".join(str(s["counts"][law]) for law in LAWS) + " |")
    lines += ["| **Total** | — | " + " | ".join(str(summary["config_totals"][law]) for law in LAWS) + " |", "",
              "Exact measured config IDs, resource ratios, dense anchors, predictions and actual losses are in summary.json.", "",
              "## Candidate coverage and minimum available storage", "",
              "Counts above exclude the dense source from pruning/quantization/KD. KD counts are 0, 1 or 2 "
              "measured smaller same-stage students. N/A means no measured candidate. Quant includes both RTN variants.", "",
              "| State | Min r: prune | Min r: channel | Min r: group | Min r: quant (combined) | Min r: KD | Min r: dense |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["candidate_coverage"]:
        law, method = row["minimum_r_by_law"], row["minimum_r_by_method"]
        values = (method["prune"], law["quant_channel"], law["quant_group"], method["quant"], method["distill"], method["dense"])
        lines.append(f"| {row['state']} | " + " | ".join(number(v, 7) for v in values) + " |")
    lines += ["", "## Rules and interpretation", ""]
    lines.extend(f"- {rule}" for rule in summary["rules"])
    lines += ["", "## Law errors from this run", "",
              "Errors average configurations once per held-out source, never repeated budget cells. "
              "Distillation counts source/student alternatives; student reuse is disclosed above.", "",
              "| Law | Configs | Math MAE | Code MAE | QA MAE | Multi objective MAE |",
              "|---|---:|---:|---:|---:|---:|"]
    for law, row in summary["law_loso_mae"].items():
        vals = ["N/A" if row["mae"][c] is None else f"{row['mae'][c]:.6f}" for c in OBJECTIVES]
        lines.append(f"| {law} | {row['n_configs']} | " + " | ".join(vals) + " |")
    for cap in OBJECTIVES:
        title = "Multi-capability: minimum worst degradation" if cap == "multi" else {"math": "Math", "code": "Code", "qa": "QA"}[cap]
        a = summary["no_clear_winner_heuristic"][cap]
        coverage = a["oracle_in_heuristic_candidates_no_winner_share"]
        n_common = summary["tables"][cap][0]["n_common_feasible"]
        lines += ["", f"## {title}", "", markdown_table(summary["tables"][cap]), "",
                  f"Common-feasible subset: {n_common}/{a['n_cells']} cells, intersecting all six policies "
                  "(MAP, ORACLE, prune-only, quant-only, distill-only, CHEAPEST). Regret is in nats; "
                  "agreement uses exactly the same own/common subsets. Empty subsets report N/A.", "",
                  f"No-clear-winner heuristic: {a['n_no_clear_winner_heuristic']}/{a['n_evaluable_cells']} "
                  "evaluable cells; oracle method in heuristic candidate set: "
                  f"{number(a['oracle_in_heuristic_candidates_share'], 2, True)} of evaluable cells; "
                  + (f"{100*coverage:.2f}% within heuristic no-clear-winner cells." if coverage is not None else "no heuristic-flagged cells.")]
    lines += ["", "## Figure", "",
              "The figure retains v60's three MAP panels (math/code/QA), state order and budget axes. "
              "Grey cross-hatching denotes INFEASIBLE; diagonal hatching denotes the no-clear-winner heuristic. "
              "MAP has 100% feasibility on this measured panel, so there are no infeasible MAP cells to hatch. "
              "Fixed-policy infeasibility is reported in the tables and per-cell JSON."]
    lines += ["", "## Data gaps and limits", ""]
    lines.extend(f"- {gap}" for gap in summary["data_gaps"])
    lines += ["", "## Dense-anchor differences", "",
              "Signed offset from the source pruning dense anchor; all nonzero differences are listed.", "",
              "| State | Anchor | Math | Code | QA |", "|---|---|---:|---:|---:|"]
    for s in summary["states"]:
        for name, delta in s["dense_anchor_offsets"].items():
            if any(v != 0 for v in delta.values()):
                lines.append(f"| {s['tag']} | {name} | " + " | ".join(f"{delta[c]:+.9f}" for c in CAPS) + " |")
    lines += ["", "## Reuse and validation", "", *summary["validation"], "",
              "## Files read (SHA-256)", "",
              "Every explicit input/source/font read is printed, hashed and rechecked before outputs. "
              "Standard Python/NumPy/matplotlib installation files are represented by package versions.", "",
              "```text"]
    lines.extend(f"{sha}  {path}" for path, sha in sorted(summary["input_sha256"].items()))
    lines += ["```", "", "## Outputs", ""]
    lines.extend(f"- `{path}`" for path in OUTPUTS)
    return "\n".join(lines) + "\n"


@proofread_table
def latex_table(summary):
    n_cells = summary["n_budget_cells_per_objective"]
    n_common = summary["tables"]["math"][0]["n_common_feasible"]
    lines = [r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Feasible leave-one-state-out selection. Coverage and oracle-method agreement "
             r"are percentages; regret is in nats. Own/common refer to the subsets defined below.}",
             r"\label{tab:selection-feasible}", r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Policy & Objective & Coverage & \multicolumn{2}{c}{Regret (nats)} & \multicolumn{2}{c}{Agreement (\%)} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r" & & (\%) & Own & Common & Own & Common \\",
             r"\midrule"]
    for policy in POLICIES:
        for cap in OBJECTIVES:
            row = next(r for r in summary["tables"][cap] if r["policy"] == policy)
            label = "All (max)" if cap == "multi" else {"math": "Math", "code": "Code", "qa": "QA"}[cap]
            values = [number(100 * row["feasibility_coverage"], 1),
                      number(row["mean_regret_own_feasible"], 4),
                      number(row["mean_regret_common_feasible"], 4),
                      number(None if row["oracle_method_agreement_own_feasible"] is None else
                             100 * row["oracle_method_agreement_own_feasible"], 1),
                      number(None if row["oracle_method_agreement_common_feasible"] is None else
                             100 * row["oracle_method_agreement_common_feasible"], 1)]
            lines.append(f"{policy} & {label} & " + " & ".join(values) + r" \\")
        if policy != POLICIES[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\begin{minipage}{\linewidth}\small",
              rf"Coverage is feasible cells / {n_cells}. Own averages only that policy's feasible cells; "
              rf"common averages the same {n_common}/{n_cells} cells feasible for all six policies, including CHEAPEST. "
              r"Agreement uses those same respective subsets. Empty subsets are N/A. "
              r"An empty candidate set returns INFEASIBLE with no source fallback. "
              r"Dense ($r=1$) is eligible for MAP/ORACLE/CHEAPEST only at $r_{\max}=1$. "
              r"Fixed policies retain only their named method; quant-only pools channel and grouped RTN. "
              r"Each deployed endpoint is an absolute loss: source dense plus response for pruning/RTN; "
              r"student dense plus student post-training change for KD. "
              r"The multi objective is $\max_c(L_c-L_{0c})$. "
              r"The unchanged no-clear-winner heuristic uses pooled LOSO MAEs and is not calibrated uncertainty. "
              r"Five-bit folds remain underidentified; measurement coverage and dense anchors vary by arm.",
              r"\end{minipage}", r"\end{table}"]
    return "\n".join(lines) + "\n"


@proofread_table
def latex_candidate_coverage(summary):
    lines = [r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Measured candidate coverage by source state. Each method column gives "
             r"configuration count / minimum available storage ratio $r$.}",
             r"\label{tab:candidate-coverage}", r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Source state & Prune & Channel RTN & Grouped RTN & KD students & Min $r$: quant \\",
             r"\midrule"]
    for state, row in zip(summary["states"], summary["candidate_coverage"]):
        tag = f"{state['size']} @ {state['step']//1000}k"
        values = [f"{row['counts'][law]} / {number(row['minimum_r_by_law'][law], 7)}"
                  for law in LAWS[:-1]]
        lines.append(tag + " & " + " & ".join(values) + " & " +
                     number(row["minimum_r_by_method"]["quant"], 7) + r" \\")
    lines += [r"\midrule", "Total & " + " & ".join(str(summary["config_totals"][law]) for law in LAWS[:-1]) +
              r" & --- \\", r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\begin{minipage}{\linewidth}\small",
              r"N/A means no measured candidate. Each state also has one dense source, minimum $r=1$; "
              r"it is excluded from method counts. KD has 0, 1 or 2 smaller same-stage measured students. "
              r"Student reuse across sources counts as separate alternatives, not independent measurements. "
              r"Quant pools channel and grouped RTN. Ratios: pruning $r=d$, channel RTN $r=b/16$, "
              r"grouped RTN $r=(b+16/g)/16$, KD $r=N_S/N_0$. "
              r"Storage is nominal transformer-matrix storage; pruning requires a sparse format (index overhead excluded). "
              r"No missing configurations are imputed. Coverage is measured availability, independent of "
              r"the no-clear-winner heuristic.", r"\end{minipage}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def figure_bytes(summary, style, audit):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.text import Text

    # Font discovery traverses a set of files. This machine has two versions
    # of Liberation Serif with identical family/style scores; sort that tie
    # deterministically before the unchanged shared style resolves a font.
    font_manager.fontManager.ttflist.sort(key=lambda entry: str(Path(entry.fname).resolve()))
    font_manager.fontManager._findfont_cached.cache_clear()
    style.setup_style()  # Exact requested font/color setup, no upstream I/O helpers.
    for weight in ("normal", "bold"):
        font = font_manager.findfont(font_manager.FontProperties(family=plt.rcParams["font.serif"], weight=weight))
        audit.read(font)
    colors = [style.COLORS["pruning"], style.COLORS["quantization"],
              style.COLORS["distillation"], PALETTE["grid"], PALETTE["grid"]]
    states = summary["states"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 6.4), sharey=True)
    fig.subplots_adjust(left=.15, right=.985, top=.84, bottom=.14, wspace=.10)
    for ax, cap in zip(axes, CAPS):
        entries = summary["cells"][cap]
        data = np.array([METHODS.index(e["policies"]["MAP"]["method"])
                         if e["policies"]["MAP"]["feasible"] else len(METHODS)
                         for e in entries]).reshape(len(states), len(BUDGETS))
        ax.imshow(data, cmap=ListedColormap(colors), vmin=-.5, vmax=4.5, interpolation="nearest", aspect="auto")
        for i, entry in enumerate(entries):
            y, x = divmod(i, len(BUDGETS))
            if not entry["policies"]["MAP"]["feasible"]:
                ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor=PALETTE["grid"],
                                       edgecolor=PALETTE["dense"], hatch="xxx", lw=0))
                continue
            if entry["no_clear_winner_heuristic"]:
                ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor="none",
                                       edgecolor=PALETTE["reference"], hatch="///", lw=0))
            if not entry["policies"]["MAP"]["oracle_method_agreement"]:
                ax.plot(x, y, "o", ms=2.8, mfc=PALETTE["white"], mec=PALETTE["reference"], mew=.7)
        ticks = range(0, len(BUDGETS), 2)
        ax.set_xticks(list(ticks), [f"{BUDGETS[i]:.2f}" for i in ticks], rotation=45, ha="right")
        ax.set_yticks(range(len(states)), [f"{s['size']} @ {s['step']//1000}k" for s in states])
        ax.set_xticks(np.arange(-.5, len(BUDGETS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(states), 1), minor=True)
        ax.grid(which="minor", color=PALETTE["white"], lw=.35, alpha=.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        # Plain text avoids mathtext subscripts being smaller than 8 pt.
        ax.set_xlabel("Storage budget r_max")
        ax.set_title(style.CAP_LABEL[cap], pad=9)
    axes[0].set_ylabel("Held-out source (size, pretraining stage)")
    handles = [Patch(facecolor=color, label=label) for color, label in zip(colors, ("Prune", "Quant", "Distill", "Dense"))]
    handles += [Patch(facecolor=PALETTE["grid"], edgecolor=PALETTE["dense"], hatch="xxx", label="INFEASIBLE"),
                Patch(facecolor=PALETTE["white"], edgecolor=PALETTE["reference"], hatch="///", label="No clear winner (heuristic)"),
                Line2D([], [], marker="o", ms=3, mfc=PALETTE["white"], mec=PALETTE["reference"], ls="", label="Oracle method differs")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .995), ncol=4,
               frameon=False, columnspacing=1.3, handlelength=1.7)
    fig.text(.53, .885, "Measured configurations only; grouped RTN and student coverage varies by state", ha="center", fontsize=9)
    fig.text(.53, .025, "Nominal matrix storage; pruning requires sparse format. Diagonal hatching: no-clear-winner heuristic (LOSO MAE).",
             ha="center", fontsize=8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            assert text.get_fontsize() >= 8
            bbox = text.get_window_extent(renderer)
            assert bbox.x0 >= -1 and bbox.y0 >= -1 and bbox.x1 <= fig.bbox.x1 + 1 and bbox.y1 <= fig.bbox.y1 + 1, text.get_text()
    result = {}
    for ext in ("pdf", "png"):
        buffer = io.BytesIO()
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {}
        fig.savefig(buffer, format=ext, dpi=300, metadata=metadata)
        result[ext] = buffer.getvalue()
    plt.close(fig)
    return result


def build(audit, modules):
    prune, group, v36, distill, style = modules
    states, prows, qrows, grows, drows, students, reproduction = load_data(audit, prune, group, distill)
    folds, maes = evaluate_predictions(states, modules, prows, qrows, grows, drows, audit)
    cells, tables, ambiguity_summary = selections(states, maes)
    audit.gap("The measured configuration grid is heterogeneous: no missing loss is interpolated or imputed. "
              "Method agreement measures method identity, not exact configuration recovery. Budget cells "
              "share states and raw runs; no independence-based confidence intervals are claimed.")
    audit.gap("No-clear-winner heuristic thresholds use same-run pooled out-of-fold MAEs, including the scored fold. "
              "They are retrospective heuristic annotations, not independently calibrated uncertainty or inputs "
              "to MAP selection. Grouped RTN uncertainty uses its own law, although both RTN variants "
              "count as the single quant method.")
    rules = [
        "State universe: all non-2.8b Pythia states with both pruning and per-channel quantization files, "
        "ordered by transformer-matrix N0 then stage. All measured configs on these states are candidates.",
        "Storage: pruning r=d (unstructured pruning realizes this only with a sparse format; no index overhead); "
        "channel RTN r=b/16; grouped RTN r=(b+16/g)/16; smaller same-stage student r=matrix_n0(S)/matrix_n0(source); "
        "dense r=1, zero change. These ratios exclude embeddings/head/vectors, channel-scale overhead, "
        "unmerged adapter overhead, runtime memory and execution cost.",
        "Pruning imports v53 fit_all/predict_all unchanged, selects the prescribed power form only: "
        "(beta.phi)*((1-d)/0.3)^gamma. Ridge=1e-3 including intercept; gamma=0.5..6.0 by 0.05; "
        "training SSE selects gamma. Training-only population standardization pools response rows and capabilities.",
        "Channel RTN reuses v36 covariates/design/predict, the v38/v49 path: per-capability training-only "
        "population standardization, configuration indicators crossed with four phi terms, OLS. "
        "Extend its bit vocabulary to [8,6,5,4,3]; underidentified 5-bit uses the same least-squares "
        "minimum-norm solve. This is the requested per-bit regression, not v49's 4/6 interpolation rule.",
        "Grouped RTN imports v55 fit_all/predict_all unchanged and uses low_order_2d only: "
        "20 ridge coefficients for phi crossed with [1,u,v,u*v,u^2]. Only its 24 original dev cells "
        "(six states, b3/b5 x g64/g256) can train; remove every cell of the held-out state. "
        "Remaining measured bit/granularity/joint cells are scored only. Predict grouped RTN only on measured source configs.",
        "Distillation imports v39 _fit/_predict unchanged: per-capability OLS on the STUDENT's standardized "
        "log N0, dense L0c, log D0. Delta=raw post_training-dense. A source fold excludes the source "
        "and ALL its smaller same-stage candidate students from distillation training. This extra purge "
        "is stricter than source-only LOSO and prevents fitting any candidate's target outcome. "
        "Prediction anchors on the student's measured dense, never the source's dense.",
        "Feasible means measured r<=r_max on 0.20,0.25,...,1.00 (1e-12 floating tolerance). MAP minimizes "
        "predicted absolute loss; ORACLE minimizes recorded actual loss. Both may choose dense only at r_max=1. "
        "Fixed policies minimize predicted loss within prune/quant/distill. Any policy with no feasible "
        "candidate returns INFEASIBLE with null config, loss, regret and agreement, including at r_max=1 "
        "if its named method has no candidates. There is no source fallback. Dense belongs only to "
        "the unchanged MAP/ORACLE/CHEAPEST candidate sets. Feasible regret is nonnegative (1e-12 tolerance).",
        "CHEAPEST minimizes r. All exact ties use method order prune,quant,distill,dense; remaining ties "
        "use smaller r then lexical config ID. Quant-only includes channel and grouped RTN.",
        "No-clear-winner heuristic: compare the predicted winners of the best two distinct feasible methods. "
        "Flag strictly when their gap is smaller than max(their respective winning-config laws' LOSO MAEs). "
        "Dense law MAE is zero. Candidate set includes the best method plus each other method satisfying "
        "that pairwise gap/MAE rule against the best. With one feasible method, the flag is false. "
        "The heuristic flag belongs to the shared state/budget candidate space. With no feasible "
        "method it is null; its summary denominator includes only evaluable cells.",
        "Feasibility coverage is the share of all 289 (state,budget) cells with a feasible choice. "
        "Own-feasible regret and oracle-method agreement average only each policy's feasible cells. "
        "Common-feasible regret and agreement average the intersection where all six compared policies "
        "(including ORACLE and CHEAPEST) are feasible, against the same unrestricted budget-feasible ORACLE. "
        "Empty subsets return null (N/A in tables), never zero. Every included (state,budget) has equal weight. Law MAEs weight "
        "each evaluated (held-out source, measured config) once, including shared student alternatives; "
        "the selected configuration is never tuned using these errors.",
        "Multi-capability objective is max_c(predicted Lc - source dense L0c), using all three capabilities "
        "on the same configuration. Its oracle minimizes max_c(actual Lc - source dense L0c); regret "
        "is the difference of these actual worst-degradation objectives. Its no-clear-winner heuristic threshold uses "
        "each law's MAE on this same scalar max objective, recomputed from held-out predictions. "
        "This is a minimax requirement, not an absolute pass/fail threshold or an average of capabilities.",
    ]
    summary = {"schema_version": 2, "analysis": "v64-selection-feasible", "protocol": "Retrospective leave-one-state-out validation of fixed "
               "per-method law families; predictions receive K0 plus dense anchors, never target compressed "
               "losses. No target fitting, measurement execution, randomness, timestamps or GPU dependencies.",
               "n_states": len(states), "n_budget_cells_per_objective": len(states)*len(BUDGETS),
               "budgets": BUDGETS, "states": states, "config_totals": {law: sum(s["counts"][law] for s in states) for law in LAWS},
               "endpoint_definition": ENDPOINT, "compared_policies": list(POLICIES),
               "candidate_coverage": candidate_coverage(states),
               "rules": rules, "law_loso_mae": maes, "tables": tables,
               "no_clear_winner_heuristic": ambiguity_summary,
               "cells": cells, "folds": folds, "students": students, "v39_reproduction": reproduction,
               "data_gaps": audit.gaps, "input_sha256": audit.hashes,
               "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                            "matplotlib": style.matplotlib.__version__},
               "validation": [
                   "- Imported unchanged: v53 fit_all/predict_all; v55 fit_all/predict_all; v39 _fit/_predict/_cv. "
                   "v36 basic_input/covariates/design_matrix/predict supply the quantization path, with the documented "
                   "5-bit vocabulary/rank exception. All source files are hashed.",
                   "- Reconstructed v39 raw deltas agree with saved delta fields; original leave-size/leave-step "
                   "MAEs reproduce summary.json to 1e-10 relative / 1e-12 absolute tolerance when the cohort is available.",
                   "- Each fold asserts source exclusion; each student prediction asserts target-student exclusion. "
                   "Prediction interfaces receive no actual loss. Every feasible policy obeys its budget and has nonnegative regret.",
                   "- Plot text is checked for >=8 pt and containment within the saved canvas. PDF creation/modification "
                   "timestamps are suppressed; duplicate font-family matches are ordered by absolute filename. "
                   "Inputs are rehashed before exclusive output creation.",
               ]}
    return summary


def validate_v60_preservation(summary, audit):
    """Read-only regression check against v60, after all fitting and selection."""
    reference = audit.json("results/v60-selection-maps/summary.json")
    audit.read("results/v60-selection-maps/summary.md")
    assert summary["n_states"] == 17
    assert summary["n_budget_cells_per_objective"] == 289
    checked = ("n_states", "n_budget_cells_per_objective", "budgets", "states", "config_totals",
               "law_loso_mae", "folds", "students", "v39_reproduction")
    for field in checked:
        # Convert NumPy fit arrays to the same representation as saved JSON.
        current = json.loads(json.dumps(summary[field], default=to_json, allow_nan=False))
        if current != reference[field]:
            raise AssertionError(f"V60 preservation failed for {field}; check input/protocol drift")
    n_retained, n_infeasible = 0, 0
    for objective in OBJECTIVES:
        entries = summary["cells"][objective]
        common = [e for e in entries if all(e["policies"][p]["feasible"] for p in POLICIES)]
        for entry, old in zip(entries, reference["cells"][objective]):
            assert (entry["state"], entry["budget"]) == (old["state"], old["budget"])
            assert entry["common_feasible"] == (entry in common)
            assert entry["no_clear_winner_heuristic"] == old["no_clear_winner"]
            for field in ("candidate_methods", "methods", "best_two_gap", "threshold", "n_feasible", "oracle_method"):
                assert entry[field] == old[field], (objective, entry["state"], entry["budget"], field)
            for policy in POLICIES:
                new, previous = entry["policies"][policy], old["policies"][policy]
                if previous["source_fallback"]:
                    assert new["status"] == "INFEASIBLE" and not new["feasible"]
                    assert all(new[k] is None for k in ("config_id", "method", "r", "predicted_score",
                                                       "actual_score", "regret", "oracle_method_agreement"))
                    n_infeasible += 1
                else:
                    assert new["status"] == "FEASIBLE" and new["feasible"]
                    assert new["r"] <= entry["budget"] + 1e-12
                    assert new["regret"] >= -1e-12
                    for field in ("config_id", "method", "r", "predicted_score", "actual_score", "regret",
                                  "oracle_method_agreement"):
                        assert new[field] == previous[field]
                    n_retained += 1
        for row in summary["tables"][objective]:
            policy = row["policy"]
            own = [e for e in entries if e["policies"][policy]["feasible"]]
            assert row["n_feasible"] == len(own)
            assert row["feasibility_coverage"] == len(own) / 289
            assert row["n_common_feasible"] == len(common)
            for suffix, subset in (("own_feasible", own), ("common_feasible", common)):
                for target, field in (("mean_regret", "regret"), ("oracle_method_agreement", "oracle_method_agreement")):
                    expected = sum(e["policies"][policy][field] for e in subset) / len(subset) if subset else None
                    actual = row[f"{target}_{suffix}"]
                    if expected is None:
                        assert actual is None
                    else:
                        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert all(r["n_kd_students"] in (0, 1, 2) for r in summary["candidate_coverage"])
    summary["v60_preservation"] = {"unchanged_fields": list(checked),
        "retained_policy_choices": n_retained, "fallbacks_replaced_by_infeasible": n_infeasible,
        "no_clear_winner_heuristic_unchanged": True,
        "all_feasible_choices_within_budget": True, "subset_metrics_independently_recomputed": True}
    summary["validation"].append(
        f"- Read-only v60 regression: all states/configs, predictions, fits, MAEs, students and budgets "
        f"match its saved summary exactly. {n_retained} retained policy choices are identical; "
        f"{n_infeasible} source fallbacks become INFEASIBLE across four objectives. "
        "The no-clear-winner heuristic is unchanged. Own/common metrics are independently recomputed. "
        "Both v60 result files are hashed and rechecked without modification.")


def main():
    if "--figures-only" in sys.argv:
        if __package__:
            from .plot_paper_appendix import generate
        else:
            from plot_paper_appendix import generate
        generate("selection_feasible")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT,
                        help="Output directory within results/v64-selection-feasible/ (exclusive creation)")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    allowed_root = ROOT.resolve() / "results/v64-selection-feasible"
    if not output_root.is_relative_to(allowed_root):
        raise ValueError(f"Outputs must stay within {allowed_root}")
    targets = [output_root / p for p in OUTPUTS]
    for path in targets:
        if not path.resolve().is_relative_to(allowed_root):
            raise ValueError(f"Output symlink escapes authorized directory: {path}")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    # Keep matplotlib's cache and configuration off all existing repository files.
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v64-mpl-", dir=output_root) as mpldir:
        os.environ["MPLCONFIGDIR"] = mpldir
        os.environ["MPLBACKEND"] = "Agg"
        audit = Audit()
        modules = imports(audit)
        summary = build(audit, modules)
        validate_v60_preservation(summary, audit)
        figures = figure_bytes(summary, modules[-1], audit)
        audit.verify()
        payloads = [json.dumps(summary, indent=2, allow_nan=False, default=to_json).encode() + b"\n",
                    report(summary).encode(), latex_table(summary).encode(), latex_candidate_coverage(summary).encode(),
                    figures["pdf"], figures["png"], Path(__file__).read_bytes()]
        for path, payload in zip(targets, payloads):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
            print(f"WROTE {path} sha256={hashlib.sha256(payload).hexdigest()}", flush=True)
        for cap in OBJECTIVES:
            print(f"\n{cap.upper()}\n{markdown_table(summary['tables'][cap])}")


if __name__ == "__main__":
    main()
