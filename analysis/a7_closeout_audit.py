#!/usr/bin/env python3
"""Read frozen A2/A3/A5 artifacts; standard-library CPU arithmetic only.

No model imports, measurement, fitting, resampling, or candidate selection.
Writes only the three requested audit reports; any failed check exits nonzero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "a2": "results/a2-curvature-interaction/summary.json",
    "counts": "results/a2-curvature-interaction/nonembedding_counts.json",
    "plan": "results/a3-corner-pools/plan.json",
    "seed": "results/a5-training-seed-noise/summary.json",
    "corners": "results/a5-corner-second-difference/summary.json",
}
OUT = Path("results/a7-closeout-audit")
DOC = Path("paper/docs/CLOSEOUT_AUDIT.md")
STRUCTURES = ("F_log", "F_curv", "F_int")
FORMS = {
    "F_log": "(a+a_prime*z)*u+(b+b_prime*z)*v",
    "F_curv": "(a+a_prime*z)*u+(b+b_prime*z)*h_p(E)",
    "F_int": "(a+a_prime*z)*u+(b+b_prime*z)*v+k*u*v",
}
PRIMARY = (("code", "training_probe:MBPP:", "I_T"),
           ("qa", "2wiki_new:", "I_U"))
READOUTS = {"qa": "2Wiki QA probe", "math": "MATH-500", "code": "MBPP"}
SIGNS = {"1": -1, "2": 1, "3": 1, "4": -1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"Expected finite number, got {value!r}")
    return value


def close(a, b, message):
    require(math.isclose(finite(a), finite(b), rel_tol=1e-10, abs_tol=1e-12), message)


def interval_flags(gain, interval):
    finite(gain)
    if interval is None:
        return None, False
    require(len(interval) == 2, "Malformed frozen interval")
    lo, hi = map(finite, interval)
    require(lo <= hi, "Reversed frozen interval")
    return lo > 0 or hi < 0, gain > 0 and lo > 0


def gate_row(result, fold, target, oracle=False):
    """Selection and metrics come exclusively from the individual held fold."""
    selection = fold.get("selection", {})
    metric = fold.get("metrics", {}).get(target, {})
    row = {
        "capability": result["capability"], "distribution": result["distribution"],
        "target": target, "split": result["split"], "held": fold["held"],
        "primary": selection.get("primary"),
        "baseline": selection.get("baseline", {}).get(target),
        "selection_method": selection.get("method"),
        "status": metric.get("status", fold["status"]),
        "primary_mae": None, "baseline_mae": None, "paired_gain": None,
        "frozen_error_interval": None, "interval_excludes_zero": None,
        "positive_gain_and_clear_interval": False,
        "source_record": f"results[{result['distribution']}, {result['split']}].folds[{fold['held']}]",
    }
    if row["status"] != "scored":
        if oracle:
            row["baseline"] = None
        return row
    require(set(metric["primary_candidates"]) == {selection["primary"]},
            "Per-fold primary differs from recorded selection")
    require(set(metric["inner_selected_baselines"]) == {row["baseline"]},
            "Per-fold baseline differs from recorded inner selection")
    if oracle:
        row["baseline"] = metric["strongest_observed_baseline"]
        bm, gain, interval = (metric[k] for k in
                             ("strongest_baseline_mae", "improvement_over_strongest", "interval_over_strongest"))
    else:
        bm, gain, interval = (metric[k] for k in
                             ("baseline_mae", "paired_improvement", "frozen_prediction_error_interval"))
    pm = metric["primary_mae"]
    close(bm - pm, gain, "Paired gain differs from difference of frozen MAEs")
    close(bm, metric["all_model_mae"][row["baseline"]], "Baseline MAE mismatch")
    close(pm, metric["all_model_mae"][row["primary"]], "Primary MAE mismatch")
    excludes, positive = interval_flags(gain, interval)
    row.update(primary_mae=pm, baseline_mae=bm, paired_gain=gain,
               frozen_error_interval=interval, interval_excludes_zero=excludes,
               positive_gain_and_clear_interval=positive)
    return row


def check_gate(a2):
    inner, diagnostic, conclusions, split_metrics = [], [], [], []
    for cap, prefix, target in PRIMARY:
        results = [r for r in a2["results"] if r["capability"] == cap
                   and r["distribution"].startswith(prefix)]
        require(len({r["distribution"] for r in results}) == 1, "Missing/ambiguous primary distribution")
        require({r["split"] for r in results} ==
                {"data_rung", "student", "largest_budget", "pool_seed"}
                and len(results) == 4, "Missing/duplicate primary split")
        for result in results:
            require(result["folds"], "Missing held folds")
            require(len({f["held"] for f in result["folds"]}) == len(result["folds"]), "Duplicate held fold")
            for fold in result["folds"]:
                inner.append(gate_row(result, fold, target))
                diagnostic.append(gate_row(result, fold, target, oracle=True))
            m = result["metrics"][target]
            flags = interval_flags(m["paired_improvement"], m["frozen_prediction_error_interval"]) if m["status"] == "scored" else (None, False)
            split_metrics.append({"capability": cap, "target": target, "split": result["split"],
                                  "status": m["status"], "paired_gain": m.get("paired_improvement"),
                                  "frozen_error_interval": m.get("frozen_prediction_error_interval"),
                                  "positive_gain_and_clear_interval": flags[1]})
        rows = [r for r in inner if r["capability"] == cap]
        scored = [r for r in rows if r["status"] == "scored"]
        split_rows = [r for r in split_metrics if r["capability"] == cap]
        wins = sum(r["positive_gain_and_clear_interval"] for r in scored)
        split_wins = sum(r["positive_gain_and_clear_interval"] for r in split_rows)
        positive = sum(r["paired_gain"] > 0 for r in scored)
        # No new launch threshold: verify the observed absence of a clear split gain.
        require(split_wins == 0, "A clear split gain requires reassessment of the launch verdict")
        label = "code × I_T (MBPP)" if cap == "code" else "QA × I_U (fresh 2wiki_new, 1% proxy)"
        conclusion = (f"{label}: the launch verdict remains not met with the development-stage baseline; "
                      f"{positive}/{len(scored)} scorable folds have positive gain, "
                      f"{wins}/{len(scored)} have positive gain and an interval clear of zero "
                      f"({len(rows)-len(scored)} unscorable), and {split_wins}/"
                      f"{sum(r['status'] == 'scored' for r in split_rows)} scorable splits have a clear positive gain.")
        conclusions.append({"capability": cap, "scorable_folds": len(scored),
                            "unscorable_folds": len(rows)-len(scored), "positive_gain_folds": positive,
                            "positive_gain_clear_interval_folds": wins, "verdict_changed": False,
                            "conclusion": conclusion})
    return {"status": "COMPLETE", "inner_selected": inner, "outer_oracle_diagnostic": diagnostic,
            "stored_split_metrics": split_metrics, "conclusions": conclusions,
            "interval_definition": a2["protocol"]["error_interval"],
            "note": "Frozen pointwise error intervals; folds overlap and are not independent replications. "
                    "QA I_U is the stored 1% tolerance proxy, not an exact fixed-budget contrast. "
                    "When inner folds were insufficient, A2 used its registered fallback settings. "
                    "Outer-oracle intervals do not cover the post-hoc baseline selection."}


def check_accounting(counts, a2):
    rows = []
    require(set(counts["students"]) == set(a2["nonembedding_counts"]), "Student count inventory mismatch")
    for student, r in counts["students"].items():
        keys = ("total_parameters", "embedding_parameters", "vision_and_projector_parameters", "non_embedding_parameters")
        require(all(type(r[k]) is int and r[k] >= 0 for k in keys), "Invalid integer parameter count")
        total, embedding, vision, decoder = (r[k] for k in keys)
        require(total - embedding - vision == decoder == a2["nonembedding_counts"][student],
                f"Parameter accounting does not reconcile: {student}")
        require(r["revision"] and r["embedding_tensors"], "Missing revision/embedding provenance")
        rows.append({"student": student, **{k: r[k] for k in keys}, "revision": r["revision"],
                     "embedding_tensors": r["embedding_tensors"],
                     "total_minus_embedding": total - embedding})
    four = next(r for r in rows if r["student"] == "gemma3-4b")
    b = lambda n: f"{n / 1e9:.3f}B"
    note = (f"For gemma3-4b, {b(four['total_parameters'])} minus {b(four['embedding_parameters'])} "
            f"equals {b(four['total_minus_embedding'])} only when the "
            f"{b(four['vision_and_projector_parameters'])} vision/projector deduction is omitted; "
            f"the round used {b(four['non_embedding_parameters'])} TEXT-DECODER NON-EMBEDDING parameters. "
            "These rounded values are calculated from the exact file counts below. "
            "The artifact combines vision tower and multimodal projector into one deduction.")
    return {"status": "COMPLETE", "rows": rows, "note": note}


def check_seed_pair(seed, a5):
    """Keep one correlated trajectory; do not consume the old sigma summaries."""
    paths = [seed["seed0"], seed["seed1"]]
    require(paths[0] != paths[1] and all("poolB" in p for p in paths), "Expected one distinct pool-B seed pair")
    students = {p.split("/")[2] for p in paths}
    require(len(students) == 1, "Seed pair students differ")
    student = students.pop()
    shared = seed["shared_checkpoints"]
    require(len(shared) == 14 and len(set(shared)) == 14, "Expected 14 unique shared checkpoints")
    require([p[0] for p in seed["per_checkpoint"]] == shared, "Seed trajectory membership/order mismatch")
    rows = []
    for checkpoint, budget, differences in seed["per_checkpoint"]:
        update = int(checkpoint.removeprefix("update-"))
        require(set(differences) == set(READOUTS), "Seed readout mismatch")
        require(type(budget) is int and budget > 0, "Invalid shared budget")
        rows.append({"checkpoint": checkpoint, "update": update, "T_supervised": budget,
                     "differences_seed1_minus_seed0": {c: finite(differences[c]) for c in READOUTS},
                     "corner": {36: 1, 74: 4}.get(update)})
    require([r["update"] for r in rows] == sorted(r["update"] for r in rows), "Unordered trajectory")
    corners = [r for r in rows if r["corner"] is not None]
    require([r["update"] for r in corners] == [36, 74], "Missing seed corner checkpoint")
    for r in corners:
        measured = a5["students"][student]["corners"][str(r["corner"])]
        require(measured["achieved_update"] == r["update"] and
                measured["achieved_supervised_tokens"] == r["T_supervised"], "Seed/A5 corner mismatch")
        require(measured["source"].startswith(seed["seed0"] + "/"), "A5 pool-B seed-0 source mismatch")
    return {"status": "COMPLETE", "student": student, "pool": "B", "training_seeds": [0, 1],
            "run_paths": paths, "run_pair_count": 1, "shared_checkpoint_count": len(rows),
            "unit_of_replication": "one pair of training runs; one correlated difference trajectory",
            "independent_checkpoint_replicates": False, "trajectory": rows, "corner_checkpoints": corners,
            "interpretation": "Descriptive only: one pair documents the signed seed differences along this "
                              "pool-B trajectory and at its two corner checkpoints. The 14 checkpoints are "
                              "correlated observations of one pair, not 14 independent replicates. One pair "
                              "cannot estimate a general training-seed noise distribution, establish interval "
                              "coverage, or establish that the registered band is generally conservative. "
                              "No sigma is calculated from these points; A5's derived sigma summaries are not reused."}


def second_difference(values):
    require(set(values) == set(SIGNS), "Need exactly four corners")
    return math.fsum(SIGNS[i] * finite(values[i]) for i in SIGNS)


def h_p(e, p):
    v = math.log1p(e)
    if p == 0:
        return v
    if abs(p) < 1e-7:
        x = p * v
        return v * (1 + x/2 + x*x/6 + x*x*x/24)
    return math.expm1(p * v) / p


def structure_second_difference(name, coefficients, coordinates, z, t_ref, p=None):
    """Derive the contrast of the stored columns, including achieved mismatch.

    On a rectangle, the u and reuse contrasts vanish exactly, leaving
    k*(u_high-u_low)*(v_low-v_high) for F_int. Do not force cancellation
    when the four supplied achieved coordinates differ along an edge.
    """
    require(name in STRUCTURES, "Unknown frozen structure")
    require(len(coefficients) == (5 if name == "F_int" else 4), "Missing coefficient")
    require(set(coordinates) == set(SIGNS), "Need exactly four coordinates")
    require(finite(t_ref) > 0, "Invalid T_ref")
    finite(z)
    a, ap, b, bp = map(finite, coefficients[:4])
    if name == "F_curv":
        finite(p)
    for t, e in coordinates.values():
        require(finite(t) >= 0 and finite(e) >= 0, "Invalid corner coordinate")
    u = {i: math.log1p(t / t_ref) for i, (t, e) in coordinates.items()}
    v = {i: math.log1p(e) for i, (t, e) in coordinates.items()}
    reuse = {i: h_p(e, p) for i, (t, e) in coordinates.items()} if name == "F_curv" else v
    du, dh = second_difference(u), second_difference(reuse)
    cross = second_difference({i: u[i] * v[i] for i in SIGNS})
    rectangle = (coordinates["1"][0] == coordinates["2"][0]
                 and coordinates["3"][0] == coordinates["4"][0]
                 and coordinates["1"][1] == coordinates["3"][1]
                 and coordinates["2"][1] == coordinates["4"][1])
    if rectangle:
        require(du == 0 and dh == 0, "Additive cancellation failed")
        cross = (u["3"] - u["1"]) * (v["1"] - v["2"])
    budget_term, reuse_term = (a + ap*z) * du, (b + bp*z) * dh
    interaction = finite(coefficients[4]) * cross if name == "F_int" else 0.0
    return {"I": math.fsum((budget_term, reuse_term, interaction)),
            "budget_mismatch_term": budget_term, "reuse_mismatch_term": reuse_term,
            "interaction_term": interaction, "is_exact_rectangle": rectangle,
            "contrast_u": du, "contrast_reuse": dh, "contrast_uv": cross}


def frozen_fit(a2, cap, distribution, name):
    matches = [(i, p) for i, p in enumerate(a2["parameter_intervals"])
               if p["capability"] == cap and p["distribution"] == distribution]
    require(len(matches) == 1, "Missing/ambiguous full-development parameters")
    index, record = matches[0]
    f = record["fits"][name]["fit"]
    require(f["name"] == name and f["fit_id"], "Wrong/missing frozen fit identity")
    require(f["standardizer"]["descriptor"] == record["selection"]["descriptor"]
            and f["lambda_"] == record["selection"]["lambda_"], "Frozen selection mismatch")
    require(f["train_keys"] and all(k[2:] == [cap, distribution] for k in f["train_keys"]),
            "Frozen fit distribution mismatch")
    return {k: f[k] for k in ("name", "coefficients", "p", "lambda_", "standardizer", "fit_id")}, \
        f"parameter_intervals[{index}].fits.{name}.fit"


def check_predictions(a2, plan, a5):
    require(a2["protocol"]["structures"] == FORMS, "Stored structure form changed; derivation must be reviewed")
    require(a5["preregistration"]["formula"] == "I = delta_3 - delta_4 - delta_1 + delta_2", "Sign convention mismatch")
    t_ref = finite(a2["protocol"]["T_ref"])
    rows, geometry, failures = [], {}, []
    require(set(a5["selected_students"]) == {"gemma3-1b", "gemma3-4b"}, "Unexpected achieved student inventory")
    for student in a5["selected_students"]:
        stored = a5["students"][student]
        require(stored["status"] == "COMPLETE", "Incomplete A5 student")
        coordinates = {}
        for pc in plan["design"]["corners"]:
            i = str(pc["corner"])
            c = stored["corners"][i]
            require(c["status"] == "VERIFIED", "Unverified achieved corner")
            require(c["pool"] == pc["pool"] and c["D_U_pool"] == pc["D_U_pool"], "Pool accounting mismatch")
            require(c["achieved_supervised_tokens"] == pc["T_supervised"], "Achieved/plan budget mismatch")
            close(c["achieved_E"], c["achieved_supervised_tokens"] / c["D_U_pool"], "Achieved reuse mismatch")
            coordinates[i] = (c["achieved_supervised_tokens"], c["achieved_E"])
        geometry[student] = {i: {"T": t, "E": e, "pool": stored["corners"][i]["pool"]}
                             for i, (t, e) in coordinates.items()}
        for cap, label in READOUTS.items():
            row = {"student": student, "capability": cap, "readout": label, "status": "COMPLETE"}
            try:
                readout = stored["readouts"][cap]
                distribution = readout["distribution"]
                require(distribution.startswith("training_probe:"), "Expected probe readout")
                responses = [stored["corners"][i]["responses"][cap] for i in SIGNS]
                require(all(r["distribution"] == distribution for r in responses), "Corner distribution mismatch")
                measured = finite(readout["I"])
                close(measured, second_difference({i: stored["corners"][i]["responses"][cap]["delta"] for i in SIGNS}),
                      "Measured I differs from frozen corner responses")
                registered = a5["preregistration"]["readouts"][cap]
                close(readout["noise_on_I"], registered["noise_on_I"], "Registered noise mismatch")
                row.update(distribution=distribution, measured_I=measured,
                           registered_noise=registered["noise_on_I"],
                           registered_quoted_disagreement=registered["predicted_disagreement"],
                           additive_rectangle_prediction=0.0, additive_zero_absolute_error=abs(measured))
                predictions = {}
                for name in STRUCTURES:
                    fit, source = frozen_fit(a2, cap, distribution, name)
                    scaler = fit["standardizer"]
                    if scaler["descriptor"] == "log_parameters":
                        descriptor = math.log(a2["nonembedding_counts"][student])
                    else:
                        require(scaler["descriptor"] == "initial_loss", "Unknown descriptor")
                        require(len({r["initial_loss"] for r in responses}) == 1,
                                "Initial-loss descriptor differs between achieved corners")
                        descriptor = responses[0]["initial_loss"]
                    require(finite(scaler["scale"]) > 0, "Missing/invalid standardizer scale")
                    z = (finite(descriptor) - finite(scaler["mean"])) / scaler["scale"]
                    prediction = structure_second_difference(name, fit["coefficients"], coordinates, z, t_ref, fit["p"])
                    prediction.update(absolute_error=abs(prediction["I"] - measured),
                                      signed_error=prediction["I"] - measured, fit=fit,
                                      source_record=source, descriptor_value=descriptor, z=z)
                    predictions[name] = prediction
                row["predictions"] = predictions
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                row.update(status="STOPPED", reason=f"Missing/invalid frozen row input: {exc}; no parameter was estimated.")
                failures.append(f"{student}/{cap}: {row['reason']}")
            rows.append(row)
    return {"status": "INCOMPLETE" if failures else "COMPLETE", "failures": failures,
            "T_ref": t_ref, "forms": FORMS, "achieved_coordinates": geometry, "rows": rows,
            "error_definition": "Absolute prediction error abs(predicted I - measured I); signed errors also stored.",
            "geometry_note": "The achieved coordinates are an approximate rectangle. The additive structures "
                             "predict exactly zero on a matched rectangle, but their frozen predictions at the "
                             "four unequal achieved coordinates include mismatch terms. The zero column is the "
                             "registered rectangle reference, not their exact achieved-coordinate evaluation. "
                             "No coordinates are projected, averaged, or interpolated.",
            "interpretation": "For the 2Wiki QA probe, both measured values are inside twice the registered noise: "
                              "failed to reject additivity. This does not identify why the fitted structures "
                              "miss the measurements. MATH-500 is marginal and has opposite measured signs across "
                              "students; MBPP remains underpowered. Students are reported separately. "
                              "The registered bands are not calibrated confidence intervals."}


def build_audit(root=ROOT):
    inputs, hashes = {}, {}
    for name, path in SOURCES.items():
        raw = (Path(root) / path).read_bytes()
        hashes[path] = hashlib.sha256(raw).hexdigest()
        inputs[name] = json.loads(raw)
    checks = {}
    calls = {
        "gate": lambda: check_gate(inputs["a2"]),
        "accounting": lambda: check_accounting(inputs["counts"], inputs["a2"]),
        "seed_pair": lambda: check_seed_pair(inputs["seed"], inputs["corners"]),
        "predictions": lambda: check_predictions(inputs["a2"], inputs["plan"], inputs["corners"]),
    }
    for name, call in calls.items():
        try:
            checks[name] = call()
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            checks[name] = {"status": "FAILED", "reason": str(exc)}
    # Detect concurrent modification and never overwrite an input artifact.
    for path, digest in hashes.items():
        require(hashlib.sha256((Path(root) / path).read_bytes()).hexdigest() == digest, f"Input changed during audit: {path}")
    return {"schema_version": "a7-closeout-audit-v1", "status": "COMPLETE" if all(
        c["status"] == "COMPLETE" for c in checks.values()) else "FAILED",
        "device": "cpu", "training_run": False, "new_measurements": 0, "fits_performed": 0,
        "new_candidates": 0, "resampling_performed": False, "source_sha256": hashes, "checks": checks}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|"]
                     + ["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in rows])


def number(value):
    return "NA" if value is None else f"{value:.6f}"


def gate_table(rows, oracle=False):
    def cells(r):
        ci = r["frozen_error_interval"]
        gain = number(r["paired_gain"]) + (f" [{number(ci[0])}, {number(ci[1])}]" if ci else " [NA]")
        label = "MBPP I_T" if r["capability"] == "code" else "fresh 2wiki_new I_U"
        excludes = {True: "yes", False: "no", None: "NA"}[r["interval_excludes_zero"]]
        return [label, r["split"], r["held"], r["primary"] or "NA", r["baseline"] or "NA",
                number(r["primary_mae"]), number(r["baseline_mae"]), gain, excludes,
                "yes" if r["positive_gain_and_clear_interval"] else "no" if r["status"] == "scored" else "unscorable"]
    return table(["Primary target", "Split", "Held", "Primary F", "OUTER oracle" if oracle else "INNER baseline",
                  "MAE F", "MAE baseline", "Gain [frozen 95% interval]", "Excludes 0?", "Positive and clear?"],
                 [cells(r) for r in rows])


def render(audit):
    lines = ["# A7 closeout audit", "", f"Status: **{audit['status']}**. CPU only; existing artifacts only; "
             "no training, new measurement, fitting, resampling, or new candidate.", "",
             "All losses, contrasts, errors, and registered noise are in native-token nats. "
             "Displayed decimals are rounded; summary.json retains full precision.", ""]
    titles = {"gate": "1. The gate baseline", "accounting": "2. Parameter accounting",
              "seed_pair": "3. One training-seed pair", "predictions": "4. Frozen predictions at achieved corners"}
    for name, title in titles.items():
        lines += [f"## {title}", ""]
        c = audit["checks"][name]
        if c["status"] == "FAILED":
            lines += [f"FAILED: {c['reason']}", ""]
            continue
        if name == "gate":
            lines += [c["note"], "", "### Development-stage baseline", "", gate_table(c["inner_selected"]), "",
                      "### DIAGNOSTIC — post-hoc OUTER-score oracle", "",
                      "This baseline was selected using the outer score and cannot authorize launch. "
                      "Its frozen interval is conditional on that choice.", "", gate_table(c["outer_oracle_diagnostic"], True), ""]
            lines += [r["conclusion"] + "\n" for r in c["conclusions"]]
            lines += ["Split-level gains [frozen intervals], read directly from A2's result metrics:", ""]
            lines += [f"- {r['capability']} / {r['split']}: {number(r['paired_gain'])} "
                      f"{r['frozen_error_interval'] if r['frozen_error_interval'] else '(unscorable)'}."
                      for r in c["stored_split_metrics"]]
        elif name == "accounting":
            lines += [c["note"], "", table(["Student", "Total", "− tied token embedding",
                      "− (vision tower + projector)", "= TEXT-DECODER NON-EMBEDDING", "Revision"],
                      [[r["student"], *(f"{r[k]:,}" for k in ("total_parameters", "embedding_parameters",
                        "vision_and_projector_parameters", "non_embedding_parameters")), r["revision"]] for r in c["rows"]])]
        elif name == "seed_pair":
            lines += [f"{c['student']}, pool {c['pool']}, seeds 0 and 1: **one run pair**, "
                      f"{c['shared_checkpoint_count']} shared checkpoints. Signed difference = delta(seed 1) − delta(seed 0).",
                      "", c["interpretation"], "", table(["Update", "Supervised tokens", "2Wiki QA probe difference",
                      "MATH-500 difference", "MBPP difference", "Corner checkpoint"],
                      [[r["update"], r["T_supervised"], *(number(r["differences_seed1_minus_seed0"][cap]) for cap in READOUTS),
                        f"corner {r['corner']}" if r["corner"] else "—"] for r in c["trajectory"]]), ""]
            lines += [f"- **Update {r['update']} (corner {r['corner']})**: " + "; ".join(
                      f"{label} {number(r['differences_seed1_minus_seed0'][cap])}" for cap, label in READOUTS.items()) + "."
                      for r in c["corner_checkpoints"]]
        else:
            lines += [c["geometry_note"], "", "### Derivation from A2's stored form", "",
                      f"u = log(1 + T/{c['T_ref']:g}), v = log(1 + E), z = (descriptor − mean)/scale. "
                      "The descriptor, mean, scale, lambda, and coefficients are the stored full-development values. "
                      "A2 saves coefficients in the original design-column units after undoing ridge column scaling; "
                      "no further coefficient standardisation is applied.", "",
                      "Write C(q) = q3 − q4 − q1 + q2, A = a + a′z, and B = b + b′z. The stored forms give:", "",
                      "- F_log: I = A C(u) + B C(v).",
                      "- F_curv: I = A C(u) + B C(h_p(E)), with h_p(E) = ((1+E)^p−1)/p and h_0(E) = v.",
                      "- F_int: I = A C(u) + B C(v) + k C(uv).", "",
                      "On an exact rectangle, C(u) = C(v) = C(h_p(E)) = 0, so F_log and F_curv give exactly 0 "
                      "and F_int gives k (u_high−u_low)(v_low−v_high), independent of A, B, and z. "
                      "At the achieved coordinates all four stored coordinates enter C directly; the small A and B "
                      "mismatch terms remain. The interaction coefficient k has no student modifier in A2's form.", ""]
            for student, corners in c["achieved_coordinates"].items():
                lines += [f"- {student}: " + "; ".join(f"corner {i} ({r['pool']}): T={r['T']}, E={r['E']:.12f}"
                          for i, r in corners.items()) + "."]
            lines += ["", "Each prediction cell gives **I / absolute error**. The zero reference applies to both additive "
                      "structures on a matched rectangle. Registered quoted disagreement is copied from A5, not used "
                      "as a coefficient or a prediction.", ""]
            rows = []
            for r in c["rows"]:
                if r["status"] != "COMPLETE":
                    rows.append([r["student"], r["readout"], "STOPPED: " + r["reason"], *(["NA"]*6)])
                    continue
                preds = r["predictions"]
                rows.append([r["student"], r["readout"],
                             *(f"{number(preds[n]['I'])} / {number(preds[n]['absolute_error'])}" for n in ("F_int", "F_log", "F_curv")),
                             f"0 / {number(r['additive_zero_absolute_error'])}", number(r["measured_I"]),
                             number(r["registered_noise"]), number(r["registered_quoted_disagreement"])])
            lines += [table(["Student", "Readout", "F_int achieved I / error", "F_log achieved I / error",
                      "F_curv achieved I / error", "Additive rectangle 0 / error", "Measured I",
                      "Registered noise on I", "Registered quoted disagreement"], rows), "",
                      "### Frozen parameter provenance", ""]
            for r in c["rows"]:
                if r["status"] != "COMPLETE":
                    continue
                f = r["predictions"]["F_int"]
                fit = f["fit"]
                s = fit["standardizer"]
                lines += [f"- {r['student']} / {r['readout']}: `{f['source_record']}`, fit `{fit['fit_id']}`; "
                          f"descriptor={s['descriptor']}, lambda={fit['lambda_']}, mean={s['mean']:.17g}, "
                          f"scale={s['scale']:.17g}, descriptor value={f['descriptor_value']:.17g}, z={f['z']:.17g}; "
                          f"[a,a′,b,b′,k]={fit['coefficients']}; k C(uv)={f['interaction_term']:.12g}, "
                          f"A C(u)+B C(v)={f['budget_mismatch_term']+f['reuse_mismatch_term']:.12g}."]
            lines += ["", "All three frozen fit records, standardizers, decompositions, and errors are retained in summary.json. "
                      "The registered quoted disagreement is not reproduced by the stored full-development F_int "
                      "evaluations at these achieved coordinates; this audit does not infer how that quoted number was obtained.",
                      "", c["interpretation"]]
        lines += [""]
    lines += ["## Reproduction and input integrity", "", "```bash",
              "python -B -m pytest -q tests/test_a7_closeout_audit.py",
              "python -B analysis/a7_closeout_audit.py", "```", "",
              "The audit command prints all four numbered checks and their tables after writing the reports. "
              "A failed check or a stopped prediction row gives a non-zero exit status. "
              "Inputs are hashed before and after the audit; A2/A3/A5 artifacts are never written.", ""]
    lines += [f"- `{p}`: `{digest}`" for p, digest in audit["source_sha256"].items()]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        audit = build_audit(args.root)
        report = render(audit)
        output = args.root / OUT
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
        (output / "summary.md").write_text(report)
        (args.root / DOC).parent.mkdir(parents=True, exist_ok=True)
        (args.root / DOC).write_text(report)
        print(report, end="")
        return 0 if audit["status"] == "COMPLETE" else 1
    except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
        print(f"A7 closeout audit FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
