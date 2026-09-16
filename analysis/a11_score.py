#!/usr/bin/env python3
"""Score A11's frozen functions after same-job dense/pruned measurements (CPU).

python -B analysis/a11_score.py --verify-only
python -B analysis/a11_score.py
No fitting, model imports, or changes to registration files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/a11-efficiency-confirmation"
CAPS = ("math", "code", "qa")
METHODS = ("power_18", "power_36", "A2_36", "median_curve_36", "median_curve_18")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_json(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_registration(out=OUT, root=ROOT):
    path = out / "predictions.json"
    expected = path.with_suffix(".json.sha256").read_text().split()[0]
    require(digest(path) == expected, "predictions.json SHA256 mismatch")
    frozen = read(path)
    plan = read(out / "plan.json")
    require(digest(out / "plan.json") == frozen["plan_sha256"], "Plan changed after freeze")
    require(digest(out / "prereg.md") == frozen["prereg_sha256"] == plan["prereg_sha256"],
            "Preregistration changed after freeze")
    require(sha_json(frozen["coefficients"]) == frozen["coefficients_sha256"], "Coefficient SHA256 mismatch")
    for relative, expected_hash in frozen["runtime_sha256"].items():
        require(digest(root / relative) == expected_hash, f"Runtime/probe file changed: {relative}")
    require(frozen["targets"] == plan["targets"], "Target inputs changed")
    require(frozen["capabilities"] == list(CAPS) and set(frozen["coefficients"]) == set(METHODS),
            "Capability/method mismatch")
    wanted = {(t["state"], d, c) for t in plan["targets"] for d in plan["densities"] for c in CAPS}
    actual = [(r["state"], r["density"], r["capability"]) for r in frozen["cells"]]
    require(len(actual) == 72 and len(set(actual)) == 72 and set(actual) == wanted, "Not the 72 registered cells")
    for cell in frozen["cells"]:
        target = next(t for t in plan["targets"] if t["state"] == cell["state"])
        require(set(cell["predictions"]) == set(METHODS), "Missing cell predictor")
        for name, prediction in cell["predictions"].items():
            require(prediction["inputs"] == {"N0": target["N0"], "D0": target["D0"], "L0,c": None},
                    "Unexpected target source inputs")
            require(prediction["coefficient_set_sha256"] == frozen["coefficients_sha256"], "Wrong coefficient set")
            expression = prediction["predicted_delta_L"]
            if name.startswith("median_curve"):
                require(expression["kind"] == "constant" and finite_number(expression["value"]), "Invalid median")
            else:
                require(expression["kind"] == "affine_standardized_anchor", "Invalid source function")
                require(all(finite_number(expression[k]) for k in ("a", "b", "mu_L", "sigma_L"))
                        and expression["sigma_L"] > 0, "Invalid anchor coefficients")
    return plan, frozen


def evaluate_prediction(prediction, dense_loss):
    expression = prediction["predicted_delta_L"]
    if expression["kind"] == "constant":
        return expression["value"]
    require(finite_number(dense_loss), "A finite same-job dense anchor is required")
    return expression["a"] + expression["b"] * ((dense_loss - expression["mu_L"]) / expression["sigma_L"])


def measurement_probe_hashes(probes):
    return {c: sha_json([{k: p[k] for k in ("prompt", "completion")} for p in probes[c][1::2]]) for c in CAPS}


def validate_losses(losses, plan):
    require(set(losses) == {"1.0", *(str(d) for d in plan["densities"])}, "Wrong dense/density loss keys")
    for density, values in losses.items():
        require(set(values) == set(CAPS), f"Wrong capabilities at density {density}")
        require(all(finite_number(v) and v >= 0 for v in values.values()), f"Invalid loss at density {density}")


def read_measurements(plan, frozen, measurements, prediction_hash):
    tables, provenance, missing = {}, {}, []
    for target in plan["targets"]:
        directory = measurements / target["state"].replace("@", "--")
        loss_path, meta_path = directory / "prune_losses.json", directory / "metadata.json"
        if not loss_path.is_file() or not meta_path.is_file():
            missing.append(target["state"])
            continue
        table, meta = read(loss_path), read(meta_path)
        require(meta["status"] == "complete" and meta["target"] == target, "Wrong/incomplete measurement state")
        require(meta["predictions_sha256"] == prediction_hash, "Measurement used another prediction freeze")
        require(meta["protocol"] == plan["protocol"], "Measurement protocol mismatch")
        require(meta["probe_sha256"] == frozen["measurement_probe_sha256"], "Measurement probe mismatch")
        require(meta["prune_losses_sha256"] == digest(loss_path), "Measurement loss file changed")
        require(datetime.fromisoformat(meta["started_utc"]) >= datetime.fromisoformat(frozen["created_utc"]),
                "Measurement predates prediction freeze")
        validate_losses(table, plan)
        tables[target["state"]] = table
        provenance[target["state"]] = {"losses_sha256": digest(loss_path), "metadata_sha256": digest(meta_path)}
    return tables, provenance, missing


def score_tables(plan, frozen, tables):
    require(set(tables) == {t["state"] for t in plan["targets"]}, "Incomplete panel")
    for table in tables.values():
        validate_losses(table, plan)
    rows, errors = [], {c: {m: [] for m in METHODS} for c in CAPS}
    for cell in frozen["cells"]:
        state, density, cap = cell["state"], cell["density"], cell["capability"]
        table = tables[state]
        dense, pruned = table["1.0"][cap], table[str(density)][cap]
        observed = pruned - dense
        predicted = {m: evaluate_prediction(p, dense) for m, p in cell["predictions"].items()}
        require(all(finite_number(v) for v in predicted.values()), "Nonfinite evaluated prediction")
        absolute_errors = {m: abs(value - observed) for m, value in predicted.items()}
        for method, error in absolute_errors.items():
            errors[cap][method].append(error)
        rows.append(dict(state=state, density=density, capability=cap, dense_loss=dense, pruned_loss=pruned,
                         observed_delta_L=observed, predicted_delta_L=predicted, absolute_errors=absolute_errors))
    by_cap = {}
    for cap in CAPS:
        require(all(len(values) == 24 for values in errors[cap].values()), "MAE needs all 24 cells")
        # Center before summation so a constant error exactly at the registered
        # margin remains that constant (fsum([0.05]*24)/24 rounds upward).
        mae = {m: values[0] + math.fsum(v-values[0] for v in values) / 24
               for m, values in errors[cap].items()}
        excess = mae["power_18"] - mae["A2_36"]
        by_cap[cap] = dict(n_cells=24, mae=mae, power_18_minus_comparator_mae={
            m: mae["power_18"] - mae[m] for m in METHODS if m != "power_18"},
            within_margin=excess <= plan["decision"]["margin_nats"],
            descriptive_mae_le_0_25={m: v <= .25 for m, v in mae.items()},
            median_beats_power_18={m: mae[m] < mae["power_18"] for m in ("median_curve_36", "median_curve_18")})
    support = all(by_cap[c]["within_margin"] for c in plan["decision"]["primary_capabilities"])
    return dict(status="complete", efficiency_claim_supported=support, decision=plan["decision"],
                by_capability=by_cap, cells=rows)


def report_text(summary):
    if summary["status"] != "complete":
        return "# A11 confirmation: pending\n\nAll 72 cells are required. " + summary["reason"] + "\n"
    lines = ["# A11 confirmation", "", "Efficiency claim " +
             ("supported" if summary["efficiency_claim_supported"] else "not supported") +
             " under the registered 0.05-nat rule on both math and code.", "",
             "| Capability | power 18 | power 36 | A2 36 | median 36 | median 18 | power 18 - A2 36 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for cap in CAPS:
        record = summary["by_capability"][cap]
        lines.append("| " + cap + " | " + " | ".join(f"{record['mae'][m]:.6f}" for m in METHODS) +
                     f" | {record['power_18_minus_comparator_mae']['A2_36']:+.6f} |")
    lines += ["", "Unweighted MAE in nats over all 24 cells per capability. QA and MAE <= 0.25 are descriptive."]
    for cap in CAPS:
        for method, wins in summary["by_capability"][cap]["median_beats_power_18"].items():
            if wins:
                lines.append(f"{cap}: {method} has lower MAE than power_18; the efficiency result does not imply overall superiority.")
    lines += ["No refitting; only same-job dense anchors were substituted into frozen functions.",
              "The four-state panel uses shared probes; no independent-seed or population uncertainty is inferred.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--registration-dir", type=Path, default=OUT)
    parser.add_argument("--measurements-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        plan, frozen = load_registration(args.registration_dir)
        if args.verify_only:
            print("Verified frozen plan, coefficients, 72 cells x 5 predictors, runtime files and probes; no model imports.")
            return 0
        prediction_hash = digest(args.registration_dir / "predictions.json")
        tables, provenance, missing = read_measurements(plan, frozen,
            args.measurements_dir or args.registration_dir / "measurements", prediction_hash)
        summary = (dict(status="pending", efficiency_claim_supported=None, missing_states=missing,
                        reason="Missing complete state measurements: " + ", ".join(missing)) if missing
                   else score_tables(plan, frozen, tables))
        summary.update(predictions_sha256=prediction_hash, measurement_sha256=provenance,
                       scored_utc=datetime.now(timezone.utc).isoformat())
        out = args.output_dir or args.registration_dir
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "summary.json", summary)
        (out / "summary.md").write_text(report_text(summary))
        print(report_text(summary))
        return 0 if summary["status"] == "complete" else 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"A11 pending / invalid inputs; no confirmation scored: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
