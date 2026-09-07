#!/usr/bin/env python3
"""Close v27 measurement definitions using saved JSON only (standard library/CPU).

Run: python analysis/v27b_scoring_readout.py
No model, tokenizer, dataset, network, training, or prediction code is imported.
Original artifacts are read-only. Both generated outputs are deterministic.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
LOSSES = ("L_full", "L_direct", "L_given")
LABELS = dict(zip(LOSSES, ("full solution loss", "direct answer loss",
                         "given-reference-reasoning answer loss")))
UNITS = ("example_token", "example_byte", "corpus_token", "corpus_byte")
BENCHMARKS = {
    "math": "MATH-500", "math_gsm8k": "GSM8K", "code": "MBPP",
    "code_humaneval": "HumanEval", "qa": "2WikiMultihopQA", "qa_hotpotqa": "HotpotQA",
}
PREFIXES = {"canonical": "", "newline": "\n", "blank_line": "\n\n",
            "answer_prefix": "Answer: ", "equivalent_prefix": "The answer is "}


def digest(value):
    """The original v27 text/probe digest convention, not a raw-file digest."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected finite numeric value")
    return float(value)


def close(actual, expected, label):
    if not math.isclose(finite(actual), finite(expected), abs_tol=1e-9, rel_tol=1e-10):
        raise ValueError(f"Stored/recomputed mismatch: {label}: {actual} vs {expected}")


def item_map(items):
    result = {}
    for item in items:
        identity = (item["measurement_index"], item["probe_sha256"])
        if identity in result:
            raise ValueError("Duplicate probe identity")
        result[identity] = item
    return result


def recompute(items):
    """Keep equal-example and corpus weighting separate in BOTH unit systems."""
    if not items:
        raise ValueError("Per-sample NLL and exact bytes required; aggregate conversion forbidden")
    nll, tokens, byte_counts = [], [], []
    for item in items:
        value = finite(item.get("sum_ce", item.get("total_nll")))
        if value < 0:
            raise ValueError("NLL must be nonnegative")
        for key in ("n_tokens", "target_bytes"):
            if type(item.get(key)) is not int or item[key] <= 0:
                raise ValueError(f"Exact positive integer {key} required; no aggregate conversion")
        nll.append(value)
        tokens.append(item["n_tokens"])
        byte_counts.append(item["target_bytes"])
    return {"n": len(items), "sum_nll": math.fsum(nll), "sum_tokens": sum(tokens),
            "sum_bytes": sum(byte_counts), "mean_total_nll": mean(nll),
            "example_token": mean(v / n for v, n in zip(nll, tokens)),
            "example_byte": mean(v / b for v, b in zip(nll, byte_counts)),
            "corpus_token": math.fsum(nll) / sum(tokens),
            "corpus_byte": math.fsum(nll) / sum(byte_counts)}


def check_group(group):
    units = recompute(group["items"])
    close(group["L_c"], units["example_token"], "L_c")
    close(group["token_weighted_L_c"], units["corpus_token"], "token_weighted_L_c")
    saved = group["units_v27_v1"]
    for old, new in {"sum_nll": "sum_nll", "sum_target_tokens": "sum_tokens",
                     "sum_target_bytes": "sum_bytes", "n_examples": "n",
                     "example_token_normalized": "example_token",
                     "token_normalized": "corpus_token", "byte_normalized": "corpus_byte"}.items():
        close(saved[old], units[new], old)
    return {**units, "original_L_c": group["L_c"]}


def benchmark_entries(payload):
    """Accept the observed benchmarks schema and a capabilities wrapper.

    Keep benchmark identity, including primary/secondary, instead of averaging
    different benchmarks within a capability. Unknown leaf schemas fail closed.
    """
    container = "benchmarks" if "benchmarks" in payload else "capabilities"
    if container not in payload:
        raise ValueError("Panel requires benchmarks or capabilities")
    found = []

    def walk(value, path):
        if not isinstance(value, dict):
            return
        if "scoring" in value:
            name = value.get("benchmark", value.get("spec", {}).get("benchmark"))
            key = next((k for k, v in BENCHMARKS.items() if v == name), None)
            if key is None:
                raise ValueError(f"Unregistered benchmark at {path}: {name}")
            if value.get("capability", key.split("_")[0]) != key.split("_")[0]:
                raise ValueError("Benchmark/capability mismatch")
            found.append((key, value, "/".join(path)))
        else:
            for key, child in value.items():
                walk(child, (*path, key))

    walk(payload[container], (container,))
    if not found or len({k for k, _, _ in found}) != len(found):
        raise ValueError("Missing or duplicate benchmark scoring groups")
    return sorted(found)


def verify_text(item, text):
    if item["target_bytes"] != len(text.encode("utf-8")) or item["target_sha256"] != digest(text):
        raise ValueError("Scored target hash/UTF-8 byte count disagrees with reference text")


def contrasts(groups):
    return {f"{left}_minus_{right}": {unit: groups[left][unit] - groups[right][unit]
                                      for unit in (*UNITS, "mean_total_nll")}
            for left, right in (("L_direct", "L_full"), ("L_given", "L_full"),
                                ("L_given", "L_direct"))}


def regions(maps, decompositions):
    """Chain rule on TOTAL NLL, never subtraction of normalized losses."""
    reasoning = []
    for identity, full in maps["L_full"].items():
        given, direct = maps["L_given"][identity], maps["L_direct"][identity]
        if any(given[k] != direct[k] for k in ("n_tokens", "target_bytes", "target_sha256")):
            raise ValueError("Direct/given answer targets differ")
        r = decompositions[identity]["r"]
        total = full["sum_ce"] - given["sum_ce"]
        count = full["n_tokens"] - given["n_tokens"]
        byte_count = full["target_bytes"] - given["target_bytes"]
        if not r:
            for loss in LOSSES:
                for field in ("sum_ce", "n_tokens", "target_bytes", "target_sha256"):
                    if maps[loss][identity][field] != full[field]:
                        raise ValueError("Empty reasoning must give identical three losses")
        else:
            if total < -1e-5 or count <= 0 or byte_count != len(r.encode("utf-8")):
                raise ValueError("Invalid chain-rule reasoning region")
            reasoning.append({"sum_ce": max(total, 0), "n_tokens": count, "target_bytes": byte_count})
    full_total = math.fsum(i["sum_ce"] for i in maps["L_full"].values())
    answer_total = math.fsum(i["sum_ce"] for i in maps["L_given"].values())
    return {"reasoning_items": len(reasoning), "empty_reasoning_items": len(maps["L_full"]) - len(reasoning),
            "reasoning": recompute(reasoning) if reasoning else None,
            "reasoning_sum_nll": full_total - answer_total,
            "answer_sum_nll": answer_total, "answer_fraction_full_nll": answer_total / full_total if full_total else None,
            "interpretation": ("Entire scored target is y; three losses coincide, no separately scored reasoning."
                               if not reasoning else "Reasoning total = full total minus given-answer total on shared token sequence.")}


def format_readout(benchmark, maps, decompositions):
    control = benchmark["format_control"]
    condition = control["conditioning"]
    if condition not in LOSSES or control["surface_prefixes"] != PREFIXES:
        raise ValueError("Unrecognized fixed format protocol")
    variants = control["variants"]
    if set(variants) != set(PREFIXES):
        raise ValueError("Incomplete format variants")
    fm = {name: item_map(group["items"]) for name, group in variants.items()}
    identities = set(fm["canonical"])
    if not identities or not identities <= set(maps[condition]):
        raise ValueError("Format cohort must be a nonempty subset of scoring cohort")
    if any(set(m) != identities for m in fm.values()):
        raise ValueError("Format variants must have identical paired cohorts")
    metrics = {name: check_group(group) for name, group in variants.items()}
    ranges, deltas = [], defaultdict(list)
    for identity in sorted(identities):
        canonical = fm["canonical"][identity]
        split = decompositions[identity]
        body = split["r"] + split["y"] if condition == "L_full" else split["y"]
        values = []
        for name, prefix in PREFIXES.items():
            item = fm[name][identity]
            verify_text(item, prefix + body)
            if (item["information_sha256"] != digest(body)
                    or item["context_tokens_sha256"] != canonical["context_tokens_sha256"]):
                raise ValueError("Format control changed information or context")
            val = item["sum_ce"] / item["n_tokens"]
            values.append(val)
            deltas[name].append(val - canonical["sum_ce"] / canonical["n_tokens"])
        for field in ("sum_ce", "n_tokens", "target_bytes", "target_sha256"):
            if canonical[field] != maps[condition][identity][field]:
                raise ValueError("Canonical format differs from selected scoring condition")
        ranges.append(max(values) - min(values))
    close(control["mean_within_item_range_nats_per_token"], mean(ranges), "format range")
    paired = control["paired_items"]
    expected = {identity[0]: j for j, identity in enumerate(sorted(identities))}
    if len(paired) != len(expected) or {p["measurement_index"] for p in paired} != set(expected):
        raise ValueError("Incomplete saved format pairs")
    for pair in paired:
        j = expected[pair["measurement_index"]]
        close(pair["range_nats_per_token"], ranges[j], "paired format range")
        for name in PREFIXES:
            close(pair["delta_from_canonical"][name], deltas[name][j], "paired format delta")
    matched = {loss: recompute([maps[loss][i] for i in sorted(identities)]) for loss in LOSSES}
    conditioning = contrasts(matched)["L_given_minus_L_direct"]
    condition_abs = mean(abs(maps["L_given"][i]["sum_ce"] / maps["L_given"][i]["n_tokens"]
                             - maps["L_direct"][i]["sum_ce"] / maps["L_direct"][i]["n_tokens"])
                         for i in identities)
    return {"conditioning": condition, "n": len(identities), "variants": metrics,
            "delta_from_canonical": {name: {u: metrics[name][u] - metrics["canonical"][u]
                                             for u in (*UNITS, "mean_total_nll")} for name in PREFIXES},
            "mean_absolute_delta_from_canonical_token": {k: mean(map(abs, v)) for k, v in deltas.items()},
            "mean_within_item_range_token": mean(ranges),
            "given_minus_direct_same_cohort": conditioning,
            "mean_absolute_conditioning_delta_token": condition_abs,
            "format_exclusions": control["format_exclusions"],
            "interpretation": "Fixed information/context; prefix tokens and bytes are included. Changes include wrapper prediction and length dilution; body-only NLL is not stored."}


def analyze_benchmark(key, benchmark):
    scoring = benchmark["scoring"]
    if set(scoring) != set(LOSSES):
        raise ValueError("Three pre-specified losses required")
    maps = {loss: item_map(scoring[loss]["items"]) for loss in LOSSES}
    if any(set(m) != set(maps["L_full"]) for m in maps.values()):
        raise ValueError("Three losses must use identical paired cohorts")
    decomposition = item_map(benchmark["probe_decompositions"])
    excluded = item_map(benchmark["conditioning_exclusions"])
    eligible = set(maps["L_full"])
    if (len(decomposition) != benchmark["n_measurement_probes"]
            or eligible & set(excluded) or eligible | set(excluded) != set(decomposition)):
        raise ValueError("Eligibility/exclusion records do not partition measurement probes")
    for loss, items in maps.items():
        for identity, item in items.items():
            split = decomposition[identity]
            if not split["supported"] or not split["reconstruction_exact"]:
                raise ValueError("Scored unsupported reference decomposition")
            verify_text(item, split["r"] + split["y"] if loss == "L_full" else split["y"])
    metrics = {loss: check_group(scoring[loss]) for loss in LOSSES}
    region = regions(maps, decomposition)
    return {"benchmark_key": key, "benchmark": BENCHMARKS[key], "capability": key.split("_")[0],
            "spec": {k: benchmark[k] for k in ("dataset", "decomposition_policy", "n_measurement_probes")},
            "n": metrics["L_full"]["n"], "conditioning_exclusions": benchmark["conditioning_exclusions"],
            "losses": metrics, "differences": contrasts(metrics), "regions": region,
            "format_control": format_readout(benchmark, maps, decomposition),
            "legacy_v6": check_group(benchmark["legacy_v6"]),
            "excludes": (["Distinct hidden reasoning versus answer loci for CODE/QA in this scoring protocol.",
                          "Three independent confirmations from the identical empty-r losses."]
                         if not region["reasoning_items"] else
                         ["Equivalence of direct-answer and given-reference-reasoning answer loss.",
                          "Interpreting normalized full-minus-given as reasoning NLL."]),
            "format_excludes": "A perfectly surface-invariant normalized reference likelihood on these items."}, maps


def order_comparison(values, left, right):
    def ordering(unit):
        # Preserve ties explicitly rather than resolving them by model name.
        levels = defaultdict(list)
        for model in sorted(values):
            levels[values[model][unit]].append(model)
        return [models for _, models in sorted(levels.items())]

    flips, ties = [], []
    for a, b in combinations(sorted(values), 2):
        dl, dr = values[a][left] - values[b][left], values[a][right] - values[b][right]
        row = {"models": [a, b], "left_difference": dl, "right_difference": dr}
        if dl * dr < 0:
            flips.append(row)
        elif (dl == 0) != (dr == 0):
            ties.append(row)
    return {"left": left, "right": right, "left_order_low_to_high": ordering(left),
            "right_order_low_to_high": ordering(right), "reversals": flips,
            "tie_changes": ties, "ordering_changed": bool(flips or ties)}


def ordering_readout(panels, raw):
    groups = defaultdict(list)
    for panel in panels:
        # Never rank different compression configurations as a tokenizer check.
        if panel["metadata"]["adapter"] is not None:
            continue
        config = (panel["metadata"]["prune_density"], panel["metadata"]["quant_bits"],
                  json.dumps(panel["protocol"], sort_keys=True), panel["probe_sha256"])
        groups[config].append(panel)
    result = []
    for members in groups.values():
        if len(members) < 2:
            continue
        if len({p["model"] for p in members}) != len(members):
            raise ValueError("Duplicate model/config in ordering comparison")
        for key in sorted(set.intersection(*(set(p["benchmarks"]) for p in members))):
            for loss in LOSSES:
                maps = {p["model"]: raw[p["id"]][key][loss] for p in members}
                common = set.intersection(*(set(m) for m in maps.values()))
                if not common:
                    raise ValueError("No matched probes for cross-model ordering")
                for identity in common:
                    signatures = {(m[identity]["target_sha256"], m[identity]["target_bytes"]) for m in maps.values()}
                    if len(signatures) != 1:
                        raise ValueError("Cross-model scored target text/bytes differ")
                values = {model: recompute([m[i] for i in sorted(common)]) for model, m in maps.items()}
                comparisons = {name: order_comparison(values, a, b) for name, a, b in (
                    ("equal_example_unit_change", "example_token", "example_byte"),
                    ("corpus_unit_change", "corpus_token", "corpus_byte"),
                    ("main_to_corpus_byte", "example_token", "corpus_byte"),
                    ("token_weighting_change", "example_token", "corpus_token"))}
                result.append({"benchmark_key": key, "loss": loss, "n_common": len(common),
                               "probe_identities": [list(i) for i in sorted(common)],
                               "original_n": {model: len(m) for model, m in maps.items()},
                               "tokenizers": {p["model"]: p["metadata"]["tokenizer"] for p in members},
                               "prune_density": members[0]["metadata"]["prune_density"],
                               "quant_bits": members[0]["metadata"]["quant_bits"],
                               "values": values, "comparisons": comparisons})
    return result


def load_evals(results, read):
    rows, snapshots = [], []
    for path in sorted((results / "v12-distill").glob("**/eval.json")):
        payload = read(path, "eval")
        if payload["version"] != 12:
            raise ValueError(f"Unexpected eval version: {path}")
        run_root = path.parent if "trajectory" not in path.relative_to(results).parts else path.parents[2]
        model, run = run_root.parent.name, run_root.name
        if payload["student_tag"] != model or payload["run_name"] != run:
            raise ValueError("Eval path/metadata identity mismatch")
        if payload["measurement_benchmarks"] != {c: BENCHMARKS[c] for c in ("math", "code", "qa")}:
            raise ValueError("Unrecognized legacy eval benchmark protocol")
        definition = "sum_target_CE / sum_target_tokens (nats/token, legacy V12)"
        if payload.get("loss_definition", definition) != definition:
            raise ValueError("Unrecognized V12 native loss definition; cannot mix endpoints")
        record = {"source": path.relative_to(results.parent).as_posix(), "version": 12,
                  "model": model, "run": run, "recipe": payload["recipe"], "teacher": payload["teacher"],
                  "n_per_domain": payload["n_per_domain"], "seed": payload.get("seed", 0),
                  "training_mode": payload.get("training_mode"),
                  "snapshot_kind": payload.get("snapshot_kind", "final"),
                  "independent_seed_unit": payload.get("independent_seed_unit", f"{model}/{run}"),
                  "measurement_tokens": payload["measurement_tokens"],
                  "measurement_samples": payload["measurement_samples"],
                  "loss_definition": definition,
                  "loss_definition_original": payload.get("loss_definition"),
                  "losses": {}}
        for cap in ("math", "code", "qa"):
            dense, post = finite(payload["dense"][cap]), finite(payload["post_training"][cap])
            close(payload["delta"][cap], post - dense, "V12 post-minus-own-dense")
            record["losses"][cap] = {"dense": dense, "post": post, "delta": post - dense,
                                     "scored_region": "reference solution (legacy truncation)" if cap == "math" else "entire reference code y" if cap == "code" else "reference answer y",
                                     "three_loss_v27_delta": None, "byte_delta": None}
        (snapshots if "trajectory" in path.relative_to(results).parts else rows).append(record)
    if not rows:
        raise ValueError("No existing final V12 eval JSON")
    signs = {}
    for cap in ("math", "code", "qa"):
        deltas = [r["losses"][cap]["delta"] for r in rows]
        signs[cap] = {"n_final_configs": len(rows), "positive": sum(x > 0 for x in deltas),
                      "negative": sum(x < 0 for x in deltas), "zero": sum(x == 0 for x in deltas),
                      "min_delta": min(deltas), "max_delta": max(deltas)}
    return {"version": "legacy-v12-native-corpus-token", "final_runs": rows,
            "dependent_snapshots_separate": snapshots, "final_config_signs": signs,
            "scope": "All local final eval.json, no outcome filtering; configs are not independent seeds. Trajectory snapshots are kept separate.",
            "unavailable": "No V27 distilled/dense pair or token-region eval NLL; no V12 per-item byte denominators. Cannot measure distillation three-loss contrasts, body-vs-wrapper attribution, or byte deltas from aggregate eval JSON.",
            "excludes": ["CODE distillation response confined to a separately scored reasoning span: its target is the code answer.",
                         "QA sign caused by adding reference reasoning at evaluation: QA target is answer-only with r empty.",
                         "Universal positive CODE change or universal negative QA change across all available distillation configurations."],
            "does_not_exclude": ["Surface-form/style contribution to CODE or QA losses.",
                                 "Different behavioral accuracy signs; these files contain reference likelihood, not accuracy."]}


def build_summary(results):
    results = Path(results).resolve()
    sources, panels, raw = [], [], {}

    def read(path, kind):
        data = path.read_bytes()
        sources.append({"path": path.relative_to(results.parent).as_posix(), "kind": kind,
                        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        return json.loads(data)

    paths = sorted(results.glob("v27-*/**/*scoring-units.json"))
    if not paths:
        raise FileNotFoundError("No existing results/v27-* scoring-units.json panel")
    for path in paths:
        payload = read(path, "v27_panel")
        if payload["version"] != 27 or payload["metric_definitions"]["MAIN"]["name"] != "L_full":
            raise ValueError("Frozen v27 MAIN must remain L_full")
        if payload["metric_definitions"]["MAIN"]["units"] != "nats/token":
            raise ValueError("Frozen MAIN must retain native token units")
        panel = {"id": path.relative_to(results.parent).as_posix(), "model": payload["model"],
                 "version": 27, "schema_container": "benchmarks" if "benchmarks" in payload else "capabilities",
                 "metric_definitions_original": payload["metric_definitions"], "protocol": payload["protocol"],
                 "probe_sha256": payload["probe_sha256"],
                 "metadata": {k: payload[k] for k in ("resolved_model", "source_checkpoint", "adapter", "prune_density", "quant_bits", "tokenizer")},
                 "benchmarks": {}}
        raw[panel["id"]] = {}
        for key, benchmark, source_path in benchmark_entries(payload):
            row, maps = analyze_benchmark(key, benchmark)
            panel["benchmarks"][key] = {"source_json_path": source_path, **row}
            raw[panel["id"]][key] = maps
        panels.append(panel)
    ordering = ordering_readout(panels, raw)
    return {"version": "v27b-readout-v1", "status": "measurement definitions closed; existing-data limits explicit",
            "measurement_policy": {
                "MAIN": {"version": "v27-native-token", "loss": "L_full", "aggregation": "mean_i(NLL_i / n_tokens_i)", "unit": "nats/token", "frozen": True},
                "loss_labels": LABELS,
                "auxiliary_versions": {"v27-direct": "L_direct: -log p(y|x)", "v27-given": "L_given: -log p(y|x,r); given-reference-reasoning answer loss", "v27-format": "Same context and information; only surface prefixes change", "v27b-example-byte": "mean_i(NLL_i / UTF8_bytes_i)", "v27-corpus-token": "sum_i NLL_i / sum_i n_tokens_i", "v27-corpus-byte": "sum_i NLL_i / sum_i UTF8_bytes_i"},
                "negative_logprob": "NLL_i = -total_logprob_i, natural logarithm; byte count is exact scored target only, excludes context and special tokens",
                "prospective_results": "Original V6/V12/V26/V28 endpoints, cohorts, aggregations, predictions and scores retain their original versions; no refits or replacement.",
                "normalization_warning": "Normalized full-minus-given is not reasoning loss; use item total NLL chain rule.",
                "execution": "CPU standard-library JSON arithmetic only; no inference or tokenizer reconstruction"},
            "panels": panels, "ordering": ordering, "distillation": load_evals(results, read),
            "sources": sources,
            "limitations": ["Local v27 scope is recorded per panel; no inference of missing dense, distillation or quantization cells.",
                            "Shared probe hashes and target bytes control item/target eligibility. Token-limited prompts can retain different text across tokenizers; exact retained prompt text/revisions are not available for full invariance verification.",
                            "Format variants are pre-specified sensitivity checks, never a replacement main endpoint. Their wrapper NLL and body NLL are not separately stored.",
                            "Point estimates describe the fixed panel; reused probes/configurations are not independent replications, and no accuracy claim follows."]}


def fmt(value, signed=False):
    return "N/A" if value is None else format(value, "+.6f" if signed else ".6f")


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(map(cell, row)) + " |" for row in rows)])


def ordering_conclusion(summary):
    entries = summary["ordering"]
    counts = {name: sum(e["comparisons"][name]["ordering_changed"] for e in entries)
              for name in ("equal_example_unit_change", "corpus_unit_change", "main_to_corpus_byte")}
    lines = [f"**Ordering conclusion:** over {len(entries)} benchmark/loss comparisons, "
             f"token→byte ordering changes in {counts['equal_example_unit_change']} at equal-example "
             f"weighting and {counts['corpus_unit_change']} at corpus weighting. Comparing MAIN's "
             f"weighting to corpus bytes changes {counts['main_to_corpus_byte']} orders "
             "(CODE/QA identities are counted separately by loss)."]
    for entry in entries:
        if entry["loss"] != "L_full":
            continue
        for reversal in entry["comparisons"]["main_to_corpus_byte"]["reversals"]:
            a, b = reversal["models"]
            diff = {u: entry["values"][a][u] - entry["values"][b][u] for u in UNITS}
            lines.append(f"{BENCHMARKS[entry['benchmark_key']]}: {a}−{b} is "
                         f"{fmt(diff['example_token'], True)} native nats/token, "
                         f"{fmt(diff['example_byte'], True)} equal-example nats/byte, "
                         f"{fmt(diff['corpus_token'], True)} corpus nats/token, and "
                         f"{fmt(diff['corpus_byte'], True)} corpus nats/byte.")
            if diff["example_token"] * diff["corpus_token"] < 0:
                lines.append("This reversal already occurs when changing example weighting to "
                             "corpus weighting with token units fixed; it does not demonstrate a "
                             "tokenizer-induced reversal.")
    if not counts["equal_example_unit_change"] and not counts["corpus_unit_change"]:
        lines.append("Fixed-weight rankings, including OLMo versus Gemma, are unit-robust in this "
                     "panel. This is not universal measurement invariance.")
    if counts["main_to_corpus_byte"]:
        lines.append("The reversals exclude a claim that every numerical cross-model ranking "
                     "survives a simultaneous unit/aggregation change.")
    return "\n\n".join(lines)


def report(summary):
    rows = [(p["model"], b) for p in summary["panels"] for b in p["benchmarks"].values()]
    sensitivity = []
    for cap in ("math", "code", "qa"):
        bs = [b for _, b in rows if b["capability"] == cap]
        if bs:
            formats = [b["format_control"]["mean_within_item_range_token"] for b in bs]
            conditioning = [b["differences"]["L_given_minus_L_direct"]["example_token"] for b in bs]
            sensitivity.append([cap, f"{fmt(min(formats))} to {fmt(max(formats))}",
                                f"{fmt(min(conditioning), True)} to {fmt(max(conditioning), True)}",
                                "excludes surface invariance; empty-r conditioning identity" if cap != "math" else
                                "excludes equivalence of direct and given-reference-reasoning answer loss"])
    wrapper_example = []
    for model, b in rows:
        if model == "olmo3-7b" and b["benchmark_key"] == "qa":
            change = b["format_control"]["delta_from_canonical"]["answer_prefix"]
            wrapper_example = [f"For a concrete denominator check, OLMo7B/2WikiMultihopQA with "
                               f"`Answer: ` changes mean token loss by {fmt(change['example_token'], True)} "
                               f"nats/token but mean total NLL by {fmt(change['mean_total_nll'], True)} "
                               "nats/example. This excludes treating the normalized change as the "
                               "same change in total sequence probability. The original answer's "
                               "body-only likelihood cannot be recovered from a wrapped-target total.", ""]
    lines = ["# Measurement definitions — v27 closed", "",
             "This is a numerical closeout of existing outputs, before new training. "
             "The MAIN endpoint remains the native-token capability loss: v27 **L_full**, "
             "the equal-example mean of target NLL/token, separately by benchmark. "
             "No model was run, no prediction endpoint was changed, and no auxiliary result was selected as MAIN.", "",
             "Numerical readout across the saved models and primary/secondary benchmarks "
             "(nats/token; the detailed paired contrasts below retain every benchmark):", "",
             table(["Capability", "Mean within-item format range", "Mean given−direct change", "Candidate interpretation excluded"], sensitivity), "",
             "The legacy V6/V12 and prospective results retain their original versions, cohorts, "
             "token units and aggregation. In particular V12 uses a corpus-token ratio; it is "
             "not replaced by v27's equal-example full-solution metric. V27 also retains `legacy_v6` "
             "separately. This document closes definitions and the available readout; missing measurements "
             "are explicitly unavailable, rather than a request to reopen v27 or launch training.", "",
             "Observed input schema: the local files use `benchmarks` (not the anticipated `capabilities` "
             "wrapper), with `metric_definitions`, per-benchmark specs, `scoring`, `format_control`, "
             "`legacy_v6`, per-item NLL/token/byte counts, and reference decompositions. The reader also "
             "accepts a capabilities wrapper containing those benchmark records.", "",
             table(["Model", "Prune density", "Quant bits", "Adapter", "Tokenizer", "Source"],
                   [[p["model"], p["metadata"]["prune_density"], p["metadata"]["quant_bits"],
                     p["metadata"]["adapter"], p["metadata"]["tokenizer"], f"`{p['id']}`"] for p in summary["panels"]]), "",
             f"There are {len(summary['panels'])} saved panels. Their absolute losses alone cannot "
             "establish a compression delta or a QA pruning sign without a matched dense anchor. "
             "Protocol and eligibility are retained explicitly below; counts are benchmark-specific. "
             "Conditioning comparisons always pair the same items, and format comparisons use the "
             "common format/scoring subset. This excludes attributing a within-benchmark contrast "
             "to different selected items.", "",
             table(["Model", "Seed", "Requested", "Probe half", "Max length", "Prompt cap"],
                   [[p["model"], p["protocol"]["probe_seed"], p["protocol"]["n_probe_requested"],
                     p["protocol"]["probe_half"], p["protocol"]["max_len"], p["protocol"]["prompt_max_tokens"]]
                    for p in summary["panels"]]), "",
             table(["Model", "Benchmark", "Scoring / available", "Format n", "Conditioning exclusions", "Extra format exclusions"],
                   [[m, b["benchmark"], f"{b['n']} / {b['spec']['n_measurement_probes']}", b["format_control"]["n"],
                     "; ".join(f"{n}: {reason}" for reason, n in sorted(Counter(x["reason"] for x in b["conditioning_exclusions"]).items())) or "none",
                     len(b["format_control"]["format_exclusions"])] for m, b in rows]), "",
             "The three losses are distinct definitions (natural logarithms, total NLL before normalization):", "",
             table(["Version", "Name", "Total NLL", "Scored region", "Conditioning"], [
                 ["v27-native-token / MAIN", "L_full — full solution loss", "−log p(r,y|x)", "r+y", "x"],
                 ["v27-direct / auxiliary", "L_direct — direct answer loss", "−log p(y|x)", "y", "x"],
                 ["v27-given / auxiliary", "L_given — given-reference-reasoning answer loss", "−log p(y|x,r)", "y", "x,r"]]), "",
             "L_given is still an **answer loss**; its conditioning differs from L_direct. "
             "For MATH-500, y contains the final boxed answer plus punctuation; for GSM8K it contains "
             "the final `####` delimiter and answer. Reference r/y are encoded separately, with identical "
             "y token IDs in all three conditions. MBPP/HumanEval use the whole reference code as y. "
             "2WikiMultihopQA/HotpotQA use the reference answer as y with benchmark context in x. "
             "CODE/QA have r empty: equality of their three losses is an identity, not three independent "
             "confirmations and not evidence that reasoning generally has no value.", "",
             table(["Model", "Capability / benchmark", "n", "L_full MAIN", "L_direct", "L_given", "direct−full", "given−full", "given−direct"],
                   [[m, f"{b['capability']} / {b['benchmark']}", b["n"],
                     *[fmt(b["losses"][l]["original_L_c"]) for l in LOSSES],
                     *[fmt(b["differences"][d]["example_token"], True) for d in ("L_direct_minus_L_full", "L_given_minus_L_full", "L_given_minus_L_direct")]] for m, b in rows]), "",
             "All numbers above are nats/token, equal-example means. The negative given−direct "
             "math contrasts exclude equivalence of the two answer-conditioning questions on this "
             "panel. A lower normalized full loss does not imply that producing reasoning is free: "
             "full and answer-only losses have different target lengths. Never interpret their "
             "normalized difference as reasoning NLL.", "",
             "True format control is separate. The context and reference body are fixed; only the "
             "outer prefix changes to a newline, blank line, `Answer: `, or `The answer is `. "
             "Internal code whitespace stays fixed. All saved controls condition as L_full. "
             "Entries below are paired mean variant−canonical changes, with the mean within-item "
             "max−min range over all five pre-specified variants. The range is a sensitivity "
             "statistic, not a rule for choosing the best format. Given−direct is recomputed on "
             "the format cohort; the final column is its mean absolute paired change.", "",
             table(["Model", "Benchmark", "newline Δ", "blank line Δ", "Answer: Δ", "The answer is Δ", "format range", "given−direct Δ", "mean |given−direct|"],
                   [[m, b["benchmark"], *[fmt(b["format_control"]["delta_from_canonical"][v]["example_token"], True) for v in list(PREFIXES)[1:]],
                     fmt(b["format_control"]["mean_within_item_range_token"]),
                     fmt(b["format_control"]["given_minus_direct_same_cohort"]["example_token"], True),
                     fmt(b["format_control"]["mean_absolute_conditioning_delta_token"])] for m, b in rows]), "",
             "These numbers exclude perfect surface invariance. They do not identify a causal share "
             "of distillation damage: there are no paired dense/distilled format panels. In math the "
             "format control scores r+y, while the conditioning contrast scores y, so their normalized "
             "magnitudes are descriptive sensitivities on different target regions. In CODE/QA, the "
             "format movement coexists with exactly zero conditioning movement because r is empty. "
             "Prefix prediction and denominator dilution are included; body-only or syntax/semantic "
             "NLL is unavailable. To expose denominator effects, the following are mean **total** "
             "NLL changes (nats/example), before length normalization:", "",
             table(["Model", "Benchmark", "newline Δ total", "blank line Δ total", "Answer: Δ total", "The answer is Δ total", "given−direct Δ total"],
                   [[m, b["benchmark"], *[fmt(b["format_control"]["delta_from_canonical"][v]["mean_total_nll"], True) for v in list(PREFIXES)[1:]],
                     fmt(b["format_control"]["given_minus_direct_same_cohort"]["mean_total_nll"], True)] for m, b in rows]), "",
             *wrapper_example,
             "Byte versions use each sample's saved total negative log probability and **exact UTF-8 "
             "bytes of its scored target**: b_i=len(target.encode('utf-8')). The readout verifies "
             "byte counts and target hashes against the saved r/y strings and prefixes. Context/BOS/EOS "
             "bytes are excluded. It does not convert an aggregate per-token mean into bytes. "
             "Both weighting conventions are explicit: equal-example token=mean_i(NLL_i/n_i), "
             "equal-example byte=mean_i(NLL_i/b_i), corpus token=ΣNLL_i/Σn_i, corpus byte=ΣNLL_i/Σb_i. "
             "The corpus-byte version is the existing v27 byte convention; the equal-example byte "
             "version isolates a unit change while keeping MAIN's example weighting.", "",
             table(["Model", "Benchmark", "Loss", "Native token mean", "Byte mean", "Corpus token", "Corpus byte", "ΣNLL", "Σtokens", "Σbytes"],
                   [[m, b["benchmark"], l, *[fmt(b["losses"][l][u]) for u in UNITS],
                     fmt(b["losses"][l]["sum_nll"]), b["losses"][l]["sum_tokens"], b["losses"][l]["sum_bytes"]]
                    for m, b in rows for l in LOSSES]), "",
             "Cross-model ordering uses the intersection of probe identities with equal scored-target "
             "hashes and bytes, separately by benchmark/loss and compression configuration. All eligible "
             "items are shared in this panel. Lower loss sorts first; ties are retained explicitly. "
             "The table groups identical CODE/QA loss identities; summary.json retains every loss.", ""]
    order_rows = []
    for entry in summary["ordering"]:
        if entry["loss"] != "L_full" and not entry["benchmark_key"].startswith("math"):
            continue
        c = entry["comparisons"]
        def order(name, side):
            return " < ".join(" = ".join(g) for g in c[name][side + "_order_low_to_high"])
        order_rows.append([BENCHMARKS[entry["benchmark_key"]], entry["loss"] if entry["benchmark_key"].startswith("math") else "all three",
                           order("equal_example_unit_change", "left"), order("equal_example_unit_change", "right"),
                           order("corpus_unit_change", "right"),
                           c["equal_example_unit_change"]["ordering_changed"], c["corpus_unit_change"]["ordering_changed"]])
    lines += [table(["Benchmark", "Loss", "Equal-example token order", "Equal-example byte order", "Corpus byte order", "Example unit flip?", "Corpus unit flip?"], order_rows), "",
              ordering_conclusion(summary), "",
              "Token-limited "
              "prompts can retain different text across tokenizers, and exact retained prompt text "
              "and immutable tokenizer revisions were not saved.", "",
              "Region accounting uses NLL_full−NLL_given **before normalization** to recover "
              "−log p(r|x). Answer NLL is the saved given-reference-reasoning answer NLL. "
              "Fractions are shares of full total NLL at these pruned checkpoints, not shares "
              "of a distillation or pruning delta.", "",
              table(["Model", "Benchmark", "Reasoning ΣNLL", "Answer ΣNLL", "Answer fraction", "Reasoning nats/token"],
                    [[m, b["benchmark"], fmt(b["regions"]["reasoning_sum_nll"]), fmt(b["regions"]["answer_sum_nll"]),
                      fmt(b["regions"]["answer_fraction_full_nll"]),
                      fmt(b["regions"]["reasoning"]["example_token"] if b["regions"]["reasoning"] else None)] for m, b in rows]), "",
              "For all CODE/QA rows the answer fraction is 1.000000 and reasoning total is 0.000000: "
              "the relevant region is the whole code answer or short QA answer, respectively. "
              "This excludes a separately scored reasoning-span explanation for the CODE response "
              "or QA sign. It does not locate effects within code syntax, whitespace, semantics, "
              "or particular QA answer tokens; those per-token likelihoods are absent.", "",
              "Distillation is read from the existing V12 eval JSON in its **original corpus-token "
              "version**, subtracting each run's own dense anchor. These are not v27 measurements. "
              "`answer_only` and `no_code_fence` below name training recipes, not different evaluation "
              "losses and not fixed-model format controls. All final configurations are reported, "
              "including sign exceptions; dependent trajectory snapshots are preserved separately "
              "in summary.json and do not enter the final-configuration counts.", ""]
    distill = summary["distillation"]
    lines += [table(["Capability", "Final configs", "Positive Δ", "Negative Δ", "Zero Δ", "Min Δ", "Max Δ"],
                    [[c, s["n_final_configs"], s["positive"], s["negative"], s["zero"], fmt(s["min_delta"], True), fmt(s["max_delta"], True)]
                     for c, s in distill["final_config_signs"].items()]), "",
              table(["Model", "Run", "CODE dense", "CODE post", "CODE Δ", "QA dense", "QA post", "QA Δ", "CODE / QA target tokens"],
                    [[r["model"], r["run"], *[fmt(r["losses"][c][v], v == "delta") for c in ("code", "qa") for v in ("dense", "post", "delta")],
                      f"{r['measurement_tokens']['code']} / {r['measurement_tokens']['qa']}"] for r in distill["final_runs"]]), "",
              "For a compact size/family view, the original gpt-5.6-luna/full/600 run in each "
              "available model is shown below (native V12 corpus nats/token). This is a named "
              "configuration slice; it does not replace the complete table or select by outcome.", "",
              table(["Model", "CODE Δ", "QA Δ"],
                    [[r["model"], fmt(r["losses"]["code"]["delta"], True), fmt(r["losses"]["qa"]["delta"], True)]
                     for r in distill["final_runs"] if r["run"] == "gpt-5.6-luna_full_600"]), "",
              "**CODE response:** the sign exceptions above exclude a universal positive CODE "
              "response across the observed models/configurations. The response concerns the code "
              "answer target, not a scored rationale. The `no_code_fence` training recipe's delta "
              "is retained in the complete table; that comparison changes training and cannot "
              "isolate a pure evaluation-format contribution.", "",
              "**QA sign:** the complete-table positive exceptions exclude an unconditional "
              "negative-QA rule. The empty-r identity excludes an evaluation-reasoning-conditioning "
              "explanation. Short QA targets (see the recorded token counts above) and the measured "
              "format sensitivity leave surface/answer-token effects plausible. Neither sign "
              "establishes a behavioral accuracy improvement or decline.", "",
              "**Identification limit at closeout:** no v27 dense/distilled pair exists locally; "
              "V12 eval JSON contains aggregate losses, not item NLL/byte lengths or token-region "
              "losses. Therefore the separate **numerical distillation deltas** of the three v27 "
              "losses, any body-versus-wrapper causal share, and V12 per-byte distillation deltas "
              "are unavailable (JSON null), not zero. The empty-r identity locates CODE/QA at "
              "the answer target but does not reconstruct new v27 measurements. Borrowing bytes "
              "or dense anchors from the pruning panel would conflate versions and is refused. "
              "The measurement definitions are closed with this boundary; the original prospective "
              "results stay in their original version.", "",
              "Reproduce using only the standard library:", "", "```bash",
              "CUDA_VISIBLE_DEVICES='' python analysis/v27b_scoring_readout.py",
              "python -m pytest -q tests/test_v27b.py", "```", "",
              "Outputs: `results/v27b-readout/summary.json` and this document. The JSON preserves "
              "unrounded values, source metric definitions, exclusions, all format variants in "
              "both units/weightings, same-item ordering comparisons, all V12 losses, and SHA-256 "
              "hashes of every input. Input files and prediction artifacts are never written. "
              "The source panel is small and fixed; no independent-seed uncertainty or general "
              "measurement-invariance claim is inferred from these descriptive values.", "",
              "Source inventory (raw-file SHA-256; paths relative to repository root):", "",
              table(["Kind", "Source", "SHA-256"], [[s["kind"], f"`{s['path']}`", s["sha256"]] for s in summary["sources"]]), ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results/v27b-readout/summary.json")
    parser.add_argument("--report", type=Path, default=ROOT / "paper/docs/MEASUREMENT_DEFINITIONS.md")
    parser.add_argument("--dry-run", action="store_true", help="Validate/read all inputs without writing")
    args = parser.parse_args(argv)
    summary = build_summary(args.results)
    if args.dry_run:
        print(json.dumps({"inputs": len(summary["sources"]), "panels": len(summary["panels"]), "writes": False}))
        return summary
    protected = {(args.results.parent / s["path"]).resolve() for s in summary["sources"]}
    outputs = [args.output.resolve(), args.report.resolve()]
    if len(set(outputs)) != 2 or any(p in protected for p in outputs):
        raise ValueError("Output paths must be distinct and must not overwrite inputs")
    # Restrict results writes to the readout version, including custom invocations.
    if args.output.resolve().parent != (args.results / "v27b-readout").resolve():
        raise ValueError("JSON output must be under results/v27b-readout")
    if args.report.suffix != ".md":
        raise ValueError("Report output must be Markdown")
    rendered = report(summary)
    for path, content in ((args.output, json.dumps(summary, indent=2, allow_nan=False) + "\n"),
                          (args.report, rendered)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"Read {len(summary['sources'])} existing JSON files; wrote {args.output} and {args.report}")
    return summary


if __name__ == "__main__":
    main()
