"""Arithmetic, pairing, units and preservation checks; no scoring/model calls."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from analysis import v27b_scoring_readout as readout


def group(items):
    u = readout.recompute(items)
    return {"items": items, "L_c": u["example_token"], "token_weighted_L_c": u["corpus_token"],
            "units_v27_v1": {"sum_nll": u["sum_nll"], "sum_target_tokens": u["sum_tokens"],
                             "sum_target_bytes": u["sum_bytes"], "n_examples": u["n"],
                             "example_token_normalized": u["example_token"],
                             "token_normalized": u["corpus_token"], "byte_normalized": u["corpus_byte"]}}


def benchmark(cap="math"):
    identity = {"measurement_index": 0, "probe_sha256": "same-probe"}
    r, y = ("ré ", "é2") if cap == "math" else ("", "é2")

    def item(text, total, tokens, **extra):
        return {**identity, "sum_ce": total, "n_tokens": tokens,
                "target_bytes": len(text.encode("utf-8")), "target_sha256": readout.digest(text), **extra}

    full = item(r + y, 14 if r else 4, 5 if r else 2)
    scoring = {"L_full": group([full]), "L_direct": group([item(y, 8 if r else 4, 2)]),
               "L_given": group([item(y, 4, 2)])}
    variants = {}
    for name, prefix in readout.PREFIXES.items():
        variants[name] = group([item(prefix + r + y, full["sum_ce"] + len(prefix),
                                    full["n_tokens"] + len(prefix),
                                    information_sha256=readout.digest(r + y),
                                    context_tokens_sha256="unchanged-context")])
    values = {name: g["L_c"] for name, g in variants.items()}
    span = max(values.values()) - min(values.values())
    return {"capability": cap, "benchmark": readout.BENCHMARKS[cap], "dataset": ["fixture", None, "test"],
            "n_measurement_probes": 1, "decomposition_policy": "fixture exact r+y",
            "probe_decompositions": [{**identity, "r": r, "y": y, "supported": True, "reconstruction_exact": True}],
            "conditioning_exclusions": [], "scoring": scoring, "legacy_v6": copy.deepcopy(scoring["L_full"]),
            "format_control": {"conditioning": "L_full", "surface_prefixes": readout.PREFIXES,
                               "variants": variants, "format_exclusions": [],
                               "mean_within_item_range_nats_per_token": span,
                               "paired_items": [{"measurement_index": 0, "range_nats_per_token": span,
                                                 "delta_from_canonical": {name: v - values["canonical"] for name, v in values.items()}}]}}


def test_utf8_bytes_and_weightings_are_independently_computed():
    # Unequal lengths prevent a mean-of-ratios / ratio-of-totals substitution.
    result = readout.recompute([{"sum_ce": 1, "n_tokens": 1, "target_bytes": 2},
                                {"sum_ce": 27, "n_tokens": 9, "target_bytes": 10}])
    assert result["example_token"] == 2
    assert result["corpus_token"] == 2.8
    assert result["example_byte"] == 1.6
    assert result["corpus_byte"] == pytest.approx(28 / 12)
    b = benchmark()
    row, _ = readout.analyze_benchmark("math", b)
    assert row["losses"]["L_full"]["sum_bytes"] == 7  # 5 chars, 7 UTF-8 bytes
    assert row["losses"]["L_full"]["example_byte"] == 2
    bad = copy.deepcopy(b)
    bad["scoring"]["L_full"]["items"][0]["target_bytes"] = 5
    with pytest.raises(ValueError, match="UTF-8 byte"):
        readout.analyze_benchmark("math", bad)


@pytest.mark.parametrize("items", [[], [{"L_c": 3, "n_tokens": 3, "target_bytes": 9}],
                                    [{"sum_ce": 3, "n_tokens": 3}],
                                    [{"sum_ce": -1, "n_tokens": 3, "target_bytes": 9}],
                                    [{"sum_ce": float("nan"), "n_tokens": 3, "target_bytes": 9}],
                                    [{"sum_ce": 3, "n_tokens": True, "target_bytes": 9}],
                                    [{"sum_ce": 3, "n_tokens": 3, "target_bytes": 0}]])
def test_refuse_aggregate_conversion_missing_lengths_and_invalid_values(items):
    with pytest.raises(ValueError):
        readout.recompute(items)


def test_three_loss_pairing_chain_rule_and_format_are_separate():
    b = benchmark()
    original = copy.deepcopy(b)
    row, _ = readout.analyze_benchmark("math", b)
    assert b == original
    assert row["differences"]["L_given_minus_L_direct"]["example_token"] == -2
    assert row["regions"]["reasoning_sum_nll"] == 10
    assert row["regions"]["reasoning"]["example_token"] == pytest.approx(10 / 3)
    assert row["losses"]["L_full"]["example_token"] - row["losses"]["L_given"]["example_token"] != 10 / 3
    assert row["format_control"]["delta_from_canonical"]["newline"]["example_token"] == pytest.approx(-0.3)
    assert row["format_control"]["delta_from_canonical"]["newline"]["mean_total_nll"] == 1
    assert row["format_control"]["given_minus_direct_same_cohort"]["example_token"] == -2
    assert readout.LABELS["L_given"] == "given-reference-reasoning answer loss"


@pytest.mark.parametrize("cap", ["code", "qa"])
def test_empty_reasoning_is_an_identity_not_a_format_control(cap):
    row, _ = readout.analyze_benchmark(cap, benchmark(cap))
    assert row["losses"]["L_full"] == row["losses"]["L_direct"] == row["losses"]["L_given"]
    assert row["regions"]["reasoning_sum_nll"] == 0
    assert row["regions"]["answer_fraction_full_nll"] == 1
    assert row["regions"]["reasoning"] is None
    assert row["format_control"]["mean_within_item_range_token"] > 0
    assert row["format_control"]["mean_absolute_conditioning_delta_token"] == 0


@pytest.mark.parametrize("mutation,match", [
    (lambda b: b["scoring"]["L_given"]["items"][0].update(probe_sha256="different"), "paired cohorts"),
    (lambda b: b["scoring"]["L_full"].update(L_c=100), "mismatch"),
    (lambda b: b["format_control"]["variants"]["newline"]["items"][0].update(information_sha256="changed"), "information or context"),
    (lambda b: b["format_control"]["variants"]["newline"]["items"][0].update(context_tokens_sha256="changed"), "information or context"),
    (lambda b: b["format_control"]["variants"]["newline"]["items"][0].update(probe_sha256="changed"), "paired cohorts"),
    (lambda b: b["format_control"]["paired_items"][0].update(range_nats_per_token=999), "paired format range"),
])
def test_refuse_stale_aggregates_and_unpaired_or_information_changing_controls(mutation, match):
    b = benchmark()
    mutation(b)
    with pytest.raises(ValueError, match=match):
        readout.analyze_benchmark("math", b)


def test_both_panel_containers_preserve_benchmarks():
    b = benchmark()
    assert readout.benchmark_entries({"benchmarks": {"math": b}})[0][0] == "math"
    wrapped = {"capabilities": {"math": {"primary": b}}}
    key, result, path = readout.benchmark_entries(wrapped)[0]
    assert key == "math" and result is b and path == "capabilities/math/primary"
    with pytest.raises(ValueError, match="duplicate"):
        readout.benchmark_entries({"capabilities": {"math": {"primary": b, "secondary": b}}})


def test_eligibility_and_exclusions_cannot_silently_drop_probes():
    b = benchmark()
    b["n_measurement_probes"] += 1
    with pytest.raises(ValueError, match="partition measurement probes"):
        readout.analyze_benchmark("math", b)


def test_ordering_distinguishes_unit_flips_weighting_flips_and_ties():
    values = {"a": {"example_token": 2, "example_byte": 1, "corpus_token": 3, "corpus_byte": 1.5},
              "b": {"example_token": 2.1, "example_byte": 1.1, "corpus_token": 2.9, "corpus_byte": 1.4}}
    assert not readout.order_comparison(values, "example_token", "example_byte")["ordering_changed"]
    assert readout.order_comparison(values, "example_token", "corpus_byte")["ordering_changed"]
    assert readout.order_comparison(values, "example_token", "corpus_token")["ordering_changed"]
    values["b"]["example_byte"] = 0.9
    assert readout.order_comparison(values, "example_token", "example_byte")["reversals"]
    values["b"]["example_byte"] = 1
    result = readout.order_comparison(values, "example_token", "example_byte")
    assert result["right_order_low_to_high"] == [["a", "b"]]
    assert result["tie_changes"] and not result["reversals"]


def test_cross_model_ordering_intersects_cohorts_and_rejects_different_text():
    panels, raw = [], {}
    for model in ("a", "b"):
        _, maps = readout.analyze_benchmark("math", benchmark())
        raw[model] = {"math": maps}
        panels.append({"id": model, "model": model, "benchmarks": {"math": {}},
                       "metadata": {"adapter": None, "prune_density": 0.7, "quant_bits": None, "tokenizer": model},
                       "protocol": {}, "probe_sha256": "same-probe-list"})
    for loss in readout.LOSSES:
        extra = copy.deepcopy(next(iter(raw["a"]["math"][loss].values())))
        extra.update(measurement_index=1, probe_sha256="only-a")
        raw["a"]["math"][loss][(1, "only-a")] = extra
    result = readout.ordering_readout(panels, raw)
    assert result[0]["n_common"] == 1
    assert result[0]["original_n"] == {"a": 2, "b": 1}
    raw["b"]["math"]["L_full"][(0, "same-probe")]["target_sha256"] = "different-text"
    with pytest.raises(ValueError, match="target text/bytes"):
        readout.ordering_readout(panels, raw)


@pytest.fixture(scope="module")
def actual_summary():
    if not (readout.ROOT / "results/v27-scoring-units").exists():
        pytest.skip("Local saved measurement panel is not bundled with this checkout")
    return readout.build_summary(readout.ROOT / "results")


def test_actual_panel_reproduces_numbers_units_and_frozen_main(actual_summary):
    s = actual_summary
    assert s["measurement_policy"]["MAIN"]["loss"] == "L_full"
    assert s["measurement_policy"]["MAIN"]["aggregation"] == "mean_i(NLL_i / n_tokens_i)"
    p = next(p for p in s["panels"] if p["model"] == "gemma3-1b")
    math = p["benchmarks"]["math"]
    assert math["n"] == 35
    assert len(math["conditioning_exclusions"]) == 29
    assert math["losses"]["L_full"]["original_L_c"] == pytest.approx(2.78820897891953)
    assert math["losses"]["L_direct"]["example_token"] == pytest.approx(7.253992894872929)
    assert math["losses"]["L_given"]["example_token"] == pytest.approx(6.576104020263012)
    assert math["losses"]["L_full"]["corpus_byte"] == pytest.approx(1.0882351114090068)
    qa = p["benchmarks"]["qa"]
    assert qa["format_control"]["mean_within_item_range_token"] == pytest.approx(4.205446, abs=5e-7)
    for entry in s["ordering"]:
        comparisons = entry["comparisons"]
        assert not comparisons["equal_example_unit_change"]["ordering_changed"]
        assert not comparisons["corpus_unit_change"]["ordering_changed"]
        assert comparisons["main_to_corpus_byte"]["ordering_changed"] == (entry["benchmark_key"] == "code")


def test_actual_eval_sign_exceptions_and_missing_regions_stay_explicit(actual_summary):
    d = actual_summary["distillation"]
    runs = {(r["model"], r["run"]): r for r in d["final_runs"]}
    r = runs["gemma3-1b", "gpt-5.6-luna_full_600"]
    assert r["losses"]["code"]["delta"] == pytest.approx(0.13831377325066452)
    assert r["losses"]["qa"]["delta"] == pytest.approx(-0.7473232071713145)
    assert r["losses"]["code"]["three_loss_v27_delta"] is None
    assert r["losses"]["qa"]["byte_delta"] is None
    assert runs["Qwen3-1.7B", "gpt-5.6-luna_full_600"]["losses"]["code"]["delta"] < 0
    assert runs["gemma3-1b", "gpt-5.6-luna_full_75_uxseen"]["losses"]["qa"]["delta"] > 0
    assert all("/trajectory/" not in r["source"] for r in d["final_runs"])
    assert all("/trajectory/" in r["source"] for r in d["dependent_snapshots_separate"])


def test_cli_is_cpu_only_deterministic_and_does_not_write_inputs(tmp_path, actual_summary):
    # A minimal copied real cohort tests executable I/O without touching source data.
    selected = [s for s in actual_summary["sources"] if s["kind"] == "v27_panel"]
    selected += [next(s for s in actual_summary["sources"] if s["path"].endswith("gemma3-1b/gpt-5.6-luna_full_600/eval.json"))]
    for source in selected:
        target = tmp_path / source["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((readout.ROOT / source["path"]).read_bytes())
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*.json")}
    output = tmp_path / "results/v27b-readout/summary.json"
    report = tmp_path / "MEASUREMENT_DEFINITIONS.md"
    argv = ["--results", str(tmp_path / "results"), "--output", str(output), "--report", str(report)]
    # Import guard fails if the readout starts reaching for scoring dependencies.
    script = '''
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'datasets', 'numpy', 'huggingface_hub'}:
        raise AssertionError('Forbidden model/data dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from analysis.v27b_scoring_readout import main
main(sys.argv[1:])
'''
    command = [sys.executable, "-c", script, *argv]
    subprocess.run([*command, "--dry-run"], check=True, cwd=readout.ROOT, capture_output=True)
    assert not output.exists() and not report.exists()
    subprocess.run(command, check=True, cwd=readout.ROOT, capture_output=True)
    first = (output.read_bytes(), report.read_bytes())
    subprocess.run(command, check=True, cwd=readout.ROOT, capture_output=True)
    assert first == (output.read_bytes(), report.read_bytes())
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    assert json.loads(output.read_text())["measurement_policy"]["MAIN"]["frozen"] is True
    with pytest.raises(ValueError, match="overwrite inputs"):
        readout.main(["--results", str(tmp_path / "results"), "--output", str(next(iter(before))), "--report", str(report)])
    panel_path = next(p for p in before if "v27-scoring-units" in p.parts)
    payload = json.loads(panel_path.read_text())
    payload["metric_definitions"]["MAIN"]["name"] = "L_given"
    panel_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="MAIN must remain L_full"):
        readout.build_summary(tmp_path / "results")
