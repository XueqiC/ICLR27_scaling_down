"""Retrospective A19 control arithmetic, denominators and exact table reproduction.

These tests read frozen inputs; all test writes go to pytest temporary directories.
Run the standalone paper mirror with --noconftest to avoid its unrelated bootstrap.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from analysis import s3_control_table as gen

ROOT = Path(__file__).resolve().parents[1]
# Set from the reviewed retrospective table, checked even without manuscript files.
TABLE_SHA256 = "ea5b4f8dc1ca7d57e7cd9314ddc1a03e1c3bd6bdcd8d9a9809baf0f8e9e918fc"


@pytest.fixture(scope="module")
def result():
    return gen.analyze(ROOT)


def candidate(cid, method, ratio, losses):
    return dict(id=cid, method=method, r=ratio,
                actual=dict(zip(gen.CAPS, losses)))


def frozen(rule, quant):
    return {"rule": {"selected": rule}, "quant-only": {"selected": quant}}


def test_hand_calculated_signed_and_retained_decompositions():
    configs = [candidate("q", "quant", .25, [4, 4, 4]),
               candidate("qbad", "quant", .25, [6, 6, 6]),
               candidate(gen.KD_ID, "distill", .25, [2, 2, 2]),
               candidate("pruned", "prune", .9, [.5, .5, .5]),
               candidate("dense:source", "dense", 1., [.25, .25, .25])]
    s0 = candidate(gen.S0_ID, "dense", .25, [1, 1, 1])
    row = gen.evaluate_cell(configs, s0, dict.fromkeys(gen.CAPS, .25),
                            "math", 1., frozen(gen.KD_ID, "qbad"))
    assert row["decomposition"] == {
        "s0_opportunity": 3., "distillation_change": -1.,
        "other_registered_methods": 1.75, "registered_opportunity": 3.75,
        "distillation_added_retaining_s0": 0., "other_methods_retaining_s0": .75,
        "augmented_opportunity": 3.75,
    }
    assert row["policies"]["rule"]["regret_registered"] == 1.75
    assert row["policies"]["quant-only"]["regret_registered"] == 5.75
    # The quantization policy's regret includes a within-quantization error of 2.
    assert row["policies"]["quant-only"]["regret_registered"] != row["decomposition"]["registered_opportunity"]
    assert row["oracles"]["quantization_plus_s0"]["candidate"] == gen.S0_ID


def test_largest_increase_is_maximum_before_minimization():
    configs = [candidate("q", "quant", .25, [1, 6, 4]),
               candidate(gen.KD_ID, "distill", .25, [3, 4, 3])]
    s0 = candidate(gen.S0_ID, "dense", .25, [4, 5, 5])
    row = gen.evaluate_cell(configs, s0, dict(zip(gen.CAPS, [1, 2, 3])),
                            "multi", .25, frozen(gen.KD_ID, "q"))
    assert row["oracles"]["quantization"]["loss"] == 4
    assert row["oracles"]["registered"]["loss"] == 2
    assert row["decomposition"]["registered_opportunity"] == 2


def test_storage_tolerance_and_exact_tie_priority():
    q = candidate("q", "quant", .25 + gen.EPS / 2, [1, 1, 1])
    assert gen.feasible([q], .25) == [q]
    q["r"] = .25 + 2 * gen.EPS
    assert gen.feasible([q], .25) == []
    configs = [candidate(m, m, .2, [1, 1, 1]) for m in reversed(gen.METHODS)]
    dense = dict.fromkeys(gen.CAPS, 1)
    for method in gen.METHODS:
        assert gen.choose(configs, "math", dense)["method"] == method
        configs = [q for q in configs if q["method"] != method]
    # Ratios precede ids; loss comparisons have no epsilon tolerance.
    configs = [candidate("a", "quant", .3, [1, 1, 1]),
               candidate("z", "quant", .25, [1, 1, 1]),
               candidate("b", "quant", .25, [1, 1, 1])]
    assert gen.choose(configs, "math", dense)["id"] == "b"
    configs[0]["actual"]["math"] -= 1e-13
    assert gen.choose(configs, "math", dense)["id"] == "a"
    assert gen.choose([], "math", dense) is None


def test_nominal_storage_uses_plan_matrix_counts(result):
    expected = [(84934656, 301989888, 15), (301989888, 1207959552, 16),
                (100270080, 697761792, 17), (697761792, 3208642560, 16)]
    for ref, (ns, nr, count) in zip(result["references"], expected):
        st = ref["storage"]
        assert (st["student_N0"], st["reference_N0"]) == (ns, nr)
        assert st["s0_ratio"] == st["distilled_ratio"] == ns / nr
        assert st["n_student_feasible"] == count
        assert st["n_budgets"] == 17
        assert len(st["student_feasible_budgets"]) + len(st["student_infeasible_budgets"]) == 17
        s0 = next(q for q in ref["candidates"] if q["id"] == gen.S0_ID)
        assert s0["method"] == "dense" and s0["retrospective"]
        assert gen.S0_ID not in ref["candidate_sets"]["registered"]
    # Reject the tempting model-name 160/410 convention even if it looks plausible.
    ref = {"N0": 301989888, "student": {"N0": 84934656},
           "candidates": [{"id": gen.KD_ID, "method": "distill", "r": 160 / 410}]}
    with pytest.raises(ValueError, match="storage differs"):
        gen.pristine_candidate(ref, {"losses": dict.fromkeys(gen.CAPS, 1)})


def test_measured_transfer_values_and_exact_source_files(result):
    expected = [(.0701969469271535, .12253387211789901, -.1404467680608361),
                (.0969706556988057, .15337532683622523, -.537547528517111),
                (.09533641460827402, .16817980513728958, -.6502427788844622),
                (.12285952417032897, .1426317537643933, -.7747447709163344)]
    directory = gen.input_directory(ROOT)
    for ref, values in zip(result["references"], expected):
        for cap, target in zip(gen.CAPS, values):
            t = ref["transfer"][cap]
            assert t["distilled_minus_s0"] == pytest.approx(target, abs=1e-14)
            assert t["distilled_minus_s0"] == t["distilled_loss"] - t["s0_loss"]
            assert t["anchor_file"] == f"results/{gen.EXPERIMENT}/anchors/{ref['reference']}__student.json"
            assert t["measurement_file"] == f"results/{gen.EXPERIMENT}/students/{ref['reference']}/measurement.json"
            for key, value in (("anchor_file", "s0_loss"), ("measurement_file", "distilled_loss")):
                source = directory / t[key].removeprefix(f"results/{gen.EXPERIMENT}/")
                assert json.loads(source.read_text())["losses"][cap] == t[value]
            assert t["units"] == "nats per native " + ref["family"] + " token"


def test_infeasible_budgets_are_null_and_frozen_selection_is_not_repaired():
    configs = [candidate("q", "quant", .21875, [3, 3, 3]),
               candidate(gen.KD_ID, "distill", .14, [2, 2, 2])]
    s0 = candidate(gen.S0_ID, "dense", .14, [1, 1, 1])
    row = gen.evaluate_cell(configs, s0, dict.fromkeys(gen.CAPS, 3),
                            "code", .2, frozen(gen.KD_ID, None))
    assert row["oracles"]["quantization"]["loss"] is None
    assert row["oracles"]["quantization_plus_s0"]["loss"] == 1
    assert row["decomposition"]["registered_opportunity"] is None
    assert row["policies"]["quant-only"]["regret_registered"] is None
    assert row["policies"]["rule"]["selected"] == gen.KD_ID
    assert row["policies"]["rule"]["regret_registered"] == 0
    assert row["policies"]["rule"]["regret_registered_plus_s0"] == 1


def test_every_denominator_and_excluded_budget(result):
    assert result["budgets"] == [i / 100 for i in range(20, 101, 5)]
    assert result["n_cells"] == len(result["rows"]) == 4 * 4 * 17
    for ref in result["references"]:
        for obj in ref["objectives"]:
            assert obj["n_budgets"] == 17
            cells = [r for r in result["rows"] if r["reference"] == ref["reference"]
                     and r["objective"] == obj["objective"]]
            paired = [r for r in cells if r["budget"] >= .25]
            common = obj["common_denominator"]
            assert common["n_included"] == 16 and common["n_total"] == 17
            assert common["included_budgets"] == gen.BUDGETS[1:]
            assert [x["budget"] for x in common["excluded"]] == [.2]
            assert "quantization" in common["excluded"][0]["reason"]
            for name in gen.SETS:
                expected = 17 if ref["reference"] == "gemma3-1b" and name != "quantization" else 16
                d = obj["candidate_sets"][name]["denominator"]
                assert d["n_total"] == 17 and d["n_included"] == expected
                assert d["included_budgets"] == gen.BUDGETS[17 - expected:]
                assert [x["budget"] for x in d["excluded"]] == ([] if expected == 17 else [.2])
                assert all("r <= budget" in x["reason"] for x in d["excluded"])
                assert obj["candidate_sets"][name]["mean_over_common_budgets"] == pytest.approx(
                    sum(r["oracles"][name]["loss"] for r in paired) / 16, abs=1e-14)
                assert obj["candidate_sets"][name]["mean_over_feasible_budgets"] == pytest.approx(
                    sum(r["oracles"][name]["loss"] for r in cells if r["oracles"][name]["feasible"]) / expected, abs=1e-14)
            for policy, values in obj["policies"].items():
                expected = 17 if ref["reference"] == "gemma3-1b" and policy == "rule" else 16
                assert values["denominator"]["n_included"] == expected
                assert values["common_denominator"] == common
                for baseline in ("registered", "registered_plus_s0"):
                    assert values[f"mean_regret_{baseline}"] == pytest.approx(
                        sum(r["policies"][policy][f"regret_{baseline}"] for r in paired) / 16, abs=1e-14)
            for name in gen.DECOMPOSITION:
                assert obj["mean_decomposition"][name] == pytest.approx(
                    sum(r["decomposition"][name] for r in paired) / 16, abs=1e-14)


def test_all_cell_and_mean_decompositions_close(result):
    for d in ([r["decomposition"] for r in result["rows"]] +
              [o["mean_decomposition"] for r in result["references"] for o in r["objectives"]]):
        if d["registered_opportunity"] is None:
            continue
        assert d["registered_opportunity"] == pytest.approx(
            d["s0_opportunity"] + d["distillation_change"] + d["other_registered_methods"], abs=1e-14)
        assert d["augmented_opportunity"] == pytest.approx(
            d["s0_opportunity"] + d["distillation_added_retaining_s0"] + d["other_methods_retaining_s0"], abs=1e-14)
        assert d["distillation_added_retaining_s0"] >= 0
    expected_qa = [.083639050790147, .537547528517111, .6502427788844622, .220057270672870]
    for ref, target in zip(result["references"], expected_qa):
        for obj in ref["objectives"]:
            d = obj["mean_decomposition"]
            if obj["objective"] == "qa":
                assert d["distillation_added_retaining_s0"] == pytest.approx(target, abs=1e-9)
            else:
                assert d["distillation_added_retaining_s0"] == 0
                assert d["distillation_change"] <= 0


def test_registered_scoring_matches_shared_contract_when_available(result):
    # The public generator remains standalone. In the working repository compare
    # every cell directly with the original module, without importing model code.
    paths = [ROOT / "analysis/s3_common.py", ROOT.parent / "analysis/s3_common.py"]
    source = next((p for p in paths if p.is_file()), None)
    if source is None:
        assert gen.METHODS == ("prune", "quant", "distill", "dense")
        return  # Hand-calculated contract tests above remain unconditional.
    spec = importlib.util.spec_from_file_location("a19_frozen_common", source)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    refs = {r["reference"]: r for r in result["references"]}
    for row in result["rows"]:
        ref = refs[row["reference"]]
        configs = [q for q in ref["candidates"] if q["id"] != gen.S0_ID]
        dense = next(q["actual"] for q in configs if q["id"] == "dense:source")
        policies = row["policies"]
        baseline = common.metrics(configs, row["objective"], dense, row["budget"],
                                  policies["rule"]["selected"], policies["quant-only"]["selected"])
        assert row["decomposition"]["registered_opportunity"] == baseline["opportunity"]
        assert row["oracles"]["registered"]["candidate"] == baseline["oracle_id"]
        for name, ids in ref["candidate_sets"].items():
            selected = common.choose(common.feasible([q for q in ref["candidates"] if q["id"] in ids], row["budget"]),
                                     row["objective"], dense, "actual")
            assert row["oracles"][name]["candidate"] == (selected["id"] if selected else None)
        for policy in policies:
            assert policies[policy]["regret_registered"] == baseline["policies"][policy]["regret"]
            assert policies[policy]["selected"] != gen.S0_ID


def test_retrospective_table_is_byte_identical(result):
    rendered = gen.render(result).encode("utf-8")
    assert hashlib.sha256(rendered).hexdigest() == TABLE_SHA256
    assert gen.render(copy.deepcopy(result)).encode("utf-8") == rendered
    # The digest above is unconditional for standalone public checkouts too.
    for path in (gen.table_path(ROOT), ROOT / "results" / gen.CONTROL / "s3_control.tex"):
        if path.exists():
            assert rendered == path.read_bytes()
    text = rendered.decode()
    assert "\\begin{table}[!htbp]" in text
    assert "\\begin{tabular*}{\\textwidth}" in text
    assert "This analysis is retrospective." in text
    assert "was not in the frozen candidate set" in text
    assert "sixteen of seventeen" in text


def test_generation_preserves_all_input_bytes_and_is_deterministic(tmp_path, result):
    source = gen.input_directory(ROOT)
    before = {p.relative_to(source): gen.sha(p) for p in source.rglob("*") if p.is_file()}
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / gen.EXPERIMENT).symlink_to(source.resolve(), target_is_directory=True)
    generated = gen.generate(tmp_path)
    assert generated["input_sha256"] == result["input_sha256"]
    table = gen.table_path(tmp_path).read_bytes()
    assert table == (tmp_path / "results" / gen.CONTROL / "s3_control.tex").read_bytes()
    first = (tmp_path / "results" / gen.CONTROL / "summary.json").read_bytes()
    gen.generate(tmp_path)
    assert first == (tmp_path / "results" / gen.CONTROL / "summary.json").read_bytes()
    after = {p.relative_to(source): gen.sha(p) for p in source.rglob("*") if p.is_file()}
    assert before == after
    assert before[Path("score_v2.json")] == gen.SCORE_SHA256
    assert before[Path("plan_v2.json")] in gen.PLAN_HASHES


@pytest.mark.parametrize("filename, message", [("plan_v2.json", "frozen plan"),
                                             ("score_v2.json", "Frozen score changed")])
def test_modified_frozen_files_are_rejected(tmp_path, filename, message):
    source = gen.input_directory(ROOT)
    for name in ("plan_v2.json", "score_v2.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes() + (b" " if name == filename else b""))
    with pytest.raises(ValueError, match=message):
        gen.analyze(ROOT, directory=tmp_path)
