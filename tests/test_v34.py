"""C17 provenance regressions: no name-only cleanliness, no double half-slicing."""
from __future__ import annotations

import builtins
import copy
import json
from pathlib import Path

import pytest

from analysis import v34_c17_scope_audit as audit
from public_inputs import require_public_inputs


def prompt(q, context="Example: supplied passage", suffix="Answer:"):
    return {"prompt": f"Context:\n{context}\n\nQuestion: {q}\n{suffix}"}


def test_question_identity_ignores_template_answer_and_unicode_whitespace():
    left = [prompt("Who is Ａ?", suffix="Think briefly, then answer concisely.")]
    right = [{"question": " who\t is a? ", "answer": "a different answer"}]
    result = audit.compare_questions(left, right)
    assert result["status"] == "overlap"
    assert result["overlap_count"] == 1
    assert len(result["matched_question_sha256"][0]) == 64


def test_old_gold_control_uses_user_message_not_assistant_answer():
    row = {"messages": [{"role": "user", "content": "Context: text\n\nQuestion: Original question?\n\nReason."},
                        {"role": "assistant", "content": "Question: fabricated answer question?"}]}
    assert audit.question(row) == "Original question?"


@pytest.mark.parametrize("left,right", [([], []), ([{"i": 1}], [{"i": 2}]),
    ([{"task_id": "old-id"}], [prompt("Known question?")]),
    ([{"question": ""}], [prompt("Known question?")])])
def test_missing_questions_and_disjoint_ids_are_unknown(left, right):
    result = audit.compare_questions(left, right)
    assert result["status"] == "unknown"
    assert result["overlap_count"] is None
    assert not result["comparison_complete"]


def test_partial_coverage_reports_observed_hit_without_certifying_remainder():
    result = audit.compare_questions([prompt("A?"), {"i": 3}], [prompt("A?")])
    assert result["overlap_count"] == 1
    assert result["status"] == "overlap"
    assert not result["comparison_complete"]


def test_question_check_is_not_answer_or_dataset_name_match():
    left = [{"question": "Who won in 2001?", "answer": "yes", "dataset": "HotpotQA", "id": 1}]
    right = [{"question": "Who won in 2002?", "answer": "yes", "dataset": "2Wiki", "id": 1}]
    assert audit.compare_questions(left, right)["overlap_count"] == 0
    right[0]["question"] = left[0]["question"]
    assert audit.compare_questions(left, right)["status"] == "overlap"


def test_shared_article_is_distinct_from_shared_question():
    a, b = prompt("Who wrote A?", "Same Title: old text"), prompt("Who wrote B?", "Same Title: new text")
    assert audit.titles(a) & audit.titles(b) == {"same title"}
    assert audit.compare_questions([a], [b])["overlap_count"] == 0


@pytest.fixture(scope="module")
def summary():
    require_public_inputs(
        "results/traces-pilot/gpt-5.6-luna_qa.jsonl",
        "results/traces-pilot/claude-sonnet-4-6_qa.jsonl",
        reason="raw teacher-trace text is deliberately not redistributed")
    # Entire default audit must work with all model/network/data-loader imports
    # prohibited. Inspect existing artifacts only; no writes in build_summary.
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "datasets", "requests", "socket", "pyarrow"}:
            raise AssertionError(f"Forbidden audit dependency: {name}")
        return original(name, *args, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "__import__", guarded)
        return audit.build_summary()


def test_raw_recomputation_and_one_third_denominator(summary):
    old = summary["old_grid"]
    assert old["respondents_per_benchmark"] == {"hotpotqa": 52, "2wiki": 52}
    assert old["hotpotqa_local_indices"] == list(range(400, 450))
    assert old["locked_grid_link"]["all_V3_item_ids_match"]
    assert old["locked_grid_link"]["cell_table_rows"] == 1065
    assert old["recovered_M0_training_questions"]["rows"] == 150
    assert summary["overlap_checks"]["old_M0_training_vs_current_training"]["overlap_count"] == 0
    assert summary["overlap_checks"]["old_M0_training_vs_current_measurement"]["overlap_count"] == 0
    m2 = old["qa_split"]["M2"]
    assert m2["n"] == 12
    assert m2["d_trained"] == pytest.approx(.19833333333333333)
    assert m2["d_transfer"] == pytest.approx(.12666666666666668)
    assert all(r["matches_saved"] for r in old["qa_split"].values())
    assert old["M2_gap_divided_by_hotpot_gain"] == pytest.approx(43/119)
    assert old["M2_panel_excess_over_transfer_fraction"] == pytest.approx(43/195)


def test_current_coverage_and_measurement_not_resliced(summary):
    current = summary["current_artifacts"]
    assert [r["rows"] for r in current["trace_files"]] == [600, 600]
    assert len(current["run_inventory"]["v12"]) == 74
    assert len(current["run_inventory"]["v16"]) == 16
    assert len(current["default_accuracy_panels"]) == 3
    assert all(p["same_ordered_panel_as_first"] for p in current["default_accuracy_panels"])
    assert not current["metadata_conflicts"]
    assert summary["current_protocol"]["odd_after_sampling_verified"]
    assert summary["current_protocol"]["expected_current_design_verified"]
    check = summary["overlap_checks"]["current_training_vs_current_measurement"]
    assert (check["left_rows"], check["left_unique_questions"], check["right_rows"]) == (1200, 600, 64)
    assert check["overlap_count"] == 0
    clean = summary["minimal_clean_set"]
    assert clean["exact_question_clean_indices"] == list(range(64))
    assert clean["stricter_rendered_title_disjoint_indices"] == [i for i in range(64) if i not in (8, 37, 41)]
    assert summary["overlap_checks"]["old_2wiki_vs_current_measurement"]["overlap_count"] is None


def test_no_cache_no_official_split_claim_and_no_global_clean_bill(summary):
    assert summary["hotpot_cache_identification"]["status"] == "not_supplied"
    assert summary["verdict"]["branch"] == "b_training_benchmark_reused"
    assert not summary["verdict"]["unconditional_clean_bill"]
    assert summary["verdict"]["original_C17_preserved"]
    report = audit.render_report(summary)
    assert "complete expected official-split mapping is not established" in report
    assert "identifies current trace questions as official train rows 0–599" not in report
    assert "C18 QA interpretation" in report and "C21 QA" in report
    assert "No model runs were performed" in report
    dependencies = summary["claim_dependency_inventory"]
    v21 = dependencies["results/v21-distill-law/summary.json"]
    assert len(v21["declared_raw_paths"]) == 11
    assert not v21["missing_paths"] and not v21["old_grid_paths"]
    assert dependencies["results/v31-uxseen/summary.json"]["reference_coverage"] == "no raw-path manifest in summary"


def test_fail_closed_on_measurement_hit_missing_evidence_or_changed_metadata():
    clean = audit.compare_questions([prompt("A?")], [prompt("B?")])
    hit = audit.compare_questions([prompt("A?")], [prompt("A?")])
    unknown = audit.compare_questions([], [prompt("A?")])
    assert audit.verdict(clean, hit, clean, True)["status"] == "measurement_overlap_requires_revalidation"
    assert audit.verdict(clean, unknown, clean, True)["status"] == "scope_incomplete"
    assert audit.verdict(clean, clean, clean, False)["status"] == "scope_incomplete"


def test_incomplete_report_cannot_repeat_clean_snapshot_prose(summary):
    changed = copy.deepcopy(summary)
    changed["verdict"]["status"] = "scope_incomplete"
    report = audit.render_report(changed)
    assert "Scope cannot be certified" in report
    assert "saved questions show no exact overlap" not in report


def test_ast_checks_executable_half_not_docstring(tmp_path):
    (tmp_path / "analysis").mkdir()
    for name in ("v12_distill.py", "v6_capability_geometry.py"):
        text = (audit.ROOT / "analysis" / name).read_text()
        if name == "v12_distill.py":
            text = text.replace("all_probes[capability][1::2]", "all_probes[capability][::2]")
        (tmp_path / "analysis" / name).write_text(text)
    protocol = audit.source_protocol(audit.Evidence(tmp_path))
    assert protocol["measurement_slices"] == ["::2"]
    assert not protocol["odd_after_sampling_verified"]


def test_optional_arrow_cache_resolves_real_ids_and_split_names(tmp_path):
    pa = pytest.importorskip("pyarrow")
    for split, qs in (("train", ["current?", "extra?"]), ("validation", ["old?"])):
        table = pa.table({"question": qs, "id": [split + str(i) for i in range(len(qs))]})
        with pa.OSFile(str(tmp_path / f"hotpot_qa-{split}.arrow"), "wb") as handle:
            with pa.ipc.new_stream(handle, table.schema) as writer:
                writer.write_table(table)
    evidence = audit.Evidence(tmp_path)
    result = audit.cache_matches(evidence, tmp_path, {"current": [prompt("CURRENT?")], "old": [{"question": "old?"}]})
    assert result["coverage"]["current"]["matches"][0]["source_id"] == "train0"
    assert result["coverage"]["old"]["matches"][0]["split"] == "validation"
    assert len(evidence.hashes) == 2
    evidence.verify_unchanged()


def test_input_hash_guard_and_strict_json(summary, tmp_path):
    assert not summary["inputs_mutated"]
    assert "results/v3-measurement-audit/split.json" in summary["input_sha256"]
    assert "paper/docs/RESULTS_LEDGER.md" in summary["input_sha256"]
    json.dumps(summary, allow_nan=False)
    p = tmp_path / "original.json"
    p.write_text("{}")
    e = audit.Evidence(tmp_path)
    e.read(p)
    p.write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match="input changed"):
        e.verify_unchanged()
