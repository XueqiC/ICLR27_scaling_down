from __future__ import annotations

import pytest

from analysis import v17_unification as v17


def test_coordinate_mappings() -> None:
    pruning = v17.coordinate_mapping("pruning", 0.4)
    assert pruning["raw_coordinate_name"] == "density"
    pruning_values = {
        key: pruning[key] for key in ("raw_coordinate", "r_storage", "r_active")
    }
    assert pruning_values == pytest.approx(
        {"raw_coordinate": 0.4, "r_storage": 0.4, "r_active": 1.0}
    )

    quantization = v17.coordinate_mapping("quantization", 8)
    assert quantization["raw_coordinate_name"] == "bits"
    quantization_values = {
        key: quantization[key] for key in ("raw_coordinate", "r_storage", "r_active")
    }
    assert quantization_values == pytest.approx(
        {"raw_coordinate": 8.0, "r_storage": 0.5, "r_active": 1.0}
    )

    distillation = v17.coordinate_mapping(
        "distillation", 2_000_000, n0_params=10_000_000
    )
    assert distillation["raw_coordinate_name"] == "student_params"
    distillation_values = {
        key: distillation[key] for key in ("raw_coordinate", "r_storage", "r_active")
    }
    assert distillation_values == pytest.approx(
        {"raw_coordinate": 2_000_000.0, "r_storage": 0.2, "r_active": 0.2}
    )


def test_coordinate_mapping_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="density"):
        v17.coordinate_mapping("pruning", 0.0)
    with pytest.raises(ValueError, match="n0_params"):
        v17.coordinate_mapping("distillation", 10)
    with pytest.raises(ValueError, match="exceed"):
        v17.coordinate_mapping("distillation", 11, n0_params=10)


def test_leave_one_out_splitter_holds_out_whole_groups() -> None:
    rows = [
        {"method": "pruning", "cell": 0},
        {"method": "pruning", "cell": 1},
        {"method": "quantization", "cell": 2},
        {"method": "distillation", "cell": 3},
    ]

    splits = v17.leave_one_out_splits(rows, "method")

    assert [split["held_out"] for split in splits] == [
        "distillation",
        "pruning",
        "quantization",
    ]
    for split in splits:
        held_out = split["held_out"]
        train_methods = {rows[index]["method"] for index in split["train_indices"]}
        test_methods = {rows[index]["method"] for index in split["test_indices"]}
        assert held_out not in train_methods
        assert test_methods == {held_out}
        assert set(split["train_indices"]).isdisjoint(split["test_indices"])
        assert sorted(split["train_indices"] + split["test_indices"]) == list(
            range(len(rows))
        )


def test_leave_largest_model_out_is_within_family() -> None:
    rows = [
        {"family": "a", "model": "a-small", "model_size_params": 1},
        {"family": "a", "model": "a-large", "model_size_params": 2},
        {"family": "b", "model": "b-only", "model_size_params": 4},
    ]

    splits = v17.leave_largest_model_out_splits(rows)

    assert len(splits) == 1
    assert splits[0]["family"] == "a"
    assert splits[0]["held_out"] == "a-large"
    assert splits[0]["test_indices"] == [1]


def test_precliff_filter_drops_crossing_and_all_more_compressed_cells() -> None:
    rows = []
    for capability, damages in {
        "math": {1.0: 0.0, 0.9: 0.1, 0.8: 1.1, 0.7: 0.2},
        "qa": {1.0: 0.0, 0.9: 0.2, 0.8: 0.3, 0.7: 0.4},
    }.items():
        for storage, damage in damages.items():
            rows.append(
                {
                    "model": "Qwen3-0.6B",
                    "method": "pruning",
                    "capability": capability,
                    "r_storage": storage,
                    "delta_L_c": damage,
                    "is_baseline": storage == 1.0,
                }
            )

    annotated, audit = v17.apply_precliff_filter(rows, cap=1.0)
    statuses = {
        (row["capability"], row["r_storage"]): row["precliff_status"]
        for row in annotated
    }

    assert statuses[("math", 1.0)] == "baseline"
    assert statuses[("math", 0.9)] == "pre_cliff"
    assert statuses[("math", 0.8)] == "post_cliff"
    # The lower-damage 0.7 cell is still post-cliff because it follows the crossing.
    assert statuses[("math", 0.7)] == "post_cliff"
    assert statuses[("qa", 0.7)] == "pre_cliff"
    assert audit["precliff_cells"] == {"pruning": 4}
    assert audit["postcliff_cells"] == {"pruning": 2}


def test_matched_storage_pairing_uses_tolerance_and_is_one_to_one() -> None:
    def row(method: str, storage: float, damage: float) -> dict:
        return {
            "model": "Qwen3-0.6B",
            "family": "qwen3",
            "method": method,
            "capability": "math",
            "r_storage": storage,
            "raw_coordinate": storage if method == "pruning" else storage * 16,
            "delta_L_c": damage,
            "is_baseline": False,
            "precliff_status": "pre_cliff",
        }

    rows = [
        row("pruning", 0.50, 5.9),
        row("pruning", 0.40, 0.4),
        row("pruning", 0.35, 0.3),
        row("pruning", 0.30, 0.2),
        row("quantization", 0.50, 0.0),
        row("quantization", 0.375, 0.01),
        row("quantization", 0.25, 0.1),
    ]

    pairs = v17.matched_storage_pairs(rows, tolerance=0.03)

    assert len(pairs) == 2
    assert sum(pair["quantization_r_storage"] == 0.375 for pair in pairs) == 1
    assert {
        (pair["pruning_r_storage"], pair["quantization_r_storage"]) for pair in pairs
    } == {
        (0.5, 0.5),
        (0.4, 0.375),
    }
    assert pairs[0]["absolute_gap"] == pytest.approx(5.9)


def test_historical_quant_ladder_excludes_metadata_but_still_rejects_invalid_bits(tmp_path):
    import json

    path = tmp_path / "gemma3-1b" / "quant_losses.json"
    path.parent.mkdir()
    losses = dict.fromkeys(v17.CAPABILITIES, 1.)
    for invalid in ("0", "17", "not_a_bit"):
        path.write_text(json.dumps({"dense": losses, invalid: losses}))
        with pytest.raises(ValueError):
            v17._load_quantization_rows(tmp_path)
    path.write_text(json.dumps({"dense": losses, "4": losses, "5": losses,
                                "_5bit_meta": {"source": "later confirmation"}}))
    rows, notes = v17._load_quantization_rows(tmp_path)
    assert len(rows) == 6 and {r["raw_coordinate"] for r in rows} == {4., 16.}
    assert any("Ignored quantization metadata" in n for n in notes)
    assert any("Excluded later bit configuration" in n for n in notes)


def test_source_referenced_delta_subtracts_source_dense_loss() -> None:
    distilled = {"math": 0.82, "code": 0.91, "qa": 4.25}
    source = {"math": 0.60, "code": 0.73, "qa": 5.79}

    assert v17.source_referenced_deltas(distilled, source) == pytest.approx(
        {"math": 0.22, "code": 0.18, "qa": -1.54}
    )


def test_assembled_distillation_keeps_self_and_four_point_source_ladder() -> None:
    rows, audit = v17.assemble_table()
    assert {row["model"] for row in rows} == v17.V17_MODELS
    assert {row["raw_coordinate"] for row in rows if row["method"] == "quantization"} == v17.V17_BITS
    self_rows = [row for row in rows if row["method"] == v17.DISTILL_SELF]
    source_rows = [row for row in rows if row["method"] == v17.DISTILL_SOURCE]

    assert self_rows
    expected_students = {
        "gemma3-270m",
        "gemma3-1b",
        "gemma3-4b",
        "gemma3-12b",
    }
    assert {row["model"] for row in source_rows} == expected_students
    assert len(source_rows) == 4 * len(v17.CAPABILITIES)
    assert audit["n_distillation_source_students"] == 4
    assert audit["distillation_source_students"] == sorted(expected_students)
    assert audit["distillation_source_audit"]["registered_ladders"]["gemma3"][
        "n_points"
    ] == 4
    assert all(row["teacher"] == "gpt-5.6-luna" for row in source_rows)
    assert all(row["run"] == "gpt-5.6-luna_full_600" for row in source_rows)
    assert audit["distillation_source_audit"]["family_status"]["olmo3"] == (
        "single_point_no_clean_same_family_source"
    )
    expected = {
        "gemma3-270m": {
            "r_storage": 0.010,
            "math": 0.81,
            "code": 0.63,
            "qa": -1.18,
        },
        "gemma3-1b": {
            "r_storage": 0.036,
            "math": 0.74,
            "code": 0.46,
            "qa": -0.96,
        },
        "gemma3-4b": {
            "r_storage": 0.157,
            "math": 0.22,
            "code": 0.15,
            "qa": -1.52,
        },
        "gemma3-12b": {
            "r_storage": 0.445,
            "math": 0.14,
            "code": 0.15,
            "qa": -1.55,
        },
    }
    for student, student_expected in expected.items():
        student_rows = [row for row in source_rows if row["model"] == student]
        assert len(student_rows) == len(v17.CAPABILITIES)
        assert student_rows[0]["r_storage"] == pytest.approx(
            student_expected["r_storage"], abs=0.001
        )
        for row in student_rows:
            assert row["delta_L_c"] == pytest.approx(
                student_expected[row["capability"]], abs=0.01
            )
    math_4b = next(
        row
        for row in source_rows
        if row["model"] == "gemma3-4b" and row["capability"] == "math"
    )
    assert math_4b["delta_L_source_c"] == pytest.approx(
        math_4b["distilled_L_c"] - math_4b["reference_L_c"]
    )
    assert math_4b["r_storage"] == pytest.approx(
        math_4b["model_size_params"] / math_4b["n0_params"]
    )
    assert math_4b["r_active"] == pytest.approx(math_4b["r_storage"])


def test_matched_axes_use_source_delta_and_exclude_self_delta() -> None:
    base = {
        "family": "gemma3",
        "capability": "math",
        "is_baseline": False,
        "precliff_status": "pre_cliff",
        "raw_coordinate": 1.0,
        "delta_reference": "own_dense",
    }
    rows = [
        {
            **base,
            "method": "pruning",
            "model": "gemma3-27b",
            "r_storage": 0.41,
            "r_active": 1.0,
            "delta_L_c": 0.4,
        },
        {
            **base,
            "method": "quantization",
            "model": "gemma3-27b",
            "r_storage": 0.40,
            "r_active": 1.0,
            "delta_L_c": 0.1,
        },
        {
            **base,
            "method": v17.DISTILL_SOURCE,
            "model": "gemma3-12b",
            "n0_model": "gemma3-27b",
            "r_storage": 0.40,
            "r_active": 0.40,
            "delta_L_c": 0.2,
            "delta_reference": "same_family_source_dense",
        },
        {
            **base,
            "method": v17.DISTILL_SELF,
            "model": "gemma3-12b",
            "n0_model": "gemma3-27b",
            "r_storage": 0.40,
            "r_active": 0.40,
            "delta_L_c": -9.0,
            "delta_reference": "student_own_dense_style_residualized",
        },
    ]

    storage_pairs = v17.matched_axis_pairs(rows, "r_storage", tolerance=0.03)
    assert any(
        v17.DISTILL_SOURCE in {pair["method_a"], pair["method_b"]}
        for pair in storage_pairs
    )
    assert all(
        v17.DISTILL_SELF not in {pair["method_a"], pair["method_b"]}
        for pair in storage_pairs
    )
    active_pairs = v17.matched_axis_pairs(rows, "r_active", tolerance=0.03)
    assert all(
        v17.DISTILL_SOURCE not in {pair["method_a"], pair["method_b"]}
        for pair in active_pairs
    )


def test_method_selection_has_distinct_storage_and_active_winners() -> None:
    def row(
        method: str,
        storage: float,
        active: float,
        damage: float,
        raw_name: str,
        raw: float,
        model: str = "gemma3-27b",
    ) -> dict:
        return {
            "method": method,
            "model": model,
            "family": "gemma3",
            "capability": "math",
            "r_storage": storage,
            "r_active": active,
            "raw_coordinate_name": raw_name,
            "raw_coordinate": raw,
            "delta_L_c": damage,
            "n0_model": "gemma3-27b",
        }

    rows = [
        row("pruning", 0.9, 1.0, 0.01, "density", 0.9),
        row("pruning", 0.7, 1.0, 0.08, "density", 0.7),
        row("pruning", 0.5, 1.0, 0.30, "density", 0.5),
        row("quantization", 0.5, 1.0, 0.0, "bits", 8.0),
        row("quantization", 0.25, 1.0, 0.10, "bits", 4.0),
        row(
            v17.DISTILL_SOURCE,
            0.4,
            0.4,
            0.10,
            "student_params",
            4.0,
            model="gemma3-12b",
        ),
        row(
            v17.DISTILL_SOURCE,
            0.2,
            0.2,
            0.20,
            "student_params",
            2.0,
            model="gemma3-4b",
        ),
        # A very favorable self-referenced value must never enter selection.
        row(
            v17.DISTILL_SELF,
            0.01,
            0.01,
            -10.0,
            "student_params",
            1.0,
            model="gemma3-1b",
        ),
    ]

    selection, _ = v17.build_method_selection_map(rows, [0.12], tolerance=0.03)
    math_rows = {
        result["cost_axis"]: result
        for result in selection
        if result["capability"] == "math"
    }

    assert math_rows["r_storage"]["cheapest_method"] == "quantization"
    assert math_rows["r_storage"]["cost_ratio"] == pytest.approx(0.25)
    assert math_rows["r_active"]["cheapest_method"] == v17.DISTILL_SOURCE
    assert math_rows["r_active"]["cost_ratio"] == pytest.approx(0.4)
    assert math_rows["r_active"]["status"] == "insufficient_matched_active_support"
