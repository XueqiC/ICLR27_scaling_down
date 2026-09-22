"""A16 storage accounting: byte formulas and the recorded summary (CPU only)."""
import json
import math
from pathlib import Path

import pytest

from analysis import a16_storage_accounting as a16

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/a16-storage-accounting/summary.json"
SHAPES = [(1000, 1000), (4000, 1000)]  # 5M parameters, 5000 rows


def test_byte_formulas():
    n = 5_000_000; rows = 5000
    assert a16.dense_bytes(SHAPES) == 2 * n
    p = a16.prune_bytes(SHAPES, 0.5)
    assert math.isclose(p["bitmap"], 2 * 0.5 * n + n / 8) and math.isclose(p["csr"], 4 * 0.5 * n + 4 * rows)
    assert math.isclose(a16.quant_bytes(SHAPES, 4), n * 0.5 + 2 * rows)
    assert math.isclose(a16.quant_bytes(SHAPES, 4, 32), n * 0.5 + 2 * n / 32)


def test_price_reproduces_nominal_for_grouped_quantization_and_adds_index_for_pruning():
    ref = SHAPES; student = [(500, 500)]
    grouped = a16.price({"id": "quant:b4_g32", "r": 4 / 16 + 1 / 32}, ref, student)
    assert math.isclose(grouped["actual"], grouped["nominal"], rel_tol=1e-12)
    pruned = a16.price({"id": "prune:d0.8", "r": 0.8}, ref, student)
    assert math.isclose(pruned["actual"], 0.8 + 1 / 16, rel_tol=1e-9) and pruned["actual_csr"] > 1
    dense = a16.price({"id": "dense:source", "r": 1.0}, ref, student)
    assert dense["actual"] == 1.0
    with pytest.raises(ValueError):
        a16.price({"id": "unknown:x", "r": 1.0}, ref, student)


@pytest.mark.skipif(not SUMMARY.exists(), reason="storage accounting not run in this checkout")
def test_recorded_summary_is_internally_consistent():
    s = json.loads(SUMMARY.read_text())
    for rid, e in s["references"].items():
        by_id = {c["id"]: c for c in e["candidates"]}
        assert by_id["dense:source"]["actual"] == 1.0
        for cid, c in by_id.items():
            if cid.startswith("quant:b"):
                assert math.isclose(c["actual"], c["nominal"], rel_tol=1e-9)
            if cid.startswith("prune"):
                assert c["actual"] > c["nominal"] and c["actual_csr"] > c["actual"]
        student = by_id["distill:new_s3"]
        assert math.isclose(student["actual"], e["student_matrix_parameters"] / e["matrix_parameters"], rel_tol=1e-9)
    f = s["feasibility_under_bytes"]
    assert f["cells"] == 272 and f["material"] == sum(1 for r in f["rule_choice_over_budget"] if r["material"])
    assert set(f["material_choices"]) == {"distill:new_s3"}
