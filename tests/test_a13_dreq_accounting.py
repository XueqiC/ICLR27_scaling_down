"""Interval/censoring semantics and equal-denominator regression checks."""
import copy
import math

import pytest

from analysis.a13_dreq_accounting import (
    METHODS, QA, build_accounting, figure_records, interval_metrics, load_inputs, summarize,
)


@pytest.mark.parametrize("value,distance,inside,signed", [
    (50, math.log(2), False, -math.log(2)), (100, 0, True, 0),
    (200, 0, True, 0), (400, 0, True, 0), (800, math.log(2), False, math.log(2)),
])
def test_finite_interval_does_not_invent_a_point_minimum(value, distance, inside, signed):
    result = interval_metrics(value, {"status": "crossed", "interval": [100, 400]})
    assert result["log_distance"] == pytest.approx(distance)
    assert result["signed_log_distance"] == pytest.approx(signed)
    assert result["inside_interval"] is inside
    assert result["interval_width_tokens"] == 300
    assert result["interval_log_width"] == pytest.approx(math.log(4))


@pytest.mark.parametrize("status,interval,value,expected,inside", [
    ("lower_bound", [400, None], 100, math.log(4), False),
    ("lower_bound", [400, None], 400, 0, True),
    ("lower_bound", [400, None], 800, 0, True),
    ("upper_bound", [0, 100], 400, math.log(4), False),
    ("upper_bound", [0, 100], 50, 0, True),
])
def test_censored_distance_is_only_a_lower_bound(status, interval, value, expected, inside):
    result = interval_metrics(value, {"status": status, "interval": interval})
    assert result["log_distance"] == pytest.approx(expected)
    assert result["inside_interval"] is inside
    assert result["distance_status"] == "censored_distance_lower_bound"
    assert result["interval_log_width"] is None
    assert result["width_status"] == "unbounded_log_interval"


def test_missing_recommendation_and_unresolved_crossing_are_not_zero_error():
    assert interval_metrics(None, {"status": "crossed", "interval": [100, 400]})["log_distance"] is None
    for status in ("replicate_ambiguous", "non_monotone"):
        result = interval_metrics(200, {"status": status, "interval": None})
        assert result["log_distance"] is None
        assert result["inside_interval"] is None
        assert result["measured_lower_bound_D_U"] is None


@pytest.fixture(scope="module")
def actual():
    score, plan, _ = load_inputs()
    rows, requests = build_accounting(score, plan)
    return score, plan, rows, requests


def test_actual_request_grid_and_primary_map(actual):
    score, plan, rows, requests = actual
    assert len(requests) == 6*4*2*3 == 144
    assert len(rows) == 144*3*2
    records = figure_records(score, plan, requests)
    assert len(records) == 18 and {r["readout"] for r in records} == set(QA)
    assert all(r["tau"] == .25 for r in records)
    for r in records:
        interval = r["plotted_interval"]
        if interval:
            lo, hi = interval
            all_cells = [c for t in r["tiers"] for c in t["replicates"]]
            assert lo == max(c["D_U"] for c in all_cells if not c["passes"])
            assert hi == min(c["D_U"] for c in all_cells if c["passes"])
            assert r["measured_crossing"]["interval"][0] <= lo < hi <= r["measured_crossing"]["interval"][1]
        elif r["measured_crossing"]["status"] in ("replicate_ambiguous", "non_monotone"):
            assert r["plotted_bound"] is None and r["unresolved_tier_ranges"]
        for tier in r["tiers"]:
            if tier["position"] == "above":
                assert {c["D_U"] for c in tier["replicates"]} == {158391}
                assert {c["data_seed"] for c in tier["replicates"]} == {None}


def test_same_denominators_keep_abstentions_and_unresolved_rows(actual):
    _, _, rows, _ = actual
    comparisons, _ = summarize(rows)
    for kind in ("boundary", "delta"):
        for group, n in (("QA distributions", 72), ("math/code", 48),
                         ("2Wiki original probe", 24), ("all requests", 144)):
            rr = [r for r in comparisons if r["kind"] == kind and r["group"] == group]
            assert len(rr) == 3
            assert len({tuple(r["common_request_ids"]) for r in rr}) == 1
            for r in rr:
                assert r["coverage"]["denominator"] == r["solved"]["denominator"] == n
                assert r["conditional"]["denominator"] == r["coverage"]["numerator"]
                assert r["conditional"]["numerator"] == r["solved"]["numerator"]
    # Historical C84 pooled numbers must decompose, not become a request denominator.
    pooled = {m: [r for r in rows if r["method"] == m] for m in METHODS}
    assert [(sum(r["recommendation_meets_loss_constraint"] is True for r in pooled[m]),
             sum(r["recommended_D_U"] is not None for r in pooled[m])) for m in METHODS] == [(126, 207), (101, 181), (112, 172)]


def test_empty_common_support_is_undefined_not_perfect_or_zero(actual):
    rows = copy.deepcopy(actual[2])
    for r in rows:
        if r["method"] == "fixed_reuse":
            r["recommended_D_U"] = None
            r["recommendation_meets_loss_constraint"] = None
    comparisons, _ = summarize(rows)
    assert all(r["common_reliability"]["denominator"] == 0 for r in comparisons)
    assert all(r["common_reliability"]["fraction"] is None for r in comparisons)

