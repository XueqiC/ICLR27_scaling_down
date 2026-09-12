import json

import numpy as np

from analysis import v13_recovery as recovery


def test_budget_schedule_parser_accepts_decimal_suffixes():
    assert recovery.parse_budgets("0.5M,2M,8M,32M") == [
        500_000,
        2_000_000,
        8_000_000,
        32_000_000,
    ]
    assert recovery.parse_budget_schedule("250k,1B") == [250_000, 1_000_000_000]


def test_recovery_power_law_fit_recovers_known_beta():
    tokens = np.array([500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000])
    expected_l_inf = 1.15
    expected_b = 24_000.0
    expected_beta = 0.68
    losses = expected_l_inf + expected_b * np.power(tokens, -expected_beta)

    fit = recovery.fit_recovery_power_law(
        tokens, losses, dense_loss=expected_l_inf - 0.03
    )

    assert fit["status"] == "ok"
    assert fit["n_points"] == len(tokens)
    np.testing.assert_allclose(fit["L_inf"], expected_l_inf, atol=1e-5)
    np.testing.assert_allclose(fit["beta"], expected_beta, atol=1e-4)
    np.testing.assert_allclose(fit["B"], expected_b, rtol=2e-3)
    np.testing.assert_allclose(fit["r2"], 1.0, atol=1e-9)
    np.testing.assert_allclose(fit["residual_vs_dense"], 0.03, atol=1e-5)


def test_recovery_power_law_fit_handles_too_few_points():
    fit = recovery.fit_recovery_power_law(
        [500_000, 2_000_000], [2.0, 1.7], dense_loss=1.0
    )

    assert fit == {
        "status": "too_few_points",
        "n_points": 2,
        "L_inf": None,
        "B": None,
        "beta": None,
        "r2": None,
        "residual_vs_dense": None,
    }


def test_trace_recovery_source_returns_mixed_teacher_domain_shape(tmp_path):
    rows_by_file = {
        "gpt-5.6-luna_math.jsonl": [
            {"prompt": "m1", "response": "a1", "teacher": "gpt-5.6-luna"},
            {"prompt": "m2", "response": "a2", "teacher": "gpt-5.6-luna"},
        ],
        "gpt-5.6-luna_code.jsonl": [
            {"prompt": "c1", "response": "code", "teacher": "gpt-5.6-luna"}
        ],
        "claude-sonnet-4-6_qa.jsonl": [
            {
                "prompt": "q1",
                "response": "answer",
                "teacher": "claude-sonnet-4-6",
            }
        ],
    }
    for filename, rows in rows_by_file.items():
        (tmp_path / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    records, metadata = recovery.select_recovery_source(
        "traces", trace_base=tmp_path, seed=17
    )

    assert len(records) == 4
    assert all(
        set(record) == {"prompt", "completion", "domain", "teacher"}
        for record in records
    )
    assert metadata["mix"] == {
        "claude-sonnet-4-6": {"qa": 1},
        "gpt-5.6-luna": {"code": 1, "math": 2},
    }
    assert metadata["mix_shape"] == {
        "teachers": 2,
        "domains": 3,
        "teacher_domain_cells": 3,
    }
    second_records, _ = recovery.select_recovery_source(
        "traces", trace_base=tmp_path, seed=17
    )
    assert records == second_records
