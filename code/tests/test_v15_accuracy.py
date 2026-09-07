import pytest

from analysis import v15_accuracy_link as accuracy


@pytest.mark.parametrize(
    ("generation", "reference"),
    [
        (r"Work. Therefore, \boxed{\frac{1}{2}}.", r"\frac{1}{2}"),
        (r"The result is \boxed{-\dfrac{3}{4}}", r"-\frac{3}{4}"),
        (r"Final answer: $\left( -2 \right)$.", "(-2)"),
        (r"The answer is 12 cm.", "12"),
        (r"Thus the area is \boxed{25\text{ cm}^2}.", "25"),
        ("Reasoning\n#### -17.", "-17"),
        ("The final value is $1,250$.", "1250"),
        ("After reducing, the answer is 1/2.", r"\frac{1}{2}"),
        (r"Answer: \frac12", r"\frac{1}{2}"),
        ("We reject 2 and obtain -9.", "-9"),
    ],
)
def test_math_extraction_and_normalization(generation, reference):
    correct, extracted, normalized_prediction, normalized_reference = (
        accuracy.math_exact_match(generation, reference)
    )

    assert extracted
    assert normalized_prediction == normalized_reference
    assert correct is True


def test_math_normalization_strips_equation_wrapper_and_latex_spacing():
    assert accuracy.normalize_math_answer(r"$x = \left\dfrac{ 2 }{ 3 }\right$.") == (
        r"\frac{2}{3}"
    )


def test_qa_squad_normalization_exact_match_and_token_f1():
    assert accuracy.normalize_qa_answer(" The, QUICK fox! ") == "quick fox"
    assert accuracy.qa_exact_match("An Eiffel Tower.", "Eiffel Tower")
    assert not accuracy.qa_exact_match("Eiffel Tower Paris", "Eiffel Tower")
    assert accuracy.qa_token_f1("Eiffel Tower Paris", "the Eiffel Tower") == pytest.approx(
        0.8
    )
    assert accuracy.qa_token_f1("", "") == 1.0
    assert accuracy.qa_token_f1("unrelated", "answer") == 0.0


def test_code_sandbox_passes_valid_completion():
    result = accuracy.run_code_tests(
        "def add(a, b):\n    return a + b",
        ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"],
        timeout_seconds=1.0,
    )

    assert result["passed"] is True
    assert result["status"] == "passed"
    assert result["returncode"] == 0


def test_code_sandbox_reports_failing_completion():
    result = accuracy.run_code_tests(
        "```python\ndef add(a, b):\n    return a - b\n```",
        ["assert add(2, 3) == 5"],
        timeout_seconds=1.0,
    )

    assert result["passed"] is False
    assert result["status"] == "failed"
    assert result["returncode"] != 0
    assert "AssertionError" in result["stderr"]


def test_code_sandbox_times_out():
    result = accuracy.run_code_tests(
        "while True:\n    pass",
        [],
        timeout_seconds=0.2,
    )

    assert result["passed"] is False
    assert result["status"] == "timeout"
    assert result["returncode"] is None


def test_fewshot_prompts_include_selected_disjoint_exemplars():
    probes = {
        "math": [{"prompt": "Problem: Measurement-only math probe\nSolution:"}],
        "code": [
            {
                "prompt": "# Task: Measurement-only code probe\n"
                "# Write a Python function.\n",
                "test_list": ["assert measured() == 1"],
            }
        ],
        "qa": [
            {
                "prompt": "Context:\nMeasurement-only context\n\n"
                "Question: Measurement-only QA probe?\nAnswer:"
            }
        ],
    }
    n_shots = accuracy.resolve_shot_counts(2)

    accuracy.validate_fewshot_disjoint(probes, n_shots)
    for domain in accuracy.CAPABILITIES:
        prompt = accuracy.build_accuracy_prompt(
            domain, probes[domain][0], n_shots[domain]
        )
        exemplars = accuracy.fewshot_exemplars(domain, n_shots[domain])

        assert len(exemplars) == 2
        assert all(
            accuracy.render_fewshot_exemplar(exemplar) in prompt
            for exemplar in exemplars
        )
        assert all(
            probes[domain][0]["prompt"]
            not in accuracy.render_fewshot_exemplar(exemplar)
            for exemplar in exemplars
        )


def test_zero_shot_prompt_is_unchanged():
    sample = {
        "prompt": "# Task: Return one.\n# Write a Python function.\n",
        "test_list": ["assert one() == 1"],
    }

    assert accuracy.build_accuracy_prompt("code", sample, 0) == sample["prompt"]


def test_code_extraction_strips_mbpp_delimiters():
    generation = "[BEGIN]\ndef one():\n    return 1\n[DONE]\nignored"

    assert accuracy.extract_code_completion(generation) == (
        "def one():\n    return 1"
    )


@pytest.mark.parametrize(
    ("domain", "generation", "expected"),
    [
        (
            "math",
            "120 / 15 = 8 packs. Therefore, \\boxed{8}\n\nProblem: Next one",
            "120 / 15 = 8 packs. Therefore, \\boxed{8}",
        ),
        ("math", "Therefore, \\boxed{8}\nProblem: Next one", "Therefore, \\boxed{8}"),
        (
            "code",
            "def one():\n    return 1\n[DONE]\nYou are an expert Python programmer",
            "def one():\n    return 1\n",
        ),
        (
            "code",
            "def one():\n    return 1\nYou are an expert Python programmer",
            "def one():\n    return 1",
        ),
        ("qa", "Paris\nQuestion: What is next?\nAnswer: London", "Paris"),
        ("qa", "Paris\nContext: A different passage", "Paris"),
    ],
)
def test_truncate_generation_keeps_first_answer_block(domain, generation, expected):
    assert accuracy.truncate_generation(domain, generation) == expected


@pytest.mark.parametrize("accuracy_benchmark", ["default", "easy"])
def test_scoring_uses_first_answer_block_and_keeps_full_generation(
    accuracy_benchmark,
):
    probes = {
        "math": [
            {
                "prompt": "Problem: Parker needs packs.\nSolution:",
                "completion": " 120 / 15 = 8. Therefore, \\boxed{8}",
                "answer": "8",
            }
        ],
        "code": [
            {
                "prompt": "# Task: Return one.\n# Write a Python function.\n",
                "completion": "def one():\n    return 1",
                "test_list": ["assert one() == 1"],
                "test_setup_code": "",
            }
        ],
        "qa": [
            {
                "prompt": "Context:\nFrance facts.\n\nQuestion: Capital?\nAnswer:",
                "completion": " Paris",
                "answer": "Paris",
                "answer_aliases": ["Paris"],
            }
        ],
    }
    generations = {
        "math": [
            "120 / 15 = 8 packs. Therefore, \\boxed{8}\n\n"
            "Problem: Hallucinated follow-on.\nSolution: \\boxed{999}"
        ],
        "code": [
            "def one():\n    return 1\n[DONE]\n"
            "You are an expert Python programmer.\n[BEGIN]\n"
            "def one():\n    return 999\n[DONE]"
        ],
        "qa": ["Paris\nQuestion: Hallucinated follow-on?\nAnswer: London"],
    }

    records, aggregates = accuracy.score_generations(
        probes, generations, accuracy_benchmark=accuracy_benchmark
    )
    by_domain = {record["domain"]: record for record in records}

    assert aggregates["math"]["accuracy"] == 1.0
    assert aggregates["code"]["pass_at_1"] == 1.0
    assert aggregates["qa"]["exact_match"] == 1.0
    assert by_domain["math"]["extracted_answer"] == "8"
    assert by_domain["code"]["executed_completion"] == (
        "def one():\n    return 1"
    )
    assert by_domain["qa"]["extracted_answer"] == "Paris"
    for domain in accuracy.CAPABILITIES:
        assert by_domain[domain]["raw_generation_full"] == generations[domain][0]
        assert by_domain[domain]["generation"] == accuracy.truncate_generation(
            domain, generations[domain][0]
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("We compute carefully.\n#### 42\nConfidence: 90%", "42"),
        ("Reasoning.\n#### -1,234", "-1,234"),
        ("Therefore, the answer is 17.", "17"),
        ("The final answer is -2.5 dollars.", "-2.5"),
        ("Work\n#### $0.125$\n", "0.125"),
    ],
)
def test_gsm8k_final_answer_extraction(text, expected):
    extracted = accuracy.extract_gsm8k_answer(text)

    assert accuracy.normalize_math_answer(extracted) == (
        accuracy.normalize_math_answer(expected)
    )


def test_triviaqa_alias_match_and_max_token_f1():
    aliases = ["NYC", "New York City", "City of New York"]

    assert accuracy.qa_alias_exact_match("The New York City", aliases)
    assert not accuracy.qa_alias_exact_match("New York State", aliases)
    assert accuracy.qa_alias_token_f1("New York", aliases) == pytest.approx(0.8)
    assert accuracy.extract_triviaqa_answer("The answer is New York City.") == (
        "New York City"
    )


def test_easy_and_default_use_distinct_output_directories(tmp_path):
    default_file = accuracy.checkpoint_output_file(
        tmp_path, "student", "dense", "default"
    )
    easy_file = accuracy.checkpoint_output_file(
        tmp_path, "student", "dense", "easy"
    )

    assert default_file == tmp_path / "student/dense/accuracy.json"
    assert easy_file == tmp_path / "student/dense__easy/accuracy.json"
    assert default_file != easy_file


def test_compressed_dense_checkpoint_name_omits_dense_prefix():
    assert accuracy._checkpoint_name(None, 0.5, None) == "prune-d0.5"
    assert accuracy._checkpoint_name(None, None, 4) == "quant-b4"
