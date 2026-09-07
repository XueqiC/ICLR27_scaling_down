#!/usr/bin/env python3
"""V15: task accuracy on the V12 capability-loss measurement probes.

The odd-indexed half of the deterministic V6 probes is used for both
completion loss and greedy task evaluation, producing paired ``(L_c, A_c)``
observations for link-function fitting.  Completion loss remains zero-shot;
only the greedy task-evaluation prompt receives fixed few-shot exemplars.

Examples:
  python3 analysis/v15_accuracy_link.py --model gemma3-270m --device cuda:0
  python3 analysis/v15_accuracy_link.py --model gemma3-270m \
      --adapter results/v12-distill/gemma3-270m/RUN/adapter
  python3 analysis/v15_accuracy_link.py \
      --v12-sweep results/v12-distill --device cuda:0
  python3 analysis/v15_accuracy_link.py --summarize
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import re
import resource
import string
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

try:
    from .v6_capability_geometry import (
        apply_global_magnitude_pruning,
        build_probes,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
    )
    from .v10_quantization import fake_quantize_per_output_channel
    from .v12_distill import (
        CAPABILITIES,
        MAX_LEN,
        measure_capability_losses,
        seed_everything,
        write_json_atomic,
    )
except ImportError:  # direct execution: python analysis/v15_accuracy_link.py
    from v6_capability_geometry import (
        apply_global_magnitude_pruning,
        build_probes,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
    )
    from v10_quantization import fake_quantize_per_output_channel
    from v12_distill import (
        CAPABILITIES,
        MAX_LEN,
        measure_capability_losses,
        seed_everything,
        write_json_atomic,
    )


ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "results/v15-accuracy"
DEFAULT_V12_ROOT = ROOT / "results/v12-distill"
N_PROBE = 128
SEED = 0
MAX_NEW_TOKENS = {"math": 512, "code": 384, "qa": 32}
DOMAIN_DEFAULT_SHOTS = {"math": 4, "code": 3, "qa": 4}
ACCURACY_BENCHMARKS = {
    "default": {
        "math": "MATH-500",
        "code": "MBPP",
        "qa": "2WikiMultihopQA",
    },
    "easy": {
        "math": "GSM8K",
        "code": "MBPP",
        "qa": "TriviaQA",
    },
}
ACCURACY_BENCHMARK_DATASETS = {
    "default": {
        "math": "HuggingFaceH4/MATH-500 test",
        "code": "google-research-datasets/mbpp full test",
        "qa": "framolfese/2WikiMultihopQA validation",
    },
    "easy": {
        "math": "openai/gsm8k main test",
        "code": "google-research-datasets/mbpp full test",
        "qa": "mandarjoshi/trivia_qa rc.nocontext validation",
    },
}
CODE_TIMEOUT_SECONDS = 10.0
CODE_MEMORY_BYTES = 512 * 1024 * 1024
CODE_FILE_BYTES = 1024 * 1024

_FENCED_CODE_RE = re.compile(
    r"```(?:python|py)?\s*\r?\n(?P<body>.*?)```", re.IGNORECASE | re.DOTALL
)
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_MATH_UNIT_RE = re.compile(
    r"(?:degrees?|radians?|percent|inches?|feet|yards?|miles?|"
    r"millimeters?|centimeters?|meters?|kilometers?|mm|cm|km|m|"
    r"seconds?|minutes?|hours?|days?|grams?|kilograms?|kg|g)"
    r"(?:\s*\^\s*\{?[-+]?\d+\}?)?\s*$",
    re.IGNORECASE,
)
DOMAIN_STOP_SEQUENCES = {
    "math": ("\nProblem:", "\n\nProblem:"),
    "code": ("[DONE]", "\nYou are an expert"),
    "qa": ("\nQuestion:", "\nContext:"),
}
_MATH_CONTINUATION_RE = re.compile(r"\n+(?=Problem:)")


# These are deliberately embedded instead of loaded at evaluation time.  The
# math examples are small canonical worked problems rather than MATH-500 rows
# (MATH-500 has no training split).  MBPP task IDs 2/3/4 are the benchmark's
# canonical prompt examples.  The QA rows were selected once from the 2Wiki
# training split with np.random.default_rng(0); only their supporting context
# is retained to keep the demonstrations short.  All three pools are disjoint
# from build_probes(), which uses MATH-500 test, MBPP test, and 2Wiki validation.
FEWSHOT = {
    "seed": SEED,
    "math": {
        "source": "hardcoded canonical worked MATH-style problems (not MATH-500)",
        "exemplars": [
            {
                "id": "canonical-linear-equation",
                "prompt": "Problem: Solve for x: 3x + 7 = 25.\nSolution:",
                "completion": (
                    " Subtracting 7 gives 3x = 18. Dividing by 3 gives "
                    "x = 6. Therefore, \\boxed{6}"
                ),
            },
            {
                "id": "canonical-fraction-sum",
                "prompt": (
                    "Problem: Compute \\frac{3}{4} + \\frac{5}{8} and simplify."
                    "\nSolution:"
                ),
                "completion": (
                    " Using denominator 8, \\frac{3}{4}=\\frac{6}{8}. Thus "
                    "\\frac{6}{8}+\\frac{5}{8}=\\frac{11}{8}. Therefore, "
                    "\\boxed{\\frac{11}{8}}"
                ),
            },
            {
                "id": "canonical-pentagon-angles",
                "prompt": (
                    "Problem: What is the sum, in degrees, of the interior "
                    "angles of a pentagon?\nSolution:"
                ),
                "completion": (
                    " An n-gon's interior angles sum to (n-2)\\cdot180^\\circ. "
                    "For n=5 this is 3\\cdot180^\\circ=540^\\circ. Therefore, "
                    "\\boxed{540}"
                ),
            },
            {
                "id": "canonical-quadratic-roots",
                "prompt": (
                    "Problem: The roots of x^2-5x+6=0 are r and s. Find r^2+s^2."
                    "\nSolution:"
                ),
                "completion": (
                    " Factoring gives (x-2)(x-3)=0, so r and s are 2 and 3. "
                    "Hence r^2+s^2=2^2+3^2=13. Therefore, \\boxed{13}"
                ),
            },
        ],
    },
    "code": {
        "source": (
            "google-research-datasets/mbpp canonical prompt examples, "
            "task_ids 2,3,4"
        ),
        "exemplars": [
            {
                "id": 2,
                "prompt": (
                    "You are an expert Python programmer, and here is your task: "
                    "Write a function to find the similar elements from the given "
                    "two tuple lists. Your code should pass these tests:\n"
                    "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n"
                    "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)\n"
                    "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) "
                    "== (13, 14)\n[BEGIN]\n"
                ),
                "completion": (
                    "def similar_elements(test_tup1, test_tup2):\n"
                    "  res = tuple(set(test_tup1) & set(test_tup2))\n"
                    "  return (res)\n[DONE]"
                ),
            },
            {
                "id": 3,
                "prompt": (
                    "You are an expert Python programmer, and here is your task: "
                    "Write a python function to identify non-prime numbers. Your "
                    "code should pass these tests:\n"
                    "assert is_not_prime(2) == False\n"
                    "assert is_not_prime(10) == True\n"
                    "assert is_not_prime(35) == True\n[BEGIN]\n"
                ),
                "completion": (
                    "import math\n\n"
                    "def is_not_prime(n):\n"
                    "    result = False\n"
                    "    for i in range(2, int(math.sqrt(n)) + 1):\n"
                    "        if n % i == 0:\n"
                    "            result = True\n"
                    "    return result\n[DONE]"
                ),
            },
            {
                "id": 4,
                "prompt": (
                    "You are an expert Python programmer, and here is your task: "
                    "Write a function to find the largest integers from a given "
                    "list of numbers using heap queue algorithm. Your code should "
                    "pass these tests:\n"
                    "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, "
                    "58], 3) == [85, 75, 65]\n"
                    "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, "
                    "58], 2) == [85, 75]\n"
                    "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, "
                    "58], 5) == [85, 75, 65, 58, 35]\n[BEGIN]\n"
                ),
                "completion": (
                    "import heapq as hq\n\n"
                    "def heap_queue_largest(nums, n):\n"
                    "  largest_nums = hq.nlargest(n, nums)\n"
                    "  return largest_nums\n[DONE]"
                ),
            },
        ],
    },
    "qa": {
        "source": (
            "framolfese/2WikiMultihopQA train; seed-0 row indices "
            "45176,85591,142437,106660"
        ),
        "exemplars": [
            {
                "id": "65b10f7a0bda11eba7f7acde48001122",
                "row_index": 45176,
                "prompt": (
                    "Context:\n"
                    "Black Moon (1934 film): Black Moon is a 1934 American pre-Code "
                    "horror film directed by Roy William Neill, and starring Jack "
                    "Holt, Fay Wray, and Dorothy Burgess.\n"
                    "Roy William Neill: Roy William Neill (4 September 1887 - 14 "
                    "December 1946) was an Irish-born American film director.\n\n"
                    "Question: When did the director of film Black Moon (1934 Film) "
                    "die?\nAnswer:"
                ),
                "completion": " 14 December 1946",
            },
            {
                "id": "307d6b8608b111ebbd85ac1f6bf848b6",
                "row_index": 85591,
                "prompt": (
                    "Context:\n"
                    "The Perfect Candidate: The Perfect Candidate is a 2019 Saudi "
                    "Arabian drama film directed by Haifaa al-Mansour.\n"
                    "Reckless (1951 film): Reckless is a 1951 Spanish drama film "
                    "directed by Jose Antonio Nieves Conde.\n"
                    "Haifaa al-Mansour: Haifaa al-Mansour was born 10 August 1974.\n"
                    "Jose Antonio Nieves Conde: Jose Antonio Nieves Conde was born "
                    "22 December 1915.\n\n"
                    "Question: Which film whose director is younger, The Perfect "
                    "Candidate or Reckless (1951 Film)?\nAnswer:"
                ),
                "completion": " The Perfect Candidate",
            },
            {
                "id": "4777c6220bdd11eba7f7acde48001122",
                "row_index": 142437,
                "prompt": (
                    "Context:\n"
                    "The Brat (1919 film): The Brat is a 1919 American silent drama "
                    "film directed by Herbert Blache.\n"
                    "Herbert Blache: Herbert Blache (5 October 1882 - 23 October "
                    "1953) was a British-born American film director.\n\n"
                    "Question: When did the director of film The Brat (1919 Film) "
                    "die?\nAnswer:"
                ),
                "completion": " 23 October 1953",
            },
            {
                "id": "852078b108dc11ebbd9cac1f6bf848b6",
                "row_index": 106660,
                "prompt": (
                    "Context:\n"
                    "Blue Demon (film): Blue Demon is a 2004 American film directed "
                    "by Daniel Grodnik.\n"
                    "Clash of the Titans (2010 film): Clash of the Titans was "
                    "directed by Louis Leterrier.\n"
                    "Daniel Grodnik: Daniel Grodnik is an American film producer.\n"
                    "Louis Leterrier: Louis Leterrier is a French film director.\n\n"
                    "Question: Are both directors of films Clash of the Titans "
                    "(2010 film) and Blue Demon (film) from the same country?\n"
                    "Answer:"
                ),
                "completion": " no",
            },
        ],
    },
}


def resolve_shot_counts(shots: int) -> dict[str, int]:
    """Apply one global shot cap to the fixed per-domain exemplar pools."""
    if shots < 0:
        raise ValueError("shots must be non-negative")
    maximum = max(DOMAIN_DEFAULT_SHOTS.values())
    if shots > maximum:
        raise ValueError(f"shots must be at most {maximum}")
    return {
        domain: min(shots, DOMAIN_DEFAULT_SHOTS[domain]) for domain in CAPABILITIES
    }


def fewshot_exemplars(domain: str, n_shots: int) -> list[Mapping]:
    """Return the fixed prefix of a domain's disjoint exemplar pool."""
    if domain not in CAPABILITIES:
        raise ValueError(f"unknown capability: {domain!r}")
    exemplars = FEWSHOT[domain]["exemplars"]
    if not 0 <= n_shots <= len(exemplars):
        raise ValueError(
            f"{domain} supports between 0 and {len(exemplars)} shots, got {n_shots}"
        )
    return list(exemplars[:n_shots])


def _code_task_text(sample: Mapping) -> str:
    prompt = str(sample["prompt"])
    prefix = "# Task: "
    suffix = "\n# Write a Python function.\n"
    if prompt.startswith(prefix) and prompt.endswith(suffix):
        return prompt[len(prefix) : -len(suffix)]
    return prompt.strip()


def _format_code_probe(sample: Mapping) -> str:
    tests = "\n".join(str(test) for test in sample.get("test_list", []))
    return (
        "You are an expert Python programmer, and here is your task: "
        f"{_code_task_text(sample)} Your code should pass these tests:\n"
        f"{tests}\n[BEGIN]\n"
    )


def render_fewshot_exemplar(exemplar: Mapping) -> str:
    return str(exemplar["prompt"]) + str(exemplar["completion"])


def build_accuracy_prompt(domain: str, sample: Mapping, n_shots: int) -> str:
    """Build only the greedy-accuracy prompt; loss probes remain unmodified."""
    exemplars = fewshot_exemplars(domain, n_shots)
    if not exemplars:
        # This exact return is what makes --shots 0 reproduce V15 zero-shot.
        return str(sample["prompt"])
    target = _format_code_probe(sample) if domain == "code" else str(sample["prompt"])
    return "\n\n".join(
        [*(render_fewshot_exemplar(exemplar) for exemplar in exemplars), target]
    )


def validate_fewshot_disjoint(
    probes: Mapping[str, Sequence[Mapping]],
    n_shots: Mapping[str, int],
) -> None:
    """Fail closed if a selected exemplar contains a measurement prompt."""
    for domain in CAPABILITIES:
        rendered = [
            render_fewshot_exemplar(exemplar)
            for exemplar in fewshot_exemplars(domain, n_shots[domain])
        ]
        for sample in probes[domain]:
            measurement_prompt = str(sample["prompt"])
            probe_strings = [measurement_prompt]
            if domain == "code":
                # Code demonstrations use canonical MBPP formatting whereas
                # V6 loss probes use ``# Task: ...``.  Compare task text too.
                probe_strings.append(_code_task_text(sample))
            if any(
                probe_string and probe_string in exemplar
                for probe_string in probe_strings
                for exemplar in rendered
            ):
                raise RuntimeError(
                    f"few-shot leakage: a {domain} measurement prompt occurs in "
                    "an exemplar"
                )


def fewshot_metadata(n_shots: Mapping[str, int]) -> dict:
    """Return auditable IDs and content hashes for the selected exemplars."""
    metadata = {"seed": FEWSHOT["seed"]}
    for domain in CAPABILITIES:
        selected = fewshot_exemplars(domain, n_shots[domain])
        metadata[domain] = {
            "k": n_shots[domain],
            "n_shots": n_shots[domain],
            "source": FEWSHOT[domain]["source"],
            "exemplars": [
                {
                    "id": exemplar["id"],
                    **(
                        {"row_index": exemplar["row_index"]}
                        if "row_index" in exemplar
                        else {}
                    ),
                    "sha256": hashlib.sha256(
                        render_fewshot_exemplar(exemplar).encode("utf-8")
                    ).hexdigest(),
                }
                for exemplar in selected
            ],
        }
    return metadata


def _last_boxed(text: str) -> str | None:
    """Return the contents of the last balanced ``\\boxed{...}``."""
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    for match in reversed(matches):
        opening = match.end() - 1
        depth = 0
        for index in range(opening, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[opening + 1 : index]
    simple = list(re.finditer(r"\\boxed\s+([^\s$.,]+)", text))
    return simple[-1].group(1) if simple else None


def extract_math_answer(text: str) -> str:
    """Extract a generated final answer, preferring the last boxed result."""
    text = str(text).strip()
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed.strip()

    if "####" in text:
        return text.rsplit("####", 1)[1].strip().splitlines()[0].strip()

    answer_matches = list(
        re.finditer(
            r"(?:final\s+answer|answer)\s*(?:(?:is)\s*)?(?:[:=]\s*)?(.+)",
            text,
            re.IGNORECASE,
        )
    )
    if answer_matches:
        candidate = answer_matches[-1].group(1).strip().splitlines()[0]
        return candidate.strip()

    displayed = re.findall(r"\$+([^$\n]+?)\$+", text)
    if displayed:
        return displayed[-1].strip()

    fractions = list(
        re.finditer(
            r"-?\\(?:dfrac|tfrac|frac)\s*\{[^{}]+\}\s*\{[^{}]+\}",
            text,
        )
    )
    slash_fractions = list(re.finditer(r"(?<![\w.])-?\d+\s*/\s*-?\d+", text))
    numbers = list(
        re.finditer(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?", text)
    )
    candidates = [*fractions, *slash_fractions, *numbers]
    if candidates:
        return max(candidates, key=lambda match: match.end()).group(0)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def extract_gsm8k_answer(text: str) -> str:
    """Extract GSM8K's final numeric answer from a reference or generation."""
    value = str(text).strip()
    if "####" in value:
        tail = value.rsplit("####", 1)[1].strip()
        final_line = tail.splitlines()[0] if tail else ""
        return extract_math_answer(final_line)
    answer_matches = list(
        re.finditer(
            r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?[ \t]*(.+)",
            value,
            re.IGNORECASE,
        )
    )
    if answer_matches:
        return extract_math_answer(answer_matches[-1].group(1).splitlines()[0])
    return extract_math_answer(value)


def normalize_math_answer(answer: str) -> str:
    """Apply benchmark-stable surface normalization to a math answer."""
    value = str(answer).strip().replace("−", "-")
    boxed = _last_boxed(value)
    if boxed is not None:
        value = boxed.strip()
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\!", "").replace("\\,", "")
    value = value.replace("\\%", "%")
    value = value.rstrip().rstrip(".").rstrip()
    value = value.strip("$").strip()
    value = re.sub(
        r"\s*\\(?:text|mathrm|mbox)\s*\{[^{}]*\}"
        r"(?:\s*\^\s*\{?[-+]?\d+\}?)?\s*$",
        "",
        value,
    )
    value = _MATH_UNIT_RE.sub("", value).strip()
    value = re.sub(r"\s*%\s*$", "", value)
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"(?<=\d),(?=\d)", "", value)

    while len(value) >= 2 and value[0] == "{" and value[-1] == "}":
        depth = 0
        balanced = True
        for index, character in enumerate(value):
            depth += character == "{"
            depth -= character == "}"
            if depth == 0 and index != len(value) - 1:
                balanced = False
                break
        if not balanced:
            break
        value = value[1:-1]

    if value.count("=") == 1:
        left, right = value.split("=", 1)
        if len(left) <= 2:
            value = right
    value = re.sub(r"^(-?)\.(\d+)$", r"\g<1>0.\2", value)
    value = re.sub(
        r"^\\frac\s*([+-]?\d)\s*([+-]?\d)$",
        r"\\frac{\1}{\2}",
        value,
    )
    slash_fraction = re.fullmatch(r"([+-]?\d+)\/([+-]?\d+)", value)
    if slash_fraction:
        value = rf"\frac{{{slash_fraction.group(1)}}}{{{slash_fraction.group(2)}}}"
    return value


def math_exact_match(generation: str, reference: str) -> tuple[bool, str, str, str]:
    extracted = extract_math_answer(generation)
    prediction = normalize_math_answer(extracted)
    target = normalize_math_answer(reference)
    return prediction == target, extracted, prediction, target


def gsm8k_exact_match(
    generation: str, reference: str
) -> tuple[bool, str, str, str]:
    extracted = extract_gsm8k_answer(generation)
    prediction = normalize_math_answer(extracted)
    target = normalize_math_answer(extract_gsm8k_answer(reference))
    return prediction == target, extracted, prediction, target


def normalize_qa_answer(answer: str) -> str:
    """SQuAD exact-match/F1 normalization."""
    lowered = str(answer).lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = _ARTICLES_RE.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def qa_exact_match(prediction: str, reference: str) -> bool:
    return normalize_qa_answer(prediction) == normalize_qa_answer(reference)


def qa_token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_qa_answer(prediction).split()
    reference_tokens = normalize_qa_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def qa_alias_exact_match(prediction: str, aliases: Sequence[str]) -> bool:
    """TriviaQA exact match against any accepted answer alias."""
    return any(qa_exact_match(prediction, alias) for alias in aliases)


def qa_alias_token_f1(prediction: str, aliases: Sequence[str]) -> float:
    """TriviaQA's maximum token F1 over all accepted answer aliases."""
    return max((qa_token_f1(prediction, alias) for alias in aliases), default=0.0)


def extract_triviaqa_answer(text: str) -> str:
    """Strip a common answer lead-in while retaining free-form entity text."""
    value = str(text).strip()
    matches = list(
        re.finditer(r"(?:final\s+)?answer\s*(?:is|:)[ \t]*(.+)", value, re.I)
    )
    if matches:
        value = matches[-1].group(1).splitlines()[0].strip()
    return value.rstrip().rstrip(".").strip()


# Descriptive aliases for notebook/test callers.
triviaqa_alias_exact_match = qa_alias_exact_match
triviaqa_alias_token_f1 = qa_alias_token_f1


def extract_code_completion(generation: str) -> str:
    """Extract Python from canonical MBPP delimiters or a Markdown fence."""
    text = str(generation)
    if "[BEGIN]" in text:
        text = text.rsplit("[BEGIN]", 1)[1]
    if "[DONE]" in text:
        text = text.split("[DONE]", 1)[0]
    matches = list(_FENCED_CODE_RE.finditer(text))
    return matches[-1].group("body").strip() if matches else text.strip()


def truncate_generation(domain: str, raw_generation: str) -> str:
    """Keep only the first answer block in a decoded domain completion."""
    text = str(raw_generation)
    if domain == "math":
        match = _MATH_CONTINUATION_RE.search(text)
        cut = match.start() if match is not None else None
    elif domain == "code":
        cut = text.find("[DONE]")
        if cut < 0:
            cut = text.find("\nYou are an expert")
        cut = cut if cut >= 0 else None
    elif domain == "qa":
        positions = [
            position
            for delimiter in DOMAIN_STOP_SEQUENCES[domain]
            if (position := text.find(delimiter)) >= 0
        ]
        cut = min(positions) if positions else None
    else:
        raise ValueError(f"unknown capability: {domain!r}")
    return text if cut is None else text[:cut]


def _set_code_resource_limits(timeout_seconds: float) -> None:
    cpu_soft = max(1, int(math.ceil(timeout_seconds)))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_soft, cpu_soft + 1))
    resource.setrlimit(resource.RLIMIT_AS, (CODE_MEMORY_BYTES, CODE_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (CODE_FILE_BYTES, CODE_FILE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _tail_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return (value or "")[-4000:]


def run_code_tests(
    completion: str,
    test_list: Sequence[str],
    test_setup_code: str = "",
    *,
    timeout_seconds: float = CODE_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Execute one MBPP completion and its tests in an isolated temp cwd."""
    if timeout_seconds <= 0:
        raise ValueError("code timeout must be positive")
    if isinstance(test_list, (str, bytes)):
        raise TypeError("test_list must be a sequence of test statements")
    program_parts = [str(test_setup_code), extract_code_completion(completion)]
    program_parts.extend(str(test) for test in test_list)
    program = "\n\n".join(part for part in program_parts if part.strip()) + "\n"

    with tempfile.TemporaryDirectory(prefix="v15-mbpp-", dir="/tmp") as temporary:
        sandbox_environment = {
            "CUDA_VISIBLE_DEVICES": "",
            "HOME": temporary,
            "LANG": "C.UTF-8",
            "PATH": os.defpath,
            "PYTHONHASHSEED": str(SEED),
            "TMPDIR": temporary,
        }
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", program],
                cwd=temporary,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=lambda: _set_code_resource_limits(timeout_seconds),
                env=sandbox_environment,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "passed": False,
                "status": "timeout",
                "returncode": None,
                "stdout": _tail_text(exc.stdout),
                "stderr": _tail_text(exc.stderr),
            }
    return {
        "passed": result.returncode == 0,
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": _tail_text(result.stdout),
        "stderr": _tail_text(result.stderr),
    }


def _triviaqa_aliases(answer: object) -> list[str]:
    if isinstance(answer, Mapping):
        raw_aliases = answer.get("aliases") or []
        if isinstance(raw_aliases, (str, bytes)):
            raw_aliases = [raw_aliases]
        candidates = [answer.get("value"), *raw_aliases]
    elif isinstance(answer, Sequence) and not isinstance(answer, (str, bytes)):
        candidates = list(answer)
    else:
        candidates = [answer]
    aliases = []
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value and value not in aliases:
            aliases.append(value)
    if not aliases:
        raise ValueError("TriviaQA row has no non-empty answer aliases")
    return aliases


def build_easy_probes(n: int = N_PROBE, seed: int = SEED) -> dict[str, list[dict]]:
    """Build GSM8K/MBPP/closed-book TriviaQA probes before half selection."""
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    probes: dict[str, list[dict]] = {}

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    indices = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    probes["math"] = []
    for index in indices:
        row = dataset[int(index)]
        reference = str(row["answer"])
        probes["math"].append(
            {
                "prompt": f"Problem: {row['question']}\nSolution:",
                "completion": " " + reference,
                "answer": extract_gsm8k_answer(reference),
            }
        )

    dataset = load_dataset(
        "google-research-datasets/mbpp", "full", split="test"
    )
    indices = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    probes["code"] = [
        {
            "prompt": (
                f"# Task: {dataset[int(index)]['text']}\n"
                "# Write a Python function.\n"
            ),
            "completion": dataset[int(index)]["code"],
            "test_list": dataset[int(index)].get("test_list", []),
            "test_setup_code": dataset[int(index)].get("test_setup_code", ""),
            "challenge_test_list": dataset[int(index)].get(
                "challenge_test_list", []
            ),
        }
        for index in indices
    ]

    dataset = load_dataset(
        "mandarjoshi/trivia_qa", "rc.nocontext", split="validation"
    )
    indices = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    probes["qa"] = []
    for index in indices:
        row = dataset[int(index)]
        aliases = _triviaqa_aliases(row["answer"])
        probes["qa"].append(
            {
                "prompt": f"Question: {row['question']}\nAnswer:",
                "completion": " " + aliases[0],
                "answer": aliases[0],
                "answer_aliases": aliases,
            }
        )
    return probes


def measurement_probes(
    accuracy_benchmark: str = "default",
) -> dict[str, list[dict]]:
    """Build the selected probes and retain the odd-indexed measurement half."""
    if accuracy_benchmark not in ACCURACY_BENCHMARKS:
        raise ValueError(
            f"unknown accuracy benchmark {accuracy_benchmark!r}; expected "
            f"one of {tuple(ACCURACY_BENCHMARKS)}"
        )
    all_probes = (
        build_probes(n=N_PROBE, seed=SEED)
        if accuracy_benchmark == "default"
        else build_easy_probes(n=N_PROBE, seed=SEED)
    )
    probes = {
        capability: all_probes[capability][1::2] for capability in CAPABILITIES
    }
    if any(not probes[capability] for capability in CAPABILITIES):
        raise RuntimeError("V6 measurement-half probes must be non-empty")
    return probes


def _batch_decode(tokenizer, token_ids: torch.Tensor) -> list[str]:
    try:
        return tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.batch_decode(token_ids, skip_special_tokens=True)


class _StopOnTokenSequences:
    """Mark each batch row finished after it emits a domain delimiter."""

    def __init__(self, stop_token_sequences: Sequence[Sequence[int]]):
        self.stop_token_sequences = tuple(
            tuple(sequence) for sequence in stop_token_sequences if sequence
        )

    def __call__(self, input_ids, scores=None, **kwargs):
        stopped = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        for sequence in self.stop_token_sequences:
            if input_ids.shape[1] < len(sequence):
                continue
            suffix = input_ids[:, -len(sequence) :]
            expected = torch.tensor(
                sequence, dtype=input_ids.dtype, device=input_ids.device
            )
            stopped |= (suffix == expected).all(dim=1)
        return stopped


def _encode_stop_sequences(tokenizer, stop_sequences: Sequence[str]) -> list[list[int]]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return []
    encoded = []
    for sequence in stop_sequences:
        token_ids = encode(sequence, add_special_tokens=False)
        if token_ids:
            encoded.append([int(token_id) for token_id in token_ids])
    return encoded


def _generate_supports_keyword(generate, keyword: str) -> bool:
    try:
        parameters = inspect.signature(generate).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == keyword
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _transformers_stopping_criteria(stop_token_sequences: Sequence[Sequence[int]]):
    if not stop_token_sequences:
        return None
    try:
        from transformers import StoppingCriteriaList
    except ImportError:
        return None
    return StoppingCriteriaList([_StopOnTokenSequences(stop_token_sequences)])


def _fallback_greedy_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
    pad_token_id: int,
    stop_token_sequences: Sequence[Sequence[int]] = (),
) -> torch.Tensor:
    """Greedy cached decoding for the language-only multimodal adapter."""
    generated: list[torch.Tensor] = []
    sequences = input_ids
    current_ids = input_ids
    current_attention = attention_mask
    past_key_values = None
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    for _ in range(max_new_tokens):
        kwargs = {
            "input_ids": current_ids,
            "attention_mask": current_attention,
            "use_cache": True,
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        outputs = model(**kwargs)
        next_tokens = outputs.logits[:, -1].argmax(dim=-1)
        next_tokens = torch.where(
            finished,
            torch.full_like(next_tokens, pad_token_id),
            next_tokens,
        )
        generated.append(next_tokens)
        if eos_token_ids:
            emitted_eos = torch.zeros_like(finished)
            for eos_token_id in eos_token_ids:
                emitted_eos |= next_tokens == eos_token_id
            finished |= emitted_eos
        for sequence in stop_token_sequences:
            if len(generated) < len(sequence):
                continue
            suffix = torch.stack(generated[-len(sequence) :], dim=1)
            expected = torch.tensor(
                sequence, dtype=input_ids.dtype, device=input_ids.device
            )
            finished |= (suffix == expected).all(dim=1)
        if bool(finished.all()):
            break

        sequences = torch.cat((sequences, next_tokens[:, None]), dim=1)
        current_attention = torch.cat(
            (
                current_attention,
                torch.ones(
                    (current_attention.shape[0], 1),
                    dtype=current_attention.dtype,
                    device=current_attention.device,
                ),
            ),
            dim=1,
        )
        past_key_values = getattr(outputs, "past_key_values", None)
        current_ids = next_tokens[:, None] if past_key_values is not None else sequences

    if not generated:
        return torch.empty((input_ids.shape[0], 0), dtype=input_ids.dtype, device=input_ids.device)
    return torch.stack(generated, dim=1)


def greedy_generate_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    device: str,
    max_new_tokens: int,
    input_max_length: int = MAX_LEN // 2,
    truncation_side: str | None = None,
    stop_sequences: Sequence[str] = (),
) -> list[str]:
    """Generate raw completions greedily from left-padded base prompts."""
    if not prompts:
        return []
    tokenizer.padding_side = "left"
    if truncation_side is not None:
        tokenizer.truncation_side = truncation_side
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif getattr(tokenizer, "bos_token", None) is not None:
            tokenizer.pad_token = tokenizer.bos_token
        else:
            raise ValueError("tokenizer needs a pad, EOS, or BOS token for batching")
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=input_max_length,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
    else:
        attention_mask = attention_mask.to(device)
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))
    stop_token_sequences = _encode_stop_sequences(tokenizer, stop_sequences)

    model.eval()
    with torch.inference_mode():
        generate = getattr(model, "generate", None)
        if callable(generate):
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "use_cache": True,
                "eos_token_id": eos,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if _generate_supports_keyword(generate, "stopping_criteria"):
                stopping_criteria = _transformers_stopping_criteria(
                    stop_token_sequences
                )
                if stopping_criteria is not None:
                    generate_kwargs["stopping_criteria"] = stopping_criteria
            output_ids = generate(
                **generate_kwargs,
            )
            new_token_ids = output_ids[:, input_ids.shape[1] :]
        else:
            new_token_ids = _fallback_greedy_generate(
                model,
                input_ids,
                attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                stop_token_sequences=stop_token_sequences,
            )
    return _batch_decode(tokenizer, new_token_ids)


def generate_domain(
    model,
    tokenizer,
    samples: Sequence[Mapping],
    domain: str,
    device: str,
    batch_size: int,
    n_shots: int,
) -> list[str]:
    generations: list[str] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        generations.extend(
            greedy_generate_batch(
                model,
                tokenizer,
                [build_accuracy_prompt(domain, sample, n_shots) for sample in batch],
                device=device,
                max_new_tokens=MAX_NEW_TOKENS[domain],
                input_max_length=(
                    MAX_LEN // 2
                    if n_shots == 0
                    else max(MAX_LEN // 2, MAX_LEN - MAX_NEW_TOKENS[domain])
                ),
                # Few-shot prefixes put the actual question at the end.  If a
                # long QA context must be truncated, retain that target suffix.
                truncation_side="left" if n_shots else None,
                stop_sequences=DOMAIN_STOP_SEQUENCES[domain],
            )
        )
    return generations


def score_generations(
    probes: Mapping[str, Sequence[Mapping]],
    generations: Mapping[str, Sequence[str]],
    accuracy_benchmark: str = "default",
) -> tuple[list[dict], dict[str, dict[str, float | int]]]:
    """Return per-probe scores and task-level accuracy aggregates."""
    if accuracy_benchmark not in ACCURACY_BENCHMARKS:
        raise ValueError(f"unknown accuracy benchmark: {accuracy_benchmark!r}")
    records: list[dict] = []
    aggregates: dict[str, dict[str, float | int]] = {}

    math_correct = 0
    for index, (sample, raw_generation_full) in enumerate(
        zip(probes["math"], generations["math"], strict=True)
    ):
        generation = truncate_generation("math", raw_generation_full)
        scorer = gsm8k_exact_match if accuracy_benchmark == "easy" else math_exact_match
        correct, extracted, prediction, target = scorer(generation, str(sample["answer"]))
        math_correct += int(correct)
        records.append(
            {
                "domain": "math",
                "probe_index": index,
                "prompt": sample["prompt"],
                "reference_completion": sample["completion"],
                "reference_answer": sample["answer"],
                "raw_generation_full": raw_generation_full,
                "generation": generation,
                "extracted_answer": extracted,
                "normalized_prediction": prediction,
                "normalized_reference": target,
                "exact_match": correct,
            }
        )
    aggregates["math"] = {
        "n": len(probes["math"]),
        "correct": math_correct,
        "accuracy": math_correct / len(probes["math"]),
    }

    code_correct = 0
    for index, (sample, raw_generation_full) in enumerate(
        zip(probes["code"], generations["code"], strict=True)
    ):
        generation = truncate_generation("code", raw_generation_full)
        execution = run_code_tests(
            generation,
            sample["test_list"],
            str(sample.get("test_setup_code", "")),
        )
        code_correct += int(execution["passed"])
        records.append(
            {
                "domain": "code",
                "probe_index": index,
                "prompt": sample["prompt"],
                "reference_completion": sample["completion"],
                "test_list": sample["test_list"],
                "test_setup_code": sample.get("test_setup_code", ""),
                "challenge_test_list": sample.get("challenge_test_list", []),
                "raw_generation_full": raw_generation_full,
                "generation": generation,
                "executed_completion": extract_code_completion(generation),
                "passed": execution["passed"],
                "execution": execution,
            }
        )
    aggregates["code"] = {
        "n": len(probes["code"]),
        "passed": code_correct,
        "pass_at_1": code_correct / len(probes["code"]),
        "accuracy": code_correct / len(probes["code"]),
    }

    qa_em_total = 0
    qa_f1_total = 0.0
    for index, (sample, raw_generation_full) in enumerate(
        zip(probes["qa"], generations["qa"], strict=True)
    ):
        generation = truncate_generation("qa", raw_generation_full)
        reference = str(sample["answer"])
        references = (
            [str(value) for value in sample.get("answer_aliases", [reference])]
            if accuracy_benchmark == "easy"
            else [reference]
        )
        prediction = (
            extract_triviaqa_answer(generation)
            if accuracy_benchmark == "easy"
            else generation
        )
        exact_match = qa_alias_exact_match(prediction, references)
        token_f1 = qa_alias_token_f1(prediction, references)
        qa_em_total += int(exact_match)
        qa_f1_total += token_f1
        records.append(
            {
                "domain": "qa",
                "probe_index": index,
                "prompt": sample["prompt"],
                "reference_completion": sample["completion"],
                "reference_answer": reference,
                "reference_answers": references,
                "raw_generation_full": raw_generation_full,
                "generation": generation,
                "extracted_answer": prediction,
                "normalized_prediction": normalize_qa_answer(prediction),
                "normalized_reference": normalize_qa_answer(reference),
                "normalized_references": [
                    normalize_qa_answer(value) for value in references
                ],
                "exact_match": exact_match,
                "token_f1": token_f1,
            }
        )
    aggregates["qa"] = {
        "n": len(probes["qa"]),
        "exact_matches": qa_em_total,
        "exact_match": qa_em_total / len(probes["qa"]),
        "token_f1": qa_f1_total / len(probes["qa"]),
    }
    return records, aggregates


def _load_adapter(model, adapter_dir: Path, resolved_model: str):
    manifest_path = adapter_dir / "delta_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "v12-full-finetune-delta-v1":
            raise ValueError(f"Unsupported V12 delta format in {manifest_path}")
        adapter_base = manifest.get("base_model")
        if adapter_base and require_compliant(str(adapter_base)) != resolved_model:
            raise ValueError(
                f"Adapter base {adapter_base!r} does not match {resolved_model!r}"
            )
        named_parameters = dict(model.named_parameters())
        with torch.no_grad():
            for entry in manifest["parameters"]:
                name = entry["name"]
                if name not in named_parameters:
                    raise RuntimeError(f"Adapter parameter {name!r} is absent from the model")
                delta = torch.load(
                    adapter_dir / entry["file"],
                    map_location="cpu",
                    weights_only=True,
                )
                parameter = named_parameters[name]
                if tuple(delta.shape) != tuple(parameter.shape):
                    raise RuntimeError(f"Adapter parameter {name!r} has the wrong shape")
                parameter.add_(delta.to(device=parameter.device, dtype=parameter.dtype))
        return model

    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No PEFT or V12 delta adapter found in {adapter_dir}")
    adapter_config = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    adapter_base = adapter_config.get("base_model_name_or_path")
    if adapter_base and require_compliant(str(adapter_base)) != resolved_model:
        raise ValueError(
            f"Adapter base {adapter_base!r} does not match {resolved_model!r}"
        )
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("peft is required to evaluate a LoRA adapter") from exc
    adapted = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    if hasattr(adapted, "merge_and_unload"):
        adapted = adapted.merge_and_unload()
    return adapted


def _apply_quantization(model, bits: int) -> None:
    parameters = language_weight_parameters(model)
    if not parameters:
        raise ValueError("No language weight matrices found for quantization")
    with torch.no_grad():
        for _, parameter in parameters:
            parameter.copy_(fake_quantize_per_output_channel(parameter, bits))


def _checkpoint_name(
    adapter: Path | None,
    prune_density: float | None,
    quant_bits: int | None,
) -> str:
    parts = []
    if adapter is None:
        if prune_density is None and quant_bits is None:
            parts.append("dense")
    elif adapter.name == "adapter":
        parts.append(adapter.parent.name)
    else:
        parts.append(adapter.name)
    if prune_density is not None:
        parts.append(f"prune-d{prune_density:g}")
    if quant_bits is not None:
        parts.append(f"quant-b{quant_bits}")
    return "_".join(parts)


def benchmark_checkpoint_name(checkpoint: str, accuracy_benchmark: str) -> str:
    """Keep easy-benchmark artifacts disjoint from the default artifacts."""
    if accuracy_benchmark not in ACCURACY_BENCHMARKS:
        raise ValueError(f"unknown accuracy benchmark: {accuracy_benchmark!r}")
    return checkpoint if accuracy_benchmark == "default" else f"{checkpoint}__easy"


def checkpoint_output_file(
    output_base: Path,
    model_tag: str,
    checkpoint: str,
    accuracy_benchmark: str,
) -> Path:
    return (
        output_base
        / model_tag
        / benchmark_checkpoint_name(checkpoint, accuracy_benchmark)
        / "accuracy.json"
    )


def evaluate_checkpoint(
    *,
    model_request: str,
    device: str,
    probes: Mapping[str, Sequence[Mapping]],
    out_file: Path,
    batch_size: int,
    adapter: Path | None = None,
    prune_density: float | None = None,
    quant_bits: int | None = None,
    checkpoint_name: str | None = None,
    n_shots: Mapping[str, int] | None = None,
    accuracy_benchmark: str = "default",
) -> dict:
    """Load, transform, loss-measure, generate, score, and persist one model."""
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if prune_density is not None and not 0.0 < prune_density <= 1.0:
        raise ValueError("prune density must be in (0, 1]")
    if quant_bits is not None and quant_bits < 2:
        raise ValueError("quantization bits must be at least 2")
    if accuracy_benchmark not in ACCURACY_BENCHMARKS:
        raise ValueError(f"unknown accuracy benchmark: {accuracy_benchmark!r}")
    shot_counts = dict(n_shots or resolve_shot_counts(4))
    if set(shot_counts) != set(CAPABILITIES):
        raise ValueError(f"n_shots must specify exactly {tuple(CAPABILITIES)}")
    for capability in CAPABILITIES:
        fewshot_exemplars(capability, shot_counts[capability])
    validate_fewshot_disjoint(probes, shot_counts)

    seed_everything(SEED)
    resolved_model = require_compliant(model_request)
    # Apply the guard to the actual identifier too, including raw adapter
    # metadata/sweep inputs rather than relying only on an alias.
    resolved_model = require_compliant(resolved_model)
    model, tokenizer = load_text_causal_lm(resolved_model, torch.bfloat16)
    if adapter is not None:
        model = _load_adapter(model, adapter, resolved_model)
    model.to(device).eval()

    pruning_threshold = None
    if prune_density is not None:
        pruning_threshold = apply_global_magnitude_pruning(
            model, prune_density, seed=SEED
        )
    if quant_bits is not None:
        _apply_quantization(model, quant_bits)

    losses, measurement_tokens = measure_capability_losses(
        model, tokenizer, probes, device
    )
    print(f"[loss] {checkpoint_name or model_request}: {losses}", flush=True)
    generations = {}
    for capability in CAPABILITIES:
        generations[capability] = generate_domain(
            model,
            tokenizer,
            probes[capability],
            capability,
            device,
            batch_size,
            shot_counts[capability],
        )
        print(
            f"[generate] {capability}: {len(generations[capability])} probes",
            flush=True,
        )
    records, aggregates = score_generations(
        probes, generations, accuracy_benchmark=accuracy_benchmark
    )
    for capability in CAPABILITIES:
        aggregates[capability]["loss"] = losses[capability]

    label = checkpoint_name or _checkpoint_name(adapter, prune_density, quant_bits)
    payload = {
        "version": 15,
        "checkpoint": label,
        "model": model_request,
        "resolved_model": resolved_model,
        "adapter": str(adapter.resolve()) if adapter is not None else None,
        "prune_density": prune_density,
        "prune_threshold": pruning_threshold,
        "quant_bits": quant_bits,
        "dtype": "bfloat16",
        "device": device,
        "seed": SEED,
        "accuracy_benchmark_mode": accuracy_benchmark,
        "accuracy_benchmark": ACCURACY_BENCHMARKS[accuracy_benchmark],
        "accuracy_benchmark_dataset": ACCURACY_BENCHMARK_DATASETS[
            accuracy_benchmark
        ],
        "probe_source": (
            "analysis.v6_capability_geometry.build_probes"
            if accuracy_benchmark == "default"
            else "analysis.v15_accuracy_link.build_easy_probes"
        ),
        "n_probe_requested": N_PROBE,
        "probe_half": "measurement (odd indices, v[1::2])",
        "measurement_samples": {
            capability: len(probes[capability]) for capability in CAPABILITIES
        },
        "measurement_tokens": measurement_tokens,
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_shots": shot_counts,
        "fewshot": fewshot_metadata(shot_counts),
        "decoding": (
            "greedy, batched, left padding, EOS and domain-delimiter stopping, "
            "post-hoc first-block truncation, no chat template"
        ),
        "losses": losses,
        "aggregates": aggregates,
        "records": records,
    }
    write_json_atomic(out_file, payload)
    print(f"[write] {out_file}", flush=True)
    del model, tokenizer
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def _v12_runs(student_dir: Path) -> list[tuple[Path, dict]]:
    runs = []
    for run_dir in sorted(student_dir.iterdir()):
        eval_path = run_dir / "eval.json"
        adapter_dir = run_dir / "adapter"
        if not run_dir.is_dir() or not eval_path.is_file() or not adapter_dir.is_dir():
            continue
        try:
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Could not read V12 metadata {eval_path}: {exc}") from exc
        runs.append((run_dir, payload))
    return runs


def run_v12_sweep(
    v12_root: Path,
    *,
    device: str,
    batch_size: int,
    output_base: Path = OUT_BASE,
    n_shots: Mapping[str, int] | None = None,
    accuracy_benchmark: str = "default",
    dry_run: bool = False,
) -> dict[str, int]:
    """Evaluate every V12 student densely once and every completed adapter."""
    if not v12_root.is_dir():
        raise FileNotFoundError(f"V12 sweep directory does not exist: {v12_root}")
    completed = 0
    skipped = 0
    probes: dict[str, list[dict]] | None = None
    for student_dir in sorted(path for path in v12_root.iterdir() if path.is_dir()):
        runs = _v12_runs(student_dir)
        if not runs:
            continue
        model_requests = {
            str(metadata.get("student") or metadata.get("resolved_student"))
            for _, metadata in runs
        }
        if len(model_requests) != 1:
            raise ValueError(
                f"V12 student directory {student_dir} names multiple models: "
                f"{sorted(model_requests)}"
            )
        model_request = model_requests.pop()
        require_compliant(require_compliant(model_request))

        dense_checkpoint = benchmark_checkpoint_name("dense", accuracy_benchmark)
        dense_file = output_base / student_dir.name / dense_checkpoint / "accuracy.json"
        if dense_file.is_file():
            skipped += 1
            print(f"[skip] {student_dir.name}/dense (accuracy.json exists)")
        elif dry_run:
            completed += 1
            print(f"[would measure] {student_dir.name}/{dense_checkpoint}")
        else:
            probes = probes or measurement_probes(accuracy_benchmark)
            evaluate_checkpoint(
                model_request=model_request,
                device=device,
                probes=probes,
                out_file=dense_file,
                batch_size=batch_size,
                checkpoint_name=f"{student_dir.name}/{dense_checkpoint}",
                n_shots=n_shots,
                accuracy_benchmark=accuracy_benchmark,
            )
            completed += 1

        for run_dir, metadata in runs:
            del metadata
            run_checkpoint = benchmark_checkpoint_name(
                run_dir.name, accuracy_benchmark
            )
            out_file = (
                output_base / student_dir.name / run_checkpoint / "accuracy.json"
            )
            if out_file.is_file():
                skipped += 1
                print(
                    f"[skip] {student_dir.name}/{run_dir.name} "
                    "(accuracy.json exists)"
                )
                continue
            if dry_run:
                completed += 1
                print(f"[would measure] {student_dir.name}/{run_checkpoint}")
                continue
            probes = probes or measurement_probes(accuracy_benchmark)
            evaluate_checkpoint(
                model_request=model_request,
                device=device,
                probes=probes,
                out_file=out_file,
                batch_size=batch_size,
                adapter=run_dir / "adapter",
                checkpoint_name=f"{student_dir.name}/{run_checkpoint}",
                n_shots=n_shots,
                accuracy_benchmark=accuracy_benchmark,
            )
            completed += 1
    return {"completed": completed, "skipped": skipped}


def write_summary(output_base: Path = OUT_BASE) -> Path:
    """Generate the V15 paired-loss/accuracy Markdown table."""
    rows = []
    if output_base.is_dir():
        for path in sorted(output_base.glob("**/accuracy.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                aggregates = payload["aggregates"]
                checkpoint = str(path.parent.relative_to(output_base))
                rows.append(
                    (
                        checkpoint,
                        float(aggregates["math"]["loss"]),
                        float(aggregates["math"]["accuracy"]),
                        float(aggregates["code"]["loss"]),
                        float(aggregates["code"]["pass_at_1"]),
                        float(aggregates["qa"]["loss"]),
                        float(aggregates["qa"]["exact_match"]),
                        float(aggregates["qa"]["token_f1"]),
                    )
                )
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
                print(f"[summary] skip malformed {path}: {exc}", file=sys.stderr)

    lines = [
        "# V15 task-accuracy/loss pairs",
        "",
        "All values use the odd-indexed measurement half of the same V6 probes.",
        "",
        "| checkpoint | L_math | A_math | L_code | A_code | L_qa | EM/F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint, l_math, a_math, l_code, a_code, l_qa, em, f1 in rows:
        lines.append(
            f"| {checkpoint} | {l_math:.6f} | {a_math:.4f} | "
            f"{l_code:.6f} | {a_code:.4f} | {l_qa:.6f} | {em:.4f}/{f1:.4f} |"
        )
    output_base.mkdir(parents=True, exist_ok=True)
    summary_path = output_base / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def print_dry_run(
    probes: Mapping[str, Sequence[Mapping]],
    n_shots: Mapping[str, int],
    accuracy_benchmark: str = "default",
) -> None:
    validate_fewshot_disjoint(probes, n_shots)
    print(
        f"probe source: {accuracy_benchmark}, n={N_PROBE}, seed={SEED}, "
        "measurement half v[1::2]"
    )
    for capability in CAPABILITIES:
        benchmark = ACCURACY_BENCHMARKS[accuracy_benchmark][capability]
        dataset = ACCURACY_BENCHMARK_DATASETS[accuracy_benchmark][capability]
        print(
            f"{capability}: benchmark={benchmark}, dataset={dataset}, "
            f"{len(probes[capability])} probes, "
            f"n_shots={n_shots[capability]}"
        )
        print(build_accuracy_prompt(
            capability, probes[capability][0], n_shots[capability]
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gemma3-270m",
        help="model-registry tag or raw Hugging Face model id",
    )
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--prune-density", type=float, default=None)
    parser.add_argument("--quant-bits", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--accuracy-benchmark",
        choices=tuple(ACCURACY_BENCHMARKS),
        default="default",
        help="paired loss/accuracy benchmark set (default: current V15 probes)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=4,
        metavar="INT",
        help=(
            "global shot cap (default: 4, resolving to math/code/qa=4/3/4; "
            "0 restores zero-shot prompts)"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--v12-sweep", type=Path, default=None, metavar="DIR")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.summarize:
        path = write_summary()
        print(path)
        return
    try:
        n_shots = resolve_shot_counts(args.shots)
    except ValueError as exc:
        parser.error(str(exc))
    if args.v12_sweep is not None and any(
        value is not None
        for value in (args.adapter, args.prune_density, args.quant_bits, args.output_dir)
    ):
        parser.error(
            "--v12-sweep cannot be combined with --adapter, compression, or --output-dir"
        )

    if args.dry_run:
        probes = measurement_probes(args.accuracy_benchmark)
        print_dry_run(probes, n_shots, args.accuracy_benchmark)
        if args.v12_sweep is None:
            require_compliant(require_compliant(args.model))
        else:
            result = run_v12_sweep(
                args.v12_sweep,
                device=args.device,
                batch_size=args.batch_size,
                n_shots=n_shots,
                accuracy_benchmark=args.accuracy_benchmark,
                dry_run=True,
            )
            print(
                f"would measure {result['completed']} run, "
                f"{result['skipped']} already complete"
            )
        return

    if args.v12_sweep is not None:
        result = run_v12_sweep(
            args.v12_sweep,
            device=args.device,
            batch_size=args.batch_size,
            n_shots=n_shots,
            accuracy_benchmark=args.accuracy_benchmark,
        )
        print(
            f"sweep complete: {result['completed']} run, {result['skipped']} skipped"
        )
        return

    probes = measurement_probes(args.accuracy_benchmark)
    resolved_model = require_compliant(require_compliant(args.model))
    tag = model_output_tag(args.model, resolved_model)
    checkpoint = _checkpoint_name(
        args.adapter, args.prune_density, args.quant_bits
    )
    if args.output_dir is None:
        output_dir = checkpoint_output_file(
            OUT_BASE,
            tag,
            checkpoint,
            args.accuracy_benchmark,
        ).parent
    else:
        output_dir = (
            args.output_dir
            if args.accuracy_benchmark == "default"
            else args.output_dir / "easy"
        )
    output_checkpoint = benchmark_checkpoint_name(
        checkpoint, args.accuracy_benchmark
    )
    evaluate_checkpoint(
        model_request=args.model,
        device=args.device,
        probes=probes,
        out_file=output_dir / "accuracy.json",
        batch_size=args.batch_size,
        adapter=args.adapter,
        prune_density=args.prune_density,
        quant_bits=args.quant_bits,
        checkpoint_name=f"{tag}/{output_checkpoint}",
        n_shots=n_shots,
        accuracy_benchmark=args.accuracy_benchmark,
    )


if __name__ == "__main__":
    main()
