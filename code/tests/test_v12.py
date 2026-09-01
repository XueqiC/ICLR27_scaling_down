from types import SimpleNamespace

import torch

from analysis import v12_distill as distill
from analysis import v12_sweep as sweep


class TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    bos_token_id = 1

    def __call__(
        self,
        text,
        return_tensors,
        truncation,
        max_length,
        add_special_tokens,
    ):
        assert return_tensors == "pt"
        assert truncation is True
        ids = [10 + ord(character) % 50 for character in text]
        if add_special_tokens:
            ids = [self.bos_token_id, *ids]
        return SimpleNamespace(
            input_ids=torch.tensor([ids[:max_length]], dtype=torch.long)
        )


def test_recipe_extractors_full_and_answer_only_domains():
    response = "Reasoning\n#### 17"
    assert distill.extract_full(response, "math") == response
    assert distill.extract_answer_only(response, "math") == "17"
    assert distill.extract_answer_only("work\nfinal line", "math") == "final line"
    assert (
        distill.extract_answer_only("first\n\n  final answer  \n", "qa")
        == "final answer"
    )


def test_answer_only_code_uses_last_fence_or_last_line():
    response = (
        "First attempt:\n```python\nreturn 1\n```\n"
        "Correction:\n```python\ndef answer():\n    return 2\n```\nDone."
    )
    assert distill.extract_answer_only(response, "code") == (
        "def answer():\n    return 2"
    )
    assert distill.extract_answer_only("reasoning\nreturn 3", "code") == "return 3"


def test_no_code_fence_removes_blocks_but_retains_surrounding_response():
    response = "Before\n```python\nprint('deleted')\n```\nAfter"
    assert distill.extract_no_code_fence(response, "code") == "Before\n\nAfter"
    assert distill.extract_no_code_fence("plain answer", "qa") == "plain answer"


def test_tokenize_sft_example_masks_prompt_and_keeps_completion_labels():
    tokenizer = TinyTokenizer()
    prompt = "ab"
    completion = "XY"
    example = distill.tokenize_sft_example(
        tokenizer, prompt, completion, max_len=16
    )

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8,
        add_special_tokens=True,
    ).input_ids[0]
    completion_ids = tokenizer(
        completion,
        return_tensors="pt",
        truncation=True,
        max_length=8,
        add_special_tokens=False,
    ).input_ids[0]

    torch.testing.assert_close(
        example["input_ids"], torch.cat((prompt_ids, completion_ids))
    )
    torch.testing.assert_close(
        example["labels"][: len(prompt_ids)],
        torch.full_like(prompt_ids, -100),
    )
    torch.testing.assert_close(example["labels"][len(prompt_ids) :], completion_ids)
    torch.testing.assert_close(
        example["attention_mask"], torch.ones_like(example["input_ids"])
    )
    assert example["n_completion_tokens"] == len(completion_ids)


def test_sweep_skips_existing_eval_and_runs_only_incomplete_config(tmp_path):
    completed = {
        "student": "gemma3-270m",
        "teacher": "gpt-5.6-luna",
        "recipe": "full",
        "n_per_domain": 10,
    }
    incomplete = ["gemma3-270m", "claude-sonnet-4-6", "answer_only", 20]
    marker = sweep.eval_path_for_config(completed, output_base=tmp_path)
    marker.parent.mkdir(parents=True)
    marker.write_text("{}\n")
    calls = []

    def fake_runner(command, check, cwd):
        calls.append((command, check, cwd))

    result = sweep.run_sweep(
        [completed, incomplete],
        output_base=tmp_path,
        runner=fake_runner,
        python_executable="python-test",
    )

    assert result == {"completed": 1, "skipped": 1}
    assert len(calls) == 1
    command, check, cwd = calls[0]
    assert command[0] == "python-test"
    assert check is True
    assert cwd == sweep.ROOT
    assert command[command.index("--teacher") + 1] == "claude-sonnet-4-6"
    assert command[command.index("--n-per-domain") + 1] == "20"
