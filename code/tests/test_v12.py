from types import SimpleNamespace

import json
import random

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


def test_training_seeds_get_distinct_paths_and_are_written_to_eval(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def save_pretrained(self, path):
            (path / "adapter_config.json").write_text("{}")

    seeded, data_seeds, training_seeds, probe_seeds = [], [], [], []
    monkeypatch.setattr(distill, "OUT_BASE", tmp_path)
    monkeypatch.setattr(distill, "seed_everything", seeded.append)

    def load_records(**kwargs):
        data_seeds.append(kwargs["seed"])
        return [{"prompt": "ab", "completion": "XY", "domain": "math"}], {}

    def probes(n, seed):
        probe_seeds.append(seed)
        return {c: [{"prompt": "ab", "completion": "XY"}] * 4
                for c in distill.CAPABILITIES}

    def train(**kwargs):
        training_seeds.append((kwargs["seed"], kwargs["data_sampling_seed"]))
        accounting = distill.TokenAccounting(kwargs["examples"])
        for index in range(len(kwargs["examples"])):
            accounting.observe(index)
        return {**accounting.snapshot(), "updates": 1, "completion_tokens_seen": 2,
                "trajectory_tokens_unreached": []}

    monkeypatch.setattr(distill, "load_sft_records", load_records)
    monkeypatch.setattr(distill, "build_probes", probes)
    monkeypatch.setattr(distill, "load_text_causal_lm", lambda *a: (Model(), TinyTokenizer()))
    monkeypatch.setattr(distill, "configure_training", lambda m: (m, "lora", ["weight"]))
    monkeypatch.setattr(distill, "_train", train)
    monkeypatch.setattr(distill, "measure_capability_losses",
                        lambda *a: ({c: 2.0 for c in distill.CAPABILITIES},
                                    {c: 4 for c in distill.CAPABILITIES}))
    paths = []
    for seed in (0, 7):
        payload = distill.run_distillation(
            "gemma3-270m", "gpt-5.6-luna", "full", ["math"], 1, 1,
            "cpu", None, seed=seed,
        )
        path = tmp_path / "gemma3-270m" / payload["run_name"] / "eval.json"
        saved = json.loads(path.read_text())
        assert saved["seed"] == saved["training_seed"] == saved["data_sampling_seed"] == seed
        assert saved["probe_seed"] == 0
        paths.append(path)
    assert paths[0] != paths[1]
    assert paths[0].parent.name == "gpt-5.6-luna_full_1"
    assert paths[1].parent.name == "gpt-5.6-luna_full_1_seed7"
    assert distill.make_run_name("teacher", "full", 1, "control", seed=7) == (
        "teacher_full_1_control_seed7"
    )
    assert seeded == data_seeds == [0, 7]
    assert training_seeds == [(0, 0), (7, 7)]
    assert probe_seeds == [0, 0]


def test_trace_shuffle_uses_requested_seed_without_changing_selected_rows(tmp_path):
    rows = [{"prompt": str(i), "response": "answer"} for i in range(12)]
    (tmp_path / "gpt-5.6-luna_math.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows)
    )
    orders = []
    for seed in (0, 7):
        records, _ = distill.load_sft_records(
            "gpt-5.6-luna", ["math"], 10, "full", tmp_path, seed=seed,
        )
        orders.append([r["prompt"] for r in records])
        expected = [str(i) for i in range(10)]
        random.Random(seed).shuffle(expected)
        assert orders[-1] == expected
    assert orders[0] != orders[1]


def test_epoch_shuffle_uses_data_sampling_seed(monkeypatch):
    import transformers

    class Scheduler:
        def step(self):
            pass

        def get_last_lr(self):
            return [1e-4]

    monkeypatch.setattr(transformers, "get_cosine_schedule_with_warmup",
                        lambda *a, **kw: Scheduler())
    examples = [{"index": i, "input_ids": torch.tensor([0, 1])} for i in range(8)]
    model = torch.nn.Linear(1, 1)
    seen = []

    def loss(model, example, device):
        seen.append(example["index"])
        return model.weight.square().sum(), 1

    monkeypatch.setattr(distill, "masked_causal_loss", loss)
    log = distill._train(model, examples, "cpu", 2, 1e-4,
                         seed=7, data_sampling_seed=13)
    expected = []
    for epoch in range(2):
        order = list(range(8))
        random.Random(13 + epoch).shuffle(order)
        expected.extend(order)
    assert seen == expected
    assert log["seed"] == log["training_seed"] == 7
    assert log["data_sampling_seed"] == 13


def test_seed_cli_default_and_override(monkeypatch):
    import sys

    calls = []
    monkeypatch.setattr(distill, "run_distillation", lambda **kw: calls.append(kw))
    argv = ["v12_distill.py", "--teacher", "gpt-5.6-luna", "--recipe", "full"]
    monkeypatch.setattr(sys, "argv", argv)
    distill.main()
    monkeypatch.setattr(sys, "argv", [*argv, "--seed", "7"])
    distill.main()
    assert [call["seed"] for call in calls] == [0, 7]
