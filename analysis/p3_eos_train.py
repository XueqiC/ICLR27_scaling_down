#!/usr/bin/env python3
"""P3 end-marker control: train a student with the frozen trainer and one change, the end-of-sequence token
appended to every training target and supervised.

The distillation trainer `analysis/v12_distill.py` is frozen: registered experiments (A3, A12, S3, V22) pin
its SHA-256, so it must not be edited. This wrapper leaves the file untouched and changes only what the
control needs, by patching two module attributes for the duration of one run:

  * `tokenize_sft_example` -> `tokenize_sft_example_eos`, the frozen tokenisation with the completion capped
    one token shorter and the tokenizer's end-of-sequence id appended as a supervised label;
  * `OUT_BASE` -> `results/p3-eos-control/runs`, so control students never enter the registered
    `results/v12-distill` namespace.

Every other argument is passed to the frozen trainer unchanged. After the run the wrapper records the
protocol in the run's `eval.json` (`"append_eos": true`, the trainer's SHA-256 and this file's name), which
`analysis/p3_eos_control.py` checks before it evaluates a control student.

    python analysis/p3_eos_train.py --student EleutherAI/pythia-410m@step120000 --teacher gpt-5.6-luna \
        --recipe full --training-mode lora --domains math,qa,code --n-per-domain 600 --epochs 2 --seed 0 \
        --lr 0.0001 --device cuda:0 --output-suffix p3-eos-s3-pythia-1.4b--step120000
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from analysis import v12_distill as v12  # noqa: E402

OUT_BASE = ROOT / "results/p3-eos-control/runs"
FROZEN_TRAINER_SHA256 = "2d1d78b3562583a38fd15cda5f61ed9fbec7563ace7d44d9ddad7bd22225bc3a"


def trainer_sha256() -> str:
    return hashlib.sha256((ROOT / "analysis/v12_distill.py").read_bytes()).hexdigest()


def tokenize_sft_example_eos(tokenizer, prompt: str, completion: str, max_len: int = v12.MAX_LEN):
    """The frozen `v12_distill.tokenize_sft_example` with one change: the completion is capped at
    half_len - 1 tokens and the tokenizer's end-of-sequence id follows it as a supervised label. Prompt
    handling, the Gemma single-BOS guard, left truncation and the label mask are the frozen ones."""
    if max_len < 2:
        raise ValueError("max_len must be at least 2")
    half_len = max_len // 2
    prompt_ids = v12._encoding_input_ids(
        tokenizer(prompt, return_tensors="pt", truncation=True, max_length=half_len, add_special_tokens=True))
    tokenizer_name = f"{getattr(tokenizer, 'name_or_path', '')} {type(tokenizer).__name__}".lower()
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if "gemma" in tokenizer_name and bos_id is not None and (prompt_ids.shape[1] == 0 or int(prompt_ids[0, 0]) != bos_id):
        bos = torch.tensor([[bos_id]], dtype=prompt_ids.dtype)
        prompt_ids = torch.cat((bos, prompt_ids), dim=1)[:, :half_len]
    completion_ids = v12._encoding_input_ids(
        tokenizer(completion, return_tensors="pt", truncation=True, max_length=half_len - 1, add_special_tokens=False))
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        raise ValueError("the end-marker control needs a tokenizer with an end-of-sequence id")
    completion_ids = torch.cat((completion_ids, torch.tensor([[eos_id]], dtype=completion_ids.dtype)), dim=1)
    input_ids = torch.cat((prompt_ids, completion_ids), dim=1)[:, -max_len:]
    n_prompt = (min(prompt_ids.shape[1], input_ids.shape[1] - completion_ids.shape[1])
                if input_ids.shape[1] > completion_ids.shape[1] else 0)
    labels = input_ids.clone()
    labels[:, :n_prompt] = -100
    return {"input_ids": input_ids[0], "attention_mask": torch.ones_like(input_ids)[0], "labels": labels[0],
            "n_completion_tokens": int((labels[:, 1:] != -100).sum())}


def record_protocol(run_dir: Path) -> None:
    """Mark a finished control run so the evaluation refuses anything trained without the end marker."""
    path = run_dir / "eval.json"
    payload = json.loads(path.read_text())
    payload.update({"append_eos": True, "trainer_sha256": trainer_sha256(),
                    "tokenization": "analysis/p3_eos_train.py:tokenize_sft_example_eos"})
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    if trainer_sha256() != FROZEN_TRAINER_SHA256:
        raise SystemExit("analysis/v12_distill.py differs from the frozen trainer; the control must not run on an edited copy")
    before = {p for p in OUT_BASE.glob("*/*") if p.is_dir()} if OUT_BASE.exists() else set()
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    with patch.object(v12, "tokenize_sft_example", tokenize_sft_example_eos), patch.object(v12, "OUT_BASE", OUT_BASE):
        v12.main()
    for run in sorted({p for p in OUT_BASE.glob("*/*") if p.is_dir()} - before):
        if (run / "eval.json").is_file():
            record_protocol(run)
            print("recorded the end-marker protocol in", run.relative_to(ROOT))


if __name__ == "__main__":
    main()
