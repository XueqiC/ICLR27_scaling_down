#!/usr/bin/env python3
"""P3a: token-level audit of end-of-answer supervision in the distilled students of the selection round.

For the student of each reference in the selection validation (S3), rebuild the training examples exactly as
analysis/v12_distill.py builds them (same teacher, recipe, domains, rows per domain and pool selection),
tokenise them with the student's tokenizer through the same function, and record for a sample of examples:
the last supervised tokens of the target (ids and strings), whether the tokenizer's end-of-sequence id
occurs anywhere among the supervised labels, the label mask (prompt tokens masked with -100, completion
tokens supervised), the tokenizer's end-of-sequence id, and the stop conditions of the generation readout
(analysis/v15_accuracy_link: the tokenizer's EOS ids and the domain stop sequences). Also counts, over the
whole pool, how many targets end in a newline or in a sentence-ending character, and the extra supervised
tokens that appending the end marker would add (one per example per epoch). CPU only; no model is loaded.

Writes results/p3-eos-control/token_audit.json and token_audit.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import v12_distill as v12  # noqa: E402
from analysis import v15_accuracy_link as v15  # noqa: E402

OUT = ROOT / "results/p3-eos-control"
STUDENTS = {  # reference -> (student model, revision, S3 run configuration)
    "pythia-410m--step120000": ("EleutherAI/pythia-160m", "step120000"),
    "pythia-1.4b--step120000": ("EleutherAI/pythia-410m", "step120000"),
    "gemma3-1b": ("google/gemma-3-270m", "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"),
    "gemma3-4b": ("google/gemma-3-1b-pt", "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"),
}
S3_CONFIG = {"teacher": "gpt-5.6-luna", "recipe": "full", "domains": ("math", "qa", "code"), "n_per_domain": 600,
             "epochs": 2, "seed": 0, "data_seed": None, "learning_rate": 1e-4, "training_mode": "lora"}
N_SHOW = 3


def read_config(ref):
    d = ROOT / "results/s3-selection-validation/students" / ref / "v12"
    evals = list(d.glob("*/*/eval.json"))
    if len(evals) != 1:
        raise SystemExit(f"{ref}: expected one eval.json, found {len(evals)}")
    e = json.loads(evals[0].read_text())
    keys = ("student", "teacher", "recipe", "domains", "n_per_domain", "epochs", "seed", "data_seed", "learning_rate", "training_mode")
    return {k: e.get(k) for k in keys}, evals[0]


def audit_reference(ref, model, revision):
    from transformers import AutoTokenizer
    cfg, path = read_config(ref)
    assert cfg["student"] == f"{model}@{revision}", (ref, cfg["student"])
    for k in ("teacher", "recipe", "n_per_domain", "epochs", "seed", "data_seed", "learning_rate", "training_mode"):
        assert cfg[k] == S3_CONFIG[k], (ref, k, cfg[k])
    assert tuple(cfg["domains"]) == S3_CONFIG["domains"], (ref, cfg["domains"])
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    records, counts = v12.load_sft_records(cfg["teacher"], cfg["domains"], cfg["n_per_domain"], cfg["recipe"],
                                           seed=cfg["seed"], data_seed=cfg["data_seed"])
    eos_id = getattr(tok, "eos_token_id", None)
    eos_ids = set(eos_id if isinstance(eos_id, (list, tuple)) else ([] if eos_id is None else [eos_id]))
    summary = {"reference": ref, "student": cfg["student"], "eval_json": str(path.relative_to(ROOT)), "config": cfg,
               "rows_per_domain": counts, "n_examples": len(records), "eos_token_id": eos_id, "eos_token": tok.convert_ids_to_tokens(eos_id) if eos_id is not None else None,
               "inference_stop": {"eos_ids": sorted(eos_ids), "stop_sequences": v15.DOMAIN_STOP_SEQUENCES["qa"]},
               "targets_with_eos_supervised": 0, "targets_ending_in_newline": 0, "targets_ending_in_sentence_end": 0,
               "supervised_tokens_per_epoch": 0, "extra_supervised_tokens_per_epoch_with_eos": 0, "examples": []}
    for i, rec in enumerate(records):
        ex = v12.tokenize_sft_example(tok, rec["prompt"], rec["completion"], max_len=v12.MAX_LEN)
        labels = ex["labels"].tolist(); ids = ex["input_ids"].tolist()
        supervised = [t for t in labels if t != -100]
        summary["supervised_tokens_per_epoch"] += ex["n_completion_tokens"]
        summary["extra_supervised_tokens_per_epoch_with_eos"] += 1
        if any(t in eos_ids for t in supervised):
            summary["targets_with_eos_supervised"] += 1
        tail = rec["completion"].rstrip(" ")
        if tail.endswith("\n"):
            summary["targets_ending_in_newline"] += 1
        if tail.rstrip().endswith((".", "!", "?", "```", "$")):
            summary["targets_ending_in_sentence_end"] += 1
        if i % (len(records) // N_SHOW) == 0 and len(summary["examples"]) < N_SHOW:
            n_prompt = sum(1 for t in labels if t == -100)
            last = supervised[-6:]
            summary["examples"].append({"domain": rec["domain"], "n_tokens": len(ids), "n_prompt_masked": n_prompt,
                                        "n_supervised": len(supervised), "last_supervised_ids": last,
                                        "last_supervised_tokens": tok.convert_ids_to_tokens(last),
                                        "target_tail_text": rec["completion"][-80:], "first_label_is_first_completion_token": labels[n_prompt] == ids[n_prompt]})
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = [audit_reference(ref, *STUDENTS[ref]) for ref in STUDENTS]
    (OUT / "token_audit.json").write_text(json.dumps(results, indent=1))
    lines = ["# P3a token-level audit of end-of-answer supervision (selection-round students)", "",
             "Training examples rebuilt with `v12_distill.load_sft_records` and tokenised with `tokenize_sft_example` "
             "(the functions the students were trained with); stop conditions from `v15_accuracy_link`.", "",
             "| Reference | Student | Examples | EOS id (token) | Targets with EOS supervised | Ending in newline | Ending in sentence end | Supervised tokens / epoch | Extra with EOS / epoch | Inference stop |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for r in results:
        lines.append(f"| {r['reference']} | {r['student']} | {r['n_examples']} | {r['eos_token_id']} ({r['eos_token']}) | {r['targets_with_eos_supervised']} | "
                     f"{r['targets_ending_in_newline']} | {r['targets_ending_in_sentence_end']} | {r['supervised_tokens_per_epoch']} | "
                     f"{r['extra_supervised_tokens_per_epoch_with_eos']} | eos {r['inference_stop']['eos_ids']} + {r['inference_stop']['stop_sequences']} |")
    lines += ["", "## Sampled examples (last supervised tokens)", ""]
    for r in results:
        for ex in r["examples"]:
            lines.append(f"- {r['reference']} / {ex['domain']}: {ex['n_tokens']} tokens, {ex['n_prompt_masked']} prompt tokens masked, {ex['n_supervised']} supervised; "
                         f"first label is the first completion token: {ex['first_label_is_first_completion_token']}; last supervised tokens {ex['last_supervised_tokens']} "
                         f"(ids {ex['last_supervised_ids']}); target tail {ex['target_tail_text']!r}")
    (OUT / "token_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
