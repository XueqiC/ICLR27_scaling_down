#!/usr/bin/env python3
"""V67: MuSiQue answerable-dev conditional QA loss on existing P2/P3 adapters.

Authoring checks (no data, pretrained models, downloads, or CUDA):
    python analysis/v67_musique_qa.py --dry-run
    python analysis/v67_musique_qa.py --selftest

Evaluation requires CUDA_VISIBLE_DEVICES and already-local base weights plus
musique_ans_v1.0_dev.jsonl (use --data-file, data/musique/, or the HF Hub cache
for dgslibisey/MuSiQue). All Hugging Face access is offline. No generation,
accuracy, new references, training, pruning, or quantization is performed.

Protocol: 128 examples, numpy default_rng(0), without replacement; supporting
paragraphs in source order as (title, [paragraph_text]) pairs; V6 HotpotQA's
exact renderer, including its 4000-character context cap and leading answer
space. V12/V48's serial, no-grad V6 completion_loss uses direct tokenization,
one Gemma prompt BOS, no completion special tokens, separate 512-token caps,
and summed conditional CE / summed completion tokens (native-token nats).

Primary QA is read from each selected checkpoint's eval.json, never rescored.
T is completion_tokens_seen from that file. As in V50, E = T / registered
D_U_completion (the denominator is absent from legacy eval.json files).
Processed-token T/E and their raw checkpoint fields are also retained.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

STUDENTS = ("gemma3-270m", "gemma3-1b", "gemma3-4b")
RUN_GLOB = "gpt-5.6-luna_full_*_p2v2*_lora_dseed*"
DATASET = "dgslibisey/MuSiQue"
DATA_FILE = "musique_ans_v1.0_dev.jsonl"
N_PROBE = 128
PROBE_SEED = 0
MAX_LEN = 1024
P3_STATES = (
    "kd_U75_s11_update-00000025", "kd_U75_s11_update-00000049",
    "kd_U450_s11_update-00000026", "kd_U450_s11_update-00000050",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def path_label(path: Path, root: Path = ROOT) -> str:
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def adapter_directory(checkpoint: Path) -> Path:
    """V12 saves both run-final and trajectory adapters in <checkpoint>/adapter."""
    adapter = checkpoint / "adapter"
    if not (adapter / "adapter_config.json").is_file() or not any(
        (adapter / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError(f"Incomplete existing adapter: {adapter}")
    return adapter


def checkpoint_row(student: str, run: Path, checkpoint: Path, selection: str,
                   pools: dict, root: Path) -> dict:
    adapter = adapter_directory(checkpoint)
    source = checkpoint / "eval.json"
    saved = read_json(source)
    if saved["student"] != student or saved["training_mode"] != "lora":
        raise ValueError(f"Student/training-mode mismatch: {source}")
    if (saved["measurement_benchmarks"]["qa"] != "2WikiMultihopQA"
            or saved["probe_seed"] != PROBE_SEED
            or saved["measurement_samples"]["qa"] != 64):
        raise ValueError(f"Unexpected primary QA protocol: {source}")
    qa = saved["post_training"]["qa"]
    if saved.get("capability_losses", {}).get("qa", qa) != qa:
        raise ValueError(f"Conflicting primary QA losses: {source}")
    pool_key = f"U{saved['n_per_domain']}_s{saved['data_seed']}"
    pool = pools[pool_key]
    pool_tokens = saved["unique_data_pool_tokens"]
    if pool["pool_processed_tokens"] != pool_tokens:
        raise ValueError(f"Registered pool size differs from checkpoint: {source}")
    processed, completion = saved["processed_tokens"], saved["completion_tokens_seen"]
    if min(processed, completion, pool_tokens, pool["D_U_completion"]) <= 0:
        raise ValueError(f"Expected a trained checkpoint with positive token counts: {source}")
    if not all(math.isfinite(x) for x in (qa, saved["dense"]["qa"])):
        raise ValueError(f"Non-finite primary QA loss: {source}")
    return {
        "student": student, "run": run.name,
        "checkpoint": path_label(checkpoint, root), "selection": selection,
        "adapter": path_label(adapter, root), "eval_json": path_label(source, root),
        "T": completion, "E": completion / pool["D_U_completion"],
        "T_definition": "eval.json:completion_tokens_seen",
        "E_definition": "T / registered D_U_completion (V50 convention)",
        "pool_register": "results/v47-p2-register/register.json",
        "pool_key": pool_key, "D_U_completion": pool["D_U_completion"],
        "processed_tokens": processed, "completion_tokens_seen": completion,
        "unique_data_tokens": saved["unique_data_tokens"],
        "unique_data_pool_tokens": pool_tokens,
        "E_processed": processed / pool_tokens, "updates": saved["updates"],
        "primary_qa_loss": qa, "primary_qa_tokens": saved["measurement_tokens"]["qa"],
        "primary_qa_source": "post_training.qa",
        "dense_primary_qa_loss": saved["dense"]["qa"],
    }


def plan_evaluations(root: Path = ROOT, students=STUDENTS) -> list[dict]:
    """Read local metadata only. Never silently substitute an earlier snapshot."""
    pools = read_json(root / "results/v47-p2-register/register.json")["pools"]
    p3 = {}
    if "gemma3-1b" in students:
        records = read_json(root / "results/v48-p3-measure/measurements.json")
        for state in P3_STATES:
            matches = [r for r in records if r["state"] == state]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one V48 record for {state}")
            p3[state] = matches[0]
    plan = []
    for student in students:
        directory = root / "results/v12-distill" / student
        runs = sorted(p for p in directory.glob(RUN_GLOB) if p.is_dir())
        if not runs:
            raise FileNotFoundError(f"No runs matching {directory / RUN_GLOB}")
        rows = []
        for run in runs:
            snapshots = [p for p in run.glob("trajectory/update-*")
                         if p.is_dir() and re.fullmatch(r"update-\d+", p.name)]
            if not snapshots:
                raise FileNotFoundError(f"No trajectory checkpoints: {run}")
            last = max(snapshots, key=lambda p: int(p.name.split("-")[1]))
            rows.append(checkpoint_row(student, run, last, "last_trajectory", pools, root))
        if student == "gemma3-1b":
            for state, record in p3.items():
                match = re.fullmatch(r"kd_U(\d+)_s(\d+)_(update-\d+)", state)
                pool, seed, update = match.groups()
                run = directory / f"gpt-5.6-luna_full_{pool}_p2v2_lora_dseed{seed}"
                row = checkpoint_row(student, run, run / "trajectory" / update,
                                     "p3", pools, root)
                if not math.isclose(row["primary_qa_loss"], record["primary"]["qa"],
                                    rel_tol=1e-7, abs_tol=1e-7):
                    raise ValueError(f"V48 pairing differs from checkpoint eval.json: {state}")
                row["state"] = state
                # If a P3 checkpoint is also a run's last, score it only once.
                existing = next((r for r in rows if r["checkpoint"] == row["checkpoint"]), None)
                if existing is None:
                    rows.append(row)
                else:
                    existing.update(state=state, selection="last_trajectory+p3")
        reference = rows[0]
        for row in rows:
            if (row["dense_primary_qa_loss"] != reference["dense_primary_qa_loss"]
                    or row["primary_qa_tokens"] != reference["primary_qa_tokens"]):
                raise ValueError(f"Inconsistent dense primary QA baseline for {student}")
        plan.append({
            "student": student, "run": "dense", "checkpoint": "dense",
            "selection": "dense", "adapter": None, "T": 0, "E": 0.0,
            "processed_tokens": 0, "completion_tokens_seen": 0, "E_processed": 0.0,
            "eval_json": reference["eval_json"], "primary_qa_source": "dense.qa",
            "primary_qa_loss": reference["dense_primary_qa_loss"],
            "primary_qa_tokens": reference["primary_qa_tokens"],
        })
        plan.extend(rows)
    return plan


def musique_probe(row: dict) -> dict:
    """Use V6's title/sentence-pair QA formatter verbatim, with support only."""
    from analysis.v6_capability_geometry import secondary_probe

    if (not isinstance(row["question"], str) or not row["question"].strip()
            or not isinstance(row["answer"], str) or not row["answer"].strip()
            or row.get("answerable", True) is False):
        raise ValueError("MuSiQue answerable dev requires a question and existing reference answer")
    pairs = [(p["title"], [p["paragraph_text"]]) for p in row["paragraphs"]
             if p["is_supporting"]]
    if not pairs:
        raise ValueError("MuSiQue example has no supporting paragraphs")
    return secondary_probe("qa_hotpotqa", {
        "question": row["question"], "answer": row["answer"], "context": pairs,
    })


def build_probes(rows: list[dict], n: int = N_PROBE, seed: int = PROBE_SEED):
    import numpy as np

    if len(rows) < n:
        raise ValueError(f"Need {n} MuSiQue dev examples, found {len(rows)}")
    indices = np.random.default_rng(seed).choice(len(rows), size=n, replace=False).tolist()
    return [musique_probe(rows[i]) for i in indices], indices


def local_data_file(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit
    for path in (ROOT / "data/musique" / DATA_FILE, ROOT / "data" / DATA_FILE):
        if path.is_file():
            return path
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(DATASET, DATA_FILE, repo_type="dataset")
    if isinstance(cached, str) and Path(cached).is_file():
        return Path(cached)
    raise FileNotFoundError(
        f"No local {DATA_FILE}; pass --data-file PATH to an existing copy. "
        "This script never downloads data or weights."
    )


def measure_qa(model, tokenizer, probes: list[dict], device: str) -> tuple[float, int]:
    """The QA-only loop from V12.measure_capability_losses, as called by V48."""
    import torch
    from analysis.v12_distill import completion_loss

    total_loss, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for sample in probes:
            loss, tokens = completion_loss(model, tokenizer, sample["prompt"],
                                           sample["completion"], device, max_len=MAX_LEN)
            if tokens <= 0 or not math.isfinite(float(loss)):
                raise RuntimeError("MuSiQue probe produced zero tokens or a non-finite loss")
            total_loss += float(loss)
            total_tokens += tokens
    if not total_tokens:
        raise RuntimeError("MuSiQue probe has no completion tokens")
    return total_loss / total_tokens, total_tokens


def summary_markdown(rows: list[dict], planned: int) -> str:
    dense = {r["student"]: r for r in rows if r["checkpoint"] == "dense"}
    lines = [
        "# V67 MuSiQue QA", "",
        f"Completed {len(rows)}/{planned} evaluations. Losses are native-token nats; "
        "delta = checkpoint minus the student's dense loss (negative is improvement).",
        f"MuSiQue answerable dev: {N_PROBE} references, probe seed {PROBE_SEED}, "
        "supporting context, V6 QA template and 1024-token limit (512 per span), batch size 1.",
        "2Wiki primary QA (64 measurement examples) is read from checkpoint eval.json; "
        "dense uses that file's dense.qa. No accuracy or generated references.",
        "T = eval.json completion_tokens_seen; E = T / registered D_U_completion "
        "(V50 convention). Raw processed-token exposure is retained in measurements.json.",
        "Trajectory snapshots from the same run are dependent measurements, not independent seeds.",
        "", "| Student | Run | Checkpoint | T | E | MuSiQue | Δ MuSiQue | 2Wiki | Δ 2Wiki | Tokens |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        baseline = dense[row["student"]]
        lines.append(
            f"| {row['student']} | {row['run']} | {Path(row['checkpoint']).name} "
            f"| {row['T']} | {row['E']:.4f} | {row['musique_loss']:.6f} "
            f"| {row['musique_loss'] - baseline['musique_loss']:+.6f} "
            f"| {row['primary_qa_loss']:.6f} "
            f"| {row['primary_qa_loss'] - baseline['primary_qa_loss']:+.6f} | {row['tokens']} |"
        )
    return "\n".join(lines) + "\n"


def evaluate(plan: list[dict], data_file: Path | None, out: Path) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible.split(",")[0].strip() == "-1":
        raise ValueError("Set CUDA_VISIBLE_DEVICES explicitly before evaluation")
    # Set before importing HF libraries; V12's shared loader has no offline kwarg.
    if os.environ.get("V67_ALLOW_ONLINE") != "1":  # cached Gemma snapshots lack refs for offline resolution on rai
        for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ[key] = "1"
    path = local_data_file(data_file)
    content = path.read_bytes()
    probes, indices = build_probes([json.loads(line) for line in content.splitlines() if line.strip()])

    import torch
    from peft import PeftModel
    from analysis import v12_distill as v12

    if not torch.cuda.is_available():
        raise RuntimeError(f"No CUDA device available under CUDA_VISIBLE_DEVICES={visible!r}")
    device = "cuda:0"  # Logical first device in CUDA_VISIBLE_DEVICES, never a physical ordinal.
    provenance = {
        "dataset": DATASET, "dataset_file": DATA_FILE, "local_data_file": str(path.resolve()),
        "data_sha256": hashlib.sha256(content).hexdigest(),
        "probe_seed": PROBE_SEED, "n_probe": N_PROBE, "probe_indices": indices,
        "max_len": MAX_LEN, "batch_size": 1, "cuda_visible_devices": visible,
        "loss_definition": "sum completion CE / sum completion tokens (native-token nats)",
    }
    measured, dense_tokens = [], {}
    for i, planned in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {planned['student']} {planned['checkpoint']}", flush=True)
        name, revision = v12.resolve_model_and_revision(planned["student"])
        base = tokenizer = model = None
        start = time.monotonic()
        try:
            # V48 reloads the base between adapters: PEFT mutates its input model.
            base, tokenizer = v12.load_text_causal_lm(name, torch.bfloat16, revision)
            base.to(device).eval()
            model = (base if planned["adapter"] is None else PeftModel.from_pretrained(
                base, str(ROOT / planned["adapter"]), local_files_only=True))
            loss, tokens = measure_qa(model, tokenizer, probes, device)
        finally:
            del model, base, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        student = planned["student"]
        if planned["checkpoint"] == "dense":
            dense_tokens[student] = tokens
        elif tokens != dense_tokens[student]:
            raise RuntimeError(f"MuSiQue token counts changed between dense and adapter: {student}")
        row = {**planned, **provenance, "musique_loss": loss, "tokens": tokens,
               "eval_s": time.monotonic() - start}
        measured.append(row)
        # Keep completed work if a subsequent checkpoint fails; summary marks progress.
        v12.write_json_atomic(out / "measurements.json", measured)
        summary = out / ".summary.md.tmp"
        summary.write_text(summary_markdown(measured, len(plan)), encoding="utf-8")
        summary.replace(out / "summary.md")
        print(json.dumps({"student": student, "checkpoint": planned["checkpoint"],
                          "musique_loss": loss, "tokens": tokens}), flush=True)


def selftest() -> None:
    """Exercise the actual renderer/scorer using only CPU tensors and toy logits."""
    from types import SimpleNamespace
    import torch
    from torch.nn import functional as F
    from analysis.v6_capability_geometry import completion_loss

    example = {
        "question": "Where was the author of River born?", "answer": "Lima",
        "paragraphs": [
            {"title": "River", "paragraph_text": "River was written by Ada.", "is_supporting": True},
            {"title": "Distractor", "paragraph_text": "Never include this.", "is_supporting": False},
            {"title": "Ada", "paragraph_text": "Ada was born in Lima.", "is_supporting": True},
        ],
    }
    long_context = {**example, "paragraphs": [
        {**example["paragraphs"][0], "paragraph_text": "Long sentence. " * 400},
        *example["paragraphs"][1:],
    ]}
    examples = [example, long_context, {**long_context, "answer": "Lima " * 600}]
    probes, indices = build_probes(examples, n=3)
    assert (probes, indices) == build_probes(examples, n=3)
    assert sorted(indices) == [0, 1, 2]
    short = probes[indices.index(0)]
    assert short["prompt"] == (
        "Context:\nRiver: River was written by Ada.\nAda: Ada was born in Lima."
        "\n\nQuestion: Where was the author of River born?\nAnswer:"
    )
    assert short["completion"] == " Lima"
    assert len(probes[indices.index(1)]["prompt"].split("\n\nQuestion:")[0][9:]) == 4000

    class GemmaInlineTokenizer:
        name_or_path, bos_token_id = "gemma-inline", 1

        def __init__(self, emit_bos):
            self.emit_bos = emit_bos
            self.calls = []

        def __call__(self, text, *, return_tensors, truncation, max_length, add_special_tokens):
            assert return_tensors == "pt" and truncation
            self.calls.append((add_special_tokens, max_length))
            ids = ([1] if add_special_tokens and self.emit_bos else [])
            ids += [ord(char) % 29 + 2 for char in text]
            return SimpleNamespace(input_ids=torch.tensor([ids[:max_length]], dtype=torch.long))

    class ToyLogits:
        def __init__(self, prompt_length, perturb_prompt=False):
            self.prompt_length, self.perturb_prompt = prompt_length, perturb_prompt

        def eval(self):
            return self

        def __call__(self, *, input_ids, use_cache):
            assert input_ids.device.type == "cpu" and use_cache is False
            self.ids = input_ids.clone()
            self.logits = torch.zeros(1, input_ids.shape[1], 32)
            self.logits[0, :, 0] = torch.linspace(-3, 3, input_ids.shape[1])
            if self.perturb_prompt:
                self.logits[:, :self.prompt_length - 1, 0] += 100
            return SimpleNamespace(logits=self.logits)

    with torch.no_grad():
        for probe in probes:
            assert "Distractor" not in probe["prompt"] and "Never include this." not in probe["prompt"]
            prompt_ids = ([1] + [ord(c) % 29 + 2 for c in probe["prompt"]])[:512]
            answer_ids = [ord(c) % 29 + 2 for c in probe["completion"]][:512]
            for emit_bos in (True, False):
                tok = GemmaInlineTokenizer(emit_bos)
                fake = ToyLogits(len(prompt_ids))
                loss, tokens = completion_loss(fake, tok, probe["prompt"], probe["completion"],
                                               "cpu", max_len=MAX_LEN)
                assert tok.calls == [(True, 512), (False, 512)]
                assert fake.ids.tolist() == [prompt_ids + answer_ids]
                assert fake.ids.shape[1] <= MAX_LEN and tokens == len(answer_ids)
                assert int((fake.ids == 1).sum()) == 1  # Exactly one prompt BOS.
                labels = fake.ids.clone()
                labels[:, :len(prompt_ids)] = -100
                expected = F.cross_entropy(fake.logits[:, :-1].transpose(1, 2), labels[:, 1:],
                                           ignore_index=-100, reduction="sum")
                assert torch.allclose(loss, expected, rtol=1e-6, atol=1e-5)
                changed, count = completion_loss(ToyLogits(len(prompt_ids), True), tok,
                                                probe["prompt"], probe["completion"], "cpu")
                assert count == tokens and torch.equal(loss, changed)  # Prompt loss is masked.
                if len(answer_ids) == 512:
                    assert fake.ids.shape[1] == 1024
        # Confirm V12-style aggregation weights completion tokens, not examples.
        loss, tokens = measure_qa(ToyLogits(1), GemmaInlineTokenizer(True), probes, "cpu")
        parts = [completion_loss(ToyLogits(1), GemmaInlineTokenizer(True), p["prompt"],
                                 p["completion"], "cpu") for p in probes]
        assert tokens == sum(n for _, n in parts)
        assert math.isclose(loss, sum(float(x) for x, _ in parts) / tokens)
    print("selftest: PASS (3 prompts; support-only context; reference answers; prompt masking; "
          "BOS; 512+512 truncation; token-weighted CE; CPU only, no model loaded)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="List planned evaluations; no data/model loading or writes")
    modes.add_argument("--selftest", action="store_true", help="Run inline CPU tests without a pretrained model or dataset")
    parser.add_argument("--students", nargs="+", choices=STUDENTS, default=list(STUDENTS))
    parser.add_argument("--data-file", type=Path, help=f"Existing local {DATA_FILE}; never downloaded")
    parser.add_argument("--out", type=Path, default=ROOT / "results/v67-musique-qa")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if len(set(args.students)) != len(args.students):
        parser.error("--students must not contain duplicates")
    try:
        plan = plan_evaluations(students=args.students)
        if args.dry_run:
            print(json.dumps({
                "dry_run": True, "n_evaluations": len(plan), "dataset": DATASET,
                "dataset_file": DATA_FILE, "n_probe": N_PROBE, "probe_seed": PROBE_SEED,
                "max_len": MAX_LEN, "batch_size": 1, "offline": True,
                "device": "first logical CUDA device selected by CUDA_VISIBLE_DEVICES",
                "output": str(args.out), "evaluations": plan,
            }, indent=2))
            return
        evaluate(plan, args.data_file, args.out)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"V67: {exc}\n")


if __name__ == "__main__":
    main()
