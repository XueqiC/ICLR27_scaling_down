#!/usr/bin/env python3
"""V71: pre-specified Gemma-3-1B QA capability-measurement scope check.

CPU authoring (local dataset/metadata reads, no pretrained model or CUDA):
    python analysis/v71_qa_scope.py --dry-run
    python analysis/v71_qa_scope.py --selftest
    python analysis/v71_qa_scope.py --dry-run --write-plan

The last command freezes register.json and writes explicitly unmeasured result
cells plus summary.md and the [H] paper table. Evaluation is a separate invocation:
    CUDA_VISIBLE_DEVICES=GPU-<caller-selected-UUID> python analysis/v71_qa_scope.py

Local Arrow caches are read directly, without cache writes/downloads. As in V67,
HF model access defaults to offline; V67_ALLOW_ONLINE=1 permits model resolution.
The caller's CUDA visibility is never changed. No training, generation, accuracy,
teacher calls, or outcome-based state/sample selection. Re-running resumes only
an identical register; the base is reloaded for every state (PEFT mutates it).
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
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

# BEGIN PRE-SPECIFIED REGISTER -- fixed before any V71 measurement.
MODEL = "gemma3-1b"
HF_ID = "google/gemma-3-1b-pt"
RUN = "results/v12-distill/gemma3-1b/gpt-5.6-luna_full_450_p2v2_lora_dseed11"
N = 256
MAX_LEN = 1024
STATES = (
    {"state": "dense", "kind": "dense", "label": "Dense"},
    {"state": "prune_d0.85", "kind": "prune", "density": 0.85, "label": "Prune d=0.85"},
    {"state": "prune_d0.7", "kind": "prune", "density": 0.7, "label": "Prune d=0.7"},
    {"state": "rtn_int4", "kind": "quant", "bits": 4, "group_size": 0, "label": "RTN int4/channel"},
    {"state": "rtn_b4_g128", "kind": "quant", "bits": 4, "group_size": 128, "label": "RTN b=4 g=128"},
    {"state": "kd_U450_s11_T35000", "kind": "adapter", "planned_T": 35000, "label": "KD T=35k"},
    {"state": "kd_U450_s11_T140000", "kind": "adapter", "planned_T": 140000, "label": "KD T=140k"},
    {"state": "kd_U450_s11_T280000", "kind": "adapter", "planned_T": 280000, "label": "KD T=280k"},
)
SETS = {
    "2wiki_new": {"dataset": "framolfese/2WikiMultihopQA", "config": None,
                  "split": "validation", "seed": 71,
                  "context": "All supplied context, exact V6 renderer, 4000-character cap"},
    "musique": {"dataset": "dgslibisey/MuSiQue", "file": "musique_ans_v1.0_dev.jsonl",
                "split": "answerable dev", "seed": 0,
                "context": "Supporting paragraphs in source order, exact V67 builder, 4000-character cap"},
    "triviaqa": {"dataset": "mandarjoshi/trivia_qa", "config": "rc.nocontext",
                 "split": "validation", "seed": 0,
                 "context": "No context supplied by this split; exact V48/V9 builder"},
}
INFORMATION = (
    "Use the supplied question and existing reference answer on every set; supply context "
    "where the set has it. 2Wiki uses all supplied paragraphs; MuSiQue uses supporting "
    "paragraphs only (V67); TriviaQA rc.nocontext supplies none (V48). No retrieved context, "
    "chat template, generated rationale, or reference in the prompt. Thus the same "
    "available-information rule is used, but these are not matched-context datasets. "
    "Context availability and support selection limit cross-set causal interpretations. "
    "V6 direct tokenization, exactly one Gemma prompt BOS, no completion special tokens, "
    "leading answer space, right truncation at 512 prompt and 512 completion tokens. "
    "The question may be truncated after a long context, exactly as in V6/V67. "
    "Batch size 1; sum conditional completion CE / sum completion tokens, native-token nats."
)
READINGS = (
    {"id": "both", "reading": "Both new primary and MuSiQue reproduce",
     "meaning": "QA loss gains extend beyond the original primary sample to a second context-supplied distribution."},
    {"id": "primary_only", "reading": "Only the new primary sample reproduces",
     "meaning": "QA loss gains replicate within 2Wiki but remain distribution-specific."},
    {"id": "neither", "reading": "Neither reproduces",
     "meaning": "The original QA gain does not replicate on either new panel; an original-sample-specific reading is warranted."},
)
READING_RULE = (
    "Apply the three readings separately at each of the three fixed KD budgets. "
    "Reproduction means delta from that set's dense loss < 0 (descriptive sign only, "
    "not statistical significance). Zero is no gain. MuSiQue-only is reported as an "
    "unanticipated fourth outcome, never forced into the three planned readings. "
    "Report all budgets and both compression arms; do not select a favorable budget. "
    "TriviaQA is the V48 control and does not determine the primary/MuSiQue reading. "
    "The three adapters are dependent checkpoints of one trajectory, not three seeds."
)
# END PRE-SPECIFIED REGISTER.


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def label(path, root=ROOT):
    path = Path(path).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as f:
        f.write(content)
    os.replace(f.name, path)


def json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def checkpoint_plan(root=ROOT):
    """Match registered processed triggers, not nearest T or best saved QA loss."""
    from analysis.v67_musique_qa import adapter_directory

    source = root / "results/v47-p2-register/register.json"
    register = read_json(source)
    protocol = register["v2"]["protocol_v2"]
    triggers = dict(zip(protocol["planned_supervised_T"], protocol["processed_triggers"], strict=True))
    pool = register["pools"]["U450_s11"]
    revision = register["student_snapshot"]
    paths = sorted((root / RUN / "trajectory").glob("update-*/eval.json"))
    saved = [(p, read_json(p)) for p in paths]
    plan = []
    for spec in STATES:
        row = dict(spec)
        if row["kind"] == "adapter":
            trigger = triggers[row["planned_T"]]
            matches = [(p, d) for p, d in saved if trigger in d.get("requested_token_milestones", [])]
            if len(matches) != 1:
                raise ValueError(f"Need exactly one trajectory checkpoint for T={row['planned_T']}, trigger={trigger}")
            path, d = matches[0]
            expected = {"student": MODEL, "teacher": "gpt-5.6-luna", "recipe": "full",
                        "n_per_domain": 450, "data_seed": 11, "training_mode": "lora",
                        "output_suffix": "_p2v2", "schedule_tokens": 1000000, "training_seed": 0,
                        "probe_seed": 0, "unique_data_pool_tokens": pool["pool_processed_tokens"]}
            if any(d.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Checkpoint protocol mismatch: {path}")
            if not (d["processed_tokens"] >= trigger and 0 < d["completion_tokens_seen"] <= d["processed_tokens"]):
                raise ValueError(f"Invalid checkpoint exposure: {path}")
            adapter = adapter_directory(path.parent)
            config = read_json(adapter / "adapter_config.json")
            if (config["base_model_name_or_path"] != HF_ID or config["peft_type"] != "LORA"
                    or config["r"] != 16 or config["lora_alpha"] != 32
                    or config["lora_dropout"] != 0.0
                    or set(config["target_modules"]) != set(protocol["lora"]["targets"])):
                raise ValueError(f"Adapter protocol mismatch: {adapter}")
            row.update(adapter=label(adapter, root), checkpoint=label(path.parent, root),
                       eval_json=label(path, root), eval_sha256=sha_file(path),
                       adapter_sha256={p.name: sha_file(p) for p in sorted(adapter.iterdir())
                                       if p.name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")},
                       trigger_processed=trigger, actual_T=d["completion_tokens_seen"],
                       processed_tokens=d["processed_tokens"], updates=d["updates"],
                       D_U_completion=pool["D_U_completion"])
        plan.append(row)
    if len({r["adapter"] for r in plan if r["kind"] == "adapter"}) != 3:
        raise ValueError("The three budgets must resolve to distinct adapters")
    return plan, revision, {"path": label(source, root), "sha256": sha_file(source)}


class RecordingRows:
    def __init__(self, rows):
        self.rows, self.indices = rows, []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        self.indices.append(int(index))
        return self.rows[int(index)]


def cached_dataset(cache, dataset, config, split):
    """Read existing Arrow files directly; never invoke HF download/cache writers."""
    from datasets import Dataset, concatenate_datasets

    cache_name = {
        "HuggingFaceH4/MATH-500": "HuggingFaceH4___math-500",
        "google-research-datasets/mbpp": "google-research-datasets___mbpp",
        "framolfese/2WikiMultihopQA": "framolfese___2_wiki_multihop_qa",
        "mandarjoshi/trivia_qa": "mandarjoshi___trivia_qa",
    }[dataset]
    candidates = []
    for info_path in sorted((cache / cache_name / (config or "default")).glob("*/*/dataset_info.json")):
        info = read_json(info_path)
        files = sorted(info_path.parent.glob(f"*-{split}.arrow"))
        if not files:
            files = sorted(info_path.parent.glob(f"*-{split}-*-of-*.arrow"))
        if files and split in info["splits"]:
            candidates.append((info_path, info, files))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Need one local Arrow cache for {dataset}/{config}/{split}; found {len(candidates)} under {cache}")
    info_path, info, files = candidates[0]
    parts = [Dataset.from_file(str(p)) for p in files]
    rows = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    if len(rows) != info["splits"][split]["num_examples"]:
        raise ValueError(f"Incomplete cached split: {info_path}")
    return rows, {"dataset": dataset, "config": config, "split": split, "n_rows": len(rows),
                  "files": {str(p): sha_file(p) for p in [info_path, *files]}}


def two_wiki_probe(row):
    """Verbatim V6 primary renderer, including its non-dict context fallback."""
    context = row.get("context")
    if isinstance(context, dict):
        parts = []
        for title, sentences in zip(context.get("title", []), context.get("sentences", context.get("content", []))):
            body = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
            parts.append(f"{title}: {body}")
        context_text = "\n".join(parts)[:4000]
    else:
        context_text = str(context)[:4000]
    answer = str(row["answer"])
    return {"prompt": f"Context:\n{context_text}\n\nQuestion: {row['question']}\nAnswer:",
            "completion": " " + answer, "answer": answer}


def recover_v6_indices(datasets_by_name, n, seed=0):
    """Run the actual V6 builder: its math/code draws advance the shared RNG."""
    from analysis import v6_capability_geometry as v6

    wrappers = {k: RecordingRows(v) for k, v in datasets_by_name.items()}
    with patch("datasets.load_dataset", side_effect=lambda name, *a, **kw: wrappers[name]):
        legacy = v6.build_probes(n, seed=seed)
    indices = wrappers[SETS["2wiki_new"]["dataset"]].indices
    if len(indices) != n or len(set(indices)) != n:
        raise ValueError("V6 builder did not return the requested unique source indices")
    return {"probe_indices": indices[0::2], "measurement_indices": indices[1::2]}, legacy["qa"]


def fresh_indices(length, excluded, n=N, seed=71):
    import numpy as np

    excluded = set(excluded)
    if any(type(i) is not int or not 0 <= i < length for i in excluded):
        raise ValueError("Invalid source dataset exclusion index")
    eligible = [i for i in range(length) if i not in excluded]
    if len(eligible) < n:
        raise ValueError(f"Need {n} disjoint examples, found {len(eligible)}")
    return np.random.default_rng(seed).choice(eligible, size=n, replace=False).tolist()


def teacher_questions(trace_base):
    """Full V12 source pools, both teachers/all domains; never only selected U450."""
    from analysis.v12_distill import TEACHERS, DOMAINS
    from analysis.v34_c17_scope_audit import normalize, question

    questions, sources = set(), []
    for teacher in TEACHERS:
        for domain in DOMAINS:
            path = trace_base / f"{teacher}_{domain}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            extracted = [question(r) for r in rows]
            if not rows or (domain == "qa" and not all(extracted)):
                raise ValueError(f"Incomplete teacher QA question coverage: {path}")
            questions.update(normalize(q) for q in extracted if q)
            sources.append({"path": label(path), "sha256": sha_file(path), "rows": len(rows),
                            "question_rows": sum(q is not None for q in extracted), "domain": domain})
    return questions, sources


def musique_panel(rows, questions, n=N, seed=0):
    from analysis.v67_musique_qa import build_probes
    from analysis.v34_c17_scope_audit import normalize

    answerable = [i for i, row in enumerate(rows) if row.get("answerable", True) is not False]
    overlapping = [i for i in answerable if normalize(rows[i]["question"]) in questions]
    eligible = [i for i in answerable if i not in set(overlapping)]
    probes, chosen = build_probes([rows[i] for i in eligible], n=n, seed=seed)
    indices = [eligible[i] for i in chosen]
    return probes, {"source_rows": len(rows), "answerable_rows": len(answerable),
                    "overlap_count": len(overlapping), "overlap_unique_questions": len({normalize(rows[i]["question"]) for i in overlapping}),
                    "excluded_indices": overlapping, "eligible_rows": len(eligible),
                    "sampled_overlap_count": len(set(indices) & set(overlapping)),
                    "indices": indices,
                    "matching": "V34 question extraction; NFKC + casefold + collapsed whitespace, punctuation preserved; exact question equality. All QA trace rows must parse. Non-QA rows without a Question marker have no QA question string. No semantic or pretraining overlap claim."}


def prepare_panels(cache, musique_file, trace_base, root=ROOT):
    from analysis import v9_capability_regions as v9

    specifications = [("HuggingFaceH4/MATH-500", None, "test"),
                      ("google-research-datasets/mbpp", "full", "test"),
                      (SETS["2wiki_new"]["dataset"], None, "validation"),
                      (SETS["triviaqa"]["dataset"], "rc.nocontext", "validation")]
    loaded, provenance = {}, []
    for name, config, split in specifications:
        loaded[name], source = cached_dataset(cache, name, config, split)
        provenance.append(source)
    qa = loaded[SETS["2wiki_new"]["dataset"]]
    count_files = sorted((root / "results/v6-capability-geometry").glob("*/probes.json"))
    if not count_files:
        raise FileNotFoundError("Missing V6 probe metadata; cannot establish exclusions")
    counts = [read_json(p) for p in count_files]
    if any(c != {"math": 64, "code": 64, "qa": 64} for c in counts):
        raise ValueError("Unexpected V6 counts; inspect the historical split before measuring")
    halves, legacy = recover_v6_indices(loaded, 128, seed=0)
    for i, probe in zip([x for pair in zip(halves["probe_indices"], halves["measurement_indices"]) for x in pair], legacy):
        if two_wiki_probe(qa[i]) != probe:
            raise ValueError("V71 primary rendering diverged from V6")
    # Legacy V6 saved counts only. Cross-check replay against saved item strings,
    # not V15's renumbered probe_index (which is not a source dataset row ID).
    saved_path = root / "results/v15-accuracy/gemma3-1b/dense/accuracy.json"
    old = read_json(saved_path)
    old_qa = [r for r in old["records"] if r["domain"] == "qa"]
    if (old["n_probe_requested"] != 128 or old["seed"] != 0 or len(old_qa) != 64
            or [(r["prompt"], r["reference_completion"]) for r in old_qa]
            != [(r["prompt"], r["completion"]) for r in legacy[1::2]]):
        raise ValueError("V6 replay disagrees with saved primary identities; refusing inferred exclusions")
    excluded = set(halves["probe_indices"] + halves["measurement_indices"])
    index_sources = []
    # Honor explicit index sets if a V6 run supplies them, in addition to replay.
    for path in sorted((root / "results/v6-capability-geometry").glob("*/qa_indices.json")):
        saved = read_json(path)
        excluded.update(saved["probe_indices"] + saved["measurement_indices"])
        index_sources.append({"path": label(path, root), "sha256": sha_file(path)})
    indices = fresh_indices(len(qa), excluded, seed=SETS["2wiki_new"]["seed"])
    # Also exclude duplicated question strings belonging to either old half.
    from analysis.v34_c17_scope_audit import normalize
    old_questions = {normalize(qa[i]["question"]) for i in excluded}
    duplicate_indices = {i for i in range(len(qa)) if normalize(qa[i]["question"]) in old_questions}
    excluded.update(duplicate_indices)
    indices = fresh_indices(len(qa), excluded, seed=SETS["2wiki_new"]["seed"])
    panels = {"2wiki_new": [two_wiki_probe(qa[i]) for i in indices]}
    meta = {"2wiki_new": {"indices": indices, "v6_seed": 0, "v6_n_probe": 128, **halves,
                           "excluded_indices": sorted(excluded), "sampled_overlap_count": 0,
                           "index_sources": index_sources,
                           "index_provenance": "V6 files contain counts, not indices. Replayed the actual V6 builder (shared math/code/QA RNG); checked all 64 measurement prompt/reference strings against V15. Exclude both halves and their duplicated question strings.",
                           "count_files": {label(p, root): sha_file(p) for p in count_files},
                           "identity_check": {"path": label(saved_path, root), "sha256": sha_file(saved_path)}}}
    questions, sources = teacher_questions(trace_base)
    rows = [json.loads(line) for line in musique_file.read_text().splitlines() if line.strip()]
    panels["musique"], meta["musique"] = musique_panel(rows, questions)
    meta["musique"].update(data_file=label(musique_file, root), data_sha256=sha_file(musique_file),
                           teacher_trace_files=sources, teacher_unique_questions=len(questions))
    wrapper = RecordingRows(loaded[SETS["triviaqa"]["dataset"]])
    with patch("datasets.load_dataset", return_value=wrapper):
        panels["triviaqa"] = v9.build_triviaqa_probes(N, seed=SETS["triviaqa"]["seed"])
    meta["triviaqa"] = {"indices": wrapper.indices, "builder": "V48 registry -> V9 build_triviaqa_probes"}
    for key, probes in panels.items():
        if len(probes) != N or len(set(meta[key]["indices"])) != N:
            raise ValueError(f"Expected {N} unique source items for {key}")
        if any(not p["completion"].strip() for p in probes):
            raise ValueError(f"Empty reference answer in {key}")
        meta[key].update(**SETS[key], n=N, prompt_reference_sha256=sha_json(probes))
    return panels, meta, provenance


def make_register(cache, musique_file, trace_base, root=ROOT):
    plan, revision, pool_source = checkpoint_plan(root)
    panels, panel_meta, data_sources = prepare_panels(cache, musique_file, trace_base, root)
    sources = ("v71_qa_scope.py", "v6_capability_geometry.py", "v9_capability_regions.py",
               "v10_quantization.py", "v12_distill.py", "v54_quant_group.py", "v67_musique_qa.py",
               "v34_c17_scope_audit.py", "v48_p3_measure.py", "model_registry.py")
    register = {"version": 71, "model": MODEL, "hf_id": HF_ID, "revision": revision,
                "dtype": "bfloat16", "states": plan, "sets": panel_meta, "n_per_set": N,
                "information_condition": INFORMATION, "readings": READINGS, "reading_rule": READING_RULE,
                "reading_source": "paper/docs/A100_EXPLORATION_PLAN.md: Round v5 / V5-M",
                "loss": "sum completion CE / sum completion tokens (native-token nats)",
                "max_len": MAX_LEN, "batch_size": 1, "prune_seed": 0,
                "pruning": "V6 sampled global magnitude threshold, 2,000,000 proportional weight samples; strict abs(w)>threshold",
                "quantization": "V54 symmetric RTN, b=4, g=0 (V10 per-output-channel) or g=128; native bf16 arithmetic and V54 padding rule",
                "weight_scope": "V6 language_weight_parameters, including embeddings and LM head; fresh dense base per state",
                "pool_register": pool_source, "dataset_sources": data_sources,
                "code_sha256": {name: sha_file(root / "analysis" / name) for name in sources}}
    return register, panels


def freeze_register(out, register):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "register.json"
    if path.exists():
        if read_json(path) != json.loads(json_text(register)):
            raise ValueError(f"Existing register differs: {path}; do not overwrite a pre-measurement register")
    else:
        # Exclusive creation makes this write-once and precedes model loading.
        with path.open("x", encoding="utf-8") as f:
            f.write(json_text(register))


def initial_results(register):
    return {"version": 71, "status": "not_measured", "register_sha256": sha_json(register),
            "register": register,
            "measurements": [{"state": s["state"], "set": key, "loss": None, "tokens": None,
                              "n": 0, "planned_n": N, "delta_from_dense": None}
                             for s in register["states"] for key in SETS]}


def validate_results(result, register):
    expected = initial_results(register)
    if (result["register_sha256"] != expected["register_sha256"]
            or sha_json(result["register"]) != expected["register_sha256"]):
        raise ValueError("Results belong to a different register")
    rows = result["measurements"]
    if [(r["state"], r["set"]) for r in rows] != [(r["state"], r["set"]) for r in expected["measurements"]]:
        raise ValueError("Results must contain exactly the ordered 8 x 3 cells")
    dense = {r["set"]: r for r in rows if r["state"] == "dense"}
    for row in rows:
        if row["loss"] is None:
            if row["tokens"] is not None or row["n"] != 0 or row["delta_from_dense"] is not None:
                raise ValueError("Unmeasured cells cannot contain fabricated counts or deltas")
        else:
            base = dense[row["set"]]
            if (not math.isfinite(row["loss"]) or row["loss"] < 0 or row["n"] != N
                    or type(row["tokens"]) is not int or row["tokens"] <= 0
                    or base["loss"] is None or row["tokens"] != base["tokens"]
                    or row["delta_from_dense"] != row["loss"] - base["loss"]):
                raise ValueError("Invalid loss, sample/token count, or dense delta")


def reading(primary, musique):
    if primary is None or musique is None:
        return "Pending"
    if primary < 0 and musique < 0:
        return READINGS[0]["reading"]
    if primary < 0:
        return READINGS[1]["reading"]
    if musique >= 0:
        return READINGS[2]["reading"]
    return "MuSiQue only (unanticipated fourth outcome)"


def reports(result):
    register, rows = result["register"], result["measurements"]
    cells = {(r["state"], r["set"]): r for r in rows}
    done = sum(r["loss"] is not None for r in rows)
    lines = ["# V71 QA scope", "", f"Status: {done}/24 cells measured. "
             "Unmeasured values are shown as --; no empirical conclusion until evaluation.", "",
             "Gemma-3-1B; eight pre-specified states; 256 fixed references per set. "
             "Delta = state loss minus dense loss on the same set; negative means improvement.", "",
             INFORMATION, "", "## Pre-specified register", "",
             "| State | Nominal supervised T | Actual supervised T | Processed trigger | Checkpoint |",
             "| --- | ---: | ---: | ---: | --- |"]
    for s in register["states"]:
        lines.append(f"| {s['label']} | {s.get('planned_T', '--')} | {s.get('actual_T', '--')} | "
                     f"{s.get('trigger_processed', '--')} | {Path(s.get('checkpoint', '--')).name} |")
    lines += ["", "## Sample and overlap audit", ""]
    primary, musique = register["sets"]["2wiki_new"], register["sets"]["musique"]
    lines += [f"2Wiki: seed {primary['seed']}; {len(primary['probe_indices'])} old probe and "
              f"{len(primary['measurement_indices'])} old measurement indices recovered; "
              f"{len(primary['excluded_indices'])} source rows excluded including duplicate questions. "
              f"Selected overlap: {primary['sampled_overlap_count']}. {primary['index_provenance']}", "",
              f"MuSiQue: seed {musique['seed']}; {musique['answerable_rows']} answerable rows checked "
              f"against {len(musique['teacher_trace_files'])} full V12 teacher pool files "
              f"({musique['teacher_unique_questions']} unique question strings). "
              f"Overlap count: {musique['overlap_count']} items / {musique['overlap_unique_questions']} "
              f"unique questions; excluded before sampling. Selected overlap: {musique['sampled_overlap_count']}. "
              + musique["matching"], "", "TriviaQA: seed 0; V48/V9 rc.nocontext validation builder.", "",
              "Source hashes, exact source indices, adapter hashes, and prompt/reference hashes are in register.json.", "",
              "## Conditional loss", "", "| State | Set | Loss | Delta from dense | Tokens | n |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for s in register["states"]:
        for key in SETS:
            r = cells[s["state"], key]
            loss = "--" if r["loss"] is None else f"{r['loss']:.6f}"
            delta = "--" if r["delta_from_dense"] is None else f"{r['delta_from_dense']:+.6f}"
            lines.append(f"| {s['label']} | {key} | {loss} | {delta} | {r['tokens'] if r['tokens'] is not None else '--'} | {r['n']} |")
    lines += ["", "## Three pre-stated readings", "", READING_RULE, ""]
    for r in READINGS:
        lines.append(f"- **{r['reading']}**: {r['meaning']}")
    lines += ["", "| KD budget | Observed reading |", "| --- | --- |"]
    for s in register["states"]:
        if s["kind"] == "adapter":
            lines.append(f"| {s['planned_T']} | {reading(cells[s['state'], '2wiki_new']['delta_from_dense'], cells[s['state'], 'musique']['delta_from_dense'])} |")
    tex = [r"\begin{table}[H]", r"\centering", r"\small",
           r"\caption{QA measurement scope on Gemma-3-1B: conditional loss and $\Delta L$ from dense (native-token nats; negative is improvement), 256 fixed references per set. KD labels denote nominal supervised-token budgets. 2Wiki supplies all context; MuSiQue supplies supporting paragraphs; TriviaQA uses V48's no-context control. "
           + (r"CPU registration only; all cells are unmeasured." if done == 0 else f"{done}/24 cells measured.") + "}",
           r"\label{tab:qa_scope}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r"& \multicolumn{2}{c}{New 2Wiki} & \multicolumn{2}{c}{MuSiQue} & \multicolumn{2}{c}{TriviaQA} \\",
           r"State & Loss & $\Delta L$ & Loss & $\Delta L$ & Loss & $\Delta L$ \\", r"\midrule"]
    for s in register["states"]:
        values = []
        for key in SETS:
            r = cells[s["state"], key]
            values.extend(["--", "--"] if r["loss"] is None else [f"{r['loss']:.3f}", f"{r['delta_from_dense']:+.3f}"])
        tex.append(s["label"] + " & " + " & ".join(values) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n", "\n".join(tex) + "\n"


def save_outputs(out, table, result):
    done = sum(r["loss"] is not None for r in result["measurements"])
    result["status"] = "complete" if done == 24 else "partial" if done else "not_measured"
    validate_results(result, result["register"])
    markdown, tex = reports(result)
    atomic_text(out / "measurements.json", json_text(result))
    atomic_text(out / "summary.md", markdown)
    atomic_text(table, tex)


def cuda_uuid():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or any(not re.fullmatch(r"GPU-[0-9a-fA-F]+(?:-[0-9a-fA-F]+)*", s.strip()) for s in visible.split(",")):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to caller-selected GPU UUID(s), not physical ordinals")
    return visible


def configure_hf():
    if os.environ.get("V67_ALLOW_ONLINE") != "1":
        for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ[key] = "1"


def apply_state(base, spec):
    from analysis import v6_capability_geometry as v6
    from analysis.v54_quant_group import _apply_grouped_quantization

    if spec["kind"] == "prune":
        return {"prune_threshold": v6.apply_global_magnitude_pruning(base, spec["density"], seed=0)}
    if spec["kind"] == "quant":
        params = v6.language_weight_parameters(base)
        dense = [p.detach().clone() for _, p in params]
        _apply_grouped_quantization(params, dense, spec["bits"], spec["group_size"])
    return {}


def evaluate(register, panels, out, table, result, root=ROOT):
    visible = cuda_uuid()
    import torch
    from peft import PeftModel
    from analysis.v12_distill import load_text_causal_lm
    from analysis.v67_musique_qa import measure_qa

    if not torch.cuda.is_available():
        raise RuntimeError(f"No CUDA available under CUDA_VISIBLE_DEVICES={visible!r}")
    cells = {(r["state"], r["set"]): r for r in result["measurements"]}
    for spec in register["states"]:
        pending = [key for key in SETS if cells[spec["state"], key]["loss"] is None]
        if not pending:
            continue
        base = tokenizer = model = None
        try:
            print(f"V71: {spec['state']} ({', '.join(pending)})", flush=True)
            base, tokenizer = load_text_causal_lm(HF_ID, torch.bfloat16, register["revision"])
            tokenizer.truncation_side = "right"
            base.to("cuda:0").eval()
            transform = apply_state(base, spec)
            model = (PeftModel.from_pretrained(base, str(root / spec["adapter"]),
                                             local_files_only=True, is_trainable=False)
                     if spec["kind"] == "adapter" else base)
            model.eval()
            for key in pending:
                loss, tokens = measure_qa(model, tokenizer, panels[key], "cuda:0")
                dense = loss if spec["state"] == "dense" else cells["dense", key]["loss"]
                cells[spec["state"], key].update(loss=loss, tokens=tokens, n=len(panels[key]),
                                               delta_from_dense=loss - dense,
                                               cuda_visible_devices=visible, **transform)
                save_outputs(out, table, result)
                print(json.dumps(cells[spec["state"], key]), flush=True)
        finally:
            del model, base, tokenizer
            gc.collect()
            torch.cuda.empty_cache()


def selftest():
    """Meaningful CPU fixtures: leakage exclusions, selection, scoring, state math, reporting."""
    import numpy as np
    import torch
    from analysis import v6_capability_geometry as v6
    from analysis.v34_c17_scope_audit import normalize
    from analysis.v54_quant_group import fake_quantize_grouped
    from analysis.v67_musique_qa import selftest as scoring_selftest

    def forbidden(*args, **kwargs):
        raise AssertionError("Authoring test attempted pretrained loading, network, or CUDA")

    with patch("torch.cuda.is_available", forbidden), patch("torch.cuda.init", forbidden), \
            patch("analysis.v12_distill.load_text_causal_lm", forbidden), \
            patch("socket.socket.connect", forbidden):
        torch.set_num_threads(1)
        scoring_selftest()  # Actual V6/V67 masking, BOS, truncation, token-weighted CE.
        wiki = [{"context": {"title": ["A"], "sentences": [["Fact."]]},
                 "question": f"Question {i}?", "answer": f"Answer {i}"} for i in range(600)]
        fixture = {"HuggingFaceH4/MATH-500": [{"problem": "P", "solution": "S", "answer": "A"}] * 500,
                   "google-research-datasets/mbpp": [{"text": "P", "code": "pass"}] * 500,
                   SETS["2wiki_new"]["dataset"]: wiki}
        halves, legacy = recover_v6_indices(fixture, 128)
        rng = np.random.default_rng(0)
        rng.choice(500, 128, replace=False)
        rng.choice(500, 128, replace=False)
        expected = rng.choice(600, 128, replace=False).tolist()
        assert halves == {"probe_indices": expected[::2], "measurement_indices": expected[1::2]}
        assert legacy == [two_wiki_probe(wiki[i]) for i in expected]
        new = fresh_indices(600, expected)
        assert new == fresh_indices(600, expected) and len(new) == 256 and not set(new) & set(expected)
        try:
            fresh_indices(128, expected)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid/undersized exclusions accepted")
        example = {"question": "Shared question?", "answer": "A", "paragraphs": [
            {"title": "Support", "paragraph_text": "Evidence", "is_supporting": True},
            {"title": "Distractor", "paragraph_text": "Do not include", "is_supporting": False}]}
        rows = [example, {**example, "question": " SHARED  QUESTION? "},
                {**example, "question": "Unique?"}, {**example, "question": "Other?"},
                {**example, "answerable": False}]
        probes, meta = musique_panel(rows, {normalize("Shared question?")}, n=2)
        assert meta["overlap_count"] == 2 and meta["overlap_unique_questions"] == 1
        assert set(meta["indices"]) == {2, 3} and meta["sampled_overlap_count"] == 0
        assert all("Distractor" not in p["prompt"] and p["completion"] == " A" for p in probes)
        weight = torch.linspace(-2, 2, 4 * 259).reshape(4, 259)
        for spec in STATES[1:5]:
            model = torch.nn.Linear(259, 4, bias=False)
            with torch.no_grad():
                model.weight.copy_(weight)
            metadata = apply_state(model, spec)
            if spec["kind"] == "prune":
                sample = v6._sample_abs_weights([("weight", weight)], seed=0)
                threshold = float(np.quantile(sample, 1 - spec["density"]))
                assert metadata["prune_threshold"] == threshold
                assert torch.equal(model.weight, weight * (weight.abs() > threshold))
            else:
                assert torch.equal(model.weight, fake_quantize_grouped(weight, 4, spec["group_size"]))
        assert reading(-1, -1) == READINGS[0]["reading"]
        assert reading(-1, 0) == READINGS[1]["reading"]
        assert reading(0, 0) == READINGS[2]["reading"]
        assert "unanticipated" in reading(1, -1) and reading(None, -1) == "Pending"
        for value in ("", "0", "-1", "GPU-abcd,1"):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": value}):
                try:
                    cuda_uuid()
                except ValueError:
                    pass
                else:
                    raise AssertionError("Invalid CUDA selector accepted")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-20b20454"}):
            assert cuda_uuid() == "GPU-20b20454"
        for online in ("0", "1"):
            with patch.dict(os.environ, {"V67_ALLOW_ONLINE": online}, clear=True):
                configure_hf()
                assert os.environ.get("HF_HUB_OFFLINE") == (None if online == "1" else "1")
        with tempfile.TemporaryDirectory(prefix="v71-selftest-") as tmp:
            out = Path(tmp)
            minimal = {"states": list(STATES), "sets": {
                "2wiki_new": {"seed": 71, **halves, "excluded_indices": expected,
                              "sampled_overlap_count": 0, "index_provenance": "Fixture"},
                "musique": {"seed": 0, **meta, "teacher_trace_files": ["fixture"], "teacher_unique_questions": 1}}}
            result = initial_results(minimal)
            freeze_register(out, minimal)
            save_outputs(out, out / "qa_scope.tex", result)
            assert result["status"] == "not_measured" and "\\begin{table}[H]" in (out / "qa_scope.tex").read_text()
            for row in result["measurements"]:
                row.update(loss=2.0 if row["state"] == "dense" else 1.0, tokens=500, n=N,
                           delta_from_dense=0.0 if row["state"] == "dense" else -1.0)
            save_outputs(out, out / "qa_scope.tex", result)
            assert result["status"] == "complete" and "24/24" in (out / "summary.md").read_text()
            result["measurements"][-1]["tokens"] += 1
            try:
                validate_results(result, minimal)
            except ValueError:
                pass
            else:
                raise AssertionError("Changing completion token counts accepted")
            try:
                freeze_register(out, {**minimal, "changed": True})
            except ValueError:
                pass
            else:
                raise AssertionError("Register overwrite accepted")
        # Nominal supervised T maps to a registered processed trigger. In
        # particular, actual T=138524 is the 140k state despite falling below it.
        with tempfile.TemporaryDirectory(prefix="v71-checkpoints-") as tmp:
            from contextlib import redirect_stdout
            from io import StringIO

            root = Path(tmp)
            targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            pool_register = {"student_snapshot": "fixture-revision", "pools": {
                "U450_s11": {"pool_processed_tokens": 400438, "D_U_completion": 118791}},
                "v2": {"protocol_v2": {"planned_supervised_T": [35000, 70000, 140000, 280000],
                                        "processed_triggers": [119000, 237000, 475000, 949000],
                                        "lora": {"targets": targets}}}}
            atomic_text(root / "results/v47-p2-register/register.json", json_text(pool_register))
            for update, trigger, actual in [(26, 119000, 36203), (50, 237000, 69539),
                                            (100, 475000, 138524), (199, 949000, 280329)]:
                checkpoint = root / RUN / "trajectory" / f"update-{update:08d}"
                metadata = {"student": MODEL, "teacher": "gpt-5.6-luna", "recipe": "full",
                            "n_per_domain": 450, "data_seed": 11, "training_mode": "lora",
                            "output_suffix": "_p2v2", "schedule_tokens": 1000000, "training_seed": 0,
                            "probe_seed": 0, "unique_data_pool_tokens": 400438, "updates": update,
                            "requested_token_milestones": [trigger], "processed_tokens": trigger + 5000,
                            "completion_tokens_seen": actual}
                atomic_text(checkpoint / "eval.json", json_text(metadata))
                atomic_text(checkpoint / "adapter/adapter_config.json", json_text({
                    "base_model_name_or_path": HF_ID, "peft_type": "LORA", "r": 16,
                    "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": targets}))
                atomic_text(checkpoint / "adapter/adapter_model.bin", "fixture, never loaded")
            plan, revision, _ = checkpoint_plan(root)
            assert [r["actual_T"] for r in plan if r["kind"] == "adapter"] == [36203, 138524, 280329]
            assert revision == "fixture-revision"
            # Exercise the orchestration with inert objects: each state gets a
            # fresh base; PEFT may mutate it; restart skips completed cells.
            register = {**minimal, "states": plan, "revision": revision}
            result = initial_results(register)
            bases, adapted = [], []

            class InertBase:
                def to(self, device):
                    assert device == "cuda:0"  # Logical address only; no tensor allocation.
                    return self

                def eval(self):
                    return self

            def fake_load(name, dtype, rev):
                assert (name, dtype, rev) == (HF_ID, torch.bfloat16, revision)
                base = InertBase()
                bases.append(base)
                return base, InertBase()

            def fake_adapter(base, path, **kwargs):
                assert base is bases[-1] and not hasattr(base, "adapter")
                assert kwargs == {"local_files_only": True, "is_trainable": False}
                base.adapter = path
                adapted.append(base)
                return base

            with patch("analysis.v12_distill.load_text_causal_lm", side_effect=fake_load), \
                    patch("peft.PeftModel.from_pretrained", side_effect=fake_adapter), \
                    patch("analysis.v67_musique_qa.measure_qa", return_value=(2.0, 500)), \
                    patch("torch.cuda.is_available", return_value=True), \
                    patch("torch.cuda.empty_cache"), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-abcd"}), \
                    redirect_stdout(StringIO()):
                with patch.dict(evaluate.__globals__, {"apply_state": lambda base, spec: {}}):
                    panels = {k: [{}] * N for k in SETS}
                    evaluate(register, panels, root / "output", root / "qa_scope.tex", result, root)
                    evaluate(register, panels, root / "output", root / "qa_scope.tex", result, root)
            assert len(bases) == 8 and len(adapted) == 3 and result["status"] == "complete"
            missing = root / RUN / "trajectory/update-00000100/eval.json"
            d = read_json(missing)
            d["requested_token_milestones"] = []
            atomic_text(missing, json_text(d))
            try:
                checkpoint_plan(root)
            except ValueError:
                pass
            else:
                raise AssertionError("Missing 140k checkpoint silently substituted")
    print("V71 selftest: PASS (CPU only; exact V6 replay; both-half exclusions; normalized trace overlap; "
          "support-only MuSiQue; actual conditional scorer; both pruning/RTN states; UUID/offline guards; "
          "write-once register; nominal checkpoint mapping; fresh adapter bases; resume; 24-cell reporting)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="CPU preflight of local data/checkpoints; no pretrained model, CUDA, network, or writes")
    modes.add_argument("--selftest", action="store_true", help="CPU fixture tests; no real datasets/models")
    parser.add_argument("--write-plan", action="store_true", help="With --dry-run only: freeze register and write explicitly unmeasured outputs")
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("HF_DATASETS_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "datasets")))
    parser.add_argument("--data-file", type=Path, help="Existing MuSiQue answerable-dev JSONL (V67 lookup if omitted)")
    parser.add_argument("--trace-base", type=Path, default=ROOT / "results/traces-pilot")
    parser.add_argument("--out", type=Path, default=ROOT / "results/v71-qa-scope")
    parser.add_argument("--table", type=Path, default=ROOT / "paper/paper/tables/qa_scope.tex")
    args = parser.parse_args(argv)
    if args.write_plan and not args.dry_run:
        parser.error("--write-plan requires --dry-run")
    if args.selftest:
        selftest()
        return
    try:
        if not args.dry_run:
            cuda_uuid()  # Reject accidental unselected GPU evaluation before imports.
        configure_hf()  # Before any HF import; follows V67_ALLOW_ONLINE behavior.
        from analysis.v67_musique_qa import local_data_file

        register, panels = make_register(args.cache_dir, local_data_file(args.data_file), args.trace_base)
        result = initial_results(register)
        if (args.out / "measurements.json").exists():
            result = read_json(args.out / "measurements.json")
            validate_results(result, register)
        if args.dry_run:
            print(json_text({"dry_run": True, "cpu_only": True, "pretrained_model_loaded": False,
                             "network_used": False, "n_evaluations": 24, "register": register}))
            if args.write_plan:
                freeze_register(args.out, register)
                save_outputs(args.out, args.table, result)
            return
        freeze_register(args.out, register)
        save_outputs(args.out, args.table, result)
        evaluate(register, panels, args.out, args.table, result)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"V71: {exc}\n")


if __name__ == "__main__":
    main()
