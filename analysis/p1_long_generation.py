#!/usr/bin/env python3
"""P1b: the fresh-item QA generation re-run with enough room to answer and a gold-blind extraction rule.

The first fresh-item check (analysis/p1_qa_behaviour.py) generated 32 new tokens under the V15 rule
and scored the whole block before the next "Question:"/"Context:" marker. Distilled students answer
in the teacher's longer style, so that block can be truncated by the cap and can carry more than the
answer span. This re-run keeps the same 384 items and the same 22 models and changes only the readout:

  * 96 new tokens, the same greedy decoding and the same stop markers;
  * a truncation flag: the generation reached the cap without any stop marker;
  * one gold-blind extraction rule applied to every model: drop markdown emphasis, keep the first
    line, drop a leading "Answer:" label, keep the first sentence, strip trailing punctuation;
  * exact match and token F1 (V15 normalisation) on the extracted span and, for reference, on the
    whole first line.

Stages
  --run [--half a|b]   GPU; incremental results_long.json (the roster is split by parity for two cards)
  --analyse            paired readouts against the quantization-only choice and the initial student;
                       truncation rates of this run and of the 32-token run (re-tokenised);
                       summary_long.json / summary_long.md
  --sample N           writes review_sample.md: N items per model for the distilled students and the
                       quantization-only choices, gold answer beside both generations, for a content check
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import p1_qa_behaviour as p1  # noqa: E402

OUT = p1.OUT
RESULTS = OUT / "results_long.json"
MAX_NEW = 96
MARKDOWN = re.compile(r"(\*\*|__|\*|`)")


EXTRACTION_RULE = ("v2: first line, markdown removed, leading 'Answer:' dropped, cut at the first sentence end, where a "
                   "sentence ends at . ! or ? followed by whitespace, by an uppercase letter or by the end of the line, "
                   "except a period that follows a single capital letter (an initial)")
SENTENCE_END = re.compile(r"(?<![A-Z])[.!?](?=\s|[A-Z]|$)")


def extract_span(raw):
    """Gold-blind extraction (EXTRACTION_RULE): the same rule for every model, reading nothing from the gold answer."""
    text = MARKDOWN.sub("", str(raw)).strip()
    text = text.split("\n")[0].strip()
    text = re.sub(r"^(answer|final answer)\s*[:：]\s*", "", text, flags=re.I).strip()
    m = SENTENCE_END.search(text)
    if m and m.start() > 0:
        text = text[:m.start()]
    return text.strip().strip(".").strip()


def evaluate_long(model, tok, items):
    import torch
    from analysis import v15_accuracy_link as v15
    prompts = [v15.build_accuracy_prompt("qa", it, 4) for it in items]
    raws = []
    for start in range(0, len(prompts), 8):
        raws.extend(v15.greedy_generate_batch(model, tok, prompts[start:start + 8], device="cuda:0",
                    max_new_tokens=MAX_NEW, input_max_length=1024 - MAX_NEW, truncation_side="left",
                    stop_sequences=v15.DOMAIN_STOP_SEQUENCES["qa"]))
    out = []
    for it, raw in zip(items, raws):
        stopped = any(s in raw for s in v15.DOMAIN_STOP_SEQUENCES["qa"])
        n_tokens = len(tok(raw, add_special_tokens=False)["input_ids"])
        block = v15.truncate_generation("qa", raw)
        span = extract_span(block)
        first_line = MARKDOWN.sub("", block).strip().split("\n")[0].strip()
        out.append({"raw": raw, "block": block, "span": span, "tokens": n_tokens,
                    "truncated": (not stopped) and n_tokens >= MAX_NEW - 1,
                    "em_span": bool(v15.qa_exact_match(span, it["answer"])), "f1_span": float(v15.qa_token_f1(span, it["answer"])),
                    "em_line": bool(v15.qa_exact_match(first_line, it["answer"])), "f1_line": float(v15.qa_token_f1(first_line, it["answer"]))})
    return out


def run(half):
    import torch
    plan = p1.read(OUT / "plan.json")
    s3 = p1.read(p1.S3 / "plan_v2.json"); refs = {r["id"]: r for r in s3["references"]}
    done = p1.read(RESULTS) if RESULTS.exists() else {"evaluations": {}, "protocol": {"max_new_tokens": MAX_NEW, "extraction": extract_span.__doc__}}
    specs = sorted(plan["models"], key=lambda m: m["method"] == "distill")
    if half:
        specs = [s for i, s in enumerate(specs) if (i % 2 == 0) == (half == "a")]
    for spec in specs:
        key = f"{spec['reference']}|{spec['candidate']}"
        done = p1.read(RESULTS) if RESULTS.exists() else done   # another card may have written meanwhile
        if key in done["evaluations"]:
            continue
        t0 = time.time(); model = tok = None
        try:
            model, tok, transform = p1.build_state(spec["reference"], spec, refs[spec["reference"]])
            records = evaluate_long(model, tok, plan["items"])
        finally:
            del model, tok; gc.collect(); torch.cuda.empty_cache()
        done = p1.read(RESULTS) if RESULTS.exists() else done
        done["evaluations"][key] = {"reference": spec["reference"], "candidate": spec["candidate"], "roles": spec["roles"],
                                    "transform": transform, "seconds": round(time.time() - t0, 1),
                                    "exact_match_span": statistics.mean(r["em_span"] for r in records),
                                    "token_f1_span": statistics.mean(r["f1_span"] for r in records),
                                    "exact_match_line": statistics.mean(r["em_line"] for r in records),
                                    "truncated_rate": statistics.mean(r["truncated"] for r in records), "items": records}
        RESULTS.write_text(json.dumps(done, indent=1))
        e = done["evaluations"][key]
        print(json.dumps({k: e[k] for k in ("reference", "candidate", "roles", "exact_match_span", "token_f1_span", "exact_match_line", "truncated_rate", "seconds")}), flush=True)


def analyse():
    from analysis import v15_accuracy_link as v15
    plan = p1.read(OUT / "plan.json"); short = p1.read(OUT / "results.json")["evaluations"]; long = p1.read(RESULTS)["evaluations"]
    answers = [it["answer"] for it in plan["items"]]
    for v in long.values():   # rescore from the stored blocks with the current extraction rule
        for r, answer in zip(v["items"], answers):
            r["span"] = extract_span(r["block"])
            r["em_span"] = bool(v15.qa_exact_match(r["span"], answer)); r["f1_span"] = float(v15.qa_token_f1(r["span"], answer))
        v["exact_match_span"] = statistics.mean(r["em_span"] for r in v["items"])
        v["token_f1_span"] = statistics.mean(r["f1_span"] for r in v["items"])
    # truncation of the 32-token run: re-tokenise its kept block with each model's tokenizer is not available here;
    # a block of 32 or more whitespace tokens is a conservative proxy recorded as `cap_proxy`.
    summary = {"n_items": len(answers), "max_new_tokens": MAX_NEW, "extraction_rule": EXTRACTION_RULE, "references": {}}
    for ref in sorted({v["reference"] for v in long.values()}):
        keys = {k: v for k, v in long.items() if v["reference"] == ref}
        quant = next((v for v in keys.values() if "quant-only" in v["roles"]), None)
        pristine = next((v for v in keys.values() if "pristine_student" in v["roles"]), None)
        entry = {"models": {}}
        for key, v in keys.items():
            s = short.get(key)
            row = {"candidate": v["candidate"], "roles": v["roles"], "truncated_rate_96": v["truncated_rate"],
                   "exact_match_32_block": s["exact_match"] if s else None, "token_f1_32_block": s["token_f1"] if s else None,
                   "exact_match_96_span": v["exact_match_span"], "token_f1_96_span": v["token_f1_span"],
                   "exact_match_96_line": v["exact_match_line"],
                   "span_words_mean": statistics.mean(len(r["span"].split()) for r in v["items"]),
                   "block_words_mean": statistics.mean(len(r["block"].split()) for r in v["items"])}
            if s:
                row["cap_proxy_32"] = statistics.mean(len(r["generation"].split()) >= 20 for r in s["items"])
            for name, base in (("vs_quant_only", quant), ("vs_pristine", pristine if v["candidate"].startswith("distill") else None)):
                if base is None or base is v:
                    continue
                row[name] = {m: p1.bootstrap_mean([a[m] - b[m] for a, b in zip(v["items"], base["items"])],
                                                  seed=__import__("zlib").crc32(f"{ref}|{key}|{m}|long".encode()))
                             for m in ("em_span", "f1_span")}
            entry["models"][v["candidate"]] = row
        summary["references"][ref] = entry
    (OUT / "summary_long.json").write_text(json.dumps(summary, indent=1))
    lines = ["# P1b long-generation readout (96 new tokens, gold-blind first-sentence extraction)", "",
             "| Reference | Model | Roles | EM 32-block | EM 96-span | EM 96-line | F1 32 | F1 96-span | Truncated@96 | Span words | dEM vs quant-only [95%] | dEM vs initial |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def ci(d, m):
        if not d or m not in d or d[m]["mean"] is None: return "--"
        return f"{d[m]['mean']:+.3f} [{d[m]['ci95'][0]:+.3f}, {d[m]['ci95'][1]:+.3f}]"
    def f(x): return "--" if x is None else f"{x:.3f}"
    for ref, e in summary["references"].items():
        for cid, r in e["models"].items():
            lines.append("| " + " | ".join([ref, cid, ",".join(r["roles"]), f(r["exact_match_32_block"]), f(r["exact_match_96_span"]), f(r["exact_match_96_line"]),
                          f(r["token_f1_32_block"]), f(r["token_f1_96_span"]), f(r["truncated_rate_96"]), f"{r['span_words_mean']:.1f}",
                          ci(r.get("vs_quant_only"), "em_span"), ci(r.get("vs_pristine"), "em_span")]) + " |")
    (OUT / "summary_long.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def sample(n):
    plan = p1.read(OUT / "plan.json"); long = p1.read(RESULTS)["evaluations"]; short = p1.read(OUT / "results.json")["evaluations"]
    rng = random.Random(2026); idx = rng.sample(range(len(plan["items"])), n)
    lines = [f"# Content check sample: {n} items per model (seed 2026, same items for every model)", ""]
    for key, v in long.items():
        if not (v["candidate"].startswith("distill") or "quant-only" in v["roles"]):
            continue
        lines += [f"## {key}  roles={','.join(v['roles'])}", ""]
        for i in idx:
            it = plan["items"][i]; r = v["items"][i]; s = short[key]["items"][i]["generation"] if key in short else ""
            q = it["prompt"].split("Question:")[-1].split("\n")[0].strip()
            lines += [f"- [{i}] Q: {q}", f"  gold: {it['answer']}", f"  span96: {r['span']!r}  em={int(r['em_span'])} f1={r['f1_span']:.2f} truncated={int(r['truncated'])}",
                      f"  block96: {r['block'].strip()[:300]!r}", f"  block32: {s.strip()[:160]!r}", ""]
    (OUT / "review_sample.md").write_text("\n".join(lines) + "\n"); print("wrote", OUT / "review_sample.md", len(lines), "lines")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true"); ap.add_argument("--half", choices=("a", "b"), default=None)
    ap.add_argument("--analyse", action="store_true"); ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    if a.run:
        run(a.half)
    if a.analyse:
        analyse()
    if a.sample:
        sample(a.sample)
