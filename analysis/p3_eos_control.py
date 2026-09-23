#!/usr/bin/env python3
"""P3b: paired end-marker control for one selection-round student.

The selection round's distilled students were trained on teacher answers with no end-of-answer marker
(results/p3-eos-control/token_audit.md). This control retrains the same student with the same data, recipe,
seed and schedule and one change, the tokenizer's end-of-sequence id appended to every target and supervised
(`v12_distill.py --append-eos`), then reads both students out on the 384 fresh question-answering items of
the fresh-item check with the two protocols of that check: 32 new tokens under the V15 rule (loss on the
reference completion, exact match on the block) and 96 new tokens with the gold-blind first-sentence
extraction (exact match, token F1, cap rate, generated tokens). The original student's readouts are the
stored ones; only the control is evaluated here. Also reports both students' probe losses from their
training records.

  --evaluate --reference pythia-1.4b--step120000   GPU: evaluate the control student; writes results.json
  --analyse                                         paired differences (item bootstrap) and summary.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import p1_qa_behaviour as p1  # noqa: E402
from analysis import p1_long_generation as p1l  # noqa: E402

OUT = ROOT / "results/p3-eos-control"
STUDENT_BASE = {"pythia-1.4b--step120000": "EleutherAI/pythia-410m", "pythia-410m--step120000": "EleutherAI/pythia-160m",
                "gemma3-1b": "google/gemma-3-270m", "gemma3-4b": "google/gemma-3-1b-pt"}
CONTROL_SUFFIX = "p3-eos-s3-"


def control_run_dir(ref):
    base = STUDENT_BASE[ref].replace("/", "--")
    runs = [d for d in (ROOT / "results/v12-distill").glob(f"{base}*/*{CONTROL_SUFFIX}{ref}*") if (d / "adapter" / "adapter_model.safetensors").is_file()]
    if len(runs) != 1:
        raise SystemExit(f"expected one control run for {ref}, found {runs}")
    return runs[0]


def s3_run_dir(ref):
    runs = list((ROOT / "results/s3-selection-validation/students" / ref / "v12").glob("*/*"))
    runs = [d for d in runs if (d / "adapter" / "adapter_model.safetensors").is_file()]
    if len(runs) != 1:
        raise SystemExit(f"expected one S3 run for {ref}, found {runs}")
    return runs[0]


def evaluate(ref):
    import gc
    import torch
    from peft import PeftModel
    from analysis.v12_distill import load_text_causal_lm
    plan = p1.read(p1.OUT / "plan.json"); items = plan["items"]
    run = control_run_dir(ref)
    cfg = p1.read(run / "eval.json")
    if not cfg.get("append_eos"):
        raise SystemExit(f"{run} was not trained with --append-eos")
    t0 = time.time()
    model, tok = load_text_causal_lm(str(p1.snapshot_dir(STUDENT_BASE[ref])), torch.bfloat16)
    model.to("cuda:0").eval().requires_grad_(False)
    model = PeftModel.from_pretrained(model, str(run / "adapter"), local_files_only=True, is_trainable=False).eval()
    short = p1.evaluate_model(model, tok, items)
    long = p1l.evaluate_long(model, tok, items)
    del model; gc.collect(); torch.cuda.empty_cache()
    out = p1.read(OUT / "results.json") if (OUT / "results.json").exists() else {"evaluations": {}}
    out["evaluations"][f"{ref}|distill:eos_control"] = {
        "reference": ref, "candidate": "distill:eos_control", "run": str(run.relative_to(ROOT)), "seconds": round(time.time() - t0, 1),
        "training": {k: cfg.get(k) for k in ("student", "teacher", "recipe", "n_per_domain", "epochs", "seed", "data_seed", "learning_rate", "training_mode", "append_eos")},
        "short": {"exact_match": statistics.mean(r["exact_match"] for r in short), "token_f1": statistics.mean(r["token_f1"] for r in short),
                  "loss_token_weighted": sum(r["loss_sum"] for r in short) / sum(r["tokens"] for r in short), "items": short},
        "long": {"exact_match_span": statistics.mean(r["em_span"] for r in long), "token_f1_span": statistics.mean(r["f1_span"] for r in long),
                 "truncated_rate": statistics.mean(r["truncated"] for r in long), "tokens_mean": statistics.mean(r["tokens"] for r in long), "items": long}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, indent=1))
    e = out["evaluations"][f"{ref}|distill:eos_control"]
    print(json.dumps({"short_em": e["short"]["exact_match"], "loss": e["short"]["loss_token_weighted"], "long_em": e["long"]["exact_match_span"],
                      "truncated": e["long"]["truncated_rate"], "tokens": e["long"]["tokens_mean"], "seconds": e["seconds"]}))


def analyse():
    import zlib
    control = p1.read(OUT / "results.json")["evaluations"]
    short_all = p1.read(p1.OUT / "results.json")["evaluations"]; long_all = p1.read(p1l.RESULTS)["evaluations"]
    summary = {"references": {}}
    for key, e in control.items():
        ref = e["reference"]
        s3_short = short_all[f"{ref}|distill:new_s3"]; s3_long = long_all[f"{ref}|distill:new_s3"]
        others = {name: (short_all[f"{ref}|{cid}"], long_all[f"{ref}|{cid}"]) for name, cid in
                  (("quant_only", next(v["candidate"] for v in short_all.values() if v["reference"] == ref and "quant-only" in v["roles"])),
                   ("pristine", "pristine:student"))}
        seed = zlib.crc32(f"{ref}|eos".encode())
        entry = {"control_run": e["run"], "training": e["training"],
                 "probe_losses": {"s3_student": p1.read(s3_run_dir(ref) / "eval.json").get("post_training"), "control": p1.read(ROOT / e["run"] / "eval.json").get("post_training")},
                 "control": {"loss": e["short"]["loss_token_weighted"], "em_32": e["short"]["exact_match"], "f1_32": e["short"]["token_f1"],
                             "em_96": e["long"]["exact_match_span"], "f1_96": e["long"]["token_f1_span"], "truncated_96": e["long"]["truncated_rate"], "tokens_96": e["long"]["tokens_mean"]},
                 "s3_student": {"loss": sum(r["loss_sum"] for r in s3_short["items"]) / sum(r["tokens"] for r in s3_short["items"]), "em_32": s3_short["exact_match"], "f1_32": s3_short["token_f1"],
                                "em_96": s3_long["exact_match_span"], "f1_96": s3_long["token_f1_span"], "truncated_96": s3_long["truncated_rate"],
                                "tokens_96": statistics.mean(r["tokens"] for r in s3_long["items"])},
                 "paired_control_minus_s3": {
                     "loss": p1.bootstrap_weighted_difference(e["short"]["items"], s3_short["items"], seed),
                     "em_32": p1.bootstrap_mean([a["exact_match"] - b["exact_match"] for a, b in zip(e["short"]["items"], s3_short["items"])], seed + 1),
                     "em_96": p1.bootstrap_mean([a["em_span"] - b["em_span"] for a, b in zip(e["long"]["items"], s3_long["items"])], seed + 2),
                     "f1_96": p1.bootstrap_mean([a["f1_span"] - b["f1_span"] for a, b in zip(e["long"]["items"], s3_long["items"])], seed + 3),
                     "truncated_96": p1.bootstrap_mean([float(a["truncated"]) - float(b["truncated"]) for a, b in zip(e["long"]["items"], s3_long["items"])], seed + 4),
                     "tokens_96": p1.bootstrap_mean([a["tokens"] - b["tokens"] for a, b in zip(e["long"]["items"], s3_long["items"])], seed + 5)},
                 "comparators": {}}
        for name, (sh, lo) in others.items():
            entry["comparators"][name] = {"candidate": sh["candidate"], "loss": sum(r["loss_sum"] for r in sh["items"]) / sum(r["tokens"] for r in sh["items"]),
                                          "em_32": sh["exact_match"], "em_96": lo["exact_match_span"], "truncated_96": lo["truncated_rate"],
                                          "control_minus_this": {"loss": p1.bootstrap_weighted_difference(e["short"]["items"], sh["items"], seed + 10),
                                                                 "em_96": p1.bootstrap_mean([a["em_span"] - b["em_span"] for a, b in zip(e["long"]["items"], lo["items"])], seed + 11)}}
        summary["references"][ref] = entry
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["# P3b end-marker control: paired readouts on the 384 fresh items", ""]
    def ci(d): return f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
    for ref, r in summary["references"].items():
        c, s = r["control"], r["s3_student"]; d = r["paired_control_minus_s3"]
        lines += [f"## {ref}: student {r['training']['student']}, control run {r['control_run']}", "",
                  "| Readout | S3 student (no marker) | Control (marker) | Control minus S3 [95%] |", "|---|---:|---:|---:|",
                  f"| Token-weighted loss (nats) | {s['loss']:.3f} | {c['loss']:.3f} | {ci(d['loss'])} |",
                  f"| Exact match, 32 tokens | {s['em_32']:.3f} | {c['em_32']:.3f} | {ci(d['em_32'])} |",
                  f"| Exact match, 96 tokens (span) | {s['em_96']:.3f} | {c['em_96']:.3f} | {ci(d['em_96'])} |",
                  f"| Token F1, 96 tokens (span) | {s['f1_96']:.3f} | {c['f1_96']:.3f} | {ci(d['f1_96'])} |",
                  f"| Reached the 96-token cap | {s['truncated_96']:.3f} | {c['truncated_96']:.3f} | {ci(d['truncated_96'])} |",
                  f"| Generated tokens (mean) | {s['tokens_96']:.1f} | {c['tokens_96']:.1f} | {ci(d['tokens_96'])} |", ""]
        for name, o in r["comparators"].items():
            lines.append(f"- Control minus {name} ({o['candidate']}): loss {ci(o['control_minus_this']['loss'])}, exact match 96 {ci(o['control_minus_this']['em_96'])} "
                         f"({name}: loss {o['loss']:.3f}, EM96 {o['em_96']:.3f}, cap {o['truncated_96']:.3f})")
        lines.append(f"- Probe losses after training (math / code / qa): S3 student {r['probe_losses']['s3_student']}; control {r['probe_losses']['control']}")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def render():
    """Appendix table: the two students' readouts on the fresh items and their paired difference."""
    from analysis.paper_table_layout import house_style, table_layout
    summary = p1.read(OUT / "summary.json")
    rows = []
    def ci(d): return f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
    WORDS = {"pythia-1.4b--step120000": "Pythia 1.4B reference, student Pythia 410M", "gemma3-1b": "Gemma 3 1B reference, student Gemma 3 270M",
             "pythia-410m--step120000": "Pythia 410M reference, student Pythia 160M", "gemma3-4b": "Gemma 3 4B reference, student Gemma 3 1B"}
    refs = list(summary["references"])
    for ref in refs:
        r = summary["references"][ref]; c, s, d = r["control"], r["s3_student"], r["paired_control_minus_s3"]
        if len(refs) > 1:
            rows.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{WORDS.get(ref, ref)}}}}} \\\\")
        rows += [f"Reference-completion loss (nats per token) & {s['loss']:.3f} & {c['loss']:.3f} & {ci(d['loss'])} \\\\",
                 f"Exact match, 32 new tokens & {s['em_32']:.3f} & {c['em_32']:.3f} & {ci(d['em_32'])} \\\\",
                 f"Exact match, 96 new tokens & {s['em_96']:.3f} & {c['em_96']:.3f} & {ci(d['em_96'])} \\\\",
                 f"Token F1, 96 new tokens & {s['f1_96']:.3f} & {c['f1_96']:.3f} & {ci(d['f1_96'])} \\\\",
                 f"Share reaching the 96-token cap & {s['truncated_96']:.3f} & {c['truncated_96']:.3f} & {ci(d['truncated_96'])} \\\\",
                 f"Generated tokens, mean & {s['tokens_96']:.1f} & {c['tokens_96']:.1f} & {ci(d['tokens_96'])} \\\\"]
        if len(refs) > 1 and ref != refs[-1]:
            rows.append(r"\midrule")
    who = (f"student of the {WORDS.get(refs[0], refs[0])}" if len(refs) == 1
           else "students of the " + " and ".join(WORDS.get(x, x).split(",")[0].replace(" reference", "") for x in refs) + " references")
    caption = (f"End-marker control on the 384 fresh question-answering items for the selection-round {who}: the original "
               "student against a student retrained on the same 1,800 teacher answers, recipe, seed and schedule with the tokenizer's "
               "end-of-sequence token appended to every target and supervised, 1,781 extra supervised tokens per epoch. Differences "
               "are control minus original, with 95 percent intervals from a paired item bootstrap; the loss is token-weighted as in Eq. 1.")
    text = ("% Generated by analysis/p3_eos_control.py --table; do not edit.\n"
            "\\begin{table}[tb]\n\\centering\\footnotesize\n"
            f"\\caption{{{caption}}}\n\\label{{tab:eos-control}}\n"
            "\\begin{tabular*}{\\textwidth}{lrrr}\n\\toprule\n"
            "Readout & Original student & Control with end marker & Difference [95\\%] \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    target = ROOT / "paper/paper/tables/eos_control.tex"
    target.write_text(house_style(table_layout(text))); print(target.read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evaluate", action="store_true"); ap.add_argument("--reference", default="pythia-1.4b--step120000")
    ap.add_argument("--analyse", action="store_true"); ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.evaluate:
        evaluate(a.reference)
    if a.analyse:
        analyse()
    if a.table:
        render()
