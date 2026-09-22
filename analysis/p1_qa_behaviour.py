#!/usr/bin/env python3
"""P1: behavioural and measurement check of the selected models on fresh question-answering items.

    python -B analysis/p1_qa_behaviour.py --plan                       # CPU: items and roster
    CUDA_VISIBLE_DEVICES=GPU-<uuid> python -B analysis/p1_qa_behaviour.py --run   # GPU

For each S3 reference, the models that the selection policies actually chose on question
answering (the distilled student, the quantization-only choice, the quantization oracle, any
other all-method oracle) plus the pristine student and the dense reference are evaluated on
N fresh 2WikiMultihopQA validation items that no earlier probe, sample, teacher trace or
few-shot exemplar used. Per item and model: the completion loss of the reference answer
(the paper's endpoint) and a greedy 4-shot generation scored by exact match and token F1
(V15 protocol). Paired per-item differences against the dense reference and against the
quantization-only choice carry percentile bootstrap intervals. Compression states are
rebuilt with the S3 recipe (V6 magnitude pruning at the S3 thresholds, V10/V54 rounding);
students load the S3 LoRA adapters. Nothing here changes any S3 file. Retrospective.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

S3 = ROOT / "results/s3-selection-validation"
OUT = ROOT / "results/p1-qa-behaviour"
N_ITEMS = 384
SEED = 2026
DATASET = "framolfese/2WikiMultihopQA"
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
SNAPSHOTS = {  # the S3 revisions, resolved on this host's cache
    "google/gemma-3-1b-pt": "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29",
    "google/gemma-3-4b-pt": "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
    "google/gemma-3-270m": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
    "EleutherAI/pythia-1.4b": "4e056d657ceede13435146a8126bbf7bda0a1b0e",
    "EleutherAI/pythia-410m": "ade882a342507b4be34455393ba750fee7a513b8",
    "EleutherAI/pythia-160m": "3df7d2c5532755f7022704e07dea87f35eeaf0e1",
}
REFERENCE_MODEL = {"pythia-410m--step120000": "EleutherAI/pythia-410m", "pythia-1.4b--step120000": "EleutherAI/pythia-1.4b",
                   "gemma3-1b": "google/gemma-3-1b-pt", "gemma3-4b": "google/gemma-3-4b-pt"}


def read(path):
    return json.loads(Path(path).read_text())


def snapshot_dir(model):
    d = HF_HOME / "hub" / ("models--" + model.replace("/", "--")) / "snapshots" / SNAPSHOTS[model]
    if not d.is_dir():
        raise FileNotFoundError(d)
    return d


def fresh_items():
    """Sample fresh validation items, excluding every index or question used before."""
    from analysis import v71_qa_scope as v71
    from analysis.v34_c17_scope_audit import normalize
    cache = HF_HOME / "datasets"
    qa, provenance = v71.cached_dataset(cache, DATASET, None, "validation")
    register = read(ROOT / "results/v71-qa-scope/register.json")["sets"]["2wiki_new"]
    excluded = set(register["indices"]) | set(register["excluded_indices"])
    questions, sources = v71.teacher_questions(ROOT / "results/traces-pilot")
    from analysis import v15_accuracy_link as v15
    exemplar_questions = {normalize(e["prompt"].split("Question: ")[-1].split("\nAnswer")[0]) for e in v15.FEWSHOT["qa"]["exemplars"]}
    indices = v71.fresh_indices(len(qa), excluded, n=len(qa) - len(excluded), seed=SEED)
    items = []
    for i in indices:
        row = qa[i]
        q = normalize(row["question"])
        if q in questions or q in exemplar_questions:
            continue
        probe = v71.two_wiki_probe(row)
        items.append({"index": int(i), "id": row.get("_id", row.get("id", str(i))), **probe, "question": row["question"]})
        if len(items) == N_ITEMS:
            break
    if len(items) != N_ITEMS:
        raise ValueError("Not enough fresh items")
    return items, {"dataset": DATASET, "split": "validation", "seed": SEED, "n": N_ITEMS, "source": provenance,
                   "excluded_indices": len(excluded), "excluded_teacher_questions": len(questions),
                   "excluded_exemplar_questions": len(exemplar_questions), "trace_sources": sources}


def roster(plan, ablation):
    """Distinct models each policy chose on question answering, per reference, plus controls."""
    chosen = {}
    for row in ablation["rows"]:
        if row["objective"] != "qa" or not row["paired"]:
            continue
        r = chosen.setdefault(row["reference"], {})
        for name in ("rule", "priority", "quant-only"):
            r.setdefault(row["policies"][name]["selected"], set()).add(name)
        r.setdefault(row["oracle_id"], set()).add("oracle")
        r.setdefault(row["quant_oracle_id"], set()).add("quant_oracle")
    out = []
    for ref in plan["references"]:
        cands = {q["id"]: q for q in ref["candidates"]}
        ids = dict(chosen[ref["id"]])
        ids.setdefault("dense:source", set()).add("dense")
        ids["pristine:student"] = {"pristine_student"}
        for cid, roles in ids.items():
            q = cands.get(cid, {"id": cid, "method": "pristine", "arm": "pristine", "r": ref["student"]["N0"] / ref["N0"]})
            out.append({"reference": ref["id"], "candidate": cid, "roles": sorted(roles), "method": q["method"],
                        "arm": q.get("arm"), "d": q.get("d"), "bit": q.get("bit"), "group_size": q.get("group_size"),
                        "r": q["r"], "family": ref["family"]})
    return out


def make_plan():
    plan = read(S3 / "plan_v2.json")
    ablation = read(ROOT / "results/s3-policy-ablation/summary.json")
    items, meta = fresh_items()
    models = roster(plan, ablation)
    OUT.mkdir(exist_ok=True)
    p1 = {"status": "PLANNED", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "items_meta": meta,
          "protocol": {"loss": "V6 completion loss of the reference answer, prompt masked, max_len 1024, bf16",
                       "generation": "V15 QA protocol: 4 fixed 2Wiki-train exemplars, greedy, 32 new tokens, "
                                     "stop on next Question/Context, exact match and token F1 after V15 normalisation",
                       "states": "S3 recipe: V6 global magnitude pruning at the reference's S3 thresholds, V10 per-channel "
                                 "or V54 grouped round-to-nearest; students are the S3 LoRA adapters on the pristine student",
                       "pairing": "per-item differences against the dense reference and against the quantization-only "
                                  "choice, 5,000-resample percentile bootstrap over items, per reference"},
          "reading": {"loss and behaviour improve": "evidence of deployment value", "loss improves, behaviour flat":
                      "the prediction optimises answer likelihood; no behavioural claim", "behaviour worsens":
                      "the loss endpoint and the behavioural endpoint disagree; application wording must change"},
          "models": models, "items": items}
    (OUT / "plan.json").write_text(json.dumps(p1, indent=1))
    print(f"{len(items)} items; {len(models)} model evaluations")
    for m in models:
        print(f"  {m['reference']:26s} {m['candidate']:20s} {','.join(m['roles'])}")


def build_state(ref_id, spec, plan_ref):
    """Load the base for this candidate and apply the S3 recipe; returns (model, tokenizer, transform)."""
    import numpy as np
    import torch
    from analysis import v6_capability_geometry as v6, v10_quantization as v10, v54_quant_group as v54
    from analysis.v12_distill import load_text_causal_lm
    student = spec["method"] in ("distill", "pristine")
    base_name = plan_ref["student"]["model"] if student else REFERENCE_MODEL[ref_id]
    model, tok = load_text_causal_lm(str(snapshot_dir(base_name)), torch.bfloat16)
    model.to("cuda:0").eval().requires_grad_(False)
    transform = {"base": base_name}
    if spec["method"] == "distill":
        from peft import PeftModel
        adapters = list((ROOT / plan_ref["student"]["output_dir"]).glob("v12/*/*/adapter"))
        if len(adapters) != 1 or not (adapters[0] / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Expected one S3 adapter with weights for {ref_id}, found {adapters}")
        model = PeftModel.from_pretrained(model, str(adapters[0]), local_files_only=True, is_trainable=False).eval()
        transform["adapter"] = str(adapters[0].relative_to(ROOT))
    elif spec["method"] in ("prune", "quant"):
        params = v6.language_weight_parameters(model)
        weights = [p.detach().cpu().clone() for _, p in params]
        if spec["method"] == "prune":
            sample = v6._sample_abs_weights(params)
            threshold = float(np.quantile(sample, 1 - spec["d"]))
            v6.apply_global_magnitude_pruning(model, spec["d"], reference_weights=weights, threshold=threshold)
            transform["prune_threshold"] = threshold
        elif spec["arm"] == "per-channel":
            v10._apply_fake_quantization(params, weights, spec["bit"])
        else:
            v54._apply_grouped_quantization(params, weights, spec["bit"], spec["group_size"])
        del weights
    return model, tok, transform


def evaluate_model(model, tok, items):
    import torch
    from analysis.v6_capability_geometry import completion_loss
    from analysis import v15_accuracy_link as v15
    losses = []
    with torch.no_grad():
        for it in items:
            loss, tokens = completion_loss(model, tok, it["prompt"], it["completion"], "cuda:0", max_len=1024)
            losses.append({"loss_sum": float(loss), "tokens": int(tokens), "loss": float(loss) / max(int(tokens), 1)})
    prompts = [v15.build_accuracy_prompt("qa", it, 4) for it in items]
    generations = []
    for start in range(0, len(prompts), 8):
        generations.extend(v15.greedy_generate_batch(model, tok, prompts[start:start + 8], device="cuda:0",
                           max_new_tokens=v15.MAX_NEW_TOKENS["qa"], input_max_length=1024 - v15.MAX_NEW_TOKENS["qa"],
                           truncation_side="left", stop_sequences=v15.DOMAIN_STOP_SEQUENCES["qa"]))
    scored = []
    for it, raw in zip(items, generations):
        gen = v15.truncate_generation("qa", raw)
        scored.append({"generation": gen, "exact_match": bool(v15.qa_exact_match(gen, it["answer"])),
                       "token_f1": float(v15.qa_token_f1(gen, it["answer"]))})
    return [{**l, **g} for l, g in zip(losses, scored)]


def run():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("P1 --run needs a GPU")
    p1 = read(OUT / "plan.json")
    plan = read(S3 / "plan_v2.json")
    refs = {r["id"]: r for r in plan["references"]}
    done = read(OUT / "results.json") if (OUT / "results.json").exists() else {"evaluations": {}}
    # Compression states first, students last: the students need the fetched adapters.
    for spec in sorted(p1["models"], key=lambda m: m["method"] == "distill"):
        key = f"{spec['reference']}|{spec['candidate']}"
        if key in done["evaluations"]:
            continue
        t0 = time.time()
        model = tok = None
        try:
            model, tok, transform = build_state(spec["reference"], spec, refs[spec["reference"]])
            records = evaluate_model(model, tok, p1["items"])
        finally:
            del model, tok
            gc.collect(); torch.cuda.empty_cache()
        total_tokens = sum(r["tokens"] for r in records)
        done["evaluations"][key] = {
            "reference": spec["reference"], "candidate": spec["candidate"], "roles": spec["roles"], "transform": transform,
            "mean_loss": sum(r["loss_sum"] for r in records) / total_tokens, "exact_match": statistics.mean(r["exact_match"] for r in records),
            "token_f1": statistics.mean(r["token_f1"] for r in records), "seconds": round(time.time() - t0, 1), "items": records}
        (OUT / "results.json").write_text(json.dumps(done, indent=1))
        e = done["evaluations"][key]
        print(json.dumps({k: e[k] for k in ("reference", "candidate", "roles", "mean_loss", "exact_match", "token_f1", "seconds")}), flush=True)
    done["status"] = "COMPLETE" if len(done["evaluations"]) == len(p1["models"]) else "PARTIAL"
    (OUT / "results.json").write_text(json.dumps(done, indent=1))


def bootstrap_mean(diffs, seed, resamples=5000):
    import numpy as np
    rng = np.random.default_rng(seed)
    x = np.asarray(diffs, dtype=float)
    if len(x) == 0:
        return {"mean": None, "ci95": None}
    idx = rng.integers(0, len(x), size=(resamples, len(x)))
    means = x[idx].mean(axis=1)
    return {"mean": float(x.mean()), "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))]}


def lenient_answer(generation):
    """Post-hoc readout: drop markdown emphasis and keep the first sentence of the generation."""
    text = generation.replace("**", "").replace("__", "").strip()
    for stop in (". ", ".\n", "\n"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip().rstrip(".")


def analyse():
    """Paired per-item differences against the dense reference and the quantization-only choice."""
    from analysis import v15_accuracy_link as v15
    p1 = read(OUT / "plan.json"); res = read(OUT / "results.json")
    ev = res["evaluations"]
    answers = [it["answer"] for it in p1["items"]]
    for v in ev.values():
        for it, answer in zip(v["items"], answers):
            it["lenient_exact_match"] = bool(v15.qa_exact_match(lenient_answer(it["generation"]), answer))
        v["lenient_exact_match"] = sum(it["lenient_exact_match"] for it in v["items"]) / len(v["items"])
    metrics = ("loss", "exact_match", "token_f1")
    summary = {"status": res.get("status"), "n_items": len(p1["items"]), "references": {}}
    for ref in sorted({m["reference"] for m in p1["models"]}):
        keys = {k: v for k, v in ev.items() if v["reference"] == ref}
        dense = next((v for v in keys.values() if "dense" in v["roles"]), None)
        quant = next((v for v in keys.values() if "quant-only" in v["roles"]), None)
        pristine = next((v for v in keys.values() if "pristine_student" in v["roles"]), None)
        entry = {"models": {}}
        for key, v in keys.items():
            row = {"candidate": v["candidate"], "roles": v["roles"], "mean_loss": v["mean_loss"],
                   "exact_match": v["exact_match"], "token_f1": v["token_f1"],
                   "lenient_exact_match": v["lenient_exact_match"]}
            for name, base in (("vs_dense", dense), ("vs_quant_only", quant),
                               ("vs_pristine", pristine if v["candidate"].startswith("distill") else None)):
                if base is None or base is v:
                    continue
                paired = {}
                for metric in metrics:
                    a = [it[metric] for it in v["items"]]; b = [it[metric] for it in base["items"]]
                    paired[metric] = bootstrap_mean([x - y for x, y in zip(a, b)], seed=zlib.crc32(f"{ref}|{key}|{metric}".encode()))
                row[name] = paired
            entry["models"][v["candidate"]] = row
        summary["references"][ref] = entry
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["# P1 QA behaviour on fresh 2Wiki items (retrospective)", "",
             f"{summary['n_items']} fresh items; per model: mean completion loss (nats/token), exact match, token F1; "
             "paired differences against the dense reference, the quantization-only choice and, for the distilled student, its pristine initialisation, with 95% bootstrap intervals. "
             "Lenient EM (post-hoc readout) strips markdown emphasis and keeps the first sentence before the same normalisation.", ""]
    for ref, e in summary["references"].items():
        lines += [f"## {ref}", "", "| Model | Roles | Loss | EM | Lenient EM | F1 | dLoss vs dense [95%] | dEM vs dense | dF1 vs dense | dLoss vs quant-only | dEM vs quant-only | dF1 vs quant-only | dLoss vs pristine | dEM vs pristine | dF1 vs pristine |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        def cell(d, m):
            if not d or m not in d or d[m]["mean"] is None: return "--"
            return f"{d[m]['mean']:+.3f} [{d[m]['ci95'][0]:+.3f}, {d[m]['ci95'][1]:+.3f}]"
        for cid, r in e["models"].items():
            lines.append("| " + " | ".join([cid, ",".join(r["roles"]), f"{r['mean_loss']:.3f}", f"{r['exact_match']:.3f}", f"{r['lenient_exact_match']:.3f}", f"{r['token_f1']:.3f}",
                         cell(r.get("vs_dense"), "loss"), cell(r.get("vs_dense"), "exact_match"), cell(r.get("vs_dense"), "token_f1"),
                         cell(r.get("vs_quant_only"), "loss"), cell(r.get("vs_quant_only"), "exact_match"), cell(r.get("vs_quant_only"), "token_f1"),
                         cell(r.get("vs_pristine"), "loss"), cell(r.get("vs_pristine"), "exact_match"), cell(r.get("vs_pristine"), "token_f1")]) + " |")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


NAMES = {"pythia-410m--step120000": "Pythia 410M, step 120000", "pythia-1.4b--step120000": "Pythia 1.4B, step 120000",
         "gemma3-1b": "Gemma 3, 1B", "gemma3-4b": "Gemma 3, 4B"}
ROLE_WORDS = {"rule": "rule", "quant-only": "quantization only", "oracle": "oracle", "quant_oracle": "quantization oracle"}
TABLE = ROOT / "paper/paper/tables/p1_behaviour.tex"


def render_table(summary):
    """Appendix table: per reference and model, fresh-item loss, exact match, token F1, and paired losses."""
    from analysis.paper_table_layout import house_style, table_layout
    fmt = lambda x: f"{x:.3f}"
    def ci(d, m):
        if not d or m not in d or d[m]["mean"] is None:
            return "--"
        return f"{d[m]['mean']:+.2f} [{d[m]['ci95'][0]:+.2f}, {d[m]['ci95'][1]:+.2f}]"
    rows = []
    refs = list(NAMES)
    for i, ref in enumerate(refs):
        models = summary["references"][ref]["models"]
        order = sorted(models, key=lambda c: (c.startswith("dense"), c.startswith("pristine"), c.startswith("distill"), c))
        for j, cid in enumerate(order):
            m = models[cid]
            name = {"dense:source": "dense reference", "pristine:student": "initial student", "distill:new_s3": "distilled student"}.get(
                cid, cid.replace("quant:channel_b", "per-channel ").replace("quant:b", "grouped ").replace("_g", " bits, group ").replace("prune:d", "pruned to density "))
            if name.startswith("per-channel "):
                name += " bits"
            roles = ", ".join(ROLE_WORDS[r] for r in ("rule", "quant-only", "oracle", "quant_oracle") if r in m["roles"])
            rows.append(" & ".join([NAMES[ref] if j == 0 else "", name + (f" ({roles})" if roles else ""), fmt(m["mean_loss"]),
                                    fmt(m["exact_match"]), fmt(m["token_f1"]), ci(m.get("vs_quant_only"), "loss"),
                                    ci(m.get("vs_pristine"), "loss")]) + r" \\")
        if i < len(refs) - 1:
            rows.append(r"\midrule")
    caption = (
        "Fresh-item check, after the selection round, of the models the policies chose for question answering, on "
        "384 2WikiMultihopQA validation items unused by any earlier probe, sample, teacher trace or few-shot "
        "exemplar: completion loss of the reference answer in nats per native token, greedy 4-shot exact match and "
        "token F1 with 32 new tokens, and paired loss differences with 95\\% bootstrap intervals over items "
        "against the quantization-only choice and, for the distilled student, its initial student; negative "
        "favours the row. Parentheses name the policies that chose the model.")
    text = ("% Generated by analysis/p1_qa_behaviour.py --table; retrospective, do not edit.\n"
            "\\begin{table}[tb]\n\\centering\\footnotesize\n"
            f"\\caption{{{caption}}}\n\\label{{tab:p1-behaviour}}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrrrr@{}}\n\\toprule\n"
            "Reference & Model (policies) & Loss & Exact match & Token F1 & Loss minus quantization only & "
            "Loss minus initial student \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    return house_style(table_layout(text))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyse", action="store_true"); ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.plan:
        make_plan()
    if a.run:
        run()
    if a.analyse:
        analyse()
    if a.table:
        TABLE.write_text(render_table(read(OUT / "summary.json")))
        print(f"wrote {TABLE}")
