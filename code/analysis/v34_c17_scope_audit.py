#!/usr/bin/env python3
"""V34: offline, CPU-only C17 provenance audit. Never load a model or dataset API.

Recompute the old QA split from saved scores; compare saved questions/prompts;
inspect current source via AST, not imports. Optional --hotpot-cache reads existing
Arrow streams with pyarrow (no datasets loader, downloads, or cache writes).
Missing identity evidence means unknown, never an empty/clean intersection.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
PRIOR = Path("prior/xueqi-intern-backup")
RUNS = PRIOR / "docs/scaling-down/results/runs"
PILOT = PRIOR / "scripts/empirical-study/scaling-down/pilot"
SWEEP = PRIOR / "scripts/empirical-study/scaling-down/sweep"
RECIPES = {"distill": "M0", "answer_only": "M1", "soft_kd": "M2", "onpolicy": "M3"}
FN = re.compile(r"^(distill|answer_only|soft_kd|onpolicy)(?:-(hotpotqa|2wiki))?-(\d+p?\d*b)-s(\d)\.json$")
QUESTION = re.compile(r"(?:^|\n)Question:\s*([^\n]+)")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(value: str) -> str:
    """Conservative exact-text matching: NFKC, casefold, collapsed whitespace.

    Keep punctuation, numbers, and articles. This is not fuzzy/semantic dedup.
    """
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def question(row: dict) -> str | None:
    if isinstance(row.get("question"), str) and row["question"].strip():
        return row["question"]
    # Last marker avoids accidentally using a Question: inside supplied context.
    prompt = row.get("prompt", "")
    if not prompt:
        prompt = next((m.get("content", "") for m in reversed(row.get("messages", []))
                       if m.get("role") == "user"), "")
    matches = QUESTION.findall(prompt)
    return matches[-1].strip() if matches else None


def context_lines(row: dict) -> list[str]:
    prompt = row.get("prompt", "")
    if not prompt.startswith("Context:\n") or "\n\nQuestion:" not in prompt:
        return []
    return [line for line in prompt.split("\n\nQuestion:", 1)[0][9:].splitlines()
            if ": " in line]


def titles(row: dict) -> set[str]:
    # Rendered title prefixes, not canonical Wikipedia IDs. Colons in titles and
    # 4000-character truncation limit this diagnostic; do not call it leakage.
    return {normalize(line.split(": ", 1)[0]) for line in context_lines(row)}


def compare_questions(left: list[dict], right: list[dict]) -> dict:
    sets = [{normalize(q) for r in rows if (q := question(r))} for rows in (left, right)]
    missing = [sum(question(r) is None for r in rows) for rows in (left, right)]
    shared = sets[0] & sets[1]
    complete = bool(left and right) and not any(missing)
    return {"status": "overlap" if shared else ("no_exact_question_overlap" if complete else "unknown"),
            "left_rows": len(left), "right_rows": len(right),
            "left_unique_questions": len(sets[0]), "right_unique_questions": len(sets[1]),
            "missing_questions": missing, "comparison_complete": complete,
            "overlap_count": len(shared) if complete or shared else None,
            "matched_question_sha256": [digest(q.encode()) for q in sorted(shared)],
            "scope": "available question text only; no semantic or pretraining-cleanliness assertion"}


def referenced_paths(value) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(set([v]) if k in ("path", "source_path") and isinstance(v, str)
                             else referenced_paths(v) for k, v in value.items()))
    if isinstance(value, list):
        return set().union(*(referenced_paths(v) for v in value))
    return set()


class Evidence:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.hashes: dict[str, str] = {}

    def label(self, path: Path) -> str:
        path = path.resolve()
        return str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)

    def read(self, path: str | Path) -> str:
        path = self.root / path
        data = path.read_bytes()
        self.hashes[self.label(path)] = digest(data)
        return data.decode("utf-8")

    def json(self, path: str | Path):
        return json.loads(self.read(path))

    def hash_file(self, path: Path) -> None:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        self.hashes[self.label(path)] = h.hexdigest()

    def verify_unchanged(self) -> None:
        before = dict(self.hashes)
        for path in before:
            self.hash_file(self.root / path)
        if before != self.hashes:
            raise RuntimeError("An input changed during the scope audit")


def source_protocol(e: Evidence) -> dict:
    v12 = ast.parse(e.read("analysis/v12_distill.py"))
    constants = {}
    for node in v12.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("TRAINING_BENCHMARKS", "MEASUREMENT_BENCHMARKS", "DEFAULT_N_PROBE", "SEED"):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    v6 = ast.parse(e.read("analysis/v6_capability_geometry.py"))
    builder = next(n for n in v6.body if isinstance(n, ast.FunctionDef) and n.name == "build_probes")
    loads = []
    for node in sorted(ast.walk(builder), key=lambda n: getattr(n, "lineno", 0)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_dataset":
            if node.args and isinstance(node.args[0], ast.Constant):
                loads.append({"dataset": ast.literal_eval(node.args[0]),
                              "config": ast.literal_eval(node.args[1]) if len(node.args) > 1 else None,
                              "split": next(ast.literal_eval(k.value) for k in node.keywords if k.arg == "split"),
                              "line": node.lineno})
    # Inspect executable probe assignments, not docstrings mentioning v[1::2].
    assignments = [n for n in ast.walk(v12) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in ("all_probes", "probes") for t in n.targets)]
    slices = [ast.unparse(n.slice) for a in assignments for n in ast.walk(a)
              if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice)]
    expected_design = (
        constants.get("TRAINING_BENCHMARKS") == {"math": "GSM8K", "code": "CodeAlpaca", "qa": "HotpotQA"}
        and constants.get("MEASUREMENT_BENCHMARKS") == {"math": "MATH-500", "code": "MBPP", "qa": "2WikiMultihopQA"}
        and [(r["dataset"], r["split"]) for r in loads] == [
            ("HuggingFaceH4/MATH-500", "test"), ("google-research-datasets/mbpp", "test"),
            ("framolfese/2WikiMultihopQA", "validation")]
        and constants.get("DEFAULT_N_PROBE") == 128 and constants.get("SEED") == 0
        and any(ast.unparse(a) == "all_probes = build_probes(DEFAULT_N_PROBE, seed=SEED)" for a in assignments))
    return {"constants": constants, "measurement_loaders_in_rng_order": loads,
            "expected_current_design_verified": expected_design,
            "measurement_assignments": [ast.unparse(n) for n in assignments], "measurement_slices": slices,
            "odd_after_sampling_verified": slices == ["1::2"],
            "index_caveat": "Odd positions in sampled probe list, not odd source dataset row IDs. "
                            "Math and code draws advance the shared seed-0 RNG before QA. "
                            "V15 probe_index is renumbered 0..63 after selection; never slice saved records again."}


def old_qa(e: Evidence) -> tuple[dict, list[dict], list[dict]]:
    saved = e.json("results/v3-measurement-audit/split.json")["qa"]
    domains = {}
    sources = {}
    for task, dirs in {
        "hotpotqa": ["hotpotqa-qwen3-2026-07-15", "answer-only-qwen3-2026-07-16",
                      "soft-kd-qwen3-2026-07-19", "onpolicy-qwen3-2026-07-21"],
        "2wiki": ["2wiki-qwen3-2026-07-22"],
    }.items():
        respondents = {}
        for folder in dirs:
            for path in sorted((e.root / RUNS / folder).glob("eval*/*.json")):
                if path.parent.name == "eval-base" and path.name.startswith("base-"):
                    key = ("base", path.stem.split("-")[1])
                elif path.parent.name == "eval" and (m := FN.match(path.name)):
                    if m[2] not in (None, task):
                        continue
                    key = (RECIPES[m[1]], m[3], int(m[4]))
                else:
                    continue
                payload = e.json(path)
                items = payload["items"]
                if len(items) != 50:
                    raise ValueError(f"Unexpected old QA panel length: {path}")
                if key in respondents:
                    raise ValueError(f"Duplicate old respondent: {task}, {key}")
                respondents[key] = items
                sources[f"{task}|{'|'.join(map(str, key))}"] = e.label(path)
        if not respondents:
            raise ValueError(f"Missing old {task} raw scores; cannot certify C17 provenance")
        domains[task] = respondents
    tr, tf = domains["hotpotqa"], domains["2wiki"]
    mean = lambda rows: statistics.mean(float(r.get("score") or 0) >= .5 for r in rows)
    recomputed = {}
    for rec in RECIPES.values():
        pairs = [(mean(tr[k]) - mean(tr[("base", k[1])]),
                  mean(tf[k]) - mean(tf[("base", k[1])]))
                 for k in sorted(tr) if k[0] == rec and k in tf
                 and ("base", k[1]) in tr and ("base", k[1]) in tf]
        if not pairs:
            raise ValueError(f"Missing old paired scores: {rec}")
        a, b = map(statistics.mean, zip(*pairs))
        row = {"n": len(pairs), "d_trained": a, "d_transfer": b, "gap": a-b,
               "ci90_saved_not_refit": saved[rec]["ci90"]}
        row["matches_saved"] = all(abs(row[k] - saved[rec][k]) < 1e-12 for k in ("n", "d_trained", "d_transfer", "gap"))
        if not row["matches_saved"]:
            raise ValueError(f"Old QA recomputation disagrees with saved V3: {rec}")
        recomputed[rec] = row
    hp_rows = next(iter(tr.values()))
    wiki_rows = next(iter(tf.values()))
    hp_questions = {normalize(question(r)) for r in hp_rows}
    wiki_ids = {r["task_id"] for r in wiki_rows}
    if any({normalize(question(r)) for r in rows} != hp_questions for rows in tr.values()):
        raise ValueError("Old HotpotQA panel changes across respondents")
    if any({r["task_id"] for r in rows} != wiki_ids for rows in tf.values()):
        raise ValueError("Old 2Wiki panel changes across respondents")
    m2 = recomputed["M2"]
    pooled = (m2["d_trained"] + m2["d_transfer"]) / 2
    return {"campaign": "8B-teacher Qwen3 M0/M1/M2/M3; 0.6/1.7/4/8B; seeds 1/2/3",
            "respondents_per_benchmark": {k: len(v) for k, v in domains.items()},
            "items_per_benchmark": 50, "hotpotqa_local_indices": sorted(r["i"] for r in hp_rows),
            "old_2wiki_task_ids": sorted(wiki_ids), "qa_split": recomputed,
            "M2_gap_divided_by_hotpot_gain": m2["gap"] / m2["d_trained"],
            "M2_equal_weight_panel_gain": pooled,
            "M2_panel_excess_over_transfer_fraction": (pooled-m2["d_transfer"]) / pooled,
            "mechanism": "same-benchmark distribution/format fitting plus correctness-filtered teacher traces; "
                         "not an identified leaked-item fraction or causal attribution estimate",
            "raw_sources": sources}, hp_rows, wiki_rows


def current_artifacts(e: Evidence) -> tuple[dict, list[dict], list[dict]]:
    traces, trace_files = [], []
    for p in sorted((e.root / "results/traces-pilot").glob("*_qa.jsonl")):
        rows = [json.loads(line) for line in e.read(p).splitlines() if line.strip()]
        traces.extend(rows)
        trace_files.append({"path": e.label(p), "rows": len(rows),
                            "unique_questions": len({normalize(question(r)) for r in rows if question(r)}),
                            "stored_fields": sorted({key for row in rows for key in row})})
    inventory = {}
    for name, pattern in (("v12", "results/v12-distill/**/eval.json"),
                          ("v16", "results/v16-style-residual/**/residual.json")):
        records = []
        for p in sorted(e.root.glob(pattern)):
            d = e.json(p)
            records.append({"path": e.label(p), **{k: d.get(k) for k in (
                "student", "teacher", "recipe", "n_per_domain", "domains", "training_benchmarks",
                "measurement_benchmarks", "probe_half", "n_probe_requested", "measurement_samples",
                "probe_source", "probe_seed", "seed", "data_selection")}})
        inventory[name] = records
    default_panels, easy_panels, measurement = [], [], []
    reference = None
    for p in sorted(e.root.glob("results/v15-accuracy/**/accuracy.json")):
        d = e.json(p)
        rows = [r for r in d.get("records", []) if r.get("domain") == "qa"]
        benchmark = d.get("accuracy_benchmark", "default")
        info = {"path": e.label(p), "qa_records": len(rows), "adapter": d.get("adapter"),
                "accuracy_benchmark": benchmark, "probe_half": d.get("probe_half")}
        if benchmark != "default" and (not isinstance(benchmark, dict) or benchmark.get("qa") != "2WikiMultihopQA"):
            easy_panels.append(info)
            continue
        signature = [(r.get("probe_index"), r.get("prompt"), r.get("reference_completion")) for r in rows]
        if reference is None:
            reference, measurement = signature, rows
        info["same_ordered_panel_as_first"] = signature == reference
        default_panels.append(info)
    conflicts = []
    expected = source_protocol(e)["constants"]["MEASUREMENT_BENCHMARKS"]
    for name, records in inventory.items():
        for row in records:
            if (row["measurement_benchmarks"] != expected or "v[1::2]" not in (row["probe_half"] or "")
                    or row["n_probe_requested"] != 128 or row["measurement_samples"] != {"math": 64, "code": 64, "qa": 64}
                    or row["probe_source"] != "analysis.v6_capability_geometry.build_probes"):
                conflicts.append(row["path"])
    return {"trace_files": trace_files, "run_inventory": inventory,
            "default_accuracy_panels": default_panels, "other_accuracy_panels": easy_panels,
            "metadata_conflicts": conflicts,
            "historical_binding": "V12/V16 record benchmark/seed/half/count, not immutable item IDs or "
                                  "ordered prompt hashes; V15 saved default prompts proxy that protocol. "
                                  "Identical counts/metadata do not prove every historical run used these bytes."}, traces, measurement


def cache_matches(e: Evidence, cache: Path | None, groups: dict[str, list[dict]]) -> dict:
    """Resolve question identities against existing HotpotQA cache, read only.

    Cache agreement is retrospective identification, not a historical generator
    manifest. Dataset-local integers are never compared across datasets/splits.
    """
    if cache is None:
        return {"status": "not_supplied", "limitation": "HotpotQA official split/source IDs unresolved without cached source rows"}
    paths = sorted(cache.rglob("*.arrow"))
    if not paths:
        return {"status": "missing", "path": str(cache), "limitation": "No existing Arrow streams"}
    if len({p.parent for p in paths}) != 1:
        raise ValueError("Select one HotpotQA cache revision; cannot merge dataset versions")
    import pyarrow as pa  # Optional local data reader; no GPU or dataset downloads.

    targets = {name: {normalize(q) for row in rows if (q := question(row))} for name, rows in groups.items()}
    matches = {name: [] for name in groups}
    wanted = set().union(*targets.values())
    files, offsets = [], defaultdict(int)
    for path in paths:
        m = re.fullmatch(r"hotpot_qa-(train|validation)(?:-\d+-of-\d+)?\.arrow", path.name)
        if not m:
            continue
        split = m[1]
        e.hash_file(path)
        rows_read = 0
        with pa.memory_map(str(path), "r") as handle:
            for batch in pa.ipc.open_stream(handle):
                questions = batch.column("question").to_pylist()
                for i, q in enumerate(questions):
                    key = normalize(q)
                    if key not in wanted:
                        continue
                    row = batch.slice(i, 1).to_pylist()[0]
                    for name, qs in targets.items():
                        if key in qs:
                            matches[name].append({"split": split, "row_index": offsets[split] + rows_read + i,
                                                  "source_id": row.get("id", row.get("_id")),
                                                  "question_sha256": digest(key.encode())})
                rows_read += batch.num_rows
        files.append({"path": e.label(path), "split": split, "rows": rows_read})
        offsets[split] += rows_read
    for p in sorted(cache.rglob("dataset_info.json")):
        e.read(p)
    coverage = {name: {"queried_unique_questions": len(qs),
                       "matched_unique_questions": len({r["question_sha256"] for r in matches[name]}),
                       "matches": matches[name]} for name, qs in targets.items()}
    return {"status": "read_existing_cache" if files else "missing", "files": files, "coverage": coverage,
            "limitation": "Cache hashes identify inspected bytes, not the historical loader revision. "
                          "Question matches identify source questions; they do not prove full-prompt identity."}


def verdict(old_train: dict, current_measure: dict, old_measure: dict, metadata_ok: bool) -> dict:
    hit = any(x["status"] == "overlap" for x in (current_measure, old_measure))
    unknown = any(x["status"] == "unknown" for x in (old_train, current_measure, old_measure)) or not metadata_ok
    return {"branch": "b_training_benchmark_reused",
            "status": "measurement_overlap_requires_revalidation" if hit else
                      ("scope_incomplete" if unknown else "scope_closed_with_provenance_limits"),
            "current_training_same_benchmark": True,
            "current_training_exact_old_eval_question_overlap": old_train["overlap_count"],
            "current_measurement_exact_question_overlap": current_measure["overlap_count"],
            "old_hotpot_eval_vs_current_measurement_overlap": old_measure["overlap_count"],
            "unconditional_clean_bill": False, "original_C17_preserved": True,
            "model_runs_performed": 0,
            "excluded_interpretations": [
                "A literal no-current-data-dependence verdict: current distillation still trains on HotpotQA.",
                "C17 proves duplicate-item leakage or says one third of current 2Wiki results are contaminated.",
                "Different benchmark names establish disjoint Wikipedia passages or historical item provenance.",
                "The current default measurement mixes 50 HotpotQA and 50 2Wiki questions like the old panel."
            ],
            "not_established": ["complete original M0/M1/M2/M3 trace lineage (original trace files absent; 150 M0 prompts recoverable through matched gold control)",
                                "every historical V12/V16/C21 measurement panel bound to immutable question hashes",
                                "old 2Wiki IDs versus current 2Wiki IDs (current source IDs absent)",
                                "semantic/paraphrase overlap, full-context overlap, or model pretraining cleanliness"]}


def build_summary(root: Path = ROOT, hotpot_cache: Path | None = None) -> dict:
    e = Evidence(root)
    # Preserve and fingerprint the original evidence. No old result/ledger writer.
    for path in ["notes/prior_inventory.md", "notes/review/irt_audit.md", "notes/review/stats_audit.md",
                 "paper/docs/RESULTS_LEDGER.md", "results/v3-measurement-audit/summary.md",
                 "results/v3-measurement-audit/irt_dif.json", "analysis/v3_measurement_audit.py",
                 "analysis/v1_recipe_strat.py", "analysis/v15_accuracy_link.py", "analysis/v16_style_residual.py",
                 "analysis/v21_distill_law.py", "analysis/v23_loss_validity.py", "analysis/v13_recovery.py",
                 "analysis/v19_links.py", "analysis/v34_c17_scope_audit.py",
                 PILOT / "gen_teacher_traces_hotpotqa.py", PILOT / "hotpotqa_util.py",
                 PILOT / "eval_hotpotqa_direct_qwen3.py", PILOT / "eval_2wiki_direct.py",
                 PILOT / "make_matched_gold_from_m0.py", SWEEP / "scaling_pilot_gold_qwen3.sh",
                 SWEEP / "scaling_pilot_qwen3_hotpotqa.sh"]:
        e.read(path)
    protocol = source_protocol(e)
    old, hp, wiki = old_qa(e)
    locked_path = RUNS / "irt-grid-fill-locked-2026-07-28/qa.json"
    locked = e.json(locked_path)
    expected_ids = {f"hotpotqa:{r['i']}" for r in hp} | {f"2wiki:{r['task_id']}" for r in wiki}
    if set(locked["item_ids"]) != expected_ids:
        raise ValueError("V3 QA panel does not match the locked prior grid")
    cells_path = RUNS / "cells-grid-fill-2026-07-28.csv"
    cells = list(csv.DictReader(e.read(cells_path).splitlines()))
    old["locked_grid_link"] = {"path": str(locked_path), "n_respondents": locked["n_respondents"],
                               "n_items": locked["n_items"], "all_V3_item_ids_match": True,
                               "cell_table": str(cells_path), "cell_table_rows": len(cells)}
    gold_path = RUNS / "gold-qwen3-2026-07-21/traces/gold-hotpotqa.jsonl"
    gold = [json.loads(line) for line in e.read(gold_path).splitlines() if line.strip()]
    old["recovered_M0_training_questions"] = {
        "source": str(gold_path), "rows": len(gold), "local_indices": sorted(r["i"] for r in gold),
        "lineage": "scaling_pilot_gold_qwen3.sh names the M0 keep-correct source; "
                   "make_matched_gold_from_m0.py preserves system/user messages verbatim, replacing assistant targets. "
                   "Derived evidence for M0 question identities, not the original traces or a manifest for all recipes."}
    current, traces, measurement = current_artifacts(e)
    overlaps = {"old_hotpot_eval_vs_current_training": compare_questions(hp, traces),
                "current_training_vs_current_measurement": compare_questions(traces, measurement),
                "old_hotpot_eval_vs_current_measurement": compare_questions(hp, measurement),
                "old_2wiki_vs_current_measurement": compare_questions(wiki, measurement),
                "old_M0_training_vs_current_training": compare_questions(gold, traces),
                "old_M0_training_vs_current_measurement": compare_questions(gold, measurement),
                "old_M0_training_vs_old_hotpot_eval": compare_questions(gold, hp)}
    train_titles = set().union(*(titles(r) for r in traces))
    measure_titles = set().union(*(titles(r) for r in measurement))
    shared_titles = train_titles & measure_titles
    flagged = [r["probe_index"] for r in measurement if titles(r) & shared_titles]
    left_lines = {normalize(line) for r in traces for line in context_lines(r)}
    right_lines = {normalize(line) for r in measurement for line in context_lines(r)}
    current["rendered_context_check"] = {
        "shared_title_prefixes": sorted(shared_titles), "shared_complete_rendered_lines": len(left_lines & right_lines),
        "measurement_indices_with_shared_title_prefix": flagged,
        "limitation": "Title-prefix diagnostic, not canonical page identity; contexts truncated at 4000 characters. "
                      "Zero equal rendered lines does not establish zero source-passage overlap."}
    cache = cache_matches(e, hotpot_cache, {"current_training": traces, "old_hotpot_eval": hp,
                                          "old_M0_training": gold, "current_measurement": measurement})
    claim_paths = ["results/v21-distill-law/summary.json", "results/v31-uxseen/summary.json",
                   "results/v31-uxseen/sameE_summary.json"]
    summaries = {p: e.json(p) for p in claim_paths}
    v21_sources = summaries[claim_paths[0]].get("sources", {})
    dependencies = {}
    for claim_path, payload in summaries.items():
        refs = sorted(referenced_paths(payload))
        missing = []
        for ref in refs:
            if (e.root / ref).is_file():
                e.read(ref)
            else:
                missing.append(ref)
        dependencies[claim_path] = {"declared_raw_paths": refs, "missing_paths": missing,
                                   "old_grid_paths": [p for p in refs if p.startswith("prior/")],
                                   "reference_coverage": "explicit paths" if refs else "no raw-path manifest in summary"}
    metadata_ok = (not current["metadata_conflicts"] and protocol["odd_after_sampling_verified"]
                   and protocol["expected_current_design_verified"]
                   and all(current["run_inventory"].values()) and bool(current["default_accuracy_panels"])
                   and all(p["same_ordered_panel_as_first"] for p in current["default_accuracy_panels"]))
    scope = verdict(overlaps["old_hotpot_eval_vs_current_training"],
                    overlaps["current_training_vs_current_measurement"],
                    overlaps["old_hotpot_eval_vs_current_measurement"], metadata_ok)
    scope["branch_interpretation"] = (
        "Branch b only at benchmark/distribution level. Known old M0/eval questions have no current "
        "training or measurement matches; item-level branch b is not demonstrated. The old mixed-panel "
        "fitting mechanism is not the current default measurement design. Neither an unconditional "
        "all-current-sets-clean branch a nor confirmed current item contamination follows.")
    if overlaps["old_M0_training_vs_current_measurement"]["status"] == "overlap":
        scope["status"] = "measurement_overlap_requires_revalidation"
    elif any(overlaps[k]["status"] == "unknown" for k in (
            "old_M0_training_vs_current_training", "old_M0_training_vs_current_measurement")):
        scope["status"] = "scope_incomplete"
    scope["no_new_model_run_needed_to_preserve_qualified_fixed_panel_observations"] = scope["status"] == "scope_closed_with_provenance_limits"
    claims = [
        {"claim": "C17", "dependence": "old locked grid and V3 8B-teacher mixed panel",
         "disposition": "Preserve original result; add this scope finding. Benchmark fitting, recipe anchor, and IRT findings stay prior-grid scoped."},
        {"claim": "C9b QA / current distillation QA behavior", "dependence": "V12 HotpotQA traces -> default 2Wiki loss; V15 behavioral support mostly easy TriviaQA",
         "disposition": "Retain numeric loss observations. Independently validate any broad QA capability/behavioral-gain claim on the frozen clean set below."},
        {"claim": "C18 QA interpretation", "dependence": "V16 uses V12 adapters, default 2Wiki probes and C4 generic control",
         "disposition": "Math/code residual row is not an old-grid QA estimate. Retain residual observations; QA residual improvement is not behavioral proof. Clean-set validation required for a QA capability interpretation."},
        {"claim": "C21 QA (including V31/V31b)", "dependence": "V12 loss/trajectory outcomes; HotpotQA training benchmark persists; V31 summaries lack per-item records",
         "disposition": "Retain signed loss/data/reuse observations conditional on the measured panel. Require clean-set revalidation before claiming independent QA transfer generality or contamination-free historical provenance."},
        {"claim": "C1 and secondary HotpotQA controls", "dependence": "V9/V23/V26/V27 include HotpotQA for dense/pruned/quantized models",
         "disposition": "Shared benchmark alone does not transfer a distillation-fitting effect to untrained compression controls. No C17-driven revalidation for those controls; audit training provenance if applying them to HotpotQA-trained adapters."},
        {"claim": "C19/C23 behavioral link", "dependence": "V15/V19 few-shot accuracy; easy QA is TriviaQA, including trained adapters",
         "disposition": "No old mixed-panel numeric input. Preserve measured benchmark-specific links; do not relabel TriviaQA as 2Wiki or use this link as proof that every QA transfer panel is clean."},
        {"claim": "C10/C11 trace recovery", "dependence": "trace-trained recovery shares current training pool; C4-only recovery does not",
         "disposition": "Apply the same QA generalization gate to trace-recovery claims; keep numerical recovery observations and C4 controls."},
    ]
    excluded_questions = {normalize(q) for t in traces+hp+gold if (q := question(t))}
    candidate_available = (metadata_ok and overlaps["current_training_vs_current_measurement"]["comparison_complete"]
                           and overlaps["old_hotpot_eval_vs_current_measurement"]["comparison_complete"]
                           and overlaps["old_M0_training_vs_current_measurement"]["comparison_complete"])
    summary = {"schema_version": 1, "audit": "v34-c17-scope", "cpu_only": True, "network_used_by_script": False,
               "normalization": "NFKC + casefold + whitespace collapse; punctuation retained; question-only identity",
               "old_grid": old, "anchor_recipe_evidence": e.json("results/v1-recipe-strat/anchor_by_recipe.json"),
               "current_protocol": protocol, "current_artifacts": current, "overlap_checks": overlaps,
               "hotpot_cache_identification": cache, "claims": claims, "v21_declared_sources": v21_sources,
               "claim_dependency_inventory": dependencies,
               "verdict": scope,
               "minimal_clean_set": {
                   "benchmark": "2WikiMultihopQA validation, fixed default measurement half",
                   "source_artifact": current["default_accuracy_panels"][0]["path"] if measurement else None,
                   "exact_question_clean_indices": [r["probe_index"] for r in measurement
                       if question(r) and normalize(question(r)) not in excluded_questions] if candidate_available else [],
                   "stricter_rendered_title_disjoint_indices": [r["probe_index"] for r in measurement
                       if r["probe_index"] not in flagged and question(r)
                       and normalize(question(r)) not in excluded_questions] if candidate_available else [],
                   "selection_independent_of_outcomes": True,
                   "protocol": "Freeze ordered prompt/reference SHA256 and recover upstream IDs/revision first. "
                               "Use all 64 question-disjoint items for a fixed-panel claim; for a corpus-independence "
                               "claim exclude shared-title items and screen full untruncated passages/near-duplicates. "
                               "Pair each relevant existing dense and distilled/recovery checkpoint on the same retained items; "
                               "keep reference loss and behavioral EM/F1 separate. No new training is indicated by C17 alone. "
                               "Reuse saved per-item outputs if available; aggregate losses cannot be subset-rescored. "
                               "If full provenance cannot be recovered, freeze an independently sourced replacement set; "
                               "do not silently declare this set clean.",
                   "validation_executed": "question/context audit only; no inference or retraining",
                   "candidate_identity_coverage_complete": candidate_available,
                   "ordered_measurement_manifest": [{"probe_index": r["probe_index"],
                       "question_sha256": digest(normalize(question(r)).encode()) if question(r) else None,
                       "prompt_reference_sha256": digest(json.dumps([r["prompt"], r["reference_completion"]], ensure_ascii=False).encode())}
                       for r in measurement]},
               "external_dataset_documentation": [
                   {"url": "https://hotpotqa.github.io/", "finding": "HotpotQA is a separate QA benchmark built using Wikipedia."},
                   {"url": "https://aclanthology.org/2020.coling-main.580/", "finding": "2Wiki combines Wikipedia/Wikidata and explicit reasoning construction; it is not a HotpotQA split."}],
               "input_sha256": e.hashes}
    e.verify_unchanged()
    summary["inputs_mutated"] = False
    return summary


def render_report(s: dict) -> str:
    old, current, cache = s["old_grid"], s["current_artifacts"], s["hotpot_cache_identification"]
    coverage = cache.get("coverage", {})
    cache_lines = []
    for name, data in coverage.items():
        counts = defaultdict(int)
        for row in data["matches"]:
            counts[row["split"]] += 1
        cache_lines.append(f"- {name}: {data['matched_unique_questions']}/{data['queried_unique_questions']} unique questions identified; split matches {dict(counts)}.")
    split_rows = [f"| {r} | {v['n']} | {v['d_trained']:+.6f} | {v['d_transfer']:+.6f} | {v['gap']:+.6f} | {v['ci90_saved_not_refit']} |"
                  for r, v in old["qa_split"].items()]
    overlap_rows = [f"| {name} | {v['left_unique_questions']} / {v['right_unique_questions']} | {v['overlap_count']} | {v['status']} |"
                    for name, v in s["overlap_checks"].items()]
    c = current["rendered_context_check"]
    clean = s["minimal_clean_set"]
    v21_refs = s["claim_dependency_inventory"]["results/v21-distill-law/summary.json"]
    claim_rows = [f"| {v['claim']} | {v['dependence']} | {v['disposition']} |" for v in s["claims"]]
    if s["verdict"]["status"] != "scope_closed_with_provenance_limits":
        return "\n".join(["# C17 scope audit — V34", "",
            f"**{s['verdict']['status']}**. Scope cannot be certified from this artifact snapshot. "
            "Keep C17 and all original results; no model runs performed. Inspect missing identities, "
            "measurement overlap and protocol/metadata conflicts in summary.json before any cleanliness claim.", "",
            "| Comparison | Unique questions left / right | Exact matches | Status |",
            "|---|---:|---:|---|", *overlap_rows, "",
            "| Claim | Data dependence | Disposition |", "|---|---|---|", *claim_rows, "",
            clean["protocol"], ""])
    train_matches = coverage.get("current_training", {}).get("matches", [])
    old_matches = coverage.get("old_hotpot_eval", {}).get("matches", [])
    identified = (len(train_matches) == 600 and {r["split"] for r in train_matches} == {"train"}
                  and {r["row_index"] for r in train_matches} == set(range(600))
                  and len(old_matches) == 50 and {r["split"] for r in old_matches} == {"validation"})
    identification = (
        "The supplied cache identifies current trace questions as official train rows 0–599, and the old "
        "50 HotpotQA evaluation questions in official validation, with no train matches. The local "
        "AFlow filename ‘test’ therefore must not be treated as an official split label. Cache identification "
        "does not recover the historical generator revision or original teacher trace membership."
        if identified else
        "The complete expected official-split mapping is not established by this invocation. "
        "Do not infer official training/validation splits from filenames or benchmark names. "
        "Inspect the cache coverage above; missing cache/source rows leave this provenance unresolved.")
    return "\n".join([
        "# C17 scope audit — V34", "",
        f"**Verdict: {s['verdict']['status']}; branch (b), training-benchmark reuse.** "
        "The observed C17 effect belongs to the old mixed QA panel. Current distillation still uses HotpotQA training data, "
        "so a literal ‘old grid only, no current data dependence’ verdict is excluded. The current default measurement "
        "uses 2Wiki, and the saved questions show no exact overlap with current traces or the old HotpotQA panel. "
        "This is a scope closeout with explicit provenance limits, not a blanket cleanliness certificate. "
        "Keep C17 and all original results. No model runs were performed.", "",
        "## Evidence and mechanism", "",
        "Entry points: [prior inventory](../../notes/prior_inventory.md), "
        "[IRT audit §c/§h](../../notes/review/irt_audit.md), "
        "[V3 source](../../analysis/v3_measurement_audit.py), "
        "[original V3 result](../../results/v3-measurement-audit/summary.md). "
        "The old frozen grid is `irt-grid-fill-locked-2026-07-28/`, with the cell inventory "
        "`cells-grid-fill-2026-07-28.csv`. V3's QA evidence specifically reuses the 8B-teacher campaign: "
        "52 respondents (4 dense + 4 recipes × 4 student sizes × 3 seeds), each evaluated on "
        "50 HotpotQA and 50 2Wiki questions. It is not a measurement of contamination across every prior-grid cell.", "",
        f"The audit also verifies all 100 V3 item IDs against the frozen QA panel ({old['locked_grid_link']['n_respondents']} "
        f"respondents) and fingerprints its {old['locked_grid_link']['cell_table_rows']}-row cell table. "
        "This binds the reanalysis to the old grid instead of relying on directory names.", "",
        "The old sweep `scaling_pilot_qwen3_hotpotqa.sh` reads the local AFlow/MetaGPT "
        "`hotpotqa_test.jsonl`: candidate training slice [0:200], evaluation slice [400:450]. "
        "The generator retains teacher responses with F1≥0.5. ‘Trained half’ means the **evaluation half from the "
        "training benchmark**, not literal training rows. Old eval JSONs preserve questions and local indices 400–449. "
        "Those local integers are not official HotpotQA row IDs; the upstream file and original teacher trace files "
        "for these four arms are absent, so complete training lineage cannot be certified. "
        "The declared slices are disjoint. The observed split is consistent with benchmark/format fitting; "
        "it does not identify duplicate leakage or causally measure the fraction attributable to fitting.", "",
        "A derived artifact does preserve **150 M0 training questions**: "
        "`gold-qwen3-2026-07-21/traces/gold-hotpotqa.jsonl`. The gold sweep explicitly names the old M0 "
        "keep-correct trace source, and `make_matched_gold_from_m0.py` copies its user/system messages verbatim "
        "while replacing the assistant target. This provides question-level M0 lineage; it does not substitute "
        "for the missing original trace bytes or establish M1/M2/M3 membership. These recovered questions "
        "are also compared with current training, current measurement, and old HotpotQA evaluation below.", "",
        "| Recipe | Paired cells | HotpotQA Δacc | 2Wiki Δacc | Difference | Saved 90% CI |",
        "|---|---:|---:|---:|---:|---|", *split_rows, "",
        "These point estimates were recomputed from the raw saved item scores (score≥0.5), matched by recipe/size/seed "
        "against each size's dense baseline, and agree with V3 within 1e-12. Original bootstrap intervals are read, not refit. "
        f"For M2, the benchmark gap / HotpotQA gain is {old['M2_gap_divided_by_hotpot_gain']:.1%}, explaining the old ‘~1/3’ shorthand. "
        f"For an equal-weight 100-item panel, the gain is {old['M2_equal_weight_panel_gain']:.4f}, and excess over the "
        f"transfer-only gain is {old['M2_panel_excess_over_transfer_fraction']:.1%} of that panel gain. "
        "Neither ratio is a contaminated-item rate. The original result is preserved; this clarifies its denominator.", "",
        "The separate [V1 anchor result](../../results/v1-recipe-strat/summary.md) uses the frozen grid's Rasch θ: "
        "QA M1 anchor mean −1.051 fails, whereas M0 +0.076/M2 +0.028/M3 +0.017 pass the original 2SE rule; "
        "pooling minus M1 passes (+0.043). This is recipe mixing, not a current dataset leak. "
        "V3's 1PL/2PL rank agreement is likewise prior-grid evidence, not a current loss calibration.", "",
        "## Current provenance and overlap", "",
        "[V12 source](../../analysis/v12_distill.py) declares training GSM8K/CodeAlpaca/HotpotQA and measurement "
        "MATH-500/MBPP/2WikiMultihopQA. The actual [V6 builder](../../analysis/v6_capability_geometry.py) loads "
        "`framolfese/2WikiMultihopQA`, split `validation`, after math and code draws from a shared seed-0 RNG. "
        "V12 takes `[1::2]` **after sampling 128 probes**, yielding 64 QA examples; these are odd sampled positions, "
        "not odd dataset rows. V15 saved indices 0–63 are already renumbered measurement examples and are not sliced again.", "",
        f"Inspected {len(current['trace_files'])} current QA trace files (600 rows/teacher in this snapshot), "
        f"{len(current['run_inventory']['v12'])} V12 eval files and {len(current['run_inventory']['v16'])} V16 residual files. "
        f"Metadata conflicts: {len(current['metadata_conflicts'])}. "
        f"{len(current['default_accuracy_panels'])} saved default V15 panels contain the same ordered 64 QA prompts/references; "
        f"{len(current['other_accuracy_panels'])} other V15 artifacts use the easy benchmark suite (QA=TriviaQA). "
        "The default panels are dense/pruned evaluations, not saved default-panel distilled behavioral results. "
        "Easy-suite behavior must not be described as 2Wiki behavior.", "",
        "Trace JSONLs contain prompt, response, teacher, model_version, timestamp and usage; no dataset/config/split/row IDs. "
        "V12 selects the first n rows per domain before training shuffles, so checking both full 600-row files "
        "covers the available smaller current prefixes. Historical trace bytes are not bound by an immutable run manifest. "
        "The source declaration alone does not prove HotpotQA's official training split; the optional existing-cache check supplies independent question identification.", "",
        f"Local HotpotQA distractor cache status: **{cache['status']}**. Only existing Arrow streams were read; no download or model/data-loader execution.", "",
        *cache_lines, "",
        identification, "",
        "The cache check also queries all 64 current measurement questions against every cached HotpotQA train/validation "
        "row, rather than just the selected 600 training questions. Its zero-match result, when available in the "
        "coverage table, is exact-question evidence for these 64 items, not a semantic whole-corpus guarantee.", "",
        "Question comparison uses Unicode NFKC, casefold and whitespace normalization, preserving punctuation. "
        "Blank/missing questions give unknown, never a clean zero. Teacher wording and answers are excluded from identity. "
        "Different dataset ID namespaces and local integer indices are not treated as proof of disjointness.", "",
        "| Comparison | Unique questions left / right | Exact matches | Status |",
        "|---|---:|---:|---|", *overlap_rows, "",
        "Old 2Wiki evals retain task IDs but not questions; current V15 examples retain questions but not source IDs. "
        "Their item intersection is unresolved (not zero). That would be reuse of old **transfer/evaluation** items, "
        "not proof of exposure to old teacher training traces.", "",
        "HotpotQA and 2Wiki are separate QA collections with different construction procedures, not two splits of "
        "one benchmark. Both use Wikipedia; 2Wiki also uses Wikidata and explicit reasoning construction. "
        "[HotpotQA authors](https://hotpotqa.github.io/); "
        "[2Wiki authors' paper](https://aclanthology.org/2020.coling-main.580/). "
        "Thus ‘different corpora’ is defensible at question-benchmark level, not as a guarantee of disjoint source articles. "
        f"Rendered current prompts share {len(c['shared_title_prefixes'])} title prefixes: "
        f"{', '.join(c['shared_title_prefixes']) or 'none'}. "
        f"There are {c['shared_complete_rendered_lines']} identical normalized rendered context lines. "
        "Title reuse is not itself answer leakage. Truncation, serialization and title-prefix parsing prevent a full passage-overlap certificate.", "",
        "## Claim disposition and minimal validation", "",
        f"The saved V21 summary explicitly names {len(v21_refs['declared_raw_paths'])} raw files "
        f"({len(v21_refs['missing_paths'])} missing, {len(v21_refs['old_grid_paths'])} under the prior tree): "
        "V12 evaluations and V6 dense anchors, all fingerprinted by this audit. V31/V31b summaries "
        "record the training design and aggregate contrasts but no raw-path manifest; their historical item binding "
        "cannot be independently reconstructed from those summaries. See the machine-readable claim dependency inventory.", "",
        "| Claim | Data dependence | Disposition |", "|---|---|---|", *claim_rows, "",
        "Branch (b) is triggered at **training-benchmark/distribution level**, not by an observed current train/eval "
        "question collision. Preserve numerical fixed-panel loss observations: C17 gives no reason to subtract one third "
        "from them, delete them, or rerun training. C18's actual ledger statement is about math/code residuals; "
        "the QA interpretation is an extension requiring its own validation. Generic-loss subtraction does not establish behavior.", "",
        s["verdict"]["branch_interpretation"], "",
        "No new model run is needed solely to preserve the present, qualified fixed-panel observations. "
        "The revalidation gate below applies to stronger independent/general QA claims and any assertion of fully "
        "verified historical cleanliness; it is not a mandatory retraining request caused by benchmark reuse.", "",
        f"Minimal existing candidate: the {len(clean['exact_question_clean_indices'])} exact-question-disjoint default "
        "2Wiki measurement examples, whose ordered prompt/reference hashes are saved in `summary.json`. "
        f"A stricter rendered-title-disjoint candidate retains {len(clean['stricter_rendered_title_disjoint_indices'])} examples "
        f"after excluding measurement indices {c['measurement_indices_with_shared_title_prefix']}. "
        "This smaller candidate still needs full-context/near-duplicate screening before a corpus-independence claim. "
        "Selection here uses provenance only, never correctness or loss. No arbitrary new sample size/power claim is made.", "",
        clean["protocol"], "",
        "Revalidation is a gate for independent QA behavioral/generalization claims and unresolved historical clean-set "
        "assertions (C9b QA, C18 QA extension, C21 QA/V31/V31b, trace-recovery QA). It is not evidence that those saved "
        "loss numbers are wrong. V12/V16 aggregate outcomes cannot establish item-level historical provenance or be "
        "recomputed on a filtered panel; saved dense-only V15 records cannot supply a missing distilled comparison. "
        "This CPU audit closes the scope question and specifies remaining validation; it does not claim to have performed that validation.", "",
        "## Excluded interpretations and limits", "",
        *[f"- {x}" for x in s["verdict"]["excluded_interpretations"]], "",
        *[f"- Not established: {x}." for x in s["verdict"]["not_established"]], "",
        "## Reproduce", "", "```bash",
        "python analysis/v34_c17_scope_audit.py",
        "# Optional read-only resolution against the existing local HotpotQA cache:",
        "python analysis/v34_c17_scope_audit.py --hotpot-cache /home/xueqi/.cache/huggingface/datasets/hotpotqa___hotpot_qa",
        "python -m pytest -q tests/test_v34.py", "```", "",
        "The first command reports cache provenance as unavailable; the second produces this snapshot's enriched "
        "split identification. Standard library only unless reading cached Arrow (pyarrow). "
        "[Machine-readable result](../../results/v34-c17-scope/summary.json) includes every input SHA256, "
        "raw respondent paths, run inventory, overlap counts, source IDs recovered from cache and clean-set indices/hashes. "
        "Inputs are hashed again before output; only the V34 summary/report are written. No network, model run, "
        "training, IRT refit, old-result mutation or claim-number correction occurs.", ""])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--hotpot-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    summary = build_summary(args.root, args.hotpot_cache)
    output = args.output_dir or args.root / "results/v34-c17-scope"
    report = args.report or args.root / "paper/docs/C17_SCOPE.md"
    output.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    report.write_text(render_report(summary))
    print(f"{summary['verdict']['status']}: {report}")


if __name__ == "__main__":
    main()
