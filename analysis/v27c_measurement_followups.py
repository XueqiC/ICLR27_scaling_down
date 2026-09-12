#!/usr/bin/env python3
"""CPU-only saved-NLL audit and V28 LOMO unit sensitivity; NEVER run a model.

python analysis/v27c_measurement_followups.py [--dry-run] [--hf-cache PATH]

Reads V27/V27b, frozen V28 folds and their V12 evals. Local Arrow/tokenizer
JSON caches reconstruct the SAME references and denominators; no downloading,
datasets/transformers/torch imports, generation, or likelihood computation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

try:
    from . import v27b_scoring_readout as readout
    from . import v28_new_source_prediction as prediction
except ImportError:
    import v27b_scoring_readout as readout
    import v28_new_source_prediction as prediction

ROOT = Path(__file__).resolve().parents[1]
CAPS = ("math", "code", "qa")
MODES = ("A", "B", "zero", "dev_mean_coefficient")
MATERIAL_RELATIVE_CHANGE = .10  # descriptive flag, not a significance test
CACHE_DATASETS = {
    "math": "HuggingFaceH4___math-500/default/0.0.0/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/math-500-test.arrow",
    "code": "google-research-datasets___mbpp/full/0.0.0/4bb6404fdc6cacfda99d4ac4205087b89d32030c/mbpp-test.arrow",
    "qa": "framolfese___2_wiki_multihop_qa/default/0.0.0/fe713bfbd1afbca1a65246741a75890405d56a3a/2_wiki_multihop_qa-validation.arrow",
}


class Inputs:
    def __init__(self, root, cache):
        self.root, self.cache = Path(root), Path(cache)
        self.sources = {}

    def record(self, path, kind):
        path = Path(path)
        label = (path.relative_to(self.root).as_posix() if path.is_relative_to(self.root)
                 else "HF_CACHE/" + path.relative_to(self.cache).as_posix())
        if label not in self.sources:
            with path.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            self.sources[label] = {"path": label, "kind": kind, "sha256": sha,
                                   "bytes": path.stat().st_size}
        return label

    def read(self, path, kind):
        self.record(path, kind)
        return json.loads(Path(path).read_text())


def reconstruct_probes(inputs):
    """V6 seed=0 sampling order, primary benchmarks only, odd half of 128.

    Arrow is read directly: no dataset loader, cache writes, or network fallback.
    Every returned probe must subsequently match a V27 item hash, including
    the MATH references excluded from V27's three-loss decomposition.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    rng = np.random.default_rng(0)
    probes = {}
    for cap, relative in CACHE_DATASETS.items():
        path = inputs.cache / "datasets" / relative
        inputs.record(path, "cached_reference_dataset")
        with pa.memory_map(str(path), "r") as stream:
            table = ipc.open_stream(stream).read_all()
            indices = rng.choice(len(table), size=min(128, len(table)), replace=False)[1::2]
            rows = table.take(pa.array(indices)).to_pylist()
        samples = []
        for row in rows:
            if cap == "math":
                sample = {"prompt": f"Problem: {row['problem']}\nSolution:",
                          "completion": " " + row["solution"], "answer": row["answer"]}
            elif cap == "code":
                sample = {"prompt": f"# Task: {row['text']}\n# Write a Python function.\n",
                          "completion": row["code"],
                          **{k: row.get(k, [] if k != "test_setup_code" else "")
                             for k in ("test_list", "test_setup_code", "challenge_test_list")}}
            else:
                context = row.get("context")
                if isinstance(context, dict):
                    parts = []
                    for title, sentences in zip(context.get("title", []),
                                                context.get("sentences", context.get("content", []))):
                        body = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
                        parts.append(f"{title}: {body}")
                    text = "\n".join(parts)[:4000]
                else:
                    text = str(context)[:4000]
                sample = {"prompt": f"Context:\n{text}\n\nQuestion: {row['question']}\nAnswer:",
                          "completion": " " + str(row["answer"]), "answer": str(row["answer"])}
            samples.append(sample)
        probes[cap] = samples
    return probes


def cached_tokenizer(inputs, hf_id):
    from tokenizers import Tokenizer

    base = inputs.cache / "hub" / ("models--" + hf_id.replace("/", "--"))
    revision = (base / "refs/main").read_text().strip()
    directory = base / "snapshots" / revision
    path = directory / "tokenizer.json"
    label = inputs.record(path, "cached_tokenizer")
    config = inputs.read(directory / "tokenizer_config.json", "cached_tokenizer_config")
    if config.get("truncation_side", "right") != "right":
        raise ValueError("Only the archived right-truncation protocol is supported")
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer, {"hf_id": hf_id, "reconstruction_revision": revision, "source": label,
                       "historical_revision_recorded": False}


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False).ids


def context_ids(tokenizer, hf_id, prompt):
    ids = tokenizer.encode(prompt, add_special_tokens=True).ids[:512]
    if "gemma" in hf_id.lower():
        bos = tokenizer.token_to_id("<bos>")
        if bos is None:
            raise ValueError("Gemma prompt BOS unavailable")
        if not ids or ids[0] != bos:
            ids = ([bos] + ids)[:512]
    if not ids:
        raise ValueError("Empty conditioning span")
    return ids


def scored_prefix(tokenizer, text, limit=512):
    """Exact original UTF-8 prefix, refusing a split byte-fallback character."""
    encoding = tokenizer.encode(text, add_special_tokens=False)
    ids = encoding.ids[:limit]
    if not ids:
        raise ValueError("Empty target")
    prefix = text
    if len(encoding.ids) > limit:
        end = encoding.offsets[limit - 1][1]
        if encoding.offsets[limit][0] < end:
            raise ValueError("Truncation splits a Unicode character")
        prefix = text[:end]
        if encode(tokenizer, prefix) != ids:
            raise ValueError("Truncated reference does not round-trip")
    return ids, prefix


def denominators(tokenizer, probes):
    result = {}
    for cap, samples in probes.items():
        items = []
        for index, sample in enumerate(samples):
            ids, text = scored_prefix(tokenizer, sample["completion"])
            items.append({"measurement_index": index, "probe_sha256": readout.digest(sample),
                          "n_tokens": len(ids), "target_bytes": len(text.encode("utf-8")),
                          "target_sha256": readout.digest(text),
                          "truncated": text != sample["completion"]})
        tokens, byte_count = (sum(i[k] for i in items) for k in ("n_tokens", "target_bytes"))
        result[cap] = {"items": items, "sum_tokens": tokens, "sum_bytes": byte_count,
                       "token_to_byte_factor": tokens / byte_count,
                       "aggregation": "corpus ratio; same 64 targets, each capped at 512 tokens"}
    return result


def audit_panel(path, payload, probes, tokenizer, tokenizer_meta, counts):
    groups, samples = [], []
    empty_n = mismatches = 0
    for cap, benchmark, _ in readout.benchmark_entries(payload):
        row, maps = readout.analyze_benchmark(cap, benchmark)
        empty = [s for s in benchmark["probe_decompositions"] if s.get("r") == ""]
        for split in empty:
            identity = (split["measurement_index"], split["probe_sha256"])
            if identity not in maps["L_full"]:
                continue
            empty_n += 1
            mismatches += int(any(maps[loss][identity] != maps["L_full"][identity]
                                  for loss in readout.LOSSES))
        if empty:
            groups.append({"benchmark": cap, "n_empty": len(empty), "losses": row["losses"],
                           "legacy_example_token": row["legacy_v6"]["example_token"],
                           "legacy_corpus_token": row["legacy_v6"]["corpus_token"]})
        if cap not in probes:
            continue
        for split, probe in zip(benchmark["probe_decompositions"], probes[cap], strict=True):
            if split["probe_sha256"] != readout.digest(probe):
                raise ValueError("Cached reference is not the original V27 probe")
        for saved, count in zip(benchmark["legacy_v6"]["items"], counts[cap]["items"], strict=True):
            for key in ("measurement_index", "probe_sha256", "n_tokens", "target_bytes"):
                if saved[key] != count[key]:
                    raise ValueError(f"Legacy denominator mismatch: {path}/{cap}/{key}")
        canonical = readout.item_map(benchmark["format_control"]["variants"]["canonical"]["items"])
        for split in benchmark["probe_decompositions"]:
            index = split["measurement_index"]
            identity = (index, split["probe_sha256"])
            if identity not in maps["L_full"]:
                continue
            probe = probes[cap][index]
            p = context_ids(tokenizer, payload["tokenizer"], probe["prompt"])
            r, y = encode(tokenizer, split["r"]), encode(tokenizer, split["y"])
            ranges = {"L_full": (p, r + y), "L_direct": (p, y), "L_given": (p + r, y)}
            if identity in canonical:
                context = ranges[benchmark["format_control"]["conditioning"]][0]
                if canonical[identity]["context_tokens_sha256"] != readout.digest(context):
                    raise ValueError("Reconstructed prompt IDs disagree with saved context hash")
            for loss, (_, target) in ranges.items():
                if len(target) != maps[loss][identity]["n_tokens"]:
                    raise ValueError("Reconstructed scoring span disagrees with saved token count")
            # Three fixed examples across the panel, chosen by identity, not loss.
            chosen = {("gemma3-1b", "code", 0), ("gemma3-4b", "qa", 0), ("olmo3-7b", "qa", 1)}
            if (payload["model"], cap, index) not in chosen:
                continue
            losses = {}
            for loss, (context, target) in ranges.items():
                item = maps[loss][identity]
                losses[loss] = {"prompt_token_ids": context,
                                "prompt_fed_decoded": tokenizer.decode(context, skip_special_tokens=False),
                                "scored_token_ids": target,
                                "scored_token_pieces": [tokenizer.id_to_token(i) for i in target],
                                "scored_text": split["r"] + split["y"] if loss == "L_full" else split["y"],
                                "input_target_span_half_open": [len(context), len(context) + len(target)],
                                "logit_span_half_open": [len(context) - 1, len(context) + len(target) - 1],
                                "sum_nll_nats": item["sum_ce"], "n_tokens": len(target),
                                "target_bytes": item["target_bytes"],
                                "loss_nats_per_token": item["sum_ce"] / len(target),
                                "normalization": "sum target CE / target tokens; prompt masked; no target BOS/EOS"}
            samples.append({"model": payload["model"], "benchmark": cap,
                            "measurement_index": index, "v6_probe_index": 2 * index + 1,
                            "probe_sha256": identity[1], "r": split["r"], "y": split["y"],
                            "prompt_original": probe["prompt"], "source": path,
                            "tokenizer": tokenizer_meta,
                            "context_hash_verified": True, "losses": losses})
    if mismatches:
        raise ValueError("Empty reasoning has unequal scoring records")
    return {"source": path, "model": payload["model"], "probe_sha256": payload["probe_sha256"],
            "protocol": payload["protocol"], "prune_density": payload["prune_density"],
            "empty_reasoning_items": empty_n, "unequal_three_loss_items": mismatches,
            "groups": groups, "samples": samples}


def convert_rows(rows, factors):
    """Change dense inputs AND signed outcomes, retaining every row and fold."""
    converted = copy.deepcopy(rows)
    for row in converted:
        factor = factors[row["model"]][row["capability"]]["token_to_byte_factor"]
        row["dense_loss"] *= factor
        row["deltas"] = {c: value * factor for c, value in row["deltas"].items()}
    return converted


def replay_arm(rows, arm, saved):
    folds, records = [], []
    for original in saved["folds"]:
        model = original["held_out"]
        train = [r for r in rows if r["model"] != model]
        target, = [r for r in rows if r["model"] == model]
        if [r["model"] for r in train] != original["fit"]["mapping"]["train_models"]:
            raise ValueError("LOMO training membership/order differs from frozen V28")
        fit = prediction.fit_arm(train, arm)
        coordinates = [r["coordinate"] for r in saved["records"] if r["model"] == model]
        calibration = prediction.CALIBRATION[arm]
        if calibration in coordinates:
            raise ValueError("Calibration coordinate in test scores")
        point = {"coordinate": calibration,
                 "loss": target["dense_loss"] + target["deltas"][str(calibration)]}
        basic = prediction.basic_input(target)
        a = prediction.predict_arm(fit, basic, coordinates)
        b = prediction.predict_arm(fit, basic, coordinates, "B", point)
        folds.append({"held_out": model, "fit": fit})
        for index, coordinate in enumerate(coordinates):
            records.append({"model": model, "coordinate": coordinate,
                            "observed_delta": target["deltas"][str(coordinate)],
                            "A": a[index], "B": b[index], "zero": 0.,
                            "dev_mean_coefficient": fit["mean_coefficient"] * prediction.shape(arm, coordinate, fit["gamma"])})
    return {"folds": folds, "records": records}


def replay_distillation(rows, cap, saved):
    folds, records = [], []
    for original in saved["folds"]:
        model = original["held_out"]
        by_id = {r["row_id"]: r for r in rows}
        train = [by_id[i] for i in original["fit"]["train_row_ids"]]
        if any(r["model"] == model for r in train):
            raise ValueError("Held-out model leaked into distillation fit")
        fit = prediction.fit_transfer(train, cap)
        cal = by_id[original["mode_B_calibration_row_id"]]
        b = cal["observed"] / prediction.transfer_shape(cap, cal["D"])
        folds.append({"held_out": model, "fit": fit,
                      "mode_B_calibration_row_id": cal["row_id"]})
        for old in saved["records"]:
            if old["model"] != model:
                continue
            row = by_id[f"{model}|{old['coordinate']}|{cap}"]
            shape = prediction.transfer_shape(cap, row["D"])
            records.append({"model": model, "coordinate": row["D"], "observed_delta": row["observed"],
                            "A": fit["coefficient"] * shape, "B": b * shape, "zero": 0.,
                            "dev_mean_coefficient": mean(r["observed"] for r in train)})
    return {"folds": folds, "records": records}


def compare_predictions(token, byte, saved, cap, factors):
    identity = lambda r: (r["model"], r["coordinate"])
    if [identity(r) for r in token["records"]] != [identity(r) for r in saved["records"]]:
        raise ValueError("Held-out score cells differ from frozen V28")
    if [identity(r) for r in byte["records"]] != [identity(r) for r in token["records"]]:
        raise ValueError("Token/byte score cells differ")
    replay_max_error = 0.
    for a, b in zip(token["records"], saved["records"], strict=True):
        for key in (*MODES, "observed_delta"):
            replay_max_error = max(replay_max_error, abs(a[key] - b[key]))
            if not math.isclose(a[key], b[key], rel_tol=2e-7, abs_tol=2e-7):
                raise ValueError(f"Native-token replay differs from frozen V28: {key}")
    metrics = {}
    zero_token = mean(abs(r["observed_delta"]) for r in token["records"])
    zero_byte = mean(abs(r["observed_delta"]) for r in byte["records"])
    for mode in MODES:
        t = mean(abs(r[mode] - r["observed_delta"]) for r in token["records"])
        b = mean(abs(r[mode] - r["observed_delta"]) for r in byte["records"])
        equivalent = mean(abs(r[mode] - r["observed_delta"]) /
                          factors[r["model"]][cap]["token_to_byte_factor"] for r in byte["records"])
        relative = equivalent / t - 1 if t else None
        scaled_frozen = mean(abs(r[mode] - r["observed_delta"]) *
                             factors[r["model"]][cap]["token_to_byte_factor"] for r in token["records"])
        metrics[mode] = {"token_mae": t, "byte_mae": b,
                         "byte_rescored_frozen_prediction_mae": scaled_frozen,
                         "byte_refit_mae_in_token_units": equivalent,
                         "refit_relative_mae_change_in_token_units": relative,
                         "material_refit_change": abs(relative) >= MATERIAL_RELATIVE_CHANGE if relative is not None else None,
                         "token_mae_over_zero": t / zero_token, "byte_mae_over_zero": b / zero_byte,
                         "token_beats_zero": t < zero_token, "byte_beats_zero": b < zero_byte}
        readout.close(t, saved["metrics"][mode]["mae"], "saved native MAE")
    return {"n_models": len(saved["folds"]), "n_test_cells": len(saved["records"]),
            "token_replay_max_abs_error": replay_max_error, "metrics": metrics,
            "token": token, "byte": byte}


def compare_coefficients(result, arm, cap, rows, factors):
    """Compare transferred amplitudes in common units; gamma is dimensionless."""
    amplitudes, gamma = [], []
    for t, b in zip(result["token"]["folds"], result["byte"]["folds"], strict=True):
        model = t["held_out"]
        if b["held_out"] != model:
            raise ValueError("Coefficient folds differ")
        factor = factors[model][cap]["token_to_byte_factor"]
        if arm == "distillation":
            token_a, byte_a = t["fit"]["coefficient"], b["fit"]["coefficient"]
        else:
            target, = [r for r in rows if r["model"] == model]
            target = prediction.basic_input(target)
            token_a = prediction.predict_mapping(t["fit"]["mapping"], target)
            target["dense_loss"] *= factor
            byte_a = prediction.predict_mapping(b["fit"]["mapping"], target)
        amplitudes.append({"held_out": model, "token_amplitude": token_a,
                           "byte_amplitude": byte_a, "byte_amplitude_in_token_units": byte_a / factor})
        if arm == "pruning":
            gt, gb = t["fit"]["gamma"], b["fit"]["gamma"]
            gamma.append({"held_out": model, "token": gt, "byte": gb, "relative_change": gb / gt - 1})
    change = (sum(abs(a["byte_amplitude_in_token_units"] - a["token_amplitude"]) for a in amplitudes)
              / sum(abs(a["token_amplitude"]) for a in amplitudes))
    return {"mode": "A", "held_out_amplitudes": amplitudes,
            "relative_l1_amplitude_change_in_token_units": change,
            "material_amplitude_change": change >= MATERIAL_RELATIVE_CHANGE,
            "definition": "sum abs(byte-fit amplitude / held-out T/B - token-fit amplitude) / sum abs(token-fit amplitude)",
            "gamma_folds": gamma,
            "gamma_max_abs_relative_change": max(abs(g["relative_change"]) for g in gamma) if gamma else None}


def build_summary(root=ROOT, cache=None):
    inputs = Inputs(root, cache or Path.home() / ".cache/huggingface")
    root = inputs.root
    frozen = inputs.read(root / "results/v28-new-source-pred/frozen_predictions.json", "frozen_lomo")
    old = inputs.read(root / "results/v27b-readout/summary.json", "v27b_readout")
    if frozen["version"] != 28 or old["version"] != "v27b-readout-v1":
        raise ValueError("Unexpected saved analysis version")
    probes = reconstruct_probes(inputs)
    factors, tokenizers = {}, {}
    for model, meta in sorted(frozen["metadata"]["models"].items()):
        tokenizer, provenance = cached_tokenizer(inputs, meta["hf_id"])
        factors[model] = denominators(tokenizer, probes)
        # Keep only the three tokenizers needed for prompt reconstruction.
        if model in {p["model"] for p in old["panels"]}:
            tokenizers[model] = tokenizer, provenance
        factors[model]["tokenizer"] = provenance
    panels = []
    for previous in old["panels"]:
        path = root / previous["id"]
        payload = inputs.read(path, "v27_panel")
        source, = [s for s in old["sources"] if s["path"] == previous["id"]]
        if inputs.sources[previous["id"]]["sha256"] != source["sha256"]:
            raise ValueError("V27 panel changed since V27b")
        if payload["metric_definitions"]["MAIN"]["units"] != "nats/token":
            raise ValueError("Main metric changed")
        tokenizer, provenance = tokenizers[payload["model"]]
        panels.append(audit_panel(previous["id"], payload, probes, tokenizer, provenance, factors[payload["model"]]))
    evals, count_checks = {}, []
    # Validate every available final eval count for development models, keeping
    # fitting membership strictly on the frozen V28 manifest below.
    for model in factors:
        for path in sorted((root / "results/v12-distill" / model).glob("*/eval.json")):
            payload = inputs.read(path, "eval_denominator_check")
            if payload.get("probe_seed") != 0 or payload.get("n_probe_requested") != 128:
                raise ValueError("Eval probe protocol differs")
            for cap in CAPS:
                if (payload["measurement_tokens"][cap] != factors[model][cap]["sum_tokens"]
                        or payload["measurement_samples"][cap] != len(probes[cap])):
                    raise ValueError(f"Eval token/sample count mismatch: {path}/{cap}")
            label = path.relative_to(root).as_posix()
            evals[label] = payload
            count_checks.append(label)
    distill_rows = []
    for source, sha in sorted(frozen["input_sha256"].items()):
        if not source.endswith("/eval.json"):
            continue
        payload = evals[source]
        if inputs.sources[source]["sha256"] != sha:
            raise ValueError("Frozen training eval changed")
        if payload["training_mode"] != "lora":
            continue  # frozen recipe exclusion, including 4B/D600
        for cap in CAPS:
            readout.close(payload["delta"][cap], payload["post_training"][cap] - payload["dense"][cap], "eval own-dense delta")
            model, budget = payload["student_tag"], payload["n_per_domain"]
            distill_rows.append({"row_id": f"{model}|{budget}|{cap}", "model": model,
                                "D": budget, "capability": cap, "observed": payload["delta"][cap],
                                "dense_loss": payload["dense"][cap], "source": source})
    sensitivity = {}
    for arm in ("pruning", "quantization", "distillation"):
        sensitivity[arm] = {}
        for cap in CAPS:
            original = frozen["methods"][arm]["by_capability"][cap]
            saved = original["lomo"]
            if arm == "distillation":
                rows = [r for r in distill_rows if r["capability"] == cap]
                byte_rows = copy.deepcopy(rows)
                for r in byte_rows:
                    factor = factors[r["model"]][cap]["token_to_byte_factor"]
                    r["observed"] *= factor
                    r["dense_loss"] *= factor
                token = replay_distillation(rows, cap, saved)
                byte = replay_distillation(byte_rows, cap, saved)
                full = {"token": prediction.fit_transfer(rows, cap), "byte": prediction.fit_transfer(byte_rows, cap)}
            else:
                rows = frozen["development_rows"][arm][cap]
                byte_rows = convert_rows(rows, factors)
                token = replay_arm(rows, arm, saved)
                byte = replay_arm(byte_rows, arm, saved)
                full = {"token": prediction.fit_arm(rows, arm), "byte": prediction.fit_arm(byte_rows, arm)}
            result = compare_predictions(token, byte, saved, cap, factors)
            result["full_development_coefficients_descriptive_only"] = full
            result["coefficient_sensitivity"] = compare_coefficients(result, arm, cap, rows, factors)
            sensitivity[arm][cap] = result
    for name in ("v27_scoring_and_units.py", "v27b_scoring_readout.py", "v28_new_source_prediction.py",
                 "v12_distill.py", "v6_capability_geometry.py", "prediction_audit.py", "v25_distill_delta.py",
                 "v27c_measurement_followups.py"):
        inputs.record(root / "analysis" / name, "analysis_code")
    return {"version": "v27c-measurement-followups-v1", "execution": "CPU; cached text/tokenization and saved-NLL arithmetic/refits only; zero model runs",
            "main_metric": old["measurement_policy"]["MAIN"],
            "empty_reasoning": {"status": "no two-loss split: all three values equal for every empty-r probe",
                                "panels": panels, "n_items": sum(p["empty_reasoning_items"] for p in panels),
                                "unequal_items": sum(p["unequal_three_loss_items"] for p in panels)},
            "prediction_sensitivity": sensitivity, "denominators": factors,
            "provenance": {"prediction_version": 28, "freeze_sha256": frozen["freeze_sha256"],
                           "reference": frozen["reference"], "measurement_original": frozen["measurement"],
                           "fold_protocol_original": frozen["protocol"],
                           "eval_token_count_checks": count_checks,
                           "probe_reconstruction": "V6 seed0 odd64; all 192 primary probe hashes checked against each V27 panel; no new probes",
                           "conversion": "L_byte = L_corpus_token * sum(scored target tokens)/sum(exact scored UTF8 bytes); same factor for own-dense and compressed losses",
                           "scope": "V28 compression-response LOMO: 12 source models; 3 Gemma3 distillation students. No prospective outcomes, V26 benchmark-transfer refits, or new candidates.",
                           "historical_limit": "V6/V10 aggregate-only losses lack per-item hashes and historical tokenizer revisions. Their conversion is conditional on the declared V6 odd64/seed0/512-target-token protocol and cached tokenizer. V27 hashes/spans and available V12 total token counts validate reconstruction, but do not retroactively certify every historical run.",
                           "units": {"token": "nats/token, corpus ratio (historical prediction endpoint)",
                                     "byte": "nats/UTF8-byte, corpus ratio; labeled sensitivity only"},
                           "materiality": {"relative_threshold": MATERIAL_RELATIVE_CHANGE,
                                           "definition": "absolute relative MAE change after converting each byte-refit error back with its own held-out model denominator; descriptive, not statistical significance"}},
            "sources": sorted(inputs.sources.values(), key=lambda s: s["path"])}


def report(summary):
    table, fmt = readout.table, readout.fmt
    audit = summary["empty_reasoning"]
    lines = ["# Measurement follow-ups (v27c)", "",
             "**The saved empty-reasoning panel has one value across all three losses.** "
             f"The audit checks {audit['n_items']} empty-r model/probe records, with {audit['unequal_items']} disagreements. "
             "The byte prediction sensitivity below replays the original V28 LOMO folds and refits their coefficients. "
             "No model was run. MAIN remains V27 L_full, mean_i(NLL_i / target tokens_i), in nats/token.", "",
             "## 1. Exact empty-reasoning audit", "",
             "The three saved panels are Gemma3-1B, Gemma3-4B, and OLMo3-7B at pruning density 0.7. "
             "References are the existing V6 seed-0 odd-index measurement probes (64 per benchmark; "
             "MBPP/HumanEval code and 2WikiMultihopQA/HotpotQA answers have r=empty). "
             "All three definitions use the same context p and target y when r is empty. "
             "V27 caches by (context IDs, target IDs), so it copies the same NLL record into all three losses. "
             "There is no prompt, scored-span, or per-sample normalization difference among them.", "",
             "The two different numbers actually present in each scoring group are `L_c` (equal-example "
             "mean of NLL/token) and `token_weighted_L_c` (sum NLL / sum target tokens). "
             "This is **example weighting**, not full versus direct/given, and neither is a per-sequence loss. "
             "`sum_ce` is the unnormalized per-sequence NLL before division. The following saved columns "
             "pinpoint this genuine two-value distinction; each column is identical across all three losses.", "",
             table(["Model", "Benchmark", "Empty r n", "L_full = L_direct = L_given (L_c)", "token_weighted_L_c"],
                   [[p["model"], g["benchmark"], g["n_empty"], fmt(g["losses"]["L_full"]["example_token"]),
                     fmt(g["losses"]["L_full"]["corpus_token"])] for p in audit["panels"] for g in p["groups"]]), "",
             "Format-control values belong to different targets: e.g. `Answer: ` is prepended to y, "
             "and its tokens/bytes enter both numerator and denominator. MATH/GSM8K have nonempty r "
             "and generally three different losses. Neither observation supplies a second empty-r conditioning loss. "
             "Without a specific contrary output, attributing an alleged two-loss split to a hidden prompt change would be unsupported.", "",
             "### Per-sample prompts and scored spans", "",
             "The following three examples use fixed model/benchmark/index identities, without loss-based selection. "
             "Prompts are reconstructed offline from the original cached references; all primary probe hashes "
             "and saved canonical context-token hashes match. Decoded strings below show exactly the retained "
             "token sequence, including special tokens. JSON escapes preserve whitespace. Authoritative integer "
             "context/target IDs for each of the three losses, raw untruncated prompt, token pieces, and hashes "
             "are in summary.json. Positions are zero-based, half-open in the concatenated model input. "
             "Prompt tokens are masked; target has no BOS/EOS. Target token at input position j is scored by logit j−1.", ""]
    for panel in audit["panels"]:
        for sample in panel["samples"]:
            lines += [f"#### {sample['model']} / {sample['benchmark']} / measurement {sample['measurement_index']} (V6 index {sample['v6_probe_index']})", "",
                      f"Source: `{sample['source']}`; probe SHA-256 `{sample['probe_sha256']}`. r=`\"\"`.", "",
                      "Prompt fed (same for L_full/L_direct/L_given; decoded retained IDs):", "", "```json",
                      json.dumps(sample["losses"]["L_full"]["prompt_fed_decoded"], ensure_ascii=False), "```", "",
                      "Scored text y (same for all three):", "", "```json", json.dumps(sample["y"], ensure_ascii=False), "```", "",
                      table(["Loss", "Input target span", "Target token IDs", "NLL (nats/sequence)", "Target tokens", "NLL/token"],
                            [[loss, str(value["input_target_span_half_open"]), str(value["scored_token_ids"]),
                              fmt(value["sum_nll_nats"]), value["n_tokens"], fmt(value["loss_nats_per_token"])]
                             for loss, value in sample["losses"].items()]), ""]
    lines += ["Both QA examples hit the 512-token prompt cap before the final question and `Answer:` cue. "
              "That retained prompt is verified against the saved token hash and is shared by all three "
              "conditions; it does not explain a difference among their losses.", "",
              "## 2. Per-byte prediction-error sensitivity", "",
              "This is the **same V28 compression-response LOMO**, including its original training models, "
              "score cells, coefficients/forms, own-dense anchors, and exclusions. Pruning: 12 models, "
              "train densities .9/.8/.7/.6, test .8/.7/.6. Quantization: the same 12 models, train bits 8/6/4, "
              "test 5/4. Distillation: 3 Gemma3 sizes, original LoRA rows, teacher gpt-5.6-luna/full, "
              "D=75/150/300/600 with full-training 4B/D600 excluded. D150 is calibration-only for test scoring. "
              "Mode A uses no compressed target calibration; B uses exactly the original one point (.9, 6 bits, D150). "
              "Every coefficient and preprocessor is fitted anew without the held-out model. No prospective outcomes enter.", "",
              "V28's historical prediction endpoint is a **corpus** ratio, sum NLL / sum target tokens, "
              "so this sensitivity preserves corpus weighting: L_byte = L_token × T/B, where T and B are "
              "the totals for the same 64 reference completions, capped at 512 target tokens. "
              "MATH uses all 64 original references, including the 29 excluded from V27's reasoning decomposition. "
              "Dense inputs, compressed losses, deltas and calibration points all change units together. "
              "This conversion does not infer byte loss from an equal-example aggregate and does not substitute "
              "V27's pruned absolute losses for historical prediction deltas.", "",
              summary["provenance"]["historical_limit"], "",
              f"All {len(summary['provenance']['eval_token_count_checks'])} available final V12 evals for development models "
              "match the reconstructed per-capability token/sample totals. V27 legacy per-item token and byte "
              "counts match for its three models. The remaining historical runs retain the conditional provenance above. "
              "Full denominator manifests and tokenizer/cache SHA-256s are in summary.json.", "",
              "Numerical MAEs in different units naturally have different scales. To distinguish rescaling "
              "from altered prediction quality, `byte refit → token MAE` divides each held-out error by its own "
              "model's T/B before averaging over the same cells. `MAE/zero` is a dimensionless comparison "
              "against zero-change, separately in each unit. A ≥10% relative change in the converted-back "
              "MAE is flagged descriptively as material; this is not an uncertainty/significance claim. "
              "A byte score of merely rescaled frozen predictions is also saved, separately from the refit.", ""]
    rows, coefficient_rows, coefficient_changes = [], [], []
    for arm, caps in summary["prediction_sensitivity"].items():
        for cap, result in caps.items():
            for mode in ("A", "B", "zero"):
                m = result["metrics"][mode]
                rows.append([arm, cap, mode, result["n_test_cells"], fmt(m["token_mae"]), fmt(m["byte_mae"]),
                             fmt(m["byte_refit_mae_in_token_units"]),
                             f"{m['refit_relative_mae_change_in_token_units']:+.2%}",
                             f"{m['token_mae_over_zero']:.4f} / {m['byte_mae_over_zero']:.4f}",
                             "yes" if m["material_refit_change"] else "no"])
            full = result["full_development_coefficients_descriptive_only"]
            field = "coefficient" if arm == "distillation" else "mean_coefficient"
            coefficient_rows.append([arm, cap, "mu" if arm == "distillation" and cap != "code" else
                                     "beta (log1p D/150)" if arm == "distillation" else
                                     "mean a at density .7" if arm == "pruning" else "mean q/1024",
                                     fmt(full["token"][field]), fmt(full["byte"][field]),
                                     fmt(full["token"].get("gamma")), fmt(full["byte"].get("gamma"))])
            sensitivity = result["coefficient_sensitivity"]
            coefficient_changes.append([arm, cap,
                                        f"{sensitivity['relative_l1_amplitude_change_in_token_units']:.2%}",
                                        "yes" if sensitivity["material_amplitude_change"] else "no",
                                        f"{sensitivity['gamma_max_abs_relative_change']:.3%}" if arm == "pruning" else "fixed / N/A"])
    prune_code = summary["prediction_sensitivity"]["pruning"]["code"]["metrics"]["A"]
    quant_code = summary["prediction_sensitivity"]["quantization"]["code"]["metrics"]["A"]
    lines += [table(["Arm", "Capability", "Mode", "Cells", "Token MAE", "Byte MAE", "Byte refit → token MAE", "Relative change", "MAE/zero token / byte", "≥10%"], rows), "",
              f"**Prediction conclusion:** pruning CODE Mode A changes from {prune_code['token_mae']:.6f} "
              f"nats/token MAE to {prune_code['byte_mae']:.6f} nats/byte; its converted-back MAE is "
              f"{prune_code['byte_refit_mae_in_token_units']:.6f}, a {prune_code['refit_relative_mae_change_in_token_units']:+.2%} "
              "change that meets the descriptive 10% threshold. Quantization CODE Mode A changes by "
              f"{quant_code['refit_relative_mae_change_in_token_units']:+.2%}, below that threshold but numerically "
              "nonzero. Other A/B converted-back MAE changes are below 3%; distillation is pure unit scaling. "
              "No A/B comparison against zero-change reverses. Stable benchmark rankings therefore coexist "
              "with a material prediction-error change in pruning CODE.", "",
              "### Law coefficients", "",
              "These full-development coefficients are descriptive; the MAEs above use the separate saved "
              "LOMO memberships and newly fitted fold coefficients, all retained in JSON. Amplitudes carry "
              "loss units, so their raw numerical changes alone are not evidence of a changed law. "
              "Pruning gamma is dimensionless and can change because model-dependent T/B changes the "
              "relative weight of curves in the profiled SSE. Quantization's exponent remains fixed at 2 "
              "in the original V28 form. Distillation uses identical Gemma target denominators within "
              "capability; a common rescaling leaves its converted-back predictions unchanged.", "",
              table(["Arm", "Capability", "Coefficient", "Token units", "Byte units", "Gamma token", "Gamma byte"], coefficient_rows), "",
              "To compare the transferred law amplitude itself, the next table uses each held-out Mode A "
              "amplitude (a, q/1024, mu, or beta), converts its byte fit back with that target's T/B, and "
              "reports sum absolute amplitude changes / sum absolute original amplitudes. This avoids "
              "unstable percentage changes of individual near-zero signed coefficients. The same 10% "
              "descriptive threshold is used. Gamma reports the maximum absolute relative change across "
              "LOMO folds. The full ridge mappings and every fold amplitude are retained in JSON.", "",
              table(["Arm", "Capability", "Amplitude change in common units", "≥10%", "Max fold gamma change"], coefficient_changes), "",
              "The 0/18 rank flips in V27b therefore do not establish prediction-error invariance. "
              "Use the matched-fold error and coefficient comparisons here for that question. "
              "V27 native-token MAIN and all frozen historical/prospective artifacts are unchanged.", "",
              "## Reproduce and provenance", "", "```bash", "python analysis/v27c_measurement_followups.py",
              "python -m pytest -q tests/test_v27c.py", "```", "",
              "Requires numpy, scipy, pyarrow, tokenizers, the existing results, and the read-only local "
              "Hugging Face cache (override with --hf-cache). Missing/mismatched references or denominators "
              "raise an error; no download or model fallback exists. --dry-run validates and computes without writes. "
              "Outputs: results/v27c/summary.json and this report. Summary includes source file hashes, "
              "frozen prediction digest, exact per-sample IDs/spans, all fold fits/predictions, conversion "
              "denominators, and native replay errors. Historical uncaptured tokenizer revisions remain explicit.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-cache", type=Path, default=Path.home() / ".cache/huggingface")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summary = build_summary(cache=args.hf_cache)
    if args.dry_run:
        print(report(summary))
        return
    output = ROOT / "results/v27c/summary.json"
    document = ROOT / "paper/docs/MEASUREMENT_FOLLOWUPS.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    document.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    document.write_text(report(summary))
    print(f"Wrote {output.relative_to(ROOT)} and {document.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
