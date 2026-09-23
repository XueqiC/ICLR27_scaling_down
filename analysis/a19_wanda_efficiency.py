#!/usr/bin/env python3
"""A19: measurement-efficiency confirmation under a second pruning criterion (Wanda), pre-registered in
results/a19-wanda-efficiency/prereg.md. The A11 design on Pythia, with the pruning criterion changed.

Nine development states (160M, 410M, 1.4B at steps 16000, 64000, 143000) pruned with Wanda at densities
{0.9, 0.8, 0.7, 0.6} (full grid, 36 configurations per capability) and {0.9, 0.7} (reduced grid, 18); four
test states (160M step 80000, 410M step 112000, 1.4B step 48000, 1B step 48000) at the six A11 densities
{0.9, 0.85, 0.8, 0.75, 0.7, 0.65}, pruned only after the freeze. Losses on the V6 probes (odd half) and on
the A18 new items; per-item loss sums and tokens, GPU seconds per stage (load, calibration, dense anchor,
every pruned evaluation) recorded. Fitting, freezing, scoring and the tables reuse analysis/a18_second_family
with this experiment's states and grids; the diagnostic applies the A11 magnitude-pruning coefficients
(power_18) to the Wanda measurements without refitting.

Stages
  --calib            cache the Wanda calibration token ids (C4 validation, 128 x 512, Pythia tokenizer) -> calib_ids.json
  --run [--states]   GPU: measure one or more states (measurements/<tag>.json); test states refused before freeze.json
  --freeze           fit every form on both development grids, write freeze.json + sha256 (refuses to overwrite)
  --score            score the frozen predictions on the measured test states (score.json, summary.md)
  --table            cost.json, per_state.md and paper/paper/tables/wanda_efficiency.tex

Hosts: any machine with the Pythia checkpoints in its HF cache (rai) or network access to download them (hpg);
set HF_HUB_CACHE for a project-local cache. Development states may be measured on several hosts and the
measurement files gathered before --freeze.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from analysis import a18_second_family as a18  # noqa: E402
from analysis import p2_wanda_panel as p2  # noqa: E402
from analysis import v53_prune_dev as p53  # noqa: E402

OUT = ROOT / "results/a19-wanda-efficiency"
A11 = ROOT / "results/a11-efficiency-confirmation/predictions.json"
CAPS = a18.CAPS
DEV_DENSITIES = (0.9, 0.8, 0.7, 0.6)
REDUCED = (0.9, 0.7)
TEST_DENSITIES = (0.9, 0.85, 0.8, 0.75, 0.7, 0.65)
REPOS = {"160m": "EleutherAI/pythia-160m", "410m": "EleutherAI/pythia-410m", "1b": "EleutherAI/pythia-1b", "1.4b": "EleutherAI/pythia-1.4b"}
STATES = {}
for _size in ("160m", "410m", "1.4b"):
    for _step in (16000, 64000, 143000):
        STATES[f"pythia-{_size}@step{_step}"] = (_size, f"step{_step}", "dev")
for _size, _step in (("160m", 80000), ("410m", 112000), ("1.4b", 48000), ("1b", 48000)):
    STATES[f"pythia-{_size}@step{_step}"] = (_size, f"step{_step}", "test")
TOKENS_PER_STEP = p53.TOKENS_PER_STEP


def read(p):
    return json.loads(Path(p).read_text())


def bind():
    """Point the A18 fitting, freezing, scoring and rendering code at this experiment."""
    a18.OUT = OUT
    a18.STATES = STATES
    a18.DEV_DENSITIES = DEV_DENSITIES
    a18.REDUCED = REDUCED
    a18.TEST_DENSITIES = TEST_DENSITIES
    a18.pythia_zero_calibration = magnitude_coefficients


def magnitude_coefficients(N0, D0, L0, c, d):
    """Diagnostic: the A11 frozen magnitude-pruning power form at half budget (power_18), no refit."""
    frozen = read(A11)["coefficients"]["power_18"]
    z = p53.standardize(p53.raw_features(N0, D0, L0), frozen["standardizer"])
    fit = frozen["fits"][c]
    return float(np.dot(fit["beta"], z) * p53.shape(d, fit["gamma"]))


def tokens_of(revision):
    return int(revision.removeprefix("step")) * TOKENS_PER_STEP


# ---------------------------------------------------------------- calibration cache
def cache_calibration():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPOS["160m"], revision="step143000")
    ids = p2.calibration_texts(tok)
    OUT.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    (OUT / "calib_ids.json").write_text(json.dumps({"tokenizer": REPOS["160m"], "n_documents": len(ids), "tokens_per_document": p2.CALIB_TOKENS,
                                                    "source": "allenai/c4 en validation, first documents with >= 512 tokens", "sha256": digest, "ids": ids}))
    print("calibration cached:", len(ids), "documents,", digest[:16])


# ---------------------------------------------------------------- measurement
def run_states(tags, device):
    import torch
    from analysis.v6_capability_geometry import build_probes, load_text_causal_lm, language_weight_parameters
    calib = read(OUT / "calib_ids.json")
    new_items = read(ROOT / "results/a18-second-family/items.json")["items"]
    probes = {c: v[1::2] for c, v in build_probes(a18.N_PROBE).items()}
    probe_items = {c: [{"prompt": p["prompt"], "completion": p["completion"]} for p in probes[c]] for c in CAPS}
    (OUT / "measurements").mkdir(parents=True, exist_ok=True)
    for tag in tags:
        size, revision, role = STATES[tag]
        target = OUT / "measurements" / f"{tag.replace('@', '__')}.json"
        if target.exists():
            print(tag, "measured"); continue
        if role == "test" and not (OUT / "freeze.json").exists():
            raise SystemExit(f"{tag} is a test state; freeze.json must exist before it is pruned")
        t0 = time.time()
        model, tok = load_text_causal_lm(REPOS[size], torch.bfloat16, revision)
        model.to(device).eval()
        params = language_weight_parameters(model)
        block = sum(p.numel() for n, p in params if ".layers." in n or ".blocks." in n)
        linears = p2.block_linears(model)
        densities = DEV_DENSITIES if role == "dev" else TEST_DENSITIES
        record = {"tag": tag, "repo": REPOS[size], "revision": revision, "role": role, "size": size, "D0": tokens_of(revision),
                  "N0_block_matrices": block, "prune_scope_parameters": sum(m.weight.numel() for m in linears.values()),
                  "criterion": "wanda", "calibration_sha256": calib["sha256"], "densities": list(densities),
                  "load_seconds": round(time.time() - t0, 1), "measurements": {}}
        t1 = time.time()
        norms = p2.activation_norms(model, tok, linears, device, calib["ids"])
        record["calibration_seconds"] = round(time.time() - t1, 1)
        t1 = time.time()
        record["measurements"]["1.0"] = {"probes": a18.measure_items(model, tok, probe_items, device),
                                         "new": a18.measure_items(model, tok, new_items, device), "seconds": round(time.time() - t1, 1)}
        print(tag, "dense", {c: round(record["measurements"]["1.0"]["probes"][c]["mean_loss"], 3) for c in CAPS},
              "calibration", record["calibration_seconds"], "s", flush=True)
        reference = {name: mod.weight.detach().clone().cpu() for name, mod in linears.items()}
        for d in densities:
            t2 = time.time()
            p2.apply_wanda(linears, norms, d, reference)
            record["measurements"][str(d)] = {"probes": a18.measure_items(model, tok, probe_items, device),
                                              "new": a18.measure_items(model, tok, new_items, device), "seconds": round(time.time() - t2, 1)}
            print(tag, d, {c: round(record["measurements"][str(d)]["probes"][c]["mean_loss"], 3) for c in CAPS}, flush=True)
        record["total_seconds"] = round(time.time() - t0, 1)
        record["tokens_evaluated"] = sum(m[s][c]["tokens"] for m in record["measurements"].values() for s in ("probes", "new") for c in CAPS)
        target.write_text(json.dumps(record, indent=1))
        del model, tok, reference, norms, linears, params
        gc.collect(); torch.cuda.empty_cache()


# ---------------------------------------------------------------- cost with calibration
_A18_COST = a18.cost   # the original A18 accounting, before render() rebinds a18.cost to this experiment's version


def cost():
    """A18's cost accounting plus the calibration pass, counted once per state in both grids."""
    c = _A18_COST()
    for tag, entry in c["development"].items():
        rec = read(OUT / "measurements" / f"{tag.replace('@', '__')}.json")
        for grid in ("full", "reduced"):
            entry[grid]["seconds"] += rec["calibration_seconds"]
            c["totals"][grid]["seconds"] += rec["calibration_seconds"]
        entry["calibration_seconds"] = rec["calibration_seconds"]
    for tag, entry in c["test"].items():
        entry["calibration_seconds"] = read(OUT / "measurements" / f"{tag.replace('@', '__')}.json")["calibration_seconds"]
    tot = c["totals"]
    tot["reduced_over_full"] = {k: tot["reduced"][k] / tot["full"][k] for k in ("seconds", "tokens", "configurations")}
    (OUT / "cost.json").write_text(json.dumps(c, indent=1))
    return c


def render():
    a18.cost = cost
    a18.FORM_WORDS = dict(a18.FORM_WORDS)
    text_target = ROOT / "paper/paper/tables/wanda_efficiency.tex"
    a18_table = ROOT / "paper/paper/tables/second_family.tex"
    keep = a18_table.read_text() if a18_table.exists() else None   # a18.render writes the A18 table path; restore it below
    a18.render()
    generated = a18_table.read_text()
    if keep is not None:
        a18_table.write_text(keep)
    generated = (generated.replace("tab:second-family", "tab:wanda-efficiency")
                 .replace("Second-family confirmation on OLMo-2", "Measurement-efficiency confirmation under Wanda pruning on Pythia")
                 .replace("(1B and 7B intermediate checkpoints used in no fit) at three densities fixed before the freeze, 0.85, 0.75 and 0.65,",
                          "(the four A11 states, in no fit) at the six A11 densities, three of them unseen in development,")
                 .replace("fitted once on the six development states", "fitted once on the nine development states")
                 .replace("Reduced (12)", "Reduced (18)").replace("Full (24)", "Full (36)")
                 .replace("Pythia coefficients, no refit", "Magnitude-pruning coefficients, no refit")
                 .replace("GPU seconds and", "GPU seconds, the Wanda calibration pass counted once per state, and")
                 .replace("dense anchors and model loading included.", "dense anchors and model loading included in both grids, as in the two earlier confirmations."))
    text_target.write_text(generated)
    print("wrote", text_target)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--states", default=""); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--freeze", action="store_true"); ap.add_argument("--score", action="store_true"); ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    bind()
    tags = [t for t in a.states.split(",") if t] or list(STATES)
    if a.calib:
        cache_calibration()
    if a.run:
        run_states(tags, a.device)
    if a.freeze:
        a18.freeze()
    if a.score:
        a18.score()
    if a.table:
        render()
