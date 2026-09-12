#!/usr/bin/env python3
"""Dense-only capability loss with the main protocol (v6 probes, measurement half, v12 loss function).
Used to obtain the K0 input L0 for a NEW source BEFORE any compressed measurement, and as a load/throughput probe."""
import sys, json, time, torch
from pathlib import Path
import analysis.v12_distill as v12
from analysis.model_registry import resolve_model_and_revision
tag = sys.argv[1]; out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
name, rev = resolve_model_and_revision(tag)
t0 = time.time(); model, tok = v12.load_text_causal_lm(name, torch.bfloat16, rev); model.to("cuda:0").eval()
probes = {c: v12.build_probes(v12.DEFAULT_N_PROBE, seed=v12.SEED)[c][1::2] for c in v12.CAPABILITIES}
torch.cuda.reset_peak_memory_stats(); t1 = time.time()
losses, ntok = v12.measure_capability_losses(model, tok, probes, "cuda:0"); t2 = time.time()
rec = {"tag": tag, "resolved": name, "revision": rev, "dense": losses, "measurement_tokens": ntok,
       "load_s": t1 - t0, "eval_s": t2 - t1, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
       "tokens_per_s": sum(ntok.values()) / (t2 - t1)}
(out / "dense.json").write_text(json.dumps(rec, indent=2)); print(json.dumps(rec, indent=1))
