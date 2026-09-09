#!/usr/bin/env python3
"""P3 measurement check (<=4 GPU-h; runs only after P2-B): same-capability PRIMARY vs SECONDARY benchmark loss
across the three arms on Gemma-3-1b. States (pre-registered, chosen before any P3 result): dense; magnitude
pruning d=0.7 (v6 protocol); RTN int4 (v10 protocol); and P2 adapters for data-seed 11 at U75 and U450 at the
two milestone snapshots (four adapters). Primary = main-protocol measurement probes (MATH-500 / MBPP / 2Wiki,
measurement half); secondary = one independent benchmark per capability, 128 samples, fixed probe seed
(GSM8K / HumanEval / HotpotQA), existing references, no accuracy. Units: native-token nats (same tokenizer)."""
import json, glob, sys, time, torch
from pathlib import Path
import analysis.v12_distill as v12
import analysis.v6_capability_geometry as v6
import analysis.v10_quantization as v10
import analysis.v9_capability_regions as v9
from analysis.model_registry import resolve_model_and_revision
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results/v48-p3-measure"; OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda:0"; N_SEC = 128
SECONDARY = {"math": "svamp", "code": "humaneval", "qa": "triviaqa"}  # NOT gsm8k/hotpotqa: those are training benchmarks
name, rev = resolve_model_and_revision("gemma3-1b")
base, tok = v12.load_text_causal_lm(name, torch.bfloat16, rev); base.to(DEV).eval()
primary = {c: v12.build_probes(v12.DEFAULT_N_PROBE, seed=v12.SEED)[c][1::2] for c in v12.CAPABILITIES}
builders = {s.name: s.builder for s in v9.PROBE_REGISTRY}
secondary = {c: builders[SECONDARY[c]](N_SEC, v9.PROBE_SEED) for c in v12.CAPABILITIES}
params = v6.language_weight_parameters(base); dense_w = [p.detach().clone() for _, p in params]
def measure(model, tag):
    t0 = time.time(); pri, npri = v12.measure_capability_losses(model, tok, primary, DEV); sec, nsec = v12.measure_capability_losses(model, tok, secondary, DEV)
    rec = {"state": tag, "primary": pri, "secondary": sec, "primary_tokens": npri, "secondary_tokens": nsec, "secondary_benchmarks": SECONDARY, "eval_s": time.time() - t0}
    print(json.dumps(rec)); return rec
results = [measure(base, "dense")]
v6.apply_global_magnitude_pruning(base, 0.7, seed=0, reference_weights=dense_w); results.append(measure(base, "prune_d0.7"))
v10._restore_dense_weights(params, dense_w); v10._apply_fake_quantization(params, dense_w, 4); results.append(measure(base, "rtn_int4"))
v10._restore_dense_weights(params, dense_w)
# P2 adapters (data-seed 11; two milestone snapshots each) -- pre-registered choice
from peft import PeftModel
for U in (75, 450):
    d = ROOT / f"results/v12-distill/gemma3-1b/gpt-5.6-luna_full_{U}_p2dev_lora_dseed11"
    snaps = sorted([p for p in glob.glob(str(d / "trajectory/update-*")) if Path(p, "adapter_config.json").exists()])
    snaps = [s for s in snaps if json.loads(Path(s, "eval.json").read_text()).get("processed_tokens", 0) > 0][:2]
    for s in snaps:
        pm = PeftModel.from_pretrained(base, s); pm.eval()
        results.append(measure(pm, f"kd_U{U}_s11_{Path(s).name}")); del pm; torch.cuda.empty_cache()
        base, tok = v12.load_text_causal_lm(name, torch.bfloat16, rev); base.to(DEV).eval(); params = v6.language_weight_parameters(base)
(OUT / "measurements.json").write_text(json.dumps(results, indent=2)); print("wrote", OUT / "measurements.json")
