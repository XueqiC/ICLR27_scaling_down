#!/usr/bin/env python3
"""Register A12 amendment 1 from the archived preparation; CPU token accounting only."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import random
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from analysis import a12_prepare as prep  # Disables CUDA and network before trainer imports.
from analysis.a12_score import build_decisions, digest, load_plan, require

a3 = prep.a3
OUT = prep.OUT
ORIGINAL = OUT / "registration_0/plan.json"
ORIGINAL_PREDICTIONS_SHA256 = "e40c51d41414eb1fb151c7f80047d723864e9eddc9c1dbd32572df1ae291f308"
CAP = 158391
SEED_RULE = ("For each former above slot, start at its originally reserved data_seed and take the first "
             "unused training seed in that 10000-seed block whose nearest completed updates meet all "
             "three registered 0.5% supervised-budget tolerances within the unchanged schedule. "
             "Selection uses integer token accounting only; no model weights, losses or A12 outcomes.")


def full_pool(counter, training_seed):
    """Price all rows with V12's no-data-seed ordering and training-seed shuffles."""
    pool = counter.price(600, None)
    records, _ = a3.load_sft_records(a3.TEACHER, a3.DOMAINS, 600, a3.RECIPE,
                                    counter.trace_base, training_seed, None)
    pool["_ledger"] = [{**counter.cache[r["prompt"], r["completion"]], "domain": r["domain"]}
                       for r in records if counter.cache[r["prompt"], r["completion"]]["completion_tokens"]]
    pool["data_sampling_seed"] = training_seed
    return pool


def select_full_pool(counter, template, start_seed, excluded):
    # The same complete membership is reused; cache tokenization across candidate seeds.
    for seed in range(start_seed, start_seed + 10000):
        if seed in excluded:
            continue
        pool = full_pool(counter, seed)
        require(pool["D_U_pool"] == CAP and pool["data_pool_sha256"] == template["data_pool_sha256"],
                "Full-pool capacity or identity changed")
        # A3's replay takes its shuffle seed from data_seed; this temporary view
        # implements V12's fallback to training seed without changing pool metadata.
        grid, schedule = a3.replay_accounting({**pool, "data_seed": seed})
        chosen = [min(grid, key=lambda r: (abs(r["supervised_tokens"] - b), r["update"]))
                  for b in prep.BUDGETS]
        if all(abs(c["supervised_tokens"] / b - 1) <= .005 for c, b in zip(chosen, prep.BUDGETS)):
            return pool, chosen, schedule
    raise ValueError("No full-pool training seed meets registered checkpoint tolerance")


def checkpoints(pool, chosen, schedule):
    seen, unique, step = set(), 0, 0
    totals = {0: 0}
    for epoch in range(math.ceil(chosen[-1]["update"] / schedule["updates_per_epoch"])):
        order = list(range(len(pool["_ledger"])))
        random.Random(pool["data_sampling_seed"] + epoch).shuffle(order)
        for start in range(0, len(order), a3.EFFECTIVE_BATCH_SIZE):
            for index in order[start:start + a3.EFFECTIVE_BATCH_SIZE]:
                item = pool["_ledger"][index]
                if item["example_id"] not in seen:
                    seen.add(item["example_id"])
                    unique += item["completion_tokens"]
            step += 1
            totals[step] = unique
    result = [{"nominal_supervised_tokens": 0, "actual_supervised_tokens": 0,
               "processed_tokens": 0, "update": 0, "D_U_seen": 0}]
    for budget, row in zip(prep.BUDGETS, chosen):
        result.append({"nominal_supervised_tokens": budget, "actual_supervised_tokens": row["supervised_tokens"],
                       "processed_tokens": row["processed_tokens"], "update": row["update"],
                       "D_U_seen": totals[row["update"]],
                       "budget_relative_error": row["supervised_tokens"] / budget - 1})
    return result


def task_table(plan):
    lines = ["| Index | Student | Tier | Pool tokens | Seed | Checkpoint budgets (supervised tokens) |",
             "|---:|---|---|---:|---|---|"]
    for tr in plan["trajectories"]:
        seed = (f"train {tr['training_seed']} (data_seed unset)" if tr["position"] == "above"
                else f"pool {tr['data_seed']}; train 0")
        budgets = " / ".join(f"{c['actual_supervised_tokens']:,}" for c in tr["checkpoints"])
        lines.append(f"| {tr['array_index']} | {tr['student']} | {tr['position']} | "
                     f"{tr['pool']['D_U_pool']:,} | {seed} | {budgets} |")
    return "\n".join(lines) + "\n"


def amendment_text(plan):
    registration = plan["amendment"]
    seeds = "; ".join(f"{s}: " + ", ".join(str(t["training_seed"]) for t in plan["trajectories"]
                     if t["student"] == s and t["position"] == "above") for s in a3.STUDENTS)
    return f"""# A12 preregistration amendment 1 — capped upper tier

Registered before any A12 training or outcomes. CPU preparation only; no jobs launched or commits made. This amendment supersedes only the upper-tier design and its associated predictions in prereg.md. The original preregistration is unchanged; the original plan is preserved at registration_0/plan.json.

The complete gpt-5.6-luna trace corpus has 600 source rows per domain and 1,781 retained examples after the registered full recipe and empty-target deletion. Both student tokenizers yield **158,391 supervised completion tokens**, the maximum independent data available in this corpus (math 65,190; code 85,417; QA 7,784). The original upper targets, 343,000 for gemma3-1b and 315,000 for gemma3-4b, cannot be realized. Repeated exposure cannot increase independent data.

The tier named **above** now uses all 158,391 supervised tokens, with n_per_domain=600. This name denotes the largest available tier; it does not guarantee placement above the predicted boundary. **The upper tier now tests the boundary only up to 158,391 (about 158k) tokens.** Requirements beyond this cap remain untested. All-failing tiers retain a right-censored lower bound; the cap is not treated as a crossing or a feasible recommendation. Any conservative-baseline fallback uses this cap without a feasibility guarantee, and savings require measured success under the original scoring rule.

Each student retains three upper-tier trajectories with DIFFERENT training seeds: **{seeds}**. Pool membership is identical across these replicates; data_seed is null and --data-seed is omitted. V12 therefore uses --seed for initialization and initial/epoch shuffles. These replicates measure training randomness conditional on the full corpus, not independent pool draws. Below and near retain their exact pools, pool seeds, training seed 0, commands, checkpoints and predictions. Their replicate variation remains pool variation; mixed sources of variation must be reported explicitly, without treating upper replicates as fresh independent data.

Training-seed selection rule: {SEED_RULE} The former upper pool seeds are archived as original_data_seed; they are not used as pool seeds. The exact selected seeds and all achieved budgets are in task-table.md and plan.json.

All 18 trajectories are realizable. The unchanged corner recipe uses strict LoRA (rank 16, alpha 32, dropout 0, q/k/v/o/gate/up/down projections), LR 1e-4, AdamW, cosine with 3% warmup, bfloat16, batch 16, max length 1024, epochs 60 and a 678,000-processed-token schedule. Each trajectory has its own update-0 reference plus nominal 50k/100k/200k supervised-token checkpoints, realized within the original 0.5% tolerance at completed optimizer updates. The processed-token CLI thresholds are replayed exactly; no schedule extension or interpolation is introduced. D_U_pool and D_U_seen remain separate, particularly before the full corpus has been encountered. All six registered readouts, tolerances, scoring, censoring and missing-data rules remain unchanged. Replicate envelopes at the upper tier now summarize training seeds, while below/near envelopes summarize pool seeds.

The candidate boundary_loglinear and baselines fixed_reuse and student_isotonic use the **same frozen A10 boundary and companion delta fits**, training rows and source hashes. All 432 upper-tier prediction records (six trajectories × four checkpoints × six readouts × three methods) are regenerated at capped D_U and replayed actual T, with delta and boundary verdicts at every registered tolerance. The other 864 prediction records and all fitted coefficients are unchanged. Recommendations and comparator ceilings are recomputed under the existing rule. No A12 outcome enters fitting or selection.

Prediction seals use SHA256 of UTF-8 json.dumps(plan['predictions'], sort_keys=True, separators=(',', ':'), allow_nan=False).

- Original predictions SHA256: `{registration['original_predictions_sha256']}`
- **Amended predictions SHA256: `{plan['predictions_sha256']}`**
- Amended decisions SHA256: `{plan['decisions_sha256']}`
- Original plan file SHA256: `{registration['original_plan_sha256']}`
- Original prereg.md SHA256: `{registration['original_prereg_sha256']}`

Reproduce on CPU with `python -B analysis/a12_amend.py`; validate all 18 real V12 parsers with `python -B analysis/a12_run.py --validate-commands`. The SLURM array remains indices 0–17, one GPU per future task. scripts/a12_sync_list.txt includes this amendment, the original plan, amended plan, task table and parser validation alongside the frozen evaluation inputs. Execution and scoring check each registered training seed. This registration authorizes no launch by the preparation script.
"""


def main():
    require(not any((OUT / "runs").glob("*")), "Cannot amend after A12 runs exist")
    source = ORIGINAL if ORIGINAL.exists() else OUT / "plan.json"
    original = load_plan(source)
    require(original["predictions_sha256"] == ORIGINAL_PREDICTIONS_SHA256,
            "Amendment requires the original A12 prediction seal")
    require(original["n_realized"] == 12, "Expected original 12-realized, six-blocked design")
    for path, expected in original["source_sha256"].items():
        require(a3.sha256(ROOT / path) == expected, f"A10/trainer source changed: {path}")
    for path, expected in original["runtime_sha256"].items():
        if path not in ("analysis/a12_run.py", "analysis/a12_score.py"):
            require(a3.sha256(ROOT / path) == expected, f"Frozen runtime input changed: {path}")
    if not ORIGINAL.exists():
        ORIGINAL.parent.mkdir(parents=True, exist_ok=True)
        ORIGINAL.write_bytes(source.read_bytes())
    plan = copy.deepcopy(original)
    excluded = set(original["excluded_seeds"]) | {t["data_seed"] for t in original["trajectories"]
                                                if t["position"] != "above"}
    for student in a3.STUDENTS:
        tokenizer, identity = a3.load_tokenizer(student)
        require(identity == original["tokenizers"][student], "Frozen tokenizer changed")
        counter = a3.PoolCounter(tokenizer)
        for tr in plan["trajectories"]:
            if tr["student"] != student or tr["position"] != "above":
                continue
            pool, chosen, schedule = select_full_pool(counter, original["capacity"][student], tr["data_seed"], excluded)
            seed = pool["data_sampling_seed"]
            excluded.add(seed)
            tr.update(original_D_U_target=tr["D_U_target"], original_data_seed=tr["data_seed"],
                      D_U_target=CAP, data_seed=None, training_seed=seed, replicate_kind="training_seed",
                      status="realized", pool=a3.public(pool),
                      pool_example_ids=sorted(x["example_id"] for x in pool["_ledger"]),
                      schedule=schedule, checkpoints=checkpoints(pool, chosen, schedule))
            argv = ["--student", student, "--teacher", "gpt-5.6-luna", "--recipe", "full",
                    "--domains", "math,qa,code", "--training-mode", "lora", "--n-per-domain", "600",
                    "--seed", str(seed), "--lr", "1e-4", "--epochs", "60", "--schedule-tokens", "678000",
                    "--save-trajectory", "--stop-after-trajectory", "--trajectory-tokens",
                    *[str(c["processed_tokens"]) for c in tr["checkpoints"][1:]],
                    "--device", "cuda:0", "--output-suffix", tr["trajectory_id"]]
            name = a3.make_run_name("gpt-5.6-luna", "full", 600, tr["trajectory_id"], seed, "lora", None)
            tr.update(v12_argv=argv, v12_command=shlex.join(["python", "-B", "analysis/v12_distill.py", *argv]),
                      native_run_dir=f"{tr['output_dir']}/v12/{student}/{name}")
            for c in tr["checkpoints"]:
                c["eval_path"] = f"{tr['native_run_dir']}/trajectory/update-{c['update']:08d}/eval.json"
            print(tr["trajectory_id"], "full pool", CAP, "training seed", seed, flush=True)
    boundaries = {(f["readout"], f["method"], f["tau"]): f for f in original["frozen_boundary_fits"]}
    deltas = {(f["readout"], f["method"]): f for f in original["frozen_delta_fits"]}
    plan["predictions"] = prep.make_predictions(plan["trajectories"], boundaries, deltas)
    unchanged = {t["trajectory_id"] for t in original["trajectories"] if t["position"] != "above"}
    require([p for p in plan["predictions"] if p["trajectory_id"] in unchanged] ==
            [p for p in original["predictions"] if p["trajectory_id"] in unchanged], "Below/near predictions changed")
    plan.update(status="prepared_amendment_1", launch_ready=True, n_realized=18, blockers=[],
                pool_rule=original["pool_rule"] + "; amendment 1 above: all 600 rows/domain, data_seed unset, training-seed shuffles",
                search_rule=original["search_rule"] + " Amendment 1 above: " + SEED_RULE)
    plan["amendment"] = {"number": 1, "path": str((OUT / "prereg_amendment_1.md").relative_to(ROOT)),
                         "original_plan": str(ORIGINAL.relative_to(ROOT)), "original_plan_sha256": a3.sha256(ORIGINAL),
                         "original_prereg_sha256": a3.sha256(OUT / "prereg.md"),
                         "original_predictions_sha256": original["predictions_sha256"],
                         "cap_supervised_tokens": CAP, "training_seed_selection_rule": SEED_RULE,
                         "replicate_kind": {"below": "pool_seed", "near": "pool_seed", "above": "training_seed"},
                         "protocol_seed_rule": "protocol.seed=0 for unchanged below/near; above overrides with trajectory.training_seed"}
    plan["pool_overlap"] = []
    for i, left in enumerate(plan["trajectories"]):
        for right in plan["trajectories"][i + 1:]:
            a, b = set(left["pool_example_ids"]), set(right["pool_example_ids"])
            plan["pool_overlap"].append({"left": left["trajectory_id"], "right": right["trajectory_id"],
                                         "shared_examples": len(a & b), "jaccard": len(a & b) / len(a | b)})
    plan["predictions_sha256"] = digest(plan["predictions"])
    plan["decisions"] = build_decisions(plan)
    plan["decisions_sha256"] = digest(plan["decisions"])
    plan["runtime_sha256"] = {p: a3.sha256(ROOT / p) for p in original["runtime_sha256"]}
    prep.write(OUT / "plan.json", plan)
    (OUT / "prereg_amendment_1.md").write_text(amendment_text(plan))
    (OUT / "task-table.md").write_text(task_table(plan))
    print("Predictions SHA256:", plan["predictions_sha256"])


if __name__ == "__main__":
    main()
