#!/usr/bin/env bash
# Caller owns CUDA_VISIBLE_DEVICES. Authoring: --selftest / --dry-run only.
# CPU sequence: v70_distill_confirm.py register; develop; freeze.
# freeze writes results/v70-distill-confirm/FREEZE_V70 containing freeze.json SHA256.
# --throughput uses a separate p2v3conf_throughput suffix; its schedule remains 1M.
# It is a timing pilot, not a resumable confirmation trajectory. Full runs start fresh.
set -euo pipefail
V70_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V70_PROJECT_ROOT="$V70_SCRIPT_DIR"
# The paper mirror places this script beside analysis/. Prefer the enclosing
# project with its register, including when invoked from that mirror.
while [[ "$V70_PROJECT_ROOT" != / ]]; do
  if [[ -f "$V70_PROJECT_ROOT/results/v47-p2-register/register.json" && -f "$V70_PROJECT_ROOT/analysis/v70_distill_confirm.py" ]]; then
    break
  fi
  V70_PROJECT_ROOT="$(dirname "$V70_PROJECT_ROOT")"
done
if [[ "$V70_PROJECT_ROOT" == / ]]; then
  V70_PROJECT_ROOT="$(dirname "$V70_SCRIPT_DIR")"
  [[ -f "$V70_PROJECT_ROOT/analysis/v70_distill_confirm.py" ]] || V70_PROJECT_ROOT="$V70_SCRIPT_DIR"
fi
cd "$V70_PROJECT_ROOT"
args=(plan --json)
dry=0
while (($#)); do
  case "$1" in
    --dry-run) args+=(--dry-run); dry=1; shift ;;
    --throughput) args+=(--throughput); shift ;;
    --student) args+=(--student "${2:?--student needs gemma3-270m or gemma3-1b}"); shift 2 ;;
    --selftest) exec python3 -B analysis/v70_distill_confirm.py --selftest ;;
    -h|--help)
      echo 'Usage: CUDA_VISIBLE_DEVICES=<caller GPU> scripts/run_v70_confirm.sh [--student gemma3-270m|gemma3-1b] [--throughput] [--dry-run]'
      echo 'CPU selftest: scripts/run_v70_confirm.sh --selftest'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if (( ! dry )); then
  : "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the caller-selected GPU}"
fi
# The complete plan must parse before any run is dispatched. No eval or shell
# command interpolation: subprocess gets the explicit V12 argv list.
python3 -B analysis/v70_distill_confirm.py "${args[@]}" | python3 -c '
import json, os, pathlib, shlex, subprocess, sys, time
plans = json.load(sys.stdin)
dry = sys.argv[1] == "1"
for item in plans:
    if dry:
        print(item["status"] + ": " + shlex.join(item["command"]) + " > " + shlex.quote(item["log"]) + " 2>&1")
        continue
    # Recheck sentinel and every frozen input before each individual trajectory.
    check = [sys.executable, "-B", "analysis/v70_distill_confirm.py", "plan", "--json", "--student", item["student"]]
    if item["throughput"]:
        check.append("--throughput")
    fresh = json.loads(subprocess.check_output(check, text=True))
    current = next(p for p in fresh if p["student"] == item["student"] and p["seed"] == item["seed"])
    if current["status"] == "skip":
        print("SKIP " + item["directory"], flush=True)
        continue
    log = pathlib.Path(item["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print("START " + item["directory"] + " log=" + str(log), flush=True)
    # Exclusive log publication also prevents simultaneous duplicate launches.
    with log.open("x") as stream:
        stream.write("COMMAND " + shlex.join(item["command"]) + "\n")
        stream.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"] + "\n")
        stream.flush()
        run = subprocess.run(item["command"], stdout=stream, stderr=subprocess.STDOUT)
        elapsed = time.monotonic() - started
        if run.returncode:
            print("FAILED rc=" + str(run.returncode) + " log=" + str(log), flush=True)
            sys.exit(run.returncode)
        directory = pathlib.Path(item["directory"])
        result = json.loads((directory / "eval.json").read_text())
        training = json.loads((directory / "train_log.json").read_text())
        if result["trajectory_tokens_unreached"]:
            raise RuntimeError("Requested trajectory checkpoints were not reached")
        tokens = result["processed_tokens"]
        tps = tokens / max(elapsed, 1e-9)
        report = {"student": item["student"], "data_seed": item["seed"], "processed_tokens": tokens,
            "completion_tokens": result["completion_tokens_seen"], "wall_seconds": elapsed,
            "processed_tokens_per_second_including_load_and_evals": tps,
            "projected_hours_per_1M_trajectory": 1000000 / tps / 3600,
            "projected_hours_six_1B_trajectories_at_this_rate": 6000000 / tps / 3600,
            "training_loop_tokens_per_second_including_checkpoint_evals": tokens / training["wall_time_seconds"],
            "projection_note": "Linear timing estimate includes load/evaluation overhead; 270M timing is not measured by the 1B pilot",
            "throughput_pilot": item["throughput"]}
        stream.write("TIMING " + json.dumps(report) + "\n")
        (directory / "v70_timing.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
' "$dry"
