#!/usr/bin/env bash
# Usage: nohup bash scripts/rsi_r1_probe_panel.sh GPU configs/r1_probe_panel_models.txt &
# PYTHON_BIN may select the installed experiment environment. Resume by rerunning.
# R1_RESOLVED_LIST / R1_LOG_DIR / R1_OUTPUT_ROOT optionally relocate artifacts.
set -euo pipefail
if [[ $# != 2 || ! $1 =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 GPU_INDEX MODEL_LIST" >&2
    exit 2
fi
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
LIST=$(realpath -- "$2")
PYTHON_BIN=${PYTHON_BIN:-"$ROOT/.venv-gemma4/bin/python"}
cd -- "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$1"
exec "$PYTHON_BIN" - "$LIST" "$ROOT" <<'PY'
import fcntl
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time

source, root = map(Path, sys.argv[1:])
if not source.is_file():
    raise SystemExit(f"Model list does not exist: {source}")
resolved = Path(os.environ.get("R1_RESOLVED_LIST", root / "configs/r1_probe_panel_models.resolved.txt"))
logs = Path(os.environ.get("R1_LOG_DIR", root / "logs"))
outputs = Path(os.environ.get("R1_OUTPUT_ROOT", root / "results/r1-probe-panel"))
if source.resolve() == resolved.resolve():
    raise SystemExit("Input must differ from the resolved copy; the original list is never mutated")
logs.mkdir(parents=True, exist_ok=True)
resolved.parent.mkdir(parents=True, exist_ok=True)


def read_rows(path):
    rows = {}
    if not path.exists():
        return rows
    for number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.partition("#")[0].split()
        if not fields:
            continue
        if len(fields) not in (3, 4) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", fields[0]):
            raise ValueError(f"Invalid line {number} in {path}")
        if fields[0] in rows:
            raise ValueError(f"Duplicate tag in {path}: {fields[0]}")
        rows[fields[0]] = fields
    return rows


def write_rows(rows):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=resolved.parent,
                                         prefix=".r1-resolved-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write("# Recorded revisions for the R1 probe panel; original list is unchanged.\n")
            for fields in rows.values():
                stream.write(" ".join(fields) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, resolved)
        fd = os.open(resolved.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


requested = read_rows(source)
failures = 0
for tag, fields in requested.items():
    with (logs / f"r1_probe_{tag}.log").open("a", buffering=1) as log:
        try:
            model_id, revision = fields[1:3]
            hostname = socket.gethostname()
            # Qwen on rsi only (user, 2026-11-07); rsi reports hostname sn4622131043.
            if "qwen" in model_id.lower() and not (hostname in ("sn4622131043",) or hostname.startswith("rsi")):
                raise RuntimeError(f"Qwen runs only on rsi; refusing on {hostname}")
            # Lock only revision recording, so separate GPU drivers can run
            # different tags while sharing one persistent set of pinned SHAs.
            with resolved.with_suffix(resolved.suffix + ".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                recorded = read_rows(resolved)
                for key, row in requested.items():
                    if key not in recorded:
                        recorded[key] = row.copy()
                previous = recorded[tag]
                if previous[1] != model_id or previous[3:] != fields[3:]:
                    raise RuntimeError(f"Resolved-list identity conflict for {tag}")
                if revision == "RESOLVE":
                    if previous[2] == "RESOLVE":
                        from huggingface_hub import HfApi
                        revision = HfApi().model_info(model_id).sha
                        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
                            raise RuntimeError(f"Invalid Hub commit for {model_id}: {revision!r}")
                    else:
                        revision = previous[2]
                elif previous[2] not in (revision, "RESOLVE"):
                    raise RuntimeError(f"Resolved-list revision conflict for {tag}")
                recorded[tag] = [tag, model_id, revision, *fields[3:]]
                write_rows(recorded)  # Commit the pin BEFORE starting measurements.
            command = [sys.executable, "-m", "analysis.r1_probe_panel",
                       "--model", model_id, "--revision", revision, "--tag", tag,
                       "--device", "cuda:0", "--reference-device", "cpu",
                       "--out", str(outputs / tag)]
            if len(fields) == 4:
                command += ["--expect-dense", fields[3]]
            print(f"[driver] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {model_id}@{revision}", file=log)
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
            print(f"[driver] exit={result.returncode}", file=log)
            if result.returncode:
                failures += 1
            print(f"[r1 driver] {tag}: exit={result.returncode}; {log.name}", flush=True)
        except Exception as exc:
            failures += 1
            print(f"[driver] {type(exc).__name__}: {exc}", file=log)
            print(f"[r1 driver] {tag}: failed; {log.name}", flush=True)
raise SystemExit(1 if failures else 0)
PY
