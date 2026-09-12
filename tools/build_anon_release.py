#!/usr/bin/env python3
"""Build an anonymised, submission-ready snapshot of the project.

Usage (from the code repository, i.e. the directory holding data_mirror/ and docs/):
    python3 tools/build_anon_release.py [--out ../anon_release] [--run-tests]

The snapshot reproduces the layout the code actually runs in, so that scripts and
tests resolve their paths:

    analysis/  tests/  configs/  scripts/  results/  data_mirror/  paper/docs/

Sources
  * the live working tree (--live, default: the parent of this repository) gives
    analysis/, tests/, configs/, scripts/, pytest.ini and the text artifacts under
    results/;
  * this repository gives docs/ (evidence records) and data_mirror/ (frozen
    registers, predictions, comparisons).

Never copied: credentials, run logs, virtual environments, model weights, LoRA
adapters, downloaded checkpoints, dataset caches, the trash directory, the
authors' internal planning and correspondence notes, and work in progress that
supports no claim in the paper.

The build then substitutes author, host, institution and path strings, drops tool
attribution trailers, writes the reader-facing files, initialises a fresh git
repository with one anonymous commit, validates that nothing was broken
structurally, and runs tools/check_anon.py, whose deny list is written
independently of the substitution table below.

Model identifiers containing a vendor name (claude-sonnet-4-6, gpt-5.6-luna) are
scientific provenance and are deliberately left alone.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Substitutions. Ordered, longest first: the earlier entry wins.
# --------------------------------------------------------------------------- #
SUBS: list[tuple[str, str]] = [
    ("/home/xueqi/hq/projects/scaling-down-law/", ""),
    ("/home/xueqi/hq/projects/scaling-down-law", "."),
    ("/home/xueqi/hq/projects", "$PROJECTS"),
    ("/home/xueqi/.cache/huggingface", "$HOME/.cache/huggingface"),
    ("/home/xueqi/hq", "$PROJECT_ROOT"),
    ("/home/xueqi", "$HOME"),
    ("/blue/yd24f.fsu/xc25.fsu/hq", "$CLUSTER_PROJECT"),
    ("/blue/fsu-compsci-dept/xc25.fsu/hq", "$CLUSTER_PROJECT"),
    ("/blue/yd24f.fsu/xc25.fsu", "$CLUSTER_PROJECT"),
    ("/blue/fsu-compsci-dept/xc25.fsu", "$CLUSTER_PROJECT"),
    ("https://github.com/XueqiC/Scaling_down_law_paper", "the paper source repository"),
    ("https://github.com/XueqiC/ICLR27_scaling_down", "this repository"),
    ("XueqiC/Scaling_down_law_paper", "the paper source repository"),
    ("XueqiC/ICLR27_scaling_down", "this repository"),
    ("Xueqi Cheng", "Anonymous Author"),
    ("Xueqi", "Anonymous"),
    ("xueqi", "anon"),
    ("xc25.fsu", "clusteruser"),
    ("yd24f.fsu", "clusteraccount"),
    ("fsu-compsci-dept", "department-account"),
    ("UF HiPerGator", "the shared cluster"),
    ("the HiPerGator", "the shared cluster"),
    ("HiPerGator", "the shared cluster"),
    ("job_hpg_", "job_cluster_"),
    ("_hpg_", "_cluster_"),
    ("hpg_", "cluster_"),
    ("HPG_BASE", "CLUSTER_BASE"),
    ("HPG", "CLUSTER"),
    ("PRC-developed", "restricted-origin"),
    ("SDL_ALLOW_PRC", "SDL_ALLOW_RESTRICTED"),
    ("non-PRC", "unrestricted"),
    # Identifier-safe: PRC also appears inside Python names such as _PRC_PATTERN,
    # so the replacement must not introduce a hyphen there.
    ("_PRC_", "_RESTRICTED_"),
    ("PRC", "restricted"),
]

# Regex substitutions, applied after the plain table: catch path shapes that vary
# (cluster allocation names change, captured tracebacks carry other users' homes).
REGEX_SUBS: list[tuple[str, str]] = [
    (r"/blue/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", "$CLUSTER_PROJECT"),
    (r"/home/[A-Za-z0-9._-]+", "$HOME"),
    (r"(?<![A-Za-z0-9])yd24f(?![A-Za-z0-9])", "clusteraccount"),
]

# Whole-word rewrites, for nicknames that are also common substrings.
WORD_SUBS: list[tuple[str, str]] = [
    (r"\bRAI-only\b", "workstation-only"),
    (r"\bRAI\b", "the workstation"),
    (r"\brai\b", "the workstation"),
    (r"\bhpg\b", "the cluster"),
]

# Trailers are removed by matching only the trailer text. Matching to the end of
# the line would be wrong: in the shell drivers these trailers sit inside a quoted
# commit message whose closing quote and following commands share the line.
# Applied to file and directory names. Kept separate from SUBS because a path must
# not gain a space, and separate from WORD_SUBS for the same reason.
PATH_SUBS: list[tuple[str, str]] = [
    (r"(?<![A-Za-z])hpg(?![A-Za-z])", "cluster"),
    (r"(?<![A-Za-z])rai(?![A-Za-z])", "workstation"),
    (r"(?<![A-Za-z])RAI(?![A-Za-z])", "workstation"),
    (r"Xueqi", "Anonymous"),
    (r"xueqi", "anon"),
    (r"xc25", "clusteruser"),
    (r"yd24f", "clusteraccount"),
    (r"fsu-compsci-dept", "department-account"),
    (r"(?<![A-Za-z])PRC(?![A-Za-z])", "restricted"),
]


DROP_LINES: list[re.Pattern[str]] = [
    re.compile(r"Co-Authored-By:[^\"\n]*"),
    re.compile(r"Claude-Session:[^\"\n]*"),
    re.compile(r"https://claude\.ai/code/session[^\s\"]*"),
]

# --------------------------------------------------------------------------- #
# What to copy.
# --------------------------------------------------------------------------- #
CODE_SUFFIXES = {".py", ".sh", ".slurm", ".json", ".jsonl", ".ini", ".cfg",
                 ".toml", ".yaml", ".yml", ".md", ".txt"}
RESULT_SUFFIXES = {".json", ".md", ".csv", ".sha256", ".npz"}
GZIP_OVER_BYTES = 2_000_000

# Work in progress for the next round; supports no claim in the paper.
EXCLUDE_LIVE_FILES = {
    "analysis/descriptor_bv.py",
    "analysis/descriptor_bv.schema.md",
    "analysis/descriptor_bv_selftest.py",
    "tests/test_descriptor_bv.py",
}
EXCLUDE_LIVE_PREFIXES = ("analysis/verify/",)

# Internal planning, progress and correspondence notes.
EXCLUDE_DOCS = {
    "docs/A100_EXPLORATION_PLAN.md",
    "docs/A100_SUPPLEMENT_PLAN.md",
    "docs/CLOSEOUT_PLAN.md",
    "docs/DAY_SUMMARY_2026-09-11.md",
    "docs/EXPERIMENT_PLAN.md",
    "docs/MANUSCRIPT_REWRITE_MAP.md",
    "docs/NEXT_ROUND_PLAN_2026-09-12.md",
    "docs/NIGHT_PROGRESS_2026-09-11.md",
    "docs/PAPER_REFRAME_PLAN.md",
    "docs/PROGRESS.md",
    "docs/RESTRUCTURE_MAP_2026-09-10.md",
    "docs/RESTRUCTURE_SUMMARY_2026-09-10.md",
    "docs/REWRITE_SUMMARY.md",
    "docs/ROUND3_SUMMARY_2026-09-10.md",
    "docs/ROUND8_SUMMARY_2026-09-11.md",
}
EXCLUDE_DOCS_PREFIXES = ("docs/inbox/",)

# Directory names never descended into, wherever they appear.
SKIP_DIRS = {
    "__pycache__", ".git", ".venv", ".venv-gemma4", "_trash", "logs", "backups",
    "node_modules", ".pytest_cache", ".mypy_cache", "snapshots", "adapter",
    "hf-cache", "trajectory_adapters",
}

# --------------------------------------------------------------------------- #
# Reader-facing files.
# --------------------------------------------------------------------------- #
README = """# Capability-conditioned scaling-down laws for LLM compression

Anonymised code and measurement artifact for the submitted paper. It holds the
measurement and analysis pipeline, the frozen predictions, the measurements they
were scored against, and the evidence records the paper's tables and figures are
generated from.

## Layout

| Path | Contents |
|---|---|
| `analysis/` | measurement and analysis pipeline, one module per experiment, named `v<N>_<topic>.py` |
| `tests/` | offline test suite; `REPRODUCE.md` says what passes without the model weights |
| `configs/` | sweep configurations |
| `scripts/` | grid drivers and cluster job scripts, as the measurements were launched |
| `results/` | per-experiment outputs: reports, summaries, registers, measurements (text artifacts only) |
| `data_mirror/` | the frozen registers, predictions and comparison files behind the paper's tables |
| `paper/docs/` | results ledger, claim-evidence matrix, measurement definitions, pre-registrations, reference verification, reviewer self-audit |

## Reading the evidence

`paper/docs/RESULTS_LEDGER.md` is the spine: one entry per experiment, in order,
recording what was frozen, when, what was measured, and how it came out, including
the entries that record failures and the ones that correct earlier entries.
`paper/docs/CLAIM_EVIDENCE_MATRIX.md` maps each claim in the paper to the artifact
supporting it. `paper/docs/EXPERIMENT_MANIFEST.md` and `paper/docs/prereg/` hold
pre-registered plans.

A file whose header says a script generated it should not be edited by hand; change
the generator and regenerate.

## What is not here

Model weights, LoRA adapters, downloaded checkpoints and dataset caches, all public
or regenerable; teacher trace files, which are third-party model outputs;
credentials; run logs; and the authors' internal planning and correspondence notes.
`ANONYMIZATION.md` lists the categories of string rewritten and what was left out.
"""

REPRODUCE = """# Reproducing

## 1. Environment

```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # analysis only, CPU
pip install -r requirements-gpu.txt      # additionally, to re-measure models
```

Measurements were taken with the GPU stack pinned in `requirements-gpu.txt` on
single GPUs (CUDA 12.8). Analysis, table generation and figure generation need
neither a GPU nor network access.

## 2. Expand the compressed summaries (once)

```
python3 expand_compressed.py
```

A few large summary files ship gzipped; some scripts and tests open them by their
plain name.

## 3. Regenerating a table or figure (CPU, minutes)

```
python3 analysis/<generator>.py
```

Generators read the frozen JSON under `data_mirror/` and `results/`. Each generated
file carries its input paths and their SHA-256 digests in a header comment, so a
regenerated table can be compared line by line with the one in the paper.

## 3. The test suite

```
python3 -m pytest tests
```

The suite was written against the authors' full working tree. In this snapshot the
tests that depend only on the shipped text artifacts pass; tests that read model
weights, LoRA adapters, dataset caches, teacher traces or the authors' internal
notes cannot pass here, because those files are not redistributable or not
regenerable from text. The counts observed when the snapshot was built are recorded
at the end of this file.

## 4. Re-measuring a model response (GPU)

1. Resolve a checkpoint through `analysis/model_registry.py`; every model is a
   public Hugging Face checkpoint pinned by revision.
2. Pruning: `analysis/v6_capability_geometry.py` applies global magnitude pruning at
   a stored threshold and scores the probe sets.
3. Quantization: `analysis/v10_quantization.py` (per-channel round-to-nearest) and
   `analysis/v54_quant_group.py` (grouped round-to-nearest).
4. Distillation: `analysis/v12_distill.py` trains a LoRA student on teacher traces.
   Generating traces needs an API endpoint and credentials, supplied through a
   `.secrets.env` file that is not part of this artifact.
5. `scripts/` holds the exact grid drivers and cluster job scripts that were used.

## 5. Freeze discipline

In every test the paper labels independent, predictions were written and hashed
before the corresponding measurement. The register and freeze files under
`data_mirror/` and `results/` carry those digests and timestamps. Nothing
overwrites a frozen file; corrections are added as new versions and recorded in the
ledger.
"""

LICENSE = """# License for peer review

This snapshot is provided for peer review of the accompanying anonymous submission.
The authors intend to release the code under a permissive open-source license on
publication.

Third-party material keeps its own terms: model checkpoints and datasets are
identified in `analysis/model_registry.py` and
`paper/docs/MEASUREMENT_DEFINITIONS.md`, and are not redistributed here.
"""

GITIGNORE = """__pycache__/
*.pyc
.venv/
.pytest_cache/
"""

EXPANDER = """#!/usr/bin/env python3
\"\"\"Expand the compressed result files.

A handful of summary files are stored gzipped to keep this artifact small. Some
analysis scripts and tests open them by their plain name. Run this once:

    python3 expand_compressed.py

It writes <name>.json next to each <name>.json.gz and leaves the archive in place,
so it is safe to run more than once.
\"\"\"
from __future__ import annotations

import gzip
import shutil
from pathlib import Path


def main() -> None:
    written = 0
    for archive in sorted(Path(__file__).resolve().parent.rglob("*.gz")):
        target = archive.with_suffix("")
        if target.exists():
            continue
        with gzip.open(archive, "rb") as source, open(target, "wb") as sink:
            shutil.copyfileobj(source, sink)
        written += 1
    print(f"expanded {written} file(s)")


if __name__ == "__main__":
    main()
"""

REQUIREMENTS = """# Analysis, table and figure generation: CPU, no network.
numpy==1.26.4
scipy==1.16.3
matplotlib==3.10.3
pytest==7.4.0
"""

REQUIREMENTS_GPU = """# Additionally needed to re-measure model responses on a single GPU.
# The paper's measurements were taken with this stack on CUDA 12.8.
torch==2.10.0
transformers==4.57.3
datasets==4.4.1
peft==0.18.0
accelerate==1.12.0
safetensors==0.7.0
huggingface-hub==0.36.0
sentencepiece==0.2.1
"""


def anon_path(rel: str) -> str:
    for pattern, new in PATH_SUBS:
        rel = re.sub(pattern, new, rel)
    return rel


def scrub(text: str) -> tuple[str, int]:
    hits = 0
    for pattern in DROP_LINES:
        text, n = pattern.subn("", text)
        hits += n
    for old, new in SUBS:
        if old in text:
            hits += text.count(old)
            text = text.replace(old, new)
    for pattern, new in REGEX_SUBS:
        text, n = re.subn(pattern, new.replace("\\", "\\\\"), text)
        hits += n
    for pattern, new in WORD_SUBS:
        text, n = re.subn(pattern, new, text)
        hits += n
    return text, hits


def walk(root: Path, top: str) -> list[str]:
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root / top):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            out.append(str((Path(dirpath) / name).relative_to(root)))
    return sorted(out)


def copy_text(src: Path, dst: Path, changed: list[tuple[str, int]], rel: str,
              gzip_big: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = src.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        shutil.copy2(src, dst)
        return
    text, hits = scrub(text)
    if gzip_big and len(text.encode("utf-8")) > GZIP_OVER_BYTES:
        with open(str(dst) + ".gz", "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
                handle.write(text.encode("utf-8"))
    else:
        dst.write_text(text, encoding="utf-8")
    if hits:
        changed.append((rel, hits))


def copy_gz(src: Path, dst: Path, changed: list[tuple[str, int]], rel: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    raw = gzip.decompress(src.read_bytes())
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        shutil.copy2(src, dst)
        return
    text, hits = scrub(text)
    with open(dst, "wb") as out:
        with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as handle:
            handle.write(text.encode("utf-8"))
    if hits:
        changed.append((rel, hits))


def validate(out: Path) -> list[str]:
    problems: list[str] = []
    for path in sorted(out.rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as exc:
            problems.append(f"{path.relative_to(out)}: python syntax: {exc}")
    for path in sorted(out.rglob("*.sh")):
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if result.returncode != 0:
            problems.append(f"{path.relative_to(out)}: shell syntax: {result.stderr.strip()[:120]}")
    for path in sorted(out.rglob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            problems.append(f"{path.relative_to(out)}: invalid json: {exc}")
    for path in sorted(out.rglob("*.json.gz")):
        try:
            json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{path.relative_to(out)}: invalid gzipped json: {exc}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../anon_release")
    parser.add_argument("--live", default="..", help="working tree holding analysis/, tests/, results/")
    parser.add_argument("--include-traces", action="store_true",
                        help="also copy teacher trace files (third-party model outputs)")
    parser.add_argument("--keep-git", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    live = (repo / args.live).resolve()
    out = (repo / args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    changed: list[tuple[str, int]] = []
    counts: dict[str, int] = {}

    for top in ("analysis", "tests", "configs", "scripts"):
        n = 0
        for rel in walk(live, top):
            if rel in EXCLUDE_LIVE_FILES or rel.startswith(EXCLUDE_LIVE_PREFIXES):
                continue
            src = live / rel
            fixture = rel.startswith("tests/fixtures/")
            if src.suffix not in CODE_SUFFIXES and not (fixture and src.suffix == ".pt"):
                continue
            if fixture and src.suffix == ".pt":
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out / rel)
                n += 1
                continue
            copy_text(src, out / anon_path(rel), changed, anon_path(rel))
            n += 1
        counts[top] = n
    if (live / "pytest.ini").exists():
        copy_text(live / "pytest.ini", out / "pytest.ini", changed, "pytest.ini")

    n = 0
    for rel in walk(live, "results"):
        src = live / rel
        if src.suffix not in RESULT_SUFFIXES:
            continue
        if not args.include_traces and "traces" in rel:
            continue
        target = out / anon_path(rel)
        if src.suffix == ".npz":
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        else:
            copy_text(src, target, changed, anon_path(rel), gzip_big=True)
        n += 1
    counts["results"] = n

    n = 0
    for rel in walk(repo, "docs"):
        if rel in EXCLUDE_DOCS or rel.startswith(EXCLUDE_DOCS_PREFIXES):
            continue
        src = repo / rel
        if src.suffix not in {".json", ".md", ".csv", ".txt"}:
            continue
        dst = out / "paper" / anon_path(rel)
        copy_text(src, dst, changed, str(dst.relative_to(out)))
        n += 1
    counts["paper/docs"] = n

    n = 0
    for rel in walk(repo, "data_mirror"):
        src = repo / rel
        if src.suffix == ".gz":
            copy_gz(src, out / anon_path(rel), changed, anon_path(rel))
        elif src.suffix in RESULT_SUFFIXES or src.suffix == ".txt":
            copy_text(src, out / anon_path(rel), changed, anon_path(rel), gzip_big=True)
        else:
            continue
        n += 1
    counts["data_mirror"] = n

    (out / "README.md").write_text(README, encoding="utf-8")
    (out / "REPRODUCE.md").write_text(REPRODUCE, encoding="utf-8")
    (out / "LICENSE-FOR-REVIEW.md").write_text(LICENSE, encoding="utf-8")
    (out / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (out / "expand_compressed.py").write_text(EXPANDER, encoding="utf-8")
    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (out / "requirements-gpu.txt").write_text(REQUIREMENTS_GPU, encoding="utf-8")

    manifest = [
        "# Anonymisation record", "",
        "This snapshot was built from the authors' working tree by a script that is",
        "not included here, because the script names the original strings. Nothing",
        "outside the directories listed in README.md was copied.", "",
        "## Categories rewritten", "",
        "| Category | Replacement |", "|---|---|",
        "| author name, in prose and inside paths | `Anonymous`, `anon` |",
        "| absolute paths under the author's home directory | `$HOME`, `$PROJECT_ROOT`, `$PROJECTS`, or a repository-relative path |",
        "| absolute paths on the shared cluster's filesystem | `$CLUSTER_PROJECT` |",
        "| cluster user and allocation accounts | `clusteruser`, `clusteraccount`, `department-account` |",
        "| institution and the named HPC system | `the shared cluster` |",
        "| workstation nickname | `the workstation` |",
        "| cluster nickname inside file and job names | `cluster` |",
        "| the authors' repository URLs | `this repository`, `the paper source repository` |",
        "| a local model-eligibility policy and its environment variable | `restricted`, `SDL_ALLOW_RESTRICTED` |",
        "", "Tool-attribution and session trailers were deleted wherever they appeared.",
        "Vendor names inside model identifiers are scientific provenance and were kept.",
        "", "## Files whose text changed", "",
        "Digests recorded inside the frozen files were computed on the originals. For the",
        "files listed here, anonymisation rewrote path strings, so a digest recomputed on",
        "this snapshot will differ from the digest stored inside it. No measurement value",
        "was altered.", "",
    ]
    for rel, hits in sorted(changed):
        manifest.append(f"- `{rel}` ({hits} substitution{'s' if hits != 1 else ''})")
    manifest += ["", "## Left out", "",
                 "- model weights, LoRA adapters, downloaded checkpoints, dataset caches",
                 "- credentials and run logs",
                 "- the authors' internal planning, progress and correspondence notes",
                 "- work in progress that supports no claim in the paper"]
    if not args.include_traces:
        manifest.append("- teacher trace files (third-party model outputs)")
    manifest.append("")
    (out / "ANONYMIZATION.md").write_text("\n".join(manifest), encoding="utf-8")

    total = sum(1 for p in out.rglob("*") if p.is_file() and ".git/" not in str(p))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and ".git/" not in str(p)) / 1_048_576
    print("copied: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"snapshot: {total} files, {size:.1f} MB, {len(changed)} files rewritten")

    problems = validate(out)
    if problems:
        print(f"STRUCTURAL CHECK FAILED: {len(problems)} file(s)")
        for problem in problems[:20]:
            print("  " + problem)
        return 1
    print("structural check passed: python, shell and json files parse")

    if args.run_tests:
        expanded: list[Path] = []
        for archive in sorted(out.rglob("*.gz")):
            target = archive.with_suffix("")
            if not target.exists():
                with gzip.open(archive, "rb") as source, open(target, "wb") as sink:
                    shutil.copyfileobj(source, sink)
                expanded.append(target)
        result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "--no-header",
                                 "-p", "no:cacheprovider"], cwd=out, capture_output=True, text=True)
        for target in expanded:
            target.unlink()
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        summary = lines[-1] if lines else "no summary"
        print("test suite: " + summary)
        with (out / "REPRODUCE.md").open("a", encoding="utf-8") as handle:
            handle.write("\n## Test counts observed when this snapshot was built\n\n"
                         f"After `python3 expand_compressed.py`:\n\n```\n{summary}\n```\n\n"
                         "The remaining failures need LoRA adapters, dataset caches, API credentials\n"
                         "or the authors' internal notes, none of which are part of this artifact.\n")

    if not args.keep_git:
        env = dict(os.environ,
                   GIT_AUTHOR_NAME="Anonymous", GIT_AUTHOR_EMAIL="anonymous@anonymous.invalid",
                   GIT_COMMITTER_NAME="Anonymous", GIT_COMMITTER_EMAIL="anonymous@anonymous.invalid",
                   GIT_AUTHOR_DATE="2026-01-01T00:00:00+0000", GIT_COMMITTER_DATE="2026-01-01T00:00:00+0000")
        subprocess.run(["git", "init", "-q"], cwd=out, check=True, env=env)
        subprocess.run(["git", "checkout", "-q", "-b", "main"], cwd=out, check=False, env=env)
        subprocess.run(["git", "add", "-A"], cwd=out, check=True, env=env)
        subprocess.run(["git", "-c", "user.name=Anonymous", "-c", "user.email=anonymous@anonymous.invalid",
                        "commit", "-q", "-m", "Anonymous artifact for peer review"], cwd=out, check=True, env=env)

    checker = repo / "tools" / "check_anon.py"
    if checker.exists():
        return subprocess.run([sys.executable, str(checker), str(out)]).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
