#!/usr/bin/env python3
"""Reconstruct a results/ tree from data_mirror/ so the generators can run.

The analysis scripts read their inputs from `results/<experiment>/<file>`, which is
the layout of the working tree the measurements were taken in. This repository
ships those inputs under `data_mirror/<experiment>/<file>`, with two differences:
large summaries are gzipped, and `@` in a file name is written `--` so the names
stay portable. This script materialises `results/` from the mirror, writing both
spellings of a state-tagged name so a script finds the one it asks for, and
expanding the gzipped summaries. It also creates the directories the table and
figure generators write into.

Existing files are never overwritten, so it is safe to re-run, and it never
touches `data_mirror/`.

    python3 bootstrap_results.py [--dry-run] [--legacy-layout]

New paper generators write to generated/{tables,figs}. Older versioned generators
use paper/paper/{tables,figs} and paper/code; opt into those compatibility paths
with --legacy-layout only when paper/ does not contain a separate manuscript repo.
"""
from __future__ import annotations

import gzip
from itertools import product
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIRROR = ROOT / "data_mirror"
RESULTS = ROOT / "results"
OUTPUT_DIRS = (ROOT / "generated" / "tables", ROOT / "generated" / "figs")
# A11 authenticates these files at their original repository-relative locations.
# Keep their authoritative copies in the experiment mirror, just like results.
A11_RUNTIME = MIRROR / "a11-efficiency-confirmation" / "runtime"


def targets(relative: Path) -> list[Path]:
    """Both spellings of a mirrored name: the portable one and the original."""
    # V12 stores portable student_tag identifiers inside every eval. Its loaders
    # enumerate run directories and require exact path/metadata identity; making
    # @step aliases would both duplicate observations and violate that identity.
    if relative.parts[0] == "v12-distill":
        return [RESULTS / relative]
    # State tags occur in directory names as well as basenames (e.g. V53 dense
    # snapshots). Preserve every spelling a frozen loader can request.
    parts = [{part, part.replace("--step", "@step")} for part in relative.parts]
    return [RESULTS.joinpath(*path) for path in sorted(product(*parts))]


def main() -> int:
    dry = "--dry-run" in sys.argv
    legacy = "--legacy-layout" in sys.argv
    output_dirs = (tuple(ROOT / "paper" / "paper" / kind for kind in ("tables", "figs"))
                   if legacy else OUTPUT_DIRS)
    if not MIRROR.is_dir():
        print(f"no data_mirror/ at {MIRROR}")
        return 1

    written, expanded, skipped = 0, 0, 0
    for source in sorted(MIRROR.rglob("*")):
        if source.is_dir():
            continue
        relative = source.relative_to(MIRROR)
        if source.suffix == ".gz":
            for target in targets(relative.with_suffix("")):
                if target.exists():
                    skipped += 1
                    continue
                if not dry:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(source, "rb") as handle, open(target, "wb") as sink:
                        shutil.copyfileobj(handle, sink)
                expanded += 1
            continue
        for target in targets(relative):
            if target.exists():
                skipped += 1
                continue
            if not dry:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            written += 1

    for directory in output_dirs:
        if not dry:
            directory.mkdir(parents=True, exist_ok=True)

    if A11_RUNTIME.is_dir():
        for source in sorted(A11_RUNTIME.rglob("*")):
            if not source.is_file():
                continue
            target = ROOT / source.relative_to(A11_RUNTIME)
            if target.exists():
                skipped += 1
            else:
                if not dry:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                written += 1

    # Generated tables that later generators read back as cross-checks.
    table_source = MIRROR / "tables"
    if table_source.is_dir():
        for source in sorted(table_source.glob("*.tex")):
            target = output_dirs[0] / source.name
            if not target.exists() and not dry:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)


    # Some scripts read and mirror themselves under a `code/` tree, the layout an
    # earlier revision of this repository used. Provide it as real directories with
    # copies: a symlink would be refused by the generators' own output guard, and
    # editing those scripts is not an option because frozen records name their digests.
    if not dry and legacy:
        bases = (ROOT / "code", ROOT / "paper" / "code")
        for base in bases:
            for name in ("analysis", "tests", "configs"):
                source = ROOT / name
                target = base / name
                if not source.is_dir():
                    continue
                target.mkdir(parents=True, exist_ok=True)
                for item in source.iterdir():
                    if item.is_file() and not (target / item.name).exists():
                        shutil.copy2(item, target / item.name)

    verb = "would write" if dry else "wrote"
    print(f"{verb} {written} file(s), {verb.split()[-1]} {expanded} gzipped summary/summaries, "
          f"left {skipped} existing file(s) alone")
    print(f"generators write LaTeX into {output_dirs[0].relative_to(ROOT)} and "
          f"figures into {output_dirs[1].relative_to(ROOT)}")
    print("Inputs that were never mirrored cannot be reconstructed; the ledger in "
          "docs/RESULTS_LEDGER.md names the artifact behind every reported number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
