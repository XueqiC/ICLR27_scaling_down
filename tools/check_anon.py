#!/usr/bin/env python3
"""Independent anonymity check for a built snapshot.

Usage: python3 tools/check_anon.py ../anon_release

Scans file contents (including gzipped JSON), file and directory names, and the
git metadata of the snapshot for author, host, institution, credential and
private-link patterns. Exits non-zero and prints every hit. Written to be read
on its own: the deny list below is the definition of what "anonymous" means for
this artifact, and it is deliberately independent of the build script's
substitution table so that a missed case shows up here.
"""
from __future__ import annotations

import gzip
import re
import subprocess
import sys
from pathlib import Path

DENY: list[tuple[str, str]] = [
    ("author name", r"[Xx]ueqi"),
    ("author surname with given name", r"\bCheng,?\s+Xueqi\b"),
    ("cluster user", r"xc25"),
    ("cluster account", r"yd24f"),
    ("institution", r"\bFSU\b|Florida State|University of Florida|\bUF\b"),
    ("cluster name", r"[Hh]i[Pp]er[Gg]ator"),
    ("cluster nickname", r"(?<![A-Za-z])hpg(?![A-Za-z])"),
    ("cluster nickname upper", r"(?<![A-Za-z])HPG(?![A-Za-z])"),
    ("workstation nickname", r"(?<![A-Za-z])RAI(?![A-Za-z])"),
    ("home path", r"/home/[a-z]"),
    ("cluster path", r"/blue/"),
    ("owner repository", r"github\.com/(?!(?:EleutherAI|huggingface|HuggingFaceTB|google-research-datasets|openai|hotpotqa|allenai|bigcode-project|sahil2801)/)[A-Za-z0-9_-]+/"),
    ("chat platform", r"\b[Dd]iscord\b"),
    ("assistant session link", r"claude\.ai/code/session"),
    ("tool attribution trailer", r"Co-Authored-By:"),
    ("email address", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:edu|com|org)"),
    ("private key block", r"-----BEGIN [A-Z ]*PRIVATE KEY"),
    ("openai-style key", r"\bsk-[A-Za-z0-9]{20,}"),
    ("github token", r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    ("aws key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("bearer token", r"[Bb]earer\s+[A-Za-z0-9._-]{20,}"),
]

# Strings that legitimately match a deny pattern and are scientific provenance
# rather than identity. Each needs a reason.
ALLOW: list[tuple[str, str]] = [
    (r"noreply@anthropic\.com", "should not appear; kept here only to fail loudly if it does"),
]

SKIP_SUFFIXES = {".png", ".pdf", ".pt", ".bin", ".safetensors", ".arrow", ".npz", ".npy", ".jpg", ".ico"}


def texts(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            if path.name == ".git":
                continue
            continue
        if ".git/" in str(path.relative_to(root)).replace("\\", "/"):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        rel = str(path.relative_to(root))
        if path.suffix == ".gz":
            try:
                yield rel, gzip.decompress(path.read_bytes()).decode("utf-8", "replace")
            except OSError:
                continue
            continue
        try:
            yield rel, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: check_anon.py <snapshot dir>")
        return 2
    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 2

    hits: list[str] = []

    for rel, text in texts(root):
        for label, pattern in DENY:
            for match in re.finditer(pattern, text):
                line = text.count("\n", 0, match.start()) + 1
                snippet = text[max(0, match.start() - 40):match.end() + 40].replace("\n", " ")
                hits.append(f"{rel}:{line}: {label}: ...{snippet}...")

    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if rel.startswith(".git/") or rel == ".git":
            continue
        for label, pattern in DENY:
            if re.search(pattern, rel):
                hits.append(f"{rel}: {label} in the path name")

    if (root / ".git").exists():
        log = subprocess.run(["git", "-C", str(root), "log", "--all", "--format=%an|%ae|%s|%b"],
                             capture_output=True, text=True)
        for label, pattern in DENY:
            for line in log.stdout.splitlines():
                if re.search(pattern, line):
                    hits.append(f"git metadata: {label}: {line[:120]}")
        remotes = subprocess.run(["git", "-C", str(root), "remote", "-v"], capture_output=True, text=True)
        if remotes.stdout.strip():
            hits.append(f"git remote configured: {remotes.stdout.strip()[:120]}")

    if hits:
        print(f"ANONYMITY CHECK FAILED: {len(hits)} hit(s)")
        for hit in hits[:80]:
            print("  " + hit)
        if len(hits) > 80:
            print(f"  ... and {len(hits) - 80} more")
        return 1

    files = sum(1 for _ in root.rglob("*") if _.is_file() and ".git/" not in str(_))
    print(f"ANONYMITY CHECK PASSED: {files} files scanned, no hits for {len(DENY)} patterns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
