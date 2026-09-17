"""Explicit prerequisites for tests of artifacts excluded from publication.

Do not catch arbitrary FileNotFoundError: an accidentally missing mirrored input
must still fail. Only callers naming a documented publication exclusion may skip.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEACHER_TRACES = tuple(
    f"results/traces-pilot/gpt-5.6-luna_{domain}.jsonl"
    for domain in ("math", "qa", "code")
)


def require_public_inputs(*paths, reason, root=ROOT):
    """Require every relative path/glob, or report what is absent and why."""
    missing = [path for path in paths if not any(p.is_file() for p in root.glob(path))]
    if missing:
        pytest.skip(f"Absent input: {', '.join(missing)}; {reason}")
