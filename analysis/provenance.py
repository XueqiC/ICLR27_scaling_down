"""Digest comparison that tolerates the published anonymisation, and nothing else.

A frozen record names the SHA-256 of the file it froze. The published copy of that
file cannot carry the same bytes, because the original contains author and host
strings that are removed before publication, so its digest differs by construction.
`data_mirror/ANONYMIZATION_DIGESTS.json` records the correspondence.

`matches(recorded, computed)` returns True when the two digests are equal, or when
the map pairs them as the same file before and after anonymisation. Any other
mismatch is still a mismatch: this widens the check by exactly the set of files
listed in the map, and by nothing else.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_MAP_NAME = "ANONYMIZATION_DIGESTS.json"
_announced: set[str] = set()


@lru_cache(maxsize=1)
def _pairs() -> dict[str, tuple[str, str]]:
    """original digest -> (anonymised digest, path), from the nearest digest map."""
    here = Path(__file__).resolve()
    for parent in [here.parent.parent, *here.parent.parent.parents]:
        candidate = parent / "data_mirror" / _MAP_NAME
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
            return {
                entry["original_sha256"]: (entry["anonymized_sha256"], rel)
                for rel, entry in data.get("files", {}).items()
                if entry.get("original_sha256") and entry.get("anonymized_sha256")
            }
    return {}


def matches(recorded: str, computed: str) -> bool:
    if recorded == computed:
        return True
    paired = _pairs().get(recorded)
    if paired and paired[0] == computed:
        if paired[1] not in _announced:
            _announced.add(paired[1])
            print(f"note: {paired[1]} is the anonymised form of the frozen file "
                  f"{recorded[:12]}; only path strings differ (see data_mirror/{_MAP_NAME})")
        return True
    return False
