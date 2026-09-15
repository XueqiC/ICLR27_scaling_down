"""Publication digests must use the audited, explicitly pairable map."""
import json

import pytest

from analysis import provenance
from analysis.paper_artifacts import Artifacts, frozen_run


@pytest.mark.parametrize("pairable,accepted", [(True, True), (False, False), (None, False), ("true", False)])
def test_digest_pairs_require_explicit_true(tmp_path, monkeypatch, pairable, accepted):
    directory = tmp_path / "results"
    directory.mkdir()
    entry = {"frozen_sha256": "frozen", "published_sha256": "published"}
    if pairable is not None:
        entry["pairable"] = pairable
    (directory / "ANONYMIZATION_DIGESTS.json").write_text(json.dumps({"files": {"artifact": entry}}))
    # An ambient/private map must never rescue a rejected or missing local pair.
    monkeypatch.setattr(provenance, "_pairs", lambda: {"frozen": ("tampered", "ambient")})
    audit = Artifacts(tmp_path)
    with frozen_run(tmp_path) as (reads, writes):
        assert audit.matches_digest("frozen", "published") is accepted
        assert not audit.matches_digest("frozen", "tampered")
        assert not audit.matches_digest("unrelated", "published")
        assert not audit.matches_digest("published", "frozen")
    assert set(audit.inputs) == {"results/ANONYMIZATION_DIGESTS.json"}
    assert reads == {str(directory / "ANONYMIZATION_DIGESTS.json")}
    assert not writes


def test_exact_digest_needs_no_map_and_missing_map_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_pairs", lambda: {"frozen": ("published", "ambient")})
    audit = Artifacts(tmp_path)
    with frozen_run(tmp_path):
        assert audit.matches_digest("identical", "identical")
        assert not audit.inputs
        assert not audit.matches_digest("frozen", "published")
