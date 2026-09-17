"""Regenerate C86 from published inputs without writing manuscript files."""
from pathlib import Path
import shutil

import pytest

from analysis import s3_paper_table as gen
from public_inputs import require_public_inputs


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path("s3-selection-validation")
SNAPSHOT = ROOT / "data_mirror/published-tables/s3_selection.tex"


@pytest.fixture
def regenerated(tmp_path, monkeypatch):
    # Read the mirror directly: an existing results/ tree may contain stale inputs
    # because bootstrap deliberately never overwrites files. Keep the same layout.
    for name in ("score_v2.json", "plan_v2.json"):
        source = ROOT / "data_mirror" / EXPERIMENT / name
        target = tmp_path / "results" / EXPERIMENT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    monkeypatch.setattr(gen, "ROOT", tmp_path)
    return gen.render().encode("utf-8")


def test_s3_table_matches_published_snapshot(regenerated):
    # These inputs are published. Accidental omissions must fail, not skip.
    assert regenerated == SNAPSHOT.read_bytes()


def test_s3_table_matches_manuscript_when_present(regenerated):
    # The separate manuscript checkout is deliberately outside the public repo.
    require_public_inputs(
        "paper/tables/s3_selection.tex",
        root=ROOT,
        reason="S3_MANUSCRIPT_NOT_DISTRIBUTED: manuscript sources live in a separate "
               "repository; the published snapshot is checked unconditionally",
    )
    assert regenerated == (ROOT / "paper/tables/s3_selection.tex").read_bytes()
